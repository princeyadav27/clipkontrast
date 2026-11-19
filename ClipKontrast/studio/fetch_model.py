"""The speech-to-text model: downloaded once, by itself, the first time it is needed.

Nothing has to be run by hand. (python -m studio.fetch_model still works, for getting it ahead of time.) The four
files add up to about 660 MB.

Hugging Face is asked first: it hands the files over about twenty-five times faster than GitHub does (measured:
about 5 MB/s against 200 KB/s), and there is no bzip2 archive to unpack afterwards. GitHub's release tarball is the
backup, for when Hugging Face cannot be reached. Each file is written next to where it will live and moved into
place only when it is whole, so a download that was cut off can never leave half a model behind; and a download
that was cut off carries on from where it stopped instead of starting the 660 MB again.
"""
import json
import shutil
import tarfile
import threading
import urllib.request
from pathlib import Path

from . import config

NAME = config.ASR_MODEL_NAME
# The model comes in two precisions and the parts are named after the one you have: the compact int8 files for a
# processor, the full-precision fp16 ones for a graphics card. Asking for the wrong names is how a machine with a
# card ended up quietly running the compact model, so the names follow the model's own name.
PRECISION = ".fp16" if NAME.endswith("-fp16") else ".int8" if NAME.endswith("-int8") else ""
FILES = ("tokens.txt", *(f"{p}{PRECISION}.onnx" for p in ("encoder", "decoder", "joiner")))
SIZE_TEXT = "about 1.3 GB" if PRECISION == ".fp16" else "about 660 MB"
HF_FILES = f"https://huggingface.co/csukuangfj/{NAME}/resolve/main/"
HF_LIST = f"https://huggingface.co/api/models/csukuangfj/{NAME}/tree/main"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NAME}.tar.bz2"
CHUNK = 1 << 20                       # a megabyte at a time: progress moves, memory does not
_lock = threading.Lock()
_total = [1]                          # every byte the model is, for the progress figure
state = {"doing": False, "percent": 0, "error": ""}


def ready() -> bool:
    """Is the whole model there: the word list and all three parts, under whichever precision name it arrived
    with (encoder.fp16.onnx, encoder.int8.onnx, or plain encoder.onnx)?"""
    d = config.ASR_MODEL_DIR
    if not (d / "tokens.txt").is_file():
        return False
    return all(any((d / f"{part}{k}.onnx").is_file() for k in (".fp16", ".int8", ""))
               for part in ("encoder", "decoder", "joiner"))


def _fetch(url: str, dest: Path, need: int, got: list) -> None:
    """One file, carried on from where an earlier attempt stopped, and only accepted at its full size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _try in (1, 2):
        have = dest.stat().st_size if dest.is_file() else 0
        if have > need:                        # not the file that was expected: start it again
            dest.unlink()
            have = 0
        if have == need:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=30) as r:
            if have and getattr(r, "status", 200) != 206:      # the server ignored the range: from the top
                have = 0
            with open(dest, "ab" if have else "wb") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    got[0] += len(buf)
                    state["percent"] = min(99, int(got[0] * 100 / max(1, _total[0])))
        if dest.stat().st_size == need:
            return
    raise OSError(f"only part of {dest.name} arrived")


def _hf_sizes() -> dict:
    """How big each file of the model is, asked of Hugging Face once (a few hundred bytes of answer)."""
    req = urllib.request.Request(HF_LIST, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        listed = json.loads(r.read().decode("utf-8"))
    return {item["path"]: int(item.get("size") or 0) for item in listed}


def _hf_names(sizes: dict) -> dict:
    """Which name each part has there. Usually the name FILES asks for; a model that leaves the precision out of
    its file names is still usable, so the part's own name is taken instead (encoder.onnx before encoder.fp16.onnx)."""
    want = {}
    for f in FILES:
        if sizes.get(f):
            want[f] = f
            continue
        part = f.split(".")[0]
        alt = [n for n in sizes if n.startswith(part) and n.endswith(".onnx")]
        if alt:
            want[f] = min(alt, key=len)
    return want


def _from_huggingface(part: Path) -> None:
    """The four files, one after another, into the folder they will live in. Anything already there is kept."""
    sizes = _hf_sizes()
    names = _hf_names(sizes)
    missing = [f for f in FILES if f not in names]
    if missing:
        raise OSError(f"the model is not on Hugging Face ({', '.join(missing)})")
    _total[0] = sum(sizes[n] for n in names.values())
    got = [0]
    part.mkdir(parents=True, exist_ok=True)
    for name in names.values():
        _fetch(HF_FILES + name, part / name, sizes[name], got)
    shutil.rmtree(config.ASR_MODEL_DIR, ignore_errors=True)      # a half-finished one from before
    part.rename(config.ASR_MODEL_DIR)


def _from_github(why) -> None:
    """The release tarball, unpacked into place once it is whole. The backup for when Hugging Face is unreachable."""
    print(f"  Hugging Face could not be used ({str(why)[-120:]}); trying GitHub instead.", flush=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.MODELS_DIR / "speech-model.tar.bz2"
    stage = config.MODELS_DIR / "speech-model.unpacking"
    req = urllib.request.Request(URL)
    try:
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("content-length") or 0)
            done = 0
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    state["percent"] = min(99, int(done * 100 / total))
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir()
        with tarfile.open(tmp, "r:bz2") as t:
            for m in t.getmembers():  # only plain files and folders that stay inside the unpacking folder
                if (m.isfile() or m.isdir()) and not m.name.startswith(("/", "..")) and ".." not in m.name.split("/"):
                    t.extract(m, stage)
        got = stage / NAME
        if not got.is_dir():
            raise OSError("the download did not contain the model")
        shutil.rmtree(config.ASR_MODEL_DIR, ignore_errors=True)
        got.rename(config.ASR_MODEL_DIR)
    finally:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)


def ensure() -> None:
    """Make sure the model is on this computer, downloading it if it is not. One download at a time: anything else
    that needs the model meanwhile waits for it here."""
    if ready():
        return
    with _lock:
        if ready():
            return
        state.update(doing=True, percent=0, error="")
        print(f"\n  Downloading the speech model (one time, {SIZE_TEXT})...\n", flush=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        part = config.MODELS_DIR / f"{NAME}.part"       # kept between attempts, so a cut-off download carries on
        try:
            try:
                _from_huggingface(part)
            except Exception as e:  # noqa: BLE001 - Hugging Face away or incomplete: the release tarball, as before
                shutil.rmtree(part, ignore_errors=True)
                _from_github(e)
            if not ready():
                raise OSError("the model did not arrive whole")
            print("  Speech model ready.\n", flush=True)
        except Exception as e:  # noqa: BLE001
            state["error"] = str(e)[-200:]
            raise RuntimeError(f"ClipKontrast couldn't download its speech model ({SIZE_TEXT}, needed once). Check "
                               "the internet connection and free disk space, then try again.") from e
        finally:
            state["doing"] = False


def main():
    if ready():
        print("Speech model already present.")
        return
    ensure()


if __name__ == "__main__":
    main()
