"""Speech to text with word timestamps, using Parakeet 0.6b through sherpa-onnx.

It runs in one of two places. On this machine's processors (the default), or, when CLIPKONTRAST_ASR_URL is set, on a
separate machine with a graphics card (studio/asr_worker.py), which does the same work many times faster. If that
machine cannot be reached, the pieces are done here instead, so transcription never stops because of it."""
import os
import queue
import re
import threading
import time

import numpy as np

from . import config

_lock = threading.Lock()

# Several copies of the speech model, each on a few cores, instead of one copy on all of them: one copy cannot keep
# a big machine busy, and several videos transcribing at once would queue behind each other inside it. On a small
# machine (under 8 cores) this is still a single copy, as before.
_CPUS = os.cpu_count() or 4
PROVIDER = os.environ.get("CLIPKONTRAST_ASR_PROVIDER", "cpu").strip().lower() or "cpu"      # "cuda" on a graphics card
THREADS = int(os.environ.get("CLIPKONTRAST_ASR_THREADS") or (4 if _CPUS >= 8 else max(2, _CPUS - 2)))
COPIES = int(os.environ.get("CLIPKONTRAST_ASR_COPIES") or (1 if PROVIDER != "cpu" else max(1, _CPUS // THREADS) if _CPUS >= 8 else 1))
_free: "queue.LifoQueue" = queue.LifoQueue()
_made = 0

# The machine with the graphics card, if there is one.
REMOTE = os.environ.get("CLIPKONTRAST_ASR_URL", "").strip().rstrip("/")
REMOTE_TOKEN = os.environ.get("CLIPKONTRAST_ASR_TOKEN", "").strip()
REMOTE_PROBLEM = ""        # why that machine is not being used, in plain words, for Settings


def _local_address(url: str) -> bool:
    """An address that cannot be read off the internet: this computer, or its own network."""
    from urllib.parse import urlsplit
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    if host in ("localhost", "::1", "0.0.0.0") or host.endswith(".local"):
        return True
    if re.fullmatch(r"127\.\d{1,3}\.\d{1,3}\.\d{1,3}", host):
        return True
    if re.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}", host) or re.fullmatch(r"192\.168\.\d{1,3}\.\d{1,3}", host):
        return True
    return bool(re.fullmatch(r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}", host))


if REMOTE and not REMOTE.startswith("https://") and not _local_address(REMOTE) \
        and os.environ.get("CLIPKONTRAST_ASR_ALLOW_INSECURE", "0") != "1":
    # A tunnel hands out an https address. Any other public address would send the shared secret in the clear.
    REMOTE_PROBLEM = ("The speech machine's address is not https, so the shared secret would travel in the clear. "
                      "ClipKontrast is not sending anything there. Use the https address the tunnel printed.")
    REMOTE = ""

_health = {"ok": None, "since": 0.0, "ms": None, "message": "", "device": "", "model": "",
           "vram_free_mb": None}


def status() -> dict:
    """What Settings shows about the speech machine, in plain words. The token is never part of it."""
    return {"remote": bool(REMOTE), "problem": REMOTE_PROBLEM, "ok": _health["ok"], "ms": _health["ms"],
            "message": _health["message"], "since": _health["since"], "device": _health["device"],
            "model": _health["model"], "vram_free_mb": _health["vram_free_mb"],
            "waiting_s": round(max(0.0, _remote_down_until - time.time()))}


def _probe_reason(e) -> str:
    """What went wrong with the speech machine, in plain words. The shared secret is never part of the answer."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code == 401:
        return ("The speech machine refused the shared secret. Both machines must have the same "
                "CLIPKONTRAST_ASR_TOKEN.")
    if code == 503:
        try:
            detail = e.response.json().get("detail") or ""
        except Exception:  # noqa: BLE001
            detail = ""
        return "The speech machine is running but cannot work yet: " + (detail or "the speech model is not loaded.")
    if code and 520 <= code <= 530:
        # Cloudflare's own numbers: the tunnel address is still there, but the notebook behind it never replied.
        return ("The speech machine's address answers, but the notebook behind it does not. Kaggle has probably "
                "stopped the notebook: start it again and put the new address it prints into "
                "CLIPKONTRAST_ASR_URL. Until then videos are transcribed on this computer.")
    if code:
        return f"The speech machine answered with an error (HTTP {code})."
    text = str(e).lower()
    if "timed out" in text or "timeout" in text:
        return ("The speech machine did not answer in time. A notebook that has just started takes a minute: try "
                "again. If it keeps saying this, the notebook has probably stopped, so start it again and put the "
                "new address it prints into CLIPKONTRAST_ASR_URL.")
    if any(k in text for k in ("connect", "resolve", "refused", "getaddrinfo", "name or service")):
        return ("Could not reach the speech machine at all. Is the notebook still running, and is its tunnel address "
                "still the one in Settings?")
    return "The speech machine could not be reached."


PROBE_WAIT = int(os.environ.get("CLIPKONTRAST_ASR_PROBE_WAIT", "60"))     # seconds to wait for one round trip


def probe() -> dict:
    """Ask the speech machine for one second of silence: the secret, the model and the card in one round trip.
    Returns what happened, in plain words."""
    if not REMOTE:
        return {"ok": False, "note": REMOTE_PROBLEM or "No speech machine is set up, so videos are transcribed on "
                                                         "this computer."}
    if not REMOTE_TOKEN:
        return {"ok": False, "note": "The speech machine's shared secret is missing (CLIPKONTRAST_ASR_TOKEN)."}
    import httpx
    began = time.time()
    try:
        # A ready machine answers in milliseconds, so a minute is generous, and a notebook that has gone keeps the
        # caller waiting no longer than that.
        r = httpx.get(REMOTE + "/probe", headers={"Authorization": "Bearer " + REMOTE_TOKEN},
                      timeout=httpx.Timeout(PROBE_WAIT, connect=10))
        r.raise_for_status()
        got = r.json()
    except Exception as e:  # noqa: BLE001 - wrong secret, dead notebook, model not loaded: all answered here
        msg = _probe_reason(e)
        _health.update(ok=False, since=time.time(), ms=None, message=msg)
        return {"ok": False, "note": msg}
    ms = got.get("ms")
    _health.update(ok=True, since=time.time(), ms=ms, message="", device=got.get("device") or "",
                   model=got.get("model") or "", vram_free_mb=got.get("vram_free_mb"))
    return {"ok": True, "ms": ms, "device": got.get("device") or "", "model": got.get("model") or "",
            "vram_free_mb": got.get("vram_free_mb"), "note": ""}
REMOTE_BATCH = int(os.environ.get("CLIPKONTRAST_ASR_REMOTE_BATCH", "8"))         # pieces sent together: the card does them at once
REMOTE_PARALLEL = int(os.environ.get("CLIPKONTRAST_ASR_REMOTE_PARALLEL", "4"))   # pieces of one video worked on at once
# The card has a ceiling: beyond a few requests at once it gets no faster. So only this many are sent to it at a
# time (all videos together), and the rest is done on this machine's own processors at the same moment: both are used.
REMOTE_SLOTS = threading.BoundedSemaphore(max(1, int(os.environ.get("CLIPKONTRAST_ASR_REMOTE_INFLIGHT", "6"))))
_remote_down_until = 0.0


def model_ready() -> bool:
    from . import fetch_model
    return fetch_model.ready()


def _model_files(d) -> dict:
    """The model's three parts. A graphics card wants the full-precision files; processors the compact ones."""
    kinds = ("fp16", "", "int8") if PROVIDER != "cpu" else ("int8", "", "fp16")
    for k in kinds:
        names = {p: d / (f"{p}.{k}.onnx" if k else f"{p}.onnx") for p in ("encoder", "decoder", "joiner")}
        if all(f.is_file() for f in names.values()):
            return {p: str(f) for p, f in names.items()}
    raise FileNotFoundError(f"No speech model files in {d}")


_lib = {"mod": None}
_lib_lock = threading.Lock()


def _sherpa():
    """The speech library, loaded once. Several copies of the model start on parallel threads, so loading is done
    under a lock: two threads loading it at the same moment would trip over each other."""
    with _lib_lock:
        if _lib["mod"] is None:
            _lib["mod"] = _load_sherpa()
        return _lib["mod"]


def _load_sherpa():
    """On Windows a file whose full path is over 260 characters cannot be loaded, which happens when ClipKontrast sits in a
    deep folder ("DLL load failed ... The filename or extension is too long"). Then the library is loaded through
    the folder's short name (C:\\Users\\...\\SITE-P~1), which points at the same files, or else from a copy of it
    in a short folder (AppData\\Local\\ClipKontrast\\lib), made once."""
    import sys
    try:
        import sherpa_onnx
        return sherpa_onnx
    except ImportError as e:
        if os.name != "nt" or "too long" not in str(e).lower():
            raise
        first = e
    import ctypes
    import importlib
    import importlib.util
    import shutil
    from pathlib import Path

    def forget():
        for k in [m for m in sys.modules if m == "sherpa_onnx" or m.startswith("sherpa_onnx.")]:
            del sys.modules[k]
        importlib.invalidate_caches()

    def load_from(folder: str):
        forget()
        sys.path.insert(0, folder)          # stays first: later imports inside the library look here too
        try:
            return importlib.import_module("sherpa_onnx")
        except ImportError:
            sys.path.remove(folder)
            forget()
            return None

    forget()
    spec = importlib.util.find_spec("sherpa_onnx")
    pkg = Path(spec.origin).parent if spec and spec.origin else None
    if pkg is None:
        raise first
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(str(pkg.parent), buf, 1024) and len(buf.value) < len(str(pkg.parent)):
        mod = load_from(buf.value)
        if mod:
            return mod
    # A copy of the library in a short folder. Same version check each start, so an update replaces it.
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ClipKontrast" / "lib"
    stamp = base / "sherpa_onnx.from"
    want = f"{pkg}|{(pkg / '__init__.py').stat().st_mtime_ns}"
    try:
        if not (base / "sherpa_onnx").is_dir() or not stamp.is_file() or stamp.read_text(encoding="utf-8") != want:
            shutil.rmtree(base / "sherpa_onnx", ignore_errors=True)
            shutil.copytree(pkg, base / "sherpa_onnx")
            stamp.write_text(want, encoding="utf-8")
        mod = load_from(str(base))
        if mod:
            return mod
    except OSError:
        pass
    raise RuntimeError("ClipKontrast is in a folder too deep for Windows to load the speech model from. Move the ClipKontrast "
                       "folder somewhere short, like C:\\clipkontrast, and start it again.") from first


def _new():
    sherpa_onnx = _sherpa()
    from . import fetch_model
    fetch_model.ensure()             # downloaded here, once, if it was never fetched
    d = config.ASR_MODEL_DIR
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **_model_files(d),
        tokens=str(d / "tokens.txt"),
        num_threads=THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider=PROVIDER,
    )


def _take():
    """A free copy of the model: one that is idle, a new one while there is room for more, else wait for one."""
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if make:
        try:
            return _new()
        except Exception:
            with _lock:
                _made -= 1
            raise
    return _free.get()


def _take_now():
    """A copy of the model if one is free (or can still be made) this instant, else None."""
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if not make:
        return None
    try:
        return _new()
    except Exception:
        with _lock:
            _made -= 1
        raise


def _recognizer():
    """Kept for callers that only need the model loaded (warm-up)."""
    rec = _take()
    _free.put(rec)
    return rec


def _chunks(audio: np.ndarray, lo=12.0, hi=24.0):
    """Split at the quietest point between lo and hi seconds so words are not cut in half."""
    sr, hop = 16000, 800  # 50 ms frames
    n = len(audio)
    frames = n // hop
    if frames == 0:
        return
    # a block of frames at a time, so a long video does not need a gigabyte of memory at once
    rms = np.empty(frames, dtype=np.float32)
    step = 12000                          # ten minutes of 50 ms frames
    for i in range(0, frames, step):
        j = min(frames, i + step)
        x = audio[i * hop: j * hop].reshape(j - i, hop).astype(np.float32)
        rms[i:j] = np.sqrt((x * x).mean(axis=1))
    pos = 0
    while pos < n:
        if n - pos <= hi * sr:
            yield pos, n
            return
        a, b = (pos + int(lo * sr)) // hop, (pos + int(hi * sr)) // hop
        cut = (a + int(np.argmin(rms[a:b]))) * hop + hop // 2
        yield pos, cut
        pos = cut


BATCH = int(os.environ.get("CLIPKONTRAST_ASR_BATCH", "2"))   # pieces decoded together. Measured on a 4-core laptop:
# two at a time is about 7% quicker than one (5.9x real time against 5.5x), and the words come back exactly the same.


def _words(tokens, timestamps, a: int, b: int) -> list:
    out = []
    off, seg_end = a / 16000.0, b / 16000.0
    cur, gap = None, False
    for tok, t in zip(tokens, timestamps):
        new_word = tok.startswith((" ", "▁")) or gap
        piece = tok.replace("▁", " ").strip()
        if not piece:
            gap = True      # a space on its own: whatever comes next starts a new word ("in" "120", not "in120")
            continue
        gap = False
        is_punct = all(not c.isalnum() for c in piece)
        if cur is None or (new_word and not is_punct):
            if cur:
                out.append(cur)
            cur = {"w": piece, "s": round(off + t, 2), "e": round(off + t + 0.12, 2)}
        else:
            cur["w"] += piece
            cur["e"] = round(off + t + 0.12, 2)
    if cur:
        cur["e"] = min(cur["e"], round(seg_end, 2))
        out.append(cur)
    return out


def decode_pieces(pieces: list, rec=None) -> list:
    """[(tokens, timestamps), ...] for float32 pieces of audio, done on this machine with one copy of the model."""
    streams = []
    rec = rec or _take()       # held only for these pieces, so videos transcribing at once take turns fairly
    try:
        for seg in pieces:
            st = rec.create_stream()
            st.accept_waveform(16000, seg)
            streams.append(st)
        rec.decode_streams(streams)
    finally:
        _free.put(rec)
    return [(list(st.result.tokens), [float(t) for t in st.result.timestamps]) for st in streams]


def _remote_ok() -> bool:
    return bool(REMOTE) and time.time() >= _remote_down_until


def _decode_remote(pieces: list) -> list:
    """The same, on the machine with the graphics card. Audio goes as 16-bit sound, the answer comes back as text.

    Tried twice: a tunnel that dropped for a second is worth one more go before the video is finished on this
    processor. If it stays away, the reason is kept for Settings, so the switch is never silent."""
    import httpx
    raw = [np.clip(seg * 32768.0, -32768, 32767).astype("<i2") for seg in pieces]
    body = b"".join(x.tobytes() for x in raw)
    headers = {"Authorization": "Bearer " + REMOTE_TOKEN, "X-Lens": ",".join(str(len(x)) for x in raw),
               "Content-Type": "application/octet-stream"}
    last = None
    for attempt in (1, 2):
        began = time.time()
        try:
            r = httpx.post(REMOTE + "/decode", content=body, timeout=httpx.Timeout(240, connect=8), headers=headers)
            r.raise_for_status()
            got = r.json()["pieces"]
            if len(got) != len(pieces):
                raise ValueError("wrong number of pieces came back")
            _health.update(ok=True, since=time.time(), ms=round((time.time() - began) * 1000), message="")
            return [(p["tokens"], p["timestamps"]) for p in got]
        except Exception as e:  # noqa: BLE001 - the card's machine is away or busy: this attempt failed
            last = e
            if attempt == 1:
                time.sleep(1.5)
    _health.update(ok=False, since=time.time(), ms=None,
                   message="The graphics card machine stopped answering. These videos are finishing on this "
                           "computer's processor, which is much slower. Check the notebook is still running and that "
                           "its address is still the one in Settings.")
    raise last


def transcribe(audio: np.ndarray, on_progress=None) -> list:
    """Returns [{'w': word, 's': start_sec, 'e': end_sec}, ...] for the whole audio.
    Speech is cut at quiet moments into 12-24 s pieces; silent pieces are skipped."""
    global _remote_down_until
    words = []
    todo = []
    for a, b in _chunks(audio):
        seg = audio[a:b].astype(np.float32) / 32768.0
        if len(seg) < 4800 or float(np.sqrt((seg ** 2).mean())) < 0.003:
            continue  # silence: skip, saves time and avoids made-up words
        todo.append((a, b, seg))
    remote = _remote_ok()
    size = max(1, REMOTE_BATCH if remote else BATCH)
    groups = [todo[i:i + size] for i in range(0, len(todo), size)]
    done = [0]

    def decode(group):
        global _remote_down_until
        segs = [seg for _, _, seg in group]
        got = None
        while got is None:
            # whichever is free first: a place on the card's machine, or a copy of the model on this one
            if remote and _remote_ok() and REMOTE_SLOTS.acquire(blocking=False):
                try:
                    got = _decode_remote(segs)
                except Exception:  # noqa: BLE001 - the card's machine is away or busy: these pieces are done here
                    _remote_down_until = time.time() + 60
                finally:
                    REMOTE_SLOTS.release()
                continue
            rec = _take_now()
            if rec is None and not (remote and _remote_ok()):
                rec = _take()                                # nothing else to wait for: wait for a copy here
            if rec is not None:
                got = []
                step = max(1, BATCH)
                try:
                    for i in range(0, len(segs), step):
                        got += decode_pieces(segs[i:i + step], rec)
                        rec = None if i + step >= len(segs) else _take()
                except Exception:
                    got = None
                    raise
                continue
            time.sleep(0.02)
        out = [_words(tok, ts, a, b) for (a, b, _), (tok, ts) in zip(group, got)]
        done[0] += 1
        if on_progress:
            on_progress(done[0] / len(groups))
        return out

    # Pieces are independent, so they go to every copy of the model that is free: a video alone on the server
    # uses all of it, and several videos share it piece by piece. The words come back in order.
    workers = REMOTE_PARALLEL if remote else COPIES
    if workers > 1 and len(groups) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(groups)), thread_name_prefix="asr") as ex:
            results = list(ex.map(decode, groups))
    else:
        results = [decode(g) for g in groups]
    for out in results:
        for piece in out:
            for w in piece:
                if words and words[-1]["w"] == w["w"] and words[-1]["s"] == w["s"]:
                    continue  # the model sometimes says the same word twice at the same moment
                words.append(w)
    # A word lasts until the next one starts, capped so pauses stay pauses.
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.4
        w["e"] = round(max(w["s"] + 0.08, min(nxt, max(w["e"], w["s"] + 0.6))), 2)
    return words
