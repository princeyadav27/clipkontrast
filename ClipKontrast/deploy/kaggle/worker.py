"""Start ClipKontrast's speech machine on a Kaggle notebook with a graphics card, and give it an address your own
computer can reach. Run this in a cell:

    !python deploy/kaggle/worker.py

What it does, in order:
  1. Points ClipKontrast's data folder at /kaggle/working, so the downloaded model survives while the session lasts.
  2. Installs the graphics-card build of the speech library (skipped with --no-install, and skipped when it is
     already there). Nothing else on the notebook is upgraded: no torch, no CUDA.
  3. Looks at the card with nvidia-smi and refuses to load the model when there is not enough memory free. The
     worker still starts: it answers every request with the reason, and your computer does the work itself.
  4. Starts the worker on port 9000, with the shared secret from the Kaggle secret CLIPKONTRAST_ASR_TOKEN.
  5. Opens a Cloudflare quick tunnel to it and prints the https address: put that address in
     CLIPKONTRAST_ASR_URL on your computer, with the same token.

The tunnel is free and needs no account. Its address changes every session and is public, which is why every route
of the worker, including its health check, needs the shared secret. The token itself is never printed.
"""
import argparse
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # the project folder
TUNNEL_BIN = Path("/kaggle/working/bin/cloudflared")
TUNNEL_URL = "https://github.com/cloudflare/cloudflared/releases/download/2026.11.1/cloudflared-linux-amd64"
TUNNEL_SHA256 = "97bb3b02d23a5980fc9c15a1a270e692da1923f103f920d5b16d4931d5ab55e7"
# Only what the worker itself imports. The notebook's own stack (torch, its CUDA, numpy's version) is left alone:
# installing the project's whole requirements.txt here would move versions Kaggle depends on.
WORKER_DEPS = ("fastapi", "uvicorn", "numpy")
# The newest graphics-card build the sherpa-onnx project publishes (checked against its wheel list: there is no CUDA
# build of the newer 2.0 releases). pip picks the file that fits this notebook's Python version.
CUDA_WHEEL = "sherpa-onnx==1.12.21+cuda12.cudnn9"
CUDA_INDEX = "https://k2-fsa.github.io/sherpa/onnx/cuda.html"
NVIDIA_LIBS = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10", "nvidia-cufft-cu12", "nvidia-curand-cu12",
               "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-nvjitlink-cu12")


def say(text=""):
    print(text, flush=True)


def secret() -> str:
    """Require a configured secret. Never leave one in notebook outputs."""
    got = os.environ.get("CLIPKONTRAST_ASR_TOKEN", "").strip()
    if got:
        return got
    try:
        from kaggle_secrets import UserSecretsClient       # only exists inside a Kaggle notebook
        got = (UserSecretsClient().get_secret("CLIPKONTRAST_ASR_TOKEN") or "").strip()
    except Exception:  # noqa: BLE001 - not Kaggle, or the secret was never added
        got = ""
    if got:
        return got
    raise SystemExit("Set CLIPKONTRAST_ASR_TOKEN in Kaggle Secrets and grant this notebook access before starting it.")


def where_data() -> Path:
    """The data folder: /kaggle/working on Kaggle (it survives the session), the project folder anywhere else."""
    got = os.environ.get("CLIPKONTRAST_DATA", "").strip()
    if got:
        return Path(got)
    return Path("/kaggle/working/clipkontrast") if Path("/kaggle/working").is_dir() else ROOT / "data"


def setup_env(data: Path, token: str) -> None:
    os.environ["CLIPKONTRAST_DATA"] = str(data)
    os.environ["DATABASE_URL"] = f"sqlite:///{(data / 'clipkontrast.db').as_posix()}"
    os.environ.setdefault("CLIPKONTRAST_MODELS", str(data / "models"))
    os.environ["CLIPKONTRAST_ASR_PROVIDER"] = os.environ.get("CLIPKONTRAST_ASR_PROVIDER", "cuda")
    os.environ["CLIPKONTRAST_ASR_MODEL"] = os.environ.get("CLIPKONTRAST_ASR_MODEL",
                                                          "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16")
    os.environ["CLIPKONTRAST_ASR_TOKEN"] = token
    os.environ["CLIPKONTRAST_FETCH_MODEL"] = "0"           # the worker fetches the model itself
    data.mkdir(parents=True, exist_ok=True)


def install(force: bool) -> None:
    """The graphics-card build of sherpa-onnx and the NVIDIA libraries it loads. Only these packages are touched."""
    try:
        import sherpa_onnx  # noqa: F401
        marker = Path(sys.prefix) / "clipkontrast-cuda.txt"
        if marker.is_file() and not force:
            say("  The graphics-card speech library is already installed.")
            return
    except ImportError:
        pass
    say("  Installing the graphics-card speech library (a minute or two)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *WORKER_DEPS], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "sherpa-onnx"], check=False)
    done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps",
                           "--force-reinstall", CUDA_WHEEL, "-f", CUDA_INDEX], check=False).returncode
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *NVIDIA_LIBS], check=False)
    if done != 0:
        say("  The graphics-card build could not be installed. The worker will say so and your computer will do "
            "the transcription instead.")
        return
    (Path(sys.prefix) / "clipkontrast-cuda.txt").write_text(CUDA_WHEEL + "\n", encoding="utf-8")


def library_path() -> str:
    """Where the NVIDIA libraries that came with pip live, so sherpa-onnx can load them however the notebook's own
    CUDA is set up."""
    import site
    roots = []
    try:                                   # a notebook's own torch ships CUDA and cuDNN next to itself
        import torch
        if (Path(torch.__file__).parent / "lib").is_dir():
            roots.append(str(Path(torch.__file__).parent / "lib"))
    except Exception:  # noqa: BLE001 - no torch, or it will not import: the pip libraries below are enough
        pass
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        n = Path(p) / "nvidia"
        if n.is_dir():
            roots += [str(n / d / "lib") for d in ("cublas", "cudnn", "cufft", "curand", "cuda_runtime",
                                                   "cuda_nvrtc", "nvjitlink") if (n / d / "lib").is_dir()]
    return ":".join(roots)


def preflight() -> None:
    say("\n  The machine, before anything is loaded:")
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        say("    card: " + ((p.stdout or "").strip() or "nvidia-smi said nothing"))
    except Exception as e:  # noqa: BLE001
        say(f"    card: none found ({str(e)[-80:]})")
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        have = [n for n in ("h264_nvenc", "h264_qsv", "libx264") if f" {n} " in (p.stdout or "")]
        say("    video encoders: " + (", ".join(have) or "none listed"))
    except Exception:  # noqa: BLE001
        say("    video encoders: ffmpeg not found")
    free = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").is_dir() else ROOT).free / 1024 ** 3
    say(f"    free disk: {free:.1f} GB")


_tunnel = {"address": ""}


def tunnel_port(port: int, *, binary=TUNNEL_BIN, found_url=None) -> subprocess.Popen:
    """Open the tunnel and keep reading its output on a thread, so its address is caught whenever it appears and
    the program never blocks on a full pipe."""
    def verified(path):
        digest = hashlib.sha256()
        with path.open("rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == TUNNEL_SHA256

    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.is_file() or not verified(binary):
        say("  Downloading the pinned tunnel program (cloudflared)...")
        with tempfile.NamedTemporaryFile(dir=binary.parent, delete=False) as f:
            tmp = Path(f.name)
        try:
            subprocess.run(["curl", "-fsSL", "--max-time", "180", "-o", str(tmp), TUNNEL_URL], check=True)
            if not verified(tmp):
                raise RuntimeError("Tunnel download did not match its checksum; refusing to run it.")
            tmp.chmod(0o700)
            tmp.replace(binary)
        finally:
            tmp.unlink(missing_ok=True)
    p = subprocess.Popen([str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def watch():
        for line in p.stdout or ():
            found = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if found:
                if found_url:
                    found_url(found.group(0))
                else:
                    _tunnel["address"] = found.group(0)

    import threading
    threading.Thread(target=watch, daemon=True).start()
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Start the speech machine and open a tunnel to it.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLIPKONTRAST_ASR_PORT", "9000")))
    ap.add_argument("--no-install", action="store_true", help="skip installing the graphics-card speech library")
    ap.add_argument("--no-tunnel", action="store_true", help="do not open a tunnel (a machine on your own network)")
    args = ap.parse_args()

    data = where_data()
    token = secret()
    setup_env(data, token)
    os.environ["CLIPKONTRAST_ASR_PORT"] = str(args.port)
    if not args.no_tunnel:
        os.environ["CLIPKONTRAST_ASR_HOST"] = "127.0.0.1"
    say(f"  Data and model folder: {data}")
    preflight()
    if not args.no_install:
        install(force=False)
    libs = library_path()
    if libs:
        os.environ["LD_LIBRARY_PATH"] = libs + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")

    say("\n  Starting the speech machine...")
    worker = subprocess.Popen([sys.executable, "-m", "studio.asr_worker"], cwd=str(ROOT),
                              env={**os.environ, "PYTHONPATH": str(ROOT)})
    tun = None
    try:
        if args.no_tunnel:
            say(f"  No tunnel. Reachable on this machine at http://127.0.0.1:{args.port}")
        else:
            tun = tunnel_port(args.port)
            say("  Waiting for the tunnel to hand out an address...")
            began = time.time()
            while not _tunnel["address"] and time.time() - began < 180 and worker.poll() is None:
                time.sleep(0.5)
            if _tunnel["address"]:
                say("\n" + "=" * 78)
                say("  Speech machine address (put this in CLIPKONTRAST_ASR_URL on your computer):")
                say("      " + _tunnel["address"])
                say("  Keep the same CLIPKONTRAST_ASR_TOKEN on both, and press Test connection in Settings, AI.")
                say("  This address changes every session. It stops working when this notebook stops.")
                say("=" * 78 + "\n")
            else:
                say("  No address yet. Is internet turned on for this notebook (Settings, right side)? If it is, the "
                    "tunnel is slow to start: leave this cell running and the address appears here when it comes.")
        while worker.poll() is None:
            if _tunnel["address"] and not _tunnel.get("said"):
                _tunnel["said"] = True
            time.sleep(2)
        say("\n  The speech machine stopped. Read the messages above for the reason.")
        return worker.returncode or 0
    except KeyboardInterrupt:
        say("\n  Stopping.")
        return 0
    finally:
        for p in (worker, tun):
            if p is not None and p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
