"""The speech-to-text machine: a tiny service that turns pieces of audio into words, meant for a machine with a
graphics card. The main ClipKontrast server sends it pieces and gets the words back (see studio/transcribe.py).

Run with:  python -m studio.asr_worker
Settings:  CLIPKONTRAST_ASR_PROVIDER=cuda, CLIPKONTRAST_ASR_TOKEN=<shared secret>, CLIPKONTRAST_ASR_PORT (9000),
           CLIPKONTRAST_ASR_BATCH (pieces decoded together on the card, default 16),
           CLIPKONTRAST_ASR_PARALLEL (requests decoded at once, default 2),
           CLIPKONTRAST_ASR_MIN_VRAM_MB (free memory needed before the model is loaded, default 4000)

It holds nothing: no accounts, no files, no database. Every route, including the health check and the probe, needs
the shared secret: a worker reached through a tunnel is a public address, and an open one would let anyone use the
card. When the card or its memory is not usable, it stays up and says so instead of dying, and every request gets a
plain answer that the main server can act on.
"""
import hmac
import os
import subprocess
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from . import config, transcribe

TOKEN = os.environ.get("CLIPKONTRAST_ASR_TOKEN", "").strip()
PORT = int(os.environ.get("CLIPKONTRAST_ASR_PORT", "9000"))
CARD_BATCH = int(os.environ.get("CLIPKONTRAST_ASR_BATCH", "16"))
PARALLEL = max(1, int(os.environ.get("CLIPKONTRAST_ASR_PARALLEL", "2")))       # requests worked on at once
MIN_VRAM_MB = int(os.environ.get("CLIPKONTRAST_ASR_MIN_VRAM_MB", "4000"))
MAX_BODY = 64 * 1024 * 1024             # about 33 minutes of sound in one request: far more than is ever sent
MAX_PIECE = 16000 * 40                  # a piece is 12 to 24 seconds; 40 is the ceiling

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_stats = {"pieces": 0, "audio_s": 0.0, "busy_s": 0.0, "since": time.time()}
_slock = threading.Lock()
_slots = threading.BoundedSemaphore(PARALLEL)
_state = {"ready": False, "note": "", "device": transcribe.PROVIDER, "model": config.ASR_MODEL_NAME}


def _allowed(request: Request) -> bool:
    """The shared secret, and nothing else. A worker with no secret set never answers anyone."""
    return bool(TOKEN) and hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + TOKEN)


def _gpu() -> dict:
    """The card's name and its memory, from nvidia-smi. Empty when there is no card to ask."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        name, total, free = [x.strip() for x in (p.stdout or "").strip().splitlines()[0].split(",")[:3]]
        return {"name": name, "total_mb": int(float(total)), "free_mb": int(float(free))}
    except Exception:  # noqa: BLE001 - no card, no driver, or no nvidia-smi: all the same answer here
        return {}


_enc = {"list": None}


def _encoders() -> list:
    """Which video encoders this machine's ffmpeg has. The notebook renders nothing, but knowing whether the card
    can encode tells the app what to expect from the same machine."""
    if _enc["list"] is None:
        try:
            p = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=30)
            got = p.stdout or ""
        except Exception:  # noqa: BLE001
            got = ""
        _enc["list"] = [n for n in ("h264_nvenc", "h264_qsv", "libx264") if f" {n} " in got]
    return list(_enc["list"])


def _plain(e) -> str:
    """The error in words that mean something on a notebook: memory, drivers, or something else."""
    text = str(e)
    low = text.lower()
    if "out of memory" in low or "cudamalloc" in low or "alloc" in low and "memory" in low:
        return ("The graphics card ran out of memory. Close other notebooks using the GPU, or set "
                "CLIPKONTRAST_ASR_PARALLEL=1, and try again.")
    if "cudnn" in low or "libcudnn" in low:
        return ("The graphics card's libraries could not be loaded (cuDNN). Run the setup cell again so the CUDA "
                "packages are installed, then start the worker.")
    if "cuda" in low or "gpu" in low:
        return ("The graphics card could not be used: " + text[-160:])
    return text[-200:] or "the speech model could not answer"


@app.get("/healthz")
def healthz(request: Request):
    if not _allowed(request):
        raise HTTPException(401, "no")
    with _slock:
        s = dict(_stats)
    gpu = _gpu()
    return {"ok": _state["ready"], "ready": _state["ready"], "note": _state["note"],
            "device": _state["device"], "model": _state["model"],
            "provider": transcribe.PROVIDER, "gpu": gpu, "encoders": _encoders(),
            "vram_free_mb": gpu.get("free_mb"), "parallel": PARALLEL,
            "pieces": s["pieces"], "audio_hours": round(s["audio_s"] / 3600, 2),
            "speed": round(s["audio_s"] / s["busy_s"], 1) if s["busy_s"] else None}


@app.get("/probe")
def probe(request: Request):
    """A round trip with one second of silence: proves the secret, the model and the card all work."""
    if not _allowed(request):
        raise HTTPException(401, "no")
    if not _state["ready"]:
        raise HTTPException(503, _state["note"] or "the speech model is not loaded")
    began = time.time()
    try:
        transcribe.decode_pieces([np.zeros(16000, dtype=np.float32)])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, _plain(e))
    return {"ok": True, "ms": round((time.time() - began) * 1000), "device": _state["device"],
            "model": _state["model"], "vram_free_mb": _gpu().get("free_mb")}


@app.post("/decode")
async def decode(request: Request):
    if not _allowed(request):
        raise HTTPException(401, "no")
    try:
        lens = [int(x) for x in request.headers.get("x-lens", "").split(",") if x]
    except ValueError:
        raise HTTPException(400, "bad lengths")
    if not lens or len(lens) > 256 or any(n <= 0 or n > MAX_PIECE for n in lens):
        raise HTTPException(400, "bad lengths")
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        raise HTTPException(413, "too much")
    body = await request.body()
    if len(body) != 2 * sum(lens) or len(body) > MAX_BODY:
        raise HTTPException(400, "lengths do not match the audio")
    if not _state["ready"]:
        # The app finishes these pieces on its own processor, and Settings shows this note.
        raise HTTPException(503, _state["note"] or "the speech model is not loaded")
    audio = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    pieces, pos = [], 0
    for n in lens:
        pieces.append(audio[pos:pos + n])
        pos += n

    def work():
        t0 = time.time()
        out = []
        for i in range(0, len(pieces), max(1, CARD_BATCH)):
            out += transcribe.decode_pieces(pieces[i:i + max(1, CARD_BATCH)])
        with _slock:
            _stats["pieces"] += len(pieces)
            _stats["audio_s"] += sum(lens) / 16000
            _stats["busy_s"] += time.time() - t0
        return out

    import anyio
    try:
        with _slots:                                  # a second request waits for the card instead of fighting for it
            got = await anyio.to_thread.run_sync(work)
    except Exception as e:  # noqa: BLE001 - a broken or overloaded card must answer, not crash the worker
        raise HTTPException(503, _plain(e))
    return {"pieces": [{"tokens": tok, "timestamps": ts} for tok, ts in got]}


def _preflight() -> None:
    """Look at the machine before loading the model. Whatever is wrong is written down and served, so the app can
    say why the card is not being used instead of the worker dying in a traceback."""
    gpu = _gpu()
    if transcribe.PROVIDER != "cpu":
        if not gpu:
            _state["note"] = ("No NVIDIA graphics card was found (nvidia-smi answered nothing). Set "
                              "CLIPKONTRAST_ASR_PROVIDER=cpu to run on the processor instead, or start this on a "
                              "machine with a card.")
        elif gpu.get("free_mb", 0) < MIN_VRAM_MB:
            _state["note"] = (f"The graphics card has only {gpu['free_mb']} MB free and {MIN_VRAM_MB} MB is needed. "
                              "Close whatever else is using it, or lower CLIPKONTRAST_ASR_MIN_VRAM_MB.")
    if _state["note"]:
        print("  " + _state["note"], flush=True)
        return                                       # stay up: every request gets this answer, plainly
    try:
        transcribe._recognizer()                     # load the model now, so the first request is not the slow one
        _state.update(ready=True, device=(gpu.get("name") or transcribe.PROVIDER))   # the card's name, not "cuda"
    except Exception as e:  # noqa: BLE001
        _state["note"] = "The speech model could not be loaded: " + _plain(e)
        print("  " + _state["note"], flush=True)


def main():
    if not TOKEN:
        raise SystemExit("Set CLIPKONTRAST_ASR_TOKEN: the speech machine refuses to start without a shared secret.")
    _preflight()
    card = _gpu()
    where = f"{card.get('name')} ({card.get('free_mb')} MB free)" if card else "this machine's processor"
    print(f"Speech machine ready on port {PORT} ({transcribe.PROVIDER}, {where})"
          + ("" if _state["ready"] else " - NOT READY, see the message above"), flush=True)
    uvicorn.run(app, host=os.environ.get("CLIPKONTRAST_ASR_HOST", "0.0.0.0"), port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
