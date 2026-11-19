"""The speech machine's door: every route needs the shared secret, and bad requests get a clear answer.

The worker can be reached from the internet through a tunnel, so a route without the secret would let anyone use
the graphics card. These checks cover that, the request shapes it accepts, the probe, and the rule that a worker
which cannot use its card says so plainly instead of dying.

Runs against the code directly, with its own empty data folder. The model is never loaded and no audio is decoded.
"""
import os, sys, tempfile
os.environ["CLIPKONTRAST_DATA"] = tempfile.mkdtemp(prefix="pc-asr-"); os.environ["DATABASE_URL"] = ""
os.environ["CLIPKONTRAST_ASR_TOKEN"] = "test-secret-that-is-long"
os.environ["CLIPKONTRAST_FETCH_MODEL"] = "0"
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from studio import asr_worker, transcribe

fails = []


def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}"))
    ok or fails.append(n)


TOKEN = os.environ["CLIPKONTRAST_ASR_TOKEN"]
GOOD = {"Authorization": "Bearer " + TOKEN}
client = TestClient(asr_worker.app)

# ---- the door
for path in ("/healthz", "/probe"):
    r = client.get(path)
    check(f"{path} without the secret is refused", r.status_code, 401)
    check(f"...and does not say anything else", TOKEN not in r.text, True)
    r = client.get(path, headers={"Authorization": "Bearer wrong-secret-here"})
    check(f"{path} with a wrong secret is refused", r.status_code, 401)

r = client.post("/decode", content=b"\x00\x00", headers={"X-Lens": "1"})
check("/decode without the secret is refused", r.status_code, 401)
r = client.post("/decode", content=b"\x00\x00", headers={"X-Lens": "1", **GOOD})
check("/decode with the secret but no model loaded says so, plainly", (r.status_code, "not loaded" in r.text),
      (503, True))
check("...and the reason it gives is the worker's own note", r.json()["detail"],
      asr_worker._state["note"] or "the speech model is not loaded")

h = client.get("/healthz", headers=GOOD)
check("the health check answers with the worker's state", (h.status_code, h.json()["ready"]), (200, False))
check("...without ever naming the secret", TOKEN not in h.text, True)
check("...and says which model and which device", (h.json()["model"], h.json()["device"]),
      (asr_worker._state["model"], transcribe.PROVIDER))

# ---- requests the worker must not try to decode
asr_worker._state.update(ready=True, note="")
bad = [("no lengths at all", "", b""), ("a zero length", "0", b""), ("lengths that do not match the audio", "4", b"\x00\x00")]
for name, lens, body in bad:
    r = client.post("/decode", content=body, headers={"X-Lens": lens, **GOOD})
    check(f"{name} is refused with 400", r.status_code, 400)
r = client.post("/decode", content=b"\x00\x00", headers={"X-Lens": ",".join(["1"] * 257), **GOOD})
check("more pieces than one request may carry is refused", r.status_code, 400)

real_max = asr_worker.MAX_BODY
asr_worker.MAX_BODY = 64
r = client.post("/decode", content=b"\x00" * 200, headers={"X-Lens": "100", **GOOD})
asr_worker.MAX_BODY = real_max
check("a request that is too big is refused with 413", r.status_code, 413)

# Streamed requests do not have to declare Content-Length. Enforce the actual byte count too.
r = client.post("/decode", content=iter([b"a", b"bc"]), headers={"X-Lens": "1", **GOOD})
check("stream must fit its declared audio shape", r.status_code, 400)

# ---- a good request, with the model stubbed out
heard = []


def fake_decode(pieces, rec=None):
    heard.append(len(pieces))
    return [([" hel", "lo"], [0.10, 0.30]) for _ in pieces]


real = transcribe.decode_pieces
transcribe.decode_pieces = fake_decode
try:
    r = client.post("/decode", content=b"\x00" * 3200, headers={"X-Lens": "800,800", **GOOD})
    check("two pieces come back as two answers", (r.status_code, len(r.json()["pieces"])), (200, 2))
    check("the words and their times survive the trip", r.json()["pieces"][0]["tokens"], [" hel", "lo"])
    check("the card was asked once for both pieces", heard, [2])
    p = client.get("/probe", headers=GOOD)
    check("the probe decodes one second of silence", (p.status_code, p.json()["ok"]), (200, True))
    check("...and reports how long the round trip took", isinstance(p.json().get("ms"), int), True)
finally:
    transcribe.decode_pieces = real

# ---- a worker that cannot use its card
asr_worker._state.update(ready=False, note="No NVIDIA graphics card was found (nvidia-smi answered nothing).")
r = client.post("/decode", content=b"\x00\x00", headers={"X-Lens": "1", **GOOD})
check("a worker without a card answers instead of dying", r.status_code, 503)
check("...with the reason, in plain words", "graphics card" in r.json()["detail"], True)
check("...and the reason never contains the secret", TOKEN not in r.text, True)

# ---- plain words for what went wrong, never the secret
class _Resp:
    def __init__(self, code, detail=""):
        self.status_code, self._detail = code, detail

    def json(self):
        return {"detail": self._detail}


class _Err(Exception):
    def __init__(self, code=None, detail="", text=""):
        super().__init__(text or f"HTTP {code}")
        self.response = _Resp(code, detail) if code else None


said = transcribe._probe_reason(_Err(401))
check("a refused secret is explained, and the secret is not repeated", ("shared secret" in said, TOKEN in said), (True, False))
said = transcribe._probe_reason(_Err(503, "the graphics card has only 900 MB free"))
check("a worker that is up but cannot work passes its own reason on", "900 MB free" in said, True)
check("a dead tunnel is described as one",
      "Could not reach" in transcribe._probe_reason(_Err(text="[Errno 11001] getaddrinfo failed")), True)
check("a timeout says to try again", "time" in transcribe._probe_reason(_Err(text="timed out")), True)
said = transcribe._probe_reason(_Err(524, text="HTTP 524"))
check("a tunnel whose notebook stopped is explained", ("stopped" in said, "524" in said), (True, False))

# ---- a secret is never sent to an address anyone can read
check("this computer and its own network are fine over http",
      [transcribe._local_address(u) for u in ("http://127.0.0.1:9000", "http://10.0.0.2:9000",
                                              "http://192.168.1.5:9000", "http://172.20.0.3:9000")],
      [True] * 4)
check("a public address over http is not",
      [transcribe._local_address(u) for u in ("http://example.com:9000", "http://8.8.8.8:9000")], [False, False])
check("https is always fine", transcribe._local_address("https://abc.trycloudflare.com"), False)

if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("All speech-machine checks passed.")
