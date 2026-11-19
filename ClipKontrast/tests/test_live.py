"""Judging a live stream in whole blocks.

The two AI judges read a live stream block by block, one after another, so every second is judged once:
nothing is slipped over and nothing is judged twice. These checks cover where a block starts and ends, the
limit the recording buffer puts on it, and the last shorter block after the stream ends or Stop is pressed.

Runs against the code directly, with its own empty data folder.
"""
import os, sys, tempfile
os.environ["CLIPKONTRAST_DATA"] = tempfile.mkdtemp(prefix="pc-live-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio.live import window_span

fails = []


def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}"))
    ok or fails.append(n)


BUF = 20 * 60          # the default: twenty minutes of stream are kept on disk
WIN = 10 * 60          # the default: one look covers ten minutes of stream

check("a block is judged once it is full", window_span(0, WIN, WIN, BUF), (0, WIN))
check("...and not a second before", window_span(0, WIN - 1, WIN, BUF), None)
check("the next block starts exactly where the last one ended", window_span(WIN, 2 * WIN, WIN, BUF), (WIN, 2 * WIN))

# three hours, judged as it records: every second covered once, in order, with no gap
spans, scanned, now = [], 0, 0
while now < 3 * 3600:
    now += 1
    span = window_span(scanned, now, WIN, BUF)
    if span:
        spans.append(span)
        scanned = span[1]
check("three hours fit into whole blocks", len(spans), 18)
check("the blocks join up with no gap", [a[1] for a in spans[:-1]], [b[0] for b in spans[1:]])
check("the blocks never overlap", [s for s in spans if s[0] >= s[1]], [])
check("coverage reaches the end of the stream", (spans[0][0], spans[-1][1]), (0, 3 * 3600))

check("a block is never longer than the buffer keeps", window_span(0, 3 * 3600, 60 * 60, BUF),
      (3 * 3600 - (BUF - 120), 3 * 3600))
check("a small buffer makes a shorter block", window_span(0, 200, 60 * 60, 6 * 60), None)
check("...and gives a four-minute block, not an hour", window_span(0, 240, 60 * 60, 6 * 60), (0, 240))
check("the oldest seconds are given up, never clipped from nothing",
      window_span(0, 6 * 60, 60 * 60, 6 * 60), (2 * 60, 6 * 60))

check("after the stream ends the last few minutes are judged too",
      window_span(WIN, WIN + 60, WIN, BUF, ended=True), (WIN, WIN + 60))
check("...but a handful of seconds waits for nothing", window_span(WIN, WIN + 20, WIN, BUF, ended=True), None)
check("nothing new is started while the last look is still running",
      window_span(WIN, WIN + 60, WIN, BUF, ended=True, busy=True), None)
check("a full block is judged even while speech is still being written down",
      window_span(WIN, WIN + WIN, WIN, BUF, busy=True), (WIN, WIN + WIN))

if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("All live block checks passed.")
