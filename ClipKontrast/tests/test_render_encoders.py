"""Which video encoder a clip is made with.

A graphics card is a bonus, never a requirement: the encoder is used only when this ffmpeg was built with it and
one small picture really came out of it. The processor encoder is always last, so a missing or broken card can
never cost a clip.

Runs against the code directly, with its own empty data folder. No real encoding happens here.
"""
import os, sys, tempfile
os.environ["CLIPKONTRAST_DATA"] = tempfile.mkdtemp(prefix="pc-enc-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio import config, render

fails = []


def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}"))
    ok or fails.append(n)


def setup(built=" h264_nvenc h264_qsv libx264 ", nvenc=None, qsv=None, want="auto"):
    render._cap["list"] = built
    works = {}
    if nvenc is not None:
        works["nvenc"] = nvenc
    if qsv is not None:
        works["qsv"] = qsv
    render._cap["works"] = works
    config.VIDEO_ENCODER = want
    return render.encoders()


check("with a working card, the card encodes and the processor is the backup",
      setup(nvenc=True, qsv=True), ["nvenc", "qsv", "x264"])
check("a card whose driver fails is skipped without trying it again",
      setup(nvenc=False, qsv=True), ["qsv", "x264"])
check("no usable card at all: the processor does all of it", setup(nvenc=False, qsv=False), ["x264"])
check("ffmpeg without the card's encoder is not asked", setup(built=" libx264 ", nvenc=True), ["x264"])

check("forced to the card, a working card is used", setup(nvenc=True, want="nvenc"), ["nvenc", "x264"])
check("...and a broken card still leaves the clip made", setup(nvenc=False, want="nvenc"), ["x264"])
check("forced to the processor", setup(nvenc=True, want="x264"), ["x264"])
check("forced to Intel Quick Sync", setup(nvenc=True, qsv=True, want="qsv"), ["qsv", "x264"])
check("an unknown setting is treated as automatic", setup(nvenc=True, qsv=True, want="something-odd"),
      ["nvenc", "qsv", "x264"])

check("the card's arguments name the card's encoder", "h264_nvenc" in render.ENCODERS["nvenc"][1], True)
check("the processor's arguments are the ones that always work",
      render.ENCODERS["x264"][1][-4:], ["-crf", "21", "-preset", "veryfast"])
check("every encoder has a name and arguments", [k for k, v in render.ENCODERS.items() if len(v) != 2], [])

if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("All encoder checks passed.")
