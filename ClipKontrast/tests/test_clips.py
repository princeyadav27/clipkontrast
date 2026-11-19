"""Does a clip work for someone who only ever sees the clip?

It has to open on a whole thought, have something happen inside it, and end on the payoff instead of
drifting into the next topic. These checks cover the rules that decide which moments are kept, which are
left out, and what the video's card says about the ones that were left out.

Runs against the code directly, with its own empty data folder.
"""
import os, sys, tempfile
os.environ["CLIPKONTRAST_DATA"] = tempfile.mkdtemp(prefix="pc-clips-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
import numpy as np
from studio import brain, jobs

fails = []


def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}"))
    ok or fails.append(n)


def talk(sentences, start=0.0, pause=0.35):
    """Words as a speech model writes them: a sentence in the talk() list ends with a full stop."""
    out, t = [], start
    for s in sentences:
        parts = s.split()
        for i, w in enumerate(parts):
            out.append({"w": w + ("." if i == len(parts) - 1 else ""), "s": round(t, 2), "e": round(t + 0.4, 2)})
            t += 0.5
        t += pause
    return out


def at(words, text):
    return next(i for i, w in enumerate(words) if w["w"].lower().startswith(text))


def loud_on(words, a, b, size=80, peak=2.6):
    """Loudness per second, with the shouting only between two words: that is where the reaction is."""
    z = np.zeros(size, dtype=np.float32)
    z[int(words[a]["s"]):int(words[b]["e"]) + 1] = peak
    return z


STREAM = talk([
    "so anyway today we are doing something a bit different",
    "i am going to bet everything on this one jump",
    "if i miss this i lose the whole run",
    "here we go",
    "no way no way he actually made it",
    "i cannot believe that worked",
    "alright let us move on to the next level",
])
setup, payoff, cooldown = at(STREAM, "going"), at(STREAM, "made"), at(STREAM, "alright")
z = loud_on(STREAM, at(STREAM, "no"), payoff)

# ---- the score: a clip that stands on its own
alone, why = brain.standalone_score(STREAM[setup]["s"], STREAM[payoff]["e"], STREAM, z)
check("a setup that runs into a loud payoff scores well", alone, lambda v: v >= 65)
check("...and has nothing held against it", why, [])

check("the same clip dragged into 'alright, moving on' is marked", 
      brain.standalone_score(STREAM[setup]["s"], STREAM[cooldown + 4]["e"], STREAM, loud_on(STREAM, at(STREAM, "no"), payoff))[1],
      lambda notes: "it runs on after the payoff" in notes)

# ---- carried on from a thought that began before the clip
carried = talk(["i bet everything and then it broke"])          # one sentence, no pause anywhere
then = at(carried, "then")
check("a clip that picks up mid-sentence is held below the floor",
      brain.standalone_score(carried[then]["s"], carried[-1]["e"], carried, np.zeros(40))[0], lambda v: v <= 34)
check("...and says why",
      brain.standalone_score(carried[then]["s"], carried[-1]["e"], carried, np.zeros(40))[1],
      lambda notes: any("carries on" in n for n in notes))
fresh = talk(["i bet everything"], pause=0.9) + talk(["and then it broke"], start=5.0)
and_, = [at(fresh, "and")]
check("the same word after a real pause is a new sentence, not a carried-on one",
      brain.standalone_score(fresh[and_]["s"], fresh[-1]["e"], fresh, np.zeros(40))[0], lambda v: v > 34)

# ---- dead air is not a clip
sparse = talk(["okay so that is the whole plan"], pause=0.0) + talk(["i think it will work"], start=14.0) + talk(["alright let us go"], start=28.0)
thin = brain.standalone_score(14.0, 31.0, sparse, np.linspace(-2, 0.4, 60))
check("a clip that is mostly silence with nothing happening is held back", thin[0], lambda v: v <= 34)
check("...and says why", thin[1], lambda notes: any("silence" in n for n in notes))

# ---- no sound information is not a fault
check("with no loudness at all, a normal clip is not punished",
      brain.standalone_score(STREAM[setup]["s"], STREAM[payoff]["e"], STREAM)[0], lambda v: v >= 55)

# ---- which moments are kept
def find(words, loudness, picks, **kw):
    """Run find_clips with the AI's answer replaced, without touching any network."""
    logs = []
    brain.ask_ai = lambda *a, **k: (picks, "model-x", 0)
    clips, info = brain.find_clips(words, loudness, kw.pop("duration", 60.0), "T", "U",
                                   max_clips=6, min_len=8, max_len=30, log=logs.append, **kw)
    return clips, info, logs


pick = [{"start": 14.0, "end": 30.0, "score": 85, "hook": "PLAN", "title": "t", "description": "d",
         "hashtags": [], "reason": "r"}]
clips, info, logs = find(sparse, np.linspace(-2, 0.4, 60), pick)
check("a moment that cannot stand on its own is not made into a clip", clips, [])
check("...the reason is written to the job log", logs, lambda ls: any("left out" in m for m in ls))
check("...and the reason is handed back to the job, not swallowed", info.get("dropped"),
      lambda d: bool(d) and isinstance(d[0], str) and d[0])

pick = [{"start": STREAM[setup]["s"], "end": STREAM[payoff]["e"], "score": 40, "hook": "ALL IN", "title": "t",
         "description": "d", "hashtags": [], "reason": "r"}]
clips, info, logs = find(STREAM, z, pick)
check("a moment that stands on its own is kept", len(clips), 1)
check("...with nothing left out", info.get("dropped"), [])
check("...and the shown score is the AI's and the check's together, not the AI's alone", clips[0]["score"], lambda v: 40 < v <= 100)

clips, info, logs = find(STREAM, np.zeros(80), pick)
check("with flat loudness (nothing to hear) a good moment is still kept", len(clips), 1)
check("...and not marked down for it", info.get("dropped"), [])

# ---- the rules the AI is given
seen = []
brain.ask_ai = lambda prompt, *a, **k: (seen.append(prompt) or ([], "model-x", 0))[1:]
brain.find_clips(STREAM, z, 60.0, "T", "U", max_clips=6, min_len=8, max_len=30, log=lambda m: None)
check("the rules ask for a clip that works in its first two seconds", "The first two seconds" in seen[0])
check("the rules name the payoff before the clip ends", "End on the payoff or the reaction to it" in seen[0])
check("the rules prefer a tight clip", "18 to 35 seconds is the sweet spot" in seen[0])
check("the rules put a floor under the score", "below 50 do not return it at all" in seen[0])
check("the rules forbid picking a moment that needs the rest of the stream", "without the rest of the stream" in seen[0])

# ---- what the card says about the ones left out
check("one moment left out is said plainly", jobs.left_out_note(["it starts in the middle of a sentence"]),
      lambda s: s.startswith("One moment was left out") and "make sense" in s and "middle of a sentence" in s)
check("several are counted, with the usual reason", jobs.left_out_note(["it is mostly silence with nothing happening"] * 3),
      lambda s: s.startswith("3 moments were left out") and "silence" in s and "make sense" in s)
check("nothing is said when nothing was left out", jobs.left_out_note([]), "")
check("blank reasons never make a note", jobs.left_out_note(["", None]), "")

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil; shutil.rmtree(os.environ["CLIPKONTRAST_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
