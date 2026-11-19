# A free graphics card for transcription: Kaggle notebooks

A Kaggle notebook with a **Tesla T4** can do the speech-to-text for ClipKontrast many times faster than a laptop,
for nothing. The notebook is not a server: it has no address of its own, it stops after a few hours, and it has a
weekly GPU quota. So ClipKontrast treats it as a helper that comes and goes — when it is there, transcription runs
on the card; when it is not, the same work happens on your own processor.

Nothing on your computer changes: it keeps the videos, the clips and the dashboard.

## What you need

- A Kaggle account (free) with **phone verification**, which is what unlocks the GPU.
- **Internet turned on** for the notebook (the *Settings* panel on the right: Internet → On). Without it the model
  cannot be downloaded and the tunnel cannot be opened.
- Accelerator set to **GPU T4 x2 or GPU T4** (the worker uses one of them).

## One-time setup

1. Go to **kaggle.com → Create → Notebook**, then on the right: **Settings → Accelerator → GPU T4**, and
   **Internet → On**.
2. Add a **secret** with the name `CLIPKONTRAST_ASR_TOKEN` and a long random value (for example the output of
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`). **Add-ons → Secrets → Add a new secret.**
   Paste the same value into your own `.env` as `CLIPKONTRAST_ASR_TOKEN`. This is the only thing protecting the
   notebook, so do not skip it and do not reuse a password.
3. Get the code into the notebook. Either:
   - upload this project as a **Kaggle dataset** and add it to the notebook, then in a cell:
     `!cp -r /kaggle/input/clipkontrast/* /kaggle/working/` and `%cd /kaggle/working`, or
   - in a cell: `!git clone <your copy of this project> /kaggle/working/clipkontrast`.

## Every session

1. Start the notebook and run one cell:

   ```
   %cd /kaggle/working/clipkontrast
   !python deploy/kaggle/worker.py
   ```

2. It installs the graphics-card speech library (a minute or two, once per session), prints what card it found,
   downloads the full-precision model the first time (about 1.3 GB), and then prints something like:

   ```
   Speech machine address (put this in CLIPKONTRAST_ASR_URL on your computer):
       https://three-words-here.trycloudflare.com
   ```

3. Copy that address into your computer's `.env` as `CLIPKONTRAST_ASR_URL`, keep the same token, restart
   ClipKontrast, then open **Settings → AI → Speech machine** and press **Test connection**. It should answer with
   the card's name, its free memory and the round-trip time.

That address is different every session, so it has to be pasted again each time. It is public (which is why the
token is required on every route), and it stops working when the notebook stops.

`worker.py` takes three flags: `--no-install` (the library is already there), `--no-tunnel` (a machine on your own
network instead of a tunnel) and `--port`.

## What to expect

- **Speed:** a T4 transcribes roughly 30–60× faster than real time with this model, against about 5–10× on a
  laptop's processor. A two-hour stream that takes 20 minutes at home takes under a minute on the card.
- **Session limits:** this is a free service. Sessions end after a while (up to about 9–12 hours, and Kaggle may
  stop an idle notebook), and the GPU has a weekly quota (about 30 hours). Plan on starting a session when you have
  a batch of videos to process.
- **Your work is on your computer.** Videos, transcripts, clips and the database stay where ClipKontrast runs; the
  notebook only turns audio into words. If the notebook dies mid-video, ClipKontrast notices, says so, and finishes
  that video on your own processor — a bit of transcription may be repeated, nothing is lost.
- **Nothing is uploaded but audio.** The worker receives 12–24 second pieces of sound and returns words. It holds
  no files and has no database.

## When something is wrong

| What you see | What it means |
|---|---|
| Settings says **Not answering · Could not reach the speech machine** | The notebook stopped, or its address changed. Start it again and paste the new address. |
| Settings says **The speech machine refused the shared secret** | `CLIPKONTRAST_ASR_TOKEN` differs between the two machines. Make them the same and restart both. |
| `No NVIDIA graphics card was found (nvidia-smi answered nothing)` | The accelerator is not set to GPU (Settings → Accelerator). Change it and restart the session. |
| `The graphics card has only N MB free` | Something else on the notebook is using the card. Restart the session, or set `CLIPKONTRAST_ASR_MIN_VRAM_MB` lower (3000 is usually enough). |
| `The graphics card ran out of memory` | Set `CLIPKONTRAST_ASR_PARALLEL=1` and `CLIPKONTRAST_ASR_BATCH=8` in the notebook and start it again. |
| The worker never prints an address | Internet is off for the notebook, or the download of the tunnel program failed. Check Settings → Internet, then run the cell again. |
| `The graphics-card build could not be installed` | Kaggle changed its Python version. The worker still starts and says it is not ready; your own processor does the work meanwhile. |
| Everything is slow | Transcription is on your own processor because the worker is not being used. Check the Speech machine line in Settings. |

## Limits worth knowing

- Kaggle's GPU is **not permanent** and the tunnel address is not stable. Nothing in ClipKontrast assumes otherwise.
- **Encoding clips on the notebook is not used.** Renders happen on your computer. The worker reports whether its
  ffmpeg has `h264_nvenc`, so you can see whether that machine could encode — ClipKontrast never depends on it.
- Only the packages the worker needs are installed. The notebook's own CUDA and torch are left alone, and
  `worker.py` touches nothing outside `/kaggle/working` and pip.
- The notebook runs the newest **graphics-card build** of the speech library (`sherpa-onnx 1.12.21`, CUDA 12, cuDNN
  9) because the project publishes no CUDA build of 2.0; the app on your computer keeps its own version. Only words
  and timestamps cross the wire, so the two do not have to match.
- Keep the notebook's secret out of screenshots and shared notebook outputs. `worker.py` never prints it.
