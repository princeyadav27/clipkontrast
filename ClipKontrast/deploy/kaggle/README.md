# Run ClipKontrast entirely on Kaggle

Use [ClipKontrast.ipynb](ClipKontrast.ipynb) for **browser → HTTPS tunnel → the entire app on Kaggle**. No local server, ASR token, second notebook, or inbound Kaggle port is required. The notebook runs the existing dashboard, API, job queue, speech model, FFmpeg renderer and SQLite database together. The older [speech worker](#a-free-graphics-card-for-transcription-kaggle-notebooks) below is a different mode for people running the app on their PC.

1. Import `ClipKontrast.ipynb` into a **private** Kaggle notebook. Select **GPU T4** and enable **Internet**. Attach this version of the project as a **private code-only dataset**: include `studio/`, `web/`, `requirements.txt` and `deploy/kaggle/`; do not include `.env`, `data/`, virtual environments or credentials. Set `PROJECT` in the cell to the mounted directory containing `requirements.txt`. A trusted checkout of this version under `/kaggle/working` works too.
2. In **Add-ons → Secrets**, add `CLIPKONTRAST_NOTEBOOK_PASSWORD` (a unique random password of 16–200 characters) and grant this notebook access. Never type the value into a cell. No AI key is required to try clipping.
3. Run the cell. It installs dependencies outside notebook outputs, checks for FFmpeg/ffprobe, loads the GPU speech model, and prints **one HTTPS `/app` link** when the backend is ready. Sign in with `owner@notebook.invalid` and your notebook password. Public registration and Google sign-in are disabled; knowing the link alone does not grant access.
4. **Upload → Make clips → open a clip → edit/re-render → download**. Files and job state are under `/kaggle/working/clipkontrast-data/`; finished MP4s are in the user's job `clips/` folders. Video rendering uses the app's encoder selection (including CPU fallback); speech uses CUDA locally in the notebook. Optional AI highlight selection contacts the provider you configure in Settings; without it, selection uses loudness.
5. Download your clips before stopping the session. Interrupt the cell to stop the server and tunnel. A new run gets a new URL; restarting the cell in the same session reuses its database. **`/kaggle/working` is session storage, not a permanent drive.** To keep exports across sessions, save the finished MP4s privately. Do **not** publish the whole working directory: it can contain source videos, transcripts, password hashes, AI keys and connected-account tokens. If you change your password in the dashboard, update the Kaggle secret before relaunching.

## Downloads, backups and recovery

Kaggle does **not** promise advance notice of termination. The dashboard keeps a session-loss warning visible from the start. Download or verify a backup while the cell is still running, not when the GPU has already disconnected.

- **One MP4:** use Download on its clip card. **Selected clips:** check the clips and use Download. **All finished clips:** Settings → Storage → Download all finished clips. These ZIPs contain the rendered videos, not the original footage.
- **One editable project:** choose **Export project** on a finished video's card. The ZIP contains `project.json` (format version, file inventory, clip timings, titles, caption/layout options and creator settings), available source video or saved source segments, transcript, rendered clips, covers and caption sidecars. A project means one video/recording. This is not a backup of logins, posting accounts, API keys, the database, speech models, transient live buffers or unfinished work.
- **Where things live:** active projects live under `clipkontrast-data/users/`; upload parts, renderer scratch space and outgoing ZIPs use `clipkontrast-data/tmp/`. Put your own project ZIPs to restore under `clipkontrast-data/imports/`. Cleanup removes application staging files, not clips, source media or imports. A downloaded file is **not** labeled as a verified cloud backup.
- **Restore without a cloud account:** attach a trusted project ZIP as a private Kaggle dataset. Before running the launcher, copy the ZIP into `/kaggle/working/clipkontrast-data/imports/` (create the directory if needed). In Settings → Storage choose **Restore local ZIP**. Keep the ZIP until you have checked the restored project. Restored projects appear in Videos with new IDs; any original creator settings are retained in project metadata, not activated as watch/post automation.

### Optional persistent storage: private Amazon S3 bucket

S3 is independent of the notebook session and is **off by default**. AWS storage, requests and readback can incur charges. Use a private bucket with public access blocked. Give this notebook a dedicated prefix; anyone with access to that prefix can read its video content. Scope its AWS credentials to list that prefix and get/put objects there (including multipart uploads and aborting incomplete multipart uploads). Do not grant public access or bucket-wide administrative permissions.

1. Add/grant `CLIPKONTRAST_BACKUP_BUCKET`, `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in Kaggle Secrets. For temporary credentials also pass `AWS_SESSION_TOKEN`. Set **`USE_S3 = True`** in the launch cell. Set its prefix to your private workspace name and use the **same bucket and prefix on subsequent sessions**. Alternatively, supply these variables along with `CLIPKONTRAST_BACKUP_PREFIX` and `CLIPKONTRAST_BACKUP_AUTO` via the environment before launch. Do not paste credentials into cell text or a tracked `.env`.
2. Set `CLIPKONTRAST_BACKUP_AUTO` to `1` for automatic backup, or `0` for manual only. The scheduler checks every 15 seconds. It checkpoints finished clips while a job is active and backs up the completed project, including later edits. Transfers are serialized; temporary failures retry after five minutes. **Wait for verification** before ending the session. A live checkpoint restores finished clips only; it does not resume the live recording.
3. In **Settings → Storage**, back up one complete project or all projects manually. Watch the transfer phase and verification percentage there; successes and failures also appear in the notification bell. A receipt refers to the version last read back from S3, not a promise against later deletion in AWS. New local edits are not marked as backed up.
4. In a fresh notebook using that prefix, choose **Find S3 snapshots → Restore as new project**. Only committed snapshots are listed. The server validates the downloaded size and SHA-256, checks the ZIP inventory and available disk space, then publishes the restored files together. A failed restore does not replace an existing project. Restoring needs space for the ZIP **and** the unpacked project, plus the app's disk reserve.

Older snapshots are retained, not pruned by ClipKontrast. Manage retention and incomplete multipart uploads in S3; if you remove a snapshot, remove its matching `verified/` marker as well. Treat snapshots as private user content.

**Troubleshooting:** No GPU or CUDA error? Check the accelerator and Kaggle's GPU quota. The launcher requires a compatible Linux Python/CUDA 12/cuDNN 9 image. If `ffmpeg` or `ffprobe` is missing, install the notebook OS's `ffmpeg` package before retrying. First launch needs room for dependencies, the ~1.3 GB model, uploads and rendered clips. If a tunnel or model download fails, check Internet access and restart the cell. This mode is for an interactive session, not permanent hosting or stable OAuth callback URLs. Use it only where allowed by Kaggle and the tunnel provider.

---

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
   %cd /kaggle/working/clipkontrast/ClipKontrast
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
