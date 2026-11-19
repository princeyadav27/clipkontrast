"""Host the private, full ClipKontrast app on Kaggle. Called by ClipKontrast.ipynb."""
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/kaggle/working/clipkontrast-data')
OWNER = 'owner@notebook.invalid'


def provision(password):
    """Create the owner before serving anything. Never adopt another database or reset a password."""
    from studio import admin, auth, config, db
    if not config.ACCOUNTS or config.PUBLIC_SIGNUP or not config.HTTPS:
        raise RuntimeError('Notebook mode requires HTTPS, sign-in, and closed registration.')
    db.init()
    with db.engine.connect() as conn:
        emails = list(conn.execute(db.sa.select(db.users.c.email)).scalars())
    if emails == [OWNER]:
        if not auth.login(OWNER, password).is_admin:
            raise ValueError('The notebook owner is not an administrator.')
    elif emails:
        raise ValueError('Not an owner-only notebook database. Use a separate data folder.')
    else:
        user, _ = auth.signup(OWNER, password, 'Notebook owner')
        auth.mark_verified(user.id)
        admin.set_admin(OWNER, True)


def serve(fd):
    # This child runs in the isolated environment; the notebook kernel is never upgraded.
    sys.path.insert(0, str(ROOT))
    import uvicorn
    from studio import transcribe, server
    provision(json.load(sys.stdin))
    print('Loading the speech model on the notebook GPU...', flush=True)
    transcribe._recognizer()  # fail here rather than handing out a URL to a broken installation
    with socket.socket(fileno=fd) as sock:
        uvicorn.Server(uvicorn.Config(server.app, proxy_headers=False, access_log=False)).run(sockets=[sock])


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def start(password, backup_env=None):
    """Start from a notebook cell. Interrupt the cell to shut down both children."""
    from deploy.kaggle import worker
    if not isinstance(password, str) or not 16 <= len(password) <= 200:
        raise ValueError('Set a private notebook password of 16 to 200 characters in Kaggle Secrets.')
    if not DATA.parent.is_dir():
        raise RuntimeError('Run this in a Kaggle notebook with Internet and the GPU accelerator enabled.')
    for name in ('ffmpeg', 'ffprobe', 'nvidia-smi', 'curl'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} is missing. Enable GPU T4 and install ffmpeg in the notebook.')
    subprocess.run(['nvidia-smi', '-L'], check=True)
    backup_env = dict(backup_env or {})
    allowed = {'CLIPKONTRAST_BACKUP_BUCKET', 'CLIPKONTRAST_BACKUP_PREFIX', 'CLIPKONTRAST_BACKUP_AUTO',
               'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_DEFAULT_REGION'}
    if backup_env.keys() - allowed:
        raise ValueError('Unknown backup setting.')
    DATA.mkdir(mode=0o700, exist_ok=True)
    (DATA / 'tmp').mkdir(exist_ok=True)
    (DATA / 'imports').mkdir(exist_ok=True)
    # Keep program downloads outside notebook outputs. Only the data survives a cell restart.
    with tempfile.TemporaryDirectory(prefix='clipkontrast-runtime-') as runtime, socket.socket() as sock:
        runtime = Path(runtime)
        subprocess.run([sys.executable, '-m', 'venv', str(runtime / 'env')], check=True)
        python = str(runtime / 'env/bin/python')
        deps = [s for s in (ROOT / 'requirements.txt').read_text().splitlines() if s and not s.startswith('sherpa-onnx==')]
        if backup_env.get('CLIPKONTRAST_BACKUP_BUCKET') or os.getenv('CLIPKONTRAST_BACKUP_BUCKET'):
            deps.append('boto3>=1.35,<2')
        subprocess.run([python, '-m', 'pip', 'install', *deps, worker.CUDA_WHEEL, '-f', worker.CUDA_INDEX], check=True)
        # Reuse CUDA/cuDNN from the Kaggle image; do not uninstall torch or modify its Python environment.
        libs = ':'.join(filter(None, (worker.library_path(), os.environ.get('LD_LIBRARY_PATH'))))
        sock.bind(('127.0.0.1', 0))  # reserve this port until the app inherits the socket
        port = sock.getsockname()[1]
        addresses = queue.Queue()
        tunnel = worker.tunnel_port(port, binary=runtime / 'cloudflared', found_url=addresses.put)
        app = None
        try:
            deadline = time.monotonic() + 120
            while tunnel.poll() is None and time.monotonic() < deadline:
                try:
                    url = addresses.get(timeout=1)
                    break
                except queue.Empty:
                    pass
            else:
                raise RuntimeError('No tunnel URL. Check notebook Internet access and restart the cell.')
            env = dict(os.environ, **backup_env, CLIPKONTRAST_NOTEBOOK='1', TMPDIR=str(DATA / 'tmp'), CLIPKONTRAST_PUBLIC_URL=url, CLIPKONTRAST_PORT=str(port),
                       CLIPKONTRAST_HOST='127.0.0.1', CLIPKONTRAST_ACCOUNTS='1', CLIPKONTRAST_PUBLIC_SIGNUP='0',
                       CLIPKONTRAST_TRUST_PROXY='', CLIPKONTRAST_DATA=str(DATA), DATABASE_URL='',
                       CLIPKONTRAST_MODELS=str(DATA / 'models'), CLIPKONTRAST_ASR_URL='', CLIPKONTRAST_ASR_PROVIDER='cuda',
                       CLIPKONTRAST_ASR_MODEL='sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16', CLIPKONTRAST_ASR_COPIES='1',
                       CLIPKONTRAST_FETCH_MODEL='0', CLIPKONTRAST_MAX_UPLOAD_MB='1024', CLIPKONTRAST_MAX_USER_GB='10',
                       CLIPKONTRAST_MAX_VIDEO_HOURS='3', CLIPKONTRAST_JOB_WORKERS='1', LD_LIBRARY_PATH=libs)
            app = subprocess.Popen([python, __file__, str(sock.fileno())], cwd=ROOT, env=env, pass_fds=(sock.fileno(),),
                                   stdin=subprocess.PIPE, text=True)
            with app.stdin:
                json.dump(password, app.stdin)
            sock.close()
            deadline = time.monotonic() + 900
            while app.poll() is None and tunnel.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if json.load(response).get('ok'):
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('App did not become ready. Check GPU, model download, free disk and tunnel logs.')
            print(f'Open {url}/app and sign in as {OWNER}. Keep this cell running.', flush=True)
            while app.poll() is None and tunnel.poll() is None:
                time.sleep(1)
            raise RuntimeError('The server or tunnel stopped. Restart the cell to obtain a new link.')
        except KeyboardInterrupt:
            print('Stopping ClipKontrast. Files without a verified backup or download can be lost when this Kaggle session ends.', flush=True)
            raise
        finally:
            stop(tunnel)
            stop(app)


if __name__ == '__main__':
    serve(int(sys.argv[1]))
