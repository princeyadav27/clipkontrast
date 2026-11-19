"""Optional private S3 backup for the single-owner notebook. One transfer at a time, off the request thread."""
import hashlib
import json
import os
import re
import threading
import time
from contextlib import closing

from . import auth, config, ctx, jobs, notices, project_archive as archive

NOTEBOOK = os.getenv('CLIPKONTRAST_NOTEBOOK') == '1'
BUCKET = os.getenv('CLIPKONTRAST_BACKUP_BUCKET', '').strip()
PREFIX = os.getenv('CLIPKONTRAST_BACKUP_PREFIX', 'clipkontrast').strip('/') + '/'
AUTO = os.getenv('CLIPKONTRAST_BACKUP_AUTO') == '1'
_ID = re.compile(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{10}-[a-f0-9]{64}')
_slot = threading.Lock()
_state = {'phase': 'Idle', 'busy': False}
_retry = {}


def enabled():
    return bool(NOTEBOOK and BUCKET and config.ACCOUNTS and not config.PUBLIC_SIGNUP)


def client():
    import boto3  # installed only when the notebook owner configures S3
    return boto3.client('s3')


def phase(text):
    global _state
    _state = {'phase': text, 'busy': True}


def receipt(jid):
    # A receipt for another bucket or workspace must not turn this project green.
    where = hashlib.sha256((BUCKET + '/' + PREFIX).encode()).hexdigest()[:20]
    return config.user_dir(ctx.uid()) / 'backup-receipts' / where / (jid + '.json')


def verified(jid, rev):
    try:
        return json.loads(receipt(jid).read_text())['revision'] == rev
    except (OSError, ValueError, KeyError):
        return False


def remember(jid, rev, ident, versions=None):
    path = receipt(jid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(revision=rev, snapshot=ident, checked=time.time(), versions=versions)))
    tmp.replace(path)


def versions(job):
    return [[c['id'], c.get('version', 0)] for c in job['clips'] if c['status'] == 'done']


def checkpoint_matches(job):
    try:
        return json.loads(receipt(job['id']).read_bytes()).get('versions') == versions(job)
    except (ValueError, OSError):
        return False


def status():
    result = []
    for job in jobs.list_jobs():
        try:
            rev = archive.plan(job['id'])[2]
            label = 'Verified in S3' if enabled() and verified(job['id'], rev) else 'Local only'
        except (ValueError, OSError):
            label = ('Finished clips verified; project still changing' if enabled() and archive.unfinished(job)
                     and versions(job) and checkpoint_matches(job) else 'Not ready for export')
        result.append({'id': job['id'], 'title': job['title'], 'label': label})
    return dict(enabled=enabled(), automatic=enabled() and AUTO, imports=imports(), projects=result, **_state)


def catalogue():
    rows = []
    for page in client().get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX + 'verified/'):
        for obj in page.get('Contents', []):
            ident = obj['Key'].removeprefix(PREFIX + 'verified/')
            if _ID.fullmatch(ident):
                rows.append({'id': ident, 'date': obj['LastModified'].isoformat()})
    return sorted(rows, key=lambda row: row['date'], reverse=True)


def digest(stream, output=None, progress=None):
    h, count = hashlib.sha256(), 0
    while chunk := stream.read(1024 * 1024):
        count += len(chunk)
        h.update(chunk)
        if output is not None:
            output.write(chunk)
        if progress:
            progress(count)
    return h.hexdigest(), count


def upload(s3, jid, checkpoint=False):
    phase('Preparing and checking project ZIP')
    rendered = versions(jobs.get(jid))
    path, rev = archive.export(jid, checkpoint)
    try:
        with path.open('rb') as f:
            sha, size = digest(f)
        ident = jid + '-' + sha
        key = PREFIX + 'projects/' + ident + '.zip'
        phase('Uploading verified archive to S3')
        transferred = 0
        lock = threading.Lock()

        def progress(n):
            nonlocal transferred
            with lock:
                transferred += n
                phase(f'Uploading to S3: {min(100, 100*transferred//size)}%')

        s3.upload_file(str(path), BUCKET, key, Callback=progress)
        phase('Reading back S3 copy for verification')
        with closing(s3.get_object(Bucket=BUCKET, Key=key)['Body']) as remote:
            got = digest(remote, progress=lambda n: phase(f'Verifying S3 copy: {100*n//size}%'))
        if got != (sha, size):
            raise archive.ProjectError('Remote copy did not match. No backup was confirmed.')
        # A partial/interrupted upload is never listed as a restorable snapshot.
        s3.put_object(Bucket=BUCKET, Key=PREFIX + 'verified/' + ident, Body=b'1')
        remember(jid, rev, ident, rendered)
    finally:
        path.unlink(missing_ok=True)


def download(s3, ident):
    if not _ID.fullmatch(ident):
        raise archive.ProjectError('Invalid snapshot.')
    s3.head_object(Bucket=BUCKET, Key=PREFIX + 'verified/' + ident)
    key = PREFIX + 'projects/' + ident + '.zip'
    size = s3.head_object(Bucket=BUCKET, Key=key)['ContentLength']
    if size <= 0:
        raise archive.ProjectError('The stored project is empty.')
    archive.space(config.TMP_DIR, size)
    path = archive.temporary()
    try:
        with closing(s3.get_object(Bucket=BUCKET, Key=key)['Body']) as remote, path.open('wb') as out:
            sha, count = digest(remote, out, lambda n: phase(f'Downloading and checking: {100*n//size}%'))
        if count != size or sha != ident.rsplit('-', 1)[1]:
            raise archive.ProjectError('Backup verification failed. Nothing was restored.')
        phase('Restoring verified files into a new project')
        jid = archive.restore(path)
        remember(jid, archive.plan(jid)[2], ident)
    finally:
        path.unlink(missing_ok=True)


def imports():
    root = config.DATA_DIR / 'imports'
    return sorted(f.name for f in root.glob('*.zip') if f.is_file() and not f.is_symlink())


def submit(action, value, automatic=False):
    if not NOTEBOOK or not auth.get_user(ctx.uid()).is_admin or (action != 'import' and not enabled()):
        raise archive.ProjectError('S3 notebook backup is not enabled.')
    if action == 'backup' and value != 'all' and not jobs.get(value):
        raise archive.ProjectError('Project not found.')
    if action == 'import' and value not in imports():
        raise archive.ProjectError('Choose an archive from the notebook imports folder.')
    if action not in ('backup', 'restore', 'import') or (action == 'restore' and not _ID.fullmatch(value)):
        raise archive.ProjectError('Unknown backup action or snapshot.')
    if not _slot.acquire(blocking=False):
        raise archive.ProjectError('A transfer is already running.')
    phase('Starting transfer')

    def run():
        global _state
        message = 'Transfer interrupted before verification'
        try:
            s3 = client() if action != 'import' else None
            if action == 'import':
                phase('Checking and restoring local project archive')
                archive.restore(config.DATA_DIR / 'imports' / value)
                message = 'Project archive restored locally'
            elif action == 'restore':
                download(s3, value)
                message = 'Verified project restored'
            else:
                ids = [value] if value != 'all' else [j['id'] for j in jobs.list_jobs()]
                if not ids:
                    raise archive.ProjectError('No projects to back up.')
                for jid in ids:
                    upload(s3, jid, checkpoint=automatic)
                message = f'{len(ids)} project backup(s) verified in S3'
        except Exception as exc:
            # Do not send cloud SDK messages (possibly signed requests) to the browser or notebook log.
            message = str(exc) if isinstance(exc, archive.ProjectError) else 'Transfer failed. Check disk space, S3 access, network and archive integrity.'
            _retry[value] = time.monotonic() + 300
        finally:
            _state = {'phase': message, 'busy': False}
            try:
                notices.add(ctx.uid(), 'backup', message, link='/app/settings/storage')
            finally:
                _slot.release()
    threading.Thread(target=ctx.bind(run), daemon=True, name='project-transfer').start()


def start():
    if not enabled() or not AUTO:
        return

    def scan():
        while True:
            time.sleep(15)
            if _slot.locked():
                continue
            for j in jobs.all_jobs():
                if not versions(j) or _retry.get(j['id'], 0) > time.monotonic():
                    continue
                with ctx.as_user(j['user']):
                    try:
                        already_saved = (checkpoint_matches(j) if archive.unfinished(j) else
                                         verified(j['id'], archive.plan(j['id'])[2]))
                    except (ValueError, OSError):
                        already_saved = False
                    if not already_saved:
                        try:
                            submit('backup', j['id'], automatic=True)
                        except ValueError:
                            pass  # a manual transfer won the slot
                        break
    threading.Thread(target=scan, daemon=True, name='automatic-backup').start()
