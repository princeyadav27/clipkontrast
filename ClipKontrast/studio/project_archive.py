"""Portable video projects. Only declared media and editing data go into an archive, never the data root."""
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from . import config, creators, ctx, jobs, load, media

VIDEO = r'(?:mp4|mkv|webm|mov|ts|flv|m4v)'
MEMBER = re.compile(rf'words(?:_en)?\.json|source\.{VIDEO}|clips/(?:thumb\.jpg|clip\d+\.(?:mp4|jpg|json|ass|src\.{VIDEO}))')
CLIP_FIELDS = 'id title hook description hashtags score reason start end src_offset src_end style layout captions highlight_words facecam fixed_for fixed_words status version made_at'.split()
JOB_FIELDS = 'title uploader duration width height created kind game lang options'.split()


class ProjectError(ValueError):
    """A safe explanation for the dashboard."""


def space(folder, need):
    if shutil.disk_usage(folder).free < need + load.MIN_FREE_DISK_GB * 1e9:
        raise ProjectError('Not enough disk space to finish safely. Free space and try again.')


def temporary():
    root = config.TMP_DIR / 'exports'
    root.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix='export-', suffix='.zip', dir=root)
    os.close(fd)
    return Path(path)


def fields(obj, names):
    return {key: obj[key] for key in names if key in obj}


def plan(jid, checkpoint=False):
    job = jobs.get(jid)
    if not job:
        raise ProjectError('Project not found.')
    root = Path(job['dir'])
    # job.json is atomically replaced by jobs.save. Do not serialize the mutable in-memory dict.
    j = json.loads((root / 'job.json').read_bytes())
    partial = unfinished(j)
    if partial:
        j['clips'] = [c for c in j['clips'] if c['status'] == 'done']
        if not checkpoint or not j['clips']:
            raise ProjectError('Wait for recording, processing and edits to finish before exporting this project.')
    info = fields(j, JOB_FIELDS)
    info.update(format=1, checkpoint=partial, original_id=jid, source=fields(j['source'], ('type', 'name', 'url')), clips=[])
    creator = creators.get(j.get('creator')) or j.get('creator_snapshot') or {}
    info['creator_settings'] = fields(creator, set(creators.DEFAULTS) - {'accounts'})
    files = {}

    def add(name, path, optional=False):
        if not MEMBER.fullmatch(name) or path.resolve().parent not in (root.resolve(), root.resolve() / 'clips') or path.is_symlink():
            raise ProjectError('Invalid project file.')
        if optional and not path.exists():
            return
        if not path.is_file() or not path.stat().st_size:
            raise ProjectError('A project file is missing or empty. No archive was completed.')
        files[name] = path

    for name in ('words.json', 'words_en.json', 'clips/thumb.jpg'):
        add(name, root / name, optional=True)
    source = j.get('source_path') or j['source'].get('path')
    if source:
        path = Path(source)
        info['source_file'] = 'source' + path.suffix.lower()
        add(info['source_file'], path)
    for c in j['clips']:
        cid = c['id']
        if not re.fullmatch(r'clip\d+', cid):
            raise ProjectError('Invalid clip identifier.')
        clip = fields(c, CLIP_FIELDS)
        if c.get('src'):
            src = Path(c['src'])
            clip['piece'] = f'clips/{cid}.src{src.suffix.lower()}'
            add(clip['piece'], src)
        if c['status'] == 'done':
            for ext in ('mp4', 'jpg', 'ass', 'json'):
                name = f'clips/{cid}.{ext}'
                add(name, root / name, optional=ext not in ('mp4',) and not (ext == 'jpg' and c.get('cover')))
        info['clips'].append(clip)
    info['files'] = {n: f.stat().st_size for n, f in files.items()}
    data = json.dumps(info, sort_keys=True, ensure_ascii=False).encode()
    stamp = hashlib.sha256(data)
    for name, path in sorted(files.items()):
        st = path.stat()
        stamp.update(f'{name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    return data, files, stamp.hexdigest()


def unfinished(j):
    return j['status'] not in ('done', 'error') or any(c['status'] not in ('done', 'error') for c in j['clips'])


def export(jid, checkpoint=False):
    data, files, rev = plan(jid, checkpoint)
    space(config.TMP_DIR, len(data) + sum(f.stat().st_size for f in files.values()) + 1024 * (len(files) + 1))
    out = temporary()
    try:
        with zipfile.ZipFile(out, 'w') as z:
            z.writestr('project.json', data)
            for name, path in files.items():
                z.write(path, name)
        with zipfile.ZipFile(out) as z:
            if z.testzip() or any(z.getinfo(n).file_size != p.stat().st_size for n, p in files.items()):
                raise ProjectError('Archive verification failed.')
        if plan(jid, checkpoint)[2] != rev:
            raise ProjectError('The project changed during export. Try again.')
        return out, rev
    except BaseException:
        out.unlink(missing_ok=True)
        raise


def restore(archive):
    """Stage under tmp, check every member, then publish a new job. Never overwrite an existing job."""
    user = config.user_dir(ctx.uid())
    parent = user / 'jobs'
    parent.mkdir(parents=True, exist_ok=True)
    jid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:10]
    dest = parent / jid
    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp, zipfile.ZipFile(archive) as z:
        stage = Path(tmp) / 'project'
        (stage / 'clips').mkdir(parents=True)
        names = set(z.namelist())
        if (len(names) != len(z.infolist()) or len(names) > 10000 or 'project.json' not in names
                or z.getinfo('project.json').file_size > 16 << 20 or any(not MEMBER.fullmatch(n) for n in names - {'project.json'})):
            raise ProjectError('Not a supported ClipKontrast project archive.')
        space(config.TMP_DIR, sum(f.file_size for f in z.infolist()))
        p = json.loads(z.read('project.json'))
        if p.get('format') != 1 or p.get('kind') not in ('video', 'live'):
            raise ProjectError('Unrecognized project format.')
        if p.get('files') != {n: z.getinfo(n).file_size for n in names - {'project.json'}}:
            raise ProjectError('The archive does not match its file inventory.')
        source = fields(p['source'], ('type', 'name', 'url'))
        if not isinstance(source.get('name', ''), str) or source.get('type') not in ('file', 'url', 'live'):
            raise ProjectError('Invalid project source.')
        if source.get('url') and not media.site_allowed(source['url']):
            raise ProjectError('Unsupported source URL.')
        job = fields(p, JOB_FIELDS)
        job.update(id=jid, dir=str(dest), user=ctx.uid(), creator=None, created=time.time(), source=source,
                   creator_snapshot=fields(p.get('creator_settings') or {}, set(creators.DEFAULTS) - {'accounts'}),
                   options={**jobs.DEFAULTS, **jobs.clean_options(p.get('options'))}, clips=[], ai={}, status='done',
                   stage='Restored project', error=None, progress=1, thumb='clips/thumb.jpg' in names)
        if p.get('source_file'):
            n = p['source_file']
            if not re.fullmatch(rf'source\.{VIDEO}', n) or n not in names:
                raise ProjectError('Missing original video.')
            job['source_path'] = str(dest / n)
            if source.get('type') == 'file':
                source['path'] = job['source_path']
        for key in ('width', 'height', 'duration'):
            if key in job and (type(job[key]) not in (int, float) or not math.isfinite(job[key]) or job[key] < 0):
                raise ProjectError('Invalid video dimensions or duration.')
        seen = set()
        for c in p['clips']:
            cid = c['id']
            if not re.fullmatch(r'clip\d+', cid) or cid in seen or c['status'] not in ('done', 'error'):
                raise ProjectError('Invalid clip in archive.')
            seen.add(cid)
            clip = fields(c, CLIP_FIELDS)
            for key in ('start', 'end', 'version', 'score', 'src_offset', 'src_end', 'made_at'):
                value = clip.get(key, 0)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ProjectError('Invalid clip timing or score.')
            if 'facecam' in clip and not (isinstance(clip['facecam'], list) and len(clip['facecam']) == 4
                    and all(type(n) in (int, float) and math.isfinite(n) for n in clip['facecam'])):
                raise ProjectError('Invalid camera coordinates.')
            if clip['status'] == 'done':
                if f'clips/{cid}.mp4' not in names or not z.getinfo(f'clips/{cid}.mp4').file_size:
                    raise ProjectError('Missing rendered clip.')
                clip['file'] = cid + '.mp4'
                if f'clips/{cid}.jpg' in names:
                    clip['cover'] = cid + '.jpg'
            if c.get('piece'):
                n = c['piece']
                if not re.fullmatch(rf'clips/{cid}\.src\.{VIDEO}', n) or n not in names:
                    raise ProjectError('Missing source segment.')
                clip['src'] = str(dest / n)
            job['clips'].append(clip)
        for name in names - {'project.json'}:
            with z.open(name) as src, (stage / name).open('xb') as out:
                shutil.copyfileobj(src, out, 1 << 20)  # ZipExtFile verifies CRC on reaching EOF
            if (stage / name).stat().st_size != z.getinfo(name).file_size:
                raise ProjectError('Incomplete project file.')
        (stage / 'job.json').write_text(json.dumps(job), encoding='utf-8')
        with jobs._lock:
            stage.rename(dest)
            jobs._load_file(dest / 'job.json', ctx.uid())
        ctx.bump()
    return jid
