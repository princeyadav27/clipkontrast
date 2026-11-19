"""Portable project snapshots and S3 readback, without network access or notebook credentials."""
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, '.')
root = tempfile.TemporaryDirectory()
os.environ.update(CLIPKONTRAST_DATA=root.name, DATABASE_URL='', CLIPKONTRAST_NOTEBOOK='1',
                  CLIPKONTRAST_BACKUP_BUCKET='private-test', CLIPKONTRAST_ACCOUNTS='1', CLIPKONTRAST_PUBLIC_SIGNUP='0')
from studio import account, backups, ctx, jobs, project_archive as archive


class S3:
    def __init__(self): self.objects = {}
    def upload_file(self, path, bucket, key, Callback=None): self.objects[key] = Path(path).read_bytes()
    def get_object(self, Bucket, Key): return {'Body': io.BytesIO(self.objects[Key])}
    def put_object(self, Bucket, Key, Body): self.objects[Key] = Body
    def head_object(self, Bucket, Key): return {'ContentLength': len(self.objects[Key])}


class Projects(unittest.TestCase):
    def setUp(self):
        ctx.set_uid('u_first_owner')
        self.job = j = jobs.create({'type': 'file', 'name': 'hello.mp4'}, {}, enqueue=False)
        self.d = d = Path(j['dir'])
        (d / 'clips').mkdir()
        (d / 'source.mp4').write_bytes(b'original')
        (d / 'words.json').write_text('[]')
        (d / 'clips/clip01.mp4').write_bytes(b'finished')
        (d / 'clips/clip01.src.ts').write_bytes(b'source piece')
        (d / 'ai.json').write_text('private!')  # even unexpected files in a job stay out
        (d / 'audio.wav').write_bytes(b'scratch')
        j.update(status='done', source_path=str(d/'source.mp4'), clips=[dict(id='clip01', status='done', version=1,
                 src=str(d/'clips/clip01.src.ts'), file='clip01.mp4', start=1, end=6, score=74)])
        jobs.save(j)
        self.id = j['id']

    def test_roundtrip(self):
        path, _ = archive.export(self.id)
        self.addCleanup(path.unlink)
        with zipfile.ZipFile(path) as z:
            self.assertEqual(set(z.namelist()), {'project.json','source.mp4','words.json','clips/clip01.mp4','clips/clip01.src.ts'})
            self.assertNotIn(root.name, z.read('project.json').decode())
        with ctx.as_user('u_fresh_owner'):
            with self.assertRaises(ValueError): archive.export(self.id)
            new = jobs.get(archive.restore(path))
            self.assertNotEqual(new['id'], self.id)
            self.assertEqual(Path(new['source_path']).read_bytes(), b'original')
            self.assertEqual(Path(new['clips'][0]['src']).read_bytes(), b'source piece')
            self.assertEqual(new['options'], self.job['options'])
            self.assertEqual((Path(new['dir'])/'clips/clip01.mp4').read_bytes(), b'finished')

    def test_backup_receipt_requires_verified_remote(self):
        s3 = S3()
        with patch.object(s3, 'get_object', return_value={'Body':io.BytesIO(b'incomplete')}):
            with self.assertRaises(ValueError): backups.upload(s3, self.id)
        self.assertFalse(backups.receipt(self.id).exists())
        self.assertFalse(any('/verified/' in k for k in s3.objects))
        backups.upload(s3, self.id)
        ident = json.loads(backups.receipt(self.id).read_bytes())['snapshot']
        self.assertTrue(backups.verified(self.id, archive.plan(self.id)[2]))
        self.job['clips'][0]['title'] = 'Changed edit'
        jobs.save(self.job)
        self.assertFalse(backups.verified(self.id, archive.plan(self.id)[2]))
        with ctx.as_user('u_new_notebook'):
            before = len(jobs.list_jobs())
            backups.download(s3, ident)
            self.assertEqual(len(jobs.list_jobs()), before + 1)
            key = next(k for k in s3.objects if '/projects/' in k)
            s3.objects[key] = s3.objects[key][:-1]
            with self.assertRaises(ValueError): backups.download(s3, ident)
            self.assertEqual(len(jobs.list_jobs()), before + 1)

    def test_checkpoint(self):
        self.job.update(status='live'); jobs.save(self.job)
        with self.assertRaises(ValueError): archive.export(self.id)
        path, _ = archive.export(self.id, checkpoint=True)
        self.addCleanup(path.unlink)
        with zipfile.ZipFile(path) as z: self.assertTrue(json.loads(z.read('project.json'))['checkpoint'])

    def test_incomplete_and_disk_full(self):
        with patch.object(archive.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, 'disk space'): archive.export(self.id)
        (self.d/'clips/clip01.mp4').unlink()
        with self.assertRaises(ValueError): archive.export(self.id)
        with self.assertRaises(ValueError): account.zip_clips([(self.id, 'clip01')])
        self.assertTrue((self.d/'source.mp4').exists())

    def test_reject_unsafe_or_incomplete_archive(self):
        path, _ = archive.export(self.id)
        self.addCleanup(path.unlink)
        with zipfile.ZipFile(path) as z: parts = {n:z.read(n) for n in z.namelist()}
        before = len(jobs.list_jobs())
        for changed in (dict(parts, **{'../outside': b'x'}), {k:v for k,v in parts.items() if k != 'words.json'}):
            with zipfile.ZipFile(path, 'w') as z:
                for name, data in changed.items(): z.writestr(name, data)
            with self.assertRaises(ValueError): archive.restore(path)
            self.assertEqual(len(jobs.list_jobs()), before)

if __name__ == '__main__': unittest.main()
