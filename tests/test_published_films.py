"""Film packages remain usable without any backend run, cache, or job history."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from volleymole.web import films
from volleymole.web.server import Workspace
from volleymole.web.progress import summary
from web_fixtures import film_fixture


class ShelfTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app = Workspace.__new__(Workspace)
        self.app.root = self.root

    def test_reading_shelf_does_not_scan_runs_or_consult_job_history(self):
        film_id = film_fixture(self.root)
        original = os.walk
        def walk(path, **kwargs):
            self.assertTrue(Path(path).is_relative_to(self.root/'data'))
            return original(path, **kwargs)
        with patch('volleymole.web.library.os.walk', side_effect=walk):
            self.assertEqual(self.app.library()['outputs'][0]['film_id'], film_id)

    def test_incomplete_changed_and_malicious_packages_are_not_shown(self):
        for change in ('missing_poster', 'changed_video', 'wrong_version', 'escape', 'symlink'):
            with self.subTest(change=change):
                shutil.rmtree(self.root/'outputs', ignore_errors=True)
                film_id = film_fixture(self.root)
                base = films.folder(self.app, film_id)
                manifest = base/'manifest.json'
                data = json.loads(manifest.read_text())
                if change == 'missing_poster': (base/'poster.jpg').unlink()
                if change == 'changed_video': (base/'video.mp4').write_bytes(b'replacement')
                if change == 'wrong_version': data['schema_version'] = 2
                if change == 'escape': data['video']['file'] = '../../../runs/private.mp4'
                if change == 'symlink':
                    video = base/'video.mp4'
                    video.rename(self.root/'other.mp4')
                    video.symlink_to(self.root/'other.mp4')
                manifest.write_text(json.dumps(data))
                self.assertEqual(self.app.library()['outputs'], [])

    def test_details_only_return_public_fields(self):
        film_id = film_fixture(self.root)
        path = films.folder(self.app, film_id)/'manifest.json'
        data = json.loads(path.read_text())
        data.update(run='/private/backend', api_key='never-return-this')
        data['highlights'][0]['source_path'] = '/private/source'
        path.write_text(json.dumps(data))
        public = json.dumps(films.detail(self.app, film_id))
        self.assertNotIn('/private', public)
        self.assertNotIn('never-return-this', public)

    def test_copying_a_complete_package_to_another_workspace_is_supported(self):
        film_id=film_fixture(self.root)
        original=films.folder(self.app,film_id)
        other=self.root/'another-workspace'
        shutil.copytree(original,other/films.SHELF/film_id,copy_function=shutil.copy)
        app=Workspace.__new__(Workspace);app.root=other
        self.assertEqual(films.entries(app)[0]['film_id'],film_id)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg and FFprobe')
class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.temp.name)/'match.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'testsrc2=size=160x240:rate=30',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '1',
                        '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                        str(cls.fixture)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app = Workspace.__new__(Workspace)
        self.app.root = self.root
        self.run = self.root/'runs/game'
        self.run.mkdir(parents=True)
        self.video = self.run/'final.mp4'
        shutil.copy2(self.fixture, self.video)
        self.write('render_report_lively.json', {'output': str(self.video)})
        self.write('edit_decision.json', {'title': '周末十佳球', 'selected': [{'rank': 1, 'title': '前排扣球'}]})
        self.write('run_config.json', {'design_suite': 'atelier'})

    def write(self, name, data):
        (self.run/name).write_text(json.dumps(data, ensure_ascii=False))

    def job(self, **changes):
        return {'command': 'run', 'values': {'output': str(self.run)}, 'status': 'succeeded',
                'started_at': time.time()-10, 'title': '周末联赛', **changes}

    def test_real_export_survives_backend_removal_and_deduplicates(self):
        original_hash = films.sha256(self.video)
        self.write('verification_lively.json', {'status': 'passed', 'videos': [
            {'path': str(self.video), 'sha256': original_hash, 'full_decode': 'passed'}]})
        ids = films.publish_job(self.root, self.job())
        self.assertEqual(ids, films.publish_job(self.root, self.job()))
        self.assertEqual(len(ids), 1)
        self.assertEqual(films.sha256(self.video), original_hash)
        shutil.rmtree(self.root/'runs')
        data = films.detail(self.app, ids[0])
        self.assertEqual(data['verification'], 'passed')
        self.assertEqual(data['highlights'], [{'rank': 1, 'title': '前排扣球'}])
        row = self.app.library()['outputs'][0]
        self.assertNotIn('run', row)
        self.assertEqual(summary(self.app, self.job(film_ids=ids), '')['verification'], 'passed')
        package = self.root/row['path']
        self.assertEqual(set(p.name for p in package.parent.iterdir()), {'video.mp4', 'poster.jpg', 'manifest.json'})
        content = package.read_bytes()
        self.assertLess(content.index(b'moov'), content.index(b'mdat'))
        self.assertEqual(films.sha256(package), data['video']['sha256'])
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(package), '-f', 'null', '-'],
                       check=True, capture_output=True)

    def test_missing_final_delivery_never_promotes_clean_render(self):
        self.write('delivery.json', {'output': str(self.run/'missing.mp4')})
        self.assertEqual(films.publish_run(self.app, self.run), [])

    def test_failed_partial_and_stale_jobs_do_not_publish(self):
        for status in ('failed', 'cancelled', 'running', 'interrupted'):
            self.assertEqual(films.publish_job(self.root, self.job(status=status)), [])
        self.assertEqual(films.publish_job(self.root, self.job(values={'output':str(self.run),'stop_after':'rank'})), [])
        self.assertEqual(films.publish_job(self.root, self.job(started_at=time.time()+10)), [])
        self.assertEqual(len(films.publish_job(self.root, self.job(started_at=time.time()+10), '[render] 复用已验证结果')), 1)

    def test_invalid_media_and_thumbnail_failure_never_leave_visible_partial_package(self):
        self.video.write_bytes(b'broken video')
        with self.assertRaises(subprocess.CalledProcessError): films.publish_run(self.app, self.run)
        self.assertEqual(self.app.library()['outputs'], [])
        shutil.copy2(self.fixture, self.video)
        original = subprocess.run
        def fail_poster(args, **kwargs):
            if str(args[-1]).endswith('poster.jpg'):
                raise subprocess.CalledProcessError(1, args)
            return original(args, **kwargs)
        with patch('volleymole.web.films.subprocess.run', side_effect=fail_poster):
            with self.assertRaises(subprocess.CalledProcessError): films.publish_run(self.app, self.run)
        self.assertEqual(list((self.root/films.SHELF).iterdir()), [])

    def test_source_mixing_cannot_publish_but_published_film_mixing_can(self):
        ids = films.publish_job(self.root, self.job())
        source = self.root/films.SHELF/ids[0]/'video.mp4'
        output = self.root/'runs/mixed.mp4'
        shutil.copy2(source, output)
        report = output.with_suffix('.audio.json')
        report.write_text(json.dumps({'source':str(self.video),'output':str(output),'video_unchanged':True}))
        job = self.job(command='meme-audio',values={'video':str(self.video),'output':str(output)})
        self.assertEqual(films.publish_job(self.root, job), [])
        job['values']['video'] = str(source)
        report.write_text(json.dumps({'source':str(source),'output':str(output),'video_unchanged':True}))
        self.assertEqual(len(films.publish_job(self.root, job)), 1)

    def test_cancel_during_packaging_removes_staging_without_publishing(self):
        with patch('volleymole.web.films.probe', wraps=films.probe):
            checks=iter((False,True))
            with self.assertRaisesRegex(ValueError,'取消'):
                films.publish_job(self.root,self.job(),cancelled=lambda:next(checks))
        self.assertEqual(self.app.library()['outputs'],[])
        self.assertEqual(list((self.root/films.SHELF).iterdir()),[])

    def test_generic_task_type_does_not_replace_work_title(self):
        ids=films.publish_job(self.root,self.job(title='单视频剪辑'))
        self.assertEqual(films.detail(self.app,ids[0])['title'],'周末十佳球')

    def test_queue_records_publication_error_without_changing_cli_success(self):
        from volleymole.web.jobs import JobManager
        from test_web import wait_until
        manager = JobManager(self.root)
        self.addCleanup(manager.close)
        with patch('volleymole.web.films.publish_job', side_effect=ValueError('封面生成失败')):
            job = manager.submit('templates','list',['templates','list'],{})
            wait_until(lambda: manager.get(job['id'])['status']=='succeeded')
        result = manager.get(job['id'])
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['film_ids'], [])
        self.assertIn('封面生成失败', result['publication_error'])
