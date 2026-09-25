"""Video UI contracts: real metadata caching and evidence-based progress."""
import json
from pathlib import Path
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from volleymole.web.server import Workspace
from volleymole.web.media import MediaCache
from volleymole.web.progress import summary
from web_fixtures import film_fixture


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.workspace=Workspace.__new__(Workspace);self.workspace.root=self.root
        self.cache=MediaCache(self.workspace)

    def write(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(data)) if isinstance(data,dict) else p.write_bytes(data)
        return p

    def test_thumbnail_uses_rotation_actual_metadata_and_invalidates_on_change(self):
        self.write('data/portrait.mp4',b'video')
        probe=SimpleNamespace(stdout=json.dumps({'format':{'duration':'75.2'},'streams':[{'codec_type':'video','width':1920,'height':1080,'side_data_list':[{'rotation':-90}]}]}))
        frame=SimpleNamespace(returncode=0,stdout=b'\xff\xd8real-frame')
        with patch('volleymole.web.media.subprocess.run',side_effect=[probe,frame,probe,frame]) as run:
            a=self.cache.info('data/portrait.mp4');b=self.cache.info('data/portrait.mp4')
            self.assertEqual(a,b);self.assertEqual(run.call_count,2)
            self.assertEqual((a['width'],a['height'],a['duration']),(1080,1920,75.2))
            self.assertTrue(self.cache.thumbnail('data/portrait.mp4').read_bytes().startswith(b'\xff\xd8'))
            self.write('data/portrait.mp4',b'replaced-video')
            c=self.cache.info('data/portrait.mp4')
            self.assertNotEqual(a['poster'],c['poster']);self.assertEqual(run.call_count,4)

    def test_thumbnail_rejects_private_external_and_nonmedia_paths(self):
        self.write('data/private.mp4',b'private');self.write('data/info.json',{})
        self.write('secret.mp4',b'private')
        (self.root/'data/alias.mp4').symlink_to(self.root/'secret.mp4')
        for path in ['../outside.mp4','data/alias.mp4','data/info.json']:
            with self.subTest(path=path),self.assertRaises(ValueError): self.cache.info(path)

    def test_missing_decoder_reports_unavailable_without_fake_metadata(self):
        self.write('data/broken.mp4',b'bad')
        with patch('volleymole.web.media.subprocess.run',side_effect=FileNotFoundError):
            result=self.cache.info('data/broken.mp4')
        self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['poster']);self.assertIsNone(result['duration'])

    def test_film_titles_and_details_live_in_the_package(self):
        film_id=film_fixture(self.root)
        self.workspace.jobs=SimpleNamespace(list=lambda:[])
        film=self.workspace.library()['outputs'][0]
        self.assertEqual(film['title'],'周末联赛 · 十佳球')
        self.assertEqual(film['film_id'],film_id)
        self.assertNotIn('run',film)

    def test_stage_completion_does_not_imply_finished_film_or_verification(self):
        started=time.time()-10
        self.write('runs/test/state.json',{'stages':{'inference':{'status':'complete','started_at':started+1},'rallies':{'status':'running','started_at':started+2},'verify':{'status':'complete','started_at':started-100}}})
        job={'values':{'output':'runs/test'},'status':'running','started_at':started}
        result=summary(self.workspace,job,'')
        self.assertEqual([s['status'] for s in result['stages']],['complete','running','unknown','unknown','unknown'])
        self.assertEqual(result['outputs'],[]);self.assertEqual(result['verification'],'unknown')
        job['status']='failed'
        self.assertEqual(summary(self.workspace,job,'')['stages'][1]['status'],'interrupted')

    def test_playback_quality_survives_cleanup_and_must_cover_current_video(self):
        film_id=film_fixture(self.root,verification='passed')
        job={'id':'job','command':'run','values':{'output':'runs/deleted'},'status':'succeeded',
             'started_at':time.time()-10,'film_ids':[film_id]}
        result=summary(self.workspace,job,'')
        self.assertEqual(len(result['outputs']),1)
        self.assertEqual(result['verification'],'passed')
        video=self.root/f'outputs/published/{film_id}/video.mp4'
        video.write_bytes(b'changed-content')
        result=summary(self.workspace,job,'')
        self.assertEqual(result['outputs'],[])
        self.assertEqual(result['verification'],'unknown')

    def test_reused_stage_does_not_promote_unpublished_backend_video(self):
        started=time.time()
        p=self.write('runs/test/final.mp4',b'old-film');os.utime(p,(started-100,started-100))
        self.write('runs/test/render_report.json',{'output':str(p)})
        self.write('runs/test/state.json',{'stages':{'render':{'status':'complete','started_at':started-100}}})
        job={'values':{'output':'runs/test'},'status':'succeeded','started_at':started}
        result=summary(self.workspace,job,'[render] 复用已验证结果')
        self.assertEqual(result['outputs'],[])
        self.assertEqual(result['stages'][3]['status'],'complete')
        self.assertEqual(result['stages'][0]['status'],'unknown')

    def test_publication_failure_is_explained_without_claiming_task_failure(self):
        job={'values':{},'status':'succeeded','publication_error':'成片收录失败：无法生成封面'}
        result=summary(self.workspace,job,'')
        self.assertEqual(result['message'],job['publication_error'])
        self.assertEqual(result['verification'],'unknown')
