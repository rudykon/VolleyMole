"""Web contracts: actual CLI forms, HTTP isolation, files and process lifecycle."""
import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from volleymole.web.catalog import COMMANDS, build_argv, catalog, fields, parser_for
from volleymole.web.jobs import JobManager, process_stamp, redact
from volleymole.web.server import Server, Workspace, listen_host


def wait_until(fn,timeout=8):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        result=fn()
        if result: return result
        time.sleep(.03)
    raise AssertionError('Timed out waiting for job state')


class CatalogTests(unittest.TestCase):
    def test_explicit_lan_listen_addresses(self):
        for value in ('127.0.0.1','localhost','172.22.13.156'):
            self.assertEqual(listen_host(value),value)
        for value in ('attacker.invalid','http://172.22.13.156','999.0.0.1','::1','0.0.0.0','224.0.0.1','255.255.255.255'):
            with self.subTest(value=value),self.assertRaises(argparse.ArgumentTypeError): listen_host(value)

    def test_catalog_covers_every_public_command_and_option(self):
        import ast
        tree=ast.parse(Path('src/volleymole/cli.py').read_text())
        commands=next(k.value.elts for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                      and n.func.attr=='add_argument' for k in n.keywords if k.arg=='choices')
        self.assertEqual(set(COMMANDS),{n.value for n in commands}-{'web'})
        for command in catalog():
            parser=parser_for(command['id'])
            public={a.dest for a in parser._actions if a.dest!='help' and not isinstance(a,argparse._SubParsersAction) and a.help!=argparse.SUPPRESS}
            self.assertEqual(public,{f['key'] for f in command['fields']})
            self.assertTrue(all(f['label']!=f['key'] for f in command['fields']))

    def test_boolean_false_keeps_template_override_and_flag_order(self):
        argv,values=build_argv('run',None,{'video':'data/a.mp4','template':'matchday','replays':False,'replay_review':'off','top_k':10},lambda p,k:Path('/tmp')/p)
        self.assertIn('--no-replays',argv)
        args=parser_for('run').parse_args(argv[1:])
        self.assertFalse(args.replays)
        self.assertEqual(args.top_k,10)

    def test_subcommand_parent_options_precede_child_and_lists_remain_arguments(self):
        argv,_=build_argv('models','verify',{'directory':'models','names':['ball','person']},lambda p,k:Path('/tmp')/p)
        self.assertEqual(argv,['models','--directory=/tmp/models','verify','ball','person'])
        argv,_=build_argv('assets','import-file',{'directory':'data/assets','archive':'data/bundle.zip'},lambda p,k:Path('/tmp')/p)
        args=parser_for('assets').parse_args(argv[1:])
        self.assertEqual(args.archive,Path('/tmp/data/bundle.zip'))

    def test_unknown_hidden_nonfinite_and_invalid_inputs_rejected(self):
        cases=[{'video':'a.mp4','unknown':'x'},{'video':'a.mp4','event_frame_cache':'x'},
               {'video':'a.mp4','top_k':7},{'video':'a.mp4','top_k':True},
               {'video':'a.mp4','replay_speed':float('nan')},{'video':'a.mp4','replays':'false'},{}]
        for values in cases:
            with self.subTest(values=values),self.assertRaises(ValueError):
                build_argv('run',None,values,lambda p,k:Path('/tmp')/p)
        with self.assertRaises(ValueError): build_argv('shell',None,{},lambda p,k:p)
        with self.assertRaises(ValueError): build_argv('models','verify',{'names':['--help']},lambda p,k:p)

    def test_paths_with_shell_metacharacters_are_literal_arguments(self):
        name='data/a $(touch x); `echo no`.mp4'
        argv,_=build_argv('run',None,{'video':name},lambda p,k:p)
        self.assertIn('--video='+name,argv)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.app=Workspace(self.root)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.app.jobs.close)

    def test_reject_traversal_symlinks_source_and_credentials(self):
        (self.root/'data').mkdir()
        (self.root/'data/link').symlink_to('/tmp')
        for path in ('../outside','/etc/passwd','data/link/x','src/volleymole/cli.py','.git/config','llm_api.json','github_token.json'):
            with self.subTest(path=path),self.assertRaises(ValueError): self.app.path(path)
        self.assertEqual(self.app.argument_path('llm_api.json','llm_config'),self.root/'llm_api.json')
        with self.assertRaises(ValueError): self.app.argument_path('llm_api.json','output')
        with self.assertRaises(ValueError): self.app.argument_path('.','output')
        self.assertEqual(self.app.path('runs/.analysis-cache'),self.root/'runs/.analysis-cache')

    def test_secret_settings_preserve_unrelated_nodes_and_existing_key(self):
        path=self.root/'llm_api.json'
        path.write_text(json.dumps({'rank':{'model':'unchanged'},'llm':{'api_key':'private-test-value'}}))
        settings={'base_url':'https://example.invalid/v1','model':'vision','api_key':''}
        result=self.app.save_settings(settings)
        self.assertTrue(result['has_key']);self.assertNotIn('private-test-value',json.dumps(result))
        doc=json.loads(path.read_text());self.assertEqual(doc['rank']['model'],'unchanged')
        self.assertEqual(doc['llm']['api_key'],'private-test-value')
        self.assertEqual(path.stat().st_mode&0o777,0o600)
        result=self.app.save_settings({**settings,'clear_key':True})
        self.assertFalse(result['has_key'])
        with self.assertRaises(ValueError): self.app.save_settings({**settings,'base_url':'javascript:alert(1)'})

    def test_redaction_covers_nested_config_and_log_headers(self):
        value=redact({'llm':{'api_key':'hide-me'},'max_output_tokens':123,'log':'Authorization: Bearer private-value\napi_key="private-value"'})
        self.assertNotIn('private-value',json.dumps(value));self.assertNotIn('hide-me',json.dumps(value))
        self.assertEqual(value['max_output_tokens'],123)

    def test_template_and_run_recovery_uses_source_not_guessed_filename(self):
        run=self.root/'runs/2026-09-20-top10';run.mkdir(parents=True)
        (self.root/'data').mkdir()
        (run/'run_config.json').write_text(json.dumps({'mode':'date_grouped_match','date':'2026-09-20','top_k':10,'quality':'720p'}))
        (run/'match_manifest.json').write_text(json.dumps({'sources':{'one':{'path':str(self.root/'data/2026.9.20.1.mp4')},'two':{'path':str(self.root/'data/2026.9.20.2.mp4')}}}))
        (run/'edit_decision.json').write_text('{"ranking_mode":"rules"}')
        result=self.app.resume('runs/2026-09-20-top10')
        self.assertEqual(result['command'],'match')
        self.assertEqual(result['values']['input_dir'],str(self.root/'data'))
        self.assertEqual(result['values']['output'],str(self.root/'runs'))
        self.assertEqual(result['values']['ranker'],'rules')

    def test_media_library_separates_sources_exports_and_intermediates(self):
        files=['data/match.MOV','data/uploads/batch/voice.wav','outputs/final.mp4',
               'runs/game/top5.mp4','runs/game/segments/part.mp4','runs/game/previews/rally.mp4',
               'outputs/segments/part.mp4','data/visual-assets/intro.mp4',
               'data/.analysis-cache/cache.mp4','data/public_benchmarks/eval.mp4','runs/game/analysis.json']
        for name in files:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'media')
        (self.root/'runs/game/render_report.json').write_text(json.dumps({'output':str(self.root/'runs/game/top5.mp4')}))
        (self.root/'runs/game/render_report_lively.json').write_text(json.dumps({'output':str(self.root/'outputs/final.mp4')}))
        result=self.app.library()
        self.assertEqual({r['path'] for r in result['outputs']},{'outputs/final.mp4','runs/game/top5.mp4'})
        self.assertEqual({r['path'] for r in result['materials']},{'data/match.MOV','data/uploads/batch/voice.wav'})
        self.assertEqual(next(r['kind'] for r in result['materials'] if r['name']=='voice.wav'),'audio')
        self.assertTrue(all('/' not in r['name'] for rows in result.values() for r in rows))

    def test_media_library_handles_moved_runs_missing_files_and_private_links(self):
        run=self.root/'runs/moved';run.mkdir(parents=True)
        (run/'top5.mp4').write_bytes(b'film')
        (run/'render_report.json').write_text('{"output":"/old/workspace/runs/moved/top5.mp4"}')
        (run/'render_report_broken.json').write_text('{')
        (run/'render_report_missing.json').write_text('{"output":"missing.mp4"}')
        (run/'render_report_list.json').write_text('[]')
        (self.root/'data').mkdir()
        (self.root/'data/empty.mp4').touch()
        (self.root/'data/secret.mp4').write_bytes(b'private')
        (self.root/'data/alias.mp4').symlink_to(self.root/'data/secret.mp4')
        (self.root/'data/external').symlink_to('/tmp')
        result=self.app.library()
        self.assertEqual([r['path'] for r in result['outputs']],['runs/moved/top5.mp4'])
        self.assertEqual(result['materials'],[])

    def test_media_library_prefers_delivered_film_over_pre_audio_render(self):
        run=self.root/'runs/with-audio';run.mkdir(parents=True)
        (run/'clean.mp4').write_bytes(b'no-voice')
        (run/'render_report_lively.json').write_text(json.dumps({'output':str(run/'clean.mp4')}))
        export=self.root/'outputs/finished.mp4';export.parent.mkdir()
        export.write_bytes(b'final-with-voice')
        (run/'delivery.json').write_text(json.dumps({'output':str(export)}))
        self.assertEqual([r['path'] for r in self.app.library()['outputs']],['outputs/finished.mp4'])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.manager=JobManager(self.root)
        self.addCleanup(self.temp.cleanup);self.addCleanup(self.manager.close)

    def test_real_cli_execution_and_history(self):
        j=self.manager.submit('templates','list',['templates','list'],{})
        done=wait_until(lambda:self.manager.get(j['id'])['status']=='succeeded')
        self.assertTrue(done)
        self.assertIn('matchday',self.manager.log(j['id']))
        self.manager.close()
        self.manager=JobManager(self.root)
        self.addCleanup(self.manager.close)
        self.assertEqual(self.manager.get(j['id'])['status'],'succeeded')

    def test_only_one_manager_per_workspace(self):
        with self.assertRaises(RuntimeError): JobManager(self.root)

    def test_failure_is_not_success_and_queue_continues(self):
        a=self.manager.submit('templates','show',['templates','show','does-not-exist'],{})
        b=self.manager.submit('templates','list',['templates','list'],{})
        wait_until(lambda:self.manager.get(b['id'])['status']=='succeeded')
        self.assertEqual(self.manager.get(a['id'])['status'],'failed')

    def test_cancel_running_process_group_and_queued_job(self):
        original=subprocess.Popen
        def slow(*args,**kwargs):
            return original([sys.executable,'-c','import time; time.sleep(30)'],**kwargs)
        with patch('volleymole.web.jobs.subprocess.Popen',side_effect=slow):
            a=self.manager.submit('templates','list',['templates','list'],{})
            wait_until(lambda:self.manager.get(a['id'])['status']=='running')
            b=self.manager.submit('templates','list',['templates','list'],{})
            self.manager.cancel(b['id']);self.manager.cancel(a['id'])
            wait_until(lambda:self.manager.get(a['id'])['status']=='cancelled')
            self.assertEqual(self.manager.get(b['id'])['status'],'cancelled')

    def test_reconnect_live_process_after_server_restart_without_relaunch(self):
        self.manager.close()
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'],start_new_session=True)
        self.addCleanup(process.wait)
        record={'id':'recovery','command':'templates','subcommand':'list','values':{},'argv':['templates','list'],
                'title':'Recovery','created_at':time.time(),'status':'running','started_at':time.time(),
                'pid':process.pid,'process_stamp':process_stamp(process.pid)}
        (self.root/'.local/web/recovery.json').write_text(json.dumps(record))
        self.manager=JobManager(self.root);self.addCleanup(self.manager.close)
        self.assertEqual(self.manager.get('recovery')['status'],'running')
        wait_until(lambda:self.manager.get('recovery')['status']=='interrupted')
        self.assertIsNone(self.manager.get('recovery')['returncode'])

    def test_cancel_waits_for_child_which_ignores_sigterm(self):
        original=subprocess.Popen
        pidfile=self.root/'child.pid'
        child_code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
        parent_code=(f'import subprocess,sys,time; from pathlib import Path; '
                     f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
                     f'time.sleep(.2); Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)')
        def slow(*args,**kwargs):
            return original([sys.executable,'-c',parent_code],**kwargs)
        with patch('volleymole.web.jobs.subprocess.Popen',side_effect=slow):
            job=self.manager.submit('templates','list',['templates','list'],{})
            wait_until(pidfile.exists)
            child_pid=int(pidfile.read_text())
            self.manager.cancel(job['id'])
            self.assertEqual(self.manager.get(job['id'])['status'],'cancelling')
            wait_until(lambda:self.manager.get(job['id'])['status']=='cancelled')
            wait_until(lambda:process_stamp(child_pid) is None)


class HTTPTests(unittest.TestCase):
    bind_host='127.0.0.1'
    connect_host='127.0.0.1'

    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.app=Workspace(cls.root)
        cls.server=Server((cls.bind_host,0),cls.app)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        (cls.root/'data').mkdir()
        (cls.root/'data/sample.mp4').write_bytes(bytes(range(256))*8)
        (cls.root/'llm_api.json').write_text('{"llm":{"api_key":"do-not-expose"}}')

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.app.jobs.close();cls.temp.cleanup()

    def request(self,path,body=None,headers=None,method=None):
        connection=http.client.HTTPConnection(self.connect_host,self.server.server_port,timeout=10)
        h={}
        if body is not None:
            if not isinstance(body,bytes): body=json.dumps(body).encode()
            h={'Content-Type':'application/json','X-VolleyMole-Token':self.app.token}
        h.update(headers or {})
        connection.request(method or ('GET' if body is None else 'POST'),path,body,headers=h)
        response=connection.getresponse();data=response.read();status=response.status;rh=dict(response.getheaders())
        connection.close();return status,rh,data

    def test_bootstrap_and_packaged_static_files(self):
        status,_,data=self.request('/api/bootstrap');self.assertEqual(status,200)
        self.assertEqual(len(json.loads(data)['commands']),10)
        for path in ('/','/static/app.js','/static/app.css','/static/previews/matchday.png'):
            status,_,body=self.request(path);self.assertEqual(status,200);self.assertTrue(body)

    def test_media_library_endpoint(self):
        status,_,data=self.request('/api/library')
        self.assertEqual(status,200)
        result=json.loads(data)
        self.assertEqual(set(result),{'outputs','materials'})
        self.assertIn('sample.mp4',[row['name'] for row in result['materials']])
        self.assertNotIn('llm_api.json',json.dumps(result))

    def test_cross_site_host_and_missing_token_are_rejected(self):
        for headers in ({'X-VolleyMole-Token':''},{'Origin':'https://attacker.invalid'},{'Host':'attacker.invalid'}):
            self.assertEqual(self.request('/api/jobs',{'command':'templates','subcommand':'list','values':{}},headers)[0],403)
        self.assertEqual(self.request('/api/bootstrap',headers={'Host':'attacker.invalid'})[0],403)

    def test_credentials_and_symlink_escape_never_served(self):
        for path in ('/media?path=llm_api.json','/api/json?path=llm_api.json','/api/files?path=../','/static/../../llm_api.json'):
            status,_,body=self.request(path);self.assertIn(status,(400,403));self.assertNotIn(b'do-not-expose',body)

    def test_media_ranges_seek_suffix_head_and_416(self):
        for header,expected in [('bytes=10-19',bytes(range(10,20))),('bytes=-5',bytes(range(251,256))),('bytes=2040-',bytes(range(248,256)))]:
            status,headers,data=self.request('/media?path=data/sample.mp4',headers={'Range':header})
            self.assertEqual(status,206);self.assertEqual(data,expected);self.assertIn('Content-Range',headers)
        for invalid in ('bytes=9999-','bytes=4-2','bytes=-0','bytes=0-2,4-8'):
            self.assertEqual(self.request('/media?path=data/sample.mp4',headers={'Range':invalid})[0],416)
        status,headers,data=self.request('/media?path=data/sample.mp4',method='HEAD')
        self.assertEqual(status,200);self.assertEqual(data,b'');self.assertEqual(headers['Content-Length'],'2048')

    def test_upload_is_streamed_non_overwriting_and_downloadable(self):
        path='/api/upload?name=2026.9.22.1.mp4&batch=testbatch'
        status,_,data=self.request(path,b'video-content',{'Content-Type':'application/octet-stream'})
        self.assertEqual(status,201);record=json.loads(data)
        self.assertEqual((self.root/record['path']).read_bytes(),b'video-content')
        self.assertEqual(self.request(path,b'other-content')[0],409)
        self.assertEqual(self.request('/api/upload?name=../bad.mp4',b'bad')[0],400)
        self.assertEqual(self.request('/api/upload?name=script.py',b'bad')[0],400)
        self.assertFalse(list((self.root/'data/uploads/testbatch').glob('*.partial')))

    def test_templates_save_conflict_validation_and_catalog(self):
        doc={'version':1,'name':'web-test','options':{'quality':'720p','replays':False}}
        self.assertEqual(self.request('/api/templates',doc)[0],201)
        self.assertEqual(self.request('/api/templates',doc)[0],409)
        self.assertEqual(self.request('/api/templates',{**doc,'name':'matchday'})[0],400)
        self.assertEqual(self.request('/api/templates',{**doc,'name':'../escape'})[0],400)
        self.assertIn(b'web-test',self.request('/api/templates')[2])

    def test_real_job_submission_polling_and_validation(self):
        status,_,data=self.request('/api/jobs',{'command':'templates','subcommand':'list','values':{}})
        self.assertEqual(status,201);job=json.loads(data)
        done=wait_until(lambda:json.loads(self.request('/api/jobs/'+job['id'])[2])['status']=='succeeded')
        self.assertTrue(done)
        self.assertEqual(self.request('/api/jobs',{'command':'run','values':{'top_k':8}})[0],400)

    def test_algorithm_config_validation_and_save(self):
        body={'kind':'algorithm','document':{'min_rally_sec':4,'weights':{'duration':25}}}
        status,_,data=self.request('/api/document',body);self.assertEqual(status,201)
        self.assertTrue((self.root/json.loads(data)['path']).is_file())
        for doc in ({'evil':1},{'weights':{'unknown':1}},{'min_rally_sec':'bad'}):
            self.assertEqual(self.request('/api/document',{'kind':'algorithm','document':doc})[0],400)


class LANHTTPTests(HTTPTests):
    """Exercise all endpoints through an explicitly bound second interface."""
    bind_host='127.0.0.2'
    connect_host='127.0.0.2'

    def test_lan_same_origin_can_submit_but_other_interfaces_cannot_impersonate_host(self):
        authority=f'{self.connect_host}:{self.server.server_port}'
        body={'command':'templates','subcommand':'list','values':{}}
        status,_,data=self.request('/api/jobs',body,{'Origin':'http://'+authority})
        self.assertEqual(status,201)
        job=json.loads(data)
        wait_until(lambda:self.app.jobs.get(job['id'])['status']=='succeeded')
        self.assertEqual(self.request('/api/jobs',body,{'Origin':f'http://127.0.0.1:{self.server.server_port}'})[0],403)
        for host in ('172.22.13.156','192.168.1.1','0.0.0.0','localhost.attacker.invalid'):
            with self.subTest(host=host):
                self.assertEqual(self.request('/api/bootstrap',headers={'Host':f'{host}:{self.server.server_port}'})[0],403)


if __name__=='__main__': unittest.main()
