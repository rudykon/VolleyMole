"""Actual video/audio sampling with a deterministic fake semantic endpoint."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from volleymole.common import APP, probe, identity, read_json, save_json
from volleymole.event_pipeline import Discovery, collection_artifacts, complete_collections
from volleymole.run_match import argument_parser
from volleymole.semantic import sampled_evidence
from test_events import event, timeline_event


class PipelineTests(unittest.TestCase):
    def test_collection_render_runs_source_picture_and_audio_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'source.mp4'
            subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i',
                'testsrc2=size=720x405:rate=30:duration=4','-f','lavfi','-i',
                'sine=frequency=440:sample_rate=48000:duration=4','-c:v','libx264','-threads','1',
                '-c:a','aac','-shortest',str(video)],check=True)
            source=probe(video)
            manifest={'source':source,'rallies':[],'config':read_json(APP/'defaults.json')}
            timeline={'events':[timeline_event(start=1.,end=3.,clip_start_sec=0.,clip_end_sec=4.)]}
            args=argument_parser().parse_args(['--video',str(video),'--model','fake-av',
                '--collection','bloopers','--style','classic','--quality','720p','--no-sound-model'])

            def render_fixture(directory,*unused,**kwargs):
                directory=Path(directory);output=directory/'top1.mp4'
                subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-filter_complex',
                    '[0:v]scale=720:405,pad=720:1280:0:875:black[v]','-map','[v]','-map','0:a:0',
                    '-c:v','libx264','-threads','1','-r','30','-c:a','aac','-t','4',str(output)],check=True)
                clip={'rank':1,'rally_id':'event_00001','path':str(output),'source_start_sec':0.,
                    'source_end_sec':4.,'duration_sec':4.,'output_frames':120}
                save_json(directory/'render_report.json',{'output':str(output),'order':'countdown',
                    'clips':[clip],'expected_duration_sec':4.})

            with patch('volleymole.media_worker.render',side_effect=render_fixture):
                reports=complete_collections(args,manifest,timeline,root/'run')
            target=root/'run/collections/bloopers'
            self.assertEqual(reports[0]['actual_count'],1)
            self.assertEqual(read_json(target/'verification.json')['status'],'passed')
            self.assertEqual(read_json(target/'alignment_verification.json')['status'],'passed')
            command=json.loads((target/'alignment.log').read_text().splitlines()[0])
            self.assertEqual(command[0],sys.executable)

    def test_single_analysis_serves_both_collections_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=30:duration=8',
                '-f','lavfi','-i','sine=frequency=440:duration=8','-c:v','libx264','-threads','1','-c:a','aac','-shortest',str(video)],check=True)
            source=probe(video);source['identity']=identity(video)
            args=argument_parser().parse_args(['--video',str(video),'--analysis-cache-dir',str(root/'cache'),
                '--model','fake-av','--collection','both','--stop-after','rank','--analysis-timeout','30','--no-sound-model'])
            manifest={'source':source,'rallies':[],'config':read_json(APP/'defaults.json')}
            calls=[]
            def request(endpoint,key,payload,timeout):
                content=payload['messages'][1]['content']
                info=json.loads(content[0]['text']);calls.append(info)
                self.assertTrue(any(p['type']=='input_audio' for p in content))
                self.assertNotIn('rule_score',json.dumps(info));self.assertNotIn('rule_score',payload['messages'][0]['content'])
                frames=[r for r in info['evidence'] if r['kind']=='frame']
                evidence=min(frames,key=lambda f:abs(f['start_sec']-3))
                canonical=event()
                from volleymole.event_schema import EVENT_FIELDS, FRAME_FIELDS
                result={key:canonical[key] for key in EVENT_FIELDS}
                for key in FRAME_FIELDS:
                    when=canonical[key.replace('_frame','_sec')]
                    result[key]=min(frames,key=lambda f:abs(f['start_sec']-when))['id']
                result['facts']=[{'kind':'observation','text':canonical['observations'][0]['text'],
                    'frame_id':evidence['id'],'support_id':None}]
                result['dimensions']=[{'dimension':key,'value':score['value'],
                    'fact_indexes':[] if score['value'] is None else [0]}
                    for key,score in canonical['dimensions'].items()]
                return {'events':[result]},{'model':'fake-av','usage':{},'finish_reason':'stop'}
            with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fake-test-key'}),patch('volleymole.semantic.request_json',side_effect=request):
                timeline=Discovery(source,root,args).finish(manifest)
                self.assertEqual(len(calls),2)
                self.assertEqual(timeline['events'][0]['review_status'],'reviewed')
                self.assertGreater(len(calls[1]['actual_sample_times']),len(calls[0]['actual_sample_times']))
                for collection in ('highlights','bloopers'):
                    result=collection_artifacts(manifest,timeline,root/collection,collection,5)
                    self.assertEqual(result['actual_count'],1)
                self.assertEqual(len(calls),2)
                again=Discovery(source,root,args).finish(manifest)
                self.assertEqual(len(calls),2)
                self.assertTrue(all(r['cached'] for r in again['requests']))
                self.assertNotIn('fake-test-key',(root/'event_timeline.json').read_text())
                # Exercise the real CLI orchestration while replacing only
                # expensive local inference and its already-covered rally builder.
                from types import SimpleNamespace
                from volleymole.common import save_json
                from volleymole.run_match import main
                def ingest(video,directory,*unused):
                    artifacts=[]
                    for name in ('analytics','tracking','player'):
                        save_json(directory/name/'provenance.json',{'mode':'fixture'})
                        artifacts.append(directory/name/'provenance.json')
                    paths={'analytics':directory/'analytics/detections.jsonl',
                           'tracking':directory/'tracking/ball.csv','player':directory/'player/index.json'}
                    for path in paths.values():path.write_text('');artifacts.append(path)
                    pts=directory/'tracking/source_pts.csv';pts.write_text('0\n');artifacts.append(pts)
                    return {k:str(v) for k,v in paths.items()},artifacts
                def build(*params, **kwargs):
                    directory=params[5]
                    save_json(directory/'match_manifest.json',manifest)
                    return manifest,[directory/'match_manifest.json']
                with patch('volleymole.run_match.ModelRegistry',return_value=SimpleNamespace(directory=root,entries={})), \
                     patch('volleymole.run_match.inference_signature',return_value={'fixture':True}), \
                     patch('volleymole.run_match.ingest_shared',side_effect=ingest), \
                     patch('volleymole.run_match.build_manifest',side_effect=build):
                    main(['--video',str(video),'--output',str(root/'cli'),'--collection','both','--model','fake-av','--no-sound-model',
                          '--llm-config',str(root/'absent-config.json'),'--analysis-cache-dir',str(root/'cache'),
                          '--analysis-timeout','30','--stop-after','rank'])
                report=read_json(root/'cli/collections_report.json')
                self.assertEqual([r['actual_count'] for r in report['collections']],[1,1])
                self.assertEqual(len(read_json(root/'cli/timing_latest.json')['collections']),2)

    def test_review_empty_rejects_coarse_event(self):
        # Exercise the replacement semantics without media work or network.
        from concurrent.futures import Future
        from volleymole.events import DIMENSIONS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=argument_parser().parse_args(['--video','unused','--model','fake','--analysis-cache-dir',str(root/'cache')])
            source={'duration_sec':20,'identity':{'sha256':'fixture'}}
            with patch.object(Discovery,'discover',return_value={'results':[], 'failures':[], 'audio':None,
                    'audio_features':{'windows':[],'status':'unknown'},'status':'complete'}):
                discovery=Discovery(source,root,args)
                discovery.future.result()
            e=event()
            discovery.future=Future();discovery.future.set_result({'results':[{'events':[e],'signature':'coarse','job':{'id':'c'},
                'requested_fps':2,'sample_times_sec':[],'request':{},'cached':False,'elapsed_sec':0}],
                'failures':[], 'audio':None, 'audio_features':{'windows':[],'status':'unknown'},'status':'complete'})
            def review(job,*args,**kwargs):
                return {'events':[],'signature':'review','job':job,'requested_fps':8,'sample_times_sec':[],
                        'request':{},'cached':False,'elapsed_sec':0}
            with patch('volleymole.event_pipeline.understand_context',side_effect=review):
                result=discovery.finish({'rallies':[],'config':read_json(APP/'defaults.json')})
            self.assertEqual(result['events'],[])


if __name__=='__main__':unittest.main()
