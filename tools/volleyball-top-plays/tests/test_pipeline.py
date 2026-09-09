import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import Stages, read_json, save_json
from ranker import rank, rule_decision
from schemas import validate_decision, local_file


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rallies=[]
        for i in range(10):
            track=f'track_{i}.json';(self.root/track).write_text('{}')
            previews=[f'{i}_{k}.jpg' for k in range(3)]
            for p in previews:(self.root/p).write_bytes(b'fake-image-for-validation')
            self.rallies.append({'rally_id':f'r{i}','start_sec':i*20+2.,'end_sec':i*20+10.,
                'safe_start_sec':i*20.,'safe_end_sec':i*20+12.,'eligible':True,'rule_score':90-i,
                'tracking_json':track,'preview_frames':previews,'duration_sec':8,'actions':['set'],
                'players':[],'ball_metrics':{'visible_ratio':.8,'trajectory_changes':5},'exclusion_reasons':[]})
        self.manifest={'source':{'duration_sec':220},'rallies':self.rallies,'config':{'preview_limit':25}}
        self.decision=rule_decision(self.rallies,5)

    def tearDown(self):self.temp.cleanup()

    def test_top5_and_top10_valid(self):
        for k in (5,10):validate_decision(rule_decision(self.rallies,k),self.manifest,self.root,k)

    def test_reject_hallucinated_ids_duplicate_ranks_nan_and_truncation(self):
        for field,value in [('rally_id','invented'),('rank',2),('rank',True),('clip_start_sec',float('nan')),
                            ('clip_start_sec',3.),('clip_end_sec',9.),('confidence',1.2),('title','x\ncommand')]:
            with self.subTest(field=field,value=value):
                decision=copy.deepcopy(self.decision);decision['selected'][0][field]=value
                with self.assertRaises(ValueError):validate_decision(decision,self.manifest,self.root,5)

    def test_missing_files_stop_render(self):
        (self.root/self.rallies[0]['preview_frames'][1]).unlink()
        with self.assertRaises(ValueError):validate_decision(self.decision,self.manifest,self.root,5)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):local_file(self.root,'../outside.jpg')

    def test_model_failure_falls_back_and_reports_every_rejection(self):
        with patch.dict('os.environ',{'VOLLEYMOLE_API_KEY':'test-token'}),patch('ranker.api_decision',side_effect=URLError('offline')):
            path,_=rank(self.manifest,self.root,5,None,'auto','https://example.invalid/v1','test-model',.1)
        decision=read_json(path)
        self.assertEqual(decision['ranking_mode'],'rules_fallback')
        self.assertEqual(len(decision['selected']),5)
        self.assertEqual(len(decision['rejected']),5)
        self.assertEqual(decision['fallback']['type'],'URLError')

    def test_invalid_successful_model_response_also_falls_back(self):
        invalid=copy.deepcopy(self.decision);invalid['selected'][0]['clip_end_sec']=999
        with patch.dict('os.environ',{'VOLLEYMOLE_API_KEY':'test-token'}),patch('ranker.api_decision',return_value=invalid):
            path,_=rank(self.manifest,self.root,5,None,'auto','https://example.invalid/v1','test-model')
        self.assertEqual(read_json(path)['ranking_mode'],'rules_fallback')

    def test_valid_model_response_used(self):
        with patch.dict('os.environ',{'VOLLEYMOLE_API_KEY':'test-token'}),patch('ranker.api_decision',return_value=self.decision):
            path,_=rank(self.manifest,self.root,5,None,'auto','https://example.invalid/v1','test-model')
        self.assertEqual(read_json(path)['ranking_mode'],'multimodal_api')

    def test_refuse_fewer_rallies_than_requested(self):
        self.manifest['rallies']=self.rallies[:4]
        with self.assertRaises(ValueError):rank(self.manifest,self.root,5,None,'rules','',None)


class ResumeTests(unittest.TestCase):
    def test_resume_invalidates_modified_outputs_and_failed_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'artifact.json';calls=[]
            def callback():
                calls.append(1);save_json(path,{'x':1});return str(path),[path]
            stages=Stages(temp)
            stages.execute('test',{'version':1},callback)
            Stages(temp).execute('test',{'version':1},callback)
            self.assertEqual(len(calls),1)
            path.write_text('corrupted')
            Stages(temp).execute('test',{'version':1},callback)
            self.assertEqual(len(calls),2)
            with self.assertRaises(RuntimeError):
                stages.execute('broken',{},lambda:(_ for _ in ()).throw(RuntimeError('test')))
            self.assertEqual(read_json(Path(temp)/'state.json')['stages']['broken']['status'],'failed')
            Stages(temp).execute('broken',{},callback)
            self.assertEqual(read_json(Path(temp)/'state.json')['stages']['broken']['status'],'complete')


if __name__=='__main__':unittest.main()
