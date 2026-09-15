import importlib.util
import copy
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('action_eval',Path(__file__).resolve().parents[1]/'scripts/evaluate_volleyball_actions.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def action(t,label='serve'):
    return {'label':label,'time_sec':t}


class ActionEvaluationTests(unittest.TestCase):
    def source_fixture(self):
        row={'video':'heldout_match/rally_01','fps':25.,'num_frames':50,
             'events':[{'label':'serve','frame':12,'xy':[.1,.2]}]}
        sample={'video':row['video'],'annotation_row':row,'fps':25.,'frames':50,
                'source_declared_num_frames':50,'frame_count_matches_annotation':True,
                'first_frame_file':'000000.jpg','match_id':'heldout_match','audio_available':False,
                'derived_video':'sample.mp4','derived_video_sha256':'unchanged-media'}
        source={'frame_count':50,'nominal_fps':'25/1','average_fps':'25/1','duration_sec':2.,
                'start_sec':0.,'video_start_sec':0.,'has_audio':False}
        return sample,source

    def test_maximum_cardinality_beats_greedy_nearest(self):
        counts=mod.match_counts([action(0),action(.2)],[action(.2),action(.4)],.2)
        self.assertEqual(counts['serve'],{'tp':2,'fp':0,'fn':0})

    def test_duplicates_and_wrong_class_do_not_inflate_recall(self):
        counts=mod.match_counts([action(1)],[action(1),action(1),action(1,'spike')],.1)
        self.assertEqual(counts['serve'],{'tp':1,'fp':1,'fn':0})
        self.assertEqual(counts['spike'],{'tp':0,'fp':1,'fn':0})

    def test_failed_prediction_is_false_negative(self):
        self.assertEqual(mod.match_counts([action(1)],[],.5)['serve'],{'tp':0,'fp':0,'fn':1})

    def test_prediction_requires_valid_time_and_frame(self):
        frames=[{'id':'frame_00000','kind':'frame'}]
        data={'events':[{**action(1),'confidence':.5,'evidence_ids':['frame_00000']}],'uncertainty':''}
        self.assertEqual(mod.validate_predictions(data,frames,2),data)
        for invalid in (float('nan'),2,-1,True):
            row={**data,'events':[{**data['events'][0],'time_sec':invalid}]}
            with self.assertRaises(ValueError):mod.validate_predictions(row,frames,2)
        data['events'][0]['evidence_ids']=['invented']
        with self.assertRaises(ValueError):mod.validate_predictions(data,frames,2)

    def test_manifest_clock_and_counts_must_match_actual_media(self):
        sample,source=self.source_fixture()
        mod.validate_sample_timing(sample,source)
        for field,value in (('fps',50.),('frames',1050),('match_id','different_match'),
                            ('source_declared_num_frames',40),('first_frame_file','000001.jpg'),
                            ('frame_count_matches_annotation',False)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                mod.validate_sample_timing({**sample,field:value},source)
        for field,value in (('average_fps','30/1'),('nominal_fps','30/1'),('duration_sec',4.),
                            ('start_sec',1.),('video_start_sec',.04),('has_audio',True)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                mod.validate_sample_timing(sample,{**source,field:value})

    def test_documented_extra_final_frame_retains_original_clock(self):
        sample,source=self.source_fixture()
        sample.update(frames=51,frame_count_matches_annotation=False)
        source.update(frame_count=51,duration_sec=2.04)
        mod.validate_sample_timing(sample,source)

    def test_completion_accounts_for_failed_and_unstarted_jobs(self):
        jobs=[{'id':'a','fps':2.},{'id':'b','fps':8.}]
        summary=mod.completion_summary(jobs,[{'id':'a'}],[{'job':'b','error':'HTTPError'}])
        self.assertFalse(summary['complete'])
        self.assertTrue(summary['completion_by_fps']['2.0']['complete'])
        self.assertFalse(summary['completion_by_fps']['8.0']['complete'])
        self.assertFalse(mod.completion_summary(jobs,[],[])['complete'])
        self.assertTrue(mod.completion_summary(jobs,[{'id':'a'},{'id':'b'}],[])['complete'])
        with self.assertRaises(ValueError):mod.completion_summary(jobs,[{'id':'a'},{'id':'a'}],[])

    def test_actual_payload_excludes_labels_and_failures_keep_false_negatives(self):
        sample,source=self.source_fixture()
        sample['annotation_row']['author_only_marker']='NEVER_TRANSMIT_THIS_LABEL'
        evidence=[{'id':'frame_00000','kind':'frame','start_sec':0.,'end_sec':0.}]
        captured=[]
        def sampler(*args,**kwargs):
            return copy.deepcopy(evidence),[{'type':'text','text':'frame_00000 source_sec=0.000000'}],[]
        def request(endpoint,key,payload,timeout):
            captured.append(payload)
            timing=json.loads(payload['messages'][1]['content'][0]['text'])
            if timing['requested_fps']==8.:raise TimeoutError('simulated failure')
            return {'events':[{**action(.48),'confidence':.9,'evidence_ids':['frame_00000']}],
                    'uncertainty':''},{'usage':{},'finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(dataset=Path(directory),output=Path(directory)/'results',
                llm_config=Path(directory)/'fake_config.json',api_base=None,model=None,
                fps=[2.,8.],width=224,concurrency=1,total_timeout=30,request_timeout=5,tolerances=[.2])
            with patch.object(mod,'load_samples',return_value=({'splits':{'test':{'clips':115}}},[sample])), \
                 patch.object(mod,'read_json',return_value={'llm':{'base_url':'https://example.invalid',
                     'model':'unit-test-visual-model','api_key':'unit-test-placeholder'}}), \
                 patch.object(mod,'digest',return_value='fixed-hash'), \
                 patch.object(mod,'probe',return_value=source), \
                 patch.object(mod,'sampled_evidence',side_effect=sampler), \
                 patch.object(mod,'request_json',side_effect=request),contextlib.redirect_stdout(io.StringIO()):
                report=mod.evaluate(args)
        self.assertEqual(len(captured),2)
        body=json.dumps(captured)
        for forbidden in ('NEVER_TRANSMIT_THIS_LABEL','annotation_row','num_events','"xy"','heldout_match'):
            self.assertNotIn(forbidden,body)
        self.assertEqual(report['completed_requests'],1)
        self.assertFalse(report['complete'])
        self.assertEqual(report['metrics']['2.0']['0.2']['micro']['tp'],1)
        self.assertEqual(report['metrics']['8.0']['0.2']['micro']['fn'],1)

    def test_cli_exits_nonzero_for_incomplete_report(self):
        report={'complete':False,'completed_requests':0,'failures':[{'job':'x'}],'metrics':{}}
        with patch.object(mod.sys,'argv',['evaluate_volleyball_actions.py']), \
             patch.object(mod,'evaluate',return_value=report),contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as raised:
            mod.main()
        self.assertNotEqual(raised.exception.code,0)


if __name__=='__main__':unittest.main()
