"""Event evidence, ranking counterexamples, budgets and shared dual-output tests."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from volleymole.events import DIMENSIONS, windows, validate_events, merge_events, score_event, select_events
from volleymole.audio_events import measure_audio
from volleymole.motion_features import relative_motion
from volleymole.semantic import bounded_map
from volleymole.schemas import validate_decision
from volleymole.common import APP, read_json, save_json
from volleymole.event_pipeline import collection_artifacts
from volleymole.presentation import build_timeline, validate_timeline


def event(start=2., end=5., **changes):
    result = {'event_type': 'unexpected_save', 'start_sec': start, 'end_sec': end,
        'clip_start_sec': start-1, 'clip_end_sec': end+1, 'peak_sec': start+1,
        'observations': [{'text': '意外救回后继续组织', 'time_sec': start+1, 'evidence_ids': ['f']}],
        'aftermath': [], 'reactions': [], 'title': '意外救回', 'confidence': .8,
        'is_rally': True, 'boundary_complete': True, 'injury_suspected': False,
        'laughter_linked': None, 'uncertainty': '快速触球次数未知',
        'dimensions': {k: {'value': None if k in ('motion_intensity', 'related_laughter') else 3,
                           'evidence_ids': [] if k in ('motion_intensity', 'related_laughter') else ['f']} for k in DIMENSIONS}}
    result.update(changes)
    return result


def timeline_event(**kwargs):
    return event(**{'source_id':'single', 'event_id':'event_00001', 'origins':['context_hash'], 'review_status':'reviewed', **kwargs})


class EventTests(unittest.TestCase):
    def test_chunks_cover_preparation_and_tail_without_rally_gate(self):
        chunks = list(windows(63, 24, 4))
        self.assertEqual(chunks, [(0,24),(20,44),(40,63)])
        self.assertEqual(list(windows(2)), [(0,2)])

    def test_missing_audio_is_unknown_and_volume_is_match_relative(self):
        self.assertEqual(measure_audio(None)['status'], 'unknown')
        self.assertEqual(measure_audio(np.zeros(32000))['status'], 'silent')
        wave = np.sin(np.arange(64000)/10).astype(np.float32)*.1
        a,b=measure_audio(wave),measure_audio(wave*2)
        self.assertAlmostEqual(a['windows'][0]['relative_db'],b['windows'][0]['relative_db'],places=5)
        self.assertIsNone(a['windows'][0]['laughter'])
        self.assertIsNone(a['windows'][0]['touch'])

    def test_camera_translation_is_removed_and_unknown_stays_unknown(self):
        before=[{'xyxy':[0,0,20,100], 'confidence':.9}]
        after=[{'xyxy':[10,0,30,100], 'confidence':.9}]
        self.assertIsNone(relative_motion(before,after,.5)['body_lengths_per_sec'])
        self.assertEqual(relative_motion(before,after,.5,np.array([[1,0,10],[0,1,0]]))['body_lengths_per_sec'],0)
        self.assertAlmostEqual(relative_motion(before,after,.5,np.array([[1,0,0],[0,1,0]]))['body_lengths_per_sec'],.2)

    def test_fabricated_refs_and_out_of_context_boundaries_rejected(self):
        e=event(); refs=[{'id':'f','kind':'frame','start_sec':3,'end_sec':3}]
        validate_events({'events':[e]},refs,0,10)
        bad=copy.deepcopy(e);bad['observations'][0]['evidence_ids']=['invented']
        with self.assertRaises(ValueError):validate_events({'events':[bad]},refs,0,10)
        bad=copy.deepcopy(e);bad['clip_end_sec']=11
        with self.assertRaises(ValueError):validate_events({'events':[bad]},refs,0,10)

    def test_laughter_requires_linked_audio_and_visual_facts(self):
        e=event();e['dimensions']['related_laughter']={'value':4,'evidence_ids':['f']}
        e['laughter_linked']=True
        refs=[{'id':'f','kind':'frame','start_sec':3,'end_sec':3}]
        with self.assertRaisesRegex(ValueError,'音画关联'):validate_events({'events':[e]},refs,0,10)
        e['observations'][0]['evidence_ids'].append('a');e['dimensions']['related_laughter']['evidence_ids'].append('a')
        refs.append({'id':'a','kind':'audio','start_sec':0,'end_sec':10})
        validate_events({'events':[e]},refs,0,10)
        e['laughter_linked']=False
        with self.assertRaises(ValueError):validate_events({'events':[e]},refs,0,10)

    def test_motion_cannot_be_guessed_from_frames(self):
        e=event();e['dimensions']['motion_intensity']={'value':4,'evidence_ids':['f']}
        with self.assertRaisesRegex(ValueError,'相机补偿'):validate_events({'events':[e]},[{'id':'f','kind':'frame','start_sec':3,'end_sec':3}],0,10)

    def test_unknown_score_is_not_zero_and_laughter_never_changes_athletics(self):
        e=timeline_event();before=score_event(e,'highlights')
        self.assertEqual(before['score'],75)
        self.assertIn('motion_intensity',before['unknown_dimensions'])
        e['dimensions']['related_laughter']['value']=4
        self.assertEqual(score_event(e,'highlights'),before)
        self.assertEqual(score_event(e,'bloopers')['score'],81.25)

    def test_short_excellent_beats_long_ordinary_without_time_penalty(self):
        short=timeline_event(start=2,end=4)
        long=timeline_event(start=10,end=60,event_id='long')
        for k in long['dimensions']:
            if long['dimensions'][k]['value'] is not None:long['dimensions'][k]['value']=1
        self.assertEqual(select_events([long,short],'highlights',5)[0][0]['event_id'],short['event_id'])
        equal=timeline_event(start=8,end=10,event_id='nearby')
        self.assertEqual(len(select_events([equal,short],'highlights',5)),2)

    def test_no_laughter_can_qualify_but_ordinary_mistake_or_injury_cannot(self):
        e=timeline_event()
        self.assertEqual(len(select_events([e],'bloopers',5)),1)
        for uncertainty in (True,None):
            self.assertFalse(select_events([{**e,'injury_suspected':uncertainty}],'bloopers',5))
        e['dimensions']['unexpected_contrast']['value']=1
        self.assertFalse(select_events([e],'bloopers',5))

    def test_dedup_does_not_add_evidence_or_scores_and_preserves_source_clocks(self):
        first=timeline_event(review_status='unreviewed',confidence=.95)
        second=timeline_event(confidence=.8)
        merged=merge_events([first,second])
        self.assertEqual(len(merged),1);self.assertEqual(merged[0]['review_status'],'reviewed')
        self.assertEqual(len(merged[0]['observations']),1)
        other={**second,'source_id':'set2'}
        self.assertEqual(len(merge_events([first,other])),2)

    def test_budget_stops_queue_and_records_failures(self):
        jobs=[{'id':str(i)} for i in range(20)]
        def worker(job):
            time.sleep(.03)
            if job['id']=='0':raise ValueError('private remote body')
            return job
        results,failures=bounded_map(jobs,worker,2,time.monotonic()+.01)
        self.assertLessEqual(len(results)+len(failures),2)
        self.assertNotIn('private',json.dumps(failures))

    def test_event_clip_contract_keeps_setup_and_reaction_without_three_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest={'source':{'duration_sec':20,'path':'unused'},'config':read_json(APP/'defaults.json'),'rallies':[]}
            decision=collection_artifacts(manifest,{'events':[timeline_event()]},root,'bloopers',5)
            self.assertEqual(decision['actual_count'],1)
            self.assertTrue(decision['shortage_reason'])
            m=read_json(root/'match_manifest.json')
            validate_decision(decision,m,root,1)
            bad=copy.deepcopy(decision);bad['selected'][0]['clip_start_sec']=2
            with self.assertRaisesRegex(ValueError,'铺垫'):validate_decision(bad,m,root,1)
            bad=copy.deepcopy(decision);bad['selected'][0]['clip_end_sec']=5
            with self.assertRaisesRegex(ValueError,'反应'):validate_decision(bad,m,root,1)
            segments=build_timeline(decision,m)
            self.assertEqual(segments[0]['peak_sec'],3)
            self.assertEqual(sum(s['kind']=='teaser' for s in segments),1)
            for segment in segments: segment['path']=str(root/'edit_decision.json')
            report={'segments':segments,'expected_duration_sec':sum(s['duration_sec'] for s in segments)}
            validate_timeline(report,decision)

    def test_shared_decoder_spool_supplies_frames_without_redecoding(self):
        from fractions import Fraction
        from volleymole.video import FramePacket
        from volleymole.event_frames import tap
        from volleymole.semantic import sampled_evidence
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            packets=[FramePacket(i,i,Fraction(1,10),0.,np.zeros((90,160,3),dtype=np.uint8)) for i in range(31)]
            self.assertEqual(len(list(tap(iter(packets),root,8))),31)
            with patch('av.open',side_effect=AssertionError('must reuse shared decode')):
                refs,content,frames=sampled_evidence({'path':'unused'},0,3,2,frame_cache=root,deadline=time.monotonic()+5)
            self.assertEqual(len(frames),6)
            self.assertEqual([round(r['start_sec'],1) for r in refs],[0,.5,1,1.5,2,2.5])
            first=root/'frame_000000000.jpg';first.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                sampled_evidence({'path':'unused'},0,3,2,frame_cache=root,deadline=time.monotonic()+5)

    def test_empty_collection_is_reported_without_filler(self):
        with tempfile.TemporaryDirectory() as tmp:
            m={'source':{'duration_sec':20},'config':read_json(APP/'defaults.json'),'rallies':[]}
            d=collection_artifacts(m,{'events':[]},tmp,'bloopers',5)
            self.assertEqual(d['actual_count'],0);self.assertEqual(d['selected'],[])


if __name__=='__main__':unittest.main()
