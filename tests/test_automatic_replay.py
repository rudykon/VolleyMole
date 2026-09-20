"""Offline protocol/orchestration tests; fake model judgments are not accuracy evidence."""
import copy
import json
import os
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch

from volleymole.common import identity, read_json, save_json
from volleymole.replay_review import review_rally, distinct_candidates
from volleymole.replay_review_schema import decode_review, decode_scan, payload
from volleymole.replay_stage import run_review, load_reviewed_manifest
from volleymole.presentation import build_timeline, validate_replay_evidence
from volleymole.run_match import argument_parser


def sample(source, start, end, fps, width, **kwargs):
    import base64
    import cv2
    import numpy as np
    _, encoded = cv2.imencode('.jpg', np.zeros((32,64,3), dtype=np.uint8))
    jpeg = base64.b64encode(encoded).decode()
    evidence = []; content = []
    for n in range(round((end-start)*fps)):
        t = start+n/fps
        if t >= end:
            break
        fid = f'frame_{n:05d}'
        evidence.append(dict(id=fid, kind='frame', start_sec=t, end_sec=t))
        content.extend([dict(type='text', text=f'{fid} source_sec={t:.6f}'),
                        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+jpeg))])
    return evidence, content, []


class Model:
    def __init__(self, result_sec=11., only_set=False, fail=False):
        self.calls = []; self.result_sec = result_sec; self.only_set = only_set; self.fail = fail

    def __call__(self, endpoint, key, payload, timeout):
        self.calls.append(copy.deepcopy(payload))
        content = payload['messages'][1]['content']; context = json.loads(content[0]['text'])
        refs = {}
        for part in content:
            match = re.fullmatch(r'(frame_\d+) source_sec=([\d.]+)', part.get('text', ''))
            if match:
                refs[match[1]] = float(match[2])
        def fid(when):return min(refs, key=lambda k: abs(refs[k]-when))
        if context['phase'] == 'scan':
            candidate = dict(action='set' if self.only_set else 'spike', preparation_frame_id=fid(8.),
                peak_frame_id=fid(10.), result_frame_id=fid(11.), excitement=4, confidence=.9,
                observation='Visible approach, contact and ball flight.')
            result = dict(candidates=[candidate] if min(refs.values()) < 8. and max(refs.values()) > 11. else [], uncertainty='')
        else:
            if self.fail:
                raise TimeoutError('a private provider message must not enter audit')
            expand = max(refs.values()) < self.result_sec+.5
            result = dict(verdict='expand' if expand else 'accept', action='spike',
                lead_frame_id=fid(8.), preparation_frame_id=fid(9.), peak_frame_id=fid(10.),
                result_frame_id=None if expand else fid(self.result_sec), tail_frame_id=None if expand else fid(self.result_sec+.5),
                need_before=False, need_after=expand, start_complete=True, end_complete=not expand,
                excitement=4, confidence=.9, preparation_observation='Approach visible.',
                contact_observation='Arm swing and outgoing ball visible.', result_observation='' if expand else 'Landing and ball outcome visible.',
                reason='Need later ball outcome.' if expand else 'Complete attack sequence.', uncertainty='')
        if context['phase'] == 'verify':
            result.update(contact_type='single_arm_attack', outcome_type='opponent_response',
                contact_before_frame_id=fid(9.875), contact_after_frame_id=fid(10.125),
                next_touch_frame_id=None if expand else fid(self.result_sec),
                outcome_before_frame_id=None if expand else fid(self.result_sec-.125),
                outcome_after_frame_id=None if expand else fid(self.result_sec+.125),
                opening_in_motion=False, ending_in_motion=False)
        return result, dict(finish_reason='stop')


class AutomaticReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); video = self.root/'fixture.mp4'; video.write_bytes(b'stable source fixture')
        self.source = dict(path=str(video), identity=identity(video), duration_sec=20., nominal_fps='30/1', start_sec=0., rotation=0)
        self.item = dict(rank=1, rally_id='r1', clip_start_sec=0., clip_end_sec=20., title='回合')
        self.settings = dict(endpoint='https://example.invalid/v1', model='configured-vision', key='SECRET_FIXTURE_KEY',
            timeout=10., max_tokens=2048, reasoning_effort=None, scan_fps=4., review_fps=8., width=768,
            chunk_sec=12., overlap_sec=2., max_context_sec=18., max_expansions=2, max_candidates=3,
            min_confidence=.65, min_excitement=3, retries=0)
        self.sampler = patch('volleymole.replay_review.sampled_evidence', side_effect=sample)
        self.sampler.start(); self.addCleanup(self.sampler.stop)

    def run_model(self, model, source=None, settings=None, motion=None):
        with patch('volleymole.replay_review.request_json', side_effect=model):
            return review_rally(self.item, source or self.source, settings or self.settings,
                                self.root/'cache', time.monotonic()+30, motion=motion)

    def test_scans_whole_rally_then_expands_past_old_cut_to_observed_result(self):
        model = Model(result_sec=14.)
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertGreaterEqual(result['review']['source_end_sec'], 14.5)
        self.assertEqual(result['review']['boundary_expansions'], 1)
        self.assertEqual([r['context'] for r in result['requests'] if r['phase']=='scan'], [[0.,12.],[10.,20.]])
        self.assertEqual(len([r for r in result['requests'] if r['phase']=='review']), 2)
        self.assertNotIn('SECRET_FIXTURE_KEY', json.dumps(model.calls))
        self.assertNotIn('SECRET_FIXTURE_KEY', Path(result['artifact']).read_text())
        for call in model.calls:
            context = json.loads(call['messages'][1]['content'][0]['text'])
            if context['candidate']:
                self.assertFalse(any(k.endswith('_frame_id') for k in context['candidate']))
                self.assertIn('peak_sec', context['candidate'])

    def test_set_only_or_unseen_result_does_not_become_reviewed_highlight(self):
        for model in (Model(only_set=True), Model(result_sec=25.)):
            with self.subTest(model=model.only_set):
                changed = {**self.settings, 'model':str(model.only_set)}
                result = self.run_model(model, settings=changed)
                self.assertEqual(result['review']['status'], 'omit')
                self.assertEqual(result['status'], 'complete' if model.only_set else 'incomplete')

    def test_short_draft_is_replaced_by_independently_verified_anchors(self):
        model = Model()
        review_calls = []
        def short_then_complete(endpoint, key, wire, timeout):
            raw, metadata = model(endpoint, key, wire, timeout)
            parts = wire['messages'][1]['content']
            job = json.loads(parts[0]['text'])
            if job['phase'] == 'review':
                review_calls.append(job)
                if len(review_calls) == 1:
                    refs = dict((m[1], float(m[2])) for p in parts
                        if (m := re.fullmatch(r'(frame_\d+) source_sec=([\d.]+)', p.get('text', ''))))
                    nearest = lambda t: min(refs, key=lambda f: abs(refs[f]-t))
                    raw.update(lead_frame_id=nearest(9.25), preparation_frame_id=nearest(9.5),
                               result_frame_id=nearest(10.75), tail_frame_id=nearest(11.125))
            return raw, metadata
        result = self.run_model(short_then_complete)
        self.assertEqual(len(review_calls), 1)
        verified = [r for r in result['attempts'] if r['phase']=='verify']
        self.assertEqual(len(verified), 1)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertEqual(result['review']['source_start_sec'], 8.)
        self.assertGreaterEqual(result['review']['source_end_sec'], 11.5)
        self.assertEqual(result['review']['boundary_expansions'], 0)
        self.assertEqual(result['review']['anchor_rechecks'], 0)

    def test_uncertain_scan_candidate_reaches_review_but_cannot_approve_itself(self):
        model = Model()
        def uncertain_scan(endpoint, key, wire, timeout):
            raw, metadata = model(endpoint, key, wire, timeout)
            if 'candidates' in raw:
                for row in raw['candidates']: row['confidence'] = .5
            return raw, metadata
        result = self.run_model(uncertain_scan)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertEqual(result['review']['confidence'], .9)
        self.assertTrue(result['attempts'])

    def test_unresolved_boundary_is_not_a_completed_negative_review(self):
        args=self.stage_fixture()
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fixture-key'}), \
                patch('volleymole.replay_review.request_json',side_effect=Model(result_sec=25.)):
            with self.assertRaisesRegex(RuntimeError,'未全部完成'):
                run_review(self.root,args)
            report=read_json(self.root/'replay_reviews.json')
            self.assertEqual(report['status'],'partial')
            self.assertEqual(report['failed_count'],1)
            self.assertEqual(report['omitted_count'],0)
            self.assertEqual(report['results']['r1']['failure']['error'],'boundary_review_incomplete')
            with self.assertRaisesRegex(ValueError,'不能绕过复核'):
                load_reviewed_manifest(self.root)

    def test_expansion_may_resume_with_a_larger_budget(self):
        model=Model(result_sec=16.)
        limited=self.run_model(model,settings={**self.settings,'max_expansions':1})
        self.assertEqual(limited['status'],'incomplete')
        self.assertEqual(limited['unresolved_boundaries'][-1]['reason'],'expansion_limit')
        complete=self.run_model(model,settings={**self.settings,'max_expansions':2})
        self.assertEqual(complete['status'],'complete')
        self.assertEqual(complete['review']['status'],'approved')
        self.assertGreaterEqual(complete['review']['source_end_sec'],16.5)

    def test_cache_reuses_frames_but_changed_model_or_source_rechecks(self):
        model = Model(); first = self.run_model(model)
        count = len(model.calls); again = self.run_model(model)
        self.assertTrue(again['cached']); self.assertEqual(len(model.calls), count)
        self.run_model(model, settings={**self.settings, 'model':'another-explicit-model'})
        self.assertGreater(len(model.calls), count); count = len(model.calls)
        path = self.root/'second.mp4'; path.write_bytes(b'different source with same local clock')
        second = self.run_model(model, source={**self.source, 'path':str(path), 'identity':identity(path)})
        self.assertGreater(len(model.calls), count)
        self.assertNotEqual(first['review']['source_sha256'], second['review']['source_sha256'])

    def test_failed_review_resumes_from_successful_scan_requests(self):
        model = Model(fail=True)
        with self.assertRaises(TimeoutError):self.run_model(model)
        model.fail = False; model.calls.clear()
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([json.loads(c['messages'][1]['content'][0]['text'])['phase'] for c in model.calls],
                         ['review', 'verify'])

    def test_modified_cached_window_is_rebuilt_from_actual_model_anchors(self):
        model=Model(); result=self.run_model(model)
        saved=read_json(result['artifact']); saved['review']['source_end_sec']+=1
        save_json(result['artifact'],saved)
        restored=self.run_model(model)
        self.assertEqual(restored['review'],result['review'])

    def test_real_pts_sampling_reaches_model_and_returns_renderable_timeline(self):
        import av
        import numpy as np
        video=Path(self.source['path'])
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=10); stream.width=160; stream.height=90; stream.pix_fmt='yuv420p'
            for n in range(200):
                pixels=np.full((90,160,3),n%200,dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        self.source.update(identity=identity(video),nominal_fps='10/1')
        self.sampler.stop()
        model=Model(result_sec=14.)
        result=self.run_model(model)
        self.assertEqual(result['review']['status'],'approved')
        self.assertGreater(result['review']['source_end_sec'],14.)
        images=[p for c in model.calls for p in c['messages'][1]['content'] if p['type']=='image_url']
        self.assertGreater(len(images),100)
        self.assertTrue(all(len(p['image_url']['url'])>500 for p in images))
        manifest=dict(source=self.source,rallies=[dict(rally_id='r1',preview_times_sec=[0.,3.,20.],replay_review=result['review'])])
        rows=build_timeline(dict(selected=[self.item]),manifest)
        replay=next(r for r in rows if r['kind']=='replay')
        self.assertGreater(replay['duration_sec'],9.)

    def test_save_can_outrank_ordinary_spike_and_overlap_duplicates_collapse(self):
        rows = [dict(action='spike',peak_sec=4.,excitement=3,confidence=.95),
                dict(action='dig',peak_sec=10.,excitement=5,confidence=.85),
                dict(action='dig',peak_sec=10.3,excitement=5,confidence=.75),
                dict(action='set',peak_sec=1.,excitement=5,confidence=1.)]
        chosen = distinct_candidates(rows,self.settings)
        self.assertEqual([c['action'] for c in chosen], ['dig','spike'])

    def test_missing_sample_interval_is_rejected_before_upload(self):
        def broken(*a, **kw):
            e,c,f = sample(*a,**kw); del e[7:13]; return e,c,f
        with patch('volleymole.replay_review.sampled_evidence',side_effect=broken), \
                patch('volleymole.replay_review.request_json') as request:
            with self.assertRaisesRegex(ValueError,'sampling_gap'):
                review_rally(self.item,self.source,self.settings,self.root/'cache',time.monotonic()+30)
            request.assert_not_called()

    def test_jump_set_claiming_spike_is_rejected_without_retrying_for_approval(self):
        model = Model()
        def jumping_set(endpoint,key,wire,timeout):
            raw,metadata = model(endpoint,key,wire,timeout)
            if json.loads(wire['messages'][1]['content'][0]['text'])['phase']=='verify':
                raw['contact_type'] = 'two_hand_set'
            return raw,metadata
        result = self.run_model(jumping_set,settings={**self.settings,'retries':1})
        self.assertEqual(result['review']['status'],'omit')
        self.assertEqual(result['status'],'complete')
        verify = [r for r in result['attempts'] if r['phase']=='verify']
        self.assertEqual(len(verify),1)
        self.assertTrue(verify[0]['quality_problem']['reject'])
        self.assertEqual(verify[0]['result']['action'],'spike')
        self.assertEqual(verify[0]['result']['contact_type'],'two_hand_set')

    def test_unknown_outcome_or_motion_at_tail_never_passes_quality_gate(self):
        for changed in (dict(outcome_type='unknown'),dict(ending_in_motion=True),
                        dict(next_touch_frame_id=None)):
            with self.subTest(changed=changed):
                model = Model()
                def unresolved(endpoint,key,wire,timeout):
                    raw,metadata = model(endpoint,key,wire,timeout)
                    if json.loads(wire['messages'][1]['content'][0]['text'])['phase']=='verify':
                        raw.update(changed)
                    return raw,metadata
                result = self.run_model(unresolved,settings={**self.settings,'model':str(changed),'max_expansions':0})
                self.assertEqual(result['review']['status'],'omit')
                self.assertEqual(result['status'],'incomplete')
                self.assertEqual(result['verified_candidates'],[])
                self.assertTrue(result['unresolved_boundaries'][-1]['problem']['need_after'])

    def test_independent_verifier_does_not_receive_draft_judgments_or_anchors(self):
        model = Model()
        def draft_with_private_hypothesis(endpoint,key,wire,timeout):
            raw,metadata = model(endpoint,key,wire,timeout)
            if json.loads(wire['messages'][1]['content'][0]['text'])['phase']=='review':
                raw['reason']='PRIVATE_DRAFT_REASON_ONLY'
                raw['contact_observation']='PRIVATE_DRAFT_CONTACT_ONLY'
            return raw,metadata
        result = self.run_model(draft_with_private_hypothesis)
        self.assertEqual(result['review']['status'],'approved')
        verify = [w for w in model.calls if json.loads(w['messages'][1]['content'][0]['text'])['phase']=='verify']
        self.assertEqual(len(verify),1)
        metadata = json.loads(verify[0]['messages'][1]['content'][0]['text'])
        self.assertEqual(metadata['focus_sec'],10.)
        self.assertIsNone(metadata['candidate'])
        self.assertIsNone(metadata['previous_boundary_problem'])
        self.assertFalse(set(metadata)&{'action','confidence','excitement','reason','times','lead_frame_id','peak_frame_id'})
        self.assertNotIn('PRIVATE_DRAFT_',json.dumps(verify[0]))
        self.assertNotEqual(result['review']['reason'],'PRIVATE_DRAFT_REASON_ONLY')

    def test_short_verified_claim_gets_handles_from_observed_pts_without_changing_events(self):
        model = Model(); calls = []
        def short_then_complete(endpoint,key,wire,timeout):
            raw,metadata = model(endpoint,key,wire,timeout)
            parts = wire['messages'][1]['content']; job=json.loads(parts[0]['text'])
            if job['phase']=='verify':
                calls.append(job)
                if len(calls)==1:
                    refs = {m[1]:float(m[2]) for p in parts
                            if (m:=re.fullmatch(r'(frame_\d+) source_sec=([\d.]+)',p.get('text','')))}
                    nearest=lambda t:min(refs,key=lambda f:abs(refs[f]-t))
                    raw.update(lead_frame_id=nearest(9.25),preparation_frame_id=nearest(9.5),
                               tail_frame_id=nearest(11.25))
            return raw,metadata
        result = self.run_model(short_then_complete)
        self.assertEqual(result['review']['status'],'approved')
        self.assertEqual(len(calls),1)
        verified = next(r for r in result['attempts'] if r['phase']=='verify')['result']
        request = next(r for r in result['requests'] if r['phase']=='verify')
        observed = [f['start_sec'] for f in read_json(request['path'])['evidence']]
        self.assertEqual(verified['times']['lead'],9.25)
        self.assertEqual(verified['times']['tail'],11.25)
        self.assertLess(result['review']['source_start_sec'],verified['times']['lead'])
        self.assertGreater(result['review']['source_end_sec'],verified['times']['tail'])
        self.assertIn(result['review']['source_start_sec'],observed)
        self.assertTrue(any(abs(t-(result['review']['source_end_sec']-1/30))<1e-8 for t in observed))
        self.assertEqual(result['review']['peak_sec'],verified['times']['peak'])
        self.assertEqual(result['review']['action_end_sec'],verified['times']['result'])
        self.assertGreaterEqual(result['review']['source_end_sec']-result['review']['peak_sec'],1.5)

    def test_all_shortlisted_candidates_are_verified_before_better_later_save_wins(self):
        calls = []
        def two_actions(endpoint,key,wire,timeout):
            calls.append(copy.deepcopy(wire))
            parts=wire['messages'][1]['content']; job=json.loads(parts[0]['text'])
            refs={m[1]:float(m[2]) for p in parts
                  if (m:=re.fullmatch(r'(frame_\d+) source_sec=([\d.]+)',p.get('text','')))}
            fid=lambda t:min(refs,key=lambda f:abs(refs[f]-t))
            if job['phase']=='scan':
                peak=6. if job['context'][0]<6. else 14.
                raw=dict(candidates=[dict(action='spike' if peak==6. else 'dig',
                    preparation_frame_id=fid(peak-2),peak_frame_id=fid(peak),result_frame_id=fid(peak+1),
                    excitement=5 if peak==6. else 4,confidence=.95 if peak==6. else .85,
                    observation='Independent visible action candidate.')],uncertainty='')
            else:
                peak=job['focus_sec'] if job['phase']=='verify' else job['candidate']['peak_sec']
                action='spike' if peak==6. else 'dig'; result_sec=peak+1
                raw=dict(verdict='accept',action=action,lead_frame_id=fid(peak-2),
                    preparation_frame_id=fid(peak-1),peak_frame_id=fid(peak),
                    result_frame_id=fid(result_sec),tail_frame_id=fid(result_sec+.5),
                    need_before=False,need_after=False,start_complete=True,end_complete=True,
                    excitement=3 if action=='spike' else 5,confidence=.85,
                    preparation_observation='Run starts after lead.',contact_observation='Actual contact visible.',
                    result_observation='Subsequent handling visible.',reason='Complete visible action.',uncertainty='')
                if job['phase']=='verify':
                    raw.update(contact_type='single_arm_attack' if action=='spike' else 'one_hand_save',
                        outcome_type='opponent_response' if action=='spike' else 'team_return',
                        contact_before_frame_id=fid(peak-.125),contact_after_frame_id=fid(peak+.125),
                        next_touch_frame_id=fid(result_sec),outcome_before_frame_id=fid(result_sec-.125),
                        outcome_after_frame_id=fid(result_sec+.125),opening_in_motion=False,ending_in_motion=False)
            return raw,dict(finish_reason='stop')
        result=self.run_model(two_actions)
        self.assertEqual([r['action'] for r in result['shortlisted']],['spike','dig'])
        self.assertEqual(len(result['verified_candidates']),2)
        self.assertEqual(result['review']['action'],'dig')
        self.assertAlmostEqual(result['review']['peak_sec'],14.)
        self.assertEqual(len([r for r in result['requests'] if r['phase']=='verify']),2)

    def test_motion_counterevidence_blocks_accepted_defense_before_next_touch(self):
        model=Model()
        def early_defense(endpoint,key,wire,timeout):
            raw,metadata=model(endpoint,key,wire,timeout)
            if 'candidates' in raw:
                for candidate in raw['candidates']:candidate['action']='dig'
            else:
                raw['action']='dig'
                if 'contact_type' in raw:raw['contact_type']='one_hand_save'
            return raw,metadata
        # A synthetic independently measured impulse occurs after the model's
        # 11.5s tail. Its existence must prevent rendering that accepted window.
        motion=dict(status='available',clip_start_sec=0.,clip_end_sec=20.,gaps=[],
            contact_hypotheses=[dict(start_sec=12.9,end_sec=13.1,time_sec=13.,observed_after_sec=13.3)],
            observed_flights=[dict(start_sec=11.2,end_sec=11.8,center_sec=11.5)])
        result=self.run_model(early_defense,settings={**self.settings,'max_expansions':0},motion=motion)
        self.assertEqual(result['review']['status'],'omit')
        self.assertEqual(result['status'],'incomplete')
        verify=next(r for r in result['attempts'] if r['phase']=='verify')
        self.assertEqual(verify['result']['verdict'],'accept')
        self.assertEqual(verify['motion_guard']['verdict'],'expand')
        self.assertGreater(verify['quality_problem']['minimum_source_end_sec'],13.)

    def stage_fixture(self, mode='required'):
        rally = dict(rally_id='r1',preview_times_sec=[0.,3.,20.])
        save_json(self.root/'match_manifest.json',dict(source=self.source,rallies=[rally]))
        save_json(self.root/'edit_decision.json',dict(selected=[self.item],ranking_mode='rules_fallback'))
        args=argument_parser().parse_args(['--video',self.source['path'],'--model','configured-vision',
            '--api-base','https://example.invalid/v1','--replay-review',mode,'--semantic-retries','0'])
        return args

    def test_stage_injects_review_into_renderer_and_detects_stale_decision(self):
        args=self.stage_fixture(); model=Model(result_sec=14.)
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fixture-key'}), \
                patch('volleymole.replay_review.request_json',side_effect=model):
            stage=run_review(self.root,args)
            self.assertEqual(stage['approved_count'],1)
            count=len(model.calls); run_review(self.root,args)
            self.assertEqual(len(model.calls),count)
        manifest,binding=load_reviewed_manifest(self.root)
        decision=read_json(self.root/'edit_decision.json'); rows=build_timeline(decision,manifest)
        replay=next(r for r in rows if r['kind']=='replay')
        self.assertGreater(replay['source_end_sec'],14.)
        validate_replay_evidence(dict(replay_policy_version=2,segments=rows),decision,manifest)
        self.assertNotIn('replay_review',read_json(self.root/'match_manifest.json')['rallies'][0])
        changed=read_json(self.root/'edit_decision.json'); changed['selected'][0]['clip_end_sec']=19.
        save_json(self.root/'edit_decision.json',changed)
        with self.assertRaisesRegex(ValueError,'剪辑单已改变'):load_reviewed_manifest(self.root,binding)

    def test_required_failure_stops_and_auto_failure_omits_unreviewed_replay(self):
        args=self.stage_fixture(); model=Model(fail=True)
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fixture-key'}), \
                patch('volleymole.replay_review.request_json',side_effect=model):
            with self.assertRaisesRegex(RuntimeError,'未全部完成'):run_review(self.root,args)
            report=read_json(self.root/'replay_reviews.json')
            self.assertEqual(report['status'],'partial')
            self.assertNotIn('private',json.dumps(report))
            with self.assertRaisesRegex(ValueError,'不能绕过复核'):load_reviewed_manifest(self.root)
            args.replay_review='auto'; run_review(self.root,args)
        manifest,_=load_reviewed_manifest(self.root)
        rows=build_timeline(read_json(self.root/'edit_decision.json'),manifest)
        self.assertFalse(any(r['kind']=='replay' for r in rows))

    def test_rules_mode_makes_no_remote_requests(self):
        args=self.stage_fixture('auto'); args.ranker='rules'
        with patch('volleymole.replay_review.request_json') as request:
            self.assertEqual(run_review(self.root,args)['status'],'disabled')
            request.assert_not_called()

    def test_valid_existing_review_can_be_used_without_network_credentials(self):
        review=self.run_model(Model())['review']; args=self.stage_fixture()
        # This fixture represents an existing assistant visual inspection; an
        # automatic model record must pass the current independent quality gate.
        review['method']='assistant_visual_review'
        manifest=read_json(self.root/'match_manifest.json'); manifest['rallies'][0]['replay_review']=review
        save_json(self.root/'match_manifest.json',manifest)
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'','OPENAI_API_KEY':''}), \
                patch('volleymole.replay_review.request_json') as request:
            result=run_review(self.root,args)
            self.assertEqual(result['approved_count'],1)
            self.assertEqual(result['results']['r1']['origin'],'existing_source_bound_review')
            request.assert_not_called()

    def test_changed_source_bytes_rejected_before_remote_call(self):
        args=self.stage_fixture(); Path(self.source['path']).write_bytes(b'changed file')
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fixture-key'}), \
                patch('volleymole.replay_review.request_json') as request:
            with self.assertRaisesRegex(ValueError,'原片身份'):run_review(self.root,args)
            request.assert_not_called()

    def test_source_replaced_between_review_and_render_is_rejected(self):
        args=self.stage_fixture()
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fixture-key'}), \
                patch('volleymole.replay_review.request_json',side_effect=Model()):
            run_review(self.root,args)
        Path(self.source['path']).write_bytes(b'replaced after review')
        with self.assertRaisesRegex(ValueError,'原片内容已改变'):load_reviewed_manifest(self.root)


class GroundingTests(unittest.TestCase):
    def test_unknown_frames_reversed_bounds_and_false_completion_rejected(self):
        evidence,content,_=sample({},0.,16.,8.,768)
        p=dict(messages=[{},dict(content=[dict(type='text',text=json.dumps(dict(phase='review')))]+content)])
        raw,_=Model()(None,None,p,10)
        decode_review(raw,evidence)
        for changes in ({'peak_frame_id':'frame_99999'},{'tail_frame_id':raw['lead_frame_id']},
                        {'need_after':True},{'end_complete':False},{'confidence':float('nan')},
                        {'excitement':True},{'result_observation':''}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                decode_review({**raw,**changes},evidence)

    def test_scan_unknown_results_are_explicit_but_invented_ids_fail(self):
        evidence,_,_=sample({},0.,12.,4.,768)
        row=dict(action='spike',preparation_frame_id='frame_00010',peak_frame_id='frame_00020',
                 result_frame_id=None,excitement=4,confidence=.9,observation='Contact observed; result outside view.')
        self.assertIsNone(decode_scan(dict(candidates=[row],uncertainty=''),evidence)[0]['action_end_sec'])
        with self.assertRaises(ValueError):
            decode_scan(dict(candidates=[{**row,'result_frame_id':'frame_99999'}],uncertainty=''),evidence)

    def test_contract_is_visible_when_gateway_does_not_enforce_response_format(self):
        evidence, content, _ = sample({}, 0., 3., 4., 768)
        for phase in ('scan', 'review'):
            wire = payload('configured-vision', phase, 'Review visible actions.', content, evidence, 2048)
            written = json.loads(wire['messages'][0]['content'].split('\n')[-1])
            self.assertEqual(written, wire['response_format']['json_schema']['schema'])
        row = dict(action='spike', preparation_frame_id='frame_00001', peak_frame_id=4,
                   result_frame_id='frame_00008', excitement=4, confidence=.9, observation='Visible action.')
        with self.assertRaisesRegex(ValueError, 'replay_unknown_frame'):
            decode_scan(dict(candidates=[row], uncertainty=''), evidence)


if __name__=='__main__':unittest.main()
