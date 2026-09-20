"""Offline counterexamples for selection policy, not evidence of model accuracy."""
import copy
import json
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch

from volleymole.common import identity
from volleymole.replay_review import review_rally, verification_problem


def sample(source, start, end, fps, width, **kwargs):
    import base64
    import cv2
    import numpy as np
    _, encoded = cv2.imencode('.jpg', np.zeros((24, 48, 3), dtype=np.uint8))
    image = 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode()
    evidence, content = [], []
    for index in range(round((end-start)*fps)):
        when = start+index/fps
        if when >= end:
            break
        fid = f'frame_{index:05d}'
        evidence.append(dict(id=fid, kind='frame', start_sec=when, end_sec=when))
        content.extend([dict(type='text', text=f'{fid} source_sec={when:.6f}'),
                        dict(type='image_url', image_url=dict(url=image))])
    return evidence, content, []


class CounterexampleModel:
    def __init__(self, candidates, overrides=None):
        self.candidates = candidates
        self.overrides = overrides or {}
        self.calls = []

    def __call__(self, endpoint, key, wire, timeout):
        self.calls.append(copy.deepcopy(wire))
        parts = wire['messages'][1]['content']
        context = json.loads(parts[0]['text'])
        refs = {m[1]: float(m[2]) for part in parts
                if (m := re.fullmatch(r'(frame_\d+) source_sec=([\d.]+)', part.get('text', '')))}
        fid = lambda when: min(refs, key=lambda name: abs(refs[name]-when))
        phase = context['phase']
        if phase == 'scan':
            rows = [dict(action=action, preparation_frame_id=fid(peak-1), peak_frame_id=fid(peak),
                         result_frame_id=fid(peak+1), excitement=score, confidence=.9,
                         observation='A visible candidate merits closer inspection.')
                    for peak, action, score in self.candidates
                    if min(refs.values()) <= peak-1 and max(refs.values()) >= peak+1]
            return dict(candidates=rows, uncertainty=''), dict(finish_reason='stop')
        focus = context['candidate']['peak_sec'] if phase == 'review' else context['focus_sec']
        candidate = min(self.candidates, key=lambda row: abs(row[0]-focus))
        original_peak, action, score = candidate
        override = self.overrides.get(original_peak, {}).get(phase, {})
        peak = override.get('peak_sec', original_peak)
        action = override.get('action', action)
        result = peak+(2 if action in ('dig', 'receive') else 1)
        raw = dict(verdict=override.get('verdict', 'accept'), action=action,
                   lead_frame_id=fid(peak-2), preparation_frame_id=fid(peak-1), peak_frame_id=fid(peak),
                   result_frame_id=fid(result), tail_frame_id=fid(result+.5),
                   need_before=False, need_after=False, start_complete=True, end_complete=True,
                   excitement=override.get('excitement', score), confidence=override.get('confidence', .9),
                   preparation_observation='The preparation is visible.',
                   contact_observation='The contact is visible.', result_observation='Later handling is visible.',
                   reason='PRIVATE_DRAFT_REASON' if phase == 'review' else 'Independent observation.',
                   uncertainty='Distant contact is uncertain.' if override.get('confidence', .9) < .65 else '')
        if phase == 'verify':
            contact = {'spike': 'single_arm_attack', 'dig': 'one_hand_save',
                       'receive': 'forearm_defense', 'set': 'two_hand_set'}[action]
            raw.update(contact_type=override.get('contact_type', contact),
                       outcome_type='team_return' if action in ('dig', 'receive') else 'opponent_response',
                       contact_before_frame_id=fid(peak-.125), contact_after_frame_id=fid(peak+.125),
                       next_touch_frame_id=fid(result), outcome_before_frame_id=fid(result-.125),
                       outcome_after_frame_id=fid(result+.125), opening_in_motion=False, ending_in_motion=False)
        return raw, dict(finish_reason='stop')


class UncertainSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root/'source.mp4'
        path.write_bytes(b'offline source identity fixture')
        self.source = dict(path=str(path), identity=identity(path), duration_sec=20.,
                           nominal_fps='30/1', start_sec=0., rotation=0)
        self.item = dict(rank=1, rally_id='fixture', clip_start_sec=0., clip_end_sec=20.)
        self.settings = dict(endpoint='https://example.invalid/v1', model='offline-counterexample',
            key='FAKE_KEY', timeout=10., max_tokens=2048, reasoning_effort=None,
            scan_fps=4., review_fps=8., verify_fps=8., width=768, chunk_sec=12., overlap_sec=2.,
            max_context_sec=18., max_expansions=2, max_candidates=3,
            min_confidence=.65, min_excitement=3, retries=0)
        self.sampling = patch('volleymole.replay_review.sampled_evidence', side_effect=sample)
        self.sampling.start()
        self.addCleanup(self.sampling.stop)
        self.runs = 0

    def run_model(self, model, *, cache=None, code=None):
        self.runs += 1
        with patch('volleymole.replay_review.request_json', side_effect=model):
            return review_rally(self.item, self.source, self.settings,
                cache or self.root/f'cache_{self.runs}', time.monotonic()+30, code=code)

    def test_uncertain_better_save_blocks_ordinary_approved_attack(self):
        model = CounterexampleModel([(6., 'spike', 3), (14., 'dig', 5)],
                                    {14.: {'verify': {'confidence': .6}}})
        result = self.run_model(model)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['review']['status'], 'omit')
        self.assertEqual(result['provisional_review']['peak_sec'], 6.)
        pending = result['unresolved_boundaries'][0]
        self.assertEqual(pending['candidate_peak_sec'], 14.)
        self.assertEqual(pending['scan_excitement'], 5)
        self.assertEqual(pending['reason'], 'insufficient_visual_confidence')
        self.assertTrue(pending['problem']['unresolved'])
        self.assertNotIn('reject', pending['problem'])
        self.assertEqual(len([x for x in result['attempts'] if x['phase'] == 'verify']), 2)
        self.assertEqual(self.settings['min_confidence'], .65)

    def test_low_confidence_reject_is_unresolved_in_either_phase(self):
        for phase in ('review', 'verify'):
            with self.subTest(phase=phase):
                model = CounterexampleModel([(6., 'spike', 3), (14., 'dig', 5)],
                    {14.: {phase: {'verdict': 'reject', 'confidence': .5}}})
                result = self.run_model(model)
                self.assertEqual(result['status'], 'incomplete')
                self.assertEqual(result['review']['status'], 'omit')
                self.assertEqual(result['unresolved_boundaries'][0]['phase'], phase)

    def test_supported_negative_does_not_block_another_valid_candidate(self):
        model = CounterexampleModel([(6., 'spike', 3), (14., 'spike', 5)],
            {14.: {'review': {'verdict': 'reject', 'action': 'set', 'confidence': .95}}})
        result = self.run_model(model)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['review']['peak_sec'], 6.)
        self.assertEqual(result['unresolved_boundaries'], [])

    def test_lower_potential_uncertainty_does_not_erase_better_verified_save(self):
        model = CounterexampleModel([(6., 'spike', 3), (14., 'dig', 5)],
                                    {6.: {'verify': {'confidence': .6}}})
        result = self.run_model(model)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['review']['peak_sec'], 14.)
        self.assertEqual(len(result['unresolved_boundaries']), 1)

    def test_low_confidence_draft_can_be_resolved_by_independent_verify(self):
        model = CounterexampleModel([(10., 'dig', 5)], {10.: {'review': {'confidence': .5}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertEqual(result['review']['confidence'], .9)

    def test_same_contact_defense_attack_flip_is_unresolved_in_both_directions(self):
        for draft, verified in (('dig', 'spike'), ('receive', 'spike'),
                                ('spike', 'dig'), ('spike', 'receive')):
            with self.subTest(draft=draft, verified=verified):
                model = CounterexampleModel([(10., draft, 4)], {10.: {'verify': {'action': verified}}})
                result = self.run_model(model)
                self.assertEqual(result['status'], 'incomplete')
                self.assertEqual(result['review']['status'], 'omit')
                self.assertEqual(result['verified_candidates'], [])
                self.assertEqual(result['unresolved_boundaries'][0]['reason'],
                                 'conflicting_action_families_for_same_contact')
                calls = [w for w in model.calls if json.loads(w['messages'][1]['content'][0]['text'])['phase'] == 'verify']
                self.assertEqual(len(calls), 1)
                context = json.loads(calls[0]['messages'][1]['content'][0]['text'])
                self.assertIsNone(context['candidate'])
                self.assertIsNone(context['previous_boundary_problem'])
                self.assertNotIn('PRIVATE_DRAFT_REASON', json.dumps(calls[0]))
                self.assertNotIn('conflicting_action_families', json.dumps(calls[0]))

    def test_half_second_contact_flip_is_still_the_same_contact(self):
        model = CounterexampleModel([(10., 'dig', 4)],
            {10.: {'verify': {'action': 'spike', 'peak_sec': 10.5}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'omit')
        self.assertEqual(result['unresolved_boundaries'][0]['reason'],
                         'conflicting_action_families_for_same_contact')

    def test_conflicting_better_save_prevents_ordinary_fallback(self):
        model = CounterexampleModel([(6., 'spike', 3), (14., 'dig', 5)],
                                    {14.: {'verify': {'action': 'spike'}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'omit')
        self.assertEqual(result['provisional_review']['peak_sec'], 6.)
        self.assertEqual(result['unresolved_boundaries'][0]['scan_excitement'], 5)
        self.assertEqual(result['unresolved_boundaries'][0]['reason'],
                         'conflicting_action_families_for_same_contact')

    def test_equal_score_uncertain_defense_blocks_ordinary_attack(self):
        for action in ('dig', 'receive'):
            with self.subTest(action=action):
                model = CounterexampleModel([(6., 'spike', 3), (14., action, 3)],
                                            {14.: {'verify': {'confidence': .6}}})
                result = self.run_model(model)
                self.assertEqual(result['review']['status'], 'omit')
                self.assertEqual(result['provisional_review']['action'], 'spike')
                self.assertEqual(result['unresolved_boundaries'][0]['scan_action'], action)

    def test_dense_defense_hypothesis_blocks_equal_score_coarse_spike(self):
        model = CounterexampleModel([(6., 'spike', 4), (14., 'spike', 4)],
            {14.: {'review': {'action': 'dig'}, 'verify': {'action': 'dig', 'confidence': .6}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'omit')
        pending = result['unresolved_boundaries'][0]
        self.assertEqual(pending['scan_action'], 'spike')
        self.assertEqual(pending['potential_actions'], ['dig', 'spike'])
        self.assertEqual(pending['potential_excitement'], 4)
        self.assertEqual(result['provisional_review']['peak_sec'], 6.)

    def test_dense_higher_excitement_is_retained_when_confidence_is_low(self):
        model = CounterexampleModel([(6., 'spike', 4), (14., 'dig', 3)],
            {14.: {'review': {'excitement': 5}, 'verify': {'excitement': 5, 'confidence': .6}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'omit')
        pending = result['unresolved_boundaries'][0]
        self.assertEqual(pending['scan_excitement'], 3)
        self.assertEqual(pending['potential_excitement'], 5)
        self.assertEqual(result['provisional_review']['excitement'], 4)
        self.assertFalse(any(r['peak_sec'] == 14. for r in result['verified_candidates']))

    def test_reverse_class_flip_retains_defense_potential_for_tied_selection(self):
        model = CounterexampleModel([(6., 'spike', 4), (14., 'spike', 4)],
                                    {14.: {'verify': {'action': 'dig'}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'omit')
        pending = result['unresolved_boundaries'][0]
        self.assertEqual(pending['reason'], 'conflicting_action_families_for_same_contact')
        self.assertIn('dig', pending['potential_actions'])
        self.assertEqual(result['provisional_review']['peak_sec'], 6.)

    def test_defense_hypothesis_with_lower_potential_does_not_block_better_attack(self):
        model = CounterexampleModel([(6., 'spike', 4), (14., 'spike', 3)],
            {14.: {'review': {'action': 'dig'}, 'verify': {'action': 'dig', 'confidence': .6}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertEqual(result['review']['peak_sec'], 6.)
        self.assertEqual(result['unresolved_boundaries'][0]['potential_excitement'], 3)
        self.assertEqual(result['unresolved_boundaries'][0]['potential_actions'], ['dig', 'spike'])

    def test_different_contact_may_change_action_after_independent_refocus(self):
        model = CounterexampleModel([(10., 'spike', 4)],
            {10.: {'verify': {'action': 'dig', 'peak_sec': 10.75}}})
        result = self.run_model(model)
        self.assertEqual(result['review']['status'], 'approved')
        self.assertEqual(result['review']['action'], 'dig')
        self.assertEqual(result['review']['peak_sec'], 10.75)
        attempts = [x for x in result['attempts'] if x['phase'] == 'verify']
        self.assertEqual([x['candidate_peak_sec'] for x in attempts], [10., 10.75])
        self.assertEqual(attempts[0]['quality_problem']['reason'], 'contact_requires_refocused_detail')

    def test_policy_changes_reuse_identical_phase_requests(self):
        model = CounterexampleModel([(10., 'dig', 4)])
        cache = self.root/'shared_cache'
        first = self.run_model(model, cache=cache, code={'policy': 'before'})
        count = len(model.calls)
        second = self.run_model(model, cache=cache, code={'policy': 'after'})
        self.assertNotEqual(first['signature'], second['signature'])
        self.assertEqual(len(model.calls), count)
        self.assertTrue(all(row['cached'] for row in second['requests']))

    def test_low_confidence_quality_finding_is_not_a_semantic_reject(self):
        # The confidence finding is independent of anchors; it must not pretend
        # to have disproved an exciting action that the camera cannot resolve.
        problem = verification_problem(dict(confidence=.6), self.settings)
        self.assertEqual(problem, dict(unresolved=True, reason='insufficient_visual_confidence'))


if __name__ == '__main__':
    unittest.main()
