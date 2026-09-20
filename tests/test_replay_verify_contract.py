"""Verifier protocol contracts; synthetic anchors do not prove visual accuracy."""
import copy
import math
import unittest

from volleymole.replay_requests import evidence_valid
from volleymole.replay_review_schema import VERIFY_SCHEMA, decode_verify, payload


class ReplayVerifyContractTests(unittest.TestCase):
    def setUp(self):
        # Irregular *actual* PTS deliberately differ from nominal frame/fps.
        self.times = [10.01,10.14,10.26,10.39,10.51,10.64,10.76,10.89,
                      11.01,11.14,11.26,11.39,11.51,11.64,11.76,11.89]
        self.evidence = [dict(id=f'frame_{index:05d}',kind='frame',start_sec=when,end_sec=when)
                         for index,when in enumerate(self.times)]
        self.valid = dict(verdict='accept',action='spike',lead_frame_id='frame_00000',
            preparation_frame_id='frame_00002',peak_frame_id='frame_00005',
            result_frame_id='frame_00011',tail_frame_id='frame_00015',
            need_before=False,need_after=False,start_complete=True,end_complete=True,
            excitement=4,confidence=.85,preparation_observation='Attacker begins approach after the lead frame.',
            contact_observation='One arm swings; the ball changes direction between the bracketing frames.',
            result_observation='Opponent forearms meet the ball and it travels upward after contact.',
            reason='Observed attack and opponent response with surrounding frames.',uncertainty='',
            contact_type='single_arm_attack',outcome_type='opponent_response',
            contact_before_frame_id='frame_00004',contact_after_frame_id='frame_00006',
            next_touch_frame_id='frame_00011',outcome_before_frame_id='frame_00010',
            outcome_after_frame_id='frame_00012',opening_in_motion=False,ending_in_motion=False)

    def test_valid_bracketing_resolves_exact_actual_pts_without_repair(self):
        original = copy.deepcopy(self.valid)
        decoded = decode_verify(self.valid,self.evidence)
        self.assertEqual(self.valid,original)
        self.assertEqual(decoded['times']['peak'],10.64)
        self.assertEqual(decoded['times']['result'],11.39)
        self.assertEqual(decoded['event_times'],dict(contact_before=10.51,contact_after=10.76,
                         next_touch=11.39,outcome_before=11.26,outcome_after=11.51))
        self.assertEqual(decoded['contact_type'],'single_arm_attack')
        self.assertFalse(decoded['opening_in_motion']); self.assertFalse(decoded['ending_in_motion'])

    def test_observation_of_set_or_unclear_contact_is_not_coerced_to_attack(self):
        # Contract validity is not permission to render a spike: the quality
        # gate must consume these unchanged semantic disagreements.
        for contact in ('two_hand_set','unclear'):
            with self.subTest(contact=contact):
                row = dict(self.valid,contact_type=contact)
                decoded = decode_verify(row,self.evidence)
                self.assertEqual(decoded['contact_type'],contact)
                self.assertNotEqual(decoded['contact_type'],'single_arm_attack')
                self.assertEqual(decoded['action'],'spike')

    def test_unknown_outcome_is_preserved_for_quality_gate(self):
        row = dict(self.valid,outcome_type='unknown',next_touch_frame_id=None)
        decoded = decode_verify(row,self.evidence)
        self.assertEqual(decoded['outcome_type'],'unknown')
        self.assertIsNone(decoded['event_times']['next_touch'])

    def test_reject_can_truthfully_report_set_without_any_event_anchors(self):
        row = dict(self.valid,verdict='reject',action='set',contact_type='two_hand_set',outcome_type='unknown',
                   start_complete=False,end_complete=False,preparation_observation='',contact_observation='',
                   result_observation='',reason='Only an ordinary two-hand set is visible.')
        for key in row:
            if key.endswith('_frame_id'):
                row[key] = None
        decoded = decode_verify(row,self.evidence)
        self.assertEqual(decoded['verdict'],'reject')
        self.assertTrue(all(value is None for value in decoded['event_times'].values()))

    def test_accept_requires_all_contact_and_outcome_bracketing_frames(self):
        required = ('contact_before_frame_id','contact_after_frame_id',
                    'outcome_before_frame_id','outcome_after_frame_id')
        for key in required:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'replay_verify_missing_event_frames'):
                decode_verify(dict(self.valid,**{key:None}),self.evidence)

    def test_missing_referenced_frame_is_not_replaced_with_nearest_frame(self):
        missing = [r for r in self.evidence if r['id']!='frame_00006']
        with self.assertRaisesRegex(ValueError,'replay_unknown_frame'):
            decode_verify(self.valid,missing)
        for value in ('frame_99999',6,'6'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'replay_unknown_frame'):
                decode_verify(dict(self.valid,contact_after_frame_id=value),self.evidence)

    def test_sampling_gap_is_rejected_even_when_named_anchors_still_exist(self):
        evidence_valid(self.evidence,10.,12.,8.,{'nominal_fps':'30000/1001'})
        # This missing frame is not a selected anchor, so JSON contract alone
        # cannot establish full temporal coverage. Request validation must act.
        missing = [r for r in self.evidence if r['id']!='frame_00008']
        decode_verify(self.valid,missing)
        with self.assertRaisesRegex(ValueError,'replay_sampling_gap'):
            evidence_valid(missing,10.,12.,8.,{'nominal_fps':'30000/1001'})

    def test_contact_and_outcome_frames_must_bracket_their_events(self):
        cases = [dict(contact_before_frame_id='frame_00005'),
                 dict(contact_after_frame_id='frame_00004'),
                 dict(contact_after_frame_id='frame_00012'),
                 dict(outcome_before_frame_id='frame_00004'),
                 dict(outcome_before_frame_id='frame_00011'),
                 dict(outcome_after_frame_id='frame_00011'),
                 dict(tail_frame_id='frame_00011')]
        for changed in cases:
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'replay_verify_event_order'):
                decode_verify(dict(self.valid,**changed),self.evidence)

    def test_next_touch_cannot_precede_peak_or_follow_claimed_result(self):
        for frame in ('frame_00004','frame_00005','frame_00012'):
            with self.subTest(frame=frame),self.assertRaisesRegex(ValueError,'replay_verify_next_touch_order'):
                decode_verify(dict(self.valid,next_touch_frame_id=frame),self.evidence)

    def test_unknown_event_type_is_protocol_error_not_unknown_observation(self):
        for changed in (dict(contact_type='unknown'),dict(contact_type='powerful_attack'),
                        dict(outcome_type='apex'),dict(outcome_type=None)):
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'replay_verify_event_type'):
                decode_verify(dict(self.valid,**changed),self.evidence)

    def test_motion_flags_require_real_booleans_and_are_not_silently_overridden(self):
        for key in ('opening_in_motion','ending_in_motion'):
            for value in (0,1,'false',None):
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'replay_verify_motion_boolean'):
                    decode_verify(dict(self.valid,**{key:value}),self.evidence)
            decoded = decode_verify(dict(self.valid,**{key:True}),self.evidence)
            self.assertTrue(decoded[key])  # quality gate must reject this claim

    def test_false_completeness_or_need_more_conflicts_with_accept(self):
        for changed in (dict(start_complete=False),dict(end_complete=False),
                        dict(need_before=True),dict(need_after=True)):
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'replay_contradictory_accept'):
                decode_verify(dict(self.valid,**changed),self.evidence)

    def test_exact_fields_and_finite_scores_are_required(self):
        for changed in (dict(extra='provider-added field'),dict(confidence=math.nan),dict(excitement=True)):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                decode_verify(dict(self.valid,**changed),self.evidence)
        row = copy.deepcopy(self.valid); del row['outcome_type']
        with self.assertRaisesRegex(ValueError,'replay_response_fields'):
            decode_verify(row,self.evidence)

    def test_verify_wire_schema_only_allows_this_requests_frame_ids(self):
        wire = payload('configured-model','verify','Independent observation.',[],self.evidence,4096,'low')
        schema = wire['response_format']['json_schema']['schema']
        self.assertEqual(set(schema['properties']),set(VERIFY_SCHEMA['properties']))
        self.assertEqual(wire['response_format']['json_schema']['name'],'replay_verify')
        ids = {r['id'] for r in self.evidence}
        for name,rule in schema['properties'].items():
            if name.endswith('_frame_id'):
                self.assertEqual(set(rule['enum']),ids|{None})
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(wire['reasoning_effort'],'low')


if __name__ == '__main__':
    unittest.main()
