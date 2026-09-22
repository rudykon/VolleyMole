import copy
import unittest
from test_meme_director import assets,clip,decision,frames,skip
from volleymole.meme_director import policy,schedule_after_action,eligible_cue


class MemeSchedulerTests(unittest.TestCase):
    def test_delay_past_contact_and_observed_result_without_changing_answer(self):
        raw=decision('nice');raw['anchor_frame_id']='f10'
        original=copy.deepcopy(raw)
        adjusted,audit=schedule_after_action(raw,frames(),clip(),assets(),policy())
        self.assertEqual(raw,original)
        self.assertEqual(adjusted['anchor_frame_id'],'f12')
        self.assertEqual(eligible_cue(adjusted,frames(),clip(),assets(),policy())[1],'eligible')
        self.assertEqual(audit['status'],'delayed_to_safe_gap')
        self.assertEqual(adjusted['facts'],raw['facts'])

    def test_skip_and_semantic_veto_never_become_use(self):
        for raw in (skip(),{**decision('nice'),'confidence':.1}):
            adjusted,_=schedule_after_action(raw,frames(),clip(),assets(),policy())
            self.assertEqual(adjusted,raw)

    def test_no_truncation_or_early_shift_to_fit(self):
        raw=decision('nice');raw['anchor_frame_id']='f35'
        adjusted,audit=schedule_after_action(raw,frames(),clip(),assets(),policy())
        self.assertEqual(adjusted,raw)
        self.assertEqual(audit['status'],'phrase_would_be_truncated')

    def test_delay_limit_and_later_contact_remain_hard_constraints(self):
        raw=decision('nice');raw['anchor_frame_id']='f10';raw['contact_frame_ids']=['f10','f18','f26']
        adjusted,_=schedule_after_action(raw,frames(),clip(),assets(),policy(),max_delay_sec=.3)
        self.assertEqual(adjusted,raw)
