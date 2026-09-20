import copy
import unittest

from volleymole.presentation import build_timeline, validate_replay_evidence
from volleymole.replay import replay_window


class ReplaySelectionTests(unittest.TestCase):
    def setUp(self):
        self.item=dict(rank=1,rally_id='r1',clip_start_sec=10.,clip_end_sec=23.,title='攻防回合')
        self.source={'identity':{'sha256':'a'*64}}
        self.rally=dict(rally_id='r1',preview_times_sec=[10.,14.,23.],action_events=[])
        self.review=dict(status='approved',method='assistant_visual_review',source_sha256='a'*64,
                         reason='Reviewed approach, contact, landing and ball aftermath.',
                         action='spike',boundary_complete=True,source_start_sec=17.,
                         action_start_sec=18.,peak_sec=19.,action_end_sec=21.,source_end_sec=22.)

    def test_midpoint_or_set_only_does_not_force_a_replay(self):
        for events in ([],[dict(action='set',start_sec=14.,end_sec=14.5,detection_frames=20)]):
            self.rally['action_events']=events
            self.assertIsNone(replay_window(self.item,self.rally,self.source))

    def test_attack_beats_set_and_receive_and_keeps_context(self):
        self.rally['action_events']=[
            dict(action='set',start_sec=12.,end_sec=12.3,detection_frames=20),
            dict(action='receive',start_sec=14.,end_sec=14.3,detection_frames=10),
            dict(action='spike',start_sec=18.,end_sec=18.4,detection_frames=3)]
        result=replay_window(self.item,self.rally,self.source)
        self.assertEqual(result['replay_action'],'spike')
        self.assertEqual((result['source_start_sec'],result['source_end_sec']),(16.5,20.4))
        self.assertFalse(result['visual_review_used'])
        self.assertIsNone(result['boundary_complete'])

    def test_clipped_action_or_single_frame_detection_is_omitted(self):
        for start,end,support in ((10.2,10.4,3),(22.,22.5,3),(17.,17.3,1)):
            self.rally['action_events']=[dict(action='spike',start_sec=start,end_sec=end,detection_frames=support)]
            self.assertIsNone(replay_window(self.item,self.rally,self.source))

    def test_review_selects_later_complete_action_independent_of_preview(self):
        self.rally['replay_review']=self.review
        result=replay_window(self.item,self.rally,self.source)
        self.assertEqual(result['peak_sec'],19.)
        self.assertEqual(result['source_end_sec'],22.)
        self.assertTrue(result['visual_review_used'])
        self.assertTrue(result['boundary_complete'])

    def test_rejects_wrong_source_or_cuts_through_reviewed_action(self):
        for changes in ({'source_sha256':'b'*64},{'source_end_sec':20.},
                        {'source_start_sec':18.5},{'peak_sec':float('nan')},
                        {'boundary_complete':False}):
            self.rally['replay_review']={**self.review,**changes}
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                replay_window(self.item,self.rally,self.source)

    def test_explicit_omission_and_source_bound_timeline_validation(self):
        self.rally['replay_review']=self.review
        decision={'selected':[self.item],'ranking_mode':'api'}
        manifest={'source':self.source,'rallies':[self.rally]}
        rows=build_timeline(decision,manifest)
        report={'replay_policy_version':2,'segments':rows}
        validate_replay_evidence(report,decision,manifest)
        changed=copy.deepcopy(report)
        next(s for s in changed['segments'] if s['kind']=='replay')['source_end_sec']-=1
        with self.assertRaises(ValueError):validate_replay_evidence(changed,decision,manifest)
        self.rally['replay_review']={**self.review,'status':'omit','reason':'No complete exciting action.'}
        rows=build_timeline(decision,manifest)
        self.assertFalse(any(s['kind']=='replay' for s in rows))
        self.assertFalse(rows[0]['visual_review_used'])


if __name__=='__main__':unittest.main()
