"""Synthetic, labelled evidence tests; these are not real-match accuracy scores."""
import unittest
import numpy as np
from volleymole.semantic import candidate_fields

from volleymole.rally_evidence import associate_auxiliary, low_motion, uncertain_gaps, preserve_separations


class EvidenceTests(unittest.TestCase):
    def test_api_summary_is_bounded_but_full_cache_identity_keeps_raw_evidence(self):
        rally = {k:0 for k in ('rally_id','start_sec','end_sec','safe_start_sec','safe_end_sec',
                               'duration_sec','actions','players','ball_metrics','rule_score')}
        rally['uncertainty'] = {'auxiliary_associations':[{'frame':i,'time_sec':i/30,
            'source':'actual_detector','vball_anchors':[0,100]} for i in range(100)],
            'occlusion_gaps':[{'start_sec':i,'end_sec':i+.2,'coordinates_synthesized':False} for i in range(20)]}
        small = candidate_fields(rally)['uncertainty']
        self.assertEqual(small['associated_detection_count'],100)
        self.assertEqual(len(small['association_examples']),3)
        self.assertEqual(small['occlusion_gap_count'],20)
        self.assertEqual(len(small['longest_occlusion_gaps']),3)
        full = candidate_fields(rally,full_evidence=True)['uncertainty']
        self.assertEqual(full,rally['uncertainty'])
        self.assertEqual(len(full['auxiliary_associations']),100)

    def test_low_motion_cannot_merge_two_separated_strong_cores(self):
        cores = [list(range(2,12)),list(range(20,30))]
        groups,cuts = preserve_separations([list(range(1,31))],cores,np.arange(40)/2,4.)
        self.assertEqual(cuts,[8.])
        self.assertEqual(groups,[list(range(1,16)),list(range(16,31))])
        for core in cores:
            self.assertTrue(any(set(core)<=set(g) for g in groups))
        self.assertEqual(preserve_separations([list(range(10))],[list(range(5))],np.arange(40)/2,4.)[1],[])

    def test_auxiliary_requires_two_actual_bounded_detections(self):
        times = np.arange(6)/30
        xy = np.array([[100.,100.],[0.,0.],[0.,0.],[160.,100.],[0.,0.],[200.,100.]])
        seen = np.array([True,False,False,True,False,True])
        rows = [{'ball':{'xyxy':[100+20*i-3,97,100+20*i+3,103],
                         'confidence':.9,'origin':'test_actual_detector'}} for i in range(6)]
        fused,visible,origins,accepted = associate_auxiliary(rows,xy,seen,times,1000)
        np.testing.assert_array_equal(xy[seen],fused[seen])
        np.testing.assert_array_equal(visible,[True,True,True,True,False,True])
        self.assertEqual([a['frame'] for a in accepted],[1,2])
        self.assertEqual(origins[4],'missing')
        np.testing.assert_array_equal(xy[4],fused[4])
        rows[2]['ball']['confidence'] = .1
        self.assertFalse(associate_auxiliary(rows,xy,seen,times,1000)[3])

    def test_auxiliary_does_not_bridge_long_or_implausible_gaps(self):
        xy = np.array([[100.,100.],[0.,0.],[0.,0.],[160.,100.]])
        rows = [{'ball':{'xyxy':[117,97,123,103],'confidence':.99}}]*4
        seen = np.array([True,False,False,True])
        for times in (np.arange(4), np.arange(4)/10000):
            fused,visible,_,accepted = associate_auxiliary(rows,xy,seen,times,1000)
            np.testing.assert_array_equal(fused,xy)
            np.testing.assert_array_equal(visible,seen)
            self.assertEqual(accepted,[])

    def test_carried_motion_rejected_despite_play_state(self):
        n = 30
        times = np.arange(n)/30
        xy = np.column_stack((200+times*200,np.full(n,412.5)))
        rows = [{'state':'play','state_confidence':.99,'players':[
            {'xyxy':[175+t*200,300,225+t*200,550],'confidence':.9},
            {'xyxy':[750,300,800,550],'confidence':.9}]} for t in times]
        low,carried = low_motion(rows,xy,np.ones(n,dtype=bool),np.full(n,.2),
                                np.full(n,337.5),times,1000,600)
        self.assertTrue(carried[1:].all())
        self.assertFalse(low[1:].any())

    def test_low_ball_requires_motion_and_state_or_nearby_action(self):
        n = 30
        times = np.arange(n)/30
        xy = np.column_stack((350+times*200,np.full(n,460.)))
        people = [{'xyxy':[200,300,250,550],'confidence':.9},
                  {'xyxy':[750,300,800,550],'confidence':.9}]
        rows = [{'state':'play','state_confidence':.9,'players':people} for _ in times]
        args = (xy,np.ones(n,dtype=bool),np.full(n,.2),np.full(n,337.5),times,1000,600)
        self.assertTrue(low_motion(rows,*args)[0].all())
        for row in rows:
            row['state'] = 'no-play'
        self.assertFalse(low_motion(rows,*args)[0].any())
        rows[15]['raw_actions'] = [{'class':'receive','confidence':.9,'xyxy':[400,350,510,510]}]
        low,_ = low_motion(rows,*args)
        self.assertTrue(low[15])
        self.assertFalse(low[0])
        self.assertFalse(low[-1])
        args = (xy,np.ones(n,dtype=bool),np.zeros(n),np.full(n,337.5),times,1000,600)
        self.assertFalse(low_motion(rows,*args)[0].any())

    def test_occlusion_is_annotation_not_synthetic_track(self):
        times = np.arange(8)/10
        visible = np.array([False,True,False,False,True,False,False,False])
        original = visible.copy()
        gaps = uncertain_gaps(times,visible,0,len(times))
        self.assertEqual(len(gaps),1)
        self.assertEqual(gaps[0]['missing_frames'],2)
        self.assertFalse(gaps[0]['coordinates_synthesized'])
        np.testing.assert_array_equal(visible,original)


if __name__=='__main__':
    unittest.main()
