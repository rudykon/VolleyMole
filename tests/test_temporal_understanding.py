import copy
import unittest

from volleymole.temporal_understanding import action_windows,grounded_predictions,video_frame_content


class TemporalUnderstandingTests(unittest.TestCase):
    def test_context_overlaps_but_ownership_covers_exactly_once(self):
        parts=action_windows(14.81,8,24,.5)
        self.assertEqual(parts[0]['own_start'],0)
        self.assertEqual(parts[-1]['own_end'],14.81)
        for i,part in enumerate(parts):
            self.assertLessEqual(part['end']-part['start'],3+1e-9)
            self.assertLessEqual(part['start'],part['own_start'])
            self.assertGreaterEqual(part['end'],part['own_end'])
            if i:self.assertEqual(parts[i-1]['own_end'],part['own_start'])
        for t in [i*.01 for i in range(1481)]:
            self.assertEqual(sum(p['own_start']<=t<p['own_end'] for p in parts),1)

    def response(self):
        return {'events':[{'label':'spike','frame_id':'f1','after_frame_id':'f2',
            'confidence':.8,'observable':True,'aftermath':'continues','observation':'Ball leaves the striking hand.'}],
            'uncertainty':'Sampled frames may miss occluded contacts.'}

    def test_times_come_from_real_frames_and_unknown_is_not_a_contact(self):
        evidence=[{'id':f'f{i}','kind':'frame','start_sec':10+i*.12} for i in range(3)]
        rows,rejected=grounded_predictions(self.response(),evidence,{'own_start':10,'own_end':11})
        self.assertEqual(rows[0]['time_sec'],10.12)
        self.assertIsNone(rows[0]['confirmed_contact'])
        data=self.response();data['events'][0]['observable']=False
        rows,rejected=grounded_predictions(data,evidence,{'own_start':10,'own_end':11})
        self.assertEqual(rows,[]);self.assertEqual(rejected[0]['reason'],'unobservable')

    def test_after_frame_must_be_later_and_references_cannot_be_invented(self):
        evidence=[{'id':f'f{i}','kind':'frame','start_sec':i*.12} for i in range(3)]
        for key,value in [('frame_id','nonexistent'),('after_frame_id','f0'),('confidence',float('nan'))]:
            data=self.response();data['events'][0][key]=value
            with self.assertRaises(ValueError):grounded_predictions(data,evidence,{'own_start':0,'own_end':1})

    def test_native_video_preserves_frame_indices_and_requests_no_resampling(self):
        content=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,QQ=='}},
                 {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,Qg=='}}]
        evidence=[{'id':'a','kind':'frame','start_sec':2.04},{'id':'b','kind':'frame','start_sec':2.16}]
        parts,extra=video_frame_content(content,evidence,{'nominal_fps':'25/1','frame_count':250,'duration_sec':10})
        self.assertEqual(parts[0]['video_url']['url'],'data:video/jpeg;base64,QQ==,Qg==')
        self.assertEqual(extra['media_io_kwargs']['video']['frames_indices'],[51,54])
        self.assertIs(extra['mm_processor_kwargs']['do_sample_frames'],False)

    def test_invalid_sampling_rejected(self):
        for args in [(1,0,24,.5),(1,8,24,3),(0,8,24,.5),(1,8,2,.5)]:
            with self.assertRaises(ValueError):action_windows(*args)

    def test_last_frame_allows_unknown_aftermath_without_discarding_action(self):
        evidence=[{'id':'f1','kind':'frame','start_sec':.96}]
        data=self.response();data['events'][0].update(after_frame_id=None,aftermath='unknown')
        rows,_=grounded_predictions(data,evidence,{'own_start':0,'own_end':1})
        self.assertEqual(rows[0]['time_sec'],.96)
        self.assertEqual(rows[0]['evidence_ids'],['f1'])
        data['events'][0]['observable']=False
        rows,rejected=grounded_predictions(data,evidence,{'own_start':0,'own_end':1})
        self.assertEqual(rows,[]);self.assertEqual(rejected[0]['reason'],'unobservable')

    def test_token_budget_invalidates_response_cache(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from volleymole.temporal_understanding import observe_actions
        evidence=[{'id':'f1','kind':'frame','start_sec':.12},{'id':'f2','kind':'frame','start_sec':.24}]
        source={'path':'unused','width':398}
        settings={'model':'fake','endpoint':'https://example.invalid','key':'test-key'}
        job={'start':0,'end':1,'own_start':0,'own_end':1}
        with tempfile.TemporaryDirectory() as tmp,patch('volleymole.temporal_understanding.digest',return_value='same'), \
             patch('volleymole.temporal_understanding.sampled_evidence',return_value=(evidence,[],[])), \
             patch('volleymole.temporal_understanding.request_json',return_value=(self.response(),{})) as request:
            first=observe_actions(source,job,Path(tmp),{**settings,'max_tokens':10},float('inf'))
            second=observe_actions(source,job,Path(tmp),{**settings,'max_tokens':999},float('inf'))
            self.assertNotEqual(first['signature'],second['signature'])
            self.assertEqual(request.call_count,2)


if __name__=='__main__':unittest.main()
