import copy
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ranker import rank
from common import read_json
from semantic import validate_reviews, visual_reviews
from upstream_worker import PLAYER_SAMPLE_FILTER
import test_pipeline


class VisionRoutingTests(unittest.TestCase):
    setUp = test_pipeline.DecisionTests.setUp
    tearDown = test_pipeline.DecisionTests.tearDown
    def review_rows(self):
        return [{'rally_id':r['rally_id'],'observation':'可见接球姿态。','uncertainty':'三帧不能确认得分。',
                 'watchability_score':80,'confidence':.6} for r in self.rallies]

    def test_text_only_primary_routes_through_visual_reviewer(self):
        error=HTTPError('https://example.invalid',400,'Bad request',{},io.BytesIO(b'{"message":"not a multimodal model"}'))
        with patch.dict('os.environ',{'VOLLEYMOLE_API_KEY':'test-token'}), \
             patch('ranker.api_decision',side_effect=[error,self.decision]) as generation, \
             patch('ranker.choose_vision_model',return_value=('vision-model',['text-model','vision-model'])), \
             patch('ranker.visual_reviews',return_value=(self.review_rows(),[],[])):
            path,_=rank(self.manifest,self.root,5,None,'auto','https://example.invalid/v1','text-model')
        result=read_json(path)
        self.assertEqual(result['ranking_mode'],'vision_then_text_api')
        self.assertEqual(result['model'],'text-model')
        self.assertEqual(result['vision_model'],'vision-model')
        self.assertIsNotNone(generation.call_args.kwargs['reviews'])

    def test_unrelated_400_does_not_change_models(self):
        error=HTTPError('https://example.invalid',400,'Bad request',{},io.BytesIO(b'{"message":"context too long"}'))
        with patch.dict('os.environ',{'VOLLEYMOLE_API_KEY':'test-token'}), \
             patch('ranker.api_decision',side_effect=error),patch('ranker.choose_vision_model') as choose:
            path,_=rank(self.manifest,self.root,5,None,'auto','https://example.invalid/v1','test-model')
        choose.assert_not_called()
        self.assertEqual(read_json(path)['ranking_mode'],'rules_fallback')

    def test_visual_review_rejects_duplicates_and_invalid_scores(self):
        data={'reviews':self.review_rows()}
        validate_reviews(data,self.rallies)
        for change in ('duplicate','nan','missing'):
            bad=copy.deepcopy(data)
            if change=='duplicate':bad['reviews'][-1]=bad['reviews'][0]
            elif change=='nan':bad['reviews'][0]['watchability_score']=float('nan')
            else:bad['reviews'].pop()
            with self.assertRaises(ValueError):validate_reviews(bad,self.rallies)

    def test_visual_batches_resume_and_invalidate_changed_images(self):
        for r in self.rallies:r['preview_times_sec']=[r['start_sec'],(r['start_sec']+r['end_sec'])/2,r['end_sec']]
        candidates=self.rallies[:5]
        response=({'reviews':self.review_rows()[:5]},{'model':'vision','finish_reason':'stop','usage':{'total_tokens':20}})
        with patch('semantic.request_json',return_value=response) as request:
            visual_reviews(candidates,self.root,'https://example.invalid/v1','vision','test-token',1,batch_size=5)
            visual_reviews(candidates,self.root,'https://example.invalid/v1','vision','test-token',1,batch_size=5)
            self.assertEqual(request.call_count,1)
            (self.root/candidates[0]['preview_frames'][0]).write_bytes(b'changed-source-frame')
            visual_reviews(candidates,self.root,'https://example.invalid/v1','vision','test-token',1,batch_size=5)
            self.assertEqual(request.call_count,2)


class PlayerTimestampTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_recorded_integer_second_matches_actual_source_frame(self):
        result=subprocess.run(['ffmpeg','-hide_banner','-f','lavfi','-i','testsrc2=size=320x180:rate=30',
            '-t','2','-vf','showinfo@before,'+PLAYER_SAMPLE_FILTER+',showinfo@after','-f','null','-'],
            capture_output=True,text=True,check=True)
        source={};sampled=[]
        for line in result.stderr.splitlines():
            match=re.search(r'pts_time:([\d.]+).*?checksum:([0-9A-F]+)',line)
            if not match:continue
            when,checksum=match.groups()
            if 'showinfo@before' in line:source[checksum]=float(when)
            if 'showinfo@after' in line:sampled.append((float(when),checksum))
        self.assertGreaterEqual(len(sampled),2)
        for when,checksum in sampled[:2]:
            self.assertLessEqual(abs(when-source[checksum]),1/30)


if __name__=='__main__':unittest.main()
