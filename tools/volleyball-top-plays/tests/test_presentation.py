import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import Stages, save_json
from presentation import build_timeline, lively_title, peak_window, validate_timeline
import test_pipeline


class PresentationTests(unittest.TestCase):
    setUp=test_pipeline.DecisionTests.setUp
    tearDown=test_pipeline.DecisionTests.tearDown

    def timeline(self,k):
        from ranker import rule_decision
        for r in self.rallies:r['preview_times_sec']=[r['start_sec'],r['start_sec']+2,r['end_sec']]
        decision=rule_decision(self.rallies,k)
        rows=build_timeline(decision,self.manifest)
        placeholder=self.root/'segment.mp4';placeholder.touch()
        for row in rows:row['path']=str(placeholder)
        return decision,{'segments':rows,'expected_duration_sec':sum(r['output_frames'] for r in rows)/30}

    def test_four_second_hook_full_rallies_replays_and_wipes(self):
        for k in (5,10):
            decision,report=self.timeline(k)
            validate_timeline(report,decision)
            rows=report['segments']
            self.assertEqual(len([r for r in rows if r['kind']=='transition']),k)
            self.assertEqual(sum(r['duration_sec'] for r in rows if r['kind']=='teaser'),4)
            self.assertTrue(all(r['duration_sec']==3 for r in rows if r['kind']=='replay'))
            self.assertTrue(all(r['duration_sec']==12 for r in rows if r['kind']=='rally'))

    def test_peak_clamps_at_clip_edges_and_handles_short_rally(self):
        item={'rank':1,'rally_id':'r','clip_start_sec':10.,'clip_end_sec':11.4}
        for peak in (10.,11.4,float('nan'),100.):
            window=peak_window(item,{'preview_times_sec':[10.,peak,11.4]})
            self.assertAlmostEqual(window['source_start_sec'],10.)
            self.assertAlmostEqual(window['source_end_sec'],11.4)

    def test_rejects_replay_outside_source_and_broken_timeline(self):
        decision,report=self.timeline(5)
        for problem in ('outside','gap','missing_replay','truncated_rally'):
            changed=copy.deepcopy(report)
            row=next(r for r in changed['segments'] if r['kind']=='replay')
            if problem=='outside':row['source_start_sec']=-1
            elif problem=='gap':row['timeline_start_frame']+=1
            elif problem=='missing_replay':changed['segments'].remove(row)
            else:next(r for r in changed['segments'] if r['kind']=='rally')['source_end_sec']-=1
            with self.subTest(problem=problem),self.assertRaises(ValueError):validate_timeline(changed,decision)

    def test_titles_are_short_playful_and_do_not_invent_scores(self):
        for text in ('长回合起跳对抗与防守站位','极低姿态接球防守','持续攻防与二传组织配合'):
            title=lively_title({'title':text})
            self.assertLessEqual(len(title),16)
            self.assertNotIn('得分',title)
            self.assertNotIn('获胜',title)


class TimingTests(unittest.TestCase):
    def test_cached_stage_reports_current_hash_check_time_not_historical_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'result.json'
            def callback():save_json(path,{'ok':True});return str(path),[path]
            stages=Stages(root);stages.execute('render',{},callback)
            stages.data['stages']['render']['elapsed_sec']=999
            with patch('common.time.monotonic',side_effect=[10.,10.02]):stages.execute('render',{},callback)
            self.assertEqual(stages.current_run[-1],{'stage':'render','status':'reused','elapsed_sec':.02})
