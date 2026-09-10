import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import Stages, save_json, digest, APP
from presentation import build_timeline, lively_title, peak_window, validate_timeline
from illustrated import chapter_label,rank_label,illustration_for,transition_opacity,title_lines,prepare_brand
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

    def test_five_second_hook_full_rallies_replays_and_title_cards(self):
        for k in (5,10):
            decision,report=self.timeline(k)
            validate_timeline(report,decision)
            rows=report['segments']
            self.assertEqual(len([r for r in rows if r['kind']=='transition']),k)
            self.assertEqual(sum(r['duration_sec'] for r in rows if r['kind']=='teaser'),5)
            self.assertTrue(all(r['duration_sec']==3 and r['title_hold_sec']>=2 for r in rows if r['kind']=='transition'))
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

    def test_title_cards_hold_opaque_and_use_playful_chapter_copy(self):
        self.assertEqual(transition_opacity(0),0)
        self.assertEqual(transition_opacity(89),0)
        self.assertTrue(all(transition_opacity(i)==1 for i in range(12,78)))
        for rank in range(1,11):
            label=chapter_label(rank,10)
            self.assertNotIn('#',label)
            self.assertFalse(any(c.isdigit() for c in label))
        self.assertEqual(title_lines('贴地救球！太拼了'),['贴地救球！','太拼了'])

    def test_rank_labels_are_explicit_chinese_for_five_and_ten(self):
        chinese=('一','二','三','四','五','六','七','八','九','十')
        for k,collection in ((5,'五佳球'),(10,'十佳球')):
            for rank in range(1,k+1):
                self.assertEqual(rank_label(rank,k),f'{collection} · 第{chinese[rank-1]}球')
            decision,report=self.timeline(k)
            for segment in report['segments']:
                number=segment.get('rank',segment.get('next_rank'))
                self.assertEqual(segment['rank_label'],rank_label(number,k))
            for kind in ('rally','replay','transition'):
                changed=copy.deepcopy(report)
                next(s for s in changed['segments'] if s['kind']==kind)['rank_label']='五佳球 · 第一球'
                with self.subTest(k=k,kind=kind),self.assertRaises(ValueError):validate_timeline(changed,decision)
        for rank,k in ((0,5),(6,5),(11,10),(1,7)):
            with self.assertRaises(ValueError):rank_label(rank,k)

    def test_action_artwork_has_distinct_choices(self):
        titles=('长回合起跳对抗','低姿救球','二传组织配合','低位防守','极低姿态接球')
        self.assertEqual(len({illustration_for({'title':t}) for t in titles}),5)

    def test_rejects_unreadably_short_transition(self):
        decision,report=self.timeline(5)
        row=next(s for s in report['segments'] if s['kind']=='transition')
        row['title_hold_sec']=.3
        with self.assertRaises(ValueError):validate_timeline(report,decision)


class TimingTests(unittest.TestCase):
    def test_cached_stage_reports_current_hash_check_time_not_historical_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'result.json'
            def callback():save_json(path,{'ok':True});return str(path),[path]
            stages=Stages(root);stages.execute('render',{},callback)
            stages.data['stages']['render']['elapsed_sec']=999
            with patch('common.time.monotonic',side_effect=[10.,10.02]):stages.execute('render',{},callback)
            self.assertEqual(stages.current_run[-1],{'stage':'render','status':'reused','elapsed_sec':.02})


class BrandingTests(unittest.TestCase):
    def test_logo_cache_reuses_exact_source_and_refreshes_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'logo.svg';png=root/'logo.png'
            source.write_text('<svg/>');png.write_bytes(b'fixture-png')
            save_json(png.with_suffix('.json'),{'source_sha256':digest(source),
                'renderer_sha256':digest(APP/'rasterize_logo.py'),'png_sha256':digest(png)})
            with patch('illustrated.LOGO_SOURCE',source),patch('illustrated.LOGO_PNG',png), \
                    patch('illustrated.subprocess.run') as render:
                prepare_brand();render.assert_not_called()
                source.write_text('<svg><!-- replacement --></svg>')
                prepare_brand();self.assertEqual(render.call_count,1)
                prepare_brand();self.assertEqual(render.call_count,1)
                png.write_bytes(b'changed-png')
                prepare_brand();self.assertEqual(render.call_count,2)
