"""Synthetic regressions for the complete-rally policy from commit 4399475."""
import unittest

import numpy as np

from volleymole.common import read_json
from volleymole.rally import build_manifest
from volleymole.ranker import shortlist
import test_rallies


class RallyRecoveryTests(unittest.TestCase):
    setUp = test_rallies.RallyTests.setUp
    tearDown = test_rallies.RallyTests.tearDown
    write_inputs = test_rallies.RallyTests.write_inputs

    def build(self, mode=None):
        options = {} if mode is None else {'selection_mode': mode}
        return build_manifest(self.source, self.root/'analytics.jsonl', self.root/'ball.csv',
                              self.root/'pts.csv', self.root/'player.json', self.root,
                              self.config, {}, **options)[0]

    def half_second_rally(self):
        for record, ball, time in zip(self.records, self.balls, self.pts):
            playing = 2<=time<2.5 or 16<=time<25
            record['state'] = 'play' if playing else 'no-play'
            ball[2] = 500+180*np.sin(time*2) if playing else 500
            ball[3] = 170+70*np.sin(time*4) if playing else 420
        self.write_inputs()

    def test_half_second_is_excluded_from_rally_selection_but_retained_for_events(self):
        self.half_second_rally()
        events = self.build('events')
        rallies = self.build('rallies')
        self.assertEqual(len(events['rallies']), len(rallies['rallies']))
        short = next(r for r in rallies['rallies'] if r['duration_sec']==.5)
        loose = next(r for r in events['rallies'] if r['rally_id']==short['rally_id'])
        self.assertFalse(short['eligible'])
        self.assertIn('回合过短', short['exclusion_reasons'])
        self.assertNotIn(short['rally_id'], [r['rally_id'] for r in shortlist(rallies, 1)])
        self.assertTrue(loose['eligible'])
        self.assertIn('短回合，需要事件复核', loose['discovery_warnings'])
        self.assertTrue((self.root/short['tracking_json']).is_file())
        self.assertEqual((short['start_sec'], short['end_sec']), (loose['start_sec'], loose['end_sec']))

    def test_low_coverage_is_a_rally_exclusion_and_an_event_warning(self):
        # Four actual detections per half-second yield three continuous links:
        # 20% supported motion passes discovery, but not the 25% rally gate.
        for index, ball in enumerate(self.balls):
            ball[1] = int(index%15<4)
        self.write_inputs()
        events = self.build('events')
        rallies = self.build('rallies')
        candidates = [r for r in rallies['rallies'] if r['duration_sec']>=self.config['min_rally_sec']
                      and r['ball_metrics']['supported_motion_ratio']>=self.config['min_flight_ratio']]
        self.assertTrue(candidates)
        for strict in candidates:
            loose = next(r for r in events['rallies'] if r['rally_id']==strict['rally_id'])
            self.assertLess(strict['ball_metrics']['visible_ratio'], self.config['min_visible_ratio'])
            self.assertFalse(strict['eligible'])
            self.assertIn('有效球轨迹不足', strict['exclusion_reasons'])
            self.assertTrue(loose['eligible'])
            self.assertIn('有效球轨迹不足，需要事件复核', loose['discovery_warnings'])

    def test_rally_components_and_sparse_player_penalty_equal_historical_formula(self):
        # Two players in rally one, six elsewhere: exercise the cubic quality
        # factor as well as duration, multiple turns, and three action groups.
        for record, time in zip(self.records, self.pts):
            if 2<=time<10:
                record['players'] = record['players'][:2]
                for start, action in ((3., 'receive'), (5., 'set'), (7., 'spike')):
                    if start<=time<start+.3:
                        record['actions'] = [{'class': action, 'confidence': .9}]
        self.write_inputs()
        manifest = self.build('rallies')
        counts = np.array([sum(p['confidence']>=.4 for p in row['players']) for row in self.records])
        normalizer = max(1., float(np.percentile(counts, 85)))
        saw_sparse = False
        for rally in manifest['rallies']:
            first, last = rally['evidence']['analytics_frames']
            track = read_json(self.root/rally['tracking_json'])
            coverage = len(track['positions'])/(last-first+1)
            participating = float(np.median(counts[first:last+1]))/normalizer
            # This synthetic source is entirely aerial during both rallies;
            # flight is above the historical cap, and there is no target jersey.
            self.assertGreater(rally['ball_metrics']['flight_ratio'], .55)
            self.assertEqual(rally['ball_metrics']['low_ball_ratio'], 0)
            features = {'duration': min(rally['duration_sec']/28, 1), 'flight': 1.,
                'turns': min(rally['ball_metrics']['trajectory_changes']/12, 1),
                'actions': min(len(rally['action_events'])/5, 1), 'coverage': coverage,
                'participation': min(participating, 1), 'focus': 0.}
            expected = {name: round(self.config['weights'][name]*value, 3)
                        for name, value in features.items()}
            quality = min(1., max(.2, participating/.85)**3)
            self.assertEqual(rally['score_components'], expected)
            self.assertEqual(rally['rule_score'], round(sum(expected.values())*quality, 2))
            self.assertGreater(rally['score_components']['duration'], 0)
            self.assertGreater(rally['score_components']['turns'], 0)
            if quality<1:
                saw_sparse = True
                self.assertEqual(len(rally['action_events']), 3)
                self.assertEqual(rally['score_components']['actions'], 6.)
                self.assertLess(rally['rule_score'], sum(expected.values()))
        self.assertTrue(saw_sparse)

    def test_quality_floor_and_saturation_are_preserved(self):
        for record, time in zip(self.records, self.pts):
            if 2<=time<10:
                record['players'] = []
        self.write_inputs()
        rallies = self.build('rallies')['rallies']
        sparse = next(r for r in rallies if r['median_player_count']==0)
        populated = next(r for r in rallies if r['median_player_count']==6)
        self.assertEqual(sparse['rule_score'], round(sum(sparse['score_components'].values())*.2**3, 2))
        self.assertEqual(populated['rule_score'], round(sum(populated['score_components'].values()), 2))

    def test_event_defaults_and_scoring_remain_unchanged(self):
        for record, time in zip(self.records, self.pts):
            if 3<=time<3.3:
                record['actions'] = [{'class': 'receive', 'confidence': .9}]
            if 2<=time<10:
                record['players'] = record['players'][:2]
        self.write_inputs()
        implicit = self.build()
        explicit = self.build('events')
        self.assertEqual(implicit, explicit)
        self.assertEqual(explicit['selection_mode'], 'events')
        for rally in explicit['rallies']:
            parts = rally['score_components']
            self.assertEqual(parts['duration'], 0)
            self.assertEqual(parts['turns'], 0)
            self.assertEqual(parts['actions'], float(bool(rally['action_events']))*self.config['weights']['actions'])
            self.assertEqual(rally['rule_score'], round(sum(parts.values()), 2))
        strict = self.build('rallies')
        self.assertEqual(strict['selection_mode'], 'rallies')
        self.assertEqual(read_json(self.root/'match_manifest.json')['selection_mode'], 'rallies')

    def test_unknown_selection_mode_is_rejected_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, '未知候选选择模式'):
            self.build('auto')


if __name__ == '__main__':
    unittest.main()
