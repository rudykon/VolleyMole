"""Meaningful benchmark guarantees; synthetic test answers are not dataset labels."""
import copy
import importlib.util
import json
import re
import tempfile
from pathlib import Path
import unittest

from volleymole.blooper_benchmark import (DIMENSIONS, prepare, validate_forms, evaluate,
                                        ordinal_alpha, cluster_mean_interval)

spec = importlib.util.spec_from_file_location('blooper_benchmark_cli',
    Path(__file__).resolve().parents[1]/'scripts/blooper_benchmark.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class BlooperBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.media = Path(self.temp.name)/'source.mp4'
        self.media.write_bytes(b'not decoded in unit tests')
        self.data = {'matches': [{'match_id': 'match-a', 'split': 'test',
            'sources': [{'source_id': 'single', 'path': str(self.media),
                         'duration_sec': 20., 'audio_available': True}],
            'events': [{'event_id': 'one', 'source_id': 'single',
                        'clip_start_sec': 0., 'clip_end_sec': 8.,
                        'title': 'MUST NOT LEAK', 'scores': {'bloopers': 99}},
                       {'event_id': 'two', 'source_id': 'single',
                        'clip_start_sec': 10., 'clip_end_sec': 18.}]}]}
        self.private, self.forms = prepare(self.data)

    def completed(self):
        forms = copy.deepcopy(self.forms)
        event_by_item = {i['item_id']: i['event_id'] for i in self.private['items']}
        for form in forms:
            form['human_provenance'] = {'annotator_id': form['assignment_id'], 'human_only': True}
            for row in form['items']:
                row['completed'] = True
                grade = 4 if event_by_item[row['item_id']] == 'one' else 0
                row['ratings'] = dict.fromkeys(DIMENSIONS, grade)
                row['flags'] = dict.fromkeys(row['flags'], False)
                row['flags']['safe_to_include'] = True
                row['observed_facts'] = 'Synthetic test fixture only, not a human dataset label.'
                row['unknown_reason'] = 'Synthetic test has no actual temporal stages.'
                if form['condition'] == 'silent':
                    row['ratings']['related_laughter'] = None
                    row['flags']['laughter_linked'] = None
                    row['flags']['unrelated_laughter'] = None
        return forms

    def predictions(self, ids):
        return {'systems': [{'system_id': 'candidate', 'matches': [
            {'match_id': 'match-a', 'ranked_event_ids': ids}]}]}

    def test_prepare_is_empty_blind_reproducible_and_separate_conditions(self):
        again = prepare(self.data)
        self.assertEqual((self.private, self.forms), again)
        self.assertEqual(len(self.forms), 6)
        for form in self.forms:
            self.assertFalse(form['human_provenance']['human_only'])
            for row in form['items']:
                self.assertFalse(row['completed'])
                self.assertTrue(all(v is None for v in row['ratings'].values()))
                self.assertNotIn('scores', row)
                self.assertNotIn('match_id', row)
                self.assertNotIn('event_id', row)
                self.assertNotIn('MUST NOT LEAK', str(form))
                self.assertNotIn(str(self.media), str(form))

    def test_same_match_or_source_cannot_leak_across_splits(self):
        bad = copy.deepcopy(self.data)
        bad['matches'].append(copy.deepcopy(bad['matches'][0]))
        with self.assertRaises(ValueError): prepare(bad)
        bad['matches'][1].update(match_id='match-b', split='dev')
        with self.assertRaisesRegex(ValueError, '相同原素材'): prepare(bad)

    def test_offline_page_embeds_only_one_empty_assignment_and_never_prefills_answers(self):
        form = copy.deepcopy(self.forms[3])
        form['private_mapping'] = {'source_path': str(self.media), 'score': 'SECRET-MODEL-SCORE'}
        form['items'][0]['source_path'] = str(self.media)
        page = cli.assignment_page(form)
        embedded = re.search(r'<script id="assignment-data" type="application/json">(.*?)</script>', page, re.S).group(1)
        clean = json.loads(embedded)
        self.assertEqual(clean, self.forms[3])
        self.assertNotIn('SECRET-MODEL-SCORE', page)
        self.assertNotIn(str(self.media), page)
        self.assertNotIn('MUST NOT LEAK', page)
        self.assertNotIn('private_mapping', page)
        self.assertNotIn('checked>', page)
        self.assertIn("connect-src 'none'", page)
        self.assertIn("select.disabled=data.condition==='silent'&&soundKeys.has(key)", page)
        self.assertTrue(all(v is None for r in clean['items'] for v in r['ratings'].values()))
        self.assertTrue(all(not r['completed'] for r in clean['items']))
        self.assertFalse(clean['human_provenance']['human_only'])
        completed = self.completed()[3]
        with self.assertRaises(ValueError): cli.assignment_page(completed)
        bad = copy.deepcopy(self.forms[3]);bad['items'][0]['media'] = 'https://external.example/private.mp4'
        with self.assertRaises(ValueError): cli.assignment_page(bad)

    def test_invalid_intervals_and_duplicate_candidates_rejected(self):
        bad = copy.deepcopy(self.data)
        bad['matches'][0]['events'][0]['clip_end_sec'] = 21
        with self.assertRaises(ValueError): prepare(bad)
        bad = copy.deepcopy(self.data)
        bad['matches'][0]['events'][1].update(clip_start_sec=0, clip_end_sec=8)
        with self.assertRaisesRegex(ValueError, '相同区间'): prepare(bad)

    def test_native_silent_does_not_create_audio_control(self):
        data = copy.deepcopy(self.data)
        data['matches'][0]['sources'][0]['audio_available'] = False
        _, forms = prepare(data)
        self.assertEqual(len(forms), 3)
        self.assertTrue(all(f['condition'] == 'silent' for f in forms))

    def test_unfilled_forms_never_become_zero_grade_or_human_truth(self):
        report = evaluate(self.private, self.forms, self.predictions(['one']))
        self.assertEqual(report['human_completed_rows'], 0)
        self.assertIsNone(report['systems_by_match'][0]['top5_fun'])
        self.assertIsNone(report['systems_by_match'][0]['ndcg5_judged_pool'])
        self.assertIsNone(report['agreement']['audio']['overall_fun']['alpha'])

    def test_fake_provenance_repeated_person_and_wrong_assignment_rejected(self):
        forms = self.completed()
        forms[0]['human_provenance']['human_only'] = False
        with self.assertRaisesRegex(ValueError, '真人'): validate_forms(self.private, forms)
        forms = self.completed()
        forms[3]['human_provenance']['annotator_id'] = forms[0]['human_provenance']['annotator_id']
        with self.assertRaisesRegex(ValueError, '同一真人'): validate_forms(self.private, forms)
        forms = self.completed()
        forms[0]['items'].reverse()
        with self.assertRaisesRegex(ValueError, '重排'): validate_forms(self.private, forms)

    def test_silent_audio_unknown_and_boundary_safety(self):
        forms = self.completed()
        forms[3]['items'][0]['ratings']['related_laughter'] = 0
        with self.assertRaisesRegex(ValueError, '静音'): validate_forms(self.private, forms)
        forms = self.completed()
        forms[0]['items'][0]['stages']['unexpected'] = {'start_sec': 2, 'end_sec': 20}
        with self.assertRaisesRegex(ValueError, '片段内'): validate_forms(self.private, forms)
        forms = self.completed()
        forms[0]['items'][0]['stages']['setup'] = {'start_sec': 4, 'end_sec': 5}
        forms[0]['items'][0]['stages']['unexpected'] = {'start_sec': 2, 'end_sec': 3}
        with self.assertRaisesRegex(ValueError, '起点顺序'): validate_forms(self.private, forms)

    def test_unknown_requires_reason_and_boolean_is_not_ordinal_rating(self):
        forms = self.completed()
        forms[0]['items'][0]['unknown_reason'] = None
        with self.assertRaisesRegex(ValueError, '未知'): validate_forms(self.private, forms)
        forms = self.completed()
        forms[0]['items'][0]['ratings']['overall_fun'] = True
        with self.assertRaisesRegex(ValueError, '整数'): validate_forms(self.private, forms)

    def test_ordinal_alpha_perfect_disagreement_missing_and_constant(self):
        self.assertEqual(ordinal_alpha([[0, 0, 0], [4, 4, 4]])['alpha'], 1)
        self.assertAlmostEqual(ordinal_alpha([[0, 4], [0, 4]])['alpha'], -.5)
        # Singleton ratings must not affect expected disagreement marginals.
        self.assertEqual(ordinal_alpha([[0, 4], [0, 4], [2, None]])['alpha'], -.5)
        self.assertIsNone(ordinal_alpha([[2, 2, 2], [2, 2]])['alpha'])
        self.assertIsNone(ordinal_alpha([[1, None], [None, None]])['alpha'])

    def test_three_known_ratings_required_and_disagreements_retained(self):
        forms = self.completed()
        item = forms[0]['items'][0]['item_id']
        rows = [r for f in forms[:3] for r in f['items'] if r['item_id'] == item]
        for row, grade in zip(rows, [0, 4, None]): row['ratings']['overall_fun'] = grade
        report = evaluate(self.private, forms)
        judgment = next(j for j in report['judgments'] if j['item_id'] == item and j['condition'] == 'audio')
        self.assertIsNone(judgment['ratings_median']['overall_fun'])
        self.assertIn('overall_fun', judgment['disputed_dimensions'])
        self.assertTrue(judgment['needs_adjudication'])

    def test_top5_uses_independent_overall_ratings_and_requires_full_pool(self):
        forms = self.completed()
        # Inverting component scores cannot change the human ranking reference.
        for form in forms:
            for row in form['items']:
                row['ratings']['unexpected_contrast'] = 4-row['ratings']['selection_suitability']
        report = evaluate(self.private, forms, self.predictions(['one', 'two']))
        audio = report['systems_by_match'][0]
        self.assertEqual(audio['ndcg5_judged_pool'], 1)
        self.assertEqual(audio['top5_suitability'], 2)
        self.assertEqual(audio['suitable_fraction_returned'], .5)
        self.assertEqual(audio['missing_slots'], 3)
        reversed_report = evaluate(self.private, forms, self.predictions(['two', 'one']))
        self.assertLess(reversed_report['systems_by_match'][0]['ndcg5_judged_pool'], 1)
        other_id = next(i['item_id'] for i in self.private['items'] if i['event_id'] == 'two')
        for form in forms[:3]:
            next(r for r in form['items'] if r['item_id'] == other_id)['ratings']['selection_suitability'] = None
        partial = evaluate(self.private, forms, self.predictions(['one']))
        self.assertEqual(partial['systems_by_match'][0]['top5_suitability'], 4)
        self.assertIsNone(partial['systems_by_match'][0]['ndcg5_judged_pool'])

    def test_predictions_require_explicit_failures_and_unique_in_pool_ids(self):
        for ids in (['one', 'one'], ['unjudged-model-event']):
            with self.assertRaises(ValueError): evaluate(self.private, self.forms, self.predictions(ids))
        missing = {'systems': [{'system_id': 'bad', 'matches': []}]}
        with self.assertRaisesRegex(ValueError, '每场'): evaluate(self.private, self.forms, missing)
        empty = evaluate(self.private, self.completed(), self.predictions([]))
        self.assertEqual(empty['systems_by_match'][0]['ndcg5_judged_pool'], 0)
        self.assertIsNone(empty['systems_by_match'][0]['top5_fun'])

    def test_zero_candidate_match_is_retained_and_requires_explicit_empty_output(self):
        data = copy.deepcopy(self.data)
        other = Path(self.temp.name)/'other.mp4'
        other.write_bytes(b'independent empty-pool audit fixture')
        empty_match = copy.deepcopy(data['matches'][0])
        empty_match['match_id'] = 'match-empty'
        empty_match['sources'][0]['path'] = str(other)
        empty_match['events'] = []
        data['matches'].append(empty_match)
        self.private, self.forms = prepare(data)
        with self.assertRaisesRegex(ValueError, '每场'):
            evaluate(self.private, self.forms, self.predictions(['one']))
        predictions = self.predictions(['one'])
        predictions['systems'][0]['matches'].append({'match_id': 'match-empty', 'ranked_event_ids': []})
        report = evaluate(self.private, self.completed(), predictions)
        self.assertEqual(report['matches'], 2)
        self.assertEqual(report['match_coverage'], {
            'declared_matches': 2, 'matches_with_candidates': 1,
            'matches_without_candidates': 1, 'empty_pool_match_ids': ['match-empty']})
        empty_rows = [r for r in report['systems_by_match'] if r['match_id'] == 'match-empty']
        self.assertEqual(len(empty_rows), 2)
        for row in empty_rows:
            self.assertEqual(row['selected_count'], 0)
            self.assertEqual(row['pool_with_reference_count'], 0)
            self.assertIsNone(row['top5_fun'])
            self.assertIsNone(row['top5_suitability'])
            self.assertIsNone(row['ndcg5_judged_pool'])
            self.assertEqual(row['ndcg_status'], 'empty_candidate_pool')
        for summary in report['system_summaries']:
            metric = summary['metrics']['ndcg5_judged_pool']
            self.assertEqual(metric['matches'], 1)
            self.assertEqual(metric['missing_matches'], 1)
            self.assertEqual(metric['mean'], 1)

    def test_cluster_bootstrap_uses_matches_and_declines_tiny_sample_ci(self):
        small = cluster_mean_interval({'a': 1, 'b': None}, samples=200)
        self.assertEqual(small['mean'], 1)
        self.assertEqual(small['missing_matches'], 1)
        self.assertIsNone(small['ci95'])
        sufficient = {str(i): i/10 for i in range(10)}
        a = cluster_mean_interval(sufficient, samples=500)
        self.assertEqual(a, cluster_mean_interval(sufficient, samples=500))
        self.assertEqual(a['resampling_unit'], 'match')
        self.assertLessEqual(a['ci95'][0], a['mean'])
        self.assertGreaterEqual(a['ci95'][1], a['mean'])


if __name__ == '__main__':
    unittest.main()
