import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('sv_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluate_svhighlights.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def meta(count=3, start=10, end=16):
    return {'label_bin_sec': 2, 'label_bins': count, 'source_trim_start_sec': start, 'source_trim_end_sec': end}


class SVHighlightsEvaluationTests(unittest.TestCase):
    def test_standard_ap_uses_recall_increments_and_grouped_ties(self):
        self.assertAlmostEqual(mod.score_metrics([1, 0, 1], [1, 1, 1])['ap_noninterpolated'], 2 / 3)
        self.assertAlmostEqual(mod.score_metrics([0, 1, 1], [3, 2, 1])['ap_noninterpolated'], 7 / 12)
        self.assertEqual(mod.score_metrics([1, 0], [2, 1])['ap_noninterpolated'], 1)
        self.assertEqual(mod.score_metrics([0, 0], [2, 1])['ap_noninterpolated'], 0)

    def test_topk_is_exactly_k_and_ties_prefer_earlier_bin(self):
        result = mod.score_metrics([0, 1, 1], [1, 1, 1])
        self.assertEqual(result['hit1'], 0)
        self.assertEqual(result['hitk_gt_positive_bins'], .5)
        self.assertEqual(result['k'], 2)

    def test_input_requires_all_expected_ids_and_exact_bin_lengths(self):
        metadata = {'volleyball_1': meta(), 'volleyball_2': meta()}
        good = {'vid': 'volleyball_1', 'pred_saliency_scores': [1, 2, 3]}
        with self.assertRaises(ValueError):
            mod.normalize_predictions([good], metadata)
        mod.normalize_predictions([good], metadata, allow_subset=True)
        for length in (2, 4):
            with self.assertRaises(ValueError):
                mod.normalize_predictions([{**good, 'pred_saliency_scores': [0] * length}], {'volleyball_1': meta()})
        with self.assertRaises(ValueError):
            mod.normalize_predictions([good, good], metadata, allow_subset=True)
        with self.assertRaises(ValueError):
            mod.normalize_predictions([{**good, 'vid': 'unknown'}], metadata, allow_subset=True)

    def test_reject_nonfinite_boolean_and_strings(self):
        for value in (float('nan'), float('inf'), '-Inf', True):
            with self.assertRaises(ValueError):
                mod.score_metrics([1], [value])

    def test_source_mapping_subtracts_trim_and_keeps_half_open_boundaries(self):
        scores, audit = mod.intervals_to_scores({'time_basis': 'source', 'events': [
            {'start_sec': 12, 'end_sec': 14, 'score': .9},
            {'start_sec': 8, 'end_sec': 10, 'score': 1},
            {'start_sec': 15, 'end_sec': 20, 'score': .6},
        ]}, meta())
        self.assertEqual(scores, [0, .9, .6])
        self.assertEqual(audit['source_trim_subtracted_sec'], 10)
        self.assertEqual(audit['clipped_event_count'], 2)
        self.assertEqual(audit['outside_evaluation_event_count'], 1)

    def test_no_label_is_created_for_unannotated_partial_tail(self):
        scores, audit = mod.intervals_to_scores({'time_basis': 'trimmed', 'events': [
            {'start_sec': 6, 'end_sec': 6.4, 'score': 1},
        ]}, meta(end=16.5))
        self.assertEqual(scores, [0, 0, 0])
        self.assertEqual(audit['unlabelled_source_tail_sec'], .5)
        self.assertEqual(audit['outside_evaluation_event_count'], 1)

    def test_decimal_trim_roundoff_does_not_add_an_adjacent_bin(self):
        scores, _ = mod.intervals_to_scores({'time_basis': 'source', 'events': [
            {'start_sec': 14.358218, 'end_sec': 16.358218, 'score': 1},
        ]}, meta(start=12.358218, end=18.358218))
        self.assertEqual(scores, [0, 1, 0])

    def test_event_scores_can_be_negative_without_background_masking(self):
        scores, _ = mod.intervals_to_scores({'time_basis': 'trimmed', 'events': [
            {'start_sec': 2, 'end_sec': 4, 'score': -5},
            {'start_sec': 2.5, 'end_sec': 3, 'score': -3},
        ]}, meta())
        self.assertEqual(scores, [0, -3, 0])

    def test_intervals_require_explicit_basis_valid_duration_and_scores(self):
        event = {'start_sec': 0, 'end_sec': 2, 'score': .7}
        with self.assertRaises(ValueError):
            mod.intervals_to_scores({'events': [event]}, meta())
        for update in ({'start_sec': -1}, {'end_sec': 0}, {'score': None}):
            with self.assertRaises(ValueError):
                mod.intervals_to_scores({'time_basis': 'trimmed', 'events': [{**event, **update}]}, meta())

    def test_loudness_adapter_only_removes_known_final_bin_and_records_silence(self):
        rows, audit = mod.loudness_predictions({'volleyball_4': [-30, '-Inf', -10, 999]}, {'volleyball_4': meta()})
        self.assertEqual(rows[0]['pred_saliency_scores'], [-30, -31, -10])
        self.assertTrue(audit[0]['removed_only_final_bin'])
        self.assertEqual(audit[0]['removed_final_bin_value'], 999)
        self.assertEqual(audit[0]['negative_infinity_silence_bins'], 1)
        with self.assertRaises(ValueError):
            mod.loudness_predictions({'volleyball_1': [0, 0, 0, 0]}, {'volleyball_1': meta()})


if __name__ == '__main__':
    unittest.main()
