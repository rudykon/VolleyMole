"""Protect original ground truth and clip-level metric semantics."""
import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script('evaluate_sound_events')
preparation = load_script('prepare_public_audio')


class PublicAudioEvaluationTests(unittest.TestCase):
    def test_perfect_order_and_fixed_threshold_are_separate(self):
        metrics = evaluation.binary_metrics([1, 1, 0, 0], [.4, .3, .2, .1])
        self.assertEqual(metrics['average_precision'], 1.)
        self.assertEqual(metrics['auroc'], 1.)
        self.assertEqual(metrics['false_negative'], 2)
        self.assertIsNone(metrics['precision_vs_single_class_labels'])

    def test_equal_scores_are_not_given_optimistic_tie_order(self):
        metrics = evaluation.binary_metrics([1, 0, 1, 0], [.7, .7, .7, .7])
        self.assertEqual(metrics['average_precision'], .5)
        self.assertEqual(metrics['auroc'], .5)
        self.assertEqual(metrics['false_positive_vs_single_class_labels'], 2)
        other = evaluation.binary_metrics([0, 0, 1, 1], [.7, .7, .7, .7])
        self.assertEqual(other, metrics)

    def test_unknown_classes_do_not_gain_zero_metric(self):
        for truth in ([], [0, 0], [1, 1]):
            metrics = evaluation.binary_metrics(truth, [.3]*len(truth))
            self.assertIsNone(metrics['average_precision'])
            self.assertIsNone(metrics['auroc'])
        self.assertIsNone(evaluation.binary_metrics([0, 0], [.3, .3])['recall'])

    def test_non_binary_and_invalid_predictions_rejected(self):
        for truth, scores in (([2], [.5]), ([1], [float('nan')]), ([1], [-.2]), ([1], [2.]), ([1], [])):
            with self.assertRaises(ValueError):
                evaluation.binary_metrics(truth, scores)

    def test_framewise_pooling_preserves_below_threshold_probabilities(self):
        class Detector:
            labels = ['Speech', 'Clapping', 'Laughter']
            def framewise(self, samples):
                return np.array([[.9, .2, .01], [.3, .1, .4]])
        self.assertEqual(evaluation.clip_scores(Detector(), np.zeros(10)), {'laughing': .4, 'clapping': .2})

    def test_fold_reports_do_not_change_original_partitions(self):
        rows = [{'fold': 2, 'category': 'laughing', 'scores': {'laughing': .9, 'clapping': .1}},
                {'fold': 4, 'category': 'dog', 'scores': {'laughing': .8, 'clapping': .1}}]
        report = evaluation.summarize(rows)
        self.assertEqual(report['fold_2']['laughing']['positive_clips'], 1)
        self.assertEqual(report['fold_4']['laughing']['nominal_negative_clips'], 1)
        self.assertIsNone(report['fold_1']['laughing']['average_precision'])
        self.assertEqual(report['background_label_disagreements']['laughing'], {'dog': 1})

    def test_changed_labels_rejected_and_original_source_overlap_audited(self):
        rows = [{'filename': f'{fold}-{category}-{take}.wav', 'fold': fold,
                 'target': category, 'category': f'class_{category}', 'take': str(take),
                 'src_file': f'{fold}-{category}-{take}'}
                for fold in range(1, 6) for category in range(50) for take in range(8)]
        preparation.validate_rows(rows)
        bad = copy.deepcopy(rows)
        bad[400]['src_file'] = rows[0]['src_file']
        self.assertEqual(preparation.validate_rows(bad), {rows[0]['src_file']: [1, 2]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root/'esc50.csv'
            with metadata.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_rows = [dict(row, audio_path=f"audio/{row['filename']}") for row in rows]
            manifest = {'dataset': 'ESC-50', 'annotation_file': metadata.name,
                        'annotation_sha256': evaluation.sha256(metadata), 'clips': manifest_rows}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            evaluation.load_manifest(root)
            manifest['clips'][0]['category'] = 'invented_label'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'changes an original annotation'):
                evaluation.load_manifest(root)


if __name__ == '__main__':
    unittest.main()
