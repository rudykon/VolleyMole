import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from volleymole.action_spotting import CLASSES, TemporalHead, dense_probabilities, proposals, soft_targets, file_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train_local_actions', ROOT/'scripts/train_action_spotter.py')
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)


class LocalActionTests(unittest.TestCase):
    def test_soft_targets_preserve_original_and_adjacent_classes(self):
        original = [{'label':'spike', 'frame':10}, {'label':'block', 'frame':11}]
        before = copy.deepcopy(original)
        targets = soft_targets(original, 30, 25)
        self.assertEqual(original, before)
        self.assertEqual(targets[10, CLASSES.index('spike')], 1)
        self.assertEqual(targets[11, CLASSES.index('block')], 1)
        self.assertGreater(targets[10, CLASSES.index('block')], .5)
        self.assertEqual(targets[0].sum(), 0)
        with self.assertRaises(ValueError): soft_targets([{'label':'serve', 'frame':30}], 30, 25)

    def test_classwise_nms_keeps_real_source_time_and_distinct_classes(self):
        scores = np.zeros((5, len(CLASSES)))
        scores[1, 3] = .9; scores[2, 3] = .8; scores[2, 4] = .9
        times = [12, 12.04, 12.08, 12.12, 12.16]
        rows = proposals(scores, times, .5, .24)
        self.assertEqual([(r['label'], r['time_sec']) for r in rows], [('spike',12.04), ('block',12.08)])
        self.assertEqual(rows[0]['raw_probability'], .9)

    def test_repeated_timestamp_and_invalid_probability_rejected(self):
        scores = np.zeros((2, len(CLASSES)))
        with self.assertRaises(ValueError): proposals(scores, [0, 0], .5, .2)
        scores[0, 0] = float('nan')
        with self.assertRaises(ValueError): proposals(scores, [0, .04], .5, .2)

    def test_padded_training_does_not_change_short_sequence_prediction(self):
        torch.manual_seed(4)
        head = TemporalHead(input_dim=12).eval()
        short = torch.randn(1, 7, 12)
        padded = torch.cat((short, torch.randn(1, 4, 12)), dim=1)
        with torch.inference_mode():
            actual = head(padded, torch.tensor([7]))[:, :7]
            expected = head(short)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    def test_dense_inference_covers_tail(self):
        head = TemporalHead(input_dim=12).eval()
        result = dense_probabilities(head, np.zeros((17, 12), dtype=np.float32), window=8, stride=4)
        self.assertEqual(result.shape, (17, len(CLASSES)))
        self.assertTrue(np.isfinite(result).all())

    def test_training_match_leakage_rejected(self):
        a = {'split':'train', 'original_matches':['shared'], 'samples':[{'video':'shared/a'}]}
        b = {'split':'val', 'original_matches':['shared'], 'samples':[{'video':'shared/b'}]}
        with self.assertRaises(ValueError): train.check_training_separation(a, b)
        with self.assertRaises(ValueError): train.check_training_separation({**a, 'split':'test'}, b)

    def test_loader_only_reads_author_requested_split_and_rejects_changes(self):
        original = {'video':'training/a', 'fps':25, 'num_frames':25, 'events':[{'frame':4, 'label':'serve'}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root/'raw/vnl_1.5'; raw.mkdir(parents=True)
            source = raw/'train.json'; source.write_text(json.dumps([original]))
            sample = {'video':original['video'], 'match_id':'training', 'split':'train',
                      'annotation_row':original, 'fps':25, 'frames':25, 'source_declared_num_frames':25,
                      'first_frame_file':'000000.jpg'}
            value = {'dataset':'VNL-STES', 'samples':[sample], 'original_annotation_files':[
                {'path':'raw/vnl_1.5/train.json', 'sha256':file_hash(source)}]}
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(value))
            result = train.load_split(manifest, 'train')
            self.assertEqual(result['split'], 'train')
            # There is deliberately no test.json in this fixture.
            sample['annotation_row'] = {**original, 'events':[]}
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError): train.load_split(manifest, 'train')

    def test_fixed_metrics_do_not_merge_same_class_duplicate_predictions(self):
        row = {'sample':{'video':'test/a', 'fps':25}, 'truth':[{'label':'serve','time_sec':.04}]}
        values = np.zeros((20, len(CLASSES)))
        values[1, 0] = .9; values[8, 0] = .8
        result, _ = train.score_rows([row], [values], .5, .12, .5)
        self.assertEqual(result['micro']['tp'], 1)
        self.assertEqual(result['micro']['fp'], 1)


if __name__ == '__main__': unittest.main()
