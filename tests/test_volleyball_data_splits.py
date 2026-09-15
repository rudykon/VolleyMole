"""Source-partition sampling stays label-blind and retains the original clock."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('vnl_acquisition', Path(__file__).resolve().parents[1]/'scripts/acquire_volleyball_benchmark.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class VolleyballSplitTests(unittest.TestCase):
    def splits(self):
        return {split: [{'video': f'{match}/rally_{i:06d}', 'events': [{'label': 'unused_by_selector'}],
                         'num_frames': i+1} for match in matches for i in range(10)]
                for split, matches in {'train': ['a', 'b'], 'val': ['c'], 'test': ['d']}.items()}

    def test_each_original_training_match_receives_fixed_id_coverage(self):
        selected = mod.select_samples(self.splits(), 'train', 0, per_match=2)
        self.assertEqual([row['video'] for row in selected],
                         ['a/rally_000000', 'a/rally_000009', 'b/rally_000000', 'b/rally_000009'])

    def test_selection_never_depends_on_events_lengths_or_original_order(self):
        original = self.splits()
        changed = copy.deepcopy(original)
        for rows in changed.values():
            rows.reverse()
            for row in rows:
                row['events'] = []
                row['num_frames'] = 99999
        for split in ('train', 'val', 'test'):
            expected = [r['video'] for r in mod.select_samples(original, split, 6)]
            self.assertEqual([r['video'] for r in mod.select_samples(changed, split, 6)], expected)

    def test_new_test_ids_exclude_every_previously_observed_id(self):
        splits = self.splits()
        excluded = {r['video'] for r in mod.select_samples(splits, 'test', 4)}
        chosen = mod.select_samples(splits, 'test', 6, excluded=excluded)
        self.assertEqual(len(chosen), 6)
        self.assertFalse(excluded & {r['video'] for r in chosen})
        with self.assertRaises(ValueError): mod.select_samples(splits, 'test', 7, excluded=excluded)

    def test_match_leakage_duplicate_ids_and_insufficient_match_ids_fail(self):
        splits = self.splits()
        splits['val'][0]['video'] = 'a/other_rally'
        with self.assertRaises(ValueError): mod.select_samples(splits, 'train', 2)
        splits = self.splits(); splits['train'].append(copy.deepcopy(splits['train'][0]))
        with self.assertRaises(ValueError): mod.select_samples(splits, 'train', 2)
        with self.assertRaises(ValueError): mod.select_samples(self.splits(), 'train', 0, per_match=11)

    def test_prepared_directories_keep_legacy_manifest_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root/'manifest.json'; old.write_text('{"prior_observed_test":true}')
            expected = old.read_bytes()
            for split in ('train_adaptation', 'val_development', 'test_final_sealed'):
                prepared = mod.prepare_layout(root, split)
                self.assertEqual((prepared/'raw').resolve(), (root/'raw').resolve())
                self.assertEqual((prepared/'videos').resolve(), (root/'videos').resolve())
            self.assertEqual(old.read_bytes(), expected)
            with self.assertRaises(ValueError): mod.prepare_layout(root, '../escape')

    def test_encoding_cannot_upsample_drop_duplicate_frames_or_add_audio(self):
        source = {'streams': [{'codec_type': 'video', 'nb_frames': '50', 'r_frame_rate': '25/1',
                              'avg_frame_rate': '25/1', 'duration': '2.0', 'start_time': '0'}]}
        mod.validate_encoded_video(source, 50, 25.)
        for key, value in (('nb_frames', '100'), ('r_frame_rate', '50/1'),
                           ('avg_frame_rate', '50/1'), ('duration', '4'), ('start_time', '.04')):
            altered = copy.deepcopy(source); altered['streams'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                mod.validate_encoded_video(altered, 50, 25.)
        source['streams'].append({'codec_type': 'audio'})
        with self.assertRaises(ValueError): mod.validate_encoded_video(source, 50, 25.)


if __name__ == '__main__':
    unittest.main()
