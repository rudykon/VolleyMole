"""Fixed original development cohorts, threshold tie handling and ZIP safety."""
import importlib.util
import io
from pathlib import Path
import struct
import unittest
import zipfile


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare_sound_calibration')
calibrate = load('calibrate_sound_thresholds')


class SoundCalibrationTests(unittest.TestCase):
    def test_fixed_cohorts_exclude_test_partition_and_esc_sources(self):
        rows = [{'fname': str(i), 'labels': prepare.TARGETS[i % 3] if i < 180 else 'Speech', 'split': 'train'}
                for i in range(300)]
        rows += [{'fname': '-1', 'labels': ','.join(prepare.TARGETS), 'split': 'val'}]
        cohorts = prepare.select_cohorts(list(reversed(rows)), {'0', '1', '2'})
        for target, cohort in cohorts.items():
            self.assertEqual(len(cohort['positive_ids']), 40)
            self.assertEqual(len(cohort['nominal_negative_ids']), 120)
            self.assertFalse({'-1', '0', '1', '2'} & set(cohort['positive_ids']+cohort['nominal_negative_ids']))
        self.assertEqual(cohorts['Laughter']['positive_ids'][0], '3')

    def test_multilabel_positives_are_preserved(self):
        rows = [{'fname': str(i), 'labels': ','.join(prepare.TARGETS) if i < 40 else 'Speech', 'split': 'train'}
                for i in range(160)]
        cohorts = prepare.select_cohorts(rows, set())
        self.assertEqual(cohorts['Laughter']['positive_ids'], cohorts['Cheering']['positive_ids'])

    def test_f2_tie_selects_highest_threshold_exactly(self):
        truth = [True]+[False]*8+[True]
        scores = [.9, .8, .7, .6, .5, .4, .3, .25, .2, .1]
        fitted = calibrate.select_f2_threshold(truth, scores)
        self.assertEqual(fitted['threshold'], .9)
        self.assertAlmostEqual(fitted['f2'], 5/9)

    def test_threshold_uses_scores_including_below_default(self):
        fitted = calibrate.select_f2_threshold([True, True, False], [.02, .01, .001])
        self.assertEqual(fitted, {'threshold': .01, 'f2': 1.})
        with self.assertRaises(ValueError):
            calibrate.select_f2_threshold([True, False], [.2, float('nan')])

    def test_zip_directory_members_reject_path_escape(self):
        for name, safe in [('FSD50K.dev_audio/123.wav', True), ('FSD50K.dev_audio/../escape.wav', False)]:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                archive.writestr(name, b'recorded-audio')
            raw = buffer.getvalue()
            end = raw.rfind(b'PK\x05\x06')
            eocd = struct.unpack_from('<4s4H2IH', raw, end)
            directory = raw[eocd[6]:eocd[6]+eocd[5]]
            if safe:
                entry = prepare.parse_directory(directory)[name]
                self.assertEqual(entry['uncompressed_size'], len(b'recorded-audio'))
            else:
                with self.assertRaises(ValueError):
                    prepare.parse_directory(directory)


if __name__ == '__main__':
    unittest.main()
