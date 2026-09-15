"""Sound adapter boundaries and protocol failures without large model assets."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from volleymole.audio_events import LocalSoundDetector


class GridSed(torch.nn.Module):
    __constants__ = ['frame_hop_samples']
    frame_hop_samples = 320

    def forward(self, waveform: torch.Tensor):
        length = waveform.shape[1] // 320 + 1
        return {'framewise_output': torch.ones((1, length, 1), device=waveform.device) * .75}


class ClipClassifier(torch.nn.Module):
    def forward(self, waveform: torch.Tensor):
        return {'clipwise_output': torch.ones((1, 1))}


class InvalidSed(torch.nn.Module):
    def forward(self, waveform: torch.Tensor):
        return {'framewise_output': torch.ones((1, 3, 1)) * float('nan')}


class TwoClassSed(torch.nn.Module):
    def forward(self, waveform: torch.Tensor):
        return {'framewise_output': torch.tensor([[[.3, .4], [.3, .4]]])}


class AudioDetectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def detector(self, model):
        path = Path(self.directory.name) / 'model.pt'
        torch.jit.save(torch.jit.script(model), str(path))
        return LocalSoundDetector(path, ['Laughter'], device='cpu')

    def test_declared_grid_is_not_stretched_and_events_are_clipped(self):
        detector = self.detector(GridSed())
        audio = np.zeros(4321, dtype=np.float32)
        values = detector.framewise(audio)
        self.assertEqual(values.shape, (14, 1))
        event = detector.detect(audio, start_sec=17.)[0]
        self.assertEqual(event['start_sec'], 17.)
        self.assertAlmostEqual(event['end_sec'], 17. + len(audio)/32000)
        self.assertEqual(event['association'], 'unknown')

    def test_short_real_length_can_be_below_output_grid(self):
        detector = self.detector(GridSed())
        event = detector.detect(np.zeros(1, dtype=np.float32))[0]
        self.assertAlmostEqual(event['end_sec'], 1/32000)

    def test_clipwise_output_cannot_masquerade_as_events(self):
        with self.assertRaisesRegex(ValueError, 'framewise_output'):
            self.detector(ClipClassifier()).framewise(np.zeros(32000, dtype=np.float32))

    def test_nonfinite_probabilities_rejected(self):
        with self.assertRaisesRegex(ValueError, '概率协议'):
            self.detector(InvalidSed()).framewise(np.zeros(32000, dtype=np.float32))

    def test_invalid_waveform_rejected_before_inference(self):
        detector = self.detector(GridSed())
        for audio in (np.array([]), np.array([np.nan]), np.zeros((2, 100))):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                detector.framewise(audio)

    def test_invalid_detection_threshold_or_offset_rejected(self):
        detector = self.detector(GridSed())
        for start, threshold in ((-1, .5), (float('nan'), .5), (0, 1.1), (0, float('nan'))):
            with self.subTest(start=start, threshold=threshold), self.assertRaises(ValueError):
                detector.detect(np.zeros(32000, dtype=np.float32), start, threshold)

    def test_class_cutoff_changes_detection_but_not_probability_or_other_class(self):
        path = Path(self.directory.name)/'two.pt'
        torch.jit.save(torch.jit.script(TwoClassSed()), str(path))
        detector = LocalSoundDetector(path, ['Laughter', 'Speech'], thresholds={'Laughter': .2})
        events = detector.detect(np.zeros(32000, dtype=np.float32))
        self.assertEqual([event['label'] for event in events], ['Laughter'])
        self.assertAlmostEqual(events[0]['probability'], .3)
        # No sidecar still uses the original 0.5 fallback for both classes.
        plain = LocalSoundDetector(path, ['Laughter', 'Speech'])
        self.assertEqual(plain.detect(np.zeros(32000, dtype=np.float32)), [])

    def test_invalid_class_cutoffs_fail_before_loading_model(self):
        for thresholds in ({'Invented': .2}, {'Laughter': -1}, {'Laughter': True},
                           {'Laughter': float('nan')}, {}, {'Laughter': 1.1}):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                LocalSoundDetector('not-loaded.pt', ['Laughter'], thresholds=thresholds)


if __name__ == '__main__':
    unittest.main()
