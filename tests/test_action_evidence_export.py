import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from fractions import Fraction

from volleymole.action_spotting import FEATURE_DIM
from volleymole.video import FramePacket

spec = importlib.util.spec_from_file_location('export_actions', Path(__file__).resolve().parents[1]/'scripts/export_action_evidence.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class FakeEncoder:
    compute_policy = {'device':'cpu', 'autocast':'disabled'}

    def encode(self, frames, batch):
        return np.zeros((len(frames), FEATURE_DIM), dtype=np.float32)


class FakeAdapter:
    def __init__(self, *args):
        self.backbone = FakeEncoder()
        self.head = nn.Linear(FEATURE_DIM, 6)
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.fill_(1)
        self.config = {'fps':25, 'window':4, 'stride':2, 'threshold':.5, 'nms_sec':.04}


class ExportTests(unittest.TestCase):
    def fixture(self, root, *, frames=7, rate='25/1', timeout=100):
        checkpoint = root/'head.pt'; checkpoint.write_bytes(b'unused')
        checkpoint.with_suffix('.manifest.json').write_text(json.dumps({'checkpoint_sha256':'fixed', 'backbone_sha256':'backbone'}))
        args = SimpleNamespace(video=root/'video.mp4', checkpoint=checkpoint, backbone=root/'backbone.pt',
            output=root/'output.json', cache=root/'cache', device='cpu', total_timeout=timeout,
            max_cache_mib=1, feature_batch=2)
        source = {'duration_sec':frames/25, 'start_sec':0, 'nominal_fps':rate,
                  'average_fps':rate, 'frame_count':frames}
        return args, source

    def packets(self, count, step=1):
        for index in range(count):
            yield FramePacket(index, index*step, Fraction(1, 25), 0, np.zeros((16, 24, 3), dtype=np.uint8))

    def run_fixture(self, args, source, packets):
        with patch.object(mod, 'probe', return_value=source), patch.object(mod, 'file_hash', return_value='fixed'), \
             patch.object(mod, 'LocalActionSpotter', FakeAdapter), patch.object(mod, 'decode', return_value=packets):
            return mod.export(args)

    def test_full_export_processes_short_tail_and_keeps_actual_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            args, source = self.fixture(Path(directory))
            result = self.run_fixture(args, source, self.packets(7))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['finalized_frames'], 7)
            for event in result['events']:
                self.assertAlmostEqual(event['time_sec'], event['source_pts']/25)
                self.assertEqual(event['observation_status'], 'candidate')
            self.assertEqual(list(args.cache.iterdir()), [])

    def test_non25fps_is_unsupported_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, source = self.fixture(Path(directory), rate='30/1')
            with patch.object(mod, 'probe', return_value=source), patch.object(mod, 'file_hash', return_value='fixed'), \
                 patch.object(mod, 'LocalActionSpotter') as loader:
                result = mod.export(args)
            loader.assert_not_called()
            self.assertEqual(result['status'], 'unsupported')
            self.assertEqual(result['events'], [])

    def test_real_pts_mismatch_is_unsupported_and_no_guessed_times(self):
        with tempfile.TemporaryDirectory() as directory:
            args, source = self.fixture(Path(directory))
            result = self.run_fixture(args, source, self.packets(7, step=2))
            self.assertEqual(result['status'], 'unsupported')
            self.assertEqual(result['events'], [])

    def test_cache_budget_yields_explicit_partial_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            args, source = self.fixture(Path(directory), frames=30)
            args.max_cache_mib = .08
            result = self.run_fixture(args, source, self.packets(30))
            self.assertEqual(result['status'], 'partial')
            self.assertLess(result['finalized_frames'], 30)
            self.assertLessEqual(result['finalized_frames'], result['processed_frames'])
            if result['events']:
                self.assertLess(max(e['time_sec'] for e in result['events']), result['covered_end_sec'])

    def test_deadline_does_not_claim_complete_empty_video(self):
        with tempfile.TemporaryDirectory() as directory:
            args, source = self.fixture(Path(directory), timeout=.01)
            with patch.object(mod.time, 'monotonic', side_effect=[0, 1, 1]):
                result = self.run_fixture(args, source, self.packets(7))
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['events'], [])


if __name__ == '__main__': unittest.main()
