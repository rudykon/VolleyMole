"""Model parallelism must not change temporal boundaries or fallback decisions."""
from fractions import Fraction
from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import MagicMock, patch, call

from volleymole.gpu_stages import GPUStages, ROLES, parse_devices
from volleymole.shared import infer_batch
from volleymole.video import FramePacket
from volleymole.telemetry import UsageMonitor
from volleymole.jersey import pin_ocr_device


DEVICES = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']


class FakeTracker:
    def __init__(self):
        self.seen = []
        self.batches = []

    def predict(self, packets):
        self.batches.append([p.index for p in packets])
        result = []
        for p in packets:
            # Deliberately history-dependent, like real radius smoothing.
            self.seen.append(p.index)
            result.append({'Frame': p.index, 'Visibility': int(p.index % 3 == 0),
                           'history': sum(self.seen[-12:])})
        return result


class FakeDetector:
    def __init__(self):
        self.batches = []

    def detect(self, images):
        self.batches.append(list(images))
        return [[{'class': 'ball', 'confidence': .8}] if i % 3 == 1 else [] for i in images]


class FakeState:
    def classify(self, images):
        return SimpleNamespace(label='play', confidence=1., probabilities={'play': 1.},
                               sampled_indices=[0, len(images)-1])


class FourGPUStageTests(unittest.TestCase):
    def test_validates_four_distinct_explicit_devices(self):
        self.assertIsNone(parse_devices(None))
        self.assertEqual(parse_devices('cuda:00, cuda:1,cuda:2,cuda:3'), DEVICES)
        for value in ('', 'cuda:0', 'cuda:0,cuda:1,cuda:2,cuda:2',
                      'cuda:00,cuda:0,cuda:2,cuda:3', 'cpu,cuda:1,cuda:2,cuda:3',
                      'cuda:-1,cuda:1,cuda:2,cuda:3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_devices(value)

    def test_all_four_stages_can_overlap_and_failures_propagate(self):
        barrier = threading.Barrier(4)
        with GPUStages(DEVICES, bind_cuda=False) as workers:
            futures = [workers.submit(role, barrier.wait, 5) for role in ROLES]
            self.assertEqual(sorted(f.result() for f in futures), [0, 1, 2, 3])
            self.assertEqual(workers.report()['max_concurrent_stage_tasks'], 4)
            def fail():
                raise RuntimeError('worker failed')
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                workers.submit('tracking', fail).result()
        self.assertEqual(workers.active, 0)

    def test_parallel_matches_serial_with_history_and_short_tail(self):
        def run(workers):
            tracker, action, person, auxiliary = FakeTracker(), FakeDetector(), FakeDetector(), FakeDetector()
            outputs = []
            for start, end in ((0,90), (90,180), (180,182)):
                packets = [FramePacket(i, i*i+17, Fraction(1,1000), .017, i) for i in range(start,end)]
                result = infer_batch(packets,FakeState(),action,person,auxiliary,tracker,workers)
                balls, actions, people, extras, states, windows = result
                outputs.append((balls, actions, people, extras, [vars(s) for s in states], windows))
            self.assertEqual(tracker.batches[-1], [180,181])
            self.assertEqual(tracker.seen, list(range(182)))
            self.assertEqual(action.batches[-1], [180,181])
            return outputs, tracker.batches, action.batches, person.batches, auxiliary.batches
        serial = run(None)
        with GPUStages(DEVICES, bind_cuda=False) as workers:
            parallel = run(workers)
        self.assertEqual(serial, parallel)

    def test_incomplete_person_batch_is_rejected(self):
        person = SimpleNamespace(detect=lambda images: [])
        packets = [FramePacket(0, 0, Fraction(1,30), 0., 0)]
        with self.assertRaisesRegex(RuntimeError, 'incomplete batch'):
            infer_batch(packets, FakeState(), FakeDetector(), person, FakeDetector(), FakeTracker())

    def test_telemetry_resets_and_reports_each_device_separately(self):
        cuda = SimpleNamespace(init=MagicMock(), reset_peak_memory_stats=MagicMock(),
            max_memory_allocated=MagicMock(return_value=1234), max_memory_reserved=MagicMock(return_value=2345),
            get_device_name=MagicMock(return_value='test GPU'))
        fake_torch = SimpleNamespace(cuda=cuda,version=SimpleNamespace(cuda='test'))
        with patch.dict('sys.modules',{'torch':fake_torch}), patch.object(UsageMonitor,'poll'):
            with UsageMonitor(DEVICES) as monitor:
                pass
            report = monitor.report()
        self.assertEqual(cuda.reset_peak_memory_stats.call_args_list,[call(d) for d in DEVICES])
        self.assertEqual(list(report['torch_devices']),DEVICES)
        self.assertNotIn('torch_peak_allocated_bytes',report)  # not a misleading sum of separate peaks

    def test_cli_rejects_ignored_or_conflicting_multi_gpu_options_before_io(self):
        from volleymole.run_match import main as run_main
        from volleymole.inference import main as infer_main
        devices = ['--devices',','.join(DEVICES)]
        for extra in (['--device','cpu'], ['--inference-mode','independent'], ['--evidence-cache','missing']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                run_main(['--video','missing.mp4',*devices,*extra])
            self.assertEqual(cm.exception.code,2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            infer_main(['--video','missing.mp4','--output','missing','--kind','analytics',*devices])
        self.assertEqual(cm.exception.code,2)

    def test_optional_ocr_does_not_reintroduce_all_device_data_parallel(self):
        class Wrapper:
            def __init__(self):
                self.module = MagicMock()
                self.module.to.return_value = self.module
        detector, recognizer = Wrapper(), Wrapper()
        reader = SimpleNamespace(detector=detector, recognizer=recognizer)
        with patch.dict('sys.modules', {'torch':SimpleNamespace(nn=SimpleNamespace(DataParallel=Wrapper))}):
            pin_ocr_device(reader, 'cuda:2')
        self.assertIs(reader.detector, detector.module)
        self.assertIs(reader.recognizer, recognizer.module)
        detector.module.to.assert_called_once_with('cuda:2')
        recognizer.module.to.assert_called_once_with('cuda:2')
