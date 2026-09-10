from concurrent.futures import Future
from fractions import Fraction
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from volleymole.gpu_stages import GPUStages
from volleymole.performance import Timings
from volleymole.pipeline import prefetch_batches, ordered_inference
from volleymole.shared import submit_batch, resolve_batch, infer_batch
from volleymole.video import FramePacket
from volleymole.common import save_json, read_json
from volleymole.media_worker import previews
from test_gpu_stages import FakeState, FakeTracker, FakeDetector, DEVICES


class PerformanceTests(unittest.TestCase):
    def test_pipeline_matches_reference_and_retains_temporal_history(self):
        def execute(depth):
            tracker=FakeTracker()
            models=(FakeState(),FakeDetector(),FakeDetector(),FakeDetector(),tracker)
            batches=[[FramePacket(i,i*i+1,Fraction(1,1000),0.,i) for i in range(s,e)]
                     for s,e in [(0,90),(90,180),(180,182)]]
            timings=Timings()
            with GPUStages(DEVICES,bind_cuda=False,auxiliary_device='cuda:0') as workers:
                if depth == 1:
                    results=[(b,infer_batch(b,*models,workers)) for b in batches]
                else:
                    with prefetch_batches(iter(batches),timings) as decoded:
                        results=list(ordered_inference(decoded,
                            lambda b:submit_batch(b,*models,workers,timings),resolve_batch,depth,timings))
            normalized=[]
            for b,(balls,actions,people,extras,states,windows) in results:
                normalized.append((balls,actions,people,extras,[vars(s) for s in states],windows))
            return normalized, tracker.seen, tracker.batches
        self.assertEqual(execute(1),execute(2))
        self.assertEqual(execute(1),execute(4))

    def test_decode_failure_is_propagated_and_owner_joined(self):
        def broken():
            yield [1]
            raise ValueError('decode failure')
        with self.assertRaisesRegex(ValueError,'decode failure'):
            with prefetch_batches(broken(),Timings()) as batches:
                list(batches)

    def test_pipeline_failure_propagates_without_reordering(self):
        def submit(batch):
            f=Future()
            if batch==2:f.set_exception(ValueError('model failure'))
            else:f.set_result(batch)
            return f
        iterator=ordered_inference([1,2,3],submit,lambda f:f.result(),2,Timings())
        self.assertEqual(next(iterator),(1,1))
        with self.assertRaisesRegex(ValueError,'model failure'):next(iterator)

    def test_preview_subset_preserves_frame_requests_and_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            save_json(root/'match_manifest.json',{'source':{'path':'video'},'rallies':[
                {'rally_id':str(i),'preview_times_sec':[i,i+.1,i+.2],
                 'preview_frames':[f'previews/{i}-{j}.jpg' for j in range(3)]} for i in range(5)]})
            with (patch('volleymole.media_worker.frame_at',return_value='pixels') as frame,
                  patch('volleymole.media_worker.cv2.imwrite',return_value=True)):
                previews(root,['1','3'],2)
            self.assertEqual(frame.call_count,6)
            self.assertEqual(read_json(root/'previews/index.json')['files'],
                             [f'previews/{i}-{j}.jpg' for i in (1,3) for j in range(3)])
