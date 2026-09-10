"""Pinned seq9 grayscale VballNet inference, with caller-owned PTS frames.

Sequence buffering/right alignment and radius logic derive from the MIT tracker.
No OpenCV video reader, CSV-per-frame pandas calls or relative CUDA paths remain.
"""
from collections import deque
from pathlib import Path
import cv2
import numpy as np

from .common import read_json
from .vball_primitives import preprocess_frames, postprocess_heatmap_output, estimate_ball_radius


def seq9_shape(shape):
    return (len(shape)==4 and shape[1:]==[9,288,512]
            and (shape[0] is None or isinstance(shape[0],str) or shape[0]==1))


class BallTracker:
    sequence_length = 9

    def __init__(self, registry, device, profile_directory=None):
        import onnxruntime as ort
        registry.verify(['vball'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.cuda = device.startswith('cuda')
        if profile_directory is not None:
            Path(profile_directory).mkdir(parents=True, exist_ok=True)
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(profile_directory)/'vball-ort')
        providers = ['CPUExecutionProvider']
        if self.cuda:
            # Unified cu128 torch loads CUDA/cuDNN with the same major versions as ORT.
            import torch
            ort.preload_dlls()
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': int(device.split(':')[1])}))
        self.session = ort.InferenceSession(str(registry.target('vball')), options, providers=providers)
        self.session.disable_fallback()
        if self.cuda and 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('CUDA requested but VballNet could not create a CUDA execution provider')
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs)!=1 or not seq9_shape(inputs[0].shape) or not seq9_shape(outputs[0].shape):
            raise ValueError('Expected pinned stateless seq9 heatmap model')
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.buffer = []
        self.previous_gray = None
        self.radius_state = {'raw_history': deque(maxlen=5), 'filtered_history': deque(maxlen=12), 'smoothed_radius': 0.}
        self.calls = self.frames = 0
        self.profile_pending = profile_directory is not None
        self.backend = {'requested': device, 'providers': self.session.get_providers()}

    def predict(self, packets):
        if not 1 <= len(packets) <= self.sequence_length:
            raise ValueError('VballNet batch must contain 1–9 source frames')
        images = [p.pixels for p in packets]
        processed = preprocess_frames(images)
        if not self.buffer:
            self.buffer = [processed[0]]*self.sequence_length
        self.buffer = (self.buffer + processed)[-self.sequence_length:]
        tensor = np.stack(self.buffer, axis=0)[None]
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        if output.shape != (1,9,288,512):
            raise RuntimeError(f'Unexpected VballNet output shape: {output.shape}')
        if not np.isfinite(output).all():
            raise RuntimeError('VballNet emitted non-finite heatmaps')
        predictions = postprocess_heatmap_output(output)[-len(packets):]
        heatmaps = output[0, -len(packets):]
        rows = []
        for packet, (visible,x,y), heatmap in zip(packets, predictions, heatmaps):
            h,w = packet.pixels.shape[:2]
            x = int(x*w/512) if visible else -1
            y = int(y*h/288) if visible else -1
            gray = cv2.cvtColor(packet.pixels, cv2.COLOR_BGR2GRAY)
            radius = estimate_ball_radius(self.previous_gray, gray, x,y,self.radius_state)[0] if visible else 0
            self.previous_gray = gray
            rows.append({'Frame': packet.index, 'Visibility': visible, 'X':x, 'Y':y, 'Radius':radius,
                         'Confidence':float(heatmap.max()), 'SourceTime':packet.source_sec,
                         'evidence': 'vball_heatmap_detection' if visible else 'vball_not_detected'})
        self.calls += 1
        self.frames += len(packets)
        if self.profile_pending:
            path = self.session.end_profiling()
            events = read_json(path)
            counts = {}
            for event in events:
                provider = event.get('args',{}).get('provider')
                if provider:
                    counts[provider] = counts.get(provider,0)+1
            self.backend.update(first_batch_profile=path, kernel_provider_counts=counts)
            self.profile_pending = False
            if self.cuda and not counts.get('CUDAExecutionProvider'):
                raise RuntimeError('VballNet first batch did not execute any CUDA kernels')
        return rows
