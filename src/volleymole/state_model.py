"""VideoMAE inference adapted from volleyball-ml-models (MIT).

Copyright (c) 2025 Masoud Masoumi Moghadam.
Source: models/GameStatusClassifierModule.py at 2374bfd, with the recorded
local strict-loading fix. Full notice: licenses/volleyball-ml-models-MIT.txt.
The GUI/logger/settings dependency graph is intentionally not imported.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from time import perf_counter
from .performance import Timings


@dataclass(frozen=True)
class StateEvidence:
    label: str
    confidence: float
    probabilities: dict
    sampled_indices: list


def uniform_indices(length, size=16):
    if length < 1:
        raise ValueError('No frames for state classification')
    if length <= size:
        return list(range(length)) + [length-1]*(size-length)
    step = (length-1)/(size-1)
    return [min(round(i*step), length-1) for i in range(size)]


class StateClassifier:
    def __init__(self, registry, device):
        import torch
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        registry.verify(['state_weights', 'state_config', 'state_processor'])
        path = str(registry.target('state_weights').parent)
        self.device = device
        self.dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.processor = VideoMAEImageProcessor.from_pretrained(path, local_files_only=True)
        self.model, info = VideoMAEForVideoClassification.from_pretrained(
            path, dtype=self.dtype, output_loading_info=True, local_files_only=True)
        problems = {key: info[key] for key in ('missing_keys', 'unexpected_keys',
            'mismatched_keys', 'error_msgs') if info.get(key)}
        if problems:
            raise RuntimeError(f'VideoMAE checkpoint did not load exactly: {problems}')
        self.labels = self.model.config.id2label
        if set(self.labels.values()) != {'play', 'no-play', 'service'}:
            raise ValueError('Unexpected game-state labels')
        self.model.to(device).eval()
        self.calls = 0
        self.timings = Timings()

    def classify(self, frames):
        import torch
        started = perf_counter()
        indices = uniform_indices(len(frames))
        # Match the deployed core preprocessing, including BGR -> RGB.
        images = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (224,224)) for i in indices]
        inputs = self.processor(images, return_tensors='pt')
        self.timings.add('preprocess', perf_counter()-started)
        started = perf_counter()
        inputs = {key: value.to(device=self.device, dtype=self.dtype) for key,value in inputs.items()}
        self.timings.add('h2d_and_cast', perf_counter()-started)
        started = perf_counter()
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0].float().cpu().numpy()
        self.timings.add('inference_and_d2h', perf_counter()-started)
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Non-finite state inference; refusing to emit fabricated UNKNOWN evidence')
        best = int(probabilities.argmax())
        self.calls += 1
        return StateEvidence(self.labels[best], float(probabilities[best]),
            {self.labels[i]: float(p) for i,p in enumerate(probabilities)}, indices)
