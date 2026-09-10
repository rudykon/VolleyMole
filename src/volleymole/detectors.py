"""YOLO inference over caller-owned frames; no model downloads or video readers.

Batch scheduling and JSON conversion are implemented inside VolleyMole. Uses
the same action/ball/person weights and thresholds as the deployed ML core.
Raw action labels (including serve/ball) and pose indices are retained.
"""
import numpy as np
from time import perf_counter
from .performance import Timings


def resolve_device(device):
    import torch
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device = 'cuda:0'
    if device.startswith('cuda'):
        index = int(device.split(':')[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f'Requested CUDA device is unavailable: {device}')
    return device


class Detector:
    def __init__(self, registry, kind, device, *, half=False, batch_size=8):
        from ultralytics import YOLO
        registry.verify([kind])
        self.model = YOLO(str(registry.target(kind))).to(device)
        self.kind, self.device = kind, device
        self.half = half and device.startswith('cuda')
        self.batch_size = batch_size
        self.frames = self.calls = 0
        self.timings = Timings()

    def detect(self, frames):
        rows = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start:start+self.batch_size]
            began = perf_counter()
            results = self.model.predict(batch, conf=.25, iou=.45, imgsz=640,
                quantize=16 if self.half else None, device=self.device, verbose=False)
            self.timings.add('predict_including_preprocess_transfers_nms', perf_counter()-began)
            if len(results) != len(batch):
                raise RuntimeError(f'{self.kind} returned an incomplete batch')
            # Ultralytics already measures these with its device-aware timers.
            # Per-result values are ms/image; sum to recover batch durations.
            for label in ('preprocess','inference','postprocess'):
                self.timings.add('library_'+label, sum((r.speed or {}).get(label,0.) for r in results)/1000.)
            began = perf_counter()
            for result in results:
                result = result.cpu()
                detections = []
                if result.boxes is not None:
                    for i, (box, confidence, class_id) in enumerate(zip(result.boxes.xyxy.numpy(),
                            result.boxes.conf.numpy(), result.boxes.cls.numpy())):
                        if not np.isfinite(box).all() or not np.isfinite(confidence):
                            raise ValueError(f'Non-finite {self.kind} result')
                        item = {'class': 'player' if self.kind=='person' else result.names[int(class_id)],
                                'class_id': int(class_id), 'confidence': float(confidence),
                                'xyxy': [float(x) for x in box]}
                        if result.keypoints is not None:
                            # Keep COCO positions, including invisible points; do not shift indices.
                            item['keypoints'] = result.keypoints.data[i].numpy().tolist()
                        detections.append(item)
                rows.append(detections)
            self.timings.add('d2h_and_python_conversion', perf_counter()-began)
            self.calls += 1
            self.frames += len(batch)
        return rows
