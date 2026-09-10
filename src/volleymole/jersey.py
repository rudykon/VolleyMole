"""Independent jersey-number evidence, implemented without volleyball-highlights code.

Consumes existing person detections. OCR is evidence, never player identity:
retain every digit observation and require separate high-confidence timestamps
before a rally can receive a focus-number bonus.
"""
import math
from pathlib import Path
import tempfile
import cv2


PLAYER_SAMPLE_FILTER = 'fps=1:round=up:start_time=0'  # legacy test contract only


def pin_ocr_device(reader, device):
    """EasyOCR wraps CUDA models in all-visible-device DataParallel by default.

    Remove that wrapper from these reader-owned instances: otherwise a reader
    requested on cuda:2 still expects its parameters on DataParallel's cuda:0.
    No global monkeypatch or extra model copy is needed for batch-one OCR.
    """
    if device.startswith('cuda'):
        import torch
        for name in ('detector', 'recognizer'):
            model = getattr(reader, name)
            if isinstance(model, torch.nn.DataParallel):
                setattr(reader, name, model.module.to(device))


def torso_box(box, width, height):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    return [max(0, int(x1+.12*w)), max(0, int(y1+.20*h)),
            min(width, math.ceil(x2-.12*w)), min(height, math.ceil(y1+.66*h))]


def digit_candidate(text, confidence):
    compact = text.strip().replace(' ', '')
    if not compact or not compact.isascii() or not compact.isdigit() or len(compact)>3:
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    return int(compact)


class JerseyReader:
    def __init__(self, registry, number, device, confidence=.75, interval=1., runtime_directory=None):
        import easyocr
        registry.verify(['ocr_recognizer', 'ocr_detector'])
        self.temporary = tempfile.TemporaryDirectory(prefix='volleymole-ocr-') if runtime_directory is None else None
        runtime = Path(runtime_directory or self.temporary.name)
        runtime.mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(['en'], gpu=device if device.startswith('cuda') else False,
            model_storage_directory=str(registry.target('ocr_recognizer').parent),
            user_network_directory=str(runtime),
            download_enabled=False, verbose=False)
        pin_ocr_device(self.reader, device)
        self.number, self.confidence, self.interval = number, confidence, interval
        self.next_sample = 0.
        self.samples, self.observations, self.detections = [], [], []
        self.calls = 0

    def consume(self, packet, people):
        if packet.time_sec + 1e-6 < self.next_sample:
            return
        scheduled = self.next_sample
        self.next_sample = (math.floor(max(0, packet.time_sec)/self.interval)+1)*self.interval
        self.samples.append({'requested_time_sec': scheduled, 'time_sec': packet.time_sec,
                             'frame': packet.index, 'person_count': len(people)})
        height, width = packet.pixels.shape[:2]
        for person_index, person in enumerate(people):
            if person.get('confidence', 0) < .4:
                continue
            roi = torso_box(person['xyxy'], width, height)
            x1,y1,x2,y2 = roi
            if x2-x1 < 12 or y2-y1 < 16:
                continue
            crop = packet.pixels[y1:y2, x1:x2]
            scale = min(4., max(1., 160/(y2-y1)))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            results = self.reader.readtext(crop, allowlist='0123456789', detail=1,
                paragraph=False, min_size=6, workers=0)
            self.calls += 1
            for text_box, text, score in results:
                score = float(score)
                number = digit_candidate(text, score)
                accepted = number == self.number and score >= self.confidence
                evidence = {'frame': packet.index, 'time_sec': packet.time_sec, 'number': number,
                    'text': text, 'confidence': score, 'bbox': person['xyxy'], 'torso_bbox': roi,
                    'ocr_polygon': [[float(x)/scale+x1, float(y)/scale+y1] for x,y in text_box],
                    'person_index': person_index, 'accepted': accepted,
                    'uncertainty': 'OCR on a torso crop is not a confirmed player identity'}
                self.observations.append(evidence)
                if accepted:
                    self.detections.append(evidence)

    def result(self):
        return {'status': 'complete', 'number': self.number, 'sample_interval_sec': self.interval,
                'sample_count': len(self.samples), 'samples': self.samples,
                'detections': self.detections, 'observations': self.observations,
                'ocr_calls': self.calls, 'implementation': 'volleymole-independent-jersey-v1',
                'note': 'No cross-frame identity claim. Rally weighting requires at least two separate high-confidence samples.'}
