"""Local action *proposals* from dense RGB evidence, never verified contacts.

The frozen ImageNet backbone has no volleyball supervision. A small temporal
head learns only from original VNL-STES training labels. Its sigmoid outputs
are uncalibrated scores, not a probability that a touch or a point is confirmed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

CLASSES = ('serve', 'receive', 'set', 'spike', 'block', 'score')
BACKBONE_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
BACKBONE_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
FEATURE_DIM = 512 * 7
FEATURE_VERSION = 'resnet18-imagenet-full224x398-pool1plus2x3-fp16rounded-v2'


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def soft_targets(events, frames, fps, sigma_sec=.08):
    """Deterministic training-only smoothing of unchanged author event frames.

    Independent columns preserve nearby or simultaneous spike/block labels.
    Evaluation always uses the original exact frame, without this smoothing.
    """
    if frames <= 0 or not math.isfinite(fps) or fps <= 0 or sigma_sec <= 0:
        raise ValueError('Invalid source clock or training target width')
    result = np.zeros((frames, len(CLASSES)), dtype=np.float32)
    sigma = sigma_sec * fps
    radius = int(math.ceil(3 * sigma))
    for event in events:
        frame, label = event['frame'], event['label']
        if type(frame) is not int or not 0 <= frame < frames or label not in CLASSES:
            raise ValueError('Original event lies outside source clock or ontology')
        lo, hi = max(0, frame-radius), min(frames, frame+radius+1)
        values = np.exp(-.5 * ((np.arange(lo, hi)-frame)/sigma)**2)
        result[lo:hi, CLASSES.index(label)] = np.maximum(result[lo:hi, CLASSES.index(label)], values)
    return result


def proposals(probabilities, times, threshold, nms_sec):
    scores, times = np.asarray(probabilities), np.asarray(times, dtype=float)
    if (scores.shape != (len(times), len(CLASSES)) or not len(times)
            or not np.isfinite(scores).all() or not np.isfinite(times).all()
            or np.any(scores < 0) or np.any(scores > 1) or np.any(np.diff(times) <= 0)
            or times[0] < 0 or not 0 < threshold < 1 or not math.isfinite(nms_sec) or nms_sec < 0):
        raise ValueError('Invalid action scores, thresholds, or source timestamps')
    selected = []
    for column, label in enumerate(CLASSES):
        values = scores[:, column]
        # Stable earlier timestamp tie-break; per-class NMS does not erase
        # a closely timed spike and block belonging to different classes.
        peaks = np.flatnonzero((values >= threshold)
            & (values >= np.r_[-np.inf, values[:-1]])
            & (values >= np.r_[values[1:], -np.inf]))
        keep = []
        for frame in sorted(peaks, key=lambda i: (-float(values[i]), int(i))):
            if all(abs(times[frame]-times[other]) > nms_sec + 1e-9 for other in keep):
                keep.append(frame)
        selected.extend({'label':label, 'time_sec':float(times[i]), 'frame_index':int(i),
                         'raw_probability':float(values[i])} for i in keep)
    return sorted(selected, key=lambda row: (row['time_sec'], row['label']))


class FrozenRGB(nn.Module):
    def __init__(self, weights, expected_sha256=None):
        super().__init__()
        self.sha256 = file_hash(weights)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError('Backbone hash differs from the trained model')
        if self.sha256 != BACKBONE_SHA256:
            raise ValueError('Expected the pinned official torchvision ResNet18 weights')
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.body = nn.Sequential(*list(model.children())[:-2]).eval()
        self.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, images):
        value = self.body((images.float()/255-self.mean)/self.std)
        return torch.cat([F.adaptive_avg_pool2d(value, 1).flatten(1),
                          F.adaptive_avg_pool2d(value, (2, 3)).flatten(1)], dim=1)

    @property
    def compute_policy(self):
        device = self.mean.device
        return {'device':str(device), 'autocast':'float16' if device.type == 'cuda' else 'disabled',
                'output_rounding':'float16 then float32', 'torch_version':str(torch.__version__)}

    @torch.inference_mode()
    def encode(self, frames_bgr, batch_size=64):
        device = self.mean.device
        result = []
        for start in range(0, len(frames_bgr), batch_size):
            rows = frames_bgr[start:start+batch_size]
            if any(row is None or row.ndim != 3 or row.shape[2] != 3 for row in rows):
                raise ValueError('Invalid decoded RGB evidence')
            array = np.stack([cv2.cvtColor(cv2.resize(row, (398, 224)), cv2.COLOR_BGR2RGB) for row in rows])
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(device)
            with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                result.append(self(tensor).float().cpu().numpy())
        # Match the on-disk feature representation exactly in live inference.
        return (np.concatenate(result).astype(np.float16).astype(np.float32) if result
                else np.empty((0, FEATURE_DIM), dtype=np.float32))


class TemporalHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden=96):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_scale', torch.ones(input_dim))
        self.project = nn.Sequential(nn.Linear(input_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.2))
        self.gru = nn.GRU(192, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=.15)
        self.classifier = nn.Linear(hidden*2, len(CLASSES))

    def forward(self, features, lengths=None):
        value = self.project((features-self.feature_mean)/self.feature_scale)
        if lengths is not None:
            value = nn.utils.rnn.pack_padded_sequence(value, lengths.cpu(), batch_first=True, enforce_sorted=False)
        value, _ = self.gru(value)
        if lengths is not None:
            value, _ = nn.utils.rnn.pad_packed_sequence(value, batch_first=True, total_length=features.shape[1])
        return self.classifier(value)


@torch.inference_mode()
def dense_probabilities(head, features, window=192, stride=96):
    if not 0 < stride <= window or len(features) == 0:
        raise ValueError('Invalid temporal evidence window')
    device = next(head.parameters()).device
    accumulated = np.zeros((len(features), len(CLASSES)), dtype=np.float64)
    counts = np.zeros(len(features), dtype=np.float64)
    head.eval()
    for start in range(0, len(features), stride):
        end = min(len(features), start+window)
        value = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)[None]
        logits = head(value).squeeze(0).cpu().numpy()
        accumulated[start:end] += logits
        counts[start:end] += 1
        if end == len(features):
            break
    mean = np.clip(accumulated/counts[:, None], -50, 50)
    return (1/(1+np.exp(-mean))).astype(np.float32)


class LocalActionSpotter:
    """Caller supplies consecutive frames at the trained rate and actual PTS.

    This adapter accepts already decoded images, so the application can share
    decoding with its detectors. No download or label loading occurs here.
    """
    def __init__(self, checkpoint, backbone, device='cpu'):
        checkpoint = Path(checkpoint)
        manifest = json.loads(checkpoint.with_suffix('.manifest.json').read_text())
        self.sha256 = file_hash(checkpoint)
        if self.sha256 != manifest['checkpoint_sha256']:
            raise ValueError('Action checkpoint hash changed')
        self.config = manifest['inference']
        if manifest['classes'] != list(CLASSES) or manifest['feature_version'] != FEATURE_VERSION:
            raise ValueError('Action model ontology or feature version mismatch')
        self.backbone = FrozenRGB(backbone, manifest['backbone_sha256']).to(device).eval()
        self.head = TemporalHead().to(device).eval()
        self.head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)

    def predict_frames(self, frames_bgr, source_times):
        times = np.asarray(source_times, dtype=float)
        if len(times) != len(frames_bgr) or not len(times):
            raise ValueError('Every action input frame needs its actual source PTS')
        expected = 1/self.config['fps']
        if len(times) > 1 and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or np.max(np.abs(np.diff(times)-expected)) > expected*.26):
            raise ValueError('Action frames must use the trained dense sampling rate')
        features = self.backbone.encode(frames_bgr)
        probabilities = dense_probabilities(self.head, features, self.config['window'], self.config['stride'])
        rows = proposals(probabilities, times, self.config['threshold'], self.config['nms_sec'])
        return [{**row, 'source_time_sec':row['time_sec'], 'model_sha256':self.sha256,
                 'feature_compute_policy':self.backbone.compute_policy,
                 'evidence_kind':'local_action_model', 'observation_status':'candidate',
                 'uncertainty':'Uncalibrated visual proposal; contact, success and score require contextual verification.'}
                for row in rows]
