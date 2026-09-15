"""Inference-only PANNs Cnn14_DecisionLevelMax, with scriptable dynamic length.

Architecture adapted from Qiuqiang Kong's MIT-licensed audioset_tagging_cnn:
https://github.com/qiuqiangkong/audioset_tagging_cnn
See licenses/PANNs-MIT.txt. All parameters, including the frozen STFT kernels
and mel matrix, must be loaded strictly from the published checkpoint. This
module does not invent labels, train a classifier, or require torchlibrosa.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _STFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real = nn.Conv1d(1, 513, 1024, stride=320, bias=False)
        self.conv_imag = nn.Conv1d(1, 513, 1024, stride=320, bias=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = F.pad(waveform[:, None, :], (512, 512), mode='reflect')
        real, imag = self.conv_real(x), self.conv_imag(x)
        return (real.square() + imag.square())[:, None, :, :].transpose(2, 3)


class _Spectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = _STFT()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.stft(waveform)


class _Logmel(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.empty(513, 64), requires_grad=False)

    def forward(self, power: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10(torch.clamp(torch.matmul(power, self.melW), min=1e-10))


class _ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool_size)


class PannsCnn14Sed(nn.Module):
    """Published framewise SED network, not the Cnn14 clipwise classifier.

    Ten-millisecond frames are nearest-neighbor interpolation of 320 ms CNN
    steps, not independent touch-level evidence. Inputs shorter than one CNN
    step are zero-padded internally and the output is cropped to source time.
    """
    __constants__ = ['frame_hop_samples', 'sample_rate', 'decision_hop_samples']

    def __init__(self):
        super().__init__()
        self.frame_hop_samples = 320
        self.decision_hop_samples = 10240
        self.sample_rate = 32000
        self.spectrogram_extractor = _Spectrogram()
        self.logmel_extractor = _Logmel()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        if waveform.dim() != 2 or waveform.shape[1] == 0:
            raise ValueError('Expected nonempty waveform [batch, samples] at 32000 Hz')
        source_frames = waveform.shape[1] // self.frame_hop_samples + 1
        if waveform.shape[1] < self.decision_hop_samples:
            waveform = F.pad(waveform, (0, self.decision_hop_samples - waveform.shape[1]))
        x = self.logmel_extractor(self.spectrogram_extractor(waveform))
        frames = x.shape[2]
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.conv_block1(x, (2, 2))
        x = self.conv_block2(x, (2, 2))
        x = self.conv_block3(x, (2, 2))
        x = self.conv_block4(x, (2, 2))
        x = self.conv_block5(x, (2, 2))
        x = self.conv_block6(x, (1, 1))
        x = torch.mean(x, dim=3)
        x = F.max_pool1d(x, 3, stride=1, padding=1) + F.avg_pool1d(x, 3, stride=1, padding=1)
        x = F.relu_(self.fc1(x.transpose(1, 2)))
        segmentwise = torch.sigmoid(self.fc_audioset(x))
        framewise = torch.repeat_interleave(segmentwise, 32, dim=1)
        tail = framewise[:, -1:, :].repeat(1, frames - framewise.shape[1], 1)
        framewise = torch.cat((framewise, tail), dim=1)[:, :source_frames, :]
        return {'framewise_output': framewise,
                'clipwise_output': torch.max(framewise, dim=1)[0]}
