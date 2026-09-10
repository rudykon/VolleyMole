"""Camera primitives adapted from fast-volleyball-tracking-inference (MIT).

Copyright (c) 2025 Alexander Sigatchov. Full notice: licenses/tracking-MIT.txt.
Only the moving-average and clamped crop used by VolleyMole are retained.
"""
import numpy as np


def smooth_values(values, method='moving_avg', window=15, polyorder=2):
    if method == 'none' or len(values) < 2:
        return values
    if method != 'moving_avg':
        raise ValueError(f'Unsupported camera smoothing: {method}')
    window = max(3, min(window, len(values)))
    if window % 2 == 0:
        window += 1
    padded = np.pad(values, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, np.ones(window) / window, mode='valid')


def crop_frame(frame, center_x, crop_width, padding='none'):
    if padding != 'none' or crop_width < 1:
        raise ValueError('Only a positive, clamped camera crop is supported')
    width = frame.shape[1]
    crop_width = min(crop_width, width)
    left = max(0, min(center_x - crop_width // 2, width - crop_width))
    return frame[:, left:left + crop_width]
