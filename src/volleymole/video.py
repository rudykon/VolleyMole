"""One timestamp contract for every inference module (display order, not frame/fps)."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from time import perf_counter
import numpy as np

from .common import probe


@dataclass(frozen=True)
class FramePacket:
    index: int
    pts: int
    time_base: Fraction
    container_start: float
    pixels: np.ndarray

    @property
    def source_sec(self):
        return float(self.pts * self.time_base)

    @property
    def time_sec(self):
        return self.source_sec - self.container_start

    def clock(self):
        return {'frame': self.index, 'pts': self.pts,
                'time_base': [self.time_base.numerator, self.time_base.denominator],
                'source_time_s': self.source_sec, 'time_s': self.time_sec}


def decode(video, *, max_frames=None, timings=None):
    import av
    import cv2
    meta = probe(video)
    previous = None
    count = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        frames = container.decode(stream)
        if max_frames is not None:
            frames = islice(frames, max_frames)
        frames = iter(frames)
        while True:
            started = perf_counter()
            try:
                frame = next(frames)
            except StopIteration:
                break
            if timings is not None:
                timings.add('decode', perf_counter() - started)
            started = perf_counter()
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f'Frame {count} has no presentation timestamp')
            when = frame.pts * frame.time_base
            if previous is not None and when <= previous:
                raise ValueError(f'Non-monotonic presentation timestamp at frame {count}')
            previous = when
            pixels = frame.to_ndarray(format='bgr24')
            if meta['rotation']:
                pixels = cv2.rotate(pixels, {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}[meta['rotation']])
            if pixels.shape[:2] != (meta['height'], meta['width']):
                raise ValueError('Decoded dimensions disagree with display rotation')
            if timings is not None:
                timings.add('decode_to_bgr', perf_counter() - started)
            yield FramePacket(count, frame.pts, Fraction(frame.time_base), meta['start_sec'], pixels)
            count += 1
    if not count:
        raise ValueError('No decoded frames')
    if max_frames is None and meta['frame_count'] and count != meta['frame_count']:
        raise ValueError(f'Incomplete decode: {count}/{meta["frame_count"]}')


def chunks(iterator, size):
    if size < 1:
        raise ValueError('Chunk size must be positive')
    iterator = iter(iterator)
    while batch := list(islice(iterator, size)):
        yield batch
