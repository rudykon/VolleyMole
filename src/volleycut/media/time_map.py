"""Actual FFmpeg fps selections, never guessed nearest packet PTS.

FFmpeg 7 logs input microseconds and the selected frame's rounded output PTS.
Validate each Writing event against the frames already read by that filter.
"""
import csv
import re
from fractions import Fraction

from ..errors import EngineError, ErrorCode
from .probe import stream_pts_ms

TIME_MAP_VERSION = 2
FIELDS = ["frame_idx", "proxy_pts_ms", "source_frame_idx", "source_pts_ms"]
READ = re.compile(r"Read frame with in pts (-?\d+), out pts (-?\d+)")
WRITE = re.compile(r"Writing frame with pts (-?\d+) to pts (-?\d+)")


def mapping_from_fps_log(log, fps, source_offset_ms=0):
    frames, rows = [], []
    rate = Fraction(str(fps))
    count = 0
    for line in log.splitlines():
        if "fps@map" not in line:
            continue
        read = READ.search(line)
        if read:
            raw, rounded = map(int, read.groups())
            frames.append((count, raw, rounded))
            count += 1
        write = WRITE.search(line)
        if write:
            selected, output = map(int, write.groups())
            eligible = [f for f in frames if f[2] <= output]
            if not eligible or eligible[-1][2] != selected:
                raise EngineError(ErrorCode.TIME_MAP_INVALID, "Cannot verify FFmpeg frame selection")
            idx, raw, _ = eligible[-1]
            rows.append({"frame_idx": len(rows),
                         "proxy_pts_ms": round(float(output / rate * 1000), 3),
                         "source_frame_idx": idx,
                         "source_pts_ms": round(raw / 1000 + source_offset_ms, 3)})
            frames = [f for f in frames if f[0] >= idx]
    if not rows:
        raise EngineError(ErrorCode.TIME_MAP_INVALID, "Missing FFmpeg fps selection trace")
    return rows


def write_time_map(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_time_map(source_video, proxy_video, out_csv, *, fps_log=None, fps=None,
                   source_offset_ms=0, cancel_check=None):
    if fps_log is None or fps is None:
        raise EngineError(ErrorCode.TIME_MAP_INVALID,
                          "Mapping requires the actual proxy generation fps trace")
    rows = mapping_from_fps_log(fps_log, fps, source_offset_ms)
    proxy_pts = stream_pts_ms(proxy_video, cancel_check=cancel_check)
    if len(proxy_pts) != len(rows) or any(abs(p - r["proxy_pts_ms"]) > 0.1
                                        for p, r in zip(proxy_pts, rows)):
        raise EngineError(ErrorCode.TIME_MAP_INVALID, "Encoded proxy PTS differs from fps trace",
                          {"proxy_frames": len(proxy_pts), "trace_frames": len(rows)})
    write_time_map(rows, out_csv)
    return rows


def load_time_map(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as stream:
        return [float(row["source_pts_ms"]) for row in csv.DictReader(stream)]
