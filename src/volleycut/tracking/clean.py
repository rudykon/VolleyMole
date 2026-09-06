"""Time/width-normalized, single-ball trajectory cleaning; no long-gap prediction.

Rule v3 also checks entire connections against the actual ROI polygon. Every interpolated timestamp
comes from the input detections table. Rejected candidates remain in detections.csv;
the optional decisions output explains what cleaning removed or split.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

from ..roi import contains, contains_segment, validate_polygon
from ..validation import boolean, dimension, number

RULE_VERSION = 3


@dataclass
class TrackConfig:
    max_speed_widths_per_sec: float = 3.2
    max_acceleration_widths_per_sec2: float = 80.0
    sharp_turn_degrees: float = 135.0
    spike_min_excursion_widths: float = 0.03
    max_gap_ms: float = 300.0
    min_observed_ms: float = 60.0
    roi_only: bool = True

    def validate(self):
        for name in ("max_speed_widths_per_sec", "max_acceleration_widths_per_sec2",
                     "spike_min_excursion_widths", "max_gap_ms", "min_observed_ms"):
            number(getattr(self, name), name, positive=True)
        number(self.sharp_turn_degrees, "sharp_turn_degrees", positive=True)
        if self.sharp_turn_degrees >= 180:
            raise ValueError("sharp_turn_degrees must be in (0, 180)")
        boolean(self.roi_only, "roi_only")


def _point(row):
    return {"frame_idx": int(row["frame_idx"]), "source_pts_ms": float(row["source_pts_ms"]),
            "x": float(row["x"]), "y": float(row["y"]), "state": "observed"}


def _distance(a, b):
    return math.hypot(b["x"] - a["x"], b["y"] - a["y"])


def _velocity(a, b, width):
    dt = (b["source_pts_ms"] - a["source_pts_ms"]) / 1000
    if dt <= 0:
        return None
    return ((b["x"] - a["x"]) / width / dt, (b["y"] - a["y"]) / width / dt)


def _turn(a, b, c, width):
    before, after = _velocity(a, b, width), _velocity(b, c, width)
    if before is None or after is None:
        return 0.0, 0.0
    norms = math.hypot(*before) * math.hypot(*after)
    cosine = sum(x * y for x, y in zip(before, after)) / norms if norms else 1.0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    dt = (c["source_pts_ms"] - a["source_pts_ms"]) / 2000
    acceleration = math.hypot(after[0] - before[0], after[1] - before[1]) / dt
    return angle, acceleration


def build_tracks(detections, fps, frame_width, config=None, *, decisions=None, roi_polygon=None, frame_height=None,
                 cancel_check: Callable | None = None):
    cfg = TrackConfig() if config is None else config
    if not isinstance(cfg, TrackConfig):
        raise ValueError("config must be TrackConfig")
    cfg.validate()
    number(fps, "fps", positive=True)
    frame_width = dimension(frame_width, "frame_width")
    polygon = validate_polygon(roi_polygon, frame_width, frame_height)
    decisions = decisions if decisions is not None else []
    by_frame, visible = {}, []
    last_idx, last_pts = -1, -math.inf
    for n, row in enumerate(detections):
        if cancel_check and n % 128 == 0:
            cancel_check()
        point = _point(row)
        idx, pts = point["frame_idx"], point["source_pts_ms"]
        if idx <= last_idx or pts < last_pts or not all(math.isfinite(point[k]) for k in ("source_pts_ms", "x", "y")):
            raise ValueError("detections must have unique ordered frame indices and finite nondecreasing PTS")
        by_frame[idx] = point
        last_idx, last_pts = idx, pts
        if point["x"] < 0 or point["y"] < 0:
            continue
        within = row.get("within_roi", True)
        if within not in (True, False, 0, 1, "0", "1", "True", "False", "true", "false"):
            raise ValueError("within_roi must be an explicit boolean/0/1")
        within = (contains(polygon, (point["x"], point["y"])) if polygon is not None else
                  within not in (False, 0, "0", "False", "false"))
        if cfg.roi_only and not within:
            decisions.append({"frame_idx": idx, "source_pts_ms": pts, "action": "reject",
                              "reason": "outside_roi"})
            continue
        visible.append(point)

    # Only remove an isolated excursion when both adjacent legs and the direct
    # connection supply evidence. A normal change of direction alone is retained.
    rejected = set()
    for i in range(1, len(visible) - 1):
        if cancel_check and i % 128 == 0:
            cancel_check()
        a, b, c = visible[i-1:i+2]
        if b["frame_idx"] != a["frame_idx"] + 1 or c["frame_idx"] != b["frame_idx"] + 1:
            continue
        dt = (c["source_pts_ms"] - a["source_pts_ms"]) / 1000
        if dt <= 0 or dt * 1000 > cfg.max_gap_ms:
            continue
        angle, acceleration = _turn(a, b, c, frame_width)
        excursion = min(_distance(a, b), _distance(b, c)) / frame_width
        direct_speed = _distance(a, c) / frame_width / dt
        if (angle >= cfg.sharp_turn_degrees and acceleration > cfg.max_acceleration_widths_per_sec2
                and excursion >= cfg.spike_min_excursion_widths
                and direct_speed <= cfg.max_speed_widths_per_sec):
            rejected.add(b["frame_idx"])
            decisions.append({"frame_idx": b["frame_idx"], "source_pts_ms": b["source_pts_ms"],
                              "action": "reject", "reason": "isolated_acceleration_direction_spike",
                              "turn_degrees": round(angle, 3), "acceleration_widths_per_sec2": round(acceleration, 3),
                              "evidence_frames": [a["frame_idx"], b["frame_idx"], c["frame_idx"]]})

    tracks, current = [], None

    def close():
        nonlocal current
        if not current:
            return
        observed = [p for p in current["points"] if p["state"] == "observed"]
        span = observed[-1]["source_pts_ms"] - observed[0]["source_pts_ms"]
        if span >= cfg.min_observed_ms:
            current["track_id"] = len(tracks) + 1
            tracks.append(current)
        else:
            decisions.append({"action": "discard_track", "reason": "insufficient_observed_duration",
                              "frames": [p["frame_idx"] for p in observed], "observed_span_ms": round(span, 3)})
        current = None

    for n, b in enumerate(visible):
        if cancel_check and n % 128 == 0:
            cancel_check()
        if b["frame_idx"] in rejected:
            continue
        if current is None:
            current = {"points": [dict(b)], "gap_links": []}
            continue
        a = current["points"][-1]
        dt_ms = b["source_pts_ms"] - a["source_pts_ms"]
        missing = b["frame_idx"] - a["frame_idx"] - 1
        velocity = _velocity(a, b, frame_width)
        speed = math.hypot(*velocity) if velocity else (0.0 if _distance(a, b) == 0 else math.inf)
        reason = None
        if dt_ms > cfg.max_gap_ms:
            reason = "gap_exceeds_limit"
        elif speed > cfg.max_speed_widths_per_sec:
            reason = "speed_exceeds_limit" if dt_ms > 0 else "duplicate_pts_disagreement"
        elif missing and any(k not in by_frame for k in range(a["frame_idx"] + 1, b["frame_idx"])):
            reason = "missing_real_frame_pts"
        elif cfg.roi_only and polygon is not None and not contains_segment(
                polygon, (a["x"], a["y"]), (b["x"], b["y"])):
            reason = "connection_crosses_roi"
        elif missing and len(current["points"]) >= 2:
            observed = [p for p in current["points"] if p["state"] == "observed"]
            if len(observed) >= 2:
                angle, acceleration = _turn(observed[-2], a, b, frame_width)
                if angle >= cfg.sharp_turn_degrees and acceleration > cfg.max_acceleration_widths_per_sec2:
                    reason = "gap_direction_acceleration_conflict"
        if reason:
            decisions.append({"action": "split", "reason": reason, "from_frame_idx": a["frame_idx"],
                              "source_pts_ms": b["source_pts_ms"],
                              "to_frame_idx": b["frame_idx"], "gap_ms": round(dt_ms, 3),
                              "speed_widths_per_sec": round(speed, 6) if math.isfinite(speed) else None})
            close()
            current = {"points": [dict(b)], "gap_links": []}
            continue
        if missing:
            for k in range(a["frame_idx"] + 1, b["frame_idx"]):
                if cancel_check and k % 128 == 0:
                    cancel_check()
                pts = by_frame[k]["source_pts_ms"]
                alpha = (pts - a["source_pts_ms"]) / dt_ms if dt_ms else 0
                current["points"].append({"frame_idx": k, "source_pts_ms": pts,
                    "x": round(a["x"] + alpha * (b["x"] - a["x"]), 12),
                    "y": round(a["y"] + alpha * (b["y"] - a["y"]), 12), "state": "interpolated"})
            current["gap_links"].append({"from_frame_idx": a["frame_idx"], "to_frame_idx": b["frame_idx"],
                                        "gap_ms": round(dt_ms, 3), "interpolated_frames": missing,
                                        "method": "linear_position_using_actual_source_pts"})
        current["points"].append(dict(b))
    close()
    return tracks


def write_tracks_jsonl(tracks, path):
    with open(path, "w", encoding="utf-8") as stream:
        for track in tracks:
            stream.write(json.dumps(track, ensure_ascii=False, allow_nan=False) + "\n")
