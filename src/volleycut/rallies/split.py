"""候选回合生成：高召回优先（核心功能需求 §2/§3.4）。

规则基于时间（毫秒）与分辨率（画幅宽度）归一化；quality_flags 由确定性规则
生成，规则版本随 manifest 记录（核心功能需求 §4 质量标志）。
"""

from __future__ import annotations

import uuid
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..roi import signed_edge_distance, validate_polygon
from ..validation import dimension, number

QUALITY_RULE_VERSION = 3


@dataclass
class RallyConfig:
    min_rally_ms: float = 1000.0
    merge_gap_ms: float = 1000.0
    head_buffer_ms: float = 1000.0
    tail_buffer_ms: float = 1000.0
    evidence_window_ms: float = 1000.0
    evidence_bin_ms: float = 100.0
    min_evidence_coverage: float = 0.3
    min_motion_widths: float = 0.01
    long_gap_ratio: float = 0.8
    max_gap_ms: float = 300.0
    roi_edge_band: float = 0.02
    spare_ball_speed_widths_per_sec: float = 6.4

    def validate(self):
        for name in ("min_rally_ms", "merge_gap_ms", "head_buffer_ms", "tail_buffer_ms",
                     "evidence_window_ms", "evidence_bin_ms", "max_gap_ms", "min_motion_widths"):
            number(getattr(self, name), name, minimum=0)
        if self.evidence_window_ms <= 0 or self.evidence_bin_ms <= 0 or self.max_gap_ms <= 0:
            raise ValueError("evidence window/bin and max_gap_ms must be positive")
        for name in ("min_evidence_coverage", "long_gap_ratio", "roi_edge_band"):
            number(getattr(self, name), name, minimum=0, maximum=1)
        number(self.spare_ball_speed_widths_per_sec, "spare_ball_speed_widths_per_sec", positive=True)


def _flags_for(
    start_ms: float,
    end_ms: float,
    points: List[Dict],
    gaps: List[Dict],
    cfg: RallyConfig,
    frame_width: int,
    frame_height: int,
    roi_polygon=None,
    cleaning_decisions=(),
) -> List[str]:
    flags: List[str] = []
    ev_start = [
        p for p in points if p["state"] == "observed" and start_ms <= p["source_pts_ms"] <= start_ms + cfg.evidence_window_ms
    ]
    ev_end = [
        p for p in points if p["state"] == "observed" and end_ms - cfg.evidence_window_ms <= p["source_pts_ms"] <= end_ms
    ]
    def coverage(pts, origin):
        bins = {int((p["source_pts_ms"] - origin) // cfg.evidence_bin_ms) for p in pts}
        total = math.ceil(cfg.evidence_window_ms / cfg.evidence_bin_ms)
        return len({b for b in bins if 0 <= b < total}) / total

    if coverage(ev_start, start_ms) < cfg.min_evidence_coverage:
        flags.append("weak_start_evidence")
    if coverage(ev_end, end_ms - cfg.evidence_window_ms) < cfg.min_evidence_coverage:
        flags.append("uncertain_end")

    longest_gap = max((g["gap_ms"] for g in gaps), default=0.0)
    if longest_gap >= cfg.long_gap_ratio * cfg.max_gap_ms:
        flags.append("long_internal_gap")

    if frame_width > 0 and frame_height > 0:
        polygon = roi_polygon if roi_polygon is not None else (
            (0, 0), (frame_width - 1, 0), (frame_width - 1, frame_height - 1), (0, frame_height - 1))
        edge = 0
        observed = [p for p in points if p["state"] == "observed"]
        for p in observed:
            distance = signed_edge_distance(polygon, (float(p["x"]), float(p["y"])))
            if distance <= cfg.roi_edge_band * frame_width:
                edge += 1
        if observed and edge / len(observed) > 0.5:
            flags.append("possible_spare_ball")
        max_speed = 0.0
        for a, b in zip(observed, observed[1:]):
            dt = (b["source_pts_ms"] - a["source_pts_ms"]) / 1000.0
            if dt <= 0:
                continue
            dist = ((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
            max_speed = max(max_speed, dist / dt / frame_width)
        if max_speed > cfg.spare_ball_speed_widths_per_sec:
            if "possible_spare_ball" not in flags:
                flags.append("possible_spare_ball")
    abnormal = {"speed_exceeds_limit", "isolated_acceleration_direction_spike", "gap_direction_acceleration_conflict",
                "connection_crosses_roi"}
    if any(d.get("reason") in abnormal and start_ms <= d.get("source_pts_ms", -1) <= end_ms
           for d in cleaning_decisions) and "possible_spare_ball" not in flags:
        flags.append("possible_spare_ball")
    return flags


def generate_rallies(
    tracks: List[Dict],
    video_duration_ms: float,
    analysis_id: str,
    video_fingerprint: str,
    frame_width: int,
    frame_height: int,
    config: Optional[RallyConfig] = None,
    *,
    video_start_ms: float = 0,
    roi_polygon=None,
    cleaning_decisions=(),
) -> Dict:
    cfg = RallyConfig() if config is None else config
    if not isinstance(cfg, RallyConfig):
        raise ValueError("config must be RallyConfig")
    cfg.validate()
    frame_width = dimension(frame_width, "frame_width")
    frame_height = dimension(frame_height, "frame_height")
    number(video_start_ms, "video_start_ms")
    number(video_duration_ms, "video_duration_ms")
    if video_duration_ms <= video_start_ms:
        raise ValueError("video end PTS must follow start PTS")
    roi_polygon = validate_polygon(roi_polygon, frame_width, frame_height)

    segments: List[Dict] = []
    for tr in tracks:
        pts = tr["points"]
        if not pts:
            continue
        segments.append(
            {
                "track_id": tr["track_id"],
                "start_ms": pts[0]["source_pts_ms"],
                "end_ms": pts[-1]["source_pts_ms"],
                "points": pts,
                "gap_links": tr.get("gap_links", []),
            }
        )
    segments.sort(key=lambda s: s["start_ms"])

    merged: List[Dict] = []
    for seg in segments:
        if merged and seg["start_ms"] - merged[-1]["end_ms"] <= cfg.merge_gap_ms:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], seg["end_ms"])
            merged[-1]["tracks"].append(seg)
        else:
            merged.append(
                {
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "tracks": [seg],
                }
            )

    rallies: List[Dict] = []
    sequence = 0
    for group in merged:
        if group["end_ms"] - group["start_ms"] < cfg.min_rally_ms:
            continue
        observed = [p for t in group["tracks"] for p in t["points"] if p["state"] == "observed"]
        if not observed:
            continue
        motion = math.hypot(max(p["x"] for p in observed) - min(p["x"] for p in observed),
                            max(p["y"] for p in observed) - min(p["y"] for p in observed)) / frame_width
        if motion < cfg.min_motion_widths:
            continue
        sequence += 1
        start_ms = max(video_start_ms, group["start_ms"] - cfg.head_buffer_ms)
        end_ms = min(video_duration_ms, group["end_ms"] + cfg.tail_buffer_ms)
        all_points = [p for t in group["tracks"] for p in t["points"]]
        all_points.sort(key=lambda p: p["source_pts_ms"])
        all_gaps = [g for t in group["tracks"] for g in t["gap_links"]]
        for a, b in zip(group["tracks"], group["tracks"][1:]):
            if b["start_ms"] > a["end_ms"]:
                all_gaps.append({"gap_ms": b["start_ms"] - a["end_ms"]})
        flags = _flags_for(
            group["start_ms"], group["end_ms"], all_points, all_gaps, cfg, frame_width, frame_height,
            roi_polygon, cleaning_decisions,
        )
        rallies.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL,
                                     f"{video_fingerprint}:{sequence}:{start_ms}:{end_ms}")),
                "sequence": sequence,
                "start_ms": int(round(start_ms)),
                "end_ms": int(round(end_ms)),
                "source": "auto",
                "status": "kept",
                "quality_flags": flags,
                "evidence": [
                    {
                        "track_id": t["track_id"],
                        "start_ms": round(t["start_ms"], 3),
                        "end_ms": round(t["end_ms"], 3),
                    }
                    for t in group["tracks"]
                ],
                "edited": False,
            }
        )

    return {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "video_fingerprint": video_fingerprint,
        "rallies": rallies,
    }
