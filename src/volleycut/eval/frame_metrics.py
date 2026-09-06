"""帧级指标：可见帧检出率、误检率、跳点率（核心功能需求 §5/§6）。

定义（版本化，随验收配置）：
- 检出命中：真值可见帧上，预测可见且中心距离 ≤ match_tol_px（默认 5px）。
- 误检：预测可见，且该帧真值不可见或距离 > match_tol_px。
- 跳点：真值可见且预测可见，但距离 > jump_tol_px（默认 max(20px, 2% 宽度)）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

FRAME_METRICS_VERSION = 2


def frame_report(
    pred_rows: List[Dict],
    gt_rows: List[Dict],
    frame_width: int,
    match_tol_px: float = 5.0,
    jump_tol_px: Optional[float] = None,
) -> Dict:
    if jump_tol_px is None:
        jump_tol_px = max(20.0, 0.02 * frame_width)

    gt_by_frame: Dict[int, Dict] = {}
    for g in gt_rows:
        if int(g["frame_idx"]) in gt_by_frame:
            raise ValueError("duplicate frame ground truth")
        gt_by_frame[int(g["frame_idx"])] = {
            "visible": bool(int(g["visible"])),
            "x": float(g["x"]) if int(g["visible"]) else None,
            "y": float(g["y"]) if int(g["visible"]) else None,
        }

    tp = fp = jump = 0
    gt_visible_total = sum(g["visible"] for g in gt_by_frame.values())
    gt_visible_detected = 0
    pred_visible_total = 0
    seen = set()

    for p in pred_rows:
        idx = int(p["frame_idx"])
        if idx in seen:
            raise ValueError("duplicate prediction frame")
        seen.add(idx)
        px = int(p["x"])
        py = int(p["y"])
        pred_visible = px >= 0 and py >= 0
        g = gt_by_frame.get(idx)

        if g is None:
            continue
        if not pred_visible:
            continue

        pred_visible_total += 1
        if g is None or not g["visible"]:
            fp += 1
            continue
        dist = ((px - g["x"]) ** 2 + (py - g["y"]) ** 2) ** 0.5
        if dist <= match_tol_px:
            tp += 1
            gt_visible_detected += 1
        else:
            fp += 1
            if dist > jump_tol_px:
                jump += 1

    def ratio(num: int, den: int) -> float:
        return round(num / den, 6) if den else 0.0

    return {
        "frame_metrics_version": FRAME_METRICS_VERSION,
        "match_tol_px": match_tol_px,
        "jump_tol_px": jump_tol_px,
        "frames_evaluated": len(gt_by_frame),
        "missing_prediction_frames": len(set(gt_by_frame) - seen),
        "gt_visible_frames": gt_visible_total,
        "pred_visible_frames": pred_visible_total,
        "true_positives": tp,
        "false_detections": fp,
        "jump_points": jump,
        "visible_recall": ratio(gt_visible_detected, gt_visible_total),
        "false_detection_rate": ratio(fp, pred_visible_total),
        "jump_point_rate": ratio(jump, pred_visible_total),
    }
