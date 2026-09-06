"""回合级指标（验收维度：精确率、召回率、边界误差、粘连率、拆碎率、误切率）。"""

from __future__ import annotations

from typing import Dict, List

from .matching import match_rallies


def rally_report(
    pred_rallies: List[Dict],
    gt_rallies: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict:
    m = match_rallies(pred_rallies, gt_rallies, iou_threshold=iou_threshold)
    n_pred = m["n_pred"]
    n_gt = m["n_gt"]
    hits = m["hits"]

    boundary_errors = []
    for h in hits:
        p = m["preds"][h["pred"]]
        g = m["gts"][h["gt"]]
        boundary_errors.append(abs(p["start_ms"] - g["start_ms"]))
        boundary_errors.append(abs(p["end_ms"] - g["end_ms"]))

    def ratio(num: int, den: int) -> float:
        return round(num / den, 6) if den else 0.0

    return {
        "matching": {
            "matching_version": m["matching_version"],
            "iou_threshold": m["iou_threshold"],
        },
        "n_pred": n_pred,
        "n_gt": n_gt,
        "precision": ratio(len(hits), n_pred),
        "recall": ratio(len(hits), n_gt),
        "hits": len(hits),
        "boundary_error_ms": {
            "mean": round(sum(boundary_errors) / len(boundary_errors), 3) if boundary_errors else None,
            "max": round(max(boundary_errors), 3) if boundary_errors else None,
        },
        "sticking": {
            "preds": len(m["sticking_preds"]),
            "gts_involved": len(m["stuck_gts"]),
            "rate_over_preds": ratio(len(m["sticking_preds"]), n_pred),
            "rate_over_gts": ratio(len(m["stuck_gts"]), n_gt),
        },
        "splitting": {
            "gts": len(m["fragmented_gts"]),
            "extra_preds": len(m["split_extras"]),
            "rate_over_gts": ratio(len(m["fragmented_gts"]), n_gt),
            "rate_over_preds": ratio(len(m["split_extras"]), n_pred),
        },
        "false_cuts": {
            "count": len(m["false_cuts"]),
            "rate": ratio(len(m["false_cuts"]), n_pred),
        },
        "missed_gts": {
            "count": len(m["missed_gts"]),
            "rate": ratio(len(m["missed_gts"]), n_gt),
        },
        "detail": {
            "hits": hits,
            "sticking_preds": m["sticking_preds"],
            "split_extras": m["split_extras"],
            "false_cuts": m["false_cuts"],
            "missed_gts": m["missed_gts"],
        },
    }
