"""预测回合与真值回合的一对一匹配（核心功能需求 §5，版本化）。

规则：按时间 IoU 配对；IoU ≥ 0.5 且唯一最大者记为命中；一个真值最多匹配一个
预测。粘连：一个预测与 ≥2 个真值 IoU ≥ 0.5。拆碎：≥2 个预测与同一真值
IoU ≥ 0.5，仅 IoU 最大者计命中，其余计误切。粘连与拆碎分别统计。
"""

from __future__ import annotations

from typing import Dict, List

MATCHING_VERSION = 2
DEFAULT_IOU_THRESHOLD = 0.5


def _iou_ms(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0.0, inter_end - inter_start)
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def match_rallies(
    pred_rallies: List[Dict],
    gt_rallies: List[Dict],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict:
    preds = [
        {"id": p.get("id", f"pred_{i}"), "start_ms": float(p["start_ms"]), "end_ms": float(p["end_ms"])}
        for i, p in enumerate(pred_rallies)
    ]
    gts = [
        {"id": g.get("id", f"gt_{i}"), "start_ms": float(g["start_ms"]), "end_ms": float(g["end_ms"])}
        for i, g in enumerate(gt_rallies)
    ]

    iou: Dict[int, Dict[int, float]] = {}
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            v = _iou_ms(p["start_ms"], p["end_ms"], g["start_ms"], g["end_ms"])
            if v >= iou_threshold:
                iou.setdefault(pi, {})[gi] = v

    sticking_preds: List[int] = []
    stuck_gts = set()
    for pi, row in iou.items():
        if len(row) >= 2:
            sticking_preds.append(pi)
            stuck_gts.update(row.keys())

    remaining: Dict[int, Dict[int, float]] = {
        pi: row for pi, row in iou.items() if pi not in sticking_preds
    }

    claims: Dict[int, List[int]] = {}
    for pi, row in remaining.items():
        for gi in row:
            claims.setdefault(gi, []).append(pi)

    hits: List[Dict] = []
    split_extras: List[int] = []
    fragmented_gts: List[int] = []
    matched_gt = set()

    for gi, pis in claims.items():
        if len(pis) == 1:
            pi = pis[0]
            hits.append({"pred": pi, "gt": gi, "iou": remaining[pi][gi]})
            matched_gt.add(gi)
        else:
            fragmented_gts.append(gi)
            best_val = max(remaining[pi][gi] for pi in pis)
            winners = [pi for pi in pis if remaining[pi][gi] == best_val]
            if len(winners) == 1:
                hits.append({"pred": winners[0], "gt": gi, "iou": best_val})
                matched_gt.add(gi)
                split_extras.extend(pi for pi in pis if pi != winners[0])
            else:
                split_extras.extend(pis)

    hit_preds = {h["pred"] for h in hits}
    false_cuts = [
        pi
        for pi in range(len(preds))
        if pi not in hit_preds and pi not in sticking_preds and pi not in split_extras
    ]
    missed_gts = [gi for gi in range(len(gts)) if gi not in matched_gt]

    return {
        "matching_version": MATCHING_VERSION,
        "iou_threshold": iou_threshold,
        "n_pred": len(preds),
        "n_gt": len(gts),
        "hits": hits,
        "sticking_preds": sticking_preds,
        "stuck_gts": sorted(stuck_gts),
        "fragmented_gts": fragmented_gts,
        "split_extras": split_extras,
        "false_cuts": false_cuts,
        "missed_gts": missed_gts,
        "preds": preds,
        "gts": gts,
    }
