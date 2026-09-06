"""Short-link correctness with explicit unverifiable links, independent frame/rally truth."""
import math

GAP_METRICS_VERSION = 1


def gap_report(tracks, truth_by_input_frame, rally_truth, match_tol_px=5.0):
    counts = {"correct": 0, "incorrect": 0, "unverifiable": 0}
    detail = []

    def truth_rally_ids(t):
        return {i for i, r in enumerate(rally_truth) if r["start_ms"] <= t <= r["end_ms"]}

    for track in tracks:
        points = {int(p["frame_idx"]): p for p in track["points"]}
        for gap in track.get("gap_links", []):
            begin, end = int(gap["from_frame_idx"]), int(gap["to_frame_idx"])
            reason, status = [], "correct"
            a, b = truth_by_input_frame.get(begin), truth_by_input_frame.get(end)
            if a and b:
                left, right = truth_rally_ids(float(a["source_pts_ms"])), truth_rally_ids(float(b["source_pts_ms"]))
                if left and right and left.isdisjoint(right):
                    status = "incorrect"
                    reason.append("crosses_independent_rally_boundaries")
            for idx in range(begin, end + 1):
                gt, point = truth_by_input_frame.get(idx), points.get(idx)
                if gt is None or point is None or str(gt["visible"]) not in {"0", "1"} or not int(gt["visible"]):
                    if status != "incorrect":
                        status = "unverifiable"
                    reason.append(f"unobservable_or_missing_frame:{idx}")
                elif math.hypot(float(point["x"]) - float(gt["x"]), float(point["y"]) - float(gt["y"])) > match_tol_px:
                    status = "incorrect"
                    reason.append(f"position_error:{idx}")
            counts[status] += 1
            detail.append({"track_id": track["track_id"], "from_frame_idx": begin, "to_frame_idx": end,
                           "status": status, "reasons": reason})
    total = sum(counts.values())
    verified = counts["correct"] + counts["incorrect"]
    return {"gap_metrics_version": GAP_METRICS_VERSION, "links": total, **counts,
            "strict_correct_rate": round(counts["correct"] / total, 6) if total else None,
            "verified_correct_rate": round(counts["correct"] / verified, 6) if verified else None,
            "verification_coverage": round(verified / total, 6) if total else None,
            "match_tol_px": match_tol_px, "detail": detail}
