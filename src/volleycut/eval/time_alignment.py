"""Compare different CFR inputs against exact source PTS, never a nearby video frame."""
import math

from .frame_metrics import frame_report

ALIGNMENT_VERSION = 1


def pts_key(value):
    pts = float(value)
    if not math.isfinite(pts):
        raise ValueError("nonfinite source PTS")
    return round(pts * 1000)  # decoded-source microseconds, as recorded in time_map.csv


def align_frames(predictions, truth, time_map):
    gt_by_pts = {}
    for row in truth:
        key = pts_key(row["source_pts_ms"])
        if key in gt_by_pts:
            raise ValueError("duplicate source PTS in native frame truth")
        if str(row["visible"]) not in {"0", "1"}:
            raise ValueError("unreviewed frame labels cannot be evaluated")
        gt_by_pts[key] = row
    predicted = {}
    for row in predictions:
        idx = int(row["frame_idx"])
        if idx in predicted:
            raise ValueError("duplicate prediction frame")
        predicted[idx] = row
    mapped, gt_for_input, gt_by_input = {}, [], {}
    first_input_for_source = {}
    outside_truth = []
    last_idx = -1
    for row in time_map:
        idx, key = int(row["frame_idx"]), pts_key(row["source_pts_ms"])
        if idx <= last_idx:
            raise ValueError("time map must have unique increasing frame indices")
        last_idx = idx
        mapped[idx] = key
        if idx in predicted and pts_key(predicted[idx]["source_pts_ms"]) != key:
            raise ValueError(f"prediction/source time-map mismatch at frame {idx}")
        gt = gt_by_pts.get(key)
        if gt is None:
            outside_truth.append(idx)
            continue
        first_input_for_source.setdefault(key, idx)
        gt_for_input.append({**gt, "frame_idx": idx})
        gt_by_input[idx] = gt
    if set(predicted) - set(mapped):
        raise ValueError("prediction frame absent from generation time map")
    if outside_truth:
        raise ValueError(f"{len(outside_truth)} input frames have no exact source-PTS truth; do not use nearest-frame labels")
    # Select the first mapped occurrence, not the best detection or the first successful
    # duplicate. Unsampled native frames remain in the source recall denominator.
    source_predictions = []
    for key, idx in first_input_for_source.items():
        if idx in predicted:
            source_predictions.append({**predicted[idx], "frame_idx": int(gt_by_pts[key]["frame_idx"])})
    return {"source_predictions": source_predictions, "input_truth": gt_for_input,
            "truth_by_input_frame": gt_by_input,
            "metadata": {"alignment_version": ALIGNMENT_VERSION, "key": "source_pts_us_exact",
                         "duplicate_policy": "first_mapped_occurrence_not_best_prediction",
                         "native_truth_frames": len(truth), "mapped_input_frames": len(mapped),
                         "source_frames_selected": len(first_input_for_source),
                         "source_frames_not_selected": len(truth) - len(first_input_for_source),
                         "duplicated_input_frames": len(mapped) - len(first_input_for_source)}}


def aligned_frame_report(predictions, truth, time_map, frame_width, match_tol_px=5.0):
    aligned = align_frames(predictions, truth, time_map)
    return {"alignment": aligned["metadata"],
            "native_source_frames": frame_report(aligned["source_predictions"], truth, frame_width, match_tol_px),
            "model_input_frames": frame_report(predictions, aligned["input_truth"], frame_width, match_tol_px)}
