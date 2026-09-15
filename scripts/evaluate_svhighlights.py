#!/usr/bin/env python3
"""Evaluate existing SVHighlights volleyball labels; never create ground truth.

Dense submissions: [{"vid": "volleyball_1", "pred_saliency_scores": [...]}].
Each list must contain exactly the published number of trimmed-video 2s bins.
Interval submissions: [{"vid": ..., "time_basis": "source" | "trimmed",
"events": [{"start_sec": ..., "end_sec": ..., "score": ...}],
"background_score": 0}]. Intervals are half-open; an overlapping bin receives
the maximum event score, with background_score used for uncovered bins.

The standard noninterpolated AP here is NOT the authors' HL-mAP implementation.
The volume baseline uses only the authors' released audio loudness features;
it is not a VolleyMole system result and does not process original broadcasts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


TAIL_BIN_VIDEOS = {
    "volleyball_4", "volleyball_12", "volleyball_13", "volleyball_15", "volleyball_24",
    "volleyball_27", "volleyball_28", "volleyball_30", "volleyball_31", "volleyball_40",
}


def finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not a boolean or string")
    return float(value)


def score_metrics(labels, scores):
    """Threshold-grouped sum(delta_recall * precision), with no interpolation."""
    if not labels or len(labels) != len(scores):
        raise ValueError("Prediction and original labels require equal nonzero bin counts")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("Original labels must be binary integers")
    scores = [finite_number(x, "score") for x in scores]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    positives = sum(labels)
    retrieved = tp = 0
    ap = 0.0
    # All predictions tied at a score enter the PR curve at the same threshold.
    # Arbitrary index ordering must not inflate AP for a constant prediction.
    for _, group in itertools.groupby(order, key=lambda i: scores[i]):
        indices = list(group)
        new_tp = sum(labels[i] for i in indices)
        tp += new_tp
        retrieved += len(indices)
        if positives:
            ap += (new_tp / positives) * (tp / retrieved)
    return {
        "ap_noninterpolated": ap,
        "hit1": float(labels[order[0]]),
        "hitk_gt_positive_bins": sum(labels[i] for i in order[:positives]) / positives if positives else 0.0,
        "k": positives,
        "bins": len(labels),
        "positive_bins": positives,
        "positive_fraction": positives / len(labels),
    }


def unique_rows(rows, name):
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("vid"), str) for row in rows):
        raise ValueError(f"{name} must be a list of objects with string vid")
    duplicate = [vid for vid, count in Counter(row["vid"] for row in rows).items() if count > 1]
    if duplicate:
        raise ValueError(f"Duplicate {name} video IDs: {duplicate}")
    return {row["vid"]: row for row in rows}


def bin_coordinate(seconds, bin_sec):
    value = seconds / bin_sec
    nearest = round(value)
    # Subtracting decimal source trim offsets can introduce tiny FP errors at
    # exact bin boundaries. Snap only numerical noise, never an actual frame.
    return float(nearest) if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9) else value


def intervals_to_scores(row, meta):
    time_basis = row.get("time_basis")
    if time_basis not in ("source", "trimmed"):
        raise ValueError("Event submissions must explicitly specify source or trimmed time_basis")
    bin_sec = finite_number(meta["label_bin_sec"], "label_bin_sec")
    count = meta["label_bins"]
    source_start = finite_number(meta["source_trim_start_sec"], "source_trim_start_sec")
    duration = finite_number(meta["source_trim_end_sec"], "source_trim_end_sec") - source_start
    if duration <= 0 or bin_sec != 2 or type(count) is not int or count <= 0:
        raise ValueError("Invalid original video metadata")
    # Some source annotations omit a partial final bin. Do not fabricate it.
    evaluation_end = min(duration, count * bin_sec)
    background = finite_number(row.get("background_score", 0.0), "background_score")
    scores = [background] * count
    covered = [False] * count
    events = row.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    audit = {"time_basis": time_basis, "source_trim_subtracted_sec": source_start if time_basis == "source" else 0,
             "event_count": len(events), "clipped_event_count": 0, "outside_evaluation_event_count": 0,
             "evaluation_end_sec": evaluation_end, "unlabelled_source_tail_sec": max(0.0, duration - evaluation_end),
             "mapping": "maximum score over positive-duration overlap with each half-open original bin"}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each event must be an object")
        start = finite_number(event.get("start_sec"), "start_sec")
        end = finite_number(event.get("end_sec"), "end_sec")
        score = finite_number(event.get("score"), "event score")
        if start < 0 or end <= start:
            raise ValueError("Event intervals require 0 <= start_sec < end_sec")
        if time_basis == "source":
            start -= source_start
            end -= source_start
        clipped_start, clipped_end = max(0.0, start), min(evaluation_end, end)
        if clipped_start != start or clipped_end != end:
            audit["clipped_event_count"] += 1
        if clipped_end <= clipped_start:
            audit["outside_evaluation_event_count"] += 1
            continue
        for index in range(math.floor(bin_coordinate(clipped_start, bin_sec)),
                           min(count, math.ceil(bin_coordinate(clipped_end, bin_sec)))):
            scores[index] = max(scores[index], score) if covered[index] else score
            covered[index] = True
    return scores, audit


def normalize_predictions(rows, metadata, allow_subset=False):
    predictions = unique_rows(rows, "prediction")
    expected, actual = set(metadata), set(predictions)
    if actual - expected or (expected - actual and not allow_subset) or not actual:
        raise ValueError(f"Prediction video IDs do not match benchmark: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    scores, audit = {}, {}
    for vid, row in predictions.items():
        if ("pred_saliency_scores" in row) == ("events" in row):
            raise ValueError("Each video must specify exactly one of pred_saliency_scores or events")
        if "pred_saliency_scores" in row:
            if row.get("time_basis", "trimmed") != "trimmed" or row.get("bin_sec", 2) != 2:
                raise ValueError("Dense scores must align to the original trimmed-video 2s bins")
            values = row["pred_saliency_scores"]
            if not isinstance(values, list) or len(values) != metadata[vid]["label_bins"]:
                raise ValueError(f"Strict bin-count mismatch for {vid}; no padding or truncation is applied")
            scores[vid] = [finite_number(v, "pred_saliency_score") for v in values]
            audit[vid] = {"format": "dense_original_2s_bins", "bin_count": len(values)}
        else:
            scores[vid], audit[vid] = intervals_to_scores(row, metadata[vid])
    return scores, audit


def loudness_predictions(volume, metadata):
    if not isinstance(volume, dict):
        raise ValueError("Original volume file must map video IDs to dBFS lists")
    rows, audit = [], []
    for vid, meta in metadata.items():
        values = volume.get(vid)
        if not isinstance(values, list):
            raise ValueError(f"Missing original volume for {vid}")
        original_count = len(values)
        expected = meta["label_bins"]
        trimmed = vid in TAIL_BIN_VIDEOS
        if original_count != expected + int(trimmed):
            raise ValueError(f"Unexpected original volume length for {vid}")
        discarded = values[-1] if trimmed else None
        retained = values[:-1] if trimmed else values[:]
        finite = [finite_number(v, "dBFS") for v in retained if v != "-Inf"]
        # -Inf is an existing dBFS silence feature, not a missing label. A finite
        # bottom score preserves ranking while keeping output JSON strict.
        floor = min(finite) - 1.0 if finite else -1.0
        scores = [floor if v == "-Inf" else finite_number(v, "dBFS") for v in retained]
        rows.append({"vid": vid, "pred_saliency_scores": scores})
        audit.append({"vid": vid, "original_volume_bins": original_count, "label_bins": expected,
                      "removed_only_final_bin": trimmed, "removed_final_bin_value": discarded,
                      "negative_infinity_silence_bins": retained.count("-Inf"), "silence_rank_floor": floor,
                      "unlabelled_source_tail_sec": max(0.0, meta["source_trim_end_sec"] - meta["source_trim_start_sec"] - expected * 2)})
    return rows, audit


def load_benchmark(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        source = directory / artifact["path"]
        if not source.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Source artifact path escapes dataset directory")
        if hashlib.sha256(source.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Source checksum mismatch: {artifact['path']}")
    metadata = unique_rows(manifest["videos"], "metadata")
    if set(metadata) != {f"volleyball_{i}" for i in range(1, 41)}:
        raise ValueError("Manifest must contain the 40 original volleyball broadcasts")
    original = unique_rows(json.loads((directory / "annotations/label.json").read_text()), "original label")
    labels = {vid: original[vid]["saliency_scores"] for vid in metadata}
    for vid, meta in metadata.items():
        values = labels[vid]
        if not isinstance(values, list) or len(values) != meta["label_bins"] or sum(values) != meta["positive_bins"]:
            raise ValueError("Original labels and manifest metadata disagree")
        if any(type(y) is not int or y not in (0, 1) for y in values):
            raise ValueError("Original labels must remain binary integers")
    return manifest, metadata, labels


def evaluate(labels, scores):
    videos = {vid: score_metrics(labels[vid], values) for vid, values in scores.items()}
    if not videos:
        raise ValueError("No evaluated videos")
    fields = ("ap_noninterpolated", "hit1", "hitk_gt_positive_bins", "positive_fraction")
    return {"macro_mean": {field: statistics.mean(row[field] for row in videos.values()) for field in fields},
            "per_video": videos, "video_count": len(videos),
            "total_bins": sum(row["bins"] for row in videos.values()),
            "positive_bins": sum(row["positive_bins"] for row in videos.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/public_benchmarks/svhighlights"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--loudness-baseline", action="store_true")
    parser.add_argument("--model-name", default="external_prediction_submission")
    parser.add_argument("--allow-subset", action="store_true", help="Explicitly permit incomplete evaluation; report missing broadcasts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, metadata, labels = load_benchmark(args.dataset)
    adaptation = []
    if args.loudness_baseline:
        prediction_path = args.dataset / "annotations/volume.json"
        rows, adaptation = loudness_predictions(json.loads(prediction_path.read_text()), metadata)
        name = "author_released_raw_audio_loudness_baseline"
    else:
        prediction_path = args.predictions
        rows = json.loads(prediction_path.read_text())
        name = args.model_name
    scores, mapping = normalize_predictions(rows, metadata, args.allow_subset)
    report = {"schema_version": 1, "dataset": "SVHighlights volleyball", "model_or_baseline": name,
              "is_volleymole_end_to_end_evaluation": False,
              "source_revision": manifest["revision"], "dataset_manifest": str(args.dataset / "manifest.json"),
              "prediction_source": str(prediction_path), "prediction_source_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "ground_truth_scope": manifest["ground_truth_scope"],
              "metric_definitions": {
                  "ap_noninterpolated": "sum of recall increments times precision at each distinct score threshold; standard noninterpolated AP, NOT authors HL-mAP",
                  "hit1": "ground truth of highest score bin; ties choose earliest bin",
                  "hitk_gt_positive_bins": "positive fraction among exactly top K bins, K=number of original positive bins; ties choose earliest bin",
                  "macro_mean": "equal weight per broadcast, not pooled bins; scores on 0..1 scale",
                  "zero_positive_policy": "AP and HitK are 0 when the source contains no positives"},
              "coverage": {"expected_videos": len(metadata), "evaluated_videos": len(scores),
                           "missing_videos": sorted(set(metadata) - set(scores)), "full_40_video_subset": len(scores) == 40},
              "source_feature_adaptations": adaptation, "prediction_mapping": mapping,
              "limitations": ["Editorial highlight alignment is a proxy, not human top-five preference or blooper judgement",
                              "Original broadcasts were not downloaded or processed by this evaluator",
                              "Loudness baseline cannot establish VolleyMole quality, audiovisual event understanding or end-to-end timing"],
              **evaluate(labels, scores)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "baseline_or_model": name,
                      "video_count": report["video_count"], "macro_mean": report["macro_mean"],
                      "removed_final_bins": sum(row["removed_only_final_bin"] for row in adaptation),
                      "silence_bins": sum(row["negative_infinity_silence_bins"] for row in adaptation)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
