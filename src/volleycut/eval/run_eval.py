"""Diagnostic metrics for independently reviewed truth; not a frozen acceptance runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .rally_metrics import rally_report
from .time_alignment import aligned_frame_report, align_frames
from .gap_metrics import gap_report


def _load_frame_gt(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "source_pts_ms": float(row["source_pts_ms"]),
                    "x": float(row["x"]) if row["x"] not in ("", "-1") else -1,
                    "y": float(row["y"]) if row["y"] not in ("", "-1") else -1,
                    "visible": int(row["visible"]),
                }
            )
    return rows


def _load_pred_detections(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "source_pts_ms": float(row["source_pts_ms"]),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "score": float(row["score"]),
                    "model": row["model"],
                    "within_roi": bool(int(row["within_roi"])),
                }
            )
    return rows


def _load_rally_gt(path: str):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("status") != "reviewed" or not doc.get("annotator"):
        raise ValueError("independent rally truth has not been reviewed")
    return doc["rallies"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="volleycut.eval")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--rallies", required=True)
    p.add_argument("--frame-gt", required=True)
    p.add_argument("--rally-gt", required=True)
    p.add_argument("--frame-width", type=int, required=True)
    p.add_argument("--time-map", required=True, help="actual input-frame to source-PTS mapping")
    p.add_argument("--tracks", required=True)
    p.add_argument("--match-tol-px", type=float, default=5.0)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    preds = _load_pred_detections(args.detections)
    frame_gt = _load_frame_gt(args.frame_gt)

    with open(args.time_map, newline="", encoding="utf-8") as stream:
        time_rows = list(csv.DictReader(stream))
    aligned = align_frames(preds, frame_gt, time_rows)
    tracks = [json.loads(line) for line in Path(args.tracks).read_text().splitlines() if line.strip()]

    pred_rallies = json.loads(Path(args.rallies).read_text(encoding="utf-8"))["rallies"]
    rally_gt = _load_rally_gt(args.rally_gt)

    report = {
        "clip_id": args.clip_id,
        "report_kind": "diagnostic_metrics_not_acceptance_decision",
        "frame_level": aligned_frame_report(preds, frame_gt, time_rows, args.frame_width, args.match_tol_px),
        "short_gap_links": gap_report(tracks, aligned["truth_by_input_frame"], rally_gt, args.match_tol_px),
        "rally_level": rally_report(pred_rallies, rally_gt, iou_threshold=args.iou_threshold),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["frame_level"], ensure_ascii=False, indent=2))
    rl = {k: v for k, v in report["rally_level"].items() if k != "detail"}
    print(json.dumps(rl, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
