"""Validate completeness/independence/integrity before freezing stage-1 truth."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .frame_metrics import FRAME_METRICS_VERSION
from .matching import MATCHING_VERSION

TRUTH_VALIDATION_VERSION = 2


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate(root, verify_images=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    errors, details, fingerprints = [], [], {}
    clips = manifest["clips"]
    checked_sources = {}
    for clip in clips:
        source = root.resolve().parents[1] / clip["source"]
        if source not in checked_sources:
            checked_sources[source] = sha(source)
        if checked_sources[source] != clip["source_sha256"]:
            errors.append(f"{clip['clip_id']}: source fingerprint mismatch")
    for split in ("tuning", "acceptance"):
        if len({c["source_sha256"] for c in clips if c["split"] == split}) < 3:
            errors.append(f"{split}: fewer than three independent source videos")
    for i, a in enumerate(clips):
        for b in clips[i + 1:]:
            if a["source_sha256"] != b["source_sha256"] or a["split"] == b["split"]:
                continue
            if max(a["start_ms"], b["start_ms"]) <= min(a["last_frame_ms"], b["last_frame_ms"]):
                errors.append(f"tuning/acceptance leakage: {a['clip_id']}, {b['clip_id']}")
    for clip in clips:
        prefix = clip["clip_id"]
        label_path = root / clip["frame_labels"]
        rally_path = root / clip["rally_labels"]
        with open(label_path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        invalid, unreviewed, corrupt = [], [], []
        if len(rows) != clip["frames"]:
            errors.append(f"{prefix}: incomplete frame table")
        pts = []
        for i, row in enumerate(rows):
            try:
                if int(row["frame_idx"]) != i:
                    raise ValueError("nonsequential frame indices")
                t = float(row["source_pts_ms"])
                if not math.isfinite(t) or (pts and t <= pts[-1]):
                    raise ValueError("invalid source PTS")
                pts.append(t)
                if row["visible"] not in {"0", "1"} or not row["annotator"].strip():
                    unreviewed.append(i)
                elif row["visible"] == "1":
                    x, y = float(row["x"]), float(row["y"])
                    width, height = clip["source_probe"]["width"], clip["source_probe"]["height"]
                    if clip["source_probe"]["rotation"] % 180:
                        width, height = height, width
                    if not (0 <= x < width and 0 <= y < height):
                        raise ValueError("visible point outside native rotated frame")
                elif row["x"] or row["y"]:
                    raise ValueError("invisible frame has coordinates")
                if verify_images and sha(label_path.parent / row["image"]) != row["image_sha256"]:
                    corrupt.append(i)
            except (ValueError, OSError, KeyError):
                invalid.append(i)
        if invalid or unreviewed or corrupt:
            errors.append(f"{prefix}: invalid={len(invalid)}, unlabeled={len(unreviewed)}, corrupt={len(corrupt)}")
        rallies = json.loads(rally_path.read_text())
        if (rallies.get("status") != "reviewed" or not _nonempty_text(rallies.get("annotator"))
                or rallies.get("independent_of_frame_labels_and_predictions") is not True):
            errors.append(f"{prefix}: independent rally review not complete")
        if not rallies["rallies"] and not (rallies.get("confirmed_no_rallies") is True
                                           and _nonempty_text(rallies.get("boundary_context_notes"))):
            errors.append(f"{prefix}: empty rally list is not a reviewed negative")
        for rally in rallies["rallies"]:
            try:
                begin, end = rally["start_ms"], rally["end_ms"]
                # Gate checks must remain active under python -O. Booleans are
                # not timestamps or indices even though bool subclasses int.
                if type(begin) is not int or type(end) is not int or begin >= end:
                    raise ValueError("invalid rally time interval")
                for edge in ("start", "end"):
                    idx = rally[f"{edge}_frame_idx"]
                    if type(idx) is not int or not 0 <= idx < len(pts):
                        raise ValueError("invalid boundary frame index")
                    if round(pts[idx]) != rally[f"{edge}_ms"]:
                        raise ValueError("boundary time does not match source PTS")
                    if not _nonempty_text(rally[f"{edge}_evidence"]):
                        raise ValueError("missing boundary evidence")
            except (KeyError, ValueError, TypeError, IndexError):
                errors.append(f"{prefix}: invalid or undocumented rally boundary")
        fingerprints[clip["frame_labels"]] = sha(label_path)
        fingerprints[clip["rally_labels"]] = sha(rally_path)
        details.append({"clip_id": prefix, "frames": len(rows), "unlabeled_frames": len(unreviewed),
                        "invalid_frames": invalid, "corrupt_images": corrupt,
                        "rally_status": rallies.get("status"), "rallies": len(rallies["rallies"])})
    return {"status": "ready_to_freeze" if not errors else "not_ready", "errors": errors,
            "clips": details, "label_sha256": fingerprints, "images_verified": verify_images,
            "truth_validation_version": TRUTH_VALIDATION_VERSION,
            "matching_version": MATCHING_VERSION, "frame_metrics_version": FRAME_METRICS_VERSION}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    result = validate(args.root)
    (args.root / "validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.freeze and result["status"] == "ready_to_freeze":
        # Exclusive creation: an already frozen acceptance set cannot be overwritten.
        frozen = {"schema_version": 1, "frozen": True, "manifest_sha256": sha(args.root / "manifest.json"),
                  **result}
        with open(args.root / "frozen_manifest.json", "x") as stream:
            json.dump(frozen, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "errors": result["errors"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready_to_freeze" else 2


if __name__ == "__main__":
    raise SystemExit(main())
