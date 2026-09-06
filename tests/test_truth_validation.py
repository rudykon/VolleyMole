"""Artificial validator fixtures only: these are NOT frozen volleyball truth."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from volleycut.eval.validate_truth import sha, validate


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def artificial_truth(tmp_path):
    project = tmp_path / "artificial-validator-project"
    root = project / "data" / "validation_work"
    root.mkdir(parents=True)
    clips = []
    for source_id in range(3):
        source = project / f"artificial-source-{source_id}.bin"
        source.write_bytes(f"Synthetic hash fixture {source_id}; not video".encode())
        for split, start in (("tuning", 100), ("acceptance", 200)):
            name = f"source{source_id}_{split}"
            packet = root / name
            packet.mkdir()
            rows = []
            for i in range(2):
                image = packet / f"frame_{i}.png"
                assert cv2.imwrite(str(image), np.full((24, 32, 3), i, dtype=np.uint8))
                rows.append({"frame_idx": i, "source_pts_ms": start + i * 10,
                             "image": image.name, "image_sha256": sha(image),
                             "visible": 1 if i == 0 else 0, "x": 8 if i == 0 else "",
                             "y": 9 if i == 0 else "", "annotator": "artificial_test_fixture",
                             "notes": "Not real annotation or quality evidence"})
            write_rows(packet / "frames.csv", rows)
            write_json(packet / "rallies.json", {
                "status": "reviewed", "annotator": "artificial_test_fixture",
                "independent_of_frame_labels_and_predictions": True,
                "confirmed_no_rallies": True,
                "boundary_context_notes": "Artificial empty-clip validation case only",
                "rallies": [],
            })
            clips.append({"clip_id": name, "source": source.name, "source_sha256": sha(source),
                          "split": split, "start_ms": start, "last_frame_ms": start + 10,
                          "frames": 2, "frame_labels": f"{name}/frames.csv",
                          "rally_labels": f"{name}/rallies.json",
                          "source_probe": {"width": 32, "height": 24, "rotation": 0}})
    write_json(root / "manifest.json", {"frozen": False, "clips": clips})
    assert validate(root)["status"] == "ready_to_freeze", "control fixture must first be valid"
    return root


def run_freeze(root, optimized=False):
    return subprocess.run([
        sys.executable, *(["-O"] if optimized else []), "-m", "volleycut.eval.validate_truth",
        str(root), "--freeze",
    ], capture_output=True, text=True, timeout=20,
       env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})


@pytest.mark.parametrize("field,value", [
    ("visible", ""), ("annotator", " "), ("frame_idx", "3"),
    ("source_pts_ms", "nan"), ("source_pts_ms", "110"),
    ("x", "nan"), ("x", "32"), ("y", "24"),
    ("visible", "0"), ("image", "missing.png"), ("image_sha256", "0" * 64),
])
def test_invalid_or_unreviewed_frames_prevent_freeze(artificial_truth, field, value):
    path = artificial_truth / "source0_tuning/frames.csv"
    rows = read_rows(path)
    rows[0][field] = value
    write_rows(path, rows)
    result = run_freeze(artificial_truth)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["status"] == "not_ready"
    assert not (artificial_truth / "frozen_manifest.json").exists()


def test_rotation_normalized_dimensions_are_enforced(artificial_truth):
    path = artificial_truth / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["clips"][0]["source_probe"]["rotation"] = 90
    write_json(path, manifest)
    path = artificial_truth / "source0_tuning/frames.csv"
    rows = read_rows(path)
    rows[0]["x"], rows[0]["y"] = "23", "31"
    write_rows(path, rows)
    assert validate(artificial_truth)["status"] == "ready_to_freeze"
    rows[0]["x"] = "24"
    write_rows(path, rows)
    assert validate(artificial_truth)["status"] == "not_ready"


@pytest.mark.parametrize("field,value", [
    ("status", "unreviewed"), ("annotator", ""), ("annotator", "  "),
    ("independent_of_frame_labels_and_predictions", False),
    ("independent_of_frame_labels_and_predictions", 1),
    ("confirmed_no_rallies", False), ("boundary_context_notes", ""),
    ("boundary_context_notes", "  "),
])
def test_empty_rallies_need_explicit_independent_review(artificial_truth, field, value):
    path = artificial_truth / "source0_tuning/rallies.json"
    doc = json.loads(path.read_text())
    doc[field] = value
    write_json(path, doc)
    assert validate(artificial_truth)["status"] == "not_ready"


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("field,value", [
    ("end_ms", 100), ("end_ms", True), ("end_ms", 111),
    ("end_frame_idx", 99), ("end_frame_idx", True),
    ("end_evidence", ""), ("end_evidence", None),
])
def test_bad_rally_boundaries_fail_even_with_python_optimization(artificial_truth, optimized, field, value):
    path = artificial_truth / "source0_tuning/rallies.json"
    doc = json.loads(path.read_text())
    doc["rallies"] = [{"start_ms": 100, "end_ms": 110, "start_frame_idx": 0, "end_frame_idx": 1,
                       "start_evidence": "Synthetic start assertion", "end_evidence": "Synthetic end assertion"}]
    doc["rallies"][0][field] = value
    write_json(path, doc)
    result = run_freeze(artificial_truth, optimized)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["status"] == "not_ready"
    assert not (artificial_truth / "frozen_manifest.json").exists()


@pytest.mark.parametrize("change,expected", [
    ("source", "source fingerprint mismatch"),
    ("overlap", "tuning/acceptance leakage"),
    ("camera_count", "fewer than three independent source videos"),
    ("frame_count", "incomplete frame table"),
])
def test_manifest_integrity_and_split_contracts(artificial_truth, change, expected):
    path = artificial_truth / "manifest.json"
    doc = json.loads(path.read_text())
    if change == "source":
        source = artificial_truth.parents[1] / doc["clips"][0]["source"]
        source.write_bytes(b"modified synthetic source")
    elif change == "overlap":
        doc["clips"][1]["start_ms"] = 110
    elif change == "camera_count":
        doc["clips"] = doc["clips"][:-1]
    else:
        doc["clips"][0]["frames"] = 3
    write_json(path, doc)
    result = validate(artificial_truth)
    assert result["status"] == "not_ready"
    assert any(expected in error for error in result["errors"])


def test_freeze_binds_inputs_and_never_overwrites_existing_record(artificial_truth):
    result = run_freeze(artificial_truth)
    assert result.returncode == 0, result.stderr
    path = artificial_truth / "frozen_manifest.json"
    frozen_bytes = path.read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen["frozen"] is True and frozen["images_verified"] is True
    assert frozen["manifest_sha256"] == sha(artificial_truth / "manifest.json")
    assert len(frozen["label_sha256"]) == 12
    for name, digest in frozen["label_sha256"].items():
        assert sha(artificial_truth / name) == digest
    assert frozen["matching_version"] and frozen["frame_metrics_version"]
    assert frozen["truth_validation_version"] == 2
    repeated = run_freeze(artificial_truth)
    assert repeated.returncode != 0
    assert path.read_bytes() == frozen_bytes


@pytest.mark.parametrize("optimized", [False, True])
def test_documented_positive_boundary_is_accepted_by_validator(artificial_truth, optimized):
    path = artificial_truth / "source0_tuning/rallies.json"
    doc = json.loads(path.read_text())
    doc["confirmed_no_rallies"] = False
    doc["rallies"] = [{"start_ms": 100, "end_ms": 110, "start_frame_idx": 0, "end_frame_idx": 1,
                       "start_evidence": "Synthetic start assertion", "end_evidence": "Synthetic end assertion"}]
    write_json(path, doc)
    result = run_freeze(artificial_truth, optimized)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready_to_freeze"
