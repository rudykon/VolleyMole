"""Read-only checks on real annotation material, not constructed quality results."""
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.local_data

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data/validation_work/2_tuning"
CONTEXT = PACKET / "context_to_150000"


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def test_real_extension_preserved_original_frame_identity_and_split():
    before = table(CONTEXT / "frames.before.csv")
    after = table(PACKET / "frames.csv")
    # Labels may legitimately be completed/corrected after extension; only frame
    # identity is immutable here. The exact pre-extension labels stay backed up.
    identity = ("frame_idx", "source_pts_ms", "image", "image_sha256")
    assert [{k: row[k] for k in identity} for row in before] == [
        {k: row[k] for k in identity} for row in after[:len(before)]]
    record = json.loads((CONTEXT / "extension.json").read_text())
    assert sha(CONTEXT / "frames.before.csv") == record["previous_labels_sha256"]
    assert sha(CONTEXT / "manifest.before.json") == record["previous_manifest_sha256"]
    assert record["labels_added"] == 0 and record["independent_of_predictions"] is True
    assert len(after) == len(before) + record["new_frames"]
    manifest = json.loads((PACKET.parent / "manifest.json").read_text())
    tuning = next(c for c in manifest["clips"] if c["clip_id"] == "2_tuning")
    acceptance = next(c for c in manifest["clips"] if c["clip_id"] == "2_acceptance")
    assert tuning["last_frame_ms"] < acceptance["start_ms"]
    assert not manifest["frozen"] and not (PACKET.parent / "frozen_manifest.json").exists()


def test_real_extension_overlap_and_all_new_images_are_identical_to_record():
    before = table(CONTEXT / "frames.before.csv")
    after = table(PACKET / "frames.csv")
    pattern = re.compile(r"showinfo@truth.*?n:\s*(\d+)\s+pts:\s*(-?\d+)\s")
    decoded = [tuple(map(int, m.groups())) for line in (CONTEXT / "extract.log").read_text().splitlines()
               if (m := pattern.search(line))]
    images = sorted((CONTEXT / "frames").glob("frame_*.png"))
    assert len(decoded) == len(images)
    assert len(images) == len(after) - len(before) + 3
    expected = before[-3:] + after[len(before):]
    for i, (row, image, (n, pts)) in enumerate(zip(expected, images, decoded)):
        assert i == n and abs(float(row["source_pts_ms"]) * 1000 - pts) < .001
        assert sha(image) == row["image_sha256"]
        if i:
            assert pts > decoded[i - 1][1]
