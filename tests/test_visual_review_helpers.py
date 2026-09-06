"""Synthetic navigation safety checks; these do not measure annotation quality."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/stage1" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def material(tmp_path, points):
    packet = tmp_path / "packet"
    packet.mkdir()
    source = np.random.default_rng(0).integers(0, 256, (180, 220, 3), dtype=np.uint8)
    assert cv2.imwrite(str(packet / "source.png"), source)
    with (packet / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_idx", "source_pts_ms", "image", "visible", "x", "y"])
        writer.writerow([0, "1000.125", "source.png", "", "", ""])
    (packet / "rallies.json").write_text('{"status":"unreviewed","rallies":[]}')
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"packets": {"test": {"manual": points}}}))
    before = {p.name: p.read_bytes() for p in packet.iterdir()}
    return packet, decisions, source, before


@pytest.mark.parametrize("point", [(110, 90), (219, 179)])
def test_manual_crop_preserves_native_pixels_and_never_writes_truth(tmp_path, monkeypatch, point):
    packet, decisions, source, before = material(tmp_path, [[0, *point]])
    out = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", ["render", str(decisions), "test", str(packet), "--out", str(out)])
    helper("render_manual_review").main()
    rendered = cv2.imread(str(out / "manual_000.png"))[208:336, :128]
    x, y = point
    left, top = max(0, min(220 - 128, x - 64)), max(0, min(180 - 128, y - 64))
    expected = source[top:top + 128, left:left + 128].copy()
    cv2.drawMarker(expected, (x - left, y - top), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
    np.testing.assert_array_equal(rendered, expected)
    assert {p.name: p.read_bytes() for p in packet.iterdir()} == before
    saved = (out / "manual_000.png").read_bytes()
    with pytest.raises(FileExistsError):
        helper("render_manual_review").main()
    assert (out / "manual_000.png").read_bytes() == saved


def test_duplicate_manual_points_are_rejected_without_truth_changes(tmp_path, monkeypatch):
    packet, decisions, _, before = material(tmp_path, [[0, 110, 90], [0, 111, 91]])
    out = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", ["render", str(decisions), "test", str(packet), "--out", str(out)])
    with pytest.raises(ValueError, match="duplicate manual points"):
        helper("render_manual_review").main()
    assert not out.exists()
    assert {p.name: p.read_bytes() for p in packet.iterdir()} == before


def test_small_crop_cap_is_only_navigation_and_preserves_gray_context(monkeypatch):
    module = helper("prepare_small_ball_review")
    frame, previous, gray = object(), object(), object()
    choices = [{"radius": 24.9}, {"radius": 25}, {"radius": 25.1}, {"radius": 40}]

    def candidates(actual_frame, actual_previous):
        assert actual_frame is frame and actual_previous is previous
        return choices, gray

    original = SimpleNamespace(candidates=candidates)

    def original_main():
        selected, actual_gray = original.candidates(frame, previous)
        assert selected == choices[:2]
        assert actual_gray is gray
        assert len(choices) == 4  # The original candidates are not rewritten.

    original.main = original_main
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda obj: None))
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda spec: original)
    module.main()
