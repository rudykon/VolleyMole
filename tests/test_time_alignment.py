"""Timing arithmetic invariants, not measured detection accuracy."""
import pytest

from volleycut.eval.time_alignment import aligned_frame_report, align_frames
from volleycut.eval.gap_metrics import gap_report


def test_duplicate_proxy_frames_do_not_cherry_pick_best_prediction():
    truth = [{"frame_idx": 0, "source_pts_ms": 1234, "visible": 1, "x": 100, "y": 100},
             {"frame_idx": 1, "source_pts_ms": 1267, "visible": 1, "x": 110, "y": 100}]
    mapping = [{"frame_idx": 0, "source_pts_ms": 1234}, {"frame_idx": 1, "source_pts_ms": 1234}]
    pred = [{**mapping[0], "x": -1, "y": -1}, {**mapping[1], "x": 100, "y": 100}]
    result = aligned_frame_report(pred, truth, mapping, 1920)
    assert result["native_source_frames"]["visible_recall"] == 0
    assert result["native_source_frames"]["gt_visible_frames"] == 2
    assert result["model_input_frames"]["visible_recall"] == 0.5
    assert result["alignment"]["source_frames_not_selected"] == 1


def test_mapping_mismatch_or_nearby_source_frame_is_rejected():
    truth = [{"frame_idx": 0, "source_pts_ms": 1000, "visible": 0, "x": -1, "y": -1}]
    mapping = [{"frame_idx": 0, "source_pts_ms": 1000.1}]
    with pytest.raises(ValueError, match="exact source"):
        align_frames([], truth, mapping)
    with pytest.raises(ValueError, match="mismatch"):
        align_frames([{**mapping[0], "source_pts_ms": 1000, "x": -1, "y": -1}], truth, mapping)


def test_missing_first_duplicate_stays_a_false_negative():
    truth = [{"frame_idx": 0, "source_pts_ms": 1000, "visible": 1, "x": 100, "y": 100}]
    mapping = [{"frame_idx": i, "source_pts_ms": 1000} for i in range(2)]
    result = aligned_frame_report([{**mapping[1], "x": 100, "y": 100}], truth, mapping, 1920)
    assert result["native_source_frames"]["missing_prediction_frames"] == 1


def gap_fixture():
    truth = {i: {"frame_idx": i, "source_pts_ms": i * 100, "visible": 1, "x": 100 + i, "y": 100} for i in range(3)}
    tracks = [{"track_id": 1, "points": [{**p, "state": "interpolated" if i == 1 else "observed"} for i, p in truth.items()],
               "gap_links": [{"from_frame_idx": 0, "to_frame_idx": 2}]}]
    return truth, tracks


def test_unobservable_gap_is_not_reported_as_correct():
    truth, tracks = gap_fixture()
    truth[1]["visible"] = 0
    report = gap_report(tracks, truth, [])
    assert report["unverifiable"] == 1 and report["strict_correct_rate"] == 0
    assert report["verified_correct_rate"] is None


def test_gap_crossing_independent_rallies_is_incorrect():
    truth, tracks = gap_fixture()
    report = gap_report(tracks, truth, [{"start_ms": 0, "end_ms": 90}, {"start_ms": 110, "end_ms": 300}])
    assert report["incorrect"] == 1
    assert "crosses_independent_rally_boundaries" in report["detail"][0]["reasons"]
