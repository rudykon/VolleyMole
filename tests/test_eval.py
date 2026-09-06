"""Arithmetic edge cases only, not model-quality or acceptance evidence."""
import pytest

from volleycut.eval.frame_metrics import frame_report
from volleycut.eval.matching import match_rallies


def test_missing_predictions_are_false_negatives():
    truth = [{"frame_idx": i, "visible": 1, "x": 10, "y": 10} for i in range(3)]
    report = frame_report([{"frame_idx": 0, "x": 10, "y": 10}], truth, 1920)
    assert report["gt_visible_frames"] == 3
    assert report["visible_recall"] == 0.333333
    assert report["missing_prediction_frames"] == 2


def test_unannotated_frames_do_not_become_false_positives():
    report = frame_report([{"frame_idx": 4, "x": 10, "y": 10}], [], 1920)
    assert report["false_detections"] == 0
    assert report["pred_visible_frames"] == 0


def test_duplicate_predictions_cannot_inflate_recall():
    pred = {"frame_idx": 0, "x": 10, "y": 10}
    with pytest.raises(ValueError, match="duplicate"):
        frame_report([pred, pred], [], 1920)


def test_split_tie_has_no_unique_hit_and_does_not_hide_missed_truth():
    report = match_rallies([{"start_ms": 0, "end_ms": 1000}] * 2,
                           [{"start_ms": 0, "end_ms": 1000}])
    assert report["hits"] == []
    assert report["missed_gts"] == [0]
    assert report["fragmented_gts"] == [0]
    assert report["false_cuts"] == []


def test_sticking_is_not_counted_as_a_successful_match():
    report = match_rallies([{"start_ms": 0, "end_ms": 2000}],
                          [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 1000, "end_ms": 2000}])
    assert report["sticking_preds"] == [0]
    assert report["hits"] == []
    assert report["missed_gts"] == [0, 1]
    assert report["false_cuts"] == []
