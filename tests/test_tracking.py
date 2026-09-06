"""Geometric invariants only; these constructed cases are not video quality evidence."""
import pytest

from volleycut.tracking import TrackConfig, build_tracks
from volleycut.rallies import RallyConfig, generate_rallies


def row(i, t, x=-1, y=100, roi=True):
    return {"frame_idx": i, "source_pts_ms": t, "x": x, "y": y if x >= 0 else -1, "within_roi": roi}


def test_live_track_cannot_bypass_short_gap_limit():
    rows = [row(i, i * 100, 100 + i if i <= 2 or i >= 8 else -1) for i in range(11)]
    decisions = []
    tracks = build_tracks(rows, 10, 1000, decisions=decisions)
    assert len(tracks) == 2
    assert any(d.get("reason") == "gap_exceeds_limit" for d in decisions)
    assert all(p["state"] == "observed" for t in tracks for p in t["points"])


def test_interpolation_uses_actual_vfr_pts_not_frame_number():
    rows = [row(0, 0, 100), row(1, 10), row(2, 90), row(3, 100, 200)]
    track = build_tracks(rows, 30, 1000)[0]
    assert [(p["source_pts_ms"], p["x"]) for p in track["points"]] == [(0, 100), (10, 110), (90, 190), (100, 200)]
    assert [p["state"] for p in track["points"]] == ["observed", "interpolated", "interpolated", "observed"]


def test_missing_pts_cannot_be_invented_for_a_gap():
    tracks = build_tracks([row(0, 0, 100), row(1, 100, 110), row(4, 200, 120), row(5, 300, 130)], 30, 1000)
    assert len(tracks) == 2
    assert not any(t["gap_links"] for t in tracks)


def test_isolated_direction_acceleration_spike_is_explained():
    rows = [row(i, i * 100 / 3, 550 if i == 4 else 100 + i * 10) for i in range(9)]
    decisions = []
    tracks = build_tracks(rows, 30, 1000, decisions=decisions)
    middle = next(p for t in tracks for p in t["points"] if p["frame_idx"] == 4)
    assert middle["state"] == "interpolated" and middle["x"] == 140
    assert any(d.get("reason") == "isolated_acceleration_direction_spike" and d["frame_idx"] == 4 for d in decisions)


def test_normal_ball_turn_is_not_removed_by_direction_alone():
    rows = [row(i, i * 100 / 3, 100 + (4 - abs(i - 4)) * 5) for i in range(9)]
    tracks = build_tracks(rows, 30, 1000)
    assert all(p["state"] == "observed" for t in tracks for p in t["points"])


def test_resolution_scaling_does_not_change_track_membership():
    rows = [row(i, i * 40, 100 + i * 12 if i not in (3, 4) else -1) for i in range(10)]
    scaled = [{**p, "x": p["x"] * 2 if p["x"] >= 0 else -1, "y": p["y"] * 2 if p["y"] >= 0 else -1} for p in rows]
    a, b = build_tracks(rows, 25, 1000), build_tracks(scaled, 50, 2000)
    assert [[(p["frame_idx"], p["state"]) for p in t["points"]] for t in a] == [
        [(p["frame_idx"], p["state"]) for p in t["points"]] for t in b]


def test_csv_false_roi_value_does_not_become_true():
    decisions = []
    tracks = build_tracks([row(i, i * 100, 100 + i, roi="0") for i in range(10)], 10, 1000, decisions=decisions)
    assert tracks == [] and len(decisions) == 10


def test_quality_flag_uses_actual_roi_edge_and_source_start():
    points = [{"frame_idx": i, "source_pts_ms": 10000 + i * 100, "x": 210, "y": 300 + i * 2, "state": "observed"}
              for i in range(16)]
    result = generate_rallies([{"track_id": 1, "points": points, "gap_links": []}], 13000, "test", "fixture", 1000, 1000,
                              video_start_ms=10000, roi_polygon=[(200, 200), (800, 200), (800, 800), (200, 800)])
    assert result["rallies"][0]["start_ms"] == 10000
    assert "possible_spare_ball" in result["rallies"][0]["quality_flags"]


def test_stationary_point_does_not_by_itself_create_a_rally():
    points = [{"frame_idx": i, "source_pts_ms": i * 100, "x": 500, "y": 500, "state": "observed"} for i in range(21)]
    result = generate_rallies([{"track_id": 1, "points": points, "gap_links": []}], 3000, "test", "fixture", 1000, 1000)
    assert result["rallies"] == []


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        build_tracks([], 30, 1920, TrackConfig(max_gap_ms=float("nan")))
