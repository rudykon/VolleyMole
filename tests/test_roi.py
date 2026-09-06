"""Constructed geometry tests are invariants, never real-video quality evidence."""
import pytest

from volleycut.roi import contains, contains_segment, signed_edge_distance, validate_polygon
from volleycut.tracking import build_tracks

U = ((0, 0), (1000, 0), (1000, 1000), (600, 1000), (600, 400), (400, 400), (400, 1000), (0, 1000))


@pytest.mark.parametrize("polygon", [
    [], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
    [(0, 0), (5, 5), (0, 5), (5, 0)],
    [(0, 0), (5, 0), (5, 5), (2, 0), (0, 5)],
    [(0, 0), (5, 0), (3, 0), (5, 5), (0, 5)],
    [(0, 0), (5, 0), (5, 5), (0, 0), (0, 5)],
    [(0, 0), (float("nan"), 0), (0, 5)],
    [(0, 0), (float("inf"), 0), (0, 5)],
    [(-1, 0), (5, 0), (0, 5)], [(False, 0), (5, 0), (0, 5)],
    [("0", 0), (5, 0), (0, 5)], [(0, 0, 1), (5, 0), (0, 5)], "0,0;5,0;0,5",
])
def test_invalid_simple_polygons_are_rejected(polygon):
    with pytest.raises(ValueError):
        validate_polygon(polygon)


def test_closed_polygon_and_collinear_forward_vertices_preserve_shape():
    p = validate_polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    assert len(p) == 5 and contains(p, (5, 5))
    assert contains_segment(p, (0, 0), (10, 0))
    assert contains_segment(p, (0, 0), (0, 0))


def test_fractional_boundary_is_not_snapped_to_integer():
    p = validate_polygon([(0.75, .75), (9.25, .75), (9.25, 9.25), (.75, 9.25)], 10, 10)
    assert not contains(p, (0.5, 5)) and contains(p, (.75, 5))
    assert signed_edge_distance(p, (.5, 5)) == -.25
    assert signed_edge_distance(p, (1, 5)) == .25
    with pytest.raises(ValueError):
        validate_polygon([(0, 0), (10, 0), (0, 9)], 10, 10)


def test_concave_segment_checked_between_not_just_at_sampled_points():
    p = validate_polygon(U, 1200, 1200)
    assert all(contains(p, point) for point in [(100, 800), (300, 800), (700, 800), (900, 800)])
    assert not contains_segment(p, (100, 800), (900, 800))
    assert contains_segment(p, (100, 300), (900, 300))
    assert contains_segment(p, (400, 400), (600, 400))
    assert not contains_segment(p, (400, 500), (600, 500))
    reverse = tuple(reversed(p))
    assert not contains_segment(reverse, (900, 800), (100, 800))


def test_narrow_notch_crossing_cannot_hide_between_sample_pts():
    p = validate_polygon([(0, 0), (1000, 0), (1000, 1000), (501, 1000),
                          (501, 500), (500, 500), (500, 1000), (0, 1000)])
    assert contains(p, (499, 800)) and contains(p, (502, 800))
    assert not contains_segment(p, (100, 800), (900, 800))


def row(i, t, x):
    return {"frame_idx": i, "source_pts_ms": t, "x": x, "y": 800 if x >= 0 else -1, "within_roi": True}


def test_real_pts_interpolation_splits_before_crossing_roi():
    rows = [row(0, 0, 100), row(1, 100, 200), row(2, 125, -1), row(3, 375, -1),
            row(4, 400, 800), row(5, 500, 900)]
    decisions = []
    tracks = build_tracks(rows, 30, 1200, frame_height=1200, roi_polygon=U, decisions=decisions)
    assert len(tracks) == 2 and all(not t["gap_links"] for t in tracks)
    assert any(d["reason"] == "connection_crosses_roi" for d in decisions)
    assert all(p["state"] == "observed" for t in tracks for p in t["points"])
    # The absent ROI case retains the same temporal/speed connection.
    assert len(build_tracks(rows, 30, 1200)) == 1


def test_adjacent_observations_cannot_draw_a_track_outside_roi():
    rows = [row(0, 0, 100), row(1, 100, 200), row(2, 300, 800), row(3, 400, 900)]
    assert len(build_tracks(rows, 30, 1200, roi_polygon=U)) == 2


def test_polygon_overrides_stale_point_flag_and_reports_rejection():
    rows = [row(i, i * 100, 500) for i in range(4)]
    decisions = []
    assert build_tracks(rows, 30, 1200, roi_polygon=U, decisions=decisions) == []
    assert all(d["reason"] == "outside_roi" for d in decisions) and len(decisions) == 4


def test_fractional_edge_interpolation_is_not_rounded_outside_roi():
    p = ((0, 0), (1000, 333.3333333333333), (1000, 1000), (0, 1000))
    rows = [{"frame_idx": i, "source_pts_ms": t, "x": x, "y": y} for i, (t, x, y) in enumerate(
        [(0, 0, 0), (25, -1, -1), (100, 300, 100)])]
    track = build_tracks(rows, 30, 1200, roi_polygon=p)[0]
    assert all(contains(p, (point["x"], point["y"])) for point in track["points"])
