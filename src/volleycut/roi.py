"""Simple polygon geometry in rotation-normalized input pixels.

Boundary points/segments are included. No integer snapping, polygon repair, or
sampling-only shortcut: even a narrow concavity between recorded PTS is checked.
EPS is only numerical tolerance (pixels), not a quality/acceptance threshold.
"""
import math
from collections.abc import Sequence

from .validation import dimension, number

ROI_RULE_VERSION = 1
EPS = 1e-9


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _on_segment(p, a, b):
    edge = _sub(b, a)
    return (abs(_cross(edge, _sub(p, a))) <= EPS * max(1, math.hypot(*edge))
            and min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS)


def _intersection_parameters(a, b, c, d):
    """Locations on AB intersecting CD, including collinear overlap endpoints."""
    r, s = _sub(b, a), _sub(d, c)
    length2 = r[0] * r[0] + r[1] * r[1]
    if length2 <= EPS * EPS:
        return [0.0] if _on_segment(a, c, d) else []
    denominator = _cross(r, s)
    delta = _sub(c, a)
    tolerance = EPS * max(1, math.hypot(*r), math.hypot(*s))
    if abs(denominator) > tolerance:
        t, u = _cross(delta, s) / denominator, _cross(delta, r) / denominator
        if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS:
            return [min(1.0, max(0.0, t))]
        return []
    if abs(_cross(delta, r)) > tolerance:
        return []
    t0 = sum(x * y for x, y in zip(delta, r)) / length2
    t1 = sum(x * y for x, y in zip(_sub(d, a), r)) / length2
    lo, hi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
    return [lo, hi] if lo <= hi else []


def validate_polygon(polygon, width=None, height=None):
    if polygon is None:
        return None
    if isinstance(polygon, (str, bytes)) or not isinstance(polygon, Sequence):
        raise ValueError("roi_polygon must be an ordered sequence of x/y pairs")
    points = []
    for i, point in enumerate(polygon):
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError(f"ROI vertex {i} must contain exactly x and y")
        points.append(tuple(float(number(v, f"ROI vertex {i}", minimum=0)) for v in point))
    # A conventional repeated closing vertex is accepted, without changing shape.
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("ROI requires at least three distinct vertices with no repeated edges")
    if width is not None:
        width = dimension(width, "frame_width")
        if any(x >= width for x, _ in points):
            raise ValueError("ROI x lies outside rotation-normalized frame pixels")
    if height is not None:
        height = dimension(height, "frame_height")
        if any(y >= height for _, y in points):
            raise ValueError("ROI y lies outside rotation-normalized frame pixels")
    edges = list(zip(points, points[1:] + points[:1]))
    area_terms = [_cross(_sub(a, points[0]), _sub(b, points[0])) for a, b in edges]
    if not all(math.isfinite(v) for v in area_terms):
        raise ValueError("ROI coordinate range is too large")
    if abs(math.fsum(area_terms)) <= EPS:
        raise ValueError("ROI must enclose nonzero area")
    for i, (a, b) in enumerate(edges):
        if math.dist(a, b) <= EPS:
            raise ValueError("ROI has a zero-length edge")
        c = edges[(i + 1) % len(edges)][1]
        # Adjacent collinear vertices are OK only when they do not double back.
        if _on_segment(c, a, b) or _on_segment(a, b, c):
            raise ValueError("ROI has overlapping adjacent edges")
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _intersection_parameters(a, b, *edges[j]):
                raise ValueError("ROI must be simple: nonadjacent edges intersect or touch")
    return tuple(points)


def contains(polygon, point):
    """Boundary-inclusive containment; polygon must already be validated."""
    if polygon is None:
        return True
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_segment(point, a, b):
            return True
        if (a[1] > y) != (b[1] > y):
            cross_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < cross_x:
                inside = not inside
    return inside


def contains_segment(polygon, a, b):
    if polygon is None:
        return True
    if not contains(polygon, a) or not contains(polygon, b):
        return False
    parameters = [0.0, 1.0]
    for c, d in zip(polygon, polygon[1:] + polygon[:1]):
        parameters.extend(_intersection_parameters(a, b, c, d))
    parameters = sorted(set(parameters))
    for lo, hi in zip(parameters, parameters[1:]):
        t = (lo + hi) / 2
        if not contains(polygon, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))):
            return False
    return True


def signed_edge_distance(polygon, point):
    distances = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        edge, delta = _sub(b, a), _sub(point, a)
        length2 = sum(v * v for v in edge)
        t = max(0.0, min(1.0, sum(x * y for x, y in zip(delta, edge)) / length2))
        distances.append(math.dist(point, (a[0] + t * edge[0], a[1] + t * edge[1])))
    distance = min(distances)
    return distance if contains(polygon, point) else -distance
