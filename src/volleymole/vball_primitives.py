"""Adapted VballNet primitives (MIT, Copyright 2025 Alexander Sigatchov).

See licenses/tracking-MIT.txt and THIRD_PARTY.md for pinned source and changes.
"""
import cv2
import numpy as np

DEFAULT_INPUT_HEIGHT = 288
DEFAULT_INPUT_WIDTH = 512
DEFAULT_HEATMAP_THRESHOLD = .5
BALL_TREND_FRAMES = 3
BALL_RADIUS_MIN = 3
BALL_RADIUS_MAX = 40
BALL_ROI_HALF_SIZE = 48

def preprocess_frames(
    frames, input_height=DEFAULT_INPUT_HEIGHT, input_width=DEFAULT_INPUT_WIDTH
):
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (input_width, input_height))
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)
    return processed


def postprocess_heatmap_output(
    output,
    threshold=DEFAULT_HEATMAP_THRESHOLD,
    input_height=DEFAULT_INPUT_HEIGHT,
    input_width=DEFAULT_INPUT_WIDTH,
    out_dim=9,
):
    results = []
    for frame_idx in range(out_dim):
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results


def build_motion_mask(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    motion_mask = cv2.morphologyEx(
        motion_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    return motion_mask


def contour_narrow_radius(contour):
    if len(contour) < 5:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return float(radius)

    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    narrow_diameter = min(width, height)
    if narrow_diameter <= 0:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return float(radius)
    return float(narrow_diameter) / 2.0


def fallback_radius(size_state):
    smoothed_radius = size_state["smoothed_radius"]
    if smoothed_radius > 0:
        return int(round(smoothed_radius))
    filtered_history = size_state["filtered_history"]
    if not filtered_history:
        return 0
    return int(round(float(np.median(filtered_history))))


def filter_ball_radius(radius, size_state):
    if radius <= 0:
        return fallback_radius(size_state)

    raw_history = size_state["raw_history"]
    filtered_history = size_state["filtered_history"]
    raw_history.append(radius)

    if not filtered_history:
        filtered_history.append(radius)
        size_state["smoothed_radius"] = float(radius)
        return radius

    baseline = size_state["smoothed_radius"]
    if baseline <= 0:
        baseline = float(np.median(filtered_history))

    trend_window = list(raw_history)[-BALL_TREND_FRAMES:]
    trend_confirmed = False
    if len(trend_window) == BALL_TREND_FRAMES:
        upper_shift = [value > baseline for value in trend_window]
        lower_shift = [value < baseline for value in trend_window]
        trend_confirmed = all(upper_shift) or all(lower_shift)

    target_radius = float(radius)
    if trend_confirmed:
        target_radius = float(np.median(trend_window))
    else:
        max_deviation = max(3.0, baseline * 0.55)
        target_radius = float(
            np.clip(radius, baseline - max_deviation, baseline + max_deviation)
        )

    alpha = 0.6 if trend_confirmed else 0.3
    smoothed_radius = baseline * (1.0 - alpha) + target_radius * alpha
    smoothed_radius = float(np.clip(smoothed_radius, BALL_RADIUS_MIN, BALL_RADIUS_MAX))

    accepted_radius = int(round(smoothed_radius))
    filtered_history.append(accepted_radius)
    size_state["smoothed_radius"] = smoothed_radius
    return accepted_radius


def estimate_ball_radius(prev_gray, gray, x_orig, y_orig, size_state):
    if prev_gray is None or x_orig < 0 or y_orig < 0:
        return 0, None

    motion_mask = build_motion_mask(prev_gray, gray)
    x1 = max(0, x_orig - BALL_ROI_HALF_SIZE)
    y1 = max(0, y_orig - BALL_ROI_HALF_SIZE)
    x2 = min(gray.shape[1], x_orig + BALL_ROI_HALF_SIZE)
    y2 = min(gray.shape[0], y_orig + BALL_ROI_HALF_SIZE)
    roi = motion_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return fallback_radius(size_state), None

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_radius(size_state), None

    center = np.array([x_orig - x1, y_orig - y1], dtype=np.float32)
    best_contour = None
    best_score = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        radius = contour_narrow_radius(contour)
        if radius < BALL_RADIUS_MIN or radius > BALL_RADIUS_MAX:
            continue
        distance = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center)
        score = distance - area * 0.02
        if best_score is None or score < best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return fallback_radius(size_state), None

    radius = contour_narrow_radius(best_contour)
    filtered_radius = filter_ball_radius(int(round(radius)), size_state)
    contour_global = best_contour + np.array([[[x1, y1]]], dtype=np.int32)
    return filtered_radius, contour_global
