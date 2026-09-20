"""Independent, conservative motion evidence for replay boundary review.

The tracker supplies display-oriented coordinates and *source PTS*, not clip
frame numbers. A change of vertical direction alone is not a touch: an ordinary
flight apex has continuous velocity. We instead fit a velocity change while
sharing smooth acceleration on either side of a possible contact. These are
touch hypotheses, also compatible with a net/floor/wall deflection; they must be
checked against the images. Missing tracks never prove that a rally has ended.

None of the public functions approves a replay. ``boundary_guard`` can contradict an
early cut or request better evidence; ``no_conflict`` is not a quality verdict.
"""
import math

import numpy as np


VERSION = 2
TIME_BASIS = 'source presentation timestamp minus container start'


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _fit(points, center, *, kink=False):
    """Fit an ordinary flight or continuous position with a velocity impulse."""
    dt = points[:, 0] - center
    design = np.column_stack((np.ones(len(dt)), dt, dt * dt))
    if kink:
        design = np.column_stack((design, np.maximum(dt, 0)))
    coef, _, _, _ = np.linalg.lstsq(design, points[:, 1:3], rcond=None)
    residual = np.linalg.norm(design @ coef - points[:, 1:3], axis=1)
    # A small number of heatmap errors must not create a large false impulse.
    # Retain most actual samples; this does not manufacture missing positions.
    keep = residual <= max(.004, float(np.quantile(residual, .85)))
    if keep.sum() >= design.shape[1]+3:
        coef, _, _, _ = np.linalg.lstsq(design[keep], points[keep, 1:3], rcond=None)
        residual = np.linalg.norm(design @ coef - points[:, 1:3], axis=1)
    return coef, float(np.quantile(residual, .85)), int((~keep).sum())


def motion_evidence(track, source, clip_start_sec, clip_end_sec):
    """Find local flight discontinuities without interpolating missing balls.

    ``track`` is a tracking/tracks/rally_*.json object. Output times remain on
    the source clock; all output intervals are inside the selected clip. Fixed
    spatial thresholds use image width, never a particular video/date/player.
    Actual frame intervals control continuity and post-contact observation.
    """
    if not (_finite(clip_start_sec) and _finite(clip_end_sec)
            and clip_start_sec < clip_end_sec):
        raise ValueError('replay_motion_invalid_clip')
    output = dict(version=VERSION, time_basis=TIME_BASIS,
                  clip_start_sec=clip_start_sec, clip_end_sec=clip_end_sec,
                  source_sha256=source.get('identity', {}).get('sha256'),
                  status='unavailable', contact_hypotheses=[], gaps=[],
                  observed_spans=[], observed_flights=[], proves_boundary_complete=False,
                  observed_ball_samples=[],
                  observed_ball_sample_columns=['source_sec', 'x_normalized_display', 'y_normalized_display'],
                  coordinate_basis='display_oriented_pixels_after_source_rotation',
                  source_rotation=source.get('rotation', 0))
    if not isinstance(track, dict) or track.get('time_basis') != TIME_BASIS:
        output['reason'] = 'missing_or_unrecognized_source_clock'
        return output
    if track.get('coordinate_interpolation_used') is not False:
        output['reason'] = 'interpolated_or_unknown_coordinates'
        return output
    width, height = source.get('width'), source.get('height')
    if not (_finite(width) and _finite(height) and width > 0 and height > 0):
        output['reason'] = 'missing_image_dimensions'
        return output
    output.update(display_width=width, display_height=height)
    rows, origins = track.get('samples', []), track.get('sample_origins', [])
    if not isinstance(rows, list) or len(rows) != len(origins):
        output['reason'] = 'missing_sample_provenance'
        return output
    samples = []
    prior_time = -math.inf
    for row, origin in zip(rows, origins):
        if (not isinstance(row, (list, tuple)) or len(row) < 3
                or not all(_finite(v) for v in row[:3])):
            output['reason'] = 'invalid_track_sample'
            return output
        time, x, y = row[:3]
        if time <= prior_time:
            output['reason'] = 'nonmonotonic_source_pts'
            return output
        prior_time = time
        # Auxiliary associations and missing detections cannot bridge a gap.
        if (clip_start_sec <= time <= clip_end_sec and origin == 'vball'
                and 0 <= x < width and 0 <= y < height):
            samples.append((time, x / width, y / width))
    # Crop hints use real observations, never contact-fit positions. Normalize
    # the axes separately: fitting uses width units for isotropic motion, while
    # image crops use [0,1] independently on display width and display height.
    output['observed_ball_samples'] = [[t, x, y*width/height] for t,x,y in samples]
    if len(samples) < 12:
        output['reason'] = 'insufficient_observed_ball_samples'
        return output
    points = np.asarray(samples, dtype=float)
    dt = float(np.median(np.diff(points[:, 0])))
    # The detector needs several genuinely observed points on each side. Sparse
    # samples cannot resolve a contact reliably, even if they can draw a curve.
    if dt <= 0 or dt > .065:
        output['reason'] = 'insufficient_tracking_time_resolution'
        return output
    gap_limit = min(.15, max(.075, dt * 3.5))
    breaks = np.flatnonzero(np.diff(points[:, 0]) > gap_limit) + 1
    spans = np.split(points, breaks)
    output['sample_interval_sec'] = dt
    output['observed_spans'] = [[float(p[0, 0]), float(p[-1, 0])] for p in spans]
    all_edges = [clip_start_sec] + [v for span in output['observed_spans'] for v in span] + [clip_end_sec]
    output['gaps'] = [dict(start_sec=float(a), end_sec=float(b), reason='ball_unobserved')
                      for a, b in zip(all_edges[::2], all_edges[1::2]) if b-a > gap_limit]
    candidates = []
    half_window = max(.24, dt * 7.1)
    exclusion = dt * .6
    for span in spans:
        if len(span) < 12:
            continue
        # Centers between observations represent contact intervals, not invented
        # observed frames. In volleyball the ball is often occluded at contact.
        centers = np.sort(np.concatenate((span[:, 0], (span[:-1, 0]+span[1:, 0])/2)))
        for center in centers:
            before = span[(span[:, 0] >= center-half_window) & (span[:, 0] < center-exclusion)]
            after = span[(span[:, 0] > center+exclusion) & (span[:, 0] <= center+half_window)]
            if len(before) < 4 or len(after) < 4:
                continue
            if min(np.ptp(before[:, 0]), np.ptp(after[:, 0])) < .12:
                continue
            local = span[np.abs(span[:, 0]-center) <= half_window]
            fit, residual, outliers = _fit(local, center, kink=True)
            vel_before, vel_after = fit[1], fit[1]+fit[3]
            jump = float(np.linalg.norm(fit[3]))
            # Shared smooth acceleration removes ordinary parabolic apexes.
            # Poor fits are retained as unknown, never interpreted as contact.
            if residual > .0045 or jump < .22:
                continue
            if max(np.linalg.norm(vel_before), np.linalg.norm(vel_after)) > 2.1:
                continue
            candidates.append(dict(
                time_sec=float(center), start_sec=float(before[-1, 0]), end_sec=float(after[0, 0]),
                observed_before_sec=float(before[0, 0]), observed_after_sec=float(after[-1, 0]),
                position_display_xy=(fit[0]*width).tolist(),
                velocity_before_widths_per_sec=vel_before.tolist(),
                velocity_after_widths_per_sec=vel_after.tolist(),
                velocity_jump_widths_per_sec=jump, fit_error_widths=residual,
                fit_downweighted_samples=outliers,
                evidence='observed_ball_velocity_discontinuity',
                interpretation='possible_contact_or_deflection',
                requires_visual_confirmation=True))
    # Adjacent fits are the same impulse, not separate touches. Prioritize the
    # lowest fit residual before impulse strength.
    groups = []
    for candidate in candidates:
        if groups and candidate['time_sec']-groups[-1][-1]['time_sec'] <= .20:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    contacts = []
    for group in groups:
        chosen = min(group, key=lambda c: c['fit_error_widths'])
        chosen['start_sec'] = max(clip_start_sec, min(c['start_sec'] for c in group))
        chosen['end_sec'] = min(clip_end_sec, max(c['end_sec'] for c in group))
        contacts.append(chosen)
    # Compact independent evidence that a cut is inside a still moving observed
    # flight. It helps visual review inspect the right side of the boundary;
    # motion by itself cannot decide whether the narrative already resolved.
    flights = []
    for span in spans:
        for center in span[::max(1, round(.12/dt)), 0]:
            local = span[np.abs(span[:, 0]-center) <= .20]
            if len(local) < 9 or min(center-local[0, 0], local[-1, 0]-center) < .12:
                continue
            fit, residual, _ = _fit(local, center)
            speed = float(np.linalg.norm(fit[1]))
            if residual <= .0035 and .035 <= speed < 2.1:
                flights.append(dict(start_sec=float(local[0, 0]), end_sec=float(local[-1, 0]),
                                    center_sec=float(center), speed_widths_per_sec=speed,
                                    velocity_widths_per_sec=fit[1].tolist(), fit_error_widths=residual,
                                    evidence='observed_moving_ball_without_resolved_impulse'))
    output.update(status='available', reason='independent_trajectory_hypotheses',
                  observed_sample_count=len(points), contact_hypotheses=contacts,
                  observed_flights=flights)
    return output


def spatial_hints(evidence, *, start_sec, end_sec):
    """Return supplementary crop hints around observed balls and lower context.

    Short stable windows follow the caller's candidate interval. Horizontal
    margins preserve nearby ball/player context; the crop extends to the bottom
    of the frame because ball coordinates do not locate the player's body. This
    deliberately broad lower context prevents a high ball from producing a crop
    containing only the ceiling. It is not a person detector or an action claim.

    Samples and boxes are already in display coordinates. ``source_rotation``
    must not be applied again. Unknown or very broad locations return no hint;
    callers still retain their complete panoramic frames.
    """
    if not (_finite(start_sec) and _finite(end_sec) and start_sec < end_sec):
        raise ValueError('replay_spatial_hint_invalid_interval')
    if (evidence.get('coordinate_basis') != 'display_oriented_pixels_after_source_rotation'
            or evidence.get('time_basis') != TIME_BASIS):
        return []
    width, height = evidence.get('display_width'), evidence.get('display_height')
    clip_start, clip_end = evidence.get('clip_start_sec'), evidence.get('clip_end_sec')
    if (not all(_finite(v) for v in (width, height, clip_start, clip_end))
            or width <= 0 or height <= 0 or clip_start >= clip_end):
        return []
    start, end = max(start_sec, clip_start), min(end_sec, clip_end)
    if start >= end:
        return []
    rows = evidence.get('observed_ball_samples', [])
    samples = []
    previous = -math.inf
    for row in rows:
        if (not isinstance(row, (list, tuple)) or len(row) != 3
                or not all(_finite(v) for v in row)):
            return []
        when, x, y = row
        if when <= previous or not 0 <= x < 1 or not 0 <= y < 1:
            return []
        previous = when
        if start <= when <= end:
            samples.append(row)
    if len(samples) < 3:
        return []
    points = np.asarray(samples, dtype=float)
    hints = []
    # A long rally must not collapse to one nearly full-width union box. Each
    # hint uses a bounded, observed neighborhood inside the candidate interval.
    window_count = max(1, math.ceil((end-start)/1.2))
    for left_time, right_time in zip(np.linspace(start, end, window_count+1)[:-1],
                                     np.linspace(start, end, window_count+1)[1:]):
        local = points[(points[:, 0] >= left_time-.25) & (points[:, 0] <= right_time+.25)]
        if len(local) < 3:
            continue
        if (local[0, 0] > left_time+.20 or local[-1, 0] < right_time-.20
                or np.max(np.diff(local[:, 0])) > .35):
            continue
        observed_left = max(float(left_time), float(local[0, 0]))
        observed_right = min(float(right_time), float(local[-1, 0]))
        if observed_right <= observed_left:
            continue
        x0, x1 = float(np.min(local[:, 1])), float(np.max(local[:, 1]))
        # A ball's horizontal extremes are kept, even if a false detection makes
        # the crop too broad. Never discard real-looking endpoints just to zoom.
        box_width = min(1., max(.42, x1-x0+.28))
        if box_width > .82:
            continue
        center = (x0+x1)/2
        left = max(0., min(1.-box_width, center-box_width/2))
        top = max(0., min(.35, float(np.min(local[:, 2]))-.12))
        hints.append(dict(start_sec=observed_left, end_sec=observed_right,
                          bbox=[left, top, left+box_width, 1.],
                          coordinate_space='normalized_display'))
    return hints


def boundary_guard(evidence, review):
    """Request context if a defensive replay ends before the next impulse.

    This is a lower-bound guard, not a substitute for identifying the receiver,
    confirming possession, or showing the attack after a rescued ball. Return
    every later hypothesis so visual review can choose the complete sequence.
    An unsupported tail always remains uncertain, never a successful ending.
    """
    result = dict(verdict='uncertain', need_after=False, reason='motion_evidence_unavailable',
                  proves_boundary_complete=False, following_contact_hypotheses=[])
    if evidence.get('status') != 'available':
        return result
    peak, end = review.get('peak_sec'), review.get('source_end_sec')
    if not (_finite(peak) and _finite(end) and peak < end):
        raise ValueError('replay_motion_invalid_review')
    if not evidence['clip_start_sec'] <= peak < end <= evidence['clip_end_sec']:
        raise ValueError('replay_motion_review_outside_selected_clip')
    # Require separation from the selected impulse's uncertain interval. No
    # arbitrary peak + replay-length rule is used to pick an ending.
    following = [c for c in evidence['contact_hypotheses'] if c['start_sec'] > peak+.30]
    result['following_contact_hypotheses'] = following
    crossing = [flight for flight in evidence.get('observed_flights', [])
                if flight['start_sec'] < end < flight['end_sec']]
    result['observed_motion_at_tail'] = min(crossing, key=lambda f: abs(f['center_sec']-end)) if crossing else None
    if review.get('action') in ('dig', 'receive') and following:
        first = following[0]
        lower_bound = first['observed_after_sec']
        if end < lower_bound:
            result.update(verdict='expand', need_after=True,
                          reason='defensive_replay_cuts_before_next_observed_impulse_and_departure',
                          minimum_source_end_sec=lower_bound,
                          first_following_contact=first)
            return result
    gap = next((gap for gap in evidence['gaps'] if gap['start_sec'] < end
                and gap['end_sec'] > max(peak, end-.25)), None)
    if gap:
        result.update(reason='tail_ball_track_unobserved', gap=gap)
        return result
    if review.get('action') in ('dig', 'receive') and not following:
        result.update(reason='no_observed_following_contact_not_proof_of_dead_ball')
        return result
    result.update(verdict='no_conflict', reason='no_independent_motion_contradiction')
    return result
