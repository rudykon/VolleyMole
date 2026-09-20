"""Select complete action sequences, independently of ranking preview frames."""
import math

ACTION_PRIORITY = {'spike': 3, 'block': 2, 'dig': 2, 'receive': 1}
REVIEW_METHODS = ('assistant_visual_review', 'human_visual_review', 'vision_sequence_review')


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def reviewed_window(item, rally, source):
    review = rally['replay_review']
    if not isinstance(review, dict):
        raise ValueError('回放复核记录必须是对象')
    if review.get('source_sha256') != source.get('identity', {}).get('sha256') or not review.get('source_sha256'):
        raise ValueError('回放复核与原片身份不一致')
    if review.get('method') not in REVIEW_METHODS or not isinstance(review.get('reason'), str) or not review['reason'].strip():
        raise ValueError('回放缺少时序复核方法或理由')
    if review.get('status') == 'omit':
        return None
    if review.get('status') != 'approved' or review.get('boundary_complete') is not True:
        raise ValueError('回放动作与结果尚未完整复核')
    keys = ('source_start_sec', 'action_start_sec', 'peak_sec', 'action_end_sec', 'source_end_sec')
    if not all(finite(review.get(k)) for k in keys):
        raise ValueError('回放复核时间必须为有限数值')
    a, action_start, peak, action_end, b = (review[k] for k in keys)
    if not item['clip_start_sec'] <= a <= action_start < peak < action_end <= b <= item['clip_end_sec']:
        raise ValueError('回放裁断动作准备或结果，或越出已选回合')
    if review.get('action') not in ACTION_PRIORITY:
        raise ValueError('精彩回放应选择进攻、拦网或防守动作')
    return dict(source_start_sec=a, source_end_sec=b, peak_sec=peak,
                action_start_sec=action_start, action_end_sec=action_end,
                replay_action=review['action'], replay_selection='reviewed_action_sequence',
                peak_evidence=review['method'], visual_review_used=True,
                boundary_complete=True, review_reason=review['reason'],
                peak_selection_uncertainty='Sequence reviewed; action timing is approximate, no scoring outcome inferred.')


def replay_window(item, rally, source):
    """Reviewed windows win; sets and arbitrary midpoints never create replays.

    Local detections remain hypotheses. Their windows retain preparation and
    aftermath and are explicitly marked unreviewed. A clipped action is omitted.
    """
    if 'replay_review' in rally:
        return reviewed_window(item, rally, source)
    start, end = item['clip_start_sec'], item['clip_end_sec']
    candidates = []
    for event in rally.get('action_events', []):
        action = event.get('action')
        a, b = event.get('start_sec'), event.get('end_sec')
        if action not in ACTION_PRIORITY or not finite(a) or not finite(b) or not start <= a <= b <= end:
            continue
        support = event.get('detection_frames', 0)
        if type(support) is not int or support < 2:
            continue
        left, right = max(start, a - 1.5), min(end, b + 2.0)
        if a-left < .75 or right-b < 1.25:
            continue
        candidates.append((ACTION_PRIORITY[action], support, b-a, a, left, right, event))
    if candidates:
        *_, left, right, event = max(candidates, key=lambda x: x[:4])
        return dict(source_start_sec=left, source_end_sec=right,
                    peak_sec=(event['start_sec']+event['end_sec'])/2,
                    action_start_sec=event['start_sec'], action_end_sec=event['end_sec'],
                    replay_action=event['action'], replay_selection='detected_action_with_context',
                    peak_evidence='local_action_event', visual_review_used=False,
                    boundary_complete=None,
                    peak_selection_uncertainty='Local action hypothesis; context retained, outcome completeness not visually confirmed.')
    # Reviewed short narrative events retain their whole setup and aftermath.
    # Long or uncertain events need a separate action review instead of a slice
    # centered on their peak timestamp.
    if (rally.get('clip_kind') == 'event' and rally.get('boundary_complete') is True
            and finite(rally.get('peak_sec')) and end-start <= 10):
        return dict(source_start_sec=start, source_end_sec=end, peak_sec=rally['peak_sec'],
                    action_start_sec=rally['start_sec'], action_end_sec=rally['end_sec'],
                    replay_action='event', replay_selection='complete_short_event',
                    peak_evidence=rally.get('peak_evidence', 'event_evidence'),
                    visual_review_used=False, boundary_complete=True,
                    peak_selection_uncertainty='Complete event context; replay priority not independently reviewed.')
    return None
