"""Fuse timestamped state, action and ball evidence into disjoint rally candidates."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from common import read_json, save_json


def grouped(indices, times, gap):
    groups = []
    for i in indices:
        if groups and times[i] - times[groups[-1][-1]] <= gap + 1e-6:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def build_manifest(source, analytics_path, ball_path, pts_path, player_path, directory, config, provenance):
    directory = Path(directory)
    with Path(analytics_path).open() as stream:
        records = [json.loads(line) for line in stream]
    with Path(ball_path).open() as stream:
        balls = list(csv.DictReader(stream))
    pts = np.loadtxt(pts_path, ndmin=1)
    if len(records) != len(balls) or len(balls) != len(pts) or len(pts) < 2:
        raise ValueError('分析帧数、轨迹帧数与源时间戳数量不一致')
    if source['frame_count'] and len(pts) != source['frame_count']:
        raise ValueError('分析没有覆盖完整视频')
    if not np.all(np.isfinite(pts)) or np.any(np.diff(pts) <= 0):
        raise ValueError('源显示时间戳不是严格递增序列')
    # Time zero is the container start, including the audio/video start offset.
    t = pts - source['start_sec']
    if t[0] < -.001 or t[-1] > source['duration_sec'] or source['duration_sec']-t[-1] > .25:
        raise ValueError('时间戳与源视频首尾不匹配')
    for i, (r, b) in enumerate(zip(records, balls)):
        if r['frame'] != i or int(b['Frame']) != i:
            raise ValueError(f'帧号缺失或重复：{i}')
        if abs(r.get('source_time_s', r['time_s'] + pts[0]) - pts[i]) > .002:
            raise ValueError(f'比赛分析时间轴错位：帧 {i}')
    xy = np.array([[float(b['X']), float(b['Y'])] for b in balls])
    visible = np.array([int(b['Visibility']) == 1 for b in balls])
    w, h = source['width'], source['height']
    visible &= np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    dt = np.diff(t, prepend=t[0]-.0333)
    distance = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=1) / w
    speed = distance / dt
    continuous = visible & np.roll(visible, 1) & (dt < .15) & (speed < 2.1)
    continuous[0] = False
    heads, counts = [], []
    for row in records:
        people = [p['xyxy'] for p in row.get('players', []) if p.get('confidence', 0) >= .4]
        counts.append(len(people))
        heads.append(float(np.median([p[1]+.15*(p[3]-p[1]) for p in people])) if people else .4*h)
    airborne = continuous & (speed > .03) & (xy[:, 1] < heads) & (xy[:, 0] > .02*w) & (xy[:, 0] < .98*w)
    play = np.array([r['state'].lower() in ('play', 'service') for r in records])
    step = config['bin_sec']
    edges = np.arange(0, source['duration_sec']+step, step)
    lo = np.searchsorted(t, edges[:-1])
    hi = np.searchsorted(t, edges[1:])
    bins = []
    for a, b, start in zip(lo, hi, edges):
        bins.append({'start_sec': float(start), 'end_sec': min(float(start+step), source['duration_sec']),
                     'flight_ratio': float(airborne[a:b].mean()) if b>a else 0,
                     'play_ratio': float(play[a:b].mean()) if b>a else 0})
    active = np.array([b['flight_ratio'] >= config['flight_ratio'] for b in bins])
    # Strong visual motion bridges transient NO_PLAY. PLAY alone cannot turn
    # stationary/held-ball detections into a rally. Retain low-state candidates.
    groups = grouped(np.flatnonzero(active), edges, config['bridge_gap_sec']+step)
    # Also retain state-led short/low-trajectory candidates for an explicit audit.
    covered = np.zeros(len(active), dtype=bool)
    for g in groups:
        covered[g[0]:g[-1]+1] = True
    state_extra = np.array([b['play_ratio'] >= .5 for b in bins]) & ~covered
    groups += grouped(np.flatnonzero(state_extra), edges, step+1e-5)
    groups.sort(key=lambda g: g[0])
    player = read_json(player_path)
    rallies, artifacts = [], []
    counts = np.array(counts)
    max_count = max(1, float(np.percentile(counts, 85)))
    for n, g in enumerate(groups, 1):
        a, b = float(edges[g[0]]), min(float(edges[g[-1]+1]), source['duration_sec'])
        s, e = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
        if e <= s:
            continue
        good = np.flatnonzero(continuous[s:e]) + s
        flight = float(airborne[s:e].mean())
        coverage = float(continuous[s:e].mean())
        events = []
        for action in ('receive', 'set', 'spike', 'block'):
            hits = [i for i in range(s,e) if any(d['class'] == action and d['confidence'] >= config['action_confidence'] for d in records[i].get('actions', []))]
            for event in grouped(hits, t, .65):
                if len(event) >= 2:
                    events.append({'action': action, 'start_sec': round(float(t[event[0]]), 4),
                                   'end_sec': round(float(t[event[-1]]), 4), 'detection_frames': len(event)})
        # Count robust vertical direction reversals, spaced >= 0.35 sec.
        turns, last_turn = [], -10.
        for i in range(s+3, e-3):
            if not continuous[i-3:i+4].all() or not airborne[i-3:i+4].any():
                continue
            v1, v2 = xy[i,1]-xy[i-3,1], xy[i+3,1]-xy[i,1]
            if v1*v2 < 0 and min(abs(v1),abs(v2)) > .004*h and t[i]-last_turn > .35:
                turns.append(float(t[i])); last_turn = t[i]
        participating = float(np.median(counts[s:e])) / max_count
        player_samples = [d for d in player.get('detections', []) if a <= d['time_sec'] < b and d['confidence'] >= config['player_confidence']]
        hit_times = {round(d['time_sec'], 3) for d in player_samples}
        focus_ratio = min(1., len(hit_times) * player.get('sample_interval_sec', 1.)/(b-a)) if len(hit_times) >= 2 else 0.
        features = {'duration': min((b-a)/28, 1), 'flight': min(flight/.55, 1),
                    'turns': min(len(turns)/12, 1), 'actions': min(len(events)/5,1),
                    'coverage': coverage, 'participation': min(participating, 1), 'focus': focus_ratio}
        parts = {key: round(config['weights'][key]*v, 3) for key,v in features.items()}
        # Sparse-player warmup is still audited, with reduced ranking priority.
        quality = min(1., max(.2, participating/.85)**3)
        score = round(sum(parts.values())*quality, 2)
        reasons = []
        if b-a < config['min_rally_sec']: reasons.append('回合过短')
        if b-a > config['max_rally_sec']: reasons.append('过长连续片段，回合边界不确定')
        if coverage < config['min_visible_ratio']: reasons.append('有效球轨迹不足')
        if flight < config['min_flight_ratio']: reasons.append('空中运动不足，疑似持球或停顿')
        if not len(good): reasons.append('没有可关联的有效轨迹')
        rid = f'rally_{n:04d}'
        track_name = f'tracking/tracks/{rid}.json'
        track = {'rally_id': rid, 'start_frame': s, 'last_frame': e-1,
                 'start_sec': a, 'end_sec': b, 'time_basis': 'source presentation timestamp minus container start',
                 'positions': [[[float(xy[i,0]), float(xy[i,1])], int(i)] for i in good],
                 'samples': [[round(float(t[i]),6), float(xy[i,0]), float(xy[i,1]), bool(airborne[i])] for i in good]}
        save_json(directory/track_name, track); artifacts.append(directory/track_name)
        peak = max(events, key=lambda d: {'spike':4,'block':3,'receive':2,'set':1}[d['action']])['start_sec'] if events else (a+b)/2
        rallies.append({'rally_id': rid, 'start_sec': a, 'end_sec': b, 'duration_sec': round(b-a,3),
                        'tracking_json': track_name, 'actions': sorted({d['action'] for d in events}), 'action_events': events,
                        'players': [{'number': player['number'], 'presence_ratio': round(focus_ratio,3), 'confirmed_samples': len(hit_times)}] if player.get('number') is not None else [],
                        'ball_metrics': {'visible_ratio': round(coverage,4), 'flight_ratio': round(flight,4), 'trajectory_changes': len(turns)},
                        'state_metrics': dict(Counter(r['state'] for r in records[s:e])),
                        'median_player_count': float(np.median(counts[s:e])), 'rule_score': score, 'score_components': parts,
                        'eligible': not reasons, 'exclusion_reasons': reasons, 'preview_times_sec': [a, min(b-.1,max(a,peak)), max(a,b-.1)],
                        'preview_frames': [f'previews/{rid}_{label}.jpg' for label in ('start','peak','end')],
                        'evidence': {'analytics_frames': [s,e-1], 'analytics_jsonl': 'analytics/detections.jsonl', 'ball_csv': 'tracking/ball.csv',
                                     'state_agreement': round(float(play[s:e].mean()),4), 'boundary_method': 'airborne_motion_with_state_audit'}})
    # Context may extend to the midpoint of an adjacent candidate, never overlap.
    for i, r in enumerate(rallies):
        # Rejected state flickers in pre-serve setup must not consume the
        # context buffer of a valid rally. Only playable neighbours constrain it.
        previous = next((q for q in reversed(rallies[:i]) if q['eligible']),None)
        following = next((q for q in rallies[i+1:] if q['eligible']),None)
        before = (previous['end_sec']+r['start_sec'])/2 if previous else 0
        after = (r['end_sec']+following['start_sec'])/2 if following else source['duration_sec']
        r['safe_start_sec'] = max(0., before, r['start_sec']-config['padding_before_sec'])
        r['safe_end_sec'] = min(source['duration_sec'], after, r['end_sec']+config['padding_after_sec'])
    manifest = {'schema_version': 1, 'project': 'VolleyMole', 'source': source, 'config': config,
                'provenance': provenance, 'rallies': rallies, 'diagnostics': {
                    'input_frames': len(t), 'time_alignment_max_error_sec': max(abs(r.get('source_time_s',r['time_s']+pts[0])-p) for r,p in zip(records,pts)),
                    'candidate_count': len(rallies), 'eligible_count': sum(r['eligible'] for r in rallies),
                    'note': '自动候选不是人工真值；状态模型短暂跳变由有效球运动补足。动作事件为合并后的检测证据，不保证真实动作类别。'}}
    save_json(directory/'analytics/activity_bins.json', bins)
    save_json(directory/'match_manifest.json', manifest)
    return manifest, artifacts + [directory/'analytics/activity_bins.json', directory/'match_manifest.json']
