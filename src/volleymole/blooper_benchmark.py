"""Blind human blooper evaluation. Preparation never creates human judgments.

The independent human overall ratings are the reference; event-model dimensions
and scores are deliberately not imported. Null means unknown, never grade zero.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

VERSION = 'volleymole-bloopers-human-v1'
DIMENSIONS = ('overall_fun', 'selection_suitability', 'unexpected_contrast',
              'related_laughter', 'narrative', 'player_reaction')
FLAGS = ('event_visible', 'boundary_complete', 'injury_suspected',
         'safe_to_include', 'laughter_linked', 'ordinary_error', 'unrelated_laughter')
STAGES = ('setup', 'unexpected', 'reaction')


def _number(value, low=0, high=1e9):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _id(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 200


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(data, seed=20260912, raters_per_condition=3):
    """Return (private mapping, independent empty assignment forms).

    Input: {matches:[{match_id, split, sources:[{source_id,path,duration_sec,
    audio_available}], events:[{event_id,source_id,clip_start_sec,clip_end_sec}]}]}.
    Real local media and explicit source audio availability are required. The
    clip interval is a sampling window, not a claimed human event annotation.
    """
    if type(seed) is not int or type(raters_per_condition) is not int or raters_per_condition < 3:
        raise ValueError('需要整数种子和每种条件至少 3 名独立评员')
    if not isinstance(data, dict) or not isinstance(data.get('matches'), list) or not data['matches']:
        raise ValueError('输入必须包含非空 matches 数组')
    matches, items, seen_matches, source_splits = [], [], set(), {}
    for match in data['matches']:
        mid, split = match.get('match_id'), match.get('split')
        if not _id(mid) or mid in seen_matches or split not in ('train', 'dev', 'test'):
            raise ValueError('比赛 ID 必须唯一且 split 为 train/dev/test；同场不能跨分区')
        seen_matches.add(mid)
        sources = {}
        for source in match.get('sources', []):
            sid = source.get('source_id')
            path = Path(source['path']).resolve()
            duration = source.get('duration_sec')
            if not _id(sid) or sid in sources or not path.is_file() or not _number(duration, .001):
                raise ValueError('素材需要唯一 source_id、现有本地文件及有效时长')
            if type(source.get('audio_available')) is not bool:
                raise ValueError('必须明确原素材是否有音轨')
            sha = _sha_file(path)
            if sha in source_splits and source_splits[sha] != (mid, split):
                raise ValueError('相同原素材不能伪装成另一比赛或跨分区')
            source_splits[sha] = (mid, split)
            sources[sid] = {'source_id': sid, 'path': str(path), 'duration_sec': duration,
                            'audio_available': source['audio_available'], 'sha256': sha}
        if not sources:
            raise ValueError('每场需要至少一个原素材')
        seen_events, seen_intervals = set(), set()
        for event in match.get('events', []):
            eid, sid = event.get('event_id'), event.get('source_id', 'single')
            start = event.get('clip_start_sec', event.get('start_sec'))
            end = event.get('clip_end_sec', event.get('end_sec'))
            if not _id(eid) or eid in seen_events or sid not in sources:
                raise ValueError('事件需要场内唯一 ID 和有效 source_id')
            duration = sources[sid]['duration_sec']
            if not _number(start, 0, duration) or not _number(end, 0, duration) or start >= end:
                raise ValueError('素材区间越界或为空')
            interval_key = (sources[sid]['sha256'], start, end)
            if interval_key in seen_intervals:
                raise ValueError('相同区间不能重复当作独立事件')
            seen_events.add(eid)
            seen_intervals.add(interval_key)
            item_id = _digest([VERSION, seed, mid, eid])[:20]
            items.append({'item_id': item_id, 'match_id': mid, 'split': split, 'event_id': eid,
                          'source_id': sid, 'source_path': sources[sid]['path'],
                          'source_sha256': sources[sid]['sha256'], 'clip_start_sec': start,
                          'clip_end_sec': end, 'duration_sec': end-start,
                          'source_has_audio': sources[sid]['audio_available'],
                          'sampling_origin': event.get('sampling_origin', 'unspecified')})
        matches.append({'match_id': mid, 'split': split,
                        'pool_scope': match.get('pool_scope', 'candidate_pool_only'),
                        'sources': list(sources.values())})
    if not items:
        raise ValueError('评审池不能为空')
    private = {'schema_version': VERSION, 'seed': seed,
               'raters_per_condition': raters_per_condition, 'matches': matches, 'items': items}
    private['benchmark_id'] = _digest(private)
    forms = []
    for condition in ('audio', 'silent'):
        for index in range(raters_per_condition):
            assignment = f'{condition}_{index+1:02d}'
            rows = []
            for item in items:
                if condition == 'audio' and not item['source_has_audio']:
                    continue
                rows.append({'item_id': item['item_id'],
                             'media': f"media/{item['item_id']}_{condition}.mp4",
                             'duration_sec': item['duration_sec'], 'completed': False,
                             'ratings': dict.fromkeys(DIMENSIONS), 'flags': dict.fromkeys(FLAGS),
                             'stages': {stage: {'start_sec': None, 'end_sec': None} for stage in STAGES},
                             'observed_facts': None, 'unknown_reason': None})
            random.Random(f'{seed}/{assignment}').shuffle(rows)
            if rows:
                forms.append({'schema_version': VERSION, 'benchmark_id': private['benchmark_id'],
                              'assignment_id': assignment, 'condition': condition,
                              'human_provenance': {'annotator_id': None, 'human_only': False},
                              'items': rows})
    # Preserve exact assignment membership/order separately from editable answers.
    private['assignments'] = [{'assignment_id': f['assignment_id'], 'condition': f['condition'],
                               'item_ids': [r['item_id'] for r in f['items']]} for f in forms]
    return private, forms


def validate_forms(private, forms):
    """Validate blinded assignments and declared human-only, independent ratings.

    Blank drafts are accepted for counting but never contribute to any metric.
    Declaration validation cannot independently verify annotators' identities.
    """
    expected = {a['assignment_id']: a for a in private['assignments']}
    items = {i['item_id']: i for i in private['items']}
    seen_assignments, seen_person_items, records = set(), set(), []
    person_conditions = {}
    for form in forms:
        aid = form.get('assignment_id')
        if form.get('schema_version') != VERSION or form.get('benchmark_id') != private['benchmark_id']:
            raise ValueError('评分文件不属于该版本的盲评包')
        if aid not in expected or aid in seen_assignments or form.get('condition') != expected[aid]['condition']:
            raise ValueError('评员分配不正确或重复导入')
        seen_assignments.add(aid)
        rows = form.get('items', [])
        if [r.get('item_id') for r in rows] != expected[aid]['item_ids']:
            raise ValueError('不能增删或重排已随机分配的评分项')
        provenance = form.get('human_provenance', {})
        for row in rows:
            item, condition = items[row['item_id']], form['condition']
            if row.get('media') != f"media/{item['item_id']}_{condition}.mp4" or row.get('duration_sec') != item['duration_sec']:
                raise ValueError('不能更换盲评素材或时长')
            if type(row.get('completed')) is not bool:
                raise ValueError('completed 必须明确为布尔值')
            for group, keys in (('ratings', DIMENSIONS), ('flags', FLAGS), ('stages', STAGES)):
                if not isinstance(row.get(group), dict) or set(row[group]) != set(keys):
                    raise ValueError(f'无效评分字段 {group}')
            for value in row['ratings'].values():
                if value is not None and (type(value) is not int or not 0 <= value <= 4):
                    raise ValueError('序数评分必须为 0–4 整数或 null')
            for value in row['flags'].values():
                if value is not None and type(value) is not bool:
                    raise ValueError('观察状态必须为 true/false/null')
            known_stages = []
            for stage in STAGES:
                bounds = row['stages'][stage]
                if not isinstance(bounds, dict) or set(bounds) != {'start_sec', 'end_sec'}:
                    raise ValueError('过程必须具有起止时间')
                start, end = bounds['start_sec'], bounds['end_sec']
                if (start is None) != (end is None):
                    raise ValueError('每个过程的边界应一起填写或一起标为未知')
                if start is not None:
                    if not _number(start, 0, item['duration_sec']) or not _number(end, start, item['duration_sec']):
                        raise ValueError('过程时间必须位于盲评片段内，使用从零开始的秒数')
                    known_stages.append((start, end))
            # Reactions can overlap the unexpected action but cannot precede setup.
            if any(a[0] > b[0] for a, b in zip(known_stages, known_stages[1:])):
                raise ValueError('铺垫、意外、反应的起点顺序不成立')
            if condition == 'silent' and (row['ratings']['related_laughter'] is not None or
                    any(row['flags'][k] is not None for k in ('laughter_linked', 'unrelated_laughter'))):
                raise ValueError('静音条件的笑声及关联必须记为未知')
            for key in ('observed_facts', 'unknown_reason'):
                if row.get(key) is not None and not isinstance(row[key], str):
                    raise ValueError('事实与未知理由须为文字或 null')
            if not row['completed']:
                continue
            person = provenance.get('annotator_id')
            if not _id(person) or provenance.get('human_only') is not True:
                raise ValueError('已完成评分必须声明独立真人填写，不能用模型输出冒充人工标签')
            if person in person_conditions and person_conditions[person] != condition:
                raise ValueError('同一真人不能参加声音与静音两个评员组')
            person_conditions[person] = condition
            if (person, item['item_id']) in seen_person_items:
                raise ValueError('同一真人不能重复评价同一素材，包括声音/静音对照')
            seen_person_items.add((person, item['item_id']))
            if not (row.get('observed_facts') or '').strip():
                raise ValueError('已完成评分必须写明直接观察，不能只填分数')
            if (any(v is None for v in row['ratings'].values()) or
                    any(v is None for v in row['flags'].values()) or
                    any(s['start_sec'] is None for s in row['stages'].values())) and not (row.get('unknown_reason') or '').strip():
                raise ValueError('未知判断须说明证据缺失原因')
            records.append({**row, 'annotator_id': person, 'condition': condition,
                            'match_id': item['match_id'], 'split': item['split']})
    return records


def ordinal_alpha(units):
    """Krippendorff ordinal alpha from pairable units and empirical midranks.

    Coincidences weight each unit by 1/(m-1). Missing values and units with fewer
    than two ratings do not enter observed or expected disagreement.
    """
    coincidence = np.zeros((5, 5), dtype=float)
    pairable = 0
    for values in units:
        values = [v for v in values if v is not None]
        if any(type(v) is not int or not 0 <= v <= 4 for v in values):
            raise ValueError('alpha 输入必须为 0–4 整数或 null')
        if len(values) < 2:
            continue
        pairable += 1
        counts = np.bincount(values, minlength=5)
        coincidence += (np.outer(counts, counts)-np.diag(counts))/(len(values)-1)
    marginals = coincidence.sum(axis=0)
    n = marginals.sum()
    result = {'alpha': None, 'pairable_items': pairable, 'pairable_ratings': int(round(n)),
              'status': 'insufficient_pairable_ratings'}
    if n < 2:
        return result
    midranks = np.cumsum(marginals)-marginals/2
    distance = (midranks[:, None]-midranks[None, :])**2
    observed = float(np.sum(coincidence*distance))
    expected = float(np.sum(np.outer(marginals, marginals)*distance)/(n-1))
    if expected == 0:
        return {**result, 'status': 'undefined_no_category_variation'}
    return {**result, 'alpha': 1-observed/expected, 'status': 'ok'}


def cluster_mean_interval(values_by_match, seed=20260912, samples=2000):
    """Equal-match macro mean; resample whole matches, never individual clips.

    Ten matches is this protocol's minimum for descriptive percentile intervals,
    not a statistical guarantee. With fewer matches return the mean only.
    """
    if type(samples) is not int or samples < 200:
        raise ValueError('bootstrap 至少 200 次')
    values = [v for v in values_by_match.values() if v is not None]
    if any(not _number(v, -1e9, 1e9) for v in values):
        raise ValueError('比赛指标必须为有限数或未知')
    result = {'mean': statistics.mean(values) if values else None, 'matches': len(values),
              'missing_matches': len(values_by_match)-len(values), 'ci95': None,
              'resampling_unit': 'match', 'bootstrap_samples': samples,
              'status': 'insufficient_matches_for_interval'}
    if len(values) >= 10:
        rng = np.random.default_rng(seed)
        draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
        result.update(ci95=np.quantile(draws, [.025, .975]).tolist(), status='descriptive_interval')
    return result


def _consensus(values, minimum=3):
    known = [v for v in values if v is not None]
    return statistics.median(known) if len(known) >= minimum else None


def _dcg(grades):
    # Linear gains avoid silently making an ordinal scale exponential utility.
    return sum(g/math.log2(index+2) for index, g in enumerate(grades))


def evaluate(private, forms, predictions=None, split='test', seed=20260912, bootstrap_samples=2000):
    """Report independent human ratings and model ranking within a judged pool.

    Predictions: {systems:[{system_id,matches:[{match_id,ranked_event_ids:[...]}]}]}.
    Every split match must be explicit, including empty outputs. Out-of-pool and
    duplicate events are errors, not implicit negatives. Candidate recall and
    factual explanation accuracy require separate exhaustive/factual references.
    """
    if split not in ('train', 'dev', 'test'):
        raise ValueError('未知分区')
    records = validate_forms(private, forms)
    items = [i for i in private['items'] if i['split'] == split]
    grouped = defaultdict(list)
    for row in records:
        if row['split'] == split:
            grouped[(row['condition'], row['item_id'])].append(row)
    judgments, agreement = [], {}
    for condition in ('audio', 'silent'):
        pool = [i for i in items if condition == 'silent' or i['source_has_audio']]
        agreement[condition] = {}
        for dimension in DIMENSIONS:
            units = [[r['ratings'][dimension] for r in grouped[(condition, i['item_id'])]] for i in pool]
            agreement[condition][dimension] = {**ordinal_alpha(units),
                'assigned_items': len(pool),
                'items_with_three_known_ratings': sum(sum(v is not None for v in unit) >= 3 for unit in units)}
        for item in pool:
            rows = grouped[(condition, item['item_id'])]
            grades = {d: _consensus([r['ratings'][d] for r in rows]) for d in DIMENSIONS}
            flag_consensus = {}
            for flag in FLAGS:
                votes = [r['flags'][flag] for r in rows if r['flags'][flag] is not None]
                counts = Counter(votes)
                # No consensus from only two known votes or an even tie.
                flag_consensus[flag] = counts[True] > counts[False] if len(votes) >= 3 and counts[True] != counts[False] else None
            disagreements = [d for d in DIMENSIONS
                             if len(v := [r['ratings'][d] for r in rows if r['ratings'][d] is not None]) >= 2
                             and max(v)-min(v) >= 2]
            contested_flags = [f for f in FLAGS if {r['flags'][f] for r in rows} >= {True, False}]
            judgments.append({'item_id': item['item_id'], 'event_id': item['event_id'],
                              'match_id': item['match_id'], 'condition': condition,
                              'completed_raters': len(rows), 'ratings_median': grades,
                              'known_ratings': {d: sum(r['ratings'][d] is not None for r in rows) for d in DIMENSIONS},
                              'flags_majority': flag_consensus,
                              'disputed_dimensions': disagreements, 'disputed_flags': contested_flags,
                              'needs_adjudication': bool(disagreements or contested_flags)})
    results = []
    # Preserve declared matches even when candidate discovery produced no items.
    # An empty review pool has no human reference; it is not a missing match.
    declared = {m['match_id']: m for m in private['matches'] if m['split'] == split}
    split_matches = set(declared)
    matches_with_candidates = {i['match_id'] for i in items}
    seen_systems = set()
    for system in (predictions or {}).get('systems', []):
        name = system.get('system_id')
        if not _id(name) or name in seen_systems:
            raise ValueError('模型 system_id 必须非空且唯一')
        seen_systems.add(name)
        rows = system.get('matches', [])
        mids = [r.get('match_id') for r in rows]
        if len(mids) != len(set(mids)) or set(mids) != split_matches:
            raise ValueError('模型必须显式提供所评估分区每场的结果；缺失结果不可静默排除')
        for predicted in rows:
            mid = predicted['match_id']
            ranking = predicted.get('ranked_event_ids')
            valid = {i['event_id'] for i in items if i['match_id'] == mid}
            if not isinstance(ranking, list) or any(not isinstance(e, str) for e in ranking) or len(ranking) != len(set(ranking)) or not set(ranking) <= valid:
                raise ValueError('榜单包含重复事件或未加入盲评池的事件')
            selected = ranking[:5]
            for condition in ('audio', 'silent'):
                judged = {j['event_id']: j for j in judgments if j['match_id'] == mid and j['condition'] == condition}
                # Native-silent matches have no audio condition. Empty candidate
                # pools still produce explicit rows for applicable conditions.
                if condition == 'audio' and not any(s['audio_available'] for s in declared[mid]['sources']):
                    continue
                observed = [judged[e]['ratings_median']['selection_suitability'] if e in judged else None for e in selected]
                fun = [judged[e]['ratings_median']['overall_fun'] if e in judged else None for e in selected]
                safe = [judged[e]['flags_majority']['safe_to_include'] if e in judged else None for e in selected]
                pool_grades = [j['ratings_median']['selection_suitability'] for j in judged.values()]
                all_selected = bool(selected) and all(v is not None for v in observed)
                all_pool = len(judged) == len(valid) and all(g is not None for g in pool_grades)
                ideal = _dcg(sorted(pool_grades, reverse=True)[:5]) if all_pool else None
                ndcg = _dcg(observed)/ideal if all_pool and all(v is not None for v in observed) and ideal else None
                results.append({'system_id': name, 'match_id': mid, 'condition': condition,
                    'selected_count': len(selected), 'pool_count': len(valid),
                    'condition_pool_count': len(judged),
                    'pool_with_reference_count': sum(g is not None for g in pool_grades),
                    'judged_selected_count': sum(v is not None for v in observed),
                    'top5_suitability': statistics.mean(observed) if all_selected else None,
                    'top5_fun': statistics.mean(fun) if fun and all(v is not None for v in fun) else None,
                    'suitable_fraction_returned': sum(v >= 3 for v in observed)/len(observed) if all_selected else None,
                    'unsafe_fraction_returned': sum(v is False for v in safe)/len(safe) if safe and all(v is not None for v in safe) else None,
                    'ndcg5_judged_pool': ndcg,
                    'ndcg_status': 'empty_candidate_pool' if not valid else
                        ('ok' if ndcg is not None else ('all_zero_reference' if ideal == 0 else 'incomplete_reference')),
                    'missing_slots': 5-len(selected)})
    summaries = []
    for name in sorted(seen_systems):
        for condition in ('audio', 'silent'):
            group = [r for r in results if r['system_id'] == name and r['condition'] == condition]
            if group:
                summaries.append({'system_id': name, 'condition': condition, 'metrics': {
                    k: cluster_mean_interval({r['match_id']: r[k] for r in group}, seed, bootstrap_samples)
                    for k in ('top5_suitability', 'top5_fun', 'suitable_fraction_returned',
                              'unsafe_fraction_returned', 'ndcg5_judged_pool')}})
    return {'schema_version': VERSION, 'benchmark_id': private['benchmark_id'], 'split': split,
            'human_completed_rows': sum(len(v) for v in grouped.values()), 'items': len(items),
            'matches': len(split_matches), 'agreement': agreement, 'judgments': judgments,
            'match_coverage': {'declared_matches': len(split_matches),
                'matches_with_candidates': len(matches_with_candidates),
                'matches_without_candidates': len(split_matches-matches_with_candidates),
                'empty_pool_match_ids': sorted(split_matches-matches_with_candidates)},
            'systems_by_match': results, 'system_summaries': summaries,
            'unsupported': ['full_match_candidate_recall_without_exhaustive_human_events',
                            'factual_explanation_accuracy', 'adjudication_overwrites',
                            'population_claims_from_small_or_model_only_pools'],
            'reference_policy': 'Only declared independent human ratings; unknown is never zero. '
                                'Median requires 3 known ratings. Model dimensions never define reference grades.'}
