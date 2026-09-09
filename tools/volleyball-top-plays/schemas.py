"""Decision contract; reject unsafe or incomplete edits before touching the renderer."""
import math
from pathlib import Path

FIELDS = {'rank': {'type': 'integer'}, 'rally_id': {'type': 'string'},
          'clip_start_sec': {'type': 'number'}, 'clip_end_sec': {'type': 'number'},
          'title': {'type': 'string'}, 'reason': {'type': 'string'}, 'confidence': {'type': 'number'}}
DECISION_SCHEMA = {'type': 'object', 'additionalProperties': False,
                   'properties': {'title': {'type': 'string'}, 'selected': {'type': 'array', 'items': {
                       'type': 'object', 'additionalProperties': False, 'properties': FIELDS, 'required': list(FIELDS)}}},
                   'required': ['title', 'selected']}


def local_file(root, name):
    root = Path(root).resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'缺失文件或越界路径：{name}')
    return path


def validate_decision(decision, manifest, directory, top_k):
    if not isinstance(decision, dict) or not isinstance(decision.get('title'), str) or not decision['title'].strip():
        raise ValueError('无效合集标题')
    selected = decision.get('selected')
    if not isinstance(selected, list) or len(selected) != top_k:
        raise ValueError(f'剪辑单必须恰好包含 {top_k} 个不同回合')
    known = {r['rally_id']: r for r in manifest['rallies'] if r['eligible']}
    seen, ranges = set(), []
    for rank, item in enumerate(selected, 1):
        if not isinstance(item, dict) or set(item) != set(FIELDS):
            raise ValueError('剪辑项必须严格符合 schema')
        if type(item['rank']) is not int or item['rank'] != rank or item['rally_id'] in seen or item['rally_id'] not in known:
            raise ValueError('无效排名、重复回合或未知 rally_id')
        seen.add(item['rally_id'])
        rally = known[item['rally_id']]
        for key in ('clip_start_sec', 'clip_end_sec', 'confidence'):
            if type(item[key]) not in (float, int) or not math.isfinite(item[key]):
                raise ValueError('剪辑参数必须是有限数值')
        a, b = item['clip_start_sec'], item['clip_end_sec']
        # Preserve the entire detected rally; the model may adjust context only.
        if not (rally['safe_start_sec'] <= a <= rally['start_sec'] < rally['end_sec'] <= b <= rally['safe_end_sec']):
            raise ValueError('模型裁断回合或剪辑时间超出安全范围')
        if not (0 <= a < b <= manifest['source']['duration_sec']) or not 0 <= item['confidence'] <= 1:
            raise ValueError('剪辑时间或置信度越界')
        if any(max(a, c) < min(b, d) - .001 for c, d in ranges):
            raise ValueError('入选片段时间重叠')
        ranges.append((a, b))
        for key, limit in [('title', 40), ('reason', 500)]:
            if not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > limit or any(ord(c)<32 for c in item[key]):
                raise ValueError(f'无效 {key}')
        local_file(directory, rally['tracking_json'])
        if len(rally['preview_frames']) != 3:
            raise ValueError('每回合必须有三张关键帧')
        for name in rally['preview_frames']:
            local_file(directory, name)
    return decision
