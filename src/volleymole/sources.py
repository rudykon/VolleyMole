"""Resolve source-local clocks; different sets may have identical timestamps."""

def source_for(manifest,rally_id):
    if 'sources' not in manifest:return manifest['source']
    rally=next((r for r in manifest['rallies'] if r['rally_id']==rally_id),None)
    if rally is None:raise ValueError(f'未知跨局回合：{rally_id}')
    key=rally.get('source_id')
    if key not in manifest['sources']:raise ValueError(f'回合缺少有效原片映射：{rally_id}')
    return manifest['sources'][key]
