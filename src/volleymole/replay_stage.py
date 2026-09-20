"""Wire automatic replay review into run/match and existing-run rendering."""
import argparse
import copy
import math
import os
from pathlib import Path
import time

from .common import APP, ROOT, digest, read_json, save_json, identity
from .replay import reviewed_window
from .replay_review import implementation, fingerprint, public_settings, review_rally, omission
from .semantic import bounded_map
from .sources import source_for


def _manual_review(rally):
    review = rally.get('replay_review')
    return isinstance(review, dict) and review.get('method') in ('assistant_visual_review', 'human_visual_review')


def _automatic_policy(code):
    return fingerprint(dict(implementation=code, stage=digest(APP/'replay_stage.py')))


def _tracking_inputs(directory, manifest, items):
    """Bind optional motion inputs before a top-level complete-cache shortcut."""
    from .schemas import local_file
    by_id = {r['rally_id']: r for r in manifest['rallies']}
    inputs = {}
    for item in items:
        rid = item['rally_id']; path = by_id[rid].get('tracking_json')
        if not path:
            inputs[rid] = None
            continue
        try:
            inputs[rid] = identity(local_file(directory, path))
        except (OSError, ValueError, TypeError):
            # Let the per-rally worker record an inaccessible track as failure;
            # it must not prevent other selected rallies from saving progress.
            inputs[rid] = dict(path=str(path), unavailable=True)
    return inputs


def _validate_automatic_result(directory, cache, row, item, source, report, motion):
    """Re-establish verifier grounding at render time, not just cache checksums."""
    from .replay_review import window_from_verified, verification_problem
    from .replay_review_schema import decode_verify
    from .replay_requests import evidence_valid, request_implementation, request_settings
    from .replay_motion import boundary_guard
    from .schemas import local_file
    cached_identity = cache.get('identity', {})
    if (cached_identity.get('implementation') != report['implementation']
            or cached_identity.get('source') != source['identity']
            or cached_identity.get('source_clock') != {k:source.get(k) for k in ('start_sec','rotation','nominal_fps','duration_sec')}
            or cached_identity.get('interval') != [item['clip_start_sec'],item['clip_end_sec']]
            or cached_identity.get('settings') != report['settings']
            or cached_identity.get('motion_sha256') != fingerprint(motion)
            or cache.get('signature') != fingerprint(cached_identity)):
        raise ValueError('自动复核与当前原片、轨迹或复核策略不一致，须重新复核')
    review = row['review']
    if cache.get('review') != review:
        raise ValueError('回放区间与自动复核证据不一致')
    if review.get('status') != 'approved':
        return
    if cache.get('status') != 'complete' or row.get('status') != 'complete':
        raise ValueError('自动回放仍有未完成的核验，不能作为已通过结果渲染')
    request = next((r for r in cache.get('requests', [])
                    if r.get('signature') == review.get('evidence_request') and r.get('phase') == 'verify'), None)
    if not request:
        raise ValueError('自动回放缺少独立事件核验，须重新复核')
    path = local_file(directory,request['path'])
    if digest(path) != request['sha256'] or not any(
            local_file(directory,a['path']) == path and a['sha256'] == request['sha256'] for a in row['artifacts']):
        raise ValueError('独立事件核验产物未绑定到本次回放')
    raw = read_json(path); request_identity = raw.get('identity', {}); job = request_identity.get('job', {})
    start,end = job.get('start'),job.get('end')
    if (request_identity.get('source') != source['identity']
            or request_identity.get('source_clock') != cached_identity['source_clock']
            or request_identity.get('implementation') != request_implementation('verify')
            or request_identity.get('settings') != request_settings(report['settings'],'verify')
            or raw.get('signature') != request['signature']
            or raw.get('signature') != fingerprint(request_identity)
            or job.get('phase') != 'verify'
            or type(start) not in (int,float) or type(end) not in (int,float)
            or not item['clip_start_sec'] <= start < end <= item['clip_end_sec']):
        raise ValueError('独立事件核验与当前原片或入选范围不一致')
    evidence_valid(raw['evidence'],start,end,request_identity['settings']['fps'],source)
    decoded = decode_verify(raw['raw'],raw['evidence'])
    decoded['_observed_frame_times'] = [frame['start_sec'] for frame in raw['evidence']]
    if decoded['verdict'] != 'accept' or verification_problem(decoded,report['settings']):
        raise ValueError('独立事件核验未通过当前动作与完整性门禁')
    expected = window_from_verified(decoded,item,source)
    if any(review.get(k) != value for k,value in expected.items()):
        raise ValueError('回放边界与独立事件核验实际锚点不一致')
    if boundary_guard(motion,review)['verdict'] == 'expand':
        raise ValueError('当前球轨迹仍显示回放裁断后续，须重新复核')


def add_arguments(parser):
    parser.add_argument('--replay-review', choices=('auto', 'required', 'off'), default='auto',
        help='精彩慢回放：auto 自动时序复核；required 复核未完成则停止；off 保留本地动作候选。rules 排名默认不发请求')
    parser.add_argument('--replay-review-timeout', type=float, default=900., help='入选回合慢回放复核总预算（秒）')
    parser.add_argument('--replay-review-concurrency', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--replay-scan-fps', type=float, default=4.)
    parser.add_argument('--replay-review-fps', type=float, default=8.)
    parser.add_argument('--replay-review-width', type=int, default=1024)
    parser.add_argument('--replay-max-expansions', type=int, choices=range(4), default=2)
    parser.add_argument('--replace-replay-reviews', action='store_true', help='对已有逐球复核记录重新执行自动复核；请求缓存仍可复用')


def validate_arguments(args):
    for k in ('replay_review_timeout', 'replay_scan_fps', 'replay_review_fps'):
        v = getattr(args, k)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f'{k} 必须为正有限数')
    if not 2 <= args.replay_scan_fps <= args.replay_review_fps <= 16:
        raise ValueError('回放采样率须满足 2 <= scan <= review <= 16')
    if not 384 <= args.replay_review_width <= 1280:
        raise ValueError('回放复核宽度须为 384–1280')


def settings_from(args):
    return dict(endpoint=args.api_base, model=args.vision_model or args.model,
        key=os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY'),
        timeout=args.api_timeout, max_tokens=getattr(args, 'semantic_max_tokens', 4096),
        reasoning_effort=getattr(args, 'semantic_reasoning_effort', None),
        scan_fps=getattr(args, 'replay_scan_fps', 4.), review_fps=getattr(args, 'replay_review_fps', 8.),
        width=getattr(args, 'replay_review_width', 1024), chunk_sec=12., overlap_sec=2., max_context_sec=18.,
        max_expansions=getattr(args, 'replay_max_expansions', 2), max_candidates=3,
        min_confidence=.65, min_excitement=3, retries=getattr(args, 'semantic_retries', 1))


def config_from(args):
    return {k:getattr(args,k) for k in ('replay_review','replay_review_timeout','replay_review_concurrency',
        'replay_scan_fps','replay_review_fps','replay_review_width','replay_max_expansions','replace_replay_reviews')}


def run_review(directory, args):
    directory = Path(directory)
    manifest = read_json(directory/'match_manifest.json'); decision = read_json(directory/'edit_decision.json')
    mode = getattr(args, 'replay_review', 'auto'); settings = settings_from(args)
    binding = dict(manifest_sha256=digest(directory/'match_manifest.json'), decision_sha256=digest(directory/'edit_decision.json'))
    applicable = args.style == 'lively' and decision.get('collection', 'highlights') == 'highlights'
    disabled = (mode == 'off' or not applicable or (args.ranker == 'rules' and mode == 'auto'))
    ready = bool(settings['endpoint'] and settings['model'] and settings['key'])
    by_id = {r['rally_id']: r for r in manifest['rallies']}
    existing = bool(decision['selected']) and not getattr(args, 'replace_replay_reviews', False) and all(
        _manual_review(by_id[i['rally_id']]) for i in decision['selected'])
    code = implementation()
    tracking_inputs = _tracking_inputs(directory,manifest,decision['selected'])
    automatic_policy = _automatic_policy(code)
    signature = fingerprint(dict(**binding, settings=public_settings(settings), mode=mode, implementation=code,
        tracking_inputs=tracking_inputs, automatic_policy_sha256=automatic_policy,
        disabled=disabled, configured=ready, replace=getattr(args, 'replace_replay_reviews', False)))
    target = directory/'replay_reviews.json'
    base = dict(schema_version=1, signature=signature, mode=mode, **binding,
        settings=public_settings(settings), implementation=code,
        tracking_inputs=tracking_inputs, automatic_policy_sha256=automatic_policy,
        results={}, omit_unreviewed=False)
    if disabled or not (ready or existing):
        report = dict(base, status='disabled' if disabled else 'unavailable',
            reason='not_applicable' if not applicable else 'explicit_off_or_rules' if disabled else 'model_or_key_not_configured',
            approved_count=0, omitted_count=0, failed_count=0)
        save_json(target, report)
        if mode == 'required' and applicable and not disabled:
            raise RuntimeError('自动慢回放复核缺少模型或 API 配置；未标记为复核通过')
        if applicable:
            print('[replay-review] '+('已关闭自动复核，使用已有记录或本地候选' if disabled else
                  '未配置视觉模型或密钥，当前仅有未自动复核的本地候选'), flush=True)
        return report
    # Verify the current bytes before reusing any source-bound result. Once per
    # unique original, rather than once per action/window.
    verified = set()
    for item in decision['selected']:
        source = source_for(manifest, item['rally_id']); source_id = source.get('identity', {})
        key = (source['path'], source_id.get('sha256'))
        if key not in verified:
            if not source_id.get('sha256') or digest(source['path']) != source_id['sha256']:
                raise ValueError('回放复核原片身份不一致；请重新分析修改后的原片')
            verified.add(key)
    if target.is_file():
        old = read_json(target)
        if old.get('signature') == signature and old.get('status') == 'complete':
            try:
                load_reviewed_manifest(directory)
                for result in old['results'].values():
                    for artifact in result.get('artifacts', []):
                        if digest(artifact['path']) != artifact['sha256']:
                            raise ValueError('review_artifact_changed')
                print('[replay-review] 复用已绑定原片的完整复核', flush=True)
                return old
            except (OSError, ValueError, KeyError, TypeError):
                pass
    deadline = time.monotonic()+getattr(args, 'replay_review_timeout', 900.)
    cache = directory/'replay_review'/'cache'
    jobs = [dict(id=i['rally_id'], item=i) for i in decision['selected']]
    def process(job):
        item = job['item']; source = source_for(manifest, item['rally_id']); rally = by_id[item['rally_id']]
        if _manual_review(rally) and not getattr(args, 'replace_replay_reviews', False):
            reviewed_window(item, rally, source)
            return dict(rally_id=item['rally_id'], status='complete', review=rally['replay_review'],
                        origin='existing_source_bound_review', artifacts=[])
        print(f"[replay-review] #{item['rank']} 通看回合并复核动作边界", flush=True)
        from .replay_motion import motion_evidence
        from .schemas import local_file
        tracking = read_json(local_file(directory, rally['tracking_json'])) if rally.get('tracking_json') else None
        motion = motion_evidence(tracking, source, item['clip_start_sec'], item['clip_end_sec'])
        result = review_rally(item, source, settings, cache, deadline, code, motion=motion)
        artifacts = [identity(result['artifact'])]+[identity(r['path']) for r in result['requests']]
        row = dict(rally_id=item['rally_id'], status='complete' if result['status']=='complete' else 'failed',
            review=result['review'], origin='automatic_sequence_review', artifacts=artifacts)
        if result['status']!='complete':
            row['failure']=dict(error='boundary_review_incomplete', unresolved_boundaries=result['unresolved_boundaries'])
        return row
    results, failures = bounded_map(jobs, process, getattr(args, 'replay_review_concurrency', 2), deadline)
    rows = {r['rally_id']: r for r in results}; failed = {r['job']: r for r in failures}
    for job in jobs:
        if job['id'] not in rows:
            source = source_for(manifest, job['id'])
            rows[job['id']] = dict(rally_id=job['id'], status='failed',
                review=omission(source, '自动时序复核未完成；本回合不生成未经复核的慢回放。'),
                failure=failed.get(job['id'], dict(error='deadline')), artifacts=[])
    report = dict(base, results={j['id']: rows[j['id']] for j in jobs}, omit_unreviewed=True,
        status='complete' if all(r['status'] == 'complete' for r in rows.values()) else 'partial',
        approved_count=sum(r['review']['status'] == 'approved' for r in rows.values()),
        omitted_count=sum(r['status'] == 'complete' and r['review']['status'] == 'omit' for r in rows.values()),
        failed_count=sum(r['status'] == 'failed' for r in rows.values()))
    if mode == 'required' and report['status'] == 'complete' and report['approved_count'] != len(jobs):
        report.update(status='partial',reason='coverage_incomplete')
    save_json(target, report)
    print(f"[replay-review] 完整动作 {report['approved_count']}；无合适回放 {report['omitted_count']}；未完成 {report['failed_count']}", flush=True)
    if mode == 'required' and report['status'] != 'complete':
        if report.get('reason') == 'coverage_incomplete':
            raise RuntimeError('精彩慢回放未覆盖全部入选回合；已保存无合适回放的复核记录，尚未渲染')
        raise RuntimeError('自动慢回放复核未全部完成；已保存进度，重跑会继续，尚未渲染')
    return report


def load_reviewed_manifest(directory, expected_binding=None):
    directory = Path(directory)
    manifest = read_json(directory/'match_manifest.json')
    path = directory/'replay_reviews.json'
    if not path.is_file():
        if expected_binding:
            raise ValueError('缺少成片所依赖的慢回放复核记录')
        selected = {i['rally_id'] for i in read_json(directory/'edit_decision.json')['selected']}
        if any(r['rally_id'] in selected and r.get('replay_review',{}).get('method') == 'vision_sequence_review'
               for r in manifest['rallies']):
            raise ValueError('已有自动慢回放缺少独立核验产物，须重新复核')
        return manifest, None
    if expected_binding and digest(path) != expected_binding.get('sha256'):
        raise ValueError('慢回放复核记录与已渲染成片不一致，须重新渲染')
    report = read_json(path)
    if report.get('mode')=='required' and report.get('status') in ('partial','unavailable'):
        raise ValueError('required 慢回放复核尚未完成，不能绕过复核直接渲染')
    if (report.get('schema_version') != 1
            or report.get('manifest_sha256') != digest(directory/'match_manifest.json')
            or report.get('decision_sha256') != digest(directory/'edit_decision.json')):
        raise ValueError('慢回放复核对应的原片清单或剪辑单已改变，须重新复核')
    decision = read_json(directory/'edit_decision.json'); items = {i['rally_id']: i for i in decision['selected']}
    rows = report['results']
    if report.get('omit_unreviewed') and set(rows) != set(items):
        raise ValueError('自动慢回放复核没有覆盖全部入选回合')
    if (report.get('mode') == 'required' and report.get('status') != 'disabled'
            and (set(rows) != set(items) or any(row.get('status') != 'complete'
                 or row.get('review',{}).get('status') != 'approved' for row in rows.values()))):
        raise ValueError('required 精彩慢回放未覆盖全部入选回合，不能绕过复核直接渲染')
    automatic = {rid for rid,row in rows.items() if row.get('origin') == 'automatic_sequence_review'}
    if automatic:
        code = implementation()
        if report.get('implementation') != code or report.get('automatic_policy_sha256') != _automatic_policy(code):
            raise ValueError('自动慢回放的复核策略已更新，须重新复核')
        tracks = _tracking_inputs(directory,manifest,decision['selected'])
        if (not isinstance(report.get('tracking_inputs'),dict)
                or any(rid not in report['tracking_inputs'] or report['tracking_inputs'][rid] != tracks.get(rid) for rid in automatic)):
            raise ValueError('自动慢回放使用的球轨迹已改变，须重新复核')
    manifest = copy.deepcopy(manifest)
    verified_sources=set()
    for rally in manifest['rallies']:
        rid = rally['rally_id']
        if rid in items and rid not in rows and rally.get('replay_review',{}).get('method') == 'vision_sequence_review':
            raise ValueError('已有自动慢回放没有当前策略的独立核验，须重新复核')
        if rid in rows:
            if rid not in items:
                raise ValueError('慢回放复核包含未入选回合')
            rally['replay_review'] = rows[rid]['review']
            source=source_for(manifest,rid)
            source_key=(source['path'],source['identity']['sha256'])
            if source_key not in verified_sources:
                if digest(source['path'])!=source['identity']['sha256']:
                    raise ValueError('复核后的原片内容已改变，不能复用回放区间')
                verified_sources.add(source_key)
            if rows[rid].get('origin')=='automatic_sequence_review':
                from .schemas import local_file
                from .replay_motion import motion_evidence
                artifacts=rows[rid].get('artifacts',[])
                if not artifacts:
                    raise ValueError('自动复核缺少时序证据产物')
                for artifact in artifacts:
                    cached=local_file(directory,artifact['path'])
                    if digest(cached)!=artifact['sha256']:
                        raise ValueError('自动复核证据产物已改变')
                tracking = read_json(local_file(directory,rally['tracking_json'])) if rally.get('tracking_json') else None
                motion = motion_evidence(tracking,source,items[rid]['clip_start_sec'],items[rid]['clip_end_sec'])
                _validate_automatic_result(directory,read_json(artifacts[0]['path']),rows[rid],items[rid],source,report,motion)
            elif (rally['replay_review'].get('method') == 'vision_sequence_review'
                  and rally['replay_review'].get('status') == 'approved'):
                raise ValueError('已有自动慢回放未经过当前独立核验，须重新复核')
            reviewed_window(items[rid], rally, source)
    return manifest, identity(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='对已有剪辑单自动寻找精彩动作并复核慢回放边界，不改选球排名')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--llm-config', type=Path, default=ROOT/'llm_api.json')
    parser.add_argument('--api-base', default=os.getenv('VOLLEYMOLE_API_BASE', 'https://api.openai.com/v1'))
    parser.add_argument('--model', default=os.getenv('VOLLEYMOLE_MODEL'))
    parser.add_argument('--vision-model', default=os.getenv('VOLLEYMOLE_VISION_MODEL'))
    parser.add_argument('--api-timeout', type=float, default=240.)
    parser.add_argument('--semantic-max-tokens', type=int, default=4096)
    parser.add_argument('--semantic-reasoning-effort', choices=('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'))
    parser.add_argument('--semantic-retries', type=int, choices=range(3), default=1)
    add_arguments(parser); parser.set_defaults(replay_review='required', style='lively', ranker='auto')
    args = parser.parse_args(argv)
    validate_arguments(args)
    if not math.isfinite(args.api_timeout) or args.api_timeout <= 0 or not 256 <= args.semantic_max_tokens <= 16384:
        parser.error('API 超时须为正有限数，max_tokens 须为 256–16384')
    if args.llm_config.is_file():
        llm = read_json(args.llm_config).get('llm', {})
        if llm.get('provider') not in (None, 'openai_compatible'):
            parser.error('只支持项目已配置的 openai_compatible 接口')
        args.api_base = llm.get('base_url', args.api_base)
        args.model = args.model or llm.get('model'); args.vision_model = args.vision_model or llm.get('vision_model')
        if llm.get('api_key'):
            os.environ['VOLLEYMOLE_API_KEY'] = llm['api_key']
    import fcntl
    with (args.run/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('该运行目录正被使用') from None
        report = run_review(args.run.resolve(), args)
    print(f"自动慢回放复核：{report['status']}；{args.run/'replay_reviews.json'}")


if __name__ == '__main__':
    main()
