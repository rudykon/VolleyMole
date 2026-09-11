"""One full-source discovery pass and at most one review per fused candidate."""
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .common import APP, digest, read_json, save_json
from .events import windows, merge_events, select_events, score_event, WEIGHTS
from .semantic import bounded_map, understand_context


def settings_from(args):
    return {'endpoint': args.api_base, 'model': args.vision_model or args.model,
        'key': os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY'),
        'timeout': args.api_timeout, 'retries': args.semantic_retries,
        'concurrency': args.semantic_concurrency, 'coarse_fps': args.coarse_fps,
        'review_fps': args.review_fps, 'modality': args.semantic_modality,
        'chunk_sec': args.event_chunk_sec, 'overlap_sec': args.event_overlap_sec,
        'audio_model': digest(args.sound_model) if args.sound_model else None,
        'audio_labels': digest(args.sound_labels) if args.sound_labels else None}


class Discovery:
    def __init__(self, source, directory, args, deadline=None, frame_cache=None):
        self.source, self.directory, self.args = source, Path(directory), args
        self.settings = settings_from(args)
        if frame_cache: self.settings['frame_cache'] = str(frame_cache)
        self.started = time.monotonic()
        self.deadline = deadline if deadline is not None else self.started+args.analysis_timeout
        self.cache = Path(args.analysis_cache_dir)/'events' if not args.no_analysis_cache else self.directory/'event_cache'
        if args.no_analysis_cache:
            self.settings['fresh_run'] = time.time_ns()
        import threading
        self.cancelled = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self.discover)

    def close(self):
        self.cancelled.set()
        self.deadline = time.monotonic()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def discover(self):
        from .audio_events import cached_audio, LocalSoundDetector
        if self.args.ranker == 'rules' or not self.settings['key'] or not self.settings['model']:
            return {'results': [], 'failures': [], 'audio': None,
                'audio_features': {'status': 'unknown', 'windows': []}, 'status': 'semantic_not_configured'}
        failures = []
        try:
            audio, features = cached_audio(self.source, self.cache, self.deadline, self.args.no_analysis_cache)
        except (OSError, TimeoutError, __import__('subprocess').SubprocessError) as exc:
            audio = None; features = {'status': 'unknown', 'windows': []}
            failures.append({'job': 'audio', 'error': type(exc).__name__})
        features['sound_events'] = []
        detector = None
        features['sound_detector_status'] = 'configured' if self.args.sound_model else 'unknown_not_configured'
        jobs = [{'id': f'coarse_{i:05d}', 'start': a, 'end': b, 'phase': 'coarse'}
                for i, (a, b) in enumerate(windows(self.source['duration_sec'], self.args.event_chunk_sec, self.args.event_overlap_sec))]
        # Model inference is serialized; each chunk proceeds to upload/API as
        # soon as its local sound observations are available.
        import threading
        sound_lock = threading.Lock()
        def process(job):
            nonlocal detector
            if self.cancelled.is_set(): raise TimeoutError('analysis cancelled')
            local = {**features, 'sound_events': []}
            if self.args.sound_model and audio is not None and time.monotonic() < self.deadline:
                with sound_lock:
                    import hashlib
                    key = hashlib.sha256(json.dumps({'source':self.source['identity'],'job':job,
                        'model':self.settings['audio_model'],'labels':self.settings['audio_labels'],
                        'code':digest(APP/'audio_events.py')},sort_keys=True).encode()).hexdigest()
                    sound_path = self.cache/'sounds'/f'{key}.json'
                    if sound_path.is_file() and not self.args.no_analysis_cache:
                        local['sound_events'] = read_json(sound_path)
                    else:
                        if detector is None: detector = LocalSoundDetector(self.args.sound_model, read_json(self.args.sound_labels))
                        local['sound_events'] = detector.detect(audio[round(job['start']*32000):round(job['end']*32000)], job['start'])
                        save_json(sound_path, local['sound_events'])
            return understand_context(job, self.source, self.cache, self.settings, audio, local, self.deadline)
        results, failed = bounded_map(jobs, process, self.settings['concurrency'], self.deadline, self.cancelled.is_set)
        features['sound_events'] = [s for result in results for s in result['local']['sound_events']]
        completed = {r['job']['id'] for r in results}
        failures += failed
        failed_ids = {f['job'] for f in failures}
        failures += [{'job': j['id'], 'status': 'unreviewed', 'error': 'deadline'} for j in jobs if j['id'] not in completed|failed_ids]
        return {'results': results, 'failures': failures, 'audio': audio, 'audio_features': features,
            'status': 'partial' if failures else 'complete', 'total_chunks': len(jobs)}

    def finish(self, manifest):
        try: discovered = self.future.result()
        finally: self.pool.shutdown(wait=True)
        coarse = []
        for result in discovered['results']:
            for event in result['events']:
                coarse.append({**event, 'source_id': 'single', 'origins': [result['signature']], 'review_status': 'unreviewed'})
        events = merge_events(coarse)
        jobs = []
        covered = set()
        for event in events:
            a, b = event['clip_start_sec'], event['clip_end_sec']
            related = [r for r in manifest['rallies'] if max(r['start_sec'], event['start_sec']) < min(r['end_sec'], event['end_sec'])]
            for rally in related:
                a = min(a, rally['safe_start_sec']); b = max(b, rally['safe_end_sec']); covered.add(rally['rally_id'])
            jobs.append({'id': event['event_id'], 'start': max(0, a-2), 'end': min(self.source['duration_sec'], b+3),
                'phase': 'review', 'candidate': {**{k: event[k] for k in ('event_type', 'uncertainty', 'peak_sec')},
                    'hypotheses': [f['text'] for f in event['observations']]}})
        for rally in manifest['rallies']:
            if rally['rally_id'] in covered: continue
            # Retain every local hypothesis, including short/occluded rallies.
            jobs.append({'id': rally['rally_id'], 'start': max(0, rally['safe_start_sec']-2),
                'end': min(self.source['duration_sec'], rally['safe_end_sec']+3), 'phase': 'review',
                'candidate': {k: rally.get(k) for k in ('start_sec', 'end_sec', 'actions', 'action_events')}})
        # Review likely selections first; every candidate may get one review,
        # never an agent loop or another final ranking API request.
        potential = {e['event_id']: max((e['dimensions'][k]['value'] or 0) for k in e['dimensions']) for e in events}
        jobs.sort(key=lambda j: (-potential.get(j['id'], 0), j['start']))
        oversized = [{'job': j['id'], 'status': 'unreviewed', 'error': 'context_exceeds_max_review_sec'}
                     for j in jobs if j['end']-j['start'] > self.args.max_review_sec]
        jobs = [j for j in jobs if j['end']-j['start'] <= self.args.max_review_sec]
        records = None
        analytics = self.directory/'analytics/detections.jsonl'
        if analytics.is_file() and discovered['status'] != 'semantic_not_configured' and time.monotonic() < self.deadline:
            with analytics.open() as stream: records = [json.loads(line) for line in stream]
            ball_path = self.directory/'tracking/ball.csv'
            self.settings['local_signature'] = [digest(analytics), digest(ball_path) if ball_path.is_file() else None]
            if ball_path.is_file():
                import csv, math
                with ball_path.open() as stream:
                    for row in csv.DictReader(stream):
                        i = int(row['Frame']); xy = [float(row['X']), float(row['Y'])]
                        if (0 <= i < len(records) and int(row['Visibility']) == 1 and all(math.isfinite(x) for x in xy)
                                and 0 <= xy[0] < self.source['width'] and 0 <= xy[1] < self.source['height']):
                            records[i]['event_ball'] = xy
        results, failures = [], []
        if discovered['status'] != 'semantic_not_configured':
            results, failures = bounded_map(jobs, lambda job: understand_context(job, self.source, self.cache,
                self.settings, discovered['audio'], discovered['audio_features'], self.deadline, records),
                self.settings['concurrency'], self.deadline)
        reviewed_ids = {r['job']['id'] for r in results}
        # A successful empty review rejects its candidate; failures retain only
        # coarse evidence, explicitly marked as unreviewed.
        timeline = [e for e in events if e['event_id'] not in reviewed_ids]
        for result in results:
            for event in result['events']:
                timeline.append({**event, 'source_id': 'single', 'origins': [result['signature']], 'review_status': 'reviewed'})
        timeline = merge_events(timeline)
        for event in timeline:
            event['scores'] = {c: score_event(event, c,
                manifest['config'].get('event_weights', WEIGHTS)[c]) for c in WEIGHTS}
        failed_ids = {f['job'] for f in failures}
        failures += [{'job': j['id'], 'status': 'unreviewed', 'error': 'deadline_or_disabled'} for j in jobs
                     if j['id'] not in reviewed_ids|failed_ids]
        report = {'schema_version': 1, 'source': self.source, 'events': timeline,
            'coarse_status': discovered['status'], 'coarse_chunks': discovered.get('total_chunks', 0),
            'coarse_completed': len(discovered['results']), 'review_completed': len(results),
            'failures': discovered['failures']+failures+oversized,
            'budget_sec': self.args.analysis_timeout, 'elapsed_sec': time.monotonic()-self.started,
            'deadline_reached': time.monotonic() >= self.deadline,
            'evidence_policy': 'Explicit PTS video frames and synchronized audio; model prose is not independent evidence.',
            'requests': [{k: r[k] for k in ('signature', 'job', 'requested_fps', 'sample_times_sec', 'request', 'cached', 'elapsed_sec')}
                for r in discovered['results']+results]}
        save_json(self.directory/'event_timeline.json', report)
        save_json(self.directory/'audio_events.json', discovered['audio_features'])
        return report


def collection_artifacts(manifest, timeline, directory, collection, top_k):
    """Adapt an event into renderer input while retaining distinct edit contracts."""
    from .schemas import validate_decision
    from .media_worker import previews
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    candidates = copy.deepcopy(timeline['events'])
    # Athletic edits still preserve ALL of an overlapping detected rally. This
    # expansion occurs before selection so disjointness is checked afterwards.
    if collection == 'highlights':
        for e in candidates:
            for r in manifest['rallies']:
                if e['source_id'] == r.get('source_id', 'single') and max(e['start_sec'], r['start_sec']) < min(e['end_sec'], r['end_sec']):
                    e['start_sec'] = min(e['start_sec'], r['start_sec']); e['end_sec'] = max(e['end_sec'], r['end_sec'])
                    e['clip_start_sec'] = min(e['clip_start_sec'], r['safe_start_sec'])
                    e['clip_end_sec'] = max(e['clip_end_sec'], r['safe_end_sec'])
    selected = select_events(candidates, collection, 5 if collection == 'bloopers' else top_k,
        manifest['config'].get('event_weights', WEIGHTS)[collection], manifest['config'].get('event_score_threshold', 55))
    requested = 5 if collection == 'bloopers' else top_k
    result = {'title': f'比赛{len(selected)}佳球' if collection == 'highlights' else f'排球趣味时刻（{len(selected)}段）',
        'selected': [], 'collection': collection, 'clip_protocol': 'rally_v1' if collection == 'highlights' else 'event_v1',
        'ranking_mode': 'event_dimensions', 'requested_count': requested, 'actual_count': len(selected),
        'shortage_reason': f'符合证据和完整性要求的素材只有 {len(selected)} 段，未凑数。' if len(selected) < requested else None,
        'review_status': {}, 'scores': {}}
    target_manifest = {**manifest, 'rallies': []}
    for rank, (event, scores) in enumerate(selected, 1):
        rid = event['event_id']
        track_name = f'tracking/tracks/{rid}.json'
        # Preserve available original ball samples; funny events without a
        # trustworthy track use the existing centered overview fallback.
        samples = []
        for r in manifest['rallies']:
            if event['source_id'] == r.get('source_id', 'single') and max(event['clip_start_sec'], r['start_sec']) < min(event['clip_end_sec'], r['end_sec']):
                path = Path(manifest.get('_directory', '.'))/r['tracking_json']
                if path.is_file(): samples.extend(read_json(path)['samples'])
        samples = sorted({tuple(s) for s in samples if event['clip_start_sec'] <= s[0] <= event['clip_end_sec']})
        save_json(directory/track_name, {'rally_id': rid, 'samples': samples})
        row = {'rally_id': rid, 'source_id': event['source_id'], 'eligible': True,
            'start_sec': event['start_sec'], 'end_sec': event['end_sec'], 'duration_sec': event['end_sec']-event['start_sec'],
            'safe_start_sec': event['clip_start_sec'], 'safe_end_sec': event['clip_end_sec'],
            'required_start_sec': event['clip_start_sec'], 'required_end_sec': event['clip_end_sec'],
            'event_type': event['event_type'], 'clip_kind': 'event' if collection == 'bloopers' else 'rally',
            'boundary_complete': event['boundary_complete'], 'injury_suspected': event['injury_suspected'],
            'tracking_json': track_name, 'actions': [], 'players': [], 'peak_sec': event['peak_sec'],
            'peak_evidence': event['origins'], 'preview_times_sec': [event['clip_start_sec'], event['peak_sec'], max(event['clip_start_sec'], event['clip_end_sec']-.1)],
            'preview_frames': [f'previews/{rid}_{label}.jpg' for label in ('start', 'peak', 'end')]}
        if 'sources' in manifest: row['source_set'] = manifest['sources'][event['source_id']]['set_number']
        target_manifest['rallies'].append(row)
        reason = '；'.join(f['text'] for f in event['observations'])
        reason = ''.join(c for c in reason if ord(c) >= 32)[:450]
        result['selected'].append({'rank': rank, 'rally_id': rid, 'clip_start_sec': event['clip_start_sec'],
            'clip_end_sec': event['clip_end_sec'], 'title': event['title'], 'reason': reason,
            'confidence': event['confidence']})
        result['scores'][rid] = scores; result['review_status'][rid] = event['review_status']
    target_manifest.pop('_directory', None)
    save_json(directory/'match_manifest.json', target_manifest)
    save_json(directory/'edit_decision.json', result)
    if selected:
        if collection == 'highlights': previews(directory, workers=2)
        validate_decision(result, target_manifest, directory, len(selected))
    return result


def complete_collections(args, manifest, timeline, directory):
    from .media_worker import render
    from .run_match import verify
    directory = Path(directory)
    manifest = {**manifest, '_directory': str(directory)}
    reports = []
    for collection in (('highlights', 'bloopers') if args.collection == 'both' else (args.collection,)):
        target = directory/'collections'/collection
        decision = collection_artifacts(manifest, timeline, target, collection, args.top_k)
        report = {'collection': collection, 'directory': str(target), 'actual_count': decision['actual_count'],
            'requested_count': decision['requested_count'], 'shortage_reason': decision['shortage_reason'], 'output': None}
        if decision['selected'] and args.stop_after != 'rank':
            from .common import Stages, identity
            stages = Stages(target)
            if args.rerun_from in ('rank','render','verify'):
                for name in (('verify',) if args.rerun_from=='verify' else ('render','verify')):
                    stages.data['stages'].pop(name,None)
            suffix = '_lively' if args.style == 'lively' else ''
            def make_video():
                render(target, args.font, args.style, args.render_workers, args.art_theme, args.title_template,
                    args.transition_style, args.design_suite, args.design_language, args.quality)
                rendered = read_json(target/f'render_report{suffix}.json')
                return rendered['output'], [rendered['output'], target/f'render_report{suffix}.json']+[s['path'] for s in rendered.get('segments', rendered['clips'])]
            signature = {'decision': digest(target/'edit_decision.json'), 'manifest': digest(target/'match_manifest.json'),
                'code': {p.name: digest(p) for p in APP.glob('*.py')}, 'font': identity(args.font),
                'render': [args.style, args.render_workers, args.art_theme, args.title_template, args.transition_style,
                    args.design_suite, args.design_language, args.quality]}
            if args.style=='lively':
                from .illustrated import asset_paths
                signature['assets'] = [identity(p) for p in asset_paths(args.art_theme,args.title_template,args.transition_style,args.design_suite)]
            report['output'] = stages.execute('render', signature, make_video)
            stages.execute('verify', {**signature, 'output': digest(report['output'])},
                lambda: verify(target, len(decision['selected']), args.style))
            report['stages'] = stages.current_run
        reports.append(report)
        print(f'{collection}：入选 {decision["actual_count"]}/{decision["requested_count"]}；{report["output"] or target}', flush=True)
    save_json(directory/'collections_report.json', {'collections': reports})
    return reports
