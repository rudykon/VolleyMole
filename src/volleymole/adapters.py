"""Package-owned inference processes and explicit, source-hash-validated cache imports."""
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import sys

from .common import APP, Stages, digest, identity, read_json, run, save_json

INFERENCE_FILES = ('common.py','video.py','inference.py','shared.py','state_model.py',
                   'detectors.py','tracker.py','vball_primitives.py','jersey.py','models.py','telemetry.py','gpu_stages.py',
                   'performance.py','pipeline.py')


def inference_signature(source, registry, device, number, confidence, devices=None, performance=None):
    import importlib.metadata as metadata
    from .detectors import resolve_device
    from .gpu_stages import parse_devices
    # Explicit device fingerprints remain reusable without a live GPU (e.g. a
    # render-only resume). The inference worker validates hardware if it runs.
    devices = parse_devices(devices)
    device = devices[0] if devices else (resolve_device(device) if device=='auto' else ('cuda:0' if device=='cuda' else device))
    packages = ('torch','torchvision','transformers','ultralytics','onnxruntime-gpu',
                'numpy','av','opencv-python-headless','easyocr')
    return {'source': {k:source[k] for k in ('sha256','bytes')}, 'models':registry.entries,
            'performance':performance or {'pipeline_depth':1,'auxiliary_device':None,'vball_engine':'ort'},
            'device':device, 'devices':devices, 'number':number, 'confidence':confidence, 'half':True,
            'code':{name:digest(APP/name) for name in INFERENCE_FILES},
            'environment':{name:metadata.version(name) for name in packages}}


def ingest_shared(video, directory, registry, signature, cache_root, force=False):
    """Content-addressed inference cache is independent of top-k, style and API calls."""
    fingerprint = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    cached = Path(cache_root).resolve()/fingerprint
    cached.mkdir(parents=True,exist_ok=True)
    groups = {'analytics':['detections.jsonl','summary.json'],
              'tracking':['ball.csv','source_pts.csv','summary.json'], 'player':['index.json']}
    with (cached/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state = Stages(cached)
        if force:
            state.data['stages'].pop('shared_inference',None)
        def compute():
            command = [sys.executable,'-m','volleymole.inference','--kind','shared','--video',video,
                '--output',cached,'--models',registry.directory,'--half']
            command += ['--devices', ','.join(signature['devices'])] if signature.get('devices') else ['--device', signature['device']]
            for name,value in signature.get('performance',{}).items():
                if value is not None:
                    command += ['--'+name.replace('_','-'),str(value)]
            if signature['number'] is not None:
                command += ['--number',signature['number'],'--confidence',signature['confidence']]
            run(command,cached/'inference.log')
            summary = read_json(cached/'summary.json')
            if summary['status'] != 'complete':
                raise ValueError('Partial inference cannot be committed to full-match cache')
            artifacts = [cached/k/name for k,names in groups.items() for name in names]
            artifacts += [cached/name for name in ('summary.json','provenance.json','telemetry.json')]
            return str(cached),artifacts
        state.execute('shared_inference',signature,compute)
        original = read_json(cached/'provenance.json')
        artifacts = []
        for kind,names in groups.items():
            out = directory/kind
            out.mkdir(exist_ok=True)
            for name in names:
                target = out/name
                shutil.copyfile(cached/kind/name,target)
                artifacts.append(target)
            save_json(out/'provenance.json', {'mode':'shared_inference_cache',
                'cache_fingerprint':fingerprint,'cache_path':str(cached),'original':original,
                'module':kind,'shared_decode':True,'cache_status':state.current_run[-1]['status']})
            artifacts.append(out/'provenance.json')
        report = {'cache_path':str(cached),'fingerprint':fingerprint,
                  'cache_status':state.current_run[-1]['status'],
                  'summary':read_json(cached/'summary.json'),
                  'historical_inference_telemetry':read_json(cached/'telemetry.json'),
                  'note':'Historical inference time is not current wall time when the cache is reused.'}
        save_json(directory/'inference_cache.json',report)
        artifacts.append(directory/'inference_cache.json')
    return {k:str(directory/k/names[0]) for k,names in groups.items()},artifacts


def validate_cache(video, cache, kind):
    cache = Path(cache)
    manifest = read_json(cache.parent/'match_manifest.json')
    expected = manifest['source']['identity']
    observed = identity(video)
    if (expected['sha256'], expected['bytes']) != (observed['sha256'], observed['bytes']):
        raise ValueError('Evidence cache belongs to different source bytes')
    provenance = read_json(cache/'provenance.json')
    origin = provenance
    while isinstance(origin,dict):
        if origin.get('parameters',{}).get('max_frames') is not None:
            raise ValueError('Partial smoke inference is not a full-match cache')
        origin = origin.get('original')
    for name in ('summary.json','index.json'):
        if (cache/name).is_file() and read_json(cache/name).get('status')=='partial_smoke':
            raise ValueError('Partial smoke inference is not a full-match cache')
    # A completed pipeline already recorded the raw artifact hashes. Validate
    # them if present; do not import a silently modified historical detection file.
    state_path = cache.parent/'state.json'
    if state_path.is_file():
        from .common import digest
        stages = read_json(state_path)['stages']
        evidence_stages = [stages[name] for name in (kind,'inference') if name in stages]
        for file in cache.glob('*'):
            # Match the module-relative suffix too, so relocating a complete
            # verified run does not discard its stored raw-evidence hashes.
            expected = {h for state in evidence_stages for p,h in state.get('artifacts',{}).items()
                        if Path(p).parts[-2:]==(kind,file.name)}
            if expected and (len(expected)!=1 or digest(file) not in expected):
                raise ValueError(f'Modified cache artifact: {file.name}')
    return provenance


def ingest(kind, video, directory, cache, python, device, number=None, confidence=.75):
    out = directory/kind
    out.mkdir(exist_ok=True)
    files = {'analytics':['detections.jsonl','summary.json'],
             'tracking':['ball.csv','source_pts.csv'], 'player':['index.json']}[kind]
    if cache is not None:
        original = validate_cache(video, cache, kind)
        inputs = [identity(Path(cache)/name) for name in files]
        for name in files:
            shutil.copyfile(Path(cache)/name, out/name)
        save_json(out/'provenance.json', {'mode':'explicit_verified_cache', 'source':identity(video),
            'raw_files':inputs, 'original':original,
            'note':'Imported inference was not recomputed with the current package models.'})
    else:
        command = [python, '-m', 'volleymole.inference', '--kind',kind, '--video',video,
                   '--output',out, '--device',device, '--half']
        if number is not None:
            command += ['--number', number, '--confidence',confidence]
        run(command, out/'inference.log')
    artifacts = [out/name for name in files] + [out/'provenance.json']
    if cache is None and (out/'telemetry.json').is_file():
        artifacts.append(out/'telemetry.json')
    return str(out/files[0]), artifacts


def ingest_analytics(video, directory, cache, python, device):
    return ingest('analytics', video, directory, cache, python, device)


def ingest_tracking(video, directory, cache, python, device):
    return ingest('tracking', video, directory, cache, python, device)


def ingest_player(video, directory, number, python, device, confidence):
    if number is None:
        out = directory/'player'
        out.mkdir(exist_ok=True)
        save_json(out/'index.json', {'status':'not_requested','number':None,'detections':[]})
        save_json(out/'provenance.json', {'project':'VolleyMole','mode':'not_requested','models':{}})
        return str(out/'index.json'), [out/'index.json',out/'provenance.json']
    return ingest('player', video, directory, None, python, device, number, confidence)
