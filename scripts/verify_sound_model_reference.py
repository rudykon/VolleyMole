#!/usr/bin/env python3
"""Compare the export to the original author code and torchlibrosa on real audio.

This optional audit needs torchlibrosa==0.1.0 and librosa==0.11.0 in an isolated
reference environment. Production inference does not need either package.
Reference code is never downloaded or executed without the pinned hash check.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
REFERENCE_FILES = {
    'models.py': '7f9af440395ace5160bbb51d654a0dc35fb887fbf5edecb12da61ff6efb306d9',
    'pytorch_utils.py': '1464fcfbfc0fe4c55f690f6b39e1c80eeed5de1e7fd1b7fd30334d304de7dbe9',
}


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=ROOT/'models/audio/reference')
    parser.add_argument('--dependency-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'models/audio/Cnn14_DecisionLevelMax_mAP=0.385.pth')
    parser.add_argument('--model', type=Path, default=ROOT/'models/audio/panns_cnn14_sed.pt')
    parser.add_argument('--audio', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/public_benchmarks/panns_reference_parity.json')
    args = parser.parse_args()
    for filename, expected in REFERENCE_FILES.items():
        if sha256(args.reference_dir/filename) != expected:
            raise ValueError(f'Author reference source differs from the reviewed version: {filename}')
    from install_sound_model import CHECKPOINT_SHA256
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from the pinned author weight file')
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference_dir.resolve()))
    import numpy as np
    import torch
    import librosa
    import importlib.metadata
    from volleymole.audio_events import decode_audio
    from volleymole.panns import PannsCnn14Sed
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('panns_author_reference', args.reference_dir/'models.py')
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference = reference_module.Cnn14_DecisionLevelMax(32000, 1024, 320, 64, 50, 14000, 527).eval()
    local = PannsCnn14Sed().eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model']
    reference.load_state_dict(state, strict=True)
    local.load_state_dict(state, strict=True)
    exported = torch.jit.load(str(args.model), map_location='cpu').eval()
    checks = []
    with torch.inference_mode():
        for path in args.audio:
            audio = decode_audio({'has_audio': True, 'path': str(path)})
            if len(audio) < 160000:
                raise ValueError('Use at least five seconds of real audio for this reference audit')
            for seconds in (0.08, 0.32, 1.25, 2.73, 5.):
                tensor = torch.from_numpy(audio[:round(seconds*32000)].copy())[None]
                # The published network needs >= one CNN step. The deployment
                # explicitly pads shorter inputs; compare that declared policy.
                reference_input = torch.nn.functional.pad(tensor, (0, max(0, 10240-tensor.shape[1])))
                frames = tensor.shape[1]//320+1
                reference_power = reference.spectrogram_extractor(reference_input)
                local_power = local.spectrogram_extractor(reference_input)
                reference_mel = reference.logmel_extractor(reference_power)
                local_mel = local.logmel_extractor(local_power)
                expected = reference(reference_input)['framewise_output'][:, :frames]
                actual = exported(tensor)['framewise_output']
                errors = {
                    'spectrogram_max_abs_error': float((reference_power-local_power).abs().max()),
                    'logmel_max_abs_error': float((reference_mel-local_mel).abs().max()),
                    'framewise_max_abs_error': float((expected-actual).abs().max()),
                }
                if expected.shape != actual.shape or not np.isfinite(list(errors.values())).all() or max(errors.values()) > 1e-5:
                    raise ValueError(f'Author reference parity failed: {path.name} {seconds}s {errors}')
                checks.append({'audio': str(path), 'audio_sha256': sha256(path), 'seconds': seconds,
                    'shape': list(actual.shape), **errors})
    report = {
        'status': 'passed', 'purpose': 'Numerical equivalence to original author model and torchlibrosa; not an accuracy metric',
        'reference_source': 'https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master/pytorch',
        'reference_file_sha256': REFERENCE_FILES, 'model_sha256': sha256(args.model),
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'dependencies': {'torch': torch.__version__, 'librosa': librosa.__version__,
            'torchlibrosa': importlib.metadata.version('torchlibrosa')}, 'checks': checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'real_audio_comparisons': len(checks), 'report': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
