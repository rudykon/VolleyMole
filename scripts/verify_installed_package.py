"""Run with an isolated installed interpreter (-I), from outside the repository."""
import argparse
from pathlib import Path
import sys
from volleymole.common import APP, save_json
from volleymole.inference import main as infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if 'site-packages' not in APP.parts:
        raise RuntimeError(f'Not testing an installed wheel: {APP}')
    for kind in ('analytics','tracking','player','shared'):
        command = ['--kind',kind,'--video',str(args.video),'--models',str(args.models),
                   '--output',str(args.output/kind),'--device',args.device,'--max-frames','30','--half']
        if kind in ('player','shared'):
            command += ['--number','12']
        infer(command)
    imported = {name:str(module.__file__) for name,module in sys.modules.items()
                if name.startswith('volleymole') and getattr(module,'__file__',None)}
    if any('/tools/' in path or not Path(path).is_relative_to(APP) for path in imported.values()):
        raise RuntimeError('Installed inference depended on external source')
    forbidden = {'ml_manager','volleyball_highlights','make_reels','run_batch_video','run_sample'}
    if forbidden.intersection(sys.modules):
        raise RuntimeError('Upstream application module was imported')
    save_json(args.output/'installed-package-proof.json', {'status':'passed','python':sys.executable,
        'isolated_interpreter':bool(sys.flags.isolated),'cwd':str(Path.cwd()),'modules':imported,
        'scope':'Real 30-frame neural inference per independent module and shared stream with OCR; not full-match accuracy.'})
    print('Installed neural package isolation passed')


if __name__=='__main__':
    main()
