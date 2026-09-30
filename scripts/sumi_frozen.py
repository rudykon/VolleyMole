"""Entry point copied into the immutable sumi template bundle."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, entry in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'非法包内路径：{name}')
        path = ROOT / relative
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError(f'定稿文件缺失或为符号链接：{name}')
        with path.open('rb') as stream:
            hashed = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != entry['bytes'] or hashed != entry['sha256']:
            raise ValueError(f'定稿文件已变更：{name}')
    return manifest


def activate(manifest):
    for package, version in manifest['render_dependencies'].items():
        actual = importlib.metadata.version(package)
        if actual != version:
            raise ValueError(f'{package} 需要 {version}，当前 {actual}；请使用配套依赖环境')
    # Both this process and rendering subprocesses use the frozen source/assets.
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONPATH'] = str(ROOT / 'src')
    os.environ['VOLLEYMOLE_ASSETS'] = str(ROOT / 'src/volleymole/assets')
    sys.path.insert(0, str(ROOT / 'src'))
    from PIL import features
    if features.version_module('freetype2') != manifest['freetype_version']:
        raise ValueError('FreeType 版本与定稿环境不同；请使用配套 Pillow 环境')


def main():
    args = sys.argv[1:]
    if not args or args == ['--help']:
        print('水墨模板 v2.0.0（墨间回合）\n'
              '  python sumi.py --verify                 校验定稿包\n'
              '  python sumi.py --preview OUTPUT.png     复现确认版预览\n'
              '  python sumi.py run [原有剪辑参数]        单视频剪辑\n'
              '  python sumi.py match [原有剪辑参数]      多视频剪辑\n'
              '使用 Python 3.12 及配套依赖，视频剪辑另需 FFmpeg 和项目模型。')
        return
    manifest = verify()
    if args == ['--verify']:
        print(f"{manifest['version']}：{len(manifest['files'])} 个文件校验通过")
        return
    activate(manifest)
    if len(args) == 2 and args[0] == '--preview':
        from PIL import Image
        from volleymole import font_support
        from volleymole.design_suites import SuiteCard
        from volleymole.presentation import lively_title
        # Prove that a machine without any system fonts can reproduce the card.
        font_support.SYSTEM_FONTS = ()
        sample = manifest['preview']
        item = sample['item']
        card = SuiteCard(item, lively_title(item, 'sumi-zh'), 10, 'sumi').image()
        with Image.open(ROOT / 'preview.png') as approved:
            if card.mode != approved.mode or card.size != approved.size or card.tobytes() != approved.tobytes():
                raise ValueError('预览与已确认版本不同，未输出替代版')
        output = Path(args[1]).expanduser().resolve()
        if output.is_relative_to(ROOT):
            raise ValueError('预览请输出到定稿包目录之外，保留原包不变')
        output.parent.mkdir(parents=True, exist_ok=True)
        card.save(output)
        print(f'与确认版逐像素一致：{output}')
        return
    if args[0] not in ('run', 'match'):
        raise ValueError('请选择 --verify、--preview、run 或 match')
    from volleymole.cli import main as cli
    font = ROOT / 'src/volleymole/assets/fonts/NotoSansCJKsc-Bold.otf'
    sys.argv = ['volleymole', args[0], '--template', str(ROOT / 'template.json'),
                '--font', str(font), *args[1:]]
    cli()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        print(f'水墨定稿包：{exc}', file=sys.stderr)
        raise SystemExit(1)
