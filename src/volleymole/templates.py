"""Portable editing presets: explicit JSON, local imports, no media or credentials."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_LIBRARY = PROJECT / 'templates'
# Wheels carry installation seeds; every active library keeps its own presets.
BUILTINS = (SOURCE_LIBRARY / 'builtin' if (PROJECT / 'pyproject.toml').is_file()
            else Path(sys.prefix) / 'share/volleymole/templates/builtin')
MAX_BYTES = 64 * 1024


def option_schema():
    from .art_themes import THEME_IDS
    from .title_templates import TEMPLATE_IDS
    from .transitions import STYLE_IDS
    from .design_suites import SUITE_IDS
    from .quality import QUALITY_IDS
    return {
        'top_k': (int, (5, 10)),
        'collection': (str, ('highlights', 'bloopers', 'both')),
        'analysis_mode': (str, ('auto', 'rallies', 'events')),
        'ranker': (str, ('auto', 'rules')),
        'focus_player': (int, None),
        'style': (str, ('classic', 'lively')),
        'art_theme': (str, THEME_IDS),
        'title_template': (str, TEMPLATE_IDS),
        'transition_style': (str, STYLE_IDS),
        'design_suite': (str, SUITE_IDS),
        'design_language': (str, ('zh', 'en')),
        'quality': (str, QUALITY_IDS),
        'replays': (bool, None),
        'replay_speed': (float, None),
        'replay_review': (str, ('auto', 'required', 'off')),
        'replay_review_timeout': (float, None),
        'replay_review_concurrency': (int, range(1, 9)),
        'replay_scan_fps': (float, None),
        'replay_review_fps': (float, None),
        'replay_review_width': (int, None),
        'replay_max_expansions': (int, range(4)),
    }


def validate(document):
    if not isinstance(document, dict) or set(document) - {'version', 'name', 'description', 'options', 'audio'}:
        raise ValueError('模板只接受 version、name、description、options、audio 字段')
    if type(document.get('version')) is not int or document['version'] != 1:
        raise ValueError('模板 version 必须为 1')
    name = document.get('name')
    if not isinstance(name, str) or not re.fullmatch(r'[\w-]{1,64}', name):
        raise ValueError('模板 name 须为 1–64 个字母、数字、中文、下划线或连字符')
    if not isinstance(document.get('description', ''), str):
        raise ValueError('模板 description 必须为字符串')
    options = document.get('options')
    schema = option_schema()
    if not isinstance(options, dict) or not options:
        raise ValueError('模板 options 必须为非空对象')
    for key, value in options.items():
        if key not in schema:
            raise ValueError(f'不支持的模板选项：{key}；路径、设备和 API 配置请在命令行指定')
        kind, choices = schema[key]
        types = (int, float) if kind is float else (kind,)
        if type(value) not in types or (kind is float and not math.isfinite(value)):
            raise ValueError(f'模板选项 {key} 类型错误，需要 {kind.__name__}')
        if choices is not None and value not in choices:
            raise ValueError(f'模板选项 {key} 无效；可选：{", ".join(map(str, choices))}')
    for key in ('replay_review_timeout', 'replay_scan_fps', 'replay_review_fps'):
        if key in options and options[key] <= 0:
            raise ValueError(f'{key} 必须为正有限数')
    if 'replay_review_width' in options and not 384 <= options['replay_review_width'] <= 1280:
        raise ValueError('replay_review_width 必须在 384–1280 之间')
    if not 2 <= options.get('replay_scan_fps', 4) <= options.get('replay_review_fps', 8) <= 16:
        raise ValueError('回放采样率须满足 2 <= scan <= review <= 16')
    if 'focus_player' in options and not 0 <= options['focus_player'] <= 999:
        raise ValueError('focus_player 必须在 0–999 之间')
    if not .5 <= options.get('replay_speed', 2/3) <= 1:
        raise ValueError('replay_speed 必须在 0.5–1.0 之间')
    if options.get('replays') is False and options.get('replay_review') == 'required':
        raise ValueError('关闭慢回放时不能要求回放复核 required')
    if options.get('style') == 'classic' and any(
        options.get(k, v) != v for k, v in (
            ('design_suite', 'custom'), ('art_theme', 'default'),
            ('title_template', 'legacy'), ('transition_style', 'fade'))
    ):
        raise ValueError('classic 不能使用 lively 的套装、插画、标题或转场')
    validate_audio(document.get('audio', {}))
    return document


def validate_audio(settings):
    if not isinstance(settings, dict) or set(settings) - {'enabled', 'max_cues', 'rms_db', 'duck_db'}:
        raise ValueError('audio 只接受 enabled、max_cues、rms_db、duck_db')
    if 'enabled' in settings and type(settings['enabled']) is not bool:
        raise ValueError('audio.enabled 必须为布尔值')
    if 'max_cues' in settings and (type(settings['max_cues']) is not int or not 1 <= settings['max_cues'] <= 20):
        raise ValueError('audio.max_cues 必须是 1–20 的整数')
    for key, low, high in (('rms_db', -36, -14), ('duck_db', -12, 0)):
        if key in settings and (type(settings[key]) not in (float, int)
                                or not math.isfinite(settings[key]) or not low <= settings[key] <= high):
            raise ValueError(f'audio.{key} 必须在 {low}–{high} 之间')
    return settings


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'模板含重复字段：{key}')
        result[key] = value
    return result


def read_template(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('模板文件不能超过 64 KiB')
    try:
        return validate(json.loads(data.decode('utf-8-sig'), object_pairs_hook=_unique_object))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('模板必须是 UTF-8 JSON 文件') from exc


def library_directory(directory=None):
    """One stable library root, independent of the caller's working directory."""
    selected = directory or os.environ.get('VOLLEYMOLE_TEMPLATES')
    if selected:
        return Path(selected).expanduser().resolve()
    if os.environ.get('VOLLEYMOLE_WORKSPACE'):
        return Path(os.environ['VOLLEYMOLE_WORKSPACE']).expanduser().resolve() / 'templates'
    if (PROJECT / 'pyproject.toml').is_file():
        return SOURCE_LIBRARY
    base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')).expanduser()
    return (base / 'volleymole/templates').resolve()


def builtin_directory(directory=None):
    root = library_directory(directory) / 'builtin'
    if root == BUILTINS:
        return root
    # A fresh workspace/installed application gets local copies once. Existing
    # files, including a user's edits, are never replaced by discovery.
    seeds = sorted(BUILTINS.glob('*.json'))
    if not seeds:
        raise ValueError('缺少内置模板，请重新安装完整应用包')
    root.mkdir(parents=True, exist_ok=True)
    for source in seeds:
        target = root / source.name
        if target.exists():
            continue
        try:
            with target.open('xb') as output, source.open('rb') as input_file:
                shutil.copyfileobj(input_file, output)
        except FileExistsError:
            pass
    return root


def local_directory(directory=None):
    return library_directory(directory) / 'custom'


def template_entries(directory=None):
    root = library_directory(directory)
    seen = set()
    # The root-level reader preserves older user libraries; new saves use custom/.
    for folder, builtin in ((builtin_directory(root), True), (root / 'custom', False), (root, False)):
        for path in sorted(folder.glob('*.json')):
            if path.stem not in seen:
                seen.add(path.stem)
                yield path, builtin


def load_template(selection, directory=None):
    selection = str(selection)
    if selection.endswith('.json') or '/' in selection or '\\' in selection:
        return read_template(Path(selection).expanduser())
    if not re.fullmatch(r'[\w-]{1,64}', selection):
        raise ValueError('请指定模板名称或 JSON 文件路径')
    for path, _ in template_entries(directory):
        if path.stem == selection:
            return read_template(path)
    raise ValueError(f'模板不存在：{selection}；用 volleymole templates list 查看')


def write_template(document, path):
    validate(document)
    data = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
    if len(data.encode('utf-8')) > MAX_BYTES:
        raise ValueError('模板文件不能超过 64 KiB')
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps a user's existing preset intact on name collisions.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)
    return path


class TemplateArgumentParser(argparse.ArgumentParser):
    """Prepend validated preset options so explicit CLI flags always win."""

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        probe.add_argument('--template')
        selection, _ = probe.parse_known_args(argv)
        document = None
        if selection.template and not any(x in ('--help', '-h') for x in argv):
            try:
                document = load_template(selection.template)
            except (ValueError, OSError) as exc:
                self.error(str(exc))
            argv = [(f'--{key}' if value else f'--no-{key}') if type(value) is bool
                    else f'--{key.replace("_", "-")}={value}'
                    for key, value in document['options'].items()] + argv
        result = super().parse_args(argv, namespace)
        result.template_snapshot = None if document is None else {
            **document,
            'options': {key: getattr(result, key) for key in option_schema()
                        if getattr(result, key, None) is not None},
        }
        return result


def argument_parser():
    parser = argparse.ArgumentParser(description='导入、导出和查看成片模板；不运行推理或 API')
    parser.add_argument('--directory', type=Path, help='统一模板库根目录；默认项目 templates，支持 VOLLEYMOLE_TEMPLATES')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='列出内置和本地模板')
    show = commands.add_parser('show', help='查看模板 JSON')
    show.add_argument('template')
    install = commands.add_parser('import', help='校验后导入本地模板库；同名文件不覆盖')
    install.add_argument('file', type=Path)
    install.add_argument('--name', help='以新名称保存')
    export = commands.add_parser('export', help='导出已有模板或运行配置，可编辑后再导入')
    export.add_argument('template', nargs='?')
    export.add_argument('--from-run', type=Path, help='已有运行目录，读取 run_config.json')
    export.add_argument('--name', help='新模板名称')
    export.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    directory = library_directory(args.directory)
    if args.command == 'list':
        for path, builtin in template_entries(directory):
            doc = read_template(path)
            print(f'{path.stem:16} {"内置" if builtin else "个人"}  {doc.get("description", "")}')
        return
    if args.command == 'show':
        print(json.dumps(load_template(args.template, directory), ensure_ascii=False, indent=2))
        return
    if args.command == 'import':
        document = read_template(args.file)
        if args.name:
            document['name'] = args.name
        validate(document)
        if (builtin_directory(directory) / f'{document["name"]}.json').exists():
            parser.error('名称与内置模板冲突，请用 --name 指定自己的名称')
        if (directory / f'{document["name"]}.json').exists():
            raise FileExistsError('旧目录已有同名个人模板，请用 --name 指定新名称')
        target = local_directory(directory) / f'{document["name"]}.json'
    else:
        if bool(args.template) == bool(args.from_run):
            parser.error('指定一个模板名称/文件，或 --from-run 目录，二者只能选一')
        if args.from_run:
            config = json.loads((args.from_run / 'run_config.json').read_text(encoding='utf-8'))
            document = {'version': 1, 'name': args.name or 'my-template',
                        'description': '从成片配置导出',
                        'options': {key: config[key] for key in option_schema()
                                    if config.get(key) is not None}}
            if isinstance(config.get('template'), dict) and 'audio' in config['template']:
                document['audio'] = config['template']['audio']
        else:
            document = load_template(args.template, directory)
        if args.name:
            document['name'] = args.name
        target = args.output
    print(write_template(document, target))


if __name__ == '__main__':
    main()
