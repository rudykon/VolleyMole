"""Deterministic glyph-aware font fallback shared by every video text path.

Keep the requested face whenever it covers the text. Fallback changes only the
font, never the event title. Unsupported text fails rather than drawing tofu.
"""
from functools import lru_cache
from pathlib import Path
import unicodedata
from struct import error as StructError
from fontTools.ttLib import TTFont, TTCollection, TTLibError

from .common import APP, identity
from .assets import asset_root

FONTS = asset_root()/'fonts'
BUNDLED_FONTS = tuple(FONTS/name for name in (
    'NotoSansCJKsc-Bold.otf', 'NotoSerifCJKsc-Regular.otf',
    'NotoSans-Bold.ttf', 'NotoSans-BoldItalic.ttf', 'NotoSerif-Regular.ttf',
    'ZCOOLKuaiLe-Regular.ttf'))
SYSTEM_FONTS = tuple(Path(path) for path in (
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
DEFAULT_FONT = next((p for p in (*BUNDLED_FONTS, *SYSTEM_FONTS) if p.is_file()), BUNDLED_FONTS[0])


class FontCoverageError(ValueError):
    """No usable installed face can faithfully render the requested text."""


def _candidates(preferred):
    preferred = Path(preferred)
    bundled = list(BUNDLED_FONTS)
    if 'serif' in preferred.name.lower():
        bundled.sort(key=lambda p: 'serif' not in p.name.lower())
    return tuple(dict.fromkeys((preferred, *bundled, *SYSTEM_FONTS)))


def _key(path):
    path = Path(path).resolve()
    stat = path.stat()
    # A replacement at the same path must invalidate both cmap and face caches.
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _codepoints(text):
    # Shaping controls and variation selectors do not have standalone glyphs.
    return frozenset(ord(c) for c in normalize_text(text) if not c.isspace()
        and c not in ('\u200c', '\u200d') and not 0xfe00 <= ord(c) <= 0xfe0f
        and not 0xe0100 <= ord(c) <= 0xe01ef)


def normalize_text(text):
    """Normalize display whitespace and accents, never rewrite event wording."""
    text = unicodedata.normalize('NFC', str(text)).replace('\r\n', '\n').replace('\r', '\n')
    # Pillow does not expand tabs: drawing one can produce a .notdef box.
    return ''.join(' ' if c.isspace() and c != '\n' else c for c in text)


def text_units(text):
    """Keep base letters, attached marks and shaping controls together.

    This is a spacing/wrapping aid for our Chinese/Latin typography, not a
    replacement for a general-purpose Unicode line-breaking engine.
    """
    units = []
    for char in normalize_text(text):
        attached = (unicodedata.category(char).startswith('M')
                    or char in ('\u200c', '\u200d') or 0x1f3fb <= ord(char) <= 0x1f3ff)
        if units and (attached or units[-1].endswith('\u200d')):
            units[-1] += char
        else:
            units.append(char)
    return units


@lru_cache(maxsize=64)
def _face_index(key):
    if Path(key[0]).suffix.lower() not in ('.ttc', '.otc'):
        return 0
    with open(key[0], 'rb') as stream:
        with TTCollection(stream, lazy=True) as collection:
            for index, font in enumerate(collection.fonts):
                if 'CJK SC' in (font['name'].getDebugName(1) or ''):
                    return index
    return 0


@lru_cache(maxsize=64)
def _coverage(key):
    with open(key[0], 'rb') as stream:
        with TTFont(stream, fontNumber=_face_index(key), lazy=True) as font:
            cmap = font.getBestCmap() or {}
            return frozenset(code for code, glyph in cmap.items() if glyph != '.notdef' and font.getGlyphID(glyph) != 0)


@lru_cache(maxsize=512)
def _face(key, size):
    from PIL import ImageFont
    return ImageFont.truetype(key[0], size, index=_face_index(key))


def missing_glyphs(text, path):
    """Return unsupported characters from the actual font cmap, not pixel guesses."""
    return frozenset(chr(code) for code in _codepoints(text)-_coverage(_key(path)))


@lru_cache(maxsize=1024)
def _select(required, keys):
    usable = []
    union = set()
    for key in keys:
        try:
            coverage = _coverage(key)
            _face(key, 16)  # Validate the actual Pillow/FreeType loading path too.
        except (OSError, ValueError, KeyError, TTLibError, StructError):
            continue
        usable.append(key)
        union.update(coverage)
        if required <= coverage:
            return key
    if not usable:
        raise FontCoverageError('没有可加载的字体；请运行 volleymole assets fetch 或安装系统 Noto CJK 字体。')
    missing = sorted(required-union)
    if missing:
        details = ', '.join(f'U+{code:04X} ({unicodedata.name(chr(code), "UNNAMED")})' for code in missing[:12])
        raise FontCoverageError(f'所有可用字体均缺少字形：{details}；请提供覆盖这些字符的字体，未替换或删除原文。')
    raise FontCoverageError('没有单一可用字体能完整显示这段混排文字；请提供覆盖全文的字体，未替换或删除原文。')


def _resolve_key(text, preferred):
    if not isinstance(text, str) or not text.strip() or not _codepoints(text):
        raise FontCoverageError('文字不能为空')
    keys = []
    for path in _candidates(preferred):
        try:
            if path.is_file():
                keys.append(_key(path))
        except OSError:
            continue
    return _select(_codepoints(text), tuple(keys))


def resolve_font(text, preferred):
    return Path(_resolve_key(text, preferred)[0])


def load_font(text, preferred, size):
    if not isinstance(size, int) or isinstance(size, bool) or size < 4:
        raise ValueError('字体字号必须是至少 4 的整数')
    return _face(_resolve_key(text, preferred), size)


def font_assets(*preferred):
    """Every available fallback is a rendering dependency, even for classic UI."""
    paths = [Path(__file__)]
    for path in (*preferred, *BUNDLED_FONTS, *SYSTEM_FONTS):
        path = Path(path)
        if path.is_file():
            paths.append(path)
    return list(dict.fromkeys(paths))


def font_fingerprint(*preferred):
    return {'policy': 'glyph_coverage_v1', 'requested': [str(p) for p in preferred],
            'assets': [identity(p) for p in font_assets(*preferred)]}


def validate_render_fonts(decision, font_path, style='classic', title_template='legacy'):
    """Preflight every displayed title before starting any video encoders."""
    from .title_templates import get_template
    from .presentation import lively_title
    from .illustrated import TITLE_FONT
    primary = Path(font_path) if style=='classic' else (
        TITLE_FONT if title_template=='legacy' else get_template(title_template).font)
    rows = []
    for item in decision['selected']:
        text = item['title'] if style=='classic' or decision.get('collection')=='bloopers' else lively_title(item, title_template)
        key = _resolve_key(text, primary)
        rows.append({'rank': item['rank'], 'text': text, 'requested_font': str(primary),
            'resolved_font': key[0], 'face_index': _face_index(key),
            'fallback': Path(key[0]) != primary.resolve()})
    return {'status': 'passed', 'policy': 'glyph_coverage_v1', 'titles': rows,
            'fingerprint': font_fingerprint(font_path, primary)}
