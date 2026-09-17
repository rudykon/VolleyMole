"""Offline coverage for legacy/classic font entry points and layout bounds."""
import tempfile
from pathlib import Path
import unittest
import unicodedata
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from volleymole.assets import asset_root
from volleymole.common import APP, DEFAULT_FONT, save_json
from volleymole.font_support import FontCoverageError, load_font
from volleymole import illustrated, media_worker


LATIN = asset_root()/'fonts/NotoSans-Bold.ttf'
SIMPLIFIED_CHINESE = asset_root()/'fonts/NotoSansCJKsc-Bold.otf'


def previous_lettering(text, size, fill, max_width, tilt=0, outline=None):
    """Unchanged legacy rendering formula for already-supported characters."""
    font = ImageFont.truetype(str(illustrated.TITLE_FONT), size)
    while font.getlength(text) > max_width-16:
        size -= 1
        font = ImageFont.truetype(str(illustrated.TITLE_FONT), size)
    stroke = 3 if outline is not None else 1
    bbox = font.getbbox(text, stroke_width=max(2, stroke))
    layer = Image.new('RGBA', (bbox[2]-bbox[0]+18, bbox[3]-bbox[1]+18))
    draw = ImageDraw.Draw(layer)
    draw.text((9-bbox[0]+2, 9-bbox[1]+3), text, font=font, fill=(6,12,25,90))
    draw.text((9-bbox[0], 9-bbox[1]), text, font=font, fill=fill, stroke_width=stroke,
              stroke_fill=outline if outline is not None else fill)
    if tilt:
        layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True)
    if layer.width > max_width:
        layer.thumbnail((max_width, layer.height), Image.Resampling.LANCZOS)
    return layer


class LegacyFontFallbackTests(unittest.TestCase):
    def test_legacy_whitespace_variants_render_as_plain_single_line_space(self):
        expected = illustrated.lettering('排球 趣味', 48, 'white', 500)
        for separator in ('\t', '\u00a0', '\u3000', '\n'):
            with self.subTest(separator=repr(separator)):
                actual = illustrated.lettering(f'排球{separator}趣味', 48, 'white', 500)
                self.assertEqual((actual.size, actual.tobytes()), (expected.size, expected.tobytes()))

    def test_title_lines_use_canonically_equivalent_text_before_splitting(self):
        nfc = 'Café排球趣味时刻'
        nfd = unicodedata.normalize('NFD', nfc)
        self.assertNotEqual(nfc, nfd)
        expected = illustrated.title_lines(nfc)
        self.assertEqual(illustrated.title_lines(nfd), expected)
        self.assertEqual(''.join(expected), nfc)

    def test_title_lines_never_split_uncomposable_combining_mark_from_base(self):
        title = '排球趣味Q\u0301精彩时刻啊'
        self.assertEqual(unicodedata.normalize('NFC', 'Q\u0301'), 'Q\u0301')
        lines = illustrated.title_lines(title)
        self.assertEqual(len(lines), 2)
        self.assertEqual(''.join(lines), title)
        self.assertIn('Q\u0301', '\n'.join(lines))
        self.assertTrue(all(not unicodedata.combining(line[0]) for line in lines))

    def test_supported_legacy_letters_keep_exact_pixels(self):
        for text in ('排球时刻', 'VOLLEY 10'):
            with self.subTest(text=text):
                args = (text, 48, (255,245,220), 330)
                old = previous_lettering(*args, tilt=2, outline=(13,22,40))
                new = illustrated.lettering(*args, tilt=2, outline=(13,22,40))
                self.assertEqual((new.size, new.tobytes()), (old.size, old.tobytes()))

    def test_legacy_letters_fallback_for_mixed_accented_title(self):
        with patch('volleymole.illustrated.load_font', wraps=load_font) as call:
            layer = illustrated.lettering('Café 排球', 48, 'white', 500)
        self.assertIsNotNone(layer.getbbox())
        self.assertEqual(call.call_args.args[0], 'Café 排球')
        chosen = load_font('Café 排球', illustrated.TITLE_FONT, 48)
        self.assertNotEqual(Path(chosen.path), illustrated.TITLE_FONT)

    def test_legacy_empty_or_unfittable_titles_fail_without_sub_four_font(self):
        for title in ('', '   ', '\n'):
            with self.subTest(title=title), self.assertRaisesRegex(ValueError, '不能为空'):
                illustrated.lettering(title, 48, 'white', 300)
        with patch('volleymole.illustrated.load_font', wraps=load_font) as call:
            with self.assertRaisesRegex(ValueError, '过长'):
                illustrated.lettering('排'*1000, 12, 'white', 60)
        self.assertEqual(min(row.args[2] for row in call.call_args_list), 4)
        with self.assertRaisesRegex(ValueError, '不能小于'):
            illustrated.lettering('球', 3, 'white', 300)

    def test_custom_header_chapter_loads_font_for_its_actual_text(self):
        item = {'rank': 1, 'title': '排球时刻'}
        with patch('volleymole.illustrated.load_font', wraps=load_font) as call:
            frames = illustrated.headers(item, LATIN, 5, item['title'])
        self.assertEqual(frames[-1].shape, (110,720,4))
        self.assertIn(('压轴登场', LATIN, 18), [row.args for row in call.call_args_list])

    def test_custom_overlays_cover_subtitle_and_speed(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('volleymole.illustrated.load_font', wraps=load_font) as call:
            for kind in ('teaser', 'replay'):
                illustrated.overlay(Path(temp)/(kind+'.png'), kind, LATIN)
            calls = [row.args for row in call.call_args_list]
            self.assertIn(('精彩抢先看  ·  好球马上来', LATIN, 19), calls)
            self.assertIn(('0.67×', LATIN, 28), calls)

    def test_legacy_card_caption_and_footer_cover_actual_text(self):
        item = {'rank': 1, 'title': '排球时刻'}
        with patch('volleymole.illustrated.prepare_brand'), \
             patch('volleymole.illustrated.load_font', wraps=load_font) as call:
            image = illustrated.title_card(item, item['title'], LATIN, 5)
        self.assertEqual(image.size, (720,1280))
        calls = [row.args for row in call.call_args_list]
        self.assertIn(('压轴登场  /  接着看这一球', LATIN, 21), calls)
        self.assertIn(('日常排球，也有高光时刻', LATIN, 21), calls)

    def test_missing_legacy_primary_is_not_a_hardcoded_asset_requirement(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp)/'missing-primary.ttf'
            with patch('volleymole.illustrated.TITLE_FONT', missing):
                assets = illustrated.asset_paths()
                layer = illustrated.lettering('排球时刻', 48, 'white', 300)
            self.assertNotIn(missing, assets)
            self.assertIn(APP/'font_support.py', assets)
            self.assertIsNotNone(layer.getbbox())

    def test_empty_header_and_card_fail_clearly(self):
        item = {'rank': 1, 'title': ''}
        for fn in (illustrated.headers, illustrated.title_card):
            with self.subTest(function=fn.__name__), self.assertRaisesRegex(ValueError, '不能为空'):
                if fn is illustrated.headers:
                    fn(item, DEFAULT_FONT, 5, '')
                else:
                    fn(item, '', DEFAULT_FONT, 5)

    def test_unsupported_characters_are_not_silently_removed(self):
        title = '排球\U0010ffff'
        with self.assertRaises(FontCoverageError):
            illustrated.lettering(title, 48, 'white', 500)
        with self.assertRaises(FontCoverageError):
            media_worker._classic_header({'rank': 1, 'title': title}, 5, DEFAULT_FONT)


class ClassicFontFallbackTests(unittest.TestCase):
    def test_classic_whitespace_variants_render_as_plain_single_line_space(self):
        item = {'rank': 1, 'title': '排球 趣味'}
        expected = media_worker._classic_header(item, 5, SIMPLIFIED_CHINESE)
        for separator in ('\t', '\u00a0', '\u3000', '\n'):
            with self.subTest(separator=repr(separator)):
                actual = media_worker._classic_header(
                    {**item, 'title': f'排球{separator}趣味'}, 5, SIMPLIFIED_CHINESE)
                self.assertEqual((actual.size, actual.tobytes()), (expected.size, expected.tobytes()))

    def test_supported_classic_header_keeps_exact_pixels(self):
        item = {'rank': 1, 'title': '排球时刻'}
        old = Image.new('RGB', (720,110), (15,23,36)); draw = ImageDraw.Draw(old)
        draw.rounded_rectangle((20,18,114,92), radius=12, fill=(248,190,54))
        draw.text((35,27), '#1', font=ImageFont.truetype(str(SIMPLIFIED_CHINESE),42), fill=(18,23,31))
        draw.text((134,18), item['title'], font=ImageFont.truetype(str(SIMPLIFIED_CHINESE),32), fill='white')
        draw.text((135,65), 'VOLLEYMOLE  /  日常排球5佳球',
                  font=ImageFont.truetype(str(SIMPLIFIED_CHINESE),20), fill=(169,187,207))
        new = media_worker._classic_header(item, 5, SIMPLIFIED_CHINESE)
        self.assertEqual(new.tobytes(), old.tobytes())

    def test_real_system_ttc_selects_simplified_chinese_face(self):
        from fontTools.ttLib import TTCollection
        path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
        if not path.is_file():
            self.skipTest('System Noto CJK font collection is not installed')
        with path.open('rb') as stream, TTCollection(stream, lazy=True) as collection:
            families = [font['name'].getDebugName(1) or '' for font in collection.fonts]
        expected_index = next((i for i, name in enumerate(families) if 'CJK SC' in name), None)
        self.assertIsNotNone(expected_index, 'Installed Noto CJK TTC must expose its SC face')
        actual = load_font('骨直令 排球', path, 40)
        expected = ImageFont.truetype(str(path), 40, index=expected_index)
        self.assertEqual(Path(actual.path), path)
        self.assertEqual(actual.index, expected_index)
        self.assertIn('CJK SC', actual.getname()[0])
        self.assertEqual(bytes(actual.getmask('骨直令 排球')), bytes(expected.getmask('骨直令 排球')))

    def test_classic_bloopers_checks_rank_title_and_caption_independently(self):
        selected = []
        def resolve(text, preferred, size):
            font = load_font(text, preferred, size)
            selected.append((text, Path(font.path)))
            return font
        with patch('volleymole.media_worker.load_font', side_effect=resolve):
            image = media_worker._classic_header({'rank': 1, 'title': '追球后摔倒'}, 1, LATIN, 'bloopers')
        self.assertEqual(image.size, (720,110))
        self.assertIn(('#1', LATIN), selected)
        chinese = [(text, path) for text, path in selected if text != '#1']
        self.assertEqual({text for text, _ in chinese}, {'追球后摔倒', 'VOLLEYMOLE  /  排球趣味时刻'})
        self.assertTrue(all(path != LATIN for _, path in chinese))

    def test_classic_empty_or_unfittable_title_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '不能为空'):
            media_worker._classic_header({'rank': 1, 'title': ' '}, 5, DEFAULT_FONT)
        with patch('volleymole.media_worker.load_font', wraps=load_font) as call:
            with self.assertRaisesRegex(ValueError, '过长'):
                media_worker._classic_header({'rank': 1, 'title': '排'*1000}, 5, DEFAULT_FONT)
        self.assertEqual(min(row.args[2] for row in call.call_args_list), 4)

    def test_lively_render_does_not_build_or_load_classic_header(self):
        class StopBeforeMedia(Exception):
            pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source.mp4'; source.write_bytes(b'fixture only')
            save_json(root/'track.json', {'samples': []})
            save_json(root/'edit_decision.json', {'selected': [{'rank': 1}]})
            item = {'rally_id': 'r1', 'rank': 1, 'title': '排球时刻', 'clip_start_sec': 0., 'clip_end_sec': 1.}
            manifest = {'source': {'path': str(source), 'width': 320, 'height': 180, 'has_audio': False},
                        'rallies': [{'rally_id': 'r1', 'tracking_json': 'track.json'}]}
            with patch('volleymole.media_worker._classic_header', side_effect=AssertionError('unused classic header')), \
                 patch('volleymole.media_worker.load_font', side_effect=AssertionError('unused classic font')), \
                 patch('volleymole.presentation.lively_headers', return_value=[np.zeros((110,720,4), dtype=np.uint8)]), \
                 patch('volleymole.media_worker.subprocess.Popen', side_effect=StopBeforeMedia):
                with self.assertRaises(StopBeforeMedia):
                    media_worker.render_clip(root, item, manifest, root/'unused-font.ttf', style='lively')


if __name__ == '__main__':
    unittest.main()
