"""Portable font-policy checks using tiny generated faces, never system fonts."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection, TTFont
from PIL import ImageFont

from volleymole import font_support as fonts


def make_font(path, characters, family='Fixture Sans', advance=600, blank=False):
    """Create only the glyphs requested, with optional blank mapped outlines."""
    characters = sorted(set(characters))
    names = ['.notdef'] + [f'g{index:04d}' for index in range(len(characters))]
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({ord(char): names[index + 1]
                               for index, char in enumerate(characters)})
    glyphs = {}
    for name in names:
        pen = TTGlyphPen(None)
        if not blank:
            pen.moveTo((80, 0))
            pen.lineTo((480, 0))
            pen.lineTo((480, 700))
            pen.lineTo((80, 700))
            pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (advance, 80) for name in names})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({'familyName': family, 'styleName': 'Regular',
                           'uniqueFontIdentifier': family + '-Regular',
                           'fullName': family + ' Regular',
                           'psName': family.replace(' ', '') + '-Regular'})
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200,
                     usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)
    return path


class FontSupportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.primary = make_font(self.root/'primary.ttf', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/.,!')
        self.fallback = make_font(self.root/'fallback.ttf',
                                  'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/.,!排球趣味中，。！？：；（）',
                                  family='Fixture CJK')
        # Empty patched system candidates make every test independent of host
        # packages, fontconfig, locale, and the repository's production assets.
        for name, value in (('BUNDLED_FONTS', (self.fallback,)), ('SYSTEM_FONTS', ())):
            patcher = patch.object(fonts, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.clear_caches()
        self.addCleanup(self.clear_caches)

    @staticmethod
    def clear_caches():
        for name in ('_coverage', '_face', '_select', '_face_index'):
            getattr(fonts, name).cache_clear()

    def test_preferred_face_wins_when_it_covers_the_whole_text(self):
        self.assertEqual(fonts.resolve_font('TOP 10!', self.primary), self.primary.resolve())
        loaded = fonts.load_font('TOP 10!', self.primary, 24)
        self.assertEqual(loaded.getname(), ('Fixture Sans', 'Regular'))
        self.assertEqual(loaded.size, 24)

    def test_missing_chinese_and_punctuation_select_full_text_fallback(self):
        text = 'TOP 10 / 排球趣味，。！？：；（）'
        self.assertEqual(fonts.resolve_font(text, self.primary), self.fallback.resolve())
        self.assertEqual(fonts.load_font(text, self.primary, 24).getname(),
                         ('Fixture CJK', 'Regular'))
        self.assertEqual(fonts.missing_glyphs(text, self.fallback), frozenset())

    def test_cmap_not_identical_notdef_pixels_determines_coverage(self):
        # A mapped blank glyph is legitimate; both it and this fixture's
        # .notdef have empty masks, so pixel heuristics cannot tell them apart.
        blank = make_font(self.root/'blank.ttf', 'A', blank=True)
        raw = ImageFont.truetype(str(blank), 24)
        self.assertEqual(bytes(raw.getmask('A')), bytes(raw.getmask('球')))
        self.assertEqual(fonts.missing_glyphs('A球球 \n\t', blank), frozenset({'球'}))
        self.assertEqual(fonts.resolve_font('A', blank), blank.resolve())
        self.assertEqual(fonts.resolve_font('A球', blank), self.fallback.resolve())

    def test_missing_preferred_font_falls_back_without_system_fonts(self):
        missing = self.root/'not-installed.ttf'
        self.assertFalse(missing.exists())
        self.assertEqual(fonts.resolve_font('中 TOP 10', missing), self.fallback.resolve())
        self.assertEqual(fonts.load_font('中 TOP 10', missing, 18).getname()[0], 'Fixture CJK')

    def test_corrupt_preferred_font_falls_back(self):
        corrupt = self.root/'corrupt.ttf'
        corrupt.write_bytes(b'not a loadable font')
        self.assertEqual(fonts.resolve_font('排球 TOP 10', corrupt), self.fallback.resolve())

    def test_parseable_but_unloadable_preferred_font_falls_back(self):
        original_face = fonts._face

        def load_or_fail(key, size):
            if key[0] == str(self.primary.resolve()):
                raise OSError('fixture FreeType failure')
            return original_face(key, size)

        with patch.object(fonts, '_face', side_effect=load_or_fail):
            self.assertEqual(fonts.resolve_font('TOP', self.primary), self.fallback.resolve())

    def test_missing_all_fonts_has_actionable_error(self):
        with patch.object(fonts, 'BUNDLED_FONTS', ()):
            with self.assertRaisesRegex(fonts.FontCoverageError, '没有可加载的字体'):
                fonts.resolve_font('排球', self.root/'missing.ttf')

    def test_corrupt_all_fonts_has_actionable_error(self):
        corrupt = self.root/'corrupt.ttf'
        corrupt.write_bytes(b'not a loadable font')
        with patch.object(fonts, 'BUNDLED_FONTS', ()):
            with self.assertRaisesRegex(fonts.FontCoverageError, '没有可加载的字体'):
                fonts.resolve_font('排球', corrupt)

    def test_corrupt_font_fallback_does_not_leak_file_handles(self):
        corrupt = self.root/'corrupt.ttf'
        corrupt.write_bytes(b'not a loadable font')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ResourceWarning)
            self.assertEqual(fonts.resolve_font('排球', corrupt), self.fallback.resolve())
            gc.collect()
        leaks = [str(row.message) for row in caught if issubclass(row.category, ResourceWarning)]
        self.assertEqual(leaks, [])

    def test_unsupported_codepoint_is_rejected_without_rewriting_text(self):
        text = 'TOP 排球 🏐'
        with self.assertRaises(fonts.FontCoverageError) as caught:
            fonts.load_font(text, self.primary, 20)
        self.assertIn('U+1F3D0', str(caught.exception))
        self.assertIn('未替换或删除原文', str(caught.exception))
        self.assertEqual(text, 'TOP 排球 🏐')

    def test_union_coverage_is_not_mistaken_for_one_complete_face(self):
        cjk_only = make_font(self.root/'cjk-only.ttf', '中', family='CJK Only')
        with patch.object(fonts, 'BUNDLED_FONTS', (cjk_only,)):
            with self.assertRaisesRegex(fonts.FontCoverageError, '没有单一可用字体'):
                fonts.resolve_font('A中', self.primary)

    def test_empty_or_non_string_text_is_rejected(self):
        for text in ('', ' \t\n', '\u200c\u200d', '\ufe0f\U000e0100', None, 123):
            with self.subTest(text=text):
                with self.assertRaisesRegex(fonts.FontCoverageError, '文字不能为空'):
                    fonts.resolve_font(text, self.primary)

    def test_font_size_minimum_and_types(self):
        for size in (-1, 0, 3, 4.0, True, False, '16', None):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, '至少 4 的整数'):
                    fonts.load_font('A', self.primary, size)
        for size in (4, 16, 64):
            with self.subTest(size=size):
                self.assertEqual(fonts.load_font('A', self.primary, size).size, size)

    def test_serif_preference_selects_serif_fallback_before_sans(self):
        serif = make_font(self.root/'fallback-serif.ttf', '中A', family='Fixture Serif')
        with patch.object(fonts, 'BUNDLED_FONTS', (self.fallback, serif)):
            self.assertEqual(fonts.resolve_font('A中', self.root/'missing-serif.ttf'), serif.resolve())

    def test_collection_uses_sc_face_for_both_coverage_and_freetype(self):
        jp = make_font(self.root/'jp.ttf', 'A', family='Fixture CJK JP')
        sc = make_font(self.root/'sc.ttf', 'A中', family='Fixture CJK SC')
        collection = TTCollection()
        collection.fonts = [TTFont(jp), TTFont(sc)]
        path = self.root/'collection.ttc'
        try:
            collection.save(path)
        finally:
            collection.close()
        self.assertEqual(fonts.missing_glyphs('A中', path), frozenset())
        self.assertEqual(fonts.resolve_font('A中', path), path.resolve())
        self.assertEqual(fonts.load_font('A中', path, 20).getname()[0], 'Fixture CJK SC')

    def test_assets_include_code_and_existing_faces_once(self):
        missing = self.root/'missing.ttf'
        with patch.object(fonts, 'BUNDLED_FONTS', (self.fallback, self.primary)), \
                patch.object(fonts, 'SYSTEM_FONTS', (self.fallback, missing)):
            assets = fonts.font_assets(self.primary, missing, self.primary)
        self.assertEqual(assets, [Path(fonts.__file__), self.primary, self.fallback])

    def test_fingerprint_is_content_addressed_and_records_requested_font(self):
        requested = self.root/'missing.ttf'
        fingerprint = fonts.font_fingerprint(requested, self.primary)
        self.assertEqual(fingerprint, fonts.font_fingerprint(requested, self.primary))
        self.assertEqual(fingerprint['policy'], 'glyph_coverage_v1')
        self.assertEqual(fingerprint['requested'], [str(requested), str(self.primary)])
        by_path = {row['path']: row for row in fingerprint['assets']}
        self.assertNotIn(str(requested), by_path)
        for path in (Path(fonts.__file__), self.primary, self.fallback):
            self.assertEqual(by_path[str(path.resolve())]['sha256'],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(by_path[str(path.resolve())]['bytes'], path.stat().st_size)

    def test_same_path_content_replacement_invalidates_coverage_face_and_fingerprint(self):
        changing = make_font(self.root/'changing.ttf', 'AX', advance=500)
        before_stat = changing.stat()
        old_face = fonts.load_font('A', changing, 20)
        old_advance = old_face.getlength('A')
        before_fingerprint = fonts.font_fingerprint(changing)
        self.assertEqual(fonts.missing_glyphs('中X', changing), frozenset({'中'}))
        # Rewrite the same path, preserving mtime and byte count; ctime still
        # changes and must prevent stale cmap/FreeType entries being reused.
        make_font(changing, 'A中', advance=700)
        os.utime(changing, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        self.assertEqual(changing.stat().st_size, before_stat.st_size)
        self.assertEqual(changing.stat().st_mtime_ns, before_stat.st_mtime_ns)
        self.assertEqual(fonts.missing_glyphs('中X', changing), frozenset({'X'}))
        self.assertEqual(fonts.resolve_font('A中', changing), changing.resolve())
        new_face = fonts.load_font('A', changing, 20)
        self.assertIsNot(old_face, new_face)
        self.assertNotEqual(old_advance, new_face.getlength('A'))
        self.assertNotEqual(before_fingerprint, fonts.font_fingerprint(changing))

    def test_mtime_only_change_reloads_face_but_keeps_content_fingerprint(self):
        before = fonts.font_fingerprint(self.primary)
        old_face = fonts.load_font('A', self.primary, 20)
        stat = self.primary.stat()
        os.utime(self.primary, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        self.assertIsNot(old_face, fonts.load_font('A', self.primary, 20))
        self.assertEqual(before, fonts.font_fingerprint(self.primary))

    def test_atomic_same_path_replacement_invalidates_cached_face(self):
        old_face = fonts.load_font('A', self.primary, 20)
        replacement = make_font(self.root/'replacement.ttf', 'A中', family='Replacement')
        os.replace(replacement, self.primary)
        self.assertEqual(fonts.resolve_font('A中', self.primary), self.primary.resolve())
        new_face = fonts.load_font('A中', self.primary, 20)
        self.assertEqual(new_face.getname()[0], 'Replacement')
        self.assertIsNot(new_face, old_face)

    def test_fallback_content_changes_are_in_the_fingerprint(self):
        before = fonts.font_fingerprint(self.primary)
        make_font(self.fallback, 'A中', family='Changed Fallback')
        self.assertNotEqual(before, fonts.font_fingerprint(self.primary))

    def test_combining_marks_require_coverage_but_shaping_controls_do_not(self):
        # This tests glyph coverage only, not emoji composition or shaping.
        controlled = 'A\ufe0f\u200dB\u200c\U000e0100'
        self.assertEqual(fonts.missing_glyphs(controlled, self.primary), frozenset())
        self.assertEqual(fonts.resolve_font(controlled, self.primary), self.primary.resolve())
        # Q + acute has no NFC precomposed equivalent: its combining mark
        # remains an actual required glyph rather than a shaping control.
        self.assertEqual(fonts.missing_glyphs('Q\u0301', self.primary), frozenset({'\u0301'}))
        accented = make_font(self.root/'accented.ttf', 'Q\u0301', family='Combining Fixture')
        with patch.object(fonts, 'BUNDLED_FONTS', (accented,)):
            self.assertEqual(fonts.resolve_font('Q\u0301', self.primary), accented.resolve())
        with self.assertRaisesRegex(fonts.FontCoverageError, r'U\+0301'):
            fonts.resolve_font('Q\u0301', self.primary)

    def test_nfc_equivalent_titles_share_coverage_without_mutating_the_title(self):
        accented = make_font(self.root/'precomposed.ttf', 'Á', family='Precomposed Fixture')
        item = {'title': 'A\u0301'}
        original = dict(item)
        for text in ('Á', item['title']):
            with self.subTest(text=text):
                self.assertEqual(fonts.missing_glyphs(text, accented), frozenset())
                self.assertEqual(fonts.resolve_font(text, accented), accented.resolve())
                self.assertEqual(fonts.load_font(text, accented, 20).getname()[0],
                                 'Precomposed Fixture')
        self.assertEqual(item, original)
        self.assertEqual(item['title'].encode('utf-8'), b'A\xcc\x81')

    def test_parallel_font_choices_and_sizes_remain_isolated(self):
        cases = [('A TOP', self.primary, 16, 'Fixture Sans'),
                 ('A中', self.primary, 24, 'Fixture CJK'),
                 ('A', self.fallback, 40, 'Fixture CJK')] * 12

        def resolve(case):
            text, preferred, size, family = case
            path = fonts.resolve_font(text, preferred)
            face = fonts.load_font(text, preferred, size)
            return path, face.getname()[0], face.size

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(resolve, cases))
        for case, (path, family, size) in zip(cases, results):
            expected_path = self.primary if case[3] == 'Fixture Sans' else self.fallback
            self.assertEqual((path, family, size), (expected_path.resolve(), case[3], case[2]))

    def test_spacing_units_keep_marks_selectors_and_joined_text_together(self):
        self.assertEqual(fonts.text_units('Q\u0301中Cafe\u0301'), ['Q\u0301', '中', 'C', 'a', 'f', 'é'])
        self.assertEqual(fonts.text_units('中\U000e0100A\ufe0f\u200dB'), ['中\U000e0100', 'A\ufe0f\u200dB'])

    def test_control_whitespace_is_rendered_as_spaces_not_missing_glyphs(self):
        self.assertEqual(fonts.normalize_text('A\tB\u2003C\r\nD'), 'A B C\nD')
        self.assertEqual(fonts.resolve_font('A\tB', self.primary), self.primary.resolve())


if __name__ == '__main__':
    unittest.main()
