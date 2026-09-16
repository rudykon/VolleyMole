"""Real installed-font regressions for all bilingual presentation templates."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from volleymole.common import APP
from volleymole.font_support import FontCoverageError, resolve_font, missing_glyphs, validate_render_fonts
from volleymole.title_templates import TEMPLATE_IDS, get_template, text_layer, template_assets, headline
from volleymole.design_suites import SUITES, SuiteCard, suite_headers
from volleymole.presentation import build_timeline, lively_headers


@unittest.skipUnless((APP/'assets/fonts/NotoSansCJKsc-Bold.otf').is_file(), 'Local presentation fonts required')
class TitleFontRecoveryTests(unittest.TestCase):
    def test_every_template_renders_distinct_chinese_glyphs_and_mixed_text(self):
        for name in TEMPLATE_IDS[1:]:
            with self.subTest(template=name):
                left=text_layer('趣',48,'white',300,name)
                right=text_layer('球',48,'white',300,name)
                self.assertNotEqual((left.size,left.tobytes()),(right.size,right.tobytes()))
                text='TOP 10 / 趣味，Café 龘𠮷 0.67×'
                path=resolve_font(text,get_template(name).font)
                self.assertFalse(missing_glyphs(text,path))
                self.assertIn(path, [p.resolve() for p in template_assets(name)])
                layer=text_layer(text,48,'white',600,name,outline='black',tracking=1.5)
                self.assertLessEqual(layer.width,600)
                box=layer.getchannel('A').getbbox()
                self.assertGreater(box[0],0);self.assertGreater(box[1],0)
                self.assertLess(box[2],layer.width);self.assertLess(box[3],layer.height)

    def test_all_english_suites_preserve_blooper_title_and_use_covering_fonts(self):
        item={'rank':1,'rally_id':'r1','title':'追球后摔倒，Café','clip_start_sec':0.,'clip_end_sec':3.}
        decision={'collection':'bloopers','selected':[item]}
        before=copy.deepcopy(decision)
        manifest={'rallies':[{'rally_id':'r1','preview_times_sec':[0.,1.,2.]}]}
        for suite in SUITES:
            with self.subTest(suite=suite.name):
                name=suite.template('en')
                timeline=build_timeline(decision,manifest,name,suite.transition)
                title=next(s['display_title'] for s in timeline if s['kind']=='transition')
                self.assertEqual(title,item['title'])
                audit=validate_render_fonts(decision,APP/'assets/fonts/NotoSans-Bold.ttf','lively',name)
                self.assertTrue(audit['titles'][0]['fallback'])
                self.assertFalse(missing_glyphs(title,audit['titles'][0]['resolved_font']))
                card=SuiteCard({**item,'collection':'bloopers'},title,1,suite.name,'en').image()
                self.assertEqual(card.size,(720,1280))
                frames=lively_headers(item,APP/'assets/fonts/NotoSans-Bold.ttf',1,suite.art,name,suite.name,'bloopers')
                expected=suite_headers({**item,'collection':'bloopers'},1,title,suite.name,'en')
                self.assertEqual(frames[-1].tobytes(),expected[-1].tobytes())
        self.assertEqual(decision,before)

    def test_pure_english_keeps_original_design_faces(self):
        for name in TEMPLATE_IDS[1:]:
            if name.endswith('-en'):
                primary=get_template(name).font
                self.assertEqual(resolve_font('BACK AND FORTH',primary),primary.resolve())

    def test_missing_template_primary_still_has_a_valid_fallback_asset_set(self):
        with tempfile.TemporaryDirectory() as temp:
            absent=Path(temp)/'missing-font.ttf'
            with patch('volleymole.title_templates.FONTS',Path(temp)):
                layer=text_layer('追球后摔倒',48,'white',600,'arena-en',font_path=absent)
                assets=template_assets('arena-en')
            self.assertIsNotNone(layer.getbbox())
            self.assertNotIn(absent,assets)
            self.assertTrue(any(p.suffix=='.otf' for p in assets))

    def test_unavailable_glyph_cannot_pass_title_preflight_or_render_as_tofu(self):
        title='排球\U0001f3d0'
        decision={'collection':'bloopers','selected':[{'rank':1,'title':title}]}
        with self.assertRaises(FontCoverageError):
            validate_render_fonts(decision,APP/'assets/fonts/NotoSans-Bold.ttf','lively','arena-en')
        for name in TEMPLATE_IDS[1:]:
            with self.subTest(template=name),self.assertRaises(FontCoverageError):
                text_layer(title,48,'white',500,name)

    def test_combining_accent_has_same_pixels_as_canonical_equivalent(self):
        for name in ('editorial-en','cinema-en','arena-en'):
            a=text_layer('Cafe\u0301 排球',48,'white',500,name,tracking=1.5)
            b=text_layer('Café 排球',48,'white',500,name,tracking=1.5)
            self.assertEqual((a.size,a.tobytes()),(b.size,b.tobytes()))

    def test_headline_normalizes_accents_before_wrapping(self):
        for name in TEMPLATE_IDS[1:]:
            a=headline('排球AAe\u0301排球AA',name,'white','red','black')
            b=headline('排球AAé排球AA',name,'white','red','black')
            self.assertEqual(a.tobytes(),b.tobytes(),name)

    def test_tracking_does_not_separate_an_uncomposable_accent(self):
        from PIL import ImageDraw
        drawn=[]
        original=ImageDraw.ImageDraw.text
        def record(draw,position,text,*args,**kwargs):
            drawn.append(text)
            return original(draw,position,text,*args,**kwargs)
        with patch.object(ImageDraw.ImageDraw,'text',record):
            layer=text_layer('Q\u0301球',48,'white',500,'cinema-en',tracking=5)
        self.assertEqual(drawn,['Q\u0301','球'])
        self.assertIsNotNone(layer.getbbox())


if __name__=='__main__':unittest.main()
