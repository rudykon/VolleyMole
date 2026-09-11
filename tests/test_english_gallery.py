"""English suite previews use existing identities with fully localized labels."""
import importlib.util
from pathlib import Path
import sys
import unittest
import shutil
import tempfile
import numpy as np
from volleymole.design_suites import SUITES,resolve_design
from volleymole.title_templates import words,display_title,get_template
from volleymole.illustrated import rank_label


class EnglishGalleryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'),'requires FFmpeg')
    def test_studio_boundary_encodes_native_output_and_normalizes_inspection(self):
        from volleymole.common import probe
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        spec=importlib.util.spec_from_file_location('preview_transitions',scripts/'preview_transitions.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'boundary.mp4'
            frame=np.full((1280,720,3),128,np.uint8)
            module.encode_frames(path,[frame,frame],2,quality='1080p')
            meta=probe(path)
            self.assertEqual((meta['width'],meta['height']),(1080,1920))
            decoded=module.decode(path,normalize=True)
            self.assertEqual(decoded.shape,(2,1280,720,3))

    def test_all_five_suites_have_english_type_and_labels(self):
        self.assertEqual(len(SUITES),5)
        keys=('teaser','teaser_sub','teaser_bottom','replay','first','last','next','footer','caption')
        for s in SUITES:
            art,title,transition=resolve_design(s.name,'en')
            self.assertEqual((art,transition),(s.art,s.transition))
            self.assertEqual(get_template(title).language,'en')
            self.assertTrue(get_template(title).font.is_file())
            for key in keys:self.assertTrue(words(title,key).isascii())
            for rank in range(1,11):self.assertTrue(rank_label(rank,10,title).isascii())
            for raw in ('长回合起跳对抗与防守站位','极低姿态接球防守','二传组织','多拍攻防'):
                self.assertTrue(display_title({'title':raw},title).isascii())

    def test_english_only_gallery_is_visible_on_open(self):
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        sys.path.insert(0,str(scripts))
        try:
            spec=importlib.util.spec_from_file_location('preview_design_suites',scripts/'preview_design_suites.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:sys.path.pop(0)
        page='<div class="toolbar"><button class="active" data-lang="zh">中文</button><button data-lang="en">English</button></div><article data-language="en" hidden><video></video></article>'
        converted=module.select_gallery_language(page,'en')
        self.assertNotIn(' hidden',converted)
        self.assertNotIn('data-lang="zh"',converted)
        self.assertIn('class="active" data-lang="en"',converted)
        self.assertEqual(module.select_gallery_language(page,'both'),page)
