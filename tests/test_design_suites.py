"""Art direction is an immutable, auditable rendering choice, not ML evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from volleymole.common import DEFAULT_FONT, digest, read_json, save_json
from volleymole.design_suites import SUITES, SUITE_IDS, EXTRA_ART, SuiteCard, get_suite, resolve_design, palette, suite_headers, suite_overlay
from volleymole.illustrated import asset_paths, headers
from volleymole.presentation import lively_title


class DesignSuiteTests(unittest.TestCase):
    def test_five_complete_identities_and_custom_compatibility(self):
        self.assertEqual(len(SUITES),5)
        self.assertEqual(resolve_design('custom','zh','ink','cinema-en','film'),('ink','cinema-en','film'))
        for s in SUITES:
            for lang in ('zh','en'):
                self.assertEqual(resolve_design(s.name,lang), (s.art,s.template(lang),s.transition))
            self.assertEqual(palette(s.art,s.name).colors[0],s.accent)
        for bad in ('../ink','',None):
            with self.assertRaises(ValueError):get_suite(bad)

    def test_generated_glass_art_preserves_alpha_and_provenance(self):
        record=read_json(EXTRA_ART.parent/'prompts.json')['assets'][0]
        self.assertEqual(digest(EXTRA_ART),record['sha256'])
        with Image.open(EXTRA_ART) as image:
            self.assertEqual(image.mode,'RGBA')
            self.assertEqual(image.getchannel('A').getextrema(),(0,255))
        self.assertIn(EXTRA_ART,asset_paths('clay','minimal-en','prism','aurora'))
        self.assertNotIn(EXTRA_ART,asset_paths('manga','arena-zh','velocity','matchday'))

    def test_title_regions_are_static_and_hero_motion_is_bounded(self):
        item={'rank':10,'title':'长回合起跳对抗与防守站位'}
        for s in SUITES:
            for lang in ('zh','en'):
                scene=SuiteCard(item,lively_title(item,s.template(lang)),10,s.name,lang)
                base=scene.frame(45)
                self.assertEqual(base.shape,(1280,720,3))
                for index in (11,15,30,60,75,78):
                    pixels=scene.frame(index)
                    np.testing.assert_array_equal(base[:480],pixels[:480])
                    np.testing.assert_array_equal(base[1060:],pixels[1060:])
                self.assertFalse(np.array_equal(scene.frame(15),scene.frame(75)))

    def test_headers_have_distinct_ranks_and_clear_live_area(self):
        for s in SUITES:
            for lang in ('zh','en'):
                for k in (5,10):
                    results=[]
                    for rank in range(1,k+1):
                        layer=suite_headers({'rank':rank},k,'VOLLEYBALL',s.name,lang)[-1]
                        self.assertGreater(float(np.mean(layer[:,:,3]==0)),.40)
                        results.append(layer[30:80,50:260].tobytes())
                    self.assertEqual(len(set(results)),k)

    def test_overlay_replay_stays_transparent(self):
        with tempfile.TemporaryDirectory() as temp:
            for s in SUITES:
                for lang in ('zh','en'):
                    path=Path(temp)/f'{s.name}-{lang}.png'
                    suite_overlay(path,'replay',0,s.name,lang)
                    with Image.open(path) as image:
                        alpha=np.asarray(image)[:,:,3]
                    self.assertGreater(float(np.mean(alpha[124:194,22:387]==0)),.6)
                    for x,y in ((370,160),(200,190),(50,190)):self.assertEqual(alpha[y,x],0)

    def test_worker_recovers_suite_and_language_with_explicit_override(self):
        from volleymole.media_worker import render
        with tempfile.TemporaryDirectory() as temp,patch('volleymole.presentation.render_lively') as mocked:
            root=Path(temp)
            save_json(root/'run_config.json',{'design_suite':'aurora','design_language':'en'})
            render(root,DEFAULT_FONT,'lively')
            self.assertEqual(mocked.call_args.kwargs['design_suite'],'aurora')
            self.assertEqual(mocked.call_args.kwargs['title_template'],'minimal-en')
            self.assertEqual(mocked.call_args.kwargs['transition_style'],'prism')
            render(root,DEFAULT_FONT,'lively',design_suite='matchday',design_language='zh')
            self.assertEqual(mocked.call_args.kwargs['title_template'],'arena-zh')
            render(root,DEFAULT_FONT,'lively',design_suite='custom',art_theme='ink',title_template='cinema-zh')
            self.assertEqual(mocked.call_args.kwargs['design_suite'],'custom')
            with self.assertRaises(ValueError):render(root,DEFAULT_FONT,'classic',design_suite='sumi')

    def test_parallel_suites_are_isolated(self):
        item={'rank':1,'title':'二传组织'}
        def render(s):return SuiteCard(item,lively_title(item,s.template()),5,s.name).frame(15).tobytes()
        expected=list(map(render,SUITES))
        with ThreadPoolExecutor(max_workers=3) as pool:self.assertEqual(expected,list(pool.map(render,SUITES)))


if __name__=='__main__':unittest.main()
