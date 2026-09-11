import tempfile
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
from PIL import Image
from volleymole.common import DEFAULT_FONT,save_json
from volleymole.illustrated import headers,rank_label,title_card,asset_paths,overlay
from volleymole.presentation import lively_title,build_timeline,validate_timeline
from volleymole.title_templates import TEMPLATE_IDS,get_template,headline,template_assets,words


class TitleTemplateTests(unittest.TestCase):
    def test_ten_bilingual_templates_and_packaged_fonts(self):
        self.assertEqual(len(TEMPLATE_IDS),11)
        for name in TEMPLATE_IDS[1:]:
            for path in template_assets(name):self.assertTrue(path.is_file(),path)
            self.assertIn(get_template(name).font,asset_paths('manga',name))
        for bad in ('editorial','en','../cinema-en',None):
            with self.assertRaises(ValueError):get_template(bad)

    def test_english_titles_and_ui_copy_contain_no_chinese_or_invented_results(self):
        for name in TEMPLATE_IDS[1:]:
            for raw in ('极低防守','低姿接球','低位防守','二传组织','长回合起跳','往返攻防','排球'):
                text=lively_title({'title':raw},name)
                self.assertEqual(len(text.splitlines()),2)
                self.assertNotIn('获胜',text)
                self.assertNotIn('得分',text)
                if name.endswith('-en'):self.assertFalse(any('\u4e00'<=c<='\u9fff' for c in text))
            for k in (5,10):
                for rank in range(1,k+1):
                    label=rank_label(rank,k,name)
                    if name.endswith('-en'):self.assertEqual(label,f'TOP {k} / NO. {rank:02d}')
        self.assertEqual(lively_title({'title':'极低防守'}),'贴地救球！太拼了')

    def test_all_rank_headers_remain_transparent_and_distinct(self):
        for name in TEMPLATE_IDS[1:]:
            for k in (5,10):
                rows=[]
                for rank in range(1,k+1):
                    layer=headers({'rank':rank,'title':'排球'},DEFAULT_FONT,k,'ON COURT','manga',name)[-1]
                    self.assertEqual(layer.shape,(110,720,4))
                    self.assertGreater(float(np.mean(layer[:,:,3]==0)),.45)
                    rows.append(layer[:,:290].tobytes())
                self.assertEqual(len(set(rows)),k)

    def test_long_headlines_fit_and_templates_are_visually_distinct(self):
        cards=[]
        for name in TEMPLATE_IDS[1:]:
            item={'rank':10,'title':'持续攻防与二传组织配合'}
            title=lively_title(item,name)
            block=headline(title,name,(20,28,35),(245,180,60),(255,248,232))
            self.assertEqual(block.size,(608,260))
            bbox=block.getchannel('A').getbbox()
            self.assertIsNotNone(bbox)
            self.assertGreater(bbox[0],0)
            self.assertLess(bbox[2],608)
            self.assertLess(bbox[3],260)
            card=title_card(item,title,DEFAULT_FONT,10,'manga',name)
            self.assertEqual(card.size,(720,1280))
            cards.append(card.tobytes())
        self.assertEqual(len(set(cards)),10)

    def test_concurrent_templates_are_isolated(self):
        def render(name):
            item={'rank':1,'title':'极低救球'}
            return title_card(item,lively_title(item,name),DEFAULT_FONT,5,'ink',name).tobytes()
        names=('editorial-zh','cinema-en','arena-en','minimal-zh')
        expected=list(map(render,names))
        with ThreadPoolExecutor(max_workers=2) as pool:self.assertEqual(list(pool.map(render,names)),expected)

    def test_saved_template_and_override_reach_renderer(self):
        from volleymole.media_worker import render
        with tempfile.TemporaryDirectory() as temp,patch('volleymole.presentation.render_lively') as target:
            directory=Path(temp);save_json(directory/'run_config.json',{'title_template':'cinema-en'})
            render(directory,DEFAULT_FONT,'lively')
            self.assertEqual(target.call_args.kwargs['title_template'],'cinema-en')
            render(directory,DEFAULT_FONT,'lively',title_template='pop-zh')
            self.assertEqual(target.call_args.kwargs['title_template'],'pop-zh')
            with self.assertRaises(ValueError):render(directory,DEFAULT_FONT,'classic',title_template='pop-en')

    def test_replay_plate_remains_transparent_in_both_languages(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in TEMPLATE_IDS[1:]:
                path=Path(temp)/f'{name}.png'
                overlay(path,'replay',DEFAULT_FONT,0,'manga',name)
                with Image.open(path) as image:
                    self.assertEqual(image.mode,'RGBA')
                    alpha=np.asarray(image)[:,:,3]
                    self.assertGreater(float(np.mean(alpha[124:194,22:387]==0)),.6)

    def test_english_timeline_validation_uses_report_template(self):
        from test_presentation import PresentationTests
        fixture=PresentationTests()
        fixture.setUp()
        try:
            decision,report=fixture.timeline(10)
            placeholder=report['segments'][0]['path']
            report['segments']=build_timeline(decision,fixture.manifest,'editorial-en')
            report['title_template']='editorial-en'
            for row in report['segments']:
                row['path']=placeholder
                self.assertTrue(row['rank_label'].startswith('TOP 10'))
            validate_timeline(report,decision)
            report['title_template']='editorial-zh'
            with self.assertRaises(ValueError):validate_timeline(report,decision)
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
