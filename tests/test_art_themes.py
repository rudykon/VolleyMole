"""Asset integrity, palette selection and isolated concurrent UI rendering."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from volleymole.art_themes import THEME_IDS, get_theme
from volleymole.common import DEFAULT_FONT, digest, read_json, save_json
from volleymole.illustrated import ART_NAMES, ART_ROOT, asset_paths, headers, illustration_for, rank_badge, title_card


class ArtThemeTests(unittest.TestCase):
    def test_default_is_backward_compatible_and_names_are_allowlisted(self):
        self.assertEqual(get_theme().root, ART_ROOT)
        self.assertEqual(len(THEME_IDS), 6)
        self.assertEqual(illustration_for({'title':'二传'}), ART_ROOT / 'volley_set.png')
        for name in ('../manga', '/tmp', '', None):
            with self.assertRaises(ValueError):
                get_theme(name)

    def test_all_thirty_five_new_assets_are_distinct_rgba_and_tracked_in_prompts(self):
        manifest=read_json(ART_ROOT / 'themes/prompts.json')
        recorded={row['file']:row for row in manifest['assets']}
        checksums=set()
        for name in THEME_IDS[1:]:
            for asset in ART_NAMES:
                path=get_theme(name).root / f'{asset}.png'
                with self.subTest(theme=name, asset=asset), Image.open(path) as image:
                    self.assertEqual(image.mode, 'RGBA')
                    self.assertEqual(image.getchannel('A').getextrema(), (0,255))
                    self.assertGreaterEqual(min(image.size), 640)
                    self.assertIn(path, asset_paths(name))
                    row=recorded[f'{name}/{asset}.png']
                    self.assertTrue(row['prompt'])
                    self.assertEqual(row['sha256'], digest(path))
                    checksums.add(digest(path))
        self.assertEqual(len(checksums), 35)
        self.assertEqual(len(recorded), 35)

    def test_every_rank_is_distinct_and_headers_remain_transparent(self):
        for name in THEME_IDS[1:]:
            for top_k in (5,10):
                badges=[]
                for rank in range(1,top_k+1):
                    badges.append(rank_badge(rank,top_k,290,name).tobytes())
                    layer=headers({'rank':rank,'title':'排球'},DEFAULT_FONT,top_k,'好球时刻，一起看！',name)[-1]
                    self.assertEqual(layer.shape,(110,720,4))
                    self.assertGreater(float(np.mean(layer[:,:,3]==0)),.45)
                self.assertEqual(len(set(badges)),top_k)

    def test_concurrent_themes_do_not_leak_palette_or_asset_selection(self):
        def render(name):
            return title_card({'rank':1,'title':'二传组织'},'配合到位，节奏拉满！',DEFAULT_FONT,5,name).tobytes()
        sequential=list(map(render,THEME_IDS))
        with ThreadPoolExecutor(max_workers=3) as pool:
            concurrent=list(pool.map(render,THEME_IDS))
        self.assertEqual(sequential,concurrent)
        self.assertEqual(len(set(sequential)),len(THEME_IDS))

    def test_render_worker_uses_saved_theme_and_explicit_override(self):
        from volleymole.media_worker import render
        with tempfile.TemporaryDirectory() as temp, patch('volleymole.presentation.render_lively') as renderer:
            directory=Path(temp)
            render(directory,DEFAULT_FONT,'lively')
            self.assertEqual(renderer.call_args.args[-1],'default')
            save_json(directory/'run_config.json',{'art_theme':'ink'})
            render(directory,DEFAULT_FONT,'lively',2)
            self.assertEqual(renderer.call_args.args[-2:],(2,'ink'))
            render(directory,DEFAULT_FONT,'lively',2,'clay')
            self.assertEqual(renderer.call_args.args[-1],'clay')
            with self.assertRaises(ValueError):
                render(directory,DEFAULT_FONT,'classic',1,'manga')


if __name__ == '__main__':
    unittest.main()
