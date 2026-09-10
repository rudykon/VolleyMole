"""Pixel-level checks for the code-native replay caption, not the AI artwork."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from illustrated import overlay,headers,composite_header


@unittest.skipUnless(importlib.util.find_spec('PIL'),'Run with the tracking Python for Pillow image tests')
class ReplayOverlayTests(unittest.TestCase):
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

    def test_replay_caption_has_no_background_plate(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'replay.png'
            overlay(path,'replay',self.font)
            with Image.open(path) as pixels:
                self.assertEqual(pixels.mode,'RGBA')
                alpha=pixels.getchannel('A')
                for point in ((370,160),(200,190),(50,190),(400,170),(500,300)):
                    self.assertEqual(alpha.getpixel(point),0,point)
                region=alpha.crop((22,124,387,194))
                self.assertGreater(region.histogram()[0]/(region.width*region.height),.6)
                self.assertEqual(alpha.crop((0,0,720,110)).getextrema(),(0,0))
                self.assertEqual(alpha.getpixel((1,112)),255)  # Replay frame retained.

    def test_replay_keeps_text_and_only_local_dark_outlines(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'replay.png'
            overlay(path,'replay',self.font)
            with Image.open(path) as pixels:
                for box in ((29,125,222,190),(225,140,325,188)):
                    region=pixels.crop(box)
                    colors=[region.getpixel((x,y)) for y in range(region.height) for x in range(region.width)]
                    self.assertTrue(any(a>0 and r>230 and g>170 for r,g,b,a in colors))
                    self.assertTrue(any(a>0 and r<40 and g<40 and b<60 for r,g,b,a in colors))


@unittest.skipUnless(importlib.util.find_spec('PIL'),'Run with the tracking Python for image tests')
class TransparentHeaderTests(unittest.TestCase):
    font=ReplayOverlayTests.font

    def test_all_five_and_ten_rank_headers_have_no_background_bar(self):
        import numpy as np
        for k in (5,10):
            for rank in range(1,k+1):
                layers=headers({'rank':rank,'title':'排球好球'},self.font,k,'好球时刻，一起看！')
                for layer in (layers[0],layers[-1]):
                    self.assertEqual(layer.shape,(110,720,4))
                    self.assertGreater(float(np.mean(layer[:,:,3]==0)),.45)
                    self.assertEqual(int(layer[5,5,3]),0)
                    self.assertEqual(int(layer[109,500,3]),0)

    def test_alpha_composite_preserves_live_pixels_and_opaque_rank_art(self):
        import numpy as np
        layer=headers({'rank':5,'title':'极低接球'},self.font,5,'贴地救球！太拼了')[-1]
        clear=layer[:,:,3]==0;solid=layer[:,:,3]==255
        for color in ((240,240,240),(25,120,200)):
            background=np.full((110,720,3),color,np.uint8)
            result=composite_header(background,layer)
            self.assertTrue(np.array_equal(result[clear],background[clear]))
            self.assertTrue(np.array_equal(result[solid],layer[:,:,:3][solid]))
            self.assertTrue(np.array_equal(background,np.full((110,720,3),color,np.uint8)))
        with self.assertRaises(ValueError):composite_header(background[:100],layer)


if __name__=='__main__':
    unittest.main()
