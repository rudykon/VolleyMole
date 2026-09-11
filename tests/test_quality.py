"""Quality selection does not change editorial timing or legacy verification."""
import tempfile
import unittest
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch
from volleymole.quality import DEFAULT_QUALITY,PRESETS,get_quality,report_dimensions
from volleymole.common import save_json,DEFAULT_FONT
from volleymole.media_worker import render


class QualityTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(DEFAULT_QUALITY,'1080p')
        self.assertEqual((get_quality().width,get_quality().height),(1080,1920))
        self.assertEqual([q.width for q in PRESETS],[720,1080,1440,2160])
        for q in PRESETS:
            self.assertEqual(q.width*16,q.height*9)
            self.assertEqual(q.width%2+q.height%2,0)
            self.assertEqual(report_dimensions({'output_quality':q.report()}),(q.width,q.height))
        self.assertEqual(sorted([q.crf for q in PRESETS],reverse=True),[q.crf for q in PRESETS])
        self.assertEqual(report_dimensions({}),(720,1280))
        with self.assertRaises(ValueError):get_quality('8k')

    def test_render_default_saved_config_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('volleymole.presentation.render_lively') as lively:
                render(root,DEFAULT_FONT,'lively')
                self.assertEqual(lively.call_args.kwargs['quality'],'1080p')
                save_json(root/'run_config.json',{'quality':'2160p'})
                render(root,DEFAULT_FONT,'lively')
                self.assertEqual(lively.call_args.kwargs['quality'],'2160p')
                render(root,DEFAULT_FONT,'lively',quality='720p')
                self.assertEqual(lively.call_args.kwargs['quality'],'720p')

    def test_invalid_quality_fails_before_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):render(Path(tmp),DEFAULT_FONT,quality='bad')

    @unittest.skipUnless(shutil.which('ffmpeg'),'requires FFmpeg')
    def test_header_reference_uses_output_sampling_and_preserves_alpha(self):
        from volleymole.check_title_cards import normalized_header
        layer=np.zeros((110,720,4),np.uint8)
        layer[20:80,50:260]=[30,90,230,255]
        self.assertIs(normalized_header(layer,720),layer)
        for q in PRESETS[1:]:
            result=normalized_header(layer,q.width)
            self.assertEqual(result.shape,layer.shape)
            np.testing.assert_array_equal(result[40,100],layer[40,100])
            self.assertEqual(int(result[0,0,3]),0)
