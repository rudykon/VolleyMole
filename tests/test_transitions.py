"""Pixel-level transition invariants; real codecs covered by preview exporter."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from volleymole.common import DEFAULT_FONT, digest, read_json, save_json
from volleymole.illustrated import asset_paths, transition_opacity
from volleymole.transitions import ASSET_ROOT, STYLE_IDS, TransitionRenderer, transition_assets, validate_style


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.before = np.full((160, 90, 3), (20, 90, 170), dtype=np.uint8)
        self.card = np.full_like(self.before, (190, 230, 250))
        self.after = np.full_like(self.before, (200, 55, 30))

    def renderer(self, style):
        return TransitionRenderer(style, self.before, self.card, self.after)

    def test_registry_and_original_generated_plates(self):
        records = read_json(ASSET_ROOT/'prompts.json')['assets']
        self.assertEqual(len(STYLE_IDS), 6)
        hashes = set()
        for row in records:
            path = ASSET_ROOT/row['file']
            self.assertIn(path, asset_paths('manga', 'arena-en', row['id']))
            self.assertTrue(row['prompt'])
            self.assertEqual(digest(path), row['sha256'])
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, 720)
                self.assertGreaterEqual(image.height, 1280)
            hashes.add(digest(path))
        self.assertEqual(len(hashes), 5)
        for bad in ('../ink', '', None):
            with self.assertRaises(ValueError): validate_style(bad)

    def test_first_last_frames_and_entire_title_hold_are_exact(self):
        for style in STYLE_IDS:
            renderer = self.renderer(style)
            np.testing.assert_array_equal(renderer.frame(0), self.before)
            np.testing.assert_array_equal(renderer.frame(89), self.after)
            for index in range(11, 79):
                np.testing.assert_array_equal(renderer.frame(index), self.card)
            for index in (-1, 90):
                with self.assertRaises(ValueError): renderer.frame(index)

    def test_default_fade_is_bitwise_compatible(self):
        renderer = self.renderer('fade')
        for index in range(90):
            opacity = transition_opacity(index)
            base = self.before if index < 45 else self.after
            expected = cv2.addWeighted(self.card, opacity, base, 1-opacity, 0)
            np.testing.assert_array_equal(renderer.frame(index), expected)

    def test_five_gestures_are_distinct_and_deterministic(self):
        results = []
        for style in STYLE_IDS[1:]:
            renderer = self.renderer(style)
            frame = renderer.frame(4)
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(frame.shape, self.card.shape)
            np.testing.assert_array_equal(frame, self.renderer(style).frame(4))
            self.assertFalse(np.array_equal(frame, renderer.frame(7)))
            results.append(frame.tobytes())
        self.assertEqual(len(set(results)), 5)

    def test_external_overlay_transparent_ends_and_opaque_cut(self):
        for style in STYLE_IDS[1:]:
            renderer = self.renderer(style)
            for reverse in (False, True):
                for p in (0., 1.):
                    self.assertEqual(int(renderer.material_layer(p, reverse)[:, :, 3].max()), 0)
                for index in (11, 12):
                    self.assertEqual(int(renderer.material_layer(index/23, reverse)[:, :, 3].min()), 255)

    def test_no_mutation_of_inputs_or_assets(self):
        hashes = [digest(transition_assets(s)[1]) for s in STYLE_IDS[1:]]
        original = tuple(a.copy() for a in (self.before, self.card, self.after))
        for style in STYLE_IDS:
            renderer = self.renderer(style)
            for i in (0, 3, 6, 15, 75, 85, 89): renderer.frame(i)[:] = 0
        for actual, expected in zip((self.before, self.card, self.after), original):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(hashes, [digest(transition_assets(s)[1]) for s in STYLE_IDS[1:]])

    def test_saved_config_and_explicit_override(self):
        from volleymole.media_worker import render
        with tempfile.TemporaryDirectory() as temp, patch('volleymole.presentation.render_lively') as mocked:
            directory = Path(temp)
            render(directory, DEFAULT_FONT, 'lively')
            self.assertEqual(mocked.call_args.kwargs['transition_style'], 'fade')
            save_json(directory/'run_config.json', {'transition_style': 'ink'})
            render(directory, DEFAULT_FONT, 'lively')
            self.assertEqual(mocked.call_args.kwargs['transition_style'], 'ink')
            render(directory, DEFAULT_FONT, 'lively', transition_style='prism')
            self.assertEqual(mocked.call_args.kwargs['transition_style'], 'prism')
            with self.assertRaises(ValueError): render(directory, DEFAULT_FONT, 'classic', transition_style='ink')


if __name__ == '__main__': unittest.main()
