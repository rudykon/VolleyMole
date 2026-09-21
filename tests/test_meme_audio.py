import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from volleymole.meme_audio import RATE, decode_audio, mix_cue, render, validate_plan, video_signature


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class MemeAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video, self.asset = self.root / 'original.mp4', self.root / 'voice.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        'testsrc2=size=160x240:rate=30:duration=3', '-f', 'lavfi', '-i',
                        'sine=frequency=330:sample_rate=48000:duration=3',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac',
                        '-ac', '2', str(self.video)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        'sine=frequency=880:sample_rate=48000:duration=0.6',
                        '-ac', '2', str(self.asset)], check=True)
        self.plan = {'version': 1, 'max_cues': 2, 'cues': [
            {'id': 'one', 'asset': 'voice.wav', 'at_sec': 1., 'source_end_sec': .4},
            {'id': 'two', 'asset': 'voice.wav', 'at_sec': 2., 'source_end_sec': .4}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_real_encode_keeps_video_timeline_and_adds_only_two_audio_cues(self):
        plan = self.root / 'plan.json'
        plan.write_text(json.dumps(self.plan))
        output = self.root / 'edition.mp4'
        report = render(self.video, plan, output)
        self.assertEqual(video_signature(self.video), video_signature(output))
        self.assertEqual(report['cue_count'], 2)
        arrays = []
        for name, path in [('before', self.video), ('after', output)]:
            pcm = self.root / f'{name}.f32'
            decode_audio(path, pcm, samples=3 * RATE)
            arrays.append(np.fromfile(pcm, dtype='<f4').reshape(-1, 2)[:, 0])
        before, after = arrays
        self.assertLess(float(np.max(np.abs(after))), .9)
        for start, end in [(.1, .85), (1.55, 1.85), (2.55, 2.85)]:
            a, b = [x[round(start * RATE):round(end * RATE)] for x in arrays]
            self.assertGreater(float(np.corrcoef(a, b)[0, 1]), .99)
            self.assertLess(float(np.sqrt(np.mean((a - b) ** 2))), .004)
        for start in [1.1, 2.1]:
            a, b = [x[round(start * RATE):round((start + .15) * RATE)] for x in arrays]
            frequency = np.sin(np.arange(len(a)) * 2 * np.pi * 880 / RATE)
            self.assertGreater(abs(float(b @ frequency)), abs(float(a @ frequency)) + 20)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            render(self.video, plan, output)

    def test_rejects_excess_density_overlap_and_truncated_phrases(self):
        cases = []
        plan = copy.deepcopy(self.plan)
        plan['cues'].append({**plan['cues'][0], 'id': 'third'})
        cases.append(plan)
        plan = copy.deepcopy(self.plan)
        plan['cues'][1]['at_sec'] = 1.1
        cases.append(plan)
        plan = copy.deepcopy(self.plan)
        plan['cues'][1]['at_sec'] = 2.8
        cases.append(plan)
        plan = copy.deepcopy(self.plan)
        plan['cues'][1]['source_end_sec'] = 1.
        cases.append(plan)
        plan = copy.deepcopy(self.plan)
        plan['cues'][1]['at_sec'] = float('nan')
        cases.append(plan)
        for plan in cases:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_plan(plan, self.root, 3.)

    def test_loud_cue_reduces_gain_instead_of_clipping_and_fades_to_original(self):
        background = np.full((RATE, 2), .8, dtype=np.float32)
        sound = np.zeros_like(background)
        sound[::4] = 1
        mixed, report = mix_cue(background, sound, -14, 0)
        self.assertLessEqual(float(np.max(mixed)), 10 ** (-1 / 20) + 1e-6)
        self.assertLess(report['additional_headroom_reduction_db'], 0)
        np.testing.assert_array_equal(mixed[0], background[0])
        np.testing.assert_array_equal(mixed[-1], background[-1])


if __name__ == '__main__':
    unittest.main()
