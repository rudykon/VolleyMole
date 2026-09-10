"""Actual FFmpeg/PyAV timeline fixtures, independent of neural weights."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import cv2
import numpy as np

from volleymole.common import probe
from volleymole.video import decode


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'requires FFmpeg')
class DisplayTimestampTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='volleymole-pts-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Matroska avoids FFmpeg 7's MP4 edit-list EOF discard on this artificial
        # sparse-frame stream. Real MP4 frame-count mismatches must still fail closed.
        self.video = self.root/'vfr-audio-offset.mkv'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=128x96:rate=20:duration=1',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1.2',
            '-vf',"select='eq(n,0)+eq(n,1)+eq(n,4)+eq(n,7)+eq(n,8)+eq(n,15)',setpts=PTS+0.2/TB",
            '-fps_mode','passthrough','-c:v','libx264','-preset','ultrafast','-threads','1',
            '-c:a','aac',str(self.video)],check=True,capture_output=True)

    def test_vfr_and_audio_start_offset_preserve_source_pts(self):
        frames = list(decode(self.video))
        reference = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',str(self.video)]))
        expected = [float(f['best_effort_timestamp_time']) for f in reference['frames']]
        self.assertEqual(len(frames),6)
        np.testing.assert_allclose([f.source_sec for f in frames],expected,atol=1e-6)
        self.assertGreater(len(set(np.round(np.diff(expected),3))),1)
        # AAC priming may shift the Matroska mux clock by one audio packet.
        # Compare actual source clocks rather than assuming the mux has no delay.
        self.assertAlmostEqual(frames[0].time_sec,expected[0]-probe(self.video)['start_sec'],places=6)
        self.assertGreater(frames[0].time_sec,.19)
        self.assertNotEqual(frames[0].time_sec,0)
        self.assertEqual(len(list(decode(self.video,max_frames=2))),2)

    def test_display_rotation_matches_ffmpeg_actual_pixels(self):
        cfr = self.root/'cfr.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=128x96:rate=20:duration=0.3',
            '-c:v','libx264','-preset','ultrafast','-threads','1',str(cfr)],check=True,capture_output=True)
        rotated = self.root/'rotated.mp4'
        subprocess.run(['ffmpeg','-v','error','-display_rotation','90','-i',str(cfr),'-c','copy',
            str(rotated)],check=True,capture_output=True)
        meta = probe(rotated)
        self.assertEqual(meta['rotation'],90)
        frames = list(decode(rotated))
        self.assertEqual(frames[0].pixels.shape[:2],(128,96))
        encoded = subprocess.check_output(['ffmpeg','-v','error','-i',str(rotated),'-frames:v','1',
            '-f','image2pipe','-vcodec','png','pipe:1'])
        reference = cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        self.assertLess(float(np.abs(reference.astype(float)-frames[0].pixels).mean()),1.)
