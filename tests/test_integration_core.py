"""Core integration contracts; neural model execution is a separate real-video gate."""
import ast
from fractions import Fraction
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import numpy as np

from volleymole.common import APP, save_json
from volleymole.models import ModelRegistry
from volleymole.video import FramePacket, chunks
from volleymole.state_model import uniform_indices
from volleymole.jersey import JerseyReader, digit_candidate, torso_box
from volleymole.tracker import BallTracker, seq9_shape


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'known-model'
        self.spec = self.root/'manifest.json'
        save_json(self.spec, {'models': {'test': {'path':'test.onnx', 'bytes':len(self.data),
            'sha256':hashlib.sha256(self.data).hexdigest()}}, 'downloads':{}})
        self.registry = ModelRegistry(self.root/'models', self.spec)

    def test_hash_verified_before_atomic_replace(self):
        with self.assertRaises(FileNotFoundError):
            self.registry.verify()
        self.registry.install('test', io.BytesIO(self.data))
        self.assertEqual(self.registry.verify()['test']['sha256'], hashlib.sha256(self.data).hexdigest())
        for bad in (b'bad-model!!', b'oversized-model-content'):
            with self.assertRaises(ValueError):
                self.registry.install('test', io.BytesIO(bad))
            self.assertEqual(self.registry.target('test').read_bytes(), self.data)
        self.assertFalse(list(self.registry.directory.glob('*.partial')))

    def test_reject_tampered_and_path_escape(self):
        self.registry.install('test', io.BytesIO(self.data))
        self.registry.target('test').write_bytes(b'x'*len(self.data))
        with self.assertRaises(ValueError):
            self.registry.verify()
        self.registry.entries['test']['path'] = '../escape.onnx'
        with self.assertRaises(ValueError):
            self.registry.target('test')

    def test_zip_fetch_reads_only_pinned_member_without_extractall(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data,'w') as archive:
            archive.writestr('safe/test.bin',self.data)
            archive.writestr('../escape.bin',b'must not be extracted')
        payload = data.getvalue()
        self.registry.entries['test'].update(download='bundle',member='safe/test.bin')
        self.registry.manifest['downloads']['bundle'] = {
            'url':'https://example.invalid/pinned.zip','format':'zip','max_bytes':4096,
            'sha256':hashlib.sha256(payload).hexdigest()}
        with patch('volleymole.models.shutil.which',return_value=None), \
                patch('volleymole.models.urlopen',return_value=io.BytesIO(payload)):
            self.registry.fetch(['test'])
        self.assertEqual(self.registry.target('test').read_bytes(),self.data)
        self.assertFalse((self.root/'escape.bin').exists())


class PacketTests(unittest.TestCase):
    def test_pinned_onnx_allows_symbolic_batch_not_symbolic_sequence(self):
        self.assertTrue(seq9_shape(['unk__404',9,288,512]))
        self.assertTrue(seq9_shape([1,9,288,512]))
        self.assertFalse(seq9_shape([1,'sequence',288,512]))
        self.assertFalse(seq9_shape([2,9,288,512]))

    def test_pts_is_source_time_minus_container_start(self):
        packet = FramePacket(6, 13500, Fraction(1,90000), .01, np.zeros((2,2,3),np.uint8))
        self.assertAlmostEqual(packet.time_sec,.14)
        self.assertEqual(packet.clock()['time_base'],[1,90000])
        self.assertNotEqual(packet.time_sec, packet.index/30)

    def test_remainders_are_preserved(self):
        self.assertEqual([len(batch) for batch in chunks(range(20),9)],[9,9,2])
        self.assertEqual(uniform_indices(1),[0]*16)
        self.assertEqual(uniform_indices(30),[round(i*29/15) for i in range(16)])
        with self.assertRaises(ValueError):
            uniform_indices(0)

    def test_vball_short_tail_is_right_aligned(self):
        class Session:
            def run(self, outputs, values):
                self.last = values['input']
                return [np.zeros((1,9,288,512),np.float32)]
        tracker = BallTracker.__new__(BallTracker)
        from volleymole.performance import Timings
        tracker.timings = Timings()
        tracker.binding = None
        tracker.session = Session()
        tracker.input_name, tracker.output_name = 'input','output'
        tracker.buffer, tracker.previous_gray = [], None
        tracker.calls = tracker.frames = 0
        tracker.profile_pending = False
        packets = [FramePacket(i,i,Fraction(1,30),0,np.full((20,20,3),i,np.uint8)) for i in range(11)]
        tracker.predict(packets[:9])
        rows = tracker.predict(packets[9:])
        self.assertEqual([r['Frame'] for r in rows],[9,10])
        np.testing.assert_allclose(tracker.session.last[0,:,0,0],np.arange(2,11)/255,atol=1e-7)
        self.assertEqual(tracker.frames,11)


class JerseyTests(unittest.TestCase):
    def test_numeric_evidence_validation(self):
        self.assertEqual(digit_candidate(' 12 ',.9),12)
        for text,score in [('I2',.9),('1234',.9),('１２',.9),('12',float('nan')),('12',1.1)]:
            self.assertIsNone(digit_candidate(text,score))
        self.assertEqual(torso_box([-10,-10,110,210],100,200),[4,34,96,136])

    def test_samples_record_actual_pts_and_raw_evidence(self):
        class OCR:
            def readtext(self,*args,**kwargs):
                return [([[0,0],[10,0],[10,10],[0,10]],'12',.95),
                        ([[0,0],[10,0],[10,10],[0,10]],'7',.4)]
        reader = JerseyReader.__new__(JerseyReader)
        reader.reader, reader.number, reader.confidence, reader.interval = OCR(),12,.75,1.
        reader.next_sample, reader.calls = 0.,0
        reader.samples,reader.observations,reader.detections = [],[],[]
        people = [{'xyxy':[0,0,100,200], 'confidence':.9}]
        for i,pts in enumerate([14,990,1014,2010]):
            reader.consume(FramePacket(i,pts,Fraction(1,1000),0,np.zeros((200,100,3),np.uint8)),people)
        result = reader.result()
        self.assertEqual([s['time_sec'] for s in result['samples']],[.014,1.014,2.01])
        self.assertEqual(len(result['detections']),3)
        self.assertEqual(len(result['observations']),6)
        self.assertTrue(all(d['number']==12 for d in result['detections']))


class PackagingTests(unittest.TestCase):
    def test_no_upstream_runtime_imports_or_path_injection(self):
        forbidden = {'ml_manager','make_reels','volleyball_highlights','run_sample','run_batch_video'}
        for path in APP.glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0],forbidden,path.name)
                if isinstance(node,ast.Call):
                    self.assertNotIn(ast.unparse(node.func),('sys.path.insert','sys.path.append','os.chdir'),path.name)
        for name in ('assets/branding/volleymole.svg','assets/branding/volleymole.png',
                     'assets/fonts/ZCOOLKuaiLe-Regular.ttf','licenses/tracking-MIT.txt',
                     'licenses/volleyball-ml-models-MIT.txt','model_manifest.json'):
            self.assertTrue((APP/name).is_file(),name)


if __name__ == '__main__':
    unittest.main()
