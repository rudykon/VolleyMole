"""Content-addressed cache keys and explicit import integrity."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from volleymole.adapters import inference_signature, validate_cache
from volleymole.common import identity, save_json, digest


class CacheTests(unittest.TestCase):
    def test_fingerprint_uses_content_models_device_and_ocr_not_output_path(self):
        source = {'path':'/first.mp4','sha256':'a'*64,'bytes':123}
        registry = SimpleNamespace(entries={'ball':{'sha256':'b'*64}})
        def signature(src=source,reg=registry,device='cuda:0',number=None,confidence=.75):
            return json.dumps(inference_signature(src,reg,device,number,confidence),sort_keys=True)
        baseline = signature()
        self.assertEqual(baseline,signature({**source,'path':'/relocated.mp4'}))
        self.assertEqual(baseline,signature(device='cuda'))
        self.assertNotEqual(baseline,signature({**source,'sha256':'c'*64}))
        self.assertNotEqual(baseline,signature(reg=SimpleNamespace(entries={'ball':{'sha256':'d'*64}})))
        self.assertNotEqual(baseline,signature(device='cpu'))
        self.assertNotEqual(baseline,signature(number=12))
        self.assertNotEqual(baseline,signature(confidence=.8))
        with patch('importlib.metadata.version',return_value='changed-version'):
            self.assertNotEqual(baseline,signature())

    def test_four_gpu_assignment_and_order_enter_cache_key_without_initializing_cuda(self):
        source = {'sha256':'a'*64,'bytes':123}
        registry = SimpleNamespace(entries={})
        with patch('volleymole.detectors.resolve_device', side_effect=AssertionError('no CUDA on cache read')):
            single = inference_signature(source,registry,'cuda:0',None,.75)
            four = inference_signature(source,registry,'auto',None,.75,'cuda:0,cuda:1,cuda:2,cuda:3')
            reordered = inference_signature(source,registry,'auto',None,.75,'cuda:1,cuda:0,cuda:2,cuda:3')
        self.assertNotEqual(single, four)
        self.assertNotEqual(four, reordered)
        self.assertIn('gpu_stages.py', four['code'])

    def test_performance_backend_and_scheduling_change_cache_key(self):
        source={'sha256':'a'*64,'bytes':123}
        registry=SimpleNamespace(entries={})
        base={'pipeline_depth':1,'auxiliary_device':None,'vball_engine':'ort'}
        def signature(options):
            return inference_signature(source,registry,'cuda:0',None,.75,performance=options)
        for key,value in [('pipeline_depth',2),('auxiliary_device','cuda:0'),('vball_engine','ort-bound')]:
            self.assertNotEqual(signature(base),signature({**base,key:value}))

    def test_explicit_shared_cache_checks_relocated_raw_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root/'source.bin'
            video.write_bytes(b'fixture, not a video')
            cache = root/'run/tracking'
            cache.mkdir(parents=True)
            raw = cache/'ball.csv'
            raw.write_text('original fixture evidence')
            save_json(cache.parent/'match_manifest.json',{'source':{'identity':identity(video)}})
            save_json(cache/'provenance.json',{'mode':'shared_inference_cache'})
            save_json(cache.parent/'state.json',{'stages':{'inference':{'status':'complete',
                'artifacts':{'/previous-location/tracking/ball.csv':digest(raw)}}}})
            validate_cache(video,cache,'tracking')
            raw.write_text('modified fixture evidence')
            with self.assertRaisesRegex(ValueError,'Modified cache'):
                validate_cache(video,cache,'tracking')

    def test_explicit_import_rejects_nested_partial_and_different_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root/'source.bin'
            video.write_bytes(b'fixture, not a video')
            cache = root/'run/analytics'
            cache.mkdir(parents=True)
            save_json(cache.parent/'match_manifest.json',{'source':{'identity':identity(video)}})
            save_json(cache/'provenance.json',{'original':{'parameters':{'max_frames':30}}})
            with self.assertRaisesRegex(ValueError,'Partial smoke'):
                validate_cache(video,cache,'analytics')
            video.write_bytes(b'different source contents')
            with self.assertRaisesRegex(ValueError,'different source bytes'):
                validate_cache(video,cache,'analytics')


if __name__=='__main__':
    unittest.main()
