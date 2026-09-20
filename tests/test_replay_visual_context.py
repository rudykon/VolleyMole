import base64
import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import time
import unittest

import av
import cv2
import numpy as np

from volleymole.replay_evidence import (
    _decode, _encode, _hint_box, _motion_box, _native_frames,
    label_frames, prepare_visual_context,
)
from volleymole.semantic import sampled_evidence


def sequence(count=8, width=320, height=180):
    evidence = []; content = []; pixels = []
    for index in range(count):
        frame = np.full((height, width, 3), (30, 60, 90), dtype=np.uint8)
        cv2.rectangle(frame, (80+index*3, 65), (100+index*3, 100), (220, 160, 20), -1)
        ref = dict(kind='frame', id=f'frame_{index:05d}', start_sec=10+index*.125, end_sec=10+index*.125)
        evidence.append(ref); pixels.append(frame)
        content.extend([dict(type='text', text=f"{ref['id']} source_sec={ref['start_sec']:.6f}"),
                        dict(type='image_url', image_url=dict(url=_encode(frame), detail='high'))])
    return evidence, content, pixels


class ReplayVisualContextTests(unittest.TestCase):
    def test_label_keeps_original_api_and_full_picture(self):
        evidence, content, pixels = sequence(1)
        result = label_frames(evidence, content)
        self.assertIs(result, content)
        decoded = _decode(result[1])
        border = decoded.shape[0]-pixels[0].shape[0]
        self.assertGreater(border, 0)
        self.assertEqual(decoded.shape[1], pixels[0].shape[1])
        # JPEG recompression is allowed; no player/ball pixels are overpainted.
        self.assertLess(np.mean(np.abs(decoded[border:].astype(float)-pixels[0])), 2.)

    def test_overview_keeps_order_identity_and_both_boundaries(self):
        evidence, content, _ = sequence(40)
        original = copy.deepcopy(content); refs = copy.deepcopy(evidence)
        result, audit = prepare_visual_context(evidence, content, overview_frames=12)
        self.assertEqual(content, original); self.assertEqual(evidence, refs)
        self.assertEqual(audit['overview'][0]['frame_id'], evidence[0]['id'])
        self.assertEqual(audit['overview'][-1]['frame_id'], evidence[-1]['id'])
        self.assertEqual(len(audit['overview']), 12)
        self.assertEqual(audit['overview'], sorted(audit['overview'], key=lambda r:r['source_sec']))
        self.assertEqual(len([p for p in result if p['type']=='image_url']), len(evidence)+1)
        self.assertEqual(audit['details'], [])
        self.assertIn('SAME frame/PTS', result[0]['text'])
        self.assertIn('sparse navigation', result[0]['text'])

    def test_disabled_auxiliary_views_preserve_individual_count(self):
        evidence, content, _ = sequence()
        result, audit = prepare_visual_context(evidence, content, overview_frames=0, detail_limit=0)
        self.assertEqual(len(result), len(content))
        self.assertEqual(audit['overview'], [])
        self.assertEqual(audit['details'], [])

    def test_local_motion_produces_bounded_crop_but_static_does_not(self):
        _, _, pixels = sequence()
        box = _motion_box(pixels)
        self.assertIsNotNone(box)
        self.assertLessEqual(box[0], 80/320)
        self.assertGreaterEqual(box[2], 121/320)
        self.assertGreaterEqual(box[2]-box[0], .45-1e-9)
        self.assertGreaterEqual(box[3]-box[1], .4-1e-9)
        self.assertIsNone(_motion_box([pixels[0]]*4))

    def test_camera_motion_is_not_mistaken_for_action_region(self):
        rng = np.random.default_rng(17)
        texture = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
        self.assertIsNone(_motion_box([np.roll(texture, 10*i, axis=1) for i in range(8)]))

    def test_spatial_hints_union_players_and_ball_and_respect_time(self):
        hints = [dict(start_sec=10., end_sec=11., bbox=[.1,.2,.2,.5], coordinate_space='normalized_display'),
                 dict(start_sec=10., end_sec=11., bbox=[.7,.05,.75,.1], coordinate_space='normalized_display')]
        box = _hint_box(hints, 10.5)
        self.assertLessEqual(box[0], .1); self.assertGreaterEqual(box[2], .75)
        self.assertLessEqual(box[1], .05); self.assertGreaterEqual(box[3], .5)
        self.assertIsNone(_hint_box(hints, 11.1))

    def test_invalid_hint_coordinate_space_and_values_rejected(self):
        for hint in (dict(start_sec=10., end_sec=11., bbox=[10,20,30,40], coordinate_space='pixels'),
                     dict(start_sec=10., end_sec=11., bbox=[-.1,0,.5,.5], coordinate_space='normalized_display'),
                     dict(start_sec=10., end_sec=11., bbox=[0,0,float('nan'),.5], coordinate_space='normalized_display')):
            with self.subTest(hint=hint), self.assertRaisesRegex(ValueError, 'replay_visual_hint'):
                _hint_box([hint], 10.5)

    def test_out_of_order_or_duplicate_frames_cannot_form_storyboard(self):
        evidence, content, _ = sequence(3)
        for modified in ([evidence[1], evidence[0], evidence[2]], [evidence[0]]*3):
            with self.assertRaisesRegex(ValueError, 'replay_visual_frame_order'):
                prepare_visual_context(modified, content)

    def test_detail_budget_and_invalid_images_rejected(self):
        evidence, content, _ = sequence(1)
        with self.assertRaisesRegex(ValueError, 'replay_visual_budget'):
            prepare_visual_context(evidence, content, detail_limit=97)
        with self.assertRaisesRegex(ValueError, 'replay_frame_label_count'):
            prepare_visual_context(evidence, [])

    def test_hint_focus_gets_consecutive_contact_detail_frames_within_twelve_budget(self):
        from volleymole.replay_evidence import _detail_plan
        frames = [dict(id=f'f{i}',start_sec=100+i*.125) for i in range(72)]
        hint = dict(start_sec=103.6,end_sec=104.4,bbox=[.2,.3,.5,1.],coordinate_space='normalized_display')
        selected,audit = _detail_plan(frames,[hint],12)
        self.assertEqual(len(selected),12)
        self.assertEqual(audit['method'],'hint_prioritized_continuous_bursts')
        self.assertEqual(audit['regions'][0]['frame_ids'],[f'f{i}' for i in range(29,36)])
        self.assertTrue(all(i in selected for i in range(29,36)))
        self.assertGreaterEqual(sum(abs(frames[i]['start_sec']-104.)<=.375 for i in selected),5)
        self.assertEqual(selected,sorted(set(selected)))

    def test_two_hint_regions_each_receive_contiguous_bracketing_sequence(self):
        from volleymole.replay_evidence import _detail_plan
        frames = [dict(id=f'f{i}',start_sec=i*.125) for i in range(96)]
        hints = [dict(start_sec=a,end_sec=b,bbox=[.2,.3,.5,1.],coordinate_space='normalized_display')
                 for a,b in [(2.,3.),(8.,9.)]]
        selected,audit = _detail_plan(frames,hints,12)
        self.assertEqual(len(selected),12)
        self.assertEqual(len(audit['regions']),2)
        for region in audit['regions']:
            indices=[int(fid[1:]) for fid in region['frame_ids']]
            self.assertEqual(len(indices),6)
            self.assertEqual(indices,list(range(indices[0],indices[0]+6)))
            self.assertEqual(region['status'],'continuous_burst')

    def test_unhinted_sequence_retains_uniform_detail_selection(self):
        from volleymole.replay_evidence import _detail_plan, _indices
        frames = [dict(id=f'f{i}',start_sec=200+i*.125) for i in range(72)]
        selected,audit = _detail_plan(frames,[],12)
        self.assertEqual(selected,_indices(72,12))
        self.assertEqual(audit['method'],'uniform')

    def test_sparse_hint_does_not_invent_neighbors_or_claim_complete_burst(self):
        from volleymole.replay_evidence import _detail_plan
        frames = [dict(id=f'f{i}',start_sec=100+i*.125) for i in range(72)]
        hint = dict(start_sec=103.99,end_sec=104.01,bbox=[.2,.3,.5,1.],coordinate_space='normalized_display')
        selected,audit = _detail_plan(frames,[hint],12)
        self.assertEqual(len(selected),12)
        self.assertEqual(audit['regions'][0]['status'],'deferred')
        self.assertEqual(audit['regions'][0]['reason'],'fewer_than_three_consecutive_hint_frames')
        self.assertEqual(audit['regions'][0]['frame_ids'],[])

    def test_excess_hint_regions_report_budget_gap_without_exceeding_limit(self):
        from volleymole.replay_evidence import _detail_plan
        frames = [dict(id=f'f{i}',start_sec=i*.125) for i in range(96)]
        hints = [dict(start_sec=a,end_sec=a+.5,bbox=[.2,.3,.5,1.],coordinate_space='normalized_display')
                 for a in [1.,3.,5.,7.,9.]]
        selected,audit = _detail_plan(frames,hints,12)
        self.assertEqual(len(selected),12)
        served=[r for r in audit['regions'] if r['status']=='continuous_burst']
        deferred=[r for r in audit['regions'] if r['status']=='deferred']
        self.assertEqual(len(served),4);self.assertEqual(len(deferred),1)
        self.assertTrue(all(len(r['frame_ids'])==3 for r in served))
        self.assertEqual(deferred[0]['reason'],'detail_budget_cannot_bracket_every_hint_region')

    def test_large_actual_pts_gap_splits_detail_bursts(self):
        from volleymole.replay_evidence import _detail_plan
        times=[0.,.125,.25,1.,1.125,1.25,1.375,1.5]
        frames=[dict(id=f'f{i}',start_sec=t) for i,t in enumerate(times)]
        hint=dict(start_sec=0.,end_sec=1.5,bbox=[.2,.3,.5,1.],coordinate_space='normalized_display')
        selected,audit=_detail_plan(frames,[hint],12)
        self.assertEqual(len(selected),len(frames))
        self.assertEqual(len(audit['regions']),2)
        for region in audit['regions']:
            indices=[int(fid[1:]) for fid in region['frame_ids']]
            self.assertTrue(all(times[b]-times[a]<=.1875 for a,b in zip(indices,indices[1:])))

    def make_video(self, root):
        path = str(Path(root)/'source.mkv')
        with av.open(path, mode='w') as container:
            stream = container.add_stream('ffv1', rate=10)
            stream.width = 320; stream.height = 180; stream.pix_fmt = 'bgr0'
            for index in range(10):
                grid = np.indices((180, 320)).sum(axis=0)%2*80+30
                pixels = np.stack([grid, grid, grid], axis=-1).astype(np.uint8)
                cv2.rectangle(pixels, (90+index*4,65), (110+index*4,100), (220,20,80), -1)
                frame = av.VideoFrame.from_ndarray(pixels, format='bgr24')
                frame.pts = index; frame.time_base = Fraction(1,10)
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        return dict(path=path, start_sec=0., rotation=0, nominal_fps='10/1', identity={'sha256':'test-source'})

    def test_native_details_keep_exact_pts_full_view_and_coordinate_audit(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.make_video(root)
            evidence, content, _ = sampled_evidence(source, .2, .9, 5, width=96)
            hint = dict(start_sec=.2,end_sec=.9,bbox=[.3,.3,.55,.6],coordinate_space='normalized_display')
            result, audit = prepare_visual_context(evidence, content, source, spatial_hints=[hint], overview_frames=3, detail_limit=3)
            self.assertEqual(len(audit['details']), 3)
            self.assertEqual(audit['source_sha256'], 'test-source')
            self.assertEqual(audit['coordinate_space'], 'normalized_display_after_source_rotation')
            original = {r['id']:r['start_sec'] for r in evidence}
            for row in audit['details']:
                self.assertEqual(row['source_sec'], original[row['frame_id']])
                self.assertEqual(row['original_size'], [320,180])
                self.assertEqual(row['method'], 'spatial_hint')
            # One overview plus all original frames; detail is a second panel.
            self.assertEqual(len([p for p in result if p['type']=='image_url']), len(evidence)+1)
            detail_part = next(p for p in result[2:] if p['type']=='image_url')
            detail_pixels = _decode(detail_part)
            full_height = _decode(content[1]).shape[0]
            self.assertGreater(detail_pixels.shape[0], full_height+28)

    def test_native_decode_rejects_nearby_frame_instead_of_mislabeling(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.make_video(root)
            with self.assertRaisesRegex(ValueError, 'replay_visual_pts_mismatch'):
                _native_frames(source, [dict(id='frame_00000',start_sec=.251)], None)
            with self.assertRaisesRegex(TimeoutError, 'replay_visual_context_deadline'):
                _native_frames(source, [dict(id='frame_00000',start_sec=.2)], time.monotonic()-1)

    def test_rotation_uses_display_coordinates_without_changing_pts(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.make_video(root); source['rotation'] = 90
            evidence, content, _ = sampled_evidence(source, .2, .9, 5, width=64)
            hint = dict(start_sec=.2,end_sec=.9,bbox=[.2,.2,.8,.8],coordinate_space='normalized_display')
            _, audit = prepare_visual_context(evidence, content, source, spatial_hints=[hint], detail_limit=2)
            self.assertEqual(len(audit['details']), 2)
            self.assertEqual(audit['details'][0]['original_size'], [180,320])
            self.assertEqual(audit['details'][0]['source_sec'], evidence[0]['start_sec'])


if __name__ == '__main__':
    unittest.main()
