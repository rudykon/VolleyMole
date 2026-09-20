"""Motion evidence must veto truncation without inventing a completed action."""
import copy
import unittest

import numpy as np

from volleymole.replay_motion import TIME_BASIS, boundary_guard, motion_evidence, spatial_hints


def trajectory(*, impulse=None, offset=0., scale=1., missing=None, noise=False):
    width, height = 1920*scale, 1080*scale
    rows = []
    for time in np.arange(0., 3.001, 1/30):
        if missing and missing[0] < time < missing[1]:
            continue
        # A natural apex at .75 sec. A genuine velocity change is optional.
        x = .15+.12*time
        y = .31-.24*time+.16*time*time
        if impulse is not None:
            y -= .70*max(0., time-impulse)
        # Deterministic source heatmap quantization, not jittered timestamps.
        x, y = x*width, y*width
        if noise:
            x, y = round(x/4)*4, round(y/4)*4
        if 0 <= x < width and 0 <= y < height:
            rows.append([float(round(time+offset, 8)), float(x), float(y), True])
    track = dict(samples=rows, sample_origins=['vball']*len(rows),
                 time_basis=TIME_BASIS, coordinate_interpolation_used=False)
    source = dict(width=width, height=height, rotation=0, identity={'sha256': 'test-source'})
    return track, source


class MotionEvidenceTest(unittest.TestCase):
    def test_natural_apex_is_not_a_touch(self):
        track, source = trajectory(noise=True)
        evidence = motion_evidence(track, source, 0, 3)
        self.assertEqual(evidence['status'], 'available')
        self.assertEqual(evidence['contact_hypotheses'], [])
        self.assertFalse(evidence['proves_boundary_complete'])

    def test_contact_interval_is_translation_and_resolution_invariant(self):
        contacts = []
        for offset, scale in [(0., 1.), (800., 1.), (800., .5)]:
            track, source = trajectory(impulse=1.6, offset=offset, scale=scale)
            evidence = motion_evidence(track, source, offset, offset+3)
            self.assertEqual(len(evidence['contact_hypotheses']), 1)
            contact = evidence['contact_hypotheses'][0]
            self.assertAlmostEqual(contact['time_sec']-offset, 1.6, delta=.04)
            self.assertLessEqual(contact['start_sec'], offset+1.6)
            self.assertGreaterEqual(contact['end_sec'], offset+1.6)
            self.assertTrue(contact['requires_visual_confirmation'])
            contacts.append(contact['time_sec']-offset)
        self.assertAlmostEqual(contacts[0], contacts[1], places=6)
        self.assertAlmostEqual(contacts[1], contacts[2], places=6)

    def test_defense_requires_observed_departure_after_next_impulse(self):
        track, source = trajectory(impulse=1.6, noise=True)
        evidence = motion_evidence(track, source, 0, 3)
        review = dict(action='dig', peak_sec=.5, source_end_sec=1.4)
        guard = boundary_guard(evidence, review)
        self.assertEqual(guard['verdict'], 'expand')
        self.assertTrue(guard['need_after'])
        self.assertGreater(guard['minimum_source_end_sec'], 1.6)
        self.assertLess(guard['minimum_source_end_sec'], 1.9)
        self.assertIsNotNone(guard['observed_motion_at_tail'])
        # More footage removes this one contradiction, not the need for review.
        review['source_end_sec'] = 2.
        guard = boundary_guard(evidence, review)
        self.assertEqual(guard['verdict'], 'no_conflict')
        self.assertFalse(guard['proves_boundary_complete'])

    def test_peak_offset_changes_no_fixed_duration_limit(self):
        track, source = trajectory(impulse=1.6)
        evidence = motion_evidence(track, source, 0, 3)
        ends = []
        for peak in [.2, .6, 1.0]:
            guard = boundary_guard(evidence, dict(action='dig', peak_sec=peak, source_end_sec=1.4))
            ends.append(guard['minimum_source_end_sec'])
        self.assertEqual(len(set(ends)), 1)

    def test_occlusion_cannot_manufacture_a_contact(self):
        track, source = trajectory(impulse=1.6, missing=(1.3, 1.9))
        evidence = motion_evidence(track, source, 0, 3)
        self.assertEqual(evidence['contact_hypotheses'], [])
        self.assertTrue(any(g['start_sec'] <= 1.3 and g['end_sec'] >= 1.9 for g in evidence['gaps']))
        guard = boundary_guard(evidence, dict(action='dig', peak_sec=.5, source_end_sec=1.7))
        self.assertEqual(guard['verdict'], 'uncertain')
        self.assertEqual(guard['reason'], 'tail_ball_track_unobserved')

    def test_missing_after_ball_never_means_rally_over(self):
        track, source = trajectory(missing=(1.2, 4.))
        evidence = motion_evidence(track, source, 0, 3)
        guard = boundary_guard(evidence, dict(action='spike', peak_sec=.5, source_end_sec=2.))
        self.assertEqual(guard['verdict'], 'uncertain')
        self.assertFalse(guard['proves_boundary_complete'])

    def test_no_following_touch_is_not_a_successful_dig_result(self):
        track, source = trajectory()
        evidence = motion_evidence(track, source, 0, 3)
        guard = boundary_guard(evidence, dict(action='receive', peak_sec=.3, source_end_sec=1.3))
        self.assertEqual(guard['verdict'], 'uncertain')
        self.assertEqual(guard['reason'], 'no_observed_following_contact_not_proof_of_dead_ball')

    def test_untrusted_clock_interpolation_and_sparse_tracks_are_unknown(self):
        base, source = trajectory()
        for field, value in [('time_basis', 'frame_number / nominal_fps'),
                             ('coordinate_interpolation_used', True),
                             ('sample_origins', [])]:
            with self.subTest(field=field):
                track = {**base, field: value}
                self.assertEqual(motion_evidence(track, source, 0, 3)['status'], 'unavailable')
        sparse = copy.deepcopy(base)
        sparse['samples'] = sparse['samples'][::4]
        sparse['sample_origins'] = sparse['sample_origins'][::4]
        self.assertEqual(motion_evidence(sparse, source, 0, 3)['reason'], 'insufficient_tracking_time_resolution')

    def test_nonmonotonic_pts_are_not_silently_sorted(self):
        track, source = trajectory()
        track['samples'][2][0] = track['samples'][1][0]
        self.assertEqual(motion_evidence(track, source, 0, 3)['reason'], 'nonmonotonic_source_pts')

    def test_auxiliary_positions_cannot_bridge_missing_raw_ball(self):
        track, source = trajectory(impulse=1.6)
        track['sample_origins'] = ['auxiliary' if 1.3 < row[0] < 1.9 else 'vball' for row in track['samples']]
        evidence = motion_evidence(track, source, 0, 3)
        self.assertEqual(evidence['contact_hypotheses'], [])

    def test_single_false_heatmap_position_does_not_create_contact(self):
        track, source = trajectory(noise=True)
        track['samples'][35][1] += 140
        track['samples'][35][2] += 100
        evidence = motion_evidence(track, source, 0, 3)
        self.assertEqual(evidence['contact_hypotheses'], [])

    def test_clip_scope_and_display_rotation_are_preserved(self):
        track, source = trajectory(impulse=1.6, offset=100.)
        source['rotation'] = 180
        evidence = motion_evidence(track, source, 101, 102.3)
        self.assertEqual(evidence['coordinate_basis'], 'display_oriented_pixels_after_source_rotation')
        self.assertEqual(evidence['source_rotation'], 180)
        for contact in evidence['contact_hypotheses']:
            self.assertGreaterEqual(contact['start_sec'], 101)
            self.assertLessEqual(contact['end_sec'], 102.3)
        with self.assertRaises(ValueError):
            boundary_guard(evidence, dict(action='dig', peak_sec=101.2, source_end_sec=103))

    def test_real_sparse_contact_regression_uses_pts_not_missing_frame_fill(self):
        # Actual measured points around the first continuation of the known
        # failure, translated to an anonymous local clock. There are no samples
        # at the contact itself; its interval must be inferred, never fabricated.
        raw = [(-.35, 1072,292),(-.316667,1065,311),(-.283333,1057,341),
               (-.25,1046,375),(-.216667,1038,401),(-.183333,1031,431),
               (-.15,1027,461),(-.05,1001,558),(.05,1005,562),
               (.083333,1012,540),(.118333,1012,521),(.151667,1027,491),
               (.185,1035,472),(.218333,1042,457),(.251667,1050,442),
               (.285,1057,427),(.318333,1065,416),(.351667,1072,405)]
        rows = [[round(t+2., 6),x,y,True] for t,x,y in raw]
        track = dict(samples=rows,sample_origins=['vball']*len(rows),
                     time_basis=TIME_BASIS,coordinate_interpolation_used=False)
        evidence = motion_evidence(track, dict(width=1920,height=1080),1.5,2.5)
        self.assertEqual(len(evidence['contact_hypotheses']),1)
        contact = evidence['contact_hypotheses'][0]
        self.assertAlmostEqual(contact['time_sec'],2.,delta=.04)
        self.assertLessEqual(contact['start_sec'],1.95)
        self.assertGreaterEqual(contact['end_sec'],2.05)
        self.assertTrue(contact['requires_visual_confirmation'])


class SpatialHintsTest(unittest.TestCase):
    @staticmethod
    def evidence(x=.7, y=.5, *, rotation=0, count=61, step=1/30):
        rows = [[round(i*step,8), float(x*1920), float(y*1080), True] for i in range(count)]
        track = dict(samples=rows, sample_origins=['vball']*len(rows),
                     time_basis=TIME_BASIS, coordinate_interpolation_used=False)
        return motion_evidence(track, dict(width=1920,height=1080,rotation=rotation),0,max(2.,count*step))

    def test_raw_coordinates_use_height_normalization(self):
        evidence = self.evidence(x=.7,y=.5)
        self.assertEqual(evidence['display_width'],1920)
        self.assertEqual(evidence['display_height'],1080)
        for when,x,y in evidence['observed_ball_samples']:
            self.assertAlmostEqual(x,.7)
            self.assertAlmostEqual(y,.5)

    def test_high_ball_crop_keeps_lower_body_context(self):
        evidence = self.evidence(x=.84,y=.03)
        hints = spatial_hints(evidence,start_sec=.2,end_sec=1.8)
        self.assertTrue(hints)
        for hint in hints:
            left,top,right,bottom = hint['bbox']
            self.assertEqual(hint['coordinate_space'],'normalized_display')
            self.assertGreaterEqual(right-left,.419999)
            self.assertLessEqual(top,.03)
            self.assertEqual(bottom,1.)
            self.assertLess(left,.84)
            self.assertGreater(right,.84)

    def test_display_coordinates_are_not_rotated_twice(self):
        first = spatial_hints(self.evidence(rotation=0),start_sec=.2,end_sec=1.8)
        rotated = spatial_hints(self.evidence(rotation=180),start_sec=.2,end_sec=1.8)
        self.assertEqual(first,rotated)

    def test_hints_are_confined_to_candidate_and_selected_clip(self):
        evidence = self.evidence()
        for start,end in [(.6,.95),(-1.,.5),(1.8,4.)]:
            hints = spatial_hints(evidence,start_sec=start,end_sec=end)
            self.assertTrue(hints)
            for hint in hints:
                self.assertGreaterEqual(hint['start_sec'],max(start,evidence['clip_start_sec']))
                self.assertLessEqual(hint['end_sec'],min(end,evidence['clip_end_sec']))
        self.assertEqual(spatial_hints(evidence,start_sec=4.,end_sec=5.),[])

    def test_long_interval_uses_local_windows_not_rally_union(self):
        evidence = self.evidence(count=301)
        for row in evidence['observed_ball_samples']:
            row[1] = .1+.08*row[0]
        hints = spatial_hints(evidence,start_sec=0.,end_sec=10.)
        self.assertGreater(len(hints),5)
        self.assertTrue(all(h['end_sec']-h['start_sec'] <= 1.2 for h in hints))
        self.assertTrue(all(h['bbox'][2]-h['bbox'][0] < .6 for h in hints))
        self.assertLess(hints[0]['bbox'][0],hints[-1]['bbox'][0])

    def test_observation_gap_does_not_supply_a_crop(self):
        evidence = self.evidence()
        evidence['observed_ball_samples'] = [r for r in evidence['observed_ball_samples'] if not .4<r[0]<1.6]
        self.assertEqual(spatial_hints(evidence,start_sec=.3,end_sec=1.7),[])

    def test_no_fitted_or_fabricated_position_substitutes_for_observations(self):
        evidence = self.evidence()
        evidence['observed_ball_samples'] = []
        evidence['contact_hypotheses'] = [dict(position_display_xy=[1000.,500.],time_sec=.5)]
        self.assertEqual(spatial_hints(evidence,start_sec=.2,end_sec=.9),[])

    def test_few_valid_observations_can_hint_without_motion_verdict(self):
        evidence = self.evidence(count=3,step=.1)
        self.assertEqual(evidence['status'],'unavailable')
        hints = spatial_hints(evidence,start_sec=0.,end_sec=.2)
        self.assertEqual(len(hints),1)

    def test_broad_or_invalid_observations_keep_panorama_only(self):
        evidence = self.evidence()
        for index,row in enumerate(evidence['observed_ball_samples']):
            row[1] = .05 if index % 2 else .95
        self.assertEqual(spatial_hints(evidence,start_sec=.2,end_sec=1.8),[])
        for bad in [float('nan'), -1., 1.2]:
            evidence = self.evidence()
            evidence['observed_ball_samples'][2][1] = bad
            self.assertEqual(spatial_hints(evidence,start_sec=.2,end_sec=1.8),[])
        with self.assertRaises(ValueError):
            spatial_hints(self.evidence(),start_sec=2.,end_sec=1.)


if __name__ == '__main__':
    unittest.main()
