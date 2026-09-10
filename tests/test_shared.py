import unittest
from volleymole.shared import fallback_indices


class SharedSchedulingTests(unittest.TestCase):
    def test_ball_detector_only_receives_missing_primary_evidence(self):
        balls = [{'Visibility':1},{'Visibility':0},{'Visibility':0},{'Visibility':0}]
        actions = [[],[{'class':'ball','confidence':.8}],
                   [{'class':'ball','confidence':.2}],[{'class':'serve','confidence':.9}]]
        self.assertEqual(fallback_indices(balls,actions),[2,3])

    def test_scheduler_rejects_frame_count_mismatch(self):
        with self.assertRaises(ValueError):
            fallback_indices([{'Visibility':0}],[])
