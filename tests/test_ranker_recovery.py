"""Legacy rally ranking must fail safely without silently choosing a model."""
import copy
import io
from http.client import IncompleteRead, RemoteDisconnected
from unittest.mock import patch
import unittest
from urllib.error import HTTPError, URLError

from volleymole.common import read_json
from volleymole.llm_transport import ResponseContractError, TransportError
from volleymole.ranker import rank, rule_decision
import test_pipeline


class RankerRecoveryTests(unittest.TestCase):
    setUp = test_pipeline.DecisionTests.setUp
    tearDown = test_pipeline.DecisionTests.tearDown
    secret = 'NEVER_LOG_API_KEY_OR_REMOTE_BODY'

    def run_rank(self, vision_model=None):
        with patch.dict('os.environ', {'VOLLEYMOLE_API_KEY': self.secret}):
            path, _ = rank(self.manifest, self.root, 5, None, 'auto',
                           'https://example.invalid/v1', 'glm-5.3-flash', 1,
                           vision_model=vision_model)
        return read_json(path), read_json(self.root/'ranking_log.json')

    def assert_fallback(self, result, log, kind):
        self.assertEqual(result['ranking_mode'], 'rules_fallback')
        self.assertEqual(len(result['selected']), 5)
        self.assertEqual(result['fallback']['type'], kind)
        self.assertEqual(log['fallback'], result['fallback'])
        self.assertEqual(log['model_policy'], 'explicit_models_only')
        self.assertEqual(log['requested_model'], 'glm-5.3-flash')
        for name in ('edit_decision.json', 'ranking_log.json'):
            self.assertNotIn(self.secret, (self.root/name).read_text())

    def test_transport_failures_preserve_safe_categories_and_fall_back(self):
        for reason in ('connection_error', 'timeout', 'io_error', 'http_error'):
            with self.subTest(reason=reason):
                error = TransportError(reason, retryable=reason!='io_error',
                                       http_status=503 if reason=='http_error' else None)
                with patch('volleymole.ranker.api_decision', side_effect=error):
                    result, log = self.run_rank()
                self.assert_fallback(result, log, 'TransportError')
                self.assertEqual(log['fallback']['transport_error'], reason)
                self.assertEqual(log['fallback']['retryable'], reason!='io_error')
                if reason=='http_error':
                    self.assertEqual(log['fallback']['http_status'], 503)

    def test_raw_connections_timeout_and_incomplete_read_are_safe(self):
        errors = [RemoteDisconnected(self.secret), IncompleteRead(self.secret.encode(), 128),
                  ConnectionResetError(self.secret), TimeoutError(self.secret), URLError(self.secret)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch('volleymole.ranker.api_decision', side_effect=error):
                    result, log = self.run_rank()
                self.assert_fallback(result, log, type(error).__name__)
                expected = 'timeout' if isinstance(error, TimeoutError) else 'connection_error'
                self.assertEqual(log['fallback']['transport_error'], expected)

    def test_response_contract_failures_preserve_reason(self):
        for reason, finish in [('incomplete_response', 'length'), ('invalid_json', 'stop'),
                               ('invalid_response_envelope', None), ('refused', 'stop')]:
            with self.subTest(reason=reason):
                error = ResponseContractError(reason, finish)
                with patch('volleymole.ranker.api_decision', side_effect=error):
                    result, log = self.run_rank()
                self.assert_fallback(result, log, 'ResponseContractError')
                self.assertEqual(log['fallback']['response_contract'], reason)
                self.assertEqual(log['fallback']['finish_reason'], finish)

    def test_unknown_diagnostic_reasons_cannot_leak_remote_text(self):
        for error in (TransportError(self.secret, retryable=False),
                      ResponseContractError(self.secret, 'stop'), ValueError(self.secret)):
            with self.subTest(error=type(error).__name__):
                with patch('volleymole.ranker.api_decision', side_effect=error):
                    result, log = self.run_rank()
                self.assert_fallback(result, log, type(error).__name__)

    def test_primary_400_never_reads_body_or_discovers_another_model(self):
        for body in ('not a multimodal model', 'context too long'):
            with self.subTest(body=body):
                stream = io.BytesIO((body+' '+self.secret).encode())
                error = HTTPError('https://example.invalid', 400, self.secret, {}, stream)
                with patch('volleymole.ranker.api_decision', side_effect=error) as generation, \
                     patch.object(error, 'read', side_effect=AssertionError('must not read body')) as read, \
                     patch('volleymole.semantic.choose_vision_model') as choose, \
                     patch('volleymole.semantic.urlopen') as network, \
                     patch('volleymole.ranker.visual_reviews') as reviews:
                    result, log = self.run_rank()
                generation.assert_called_once()
                read.assert_not_called()
                choose.assert_not_called()
                network.assert_not_called()
                reviews.assert_not_called()
                self.assertTrue(stream.closed)
                self.assert_fallback(result, log, 'HTTPError')
                self.assertEqual(log['fallback']['http_status'], 400)
                self.assertIsNone(log['routing'])
                self.assertIsNone(log['requested_vision_model'])

    def test_explicit_vision_model_is_the_only_secondary_route(self):
        observations = [{'rally_id': r['rally_id'], 'observation': '可见接球姿态。',
                         'uncertainty': '三帧不能确认得分。', 'watchability_score': 80,
                         'confidence': .6} for r in self.rallies]
        with patch('volleymole.ranker.visual_reviews', return_value=(observations, [], [])) as reviews, \
             patch('volleymole.ranker.api_decision', return_value=self.decision) as generation, \
             patch('volleymole.semantic.choose_vision_model') as choose:
            result, log = self.run_rank(vision_model='glm-5.3-flash')
        self.assertEqual(result['ranking_mode'], 'vision_then_text_api')
        self.assertEqual(result['model'], 'glm-5.3-flash')
        self.assertEqual(result['vision_model'], 'glm-5.3-flash')
        self.assertEqual(reviews.call_args.args[3], 'glm-5.3-flash')
        self.assertEqual(generation.call_args.args[5], 'glm-5.3-flash')
        self.assertEqual(generation.call_args.kwargs['reviews'], observations)
        self.assertEqual(log['routing']['reason'], 'explicit_vision_model')
        choose.assert_not_called()

    def test_explicit_review_failure_does_not_try_another_model(self):
        with patch('volleymole.ranker.visual_reviews', side_effect=IncompleteRead(self.secret.encode())) as reviews, \
             patch('volleymole.ranker.api_decision') as generation, \
             patch('volleymole.semantic.choose_vision_model') as choose:
            result, log = self.run_rank(vision_model='glm-5.3-flash')
        reviews.assert_called_once()
        generation.assert_not_called()
        choose.assert_not_called()
        self.assert_fallback(result, log, 'IncompleteRead')
        self.assertEqual(log['requested_vision_model'], 'glm-5.3-flash')


class RuleDiversityRecoveryTests(unittest.TestCase):
    def rally(self, name, score, start, actions, source_id=None):
        result = {'rally_id': name, 'rule_score': score, 'start_sec': start,
                  'end_sec': start+8, 'safe_start_sec': start-1, 'safe_end_sec': start+10,
                  'duration_sec': 8, 'actions': actions,
                  'ball_metrics': {'visible_ratio': .8, 'trajectory_changes': 5}}
        if source_id is not None:
            result['source_id'] = source_id
        return result

    def selected_ids(self, candidates, count):
        return [item['rally_id'] for item in rule_decision(candidates, count)['selected']]

    def test_same_source_nearby_rallies_get_legacy_time_penalty(self):
        candidates = [self.rally('first', 100, 100, ['serve']),
                      self.rally('near', 99, 140, ['receive']),
                      self.rally('far', 97, 300, ['spike'])]
        self.assertEqual(self.selected_ids(candidates, 2), ['first', 'far'])

    def test_same_action_has_one_point_penalty_even_when_times_are_far_apart(self):
        candidates = [self.rally('first', 100, 100, ['serve']),
                      self.rally('same', 99, 300, ['serve']),
                      self.rally('different', 98.5, 500, ['receive'])]
        self.assertEqual(self.selected_ids(candidates, 2), ['first', 'different'])

    def test_cross_source_local_times_do_not_trigger_time_penalty_or_overlap(self):
        candidates = [self.rally('set1', 100, 100, ['serve'], 'set-0001'),
                      self.rally('set2', 99, 100, ['receive'], 'set-0002'),
                      self.rally('later', 98, 300, ['spike'], 'set-0001')]
        self.assertEqual(self.selected_ids(candidates, 2), ['set1', 'set2'])

    def test_action_penalty_remains_global_across_sources(self):
        candidates = [self.rally('first', 100, 100, ['serve'], 'set-0001'),
                      self.rally('same_action', 99, 300, ['serve'], 'set-0002'),
                      self.rally('different', 98.5, 500, ['receive'], 'set-0003')]
        self.assertEqual(self.selected_ids(candidates, 2), ['first', 'different'])

    def test_time_penalty_is_strictly_under_ninety_seconds(self):
        candidates = [self.rally('first', 100, 100, ['serve']),
                      self.rally('within', 99, 189.999, ['receive']),
                      self.rally('boundary', 98, 190, ['spike'])]
        self.assertEqual(self.selected_ids(candidates, 2), ['first', 'boundary'])

    def test_time_and_action_penalties_accumulate_over_already_selected_rallies(self):
        candidates = [self.rally('first', 100, 100, ['serve']),
                      self.rally('second', 99, 300, ['receive']),
                      self.rally('both_penalties', 98, 320, ['serve']),
                      self.rally('diverse', 96.5, 500, ['spike'])]
        self.assertEqual(self.selected_ids(candidates, 3), ['first', 'second', 'diverse'])

    def test_high_adjusted_score_cannot_bypass_same_source_clip_overlap(self):
        candidates = [self.rally('first', 100, 100, ['serve']),
                      self.rally('overlap', 99, 105, ['receive']),
                      self.rally('separate', 98, 125, ['spike'])]
        self.assertEqual(self.selected_ids(candidates, 2), ['first', 'separate'])
        with self.assertRaisesRegex(ValueError, '不重叠'):
            rule_decision(candidates, 3)

    def test_missing_and_explicit_single_source_ids_share_the_same_clock(self):
        candidates = [self.rally('first', 100, 100, ['serve']),
                      self.rally('near', 99, 140, ['receive'], 'single'),
                      self.rally('far', 97, 300, ['spike'], 'single')]
        self.assertEqual(self.selected_ids(candidates, 2), ['first', 'far'])

    def test_single_source_matches_historical_4399475_selection_without_mutation(self):
        # The historical caller provided shortlist's descending-score order.
        candidates = [self.rally(f'r{i}', score, start, actions) for i, (score, start, actions) in enumerate([
            (100, 100, ['serve']), (99, 130, ['receive']), (98, 160, ['serve']),
            (97.5, 300, ['spike']), (97, 330, ['receive']), (96.5, 500, ['serve']),
            (96, 530, ['spike']), (95.5, 700, ['receive']), (95, 730, ['serve']),
            (94.5, 900, ['spike']), (94, 930, ['receive']), (93.5, 1100, ['serve']),
        ])]
        before = copy.deepcopy(candidates)
        for count in (5, 10):
            pool = list(candidates); historical = []
            while pool and len(historical) < count:
                def historical_adjusted(r):
                    penalty = sum(3 for s in historical if abs(s['start_sec']-r['start_sec']) < 90)
                    penalty += sum(1 for s in historical if s['actions'] == r['actions'])
                    return r['rule_score']-penalty
                best = max(pool, key=historical_adjusted); historical.append(best); pool.remove(best)
            with self.subTest(count=count):
                decision = rule_decision(candidates, count)
                self.assertEqual([r['rally_id'] for r in decision['selected']],
                                 [r['rally_id'] for r in historical])
                for item, raw in zip(decision['selected'], historical):
                    self.assertEqual((item['clip_start_sec'], item['clip_end_sec']),
                                     (raw['safe_start_sec'], raw['safe_end_sec']))
        self.assertEqual(candidates, before)


if __name__ == '__main__':
    unittest.main()
