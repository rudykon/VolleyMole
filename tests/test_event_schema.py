import copy
import io
import json
import unittest
from email.message import Message
from unittest.mock import patch

from volleymole.event_schema import event_response_format, event_request_payload
from volleymole.events import DIMENSIONS, validate_events
from volleymole.semantic import request_json
from test_events import event


class EventSchemaTests(unittest.TestCase):
    evidence = [{'id': 'f', 'kind': 'frame', 'start_sec': 3., 'end_sec': 3.}]

    def test_all_objects_are_closed_and_required_fields_match_local_contract(self):
        format_ = event_response_format(self.evidence, 0, 10)
        self.assertEqual(format_['type'], 'json_schema')
        self.assertTrue(format_['json_schema']['strict'])
        schema = format_['json_schema']['schema']
        self.assertEqual(schema['type'], 'object')
        self.assertEqual(schema['required'], ['events'])
        item = schema['properties']['events']['items']
        self.assertEqual(set(item['required']), set(event()))
        self.assertEqual(set(item['properties']['dimensions']['required']), set(DIMENSIONS))
        self.assertEqual(item['properties']['title']['maxLength'], 40)
        self.assertEqual(item['properties']['end_sec'], {'type': 'number', 'minimum': 0, 'maximum': 10})
        self.assertEqual(schema['$defs']['evidence_id']['enum'], ['f'])
        def check(node):
            if isinstance(node, dict):
                if node.get('type') == 'object':
                    self.assertIs(node['additionalProperties'], False)
                    self.assertEqual(set(node['required']), set(node['properties']))
                if '$ref' in node:
                    self.assertIn(node['$ref'].removeprefix('#/$defs/'), schema['$defs'])
                for child in node.values(): check(child)
            elif isinstance(node, list):
                for child in node: check(child)
        check(schema)

    def test_unobserved_sound_and_motion_remain_unknown_in_schema(self):
        schema = event_response_format(self.evidence, 0, 10)['json_schema']['schema']
        dims = schema['properties']['events']['items']['properties']['dimensions']['properties']
        for name in ('motion_intensity', 'related_laughter'):
            self.assertEqual(dims[name]['$ref'], '#/$defs/unknown_score')
        measured = self.evidence + [
            {'id': 'a', 'kind': 'audio', 'start_sec': 0, 'end_sec': 10},
            {'id': 'm', 'kind': 'local_motion', 'start_sec': 2, 'end_sec': 3, 'measured': True}]
        schema = event_response_format(measured, 0, 10)['json_schema']['schema']
        dims = schema['properties']['events']['items']['properties']['dimensions']['properties']
        self.assertEqual(dims['motion_intensity']['$ref'], '#/$defs/score')
        self.assertEqual(dims['related_laughter']['$ref'], '#/$defs/score')

    def test_previous_bad_shapes_and_fabricated_evidence_are_not_coerced(self):
        valid = {'events': [event()]}
        self.assertEqual(validate_events(valid, self.evidence, 0, 10), valid['events'])
        bad_score = copy.deepcopy(valid)
        bad_score['events'][0]['dimensions']['action_value'] = 3
        bad_ref = copy.deepcopy(valid)
        bad_ref['events'][0]['observations'][0]['evidence_ids'] = ['invented']
        bad_bounds = copy.deepcopy(valid)
        bad_bounds['events'][0]['clip_end_sec'] = 11
        for bad in ([{'events': []}], {'events': [], 'coarse_summary': {}}, bad_score, bad_ref, bad_bounds):
            before = copy.deepcopy(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_events(bad, self.evidence, 0, 10)
            self.assertEqual(bad, before)

    def test_wire_schema_hash_and_gateway_degradation_are_auditable(self):
        payload = event_request_payload('glm-5.3-flash', 'fixture', [], self.evidence, 0, 10, reasoning_effort='low')
        for header, expected in ((None, None), ('model_unsupported', 'model_unsupported'),
                                  ('schema_keywords_stripped', 'schema_keywords_stripped'),
                                  ('UNTRUSTED_SECRET', 'other')):
            response = io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"events":[]}'}}]}).encode())
            response.headers = Message()
            if header is not None: response.headers['X-Structured-Output-Degraded'] = header
            with patch('volleymole.semantic.urlopen', return_value=response):
                data, meta = request_json('https://example.invalid', 'test-secret', payload, 2)
            self.assertEqual(data, {'events': []})
            self.assertEqual(meta['structured_output_degraded'], expected)
            self.assertEqual(meta['response_format'], 'json_schema')
            self.assertEqual(len(meta['response_schema_sha256']), 64)
            self.assertNotIn('UNTRUSTED_SECRET', json.dumps(meta))
            self.assertNotIn('test-secret', json.dumps(meta))

    def test_reasoning_is_opt_in_and_payload_does_not_modify_evidence(self):
        before = copy.deepcopy(self.evidence)
        payload = event_request_payload('glm-5.3-flash', 'fixture', [], self.evidence, 0, 10)
        self.assertNotIn('reasoning_effort', payload)
        self.assertEqual(self.evidence, before)
        with self.assertRaises(ValueError): event_response_format([], 0, 10)


if __name__ == '__main__': unittest.main()
