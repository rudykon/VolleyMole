"""Anchored event wire protocol must retain, not synthesize, observed evidence."""
import copy
import unittest

from volleymole.event_schema import WIRE_PROTOCOL, decode_event_response, wire_event_response_format
from volleymole.events import DIMENSIONS, validate_events


def evidence():
    return [
        *({'id': f'f{i}', 'kind': 'frame', 'start_sec': 100. + i,
           'end_sec': 100. + i} for i in range(5)),
        {'id': 'sound', 'kind': 'sound_event', 'start_sec': 102.5, 'end_sec': 103.5},
        {'id': 'motion', 'kind': 'local_motion', 'start_sec': 101., 'end_sec': 102.,
         'measured': True},
        {'id': 'unmeasured', 'kind': 'local_motion', 'start_sec': 101., 'end_sec': 102.,
         'measured': False},
        {'id': 'audio', 'kind': 'audio', 'start_sec': 100., 'end_sec': 105.},
    ]


def wire_event():
    dimensions = {name: {'value': None, 'fact_indexes': []} for name in DIMENSIONS}
    dimensions['action_value'] = {'value': 3, 'fact_indexes': [0, 1]}
    return {
        'event_type': 'observed_action', 'start_frame': 'f1', 'end_frame': 'f3',
        'clip_start_frame': 'f0', 'clip_end_frame': 'f4', 'peak_frame': 'f2',
        'title': '画面动作', 'confidence': .8, 'is_rally': True,
        'boundary_complete': True, 'injury_suspected': False, 'laughter_linked': None,
        'uncertainty': '快速触球无法由采样确认',
        'facts': [
            {'kind': 'observation', 'text': '球员伸出双臂', 'frame_id': 'f1', 'support_id': None},
            {'kind': 'observation', 'text': '球员向左跨步', 'frame_id': 'f2', 'support_id': None},
            {'kind': 'aftermath', 'text': '球员站直', 'frame_id': 'f3', 'support_id': None},
            {'kind': 'reaction', 'text': '同伴转头', 'frame_id': 'f4', 'support_id': None},
        ],
        'dimensions': dimensions,
    }


def named_wire_event():
    event = wire_event()
    event['dimensions'] = [{'dimension': name, **score}
                           for name, score in event['dimensions'].items()]
    return event


class EventWireTests(unittest.TestCase):
    def decode(self, event=None, refs=None):
        return decode_event_response({'events': [wire_event() if event is None else event]},
                                     evidence() if refs is None else refs, 100., 105.)

    def test_nonempty_wire_derives_exact_source_times_and_keeps_business_validation(self):
        result = self.decode()
        event = result['events'][0]
        self.assertEqual((event['start_sec'], event['end_sec'], event['peak_sec']), (101., 103., 102.))
        self.assertEqual((event['clip_start_sec'], event['clip_end_sec']), (100., 104.))
        self.assertEqual(event['observations'][0], {
            'text': '球员伸出双臂', 'time_sec': 101., 'evidence_ids': ['f1']})
        self.assertEqual(event['dimensions']['action_value'], {'value': 3, 'evidence_ids': ['f1', 'f2']})
        self.assertEqual(event['aftermath'][0]['time_sec'], 103.)
        self.assertEqual(event['reactions'][0]['time_sec'], 104.)
        self.assertEqual(validate_events(result, evidence(), 100., 105.), [event])

    def test_wire_and_original_evidence_are_not_mutated(self):
        data, refs = {'events': [wire_event()]}, evidence()
        before = copy.deepcopy((data, refs))
        decode_event_response(data, refs, 100., 105.)
        self.assertEqual((data, refs), before)

    def test_named_dimension_array_maps_by_name_not_position(self):
        event = named_wire_event()
        event['dimensions'].reverse()
        self.assertEqual(self.decode(event), self.decode(wire_event()))
        self.assertEqual(self.decode(event)['events'][0]['dimensions']['action_value']['value'], 3)

    def test_named_dimension_array_and_evidence_are_not_mutated(self):
        data, refs = {'events': [named_wire_event()]}, evidence()
        before = copy.deepcopy((data, refs))
        result = decode_event_response(data, refs, 100., 105.)
        result['events'][0]['dimensions']['action_value']['evidence_ids'].append('not_input')
        self.assertEqual((data, refs), before)

    def test_unnamed_dimension_array_is_never_assigned_by_position(self):
        event = wire_event()
        event['dimensions'] = list(event['dimensions'].values())
        with self.assertRaisesRegex(ValueError, '维度必须显式命名'):
            self.decode(event)

    def test_named_dimensions_reject_duplicate_missing_and_extra_entries(self):
        duplicate = named_wire_event()
        duplicate['dimensions'][-1] = copy.deepcopy(duplicate['dimensions'][0])
        missing = named_wire_event(); missing['dimensions'].pop()
        extra = named_wire_event(); extra['dimensions'].append(copy.deepcopy(extra['dimensions'][0]))
        for event in (duplicate, missing, extra):
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.decode(event)

    def test_named_dimensions_reject_unknown_names_and_name_coercion(self):
        for name in ('unknown', 'Action_Value', ' action_value', 'action_value ',
                     '动作价值', 0, True, None, ['action_value']):
            event = named_wire_event(); event['dimensions'][0]['dimension'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.decode(event)

    def test_named_dimensions_reject_extra_fields_missing_fields_and_mixed_shapes(self):
        variants = []
        for key in ('dimension', 'value', 'fact_indexes'):
            event = named_wire_event(); del event['dimensions'][0][key]; variants.append(event)
        event = named_wire_event(); event['dimensions'][0]['name'] = 'action_value'; variants.append(event)
        event = named_wire_event(); event['dimensions'][0] = {'action_value': {'value': 3, 'fact_indexes': [0]}}; variants.append(event)
        event = named_wire_event(); event['dimensions'][0] = 3; variants.append(event)
        for event in variants:
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.decode(event)

    def test_named_dimensions_retain_null_and_score_evidence_constraints(self):
        for changes in ({'value': None, 'fact_indexes': [0]},
                        {'value': 3, 'fact_indexes': []},
                        {'value': True, 'fact_indexes': [0]},
                        {'value': 5, 'fact_indexes': [0]},
                        {'value': 3, 'fact_indexes': [True]},
                        {'value': 3, 'fact_indexes': [4]}):
            event = named_wire_event(); event['dimensions'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.decode(event)

    def test_named_dimensions_require_unknown_without_audio_or_measured_motion(self):
        frames = [ref for ref in evidence() if ref['kind'] == 'frame']
        self.decode(named_wire_event(), frames)
        for name in ('related_laughter', 'motion_intensity'):
            for value in (0, 3):
                event = named_wire_event()
                score = next(score for score in event['dimensions'] if score['dimension'] == name)
                score.update(value=value, fact_indexes=[0])
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.decode(event, frames)
        event = named_wire_event()
        event['facts'][1]['support_id'] = 'unmeasured'
        next(score for score in event['dimensions'] if score['dimension'] == 'motion_intensity').update(
            value=3, fact_indexes=[1])
        with self.assertRaises(ValueError):
            self.decode(event)

    def test_legacy_dimension_object_still_requires_exact_nine_names(self):
        missing = wire_event(); del missing['dimensions']['player_reaction']
        extra = wire_event(); extra['dimensions']['unknown'] = {'value': None, 'fact_indexes': []}
        for event in (missing, extra):
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.decode(event)

    def test_explicit_empty_is_distinct_from_malformed_response(self):
        self.assertEqual(decode_event_response({'events': []}, evidence(), 100., 105.), {'events': []})
        for data in ({}, [], {'events': [], 'summary': 'ignored'}, {'events': None}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_event_response(data, evidence(), 100., 105.)

    def test_missing_or_extra_event_and_fact_fields_are_rejected(self):
        variants = []
        for key in ('peak_frame', 'facts', 'dimensions', 'uncertainty'):
            event = wire_event(); del event[key]; variants.append(event)
        event = wire_event(); event['start_sec'] = 101.; variants.append(event)
        event = wire_event(); del event['facts'][0]['support_id']; variants.append(event)
        event = wire_event(); event['facts'][0]['time_sec'] = 101.; variants.append(event)
        for event in variants:
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.decode(event)

    def test_unknown_frame_or_support_id_and_nonframe_anchor_are_rejected(self):
        for key, value in (('start_frame', 'invented'), ('start_frame', 'sound')):
            event = wire_event(); event[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.decode(event)
        for key, value in (('frame_id', 'invented'), ('frame_id', 'audio'),
                           ('support_id', 'invented'), ('support_id', 'f2')):
            event = wire_event(); event['facts'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.decode(event)

    def test_source_pts_not_frame_id_arithmetic_drives_time(self):
        refs = evidence()
        frame_times = [100.03, 100.8, 101.67, 103.22, 104.89]
        for ref, when in zip(refs[:5], frame_times): ref['start_sec'] = ref['end_sec'] = when
        result = self.decode(refs=refs)['events'][0]
        self.assertEqual(result['peak_sec'], 101.67)
        self.assertEqual(result['observations'][0]['time_sec'], 100.8)
        self.assertEqual(result['clip_end_sec'], 104.89)

    def test_order_errors_are_not_silently_sorted_or_fixed_by_envelope(self):
        for changes in ({'start_frame': 'f3', 'end_frame': 'f1'},
                        {'end_frame': 'f1'}, {'peak_frame': 'f0'},
                        {'clip_start_frame': 'f2'}, {'clip_end_frame': 'f2'}):
            event = wire_event(); event.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.decode(event)

    def test_clip_envelope_covers_facts_without_altering_actions_or_boundary_claim(self):
        event = wire_event()
        event.update(clip_start_frame='f1', clip_end_frame='f3', boundary_complete=False)
        event['facts'].append({'kind': 'observation', 'text': '球员面向场地',
                               'frame_id': 'f0', 'support_id': None})
        result = self.decode(event)['events'][0]
        self.assertEqual((result['clip_start_sec'], result['clip_end_sec']), (100., 104.))
        self.assertEqual((result['start_sec'], result['end_sec'], result['peak_sec']), (101., 103., 102.))
        self.assertIs(result['boundary_complete'], False)

    def test_scores_only_resolve_explicit_existing_fact_indexes(self):
        for index in (-1, 4, 100, True, False, 1., '1'):
            event = wire_event(); event['dimensions']['action_value']['fact_indexes'] = [index]
            with self.subTest(index=index), self.assertRaises(ValueError): self.decode(event)
        event = wire_event(); event['dimensions']['action_value']['fact_indexes'] = []
        with self.assertRaises(ValueError): self.decode(event)

    def test_fact_indexes_cannot_escape_to_another_event(self):
        first, second = wire_event(), wire_event()
        first['dimensions']['action_value']['fact_indexes'] = [4]
        with self.assertRaises(ValueError):
            decode_event_response({'events': [first, second]}, evidence(), 100., 105.)

    def test_unknown_scores_remain_unknown_without_synthetic_references(self):
        event = self.decode()['events'][0]
        self.assertEqual(event['dimensions']['related_laughter'], {'value': None, 'evidence_ids': []})
        self.assertEqual(event['dimensions']['motion_intensity'], {'value': None, 'evidence_ids': []})

    def test_audio_fact_uses_exact_frame_time_not_audio_interval_midpoint(self):
        event = wire_event()
        event['facts'][3]['support_id'] = 'audio'
        event['dimensions']['related_laughter'] = {'value': 3, 'fact_indexes': [3]}
        event['laughter_linked'] = True
        result = self.decode(event)['events'][0]
        self.assertEqual(result['reactions'][0]['time_sec'], 104.)
        self.assertEqual(result['reactions'][0]['evidence_ids'], ['f4', 'audio'])
        self.assertEqual(result['dimensions']['related_laughter']['evidence_ids'], ['f4', 'audio'])
        self.assertEqual(result['clip_end_sec'], 104.)

    def test_sound_support_must_overlap_fact_frame_without_relaxed_time_tolerance(self):
        event = wire_event(); event['facts'][0]['support_id'] = 'sound'
        with self.assertRaises(ValueError): self.decode(event)
        event['facts'][2]['support_id'] = 'sound'; event['facts'][0]['support_id'] = None
        result = self.decode(event)['events'][0]
        self.assertEqual(result['aftermath'][0]['evidence_ids'], ['f3', 'sound'])

    def test_laughter_still_requires_audio_support_and_explicit_link(self):
        event = wire_event()
        event['dimensions']['related_laughter'] = {'value': 3, 'fact_indexes': [2]}
        event['laughter_linked'] = True
        with self.assertRaises(ValueError): self.decode(event)
        event['facts'][2]['support_id'] = 'sound'
        self.decode(event)
        for linked in (False, None):
            event['laughter_linked'] = linked
            with self.subTest(linked=linked), self.assertRaises(ValueError): self.decode(event)

    def test_motion_still_requires_actual_camera_compensated_measurement(self):
        event = wire_event()
        event['dimensions']['motion_intensity'] = {'value': 3, 'fact_indexes': [1]}
        with self.assertRaises(ValueError): self.decode(event)
        event['facts'][1]['support_id'] = 'unmeasured'
        with self.assertRaises(ValueError): self.decode(event)
        event['facts'][1]['support_id'] = 'motion'
        result = self.decode(event)['events'][0]
        self.assertEqual(result['dimensions']['motion_intensity']['evidence_ids'], ['f2', 'motion'])

    def test_direct_visual_observation_is_required_even_with_aftermath(self):
        event = wire_event()
        for fact in event['facts']: fact['kind'] = 'aftermath'
        with self.assertRaises(ValueError): self.decode(event)

    def test_invalid_fact_kind_and_empty_text_are_rejected(self):
        for key, value in (('kind', 'hypothesis'), ('text', ''), ('frame_id', None)):
            event = wire_event(); event['facts'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.decode(event)

    def test_compact_schema_is_closed_and_has_bounded_output(self):
        format_ = wire_event_response_format(evidence(), 100., 105.)
        self.assertEqual(format_['type'], 'json_schema')
        self.assertIs(format_['json_schema']['strict'], True)
        self.assertEqual(WIRE_PROTOCOL, 'frame_anchors_v3')
        self.assertEqual(format_['json_schema']['name'], 'volleyball_event_anchors_v3')
        schema = format_['json_schema']['schema']
        event = schema['properties']['events']['items']
        self.assertEqual(schema['properties']['events']['maxItems'], 3)
        self.assertEqual(event['properties']['facts']['maxItems'], 12)
        self.assertNotIn('start_sec', event['properties'])
        self.assertEqual(set(event['required']), set(wire_event()))
        dimensions = event['properties']['dimensions']
        self.assertEqual(dimensions['type'], 'array')
        self.assertEqual((dimensions['minItems'], dimensions['maxItems']), (9, 9))
        self.assertEqual(dimensions['items'], {'$ref': '#/$defs/score'})
        self.assertEqual(set(schema['$defs']), {'frame_id', 'score'})
        for branch in schema['$defs']['score']['anyOf']:
            self.assertEqual(set(branch['required']), {'dimension', 'value', 'fact_indexes'})
            self.assertEqual(set(branch['properties']['dimension']['enum']), set(DIMENSIONS))
        def check(node):
            if isinstance(node, dict):
                if node.get('type') == 'object':
                    self.assertIs(node['additionalProperties'], False)
                    self.assertEqual(set(node['required']), set(node['properties']))
                for child in node.values(): check(child)
            elif isinstance(node, list):
                for child in node: check(child)
        check(schema)

    def test_schema_forbids_nonnull_scores_without_audio_or_measured_motion(self):
        refs = [ref for ref in evidence() if ref['kind'] == 'frame' or ref['id'] == 'unmeasured']
        schema = wire_event_response_format(refs, 100., 105.)['json_schema']['schema']
        unknown, known = schema['$defs']['score']['anyOf']
        self.assertEqual(set(unknown['properties']['dimension']['enum']), set(DIMENSIONS))
        self.assertEqual(unknown['properties']['value'], {'type': 'null'})
        self.assertEqual(unknown['properties']['fact_indexes']['maxItems'], 0)
        self.assertEqual(set(known['properties']['dimension']['enum']),
                         set(DIMENSIONS)-{'related_laughter', 'motion_intensity'})
        self.assertEqual(known['properties']['fact_indexes']['minItems'], 1)


if __name__ == '__main__': unittest.main()
