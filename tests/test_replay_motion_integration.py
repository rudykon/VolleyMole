"""Cache and render gates must consume current independent replay evidence."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_automatic_replay as fixtures

from volleymole.common import identity, read_json, save_json
from volleymole.replay_motion import TIME_BASIS
from volleymole.replay_review import review_rally
from volleymole.replay_stage import run_review, load_reviewed_manifest
from volleymole.run_match import argument_parser


class ReplayMotionIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.video = self.root/'source.mp4'
        self.video.write_bytes(b'local video identity fixture; pixels supplied by sampler')
        self.source = dict(path=str(self.video),identity=identity(self.video),duration_sec=20.,
                           nominal_fps='30/1',start_sec=0.,rotation=0,width=1920,height=1080)
        self.item = dict(rank=1,rally_id='r1',clip_start_sec=0.,clip_end_sec=20.,title='test')
        self.track = self.root/'tracking.json'
        save_json(self.track,dict(time_basis=TIME_BASIS,coordinate_interpolation_used=False,
                  samples=[[round(i/30,8),900.,540.,True] for i in range(601)],
                  sample_origins=['vball']*601))
        self.manifest = dict(source=self.source,rallies=[dict(rally_id='r1',tracking_json='tracking.json')])
        save_json(self.root/'match_manifest.json',self.manifest)
        save_json(self.root/'edit_decision.json',dict(selected=[self.item],ranking_mode='rules_fallback'))
        self.args = argument_parser().parse_args(['--video',str(self.video),'--model','configured-vision',
            '--api-base','https://example.invalid/v1','--replay-review','required','--semantic-retries','0'])
        self.model = fixtures.Model()
        patches = [patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'fixture-key'}),
                   patch('volleymole.replay_review.sampled_evidence',side_effect=fixtures.sample),
                   patch('volleymole.replay_review.request_json',side_effect=self.model),
                   # This test exercises decisions and files, not native decode.
                   patch('volleymole.replay_requests.prepare_visual_context',
                         side_effect=lambda evidence,content,*a,**kw:(content,dict(frame_count=len(evidence))))]
        for mock in patches:
            mock.start(); self.addCleanup(mock.stop)

    def test_changed_track_invalidates_top_level_complete_cache_and_render(self):
        with patch('volleymole.replay_stage.review_rally',wraps=review_rally) as reviewer:
            first = run_review(self.root,self.args)
            self.assertEqual(reviewer.call_count,1)
            changed = read_json(self.track)
            for row in changed['samples']: row[1] += 80.
            save_json(self.track,changed)
            with self.assertRaisesRegex(ValueError,'球轨迹已改变'):
                load_reviewed_manifest(self.root)
            second = run_review(self.root,self.args)
            self.assertEqual(reviewer.call_count,2)
        self.assertNotEqual(first['tracking_inputs'],second['tracking_inputs'])
        self.assertNotEqual(first['signature'],second['signature'])
        load_reviewed_manifest(self.root)

    def test_unchanged_current_evidence_reuses_complete_result_without_model(self):
        first = run_review(self.root,self.args); calls = len(self.model.calls)
        with patch('volleymole.replay_stage.review_rally') as reviewer:
            second = run_review(self.root,self.args)
            reviewer.assert_not_called()
        self.assertEqual(first,second)
        self.assertEqual(len(self.model.calls),calls)
        load_reviewed_manifest(self.root)

    def test_missing_track_becomes_failed_review_not_cached_approval(self):
        run_review(self.root,self.args); self.track.unlink()
        with self.assertRaisesRegex(RuntimeError,'未全部完成'):
            run_review(self.root,self.args)
        report = read_json(self.root/'replay_reviews.json')
        self.assertEqual(report['failed_count'],1)
        self.assertEqual(report['approved_count'],0)
        with self.assertRaisesRegex(ValueError,'不能绕过复核'):
            load_reviewed_manifest(self.root)

    def test_required_rejects_valid_omission_while_explicit_auto_allows_it(self):
        self.model.only_set = True
        with self.assertRaisesRegex(RuntimeError,'未覆盖全部入选回合'):
            run_review(self.root,self.args)
        report = read_json(self.root/'replay_reviews.json')
        self.assertEqual(report['status'],'partial')
        self.assertEqual(report['reason'],'coverage_incomplete')
        self.assertEqual(report['approved_count'],0)
        self.assertEqual(report['omitted_count'],1)
        self.assertEqual(report['failed_count'],0)
        self.assertEqual(report['results']['r1']['review']['status'],'omit')
        with self.assertRaisesRegex(ValueError,'不能绕过复核'):
            load_reviewed_manifest(self.root)
        calls = len(self.model.calls)
        self.args.replay_review='auto'
        report = run_review(self.root,self.args)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['omitted_count'],1)
        self.assertEqual(len(self.model.calls),calls)
        manifest,_ = load_reviewed_manifest(self.root)
        self.assertEqual(manifest['rallies'][0]['replay_review']['status'],'omit')

    def test_required_cannot_forgive_omission_by_only_changing_report_status(self):
        self.model.only_set=True
        self.args.replay_review='auto'
        report=run_review(self.root,self.args)
        report['mode']='required'; report['status']='complete'
        save_json(self.root/'replay_reviews.json',report)
        with self.assertRaisesRegex(ValueError,'未覆盖全部入选回合'):
            load_reviewed_manifest(self.root)

    def test_old_policy_requires_revalidation_even_if_artifact_hashes_match(self):
        report = run_review(self.root,self.args); calls = len(self.model.calls)
        del report['automatic_policy_sha256']; save_json(self.root/'replay_reviews.json',report)
        with self.assertRaisesRegex(ValueError,'复核策略已更新'):
            load_reviewed_manifest(self.root)
        renewed = run_review(self.root,self.args)
        self.assertIn('automatic_policy_sha256',renewed)
        self.assertEqual(len(self.model.calls),calls)

    def test_legacy_embedded_automatic_approval_is_not_a_manual_reference(self):
        report = run_review(self.root,self.args)
        manifest = copy.deepcopy(self.manifest)
        manifest['rallies'][0]['replay_review'] = report['results']['r1']['review']
        save_json(self.root/'match_manifest.json',manifest)
        (self.root/'replay_reviews.json').unlink()
        with self.assertRaisesRegex(ValueError,'缺少独立核验'):
            load_reviewed_manifest(self.root)
        self.args.model = self.args.vision_model = None
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'','OPENAI_API_KEY':''}):
            with self.assertRaisesRegex(RuntimeError,'缺少模型'):
                run_review(self.root,self.args)

    def test_human_reference_still_works_without_tracking_or_api(self):
        report = run_review(self.root,self.args)
        manifest = copy.deepcopy(self.manifest)
        manual = copy.deepcopy(report['results']['r1']['review']); manual['method']='assistant_visual_review'
        manifest['rallies'][0]['replay_review'] = manual
        save_json(self.root/'match_manifest.json',manifest); self.track.unlink()
        self.args.model = self.args.vision_model = None
        with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'','OPENAI_API_KEY':''}), \
                patch('volleymole.replay_stage.review_rally') as reviewer:
            result = run_review(self.root,self.args)
            reviewer.assert_not_called()
        self.assertEqual(result['approved_count'],1)
        load_reviewed_manifest(self.root)

    def _rewrite_cache(self, report, cache):
        first = report['results']['r1']['artifacts'][0]
        save_json(first['path'],cache)
        report['results']['r1']['artifacts'][0] = identity(first['path'])
        save_json(self.root/'replay_reviews.json',report)

    def test_hash_consistent_cache_without_verify_phase_is_rejected(self):
        report = run_review(self.root,self.args)
        cache = read_json(report['results']['r1']['artifacts'][0]['path'])
        for request in cache['requests']:
            if request['phase']=='verify': request['phase']='review'
        self._rewrite_cache(report,cache)
        with self.assertRaisesRegex(ValueError,'缺少独立事件核验'):
            load_reviewed_manifest(self.root)

    def test_matching_sidecar_and_cache_cannot_invent_new_render_anchors(self):
        report = run_review(self.root,self.args)
        cache = read_json(report['results']['r1']['artifacts'][0]['path'])
        report['results']['r1']['review']['source_end_sec'] += .75
        cache['review'] = copy.deepcopy(report['results']['r1']['review'])
        self._rewrite_cache(report,cache)
        with self.assertRaisesRegex(ValueError,'实际锚点不一致'):
            load_reviewed_manifest(self.root)

    def test_hash_consistent_set_claim_still_fails_semantic_quality_gate(self):
        report = run_review(self.root,self.args)
        cache = read_json(report['results']['r1']['artifacts'][0]['path'])
        verify = next(r for r in cache['requests'] if r['phase']=='verify')
        raw = read_json(verify['path']); raw['raw']['contact_type']='two_hand_set'
        save_json(verify['path'],raw); fresh = identity(verify['path']); verify['sha256']=fresh['sha256']
        report['results']['r1']['artifacts'] = [fresh if a['path']==fresh['path'] else a
                                               for a in report['results']['r1']['artifacts']]
        self._rewrite_cache(report,cache)
        with self.assertRaisesRegex(ValueError,'未通过当前动作与完整性门禁'):
            load_reviewed_manifest(self.root)


if __name__ == '__main__':
    unittest.main()
