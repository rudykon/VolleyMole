import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from volleymole.common import read_json,save_json
from volleymole.match_collection import discover_matches,parse_name,merge_manifests,main
from volleymole.ranker import rule_decision,shortlist
from volleymole.schemas import validate_decision
from volleymole.sources import source_for
from volleymole.presentation import build_timeline


class DiscoveryTests(unittest.TestCase):
    def test_numeric_order_and_normalized_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('2026.1.6.10.MOV','2026.01.06.2.mp4','2026.1.6.1.mp4','2026.9.2.1.MOV','notes.txt'):(root/name).touch()
            groups=discover_matches(root)
            self.assertEqual(list(groups),['2026-01-06','2026-09-02'])
            self.assertEqual([n for n,_ in groups['2026-01-06']],[1,2,10])
            (root/'2026.01.06.1.MOV').touch()
            with self.assertRaises(ValueError):discover_matches(root)

    def test_bad_names_dates_and_set_numbers_fail_closed(self):
        for name in ('2026.2.30.1.mp4','2026.1.6.0.mp4','match.mp4','2026.1.6.-1.mp4'):
            with self.subTest(name=name),self.assertRaises(ValueError):parse_name(name)

    def test_list_does_not_start_analysis_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'2026.1.6.1.mp4').touch()
            with patch('volleymole.match_collection.execute_match') as execute:
                main(['--input-dir',str(root),'--list','--output',str(root/'out')])
                execute.assert_not_called()
            self.assertFalse((root/'out').exists())


class MultiSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.parts=[]
        for number in (1,2):
            part=self.root/'parts'/f'set-{number:04d}';part.mkdir(parents=True)
            self.parts.append((number,part));rallies=[]
            for i in range(6):
                track=f'track-{i}.json';save_json(part/track,{'samples':[[i*20+2,30,50,True]]})
                images=[f'{i}-{n}.jpg' for n in range(3)]
                for name in images:(part/name).write_bytes(b'fixture')
                rallies.append({'rally_id':f'r{i}','start_sec':i*20+2.,'end_sec':i*20+8.,
                    'safe_start_sec':i*20.,'safe_end_sec':i*20+10.,'duration_sec':6.,'eligible':True,
                    'rule_score':number*100-i,'tracking_json':track,'preview_frames':images,
                    'preview_times_sec':[i*20+2,i*20+4,i*20+7], 'actions':['set'],'players':[],
                    'ball_metrics':{'visible_ratio':.8,'trajectory_changes':5},'exclusion_reasons':[]})
            save_json(part/'match_manifest.json',{'source':{'path':str(part/'source.mp4'),'duration_sec':120.,
                'has_audio':number==1,'width':1920,'height':1080},'config':{'preview_limit':25},'rallies':rallies})
        self.manifest=merge_manifests(self.parts,self.root,'2026-01-06')

    def tearDown(self):self.temp.cleanup()

    def test_namespaced_ids_and_unchanged_local_evidence(self):
        self.assertEqual(len({r['rally_id'] for r in self.manifest['rallies']}),12)
        for r in self.manifest['rallies']:
            self.assertTrue((self.root/r['tracking_json']).is_file())
            source=source_for(self.manifest,r['rally_id'])
            self.assertEqual(source['set_number'],r['source_set'])
        self.assertEqual(self.manifest['rallies'][0]['start_sec'],self.manifest['rallies'][6]['start_sec'])
        self.assertEqual(read_json(self.parts[1][1]/'track-0.json')['samples'][0][0],2)
        self.assertEqual(read_json(self.parts[1][1]/'match_manifest.json')['rallies'][0]['rally_id'],'r0')

    def test_global_top10_not_per_set_top5_and_overlapping_clocks_allowed(self):
        decision=rule_decision(shortlist(self.manifest,10),10)
        validate_decision(decision,self.manifest,self.root,10)
        self.assertEqual(sum(i['rally_id'].startswith('set-0002') for i in decision['selected']),6)
        timeline=build_timeline(decision,self.manifest)
        for segment in timeline:
            if 'rally_id' in segment:
                self.assertEqual(segment['source_set'],source_for(self.manifest,segment['rally_id'])['set_number'])
        self.assertEqual(len([s for s in timeline if s['kind']=='rally']),10)

    def test_same_source_overlap_invalid_and_local_duration_enforced(self):
        manifest=copy.deepcopy(self.manifest)
        decision=rule_decision(manifest['rallies'],10)
        first=decision['selected'][0];second=decision['selected'][1]
        r=next(r for r in manifest['rallies'] if r['rally_id']==second['rally_id'])
        a=next(r for r in manifest['rallies'] if r['rally_id']==first['rally_id'])
        for key in ('start_sec','end_sec','safe_start_sec','safe_end_sec'):r[key]=a[key]
        second['clip_start_sec']=first['clip_start_sec'];second['clip_end_sec']=first['clip_end_sec']
        with self.assertRaises(ValueError):validate_decision(decision,manifest,self.root,10)
        manifest=copy.deepcopy(self.manifest);decision=rule_decision(manifest['rallies'],10)
        source_for(manifest,decision['selected'][0]['rally_id'])['duration_sec']=1
        with self.assertRaises(ValueError):validate_decision(decision,manifest,self.root,10)

    def test_missing_source_and_inconsistent_config_rejected(self):
        broken=copy.deepcopy(self.manifest);broken['rallies'][0]['source_id']='missing'
        with self.assertRaises(ValueError):source_for(broken,broken['rallies'][0]['rally_id'])
        path=self.parts[1][1]/'match_manifest.json';data=read_json(path);data['config']['preview_limit']=99;save_json(path,data)
        with self.assertRaises(ValueError):merge_manifests(self.parts,self.root,'2026-01-06')

    def test_less_than_ten_global_candidates_never_fabricates(self):
        self.manifest['rallies']=self.manifest['rallies'][:9]
        with self.assertRaises(ValueError):shortlist(self.manifest,10)
