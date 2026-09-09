"""Synthetic timeline tests, independent of match-specific model outputs."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import APP, read_json, save_json
from rally import build_manifest


class RallyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source={'duration_sec':30.,'width':1000,'height':600,'frame_count':900,'start_sec':0}
        self.config=read_json(APP/'defaults.json')
        self.records=[];self.balls=[];self.pts=np.arange(900)/30+.01
        for i,t in enumerate(self.pts):
            playing=2<=t<10 or 16<=t<25
            # Real changing aerial trajectory vs held ball between rallies.
            x=500+180*np.sin(t*2) if playing else 500
            y=170+70*np.sin(t*4) if playing else 420
            self.records.append({'frame':i,'time_s':t-.01,'source_time_s':t,
                'state':'play' if playing and not 5<t<6 else 'no-play','actions':[],
                'players':[{'xyxy':[200,300,250,550],'confidence':.9}]*6})
            self.balls.append([i,1,x,y,0])
        self.write_inputs()
        save_json(self.root/'player.json',{'number':None,'detections':[]})

    def tearDown(self):self.tmp.cleanup()

    def write_inputs(self):
        with (self.root/'analytics.jsonl').open('w') as f:
            for r in self.records:f.write(json.dumps(r)+'\n')
        with (self.root/'ball.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['Frame','Visibility','X','Y','Radius']);w.writerows(self.balls)
        np.savetxt(self.root/'pts.csv',self.pts,fmt='%.8f')

    def build(self):
        return build_manifest(self.source,self.root/'analytics.jsonl',self.root/'ball.csv',self.root/'pts.csv',
                              self.root/'player.json',self.root,self.config,{})[0]

    def test_bridges_short_state_error_but_splits_retrieval(self):
        m=self.build();valid=[r for r in m['rallies'] if r['eligible']]
        self.assertEqual(len(valid),2)
        self.assertLessEqual(valid[0]['start_sec'],2.5)
        self.assertGreaterEqual(valid[0]['end_sec'],9.5)
        self.assertLess(valid[0]['safe_end_sec'],valid[1]['safe_start_sec'])
        self.assertTrue(all(r['ball_metrics']['visible_ratio']>.8 for r in valid))

    def test_misaligned_cache_rejected(self):
        self.records[450]['source_time_s']+=.2;self.write_inputs()
        with self.assertRaisesRegex(ValueError,'错位'):self.build()

    def test_missing_frame_rejected(self):
        self.balls.pop(150);self.write_inputs()
        with self.assertRaisesRegex(ValueError,'不一致'):self.build()

    def test_stationary_ball_never_qualifies_on_play_state_alone(self):
        for r in self.records:r['state']='play'
        for b in self.balls:b[2]=500;b[3]=150
        self.write_inputs();m=self.build()
        self.assertEqual(sum(r['eligible'] for r in m['rallies']),0)

    def test_player_requires_repeated_confident_evidence(self):
        baseline=self.build()['rallies'][0]['rule_score']
        save_json(self.root/'player.json',{'number':12,'sample_interval_sec':1.,'detections':[
            {'time_sec':3.,'confidence':.95},{'time_sec':4.,'confidence':.98},{'time_sec':5.,'confidence':.2}]})
        r=self.build()['rallies'][0]
        self.assertGreater(r['rule_score'],baseline)
        self.assertEqual(r['players'][0]['confirmed_samples'],2)
        save_json(self.root/'player.json',{'number':12,'detections':[{'time_sec':3.,'confidence':.99}]})
        self.assertEqual(self.build()['rallies'][0]['rule_score'],baseline)


if __name__=='__main__':unittest.main()
