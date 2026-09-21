"""Encode real single-view clips and verify their source picture and sound."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from volleymole.common import probe, read_json, save_json
from volleymole.font_support import DEFAULT_FONT
from volleymole.media_worker import concatenate_segments, frame_at, render_clip
from volleymole.presentation import effect_clip
from volleymole.check_alignment import check, single_view_error


@unittest.skipUnless(shutil.which('ffmpeg'), 'requires FFmpeg')
class SingleViewTests(unittest.TestCase):
    def test_encoded_rallies_replays_and_teasers_keep_one_source_picture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);video=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'testsrc2=size=960x540:rate=30:duration=3','-f','lavfi','-i',
                'sine=frequency=391:sample_rate=48000:duration=3',
                '-c:v','libx264','-crf','12','-threads','2','-c:a','aac',str(video)],check=True)
            source=probe(video)
            rally={'rally_id':'r1','tracking_json':'track.json'}
            manifest={'source':source,'rallies':[rally]}
            item={'rally_id':'r1','rank':1,'title':'排球','clip_start_sec':0.,'clip_end_sec':3.}
            for style,quality in [('classic','720p'),('lively','1080p')]:
                with self.subTest(style=style,quality=quality):
                    directory=root/style;directory.mkdir()
                    save_json(directory/'match_manifest.json',manifest)
                    save_json(directory/'edit_decision.json',{'selected':[item]})
                    save_json(directory/'track.json',{'samples':[[i/30,320+i*3,270,True] for i in range(90)]})
                    # UI artwork is separately tested; this regression needs no
                    # unpublished fonts or illustration bundle on the CI runner.
                    header=np.zeros((110,720,4),np.uint8);header[20:90,20:150]=[220,20,220,255]
                    with patch('volleymole.presentation.lively_headers',return_value=[header]):
                        clip=render_clip(directory,item,manifest,DEFAULT_FONT,style=style,quality=quality)
                    self.assertEqual(clip['view_layout'],'single')
                    self.assertIsNone(clip['detail_layout']['overview_top'])
                    self.assertEqual(clip['simultaneous_views'],1)
                    self.assertEqual(clip['output_frames'],90)
                    self.assertGreater(len(set(clip['camera_crop_left'])),10)
                    for when in (.5,1.3,2.4):
                        picture=frame_at(clip['path'],when,720)
                        result=single_view_error(source,clip,when,picture)
                        self.assertLess(result['picture_mean_abs_error'],12)
                        # The old full-court inset must be detected as the wrong
                        # picture even if the upper follow-camera image is valid.
                        picture[875:]=cv2.resize(frame_at(video,when,960),(720,405))
                        self.assertGreater(single_view_error(source,clip,when,picture)['picture_mean_abs_error'],12)
                    clip['timeline_start_sec']=0.
                    segments=[dict(clip,kind='rally',playback_rate=1.,index=0)]
                    if style=='lively':
                        offset=3.
                        for kind,start,end,rate in [('replay',.4,2.4,2/3),('teaser',.5,1.5,1.)]:
                            overlay=directory/f'{kind}.png';Image.new('RGBA',(720,1280)).save(overlay)
                            segment={'index':len(segments),'rank':1,'rally_id':'r1','kind':kind,
                                'source_start_sec':start,'source_end_sec':end,'playback_rate':rate,
                                'duration_sec':(end-start)/rate,'output_frames':round((end-start)/rate*30),
                                'timeline_start_sec':offset,'path':str(directory/f'{kind}.mp4')}
                            effect_clip(clip['path'],Path(segment['path']),segment,start,overlay,quality=quality)
                            self.assertEqual(probe(segment['path'])['frame_count'],segment['output_frames'])
                            segments.append(segment);offset+=segment['duration_sec']
                    output=directory/'compilation.mp4'
                    concatenate_segments(segments,output,quality)
                    report={'clips':[clip],'segments':segments,'output':str(output),'view_layout':'single'}
                    suffix='_lively' if style=='lively' else ''
                    save_json(directory/f'render_report{suffix}.json',report)
                    check(directory,style)
                    verified=read_json(directory/f'alignment_verification{suffix}.json')
                    self.assertEqual(verified['status'],'passed')
                    self.assertIn('single-view',verified['method'])
                    self.assertEqual(len(verified['effects']),2 if style=='lively' else 0)

    def test_center_fallback_is_a_single_view_and_bad_camera_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);video=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'testsrc2=size=640x360:rate=30:duration=1','-c:v','libx264','-threads','2',str(video)],check=True)
            source=probe(video)
            item={'rank':1,'rally_id':'r','title':'排球','clip_start_sec':0.,'clip_end_sec':1.}
            manifest={'source':source,'rallies':[{'rally_id':'r','tracking_json':'track.json'}]}
            save_json(root/'edit_decision.json',{'selected':[item]});save_json(root/'track.json',{'samples':[]})
            clip=render_clip(root,item,manifest,DEFAULT_FONT)
            self.assertEqual(clip['crop_mode'],'center_fallback')
            self.assertEqual(len(set(clip['camera_crop_left'])),1)
            pixels=frame_at(clip['path'],.5,720)
            self.assertLess(single_view_error(source,clip,.5,pixels)['picture_mean_abs_error'],12)
            clip['camera_crop_left']=[]
            with self.assertRaises(ValueError):single_view_error(source,clip,.5,pixels)


if __name__=='__main__':unittest.main()
