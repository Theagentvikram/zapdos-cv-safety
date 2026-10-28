import copy
import unittest
from src.ppe_pipeline.reference import score_reference, validate_reference
from src.ppe_pipeline.calibration import assess_calibration
from src.ppe_pipeline.verification import head_crop
import numpy as np
from pathlib import Path
from src.ppe_pipeline.pipeline import camera_for, output_component


class CameraSelectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = dict(id='known', scene='floor', match='original_', zones='floor.json',
                           helmet_absence_validated=True, vest_absence_validated=False)
        self.cfg = {'cameras': [self.camera]}

    def test_explicit_camera_loads_saved_policy_for_new_filename(self):
        result = camera_for(Path('renamed.mp4'), self.cfg, 'known')
        self.assertEqual(result['zones'], 'floor.json')
        self.assertTrue(result['helmet_absence_validated'])
        self.assertIsNot(result, self.camera)

    def test_unknown_camera_does_not_borrow_filename_policy(self):
        result = camera_for(Path('original_video.mp4'), self.cfg, 'new')
        self.assertNotIn('zones', result)
        self.assertFalse(result['helmet_absence_validated'])

    def test_long_output_names_are_short_and_collision_resistant(self):
        first = output_component('camera_' * 30 + 'a')
        self.assertLessEqual(len(first), 40)
        self.assertEqual(first, output_component('camera_' * 30 + 'a'))
        self.assertNotEqual(first, output_component('camera_' * 30 + 'b'))


class ReferenceTests(unittest.TestCase):
    def data(self):
        return {"reviewed":True,"human_reviewed":False,"reviewer":{"kind":"agent"},
                "frames":[{"id":"a","reviewed":True,"camera_id":"cam","split":"test",
                           "persons":[{"person_id":"P1","bbox":[0,0,100,200],"helmet":"off","vest":"unknown"}]}]}

    def test_unknown_predictions_are_abstentions_not_correct_negatives(self):
        r=score_reference(self.data(),{"a":[{"bbox":[0,0,100,200],"helmet":"unknown"}]})
        self.assertEqual(r['test']['ppe']['helmet']['abstained'],1)
        self.assertEqual(r['test']['ppe']['helmet']['correct'],0)
        self.assertEqual(r['test']['ppe']['vest']['known_labels'],0)
        self.assertFalse(r['human_reviewed'])

    def test_missing_people_count_as_misses(self):
        r=score_reference(self.data(),{"a":[]})
        self.assertEqual(r['test']['persons']['missed'],1)
        self.assertEqual(r['test']['ppe']['helmet']['abstained'],1)

    def test_extra_person_is_false_positive(self):
        r=score_reference(self.data(),{'a':[{'bbox':[0,0,100,200],'helmet':'on'},{'bbox':[500,0,600,200]}]})
        self.assertEqual(r['test']['persons']['false'],1)
        self.assertEqual(r['test']['ppe']['helmet']['false_on'],1)

    def test_camera_split_leakage_is_rejected(self):
        d=self.data(); f=copy.deepcopy(d['frames'][0]); f.update(id='b',split='development'); d['frames'].append(f)
        with self.assertRaises(ValueError):validate_reference(d)

    def test_crop_abstains_on_tiny_heads(self):
        self.assertIsNone(head_crop(np.zeros((300,300,3),np.uint8),(0,0,10,30)))


class CalibrationTests(unittest.TestCase):
    def evidence(self):
        return {'reviewed':True,'human_reviewed':True,'reviewer':{'kind':'human'},
                'complete_exposure_reviewed':True,'camera_id':'cam','type':'no_helmet',
                'exposure_hours':500,'true':25,'false':0,'duplicates':0,'missed':0}

    def test_agent_labels_cannot_qualify_camera(self):
        e=self.evidence(); e['reviewer']['kind']='agent'
        self.assertFalse(assess_calibration(e)['eligible'])

    def test_short_zero_alert_run_cannot_qualify_camera(self):
        e=self.evidence(); e['exposure_hours']=15/3600
        self.assertFalse(assess_calibration(e)['eligible'])

    def test_false_and_duplicate_alerts_fail(self):
        for key in ('false','duplicates'):
            e=self.evidence(); e[key]=1
            self.assertFalse(assess_calibration(e)['eligible'])

    def test_sufficient_reviewed_exposure_and_events_can_pass(self):
        result=assess_calibration(self.evidence())
        self.assertTrue(result['eligible'])
        self.assertAlmostEqual(result['required_zero_false_alert_hours'],479.3171637686384)

    def test_invalid_exposure_fails(self):
        e=self.evidence();e['exposure_hours']=float('nan')
        with self.assertRaises(ValueError):assess_calibration(e)
