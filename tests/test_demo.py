import unittest

from demo import clip_frames, stable_clip
from monitoring_settings import DEFAULTS, earliest_reconcile_seconds, validate_settings


def sample(count, index, digest=None):
    return {'count': count, 'quality': {'brightness': 100, 'contrast': 40,
        'sharpness': 80, 'scene': [100] * 48, 'digest': digest or str(index)}}


class DemoEvidenceTests(unittest.TestCase):
    def test_demo_samples_first_25_seconds_at_shared_interval(self):
        from unittest.mock import patch
        import cv2
        import numpy as np
        class Capture:
            def __init__(self, frames): self.frames=frames; self.position=0; self.released=False
            def isOpened(self): return True
            def get(self, key):
                return 10 if key==cv2.CAP_PROP_FPS else self.frames
            def set(self, key, value): self.position=value; return True
            def read(self): return True, np.full((12,12,3),self.position%255,dtype=np.uint8)
            def release(self): self.released=True
        valid=Capture(250)
        with patch('demo.cv2.VideoCapture',return_value=valid):
            samples=clip_frames('before.mp4',5)
        self.assertEqual([offset for offset,_ in samples],[0,5,10,15,20])
        self.assertEqual([int(frame[0,0,0]) for _,frame in samples],[0,50,100,150,200])
        self.assertTrue(valid.released)
        short=Capture(240)
        with patch('demo.cv2.VideoCapture',return_value=short):
            with self.assertRaisesRegex(ValueError,'25 seconds'):
                clip_frames('short.mp4',5)
        self.assertTrue(short.released)

    def test_repeated_or_unstable_clip_holds_result(self):
        with self.assertRaisesRegex(ValueError, 'repeat'):
            stable_clip([sample(10, i, 'same') for i in range(5)], known=10)
        with self.assertRaisesRegex(ValueError, 'unstable'):
            stable_clip([sample(count, i) for i, count in enumerate((3, 5, 7, 9, 11))],
                        error_band=1)
        with self.assertRaisesRegex(ValueError, 'starting stock'):
            stable_clip([sample(8, i) for i in range(5)], known=10)

    def test_timing_bounds_and_first_possible_census(self):
        self.assertEqual(earliest_reconcile_seconds(DEFAULTS), 182)
        self.assertEqual(earliest_reconcile_seconds({**DEFAULTS,
            'median_window_seconds':30,'consistent_windows':2}), 61)
        with self.assertRaises(ValueError):
            validate_settings({**DEFAULTS,'snapshot_seconds':0})
        with self.assertRaises(ValueError):
            validate_settings({**DEFAULTS,'zero_windows':2})
        self.assertNotIn('demo_snapshot_seconds', DEFAULTS)
        self.assertEqual(validate_settings({**DEFAULTS,'demo_snapshot_seconds':2}), DEFAULTS)
