import unittest

from demo import stable_clip
from monitoring_settings import DEFAULTS, earliest_reconcile_seconds, validate_settings


def sample(count, index, digest=None):
    return {'count': count, 'quality': {'brightness': 100, 'contrast': 40,
        'sharpness': 80, 'scene': [100] * 48, 'digest': digest or str(index)}}


class DemoEvidenceTests(unittest.TestCase):
    def test_repeated_or_unstable_clip_holds_result(self):
        with self.assertRaisesRegex(ValueError, 'repeat'):
            stable_clip([sample(10, i, 'same') for i in range(5)], known=10)
        with self.assertRaisesRegex(ValueError, 'unstable'):
            stable_clip([sample(count, i) for i, count in enumerate((3, 5, 7, 9, 11))],
                        error_band=1)
        with self.assertRaisesRegex(ValueError, 'starting stock'):
            stable_clip([sample(8, i) for i in range(5)], known=10)

    def test_timing_bounds_and_first_possible_census(self):
        self.assertEqual(earliest_reconcile_seconds(DEFAULTS), 190)
        self.assertEqual(earliest_reconcile_seconds({**DEFAULTS,
            'median_window_seconds':30,'consistent_windows':2}), 65)
        with self.assertRaises(ValueError):
            validate_settings({**DEFAULTS,'snapshot_seconds':0})
        with self.assertRaises(ValueError):
            validate_settings({**DEFAULTS,'zero_windows':2})
