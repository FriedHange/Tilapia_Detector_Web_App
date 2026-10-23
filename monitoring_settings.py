"""Validated installation-wide camera and census timing."""
import math

DEFAULTS = {
    'snapshot_seconds': 5,
    'median_window_seconds': 60,
    'consistent_windows': 3,
    'zero_windows': 10,
    'demo_snapshot_seconds': 2,
    'demo_reconcile_seconds': 30,
}
LIMITS = {
    'snapshot_seconds': (1, 5),
    'median_window_seconds': (30, 300),
    'consistent_windows': (2, 6),
    'zero_windows': (6, 20),
    'demo_snapshot_seconds': (1, 2),
    'demo_reconcile_seconds': (20, 60),
}


def validate_settings(data):
    if not isinstance(data, dict) or set(data) != set(DEFAULTS):
        raise ValueError('Send every monitoring timing setting.')
    result = {}
    for key, (low, high) in LIMITS.items():
        value = data[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key} must be a whole number from {low} to {high}.')
        result[key] = value
    if result['zero_windows'] < result['consistent_windows']:
        raise ValueError('Empty-tank confirmation cannot be shorter than regular confirmation.')
    return result


def minimum_samples(settings):
    return max(6, math.ceil(settings['median_window_seconds'] / settings['snapshot_seconds']) + 1)


def earliest_reconcile_seconds(settings):
    # Non-overlapping windows begin after the prior window completes.
    return (settings['median_window_seconds'] + settings['snapshot_seconds']) * settings['consistent_windows'] - settings['snapshot_seconds']
