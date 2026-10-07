"""Fresh-frame quality gates and conservative, non-overlapping census windows."""
import hashlib
import math
import statistics
from collections import deque

import cv2
import numpy as np


def frame_quality(frame):
    gray = cv2.cvtColor(cv2.resize(frame,(160,120)),cv2.COLOR_BGR2GRAY)
    return {'brightness':float(gray.mean()),'contrast':float(gray.std()),
            'sharpness':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
            'scene':cv2.resize(gray,(8,6)).astype(float).ravel().tolist(),
            'digest':hashlib.sha256(frame.tobytes()).hexdigest()}


def good_quality(quality, baseline=None):
    if quality['brightness']<20 or quality['brightness']>240 or quality['contrast']<8 or quality['sharpness']<10:
        return False
    if baseline:
        if abs(quality['brightness']-baseline['brightness'])>max(25,baseline['brightness']*.4):
            return False
        if quality['sharpness']<baseline['sharpness']*.25 or quality['contrast']<baseline['contrast']*.35:
            return False
        original = np.array(baseline['scene'])
        current = np.array(quality['scene'])
        # Normalized coarse scene changes reject moved/covered views without using fish detections as proof of visibility.
        original -= original.mean(); current -= current.mean()
        scale = max(8,float(original.std()))
        if float(np.abs(current-original).mean())/scale>1.5:
            return False
    return True


class CensusWindow:
    def __init__(self, error_band=0, window_seconds=60, minimum_samples=12, required_windows=3, zero_windows=10):
        self.error_band = error_band
        self.window_seconds = window_seconds
        self.minimum_samples = minimum_samples
        self.required_windows, self.zero_windows = required_windows, zero_windows
        self.samples = deque()
        self.candidate = None
        self.windows = 0
        self.last_sample = None
        self.last_window = None

    def reset(self):
        self.samples.clear(); self.candidate=None; self.windows=0; self.last_sample=None; self.last_window=None

    def add(self, count, observed_at, valid=True):
        if not valid:
            self.reset()
            return None
        if self.last_sample is not None and observed_at <= self.last_sample:
            return None  # Never accept cached/replayed inference as independent evidence.
        if self.last_sample is not None and observed_at-self.last_sample>10:
            self.reset()
        self.last_sample = observed_at
        self.samples.append((observed_at,count))
        if len(self.samples)<self.minimum_samples or observed_at-self.samples[0][0]<self.window_seconds:
            return None
        values = [sample[1] for sample in self.samples]
        estimate = int(round(statistics.median(values)))
        stable = sum(abs(value-estimate)<=self.error_band for value in values)/len(values)>=.8
        self.samples.clear()
        if not stable:
            self.candidate=None; self.windows=0
            return None
        self.last_window = estimate
        if self.candidate is None or abs(self.candidate-estimate)>self.error_band:
            self.candidate=estimate; self.windows=1
        else:
            self.windows+=1
        if self.windows >= (self.zero_windows if estimate==0 else self.required_windows):
            self.windows=0
            return estimate
        return None


def validation_profile(samples, known_count, fingerprint):
    if len(samples)<12 or samples[-1]['at']-samples[0]['at']<60:
        raise ValueError('Validation needs a full minute of reliable camera observations.')
    values = [sample['count'] for sample in samples]
    middle = statistics.median(values)
    deviations = sorted(abs(v-middle) for v in values)
    band = int(math.ceil(abs(middle-known_count)+deviations[min(len(deviations)-1,int(len(deviations)*.95))]))
    tolerance = max(1,math.ceil(known_count*.05))
    if known_count>0 and middle==0 or band>tolerance or sum(abs(v-known_count)<=tolerance for v in values)/len(values)<.9:
        raise ValueError('Camera counts do not match known stock reliably. Improve the whole-tank view and try again.')
    quality = {key:statistics.median([s['quality'][key] for s in samples]) for key in ('brightness','contrast','sharpness')}
    quality['scene'] = np.median([s['quality']['scene'] for s in samples],axis=0).tolist()
    return {'known_count':known_count,'error_band':band,'quality':quality,'profile':fingerprint}
