"""Explicit, isolated recorded-video rehearsal for the defense demonstration."""
import asyncio
import math
import statistics
import time
import uuid

import cv2
import numpy as np

from census import frame_quality, good_quality
from detection_profile import fingerprint
from farm_database import now_text


def clip_frames(path, seconds):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or frames / fps < 9:
            raise ValueError('Use a readable clip of about 10 seconds showing the whole aquarium.')
        selected = []
        for offset in range(0, 10, seconds):
            if offset * fps >= frames:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(offset * fps))
            ok, frame = capture.read()
            if not ok:
                break
            selected.append(frame)
        if len(selected) < 5:
            raise ValueError('The clip needs at least five distinct sampled moments.')
        return selected
    finally:
        capture.release()


def stable_clip(samples, known=None, baseline_quality=None, error_band=None):
    if len(samples) < 5 or len({s['quality']['digest'] for s in samples}) < 5:
        raise ValueError('Clip frames repeat or are too short for a reliable count.')
    if not all(good_quality(s['quality'], baseline_quality) for s in samples):
        raise ValueError('Clip view is blurred, dark, obstructed, or differs from the baseline view.')
    counts = [int(s['count']) for s in samples]
    middle = int(round(statistics.median(counts)))
    tolerance = max(1, math.ceil((known if known is not None else middle) * .05))
    allowed = error_band if error_band is not None else tolerance
    if sum(abs(value-middle) <= allowed for value in counts) / len(counts) < .8:
        raise ValueError('Clip counts are unstable. Use a clear whole-aquarium recording.')
    if known is not None and (abs(middle-known) > tolerance or
                              sum(abs(value-known) <= tolerance for value in counts) / len(counts) < .8):
        raise ValueError('Before clip does not reliably match the independently verified starting stock.')
    quality = {key: statistics.median(s['quality'][key] for s in samples)
               for key in ('brightness', 'contrast', 'sharpness')}
    quality['scene'] = np.median([s['quality']['scene'] for s in samples], axis=0).tolist()
    inliers = [abs(value-middle) for value in counts if abs(value-middle) <= allowed]
    return middle, quality, max(1, max(inliers))


class DemoManager:
    def __init__(self, accounts, production):
        self.accounts, self.production = accounts, production
        self.task = None
        self.lock = asyncio.Lock()
        self.phase = 'idle'
        self.message = 'Configure the demo farm and clips, then start the before phase.'
        self.before_count = None
        self.before_quality = None
        self.error_band = None
        self.run_id = None
        self.updated_at = None
        self.preview_frame = None
        self.reconcile_at = None

    async def status(self):
        return {'farm': await self.accounts.demo_farm(), 'setup': await self.accounts.demo_setup(),
                'phase': self.phase, 'message': self.message, 'before_count': self.before_count,
                'run_id': self.run_id, 'updated_at': self.updated_at, 'simulated': True,
                'preview_frame': self.preview_frame,
                'remaining_seconds': max(0, math.ceil(self.reconcile_at-time.monotonic())) if self.reconcile_at else None}

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.task = None

    async def reset(self):
        async with self.lock:
            await self.stop()
            setup = await self.accounts.demo_setup()
            farm = await self.accounts.demo_farm()
            if setup and farm and setup['farm_id'] == farm['id']:
                database = await self.accounts.get_database(farm['id'])
                await database.reset_demo_run(setup['tank_id'], setup['starting_count'])
            self.phase = 'idle'
            self.message = 'Demo records cleared; uploaded clips are ready to reuse.'
            self.before_count = self.before_quality = self.error_band = self.run_id = None
            self.preview_frame = self.reconcile_at = None
            self.updated_at = now_text()
        return await self.status()

    async def start_phase(self, phase):
        if phase not in ('before', 'after'):
            raise ValueError('Choose the before or after phase.')
        async with self.lock:
            if self.production.enabled:
                raise ValueError('Turn off Production Mode before running the recorded demo.')
            setup = await self.accounts.demo_setup()
            if not setup:
                raise ValueError('Set up the demo tank and both clips first.')
            if self.task and not self.task.done():
                raise ValueError('The previous demo phase is still running.')
            if phase == 'after' and self.phase != 'before_ready':
                raise ValueError('Validate the before clip first.')
            if phase == 'before' and self.phase not in ('idle', 'held'):
                raise ValueError('Reset or finish the current demo run before starting again.')
            if phase == 'before':
                farm = await self.accounts.demo_farm()
                database = await self.accounts.get_database(farm['id'])
                await database.reset_demo_run(setup['tank_id'], setup['starting_count'])
                self.before_count = self.before_quality = self.error_band = None
                self.preview_frame = None
                self.run_id = uuid.uuid4().hex
            self.phase = phase + '_processing'
            self.message = 'Sampling distinct moments from the recorded clip.'
            self.updated_at = now_text()
            self.task = asyncio.create_task(self._run(phase, setup, time.monotonic()))
        return await self.status()

    async def _run(self, phase, setup, switched_at):
        try:
            farm_id = setup['farm_id']
            database = await self.accounts.get_database(farm_id)
            settings = self.production.settings
            frames = await asyncio.to_thread(clip_frames, setup[phase+'_source'], settings['demo_snapshot_seconds'])
            samples = []
            seen = set()
            async with self.production._context(farm_id, database):
                tank = await database.get_tank(setup['tank_id'])
                last_log = 0
                for index, frame in enumerate(frames, 1):
                    quality = await asyncio.to_thread(frame_quality, frame)
                    if quality['digest'] in seen or not good_quality(
                            quality, self.before_quality if phase == 'after' else None):
                        raise ValueError('Clip contains repeated, blurred, dark, or mismatched aquarium frames.')
                    seen.add(quality['digest'])
                    telemetry, last_log, _ = await self.production._infer(
                        frame, 'recorded_demo_'+self.run_id, index, last_log, tank,
                        snapshot_seconds=settings['demo_snapshot_seconds'], source_name='recorded_demo')
                    if telemetry.get('frame'):
                        self.preview_frame = telemetry['frame']
                    samples.append({'count': telemetry['live_count'], 'quality': quality})
            if phase == 'before':
                observed, quality, band = stable_clip(samples, known=setup['starting_count'])
                self.before_count, self.before_quality, self.error_band = observed, quality, band
                profile = {'known_count': setup['starting_count'], 'error_band': band,
                           'quality': quality, 'profile': fingerprint(), 'source': 'recorded_demo',
                           'validated_at': now_text()}
                await database.configure_monitoring(setup['tank_id'], validation=profile)
                self.phase = 'before_ready'
                self.message = f'Before clip validated at {observed} fish. Start the after phase.'
            else:
                observed, _, _ = stable_clip(samples, baseline_quality=self.before_quality,
                                             error_band=self.error_band)
                if observed == 0 and self.before_count > 0:
                    raise ValueError('An empty-tank result needs a longer live validation; demo inventory is held.')
                self.phase = 'after_waiting'
                self.message = f'After clip counted {observed} fish. Waiting for the demo reconciliation interval.'
                self.updated_at = now_text()
                self.reconcile_at = switched_at + settings['demo_reconcile_seconds']
                await asyncio.sleep(max(0, settings['demo_reconcile_seconds'] -
                                        (time.monotonic() - switched_at)))
                config = next(c for c in await database.monitoring_configs() if c['tank_id'] == setup['tank_id'])
                await database.reconcile_census(setup['tank_id'], observed, self.run_id,
                    config['revision'], self.before_count, now_text(),
                    {'source': 'recorded_demo', 'simulated': True, 'demo_run_id': self.run_id,
                     'before_clip': setup['before_source'], 'after_clip': setup['after_source']})
                self.phase = 'complete'
                self.reconcile_at = None
                self.message = f'Simulated demo reconciliation complete: {self.before_count} to {observed} fish.'
            self.updated_at = now_text()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.phase = 'held'
            self.reconcile_at = None
            self.message = str(exc) if isinstance(exc, ValueError) else 'Demo counting failed. Check the clip and counting service, then retry.'
            self.updated_at = now_text()
