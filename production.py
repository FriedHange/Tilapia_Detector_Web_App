"""Installation-wide production supervisor and farm-owned live camera workers."""
import asyncio
import contextlib
import json
import logging
import time
import uuid
from urllib.parse import urlsplit
from dataclasses import dataclass, field

from access import current_actor, current_database, current_farm, current_session
from camera_process import ProcessCamera
from census import CensusWindow, frame_quality, good_quality, validation_profile
from detection_profile import fingerprint
from farm_database import now_text

log = logging.getLogger('tilapia.production')


def is_live(source):
    return bool(source) and (str(source).isdigit() or str(source).startswith(('rtsp://','rtsps://')))


def source_key(source):
    if str(source).isdigit():
        return 'camera:'+str(int(source))
    parsed=urlsplit(source)
    return (parsed.scheme,parsed.hostname.lower(),parsed.port or 554,parsed.path,parsed.query)


@dataclass
class TankWorker:
    farm: str
    config: dict
    task: object = None
    status: str = 'starting'
    message: str = 'Opening live camera.'
    telemetry: dict = field(default_factory=dict)
    subscribers: set = field(default_factory=set)

    def publish(self, event):
        for queue in tuple(self.subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)

    def set_status(self, status, message):
        self.status, self.message = status, message
        if status != 'running':
            self.telemetry = {}
        self.publish({'type':'stream_state','tank_id':self.config['tank_id'],
                      'status':status,'message':message,'source_kind':'camera','production':True})


class ProductionMonitor:
    def __init__(self, accounts, state_factory, processor, engine_ready, capture_factory=ProcessCamera,window_factory=CensusWindow):
        self.accounts, self.state_factory, self.processor = accounts,state_factory,processor
        self.engine_ready, self.capture_factory = engine_ready,capture_factory
        self.window_factory=window_factory
        self.enabled = False
        self.workers, self.validations, self.validation_status = {},{},{}
        self.validation_sources={}
        self.task = None
        self.lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()
        self.inference_started = None
        self.last_prune = 0
        self.heartbeat = None
        self.instance_lock = None

    async def start(self):
        # One server process owns cameras. Uvicorn must use one worker.
        self.instance_lock = open(self.accounts.path.with_suffix('.monitor.lock'),'a+b')
        try:
            import os
            if os.name == 'nt':
                import msvcrt
                self.instance_lock.seek(0); self.instance_lock.write(b'0'); self.instance_lock.flush(); self.instance_lock.seek(0)
                msvcrt.locking(self.instance_lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.instance_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.instance_lock.close(); self.instance_lock=None
            raise RuntimeError('Another server already owns this installation. Run one Uvicorn worker.')
        self.enabled = await self.accounts.production_enabled()
        self.heartbeat = self.accounts.path.parent/'monitor-heartbeat.json'
        self.task = asyncio.create_task(self._supervise())

    async def close(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        for task in self.validations.values():
            task.cancel()
        await asyncio.gather(*self.validations.values(),return_exceptions=True)
        self.validations.clear()
        self.validation_sources.clear()
        for worker in list(self.workers.values()):
            await self._stop(worker)
        self.workers.clear()
        self.validation_status.clear()
        if self.instance_lock:
            self.instance_lock.close(); self.instance_lock=None
        if self.heartbeat:
            self.heartbeat.unlink(missing_ok=True)

    async def set_enabled(self, enabled, actor):
        async with self.lock:
            await self.accounts.set_production(enabled,actor)
            self.enabled = enabled
            if not enabled:
                for worker in list(self.workers.values()):
                    await self._stop(worker)
                self.workers.clear()
        await self.sync()

    async def _stop(self, worker):
        worker.set_status('stopped','Monitoring paused.')
        if worker.task:
            worker.task.cancel()
            await asyncio.gather(worker.task,return_exceptions=True)

    async def remove_tank(self, farm, tank_id):
        async with self.lock:
            key = (farm, tank_id)
            validation = self.validations.get(key)
            if validation:
                validation.cancel()
                await asyncio.gather(validation, return_exceptions=True)
            self.validations.pop(key, None)
            self.validation_status.pop(key, None)
            self.validation_sources.pop(key, None)
            worker = self.workers.pop(key, None)
            if worker:
                await self._stop(worker)
        await self.sync()

    async def sync(self):
        async with self.lock:
            desired = {}
            if self.enabled:
                used_sources = set(self.validation_sources.values())
                for farm in await self.accounts.rows('SELECT id FROM farms ORDER BY created_at,id'):
                    database = await self.accounts.get_database(farm['id'])
                    for config in await database.monitoring_configs():
                        key=(farm['id'],config['tank_id'])
                        if not config['enabled'] or config['status']=='inactive' or key in self.validations:
                            continue
                        profile = json.loads(config['validation']) if config['validation'] else None
                        if not is_live(config['live_source']) or not profile or profile.get('profile')!=fingerprint():
                            await database.monitoring_alert('setup:'+config['tank_id'],
                                'Set a live camera and validate its whole-tank census before automatic updates.',config['tank_id'])
                            continue
                        await database.monitoring_alert('setup:'+config['tank_id'])
                        source = source_key(config['live_source'])
                        if source in used_sources:
                            await database.monitoring_alert('duplicate:'+config['tank_id'],'This camera is already assigned to another monitored tank.',config['tank_id'])
                            continue
                        used_sources.add(source)
                        await database.monitoring_alert('duplicate:'+config['tank_id'])
                        desired[key] = config
            for key, worker in list(self.workers.items()):
                config = desired.get(key)
                if not config or config['revision'] != worker.config['revision'] or worker.task.done():
                    await self._stop(worker); self.workers.pop(key,None)
            for key,config in desired.items():
                if key not in self.workers:
                    worker = TankWorker(key[0],config)
                    self.workers[key]=worker
                    worker.task=asyncio.create_task(self._run(worker))

    async def _supervise(self):
        while True:
            try:
                await self.sync()
                self.heartbeat.write_text(json.dumps({'at':time.time(),'production':self.enabled,
                    'healthy':self.inference_started is None or time.monotonic()-self.inference_started<120}),encoding='utf-8')
                if time.monotonic()-self.last_prune>3600:
                    for database in list(self.accounts.databases.values()):
                        await database.prune_monitoring()
                    self.last_prune=time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception('Monitoring supervisor encountered an error; retrying.')
            await asyncio.sleep(2)

    @contextlib.asynccontextmanager
    async def _context(self, farm, database):
        contexts=[(current_farm,current_farm.set(farm)),(current_database,current_database.set(database)),
                  (current_actor,current_actor.set(None)),(current_session,current_session.set(self.state_factory()))]
        try:
            yield
        finally:
            for variable,token in reversed(contexts):
                variable.reset(token)

    async def _infer(self, frame, session, index, last_log, tank):
        if not self.engine_ready():
            raise ConnectionError('Combined counting is unavailable. Ask Admin to check calibration.')
        # FIFO lock and newest-frame capture prevent one tank from building a GPU backlog.
        async with self.inference_lock:
            self.inference_started=time.monotonic()
            try:
                return await self.processor(frame,session,index,last_log,'production_camera',tank=tank,
                                            run_inference=True,cached_result=None)
            finally:
                self.inference_started=None

    async def _run(self, worker):
        config=worker.config; tank_id=config['tank_id']
        database=await self.accounts.get_database(worker.farm)
        profile=json.loads(config['validation'])
        window=self.window_factory(profile['error_band'])
        baseline=config['accepted_count'] if config['accepted_revision']==config['revision'] else profile['known_count']
        since=config['accepted_at'] or config['updated_at']
        session='production_'+uuid.uuid4().hex
        last_log=0; index=0; retry=1; previous_digest=None; same_since=time.monotonic()
        coverage_at=time.monotonic()
        async with self._context(worker.farm,database):
            while True:
                camera=self.capture_factory(config['live_source'])
                try:
                    worker.set_status('starting' if retry==1 else 'reconnecting','Opening live camera.' if retry==1 else 'Camera unavailable. Reconnecting automatically.')
                    await camera.open()
                    while True:
                        frame,captured=await camera.read()
                        tank=await database.get_tank(tank_id)
                        quality=await asyncio.to_thread(frame_quality,frame)
                        if quality['digest'] != previous_digest:
                            previous_digest=quality['digest']; same_since=captured
                        valid=good_quality(quality,profile['quality']) and captured-same_since<10
                        index+=1
                        telemetry,last_log,_=await self._infer(frame,session,index,last_log,tank)
                        valid = valid and time.monotonic()-captured<10
                        elapsed=time.monotonic()-coverage_at; coverage_at=time.monotonic()
                        await database.monitoring_coverage(tank_id,valid,min(10,elapsed))
                        census=window.add(telemetry['live_count'],captured,valid)
                        if valid:
                            retry=1
                            await database.monitoring_alert('camera:'+tank_id)
                            await database.monitoring_alert('quality:'+tank_id)
                        else:
                            await database.monitoring_alert('quality:'+tank_id,'Counting unreliable. Saved population is held until clear, fresh whole-tank frames return.',tank_id)
                        if census is not None and (abs(census-baseline)>profile['error_band'] or census==0 and baseline>0):
                            old_population=tank['current_count']
                            try:
                                tank=await database.reconcile_census(tank_id,census,uuid.uuid4().hex,config['revision'],baseline,since,
                                    {'profile':profile['profile'],'error_band':profile['error_band']})
                                baseline=census; since=now_text()
                                await database.monitoring_alert('pending:'+tank_id)
                                await self.accounts.audit(None,worker.farm,'automatic_census',json.dumps({'tank_id':tank_id,'population':tank['current_count'],'observed':census}))
                                if old_population and old_population-tank['current_count']>=max(1,old_population*.2):
                                    await database.monitoring_alert('loss:'+tank_id,'Population fell by at least 20%. Review estimated losses and any unrecorded dispersal.',tank_id)
                            except ValueError as exc:
                                valid=False; window.reset()
                                await database.monitoring_alert('pending:'+tank_id,str(exc),tank_id)
                        worker.status='running'; worker.message='Monitoring. Automatic census active.' if valid else 'Counting unreliable. Inventory updates are held.'
                        telemetry.update(production=True,source_kind='camera',saved_population=tank['current_count'],
                            count_reliable=valid,census_population=baseline,error_band=profile['error_band'],
                            message=worker.message)
                        worker.telemetry=telemetry; worker.publish(telemetry)
                        await asyncio.sleep(.1)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    window.reset()
                    worker.set_status('reconnecting','Camera or counting service unavailable. Retrying automatically; saved population is held.')
                    await database.monitoring_alert('camera:'+tank_id,worker.message,tank_id)
                    log.exception('Live monitoring interrupted for tank %s; retrying.',tank_id)
                    await database.monitoring_coverage(tank_id,False,time.monotonic()-coverage_at)
                    coverage_at=time.monotonic()
                finally:
                    await camera.close()
                await database.monitoring_coverage(tank_id,False,retry)
                await asyncio.sleep(retry)
                coverage_at=time.monotonic()
                retry=min(60,retry*2)

    async def validate(self, farm, tank_id, known_count, whole_view):
        async with self.lock:
            await self._begin_validation(farm,tank_id,known_count,whole_view)

    async def _begin_validation(self, farm, tank_id, known_count, whole_view):
        if whole_view is not True:
            raise ValueError('Confirm that the camera shows the entire tank population.')
        database=await self.accounts.get_database(farm)
        configs=await database.monitoring_configs()
        config=next((c for c in configs if c['tank_id']==tank_id),None)
        if config and config['status'] == 'inactive':
            raise ValueError('This tank has been removed.')
        if not config or not is_live(config['live_source']):
            raise ValueError('Assign a live camera before validating.')
        if isinstance(known_count,bool) or not isinstance(known_count,int) or known_count!=config['current_count']:
            raise ValueError('First correct saved population to your independently verified fish count.')
        key=(farm,tank_id)
        if key in self.validations:
            raise ValueError('Census validation is already running.')
        for other_key,worker in self.workers.items():
            if other_key!=key and source_key(worker.config['live_source'])==source_key(config['live_source']):
                raise ValueError('This camera is already monitoring another tank.')
        if source_key(config['live_source']) in self.validation_sources.values():
            raise ValueError('This camera is already being validated.')
        if key in self.workers:
            await self._stop(self.workers.pop(key))
        await database.configure_monitoring(tank_id,invalidate=True,expected_revision=config['revision'])
        config={**config,'revision':config['revision']+1}
        self.validation_sources[key]=source_key(config['live_source'])
        self.validation_status[key]={'status':'validating','message':'Keep the whole-tank view steady for at least one minute.','revision':config['revision']}
        self.validations[key]=asyncio.create_task(self._validate_job(farm,database,config,known_count))

    async def _validate_job(self,farm,database,config,known):
        key=(farm,config['tank_id']); camera=self.capture_factory(config['live_source'])
        samples=[]; deadline=time.monotonic()+180; last_digest=None; frozen_since=time.monotonic()
        try:
            async with self._context(farm,database):
                await camera.open(); last_log=0; index=0
                while time.monotonic()<deadline:
                    frame,captured=await camera.read()
                    quality=await asyncio.to_thread(frame_quality,frame)
                    if quality['digest']!=last_digest:
                        last_digest=quality['digest']; frozen_since=captured
                    if not good_quality(quality) or captured-frozen_since>=10:
                        samples=[]
                        await asyncio.sleep(.2)
                        continue
                    index+=1
                    telemetry,last_log,_=await self._infer(frame,'validation_'+str(key),index,last_log,await database.get_tank(config['tank_id']))
                    if time.monotonic()-captured>=10:
                        samples=[]; continue
                    samples.append({'at':captured,'count':telemetry['live_count'],'quality':quality})
                    if len(samples)>=12 and captured-samples[0]['at']>=60:
                        profile=validation_profile(samples,known,fingerprint())
                        profile['validated_at']=now_text()
                        if (await database.get_tank(config['tank_id']))['current_count']!=known:
                            raise ValueError('Population changed during validation. Verify it and try again.')
                        await database.configure_monitoring(config['tank_id'],validation=profile,expected_revision=config['revision'])
                        self.validation_status[key]={'status':'validated','message':'Whole-tank census validated. Automatic updates use the measured error band.','revision':config['revision']}
                        return
                    await asyncio.sleep(.2)
                raise ValueError('Could not collect a reliable full minute. Improve camera visibility and try again.')
        except asyncio.CancelledError:
            self.validation_status[key]={'status':'setup_required','message':'Validation cancelled.','revision':config['revision']}
            raise
        except Exception as exc:
            self.validation_status[key]={'status':'setup_required','message':str(exc) if isinstance(exc,ValueError) else 'Camera validation failed. Check the camera and try again.','revision':config['revision']}
        finally:
            await camera.close()
            self.validations.pop(key,None)
            self.validation_sources.pop(key,None)

    async def status(self, farm):
        database=await self.accounts.get_database(farm)
        configs=await database.monitoring_configs()
        tanks=[]
        for config in configs:
            key=(farm,config['tank_id']); worker=self.workers.get(key)
            profile=json.loads(config['validation']) if config['validation'] else None
            valid=bool(profile and profile.get('profile')==fingerprint())
            item={'tank_id':config['tank_id'],'enabled':bool(config['enabled']),'live_source':config['live_source'],
                  'video_source':config['video_source'],'demonstration_source':config['camera_source'],
                  'validated':valid,'error_band':profile['error_band'] if valid else None,
                  'status':worker.status if worker else 'paused' if not config['enabled'] else 'setup_required' if not valid else 'ready',
                  'message':worker.message if worker else 'Production Mode is off.' if not self.enabled else 'Validate a live whole-tank view.' if not valid else 'Monitoring paused.',
                  'current_count':config['current_count']}
            progress=self.validation_status.get(key,{})
            if progress.get('revision')==config['revision'] and progress.get('status')!='validated' and (key in self.validations or not valid):
                item.update({name:value for name,value in progress.items() if name!='revision'})
            if worker and worker.telemetry:
                item.update(last_frame_at=worker.telemetry['last_frame_at'],count_reliable=worker.telemetry['count_reliable'],census_population=worker.telemetry['census_population'])
            tanks.append(item)
        return {'production':self.enabled,'tanks':tanks,**(await database.monitoring_report())}

    async def subscribe(self,farm,tank_id):
        worker=self.workers.get((farm,tank_id))
        queue=asyncio.Queue(maxsize=2)
        if worker:
            worker.subscribers.add(queue)
            queue.put_nowait(worker.telemetry or {'type':'stream_state','status':worker.status,'message':worker.message,'tank_id':tank_id,'production':True})
        return worker,queue
