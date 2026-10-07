"""Automatic census regressions use disposable farms and deterministic camera evidence."""
import asyncio
import json
import tempfile
import time
import unittest
import uuid
from datetime import datetime,timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from access import Accounts
from census import CensusWindow,frame_quality,good_quality,validation_profile
from detection_profile import fingerprint
from farm_database import FarmDatabase,now_text
from production import ProductionMonitor,source_key
import test_access as access_fixture
import app


class CensusTests(unittest.TestCase):
    def test_aliases_share_camera_identity(self):
        self.assertEqual(source_key('0'),source_key('00'))
        self.assertEqual(source_key('rtsp://user:secret@CAMERA/live'),source_key('rtsp://camera:554/live'))
    def test_stability_requires_three_distinct_windows(self):
        window=CensusWindow(window_seconds=2,minimum_samples=3)
        results=[window.add(90,t) for t in range(9)]
        self.assertEqual([r for r in results if r is not None],[90])

    def test_bad_frames_and_gaps_reset_evidence_and_replays_do_not_advance(self):
        window=CensusWindow(window_seconds=2,minimum_samples=3)
        for t in range(3):window.add(90,t)
        for _ in range(50):self.assertIsNone(window.add(90,2))
        self.assertEqual(window.windows,1)
        window.add(0,3,False)
        self.assertEqual(window.windows,0)
        window.add(90,30)
        self.assertEqual(window.windows,0)

    def test_zero_requires_longer_confirmation(self):
        window=CensusWindow(window_seconds=2,minimum_samples=3)
        self.assertTrue(all(window.add(0,t) is None for t in range(29)))
        self.assertEqual(window.add(0,29),0)

    def test_unstable_counts_never_commit(self):
        window=CensusWindow(window_seconds=2,minimum_samples=3)
        self.assertTrue(all(window.add([90,50,110][t%3],t) is None for t in range(100)))

    def test_quality_rejects_dark_blurred_and_moved_views(self):
        rng=np.random.default_rng(2026)
        frame=rng.integers(30,210,(120,160,3),dtype=np.uint8)
        quality=frame_quality(frame)
        self.assertTrue(good_quality(quality))
        self.assertFalse(good_quality(frame_quality(np.zeros_like(frame)),quality))
        self.assertFalse(good_quality(frame_quality(np.full_like(frame,120)),quality))
        moved=dict(quality,scene=[255.0 if i%2 else 0.0 for i in range(48)])
        self.assertFalse(good_quality(moved,quality))

    def test_validation_measures_error_band_and_rejects_bad_known_count(self):
        quality=frame_quality(np.random.default_rng(1).integers(20,220,(120,160,3),dtype=np.uint8))
        samples=[{'at':i*5,'count':100+i%2,'quality':quality} for i in range(13)]
        profile=validation_profile(samples,100,'calibration')
        self.assertEqual(profile['error_band'],1)
        with self.assertRaises(ValueError):validation_profile(samples,200,'calibration')


class ReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db=FarmDatabase(':memory:'); await self.db.connect()
        await self.tank('A',100)

    async def asyncTearDown(self):await self.db.close()

    async def tank(self,code,count):
        await self.db.create_tank({'tank_id':code,'name':code,'current_count':count,'camera_source':'0' if code=='A' else '1'})
        await self.db.configure_monitoring(code,validation={'known_count':count,'error_band':0,'profile':fingerprint()})

    async def observe(self,count,previous=100,code='A',operation=None):
        config=next(c for c in await self.db.monitoring_configs() if c['tank_id']==code)
        return await self.db.reconcile_census(code,count,operation or uuid.uuid4().hex,config['revision'],previous,now_text())

    async def sale(self,count=10,**extra):
        return await self.db.commit_dispersal({'tank_id':'A','dispersal_id':uuid.uuid4().hex,'count':count,'recipient':'Buyer',**extra})

    async def test_loss_updates_population_mortality_and_feed_exactly_once(self):
        result=await self.observe(90,operation='same')
        await self.observe(90,operation='same')
        report=await self.db.get_production_report()
        self.assertEqual(result['current_count'],90)
        self.assertEqual(report['estimated_mortality_count'],10)
        self.assertEqual(report['confirmed_mortality_count'],0)
        self.assertEqual(report['mortality_rate_pct'],10)
        self.assertEqual(report['total_feed_kg'],.011)

    async def test_recovered_count_reverses_estimate_without_new_stocking(self):
        await self.observe(90)
        await self.observe(100,90)
        report=await self.db.get_production_report()
        self.assertEqual(report['total_population'],100)
        self.assertEqual(report['estimated_mortality_count'],0)
        self.assertEqual(report['mortality_denominator'],100)
        self.assertEqual((await self.db.census_history())[1]['remaining'],0)

    async def test_dispersal_after_camera_reclassifies_without_double_deduction(self):
        await self.observe(90)
        await self.sale()
        report=await self.db.get_production_report()
        self.assertEqual(report['total_population'],90)
        self.assertEqual(report['estimated_mortality_count'],0)
        self.assertEqual(report['total_dispersed_count'],10)

    async def test_before_dispersal_does_not_restore_fish_and_partial_changes_match(self):
        await self.sale()
        await self.observe(100)
        self.assertEqual((await self.db.get_tank('A'))['current_count'],90)
        await self.observe(95,100)
        await self.observe(90,95)
        report=await self.db.get_production_report()
        self.assertEqual(report['total_population'],90)
        self.assertEqual(report['estimated_mortality_count'],0)

    async def test_unexplained_loss_alongside_pending_dispersal_is_mortality(self):
        await self.sale()
        await self.observe(85)
        report=await self.db.get_production_report()
        self.assertEqual(report['total_population'],85)
        self.assertEqual(report['estimated_mortality_count'],5)

    async def test_transfer_after_both_observations_does_not_duplicate_arrivals(self):
        await self.tank('B',50)
        await self.observe(90)
        await self.observe(60,50,'B')
        await self.sale(type='transfer',destination_tank_id='B')
        report=await self.db.get_production_report()
        self.assertEqual(report['total_population'],150)
        self.assertEqual(report['estimated_mortality_count'],0)
        self.assertEqual((await self.db.get_tank('B'))['current_count'],60)
        incoming=next(r for r in (await self.db.get_tank_mortality_analytics(days=1))['records'] if r['tank_id']=='B')
        self.assertEqual(incoming['additions'],60) # initial 50 plus transfer 10, not two arrivals

    async def test_transfer_before_observation_is_expected_on_both_tanks(self):
        await self.tank('B',50)
        await self.sale(type='transfer',destination_tank_id='B')
        await self.observe(50,50,'B')
        await self.observe(60,50,'B')
        await self.observe(90)
        self.assertEqual((await self.db.get_production_report())['total_population'],150)
        self.assertEqual((await self.db.get_production_report())['estimated_mortality_count'],0)

    async def test_confirmed_mortality_reclassifies_camera_loss(self):
        await self.observe(90)
        await self.db.record_mortality('A',{'count':10,'operation_id':'confirmed'})
        await self.db.record_mortality('A',{'count':10,'operation_id':'confirmed'})
        report=await self.db.get_production_report()
        self.assertEqual(report['total_population'],90)
        self.assertEqual(report['estimated_mortality_count'],0)
        self.assertEqual(report['confirmed_mortality_count'],10)

    async def test_zero_confirmed_check_does_not_consume_or_reject_multiple_estimates(self):
        await self.observe(95)
        await self.observe(90,95)
        await self.db.record_mortality('A',{'count':0})
        report=await self.db.get_production_report()
        self.assertEqual(report['estimated_mortality_count'],10)
        self.assertEqual(report['confirmed_mortality_count'],0)

    async def test_stocking_after_automatic_arrival_does_not_duplicate_population(self):
        await self.observe(110)
        await self.db.stock_fish('A',{'count':10,'operation_id':'stock'})
        self.assertEqual((await self.db.get_tank('A'))['current_count'],110)

    async def test_net_zero_recorded_movements_cannot_hide_unexplained_losses(self):
        await self.db.stock_fish('A',{'count':10})
        await self.sale()
        await self.observe(95)
        self.assertEqual((await self.db.get_production_report())['estimated_mortality_count'],5)

    async def test_camera_changes_and_manual_corrections_invalidate_census(self):
        await self.db.update_tank('A',{'current_count':110,'adjustment_note':'Verified count'})
        with self.assertRaises(ValueError):await self.observe(105)
        self.assertIsNone((await self.db.monitoring_configs())[0]['validation'])

    async def test_manual_correction_recovers_estimates_without_leaving_double_counted_losses(self):
        await self.observe(90)
        await self.db.update_tank('A',{'current_count':100,'adjustment_note':'Verified camera undercount'})
        report=await self.db.get_production_report()
        self.assertEqual(report['estimated_mortality_count'],0)
        self.assertEqual(report['total_population'],100)
        self.assertIsNone(report['mortality_rate_pct'])

    async def test_concurrent_censuses_compare_accepted_version_before_writing(self):
        results=await asyncio.gather(self.observe(95),self.observe(90),return_exceptions=True)
        self.assertEqual(sum(isinstance(r,ValueError) for r in results),1)
        self.assertIn((await self.db.get_tank('A'))['current_count'],(90,95))

    async def test_accelerated_endurance_retries_and_recoveries_preserve_ledger(self):
        rng=np.random.default_rng(10); previous=100
        for index in range(150):
            count=max(0,previous+int(rng.integers(-4,5)))
            operation='endurance:'+str(index)
            await self.observe(count,previous,operation=operation)
            await self.observe(count,previous,operation=operation)
            self.assertEqual((await self.db.get_tank('A'))['current_count'],count)
            previous=count
        ledger=await self.db._rows('SELECT SUM(delta) AS population FROM inventory_movements')
        self.assertEqual(ledger[0]['population'],previous)
        self.assertEqual(len(await self.db.census_history()),150)

    async def test_late_record_preserves_original_mortality_date(self):
        yesterday=(datetime.strptime(now_text(),'%Y-%m-%d %H:%M:%S')-timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        await self.db._conn.execute('UPDATE inventory_baselines SET date=?',(yesterday[:10],))
        await self.db._conn.execute('UPDATE inventory_movements SET timestamp=?',(yesterday,))
        await self.db._conn.commit()
        with patch('production_records.stamp_now',return_value=yesterday):await self.observe(90,operation='old')
        await self.db.record_mortality('A',{'count':10,'census_event_id':'old'})
        rows=(await self.db.get_tank_mortality_analytics(days=2))['records']
        old=next(r for r in rows if r['date']==yesterday[:10])
        self.assertEqual(old['confirmed_mortality_count'],10)
        self.assertEqual(old['estimated_mortality_count'],0)
        self.assertEqual((await self.db.get_tank('A'))['current_count'],90)

    async def test_dispersal_matches_multiple_losses_chronologically(self):
        await self.observe(95,operation='one')
        await self.observe(90,95,operation='two')
        await self.sale(count=3)
        events=await self.db.census_history()
        self.assertEqual({e['id']:e['remaining'] for e in events},{'one':2,'two':5})
        await self.sale(count=2)
        self.assertEqual((await self.db.get_tank('A'))['current_count'],90)
        self.assertEqual((await self.db.get_production_report())['estimated_mortality_count'],5)
        await self.sale(count=7)
        self.assertEqual((await self.db.get_tank('A'))['current_count'],88)
        self.assertEqual((await self.db.get_production_report())['estimated_mortality_count'],0)

    async def test_missing_coverage_is_not_zero_mortality_and_alerts_deduplicate(self):
        self.assertIsNone((await self.db.get_production_report())['mortality_rate_pct'])
        await self.db.monitoring_coverage('A',False,60)
        self.assertIsNone((await self.db.get_production_report())['mortality_rate_pct'])
        await self.db.monitoring_coverage('A',True,60)
        self.assertEqual((await self.db.get_production_report())['mortality_rate_pct'],0)
        await self.db.monitoring_alert('camera:A','Offline','A')
        await self.db.monitoring_alert('camera:A','Still offline','A')
        self.assertEqual(len((await self.db.monitoring_report())['alerts']),1)
        await self.db.monitoring_alert('camera:A')
        self.assertIsNotNone((await self.db.monitoring_report())['alerts'][0]['resolved_at'])

    async def test_retention_prunes_telemetry_and_preserves_inventory_and_daily_summaries(self):
        old=(datetime.strptime(now_text(),'%Y-%m-%d %H:%M:%S')-timedelta(days=31)).strftime('%Y-%m-%d %H:%M:%S')
        for source in ('production_camera','tank_video','uploaded_image'):
            await self.db._conn.execute('INSERT INTO detection_events(timestamp,source) VALUES(?,?)',(old,source))
        await self.db._conn.commit()
        await self.db.monitoring_coverage('A',True,60)
        await self.db.prune_monitoring()
        self.assertEqual([r['source'] for r in await self.db._rows('SELECT source FROM detection_events')],['uploaded_image'])
        self.assertEqual((await self.db.get_tank('A'))['current_count'],100)
        self.assertEqual(len((await self.db.monitoring_report())['coverage']),1)


class ProductionAPITests(unittest.TestCase):
    setUp=access_fixture.AccessTests.setUp
    tearDown=access_fixture.AccessTests.tearDown
    login=access_fixture.AccessTests.login
    request=access_fixture.AccessTests.request
    new_farmer=access_fixture.AccessTests.new_farmer

    def test_mode_defaults_off_is_admin_only_and_rejects_all_recorded_sources(self):
        self.assertFalse(self.request(self.alice,'GET','/api/production').json()['enabled'])
        self.assertEqual(self.request(self.alice,'PUT','/api/admin/production',json={'enabled':True}).status_code,403)
        self.assertEqual(self.request(self.admin,'PUT','/api/admin/production',json={'enabled':'false'}).status_code,400)
        response=self.request(self.admin,'PUT','/api/admin/production',json={'enabled':True})
        self.assertEqual(response.status_code,200,response.text)
        for path in ('/api/upload/image','/api/upload/video','/api/tanks/upload-video','/api/evaluate-sample'):
            self.assertEqual(self.request(self.alice,'POST',path,files={'file':('test.mp4',b'video')}).status_code,403 if path=='/api/evaluate-sample' else 409)
        response=self.request(self.alice,'GET','/api/monitoring')
        self.assertEqual(response.json()['tanks'][0]['status'],'setup_required')
        other=self.request(self.alice,'GET','/api/monitoring?farm_id='+self.bob['user']['farm_id'])
        self.assertEqual(other.status_code,404)
        self.request(self.admin,'PUT','/api/admin/production',json={'enabled':False})

    def test_api_cannot_forge_validation_or_override_fixed_settings(self):
        self.assertEqual(self.request(self.alice,'PUT','/api/tanks/TANK-01/monitoring',json={'live_source':'sample.mp4'}).status_code,400)
        self.request(self.alice,'PUT','/api/tanks/TANK-01/monitoring',json={'validation':{'known_count':1000},'enabled':True})
        self.assertFalse(self.request(self.alice,'GET','/api/monitoring').json()['tanks'][0]['validated'])
        self.assertEqual(self.request(self.alice,'PUT','/api/tanks/TANK-01/monitoring',json={'conf':.1}).status_code,422)

    def test_shared_websocket_viewers_and_logout_leave_background_camera_running(self):
        captures=[]
        class Camera:
            def __init__(self,source):captures.append(self);self.closed=False
            async def open(self):pass
            async def read(self):
                await asyncio.sleep(.01)
                return np.random.default_rng(int(time.monotonic()*1000)).integers(20,220,(120,160,3),dtype=np.uint8),time.monotonic()
            async def close(self):self.closed=True
        async def processor(frame,session,index,last_log,source,**kwargs):
            return {'type':'telemetry','live_count':1000,'last_frame_at':now_text(),'frame':'fixture','frame_idx':index},time.time(),None
        async def configure():
            manager=app.app.state.production
            manager.capture_factory=Camera;manager.processor=processor
            database=await app.accounts.get_database(self.alice['user']['farm_id'])
            frame=np.random.default_rng(3).integers(20,220,(120,160,3),dtype=np.uint8)
            await database.configure_monitoring('TANK-01',live_source='0')
            await database.configure_monitoring('TANK-01',validation={'known_count':1000,'error_band':0,'profile':fingerprint(),'quality':frame_quality(frame)})
        self.client.portal.call(configure)
        self.request(self.admin,'PUT','/api/admin/production',json={'enabled':True})
        url='/ws/live/TANK-01?csrf='+self.alice['csrf']
        headers={'Cookie':'tilapia_session='+self.alice['cookie']}
        from starlette.websockets import WebSocketDisconnect
        with self.client.websocket_connect(url,headers=headers) as first,self.client.websocket_connect(url,headers=headers) as second:
            for socket in (first,second):
                self.assertEqual(socket.receive_json()['type'],'tank_bound')
                for _ in range(30):
                    if socket.receive_json().get('type')=='telemetry':break
                else:self.fail('No live telemetry')
            self.assertEqual(len(captures),1)
            self.request(self.alice,'POST','/api/auth/logout')
            with self.assertRaises(WebSocketDisconnect) as closed:
                while True:first.receive_json()
            self.assertEqual(closed.exception.code,4401)
        async def check():
            worker=app.app.state.production.workers[(self.alice['user']['farm_id'],'TANK-01')]
            return worker.status,len(worker.subscribers)
        self.assertEqual(self.client.portal.call(check),('running',0))
        self.assertFalse(captures[0].closed)
        self.request(self.admin,'PUT','/api/admin/production',json={'enabled':False})
        self.assertTrue(captures[0].closed)


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); root=Path(self.temp.name)
        self.accounts=Accounts(root/'accounts.db',root/'farms',root/'legacy.db')
        await self.accounts.connect()
        self.database=await self.accounts.get_database('legacy')
        self.captures=[]
        self.count=100
        owner=self
        class Camera:
            def __init__(self,source):self.source=source;self.closed=False;owner.captures.append(self)
            async def open(self):pass
            async def read(self):
                await asyncio.sleep(.01)
                if self.source=='1':raise ConnectionError('Disconnected fixture')
                frame=np.random.default_rng(len(owner.captures)+int(time.monotonic()*100)).integers(20,220,(120,160,3),dtype=np.uint8)
                return frame,time.monotonic()
            async def close(self):self.closed=True
        async def processor(frame,session,index,last_log,source,**kwargs):
            return {'live_count':owner.count,'last_frame_at':now_text(),'frame':'fixture','frame_idx':index},time.time(),None
        self.manager=ProductionMonitor(self.accounts,lambda:object(),processor,lambda:True,capture_factory=Camera,
            window_factory=lambda error:CensusWindow(error,window_seconds=.01,minimum_samples=2))
        await self.manager.start()
        for name,source in [('A','0'),('B','1')]:
            await self.database.create_tank({'tank_id':name,'name':name,'current_count':100,'camera_source':source})
            frame=np.random.default_rng(3).integers(20,220,(120,160,3),dtype=np.uint8)
            quality=frame_quality(frame)
            await self.database.configure_monitoring(name,validation={'known_count':100,'error_band':0,'profile':fingerprint(),'quality':quality})

    async def asyncTearDown(self):
        await self.manager.close(); await self.accounts.close(); self.temp.cleanup()

    async def test_removal_releases_worker_and_validation_without_stopping_another_tank(self):
        await self.database.configure_monitoring('B',enabled=False)
        profile=json.loads((await self.database.monitoring_configs())[0]['validation'])
        await self.database.create_tank({'tank_id':'C','name':'C','current_count':100,'camera_source':'2'})
        await self.database.configure_monitoring('C',validation=profile)
        await self.manager.set_enabled(True,{'id':'test'})
        for _ in range(100):
            if all(self.manager.workers[('legacy',tid)].status=='running' for tid in ('A','C')):break
            await asyncio.sleep(.02)
        other=self.manager.workers[('legacy','C')]
        original=self.manager.workers[('legacy','A')]
        self.assertEqual(other.status,'running')
        await self.database.create_tank({'tank_id':'V','name':'V','current_count':100,'camera_source':'3'})
        await self.manager.validate('legacy','V',100,True)
        await asyncio.sleep(.03)
        validation=self.manager.validations[('legacy','V')]
        await self.database.delete_tank('V',remove_stock=True,expected_count=100)
        await self.manager.remove_tank('legacy','V')
        self.assertTrue(validation.cancelled())
        self.assertNotIn(('legacy','V'),self.manager.validation_sources)
        self.assertNotIn(('legacy','V'),self.manager.validation_status)
        self.assertTrue(all(c.closed for c in self.captures if c.source=='3'))
        await self.database.delete_tank('A',remove_stock=True,expected_count=100)
        await self.manager.remove_tank('legacy','A')
        self.assertNotIn(('legacy','A'),self.manager.workers)
        self.assertTrue(original.task.done())
        self.assertTrue(all(c.closed for c in self.captures if c.source=='0'))
        self.assertIs(self.manager.workers[('legacy','C')],other)
        previous=other.telemetry['frame_idx']
        for _ in range(100):
            if other.telemetry.get('frame_idx',0)>previous:break
            await asyncio.sleep(.02)
        self.assertGreater(other.telemetry['frame_idx'],previous)

    async def test_workers_run_without_viewers_and_disconnect_does_not_stop_other_tanks(self):
        with self.assertLogs('tilapia.production',level='ERROR'):
            await self.manager.set_enabled(True,{'id':'test'})
            for _ in range(60):
                if self.manager.workers[('legacy','A')].status=='running' and self.manager.workers[('legacy','B')].status=='reconnecting':break
                await asyncio.sleep(.05)
        first=self.manager.workers[('legacy','A')]
        self.assertEqual(first.status,'running')
        self.assertEqual(len(first.subscribers),0)
        _,queue=await self.manager.subscribe('legacy','A')
        self.assertTrue((await queue.get())['production'])
        first.subscribers.discard(queue)
        old=first.telemetry['frame_idx']
        for _ in range(100):
            if first.telemetry.get('frame_idx',0)>old:break
            await asyncio.sleep(.05)
        self.assertGreater(first.telemetry['frame_idx'],old)
        self.assertEqual(self.manager.workers[('legacy','B')].status,'reconnecting')
        self.assertEqual((await self.database.get_tank('B'))['current_count'],100)
        await self.database.configure_monitoring('A',enabled=False); await self.manager.sync()
        self.assertNotIn(('legacy','A'),self.manager.workers)
        self.assertTrue(next(c for c in self.captures if c.source=='0').closed)

    async def test_restart_restores_mode_and_pause_settings_and_turnoff_releases_cameras(self):
        await self.database.configure_monitoring('B',enabled=False)
        await self.manager.set_enabled(True,{'id':'test'})
        await asyncio.sleep(.15)
        await self.manager.close()
        await self.manager.start(); await self.manager.sync()
        self.assertTrue(self.manager.enabled)
        self.assertIn(('legacy','A'),self.manager.workers)
        self.assertNotIn(('legacy','B'),self.manager.workers)
        await self.manager.set_enabled(False,{'id':'test'})
        self.assertEqual(self.manager.workers,{})
        self.assertTrue(all(c.closed for c in self.captures))

    async def test_autonomous_worker_updates_population_without_viewers_and_recovers_counts(self):
        await self.database.configure_monitoring('B',enabled=False)
        self.count=90
        await self.manager.set_enabled(True,{'id':'test'})
        for _ in range(60):
            if (await self.database.get_tank('A'))['current_count']==90:break
            await asyncio.sleep(.05)
        self.assertEqual((await self.database.get_tank('A'))['current_count'],90)
        self.assertEqual((await self.database.get_production_report())['estimated_mortality_count'],10)
        self.assertEqual(len(self.manager.workers[('legacy','A')].subscribers),0)
        self.count=100
        for _ in range(60):
            if (await self.database.get_tank('A'))['current_count']==100:break
            await asyncio.sleep(.05)
        self.assertEqual((await self.database.get_tank('A'))['current_count'],100)
        self.assertEqual((await self.database.get_production_report())['estimated_mortality_count'],0)
