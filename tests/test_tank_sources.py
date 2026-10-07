"""Unified sources, automatic tank identities and raw viewer previews."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from farm_database import FarmDatabase


class TankSourcesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'farm.db'
        self.db=FarmDatabase(self.path)
        await self.db.connect()
        await self.db.create_tank({'tank_id':'TANK-01','name':'Original','current_count':100,'camera_source':'0'})
        await self.db.configure_monitoring('TANK-01',validation={'known_count':100,'profile':'fixture','error_band':0})

    async def asyncTearDown(self):
        await self.db.close()
        self.temp.cleanup()

    async def config(self):
        return next(c for c in await self.db.monitoring_configs() if c['tank_id']=='TANK-01')

    async def test_video_retains_physical_camera_and_validation(self):
        before=await self.config()
        await self.db.update_tank('TANK-01',{'source':{'type':'video','value':'fixture.mp4'}})
        after=await self.config()
        self.assertEqual((after['live_source'],after['video_source'],after['camera_source']),('0','fixture.mp4','fixture.mp4'))
        self.assertEqual((after['revision'],after['validation']),(before['revision'],before['validation']))
        await self.db.update_tank('TANK-01',{'source':{'type':'usb','value':'0'}})
        self.assertEqual((await self.config())['video_source'],'fixture.mp4')

    async def test_production_camera_edit_preserves_video_and_requires_validation(self):
        await self.db.update_tank('TANK-01',{'source':{'type':'video','value':'fixture.mp4'}})
        await self.db.update_tank('TANK-01',{'source':{'type':'rtsp','value':'rtsp://camera/live'}},production=True)
        config=await self.config()
        self.assertEqual(config['camera_source'],'fixture.mp4')
        self.assertEqual(config['live_source'],'rtsp://camera/live')
        self.assertIsNone(config['validation'])
        with self.assertRaises(ValueError):
            await self.db.update_tank('TANK-01',{'source':{'type':'video','value':'other.mp4'}},production=True)
        self.assertEqual((await self.config())['video_source'],'fixture.mp4')

    async def test_invalid_population_correction_rolls_back_camera_and_fields(self):
        with self.assertRaises(ValueError):
            await self.db.update_tank('TANK-01',{'name':'Changed','current_count':99,'source':{'type':'usb','value':'1'}})
        config=await self.config()
        self.assertEqual((config['live_source'],config['name'],config['current_count']),('0','Original',100))
        self.assertIsNotNone(config['validation'])

    async def test_clear_source_retains_video_and_invalidates_camera(self):
        await self.db.update_tank('TANK-01',{'source':{'type':'video','value':'fixture.mp4'}})
        await self.db.update_tank('TANK-01',{'source':{'type':'none','value':''}})
        config=await self.config()
        self.assertEqual((config['camera_source'],config['live_source'],config['video_source']),('','','fixture.mp4'))
        self.assertIsNone(config['validation'])

    async def test_legacy_live_source_route_keeps_one_camera_assignment(self):
        await self.db.configure_monitoring('TANK-01',live_source='2')
        self.assertEqual((await self.db.get_tank('TANK-01'))['camera_source'],'2')
        await self.db.update_tank('TANK-01',{'camera_source':'fixture.mp4'})
        await self.db.configure_monitoring('TANK-01',live_source='3')
        config=await self.config()
        self.assertEqual((config['camera_source'],config['live_source']),('fixture.mp4','3'))

    async def test_identity_allocation_is_atomic_and_includes_archived_codes(self):
        await self.db.create_tank({'tank_id':'TANK-02','name':'Archived'})
        await self.db.update_tank('TANK-02',{'status':'inactive'})
        other=FarmDatabase(self.path)
        await other.connect()
        try:
            tanks=await asyncio.gather(self.db.create_tank({}),other.create_tank({}))
            self.assertEqual({t['tank_id'] for t in tanks},{'TANK-03','TANK-04'})
            self.assertEqual({t['name'] for t in tanks},{'Tank 03','Tank 04'})
        finally:
            await other.close()
        renamed=await self.db.create_tank({'name':'Nursery east'})
        self.assertEqual((renamed['tank_id'],renamed['name']),('TANK-05','Nursery east'))

    async def test_old_source_migration_prefers_production_camera_and_keeps_video(self):
        await self.db.create_tank({'tank_id':'VIDEO','name':'Video','camera_source':'fixture.mp4'})
        await self.db._conn.execute("UPDATE monitoring_config SET live_source='1' WHERE tank_id='TANK-01'")
        await self.db._conn.execute('ALTER TABLE monitoring_config DROP COLUMN video_source')
        await self.db._conn.commit()
        await self.db.close()
        await self.db.connect()
        self.assertEqual((await self.db.get_tank('TANK-01'))['camera_source'],'1')
        self.assertIsNotNone((await self.config())['validation'])
        video=next(c for c in await self.db.monitoring_configs() if c['tank_id']=='VIDEO')
        self.assertEqual((video['camera_source'],video['video_source']),('fixture.mp4','fixture.mp4'))


class RawPreviewTests(unittest.TestCase):
    def test_raw_preview_contains_no_annotation_and_inference_runs_once(self):
        import app
        from tracker import FingerlngTracker
        frame=np.full((80,120,3),80,dtype=np.uint8)
        detections=[{'x1':20,'y1':20,'x2':65,'y2':60,'conf':.9,'class_name':'Fish'}]
        with patch.object(app,'_infer_frame',return_value=([],detections,None)) as infer:
            annotated,_,boxes,cached=app._process_frame_worker(frame,True,None,FingerlngTracker(),None,None,[])
        import base64
        raw=cv2.imdecode(np.frombuffer(base64.b64decode(cached['raw_frame']),dtype=np.uint8),cv2.IMREAD_COLOR)
        drawn=cv2.imdecode(np.frombuffer(base64.b64decode(annotated),dtype=np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(infer.call_count,1)
        self.assertTrue(np.all(raw==80))
        self.assertFalse(np.array_equal(raw,drawn))
        self.assertEqual((cached['frame_width'],cached['frame_height']),(120,80))
        self.assertEqual(len(boxes),1)
