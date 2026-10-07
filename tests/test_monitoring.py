"""Concurrent source playback and lifecycle regressions using isolated farms."""
import asyncio
import threading
import unittest
from unittest.mock import patch

import numpy as np

import app
from monitoring import capture_io
import test_access as access_fixture


class NativeCaptureCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_finishes_native_read_before_resource_cleanup(self):
        entered, release = threading.Event(), threading.Event()
        finished = []
        def read():
            entered.set()
            release.wait(2)
            finished.append(True)
        task = asyncio.create_task(capture_io(read))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(finished, [True])


class MonitoringTests(unittest.TestCase):
    # Reuse the existing private-account fixture without rerunning its test methods.
    setUp = access_fixture.AccessTests.setUp
    tearDown = access_fixture.AccessTests.tearDown
    login = access_fixture.AccessTests.login
    request = access_fixture.AccessTests.request
    new_farmer = access_fixture.AccessTests.new_farmer

    def websocket(self, tank):
        return self.client.websocket_connect('/ws/live/'+tank+'?csrf='+self.alice['csrf'],
            headers={'Cookie':'tilapia_session='+self.alice['cookie']})

    def receive(self, websocket, kind, predicate=lambda event: True):
        for _ in range(80):
            event = websocket.receive_json()
            if event.get('type') == kind and predicate(event):
                return event
        self.fail('Expected stream event was not received.')

    def configure(self, tank, name):
        response = self.request(self.alice,'POST','/api/tanks/upload-video',
            files={'file':(name,b'isolated fixture footage','video/mp4')})
        source = response.json()['filepath']
        response = self.request(self.alice,'PUT','/api/tanks/'+tank,json={'camera_source':source})
        self.assertEqual(response.status_code,200,response.text)
        return source

    def test_parallel_videos_loop_reset_and_stop_independently(self):
        self.request(self.alice,'POST','/api/tanks',json={'tank_id':'TANK-02','name':'Second nursery','current_count':500})
        first_source = self.configure('TANK-01','first.mp4')
        second_source = self.configure('TANK-02','second.mp4')
        captures = []
        class Capture:
            kind = 'video'
            def __init__(self, source):
                self.source=source; self.index=0; self.released=False; captures.append(self)
            def open(self): return True
            def frame_interval(self): return .04
            def read(self):
                if self.index==2:return False,None
                value=(0,100)[self.index] if self.source==first_source else 200
                self.index+=1
                return True,np.full((96,96,3),value,dtype=np.uint8)
            def rewind(self):self.index=0
            def release(self):self.released=True
        def infer(frame,*args):
            marker=int(frame[0,0,0])
            xs=[10] if marker==0 else [65] if marker==100 else [15,60]
            boxes=[{'x1':x,'y1':10,'x2':x+10,'y2':20,'conf':.99,'class_id':0,'class_name':'Tilapia fingerling'} for x in xs]
            return [],boxes,None
        with patch.object(app,'TankCapture',Capture), patch.object(app,'_infer_frame',side_effect=infer):
            with self.websocket('TANK-01') as first,self.websocket('TANK-02') as second:
                first.receive_json();second.receive_json()
                first.send_json({'type':'start_stream'});second.send_json({'type':'start_stream'})
                crossed=self.receive(first,'telemetry',lambda event:event['count_in']==1)
                self.assertEqual(crossed['live_count'],1)
                other=self.receive(second,'telemetry')
                self.assertEqual(other['live_count'],2)
                self.assertEqual(other['count_in'],0)
                self.assertEqual(other['tank_id'],'TANK-02')
                looped=self.receive(first,'telemetry',lambda event:event['playback_cycle']>0)
                self.assertEqual(looped['count_in'],0)
                self.assertGreater(looped['frame_idx'],crossed['frame_idx'])
                self.assertEqual(looped['source_kind'],'video')
                self.assertAlmostEqual(looped['occupancy_pct'],.1)
                first.send_json({'type':'stop_stream'})
                self.receive(first,'stream_state',lambda event:event['status']=='stopped')
                self.assertTrue(next(c for c in captures if c.source==first_source).released)
                self.assertFalse(next(c for c in captures if c.source==second_source).released)
                continuing=self.receive(second,'telemetry',lambda event:event['frame_idx']>other['frame_idx'])
                self.assertEqual(continuing['live_count'],2)
                second.send_json({'type':'stop_stream'})
                self.receive(second,'stream_state',lambda event:event['status']=='stopped')
        self.assertTrue(all(c.released for c in captures))
        self.assertEqual(self.request(self.alice,'GET','/api/reports').json()['total_population'],1500)

    def test_unassigned_and_failed_sources_acknowledge_errors_and_release(self):
        with self.websocket('TANK-01') as websocket:
            websocket.receive_json();websocket.send_json({'type':'start_stream'})
            error=self.receive(websocket,'stream_state')
            self.assertEqual(error['status'],'error')
            self.assertIn('Set a camera',error['message'])
        self.configure('TANK-01','broken.mp4')
        from unittest.mock import MagicMock
        capture=MagicMock(kind='video');capture.open.return_value=False
        with patch.object(app,'TankCapture',return_value=capture):
            with self.websocket('TANK-01') as websocket:
                websocket.receive_json();websocket.send_json({'type':'start_stream'})
                error=self.receive(websocket,'stream_state',lambda event:event['status']=='error')
                self.assertIn('Could not open',error['message'])
        capture.release.assert_called_once()

    def test_camera_disconnect_does_not_stop_another_tanks_video(self):
        from unittest.mock import MagicMock
        self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'camera_source':'1'})
        self.request(self.alice,'POST','/api/tanks',json={'tank_id':'TANK-02','name':'Video nursery'})
        self.configure('TANK-02','ongoing.mp4')
        frame=np.zeros((96,96,3),dtype=np.uint8)
        camera=MagicMock();camera.isOpened.return_value=True
        camera.read.side_effect=[(True,frame),(False,None)]
        video=MagicMock();video.isOpened.return_value=True
        video.read.return_value=(True,frame);video.get.return_value=10
        def open_capture(source,*args):return camera if source==1 else video
        with patch('monitoring.cv2.VideoCapture',side_effect=open_capture):
            with self.websocket('TANK-01') as first,self.websocket('TANK-02') as second:
                first.receive_json();second.receive_json()
                first.send_json({'type':'start_stream'});second.send_json({'type':'start_stream'})
                self.assertEqual(self.receive(first,'telemetry')['source_kind'],'camera')
                error=self.receive(first,'stream_state',lambda event:event['status']=='error')
                self.assertIn('camera stopped sending frames',error['message'])
                first.send_json({'type':'stop_stream'})
                self.receive(first,'stream_state',lambda event:event['status']=='stopped')
                camera.release.assert_called_once()
                other=self.receive(second,'telemetry')
                self.assertEqual(other['source_kind'],'video')
                second.send_json({'type':'stop_stream'})
                self.receive(second,'stream_state',lambda event:event['status']=='stopped')
                video.release.assert_called_once()
