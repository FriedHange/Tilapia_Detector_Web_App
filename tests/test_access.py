import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app
from access import accounts
from detection_profile import DETECTORS
from test_detection_settings import FakeModel


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [patch.object(accounts, 'path', root / 'accounts.db'),
            patch.object(accounts, 'farm_directory', root / 'farms'), patch.object(accounts, 'legacy_path', root / 'legacy.db'),
            patch.dict('os.environ', {'TILAPIA_ADMIN_USERNAME': 'test-admin', 'TILAPIA_ADMIN_PASSWORD': 'Test admin password 2026!'}),
            patch.object(app, '_auto_load_models')]
        for item in self.patches:
            item.start()
        self.old_pool, self.old_active = app.state.model_pool, app.state.active_models
        app.state.model_pool = {name: FakeModel([([0,0,10,10], .99)]) for name in DETECTORS}
        app.state.active_models = list(DETECTORS)
        self.client = TestClient(app.app)
        self.client.__enter__()
        self.admin = self.login('test-admin', 'Test admin password 2026!')
        self.alice = self.new_farmer('alice', 'Alice farm')
        self.bob = self.new_farmer('bob', 'Bob farm')

    def tearDown(self):
        self.client.__exit__(None, None, None)
        app.state.model_pool, app.state.active_models = self.old_pool, self.old_active
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def login(self, username, password):
        response = self.client.post('/api/auth/login', json={'username': username, 'password': password})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        return {'cookie': response.cookies['tilapia_session'], 'csrf': body['csrf'], 'user': body['user']}

    def request(self, identity, method, path, **kwargs):
        headers = {'Cookie': 'tilapia_session=' + identity['cookie'], 'X-CSRF-Token': identity['csrf']}
        headers.update(kwargs.pop('headers', {}))
        return self.client.request(method, path, headers=headers, **kwargs)

    def new_farmer(self, username, farm):
        response = self.request(self.admin, 'POST', '/api/admin/accounts', json={'username':username,
            'display_name':username.title(),'farm_name':farm,'password':'Initial password 2026!'})
        self.assertEqual(response.status_code, 200, response.text)
        farmer = self.login(username, 'Initial password 2026!')
        blocked = self.request(farmer, 'GET', '/api/reports')
        self.assertEqual(blocked.status_code, 403)
        response = self.request(farmer, 'POST', '/api/auth/password', json={'current_password':'Initial password 2026!', 'new_password':'Farmer password 2026!'})
        self.assertEqual(response.status_code, 200, response.text)
        farmer = self.login(username, 'Farmer password 2026!')
        response = self.request(farmer, 'POST', '/api/tanks', json={'tank_id':'TANK-01','name':farm+' nursery','current_count':1000})
        self.assertEqual(response.status_code, 200, response.text)
        return farmer

    def test_same_tank_codes_and_reports_are_private(self):
        a = self.request(self.alice, 'GET', '/api/reports').json()
        b = self.request(self.bob, 'GET', '/api/reports').json()
        self.assertIn('Alice farm', a['tanks'][0]['name'])
        self.assertIn('Bob farm', b['tanks'][0]['name'])
        response = self.request(self.alice, 'GET', '/api/reports?farm_id='+self.bob['user']['farm_id'])
        self.assertEqual(response.status_code, 404)
        response = self.request(self.alice, 'POST', '/api/tanks/TANK-01/mortality', json={'count':10})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.request(self.bob, 'GET', '/api/reports').json()['total_population'], 1000)

    def test_structured_sources_are_farm_private_live_only_in_production_and_not_audited(self):
        upload=self.request(self.alice,'POST','/api/tanks/upload-video',files={'file':('own.mp4',b'fixture','video/mp4')})
        path=upload.json()['filepath']
        denied=self.request(self.bob,'PUT','/api/tanks/TANK-01',json={'source':{'type':'video','value':path}})
        self.assertEqual(denied.status_code,400)
        bad=self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'source':{'type':'usb','value':path}})
        self.assertEqual(bad.status_code,400)
        conflict=self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'camera_source':'0','source':{'type':'usb','value':'1'}})
        self.assertEqual(conflict.status_code,400)
        camera='rtsp://example-user:example-secret@fixture-host/live'
        saved=self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'source':{'type':'rtsp','value':camera}})
        self.assertEqual(saved.status_code,200,saved.text)
        audit=self.request(self.admin,'GET','/api/admin/audit').text
        self.assertNotIn('example-secret',audit)
        video=self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'source':{'type':'video','value':path}})
        self.assertEqual(video.status_code,200,video.text)
        config=self.request(self.alice,'GET','/api/monitoring').json()['tanks'][0]
        self.assertEqual((config['live_source'],config['video_source']),(camera,path))
        self.request(self.admin,'PUT','/api/admin/production',json={'enabled':True})
        denied=self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'source':{'type':'video','value':path}})
        self.assertEqual(denied.status_code,400)
        self.assertEqual(self.request(self.alice,'GET','/api/monitoring').json()['tanks'][0]['live_source'],camera)

    def test_automatic_names_and_codes_are_private_per_farm(self):
        first=self.request(self.alice,'POST','/api/tanks',json={})
        second=self.request(self.bob,'POST','/api/tanks',json={})
        self.assertEqual(first.status_code,200,first.text)
        self.assertEqual(second.status_code,200,second.text)
        self.assertEqual((first.json()['tank_id'],first.json()['name']),('TANK-02','Tank 02'))
        self.assertEqual(second.json()['tank_id'],'TANK-02')

    def test_admin_overview_requires_selection_for_mutations(self):
        fresh_legacy = self.request(self.admin, 'GET', '/api/reports', headers={'X-Farm-ID':'legacy'})
        self.assertEqual(fresh_legacy.json()['tanks'], [])
        summary = self.request(self.admin, 'GET', '/api/admin/dashboard')
        self.assertEqual(summary.status_code, 200, summary.text)
        self.assertEqual(summary.json()['summary']['total_population'], 2000)
        response = self.request(self.admin, 'POST', '/api/tanks/TANK-01/mortality', json={'count':1})
        self.assertEqual(response.status_code, 400)
        response = self.request(self.admin, 'POST', '/api/tanks/TANK-01/mortality', headers={'X-Farm-ID':self.alice['user']['farm_id']}, json={'count':1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.request(self.alice, 'GET', '/api/reports').json()['total_population'], 999)

    def test_user_cannot_access_admin_benchmark_or_change_settings(self):
        for path in ('/api/admin/dashboard','/api/admin/accounts','/api/evaluation-benchmarks/grouped'):
            self.assertEqual(self.request(self.alice, 'GET', path).status_code, 403)
        self.assertEqual(self.request(self.alice, 'POST', '/api/evaluate-sample').status_code, 403)
        for path in ('/api/config','/api/reprocess-current','/api/monitoring/mode'):
            self.assertEqual(self.request(self.alice,'POST',path,json={'conf':.01}).status_code,422)
        self.assertEqual(self.request(self.admin,'POST','/api/models/set-active',json={'active':[]}).status_code,410)

    def test_csrf_and_expired_sessions_are_rejected(self):
        response = self.request(self.alice, 'POST', '/api/tanks/TANK-01/mortality', headers={'X-CSRF-Token':'bad'},json={'count':0})
        self.assertEqual(response.status_code,403)
        response = self.request(self.alice, 'POST', '/api/tanks/TANK-01/mortality', headers={'Origin':'https://another.example'},json={'count':0})
        self.assertEqual(response.status_code,403)
        self.request(self.alice, 'POST', '/api/auth/logout')
        self.assertEqual(self.request(self.alice, 'GET', '/api/reports').status_code,401)

    def test_private_media_and_server_path_validation(self):
        for path in ('/api/tanks/upload-video', '/api/upload/video'):
            rejected = self.request(self.alice, 'POST', path, files={'file':('page.html',b'<html>Not a video</html>','text/html')})
            self.assertEqual(rejected.status_code, 400, rejected.text)
        response = self.request(self.alice, 'POST', '/api/tanks/upload-video', files={'file':('test.mp4',b'private test video','video/mp4')})
        self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        self.assertEqual(self.request(self.alice,'GET',result['url']).content,b'private test video')
        self.assertEqual(self.request(self.bob,'GET',result['url']).status_code,404)
        self.assertEqual(self.request(self.alice,'PUT','/api/tanks/TANK-01',json={'camera_source':str(Path('app.py').resolve())}).status_code,400)
        self.assertEqual(self.request(self.alice,'GET','/uploads/test.mp4').status_code,404)

    def test_exports_match_mortality_and_do_not_mix_farms(self):
        self.request(self.alice,'POST','/api/tanks/TANK-01/mortality',json={'count':10})
        response=self.request(self.alice,'GET','/api/analytics/mortality/export/csv?days=1')
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('Alice farm',response.text)
        self.assertNotIn('Bob farm',response.text)
        self.assertIn('1.0',response.text)
        report=self.request(self.alice,'GET','/api/reports/export/csv')
        self.assertEqual(report.status_code,200,report.text)
        self.assertIn('1.0%',report.text)

    def test_upload_cache_and_output_hide_detector_identities(self):
        import cv2
        import numpy as np
        _, encoded=cv2.imencode('.jpg',np.zeros((32,32,3),dtype=np.uint8))
        with patch.object(app,'_annotate_frame',side_effect=lambda frame,*args,**kwargs:frame):
            response=self.request(self.alice,'POST','/api/upload/image',files={'file':('fish.jpg',encoded.tobytes(),'image/jpeg')})
        self.assertEqual(response.status_code,200,response.text)
        self.assertNotIn('yolo',response.text.lower())
        self.assertTrue(self.request(self.alice,'GET','/api/cached-image/status').json()['has_cached_image'])
        self.assertFalse(self.request(self.bob,'GET','/api/cached-image/status').json()['has_cached_image'])
        self.assertEqual(self.request(self.bob,'GET','/api/records').json(),[])

    def test_photo_and_reprocessing_return_raw_previews_with_separate_detections(self):
        import base64
        import cv2
        import numpy as np
        _,encoded=cv2.imencode('.png',np.full((32,32,3),80,dtype=np.uint8))
        response=self.request(self.alice,'POST','/api/upload/image',files={'file':('fish.png',encoded.tobytes(),'image/png')})
        self.assertEqual(response.status_code,200,response.text)
        reprocessed=self.request(self.alice,'POST','/api/reprocess-current',json={})
        self.assertEqual(reprocessed.status_code,200,reprocessed.text)
        for result in (response.json(),reprocessed.json()):
            raw=cv2.imdecode(np.frombuffer(base64.b64decode(result['raw_frame']),dtype=np.uint8),cv2.IMREAD_COLOR)
            annotated=cv2.imdecode(np.frombuffer(base64.b64decode(result['annotated_frame']),dtype=np.uint8),cv2.IMREAD_COLOR)
            self.assertTrue(np.all(raw==80))
            self.assertFalse(np.array_equal(raw,annotated))
            self.assertEqual((result['frame_width'],result['frame_height']),(32,32))
            self.assertTrue(result['detections'])

    def test_tank_removal_is_private_confirmed_and_detects_population_changes(self):
        path='/api/tanks/TANK-01'
        response=self.request(self.alice,'DELETE',path)
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.request(self.alice,'DELETE',path+'?farm_id='+self.bob['user']['farm_id']+'&remove_stock=true&expected_count=1000').status_code,404)
        self.request(self.alice,'POST',path+'/mortality',json={'count':1})
        self.assertEqual(self.request(self.alice,'DELETE',path+'?remove_stock=true&expected_count=1000').status_code,409)
        self.assertEqual(self.request(self.alice,'DELETE',path+'?remove_stock=true&expected_count=999').status_code,200)
        self.assertEqual(self.request(self.alice,'DELETE',path+'?remove_stock=true&expected_count=999').status_code,200)
        self.assertEqual(self.request(self.bob,'GET',path).json()['current_count'],1000)
        self.assertEqual(self.request(self.alice,'PUT',path+'/monitoring',json={'enabled':True}).status_code,400)
        self.assertEqual(self.request(self.alice,'POST',path+'/validate-census',json={'known_count':0,'whole_view':True}).status_code,400)

    def test_production_csv_exports_the_complete_dispersal_history(self):
        import csv
        import io
        async def seed_history():
            database = await accounts.get_database(self.alice['user']['farm_id'])
            rows = [(f'EXPORT-{index}', 'TANK-01', '', 'Buyer', 'sale', 1, 'per_fish', 1, 1,
                     '2026-01-01', '2026-01-01') for index in range(1001)]
            await database._conn.executemany('INSERT INTO dispersals VALUES(?,?,?,?,?,?,?,?,?,?,?)', rows)
            await database._conn.commit()
        self.client.portal.call(seed_history)
        report = self.request(self.alice, 'GET', '/api/reports')
        self.assertEqual(len(report.json()['dispersals']), 1000)
        exported = self.request(self.alice, 'GET', '/api/reports/export/csv')
        self.assertEqual(exported.status_code, 200, exported.text)
        rows = list(csv.reader(io.StringIO(exported.text)))
        self.assertEqual(sum(bool(row) and row[0].startswith('EXPORT-') for row in rows), 1001)
        self.assertNotIn('Bob farm', exported.text)

    def test_websocket_permissions_and_locked_settings(self):
        headers={'Cookie':'tilapia_session='+self.alice['cookie']}
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/live/TANK-01?csrf=bad',headers=headers):
                pass
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/live/TANK-01?csrf='+self.alice['csrf']+'&farm_id='+self.bob['user']['farm_id'],headers=headers):
                pass
        with self.client.websocket_connect('/ws/live/TANK-01?csrf='+self.alice['csrf'],headers=headers) as websocket:
            initial=websocket.receive_json()
            self.assertIn('Alice farm',initial['tank']['name'])
            websocket.send_json({'type':'set_conf','value':.01})
            self.assertEqual(websocket.receive_json()['type'],'error')

    def test_simultaneous_counting_connections_have_independent_trackers(self):
        import base64
        import cv2
        import numpy as np
        headers={'Cookie':'tilapia_session='+self.alice['cookie']}
        url='/ws/live/TANK-01?csrf='+self.alice['csrf']
        _, encoded=cv2.imencode('.jpg',np.zeros((32,32,3),dtype=np.uint8))
        frame={'type':'frame','data':base64.b64encode(encoded).decode()}
        with self.client.websocket_connect(url,headers=headers) as first, self.client.websocket_connect(url,headers=headers) as second:
            first.receive_json();second.receive_json()
            for model in app.state.model_pool.values():model.detections=[([5,5,15,15],.99)]
            first.send_json(frame)
            self.assertEqual(first.receive_json()['count_in'],0)
            for model in app.state.model_pool.values():model.detections=[([20,5,30,15],.99)]
            first.send_json(frame)
            self.assertEqual(first.receive_json()['count_in'],1)
            second.send_json(frame)
            self.assertEqual(second.receive_json()['count_in'],0)
            second.send_json({'type':'reset_counts'})
            second.receive_json()
            first.send_json(frame)
            self.assertEqual(first.receive_json()['count_in'],1)

    def test_user_pages_and_model_names_absent_from_templates(self):
        self.assertEqual(self.request(self.alice,'GET','/user').status_code,200)
        response=self.request(self.alice,'GET','/admin',follow_redirects=False)
        self.assertEqual(response.status_code,303)
        self.assertEqual(response.headers['location'],'/user')
        for path in ('/','/landing','/login','/user'):
            response=self.request(self.alice,'GET',path)
            self.assertNotIn('yolo',response.text.lower())

    def test_admin_benchmark_uses_fixed_settings_and_anonymous_labels(self):
        import cv2
        import numpy as np
        _, encoded=cv2.imencode('.jpg',np.zeros((32,32,3),dtype=np.uint8))
        response=self.request(self.admin,'POST','/api/evaluate-sample',data={'actual_count':'2'},
            files={'image_file':('benchmark.jpg',encoded.tobytes(),'image/jpeg')})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(len(response.json()['models']),4)
        self.assertEqual(response.json()['models'][-1]['model_name'],'Combined Counting')
        self.assertEqual(response.json()['models'][-1]['predicted_count'],1)
        self.assertNotIn('yolo',response.text.lower())
        results=self.request(self.admin,'GET','/api/evaluation-benchmarks/grouped')
        self.assertEqual(len(results.json()),1)
        export=self.request(self.admin,'GET','/api/evaluation-benchmarks/export/csv')
        self.assertEqual(export.status_code,200,export.text)
        self.assertNotIn('yolo',export.text.lower())
        denied=self.request(self.admin,'POST','/api/evaluate-sample',data={'actual_count':'2','conf':'.01'},
            files={'image_file':('benchmark.jpg',encoded.tobytes(),'image/jpeg')})
        self.assertEqual(denied.status_code,422)

    def test_disable_account_revokes_existing_login(self):
        response=self.request(self.admin,'PUT','/api/admin/accounts/'+self.alice['user']['id'],json={'active':False})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.request(self.alice,'GET','/api/reports').status_code,401)
