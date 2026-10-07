"""Exercise real GPU counting against the isolated browser fixture server."""
import argparse
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

import cv2
import httpx


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    args=parser.parse_args()
    split=args.dataset/'test'
    data=json.loads((split/'_annotations.coco.json').read_text())
    image=data['images'][0]
    image_path=split/image['file_name']
    expected=sum(a['image_id']==image['id'] for a in data['annotations'])
    results={}
    with httpx.Client(base_url=args.url,timeout=120) as client:
        signed=client.post('/api/auth/login',json={'username':'preview-farmer','password':'Preview farmer password!'})
        signed.raise_for_status()
        client.headers['X-CSRF-Token']=signed.json()['csrf']
        before=client.get('/api/reports').json()['total_population']
        times=[]
        for _ in range(3):
            with image_path.open('rb') as source:
                start=time.perf_counter()
                response=client.post('/api/upload/image',data={'tank_id':'TANK-01'},files={'file':(image_path.name,source,'image/jpeg')})
                response.raise_for_status()
                times.append(round((time.perf_counter()-start)*1000,2))
            body=response.json()
            assert 'yolo' not in response.text.lower()
            assert len(body['model_metrics'])==3
            assert body['annotated_frame']
        results['image']={'actual':expected,'counted':body['count'],'request_ms':times}
        # Use the actual labeled frame for a short reproducible video and camera.
        frame=cv2.imread(str(image_path))
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'counting.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(frame.shape[1],frame.shape[0]))
            for _ in range(6):writer.write(frame)
            writer.release()
            with video.open('rb') as source:
                response=client.post('/api/upload/video',files={'file':('counting.mp4',source,'video/mp4')})
                response.raise_for_status()
            events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            assert events[-1]['done'] and events[-1]['total_frames']==6
            assert all(event['live_count']>0 for event in events if 'live_count' in event)
            results['video']={'frames':events[-1]['total_frames'],'last_count':events[-2]['live_count']}
            with video.open('rb') as source:
                uploaded=client.post('/api/tanks/upload-video',files={'file':('camera-fixture.mp4',source,'video/mp4')})
                uploaded.raise_for_status()
            client.put('/api/tanks/TANK-01',json={'camera_source':uploaded.json()['filepath']}).raise_for_status()
        after=client.get('/api/reports').json()['total_population']
        assert before==after, 'Counting changed saved stock automatically'
        results['inventory_unchanged']=True
        from websockets.sync.client import connect
        query=urlencode({'csrf':signed.json()['csrf']})
        websocket_url=args.url.replace('http://','ws://').replace('https://','wss://')+'/ws/live/TANK-01?'+query
        with connect(websocket_url,additional_headers={'Cookie':'tilapia_session='+client.cookies['tilapia_session']},max_size=10*1024*1024) as socket:
            socket.recv(timeout=30)
            socket.send(json.dumps({'type':'start_stream'}))
            samples=[]
            start=time.perf_counter()
            while len(samples)<30:
                event=json.loads(socket.recv(timeout=30))
                if event['type']=='error':raise AssertionError(event['message'])
                if event['type']=='telemetry':
                    assert 'yolo' not in json.dumps(event).lower()
                    assert len(event['model_metrics'])==3
                    samples.append(event)
            results['live']={'updates':len(samples),'elapsed_seconds':round(time.perf_counter()-start,3),
                'last_count':samples[-1]['live_count'],'reported_updates_per_second':samples[-1]['fps']}
            socket.send(json.dumps({'type':'stop_stream'}))
    with httpx.Client(base_url=args.url,timeout=120) as client:
        login=client.post('/api/auth/login',json={'username':'preview-admin','password':'Preview admin password!'})
        login.raise_for_status()
        client.headers['X-CSRF-Token']=login.json()['csrf']
        with image_path.open('rb') as source:
            benchmark=client.post('/api/evaluate-sample',data={'actual_count':str(expected)},files={'image_file':(image_path.name,source,'image/jpeg')})
            benchmark.raise_for_status()
        assert 'yolo' not in benchmark.text.lower()
        assert len(benchmark.json()['models'])==4
        results['benchmark']=[{'engine':r['model_name'],'counted':r['predicted_count'],'count_error':r['mae']} for r in benchmark.json()['models']]
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
