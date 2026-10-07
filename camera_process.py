"""Disposable camera process: a blocked driver never owns the server thread."""
import asyncio
import multiprocessing
import queue
import time


def _capture(source, frames):
    # Import in the child; do not import app or load GPU models here.
    import cv2
    from monitoring import TankCapture
    camera = TankCapture(source)
    parent = multiprocessing.parent_process()
    try:
        if not camera.open():
            frames.put(('error',None,time.monotonic()))
            return
        camera.capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        interval=camera.frame_interval() if camera.kind=='video' else .005
        while parent is None or parent.is_alive():
            ok, frame = camera.read()
            if not ok or frame is None:
                frames.put(('error',None,time.monotonic()))
                return
            height,width=frame.shape[:2]
            if max(height,width)>640:
                scale=640/max(height,width)
                frame=cv2.resize(frame,(max(1,round(width*scale)),max(1,round(height*scale))))
            ok, encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok:
                continue
            packet = ('frame',encoded.tobytes(),time.monotonic())
            try:
                frames.put_nowait(packet)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frames.put_nowait(packet)
                except queue.Full:
                    pass
            time.sleep(interval)
    finally:
        camera.release()


class ProcessCamera:
    def __init__(self, source, timeout=8):
        self.source, self.timeout = source, timeout
        self.process = None
        self.frames = None

    async def open(self):
        context = multiprocessing.get_context('spawn')
        self.frames = context.Queue(maxsize=1)
        self.process = context.Process(target=_capture,args=(self.source,self.frames),daemon=True)
        self.process.start()

    async def read(self):
        import cv2
        import numpy as np
        try:
            packet = await asyncio.to_thread(self.frames.get,True,self.timeout)
        except (queue.Empty,EOFError,OSError):
            raise ConnectionError('Camera stopped sending frames.')
        kind, data, captured = packet
        if kind != 'frame' or time.monotonic()-captured>10:
            raise ConnectionError('Camera frames are unavailable or delayed.')
        frame = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise ConnectionError('Camera frame could not be read.')
        return frame, captured

    async def close(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            await asyncio.to_thread(self.process.join,2)
            if self.process.is_alive():
                self.process.kill()
                await asyncio.to_thread(self.process.join,2)
            self.process.close()
            self.process = None
        if self.frames:
            self.frames.cancel_join_thread()
            self.frames.close()
            self.frames = None
