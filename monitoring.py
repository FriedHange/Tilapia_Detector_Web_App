"""Capture lifecycle helpers for independent camera and recorded-video streams."""
import asyncio
import math
import threading

import cv2

CAMERA_SNAPSHOT_SECONDS = 1.0
CAMERA_POLL_SECONDS = 1.0


async def capture_io(function, *args):
    """Finish a native capture operation before cancellation releases its resource."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class TankCapture:
    def __init__(self, source):
        self.source = source
        self.kind = "camera" if str(source).isdigit() or str(source).startswith(("rtsp://", "rtsps://")) else "video"
        self.capture = None
        self.lock = threading.Lock()

    def open(self):
        with self.lock:
            if str(self.source).isdigit():
                self.capture = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
                if not self.capture.isOpened():
                    self.capture.release()
                    self.capture = cv2.VideoCapture(int(self.source))
            elif self.kind == "camera":
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
            else:
                self.capture = cv2.VideoCapture(self.source)
            if self.kind == "camera" and self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return self.capture.isOpened()

    def read(self):
        with self.lock:
            return self.capture.read()

    def rewind(self):
        with self.lock:
            return self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def frame_interval(self):
        with self.lock:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            return 1 / fps if math.isfinite(fps) and 0 < fps <= 240 else .04

    def video_info(self):
        with self.lock:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            interval = 1 / fps if math.isfinite(fps) and 0 < fps <= 240 else .04
            frames = int(count) if math.isfinite(count) and count > 0 else 0
            return interval, frames

    def read_at(self, frame_index):
        with self.lock:
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                return self.capture.read()
            return self.capture.read()

    def release(self):
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
