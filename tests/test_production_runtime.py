"""Native capture/process cleanup and guarded boot, without accessing physical cameras."""
import asyncio
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import cv2
import numpy as np

from camera_process import ProcessCamera
import production_launcher
from windows_job import ProcessJob


class BootTests(unittest.TestCase):
    def test_boot_guard_exits_without_spawning_or_creating_data_when_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'absent'
            with patch.object(production_launcher,'DATA',root),patch.object(sys,'argv',['runner','--boot']),patch('production_launcher.subprocess.Popen') as spawn:
                production_launcher.main()
                spawn.assert_not_called()
                self.assertFalse(root.exists())

    def test_boot_guard_reads_only_explicit_saved_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with closing(sqlite3.connect(root/'accounts.db')) as connection:
                connection.execute('CREATE TABLE installation_settings(key TEXT,value TEXT)')
                connection.execute("INSERT INTO installation_settings VALUES('production','0')")
                connection.commit()
            with patch.object(production_launcher,'DATA',root):
                self.assertFalse(production_launcher.enabled())
                with closing(sqlite3.connect(root/'accounts.db')) as connection:
                    connection.execute("UPDATE installation_settings SET value='1'")
                    connection.commit()
                self.assertTrue(production_launcher.enabled())

    @unittest.skipUnless(os.name=='nt','Windows process ownership')
    def test_windows_job_closes_owned_server_and_camera_process_tree(self):
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=None
        try:
            with ProcessJob() as job:
                code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
                parent=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True)
                job.attach(parent)
                child_pid=int(parent.stdout.readline().strip())
                handle=kernel.OpenProcess(0x100000,False,child_pid)
                self.assertTrue(handle)
                self.assertIsNone(parent.poll())
            self.assertIsNotNone(parent.wait(timeout=5))
            self.assertEqual(kernel.WaitForSingleObject(handle,5000),0)
            parent.stdout.close()
        finally:
            if handle:kernel.CloseHandle(handle)


class NativeProcessCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_native_capture_processes_read_and_release_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'capture.mp4')
            writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),20,(160,120))
            self.assertTrue(writer.isOpened())
            rng=np.random.default_rng(5)
            for _ in range(100):writer.write(rng.integers(20,220,(120,160,3),dtype=np.uint8))
            writer.release()
            first=ProcessCamera(path);second=ProcessCamera(path)
            try:
                await first.open();await second.open()
                frame1,_=await first.read();frame2,_=await second.read()
                self.assertEqual(frame1.shape,(120,160,3));self.assertEqual(frame2.shape,frame1.shape)
                self.assertNotEqual(first.process.pid,second.process.pid)
                await first.close()
                self.assertIsNone(first.process)
                frame2,_=await second.read()
                self.assertEqual(frame2.shape,(120,160,3))
                self.assertTrue(second.process.is_alive())
            finally:
                await first.close();await second.close()
            self.assertIsNone(second.process)
