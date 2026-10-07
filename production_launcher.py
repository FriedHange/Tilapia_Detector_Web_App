"""Production runner with guarded Windows boot and process-health restart.

Dependencies are installed separately. Never enables Production Mode itself.
"""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import threading
import os
from windows_job import ProcessJob
from contextlib import closing

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'private_data'


def enabled():
    database=DATA/'accounts.db'
    if not database.is_file():
        return False
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as connection:
        try:
            row=connection.execute("SELECT value FROM installation_settings WHERE key='production'").fetchone()
            return bool(row and row[0]=='1')
        except sqlite3.OperationalError:
            return False


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name=='nt':
        # Terminate this server's tree, including any stalled native camera children.
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    else:
        import signal
        os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.kill();process.wait()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--boot',action='store_true',help='Exit unless Production Mode was enabled.')
    args=parser.parse_args()
    if args.boot and not enabled():
        return
    DATA.mkdir(exist_ok=True)
    log=logging.getLogger('runner'); log.setLevel(logging.INFO)
    handler=RotatingFileHandler(DATA/'production-runner.log',maxBytes=2_000_000,backupCount=5,encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s')); log.addHandler(handler)
    console=logging.getLogger('console'); console.setLevel(logging.INFO)
    console.addHandler(RotatingFileHandler(DATA/'production-console.log',maxBytes=2_000_000,backupCount=5,encoding='utf-8'))
    # app config rotates server output; supervise without reinstalling dependencies or opening a browser.
    while not args.boot or enabled():
        heartbeat=DATA/'monitor-heartbeat.json'; heartbeat.unlink(missing_ok=True)
        with ProcessJob() as job:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','app:app','--host','0.0.0.0','--port','8000',
                '--workers','1','--log-config',str(ROOT/'production_logging.json')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',start_new_session=os.name!='nt')
            try:
                job.attach(process)
            except Exception:
                stop_process(process)
                raise
            def drain(stream):
                for line in stream:
                    console.info(line.rstrip())
                stream.close()
            reader=threading.Thread(target=drain,args=(process.stdout,),daemon=True); reader.start()
            started=time.time(); last_good=started
            try:
                while process.poll() is None:
                    time.sleep(5)
                    if heartbeat.exists():
                        try:
                            status=json.loads(heartbeat.read_text(encoding='utf-8'))
                            if status.get('healthy'):
                                last_good=status['at']
                        except (ValueError,OSError):
                            pass
                    if time.time()-last_good>180:
                        log.error('Server heartbeat stalled; restarting process.')
                        stop_process(process)
            except KeyboardInterrupt:
                stop_process(process)
                return
            reader.join(timeout=2)
            log.warning('Server exited (%s). Retrying in ten seconds.',process.returncode)
        time.sleep(10)


if __name__=='__main__':
    main()
