"""Isolated browser fixture server; does not change real accounts or farm data."""
import asyncio
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from access import accounts
from farm_database import now_text

fixture_root = Path(tempfile.mkdtemp(prefix="tilapia-browser-"))
accounts.path = fixture_root / "accounts.db"
accounts.farm_directory = fixture_root / "farms"
accounts.legacy_path = fixture_root / "legacy.db"
original_lifespan = app.app.router.lifespan_context


@asynccontextmanager
async def preview_lifespan(application):
    async with original_lifespan(application):
        await accounts.create_admin("preview-admin", "Preview admin password!")
        admin = (await accounts.rows("SELECT * FROM users WHERE role='admin'"))[0]
        farmer = await accounts.create_farmer({"username": "preview-farmer", "display_name": "Preview farmer", "farm_name": "Browser test farm", "password": "Preview farmer password!"}, admin)
        await accounts.conn.execute("UPDATE users SET must_change_password=0")
        await accounts.conn.commit()
        database = await accounts.get_database(farmer["farm_id"])
        await database.create_tank({"tank_id": "TANK-01", "name": "Browser test nursery", "current_count": 1000, "max_capacity":1500, "avg_weight_g":2.5, "feed_rate_pct":.05})
        await database.create_tank({"tank_id": "TANK-02", "name": "Browser test rearing tank", "current_count": 500, "max_capacity":1000, "avg_weight_g":5, "feed_rate_pct":.04})
        import cv2
        directory = accounts.media_directory(farmer["farm_id"])
        for index in (1, 2):
            frame = cv2.imread(str(app.STATIC_DIR / 'media' / f'dataset_sample_{index}.jpg'))
            frame = cv2.resize(frame, (640, 480))
            video = directory / f'preview-tank-{index}.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
            for _ in range(8):
                writer.write(frame)
            writer.release()
            await database.update_tank(f'TANK-0{index}', {'camera_source': str(video)})
        await database.record_mortality("TANK-01", {"count":10})
        await database.record_mortality("TANK-02", {"count":0})
        feed = await database.create_feed_item({"name":"Test starter pellets", "low_stock_kg":5})
        await database.feed_transaction({"kind":"purchase", "item_id":feed["id"], "quantity_kg":20, "cost_php":600})
        await database.feed_transaction({"kind":"feeding", "item_id":feed["id"], "tank_id":"TANK-01", "quantity_kg":.125})
        if '--display-fixture' in sys.argv:
            # Opt-in virtual live camera; no physical device is opened in browser QA.
            import time
            import numpy as np
            from census import frame_quality
            from detection_profile import fingerprint
            frame=cv2.imread(str(app.STATIC_DIR/'media'/'dataset_sample_1.jpg'))
            frame=cv2.resize(frame,(640,480))
            class PreviewCamera:
                def __init__(self,source): self.index=0
                async def open(self): pass
                async def read(self):
                    await asyncio.sleep(.1)
                    self.index+=1
                    view=frame.copy()
                    view[-1,-1]=np.array([self.index%255,80,80],dtype=np.uint8)
                    return view,time.monotonic()
                async def close(self): pass
            original_capture=application.state.production.capture_factory
            application.state.production.capture_factory=lambda source: PreviewCamera(source) if source=='rtsp://preview-only/live' else original_capture(source)
            await database.create_tank({'tank_id':'VIEW-01','name':'Virtual camera fixture','current_count':20,'max_capacity':100,'camera_source':'rtsp://preview-only/live'})
            await database.configure_monitoring('VIEW-01',validation={'known_count':20,'error_band':0,'profile':fingerprint(),'quality':frame_quality(frame)})
        yield


app.app.router.lifespan_context = preview_lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app.app, host="127.0.0.1", port=8001)
