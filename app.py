"""
app.py — FastAPI Server for Tilapia Fingerling Counter Web Application
=======================================================================
Endpoints
---------
  GET  /                        → Serve dashboard HTML
  GET  /api/models              → List available .pt model files
  POST /api/models/load         → Load a model into the inference pool
  POST /api/models/unload       → Unload a model by name
  GET  /api/config              → Current conf/IoU/capacity settings
  POST /api/config              → Update conf/IoU/capacity
  WS   /ws/live                 → Bidirectional: frames in, telemetry out
  POST /api/upload/image        → Detect on a single uploaded image
  POST /api/upload/video        → Detect on an uploaded video (SSE progress)
  GET  /api/analytics/summary   → Summary statistics from DB
  GET  /api/analytics/timeseries→ Time-series for Chart.js
  GET  /api/analytics/hourly    → Hourly pattern for bar chart
  GET  /api/analytics/model-stats → Per-model usage stats
  GET  /api/analytics/export/csv → Streaming CSV download
  GET  /api/analytics/events    → Recent events table (paginated)
  DELETE /api/analytics/events  → Clear all events
  POST /api/evaluate            → Run model benchmark
  GET  /api/evaluate/results    → Retrieve stored benchmark results
  GET  /api/evaluate/latest     → Latest result per model (comparison view)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import statistics
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, Form, HTTPException, BackgroundTasks,
)
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from starlette.requests import Request
import torch
from ultralytics import YOLO

from database import Database, db
from evaluator import ModelEvaluator
from tracker import FingerlngTracker, nms_dedup

# ---------------------------------------------------------------------------
# Device Detection (NVIDIA GPU / CUDA)
# ---------------------------------------------------------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[SYSTEM] Acceleration device: {DEVICE_NAME} (device={DEVICE}, CUDA available={torch.cuda.is_available()})")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Application state (shared across requests)
# ---------------------------------------------------------------------------
class AppState:
    """Centralised mutable state for the FastAPI application."""

    def __init__(self):
        # Loaded YOLO models  {model_name: YOLO}
        self.model_pool: dict[str, YOLO] = {}

        # Which models are active for live inference
        self.active_models: list[str] = []

        # Inference hardware
        self.device: int | str = DEVICE
        self.device_name: str  = DEVICE_NAME

        # Global inference parameters
        self.conf:         float = 0.25
        self.iou:          float = 0.45
        self.tub_capacity: int   = 75

        # Tracker (one per live session)
        self.tracker: FingerlngTracker = FingerlngTracker()

        # Counting line (relative coords, updated by WS messages)
        self.line_rel: tuple = ((0.5, 0.0), (0.5, 1.0))

        # FPS measurement
        self._frame_times: list[float] = []
        self._fps_window:  int = 30

    def record_frame_time(self):
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) > self._fps_window:
            self._frame_times = self._frame_times[-self._fps_window:]

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def density_info(self, count: int) -> tuple[float, str]:
        pct = (count / max(1, self.tub_capacity)) * 100.0
        if pct > 100.0:
            status = "Overstocked"
        elif pct > 80.0:
            status = "High Density"
        else:
            status = "Optimal"
        return round(pct, 1), status


state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — DB connect/close + auto-load models
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: open DB → auto-load models → yield → close DB."""
    await db.connect()
    _auto_load_models()
    yield
    await db.close()


def _auto_load_models():
    """Load all .pt files from MODELS_DIR on startup."""
    if not MODELS_DIR.exists():
        return
    for pt in sorted(MODELS_DIR.glob("*.pt")):
        name = pt.stem
        if name not in state.model_pool:
            try:
                state.model_pool[name] = YOLO(str(pt))
                state.active_models.append(name)
                print(f"[BOOT] Loaded model: {name} (device={state.device_name})")
            except Exception as e:
                print(f"[BOOT] Failed to load {name}: {e}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tilapia Fingerling Counter",
    description="Aquaculture monitoring dashboard — YOLOv8/v9/v10 comparison",
    version="2.0.0",
    lifespan=lifespan,
)

templates = None  # HTML is served directly (no Jinja2 needed)

# ---------------------------------------------------------------------------
# Utility — run model inference (sync, called in thread executor)
# ---------------------------------------------------------------------------
def _infer_frame(
    frame: np.ndarray,
    tracker_enabled: bool = True,
) -> tuple[list[dict], list[dict], list]:
    """
    Run all active models on a frame, apply cross-model NMS.

    Returns
    -------
    raw_per_model  : list of {model_name, count}
    deduped_boxes  : cross-model NMS deduplicated flat list of box dicts
    first_results  : the raw Ultralytics Results object from the first
                     active model (used by the tracker for ByteTrack IDs)
    """
    all_boxes: list[dict] = []
    raw_per_model: list[dict] = []
    first_results_obj = None

    for model_name in state.active_models:
        model = state.model_pool.get(model_name)
        if model is None:
            continue
        try:
            if tracker_enabled:
                results = model.track(
                    frame,
                    verbose=False,
                    conf=state.conf,
                    iou=state.iou,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=state.device,
                )
            else:
                results = model(frame, verbose=False, conf=state.conf, iou=state.iou, device=state.device)

            r = results[0]
            # Keep first model's results for tracker update
            if first_results_obj is None:
                first_results_obj = r

            names = r.names if isinstance(r.names, dict) else {}
            model_boxes: list[dict] = []

            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    cls_id = int(r.boxes.cls[i])
                    conf_val = float(r.boxes.conf[i])
                    tid = None
                    if r.boxes.id is not None:
                        tid = int(r.boxes.id[i])

                    model_boxes.append(
                        {
                            "model": model_name,
                            "track_id": tid,
                            "class_id": cls_id,
                            "class_name": names.get(cls_id, str(cls_id)),
                            "conf": conf_val,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        }
                    )

            all_boxes.extend(model_boxes)
            raw_per_model.append({"model": model_name, "count": len(model_boxes)})

        except Exception as exc:
            print(f"[INFERENCE] {model_name} error: {exc}")

    # Cross-model NMS deduplication
    deduped = nms_dedup(all_boxes, iou_thresh=state.iou)
    return raw_per_model, deduped, first_results_obj


def _annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    tracker: FingerlngTracker,
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, confidence labels, centroid trails,
    and the virtual counting line onto the frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # --- Counting line ---
    (lx1, ly1), (lx2, ly2) = tracker.get_line_pixels(w, h)
    cv2.line(annotated, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
    cv2.putText(annotated, f"IN: {tracker.count_in}  OUT: {tracker.count_out}",
                (lx1 + 6, ly1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # --- Detections ---
    _PALETTE = _make_palette(128)

    for d in detections:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        tid  = d.get("track_id") or 0
        conf = d["conf"]
        cls_name = d.get("class_name", "obj")

        color_bgr = _PALETTE[tid % len(_PALETTE)]
        label = f"#{tid} {cls_name} {conf:.2f}"

        # Trail
        trail = d.get("trail", [])
        for k in range(1, len(trail)):
            cv2.line(
                annotated,
                (int(trail[k - 1][0]), int(trail[k - 1][1])),
                (int(trail[k][0]),     int(trail[k][1])),
                color_bgr, 2,
            )

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated,
                      (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1),
                      color_bgr, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


_PALETTE_CACHE: list[tuple] = []

def _make_palette(n: int) -> list[tuple]:
    global _PALETTE_CACHE
    if len(_PALETTE_CACHE) >= n:
        return _PALETTE_CACHE
    palette = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p, q, t = 0.0, 1 - f, f
        v = 0.85
        s = 0.75
        r, g, b = {
            0: (v, t * s * v + p, p),
            1: (q * s * v + p, v, p),
            2: (p, v, t * s * v + p),
            3: (p, q * s * v + p, v),
            4: (t * s * v + p, p, v),
            5: (v, p, q * s * v + p),
        }.get(h_i % 6, (v, v, v))
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    _PALETTE_CACHE = palette
    return palette


def _frame_to_b64(frame: np.ndarray, quality: int = 75) -> str:
    """JPEG-encode a BGR frame and return a base64 string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _b64_to_frame(b64: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG string to a BGR numpy array."""
    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# REST — root page
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# REST — models management
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def list_models():
    available = [p.stem for p in sorted(MODELS_DIR.glob("*.pt"))]
    loaded = list(state.model_pool.keys())
    active = state.active_models
    return {
        "available": available,
        "loaded": loaded,
        "active": active,
        "device": str(state.device),
        "device_name": state.device_name,
        "is_cuda": torch.cuda.is_available(),
    }


@app.post("/api/models/load")
async def load_model(body: dict):
    """
    Body: {"name": "yolov8n"}  OR  {"name": "yolov8n", "path": "/abs/path.pt"}
    """
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Model name required")

    if name in state.model_pool:
        if name not in state.active_models:
            state.active_models.append(name)
        return {"status": "already loaded", "name": name}

    custom_path = body.get("path")
    pt_path = Path(custom_path) if custom_path else (MODELS_DIR / f"{name}.pt")
    if not pt_path.exists():
        raise HTTPException(404, f"Model file not found: {pt_path}")

    try:
        model = await asyncio.to_thread(YOLO, str(pt_path))
        state.model_pool[name] = model
        if name not in state.active_models:
            state.active_models.append(name)
        return {"status": "loaded", "name": name}
    except Exception as exc:
        raise HTTPException(500, f"Failed to load model: {exc}")


@app.post("/api/models/unload")
async def unload_model(body: dict):
    name = body.get("name", "").strip()
    state.model_pool.pop(name, None)
    if name in state.active_models:
        state.active_models.remove(name)
    return {"status": "unloaded", "name": name}


@app.post("/api/models/set-active")
async def set_active_models(body: dict):
    """Body: {"active": ["yolov8n", "yolov9c"]}"""
    names = body.get("active", [])
    state.active_models = [n for n in names if n in state.model_pool]
    return {"active": state.active_models}


# ---------------------------------------------------------------------------
# REST — configuration
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    (rx1, ry1), (rx2, ry2) = state.tracker._line_rel
    return {
        "conf": state.conf,
        "iou": state.iou,
        "tub_capacity": state.tub_capacity,
        "line": {"rx1": rx1, "ry1": ry1, "rx2": rx2, "ry2": ry2},
        "device": str(state.device),
        "device_name": state.device_name,
        "is_cuda": torch.cuda.is_available(),
    }


@app.post("/api/config")
async def update_config(body: dict):
    if "conf" in body:
        state.conf = float(body["conf"])
    if "iou" in body:
        state.iou = float(body["iou"])
    if "tub_capacity" in body:
        state.tub_capacity = int(body["tub_capacity"])
    if "line" in body:
        ln = body["line"]
        state.tracker.set_line(
            float(ln.get("rx1", 0.5)), float(ln.get("ry1", 0.0)),
            float(ln.get("rx2", 0.5)), float(ln.get("ry2", 1.0)),
        )
    return {"status": "ok"}


@app.post("/api/config/reset-counts")
async def reset_counts():
    state.tracker.reset_counts()
    return {"status": "counts reset", "count_in": 0, "count_out": 0}


# ---------------------------------------------------------------------------
# Frame processing helper
# ---------------------------------------------------------------------------
async def _process_frame_payload(
    frame: np.ndarray,
    session_id: str,
    frame_idx: int,
    last_db_log: float,
    source_name: str = "live_webcam",
) -> tuple[dict, float]:
    """Process a single BGR frame, log to DB at 1Hz, and produce the telemetry dictionary."""
    state.record_frame_time()
    h, w = frame.shape[:2]

    # Run inference + ByteTrack in thread pool
    _, deduped, raw_results = await asyncio.to_thread(_infer_frame, frame, True)

    # Update tracker for line-crossing logic using the first model's Results
    if raw_results is not None:
        tracked_dets = state.tracker.update(raw_results, w, h, conf_thresh=state.conf)
        trail_map = {d["track_id"]: d["trail"] for d in tracked_dets}
        crossed_in_ids  = {d["track_id"] for d in tracked_dets if d["crossed_in"]}
        crossed_out_ids = {d["track_id"] for d in tracked_dets if d["crossed_out"]}
        for box in deduped:
            tid = box.get("track_id")
            box["trail"]       = trail_map.get(tid, [])
            box["crossed_in"]  = tid in crossed_in_ids
            box["crossed_out"] = tid in crossed_out_ids

    track_ids = [d.get("track_id") for d in deduped if d.get("track_id")]

    # Annotate frame
    annotated = await asyncio.to_thread(
        _annotate_frame, frame, deduped, state.tracker
    )

    # Telemetry
    confs = [d["conf"] for d in deduped]
    avg_conf = statistics.fmean(confs) if confs else 0.0
    live_count = len(deduped)
    density_pct, status_level = state.density_info(live_count)
    fps = state.fps

    # DB log at 1 Hz
    now = time.time()
    new_last_db_log = last_db_log
    if now - last_db_log >= 1.0:
        model_str = ",".join(state.active_models)
        await db.log_event(
            session_id=session_id,
            source=source_name,
            frame_idx=frame_idx,
            fingerling_count=live_count,
            count_in=state.tracker.count_in,
            count_out=state.tracker.count_out,
            avg_conf=avg_conf,
            density_pct=density_pct,
            status_level=status_level,
            model_name=model_str,
            boxes=deduped,
            track_ids=track_ids,
        )
        new_last_db_log = now

    # Encode annotated frame
    frame_b64 = await asyncio.to_thread(_frame_to_b64, annotated, 72)

    # Slim down detection list for JSON
    det_slim = [
        {
            "track_id": d.get("track_id"),
            "class_name": d.get("class_name"),
            "conf": round(d["conf"], 3),
            "x1": round(d["x1"]), "y1": round(d["y1"]),
            "x2": round(d["x2"]), "y2": round(d["y2"]),
            "crossed_in": d.get("crossed_in", False),
            "crossed_out": d.get("crossed_out", False),
        }
        for d in deduped
    ]

    telemetry = {
        "type":        "telemetry",
        "live_count":  live_count,
        "count_in":    state.tracker.count_in,
        "count_out":   state.tracker.count_out,
        "fps":         round(fps, 1),
        "density_pct": density_pct,
        "status":      status_level,
        "avg_conf":    round(avg_conf, 3),
        "frame":       frame_b64,
        "detections":  det_slim,
        "frame_idx":   frame_idx,
    }
    return telemetry, new_last_db_log


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    session_id = f"ws_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    frame_idx = 0
    last_db_log = 0.0

    server_stream_task: Optional[asyncio.Task] = None
    stream_running = asyncio.Event()

    async def _stream_loop(source_val):
        nonlocal frame_idx, last_db_log
        cap = None
        try:
            # If digit or int, treat as camera device index
            if isinstance(source_val, int) or (isinstance(source_val, str) and source_val.isdigit()):
                dev_idx = int(source_val)
                cap = await asyncio.to_thread(cv2.VideoCapture, dev_idx, cv2.CAP_DSHOW)
                if not cap or not cap.isOpened():
                    if cap:
                        cap.release()
                    cap = await asyncio.to_thread(cv2.VideoCapture, dev_idx)
            else:
                cap = await asyncio.to_thread(cv2.VideoCapture, source_val)

            if not cap or not cap.isOpened():
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"Could not open camera/stream source: {source_val}"
                }))
                return

            await ws.send_text(json.dumps({
                "type": "info",
                "message": f"Server camera stream active: {source_val}"
            }))

            while stream_running.is_set():
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret or frame is None:
                    await asyncio.sleep(0.04)
                    continue

                if not state.active_models:
                    await asyncio.sleep(0.1)
                    continue

                frame_idx += 1
                telemetry, last_db_log = await _process_frame_payload(
                    frame, session_id, frame_idx, last_db_log, f"server_cam_{source_val}"
                )
                await ws.send_text(json.dumps(telemetry))
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as err:
            try:
                await ws.send_text(json.dumps({"type": "error", "message": f"Stream error: {err}"}))
            except Exception:
                pass
        finally:
            if cap:
                await asyncio.to_thread(cap.release)

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            # --- Control messages ---
            if msg_type == "set_conf":
                state.conf = float(msg.get("value", state.conf))
                continue
            if msg_type == "set_iou":
                state.iou = float(msg.get("value", state.iou))
                continue
            if msg_type == "set_capacity":
                state.tub_capacity = int(msg.get("value", state.tub_capacity))
                continue
            if msg_type == "set_line":
                state.tracker.set_line(
                    float(msg.get("rx1", 0.5)), float(msg.get("ry1", 0.0)),
                    float(msg.get("rx2", 0.5)), float(msg.get("ry2", 1.0)),
                )
                continue
            if msg_type == "set_models":
                names = msg.get("active", [])
                state.active_models = [n for n in names if n in state.model_pool]
                continue
            if msg_type == "reset_counts":
                state.tracker.reset_counts()
                await ws.send_text(json.dumps(
                    {"type": "telemetry", "count_in": 0, "count_out": 0,
                     "live_count": 0, "fps": 0.0}
                ))
                continue

            # --- Server stream start/stop ---
            if msg_type == "start_stream":
                if server_stream_task and not server_stream_task.done():
                    stream_running.clear()
                    server_stream_task.cancel()
                    try:
                        await server_stream_task
                    except Exception:
                        pass
                stream_running.set()
                src_val = msg.get("source", 0)
                server_stream_task = asyncio.create_task(_stream_loop(src_val))
                continue

            if msg_type == "stop_stream":
                stream_running.clear()
                if server_stream_task and not server_stream_task.done():
                    server_stream_task.cancel()
                    try:
                        await server_stream_task
                    except Exception:
                        pass
                continue

            # --- Client Frame message ---
            if msg_type != "frame":
                continue

            b64 = msg.get("data", "")
            frame = await asyncio.to_thread(_b64_to_frame, b64)
            if frame is None:
                await ws.send_text(json.dumps({"type": "error", "message": "Bad frame"}))
                continue

            if not state.active_models:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "No active models. Load a model first."
                }))
                continue

            frame_idx += 1
            telemetry, last_db_log = await _process_frame_payload(
                frame, session_id, frame_idx, last_db_log, "browser_webcam"
            )
            await ws.send_text(json.dumps(telemetry))

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected — session {session_id}")
    except Exception as exc:
        print(f"[WS] Error: {exc}")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        stream_running.clear()
        if server_stream_task and not server_stream_task.done():
            server_stream_task.cancel()
            try:
                await server_stream_task
            except Exception:
                pass


# ---------------------------------------------------------------------------
# REST — image upload detection
# ---------------------------------------------------------------------------
@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if not state.active_models:
        raise HTTPException(400, "No active models loaded")

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Cannot decode image")

    _, deduped, _ = await asyncio.to_thread(_infer_frame, frame, False)
    annotated  = await asyncio.to_thread(_annotate_frame, frame, deduped, state.tracker)

    confs       = [d["conf"] for d in deduped]
    avg_conf    = statistics.fmean(confs) if confs else 0.0
    live_count  = len(deduped)
    density_pct, status_level = state.density_info(live_count)

    # DB log
    session_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await db.log_event(
        session_id=session_id,
        source="image_upload",
        frame_idx=1,
        fingerling_count=live_count,
        count_in=0, count_out=0,
        avg_conf=avg_conf,
        density_pct=density_pct,
        status_level=status_level,
        model_name=",".join(state.active_models),
        boxes=deduped,
        track_ids=[],
    )

    frame_b64 = await asyncio.to_thread(_frame_to_b64, annotated, 85)
    return JSONResponse({
        "count":       live_count,
        "avg_conf":    round(avg_conf, 3),
        "density_pct": density_pct,
        "status":      status_level,
        "detections":  [
            {k: v for k, v in d.items() if k != "trail"}
            for d in deduped
        ],
        "annotated_frame": frame_b64,
        "filename": file.filename,
    })


# ---------------------------------------------------------------------------
# REST — video upload detection (SSE streaming progress)
# ---------------------------------------------------------------------------
@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...),
                       background_tasks: BackgroundTasks = None):
    if not state.active_models:
        raise HTTPException(400, "No active models loaded")

    # Save to disk
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    contents = await file.read()
    save_path.write_bytes(contents)

    async def _stream():
        cap = cv2.VideoCapture(str(save_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        session_id = f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracker = FingerlngTracker()
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            _, deduped, _ = await asyncio.to_thread(_infer_frame, frame, False)
            confs = [d["conf"] for d in deduped]
            avg_conf = statistics.fmean(confs) if confs else 0.0
            live_count = len(deduped)
            density_pct, status_level = state.density_info(live_count)

            elapsed = time.time() - t_start
            calc_fps = round(frame_idx / elapsed, 1) if elapsed > 0.05 else 0.0

            # Log every 5 frames
            if frame_idx % 5 == 0:
                await db.log_event(
                    session_id=session_id,
                    source="video_upload",
                    frame_idx=frame_idx,
                    fingerling_count=live_count,
                    count_in=0, count_out=0,
                    avg_conf=avg_conf,
                    density_pct=density_pct,
                    status_level=status_level,
                    model_name=",".join(state.active_models),
                    boxes=deduped,
                    track_ids=[],
                )

            progress = round(frame_idx / total * 100, 1)

            # High-throughput preview optimization:
            # Full YOLO inference runs on EVERY frame, but thumbnail JPEG encoding &
            # base64 streaming is throttled to ~10 FPS (every 3 frames) + final/first frames.
            # This prevents SSE event buffer exhaustion and browser DOM freezes on 900+ frames.
            is_thumb_frame = (frame_idx == 1 or frame_idx % 3 == 0 or frame_idx == total)
            thumb_b64 = None
            if is_thumb_frame:
                annotated = await asyncio.to_thread(_annotate_frame, frame, deduped, tracker)
                thumb_b64 = await asyncio.to_thread(_frame_to_b64, annotated, 55)

            payload_data = {
                "frame_idx": frame_idx,
                "total": total,
                "progress": progress,
                "live_count": live_count,
                "density_pct": round(density_pct, 1),
                "fps": calc_fps,
                "avg_conf": round(avg_conf, 3),
                "status": status_level,
            }
            if thumb_b64 is not None:
                payload_data["thumb"] = thumb_b64

            yield f"data: {json.dumps(payload_data)}\n\n"

        cap.release()
        save_path.unlink(missing_ok=True)
        yield f"data: {json.dumps({'done': True, 'total_frames': frame_idx})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# REST — analytics
# ---------------------------------------------------------------------------
@app.get("/api/analytics/summary")
async def analytics_summary():
    return await db.query_summary()


@app.get("/api/analytics/timeseries")
async def analytics_timeseries(hours: Optional[int] = None, limit: int = 300):
    rows = await db.query_timeseries(hours=hours, limit=limit)
    return rows


@app.get("/api/analytics/hourly")
async def analytics_hourly():
    return await db.query_hourly()


@app.get("/api/analytics/model-stats")
async def analytics_model_stats():
    return await db.query_per_model_stats()


@app.get("/api/analytics/events")
async def analytics_events(limit: int = 200):
    return await db.query_recent_events(limit=limit)


@app.delete("/api/analytics/events")
async def analytics_clear_events():
    await db.delete_all_events()
    return {"status": "cleared"}


@app.get("/api/analytics/export/csv")
async def analytics_export_csv():
    """Stream all detection events as a downloadable CSV."""
    events = await db.query_recent_events(limit=100_000)
    import csv, io

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "timestamp", "source", "fingerling_count",
            "count_in", "count_out", "density_pct", "status_level",
            "avg_confidence", "model_name",
        ])
        for e in events:
            writer.writerow([
                e.get("id"), e.get("timestamp"), e.get("source"),
                e.get("fingerling_count"), e.get("count_in", 0),
                e.get("count_out", 0), e.get("density_pct"),
                e.get("status_level"), e.get("avg_confidence"),
                e.get("model_name"),
            ])
        yield buf.getvalue()

    filename = f"tilapia_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# REST — model evaluation benchmark
# ---------------------------------------------------------------------------
@app.post("/api/evaluate")
async def run_evaluation(body: dict):
    """
    Body:
    {
      "dataset_path": "C:/path/to/dataset",
      "model_names":  ["yolov8n", "yolov9c", "yolov10n"],
      "conf":         0.25,
      "iou_thresh":   0.5,
      "gt_csv":       null
    }
    """
    dataset_path = body.get("dataset_path", "")
    model_names  = body.get("model_names", list(state.model_pool.keys()))
    conf         = float(body.get("conf", 0.25))
    iou_thresh   = float(body.get("iou_thresh", 0.5))
    gt_csv       = body.get("gt_csv")

    if not dataset_path and not gt_csv:
        raise HTTPException(400, "dataset_path or gt_csv is required")

    evaluator = ModelEvaluator(state.model_pool)
    try:
        results = await asyncio.to_thread(
            evaluator.run_benchmark,
            dataset_path or "",
            model_names,
            conf,
            iou_thresh,
            gt_csv,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    # Persist each model result
    saved_ids = []
    for r in results:
        if "error" not in r:
            rid = await db.save_evaluation_run(r)
            saved_ids.append(rid)

    return {"results": results, "saved_run_ids": saved_ids}


@app.get("/api/evaluate/results")
async def evaluation_results(limit: int = 50):
    return await db.query_evaluation_runs(limit=limit)


@app.get("/api/evaluate/latest")
async def evaluation_latest():
    """Latest benchmark result per model for side-by-side comparison."""
    return await db.query_latest_eval_per_model()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
