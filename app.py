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
import hashlib
import io
import json
import re
import statistics
import time
import uuid
import os
import secrets
from functools import wraps
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, Form, HTTPException, BackgroundTasks, Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import torch
from ultralytics import YOLO

from database import Database
from farm_database import FarmDatabase, TankPopulationChanged
from access import (accounts, DatabaseProxy, StateProxy, current_actor, current_database,
                    current_session, current_farm, digest)
from detection_profile import (load_profile, settings_for, fingerprint, public_value,
                               INFERENCE_LOCK, DETECTORS)

db = DatabaseProxy(FarmDatabase(Path(__file__).parent / "tilapia_web_analytics.db", seed=True))
from evaluator import ModelEvaluator, parse_yolo_label_text, compute_academic_metrics
from tracker import FingerlngTracker, nms_dedup
from monitoring import TankCapture, capture_io
from production import ProductionMonitor

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
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

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
        self.conf:         float = 0.50
        self.iou:          float = 0.50
        self.tub_capacity: int   = 100

        # Tracker (one per live session)
        self.tracker: FingerlngTracker = FingerlngTracker()

        # Counting line (relative coords, updated by WS messages)
        self.line_rel: tuple = ((0.5, 0.0), (0.5, 1.0))

        # In-memory image cache for dynamic slider tuning & evaluation
        self.cached_frame: Optional[np.ndarray] = None
        self.cached_filename: Optional[str] = None
        self.cached_session_id: Optional[str] = None
        self.cached_event_id: Optional[int] = None

        # FPS measurement
        self._frame_times: list[float] = []
        self._fps_window:  int = 30

        self.stream_max_dimension: int = 640  # Max dimension for live stream & inference (e.g. 640x360)
        self.inference_stride: int = 2        # 1 = every frame, 2 = every 2nd frame (smooth motion, 2x throughput)
        self.monitoring_mode: str = "ensemble"

        # Live Classification Overlay Display Controls
        self.show_boxes: bool = True
        self.show_labels: bool = True
        self.show_conf: bool = False
        self.show_trails: bool = True

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


state = StateProxy(AppState())
_SESSION_STATES = {}


def serialized_inference(function):
    @wraps(function)
    def run(*args, **kwargs):
        with INFERENCE_LOCK:
            return function(*args, **kwargs)
    return run

# ---------------------------------------------------------------------------
# Lifespan — DB connect/close + auto-load models
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: open DB → auto-load models → yield → close DB."""
    await accounts.connect()
    db.fallback = await accounts.get_database("legacy")
    _auto_load_models()
    app.state.production = ProductionMonitor(accounts, AppState, _process_frame_payload,
        lambda: len(state.active_models)==len(DETECTORS))
    await app.state.production.start()
    try:
        yield
    finally:
        await app.state.production.close()
        _SESSION_STATES.clear()
        await accounts.close()


def _auto_load_models():
    """Load all .pt files from MODELS_DIR on startup."""
    if not MODELS_DIR.exists():
        return
    profile = load_profile()
    for pt in sorted(MODELS_DIR.glob("*.pt")):
        name = pt.stem
        if name not in DETECTORS:
            continue
        expected = profile.get("models", {}).get(name, {}).get("weight_sha256")
        if profile.get("calibrated") and expected:
            checksum = hashlib.sha256()
            with pt.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != expected:
                print(f"[BOOT] Calibration does not match {name}; recalibrate before enabling this engine.")
                state.model_pool.pop(name, None)
                continue
        if name not in state.model_pool:
            try:
                state.model_pool[name] = YOLO(str(pt))
                print(f"[BOOT] Loaded model: {name} (device={state.device_name})")
            except Exception as e:
                print(f"[BOOT] Failed to load {name}: {e}")
    state.active_models = [name for name in DETECTORS if name in state.model_pool]
    state.monitoring_mode = "ensemble"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tilapia Fingerling Counter",
    description="Private farm management and automated fingerling counting",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    # Prevent aggressive caching in Firefox and other browsers for local dashboard & API
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

templates = None  # HTML is served directly (no Jinja2 needed)

# ---------------------------------------------------------------------------
# Utility — run model inference (sync, called in thread executor)
# ---------------------------------------------------------------------------
@serialized_inference
def _infer_frame(
    frame: np.ndarray,
    tracker_enabled: bool = True,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    active_models: Optional[list[str]] = None,
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
    profile = load_profile()
    explicit_conf, explicit_iou = conf, iou
    iou = profile["ensemble_iou"] if iou is None else iou
    active_models = list(state.active_models if active_models is None else active_models)

    for model_name in active_models:
        model = state.model_pool.get(model_name)
        if model is None:
            continue
        t0 = time.perf_counter()
        fixed = settings_for(model_name, profile)
        model_conf = fixed["conf"] if explicit_conf is None else explicit_conf
        model_iou = fixed["iou"] if explicit_iou is None else explicit_iou
        try:
            if tracker_enabled:
                results = model.track(
                    frame,
                    verbose=False,
                    conf=model_conf,
                    iou=model_iou,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=state.device,
                )
            else:
                results = model(frame, verbose=False, conf=model_conf, iou=model_iou, device=state.device, imgsz=640, max_det=1000)

            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0

            r = results[0]
            # Keep first model's results for tracker update
            if first_results_obj is None:
                first_results_obj = r

            names = r.names if isinstance(r.names, dict) else {}
            model_boxes: list[dict] = []
            confs: list[float] = []

            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    cls_id = int(r.boxes.cls[i])
                    class_name = str(names.get(cls_id, "")).lower()
                    if "tilapia" not in class_name and "fingerling" not in class_name:
                        continue
                    conf_val = float(r.boxes.conf[i])
                    tid = None
                    if r.boxes.id is not None:
                        tid = int(r.boxes.id[i])

                    model_boxes.append(
                        {
                            "model": model_name,
                            "track_id": tid,
                            "class_id": 0,
                            "class_name": "Tilapia fingerling",
                            "conf": conf_val,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        }
                    )
                    confs.append(conf_val)

            all_boxes.extend(model_boxes)
            avg_c = (sum(confs) / len(confs)) if confs else 0.0
            raw_per_model.append({
                "model": model_name,
                "model_name": model_name,
                "count": len(model_boxes),
                "avg_conf": round(avg_c, 3),
                "inference_ms": round(lat_ms, 1),
            })

        except Exception as exc:
            print(f"[INFERENCE] {model_name} error: {exc}")

    # Cross-model NMS deduplication
    deduped = nms_dedup(all_boxes, iou_thresh=iou) if len(active_models) > 1 else all_boxes
    return raw_per_model, deduped, first_results_obj


def _annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    tracker: FingerlngTracker,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_conf: bool = False,
    show_trails: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, confidence labels, centroid trails,
    and centroid indicators onto the frame based on overlay display toggles.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # --- Detections ---
    _PALETTE = _make_palette(128)

    for d in detections:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        tid  = d.get("track_id") or 0
        conf = d["conf"]
        cls_name = d.get("class_name", "obj")
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        color_bgr = _PALETTE[tid % len(_PALETTE)]

        # 1. Motion Trail
        if show_trails:
            trail = d.get("trail", [])
            for k in range(1, len(trail)):
                cv2.line(
                    annotated,
                    (int(trail[k - 1][0]), int(trail[k - 1][1])),
                    (int(trail[k][0]),     int(trail[k][1])),
                    color_bgr, 2,
                )

        # 2. Bounding Box
        if show_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        elif not show_labels:
            # If both boxes and labels are turned off, draw a neat centroid dot to prevent visual blindness
            cv2.circle(annotated, (cx, cy), 3, color_bgr, -1)

        # 3. Label & Confidence Tag
        if show_labels:
            if show_conf:
                label = f"#{tid} {cls_name} {int(round(conf * 100))}%"
            else:
                label = f"#{tid} {cls_name}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated,
                          (x1, max(0, y1 - th - 7)), (x1 + tw + 5, y1),
                          color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
        elif show_conf:
            # User only wants confidence percentage without full label
            label = f"{int(round(conf * 100))}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(annotated,
                          (x1, max(0, y1 - th - 5)), (x1 + tw + 4, y1),
                          color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
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
# REST — Landing Page and Dashboard routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
async def landing_page():
    landing_file = BASE_DIR / "templates" / "landing.html"
    if landing_file.exists():
        html = landing_file.read_text(encoding="utf-8")
    else:
        html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/app", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/counter", response_class=HTMLResponse)
async def dashboard_page():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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
        "mode": state.monitoring_mode,
        "stride": state.inference_stride,
        "stream_max_dimension": state.stream_max_dimension,
        "show_boxes": state.show_boxes,
        "show_labels": state.show_labels,
        "show_conf": state.show_conf,
        "show_trails": state.show_trails,
        "device": str(state.device),
        "device_name": state.device_name,
        "is_cuda": torch.cuda.is_available(),
    }


@app.post("/api/monitoring/mode")
async def set_monitoring_mode(body: dict):
    """
    Configure live monitoring performance mode, inference stride, stream resolution, and overlays.
    Body: {
        "mode": "turbo_v8" | "turbo_v10" | "accuracy_v9" | "ensemble",
        "stride": 1 | 2 | 3,
        "max_dimension": 640 | 720 | 1080,
        "show_boxes": bool, "show_labels": bool, "show_conf": bool, "show_trails": bool
    }
    """
    mode = body.get("mode")
    if mode:
        mode_str = str(mode).strip().lower()
        if mode_str == "turbo_v8":
            if "yolov8n" in state.model_pool:
                state.active_models = ["yolov8n"]
            state.monitoring_mode = "turbo_v8"
        elif mode_str == "turbo_v10":
            if "yolov10n" in state.model_pool:
                state.active_models = ["yolov10n"]
            state.monitoring_mode = "turbo_v10"
        elif mode_str == "accuracy_v9":
            if "yolov9c" in state.model_pool:
                state.active_models = ["yolov9c"]
            state.monitoring_mode = "accuracy_v9"
        elif mode_str == "ensemble":
            order = ["yolov8n", "yolov10n", "yolov9c"]
            state.active_models = [m for m in order if m in state.model_pool] or list(state.model_pool.keys())
            state.monitoring_mode = "ensemble"

    if "stride" in body:
        try:
            stride_val = int(body["stride"])
            state.inference_stride = max(1, min(5, stride_val))
        except (ValueError, TypeError):
            pass

    if "max_dimension" in body:
        try:
            dim_val = int(body["max_dimension"])
            state.stream_max_dimension = max(320, min(1920, dim_val))
        except (ValueError, TypeError):
            pass

    if "show_boxes" in body:
        state.show_boxes = bool(body["show_boxes"])
    if "show_labels" in body:
        state.show_labels = bool(body["show_labels"])
    if "show_conf" in body:
        state.show_conf = bool(body["show_conf"])
    if "show_trails" in body:
        state.show_trails = bool(body["show_trails"])

    return {
        "status": "updated",
        "mode": state.monitoring_mode,
        "active": state.active_models,
        "active_models": state.active_models,
        "stride": state.inference_stride,
        "stream_max_dimension": state.stream_max_dimension,
        "show_boxes": state.show_boxes,
        "show_labels": state.show_labels,
        "show_conf": state.show_conf,
        "show_trails": state.show_trails,
    }


@app.post("/api/monitoring/overlay")
async def set_monitoring_overlay(body: dict):
    """
    Toggle live classification overlay visibility: boxes, labels, confidence percentage, and trails.
    """
    if "show_boxes" in body:
        state.show_boxes = bool(body["show_boxes"])
    if "show_labels" in body:
        state.show_labels = bool(body["show_labels"])
    if "show_conf" in body:
        state.show_conf = bool(body["show_conf"])
    if "show_trails" in body:
        state.show_trails = bool(body["show_trails"])

    return {
        "status": "updated",
        "show_boxes": state.show_boxes,
        "show_labels": state.show_labels,
        "show_conf": state.show_conf,
        "show_trails": state.show_trails,
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
    return {"tub_capacity": state.tub_capacity, "line": {"rx1": rx1, "ry1": ry1, "rx2": rx2, "ry2": ry2},
        "settings_fixed": True}


@app.post("/api/config")
async def update_config(body: dict):
    if set(body) & {"conf", "iou", "active_models"}:
        raise HTTPException(422, "Counting settings are fixed by calibration.")
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
def _process_frame_worker(
    frame: np.ndarray,
    run_inference: bool,
    cached_result: Optional[dict],
    tracker: FingerlngTracker,
    conf: float,
    iou: float,
    active_models: list[str],
    max_dim: int = 640,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
    show_trails: bool = True,
) -> tuple[str, list[dict], list[dict], dict]:
    """
    Synchronous worker combining downscaling, inference (or cached detection reuse),
    tracking update, bounding box/line annotation, and JPEG base64 encoding
    in a single execution pass to eliminate thread-pool overhead.
    """
    h, w = frame.shape[:2]
    if max_dim and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    settings = (conf, iou, tuple(active_models), fingerprint())
    if run_inference or cached_result is None or cached_result.get("settings") != settings:
        raw_per_model, deduped, raw_results = _infer_frame(frame, False, conf, iou, active_models)
        deduped = tracker.update_fused(deduped, w, h)
        new_cached = {
            "raw_per_model": raw_per_model,
            "deduped": deduped,
            "settings": settings,
        }
    else:
        raw_per_model = cached_result.get("raw_per_model", [])
        deduped = cached_result.get("deduped", [])
        new_cached = cached_result

    annotated = _annotate_frame(
        frame, deduped, tracker,
        show_boxes=show_boxes,
        show_labels=show_labels,
        show_conf=show_conf,
        show_trails=show_trails,
    )
    frame_b64 = _frame_to_b64(annotated, 72)
    # One raw preview per processed frame serves every viewer's own overlay choices.
    new_cached['raw_frame'] = _frame_to_b64(frame, 72)
    new_cached['frame_width'],new_cached['frame_height'] = w,h
    return frame_b64, raw_per_model, deduped, new_cached


async def _process_frame_payload(
    frame: np.ndarray,
    session_id: str,
    frame_idx: int,
    last_db_log: float,
    source_name: str = "live_webcam",
    tank: Optional[dict] = None,
    run_inference: bool = True,
    cached_result: Optional[dict] = None,
) -> tuple[dict, float, dict]:
    """Process a single BGR frame, log to DB at 1Hz, and produce the contextual telemetry dictionary."""
    state.record_frame_time()

    # Execute downscale/inference/tracking/annotation/encoding in one thread dispatch
    frame_b64, raw_per_model, deduped, new_cached_result = await capture_io(
        _process_frame_worker,
        frame,
        run_inference,
        cached_result,
        state.tracker,
        None,
        None,
        state.active_models,
        state.stream_max_dimension,
        state.show_boxes,
        state.show_labels,
        state.show_conf,
        state.show_trails,
    )

    # Telemetry
    confs = [d["conf"] for d in deduped]
    avg_conf = statistics.fmean(confs) if confs else 0.0
    live_count = len(deduped)
    cap = tank.get("max_capacity", state.tub_capacity) if tank else state.tub_capacity
    density_pct = round((live_count / max(1, cap)) * 100.0, 1)
    status_level = "Overstocked" if density_pct > 100 else ("High Density" if density_pct > 80 else "Optimal")
    fps = state.fps

    # Dynamic metrics calculated in context of this tank
    avg_w = float(tank.get("avg_weight_g", 250.0)) if tank else 250.0
    feed_rate = float(tank.get("feed_rate_pct", 0.05)) if tank else 0.05
    preview_biomass_kg = round((live_count * avg_w) / 1000.0, 2)
    preview_daily_feed_kg = round(preview_biomass_kg * feed_rate, 2)
    preview_valuation_php = round(preview_biomass_kg * 160.0, 2)

    # DB log at 1 Hz in background task (non-blocking)
    now = time.time()
    new_last_db_log = last_db_log
    if now - last_db_log >= 1.0:
        model_str = ",".join(state.active_models)
        track_ids = [d.get("track_id") for d in deduped if d.get("track_id")]
        t_id = tank.get("tank_id") if tank else "TANK-01"
        t_name = tank.get("name") if tank else "Monitoring Channel (Blue Tub - 60L)"
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
                model_metrics=raw_per_model,
                tank_id=t_id,
                tank_name=t_name,
                save_boxes=False,
            )
        new_last_db_log = now

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
            "trail": [[float(x),float(y)] for x,y in d.get('trail',[])],
        }
        for d in deduped
    ]

    telemetry = {
        "type":                  "telemetry",
        "live_count":            live_count,
        "count_in":              state.tracker.count_in,
        "count_out":             state.tracker.count_out,
        "fps":                   round(fps, 1),
        "density_pct":           density_pct,
        "occupancy_pct":         density_pct,
        "last_frame_at":         datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status":                status_level,
        "avg_conf":              round(avg_conf, 3),
        "frame":                 frame_b64,
        "raw_frame":             new_cached_result['raw_frame'],
        "frame_width":           new_cached_result['frame_width'],
        "frame_height":          new_cached_result['frame_height'],
        "detections":            det_slim,
        "model_metrics":         raw_per_model,
        "frame_idx":             frame_idx,
        "tank_id":               tank.get("tank_id") if tank else None,
        "tank_name":             tank.get("name") if tank else None,
        "avg_weight_g":          avg_w,
        "feed_rate_pct":         feed_rate,
        "preview_biomass_kg":    preview_biomass_kg,
        "preview_daily_feed_kg": preview_daily_feed_kg,
        "preview_valuation_php": preview_valuation_php,
    }
    return telemetry, new_last_db_log, new_cached_result


async def _handle_websocket_live(ws: WebSocket, initial_tank_id: Optional[str] = None):
    """Unified WebSocket handler supporting tank-specific streams and routing."""
    if initial_tank_id and not await db.get_tank(initial_tank_id):
        await ws.close(code=4404)
        return
    await ws.accept()
    session_id = f"ws_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    frame_idx = 0
    last_db_log = 0.0
    current_tank = await db.get_tank(initial_tank_id) if initial_tank_id else None

    server_stream_task: Optional[asyncio.Task] = None
    stream_running = asyncio.Event()
    cached_client_result: Optional[dict] = None

    async def stream_state(status, message="", **extra):
        await ws.send_text(json.dumps({"type": "stream_state", "status": status,
            "tank_id": current_tank.get("tank_id") if current_tank else None,
            "message": message, **extra}))

    async def stop_stream(acknowledge=True):
        stream_running.clear()
        if server_stream_task and not server_stream_task.done():
            server_stream_task.cancel()
            try:
                await server_stream_task
            except asyncio.CancelledError:
                pass
        if acknowledge:
            await stream_state("stopped", "Monitoring stopped.")

    async def _stream_loop(source_val):
        nonlocal frame_idx, last_db_log
        capture = TankCapture(source_val)
        cached_result = None
        playback_cycle = 0
        try:
            await stream_state("starting", "Opening the tank source.", source_kind=capture.kind)
            if not await capture_io(capture.open):
                await stream_state("error", "Could not open this tank source. Check its settings or whether the camera is in use.")
                return
            await stream_state("running", "Monitoring active.", source_kind=capture.kind, playback_cycle=0)
            interval = await capture_io(capture.frame_interval) if capture.kind == "video" else .005
            while stream_running.is_set():
                if app.state.production.enabled:
                    await ws.close(code=1012)
                    break
                if not await accounts.authenticate(ws.cookies.get("tilapia_session")):
                    await ws.close(code=4401)
                    break
                started = time.monotonic()
                ret, frame = await capture_io(capture.read)
                if not ret or frame is None:
                    if capture.kind != "video":
                        await stream_state("error", "The camera stopped sending frames. Check its connection and restart.")
                        break
                    await capture_io(capture.rewind)
                    ret, frame = await capture_io(capture.read)
                    if not ret or frame is None:
                        await stream_state("error", "This video contains no readable footage. Choose another video.")
                        break
                    playback_cycle += 1
                    state.tracker.reset_counts()
                    cached_result = None
                    await stream_state("running", "Recorded video restarted; crossings reset.",
                        source_kind="video", playback_cycle=playback_cycle)
                if not state.active_models:
                    await stream_state("error", "Counting is unavailable. Ask Admin to check the counting service.")
                    break
                frame_idx += 1
                do_inference = (frame_idx % state.inference_stride == 0) or (cached_result is None)
                telemetry, last_db_log, cached_result = await _process_frame_payload(
                    frame, session_id, frame_idx, last_db_log, "tank_" + capture.kind,
                    tank=current_tank, run_inference=do_inference, cached_result=cached_result)
                telemetry.update(source_kind=capture.kind, playback_cycle=playback_cycle)
                await ws.send_text(json.dumps(public_value(telemetry)))
                await asyncio.sleep(max(.005, interval - (time.monotonic() - started)))
        except asyncio.CancelledError:
            pass
        except Exception:
            try:
                await stream_state("error", "Monitoring failed. Check this tank's source and restart.")
            except Exception:
                pass
        finally:
            stream_running.clear()
            await capture_io(capture.release)

    try:
        if current_tank:
            await ws.send_text(json.dumps({
                "type": "tank_bound",
                "tank": current_tank
            }))

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")
            if not await accounts.authenticate(ws.cookies.get("tilapia_session")):
                await ws.close(code=4401)
                break

            # --- Tank association ---
            if msg_type == "set_tank":
                tid = msg.get("tank_id")
                if tid:
                    await stop_stream(acknowledge=False)
                    current_tank = await db.get_tank(tid)
                    if current_tank:
                        await ws.send_text(json.dumps({
                            "type": "tank_bound",
                            "tank": current_tank
                        }))
                continue

            # --- Control messages ---
            if not await accounts.authenticate(ws.cookies.get("tilapia_session")):
                await ws.close(code=4401)
                break
            if msg_type in ("set_conf", "set_iou", "set_models"):
                await ws.send_text(json.dumps({"type": "error", "message": "Counting settings are fixed by calibration."}))
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
            if msg_type == "set_overlay":
                if "show_boxes" in msg:
                    state.show_boxes = bool(msg["show_boxes"])
                if "show_labels" in msg:
                    state.show_labels = bool(msg["show_labels"])
                if "show_conf" in msg:
                    state.show_conf = bool(msg["show_conf"])
                if "show_trails" in msg:
                    state.show_trails = bool(msg["show_trails"])
                continue
            if msg_type == "reset_counts":
                state.tracker.reset_counts()
                await ws.send_text(json.dumps(
                    {"type": "telemetry", "count_in": 0, "count_out": 0,
                     "live_count": 0, "fps": 0.0, "tank_id": current_tank.get("tank_id") if current_tank else None}
                ))
                continue

            # --- Server stream start/stop ---
            if msg_type == "start_stream":
                if app.state.production.enabled:
                    await ws.close(code=1012)
                    break
                await stop_stream(acknowledge=False)
                if current_tank:
                    current_tank = await db.get_tank(current_tank["tank_id"])
                if not current_tank or current_tank["status"] == "inactive":
                    await stream_state("error", "Select an active tank first.")
                    continue
                try:
                    src_val = validate_source(current_tank["camera_source"], current_farm.get())
                    if not src_val:
                        raise ValueError("Set a camera or upload a video for this tank first.")
                except ValueError as exc:
                    await stream_state("error", str(exc))
                    continue
                state.tracker.reset_counts()
                state._frame_times.clear()
                stream_running.set()
                server_stream_task = asyncio.create_task(_stream_loop(src_val))
                continue

            if msg_type == "stop_stream":
                await stop_stream()
                continue

            # --- Client Frame message ---
            if msg_type != "frame":
                continue
            if app.state.production.enabled:
                await ws.close(code=1012)
                break

            if not current_tank:
                await ws.send_text(json.dumps({"type": "error", "message": "Select a tank before counting."}))
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
            frame_idx += 1
            do_inference = (frame_idx % state.inference_stride == 0) or (cached_client_result is None)
            src_tag = f"webcam_{current_tank['tank_id']}" if current_tank else "browser_webcam"
            telemetry, last_db_log, cached_client_result = await _process_frame_payload(
                frame, session_id, frame_idx, last_db_log, src_tag,
                tank=current_tank, run_inference=do_inference, cached_result=cached_client_result
            )
            await ws.send_text(json.dumps(public_value(telemetry)))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        await stop_stream(acknowledge=False)


@app.websocket("/ws/live/{tank_id}")
async def websocket_live_tank(ws: WebSocket, tank_id: str):
    await _protected_websocket(ws, tank_id)


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await _protected_websocket(ws, None)


# ---------------------------------------------------------------------------
# REST — image upload detection
# ---------------------------------------------------------------------------
@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...), tank_id: Optional[str] = Form(None)):
    if not state.active_models:
        raise HTTPException(400, "No active models loaded")
    tank = await db.get_tank(tank_id) if tank_id else None
    if tank_id and not tank:
        raise HTTPException(404, "Tank not found.")
    if tank:
        state.tub_capacity = tank["max_capacity"]

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Cannot decode image")

    raw_per_model, deduped, _ = await asyncio.to_thread(_infer_frame, frame, False)
    annotated  = await asyncio.to_thread(_annotate_frame, frame, deduped, state.tracker)

    confs       = [d["conf"] for d in deduped]
    avg_conf    = statistics.fmean(confs) if confs else 0.0
    live_count  = len(deduped)
    density_pct, status_level = state.density_info(live_count)

    # DB log
    session_id = f"img_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}"
    event_id = await db.log_event(
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
        model_metrics=raw_per_model,
        tank_id=tank_id,
        tank_name=tank["name"] if tank else None,
    )

    # In-memory cache for dynamic slider tuning and instant evaluation
    state.cached_frame = frame.copy()
    state.cached_filename = file.filename
    state.cached_session_id = session_id
    state.cached_event_id = event_id

    frame_b64 = await asyncio.to_thread(_frame_to_b64, annotated, 85)
    return JSONResponse({
        "event_id":    event_id,
        "count":       live_count,
        "avg_conf":    round(avg_conf, 3),
        "density_pct": density_pct,
        "status":      status_level,
        "model_metrics": raw_per_model,
        "detections":  [
            {k: v for k, v in d.items() if k != "trail"}
            for d in deduped
        ],
        "annotated_frame": frame_b64,
        "raw_frame": await asyncio.to_thread(_frame_to_b64, frame, 85),
        "frame_width": frame.shape[1], "frame_height": frame.shape[0],
        "filename": file.filename,
    })


@app.post("/api/reprocess-current")
async def reprocess_current(body: Optional[dict] = None):
    """
    Dynamic Slider Tuning: Re-runs inference instantly on the in-memory cached image
    when Confidence or IoU threshold sliders are tuned on the frontend.
    Does NOT require the user to re-upload the image.
    """
    if state.cached_frame is None:
        raise HTTPException(404, "No image currently cached in server session. Please upload an image first.")

    if not state.active_models:
        raise HTTPException(400, "No active models loaded")

    params = body or {}

    if set(params) & {"conf", "iou", "active_models"}:
        raise HTTPException(422, "Counting settings are fixed by calibration.")

    raw_per_model, deduped, _ = await asyncio.to_thread(_infer_frame, state.cached_frame, False)
    annotated = await asyncio.to_thread(_annotate_frame, state.cached_frame, deduped, state.tracker)

    confs = [d["conf"] for d in deduped]
    avg_conf = statistics.fmean(confs) if confs else 0.0
    live_count = len(deduped)
    density_pct, status_level = state.density_info(live_count)

    frame_b64 = await asyncio.to_thread(_frame_to_b64, annotated, 85)
    return JSONResponse({
        "count": live_count,
        "avg_conf": round(avg_conf, 3),
        "density_pct": density_pct,
        "status": status_level,
        "model_metrics": raw_per_model,
        "detections": [
            {k: v for k, v in d.items() if k != "trail"}
            for d in deduped
        ],
        "annotated_frame": frame_b64,
        "raw_frame": await asyncio.to_thread(_frame_to_b64, state.cached_frame, 85),
        "frame_width": state.cached_frame.shape[1], "frame_height": state.cached_frame.shape[0],
        "filename": state.cached_filename,
        "conf": state.conf,
        "iou": state.iou,
        "cached": True,
    })


@app.get("/api/cached-image/status")
async def cached_image_status():
    """Check whether an in-memory image is currently cached."""
    has_image = state.cached_frame is not None
    return {
        "has_cached_image": has_image,
        "filename": state.cached_filename,
        "conf": state.conf,
        "iou": state.iou,
    }


# ---------------------------------------------------------------------------
# REST — video upload detection (SSE streaming progress)
# ---------------------------------------------------------------------------
def _video_suffix(filename):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"):
        raise HTTPException(400, "Choose a supported video file, such as MP4, MOV or AVI.")
    return suffix


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...),
                       background_tasks: BackgroundTasks = None):
    if not state.active_models:
        raise HTTPException(400, "No active models loaded")

    # Save to disk
    save_path = accounts.media_directory(current_farm.get()) / (uuid.uuid4().hex + _video_suffix(file.filename))
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Choose a video that contains footage.")
    save_path.write_bytes(contents)

    async def _stream():
        cap = cv2.VideoCapture(str(save_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        session_id = f"vid_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}"
        tracker = FingerlngTracker()
        t_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                raw_per_model, deduped, _ = await asyncio.to_thread(_infer_frame, frame, False)
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
                        model_metrics=raw_per_model,
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

        finally:
            cap.release()
            save_path.unlink(missing_ok=True)
        yield f"data: {json.dumps({'done': True, 'total_frames': frame_idx})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# REST — Tank video file upload & available video files
# ---------------------------------------------------------------------------
@app.post("/api/tanks/upload-video")
async def upload_tank_video(file: UploadFile = File(...)):
    """Uploads a video file to serve as a tank camera source."""
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    clean_name = re.sub(r'[^a-zA-Z0-9_\.-]', '_', file.filename)
    unique_name = f"{uuid.uuid4().hex}{_video_suffix(clean_name)}"
    save_path = accounts.media_directory(current_farm.get()) / unique_name
    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty video file")
    
    save_path.write_bytes(contents)
    size_mb = round(len(contents) / (1024 * 1024), 2)
    relative_path = str(save_path)
    
    return {
        "status": "success",
        "filename": file.filename,
        "filepath": relative_path,
        "relative_path": relative_path,
        "size_mb": size_mb,
        "size_bytes": len(contents),
        "url": f"/media/{unique_name}?farm_id={current_farm.get()}"
    }


@app.get("/api/tanks/available-videos")
async def get_available_tank_videos():
    """Returns local video files available to be selected as tank camera sources."""
    directory = accounts.media_directory(current_farm.get())
    return [{"name": file.name, "path": str(file), "url": f"/media/{file.name}?farm_id={current_farm.get()}"}
            for file in directory.iterdir() if file.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")]


@app.post("/api/tanks")
async def api_create_tank(body: dict):
    """Create a new tank."""
    try:
        production=bool(getattr(app.state,'production',None) and app.state.production.enabled)
        body=validate_tank_selection(body,current_farm.get(),production)
        saved=await db.create_tank(body,production=production)
        if getattr(app.state,'production',None):
            await app.state.production.sync()
        return saved
    except ValueError as err:
        raise HTTPException(400, str(err))


@app.get("/api/tanks")
async def api_get_tanks():
    return await db.get_tanks()


@app.get("/api/tanks/{tank_id}")
async def api_get_tank(tank_id: str):
    """Get single tank with real-time biomass, feed KG, and PHP valuation."""
    tank = await db.get_tank(tank_id)
    if not tank:
        raise HTTPException(404, f"Tank '{tank_id}' not found")
    return tank


@app.put("/api/tanks/{tank_id}")
async def api_update_tank(tank_id: str, body: dict):
    """Update tank specifications or capacity."""
    try:
        production=bool(getattr(app.state,'production',None) and app.state.production.enabled)
        body=validate_tank_selection(body,current_farm.get(),production)
        updated = await db.update_tank(tank_id, body,production=production)
        if not updated:
            raise HTTPException(404, f"Tank '{tank_id}' not found")
        if getattr(app.state,'production',None):
            await app.state.production.sync()
        return updated
    except ValueError as err:
        raise HTTPException(400, str(err))


@app.delete("/api/tanks/{tank_id}")
async def api_delete_tank(tank_id: str, remove_stock: bool = False, expected_count: Optional[int] = Query(None, ge=0)):
    """Remove from active inventory while retaining history."""
    try:
        deleted = await db.delete_tank(tank_id, remove_stock=remove_stock, expected_count=expected_count)
    except TankPopulationChanged as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not deleted:
        raise HTTPException(404, f"Tank '{tank_id}' not found")
    if getattr(app.state, 'production', None):
        await app.state.production.remove_tank(current_farm.get(), tank_id)
    return {"status": "deleted", "tank_id": tank_id}


@app.post("/api/dispersal/commit")
async def api_commit_dispersal(body: dict):
    """
    Creates dispersal entry, logs production, and updates tank count atomically.
    Hard Rule: Never persist a fish count to tank inventory or production logs
    unless linked to a valid dispersal_id. Live vision counts remain transient preview telemetry only.
    """
    dispersal_id = (body.get("dispersal_id") or "").strip()
    if not dispersal_id:
        raise HTTPException(
            status_code=400,
            detail="Dispersal ID is strictly required to persist count to tank inventory. Live counts are preview telemetry only."
        )

    try:
        result = await db.commit_dispersal(body)
        return JSONResponse(result)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Dispersal commit transaction failed: {str(err)}")


@app.get("/api/reports")
async def api_reports():
    """Aggregated farm metrics, feed schedule, and dispersal ledger."""
    return await db.get_production_report()


@app.get("/api/reports/production")
async def api_production_report():
    """Returns population per tank, mortality %, total feed (KG), inventory PHP value, and dispersal earnings."""
    return await db.get_production_report()


@app.get("/api/reports/export/csv")
async def export_production_report_csv():
    """Download complete farm production report & dispersal ledger as CSV."""
    import csv, io
    rep = await db.get_production_report(include_all_dispersals=True)
    tanks = rep.get("population_per_tank", [])
    dispersals = rep.get("dispersals", [])

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Section 1: Executive KPI Summary
    writer.writerow(["=== AQUACULTURE FARM PRODUCTION SUMMARY ==="])
    writer.writerow(["Generated At", datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Total Live Population", rep.get("total_population", 0)])
    writer.writerow(["Total Biomass (KG)", rep.get("total_biomass_kg", 0.0)])
    writer.writerow(["Total Daily Feed (KG/day)", rep.get("total_feed_kg", 0.0)])
    writer.writerow(["Inventory Valuation (PHP)", rep.get("inventory_php_value", 0.0)])
    writer.writerow(["Total Dispersal Sales (PHP)", rep.get("dispersal_earnings", 0.0)])
    writer.writerow(["Today's Loss Rate Including Estimates (%)", (f"{rep['mortality_rate_pct']}%" if rep.get("mortality_rate_pct") is not None else "N/A")])
    writer.writerow(['Estimated Mortality Today',rep.get('estimated_mortality_count',0)])
    writer.writerow(['Confirmed Deaths Today',rep.get('confirmed_mortality_count',0)])
    writer.writerow(["Reporting Date", rep.get("report_date")])
    writer.writerow(["Mortality Checks Recorded", rep.get("mortality_checked_tanks", 0)])
    writer.writerow(["Tanks Requiring Checks", rep.get("mortality_expected_tanks", 0)])
    writer.writerow(["Daily Mortality Formula", "Estimated losses plus confirmed deaths / (opening fish + incoming fish) * 100; observation gaps are not zero deaths"])
    writer.writerow([])

    # Section 2: Tanks Inventory Breakdown
    writer.writerow(["=== TANKS INVENTORY & FEED SCHEDULE ==="])
    writer.writerow([
        "Tank ID", "Name", "Status", "Current Count", "Max Capacity",
        "Capacity %", "Avg Weight (g)", "Biomass (KG)", "Daily Feed (KG)", "Valuation (PHP)"
    ])
    for t in tanks:
        writer.writerow([
            t.get("tank_id"), t.get("name"), t.get("status"),
            t.get("current_count"), t.get("max_capacity"),
            f"{t.get('capacity_pct', 0)}%", t.get("avg_weight_g"),
            t.get("biomass_kg"), t.get("daily_feed_kg"), t.get("valuation_php")
        ])
    writer.writerow([])

    # Section 3: Dispersal Revenue Ledger
    writer.writerow(["=== DISPERSAL REVENUE LEDGER ==="])
    writer.writerow([
        "Dispersal ID", "Date", "Tank ID", "Tank Name", "Batch Code",
        "Recipient", "Type", "Price Unit", "Unit Price (PHP)", "Count Dispersed", "Total Revenue (PHP)"
    ])
    for d in dispersals:
        writer.writerow([
            d.get("dispersal_id"), d.get("date"), d.get("tank_id"), d.get("tank_name"),
            d.get("batch_code"), d.get("recipient"), d.get("type"),
            d.get("price_unit"), d.get("unit_price_php"), d.get("count"), d.get("total_revenue_php")
        ])

    csv_content = buf.getvalue()
    filename = f"tilapia_farm_production_report_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/analytics/mortality")
@app.get("/api/analytics/population-mortality")
@app.get("/api/reports/mortality")
async def get_mortality_analytics(
    tank_id: Optional[str] = None,
    days: int = 14,
    severity: Optional[str] = None,
):
    """Retrieve per-tank daily population, mortality counts, rates, timeline curves, and filtered records."""
    return await db.get_tank_mortality_analytics(tank_id=tank_id, days=days, severity=severity)


@app.get("/api/analytics/mortality/export/csv")
@app.get("/api/analytics/population-mortality/export/csv")
async def export_mortality_csv(
    tank_id: Optional[str] = None,
    days: int = 30,
    severity: Optional[str] = None,
):
    """Export filtered per-tank daily mortality data as CSV."""
    data = await db.get_tank_mortality_analytics(tank_id=tank_id, days=days, severity=severity)
    records = data.get("records", [])
    import csv, io

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "date", "tank_id", "tank_name", "population",
            "mortality_count", "mortality_rate_pct", "severity", "status", "dispersal_id",
            'estimated_mortality_count','confirmed_mortality_count','valid_monitoring_seconds','observed_monitoring_seconds'
        ])
        for r in records:
            writer.writerow([
                r.get("date"),
                r.get("tank_id"),
                r.get("tank_name"),
                r.get("population"),
                r.get("mortality_count"),
                r.get("mortality_rate_pct"),
                r.get("severity"),
                r.get("status_label"),
                r.get("dispersal_id"),
                r.get('estimated_mortality_count'),r.get('confirmed_mortality_count'),
                r.get('valid_monitoring_seconds'),r.get('observed_monitoring_seconds'),
            ])
        yield buf.getvalue()

    filename = f"tilapia_mortality_report_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# REST — analytics & records (aliased to avoid tracking-protection blocklists in Firefox)
# ---------------------------------------------------------------------------
@app.get("/api/stats/summary")
@app.get("/api/analytics/summary")
async def analytics_summary():
    return await db.query_summary()


@app.get("/api/stats/timeseries")
@app.get("/api/analytics/timeseries")
async def analytics_timeseries(hours: Optional[int] = None, limit: int = 300):
    rows = await db.query_timeseries(hours=hours, limit=limit)
    return rows


@app.get("/api/stats/hourly")
@app.get("/api/analytics/hourly")
async def analytics_hourly():
    return await db.query_hourly()


@app.get("/api/stats/model-stats")
@app.get("/api/analytics/model-stats")
async def analytics_model_stats():
    return await db.query_per_model_stats()


@app.get("/api/records")
@app.get("/api/analytics/events")
async def analytics_events(limit: int = 500):
    return await db.query_recent_events(limit=limit)


@app.get("/api/records/{event_id}")
@app.get("/api/analytics/events/{event_id}")
async def get_event_detail(event_id: int):
    """Retrieve full details for an individual detection event including model metrics and boxes."""
    detail = await db.query_event_detail(event_id)
    if not detail:
        raise HTTPException(404, f"Record #{event_id} not found")
    return detail


@app.post("/api/records/{event_id}/ground-truth")
@app.post("/api/analytics/events/{event_id}/ground-truth")
async def set_event_ground_truth(event_id: int, body: dict):
    """Update user-verified ground-truth fish count for a record and re-evaluate model accuracies."""
    if "ground_truth_count" not in body:
        raise HTTPException(400, "ground_truth_count is required")
    try:
        gt_count = int(body["ground_truth_count"])
    except ValueError:
        raise HTTPException(400, "ground_truth_count must be an integer")
    if gt_count < 0:
        raise HTTPException(400, "ground_truth_count must be non-negative")

    updated = await db.update_event_ground_truth(event_id, gt_count)
    if not updated:
        raise HTTPException(404, f"Record #{event_id} not found")
    return updated


@app.get("/api/analytics/tank-telemetry")
@app.get("/api/stats/tank-telemetry")
async def analytics_tank_telemetry(
    tank_id: Optional[str] = None,
    resolution: str = "minute",
    limit: int = 120,
):
    """
    Multi-granularity tank monitoring analytics timeseries:
    resolution = 'minute' | 'hour' | 'day'
    """
    return await db.query_tank_monitoring_telemetry(
        tank_id=tank_id,
        resolution=resolution,
        limit=limit,
    )


@app.get("/api/analytics/tank-telemetry/export/csv")
async def export_tank_telemetry_csv(
    tank_id: Optional[str] = None,
    resolution: str = "minute",
    limit: int = 5000,
):
    """Export multi-granularity (minute/hour/day) tank monitoring records as CSV."""
    data = await db.query_tank_monitoring_telemetry(
        tank_id=tank_id,
        resolution=resolution,
        limit=limit,
    )
    buckets = data.get("buckets", [])
    import csv, io

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "time_slot", "display_label", "tank_id", "tank_name",
            "samples", "avg_count", "peak_count", "min_count",
            "count_in", "count_out", "avg_density_pct", "avg_confidence", "latest_timestamp"
        ])
        for b in buckets:
            writer.writerow([
                b.get("time_slot"),
                b.get("display_label"),
                b.get("tank_id"),
                b.get("tank_name"),
                b.get("samples"),
                b.get("avg_count"),
                b.get("peak_count"),
                b.get("min_count"),
                b.get("count_in"),
                b.get("count_out"),
                b.get("avg_density"),
                b.get("avg_conf"),
                b.get("latest_ts"),
            ])
        yield buf.getvalue()

    filename = f"tilapia_tank_telemetry_{resolution}_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/database/reset")
async def reset_database_endpoint():
    """
    Delete all database records (events, bounding boxes, evaluation runs, benchmarks,
    dispersals, production logs, sessions) and reset tanks current_count to 0.
    Reclaims disk space with SQLite VACUUM.
    """
    await db.reset_all_data()
    _bench_viz_cache_invalidate()
    return {
        "status": "success",
        "message": "All database records purged and tank counts reset to 0.",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    }


@app.delete("/api/records")
@app.delete("/api/analytics/events")
async def analytics_clear_events():
    await db.delete_all_events()
    _bench_viz_cache_invalidate()
    return {"status": "cleared"}


@app.get("/api/records/export/csv")
@app.get("/api/analytics/export/csv")
async def analytics_export_csv():
    """Stream all detection events as a downloadable CSV."""
    events = await db.query_recent_events(limit=100_000)
    import csv, io

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "timestamp", "tank_id", "tank_name", "source", "fingerling_count",
            "count_in", "count_out", "density_pct", "status_level",
            "avg_confidence", "model_name", "ground_truth_count",
        ])
        for e in events:
            writer.writerow([
                e.get("id"), e.get("timestamp"), e.get("tank_id"), e.get("tank_name"), e.get("source"),
                e.get("fingerling_count"), e.get("count_in", 0),
                e.get("count_out", 0), e.get("density_pct"),
                e.get("status_level"), e.get("avg_confidence"),
                e.get("model_name"), e.get("ground_truth_count"),
            ])
        yield buf.getvalue()

    filename = f"tilapia_telemetry_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}.csv"
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
    dataset_path = (body.get("dataset_path") or "").strip()
    model_names  = body.get("model_names") or list(state.model_pool.keys())
    conf         = float(body.get("conf", 0.25))
    iou_thresh   = float(body.get("iou_thresh", 0.5))
    gt_csv       = (body.get("gt_csv") or "").strip() or None

    if not dataset_path and not gt_csv:
        raise HTTPException(400, "Please provide a Dataset Path or GT CSV path.")

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
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Benchmark execution failed: {str(exc)}")

    # Persist each model result
    saved_ids = []
    for r in results:
        if "error" not in r:
            rid = await db.save_evaluation_run(r)
            saved_ids.append(rid)

    return {"results": results, "saved_run_ids": saved_ids}


@app.post("/api/evaluate/from-records")
async def evaluate_from_records(limit: int = 500):
    """Instant benchmark comparison aggregated from saved records."""
    bench = await db.query_benchmark_from_records(limit=limit)
    if not bench:
        raise HTTPException(
            status_code=400,
            detail="No detection records found in database to benchmark. Please capture detections or upload media first."
        )
    return {"results": bench}


@app.post("/api/evaluate/upload-dataset")
async def evaluate_uploaded_dataset(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou_thresh: float = Form(0.50),
):
    """Accept a .zip dataset containing images/ (and optional labels/), extract and evaluate."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Dataset file must be a .zip archive")
    import tempfile, zipfile
    contents = await file.read()
    temp_dir = tempfile.mkdtemp(prefix="tilapia_dataset_")
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            zf.extractall(temp_dir)
        evaluator = ModelEvaluator(state.model_pool)
        results = await asyncio.to_thread(
            evaluator.run_benchmark,
            temp_dir,
            list(state.model_pool.keys()),
            conf,
            iou_thresh,
        )
        for r in results:
            if "error" not in r:
                await db.save_evaluation_run(r)
        return {"results": results}
    except Exception as exc:
        raise HTTPException(400, f"Benchmark dataset error: {str(exc)}")
    finally:
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@app.get("/api/evaluate/results")
async def evaluation_results(limit: int = 50):
    return await db.query_evaluation_runs(limit=limit)


@app.get("/api/evaluate/latest")
async def evaluation_latest():
    """Latest benchmark result per model for side-by-side comparison."""
    return await db.query_latest_eval_per_model()


@app.delete("/api/evaluate/results")
async def clear_evaluation_results():
    """Clear all stored evaluation benchmark runs and results."""
    async with db._lock:
        await db._conn.execute("DELETE FROM evaluation_runs")
        await db._conn.execute("DELETE FROM evaluation_benchmarks")
        await db._conn.execute("DELETE FROM evaluation_previews")
        await db._conn.commit()
    _bench_viz_cache_invalidate()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Benchmark visualization cache (server-side, ETag-aware)
# ---------------------------------------------------------------------------
_BENCH_VIZ_CACHE: dict[str, dict] = {}      # batch_id -> {"etag": str, "body": str}
_BENCH_VIZ_CACHE_MAX: int = 32              # FIFO cap to bound memory


def _bench_viz_cache_put(batch_id: str, etag: str, body: str):
    """Store a pre-serialized visualization payload keyed by benchmark batch id."""
    _BENCH_VIZ_CACHE[batch_id] = {"etag": etag, "body": body}
    while len(_BENCH_VIZ_CACHE) > _BENCH_VIZ_CACHE_MAX:
        _BENCH_VIZ_CACHE.pop(next(iter(_BENCH_VIZ_CACHE)))


def _bench_viz_cache_invalidate():
    """Drop all cached visualization payloads (call after deleting benchmark data)."""
    _BENCH_VIZ_CACHE.clear()


# ---------------------------------------------------------------------------
# REST — Academic Evaluation & Benchmark Mode (Per-Scan / Per-Sample)
# ---------------------------------------------------------------------------
@app.post("/api/evaluate-sample")
async def evaluate_sample_endpoint(
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    use_cached_image: bool = Form(False),
    annotation_file: Optional[UploadFile] = File(None),
    actual_count: Optional[int] = Form(None),
    conf: float = Form(0.50),
    iou_thresh: float = Form(0.50),
    models: Optional[str] = Form(None),
):
    """
    Dedicated Academic Evaluation / Benchmark Mode.
    Option A: Quick Ground Truth Count (scalar actual_count)
    Option B: Annotated Test Batch / Image (YOLO .txt annotation_file with IoU matching)
    Computes Precision, Recall, F1-Score, Counting Error (MAE, MAPE), and Confusion Matrix.
    Persists evaluation runs into evaluation_benchmarks SQLite table.
    """
    submitted = await request.form()
    if set(submitted) & {"conf", "iou_thresh", "models"}:
        raise HTTPException(422, "Counting settings are fixed by calibration.")
    # 1. Resolve image
    frame = None
    source_name = "cached_image"
    if image_file and image_file.filename:
        contents = await image_file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Cannot decode uploaded image file")
        source_name = image_file.filename
        # Cache this frame in memory
        state.cached_frame = frame.copy()
        state.cached_filename = image_file.filename
    elif use_cached_image or state.cached_frame is not None:
        if state.cached_frame is None:
            raise HTTPException(400, "No image currently cached in memory. Please upload an image first.")
        frame = state.cached_frame.copy()
        source_name = state.cached_filename or "cached_image"
    else:
        raise HTTPException(400, "Please upload an image file or enable use_cached_image.")

    # 2. Resolve Ground Truth
    gt_boxes = None
    gt_count_val = None

    if annotation_file and annotation_file.filename:
        ann_bytes = await annotation_file.read()
        ann_text = ann_bytes.decode("utf-8", errors="ignore")
        gt_boxes = parse_yolo_label_text(ann_text, frame.shape[1], frame.shape[0])
        gt_count_val = len(gt_boxes)
    elif actual_count is not None:
        if actual_count < 0:
            raise HTTPException(400, "Actual fish count cannot be negative.")
        gt_count_val = int(actual_count)
    else:
        raise HTTPException(400, "Provide the actual fish count or an annotation file.")

    # 3. Resolve Models
    if models:
        try:
            parsed_models = json.loads(models) if models.strip().startswith("[") else [m.strip() for m in models.split(",") if m.strip()]
        except Exception:
            parsed_models = [m.strip() for m in models.split(",") if m.strip()]
        model_names = [m for m in parsed_models if m in state.model_pool]
    else:
        model_names = state.active_models or list(state.model_pool.keys())

    if not model_names:
        raise HTTPException(400, "No valid models selected for evaluation")

    evaluator = ModelEvaluator(state.model_pool)
    profile = load_profile()
    def benchmark():
        with INFERENCE_LOCK:
            result = evaluator.evaluate_sample(frame, model_names, gt_count=gt_count_val, gt_boxes=gt_boxes,
                source_name=source_name, profiles=profile["models"])
            _, fused, _ = _infer_frame(frame, False)
            metrics = compute_academic_metrics([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in fused],
                gt_boxes=gt_boxes, gt_count=gt_count_val, iou_thresh=.5)
            metrics.update({"model_name": "ensemble", "confidence_threshold": None,
                "iou_threshold": profile["ensemble_iou"], "boxes": [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in fused],
                "boxes_conf": [b["conf"] for b in fused], "inference_ms": sum(m["inference_ms"] for m in result["models"]),
                "source_name": source_name})
            result["models"].append(metrics)
            return result
    result = await asyncio.to_thread(benchmark)

    # 4. Persist to evaluation_benchmarks table in SQLite
    saved_ids = []
    batch_ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S.%f")
    orig_h, orig_w = frame.shape[:2]
    for m in result["models"]:
        mape_val = float(m.get("mape", 0.0))
        acc_pct = float(m.get("accuracy_pct", round(max(0.0, 100.0 - mape_val), 2)))
        inf_ms = float(m.get("latency_ms", m.get("inference_ms", 0.0)))
        m["accuracy_pct"] = acc_pct
        m["inference_ms"] = inf_ms
        bench_row = {
            "timestamp": batch_ts,
            "model_name": m["model_name"],
            "confidence_threshold": m.get("confidence_threshold"),
            "iou_threshold": m.get("iou_threshold"),
            "actual_count": m["actual_count"],
            "predicted_count": m["predicted_count"],
            "mae": m["mae"],
            "mape": mape_val,
            "accuracy_pct": acc_pct,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1"],
            "inference_ms": inf_ms,
            "confusion_matrix": m["confusion_matrix"],
            "source_name": source_name,
            "evaluation_mode": m["evaluation_mode"],
            "boxes": m.get("boxes") or [],
            "boxes_conf": m.get("boxes_conf") or [],
        }
        rid = await db.save_evaluation_benchmark(bench_row)
        saved_ids.append(rid)

    # 5. Persist one downscaled source-image preview per batch (shared by all
    #    models of the batch) so the visualizer can re-render the boxes later.
    preview_frame = frame.copy()
    preview_max = 640
    if max(preview_frame.shape[:2]) > preview_max:
        pv_h, pv_w = preview_frame.shape[:2]
        pv_scale = preview_max / float(max(pv_h, pv_w))
        preview_frame = cv2.resize(
            preview_frame,
            (int(round(pv_w * pv_scale)), int(round(pv_h * pv_scale))),
            interpolation=cv2.INTER_AREA,
        )
    preview_b64 = await asyncio.to_thread(_frame_to_b64, preview_frame, 70)
    try:
        await db.save_evaluation_preview(
            f"{batch_ts}_{source_name}",
            source_name,
            preview_b64,
            orig_w,
            orig_h,
            gt_boxes,
        )
    except Exception as exc:
        print(f"[EVALUATE] Failed to store batch preview image: {exc}")

    result["saved_run_ids"] = saved_ids
    result["batch_timestamp"] = batch_ts
    return JSONResponse(result)


@app.get("/api/evaluation-benchmarks")
async def get_evaluation_benchmarks(limit: int = 100):
    """Retrieve saved academic evaluation benchmark records."""
    return await db.query_evaluation_benchmarks(limit=limit)


@app.get("/api/evaluation-benchmarks/grouped")
async def get_evaluation_benchmarks_grouped(limit: int = 50):
    """Retrieve saved academic evaluation benchmark records grouped by batch/timestamp for thesis side-by-side comparison."""
    return await db.query_evaluation_benchmarks_grouped(limit_batches=limit)


@app.get("/api/evaluation-benchmarks/visualize")
async def visualize_benchmark_batch(request: Request, batch_id: str):
    """
    Bounding-box render comparison for one benchmark batch: the downscaled source
    image plus the boxes each model rendered at evaluation time.
    Responses are served from an in-memory cache and carry an ETag, so repeat
    views revalidate cheaply (304) instead of re-querying the database.
    """
    key = f"{current_farm.get()}:{str(batch_id).strip()}"
    entry = _BENCH_VIZ_CACHE.get(key)

    if entry is None:
        payload = await db.query_evaluation_visualization(str(batch_id).strip())
        if payload is None:
            raise HTTPException(404, f"Benchmark batch '{key}' not found")
        body = json.dumps(payload)
        etag = f'"benchviz-{hashlib.md5(body.encode("utf-8")).hexdigest()}"'
        entry = {"etag": etag, "body": body}
        _bench_viz_cache_put(key, entry["etag"], entry["body"])

    if request.headers.get("if-none-match") == entry["etag"]:
        return Response(status_code=304, headers={"ETag": entry["etag"]})

    return Response(
        content=entry["body"],
        media_type="application/json",
        headers={"ETag": entry["etag"]},
    )


@app.delete("/api/evaluation-benchmarks")
async def delete_evaluation_benchmarks():
    """Clear all saved academic evaluation benchmark records."""
    await db.delete_evaluation_benchmarks()
    _bench_viz_cache_invalidate()
    return {"status": "cleared"}


@app.get("/api/evaluation-benchmarks/export/csv")
async def export_evaluation_benchmarks_csv():
    """Export academic evaluation benchmark history as CSV for thesis defense analysis."""
    runs = await db.query_evaluation_benchmarks(limit=50_000)
    import csv, io

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "timestamp", "model_name", "confidence_threshold",
            "iou_threshold", "actual_count", "predicted_count", "mae",
            "mape_pct", "accuracy_pct", "precision", "recall", "f1_score",
            "inference_ms", "true_positives", "false_positives", "false_negatives",
            "evaluation_mode", "source_name"
        ])
        for r in runs:
            cm = r.get("confusion_matrix") or {}
            writer.writerow([
                r.get("id"),
                r.get("timestamp"),
                r.get("model_name"),
                r.get("confidence_threshold"),
                r.get("iou_threshold"),
                r.get("actual_count"),
                r.get("predicted_count"),
                r.get("mae"),
                r.get("mape"),
                r.get("accuracy_pct"),
                r.get("precision"),
                r.get("recall"),
                r.get("f1_score"),
                r.get("inference_ms"),
                cm.get("tp", 0),
                cm.get("fp", 0),
                cm.get("fn", 0),
                r.get("evaluation_mode"),
                r.get("source_name"),
            ])
        yield buf.getvalue()

    filename = f"tilapia_academic_benchmarks_{datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# Account/farm boundary and management routes
# ---------------------------------------------------------------------------
from farm_api import register, websocket_context, validate_source, validate_tank_selection
register(app, state, AppState, _SESSION_STATES, db)


async def _protected_websocket(ws, tank_id):
    contexts = await websocket_context(ws, AppState)
    if contexts is None:
        return
    try:
        if app.state.production.enabled:
            await _production_websocket(ws,tank_id)
        else:
            await _handle_websocket_live(ws, tank_id)
    finally:
        for variable, token in reversed(contexts):
            variable.reset(token)


async def _production_websocket(ws,tank_id):
    """Authenticated viewers subscribe; only explicit controls change durable running state."""
    import contextlib
    manager=app.state.production
    farm=current_farm.get()
    if not tank_id or not await db.get_tank(tank_id):
        await ws.close(code=4404)
        return
    await ws.accept()
    await ws.send_json({'type':'tank_bound','tank':await db.get_tank(tank_id)})
    attached=None; queue=None

    async def send_updates():
        nonlocal attached,queue
        last_status=None
        while True:
            if not manager.enabled:
                await ws.close(code=1012)
                return
            if not await accounts.authenticate(ws.cookies.get('tilapia_session')):
                await ws.close(code=4401)
                return
            worker=manager.workers.get((farm,tank_id))
            if worker is not attached:
                if attached:
                    attached.subscribers.discard(queue)
                attached,queue=await manager.subscribe(farm,tank_id)
            if attached:
                try:
                    event=await asyncio.wait_for(queue.get(),2)
                    await ws.send_json(public_value(event))
                except asyncio.TimeoutError:
                    pass
            else:
                status=next(t for t in (await manager.status(farm))['tanks'] if t['tank_id']==tank_id)
                value=(status['status'],status['message'])
                if value!=last_status:
                    await ws.send_json({'type':'stream_state','status':'stopped' if status['status']=='paused' else status['status'],
                        'message':status['message'],'tank_id':tank_id,'production':True})
                    last_status=value
                await asyncio.sleep(1)

    sending=asyncio.create_task(send_updates())
    receiving=None
    try:
        while True:
            receiving=asyncio.create_task(ws.receive_json())
            done,_=await asyncio.wait([sending,receiving],return_when=asyncio.FIRST_COMPLETED)
            if sending in done:
                await sending
                break
            message=receiving.result()
            user=await accounts.authenticate(ws.cookies.get('tilapia_session'))
            if not user:
                await ws.close(code=4401); break
            if message.get('type') in ('start_stream','stop_stream'):
                config=next(c for c in await db.monitoring_configs() if c['tank_id']==tank_id)
                if message['type']=='start_stream' and not config['validation']:
                    await ws.send_json({'type':'error','message':'Validate this live camera census before starting Production monitoring.'})
                    continue
                await db.configure_monitoring(tank_id,enabled=message['type']=='start_stream')
                await accounts.audit(user['id'],farm,'monitoring_control',json.dumps({'tank_id':tank_id,'enabled':message['type']=='start_stream'}))
                await manager.sync()
            elif message.get('type') not in ('subscribe','set_overlay'):
                await ws.send_json({'type':'error','message':'Production Mode accepts only live camera monitoring.'})
    except WebSocketDisconnect:
        pass
    finally:
        if receiving and not receiving.done():
            receiving.cancel()
        sending.cancel()
        await asyncio.gather(*([receiving] if receiving else []),sending,return_exceptions=True)
        if attached:
            attached.subscribers.discard(queue)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
