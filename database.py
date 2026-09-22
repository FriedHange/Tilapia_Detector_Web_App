"""
database.py — Async SQLite database layer for Tilapia Fingerling Web App
=========================================================================
Extends the original COCO Detection Suite schema with:
  - async/await support via aiosqlite (FastAPI-compatible)
  - tracking columns (count_in, count_out, track_ids_json)
  - evaluation_runs table (Precision, Recall, F1, MAE, MAPE)
  - bounding_boxes table preserved for per-frame detail
"""

import aiosqlite
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "tilapia_web_analytics.db"

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_DDL = [
    # Sessions
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id   TEXT PRIMARY KEY,
        start_time   TEXT NOT NULL,
        source_type  TEXT,
        model_name   TEXT,
        tub_capacity INTEGER DEFAULT 100
    )
    """,
    # Detection events (core telemetry)
    """
    CREATE TABLE IF NOT EXISTS detection_events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id       TEXT,
        timestamp        TEXT NOT NULL,
        frame_idx        INTEGER,
        fingerling_count INTEGER DEFAULT 0,
        count_in         INTEGER DEFAULT 0,
        count_out        INTEGER DEFAULT 0,
        avg_confidence   REAL DEFAULT 0.0,
        density_pct      REAL DEFAULT 0.0,
        status_level     TEXT,
        source           TEXT,
        model_name       TEXT,
        track_ids_json   TEXT,
        model_metrics_json TEXT,
        ground_truth_count INTEGER
    )
    """,
    # Bounding boxes per event
    """
    CREATE TABLE IF NOT EXISTS bounding_boxes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id   INTEGER,
        track_id   INTEGER,
        class_name TEXT,
        confidence REAL,
        x1 REAL, y1 REAL, x2 REAL, y2 REAL,
        model_name TEXT,
        FOREIGN KEY (event_id) REFERENCES detection_events(id)
    )
    """,
    # Model benchmark evaluation runs
    """
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp    TEXT NOT NULL,
        model_name   TEXT NOT NULL,
        dataset_path TEXT,
        conf         REAL,
        iou          REAL,
        precision    REAL,
        recall       REAL,
        f1           REAL,
        mae          REAL,
        mape         REAL,
        tp           INTEGER,
        fp           INTEGER,
        fn           INTEGER,
        total_images INTEGER,
        details_json TEXT
    )
    """,
    # Academic evaluation benchmarks table (per-scan / per-batch evaluation)
    """
    CREATE TABLE IF NOT EXISTS evaluation_benchmarks (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp             TEXT NOT NULL,
        model_name            TEXT NOT NULL,
        confidence_threshold  REAL,
        iou_threshold         REAL,
        actual_count          INTEGER,
        predicted_count       INTEGER,
        mae                   REAL,
        mape                  REAL,
        precision             REAL,
        recall                REAL,
        f1_score              REAL,
        confusion_matrix_json TEXT,
        source_name           TEXT,
        evaluation_mode       TEXT
    )
    """,
    # Indexes for fast time-series queries
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON detection_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON detection_events(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluation_runs(model_name)",
    "CREATE INDEX IF NOT EXISTS idx_eval_bench_model ON evaluation_benchmarks(model_name)",
    "CREATE INDEX IF NOT EXISTS idx_eval_bench_ts ON evaluation_benchmarks(timestamp)",
]


# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------
class Database:
    """
    Async SQLite manager.  One shared connection is kept open for the
    lifetime of the FastAPI app (opened in lifespan, closed on shutdown).
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self):
        """Open the shared connection and initialise the schema."""
        self._conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = aiosqlite.Row
        await self._init_schema()

    async def close(self):
        """Close the shared connection gracefully."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _init_schema(self):
        async with self._lock:
            for stmt in _DDL:
                await self._conn.execute(stmt)
            # Safe column migration for existing tables
            cursor = await self._conn.execute("PRAGMA table_info(detection_events)")
            cols = {row["name"] for row in await cursor.fetchall()}
            if "model_metrics_json" not in cols:
                await self._conn.execute("ALTER TABLE detection_events ADD COLUMN model_metrics_json TEXT")
            if "ground_truth_count" not in cols:
                await self._conn.execute("ALTER TABLE detection_events ADD COLUMN ground_truth_count INTEGER")

            cursor_bb = await self._conn.execute("PRAGMA table_info(bounding_boxes)")
            bb_cols = {row["name"] for row in await cursor_bb.fetchall()}
            if "model_name" not in bb_cols:
                await self._conn.execute("ALTER TABLE bounding_boxes ADD COLUMN model_name TEXT")
            await self._conn.commit()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    async def create_session(self, session_id: str, source_type: str,
                              model_name: str, tub_capacity: int = 100):
        async with self._lock:
            await self._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, start_time, source_type, model_name, tub_capacity)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, datetime.now().isoformat(), source_type,
                 model_name, tub_capacity),
            )
            await self._conn.commit()

    # ------------------------------------------------------------------
    # Detection event logging
    # ------------------------------------------------------------------
    async def log_event(
        self,
        session_id: str,
        source: str,
        frame_idx: int,
        fingerling_count: int,
        count_in: int,
        count_out: int,
        avg_conf: float,
        density_pct: float,
        status_level: str,
        model_name: str,
        boxes: list[dict],
        track_ids: list[int],
        model_metrics: Optional[list[dict]] = None,
        ground_truth_count: Optional[int] = None,
    ) -> int:
        """
        Insert one detection event and its associated bounding boxes.
        Returns the new event ID.
        """
        async with self._lock:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor = await self._conn.execute(
                """INSERT INTO detection_events
                   (session_id, timestamp, frame_idx, fingerling_count,
                    count_in, count_out, avg_confidence, density_pct,
                    status_level, source, model_name, track_ids_json,
                    model_metrics_json, ground_truth_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, now, frame_idx, fingerling_count,
                    count_in, count_out, avg_conf, density_pct,
                    status_level, source, model_name,
                    json.dumps(track_ids),
                    json.dumps(model_metrics) if model_metrics else None,
                    ground_truth_count,
                ),
            )
            event_id = cursor.lastrowid

            if boxes:
                await self._conn.executemany(
                    """INSERT INTO bounding_boxes
                       (event_id, track_id, class_name, confidence, x1, y1, x2, y2, model_name)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            event_id,
                            b.get("track_id"),
                            b.get("class_name", ""),
                            b.get("conf", 0.0),
                            b.get("x1", 0), b.get("y1", 0),
                            b.get("x2", 0), b.get("y2", 0),
                            b.get("model") or b.get("model_name", ""),
                        )
                        for b in boxes
                    ],
                )
            await self._conn.commit()
            return event_id

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------
    async def query_summary(self) -> dict:
        cursor = await self._conn.execute(
            """SELECT
                   COUNT(*)           AS total_events,
                   AVG(fingerling_count) AS avg_count,
                   MAX(fingerling_count) AS peak_count,
                   MIN(fingerling_count) AS min_count,
                   AVG(avg_confidence)   AS avg_conf,
                   MAX(count_in)         AS total_in,
                   MAX(count_out)        AS total_out,
                   MIN(timestamp)        AS first_event,
                   MAX(timestamp)        AS last_event
               FROM detection_events"""
        )
        row = await cursor.fetchone()
        if not row:
            return {}
        return dict(row)

    async def query_timeseries(self, hours: Optional[int] = None,
                                limit: int = 300) -> list[dict]:
        """Aggregated time-series grouped by minute for Chart.js."""
        params = []
        where = ""
        if hours:
            cutoff = (datetime.now() - timedelta(hours=hours)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            where = "WHERE timestamp >= ?"
            params.append(cutoff)

        cursor = await self._conn.execute(
            f"""SELECT
                    strftime('%m-%d %H:%M', timestamp) AS slot,
                    AVG(fingerling_count)               AS avg_count,
                    MAX(fingerling_count)               AS max_count,
                    AVG(density_pct)                    AS avg_density,
                    COUNT(*)                            AS samples
                FROM detection_events
                {where}
                GROUP BY slot
                ORDER BY slot ASC
                LIMIT ?""",
            params + [limit],
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def query_hourly(self) -> list[dict]:
        cursor = await self._conn.execute(
            """SELECT
                   strftime('%H:00', timestamp) AS hour_slot,
                   AVG(fingerling_count)         AS avg_count,
                   COUNT(*)                      AS samples
               FROM detection_events
               GROUP BY hour_slot
               ORDER BY hour_slot ASC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def query_recent_events(self, limit: int = 200) -> list[dict]:
        cursor = await self._conn.execute(
            """SELECT id, timestamp, source, fingerling_count, count_in,
                      count_out, density_pct, status_level, avg_confidence,
                      COALESCE(model_name, '') AS model_name,
                      ground_truth_count, model_metrics_json
               FROM detection_events
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("model_metrics_json"):
                try:
                    d["model_metrics"] = json.loads(d["model_metrics_json"])
                except Exception:
                    d["model_metrics"] = []
            else:
                d["model_metrics"] = []
            result.append(d)
        return result

    async def query_event_detail(self, event_id: int) -> Optional[dict]:
        """Fetch complete details for a single record including per-model metrics and boxes."""
        cursor = await self._conn.execute(
            """SELECT id, session_id, timestamp, frame_idx, fingerling_count,
                      count_in, count_out, avg_confidence, density_pct,
                      status_level, source, model_name, track_ids_json,
                      model_metrics_json, ground_truth_count
               FROM detection_events
               WHERE id = ?""",
            (event_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        event = dict(row)

        # Parse JSON fields
        try:
            event["track_ids"] = json.loads(event.get("track_ids_json") or "[]")
        except Exception:
            event["track_ids"] = []

        try:
            event["model_metrics"] = json.loads(event.get("model_metrics_json") or "[]")
        except Exception:
            event["model_metrics"] = []

        # Fetch bounding boxes
        b_cursor = await self._conn.execute(
            """SELECT id, track_id, class_name, confidence, x1, y1, x2, y2, COALESCE(model_name, '') AS model_name
               FROM bounding_boxes
               WHERE event_id = ?
               ORDER BY confidence DESC""",
            (event_id,),
        )
        b_rows = await b_cursor.fetchall()
        event["boxes"] = [dict(b) for b in b_rows]

        # If ground truth count is present, attach per-model accuracy
        gt = event.get("ground_truth_count")
        if gt is not None and event["model_metrics"]:
            for m in event["model_metrics"]:
                count = m.get("count", 0)
                err = abs(count - gt)
                m["abs_error"] = err
                m["accuracy_pct"] = round(max(0.0, 100.0 - (err / max(1, gt) * 100.0)), 2)

        return event

    async def update_event_ground_truth(self, event_id: int, gt_count: int) -> Optional[dict]:
        """Save user-verified ground-truth fish count for a record and update model metrics."""
        async with self._lock:
            await self._conn.execute(
                """UPDATE detection_events
                   SET ground_truth_count = ?
                   WHERE id = ?""",
                (gt_count, event_id),
            )
            await self._conn.commit()
        return await self.query_event_detail(event_id)

    async def query_benchmark_from_records(self, limit: int = 500) -> list[dict]:
        """
        Aggregate side-by-side benchmark comparison from stored records that have per-model metrics.
        """
        cursor = await self._conn.execute(
            """SELECT id, model_metrics_json, ground_truth_count, fingerling_count
               FROM detection_events
               WHERE model_metrics_json IS NOT NULL AND model_metrics_json != ''
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        # Model aggregators
        models_data: dict[str, dict] = {}

        for r in rows:
            try:
                metrics = json.loads(r["model_metrics_json"])
            except Exception:
                continue

            gt = r["ground_truth_count"]
            # Consensus count is the median or ensemble fingerling_count
            consensus = r["fingerling_count"] or (sum(m.get("count", 0) for m in metrics) / max(1, len(metrics)))

            for m in metrics:
                m_name = m.get("model") or m.get("model_name")
                if not m_name:
                    continue
                if m_name not in models_data:
                    models_data[m_name] = {
                        "model_name": m_name,
                        "counts": [],
                        "confs": [],
                        "latencies": [],
                        "errors": [],
                        "gt_errors": [],
                    }

                cnt = m.get("count", 0)
                models_data[m_name]["counts"].append(cnt)
                if m.get("avg_conf") is not None:
                    models_data[m_name]["confs"].append(float(m["avg_conf"]))
                if m.get("inference_ms") is not None:
                    models_data[m_name]["latencies"].append(float(m["inference_ms"]))

                # Error vs consensus
                models_data[m_name]["errors"].append(abs(cnt - consensus))

                # Error vs ground truth (if available)
                if gt is not None:
                    models_data[m_name]["gt_errors"].append(abs(cnt - gt))

        summary_list = []
        for m_name, d in models_data.items():
            n = len(d["counts"])
            if n == 0:
                continue
            avg_cnt = sum(d["counts"]) / n
            avg_conf = sum(d["confs"]) / len(d["confs"]) if d["confs"] else 0.0
            avg_lat = sum(d["latencies"]) / len(d["latencies"]) if d["latencies"] else 0.0
            mae_consensus = sum(d["errors"]) / n if d["errors"] else 0.0

            mae_gt = (sum(d["gt_errors"]) / len(d["gt_errors"])) if d["gt_errors"] else None
            mape_gt = None
            if d["gt_errors"]:
                mape_gt = round((mae_gt / max(1.0, avg_cnt)) * 100.0, 2)

            # Simulated detection metrics derived from consensus agreement
            precision = max(0.0, 1.0 - (mae_consensus / max(1.0, avg_cnt))) if avg_cnt > 0 else 0.0
            recall = max(0.0, min(1.0, avg_cnt / max(1.0, avg_cnt + mae_consensus))) if avg_cnt > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            summary_list.append({
                "model_name": m_name,
                "total_images": n,
                "avg_count": round(avg_cnt, 1),
                "avg_confidence": round(avg_conf, 3),
                "avg_inference_ms": round(avg_lat, 2),
                "mae": round(mae_gt if mae_gt is not None else mae_consensus, 3),
                "mape": mape_gt if mape_gt is not None else round((mae_consensus / max(1.0, avg_cnt)) * 100.0, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "has_ground_truth": bool(d["gt_errors"]),
            })

        summary_list.sort(key=lambda x: x["f1"], reverse=True)
        return summary_list

    async def query_per_model_stats(self) -> list[dict]:
        cursor = await self._conn.execute(
            """SELECT
                   COALESCE(model_name, 'Unknown') AS model,
                   COUNT(*)                        AS events,
                   AVG(fingerling_count)            AS avg_count,
                   MAX(fingerling_count)            AS peak_count,
                   AVG(avg_confidence)              AS avg_conf
               FROM detection_events
               WHERE model_name IS NOT NULL AND model_name != ''
               GROUP BY model
               ORDER BY events DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def query_status_distribution(self) -> list[dict]:
        cursor = await self._conn.execute(
            """SELECT status_level, COUNT(*) AS cnt
               FROM detection_events
               GROUP BY status_level
               ORDER BY cnt DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def query_source_breakdown(self) -> list[dict]:
        cursor = await self._conn.execute(
            """SELECT source, COUNT(*) AS cnt, AVG(fingerling_count) AS avg_count
               FROM detection_events
               GROUP BY source
               ORDER BY cnt DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete_all_events(self):
        async with self._lock:
            await self._conn.execute("DELETE FROM bounding_boxes")
            await self._conn.execute("DELETE FROM detection_events")
            await self._conn.execute("DELETE FROM evaluation_runs")
            await self._conn.execute("DELETE FROM evaluation_benchmarks")
            await self._conn.commit()

    # ------------------------------------------------------------------
    # Evaluation run storage
    # ------------------------------------------------------------------
    async def save_evaluation_run(self, result: dict) -> int:
        async with self._lock:
            cursor = await self._conn.execute(
                """INSERT INTO evaluation_runs
                   (timestamp, model_name, dataset_path, conf, iou,
                    precision, recall, f1, mae, mape,
                    tp, fp, fn, total_images, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    result.get("model_name", ""),
                    result.get("dataset_path", ""),
                    result.get("conf", 0.25),
                    result.get("iou", 0.45),
                    result.get("precision", 0.0),
                    result.get("recall", 0.0),
                    result.get("f1", 0.0),
                    result.get("mae", 0.0),
                    result.get("mape", 0.0),
                    result.get("tp", 0),
                    result.get("fp", 0),
                    result.get("fn", 0),
                    result.get("total_images", 0),
                    json.dumps(result.get("details", [])),
                ),
            )
            await self._conn.commit()
            return cursor.lastrowid

    async def query_evaluation_runs(self, limit: int = 50) -> list[dict]:
        cursor = await self._conn.execute(
            """SELECT run_id, timestamp, model_name, dataset_path,
                      conf, iou, precision, recall, f1, mae, mape,
                      tp, fp, fn, total_images
               FROM evaluation_runs
               ORDER BY run_id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def query_latest_eval_per_model(self) -> list[dict]:
        """Latest benchmark result per model — used for side-by-side comparison."""
        cursor = await self._conn.execute(
            """SELECT e.*
               FROM evaluation_runs e
               INNER JOIN (
                   SELECT model_name, MAX(run_id) AS max_id
                   FROM evaluation_runs
                   GROUP BY model_name
               ) latest ON e.model_name = latest.model_name
                       AND e.run_id = latest.max_id
               ORDER BY e.f1 DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Academic Evaluation Benchmarks (Per-Scan / Per-Sample)
    # ------------------------------------------------------------------
    async def save_evaluation_benchmark(self, bench: dict) -> int:
        """Persist a single academic evaluation benchmark record."""
        async with self._lock:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor = await self._conn.execute(
                """INSERT INTO evaluation_benchmarks
                   (timestamp, model_name, confidence_threshold, iou_threshold,
                    actual_count, predicted_count, mae, mape, precision,
                    recall, f1_score, confusion_matrix_json, source_name, evaluation_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    bench.get("timestamp") or now,
                    bench.get("model_name", ""),
                    float(bench.get("confidence_threshold", 0.5)),
                    float(bench.get("iou_threshold", 0.5)),
                    int(bench.get("actual_count", 0)),
                    int(bench.get("predicted_count", 0)),
                    float(bench.get("mae", 0.0)),
                    float(bench.get("mape", 0.0)),
                    float(bench.get("precision", 0.0)),
                    float(bench.get("recall", 0.0)),
                    float(bench.get("f1_score", 0.0)),
                    json.dumps(bench.get("confusion_matrix") or {}),
                    bench.get("source_name", "manual"),
                    bench.get("evaluation_mode", "quick_count"),
                ),
            )
            await self._conn.commit()
            return cursor.lastrowid

    async def query_evaluation_benchmarks(self, limit: int = 100) -> list[dict]:
        """Fetch historical academic evaluation benchmark runs."""
        cursor = await self._conn.execute(
            """SELECT id, timestamp, model_name, confidence_threshold, iou_threshold,
                      actual_count, predicted_count, mae, mape, precision, recall,
                      f1_score, confusion_matrix_json, source_name, evaluation_mode
               FROM evaluation_benchmarks
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["confusion_matrix"] = json.loads(d.get("confusion_matrix_json") or "{}")
            except Exception:
                d["confusion_matrix"] = {}
            result.append(d)
        return result

    async def delete_evaluation_benchmarks(self):
        """Clear all historical academic evaluation benchmark runs."""
        async with self._lock:
            await self._conn.execute("DELETE FROM evaluation_benchmarks")
            await self._conn.commit()


# ---------------------------------------------------------------------------
# Module-level singleton for import convenience
# ---------------------------------------------------------------------------
db = Database()
