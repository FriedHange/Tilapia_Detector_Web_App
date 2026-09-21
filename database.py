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
        track_ids_json   TEXT
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
    # Indexes for fast time-series queries
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON detection_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON detection_events(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluation_runs(model_name)",
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
                    status_level, source, model_name, track_ids_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, now, frame_idx, fingerling_count,
                    count_in, count_out, avg_conf, density_pct,
                    status_level, source, model_name,
                    json.dumps(track_ids),
                ),
            )
            event_id = cursor.lastrowid

            if boxes:
                await self._conn.executemany(
                    """INSERT INTO bounding_boxes
                       (event_id, track_id, class_name, confidence, x1, y1, x2, y2)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            event_id,
                            b.get("track_id"),
                            b.get("class_name", ""),
                            b.get("conf", 0.0),
                            b.get("x1", 0), b.get("y1", 0),
                            b.get("x2", 0), b.get("y2", 0),
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
                      COALESCE(model_name, '') AS model_name
               FROM detection_events
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

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


# ---------------------------------------------------------------------------
# Module-level singleton for import convenience
# ---------------------------------------------------------------------------
db = Database()
