"""Farm operations, recorded inventory, feeding and daily mortality.

Each farm has its own SQLite file. This keeps the existing analytics queries
inside the same isolation boundary as the newer management operations.
"""
from __future__ import annotations

import math
import re
import uuid
import asyncio
from functools import wraps
from datetime import datetime, timedelta, timezone

from database import Database
from production_records import ProductionRecords

MANILA = timezone(timedelta(hours=8))


class TankPopulationChanged(ValueError):
    """The population shown when deletion was confirmed is no longer current."""


def now_text():
    return datetime.now(MANILA).strftime("%Y-%m-%d %H:%M:%S")


def integer(value, label, minimum=0):
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or number < minimum:
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a whole number of at least {minimum}.")


def number(value, label, minimum=0):
    try:
        result = float(value)
        if not math.isfinite(result) or result < minimum:
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number of at least {minimum}.")


def consistent_read(function):
    @wraps(function)
    async def read(self, *args, **kwargs):
        await self._ensure_conn()
        task = asyncio.current_task()
        if getattr(self, "_reader", None) is task:
            return await function(self, *args, **kwargs)
        async with self._lock:
            self._reader = task
            try:
                return await function(self, *args, **kwargs)
            finally:
                self._reader = None
    return read


class FarmDatabase(ProductionRecords, Database):
    def __init__(self, db_path, *, seed=False):
        super().__init__(db_path, seed_defaults=seed)
        self.seed = seed

    async def _init_schema(self):
        await super()._init_schema()
        await self._conn.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS inventory_baselines (
                tank_id TEXT PRIMARY KEY REFERENCES tanks(tank_id),
                date TEXT NOT NULL, population INTEGER NOT NULL, known INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inventory_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT UNIQUE NOT NULL,
                tank_id TEXT NOT NULL REFERENCES tanks(tank_id),
                kind TEXT NOT NULL, delta INTEGER NOT NULL,
                timestamp TEXT NOT NULL, note TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_movements_tank_date ON inventory_movements(tank_id,timestamp);
            CREATE TABLE IF NOT EXISTS mortality_checks (
                tank_id TEXT NOT NULL REFERENCES tanks(tank_id), date TEXT NOT NULL,
                recorded_at TEXT NOT NULL, PRIMARY KEY(tank_id,date)
            );
            CREATE TABLE IF NOT EXISTS feed_items (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, stock_kg REAL NOT NULL DEFAULT 0,
                low_stock_kg REAL NOT NULL DEFAULT 5
            );
            CREATE TABLE IF NOT EXISTS feed_transactions (
                id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES feed_items(id),
                tank_id TEXT REFERENCES tanks(tank_id), kind TEXT NOT NULL,
                quantity_kg REAL NOT NULL, cost_php REAL NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL, note TEXT NOT NULL DEFAULT ''
            );
        """)
        # Existing populations are a snapshot, not evidence of past opening stock.
        await self._conn.execute("""INSERT OR IGNORE INTO inventory_baselines(tank_id,date,population,known)
            SELECT tank_id, ?, current_count, 0 FROM tanks""", (now_text()[:10],))
        # Import only actual historical losses. Their population denominator is
        # unknown; these entries must not deduct stock from the existing snapshot.
        await self._conn.execute("""INSERT OR IGNORE INTO inventory_movements
            (operation_id,tank_id,kind,delta,timestamp,note)
            SELECT 'legacy-deaths:'||l.id,l.tank_id,'legacy_mortality',-l.mortality_count,l.timestamp,'Historical production log'
            FROM tank_production_logs l WHERE l.mortality_count>0 AND EXISTS(SELECT 1 FROM tanks t WHERE t.tank_id=l.tank_id)
            AND NOT EXISTS(SELECT 1 FROM inventory_movements m WHERE m.operation_id='deaths:'||l.dispersal_id)""")
        await self.init_production_records()
        await self._conn.commit()

    async def _rows(self, sql, args=()):
        cursor = await self._conn.execute(sql, args)
        return [dict(row) for row in await cursor.fetchall()]

    def _enrich_tank(self, data):
        data = super()._enrich_tank(data)
        data["biomass_kg"] = round(data["current_count"] * data["avg_weight_g"] / 1000, 6)
        data["daily_feed_kg"] = round(data["biomass_kg"] * data["feed_rate_pct"], 6)
        data["valuation_php"] = round(data["biomass_kg"] * 160, 2)
        return data

    @consistent_read
    async def get_tanks(self):
        return await super().get_tanks()

    @consistent_read
    async def get_tank(self, tank_id):
        return await super().get_tank(tank_id)

    async def _tank(self, tank_id):
        rows = await self._rows("SELECT * FROM tanks WHERE tank_id=?", (tank_id,))
        if not rows:
            raise ValueError("Tank not found.")
        if rows[0]['status'] == 'inactive':
            raise ValueError('This tank has been removed. Its history remains available.')
        return rows[0]

    @staticmethod
    def _source_choice(data, old_source='', live='', video='', *, production=False):
        selected = data.get('source')
        if selected is None:
            value = str(data.get('camera_source',old_source)).strip()
            if 'camera_source' not in data:
                return old_source,live,video
            kind = 'none' if not value else 'usb' if value.isdigit() else 'rtsp' if value.startswith(('rtsp://','rtsps://')) else 'video'
        else:
            if not isinstance(selected,dict) or selected.get('type') not in ('none','usb','rtsp','video'):
                raise ValueError('Choose a valid source type.')
            kind,value = selected['type'],str(selected.get('value','')).strip()
        if kind=='video':
            if production:
                raise ValueError('Production Mode accepts only live cameras.')
            if not value:
                raise ValueError('Upload or choose a video first.')
            return value,live,value
        if kind=='none':
            return '', '', video
        if kind=='usb':
            if not value.isdigit() or int(value)>20:
                raise ValueError('Choose a camera number between 0 and 20.')
            value=str(int(value))
        elif not value.startswith(('rtsp://','rtsps://')):
            raise ValueError('Enter an RTSP network camera address.')
        # Editing the production camera preserves a selected demonstration video.
        demo = old_source if production and old_source and not old_source.isdigit() and not old_source.startswith(('rtsp://','rtsps://')) else value
        return demo,value,video

    async def create_tank(self, data, *, production=False):
        await self._ensure_conn()
        name = str(data.get("name", "")).strip()
        if len(name) > 120:
            raise ValueError("Enter a tank name (up to 120 characters).")
        tank_id = str(data.get("tank_id") or '').strip()
        if tank_id and not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", tank_id):
            raise ValueError("Use up to 80 letters, digits, dots, dashes or underscores for the tank code.")
        count = integer(data.get("current_count", 0), "Initial fish count")
        capacity = integer(data.get("max_capacity", 1000), "Capacity", 1)
        weight = number(data.get("avg_weight_g", 2.5), "Average weight", 0.001)
        feed = number(data.get("feed_rate_pct", .05), "Feed rate", .001)
        if feed > .15:
            raise ValueError("Feed rate must be between 0.1% and 15%.")
        source,live,video = self._source_choice(data,production=production)
        stamp = now_text()
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                if not tank_id:
                    codes = await self._rows('SELECT tank_id FROM tanks')
                    sequence = max((int(m[1]) for row in codes if (m:=re.fullmatch(r'TANK-(\d+)',row['tank_id']))),default=0)+1
                    tank_id=f'TANK-{sequence:02d}'
                if not name:
                    suffix=tank_id[5:] if tank_id.startswith('TANK-') else tank_id
                    name=f'Tank {suffix}'
                await self._conn.execute("""INSERT INTO tanks(tank_id,name,camera_source,max_capacity,
                    current_count,avg_weight_g,feed_rate_pct,status) VALUES(?,?,?,?,?,?,?,?)""",
                    (tank_id, name, source, capacity, count, weight, feed, "active"))
                await self._conn.execute("INSERT INTO inventory_baselines VALUES(?,?,?,1)", (tank_id, stamp[:10], 0))
                if count:
                    await self._movement(tank_id, "stocking", count, uuid.uuid4().hex, "Initial stocking", stamp)
                await self._conn.execute('INSERT INTO monitoring_config(tank_id,live_source,video_source,updated_at) VALUES(?,?,?,?)', (tank_id,live,video,stamp))
                await self._conn.commit()
            except Exception as exc:
                await self._conn.rollback()
                if "UNIQUE" in str(exc):
                    raise ValueError("This tank code already exists in your farm.") from exc
                raise
        return await self.get_tank(tank_id)

    async def update_tank(self, tank_id, data, *, production=False):
        await self._ensure_conn()
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                tank = await self._tank(tank_id)
                config=(await self._rows('SELECT * FROM monitoring_config WHERE tank_id=?',(tank_id,)))[0]
                source,live,video=self._source_choice(data,tank['camera_source'],config['live_source'],config['video_source'],production=production)
                name = str(data.get("name", tank["name"])).strip()
                if not name or len(name) > 120:
                    raise ValueError("Enter a tank name (up to 120 characters).")
                count = integer(data.get("current_count", tank["current_count"]), "Fish count")
                capacity = integer(data.get("max_capacity", tank["max_capacity"]), "Capacity", 1)
                weight = number(data.get("avg_weight_g", tank["avg_weight_g"]), "Average weight", .001)
                feed = number(data.get("feed_rate_pct", tank["feed_rate_pct"]), "Feed rate", .001)
                if feed > .15:
                    raise ValueError("Feed rate must be between 0.1% and 15%.")
                status = data.get("status", tank["status"])
                if status not in ("active", "quarantine", "inactive"):
                    raise ValueError("Invalid tank status.")
                if status == "inactive" and count:
                    raise ValueError("Disperse or transfer all fish before archiving this tank.")
                if count != tank["current_count"]:
                    note = str(data.get("adjustment_note", "")).strip()
                    if not note:
                        raise ValueError("Explain the population correction before saving.")
                    delta=count-tank['current_count']; operation=uuid.uuid4().hex
                    recovered=await self._consume_census(tank_id,'estimated_mortality' if delta>0 else 'auto_incoming',abs(delta),operation,'Recovered camera count')
                    await self._movement(tank_id, "adjustment", delta-(recovered if delta>0 else -recovered), operation, note)
                if live != config['live_source'] or count != tank['current_count']:
                    await self._conn.execute('UPDATE monitoring_config SET revision=revision+1,validation=NULL,updated_at=? WHERE tank_id=?', (now_text(),tank_id))
                    await self._conn.execute('DELETE FROM accepted_census WHERE tank_id=?', (tank_id,))
                    await self._conn.execute('DELETE FROM camera_expectations WHERE tank_id=?', (tank_id,))
                await self._conn.execute('UPDATE monitoring_config SET live_source=?,video_source=? WHERE tank_id=?',(live,video,tank_id))
                await self._conn.execute("""UPDATE tanks SET name=?, camera_source=?, max_capacity=?,
                    current_count=?, avg_weight_g=?, feed_rate_pct=?, status=? WHERE tank_id=?""",
                    (name, source, capacity, count, weight, feed, status, tank_id))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return await self.get_tank(tank_id)

    async def delete_tank(self, tank_id, *, remove_stock=False, expected_count=None):
        # Retain history and permanent codes; removal is neither deaths nor dispersal.
        await self._ensure_conn()
        async with self._lock:
            try:
                await self._conn.execute('BEGIN IMMEDIATE')
                rows = await self._rows('SELECT * FROM tanks WHERE tank_id=?', (tank_id,))
                if not rows:
                    await self._conn.rollback()
                    return False
                tank = rows[0]
                if tank['status'] == 'inactive':
                    await self._conn.rollback()
                    return True
                if expected_count is not None and tank['current_count'] != expected_count:
                    raise TankPopulationChanged('Population changed. Review the latest population before deleting this tank.')
                if tank['current_count'] and (not remove_stock or expected_count is None):
                    raise ValueError('Confirm removal of the remaining stock before deleting this tank.')
                await self._movement(tank_id, 'tank_removed', -tank['current_count'], 'tank-removed:'+tank_id,
                                     'Tank removed; remaining stock excluded from active inventory')
                await self._conn.execute("UPDATE tanks SET current_count=0,status='inactive' WHERE tank_id=?", (tank_id,))
                await self._conn.execute('UPDATE monitoring_config SET enabled=0,validation=NULL,revision=revision+1,updated_at=? WHERE tank_id=?', (now_text(),tank_id))
                await self._conn.execute('DELETE FROM accepted_census WHERE tank_id=?', (tank_id,))
                await self._conn.execute('DELETE FROM camera_expectations WHERE tank_id=?', (tank_id,))
                await self._conn.execute('UPDATE monitoring_alerts SET resolved_at=? WHERE tank_id=? AND resolved_at IS NULL', (now_text(),tank_id))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return True

    async def _movement(self, tank_id, kind, delta, operation_id, note="", stamp=None):
        await self._conn.execute("""INSERT INTO inventory_movements
            (operation_id,tank_id,kind,delta,timestamp,note) VALUES(?,?,?,?,?,?)""",
            (operation_id, tank_id, kind, delta, stamp or now_text(), note))

    async def _duplicate(self, operation_id, tank_id, kind, delta):
        rows = await self._rows("SELECT * FROM inventory_movements WHERE operation_id=?", (operation_id,))
        if not rows:
            return False
        if (rows[0]["tank_id"], rows[0]["kind"], rows[0]["delta"]) != (tank_id, kind, delta):
            raise ValueError("This operation reference was already used for a different record. Refresh and try again.")
        return True

    async def stock_fish(self, tank_id, data):
        await self._ensure_conn()
        count = integer(data.get("count"), "Fish to stock", 1)
        operation = str(data.get("operation_id") or uuid.uuid4().hex)
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                await self._tank(tank_id)
                if not await self._duplicate(operation, tank_id, "stocking", count):
                    accounted = await self._consume_census(tank_id,'auto_incoming',count,operation,'Confirmed stocking',data.get('census_event_id'))
                    await self._movement(tank_id, "stocking", count, operation, str(data.get("note", "")))
                    await self._date_classification(tank_id,'stocking',operation,now_text())
                    await self._conn.execute("UPDATE tanks SET current_count=current_count+? WHERE tank_id=?", (count-accounted, tank_id))
                    await self._expect_camera(tank_id,count-accounted,operation)
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return await self.get_tank(tank_id)

    async def record_mortality(self, tank_id, data):
        await self._ensure_conn()
        count = integer(data.get("count"), "New deaths")
        operation = str(data.get("operation_id") or uuid.uuid4().hex)
        stamp = now_text()
        if data.get("date", stamp[:10]) != stamp[:10]:
            raise ValueError("Daily checks must be recorded for today. Historical data is not estimated.")
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                tank = await self._tank(tank_id)
                if not await self._duplicate(operation, tank_id, "mortality", -count):
                    accounted = await self._consume_census(tank_id,'estimated_mortality',count,operation,'Confirmed mortality',data.get('census_event_id'))
                    if count-accounted > tank["current_count"]:
                        raise ValueError("Deaths cannot exceed the saved fish population.")
                    await self._movement(tank_id, "mortality", -count, operation, str(data.get("note", "")), stamp)
                    await self._date_classification(tank_id,'mortality',operation,stamp)
                    await self._conn.execute("UPDATE tanks SET current_count=current_count-? WHERE tank_id=?", (count-accounted, tank_id))
                    await self._expect_camera(tank_id,-(count-accounted),operation)
                    await self._conn.execute("INSERT OR REPLACE INTO mortality_checks VALUES(?,?,?)", (tank_id, stamp[:10], stamp))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return await self.get_tank(tank_id)

    async def commit_dispersal(self, data):
        await self._ensure_conn()
        dispersal_id = str(data.get("dispersal_id", "")).strip()
        if not dispersal_id or len(dispersal_id) > 120:
            raise ValueError("Enter a unique dispersal reference (up to 120 characters).")
        tank_id = data.get("tank_id")
        count = integer(data.get("count"), "Fish to disperse", 1)
        mortality = integer(data.get("mortality_count", 0), "Deaths")
        kind = data.get("type", "sale")
        if kind not in ("sale", "transfer"):
            raise ValueError("Dispersal must be a sale or transfer.")
        price = number(data.get("unit_price_php", 0), "Unit price")
        price_unit = data.get("price_unit", "per_fish")
        if price_unit not in ("per_fish", "per_kg"):
            raise ValueError("Invalid price unit.")
        recipient = str(data.get("recipient", "")).strip()
        if not recipient:
            raise ValueError("Enter the buyer or recipient.")
        stamp = now_text()
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                tank = await self._tank(tank_id)
                if await self._rows("SELECT dispersal_id FROM dispersals WHERE dispersal_id=?", (dispersal_id,)):
                    raise ValueError("This dispersal reference already exists.")
                accounted = await self._consume_census(tank_id,'estimated_mortality',count,'dispersal:'+dispersal_id,'Dispersal explains loss',data.get('census_event_id'))
                death_accounted = await self._consume_census(tank_id,'estimated_mortality',mortality,'deaths:'+dispersal_id,'Confirmed mortality',data.get('mortality_event_id')) if mortality else 0
                if count + mortality - accounted - death_accounted > tank["current_count"]:
                    raise ValueError("Dispersal and deaths exceed saved population. Verify stock and any matching camera loss.")
                destination = data.get("destination_tank_id") if kind == "transfer" else None
                if destination:
                    if destination == tank_id:
                        raise ValueError("Choose a different destination tank.")
                    await self._tank(destination)
                biomass = count * tank["avg_weight_g"] / 1000
                revenue = round((count if price_unit == "per_fish" else biomass) * price, 2) if kind == "sale" else 0
                await self._conn.execute("""INSERT INTO dispersals VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (dispersal_id, tank_id, data.get("batch_code", ""), recipient, kind, price, price_unit, count, revenue, stamp, stamp))
                await self._movement(tank_id, "dispersal", -count, "dispersal:" + dispersal_id, recipient, stamp)
                await self._date_classification(tank_id,'dispersal','dispersal:'+dispersal_id,stamp)
                await self._expect_camera(tank_id,-(count-accounted),'dispersal:'+dispersal_id)
                if destination:
                    incoming_accounted = await self._consume_census(destination,'auto_incoming',count,'transfer:'+dispersal_id,'Transfer explains arrival',data.get('arrival_event_id'))
                    await self._movement(destination, "transfer_in", count, "transfer:" + dispersal_id, tank_id, stamp)
                    await self._date_classification(destination,'transfer_in','transfer:'+dispersal_id,stamp)
                    await self._conn.execute("UPDATE tanks SET current_count=current_count+? WHERE tank_id=?", (count-incoming_accounted, destination))
                    await self._expect_camera(destination,count-incoming_accounted,'transfer:'+dispersal_id)
                if mortality:
                    await self._movement(tank_id, "mortality", -mortality, "deaths:" + dispersal_id, "During dispersal", stamp)
                    await self._date_classification(tank_id,'mortality','deaths:'+dispersal_id,stamp)
                    await self._conn.execute("INSERT OR REPLACE INTO mortality_checks VALUES(?,?,?)", (tank_id, stamp[:10], stamp))
                    await self._expect_camera(tank_id,-(mortality-death_accounted),'deaths:'+dispersal_id)
                remaining = tank["current_count"] - count - mortality + accounted + death_accounted
                daily_feed = remaining * tank["avg_weight_g"] / 1000 * tank["feed_rate_pct"]
                # Legacy event rate is retained only as an event-level measure.
                event_rate = round(mortality / tank["current_count"] * 100, 2) if tank["current_count"] else 0
                await self._conn.execute("""INSERT INTO tank_production_logs
                    (tank_id,recorded_count,mortality_count,mortality_rate_pct,daily_feed_kg,total_php,dispersal_id,timestamp)
                    VALUES(?,?,?,?,?,?,?,?)""", (tank_id, count, mortality, event_rate, daily_feed, revenue, dispersal_id, stamp))
                await self._conn.execute("UPDATE tanks SET current_count=? WHERE tank_id=?", (remaining, tank_id))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return {"dispersal": {"dispersal_id": dispersal_id, "count": count, "total_revenue_php": revenue},
                "updated_tank": await self.get_tank(tank_id)}

    @consistent_read
    async def get_food(self):
        await self._ensure_conn()
        items = await self._rows("SELECT * FROM feed_items ORDER BY name")
        logs = await self._rows("""SELECT f.*,i.name AS item_name,t.name AS tank_name
            FROM feed_transactions f JOIN feed_items i ON i.id=f.item_id
            LEFT JOIN tanks t ON t.tank_id=f.tank_id ORDER BY timestamp DESC,id DESC LIMIT 200""")
        tanks = await self.get_tanks()
        return {"items": items, "records": logs, "schedule": [t for t in tanks if t["status"] != "inactive"]}

    async def create_feed_item(self, data):
        await self._ensure_conn()
        name = str(data.get("name", "")).strip()
        if not name or len(name) > 120:
            raise ValueError("Enter a feed name (up to 120 characters).")
        warning = number(data.get("low_stock_kg", 5), "Low-stock warning")
        item_id = uuid.uuid4().hex
        async with self._lock:
            await self._conn.execute("INSERT INTO feed_items VALUES(?,?,0,?)", (item_id, name, warning))
            await self._conn.commit()
        return {"id": item_id, "name": name, "stock_kg": 0, "low_stock_kg": warning}

    async def feed_transaction(self, data):
        await self._ensure_conn()
        qty = number(data.get("quantity_kg"), "Feed quantity", .001)
        cost = number(data.get("cost_php", 0), "Purchase cost")
        kind = data.get("kind")
        if kind not in ("purchase", "feeding"):
            raise ValueError("Choose purchase or feeding.")
        item_id = data.get("item_id")
        operation = str(data.get("operation_id") or uuid.uuid4().hex)
        tank_id = data.get("tank_id") if kind == "feeding" else None
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                existing = await self._rows("SELECT * FROM feed_transactions WHERE id=?", (operation,))
                if existing:
                    old = existing[0]
                    if (old["item_id"], old["tank_id"], old["kind"], old["quantity_kg"]) != (item_id, tank_id, kind, qty):
                        raise ValueError("This operation reference was already used for a different feeding record.")
                    await self._conn.commit()
                    return {"status": "already_recorded"}
                items = await self._rows("SELECT * FROM feed_items WHERE id=?", (item_id,))
                if not items:
                    raise ValueError("Feed item not found.")
                if kind == "feeding":
                    await self._tank(tank_id)
                    if qty > items[0]["stock_kg"] + 1e-9:
                        raise ValueError("Not enough feed stock. Record a purchase first.")
                await self._conn.execute("INSERT INTO feed_transactions VALUES(?,?,?,?,?,?,?,?)",
                    (operation, item_id, tank_id, kind, qty, cost if kind == "purchase" else 0, now_text(), str(data.get("note", ""))))
                await self._conn.execute("UPDATE feed_items SET stock_kg=ROUND(stock_kg+?,6) WHERE id=?",
                    (qty if kind == "purchase" else -qty, item_id))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return {"status": "recorded"}

    @consistent_read
    async def get_tank_mortality_analytics(self, tank_id=None, days=14, severity=None):
        await self._ensure_conn()
        days = integer(days, "Days", 1)
        if days > 366:
            raise ValueError("Choose a period of at most 366 days.")
        if severity not in (None, "all", "normal", "elevated", "critical", "unrecorded"):
            raise ValueError("Invalid mortality filter.")
        tanks = await self.get_tanks()
        if tank_id and tank_id != "all":
            tanks = [t for t in tanks if t["tank_id"] == tank_id]
            if not tanks:
                raise ValueError("Tank not found.")
        today = datetime.now(MANILA).date()
        dates = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
        baselines = {r["tank_id"]: r for r in await self._rows("SELECT * FROM inventory_baselines")}
        movements = await self._rows("SELECT * FROM inventory_movements ORDER BY timestamp,id")
        removed = {r['tank_id']:r['timestamp'][:10] for r in movements if r['kind']=='tank_removed'}
        by_day = {}
        for row in movements:
            by_day.setdefault((row["tank_id"], row["timestamp"][:10]), []).append(row)
        checks = {(r["tank_id"], r["date"]) for r in await self._rows("SELECT * FROM mortality_checks")}
        coverage = {(r['tank_id'],r['date']):r for r in await self._rows('SELECT * FROM monitoring_daily')}
        records, series, summaries = [], {}, []
        colors = ["#087f8c", "#188657", "#a86e0a", "#775bc3", "#b44f78"]
        for index, tank in enumerate(tanks):
            tid = tank["tank_id"]
            base = baselines.get(tid)
            population = base["population"] if base else None
            start = min(dates[0], base["date"]) if base else dates[0]
            cursor_day = datetime.strptime(start, "%Y-%m-%d").date()
            tank_records = []
            while cursor_day <= today:
                day = cursor_day.isoformat()
                events = by_day.get((tid, day), [])
                confirmed = -sum(r['delta'] for r in events if r['kind'] in ('mortality','legacy_mortality'))
                estimated = -sum(r['delta'] for r in events if r['kind'] in ('estimated_mortality','estimated_mortality_reversal'))
                if not base or day < base["date"]:
                    opening, closing, denominator = None, None, None
                    additions = 0
                    deaths = confirmed + estimated
                else:
                    opening = population
                    additions = sum(r["delta"] for r in events if r["kind"] in ("stocking", "transfer_in",'auto_incoming','auto_incoming_reversal'))
                    deaths = confirmed + estimated
                    closing = opening + sum(r["delta"] for r in events if r["kind"] != "legacy_mortality")
                    known = (day > base["date"] or base["known"]) and not any(r["kind"] == "adjustment" for r in events)
                    denominator = opening + additions if known else None
                    population = closing
                monitored = coverage.get((tid,day),{}).get('valid_seconds',0) >= 60 or any(r['kind'] in ('estimated_mortality','estimated_mortality_reversal','auto_incoming','auto_incoming_reversal') for r in events)
                checked = (tid, day) in checks or confirmed > 0 or estimated > 0 or monitored or any(r["kind"] == "legacy_mortality" for r in events)
                rate = round(deaths / denominator * 100, 2) if checked and denominator else None
                sev = "unrecorded" if rate is None else "normal" if rate < 1 else "elevated" if rate <= 3 else "critical"
                if day in dates:
                    tank_records.append({"date": day, "date_label": cursor_day.strftime("%b %d"), "tank_id": tid,
                        "tank_name": tank["name"], "population": closing, "opening_population": opening,
                        "additions": additions, "denominator": denominator, "mortality_count": deaths if checked else None,
                        'confirmed_mortality_count':confirmed,'estimated_mortality_count':estimated,
                        'valid_monitoring_seconds':coverage.get((tid,day),{}).get('valid_seconds',0),
                        'observed_monitoring_seconds':coverage.get((tid,day),{}).get('observed_seconds',0),
                        "eligible": (bool(base and day >= base["date"]) or checked) and day <= removed.get(tid, day),
                        "mortality_rate_pct": rate, "checked": checked, "severity": sev,
                        "status_label": "Not recorded" if not checked else "N/A" if rate is None else 'Includes estimates' if estimated else 'Monitored (partial day)' if monitored and (tid,day) not in checks else "Recorded" if sev == "normal" else sev.title(),
                        "dispersal_id": "DAILY-CHECK" if checked else None})
                cursor_day += timedelta(days=1)
            filtered = [r for r in tank_records if severity in (None, "all") or r["severity"] == severity]
            records.extend(filtered)
            series[tid] = {"tank_id": tid, "tank_name": tank["name"], "color": colors[index % len(colors)],
                "populations": [r["population"] for r in tank_records],
                "mortality_counts": [r["mortality_count"] for r in tank_records],
                "mortality_rates": [r["mortality_rate_pct"] for r in tank_records]}
            valid = [r for r in filtered if r["mortality_rate_pct"] is not None]
            den = sum(r["denominator"] for r in valid)
            avg = round(sum(r["mortality_count"] for r in valid) / den * 100, 2) if den else None
            summaries.append({"tank_id": tid, "tank_name": tank["name"], "current_population": tank["current_count"],
                "total_mortalities": sum(r["mortality_count"] or 0 for r in filtered), "avg_mortality_rate_pct": avg,
                "latest_rate_pct": tank_records[-1]["mortality_rate_pct"], "status": tank_records[-1]["status_label"]})
        records.sort(key=lambda r: (r["date"], r["tank_id"]), reverse=True)
        valid = [r for r in records if r["mortality_rate_pct"] is not None]
        denominator = sum(r["denominator"] for r in valid)
        rate = round(sum(r["mortality_count"] for r in valid) / denominator * 100, 2) if denominator else None
        peak = max(valid, key=lambda r: r["mortality_rate_pct"]) if valid else None
        return {"filters": {"tank_id": tank_id or "all", "days": days, "severity": severity or "all"},
            "timeline": [datetime.strptime(d, "%Y-%m-%d").strftime("%b %d") for d in dates], "timeline_dates": dates,
            "tank_series": series, "records": records, "tanks": tanks,
            "summary": {"total_population": sum(t["current_count"] for t in tanks),
                'estimated_mortality_count':sum(r['estimated_mortality_count'] for r in records),
                'confirmed_mortality_count':sum(r['confirmed_mortality_count'] for r in records),
                "total_mortalities": sum(r["mortality_count"] or 0 for r in records), "avg_mortality_rate_pct": rate,
                "denominator": denominator, "recorded_days": sum(r["checked"] for r in records),
                "expected_days": sum(r["eligible"] for r in records), "peak_rate_pct": peak["mortality_rate_pct"] if peak else None,
                "peak_date": peak["date"] if peak else "N/A", "peak_tank": peak["tank_name"] if peak else "N/A",
                "overall_status": "Not recorded" if rate is None else "Recorded", "tanks_summary": summaries}}

    @consistent_read
    async def get_production_report(self, include_all_dispersals=False):
        await self._ensure_conn()
        tanks = await self.get_tanks()
        active = [t for t in tanks if t["status"] != "inactive"]
        dispersals = await self.get_dispersals(limit=-1 if include_all_dispersals else 1000)
        mortality = await self.get_tank_mortality_analytics(days=7)
        daily = await self.get_tank_mortality_analytics(days=1)
        food = await self.get_food()
        rate = daily["summary"]["avg_mortality_rate_pct"]
        totals = (await self._rows("""SELECT COALESCE(SUM(CASE WHEN type='sale' THEN total_revenue_php ELSE 0 END),0) AS revenue,
            COALESCE(SUM(count),0) AS fish FROM dispersals"""))[0]
        active_ids = {t["tank_id"] for t in active}
        summary = {"total_population": sum(t["current_count"] for t in active),
            "total_biomass_kg": round(sum(t["biomass_kg"] for t in active), 3),
            "total_feed_kg": round(sum(t["daily_feed_kg"] for t in active), 3),
            "inventory_php_value": round(sum(t["valuation_php"] for t in active), 2),
            "dispersal_earnings": round(totals["revenue"], 2),
            "total_dispersed_count": totals["fish"], "mortality_rate_pct": rate,
            "mortality_count": daily["summary"]["total_mortalities"], "mortality_denominator": daily["summary"]["denominator"],
            'estimated_mortality_count':daily['summary']['estimated_mortality_count'],
            'confirmed_mortality_count':daily['summary']['confirmed_mortality_count'],
            "mortality_checked_tanks": sum(r["checked"] for r in daily["records"] if r["tank_id"] in active_ids), "mortality_expected_tanks": len(active),
            "report_date": now_text()[:10], "feed_stock_kg": round(sum(i["stock_kg"] for i in food["items"]), 3),
            "low_feed_items": sum(i["stock_kg"] <= i["low_stock_kg"] for i in food["items"])}
        trends = []
        for day in mortality["timeline_dates"]:
            rows = [r for r in mortality["records"] if r["date"] == day]
            populations = [r["population"] for r in rows]
            trends.append({"date": day, "label": datetime.strptime(day, "%Y-%m-%d").strftime("%b %d"),
                "population": sum(populations) if populations and all(p is not None for p in populations) else None,
                'estimated_mortality':sum(r['estimated_mortality_count'] for r in rows) if any(r['checked'] for r in rows) else None,
                'confirmed_mortality':sum(r['confirmed_mortality_count'] for r in rows) if any(r['checked'] for r in rows) else None,
                "mortality": sum(r["mortality_count"] or 0 for r in rows) if any(r["checked"] for r in rows) else None})
        return {**summary, "summary": summary, "tanks": tanks, "population_per_tank": tanks,
            "feed_schedule": active, "dispersals": dispersals, "dispersal_ledger": dispersals,
            "production_logs": await self.get_tank_production_logs(limit=1000), "daily_trends": trends,
            "valuation_breakdown": [{"tank_id": t["tank_id"], "label": t["name"], "valuation_php": t["valuation_php"]} for t in tanks]}
