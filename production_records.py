"""Transactional census reconciliation and durable monitoring records.

Reclassification uses compensating movements at the original observation date.
Original evidence is never deleted; current stock and historical totals reconcile.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone


def stamp_now():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')


class ProductionRecords:
    async def init_production_records(self):
        columns = await self._rows('PRAGMA table_info(monitoring_config)')
        migrate_sources = not any(c['name']=='video_source' for c in columns)
        await self._conn.executescript('''
            CREATE TABLE IF NOT EXISTS monitoring_config (
                tank_id TEXT PRIMARY KEY REFERENCES tanks(tank_id),
                enabled INTEGER NOT NULL DEFAULT 1, live_source TEXT NOT NULL DEFAULT '',
                revision INTEGER NOT NULL DEFAULT 1, validation TEXT,
                video_source TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS census_events (
                id TEXT PRIMARY KEY, tank_id TEXT NOT NULL REFERENCES tanks(tank_id),
                kind TEXT NOT NULL, count INTEGER NOT NULL, remaining INTEGER NOT NULL,
                timestamp TEXT NOT NULL, revision INTEGER NOT NULL, observation TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS census_allocations (
                id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES census_events(id),
                operation_id TEXT NOT NULL, count INTEGER NOT NULL, reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS camera_expectations (
                operation_id TEXT PRIMARY KEY, tank_id TEXT NOT NULL REFERENCES tanks(tank_id),
                remaining INTEGER NOT NULL, timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS accepted_census (
                tank_id TEXT PRIMARY KEY REFERENCES tanks(tank_id), revision INTEGER NOT NULL,
                population INTEGER NOT NULL, observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_census_remaining ON census_events(tank_id,kind,remaining,timestamp);
            CREATE INDEX IF NOT EXISTS idx_expected_camera ON camera_expectations(tank_id,timestamp);
            CREATE TABLE IF NOT EXISTS monitoring_daily (
                tank_id TEXT NOT NULL REFERENCES tanks(tank_id), date TEXT NOT NULL,
                valid_seconds REAL NOT NULL DEFAULT 0, observed_seconds REAL NOT NULL DEFAULT 0,
                last_observation TEXT, PRIMARY KEY(tank_id,date)
            );
            CREATE TABLE IF NOT EXISTS monitoring_alerts (
                id TEXT PRIMARY KEY, tank_id TEXT, alert_key TEXT NOT NULL,
                message TEXT NOT NULL, created_at TEXT NOT NULL, resolved_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_active_monitor_alert
                ON monitoring_alerts(alert_key) WHERE resolved_at IS NULL;
        ''')
        if columns and migrate_sources:
            await self._conn.execute("ALTER TABLE monitoring_config ADD COLUMN video_source TEXT NOT NULL DEFAULT ''")
        await self._conn.execute('''INSERT OR IGNORE INTO monitoring_config
            (tank_id,live_source,updated_at) SELECT tank_id,
            CASE WHEN camera_source GLOB '[0-9]*' AND camera_source NOT GLOB '*[^0-9]*'
                OR camera_source LIKE 'rtsp://%' OR camera_source LIKE 'rtsps://%'
                THEN camera_source ELSE '' END, ? FROM tanks''', (stamp_now(),))
        if migrate_sources:
            await self._conn.execute('''UPDATE monitoring_config SET video_source=(SELECT camera_source FROM tanks
                WHERE tanks.tank_id=monitoring_config.tank_id) WHERE EXISTS(SELECT 1 FROM tanks
                WHERE tanks.tank_id=monitoring_config.tank_id AND camera_source!=''
                AND NOT (camera_source GLOB '[0-9]*' AND camera_source NOT GLOB '*[^0-9]*')
                AND camera_source NOT LIKE 'rtsp://%' AND camera_source NOT LIKE 'rtsps://%')''')
            await self._conn.execute('''UPDATE tanks SET camera_source=(SELECT live_source FROM monitoring_config
                WHERE monitoring_config.tank_id=tanks.tank_id) WHERE EXISTS(SELECT 1 FROM monitoring_config
                WHERE monitoring_config.tank_id=tanks.tank_id AND live_source!='')
                AND (camera_source='' OR camera_source GLOB '[0-9]*' AND camera_source NOT GLOB '*[^0-9]*'
                    OR camera_source LIKE 'rtsp://%' OR camera_source LIKE 'rtsps://%')''')

    async def monitoring_configs(self):
        await self._ensure_conn()
        async with self._lock:
            return await self._rows('''SELECT c.*, t.name,t.status,t.current_count,t.camera_source,
                a.population AS accepted_count,a.observed_at AS accepted_at,a.revision AS accepted_revision
                FROM monitoring_config c JOIN tanks t USING(tank_id) LEFT JOIN accepted_census a USING(tank_id)''')

    async def configure_monitoring(self, tank_id, *, enabled=None, live_source=None, validation=None,
                                   invalidate=False, expected_revision=None):
        await self._ensure_conn()
        async with self._lock:
            try:
                await self._conn.execute('BEGIN IMMEDIATE')
                await self._tank(tank_id)
                await self._conn.execute('INSERT OR IGNORE INTO monitoring_config(tank_id,updated_at) VALUES(?,?)',
                                         (tank_id, stamp_now()))
                old = (await self._rows('SELECT * FROM monitoring_config WHERE tank_id=?', (tank_id,)))[0]
                if expected_revision is not None and old['revision'] != expected_revision:
                    raise ValueError('Camera settings changed. Validate the new view again.')
                changed = live_source is not None and live_source != old['live_source']
                if changed or invalidate:
                    await self._conn.execute('DELETE FROM accepted_census WHERE tank_id=?', (tank_id,))
                    await self._conn.execute('DELETE FROM camera_expectations WHERE tank_id=?', (tank_id,))
                if changed:
                    await self._conn.execute('''UPDATE tanks SET camera_source=? WHERE tank_id=?
                        AND (camera_source='' OR camera_source GLOB '[0-9]*' AND camera_source NOT GLOB '*[^0-9]*'
                             OR camera_source LIKE 'rtsp://%' OR camera_source LIKE 'rtsps://%')''',(live_source,tank_id))
                await self._conn.execute('''UPDATE monitoring_config SET enabled=?,live_source=?,revision=?,
                    validation=?,updated_at=? WHERE tank_id=?''',
                    (old['enabled'] if enabled is None else int(enabled),
                     old['live_source'] if live_source is None else live_source,
                     old['revision'] + int(changed or invalidate),
                     None if changed or invalidate else json.dumps(validation) if validation is not None else old['validation'],
                     stamp_now(), tank_id))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise

    async def _consume_census(self, tank_id, kind, count, operation, reason, event_id=None):
        """Called inside the inventory transaction; returns already-accounted quantity."""
        if not count:
            return 0
        cutoff = (datetime.strptime(stamp_now(), '%Y-%m-%d %H:%M:%S') - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        rows = await self._rows('''SELECT * FROM census_events WHERE tank_id=? AND kind=? AND remaining>0
            AND (id=? OR (? IS NULL AND timestamp>=?)) ORDER BY timestamp,id''',
            (tank_id, kind, event_id, event_id, cutoff))
        if event_id and not rows:
            raise ValueError('That camera change is unavailable or belongs to another tank.')
        if not event_id and reason not in ('Recovered camera count', 'Dispersal explains loss') and len(rows)>1 and sum(r['remaining'] for r in rows)>count:
            raise ValueError('Several camera changes could explain this activity. Select the matching camera event.')
        # Multiple events are chronological parts of the same count reconciliation.
        used = 0
        for event in rows:
            qty = min(count - used, event['remaining'])
            if qty <= 0:
                break
            allocation = uuid.uuid4().hex
            reversal = qty if kind == 'estimated_mortality' else -qty
            await self._movement(tank_id, kind + '_reversal', reversal, allocation,
                                 reason + ': ' + operation, event['timestamp'])
            await self._conn.execute('INSERT INTO census_allocations VALUES(?,?,?,?,?,?)',
                                     (allocation, event['id'], operation, qty, reason, stamp_now()))
            await self._conn.execute('UPDATE census_events SET remaining=remaining-? WHERE id=?', (qty, event['id']))
            used += qty
        if kind=='estimated_mortality' and used and not await self._rows("SELECT id FROM census_events WHERE tank_id=? AND kind='estimated_mortality' AND remaining>0 LIMIT 1",(tank_id,)):
            await self._conn.execute("UPDATE monitoring_alerts SET resolved_at=? WHERE alert_key=? AND resolved_at IS NULL",(stamp_now(),'loss:'+tank_id))
        return used

    async def _expect_camera(self, tank_id, delta, operation):
        config = await self._rows('SELECT validation FROM monitoring_config WHERE tank_id=?', (tank_id,))
        if delta and config and config[0]['validation']:
            await self._conn.execute('INSERT INTO camera_expectations VALUES(?,?,?,?)',
                                     (operation,tank_id,delta,stamp_now()))

    async def _date_classification(self, tank_id, kind, operation, stamp):
        """Move the classified portion to its evidence date without changing net stock."""
        rows = await self._rows('''SELECT a.count,e.timestamp FROM census_allocations a
            JOIN census_events e ON e.id=a.event_id WHERE a.operation_id=?''', (operation,))
        sign = 1 if kind in ('stocking','transfer_in') else -1
        for row in rows:
            if row['timestamp'][:10] != stamp[:10]:
                await self._movement(tank_id,kind,-sign*row['count'],uuid.uuid4().hex,'Reclassification date',stamp)
                await self._movement(tank_id,kind,sign*row['count'],uuid.uuid4().hex,'Original camera evidence date',row['timestamp'])

    async def _explain_camera(self, tank_id, delta):
        # Offset opposing recorded movements. A census observes net tank population,
        # not individual fish identities or the order of physical transfers.
        all_rows = await self._rows('SELECT * FROM camera_expectations WHERE tank_id=? AND remaining!=0 ORDER BY timestamp,operation_id',(tank_id,))
        positive=[dict(r) for r in all_rows if r['remaining']>0]
        negative=[dict(r) for r in all_rows if r['remaining']<0]
        for incoming in positive:
            for outgoing in negative:
                qty=min(incoming['remaining'],-outgoing['remaining'])
                if qty:
                    incoming['remaining']-=qty; outgoing['remaining']+=qty
                    await self._conn.execute('UPDATE camera_expectations SET remaining=? WHERE operation_id=?',(incoming['remaining'],incoming['operation_id']))
                    await self._conn.execute('UPDATE camera_expectations SET remaining=? WHERE operation_id=?',(outgoing['remaining'],outgoing['operation_id']))
        rows = await self._rows('SELECT * FROM camera_expectations WHERE tank_id=? AND remaining*?>0 ORDER BY timestamp,operation_id', (tank_id,delta))
        explained = 0
        for row in rows:
            qty = min(abs(delta-explained),abs(row['remaining'])) * (1 if delta>0 else -1)
            await self._conn.execute('UPDATE camera_expectations SET remaining=remaining-? WHERE operation_id=?', (qty,row['operation_id']))
            explained += qty
            if explained == delta:
                break
        return explained

    async def reconcile_census(self, tank_id, observed, operation_id, revision, previous_observed, since,
                               evidence=None):
        """Exactly-once population delta relative to the last accepted camera census.

        Pending human movements explain matching camera changes in either order.
        Positive changes first undo eligible losses; decreases first undo increases.
        """
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError('Census must be a nonnegative whole number.')
        await self._ensure_conn()
        async with self._lock:
            try:
                await self._conn.execute('BEGIN IMMEDIATE')
                tank = await self._tank(tank_id)
                config = await self._rows('SELECT * FROM monitoring_config WHERE tank_id=?', (tank_id,))
                if not config or config[0]['revision'] != revision or not config[0]['validation'] or not config[0]['enabled']:
                    raise ValueError('Automatic counting requires the current validated camera.')
                duplicate=await self._rows('SELECT * FROM census_events WHERE id=?', (operation_id,))
                if duplicate:
                    old=duplicate[0]; observation=json.loads(old['observation'])
                    if old['tank_id']!=tank_id or old['revision']!=revision or observation['observed']!=observed or observation['previous']!=previous_observed:
                        raise ValueError('This census reference already belongs to a different observation.')
                    await self._conn.commit()
                    return tank
                cutoff=(datetime.strptime(stamp_now(),'%Y-%m-%d %H:%M:%S')-timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
                if await self._rows('SELECT operation_id FROM camera_expectations WHERE tank_id=? AND remaining!=0 AND timestamp<?',(tank_id,cutoff)):
                    raise ValueError('An expected fish movement is overdue. Verify saved population and revalidate the census.')
                accepted = await self._rows('SELECT * FROM accepted_census WHERE tank_id=?', (tank_id,))
                if accepted and (accepted[0]['revision'] != revision or accepted[0]['population'] != previous_observed):
                    raise ValueError('Another census updated this tank. Reload the accepted count.')
                camera_delta = observed - previous_observed
                explained = await self._explain_camera(tank_id, camera_delta) if camera_delta else 0
                delta = camera_delta - explained
                # Persist acceptance even for zero net change, so crash/retry cannot replay it.
                kind = 'auto_incoming' if delta > 0 else 'estimated_mortality' if delta < 0 else 'census'
                qty = abs(delta)
                recovered = await self._consume_census(tank_id,
                    'estimated_mortality' if delta > 0 else 'auto_incoming', qty,
                    operation_id, 'Recovered camera count') if delta else 0
                if tank['current_count'] + delta < 0:
                    raise ValueError('Camera change conflicts with saved movements. Review the pending activity.')
                remaining = qty - recovered
                await self._conn.execute('INSERT INTO census_events VALUES(?,?,?,?,?,?,?,?)',
                    (operation_id, tank_id, kind, qty, remaining, stamp_now(), revision,
                     json.dumps({'observed':observed, 'previous':previous_observed, 'explained':explained, **(evidence or {})})))
                if remaining:
                    await self._movement(tank_id, kind, remaining if delta > 0 else -remaining,
                                         operation_id, 'Reliable whole-tank camera census',stamp_now())
                await self._conn.execute('UPDATE tanks SET current_count=current_count+? WHERE tank_id=?', (delta, tank_id))
                await self._conn.execute('INSERT OR REPLACE INTO accepted_census VALUES(?,?,?,?)', (tank_id,revision,observed,stamp_now()))
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
        return await self.get_tank(tank_id)

    async def census_history(self, tank_id=None, unresolved=False,limit=200,offset=0):
        await self._ensure_conn()
        async with self._lock:
            return await self._rows('''SELECT id,tank_id,kind,count,remaining,timestamp,observation
                FROM census_events WHERE (? IS NULL OR tank_id=?) AND (?=0 OR remaining>0)
                ORDER BY timestamp DESC,id DESC LIMIT ? OFFSET ?''', (tank_id,tank_id,int(unresolved),limit,offset))

    async def monitoring_coverage(self, tank_id, valid, seconds):
        async with self._lock:
            stamp = stamp_now()
            await self._conn.execute('''INSERT INTO monitoring_daily VALUES(?,?,?,?,?)
                ON CONFLICT(tank_id,date) DO UPDATE SET valid_seconds=valid_seconds+excluded.valid_seconds,
                observed_seconds=observed_seconds+excluded.observed_seconds,last_observation=excluded.last_observation''',
                (tank_id, stamp[:10], seconds if valid else 0, seconds, stamp))
            await self._conn.commit()

    async def monitoring_alert(self, key, message=None, tank_id=None):
        async with self._lock:
            if message:
                await self._conn.execute('''INSERT OR IGNORE INTO monitoring_alerts
                    (id,tank_id,alert_key,message,created_at) VALUES(?,?,?,?,?)''',
                    (uuid.uuid4().hex,tank_id,key,message,stamp_now()))
                await self._conn.execute('UPDATE monitoring_alerts SET message=? WHERE alert_key=? AND resolved_at IS NULL', (message,key))
            else:
                await self._conn.execute('UPDATE monitoring_alerts SET resolved_at=? WHERE alert_key=? AND resolved_at IS NULL', (stamp_now(),key))
            await self._conn.commit()

    async def monitoring_report(self):
        await self._ensure_conn()
        async with self._lock:
            return {'alerts':await self._rows('SELECT * FROM monitoring_alerts ORDER BY resolved_at IS NULL DESC,created_at DESC LIMIT 100'),
                    'coverage':await self._rows('SELECT * FROM monitoring_daily WHERE date=?', (stamp_now()[:10],))}

    async def prune_monitoring(self):
        cutoff = (datetime.strptime(stamp_now(), '%Y-%m-%d %H:%M:%S') - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        async with self._lock:
            # Only live telemetry expires. Inventory, audits, evaluations, daily summaries remain.
            await self._conn.execute('DELETE FROM bounding_boxes WHERE event_id IN (SELECT id FROM detection_events WHERE timestamp<? AND source IN (\'tank_camera\',\'tank_video\',\'production_camera\'))', (cutoff,))
            await self._conn.execute('DELETE FROM detection_events WHERE timestamp<? AND source IN (\'tank_camera\',\'tank_video\',\'production_camera\')', (cutoff,))
            await self._conn.commit()
