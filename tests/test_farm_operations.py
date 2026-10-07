import asyncio
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from farm_database import FarmDatabase, now_text
from calibrate import coco_split


class FarmOperationsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = FarmDatabase(':memory:')
        await self.db.connect()

    async def asyncTearDown(self):
        await self.db.close()

    async def tank(self, count=1000, code='TANK-01'):
        return await self.db.create_tank({'tank_id': code, 'name': code, 'current_count': count})

    async def test_empty_database_has_no_invented_tanks_losses_or_rates(self):
        report = await self.db.get_production_report()
        mortality = await self.db.get_tank_mortality_analytics(days=7)
        self.assertEqual(report['tanks'], [])
        self.assertEqual(mortality['summary']['total_mortalities'], 0)
        self.assertIsNone(report['mortality_rate_pct'])
        self.assertTrue(all(t['mortality'] is None and t['population'] is None for t in report['daily_trends']))

    async def test_new_empty_farm_remains_empty_after_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'farm.db'
            farm = FarmDatabase(path)
            await farm.connect()
            await farm.close()
            await farm.connect()
            self.assertEqual(await farm.get_tanks(), [])
            await farm.close()

    async def test_legacy_losses_survive_without_invented_denominators(self):
        from database import Database
        from datetime import datetime
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.db'
            old = Database(path)
            await old.connect()
            day = (datetime.strptime(now_text()[:10], '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            await old._conn.execute("UPDATE tanks SET current_count=900 WHERE tank_id='TANK-01'")
            await old._conn.execute("""INSERT INTO dispersals VALUES('OLD','TANK-01','','Buyer','sale',1,'per_fish',90,90,?,?)""", (day,day))
            await old._conn.execute("""INSERT INTO tank_production_logs(tank_id,recorded_count,mortality_count,mortality_rate_pct,daily_feed_kg,total_php,dispersal_id,timestamp)
                VALUES('TANK-01',90,10,1,0,90,'OLD',?)""", (day+' 10:00:00',))
            await old._conn.commit()
            await old.close()
            farm = FarmDatabase(path,seed=True)
            await farm.connect()
            result=await farm.get_tank_mortality_analytics(tank_id='TANK-01',days=2)
            historical=next(r for r in result['records'] if r['date']==day)
            self.assertEqual(historical['mortality_count'],10)
            self.assertIsNone(historical['mortality_rate_pct'])
            self.assertEqual((await farm.get_tank('TANK-01'))['current_count'],900)
            await farm.close()
            await farm.connect()
            self.assertEqual((await farm.get_tank_mortality_analytics(days=2))['summary']['total_mortalities'],10)
            await farm.close()

    async def test_recorded_deaths_use_opening_and_incoming_stock(self):
        await self.tank()
        await self.db.record_mortality('TANK-01', {'count': 10})
        report = await self.db.get_production_report()
        self.assertEqual(report['mortality_rate_pct'], 1)
        self.assertEqual(report['total_population'], 990)
        await self.db.stock_fish('TANK-01', {'count': 100})
        report = await self.db.get_production_report()
        self.assertEqual(report['mortality_rate_pct'], .91)
        self.assertEqual(report['mortality_denominator'], 1100)

    async def test_farm_rate_is_weighted_not_average_of_tank_rates(self):
        await self.tank(1000, 'A')
        await self.tank(100, 'B')
        await self.db.record_mortality('A', {'count': 10})
        await self.db.record_mortality('B', {'count': 10})
        self.assertEqual((await self.db.get_production_report())['mortality_rate_pct'], 1.82)

    async def test_zero_check_missing_check_and_empty_tank_are_distinct(self):
        await self.tank(100, 'A')
        await self.tank(100, 'B')
        await self.tank(0, 'C')
        await self.db.record_mortality('A', {'count': 0})
        await self.db.record_mortality('C', {'count': 0})
        data = await self.db.get_tank_mortality_analytics(days=1)
        records = {r['tank_id']: r for r in data['records']}
        self.assertEqual(records['A']['mortality_rate_pct'], 0)
        self.assertIsNone(records['B']['mortality_count'])
        self.assertFalse(records['B']['checked'])
        self.assertIsNone(records['C']['mortality_rate_pct'])
        self.assertTrue(records['C']['checked'])

    async def test_dispersals_are_not_deaths_and_excess_stock_rolls_back(self):
        await self.tank()
        await self.db.commit_dispersal({'tank_id': 'TANK-01', 'dispersal_id': 'S1', 'count': 100, 'mortality_count': 10, 'recipient': 'Buyer'})
        await self.db.commit_dispersal({'tank_id': 'TANK-01', 'dispersal_id': 'S2', 'count': 100, 'recipient': 'Buyer'})
        data = await self.db.get_production_report()
        self.assertEqual(data['total_population'], 790)
        self.assertEqual(data['mortality_count'], 10)
        self.assertEqual(data['mortality_rate_pct'], 1)
        with self.assertRaises(ValueError):
            await self.db.commit_dispersal({'tank_id': 'TANK-01', 'dispersal_id': 'S3', 'count': 791, 'recipient': 'Buyer'})
        self.assertEqual((await self.db.get_tank('TANK-01'))['current_count'], 790)
        self.assertEqual(len(await self.db.get_dispersals()), 2)

    async def test_internal_transfer_updates_both_tanks_without_mortality(self):
        await self.tank(1000, 'A')
        await self.tank(100, 'B')
        await self.db.commit_dispersal({'tank_id': 'A', 'destination_tank_id': 'B', 'type': 'transfer', 'count': 100, 'dispersal_id': 'T1', 'recipient': 'B'})
        self.assertEqual((await self.db.get_tank('A'))['current_count'], 900)
        self.assertEqual((await self.db.get_tank('B'))['current_count'], 200)
        self.assertEqual((await self.db.get_production_report())['mortality_count'], 0)

    async def test_feed_stock_transactions_are_atomic_and_idempotent(self):
        await self.tank()
        item = await self.db.create_feed_item({'name': 'Starter pellets'})
        await self.db.feed_transaction({'kind': 'purchase', 'item_id': item['id'], 'quantity_kg': 10})
        feeding = {'kind': 'feeding', 'item_id': item['id'], 'tank_id': 'TANK-01', 'quantity_kg': 3, 'operation_id': 'one-feeding'}
        await self.db.feed_transaction(feeding)
        await self.db.feed_transaction(feeding)
        with self.assertRaises(ValueError):
            await self.db.feed_transaction({**feeding, 'quantity_kg': 8, 'operation_id': 'too-much'})
        food = await self.db.get_food()
        self.assertEqual(food['items'][0]['stock_kg'], 7)
        self.assertEqual(len(food['records']), 2)

    async def test_duplicate_mortality_request_deducts_once(self):
        await self.tank()
        data = {'count': 10, 'operation_id': 'same-check'}
        await asyncio.gather(self.db.record_mortality('TANK-01', data), self.db.record_mortality('TANK-01', data))
        self.assertEqual((await self.db.get_tank('TANK-01'))['current_count'], 990)

    async def test_inventory_correction_marks_daily_denominator_unknown(self):
        await self.tank()
        await self.db.record_mortality('TANK-01', {'count': 10})
        with self.assertRaises(ValueError):
            await self.db.update_tank('TANK-01', {'current_count': 900})
        await self.db.update_tank('TANK-01', {'current_count': 900, 'adjustment_note': 'Confirmed recount'})
        self.assertIsNone((await self.db.get_production_report())['mortality_rate_pct'])
        self.assertEqual((await self.db.get_production_report())['mortality_count'], 10)

    async def test_validation_rejects_fractional_negative_and_nonfinite_inputs(self):
        await self.tank()
        for value in (-1, 1.5, 'bad', float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                await self.db.record_mortality('TANK-01', {'count': value})
        with self.assertRaises(ValueError):
            await self.db.record_mortality('TANK-01', {'count': 1001})
        self.assertEqual((await self.db.get_tank('TANK-01'))['current_count'], 1000)

    async def test_filters_limit_records_and_weighted_rate(self):
        await self.tank(1000, 'A')
        await self.tank(100, 'B')
        await self.db.record_mortality('A', {'count': 10})
        await self.db.record_mortality('B', {'count': 5})
        data = await self.db.get_tank_mortality_analytics(tank_id='B', days=1, severity='critical')
        self.assertEqual(len(data['records']), 1)
        self.assertEqual(data['summary']['avg_mortality_rate_pct'], 5)
        with self.assertRaises(ValueError):
            await self.db.get_tank_mortality_analytics(tank_id='unknown')

    async def test_historical_opening_uses_ledger_and_not_current_count(self):
        await self.tank()
        # Move the initial stock into yesterday to exercise real date progression.
        from datetime import datetime
        yesterday = (datetime.strptime(now_text()[:10], '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        await self.db._conn.execute('UPDATE inventory_baselines SET date=?', (yesterday,))
        await self.db._conn.execute("UPDATE inventory_movements SET timestamp=?", (yesterday + ' 09:00:00',))
        await self.db._conn.commit()
        await self.db.record_mortality('TANK-01', {'count': 10})
        data = await self.db.get_tank_mortality_analytics(days=2)
        self.assertEqual(data['records'][0]['opening_population'], 1000)
        self.assertEqual(data['records'][0]['population'], 990)
        self.assertIsNone(data['records'][1]['mortality_count'])


class CocoTests(unittest.TestCase):
    def test_coco_category_mapping_and_xywh_conversion(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'one.jpg').write_bytes(b'fixture')
            (path / '_annotations.coco.json').write_text(json.dumps({'categories': [{'id': 1, 'name': 'Tilapia-Fingerlings'}],
                'images': [{'id': 7, 'file_name': 'one.jpg', 'width': 100, 'height': 100}],
                'annotations': [{'image_id': 7, 'category_id': 1, 'bbox': [10,20,5,6]}]}))
            images, checksum = coco_split(path)
            self.assertEqual(images[0]['boxes'], [[10,20,15,26]])
            self.assertEqual(len(checksum), 64)
