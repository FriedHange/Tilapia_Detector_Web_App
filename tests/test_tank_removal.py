"""Tank removal retains history and never classifies remaining stock as mortality."""
import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from farm_database import FarmDatabase, TankPopulationChanged, now_text


class TankRemovalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'farm.db'
        self.db = FarmDatabase(self.path)
        await self.db.connect()
        await self.db.create_tank({'tank_id':'TANK-01','name':'Nursery','current_count':100,'camera_source':'0'})
        await self.db.configure_monitoring('TANK-01',validation={'known_count':100,'profile':'fixture','error_band':0})

    async def asyncTearDown(self):
        await self.db.close()
        self.temp.cleanup()

    async def test_stock_removal_preserves_deaths_history_and_does_not_add_mortality(self):
        await self.db.record_mortality('TANK-01',{'count':5})
        self.assertTrue(await self.db.delete_tank('TANK-01',remove_stock=True,expected_count=95))
        tank=await self.db.get_tank('TANK-01')
        self.assertEqual((tank['current_count'],tank['status']),(0,'inactive'))
        report=await self.db.get_production_report()
        self.assertEqual((report['total_population'],report['total_dispersed_count']),(0,0))
        day=(await self.db.get_tank_mortality_analytics(days=1))['records'][0]
        self.assertEqual((day['population'],day['confirmed_mortality_count'],day['estimated_mortality_count'],day['mortality_rate_pct']),(0,5,0,5))
        config=(await self.db.monitoring_configs())[0]
        self.assertEqual(config['enabled'],0)
        self.assertIsNone(config['validation'])
        movements=await self.db._rows("SELECT * FROM inventory_movements WHERE kind='tank_removed'")
        self.assertEqual(movements[0]['delta'],-95)

    async def test_changed_population_rolls_back_and_requires_confirmation(self):
        await self.db.record_mortality('TANK-01',{'count':2})
        with self.assertRaises(TankPopulationChanged):
            await self.db.delete_tank('TANK-01',remove_stock=True,expected_count=100)
        for options in ({},{'remove_stock':True},{'expected_count':98}):
            with self.assertRaises(ValueError):await self.db.delete_tank('TANK-01',**options)
        self.assertEqual((await self.db.get_tank('TANK-01'))['current_count'],98)
        self.assertIsNotNone((await self.db.monitoring_configs())[0]['validation'])
        self.assertEqual(await self.db._rows("SELECT * FROM inventory_movements WHERE kind='tank_removed'"),[])

    async def test_concurrent_retries_record_removal_once_and_reserve_code(self):
        other=FarmDatabase(self.path)
        await other.connect()
        try:
            result=await asyncio.gather(*(database.delete_tank('TANK-01',remove_stock=True,expected_count=100) for database in (self.db,other)))
            self.assertEqual(result,[True,True])
        finally:await other.close()
        self.assertEqual(len(await self.db._rows("SELECT * FROM inventory_movements WHERE kind='tank_removed'")),1)
        self.assertEqual((await self.db.create_tank({}))['tank_id'],'TANK-02')

    async def test_empty_removal_preserves_history_and_rejects_new_operations(self):
        await self.db.create_tank({'tank_id':'EMPTY','name':'Empty'})
        self.assertTrue(await self.db.delete_tank('EMPTY'))
        operations=[self.db.stock_fish('EMPTY',{'count':1}),self.db.record_mortality('EMPTY',{'count':0}),
            self.db.update_tank('EMPTY',{'status':'active'}),self.db.configure_monitoring('EMPTY',enabled=True),
            self.db.commit_dispersal({'tank_id':'TANK-01','count':1,'recipient':'Other','type':'transfer','destination_tank_id':'EMPTY','dispersal_id':'blocked'})]
        for operation in operations:
            with self.assertRaisesRegex(ValueError,'removed'):await operation
        self.assertEqual((await self.db.get_tank('TANK-01'))['current_count'],100)
        self.assertFalse(await self.db.delete_tank('missing'))

    async def test_removed_tank_no_longer_accrues_daily_eligibility(self):
        today=datetime.strptime(now_text(),'%Y-%m-%d %H:%M:%S')
        before=(today-timedelta(days=2)).strftime('%Y-%m-%d')
        yesterday=(today-timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        await self.db._conn.execute('UPDATE inventory_baselines SET date=?,population=100',(before,))
        await self.db._conn.execute("UPDATE inventory_movements SET timestamp=? WHERE kind='stocking'",((today-timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S'),))
        await self.db._conn.commit()
        with patch('farm_database.now_text',return_value=yesterday):
            await self.db.delete_tank('TANK-01',remove_stock=True,expected_count=100)
        records=sorted((await self.db.get_tank_mortality_analytics(days=3))['records'],key=lambda r:r['date'])
        self.assertEqual([r['eligible'] for r in records],[True,True,False])
        self.assertEqual([r['population'] for r in records],[100,0,0])
        self.assertTrue(all(r['confirmed_mortality_count']==0 and r['estimated_mortality_count']==0 for r in records))
