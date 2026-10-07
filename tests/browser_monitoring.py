"""Real parallel monitoring browser checks against the disposable preview server."""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--executable',required=True)
    args=parser.parse_args()
    artifacts=Path(__file__).resolve().parents[1]/'private_data'/'browser_qa'
    artifacts.mkdir(parents=True,exist_ok=True)
    errors=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True,executable_path=args.executable)
        page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        page.set_default_timeout(30000)
        page.on('pageerror',lambda error:errors.append(str(error)))
        sockets=[]
        page.on('websocket',lambda socket:sockets.append(socket))
        def sign_in(username,password):
            page.goto(args.url+'/login')
            page.locator('#username').fill(username);page.locator('#password').fill(password)
            page.get_by_role('button',name='Sign in',exact=True).click()
        sign_in('preview-farmer','Preview farmer password!')
        expect(page.get_by_role('heading',name='Dashboard',exact=True)).to_be_visible()
        page.get_by_role('button',name='Tank Management',exact=True).click()
        before=page.evaluate("api('/api/reports').then(r=>r.total_population)")
        original=page.evaluate("State.report.tanks.find(t=>t.tank_id==='TANK-02').camera_source")
        first=page.locator('[data-monitor-tank="TANK-01"]')
        second=page.locator('[data-monitor-tank="TANK-02"]')
        expect(page.get_by_role('button',name='Start all',exact=True)).to_be_visible()
        page.get_by_role('button',name='Start all',exact=True).click()
        page.wait_for_function("['TANK-01','TANK-02'].every(id=>freshMonitoring(tankStream(id)))")
        assert len(sockets)==2
        assert page.evaluate("['TANK-01','TANK-02'].every(id=>tankStream(id).data.live_count>0)")
        expect(first.locator('[data-monitor-field=count]')).not_to_have_text('—')
        expect(second.locator('[data-monitor-field=count]')).not_to_have_text('—')
        expect(page.get_by_role('button',name='Stop all',exact=True)).to_be_visible()
        first.locator('.monitoring-preview').click()
        expect(page.locator('#countDialog')).to_be_visible()
        expect(page.locator('#occupancyValue')).to_contain_text('%')
        page.locator('#closeCount').click()
        assert len(sockets)==2,'Enlarging created an unnecessary connection'
        # Stop and restart one tank while the second keeps updating.
        previous=page.evaluate("tankStream('TANK-02').data.frame_idx")
        page.evaluate("Promise.all([toggleTankMonitoring('TANK-01'),toggleTankMonitoring('TANK-01')])")
        expect(first.get_by_role('button',name='Play video',exact=True)).to_be_visible()
        expect(first.locator('[data-monitor-field=count]')).to_have_text('—')
        page.wait_for_function("old=>tankStream('TANK-02').data.frame_idx>old",arg=previous)
        first.get_by_role('button',name='Play video',exact=True).click()
        page.wait_for_function("freshMonitoring(tankStream('TANK-01'))")
        page.wait_for_function("['TANK-01','TANK-02'].every(id=>tankStream(id).data.playback_cycle>0)")
        # Measure actual concurrent delivery rather than claiming a target FPS.
        initial=page.evaluate("['TANK-01','TANK-02'].map(id=>tankStream(id).data.frame_idx)")
        started=time.perf_counter();page.wait_for_timeout(5000)
        elapsed=time.perf_counter()-started
        final=page.evaluate("['TANK-01','TANK-02'].map(id=>({tank:id,frame:tankStream(id).data.frame_idx,count:tankStream(id).data.live_count}))")
        measured=[{'tank':row['tank'],'visible_fish':row['count'],'delivered_updates_per_second':round((row['frame']-initial[index])/elapsed,2)} for index,row in enumerate(final)]
        assert all(row['delivered_updates_per_second']>0 for row in measured)
        # Dashboard re-renders and module navigation retain the existing sockets.
        page.evaluate("window.savedMonitoringSocket=tankStream('TANK-01').ws")
        page.get_by_role('button',name='Food Management',exact=True).click()
        expect(page.get_by_role('heading',name="Today's estimated feed requirements",exact=True)).to_be_visible()
        expect(page.get_by_role('button',name='Record feeding',exact=True)).to_have_count(0)
        expect(page.get_by_role('button',name='Record purchase',exact=True)).to_have_count(0)
        previous=page.evaluate("tankStream('TANK-01').data.frame_idx")
        page.wait_for_function("old=>tankStream('TANK-01').data.frame_idx>old",arg=previous)
        page.get_by_role('button',name='Tank Management',exact=True).click()
        page.evaluate('refresh()')
        assert page.evaluate("window.savedMonitoringSocket===tankStream('TANK-01').ws")
        assert len(sockets)==3
        page.screenshot(path=str(artifacts/'parallel-monitoring-desktop.png'),full_page=True)
        # A stalled connection must not present its old numbers as current.
        page.evaluate("window.savedHandler=tankStream('TANK-01').ws.onmessage;tankStream('TANK-01').ws.onmessage=()=>{}")
        page.get_by_role('button',name='Tank Management',exact=True).click()
        expect(first.locator('[data-monitor-field=occupancy]')).to_contain_text('%')
        expect(first.locator('[data-monitor-field=occupancy]')).to_have_text('—',timeout=15000)
        expect(second.locator('[data-monitor-field=occupancy]')).to_contain_text('%')
        page.get_by_role('button',name='Tank Management',exact=True).click()
        expect(first.locator('[data-monitor-field=status]')).to_have_text('Delayed',timeout=15000)
        expect(first.locator('[data-monitor-field=count]')).to_have_text('—')
        expect(second.locator('[data-monitor-field=status]')).to_have_text('Monitoring')
        first.locator('.monitoring-preview').click()
        expect(page.locator('#countValue')).to_have_text('—')
        expect(page.locator('#useCount')).to_be_disabled()
        page.locator('#closeCount').click()
        page.evaluate("tankStream('TANK-01').ws.onmessage=window.savedHandler")
        page.wait_for_function("freshMonitoring(tankStream('TANK-01'))")
        # Photo counting replaces only the selected tank view, never other feeds.
        second.locator('.monitoring-preview').click()
        photo=Path(__file__).resolve().parents[1]/'static'/'media'/'dataset_sample_1.jpg'
        page.locator('#countPhoto').set_input_files(str(photo))
        expect(page.locator('#countStatus')).to_have_text('Photo estimate. Review it before saving.')
        expect(page.locator('#useCount')).to_be_enabled()
        assert page.evaluate("window.savedMonitoringSocket===tankStream('TANK-01').ws")
        expect(page.locator('#useCount')).not_to_be_visible()
        page.locator('#closeCount').click()
        # Upload a broken replacement through the tank view; only that tank fails.
        second.locator('.monitoring-preview').click()
        page.locator('#countVideo').set_input_files({'name':'broken.mp4','mimeType':'video/mp4','buffer':b'not readable footage'})
        page.wait_for_function("tankStream('TANK-02')?.status==='error'")
        expect(page.locator('#toggleCamera')).to_have_text('Play video')
        expect(page.locator('#countValue')).to_have_text('—')
        page.locator('#closeCount').click()
        expect(first.locator('[data-monitor-field=status]')).to_have_text('Monitoring')
        # Restore and replace a source with usable footage through the same view.
        second.locator('.monitoring-preview').click()
        page.locator('#countVideo').set_input_files({'name':'replacement.mp4','mimeType':'video/mp4','buffer':Path(original).read_bytes()})
        page.wait_for_function("freshMonitoring(tankStream('TANK-02'))")
        page.locator('#closeCount').click()
        assert page.evaluate("window.savedMonitoringSocket===tankStream('TANK-01').ws")
        assert page.evaluate("api('/api/reports').then(r=>r.total_population)")==before
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Mobile grid overflow'
        page.screenshot(path=str(artifacts/'parallel-monitoring-mobile.png'),full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        page.get_by_role('button',name='Stop all',exact=True).click()
        page.wait_for_function("[...State.streams.values()].every(e=>!isMonitoring(e))")
        page.get_by_role('button',name='Start all',exact=True).click()
        page.wait_for_function("freshMonitoring(tankStream('TANK-01'))")
        page.get_by_role('button',name='Sign out',exact=True).click()
        expect(page.locator('#username')).to_be_visible()
        sign_in('preview-admin','Preview admin password!')
        page.locator('#farmSelector').select_option(label='Browser test farm')
        expect(page.get_by_role('heading',name='Dashboard',exact=True)).to_be_visible()
        page.get_by_role('button',name='Tank Management',exact=True).click()
        page.get_by_role('button',name='Start all',exact=True).click()
        page.wait_for_function("freshMonitoring(tankStream('TANK-01'))")
        page.locator('#farmSelector').select_option('')
        expect(page.get_by_role('heading',name='Farm administration',exact=True)).to_be_visible()
        assert page.evaluate('State.streams.size')==0
        browser.close()
    if errors:raise AssertionError('Browser JavaScript errors: '+json.dumps(errors))
    (artifacts/'monitoring_results.json').write_text(json.dumps({'concurrent_streams':measured,'saved_population_unchanged':True},indent=2),encoding='utf-8')
    print('Passed: parallel real video inference, rapid independent toggles, enlargement, looping, navigation, refresh, delayed frames, photo counting, source replacement/failure, mobile, farm switch and sign-out.')
    print(json.dumps(measured))


if __name__=='__main__':main()
