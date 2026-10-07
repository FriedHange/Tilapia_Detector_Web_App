"""Production UI and tooltip checks against the isolated preview server only."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect


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
        admin=browser.new_page(viewport={'width':1440,'height':1000})
        admin.on('pageerror',lambda error:errors.append(str(error)))
        def login(page,user,password):
            page.goto(args.url+'/login')
            page.locator('#username').fill(user);page.locator('#password').fill(password)
            page.get_by_role('button',name='Sign in',exact=True).click()
        login(admin,'preview-admin','Preview admin password!')
        expect(admin.get_by_role('heading',name='Farm administration',exact=True)).to_be_visible()
        # Recover a disposable fixture left on by an interrupted previous test run.
        if admin.evaluate("api('/api/production').then(v=>v.enabled)"):
            admin.evaluate("api('/api/admin/production',{method:'PUT',body:{enabled:false}})")
            admin.evaluate('refresh()')
        assert admin.evaluate("api('/api/production').then(v=>!v.enabled && !v.startup_configured)")
        admin.get_by_role('button',name='Turn Production Mode on',exact=True).hover()
        expect(admin.get_by_role('tooltip')).to_contain_text('whole installation')
        admin.get_by_role('button',name='Turn Production Mode on',exact=True).click()
        admin.get_by_role('button',name='Enable Production Mode',exact=True).click()
        expect(admin.locator('#formDialog')).not_to_be_visible()
        expect(admin.get_by_role('button',name='Turn Production Mode off',exact=True)).to_be_visible()
        expect(admin.get_by_role('button',name='Benchmarks',exact=True)).to_have_count(0)
        admin.locator('#farmSelector').select_option(label='Browser test farm')
        expect(admin.get_by_role('heading',name='Dashboard',exact=True)).to_be_visible()
        expect(admin.locator('.monitoring-tile')).to_have_count(0)
        assert admin.locator('main svg[role=img]').count()>=2
        original=admin.evaluate("api('/api/reports').then(r=>r.tanks.find(t=>t.tank_id==='TANK-01').camera_source)")
        admin.get_by_role('button',name='Tank Management',exact=True).click()
        expect(admin.get_by_role('heading',name='Tank Management',exact=True)).to_be_visible()
        first=admin.locator('[data-monitor-tank=TANK-01]')
        expect(first.get_by_role('button',name='Start camera',exact=True)).to_be_disabled()
        expect(first.get_by_role('button',name='Stock fish',exact=True)).to_have_count(0)
        expect(first.get_by_role('button',name='Mortality check',exact=True)).to_have_count(0)
        first.get_by_role('button',name='Edit Tank',exact=True).click()
        expect(admin.locator('[name=video_upload]')).to_have_count(0)
        expect(admin.locator('[name=video_source]')).to_have_count(0)
        expect(admin.locator('[name=source_type]')).to_be_visible()
        assert admin.locator('[name=source_type] option').all_text_contents()==['No source','USB camera','Network camera']
        admin.locator('#cancelDialog').click()
        first.locator('.monitoring-preview').click()
        expect(admin.locator('#countPhoto').locator('..')).not_to_be_visible()
        expect(admin.locator('#countVideo').locator('..')).not_to_be_visible()
        expect(admin.locator('#useCount')).not_to_be_visible()
        admin.locator('#closeCount').click()
        rejected=admin.evaluate("async()=>{const response=await fetch('/api/tanks/TANK-01',{method:'PUT',headers:{'Content-Type':'application/json','X-CSRF-Token':State.csrf,'X-Farm-ID':State.farm},body:JSON.stringify({camera_source:State.report.tanks[0].demonstration_source})});return response.status;}")
        assert rejected==400
        assert admin.evaluate("api('/api/reports').then(r=>r.tanks.find(t=>t.tank_id==='TANK-01').camera_source)")==original
        admin.screenshot(path=str(artifacts/'production-tanks-desktop.png'),full_page=True)
        farmer=browser.new_page(viewport={'width':390,'height':844},has_touch=True)
        farmer.on('pageerror',lambda error:errors.append(str(error)))
        login(farmer,'preview-farmer','Preview farmer password!')
        expect(farmer.get_by_role('heading',name='Dashboard',exact=True)).to_be_visible()
        expect(farmer.get_by_role('button',name='Turn Production Mode off',exact=True)).to_have_count(0)
        farmer.get_by_role('button',name='Open tank monitoring',exact=True).click()
        expect(farmer.get_by_role('heading',name='Tank Management',exact=True)).to_be_visible()
        enlarge=farmer.locator('[data-monitor-tank=TANK-01] .tank-title')
        enlarge.dispatch_event('pointerdown',{'pointerType':'touch'})
        farmer.wait_for_timeout(600)
        expect(farmer.get_by_role('tooltip')).to_contain_text('live view')
        enlarge.dispatch_event('pointerup',{'pointerType':'touch'})
        enlarge.dispatch_event('click')
        expect(farmer.locator('#countDialog')).not_to_be_visible()
        farmer.locator('#menuButton').focus()
        enlarge.focus()
        expect(farmer.get_by_role('tooltip')).to_contain_text('live view')
        expect(enlarge).to_have_attribute('aria-describedby','action-help')
        farmer.keyboard.press('Escape')
        expect(farmer.get_by_role('tooltip')).not_to_be_visible()
        assert farmer.evaluate('document.documentElement.scrollWidth<=innerWidth')
        farmer.screenshot(path=str(artifacts/'production-tanks-mobile.png'),full_page=True)
        admin.get_by_role('button',name='Turn Production Mode off',exact=True).click()
        admin.get_by_role('button',name='Disable Production Mode',exact=True).click()
        expect(admin.locator('#formDialog')).not_to_be_visible()
        expect(admin.get_by_role('button',name='Benchmarks',exact=True)).to_be_visible()
        admin.evaluate('refresh()')
        assert admin.evaluate("State.report.tanks.find(t=>t.tank_id==='TANK-01').camera_source")==original
        admin.get_by_role('button',name='Dashboard',exact=True).click()
        expect(admin.get_by_role('heading',name='Dashboard',exact=True)).to_be_visible()
        admin.screenshot(path=str(artifacts/'production-dashboard-desktop.png'),full_page=True)
        browser.close()
    if errors:raise AssertionError(json.dumps(errors))
    print('Passed: Admin-only installation mode, live-only sources, preserved videos, dashboard graphs, removed stock/mortality buttons, desktop/mobile and hover/focus/touch help. Startup was not activated.')


if __name__=='__main__':main()
