"""Settings and recorded-demo UI smoke against tests/serve_preview.py."""
import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--executable', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=args.executable)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url + '/login')
        page.locator('#username').fill('preview-admin')
        page.locator('#password').fill('Preview admin password!')
        page.get_by_role('button', name='Sign in', exact=True).click()
        expect(page.get_by_role('heading', name='Farm administration')).to_be_visible()
        page.get_by_role('button', name='Settings', exact=True).click()
        expect(page.get_by_role('heading', name='Settings', exact=True)).to_be_visible()
        expect(page.get_by_text('Live aquarium snapshots')).to_be_visible()
        page.get_by_role('button', name='Edit timing').click()
        page.locator('#dialogForm [name=median_window_seconds]').fill('30')
        page.locator('#dialogForm [name=consistent_windows]').fill('2')
        page.get_by_role('button', name='Save timing').click()
        expect(page.locator('#formDialog')).not_to_be_visible()
        expect(page.get_by_text('about 65 seconds')).to_be_visible()
        page.get_by_role('button', name='Create demo farm').click()
        expect(page.get_by_role('button', name='Set up tank and clips')).to_be_visible()
        upload = page.locator('#demoClipFile')
        for index in range(2):
            upload.set_input_files(str(root / 'sample.mp4'))
            page.get_by_role('button', name='Upload clip').click()
            expect(page.get_by_text('Demo clip uploaded.')).to_be_visible()
            page.wait_for_function('(count) => State.demoMedia.length >= count', arg=index + 1)
        page.get_by_role('button', name='Set up tank and clips').click()
        expect(page.locator('#dialogForm [name=before_source] option')).to_have_count(2)
        page.locator('#dialogForm [name=starting_count]').fill('10')
        page.locator('#dialogForm [name=after_source]').select_option(index=1)
        page.locator('#dialogForm [name=whole_view]').check()
        page.get_by_role('button', name='Save demo setup').click()
        expect(page.locator('#formDialog')).not_to_be_visible()
        expect(page.get_by_text('Known starting stock')).to_be_visible()
        page.get_by_role('button', name='Open demo farm').click()
        expect(page.get_by_text('SIMULATED DEFENSE DEMO', exact=False).first).to_be_visible()
        assert page.evaluate('State.report.simulated') is True
        assert not errors, errors
        browser.close()
    print('Passed: timing form, two clip uploads, guided setup, and simulated demo banner.')


if __name__ == '__main__':
    main()
