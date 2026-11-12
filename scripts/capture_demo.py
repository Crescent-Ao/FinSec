"""Capture the real local trajectory viewer; requires optional Playwright + Pillow."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import json

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'docs/assets/demo'
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / 'demo/cases.json').read_text())
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(ROOT / 'demo')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    frames = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 1000}, device_scale_factor=1)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
            page.wait_for_selector('#content:not([hidden])')
            assert page.locator('#cases option').count() == 3
            for index, name in enumerate(['ecb', 'digest', 'kdf']):
                page.select_option('#cases', str(index))
                page.wait_for_function('(file) => document.querySelector("#title").textContent === file', arg=cases[index]['file'])
                expected_tools = sum(t['kind'] == 'tool' for t in cases[index]['result']['trace'])
                assert page.locator('.step.tool').count() == expected_tools
                assert page.locator('#source').inner_text() == cases[index]['input_source'].rstrip('\n') or page.locator('#source').inner_text() == cases[index]['input_source']
                page.screenshot(path=str(output / (name + '.png')), full_page=True)
                page.evaluate('window.scrollTo(0,0)')
                temporary = output / (name + '-frame.png')
                page.screenshot(path=str(temporary))
                frame = Image.open(temporary).convert('RGB')
                frame.thumbnail((900, 704))
                frames.append(frame.quantize(colors=96))
                temporary.unlink()
            page.set_viewport_size({'width': 390, 'height': 844})
            page.select_option('#cases', '0')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            page.set_input_files('#file', str(ROOT / 'demo/cases.json'))
            assert page.locator('#cases option').count() == 3
            assert not errors, errors
            browser.close()
        frames[0].save(output / 'agent_walkthrough.gif', save_all=True, append_images=frames[1:], duration=3000, loop=0, optimize=True)
        assert all(path.stat().st_size < 2_000_000 for path in output.iterdir())
        print('PASS: 3 real-case renders, tool counts, source fidelity, local import and mobile layout; screenshots + GIF saved.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
