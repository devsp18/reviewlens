#!/usr/bin/env python
"""One-off dev tool: screenshot every page at http://localhost:8765 into docs/screenshots/.
Not part of the app - requires `pip install playwright && playwright install chromium`."""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
BASE_URL = "http://localhost:8765"

PAGES = {
    "home": "",
    "dashboard": "Dashboard",
    "backlog": "Backlog",
    "review_explorer": "Review_Explorer",
    "labeling_studio": "Labeling_Studio",
    "eval_lab": "Eval_Lab",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        for name, path in PAGES.items():
            page.goto(f"{BASE_URL}/{path}", wait_until="networkidle")
            time.sleep(2)  # let Plotly charts finish animating in
            out = OUT_DIR / f"{name}.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"saved {out}")
        browser.close()


if __name__ == "__main__":
    main()
