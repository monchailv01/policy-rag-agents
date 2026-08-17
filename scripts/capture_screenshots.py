#!/usr/bin/env python3
"""Drive the web UI with a headless browser and save the deliverable screenshots.

Requires the server to be running (``python server.py``) and the dev extras
(``pip install -r requirements-dev.txt && playwright install chromium``).

    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"
BASE_URL = "http://127.0.0.1:8100"
VIEWPORT = {"width": 1680, "height": 1050}
ANSWER_TIMEOUT_MS = 180_000

#: (filename, [questions...], inspector tab to open, where to scroll the log)
SHOTS: list[tuple[str, list[str], str, str]] = [
    (
        "01-international-travel-en",
        ["What is the policy on international travel?"],
        "pipeline",
        "top",
    ),
    (
        "02-per-diem-multi-policy",
        ["How much per diem do I get for a five-day trip to Tokyo, and by when must I file the claim?"],
        "snippets",
        "top",
    ),
    (
        "03-thai-query",
        ["ขอลาพักร้อนติดกัน 5 วัน ต้องยื่นล่วงหน้ากี่วัน และถ้าลาป่วยต้องใช้ใบรับรองแพทย์เมื่อไหร่"],
        "pipeline",
        "top",
    ),
    (
        "04-cross-policy-ai",
        ["Can I paste a customer's account number into ChatGPT to summarise it?"],
        "snippets",
        "top",
    ),
    (
        "05-out-of-scope",
        ["What is the company's pet insurance benefit?"],
        "pipeline",
        "top",
    ),
    (
        "06-multi-turn-memory",
        [
            "What is the per diem for Europe?",
            "แล้วถ้าไปญี่ปุ่นล่ะ",
            "ตอนแรกที่ผมถามยุโรป ตอบว่าเท่าไหร่นะ",
        ],
        "pipeline",
        "bottom",
    ),
    (
        "07-knowledge-base-viewer",
        [],
        "kb",
        "top",
    ),
]


def ask(page, question: str) -> None:
    """Type a question and wait until the pipeline has finished the turn."""
    before = page.locator(".turn").count()
    page.fill("#q", question)
    page.click("#send")
    page.wait_for_function(
        f"document.querySelectorAll('.turn').length >= {before + 2}",
        timeout=ANSWER_TIMEOUT_MS,
    )
    page.wait_for_selector("#send:not([disabled])", timeout=ANSWER_TIMEOUT_MS)
    page.wait_for_timeout(600)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(BASE_URL, wait_until="networkidle")
        page.wait_for_selector(".chip", timeout=30_000)

        for name, questions, tab, scroll in SHOTS:
            print(f"-> {name}", flush=True)
            page.click("#new-chat")
            page.wait_for_timeout(200)
            for question in questions:
                ask(page, question)
            page.click(f'.tab[data-tab="{tab}"]')
            if tab == "kb":
                page.click(".pol-h")  # expand the first policy so text is visible
            elif tab == "snippets":
                page.click(".snip-h")  # expand the top-scoring snippet
            # Long answers scroll the question out of view; show the top of the
            # exchange instead, which is what the screenshot is meant to prove.
            page.evaluate(
                "document.getElementById('log').scrollTop = "
                + ("1e9" if scroll == "bottom" else "0")
            )
            page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / f"{name}.png"))

        browser.close()
    print(f"\nsaved {len(SHOTS)} screenshots to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
