"""Browser tests for the read-only console, driven by the real demo pipeline.

Requires the optional ``e2e`` extra (Playwright) and a Chromium binary. Set
``PLAYWRIGHT_CHROMIUM_EXECUTABLE`` when Playwright's bundled browser is not installed.
Set ``QUANT_ALMANAC_SCREENSHOT_DIR`` to also save full-page screenshots of each view.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from quant_signal.api.app import create_app
from quant_signal.api.deps import get_governance_service, get_signal_service
from quant_signal.demo import demo_settings, run_demo
from quant_signal.governance import GovernanceService
from quant_signal.serving.service import SignalService

sync_api = pytest.importorskip("playwright.sync_api", reason="install the e2e extra")

pytestmark = pytest.mark.e2e


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def console(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    workdir = tmp_path_factory.mktemp("demo")
    result = run_demo(workdir, reset=True, echo=False)
    settings = demo_settings(workdir)
    app = create_app()
    app.dependency_overrides[get_governance_service] = lambda: GovernanceService(settings=settings)
    app.dependency_overrides[get_signal_service] = lambda: SignalService(settings=settings)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "console server did not start"
    yield {"base": f"http://127.0.0.1:{port}", "ids": result.ids, "workdir": workdir}
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    executable = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    with sync_api.sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(
                **({"executable_path": executable} if executable else {})
            )
        except sync_api.Error as exc:
            pytest.skip(
                "Chromium is not available to Playwright; run `playwright install chromium` "
                f"or set PLAYWRIGHT_CHROMIUM_EXECUTABLE ({str(exc).splitlines()[0]})"
            )
        yield launched
        launched.close()


def _screenshot(page: Any, name: str) -> None:
    directory = os.getenv("QUANT_ALMANAC_SCREENSHOT_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(directory) / f"{name}.png"), full_page=True)


def test_operator_traces_champion_and_blocked_build(
    console: dict[str, Any],
    browser: Any,
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
    page = context.new_page()
    errors: list[str] = []
    page.on(
        "console", lambda message: errors.append(message.text) if message.type == "error" else None
    )
    champion_id = console["ids"]["champion_5d_model_version_id"]

    page.goto(console["base"] + "/console")
    assert page.get_by_test_id("synthetic-banner").is_visible()
    assert page.get_by_test_id("blocked-count").inner_text() == "1"
    assert page.get_by_test_id("model-count").inner_text() == "6"
    _screenshot(page, "console-index")

    page.locator(f'a[href="/console/models/{champion_id}"]').first.click()
    page.wait_for_url(f"**/console/models/{champion_id}")
    assert page.get_by_test_id("model-id").inner_text() == champion_id
    stepper = page.get_by_test_id("lineage-stepper")
    assert stepper.locator("li").count() == 8
    assert stepper.locator('li[data-stage="dataset"]').get_attribute("data-status") == "pass"
    verification = page.get_by_test_id("verification")
    integrity = verification.locator('[data-check-id="artifact.model_integrity"]')
    assert not integrity.is_visible(), "passing checks should start collapsed"
    verification.get_by_test_id("passing-checks").locator("> summary").click()
    assert integrity.is_visible()
    assert integrity.get_attribute("data-status") == "pass"
    integrity.locator("details.evidence summary").click()
    assert "actual_sha256" in integrity.locator("pre").first.inner_text()
    assert page.get_by_test_id("limitations").locator('li[data-severity="high"]').count() >= 1
    assert page.get_by_test_id("reproducibility").locator('[data-status="pass"]').count() >= 1
    _screenshot(page, "console-model")

    page.goto(console["base"] + "/console")
    page.locator('tr[data-stage="dataset_build"][data-status="fail"] a').click()
    assert page.get_by_test_id("blocked-notice").is_visible()
    failed = page.locator('.check[data-status="fail"]')
    assert {failed.nth(i).get_attribute("data-check-id") for i in range(failed.count())} == {
        "bars.positive_prices",
        "bars.ohlc_consistency",
        "bars.calendar_alignment",
    }
    _screenshot(page, "console-blocked-build")
    assert errors == []
    context.close()


def test_model_page_fits_a_phone_viewport(console: dict[str, Any], browser: Any) -> None:
    context = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    page = context.new_page()
    champion_id = console["ids"]["champion_5d_model_version_id"]
    page.goto(f"{console['base']}/console/models/{champion_id}")
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    _screenshot(page, "console-model-mobile")
    context.close()


def test_demo_summary_records_every_step(console: dict[str, Any]) -> None:
    summary = json.loads((console["workdir"] / "demo_summary.json").read_text("utf-8"))
    steps = {step["step"]: step["exit_code"] for step in summary["steps"]}
    assert steps["build dataset with defective ticker (expected: blocked)"] == 1
    assert steps["reproduce 5d champion from artifacts"] == 0
    assert summary["data"].startswith("SYNTHETIC")
