"""The screenshot-capture watchdog must survive GIL starvation.

Native fits can hold the GIL, so termination and stack dumps must use the
C-level faulthandler timer. The whole suite has a ten-minute budget, below
its twelve-minute CI backstop.
"""

from __future__ import annotations

import faulthandler
from pathlib import Path

import yaml

from docs.screenshots import capture


def test_watchdog_uses_faulthandler_timer(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: calls.append(("enable", a, k)))
    monkeypatch.setattr(
        faulthandler,
        "dump_traceback_later",
        lambda *a, **k: calls.append(("later", a, k)),
    )

    capture._start_watchdog(timeout_s=123)

    later = [c for c in calls if c[0] == "later"]
    assert later, "watchdog must arm faulthandler.dump_traceback_later"
    _, args, kwargs = later[0]
    # The timeout is passed positionally; exit=True is what actually kills the
    # process (a bare dump would log and then hang forever again).
    assert args and args[0] == 123
    assert kwargs.get("exit") is True


def test_watchdog_does_not_spawn_a_python_daemon_thread(monkeypatch):
    # Guard against a regression back to a threading.Thread daemon, which is the
    # GIL-starvable design that caused the multi-hour hang.
    import threading

    created: list = []
    real_thread = threading.Thread

    def _spy(*args, **kwargs):
        created.append((args, kwargs))
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", _spy)
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: None)

    capture._start_watchdog(timeout_s=5)

    assert created == [], "watchdog must not rely on a Python daemon thread"


def test_default_capture_budget_precedes_ci_backstop(monkeypatch):
    calls = []
    monkeypatch.setattr(faulthandler, "enable", lambda: None)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls.append((a, k)))
    capture._start_watchdog()
    assert calls == [((600,), {"exit": True})]
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/docs-pages.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["build-docs"]["steps"]
        if step["name"] == "Generate GUI screenshots"
    )
    assert step["timeout-minutes"] == 12
    assert calls[0][0][0] < step["timeout-minutes"] * 60
