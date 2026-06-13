"""Tests for fit-panel helpers and the shared fit-call worker glue."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QObject, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.gui.panels.fit_panel import _format_param_label, _start_fit_call  # noqa: E402
from asymmetry.gui.tasks import TaskRunner  # noqa: E402


def test_format_param_label_known_and_unknown() -> None:
    assert _format_param_label("A0") == "A₀ (%)"
    assert _format_param_label("Lambda") == "λ (μs⁻¹)"
    assert _format_param_label("custom") == "custom"


def _run_fit_call(call, timeout_ms: int = 5_000) -> dict:
    """Drive *call* through _start_fit_call on a real TaskRunner to a terminal.

    Returns the captured terminal callback payload (``finished`` / ``error`` /
    ``cancelled``); the callbacks fire on the GUI thread via the runner's relay.
    """
    holder = QObject()
    holder._fit_call_runner = TaskRunner(holder)
    out: dict = {}
    loop = QEventLoop()

    def _finished(result):
        out["finished"] = result
        loop.quit()

    def _error(message):
        out["error"] = message
        loop.quit()

    def _cancelled():
        out["cancelled"] = True
        loop.quit()

    _start_fit_call(holder, call, on_finished=_finished, on_error=_error, on_cancelled=_cancelled)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    holder._fit_call_runner.shutdown()
    return out


def test_fit_call_marshals_engine_result(qapp: QApplication) -> None:
    """A successful engine call's return value reaches on_finished verbatim.

    Mirrors a global fit, whose engine returns the ``(results, global)`` tuple
    the launch unpacks into the finished handler.
    """

    def call(*, cancel_callback=None):
        assert callable(cancel_callback)
        return ({1: "ok"}, {"A0": 0.2})

    out = _run_fit_call(call)

    assert out.get("finished") == ({1: "ok"}, {"A0": 0.2})


def test_fit_call_reports_formatted_error(qapp: QApplication) -> None:
    """An engine exception reaches on_error, reformatted not crashed."""

    def call(*, cancel_callback=None):
        raise RuntimeError("boom")

    out = _run_fit_call(call)

    assert "error" in out
    assert "boom" in out["error"]
