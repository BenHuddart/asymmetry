"""Headless guidance routing for the ALC scan-build / RF-fit paths.

The integral-scan (ALC) build and the RF-resonance fit raise *guidance* (e.g.
"select at least two runs", "enable RF mode first") through ``QMessageBox``.
``QMessageBox.information``/``warning`` are **modal** (``.exec()``), which blocks
the Qt event loop until a user dismisses them. Under
``QT_QPA_PLATFORM=offscreen`` (CI / automated offscreen driving) there is no
user, so a modal hangs the process forever — the root cause of two full-corpus
evaluation runs (TCNQ, benzene) stalling with flat CPU.

These tests pin the fix: when headless, guidance is routed to the log (a
non-blocking sink) and **no modal ``.exec()`` fires**, while the message is not
swallowed. On a real display the modal path is unchanged.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.mainwindow import MainWindow

_FB = RepresentationType.TIME_FB_ASYMMETRY


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


@pytest.fixture
def no_modals(monkeypatch):
    """Make any modal ``QMessageBox`` an immediate test failure.

    If the headless routing regresses and a real ``.exec()`` is reached, the
    offscreen process would hang forever; here it fails loudly and promptly
    instead. The fixture therefore doubles as the "no blocking modal" assertion.
    """

    def _boom(*_args, **_kwargs):  # pragma: no cover - only hit on regression
        raise AssertionError("a modal QMessageBox fired on the headless path")

    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(_boom))
    monkeypatch.setattr(mw_module.QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(mw_module.QMessageBox, "critical", staticmethod(_boom))


def _ds(run_number: int, fwd: float, bwd: float, field: float) -> MuonDataset:
    n = 4
    meta = {"run_number": run_number, "field": field, "temperature": 10.0}
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.full(n, fwd), bin_width=0.01),
            Histogram(counts=np.full(n, bwd), bin_width=0.01),
        ],
        metadata=dict(meta),
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "bunching_factor": 1,
        },
    )
    t = np.arange(n) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros(n),
        error=np.full(n, 0.01),
        metadata=dict(meta),
        run=run,
    )


def test_is_headless_true_offscreen(qapp: QApplication):
    # The test suite runs under QT_QPA_PLATFORM=offscreen; the detector must
    # report headless so guidance avoids the modal path.
    assert qapp.platformName() in {"offscreen", "minimal", ""}
    assert MainWindow._is_headless() is True


def test_scan_requested_under_two_runs_logs_without_modal(mainwindow: MainWindow, no_modals):
    """<2 selected runs: guidance must reach the log, not a blocking modal."""
    mw = mainwindow
    mw._plot_workspace.set_active_view("integral_scan")
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0)])  # one run only

    start = time.monotonic()
    mw._on_scan_requested()  # must return promptly (no_modals guards blocking)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, "scan request blocked instead of returning promptly"
    log_text = mw._log_panel.to_plain_text()
    assert "Integral scan" in log_text
    assert "at least two runs" in log_text


def test_scan_requested_wrong_representation_logs_without_modal(mainwindow: MainWindow, no_modals):
    """Non-FB representation: the wrong-mode guidance must not block headless."""
    mw = mainwindow
    mw._plot_workspace.set_active_view("frequency")  # not F-B asymmetry

    mw._on_scan_requested()

    log_text = mw._log_panel.to_plain_text()
    assert "Forward-Backward asymmetry only" in log_text


def test_fit_rf_without_rf_mode_logs_without_modal(mainwindow: MainWindow, no_modals):
    """RF fit before enabling RF mode: guidance via the log, no modal hang."""
    mw = mainwindow
    mw._plot_workspace.set_active_view("integral_scan")
    mw._alc_rf_mode = False

    start = time.monotonic()
    mw._on_fit_rf()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, "RF fit blocked instead of returning promptly"
    log_text = mw._log_panel.to_plain_text()
    assert "RF resonance" in log_text


def test_guidance_message_routes_to_log_when_headless(mainwindow: MainWindow, no_modals):
    """The central helper logs (info + warning) instead of exec-ing a modal."""
    mw = mainwindow
    mw._guidance_message("Title A", "an informational note")
    mw._guidance_message("Title B", "a warning note", warning=True)

    log_text = mw._log_panel.to_plain_text()
    assert "Title A: an informational note" in log_text
    assert "Title B: a warning note" in log_text


def test_alc_notify_still_suppressed_while_loading(mainwindow: MainWindow, no_modals):
    """Project-restore suppression is preserved on top of headless routing."""
    mw = mainwindow
    mw._alc_loading = True
    before = mw._log_panel.entry_count()
    mw._alc_notify("Baseline", "should be suppressed")
    assert mw._log_panel.entry_count() == before
    assert "should be suppressed" not in mw._log_panel.to_plain_text()
