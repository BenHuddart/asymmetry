"""RED target for branch ``fix/batch-robustness``.

Round-2 GUI finding (Spin Glass, ``_findings/windows-gui/SpinGlass_YMnAl.md``):
a 15-run batch fit where 2 runs failed to converge reported "Batch fit failed"
and created NO FitSeries at all — discarding the 13 runs that *did* converge.

Root cause: ``FitPanel._on_fit_finished`` (``fit_panel.py`` ~L5249) only emits the
series when ``all(r.success for r in results_dict.values())``; any failure takes
the abort branch.

Desired behaviour: on partial success, still build the series from the converged
members and surface the failures as a warning. This test asserts
``_emit_global_fit_success`` is invoked with its ``successful`` kwarg holding
*only* the converged runs when one member fails (``results_dict`` carries every
member, converged or not, so the results card can show a chip per run).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _dataset(run_number: int) -> MuonDataset:
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": 110.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    return MuonDataset(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01, 0.01]),
        {"run_number": run_number},
        run,
    )


def _ok(run_number: int) -> FitResult:
    return FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        parameters=ParameterSet([Parameter("A", 0.2), Parameter("Lambda", 0.5)]),
        uncertainties={"A": 0.01, "Lambda": 0.02},
    )


def _failed() -> FitResult:
    return FitResult(success=False, message="call limit reached, invalid parameters")


def test_partial_batch_failure_still_emits_series_for_successes(mw, monkeypatch):
    # The batch completion handler lives on the GlobalFitTab ("Batch" tab).
    panel = mw._fit_panel._global_tab
    panel._datasets = [_dataset(10), _dataset(11)]
    panel._current_model = object()
    panel._current_global_params = []

    captured: dict[str, object] = {}

    def _record(**kwargs):
        captured["called"] = True
        # ``results_dict`` now carries every member (the results card shows a
        # chip per run, failures included); ``successful`` is the filtered set
        # that actually becomes the emitted series.
        captured["results_dict"] = kwargs.get("results_dict")
        captured["successful"] = kwargs.get("successful")

    monkeypatch.setattr(panel, "_emit_global_fit_success", _record)

    # Run 10 converged, run 11 failed.
    results_dict = {10: _ok(10), 11: _failed()}
    panel._on_fit_finished(results_dict, [])

    assert captured.get("called"), "no series emitted — the whole batch was discarded"
    assert set(captured["successful"]) == {10}, "series must keep only the converged run(s)"
    assert set(captured["results_dict"]) == {10, 11}, "the card must still see every member"
