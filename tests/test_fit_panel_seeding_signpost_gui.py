"""Batch-fit seeding signpost (GUI).

When a batch's ν(T)/A(T) trend shows the near-transition collapse/outlier signature,
the batch tab must signpost the per-run warm-start ("Per-run seeds…") and offer to
apply the descending-frequency seeds the diagnostics computed — automating the proven
manual cure for the EuO bistability instead of leaving the user with a corrupted trend
and no guidance.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import GlobalFitTab

_MODEL = SimpleNamespace(param_names=["A_1", "frequency", "lambda", "A_bg"])
# Run -> (temperature, fitted amplitude, fitted frequency); run 2944 collapsed.
_TREND = {
    2960: (10.0, 25.0, 30.0),
    2955: (30.0, 24.0, 26.0),
    2950: (50.0, 22.0, 18.0),
    2945: (60.0, 20.0, 12.0),
    2944: (63.0, 0.1, 30.5),  # spurious branch
    2943: (65.5, 18.0, 10.5),
}


def _result(amp: float, freq: float, success: bool = True) -> FitResult:
    ps = ParameterSet()
    ps.add(Parameter(name="A_1", value=amp))
    ps.add(Parameter(name="frequency", value=freq))
    ps.add(Parameter(name="lambda", value=0.1))
    return FitResult(success=success, reduced_chi_squared=1.0, parameters=ps)


def _attach_trend(tab: GlobalFitTab) -> dict[int, FitResult]:
    time = np.linspace(0.1, 8.0, 8)
    tab._datasets = [
        MuonDataset(
            time=time,
            asymmetry=np.zeros_like(time),
            error=np.ones_like(time),
            metadata={"run_number": run, "temperature": temp},
        )
        for run, (temp, _a, _f) in _TREND.items()
    ]
    tab._current_model = _MODEL
    return {run: _result(amp, freq) for run, (_t, amp, freq) in _TREND.items()}


def test_per_run_seeds_button_and_signpost_share_wording(qapp: QApplication) -> None:
    """The per-run batch button reads "Per-run seeds…", matching the signpost.

    The signpost tells a struggling user to open the per-run warm-start; the
    button it points at must use the same words rather than the old generic
    "Initial Values…" label.
    """
    tab = GlobalFitTab(member_kind="runs")

    assert tab._initial_values_btn.text() == "Per-run seeds…"
    assert tab._open_initial_values_from_signpost_btn.text() == "Open per-run seeds…"


def test_signpost_shows_and_suggests_seeds_on_collapse(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    results = _attach_trend(tab)

    tab._update_seeding_signpost(_MODEL, results)

    assert tab._seeding_signpost.isVisibleTo(tab._seeding_signpost.parentWidget() or tab)
    # The collapsed run is offered a descending frequency warm-start.
    assert 2944 in tab._suggested_series_seeds
    assert "frequency" in tab._suggested_series_seeds[2944]
    assert tab._apply_suggested_seeds_btn.isEnabled()


def test_signpost_hidden_on_clean_trend(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    time = np.linspace(0.1, 8.0, 8)
    tab._datasets = [
        MuonDataset(
            time=time,
            asymmetry=np.zeros_like(time),
            error=np.ones_like(time),
            metadata={"run_number": 100 + i, "temperature": float(i)},
        )
        for i in range(6)
    ]
    tab._current_model = _MODEL
    clean = {100 + i: _result(25.0 - i, 30.0 - 2.0 * i) for i in range(6)}

    tab._update_seeding_signpost(_MODEL, clean)

    assert tab._seeding_signpost.isHidden()
    assert tab._suggested_series_seeds == {}


def test_apply_suggested_seeds_fills_initial_values_and_switches_mode(
    qapp: QApplication,
) -> None:
    tab = GlobalFitTab(member_kind="runs")
    results = _attach_trend(tab)
    tab._update_seeding_signpost(_MODEL, results)
    assert tab._suggested_series_seeds  # precondition

    reran: list[bool] = []
    tab._run_global_fit = lambda: reran.append(True)  # isolate the merge/switch
    emitted: list[str] = []
    tab.batch_seeding_mode_changed.connect(emitted.append)

    tab._apply_suggested_series_seeds()

    # Seeds merged into the per-run seed table.
    assert 2944 in tab._user_initial_values_by_run
    assert "frequency" in tab._user_initial_values_by_run[2944]
    # Switched to Independent seeds (honours per-run seeds) and synced the menu.
    assert tab._batch_seeding_mode == "as_provided"
    assert emitted == ["as_provided"]
    # The signpost is dismissed and the batch re-runs.
    assert tab._seeding_signpost.isHidden()
    assert reran == [True]


def test_signpost_skips_short_batches(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    tab._current_model = _MODEL
    tab._datasets = []
    tab._update_seeding_signpost(_MODEL, {1: _result(25.0, 30.0), 2: _result(0.1, 30.0)})
    assert tab._seeding_signpost.isHidden()
