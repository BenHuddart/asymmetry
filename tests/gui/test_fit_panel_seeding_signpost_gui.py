"""Batch-fit seeding advice on the results card (GUI).

When a batch's ν(T)/A(T) trend shows the near-transition collapse/outlier signature,
the batch tab must say so on its results card and arm "Use as seeds" with the
descending-frequency seeds the diagnostics computed — automating the proven manual
cure for the EuO bistability instead of leaving the user with a corrupted trend and
no guidance.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit.global_tab import (
    USE_AS_SEEDS_ACTION,
    FitLaunch,
    GlobalFitTab,
)

pytestmark = [pytest.mark.gui]

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


def _launch(tab: GlobalFitTab) -> FitLaunch:
    """The launch the completion is read against: the stub model and members."""
    return FitLaunch(model=_MODEL, global_params=(), datasets=tuple(tab._datasets))


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
    return {run: _result(amp, freq) for run, (_t, amp, freq) in _TREND.items()}


def test_per_run_seeds_button_names_the_warm_start(qapp: QApplication) -> None:
    """The seeding row's button reads "Per-run seeds…" — what the advice points at."""
    tab = GlobalFitTab(member_kind="runs")

    assert tab._initial_values_btn.text() == "Per-run seeds…"


def test_advice_names_the_collapse_and_suggests_seeds(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    results = _attach_trend(tab)

    advice = tab._series_seeding_advice(_launch(tab), results)

    assert "trend has outliers" in advice
    # The collapsed run is offered a descending frequency warm-start.
    assert 2944 in tab._suggested_series_seeds
    assert "frequency" in tab._suggested_series_seeds[2944]


def test_converged_batch_card_carries_the_advice_and_arms_use_as_seeds(
    qapp: QApplication,
) -> None:
    """The signpost's sentence is the card's body, and it arms the hand-off."""
    tab = GlobalFitTab(member_kind="runs")
    results = _attach_trend(tab)
    # The stub model carries no callable; the card is what is under test.
    tab._results_with_curves = lambda model, results_dict, datasets: {}

    tab._on_fit_finished(_launch(tab), results, ParameterSet())

    assert "trend has outliers" in tab._results_card.content_html()
    assert tab._results_card._actions[USE_AS_SEEDS_ACTION].isEnabled()


def test_no_advice_on_a_clean_trend(qapp: QApplication) -> None:
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
    clean = {100 + i: _result(25.0 - i, 30.0 - 2.0 * i) for i in range(6)}

    assert tab._series_seeding_advice(_launch(tab), clean) == ""
    assert tab._suggested_series_seeds == {}


def test_apply_suggested_seeds_fills_initial_values_and_switches_mode(
    qapp: QApplication,
) -> None:
    tab = GlobalFitTab(member_kind="runs")
    results = _attach_trend(tab)
    tab._series_seeding_advice(_launch(tab), results)
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
    # The batch re-runs with the warm start.
    assert reran == [True]


def test_advice_skips_short_batches(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    tab._datasets = []
    short = {1: _result(25.0, 30.0), 2: _result(0.1, 30.0)}
    assert tab._series_seeding_advice(_launch(tab), short) == ""
