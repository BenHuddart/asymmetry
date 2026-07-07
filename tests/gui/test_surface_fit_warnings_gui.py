"""GUI surfacing for the engine's advisory fit warnings (fix/surface-fit-warnings).

The engine carries its advisory warnings (the percent/fraction scale trap and the
fixed-frequency σ-inflation trap) on ``FitResult.warnings``; these tests pin that the
fit panel *renders* that list — the single-fit result box and the batch-success box.
The engine itself is stubbed so the surfacing path is tested independently of whether
a given dataset happens to trip a guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab

pytestmark = [pytest.mark.gui]

_WARNING = (
    "Fixed-frequency trap: parameter 'frequency' is pinned at 6 MHz, ~11% away from "
    "γ_μ·B = 5.422 MHz implied by the run's 400 G field."
)


def _dataset(run_number: int = 1) -> MuonDataset:
    t = np.linspace(0.0, 8.0, 200)
    err = np.full_like(t, 0.01)
    y = 0.2 * np.exp(-0.5 * t)
    return MuonDataset(time=t, asymmetry=y, error=err, metadata={"run_number": run_number})


def _result_with_warning(model, *, warning: str = _WARNING) -> FitResult:
    names = list(model.param_names)
    return FitResult(
        success=True,
        chi_squared=190.0,
        reduced_chi_squared=1.0,
        dof=190,
        parameters=ParameterSet([Parameter(name=p, value=1.0) for p in names]),
        uncertainties={p: 0.01 for p in names},
        warnings=[warning],
    )


def test_single_fit_surfaces_engine_warning(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_dataset(_dataset())
    model = tab._composite_model

    def _fit(ds, model_fn, parameters, *, minos=False, cancel_callback=None):
        return _result_with_warning(model)

    tab._fit_engine = SimpleNamespace(fit=_fit)
    tab._run_fit()
    assert tab.wait_for_fit()

    rendered = tab._result_label.text()
    # The result box shows the converged line AND the advisory warning beneath it.
    assert "Fit converged" in rendered
    assert "⚠" in rendered
    assert "Fixed-frequency trap" in rendered


def test_batch_fit_surfaces_engine_warnings_deduped(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    model = tab._composite_model
    # Two runs that both trip the same guard: the message must render once.
    results_dict = {
        1: _result_with_warning(model),
        2: _result_with_warning(model),
    }

    tab._emit_global_fit_success(
        model=model,
        results_dict=results_dict,
        fitted_global=ParameterSet(),
        global_param_names=[],
    )

    rendered = tab._result_text.toHtml()
    assert "Batch fit converged" in rendered
    assert "Fixed-frequency trap" in rendered
    # Deduped: the identical warning fired for both runs but is shown a single time.
    assert rendered.count("Fixed-frequency trap") == 1
