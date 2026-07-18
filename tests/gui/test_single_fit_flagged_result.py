"""A converged-but-flagged single fit is surfaced (greyed curve + message).

MINUIT can return a perfectly plottable minimum yet set ``success=False`` (an
invalid-minimum flag, e.g. a degenerate baseline). The single-fit tab used to
early-return on ``not result.success``, silently drawing nothing. It now draws
the curve via the preview overlay and explains the flag, while still refusing to
*record* the fit. Regression for the maleic-acid corpus sweep.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.core.fitting.composite import CompositeModel  # noqa: E402
from asymmetry.core.fitting.engine import FitResult  # noqa: E402
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.gui.panels.fit.single_tab import SingleFitTab  # noqa: E402
from asymmetry.gui.panels.fit.tab_base import _fit_result_is_usable  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dataset() -> MuonDataset:
    t = np.linspace(0.0, 4.0, 80)
    a = 0.2 * np.exp(-0.4 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 101})


def _params() -> ParameterSet:
    return ParameterSet(
        [
            Parameter("A_1", 0.2, min=0.0, max=1.0),
            Parameter("Lambda", 0.4, min=0.0, max=5.0),
            Parameter("A_bg", 0.01, min=-0.5, max=0.5),
        ]
    )


def _flagged_result() -> FitResult:
    # success=False but a fully usable minimum: finite params + finite chi².
    return FitResult(
        success=False,
        chi_squared=12.0,
        reduced_chi_squared=1.6,
        parameters=_params(),
        uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
        message="Fit failed: invalid minimum",
    )


# --- pure predicate ----------------------------------------------------------


def test_flagged_but_finite_result_is_usable() -> None:
    assert _fit_result_is_usable(_flagged_result()) is True


def test_result_without_parameters_is_not_usable() -> None:
    assert _fit_result_is_usable(FitResult(success=False, message="iminuit import error")) is False


def test_result_with_nonfinite_value_is_not_usable() -> None:
    params = ParameterSet([Parameter("A_1", float("nan"), min=0.0, max=1.0)])
    assert _fit_result_is_usable(FitResult(success=False, parameters=params)) is False


# --- tab behaviour -----------------------------------------------------------


def _prime_tab(tab: SingleFitTab, dataset: MuonDataset) -> CompositeModel:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    tab._current_dataset = dataset
    tab._composite_model = model
    return model


def test_flagged_fit_draws_greyed_preview_and_warns(app: QApplication) -> None:
    tab = SingleFitTab()
    dataset = _dataset()
    model = _prime_tab(tab, dataset)

    previews: list[object] = []
    completed: list[object] = []
    tab.preview_requested.connect(lambda *a: previews.append(a))
    tab.fit_completed.connect(lambda *a: completed.append(a))

    tab._apply_single_fit_result(
        _flagged_result(), _params(), dataset, model, tab._model_generation
    )

    # A greyed preview curve is drawn; nothing is recorded (no fit_completed,
    # no cached converged result — so Add-to-Series/pull-diagnostic stay off).
    assert len(previews) == 1
    assert completed == []
    assert tab._last_fit_result is None
    text = tab._result_label.text()
    assert "did not fully converge" in text
    assert "not recorded" in text
    tab.deleteLater()


def test_genuine_failure_draws_nothing(app: QApplication) -> None:
    tab = SingleFitTab()
    dataset = _dataset()
    model = _prime_tab(tab, dataset)

    previews: list[object] = []
    tab.preview_requested.connect(lambda *a: previews.append(a))

    tab._apply_single_fit_result(
        FitResult(success=False, message="iminuit import error"),
        _params(),
        dataset,
        model,
        tab._model_generation,
    )

    assert previews == []
    assert "Fit failed" in tab._result_label.text()
    tab.deleteLater()
