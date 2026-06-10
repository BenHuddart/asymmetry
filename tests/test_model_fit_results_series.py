"""Phase 3 of the model-fit follow-ons: cross-group fit outputs recorded as a
trendable computed FitSeries (results-table recursion).

These exercise the MainWindow integration that turns a cross-group fit's local
and global parameters into a model-less ``FitSeries`` in the project model, so
the outputs of a trend fit can themselves be trended.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.fitting.parameter_models import (  # noqa: E402
    CrossGroupFitResult,
    ParameterCompositeModel,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.core.representation import RepresentationType  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    win = MainWindow()
    # Force a representation so the trend panel accepts recorded series.
    win._active_representation_type = lambda: RepresentationType.TIME_FB_ASYMMETRY
    return win


def _cross_group_result() -> CrossGroupFitResult:
    """A successful two-group fit: shared global 'm', per-group local 'b'."""
    g0 = ParameterSet([Parameter(name="b", value=1.0)])
    g1 = ParameterSet([Parameter(name="b", value=3.0)])
    return CrossGroupFitResult(
        success=True,
        chi_squared=8.0,
        reduced_chi_squared=1.1,
        global_parameters=ParameterSet([Parameter(name="m", value=2.0)]),
        local_parameters={"g0": g0, "g1": g1},
        global_uncertainties={"m": 0.05},
        local_uncertainties={"g0": {"b": 0.1}, "g1": {"b": 0.2}},
        message="Fit successful",
    )


def _groups():
    return [
        SimpleNamespace(group_id="g0", group_name="A", group_variable_value=10.0),
        SimpleNamespace(group_id="g1", group_name="B", group_variable_value=20.0),
    ]


def test_records_trendable_results_series(mainwindow: MainWindow) -> None:
    output = SimpleNamespace(fit_result=_cross_group_result(), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)

    modelfit = [
        s for s in mainwindow._project_model.batches.values() if s.batch_id.startswith("modelfit-")
    ]
    assert len(modelfit) == 1
    series = modelfit[0]
    assert series.is_computed  # no canonical model
    assert series.display_name("x") == "Model fit: lambda vs field"

    rows = mainwindow._build_series_rows(series)
    # Two group rows + one globals row.
    assert len(rows) == 3
    group_rows = [r for r in rows if r["run_label"] in {"A", "B"}]
    globals_rows = [r for r in rows if r["run_label"] == "globals"]
    assert len(group_rows) == 2 and len(globals_rows) == 1

    # Field fit -> local params trend against temperature (the orthogonal axis).
    by_label = {r["run_label"]: r for r in group_rows}
    assert by_label["A"]["temperature"] == pytest.approx(10.0)
    assert by_label["B"]["temperature"] == pytest.approx(20.0)
    # Local 'b' differs per group; shared global 'm' is a constant column.
    assert by_label["A"]["values"]["b"] == pytest.approx(1.0)
    assert by_label["B"]["values"]["b"] == pytest.approx(3.0)
    assert by_label["A"]["values"]["m"] == pytest.approx(2.0)
    assert by_label["A"]["errors"]["b"] == pytest.approx(0.1)
    # Globals row carries χ²ᵣ and is off the trend axis.
    assert "chi2_r" in globals_rows[0]["values"]
    assert np.isnan(globals_rows[0]["temperature"])


def test_recursion_trend_of_results(mainwindow: MainWindow) -> None:
    """The headline acceptance: trend the outputs of a trend fit."""
    output = SimpleNamespace(fit_result=_cross_group_result(), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    series = next(
        s for s in mainwindow._project_model.batches.values() if s.batch_id.startswith("modelfit-")
    )
    rows = [r for r in mainwindow._build_series_rows(series) if r["run_label"] in {"A", "B"}]

    x = np.array([r["temperature"] for r in rows], dtype=float)
    y = np.array([r["values"]["b"] for r in rows], dtype=float)
    yerr = np.array([r["errors"]["b"] for r in rows], dtype=float)
    model = ParameterCompositeModel(["Linear"], [])
    params = ParameterSet([Parameter(name=n, value=0.0) for n in model.param_names])
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success  # a trend fit on the trend outputs runs


def test_rerun_replaces_results_series(mainwindow: MainWindow) -> None:
    output = SimpleNamespace(fit_result=_cross_group_result(), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    modelfit = [
        s for s in mainwindow._project_model.batches.values() if s.batch_id.startswith("modelfit-")
    ]
    assert len(modelfit) == 1  # replaced, not duplicated


def test_results_series_survives_trend_panel_refresh(mainwindow: MainWindow) -> None:
    """The series is first-class in the project model, so a trend-panel refresh
    (the pull path that wipes panel-only series) keeps it."""
    output = SimpleNamespace(fit_result=_cross_group_result(), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    mainwindow._refresh_trend_panel()
    panel_ids = set(mainwindow._fit_parameters_panel._group_fit_results.keys())
    modelfit_ids = {
        s.batch_id
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-")
    }
    assert modelfit_ids
    assert modelfit_ids <= panel_ids
