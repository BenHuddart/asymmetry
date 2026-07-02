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
        s
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-") and not s.batch_id.startswith("modelfit-globals")
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
        s
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-") and not s.batch_id.startswith("modelfit-globals")
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
    first_id = next(
        s.batch_id
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-") and not s.batch_id.startswith("modelfit-globals")
    )
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    modelfit = [
        s
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-") and not s.batch_id.startswith("modelfit-globals")
    ]
    assert len(modelfit) == 1  # replaced, not duplicated
    # The id is a deterministic function of the logical key (stable across
    # sessions), so the re-run lands on the same batch and replaces it.
    assert modelfit[0].batch_id == first_id


def test_results_series_is_json_safe(mainwindow: MainWindow) -> None:
    """The computed series must serialise to strict JSON — the off-axis globals
    coordinate is stored as null, not a non-standard NaN token."""
    import json

    output = SimpleNamespace(fit_result=_cross_group_result(), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    series = next(
        s
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-") and not s.batch_id.startswith("modelfit-globals")
    )
    # allow_nan=False raises ValueError if any NaN/Infinity is present.
    json.dumps(series.to_dict(), allow_nan=False)


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
        if s.batch_id.startswith("modelfit-") and not s.batch_id.startswith("modelfit-globals")
    }
    assert modelfit_ids
    assert modelfit_ids <= panel_ids


def test_restore_shows_only_active_representation_chips(
    mainwindow: MainWindow, qapp: QApplication, tmp_path
) -> None:
    """F21a/b: on project load the trend panel must show the *active*
    representation's chips, re-derived from the project model — not whichever
    representation's chips happened to be serialized in the panel cache.

    Reproduces the audit symptom: the F-B Parameters view showed foreign chips
    (an integral scan / a frequency model fit) while its own series chip was
    missing, because the serialized panel cache holds only one representation
    and was restored blindly.
    """
    output = SimpleNamespace(fit_result=_cross_group_result(), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), output)
    mainwindow._refresh_trend_panel()
    active_ids = {s.batch_id for s in mainwindow._project_model.batches.values()}
    assert active_ids  # the active representation genuinely has series

    state = mainwindow.collect_project_state()

    # Inject a stale foreign-representation chip into the serialized trend-panel
    # cache. It is NOT a batch of the project model, so a correct restore must
    # drop it rather than surface it in the active representation's view.
    foreign_id = "freq-foreign-chip"
    state["fit_parameters_state"]["group_fit_results"][foreign_id] = {
        "group_id": foreign_id,
        "group_name": "GaussianPeak · 2960",
        "global_params": None,
        "rows": [
            {
                "run_number": 2960,
                "run_label": "2960",
                "field": 0.0,
                "temperature": 5.0,
                "values": {"nu0": 30.0},
                "errors": {"nu0": 0.1},
                "combined_from": None,
                "covariance": None,
                "custom_values": {},
            }
        ],
        "varying_params": ["nu0"],
        "composite_parameters": [],
        "inferred_x_key": "temperature",
        "model_fits": {},
        "plot_annotations": [],
        "global_param_uncertainties": {},
    }

    restored = MainWindow()
    restored._active_representation_type = lambda: RepresentationType.TIME_FB_ASYMMETRY
    restored.restore_project_state(state, str(tmp_path / "roundtrip.asymp"))

    panel_ids = set(restored._fit_parameters_panel._group_fit_results.keys())
    assert foreign_id not in panel_ids  # stale foreign chip dropped
    assert active_ids <= panel_ids  # own series re-derived from the project model


# ---------------------------------------------------------------------------
# Phase D — cross-fit global accumulation (Global summary series)
# ---------------------------------------------------------------------------


def _result_with_globals(m_value: float, chi2r: float = 1.0) -> CrossGroupFitResult:
    return CrossGroupFitResult(
        success=True,
        chi_squared=4.0,
        reduced_chi_squared=chi2r,
        global_parameters=ParameterSet([Parameter(name="m", value=m_value)]),
        local_parameters={"g0": ParameterSet([Parameter(name="b", value=1.0)])},
        global_uncertainties={"m": 0.05},
        local_uncertainties={"g0": {"b": 0.1}},
        message="Fit successful",
    )


def _accumulator(win: MainWindow):
    accum = [
        s for s in win._project_model.batches.values() if s.batch_id.startswith("modelfit-globals")
    ]
    return accum[0] if accum else None


def test_global_summary_accumulates_distinct_fits(mainwindow: MainWindow) -> None:
    """Two distinct cross-group fits add two members to the singleton Global
    summary series, each carrying that fit's globals + χ²ᵣ + fit_index."""
    mainwindow._record_model_fit_results_series(
        "lambda", _groups(), SimpleNamespace(fit_result=_result_with_globals(2.0), x_key="field")
    )
    mainwindow._record_model_fit_results_series(
        "nu", _groups(), SimpleNamespace(fit_result=_result_with_globals(5.0), x_key="temperature")
    )

    accum = _accumulator(mainwindow)
    assert accum is not None
    assert accum.display_name("x") == "Global summary"
    assert accum.is_computed

    rows = mainwindow._build_series_rows(accum)
    assert len(rows) == 2
    # Each row carries its fit's global 'm', χ²ᵣ and a distinct fit_index.
    m_values = sorted(r["values"]["m"] for r in rows)
    assert m_values == pytest.approx([2.0, 5.0])
    indices = sorted(r["values"]["fit_index"] for r in rows)
    assert indices == pytest.approx([1.0, 2.0])
    assert all("chi2_r" in r["values"] for r in rows)
    # Rows sit off both physical axes (trended against fit_index / a global).
    assert all(np.isnan(r["field"]) and np.isnan(r["temperature"]) for r in rows)


def test_global_summary_is_trendable_against_fit_index(mainwindow: MainWindow) -> None:
    """The headline acceptance for item D: a global parameter is trendable
    across successive fits (vs fit_index)."""
    for idx, m in enumerate((1.0, 2.0, 3.0)):
        mainwindow._record_model_fit_results_series(
            f"p{idx}",
            _groups(),
            SimpleNamespace(fit_result=_result_with_globals(m), x_key="field"),
        )
    accum = _accumulator(mainwindow)
    rows = mainwindow._build_series_rows(accum)
    rows.sort(key=lambda r: r["values"]["fit_index"])

    x = np.array([r["values"]["fit_index"] for r in rows], dtype=float)
    y = np.array([r["values"]["m"] for r in rows], dtype=float)
    yerr = np.array([r["errors"].get("m", 0.05) for r in rows], dtype=float)
    model = ParameterCompositeModel(["Linear"], [])
    params = ParameterSet([Parameter(name=n, value=0.0) for n in model.param_names])
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    assert len(x) == 3


def test_global_summary_rerun_replaces_row(mainwindow: MainWindow) -> None:
    """Re-running an existing fit updates its row in place (count unchanged,
    fit_index preserved); a genuinely new fit appends (fit_index = max + 1)."""
    out_a = SimpleNamespace(fit_result=_result_with_globals(2.0), x_key="field")
    mainwindow._record_model_fit_results_series("lambda", _groups(), out_a)
    mainwindow._record_model_fit_results_series(
        "nu", _groups(), SimpleNamespace(fit_result=_result_with_globals(5.0), x_key="field")
    )
    accum = _accumulator(mainwindow)
    assert len(accum.member_run_numbers) == 2

    # Re-run fit A with a different global value: same logical key -> same row.
    mainwindow._record_model_fit_results_series(
        "lambda", _groups(), SimpleNamespace(fit_result=_result_with_globals(2.5), x_key="field")
    )
    accum = _accumulator(mainwindow)
    assert len(accum.member_run_numbers) == 2  # replaced, not appended
    rows = {r["run_label"]: r for r in mainwindow._build_series_rows(accum)}
    lam = rows["lambda vs field"]
    assert lam["values"]["m"] == pytest.approx(2.5)  # updated value
    assert lam["values"]["fit_index"] == pytest.approx(1.0)  # original position kept


def test_global_summary_roundtrips_and_is_json_safe(mainwindow: MainWindow) -> None:
    """The accumulator persists (computed series) and serialises to strict JSON;
    a reload preserves both rows and their fit_index."""
    import json

    mainwindow._record_model_fit_results_series(
        "lambda", _groups(), SimpleNamespace(fit_result=_result_with_globals(2.0), x_key="field")
    )
    mainwindow._record_model_fit_results_series(
        "nu", _groups(), SimpleNamespace(fit_result=_result_with_globals(5.0), x_key="field")
    )
    accum = _accumulator(mainwindow)
    # Strict JSON: off-axis coordinates stored as null, never a NaN token.
    json.dumps(accum.to_dict(), allow_nan=False)

    from asymmetry.core.representation.series import FitSeries

    restored = FitSeries.from_dict(accum.to_dict())
    rows = mainwindow._build_series_rows(restored)
    assert len(rows) == 2
    assert sorted(r["values"]["fit_index"] for r in rows) == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# Phase B — single-fit-range export (one trendable row per ModelFitRange)
# ---------------------------------------------------------------------------


def _single_fit_two_ranges() -> object:
    """A single-series ParameterModelFit with two windows over one trace."""
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterModelFit,
        ParameterModelFitResult,
    )

    model = ParameterCompositeModel(["Linear"], [])

    def _range(x_min, x_max, m_val):
        res = ParameterModelFitResult(
            success=True,
            chi_squared=2.0,
            reduced_chi_squared=1.0,
            parameters=ParameterSet(
                [Parameter(name="m", value=m_val), Parameter(name="b", value=0.5)]
            ),
            uncertainties={"m": 0.05, "b": 0.1},
        )
        params = ParameterSet([Parameter(name="m", value=m_val), Parameter(name="b", value=0.5)])
        return ModelFitRange(x_min=x_min, x_max=x_max, model=model, parameters=params, result=res)

    return ParameterModelFit(
        parameter_name="lambda",
        x_key="temperature",
        ranges=[_range(0.0, 40.0, 2.0), _range(60.0, 100.0, 3.0)],
    )


def _single_series(win: MainWindow):
    matches = [
        s for s in win._project_model.batches.values() if s.batch_id.startswith("modelfit-single-")
    ]
    return matches[0] if matches else None


def test_single_fit_ranges_become_trendable_rows(mainwindow: MainWindow) -> None:
    """A two-range single fit yields a results series with one row per range,
    each carrying that range's params + χ²ᵣ, indexed by the window centre."""
    fit = _single_fit_two_ranges()
    mainwindow._record_single_model_fit_results_series("lambda", "temperature", fit)

    series = _single_series(mainwindow)
    assert series is not None
    assert series.is_computed
    assert series.display_name("x") == "Model fit (single): lambda vs temperature"

    rows = mainwindow._build_series_rows(series)
    assert len(rows) == 2
    by_center = {round(r["values"]["range_center"], 6): r for r in rows}
    assert set(by_center) == {20.0, 80.0}  # window centres
    # The x_key is temperature, so the centre lands on the temperature axis.
    assert by_center[20.0]["temperature"] == pytest.approx(20.0)
    assert by_center[80.0]["temperature"] == pytest.approx(80.0)
    # Each row carries that range's fitted params + χ²ᵣ.
    assert by_center[20.0]["values"]["m"] == pytest.approx(2.0)
    assert by_center[80.0]["values"]["m"] == pytest.approx(3.0)
    assert by_center[20.0]["values"]["chi2_r"] == pytest.approx(1.0)
    assert by_center[20.0]["errors"]["m"] == pytest.approx(0.05)


def test_single_fit_results_recursion(mainwindow: MainWindow) -> None:
    """Headline acceptance for item B: trend a single fit's per-range outputs
    (a fitted parameter vs the window centre)."""
    fit = _single_fit_two_ranges()
    mainwindow._record_single_model_fit_results_series("lambda", "temperature", fit)
    series = _single_series(mainwindow)
    rows = mainwindow._build_series_rows(series)
    rows.sort(key=lambda r: r["values"]["range_center"])

    x = np.array([r["values"]["range_center"] for r in rows], dtype=float)
    y = np.array([r["values"]["m"] for r in rows], dtype=float)
    yerr = np.array([r["errors"]["m"] for r in rows], dtype=float)
    model = ParameterCompositeModel(["Linear"], [])
    params = ParameterSet([Parameter(name=n, value=0.0) for n in model.param_names])
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    assert len(x) == 2


def test_single_fit_results_rerun_replaces(mainwindow: MainWindow) -> None:
    fit = _single_fit_two_ranges()
    mainwindow._record_single_model_fit_results_series("lambda", "temperature", fit)
    first_id = _single_series(mainwindow).batch_id
    mainwindow._record_single_model_fit_results_series("lambda", "temperature", fit)
    matches = [
        s
        for s in mainwindow._project_model.batches.values()
        if s.batch_id.startswith("modelfit-single-")
    ]
    assert len(matches) == 1  # replaced, not duplicated
    assert matches[0].batch_id == first_id


def test_single_fit_results_roundtrip_json_safe(mainwindow: MainWindow) -> None:
    import json

    fit = _single_fit_two_ranges()
    mainwindow._record_single_model_fit_results_series("lambda", "temperature", fit)
    series = _single_series(mainwindow)
    json.dumps(series.to_dict(), allow_nan=False)

    from asymmetry.core.representation.series import FitSeries

    restored = FitSeries.from_dict(series.to_dict())
    rows = mainwindow._build_series_rows(restored)
    assert len(rows) == 2
    assert sorted(round(r["values"]["range_center"], 6) for r in rows) == [20.0, 80.0]
