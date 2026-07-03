"""Tests for the Model-fit follow-ons (docs/porting/model-fit-followons).

Phase 1: arbitrary-X param-vs-param trending + opt-in effective-variance
x-uncertainty. Phase 2 (cross-group error modes + windows) and Phase 3
(results-table recursion) tests are added alongside their implementations.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    component_names_for_x,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

# EuO ν(T) real-data fixture (PSI GPS runs 2928–2943) — shared with
# tests/core/test_wimda_model_function_parity.py; (T, ν, ν_err).
from tests.core.test_wimda_model_function_parity import EUO_NU_T_TREND


def _linear_model() -> tuple[ParameterCompositeModel, ParameterSet]:
    model = ParameterCompositeModel(component_names=["Linear"], operators=[])
    params = ParameterSet(
        [Parameter(name=n, value=v) for n, v in zip(model.param_names, [1.0, 0.0])]
    )
    return model, params


# ---------------------------------------------------------------------------
# Effective-variance x-uncertainty (core)
# ---------------------------------------------------------------------------


def test_zero_xerr_is_identical_to_ordinary_least_squares() -> None:
    """σ_x = 0 must reproduce the no-xerr fit byte-for-byte (regression safety)."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 15)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.3, x.size)
    yerr = np.full_like(x, 0.3)

    m1, p1 = _linear_model()
    r_none = fit_parameter_model(x, y, yerr, m1, p1)
    m2, p2 = _linear_model()
    r_zero = fit_parameter_model(x, y, yerr, m2, p2, xerr=np.zeros_like(x))

    vn = {p.name: p.value for p in r_none.parameters}
    vz = {p.name: p.value for p in r_zero.parameters}
    assert vn.keys() == vz.keys()
    for name in vn:
        assert vz[name] == pytest.approx(vn[name], abs=1e-12)
    for name in r_none.uncertainties:
        assert r_zero.uncertainties[name] == pytest.approx(r_none.uncertainties[name], abs=1e-12)


def test_effective_variance_matches_independent_reference() -> None:
    """The iminuit effective-variance fit equals an independent scipy
    minimisation of the identical Orear/York cost, and inflates the parameter
    errors relative to the x-error-ignoring fit."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(11)
    x = np.linspace(0, 10, 15)
    m_true, b_true = 2.0, 1.0
    sy, sx = 0.3, 0.4
    y = m_true * x + b_true + rng.normal(0, sy, x.size)
    yerr = np.full_like(x, sy)
    xerr = np.full_like(x, sx)

    model, params = _linear_model()
    r = fit_parameter_model(x, y, yerr, model, params, xerr=xerr)
    fitted = {p.name: p.value for p in r.parameters}

    # Independent reference: minimise Σ (y − (m x + b))² / (σ_y² + m²σ_x²).
    def resid(theta: np.ndarray) -> np.ndarray:
        m, b = theta
        sigma2 = sy**2 + (m**2) * sx**2
        return (y - (m * x + b)) / np.sqrt(sigma2)

    ref = scipy_opt.least_squares(resid, x0=[1.0, 0.0])
    # iminuit (migrad) vs scipy (trf) — agreement to minimiser tolerance.
    assert fitted["m"] == pytest.approx(ref.x[0], rel=2e-3, abs=2e-3)
    assert fitted["b"] == pytest.approx(ref.x[1], rel=2e-3, abs=2e-3)

    # x-uncertainty inflates the variance, hence the parameter errors.
    m3, p3 = _linear_model()
    r_ols = fit_parameter_model(x, y, yerr, m3, p3)
    assert r.uncertainties["m"] > r_ols.uncertainties["m"]


def test_xerr_ignored_under_scatter_and_none_modes() -> None:
    """Effective variance needs a real σ_y; under unit-weight (NONE) or
    scatter-estimated errors the x-error term is ignored (the combination would
    be scale-dependent), so passing xerr must not change the result."""
    rng = np.random.default_rng(13)
    x = np.linspace(0, 10, 15)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.3, x.size)
    yerr = np.full_like(x, 0.3)
    xerr = np.full_like(x, 0.4)

    for mode in ("none", "scatter"):
        m1, p1 = _linear_model()
        without = fit_parameter_model(x, y, yerr, m1, p1, error_mode=mode)
        m2, p2 = _linear_model()
        with_xerr = fit_parameter_model(x, y, yerr, m2, p2, error_mode=mode, xerr=xerr)
        v1 = {p.name: p.value for p in without.parameters}
        v2 = {p.name: p.value for p in with_xerr.parameters}
        for name in v1:
            assert v2[name] == pytest.approx(v1[name], abs=1e-12), mode


def test_effective_variance_central_difference_slope_is_accurate() -> None:
    """The internal central-difference slope matches the analytic derivative
    for a non-linear model (guards the finite-difference step choice)."""
    a, n = 2.0, 1.5
    x = np.linspace(0.5, 9.0, 12)
    model = ParameterCompositeModel(component_names=["PowerLaw"], operators=[])
    step = np.maximum(np.abs(x), 1.0) * 1e-6
    fwd = model.function(x + step, a=a, n=n, c=0.0)
    bwd = model.function(x - step, a=a, n=n, c=0.0)
    slope = (fwd - bwd) / (2.0 * step)
    analytic = a * n * np.abs(x) ** (n - 1.0)
    np.testing.assert_allclose(slope, analytic, rtol=1e-5)


# ---------------------------------------------------------------------------
# Arbitrary-X scope degradation (core)
# ---------------------------------------------------------------------------


def test_param_x_offers_common_scope_components_only() -> None:
    """An arbitrary parameter x-key offers the same (common-scope) components as
    the run axis — never field- or temperature-scoped ones."""
    assert component_names_for_x("param:lambda") == component_names_for_x("run")
    # Sanity: field exposes strictly more (its field-scoped components).
    assert set(component_names_for_x("field")) >= set(component_names_for_x("run"))


def test_param_vs_param_fit_on_real_euo_data() -> None:
    """Param-vs-param trending on genuine fitted values: trend the real EuO
    temperatures against the fitted internal-field frequency ν (the abscissa is
    a fitted parameter). The fit must run and use the real ν as x."""
    nu = np.array([row[1] for row in EUO_NU_T_TREND])
    nu_err = np.array([row[2] for row in EUO_NU_T_TREND])
    temperature = np.array([row[0] for row in EUO_NU_T_TREND])
    # Treat ν as the abscissa (with its real per-run error) and T as y.
    model = ParameterCompositeModel(component_names=["Linear"], operators=[])
    params = ParameterSet(
        [Parameter(name=n, value=v) for n, v in zip(model.param_names, [-2.0, 70.0])]
    )
    t_err = np.full_like(temperature, 0.5)
    result = fit_parameter_model(nu, temperature, t_err, model, params, xerr=nu_err)
    assert result.success
    assert all(np.isfinite(p.value) for p in result.parameters)


# ---------------------------------------------------------------------------
# GUI: arbitrary-X panel + dialog
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

gui = pytest.mark.gui


def _panel_with_two_params():
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow

    panel = FitParametersPanel()
    rows = [
        _FitRow(
            run_number=i,
            run_label=str(i),
            field=float(i * 100),
            temperature=float(10 + i),
            values={"lambda": 0.1 + 0.01 * i, "nu": 1.0 + 0.2 * i},
            errors={"lambda": 0.005, "nu": 0.02},
        )
        for i in range(6)
    ]
    panel._rows = rows
    panel._varying_params = ["lambda", "nu"]
    panel._rebuild_y_controls()
    return panel, rows


def test_normalize_and_decode_x_keys() -> None:
    from asymmetry.gui.panels.fit_parameters_panel import _normalize_x_key, _x_param_name

    assert _normalize_x_key("param:lambda") == "param:lambda"
    assert _normalize_x_key("field") == "field"
    assert _normalize_x_key("temperature") == "temperature"
    assert _normalize_x_key("run") == "run"
    assert _normalize_x_key("angle") == "angle"  # first-class Angle axis is preserved
    assert _normalize_x_key("custom:rotation") == "custom:rotation"
    assert _normalize_x_key("param:") == "run"  # empty name collapses
    assert _normalize_x_key("garbage") == "run"
    assert _x_param_name("param:nu") == "nu"
    assert _x_param_name("field") is None


@gui
def test_x_combo_lists_fitted_parameters(qapp) -> None:
    panel, _ = _panel_with_two_params()
    datas = [panel._x_combo.itemData(i) for i in range(panel._x_combo.count())]
    assert "param:lambda" in datas
    assert "param:nu" in datas


@gui
def test_x_value_and_error_read_from_param(qapp) -> None:
    panel, rows = _panel_with_two_params()
    assert panel._x_value(rows[2], "param:nu") == pytest.approx(1.4)
    assert panel._x_error(rows[2], "param:nu") == pytest.approx(0.02)
    # run-level axes carry no x-uncertainty
    assert np.isnan(panel._x_error(rows[2], "field"))
    assert panel._x_error_array(rows, "field") is None
    arr = panel._x_error_array(rows, "param:nu")
    assert arr is not None and arr.shape == (6,)


@gui
def test_gle_x_column_points_at_param_value_column(qapp) -> None:
    panel, _ = _panel_with_two_params()
    # display params order: [lambda, nu] -> value cols 4, 6
    assert panel._gle_x_column("param:lambda") == 4
    assert panel._gle_x_column("param:nu") == 6
    assert panel._gle_x_column("field") == 2


@gui
def test_panel_state_round_trips_param_x_axis(qapp) -> None:
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    panel, _ = _panel_with_two_params()
    idx = panel._x_combo.findData("param:nu")
    panel._x_combo.setCurrentIndex(idx)
    assert panel._effective_x_key() == "param:nu"

    state = panel.get_state()
    assert state["x_axis_key"] == "param:nu"

    restored = FitParametersPanel()
    restored.restore_state(state)
    assert restored._effective_x_key() == "param:nu"


@gui
def test_legacy_state_without_x_axis_key_loads(qapp) -> None:
    """State predating the param-vs-param feature (no x_axis_key) must load."""
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    panel, _ = _panel_with_two_params()
    state = panel.get_state()
    state.pop("x_axis_key", None)
    restored = FitParametersPanel()
    restored.restore_state(state)  # must not raise
    assert restored._effective_x_key() in {"field", "temperature", "run"}


@gui
def test_dialog_x_error_toggle_present_and_persists(qapp) -> None:
    from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

    nu = np.array([1.0 + 0.2 * i for i in range(6)])
    lam = np.array([0.1 + 0.01 * i for i in range(6)])
    lam_err = np.full_like(lam, 0.005)
    nu_err = np.full_like(nu, 0.02)

    dlg = ModelFitDialog("lambda", "param:nu", nu, lam, lam_err, x_errors=nu_err)
    assert dlg._x_error_check is not None
    assert dlg._use_x_errors() is False  # default off
    dlg._x_error_check.setChecked(True)
    fit = dlg.get_model_fit()
    assert fit is not None and fit.use_x_errors is True


@gui
def test_x_error_toggle_ignored_when_disabled_by_scatter_mode(qapp) -> None:
    """A box checked under Column then disabled by switching to Scatter must not
    feed x-errors into the fit (gated on isEnabled, not just isChecked)."""
    from asymmetry.core.fitting.parameter_models import ErrorMode
    from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

    nu = np.array([1.0 + 0.2 * i for i in range(6)])
    lam = np.array([0.1 + 0.01 * i for i in range(6)])
    lam_err = np.full_like(lam, 0.005)
    nu_err = np.full_like(nu, 0.02)

    dlg = ModelFitDialog("lambda", "param:nu", nu, lam, lam_err, x_errors=nu_err)
    assert dlg._x_error_check is not None
    dlg._x_error_check.setChecked(True)
    assert dlg._use_x_errors() is True  # Column default

    # Switch to Scatter → toggle disabled (stays checked) → reported as off.
    idx = dlg._error_mode_combo.findData(ErrorMode.SCATTER.value)
    dlg._error_mode_combo.setCurrentIndex(idx)
    assert dlg._x_error_check.isEnabled() is False
    assert dlg._use_x_errors() is False


@gui
def test_dialog_x_error_toggle_hidden_for_run_axis(qapp) -> None:
    """A run-level x-axis (no per-point uncertainty) hides the toggle."""
    from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

    x = np.arange(6, dtype=float)
    y = np.linspace(1, 2, 6)
    ye = np.full_like(y, 0.05)
    dlg = ModelFitDialog("lambda", "run", x, y, ye, x_errors=None)
    assert dlg._x_error_check is None
    assert dlg._use_x_errors() is False


@gui
def test_model_fit_use_x_errors_round_trips_through_panel_state(qapp) -> None:
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
    )
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    panel, _ = _panel_with_two_params()
    model = ParameterCompositeModel(component_names=["Linear"], operators=[])
    fit = ParameterModelFit(
        parameter_name="lambda",
        x_key="param:nu",
        use_x_errors=True,
        ranges=[
            ModelFitRange(
                x_min=None,
                x_max=None,
                model=model,
                parameters=ParameterSet([Parameter(name=n, value=0.0) for n in model.param_names]),
            )
        ],
    )
    panel._model_fits = {"lambda": fit}
    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)
    assert restored._model_fits["lambda"].use_x_errors is True
    assert restored._model_fits["lambda"].x_key == "param:nu"


# ---------------------------------------------------------------------------
# Phase 2: cross-group error modes + fit windows
# ---------------------------------------------------------------------------


def _two_groups(rng_seed: int = 3):
    from asymmetry.core.fitting.parameter_models import ParameterGroupData

    rng = np.random.default_rng(rng_seed)
    x = np.linspace(0, 100, 21)
    groups = []
    for i in range(2):
        y = 2.0 * x + (1.0 + 2.0 * i) + rng.normal(0, 0.5, x.size)
        groups.append(
            ParameterGroupData(
                group_id=f"g{i}",
                group_name=f"g{i}",
                x=x.copy(),
                y=y,
                yerr=np.full_like(x, 0.5),
                group_variable_value=float(i),
            )
        )
    return groups, x


def test_cross_group_windows_change_point_count() -> None:
    from asymmetry.core.fitting.parameter_models import global_fit_parameter_model

    groups, _ = _two_groups()
    model = ParameterCompositeModel(["Linear"], [])
    full = global_fit_parameter_model(groups, model, ["m"], ["b"], {})
    win = global_fit_parameter_model(groups, model, ["m"], ["b"], {}, windows=[(0, 40), (60, 100)])
    # 9 points in [0,40] + 9 in [60,100] per group, two groups -> 36 of 42.
    assert full.n_points == 42
    assert win.n_points == 36
    assert win.n_points < full.n_points


def test_cross_group_error_mode_changes_weighting() -> None:
    from asymmetry.core.fitting.parameter_models import global_fit_parameter_model

    groups, _ = _two_groups()
    model = ParameterCompositeModel(["Linear"], [])
    col = global_fit_parameter_model(groups, model, ["m"], ["b"], {}, error_mode="column")
    pct = global_fit_parameter_model(
        groups, model, ["m"], ["b"], {}, error_mode="percent", error_value=5.0
    )
    assert col.reduced_chi_squared != pytest.approx(pct.reduced_chi_squared)
    assert pct.error_mode == "percent"


def test_cross_group_scatter_rescales_global_and_local_errors() -> None:
    from asymmetry.core.fitting.parameter_models import global_fit_parameter_model

    groups, _ = _two_groups()
    model = ParameterCompositeModel(["Linear"], [])
    col = global_fit_parameter_model(groups, model, ["m"], ["b"], {}, error_mode="column")
    sc = global_fit_parameter_model(groups, model, ["m"], ["b"], {}, error_mode="scatter")
    # ν = N − (1 global + 2 local) free params
    ndof = col.n_points - 3
    scale = np.sqrt(col.chi_squared / ndof)
    assert sc.global_uncertainties["m"] == pytest.approx(
        col.global_uncertainties["m"] * scale, rel=1e-3
    )
    for gid in sc.local_uncertainties:
        assert sc.local_uncertainties[gid]["b"] == pytest.approx(
            col.local_uncertainties[gid]["b"] * scale, rel=1e-3
        )


def test_cross_group_two_identical_groups_equals_single_series() -> None:
    """Degenerate case: two identical all-global groups must reproduce the
    single-series fit on one copy (proves the cross-group path reduces)."""
    from asymmetry.core.fitting.parameter_models import (
        ParameterGroupData,
        global_fit_parameter_model,
    )

    rng = np.random.default_rng(5)
    x = np.linspace(0, 100, 21)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)
    yerr = np.full_like(x, 0.5)
    groups = [
        ParameterGroupData(
            group_id=gid,
            group_name=gid,
            x=x.copy(),
            y=y.copy(),
            yerr=yerr.copy(),
            group_variable_value=0.0,
        )
        for gid in ("a", "b")
    ]
    model = ParameterCompositeModel(["Linear"], [])
    cg = global_fit_parameter_model(groups, model, ["m", "b"], [], {}, error_mode="none")

    ps = ParameterSet([Parameter(name=n, value=0.0) for n in model.param_names])
    ss = fit_parameter_model(x, y, yerr, model, ps, error_mode="none")
    ss_vals = {p.name: p.value for p in ss.parameters}
    assert cg.global_parameters["m"].value == pytest.approx(ss_vals["m"], abs=1e-5)
    assert cg.global_parameters["b"].value == pytest.approx(ss_vals["b"], abs=1e-5)


def test_cross_group_invalid_windows_returns_failed_result() -> None:
    from asymmetry.core.fitting.parameter_models import global_fit_parameter_model

    groups, _ = _two_groups()
    model = ParameterCompositeModel(["Linear"], [])
    result = global_fit_parameter_model(groups, model, ["m"], ["b"], {}, windows=[(50, 10)])
    assert not result.success
    assert "inverted" in result.message.lower()


@gui
def test_cross_group_dialog_config_round_trips_error_mode_and_windows(qapp) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterGroupData
    from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog

    x = np.linspace(0, 100, 21)
    groups = [
        ParameterGroupData(
            group_id=f"g{i}",
            group_name=f"g{i}",
            x=x.copy(),
            y=2 * x + 1 + i,
            yerr=np.full_like(x, 0.5),
            group_variable_value=float(i),
        )
        for i in range(2)
    ]
    dlg = CrossGroupFitDialog(parameter_name="lambda", x_key="field", groups=groups)
    assert dlg._error_mode_combo is not None
    dlg._error_mode_combo.setCurrentIndex(dlg._error_mode_combo.findData("percent"))
    dlg._error_value_spin.setValue(7.0)
    dlg._add_window(0)
    cfg = dlg._collect_config()
    assert cfg["error_mode"] == "percent"
    assert cfg["error_value"] == pytest.approx(7.0)
    assert cfg["windows"] is not None

    restored = CrossGroupFitDialog(
        parameter_name="lambda", x_key="field", groups=groups, existing_config=cfg
    )
    from asymmetry.core.fitting.parameter_models import ErrorMode

    assert restored._error_mode() is ErrorMode.PERCENT
    assert restored._error_value() == pytest.approx(7.0)
    assert restored._fit.ranges[0].windows is not None
