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
# tests/test_wimda_model_function_parity.py; (T, ν, ν_err).
from tests.test_wimda_model_function_parity import EUO_NU_T_TREND


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
