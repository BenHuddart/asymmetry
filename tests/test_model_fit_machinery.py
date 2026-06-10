"""Tests for the Model-layer fitting machinery (WiMDA parity Phase 2).

Covers the error-mode selector (Column/Percent/Absolute/None/Scatter), the
union multi-range windows on ModelFitRange, and the shared χ² fit-quality
helper. Oracle values from docs/porting/model-function-parity/test-data.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.fit_quality import assess_fit_quality
from asymmetry.core.fitting.parameter_models import (
    ErrorMode,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    apply_error_mode,
    evaluate_parameter_model_fit,
    fit_parameter_model,
    range_mask,
    validate_fit_windows,
    windows_mask,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _linear_scatter_data() -> tuple[np.ndarray, np.ndarray]:
    """The frozen linear oracle of test-data.md §1.6 (seed 42)."""
    rng = np.random.default_rng(42)
    x = np.linspace(1.0, 10.0, 12)
    y = 2.0 + 0.7 * x + rng.normal(0.0, 0.35, x.size)
    return x, y


# ---------------------------------------------------------------------------
# Error modes
# ---------------------------------------------------------------------------


def test_apply_error_mode_assignments() -> None:
    y = np.array([0.0, 2.0, -4.0])
    yerr = np.array([0.1, 0.2, 0.3])

    np.testing.assert_allclose(apply_error_mode(y, yerr, ErrorMode.COLUMN), yerr)
    assert apply_error_mode(y, None, ErrorMode.COLUMN) is None
    # Percent: proportional to |y|; y = 0 points get sigma = 0 (masked later).
    np.testing.assert_allclose(apply_error_mode(y, yerr, "percent", 10.0), [0.0, 0.2, 0.4])
    np.testing.assert_allclose(apply_error_mode(y, yerr, "absolute", 0.5), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(apply_error_mode(y, yerr, "none"), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(apply_error_mode(y, yerr, "scatter"), [1.0, 1.0, 1.0])
    # WiMDA fallback: invalid/missing value -> sigma = 1.
    np.testing.assert_allclose(apply_error_mode(y, None, "absolute", None), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(apply_error_mode(y, None, "absolute", 0.0), [1.0, 1.0, 1.0])


def test_percent_mode_masks_zero_y_points() -> None:
    # A y = 0 point carries no information in percent mode (sigma = 0) and is
    # dropped by the validity mask rather than crashing the fit (D9).
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 0.0, 6.0, 8.0])
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    result = fit_parameter_model(x, y, None, model, params, error_mode="percent", error_value=5.0)
    assert result.success
    assert result.n_points == 3
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["m"], 2.0, rtol=1e-6)
    np.testing.assert_allclose(fitted["b"], 0.0, atol=1e-6)


def test_error_floor_applies_only_in_column_mode() -> None:
    # One absurdly small propagated error would dominate an unfloored fit.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.2, 0.8, 1.1, 5.0])  # last point is an outlier
    yerr = np.array([0.1, 0.1, 0.1, 0.1, 1e-12])
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.0)])

    column = fit_parameter_model(x, y, yerr, model, params, error_mode="column")
    assert column.success
    c_column = next(p.value for p in column.parameters if p.name == "c")
    # Floored (sigma >= 0.05): the outlier no longer dominates entirely.
    assert c_column < 4.0

    # Absolute mode with a tiny constant is honoured verbatim (no floor): the
    # weighting is uniform, so the fit is the plain mean including the outlier.
    absolute = fit_parameter_model(
        x, y, yerr, model, params, error_mode="absolute", error_value=1e-6
    )
    assert absolute.success
    c_absolute = next(p.value for p in absolute.parameters if p.name == "c")
    np.testing.assert_allclose(c_absolute, np.mean(y), rtol=1e-6)


def test_scatter_mode_matches_unweighted_ols_with_rescaled_errors() -> None:
    x, y = _linear_scatter_data()
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    result = fit_parameter_model(x, y, None, model, params, error_mode="scatter")
    assert result.success
    assert result.error_mode == "scatter"
    fitted = {p.name: p.value for p in result.parameters}
    # Frozen OLS oracle (test-data.md §1.6).
    np.testing.assert_allclose(fitted["b"], 1.84514937, rtol=1e-5)
    np.testing.assert_allclose(fitted["m"], 0.71914734, rtol=1e-5)
    np.testing.assert_allclose(result.reduced_chi_squared, 0.1202971846, rtol=1e-5)
    np.testing.assert_allclose(result.uncertainties["b"], 0.21917779, rtol=1e-4)
    np.testing.assert_allclose(result.uncertainties["m"], 0.03544948, rtol=1e-4)


def test_scatter_mode_is_fixed_point_of_wimda_estimate_iteration() -> None:
    """WiMDA's Estimate mode refits with sigma = errabs, then sets
    errabs <- errabs*sqrt(chi2r) until stationary. One explicit iteration from
    any start lands on errabs* = sqrt(chi2_unweighted/nu), where the parameter
    errors equal the scatter-mode output — tested, not asserted in prose."""
    x, y = _linear_scatter_data()
    model = ParameterCompositeModel(["Linear"])

    def fit_with_errabs(errabs: float):
        params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
        return fit_parameter_model(
            x, y, None, model, params, error_mode="absolute", error_value=errabs
        )

    errabs = 2.0  # arbitrary start
    first = fit_with_errabs(errabs)
    errabs *= np.sqrt(first.reduced_chi_squared)
    second = fit_with_errabs(errabs)

    # One step reaches the fixed point: chi2r = 1 and errabs is stationary.
    np.testing.assert_allclose(second.reduced_chi_squared, 1.0, rtol=1e-6)
    np.testing.assert_allclose(errabs, 0.3468388453, rtol=1e-5)  # sqrt(chi2_1/nu)
    next_errabs = errabs * np.sqrt(second.reduced_chi_squared)
    np.testing.assert_allclose(next_errabs, errabs, rtol=1e-6)

    # Parameter values are independent of the uniform sigma; the converged
    # errors equal the one-pass scatter-mode errors.
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    scatter = fit_parameter_model(x, y, None, model, params, error_mode="scatter")
    for name in ("m", "b"):
        first_val = next(p.value for p in first.parameters if p.name == name)
        second_val = next(p.value for p in second.parameters if p.name == name)
        np.testing.assert_allclose(first_val, second_val, rtol=1e-6)
        np.testing.assert_allclose(
            second.uncertainties[name], scatter.uncertainties[name], rtol=1e-4
        )


def test_none_mode_same_parameters_as_absolute_different_errors() -> None:
    x, y = _linear_scatter_data()
    model = ParameterCompositeModel(["Linear"])

    def run(mode: str, value: float | None = None):
        params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
        return fit_parameter_model(x, y, None, model, params, error_mode=mode, error_value=value)

    none_result = run("none")
    absolute_result = run("absolute", 0.5)
    assert none_result.success and absolute_result.success
    for name in ("m", "b"):
        nv = next(p.value for p in none_result.parameters if p.name == name)
        av = next(p.value for p in absolute_result.parameters if p.name == name)
        np.testing.assert_allclose(nv, av, rtol=1e-6)
        # Uniform sigma scales the errors linearly: 0.5x sigma -> 0.5x error.
        np.testing.assert_allclose(
            absolute_result.uncertainties[name],
            0.5 * none_result.uncertainties[name],
            rtol=1e-4,
        )


def test_column_mode_remains_default_behaviour() -> None:
    x = np.linspace(0.0, 10.0, 20)
    y = 3.0 * x + 1.0
    yerr = np.full_like(x, 0.05)
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    default = fit_parameter_model(x, y, yerr, model, params)
    assert default.error_mode == "column"
    assert default.n_points == x.size


# ---------------------------------------------------------------------------
# Union multi-range windows
# ---------------------------------------------------------------------------


def test_windows_mask_or_combination_and_fallback() -> None:
    x = np.linspace(0.0, 10.0, 11)
    mask = windows_mask(x, [(0.0, 3.0), (7.0, 10.0)])
    np.testing.assert_array_equal(
        mask, [True, True, True, True, False, False, False, True, True, True, True]
    )
    # Overlapping windows are a plain OR; no double counting possible.
    np.testing.assert_array_equal(
        windows_mask(x, [(0.0, 5.0), (3.0, 8.0)]), windows_mask(x, [(0.0, 8.0)])
    )
    # Fallback to x_min/x_max when no windows.
    np.testing.assert_array_equal(windows_mask(x, None, 2.0, 4.0), (x >= 2.0) & (x <= 4.0))
    np.testing.assert_array_equal(windows_mask(x, None, None, None), np.ones_like(x, dtype=bool))


def test_validate_fit_windows_rejects_bad_input() -> None:
    assert validate_fit_windows(None) is None
    assert validate_fit_windows([]) is None
    assert validate_fit_windows([(1.0, 2.0)]) == [(1.0, 2.0)]
    with pytest.raises(ValueError, match="inverted"):
        validate_fit_windows([(3.0, 1.0)])
    with pytest.raises(ValueError, match="finite"):
        validate_fit_windows([(0.0, float("nan"))])


def test_single_window_equals_min_max_bounds() -> None:
    x = np.linspace(0.0, 100.0, 51)
    y = 0.1 * x + 2.0
    model = ParameterCompositeModel(["Linear"])

    def run(**kwargs):
        params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
        return fit_parameter_model(x, y, None, model, params, **kwargs)

    bounded = run(x_min=20.0, x_max=80.0)
    windowed = run(windows=[(20.0, 80.0)])
    assert bounded.n_points == windowed.n_points
    for name in ("m", "b"):
        bv = next(p.value for p in bounded.parameters if p.name == name)
        wv = next(p.value for p in windowed.parameters if p.name == name)
        np.testing.assert_allclose(wv, bv, rtol=1e-9)


def test_union_windows_recover_critical_divergence_excluding_tc() -> None:
    """The canonical WiMDA multi-range use: fit λ(T) across the divergence
    while excluding the critical region around Tc (test-data.md §1.8)."""
    rng = np.random.default_rng(7)
    true = dict(a=2.0, Tc=69.2, nu=0.7, c=0.05)
    t = np.linspace(40.0, 100.0, 61)
    lam = true["a"] * np.abs(t - true["Tc"]) ** (-true["nu"]) + true["c"]
    lam_noisy = lam * (1.0 + rng.normal(0.0, 0.03, t.size))
    lam_err = 0.03 * lam

    model = ParameterCompositeModel(["CriticalDivergence"])
    params = ParameterSet(
        [
            Parameter("a", value=1.0, min=0.0, max=100.0),
            Parameter("Tc", value=68.0, min=60.0, max=80.0),
            Parameter("nu", value=1.0, min=0.1, max=3.0),
            Parameter("c", value=0.0, min=0.0, max=10.0),
        ]
    )
    result = fit_parameter_model(
        t, lam_noisy, lam_err, model, params, windows=[(40.0, 64.0), (74.0, 100.0)]
    )
    assert result.success
    assert result.n_points == int(np.sum((t <= 64.0) | (t >= 74.0)))
    fitted = {p.name: p.value for p in result.parameters}
    for name in ("a", "Tc", "nu", "c"):
        tolerance = 4.0 * result.uncertainties.get(name, 0.0) + 1e-6
        assert abs(fitted[name] - true[name]) < max(tolerance, 0.15 * abs(true[name]) + 0.02), (
            name,
            fitted[name],
        )


def test_empty_window_union_reports_no_points() -> None:
    x = np.linspace(0.0, 10.0, 11)
    y = x.copy()
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    result = fit_parameter_model(x, y, None, model, params, windows=[(20.0, 30.0)])
    assert not result.success
    assert "No valid points" in result.message


def test_range_mask_and_curve_sampling_over_window_envelope() -> None:
    x = np.linspace(0.0, 10.0, 21)
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=2.0), Parameter("b", value=1.0)])
    fit_range = ModelFitRange(
        x_min=0.0,
        x_max=10.0,
        model=model,
        parameters=params,
        windows=[(1.0, 3.0), (7.0, 9.0)],
    )
    mask = range_mask(x, fit_range)
    np.testing.assert_array_equal(mask, ((x >= 1.0) & (x <= 3.0)) | ((x >= 7.0) & (x <= 9.0)))

    # evaluate_parameter_model_fit draws the curve across the full envelope so
    # the model is shown continuously through the excluded gap.
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    fit_range.result = ParameterModelFitResult(success=True, parameters=params)
    fit = ParameterModelFit(parameter_name="Lambda", x_key="temperature", ranges=[fit_range])
    curves = evaluate_parameter_model_fit(fit, num_points=50)
    assert len(curves) == 1
    np.testing.assert_allclose(curves[0].x.min(), 1.0)
    np.testing.assert_allclose(curves[0].x.max(), 9.0)


# ---------------------------------------------------------------------------
# χ² fit quality
# ---------------------------------------------------------------------------


def test_fit_quality_band_oracle() -> None:
    # scipy.stats.chi2.ppf oracle at R = 0.95 (test-data.md §1.5).
    expected = {
        5: (0.16624232269733252, 2.566500398806005),
        10: (0.3246972780236841, 2.0483177350807393),
        20: (0.4795388696132433, 1.708480345141917),
        50: (0.6471472739131731, 1.4284039037501284),
    }
    for dof, (low, high) in expected.items():
        quality = assess_fit_quality(float(dof), dof)
        np.testing.assert_allclose(quality.band_low, low, rtol=1e-10)
        np.testing.assert_allclose(quality.band_high, high, rtol=1e-10)


def test_fit_quality_verdicts_match_wimda_semantics() -> None:
    # Two-sided CDF test at R = 0.95: below 2.5 % -> overdone, above 97.5 % -> poor.
    assert assess_fit_quality(2.0, 10).verdict == "overdone"
    assert assess_fit_quality(25.0, 10).verdict == "poor"
    assert assess_fit_quality(9.0, 10).verdict == "good"
    good = assess_fit_quality(9.0, 10)
    np.testing.assert_allclose(good.chi2_reduced, 0.9)
    assert good.dof == 10


def test_fit_quality_edge_cases() -> None:
    assert assess_fit_quality(5.0, 0).verdict is None
    assert np.isnan(assess_fit_quality(5.0, 0).band_low)
    assert assess_fit_quality(float("nan"), 10).verdict is None
    assert assess_fit_quality(-1.0, 10).verdict is None
    # Confidence clamped to [0.5, 0.999] like WiMDA's Rgoodfit.
    assert assess_fit_quality(9.0, 10, confidence=0.2).confidence == 0.5
    assert assess_fit_quality(9.0, 10, confidence=1.0).confidence == 0.999


def test_fit_quality_importable_without_qt() -> None:
    import importlib

    module = importlib.import_module("asymmetry.core.fitting.fit_quality")
    source = open(module.__file__).read()
    assert "PySide6" not in source
    assert "asymmetry.gui" not in source


# ---------------------------------------------------------------------------
# Review fixes: edge-case and contract regressions
# ---------------------------------------------------------------------------


def test_percent_mode_zero_value_falls_back_instead_of_masking_all() -> None:
    # A zero/negative percentage is invalid input, not "zero error everywhere":
    # it falls back to 1 % so the fit cannot fail with a misleading
    # "No valid points" message.
    sigma = apply_error_mode(np.array([2.0, -4.0]), None, "percent", 0.0)
    np.testing.assert_allclose(sigma, [0.02, 0.04])

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    result = fit_parameter_model(x, y, None, model, params, error_mode="percent", error_value=0.0)
    assert result.success
    assert result.n_points == 4


def test_fit_parameter_model_returns_failed_result_for_invalid_windows() -> None:
    # Bad range inputs keep the documented failure contract: a failed result,
    # never an exception (scripted callers check result.success).
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    result = fit_parameter_model(x, y, None, model, params, windows=[(5.0, 1.0)])
    assert not result.success
    assert "inverted" in result.message


def test_scatter_mode_zero_dof_reports_indeterminate_errors() -> None:
    # An exact interpolation has no scatter to estimate errors from; the
    # rescale must not collapse the uncertainties toward zero.
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 2.0])
    model = ParameterCompositeModel(["Polynomial"])
    params = ParameterSet(
        [
            Parameter("c0", value=0.1),
            Parameter("c1", value=0.1),
            Parameter("c2", value=0.1),
            Parameter("c3", value=0.0, fixed=True),
            Parameter("c4", value=0.0, fixed=True),
            Parameter("c5", value=0.0, fixed=True),
        ]
    )
    result = fit_parameter_model(x, y, None, model, params, error_mode="scatter")
    assert result.success
    assert result.uncertainties == {}
    assert "no degrees of freedom" in result.message


def test_evaluate_skips_ranges_with_invalid_windows() -> None:
    # Inverted windows (e.g. mid-edit state committed from the dialog) must
    # not raise inside plotting paths — the curve is skipped instead.
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=2.0), Parameter("b", value=1.0)])
    fit_range = ModelFitRange(
        x_min=0.0,
        x_max=10.0,
        model=model,
        parameters=params,
        result=ParameterModelFitResult(success=True, parameters=params),
        windows=[(5.0, 1.0)],
    )
    fit = ParameterModelFit(parameter_name="Lambda", x_key="temperature", ranges=[fit_range])
    assert evaluate_parameter_model_fit(fit) == []


def test_parse_fit_windows_is_lenient() -> None:
    from asymmetry.core.fitting.parameter_models import parse_fit_windows

    assert parse_fit_windows(None) is None
    assert parse_fit_windows("nonsense") is None
    assert parse_fit_windows([]) is None
    assert parse_fit_windows([[1, 4], [7, 10]]) == [(1.0, 4.0), (7.0, 10.0)]
    # Malformed entries are dropped, not fatal (saved state must always load).
    assert parse_fit_windows([[1, 4], ["bad", 2], [3]]) == [(1.0, 4.0)]
    assert parse_fit_windows([["bad", "worse"]]) is None


def test_effective_range_bounds_envelope_and_fallback() -> None:
    from asymmetry.core.fitting.parameter_models import effective_range_bounds

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    plain = ModelFitRange(x_min=2.0, x_max=8.0, model=model, parameters=params)
    assert effective_range_bounds(plain) == (2.0, 8.0)
    windowed = ModelFitRange(
        x_min=2.0, x_max=8.0, model=model, parameters=params, windows=[(1.0, 3.0), (7.0, 12.0)]
    )
    assert effective_range_bounds(windowed) == (1.0, 12.0)
    with pytest.raises(ValueError, match="inverted"):
        effective_range_bounds(
            ModelFitRange(
                x_min=None, x_max=None, model=model, parameters=params, windows=[(5.0, 1.0)]
            )
        )
