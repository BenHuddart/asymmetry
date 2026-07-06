"""Tests for Bayesian next-point suggestion (experiment_design.py).

Oracles from docs/studies/bed-next-point-suggestion.md §6 and the numeric
prototype in docs/studies/prototype-results.md: D-optimal designs for a
straight line sit at the interval extremes; c-optimal for a parameter
concentrates where that parameter's sensitivity is largest (the intercept
near x=0, an Arrhenius activation energy at the low-T/high-1/T extreme, an
order parameter's T_c just below the fitted T_c); event-count solves are
monotone in the goal and report unreachability rather than absurd counts;
degenerate inputs never raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.experiment_design import (
    calibrate_events_for_goal,
    calibrate_suggestion,
    suggest_next_point,
)
from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

# ---------------------------------------------------------------------------
# Shared fixtures (built once per test via helper functions — fast, <60s total)
# ---------------------------------------------------------------------------


def _line_fit(seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 9.0, 40)
    true_m, true_b = 2.0, 1.0
    yerr = np.full_like(x, 0.05)
    y = true_m * x + true_b + rng.normal(0.0, 0.05, size=x.shape)

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet(
        [
            Parameter("m", value=1.0, min=-10.0, max=10.0),
            Parameter("b", value=0.0, min=-10.0, max=10.0),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    return model, result, x, yerr


def _arrhenius_fit(seed: int = 2):
    rng = np.random.default_rng(seed)
    x = np.linspace(100.0, 300.0, 25)
    true_a, true_ea = 10.0, 20.0
    model = ParameterCompositeModel(["Arrhenius"])
    y_true = model.function(x, a=true_a, Ea=true_ea)
    yerr = np.maximum(0.03 * np.abs(y_true), 1e-6)
    y = y_true + rng.normal(0.0, yerr)

    params = ParameterSet(
        [
            Parameter("a", value=8.0, min=0.1, max=100.0),
            Parameter("Ea", value=15.0, min=0.1, max=100.0),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    return model, result, x, yerr


def _order_parameter_fit(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(10.0, 95.0, 10)
    model = ParameterCompositeModel(["OrderParameter"])
    y_true = model.function(x, y0=1.0, Tc=100.0, beta=0.35, alpha=2.0)
    yerr = np.full_like(x, 0.02)
    y = y_true + rng.normal(0.0, 0.02, size=x.shape)

    params = ParameterSet(
        [
            Parameter("y0", value=0.9),
            Parameter("Tc", value=105.0, min=1.0),
            Parameter("beta", value=0.4, min=0.01, max=2.0),
            Parameter("alpha", value=1.5, min=0.1, max=4.0),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params, extra_starts=2)
    assert result.success
    assert result.covariance is not None
    return model, result, x, y, yerr


# ---------------------------------------------------------------------------
# Oracle 1-2: straight line
# ---------------------------------------------------------------------------


def test_line_d_optimal_best_x_at_boundary() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 0.0, 10.0, target=None
    )
    assert suggestion.best_x == pytest.approx(0.0) or suggestion.best_x == pytest.approx(10.0)


def test_line_c_optimal_intercept_best_x_near_zero() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 0.0, 10.0, target="b"
    )
    assert abs(suggestion.best_x) < 1.0


# ---------------------------------------------------------------------------
# Oracle 3: Arrhenius, c-optimal for Ea at the low-T extreme
# ---------------------------------------------------------------------------


def test_arrhenius_c_optimal_ea_best_x_at_low_temperature_extreme() -> None:
    model, result, x, yerr = _arrhenius_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 100.0, 300.0, target="Ea"
    )
    assert suggestion.best_x == pytest.approx(100.0, abs=5.0)


# ---------------------------------------------------------------------------
# Oracle 4: order parameter, c-optimal for Tc
# ---------------------------------------------------------------------------


def test_order_parameter_c_optimal_tc_best_x_between_data_and_tc() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    fitted_tc = next(p.value for p in result.parameters if p.name == "Tc")
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 5.0, 120.0, target="Tc"
    )
    # Oracle: information about Tc concentrates just below the fitted Tc,
    # ahead of the measured span but not past the transition itself.
    assert float(x.max()) < suggestion.best_x < fitted_tc


def test_order_parameter_utility_negligible_far_above_tc() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 5.0, 120.0, target="Tc"
    )
    peak_utility = float(np.max(suggestion.utility))
    far_above_tc = suggestion.x_candidates > 110.0
    assert np.any(far_above_tc)
    utility_far_above = suggestion.utility[far_above_tc]
    assert np.all(utility_far_above < 0.05 * peak_utility)


# ---------------------------------------------------------------------------
# Oracle 5: event-count solve
# ---------------------------------------------------------------------------


def test_event_count_solve_loose_goal_gives_finite_reachable_factor() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    suggestion = suggest_next_point(
        model,
        result.parameters,
        result.covariance,
        x,
        yerr,
        5.0,
        120.0,
        target="Tc",
        sigma_goal=0.5,
    )
    assert suggestion.events_factor_to_target is not None
    assert np.isfinite(suggestion.events_factor_to_target)
    assert suggestion.events_factor_to_target > 0.0
    assert suggestion.target_unreachable is False


def test_event_count_solve_absurd_goal_is_unreachable_with_floor_warning() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    suggestion = suggest_next_point(
        model,
        result.parameters,
        result.covariance,
        x,
        yerr,
        5.0,
        120.0,
        target="Tc",
        sigma_goal=1e-6,
    )
    assert suggestion.target_unreachable is True
    assert any("floor" in w.lower() for w in suggestion.warnings)


def test_event_count_solve_floor_sigma_finite_and_le_prior() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    names, matrix = result.covariance
    prior_sigma_tc = float(np.sqrt(np.array(matrix)[names.index("Tc"), names.index("Tc")]))
    suggestion = suggest_next_point(
        model,
        result.parameters,
        result.covariance,
        x,
        yerr,
        5.0,
        120.0,
        target="Tc",
        sigma_goal=0.5,
    )
    assert suggestion.floor_sigma is not None
    assert np.isfinite(suggestion.floor_sigma)
    assert suggestion.floor_sigma <= prior_sigma_tc + 1e-9


# ---------------------------------------------------------------------------
# Oracle 6: re-measure competes
# ---------------------------------------------------------------------------


def test_measured_x_values_within_range_appear_in_candidates() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 0.0, 10.0, target=None
    )
    in_range_measured = x[(x >= 0.0) & (x <= 10.0)]
    for value in in_range_measured[:5]:
        assert np.any(np.isclose(suggestion.x_candidates, value, atol=1e-9))


# ---------------------------------------------------------------------------
# Oracle 7: extrapolation flagged
# ---------------------------------------------------------------------------


def test_extrapolation_flagged_outside_measured_span() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, -5.0, 15.0, target=None
    )
    assert np.any(suggestion.extrapolated)
    assert any("extend" in w.lower() or "extrapolat" in w.lower() for w in suggestion.warnings)


def test_no_extrapolation_warning_when_range_inside_measured_span() -> None:
    model, result, x, yerr = _line_fit()
    x_min, x_max = float(x.min()) + 0.5, float(x.max()) - 0.5
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, x_min, x_max, target=None
    )
    assert not np.any(suggestion.extrapolated)
    assert not any("extrapolat" in w.lower() for w in suggestion.warnings)


# ---------------------------------------------------------------------------
# Degradation: never raises (§6 item 6)
# ---------------------------------------------------------------------------


def test_none_covariance_gives_empty_suggestion_with_warning() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(model, result.parameters, None, x, yerr, 0.0, 10.0)
    assert suggestion.x_candidates.size == 0
    assert suggestion.utility.size == 0
    assert np.isnan(suggestion.best_x)
    assert suggestion.warnings


def test_mismatched_covariance_shape_gives_empty_suggestion_with_warning() -> None:
    model, result, x, yerr = _line_fit()
    bad_covariance = (["m", "b"], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    suggestion = suggest_next_point(model, result.parameters, bad_covariance, x, yerr, 0.0, 10.0)
    assert suggestion.x_candidates.size == 0
    assert np.isnan(suggestion.best_x)
    assert suggestion.warnings


def test_mismatched_covariance_names_gives_empty_suggestion_with_warning() -> None:
    model, result, x, yerr = _line_fit()
    names, matrix = result.covariance
    bad_covariance = (names + ["extra"], matrix)
    suggestion = suggest_next_point(model, result.parameters, bad_covariance, x, yerr, 0.0, 10.0)
    assert suggestion.x_candidates.size == 0
    assert suggestion.warnings


def test_non_finite_covariance_entries_gives_empty_suggestion_with_warning() -> None:
    model, result, x, yerr = _line_fit()
    names, matrix = result.covariance
    bad_matrix = [[float("nan"), 0.0], [0.0, 1.0]]
    suggestion = suggest_next_point(
        model, result.parameters, (names, bad_matrix), x, yerr, 0.0, 10.0
    )
    assert suggestion.x_candidates.size == 0
    assert suggestion.warnings


def test_target_not_a_free_parameter_gives_empty_suggestion_with_warning() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 0.0, 10.0, target="not_a_param"
    )
    assert suggestion.x_candidates.size == 0
    assert np.isnan(suggestion.best_x)
    assert suggestion.warnings


def test_all_nonfinite_or_zero_yerr_gives_empty_suggestion_with_warning() -> None:
    model, result, x, _yerr = _line_fit()
    bad_yerr = np.zeros_like(x)
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, bad_yerr, 0.0, 10.0
    )
    assert suggestion.x_candidates.size == 0
    assert suggestion.warnings


def test_x_max_not_greater_than_x_min_gives_empty_suggestion_with_warning() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 10.0, 10.0
    )
    assert suggestion.x_candidates.size == 0
    assert suggestion.warnings


def test_barely_constrained_fit_warns_but_still_suggests() -> None:
    # §6 item 6 warn-but-suggest contract: n_points <= n_free + 1.
    model, result, x, _y, yerr = _order_parameter_fit()
    few_x = x[:5]
    few_yerr = yerr[:5]
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, few_x, few_yerr, 5.0, 120.0, target="Tc"
    )
    assert any("coarse scan" in w.lower() or "unreliable" in w.lower() for w in suggestion.warnings)
    assert not np.isnan(suggestion.best_x)
    assert suggestion.x_candidates.size > 0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibrate_suggestion_line_model_well_behaved() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 0.0, 10.0, target="b"
    )
    y_true_slope, y_true_intercept = 2.0, 1.0
    y = y_true_slope * x + y_true_intercept

    calibration = calibrate_suggestion(
        model,
        result.parameters,
        x,
        y,
        yerr,
        suggestion.best_x,
        target="b",
        events_factor=1.0,
        n_trials=12,
        seed=0,
        predicted_post_sigma=suggestion.predicted_post_sigma,
    )

    assert np.isfinite(calibration.realized_post_sigma)
    assert calibration.realized_post_sigma_p16 <= calibration.realized_post_sigma
    assert calibration.realized_post_sigma <= calibration.realized_post_sigma_p84
    assert calibration.n_failed <= calibration.n_trials // 2
    assert calibration.calibration_ratio is not None
    assert 0.3 < calibration.calibration_ratio < 3.0


def test_calibrate_suggestion_degenerate_events_factor_zero() -> None:
    model, result, x, yerr = _line_fit()
    y = 2.0 * x + 1.0
    calibration = calibrate_suggestion(
        model, result.parameters, x, y, yerr, 5.0, target="b", events_factor=0.0, n_trials=10
    )
    assert np.isnan(calibration.realized_post_sigma)
    assert calibration.warnings


def test_calibrate_suggestion_degenerate_n_trials_zero() -> None:
    model, result, x, yerr = _line_fit()
    y = 2.0 * x + 1.0
    calibration = calibrate_suggestion(
        model, result.parameters, x, y, yerr, 5.0, target="b", events_factor=1.0, n_trials=0
    )
    assert np.isnan(calibration.realized_post_sigma)
    assert calibration.warnings


def test_calibrate_events_for_goal_line_model_reachable() -> None:
    model, result, x, yerr = _line_fit()
    y = 2.0 * x + 1.0
    factor, calibration = calibrate_events_for_goal(
        model,
        result.parameters,
        x,
        y,
        yerr,
        5.0,
        target="b",
        sigma_goal=0.05,
        initial_events_factor=1.0,
        n_trials=10,
        seed=0,
    )
    assert factor is not None
    assert np.isfinite(factor)
    assert calibration.realized_post_sigma <= 0.05 + 1e-9


def test_calibrate_events_for_goal_impossible_goal_returns_none() -> None:
    model, result, x, yerr = _line_fit()
    y = 2.0 * x + 1.0
    factor, calibration = calibrate_events_for_goal(
        model,
        result.parameters,
        x,
        y,
        yerr,
        5.0,
        target="b",
        sigma_goal=1e-9,
        initial_events_factor=1.0,
        n_trials=10,
        seed=0,
        max_events_factor=4.0,
    )
    assert factor is None
    assert calibration is not None


# ---------------------------------------------------------------------------
# Ranking honesty (cheap version of prototype case 4)
# ---------------------------------------------------------------------------


def test_ranking_honesty_order_parameter_best_x_far_outranks_flat_region() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    suggestion = suggest_next_point(
        model, result.parameters, result.covariance, x, yerr, 5.0, 120.0, target="Tc"
    )
    best_utility = float(suggestion.utility[np.argmax(suggestion.utility)])

    flat_index = int(np.argmin(np.abs(suggestion.x_candidates - 20.0)))
    flat_utility = float(suggestion.utility[flat_index])

    assert flat_utility <= 0.0 or best_utility >= 50.0 * flat_utility
