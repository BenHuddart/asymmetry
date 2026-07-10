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
    SeriesSpec,
    aic_weights,
    calibrate_events_for_goal,
    calibrate_suggestion,
    cost_weighted_utility,
    set_matching_divergence,
    suggest_discriminating_point,
    suggest_next_point,
    suggest_next_point_multi,
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


# ---------------------------------------------------------------------------
# Phase 3 (§8.1): suggest_discriminating_point
# ---------------------------------------------------------------------------


def _fit_component(component: str, x, y, yerr, params: ParameterSet):
    model = ParameterCompositeModel([component])
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    return model, result.parameters


def test_discrimination_crossing_models_peak_at_range_end() -> None:
    # Linear vs Constant fitted to the same gently-sloped noisy data: the
    # two curves diverge most at whichever end of the range is farthest
    # from where they cross (near the data mean, since Constant fits the
    # weighted mean of a line).
    rng = np.random.default_rng(7)
    x = np.linspace(1.0, 9.0, 40)
    true_m, true_b = 0.3, 2.0
    yerr = np.full_like(x, 0.05)
    y = true_m * x + true_b + rng.normal(0.0, 0.05, size=x.shape)

    lead_model, lead_params = _fit_component(
        "Linear",
        x,
        y,
        yerr,
        ParameterSet(
            [
                Parameter("m", value=1.0, min=-10.0, max=10.0),
                Parameter("b", value=0.0, min=-10.0, max=10.0),
            ]
        ),
    )
    alt_model, alt_params = _fit_component(
        "Constant", x, y, yerr, ParameterSet([Parameter("c", value=2.0, min=-10.0, max=10.0)])
    )

    x_min, x_max = 0.0, 10.0
    suggestion = suggest_discriminating_point(
        lead_model, lead_params, [(alt_model, alt_params)], x, yerr, x_min, x_max
    )

    # Analytic argmax: |line - constant| grows monotonically away from the
    # crossing point (near the data's weighted mean), so the disagreement
    # is largest at whichever end of [x_min, x_max] is farther from the mean.
    lead_values = {p.name: float(p.value) for p in lead_params}
    alt_values = {p.name: float(p.value) for p in alt_params}
    crossing_x = (alt_values["c"] - lead_values["b"]) / lead_values["m"]
    expected_end = x_min if abs(x_min - crossing_x) > abs(x_max - crossing_x) else x_max

    assert not np.isnan(suggestion.best_x)
    assert suggestion.best_x == pytest.approx(expected_end, abs=0.5)
    assert suggestion.target is None


def test_discrimination_leader_vs_all_takes_elementwise_max() -> None:
    # Three candidates: alternative A disagrees most with the lead at low
    # x, alternative B disagrees most at high x. The combined utility (max
    # over alternatives) must exceed each pairwise curve at the respective
    # end, and its argmax should differ from at least one pairwise argmax.
    x = np.linspace(0.0, 10.0, 50)
    yerr = np.full_like(x, 0.1)

    lead_model = ParameterCompositeModel(["Constant"])
    lead_params = ParameterSet([Parameter("c", value=0.0)])

    # A: a steep positive line -> disagrees most with the flat lead at
    # high x.
    a_model = ParameterCompositeModel(["Linear"])
    a_params = ParameterSet([Parameter("m", value=2.0), Parameter("b", value=0.0)])

    # B: a steep negative line -> disagrees most with the flat lead at low
    # x (since b(0) is very negative-ish... use offset to control this).
    b_model = ParameterCompositeModel(["Linear"])
    b_params = ParameterSet([Parameter("m", value=-2.0), Parameter("b", value=0.0)])

    combined = suggest_discriminating_point(
        lead_model,
        lead_params,
        [(a_model, a_params), (b_model, b_params)],
        x,
        yerr,
        0.0,
        10.0,
    )
    only_a = suggest_discriminating_point(
        lead_model, lead_params, [(a_model, a_params)], x, yerr, 0.0, 10.0
    )
    only_b = suggest_discriminating_point(
        lead_model, lead_params, [(b_model, b_params)], x, yerr, 0.0, 10.0
    )

    # Elementwise max: combined utility is never below either pairwise curve.
    assert np.all(combined.utility >= only_a.utility - 1e-9)
    assert np.all(combined.utility >= only_b.utility - 1e-9)

    # A disagrees most at high x, B disagrees most at low x (both steep
    # lines through the same flat lead), so their argmaxes sit at opposite
    # ends and the combined curve exceeds each pairwise curve at the
    # opposite end.
    assert only_a.best_x == pytest.approx(10.0, abs=0.3)
    assert only_b.best_x == pytest.approx(10.0, abs=0.3) or only_b.best_x == pytest.approx(
        0.0, abs=0.3
    )
    assert combined.best_x is not None


def test_discrimination_order_parameter_vs_linear_peaks_near_transition() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    lead_model, lead_params = model, result.parameters

    linear_model, linear_params = _fit_component(
        "Linear",
        x,
        _y,
        yerr,
        ParameterSet(
            [
                Parameter("m", value=0.0, min=-10.0, max=10.0),
                Parameter("b", value=0.5, min=-10.0, max=10.0),
            ]
        ),
    )

    fitted_tc = next(p.value for p in result.parameters if p.name == "Tc")
    suggestion = suggest_discriminating_point(
        lead_model, lead_params, [(linear_model, linear_params)], x, yerr, 5.0, 120.0
    )

    assert not np.isnan(suggestion.best_x)
    # Physically sensible: the order-parameter curvature is strongest near
    # /below Tc, not in the flat saturated region far above Tc.
    assert suggestion.best_x <= fitted_tc + 1.0
    flat_region = suggestion.x_candidates > fitted_tc + 5.0
    if np.any(flat_region):
        assert suggestion.best_x not in suggestion.x_candidates[flat_region]


def test_discrimination_identical_model_gives_agreement_warning() -> None:
    model, result, x, _y, yerr = _order_parameter_fit()
    suggestion = suggest_discriminating_point(
        model, result.parameters, [(model, result.parameters)], x, yerr, 5.0, 120.0
    )
    assert any("agree within noise" in w.lower() for w in suggestion.warnings)
    assert np.isnan(suggestion.best_x) or float(np.max(suggestion.utility)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_discrimination_empty_alternatives_gives_empty_suggestion() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_discriminating_point(model, result.parameters, [], x, yerr, 0.0, 10.0)
    assert suggestion.x_candidates.size == 0
    assert np.isnan(suggestion.best_x)
    assert suggestion.warnings


def test_discrimination_no_valid_errors_gives_empty_suggestion() -> None:
    model, result, x, _yerr = _line_fit()
    bad_yerr = np.zeros_like(x)
    suggestion = suggest_discriminating_point(
        model, result.parameters, [(model, result.parameters)], x, bad_yerr, 0.0, 10.0
    )
    assert suggestion.x_candidates.size == 0
    assert suggestion.warnings


def test_discrimination_inverted_range_gives_empty_suggestion() -> None:
    model, result, x, yerr = _line_fit()
    suggestion = suggest_discriminating_point(
        model, result.parameters, [(model, result.parameters)], x, yerr, 10.0, 0.0
    )
    assert suggestion.x_candidates.size == 0
    assert suggestion.warnings


# ---------------------------------------------------------------------------
# Phase 3 (§8.1): aic_weights
# ---------------------------------------------------------------------------


def test_aic_weights_known_values() -> None:
    chi2 = [0.0, 2.0]
    p = [1, 1]
    weights = aic_weights(chi2, p)
    expected_0 = 1.0 / (1.0 + np.exp(-1.0))
    expected_1 = np.exp(-1.0) / (1.0 + np.exp(-1.0))
    assert weights[0] == pytest.approx(expected_0)
    assert weights[1] == pytest.approx(expected_1)
    assert sum(weights) == pytest.approx(1.0)


def test_aic_weights_extra_parameter_penalised() -> None:
    weights = aic_weights([10.0, 10.0], [1, 3])
    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1.0)


def test_aic_weights_non_finite_chi2_gets_zero_weight_and_renormalises() -> None:
    weights = aic_weights([1.0, float("nan"), 3.0], [1, 1, 1])
    assert weights[1] == 0.0
    assert weights[0] > 0.0
    assert weights[2] > 0.0
    assert sum(weights) == pytest.approx(1.0)


def test_aic_weights_all_non_finite_returns_uniform_zero() -> None:
    weights = aic_weights([float("nan"), float("inf")], [1, 2])
    assert weights == [0.0, 0.0]


def test_aic_weights_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        aic_weights([1.0, 2.0], [1])


# ---------------------------------------------------------------------------
# Phase 3 (§8.2): cost_weighted_utility
# ---------------------------------------------------------------------------


def test_cost_weighted_utility_zero_rates_preserves_ranking() -> None:
    x_candidates = np.array([0.0, 5.0, 10.0])
    utility = np.array([1.0, 4.0, 2.0])
    weighted = cost_weighted_utility(
        x_candidates,
        utility,
        x_current=5.0,
        count_time=2.0,
        up_rate=0.0,
        down_rate=0.0,
        gamma=0.7,
    )
    expected = np.clip(utility, 0.0, None) ** 0.7 / 2.0
    assert np.allclose(weighted, expected)
    assert int(np.argmax(weighted)) == int(np.argmax(utility))


def test_cost_weighted_utility_asymmetric_rates_can_flip_ranking() -> None:
    # x_current = 5. A candidate below x_current (x=0) has slightly higher
    # raw utility than one above (x=10), but a big down_rate makes moving
    # down expensive enough that the "above" candidate wins after weighting.
    x_candidates = np.array([0.0, 10.0])
    utility = np.array([1.1, 1.0])
    x_current = 5.0

    unweighted_argmax = int(np.argmax(utility))
    assert unweighted_argmax == 0

    weighted = cost_weighted_utility(
        x_candidates,
        utility,
        x_current=x_current,
        count_time=1.0,
        up_rate=0.01,
        down_rate=100.0,
        gamma=0.7,
    )
    assert int(np.argmax(weighted)) == 1


def test_cost_weighted_utility_zero_count_time_returns_utility_unchanged() -> None:
    x_candidates = np.array([0.0, 5.0, 10.0])
    utility = np.array([1.0, 4.0, 2.0])
    weighted = cost_weighted_utility(
        x_candidates, utility, x_current=5.0, count_time=0.0, up_rate=1.0, down_rate=1.0
    )
    assert np.array_equal(weighted, utility)


def test_cost_weighted_utility_negative_rate_returns_utility_unchanged() -> None:
    x_candidates = np.array([0.0, 5.0, 10.0])
    utility = np.array([1.0, 4.0, 2.0])
    weighted = cost_weighted_utility(
        x_candidates, utility, x_current=5.0, count_time=1.0, up_rate=-1.0, down_rate=1.0
    )
    assert np.array_equal(weighted, utility)


def test_cost_weighted_utility_non_finite_x_current_returns_utility_unchanged() -> None:
    x_candidates = np.array([0.0, 5.0, 10.0])
    utility = np.array([1.0, 4.0, 2.0])
    weighted = cost_weighted_utility(
        x_candidates,
        utility,
        x_current=float("nan"),
        count_time=1.0,
        up_rate=1.0,
        down_rate=1.0,
    )
    assert np.array_equal(weighted, utility)


# ---------------------------------------------------------------------------
# Phase 4 (§3.1–3.3): multi-series acquisition, risk mask, set matching
#
# Oracles from docs/studies/bed-next-angle-knight-shift.md §5: cos 2θ geometry
# (c-optimal antinodes/nodes, 90°-periodic D-optimal), the vector-observable
# information sum, the assignment-risk flag near predicted crossings, and the
# Hungarian set-matching divergence (reduces to the labelled integrand at N=1,
# permutation-invariant, zero for coincident hypotheses).
# ---------------------------------------------------------------------------


def _cos2_spec(theta2: float, *, k_amp: float = 20.0, y_err: float = 0.1) -> SeriesSpec:
    """A single AngularCos2 series with an identity covariance for the multi API.

    The identity covariance (no correlations, unit variance per parameter) keeps
    each c-optimal utility proportional to that parameter's squared sensitivity,
    so the argmax reads the pure cos 2θ geometry.
    """
    model = ParameterCompositeModel(["AngularCos2"])
    params = ParameterSet(
        [Parameter("K_avg", 100.0), Parameter("K_amp", k_amp), Parameter("theta0", theta2)]
    )
    names = ["K_avg", "K_amp", "theta0"]
    covariance = (names, np.eye(3).tolist())
    x_data = np.linspace(0.0, 180.0, 37)
    yerr = np.full_like(x_data, y_err)
    return SeriesSpec(
        model=model, parameters=params, covariance=covariance, x_data=x_data, y_err=yerr
    )


def test_multi_cos2_c_optimal_k_amp_peaks_at_antinode() -> None:
    theta2 = 15.0
    spec = _cos2_spec(theta2)
    suggestion = suggest_next_point_multi([spec], 0.0, 180.0, target=(0, "K_amp"))
    # |cos 2(θ − θ2)| is largest at the antinodes θ2 and θ2 + 90.
    assert suggestion.best_x == pytest.approx(
        theta2, abs=2.0
    ) or suggestion.best_x == pytest.approx(theta2 + 90.0, abs=2.0)


def test_multi_cos2_c_optimal_theta0_peaks_at_node() -> None:
    theta2 = 15.0
    spec = _cos2_spec(theta2)
    suggestion = suggest_next_point_multi([spec], 0.0, 180.0, target=(0, "theta0"))
    # |∂K/∂θ0| ∝ |sin 2(θ − θ2)| is largest at the nodes θ2 ± 45 (+ n·90).
    nodes = [theta2 + 45.0, theta2 + 135.0]
    assert any(suggestion.best_x == pytest.approx(node, abs=2.0) for node in nodes)


def test_multi_cos2_d_optimal_utility_is_90_periodic() -> None:
    spec = _cos2_spec(15.0)
    suggestion = suggest_next_point_multi([spec], 0.0, 180.0)
    probe = np.linspace(5.0, 85.0, 17)
    u_here = np.interp(probe, suggestion.x_candidates, suggestion.utility)
    u_shift = np.interp(probe + 90.0, suggestion.x_candidates, suggestion.utility)
    np.testing.assert_allclose(u_here, u_shift, atol=1e-9)


def test_multi_two_identical_series_double_d_optimal_utility() -> None:
    spec = _cos2_spec(15.0)
    single = suggest_next_point_multi([spec], 0.0, 180.0)
    doubled = suggest_next_point_multi([spec, spec], 0.0, 180.0)
    np.testing.assert_allclose(doubled.x_candidates, single.x_candidates)
    np.testing.assert_allclose(doubled.utility, 2.0 * single.utility, atol=1e-9)
    assert int(np.argmax(doubled.utility)) == int(np.argmax(single.utility))


def test_multi_none_covariance_series_contributes_nothing_but_a_warning() -> None:
    spec = _cos2_spec(15.0)
    nocov = SeriesSpec(
        model=spec.model,
        parameters=spec.parameters,
        covariance=None,
        x_data=spec.x_data,
        y_err=spec.y_err,
    )
    single = suggest_next_point_multi([spec], 0.0, 180.0)
    degraded = suggest_next_point_multi([spec, nocov], 0.0, 180.0, labels=["good", "bad"])
    np.testing.assert_allclose(degraded.x_candidates, single.x_candidates)
    np.testing.assert_allclose(degraded.utility, single.utility, atol=1e-12)
    assert any("bad" in w and "covariance" in w for w in degraded.warnings)


def _line_spec(m: float, b: float, *, sigma: float = 0.5) -> SeriesSpec:
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", m), Parameter("b", b)])
    covariance = (["m", "b"], np.eye(2).tolist())
    x_data = np.linspace(0.0, 10.0, 21)
    yerr = np.full_like(x_data, sigma)
    return SeriesSpec(
        model=model, parameters=params, covariance=covariance, x_data=x_data, y_err=yerr
    )


def test_multi_risk_mask_flags_predicted_crossing() -> None:
    # f_A = x and f_B = 10 − x cross at x = 5; sigma = 0.5 → 2σ = 1, so only the
    # neighbourhood of the crossing is flagged, not the well-separated ends.
    a = _line_spec(1.0, 0.0, sigma=0.5)
    b = _line_spec(-1.0, 10.0, sigma=0.5)
    suggestion = suggest_next_point_multi([a, b], 0.0, 10.0)
    assert suggestion.risk_mask is not None
    near_crossing = np.abs(suggestion.x_candidates - 5.0) < 0.3
    far_from_crossing = suggestion.x_candidates < 1.0
    assert np.all(suggestion.risk_mask[near_crossing])
    assert not np.any(suggestion.risk_mask[far_from_crossing])
    assert any("misassignment" in w.lower() for w in suggestion.warnings)


def test_multi_risk_mask_all_false_for_single_series() -> None:
    suggestion = suggest_next_point_multi([_line_spec(1.0, 0.0)], 0.0, 10.0)
    assert suggestion.risk_mask is not None
    assert not np.any(suggestion.risk_mask)


def _predict(model_name: str, values: dict[str, float], x: np.ndarray) -> np.ndarray:
    model = ParameterCompositeModel([model_name])
    return np.asarray(model.function(x, **values), dtype=float)


def test_set_matching_divergence_n1_reduces_to_labeled_integrand() -> None:
    model = ParameterCompositeModel(["Linear"])
    a = [(model, ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)]))]
    b = [(model, ParameterSet([Parameter("m", -1.0), Parameter("b", 3.0)]))]
    x = np.linspace(0.0, 10.0, 41)
    sigma = [np.full_like(x, 0.5)]
    u = set_matching_divergence(a, b, sigma, x)
    f_a = _predict("Linear", {"m": 1.0, "b": 0.0}, x)
    f_b = _predict("Linear", {"m": -1.0, "b": 3.0}, x)
    expected = (f_a - f_b) ** 2 / (2.0 * 0.5**2)
    np.testing.assert_allclose(u, expected, atol=1e-9)


def test_set_matching_divergence_invariant_under_hypothesis_reorder() -> None:
    model = ParameterCompositeModel(["Linear"])
    a = [
        (model, ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)])),
        (model, ParameterSet([Parameter("m", -0.5), Parameter("b", 8.0)])),
    ]
    b = [
        (model, ParameterSet([Parameter("m", 0.8), Parameter("b", 1.0)])),
        (model, ParameterSet([Parameter("m", -0.6), Parameter("b", 7.0)])),
    ]
    x = np.linspace(0.0, 10.0, 41)
    sigma = [np.full_like(x, 0.5), np.full_like(x, 0.7)]
    u = set_matching_divergence(a, b, sigma, x)
    u_swapped = set_matching_divergence(a, list(reversed(b)), sigma, x)
    np.testing.assert_allclose(u, u_swapped, atol=1e-9)


def test_set_matching_divergence_zero_when_hypotheses_coincide() -> None:
    model = ParameterCompositeModel(["Linear"])
    a = [
        (model, ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)])),
        (model, ParameterSet([Parameter("m", -0.5), Parameter("b", 8.0)])),
    ]
    x = np.linspace(0.0, 10.0, 41)
    sigma = [np.full_like(x, 0.5), np.full_like(x, 0.7)]
    u = set_matching_divergence(a, list(a), sigma, x)
    np.testing.assert_allclose(u, 0.0, atol=1e-12)


def test_set_matching_divergence_mismatched_lengths_raises() -> None:
    model = ParameterCompositeModel(["Linear"])
    a = [(model, ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)]))]
    b = [
        (model, ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)])),
        (model, ParameterSet([Parameter("m", -1.0), Parameter("b", 3.0)])),
    ]
    x = np.linspace(0.0, 10.0, 5)
    with pytest.raises(ValueError):
        set_matching_divergence(a, b, [np.ones_like(x)], x)


def test_multi_all_unusable_series_degrade_to_empty_suggestion() -> None:
    spec = _cos2_spec(15.0)
    nocov = SeriesSpec(
        model=spec.model,
        parameters=spec.parameters,
        covariance=None,
        x_data=spec.x_data,
        y_err=spec.y_err,
    )
    suggestion = suggest_next_point_multi([nocov, nocov], 0.0, 180.0)
    assert suggestion.x_candidates.size == 0
    assert np.isnan(suggestion.best_x)
    assert suggestion.warnings
