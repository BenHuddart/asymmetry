"""Conditional pull diagnostics through production generation/reduction/refit."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.pull_diagnostic import (
    ParameterPull,
    RefitOutcome,
    run_pull_distribution,
    run_pull_request,
)

N_BINS = 1500
BIN_WIDTH = 0.016
T0_BIN = 40


def _template() -> Run:
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=T0_BIN,
            good_bin_start=T0_BIN + 5,
            good_bin_end=N_BINS - 10,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": T0_BIN,
        "first_good_bin": T0_BIN + 5,
        "last_good_bin": N_BINS - 10,
        "good_frames": 18000.0,
    }
    return Run(run_number=4321, histograms=histograms, grouping=grouping)


def _exp_model(t: np.ndarray, a0: float = 20.0, rate: float = 0.5) -> np.ndarray:
    return a0 * np.exp(-rate * t)


def _make_refit(a0_start: float, rate_start: float, t_max: float = 8.0):
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    def refit(dataset, cancelled):
        params = ParameterSet(
            [
                Parameter(name="a0", value=a0_start, min=0.0, max=100.0),
                Parameter(name="rate", value=rate_start, min=0.0, max=10.0),
            ]
        )
        result = FitEngine().fit(
            dataset, _exp_model, params, t_max=t_max, cancel_callback=cancelled
        )
        if not result.success:
            return RefitOutcome(False, message=result.message)
        values = {p.name: p.value for p in result.parameters}
        return RefitOutcome(True, values, dict(result.uncertainties))

    return refit


class TestParameterPull:
    def test_statistics_of_a_known_normal_sample(self) -> None:
        rng = np.random.default_rng(0)
        pull = ParameterPull("x", truth=0.0, pulls=rng.normal(0.0, 1.0, 5000))
        assert abs(pull.mean) < 4.0 * pull.mean_uncertainty
        assert abs(pull.width - 1.0) < 4.0 * pull.width_uncertainty
        assert "sample width" in pull.verdict()

    def test_narrow_sample_has_no_directional_certification(self) -> None:
        rng = np.random.default_rng(1)
        pull = ParameterPull("x", truth=0.0, pulls=rng.normal(0.0, 0.5, 4000))
        assert "OVER-estimated" not in pull.verdict()


class TestRunPullDistribution:
    def test_pulls_centre_on_standard_normal(self) -> None:
        pytest.importorskip("iminuit")
        truth = {"a0": 21.0, "rate": 0.6}
        n_seeds = 100
        result = run_pull_distribution(
            _template(),
            _exp_model,
            truth,
            _make_refit(a0_start=15.0, rate_start=1.0),
            total_events=2.0e6,
            n_seeds=n_seeds,
            time_range=(None, 8.0),
        )
        assert result.n_usable >= 0.9 * n_seeds
        for name in truth:
            pull = result.parameters[name]
            assert abs(pull.mean) < 4.0 / np.sqrt(pull.n), name
            sigma_var = np.sqrt(2.0 / (pull.n - 1))
            assert abs(pull.width**2 - 1.0) < 4.0 * sigma_var, name

    def test_progress_callback_reports_each_seed(self) -> None:
        pytest.importorskip("iminuit")
        seen: list[tuple[int, int]] = []
        run_pull_distribution(
            _template(),
            _exp_model,
            {"a0": 21.0, "rate": 0.6},
            _make_refit(15.0, 1.0),
            total_events=1.0e6,
            n_seeds=5,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    def test_non_convergence_is_counted_not_fatal(self) -> None:
        result = run_pull_distribution(
            _template(),
            _exp_model,
            {"a0": 21.0, "rate": 0.6},
            lambda dataset, cancelled: None,  # always "fails"
            total_events=1.0e6,
            n_seeds=4,
        )
        assert result.n_usable == 0
        assert result.parameters["a0"].n == 0
        assert "fewer than two" in result.verdict()


class TestEarlyStop:
    def test_should_continue_false_stops_after_one_seed(self) -> None:
        completed = []

        def stop_after_one() -> bool:
            return not completed

        result = run_pull_distribution(
            _template(),
            _exp_model,
            {"a0": 21.0, "rate": 0.6},
            lambda dataset, cancelled: None,
            total_events=1.0e6,
            n_seeds=50,
            should_continue=stop_after_one,
            progress=lambda done, total: completed.append(done),
        )
        # Requested seeds remain 50; the authoritative attempt record contains one.
        assert result.n_attempted == 1


def test_original_three_seed_nan_production_entry():
    template = Run(
        446,
        [Histogram(np.zeros(100), 0.1, good_bin_end=99) for _ in range(2)],
        grouping={
            "groups": {1: [1], 2: [2]},
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 99,
            "good_frames": 1000,
        },
    )
    result = run_pull_distribution(
        template,
        lambda t, a: np.full_like(t, a),
        {"a": 20.0},
        lambda dataset, cancelled: RefitOutcome(True, {"a": float("nan")}, {"a": 1.0}),
        total_events=100000,
        n_seeds=3,
    )
    assert result.parameters["a"].n == 0
    assert "OVER-estimated" not in result.verdict()


def _constant(t, a=20.0, b=0.0):
    return np.full_like(t, a + b)


def _small_template():
    return Run(
        446,
        [Histogram(np.zeros(100), 0.1, good_bin_end=99) for _ in range(2)],
        grouping={
            "groups": {1: [1], 2: [2]},
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 99,
        },
    )


@pytest.mark.parametrize(
    "outcome,reason",
    [
        (None, "no_refit_result"),
        (RefitOutcome(False, message="call limit"), "unsuccessful_refit"),
        (RefitOutcome(True, {}, {"a": 1}), "missing_value:a"),
        (RefitOutcome(True, {"a": 20}, {}), "missing_error:a"),
        *[
            (RefitOutcome(True, {"a": v}, {"a": 1}), "nonfinite_value:a")
            for v in (np.nan, np.inf, -np.inf)
        ],
        *[
            (RefitOutcome(True, {"a": 20}, {"a": v}), "invalid_error:a")
            for v in (np.nan, np.inf, 0, -1)
        ],
        (RefitOutcome(True, {"a": 1e308}, {"a": 1e-308}), "nonfinite_pull:a"),
    ],
)
def test_unusable_outcomes_have_seed_and_reason(outcome, reason):
    result = run_pull_distribution(
        _small_template(),
        _constant,
        {"a": 20},
        lambda ds, cancel: outcome,
        total_events=100000,
        n_seeds=1,
        seed_start=71,
    )
    assert result.n_usable == 0
    assert result.n_attempted == 1
    assert reason in result.attempts[0].reason
    assert result.attempts[0].seed == 71
    assert not result.attempts[0].pulls
    assert "OVER-estimated" not in result.verdict()


def test_mixed_atomic_vector_and_independent_summary():
    outcomes = iter(
        [
            RefitOutcome(True, {"a": 21, "b": 2}, {"a": 1, "b": 1}),
            RefitOutcome(True, {"a": 23, "b": -2}, {"a": 1, "b": 1}),
            RefitOutcome(True, {"a": 25, "b": np.nan}, {"a": 1, "b": 1}),
            None,
            RefitOutcome(False),
        ]
    )
    result = run_pull_distribution(
        _small_template(),
        _constant,
        {"a": 20, "b": 0},
        lambda ds, cancel: next(outcomes),
        total_events=100000,
        n_seeds=5,
    )
    assert (result.n_attempted, result.n_fit_success, result.n_failed, result.n_usable) == (
        5,
        3,
        2,
        2,
    )
    assert sum(result.failure_reasons.values()) + result.n_usable == 5
    np.testing.assert_array_equal(result.parameters["a"].pulls, [1, 3])
    np.testing.assert_array_equal(result.parameters["b"].pulls, [2, -2])
    assert result.parameters["a"].mean == 2
    assert result.parameters["a"].width == pytest.approx(np.sqrt(2))
    assert result.parameters["b"].width == pytest.approx(np.sqrt(8))


def test_seed71_strict_json_round_trip_replay_and_source_omissions():
    import json

    from asymmetry.core.pull_diagnostic import PullDistribution, make_pull_request, run_pull_request

    source = _small_template()
    source.grouping.update(
        deadtime_correction=True, dead_time_us=[0.1, 0.1], background_correction=True
    )
    truth = {"a": 20}
    request = make_pull_request(
        source, _constant, truth, total_events=100000, n_seeds=3, seed_start=71
    )
    source.histograms[0].counts[:] = 999
    source.grouping["alpha"] = 2
    truth["a"] = 99
    seen = []

    def refit(ds, cancel):
        seen.append(ds.asymmetry.copy())
        return RefitOutcome(True, {"a": float(np.mean(ds.asymmetry))}, {"a": 1})

    first = run_pull_request(request, _constant, refit)
    restored = PullDistribution.from_dict(json.loads(json.dumps(first.to_dict(), allow_nan=False)))
    replay = run_pull_request(restored.request, _constant, refit)
    for a, b in zip(seen[:3], seen[3:], strict=True):
        np.testing.assert_array_equal(a, b)
    assert first.to_dict() == restored.to_dict() == replay.to_dict()
    assert [a.seed for a in first.attempts] == [71, 72, 73]
    assert first.request.to_dict()["source_context"]["deadtime_correction"] is True
    assert len(first.recipes) == 1
    recipe = first.recipes[0].to_dict()
    assert "deadtime" in str(recipe)
    assert "Poisson" in " ".join(request.to_dict()["assumptions"])
    assert restored.request.to_dict()["parameters"] == {"a": 20}


@pytest.mark.parametrize(
    "settings",
    [
        {"seed_start": -1},
        {"seed_start": 1.5},
        {"n_seeds": 0},
        {"total_events": np.inf},
        {"background_per_bin": -1},
        {"track": []},
        {"track": ["a", "a"]},
        {"track": ["missing"]},
        {"time_range": (2, 1)},
        {"time_range": (100, 101)},
    ],
)
def test_invalid_requests_fail_before_sampling(monkeypatch, settings):
    from asymmetry.core import pull_diagnostic

    monkeypatch.setattr(
        pull_diagnostic, "simulate_run", lambda *a, **k: pytest.fail("sampled invalid request")
    )
    with pytest.raises(ValueError):
        run_pull_distribution(
            _small_template(),
            _constant,
            {"a": 20},
            lambda ds, cancel: None,
            **({"total_events": 100000} | settings),
        )


@pytest.mark.parametrize("sample", [[0, 0, 1], [1, 2, 12], [4, 5], [1]])
def test_shift_skew_boundary_samples_never_certify(sample):
    summary = ParameterPull("a", 0, np.asarray(sample)).verdict()
    assert not any(word in summary for word in ("calibrated", "unbiased", "OVER", "UNDER"))


def test_pre_cancel_and_mid_refit_cancellation():
    from asymmetry.core.fitting.engine import FitCancelledError

    result = run_pull_distribution(
        _small_template(),
        _constant,
        {"a": 20},
        lambda ds, cancel: pytest.fail("refit"),
        total_events=100000,
        should_continue=lambda: False,
    )
    assert result.n_attempted == 0 and result.stopped

    def interrupted(ds, cancel):
        raise FitCancelledError("stop")

    result = run_pull_distribution(
        _small_template(), _constant, {"a": 20}, interrupted, total_events=100000
    )
    assert result.stopped and result.n_attempted == 1 and result.n_failed == 0
    assert result.attempts[0].reason == "cancelled"
    assert "partial" in result.verdict()
    with pytest.raises(RuntimeError, match="coding error"):

        def broken(ds, cancel):
            raise RuntimeError("coding error")

        run_pull_distribution(_small_template(), _constant, {"a": 20}, broken, total_events=100000)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda t: t.grouping.update(beta=0.8), "beta"),
        (lambda t: t.grouping.update(period_inputs=[{}]), "period"),
        (lambda t: t.metadata.update(combination={"mode": "sum"}), "arithmetic"),
        (lambda t: setattr(t.histograms[0], "t0_time_us", 0.051), "fractional"),
    ],
)
def test_unrepresentable_mean_contracts_refused(mutation, match):
    template = _small_template()
    mutation(template)
    with pytest.raises(ValueError, match=match):
        run_pull_distribution(
            template, _constant, {"a": 20}, lambda ds, cancel: None, total_events=100000
        )


def test_supplied_error_constant_and_gaussian_weighted_oracles():
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    # The native Mantid/musrfit control uses these same three supplied errors.
    x, y, sigma = np.arange(3.0), np.array([2.0, 4.0, 7.0]), np.array([1.0, 2.0, 1.0])
    result = FitEngine().fit(
        MuonDataset(x, y, sigma), lambda t, a: np.full_like(t, a), ParameterSet([Parameter("a", 3)])
    )
    assert result.success
    assert result.parameters["a"].value == pytest.approx(40 / 9, abs=1e-5)
    assert result.uncertainties["a"] == pytest.approx(2 / 3, abs=1e-5)
    assert result.chi_squared == pytest.approx(113 / 9, abs=1e-8)
    # A fixed Gaussian shape is linear in its sole fitted amplitude; weighted
    # normal equations independently determine both the minimum and curvature.
    t = np.array([0.0, 0.5, 1.0, 1.5])
    shape = np.exp(-(t**2) / 2)
    y, sigma = np.array([21.0, 17.0, 13.0, 6.0]), np.array([1.0, 2.0, 1.5, 1.0])
    information = np.sum((shape / sigma) ** 2)
    oracle = np.sum(y * shape / sigma**2) / information
    fit = FitEngine().fit(
        MuonDataset(t, y, sigma),
        lambda t, a: a * np.exp(-(t**2) / 2),
        ParameterSet([Parameter("a", 15)]),
    )
    assert fit.success
    assert fit.parameters["a"].value == pytest.approx(oracle, abs=1e-5)
    assert fit.uncertainties["a"] == pytest.approx(information**-0.5, abs=1e-5)


def test_alpha_nonunit_first_order_variance_and_percent_boundary():
    from asymmetry.core.simulate import reduce_run_to_dataset
    from asymmetry.core.transform.asymmetry import compute_asymmetry

    f, b, alpha = 160.0, 80.0, 1.5
    expected = (f - alpha * b) / (f + alpha * b)
    variance = 4 * alpha**2 * f * b * (f + b) / (f + alpha * b) ** 4
    value, error = compute_asymmetry(np.array([f]), np.array([b]), alpha=alpha)
    assert value[0] == pytest.approx(expected, rel=1e-14)
    assert error[0] ** 2 == pytest.approx(variance, rel=1e-14)
    template = _small_template()
    template.grouping["alpha"] = alpha
    template.histograms[0].counts[:] = f
    template.histograms[1].counts[:] = b
    ds = reduce_run_to_dataset(template)
    np.testing.assert_allclose(ds.asymmetry, 100 * expected, rtol=1e-14)
    np.testing.assert_allclose(ds.error**2, 10000 * variance, rtol=1e-14)


def test_bounded_ideal_poisson_constant_production_anchor():
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.pull_refit import EngineRefit

    # Predeclared stream1071..1134, N64: SE(mean)~.125 and SE(SD)~.089
    # under the normal reference. Four-reference-SE tolerances are bounded
    # regression checks at this informative plug-in point, not certification.
    refit = EngineRefit(_constant, ParameterSet([Parameter("a", 18)]))
    result = run_pull_distribution(
        _small_template(),
        _constant,
        {"a": 20},
        refit,
        total_events=1e6,
        n_seeds=64,
        seed_start=1071,
        refit_configuration=refit.configuration(),
    )
    assert result.n_usable >= 60, result.verdict()
    pull = result.parameters["a"]
    assert abs(pull.mean) < 4 / np.sqrt(pull.n)
    assert abs(pull.width**2 - 1) < 4 * np.sqrt(2 / (pull.n - 1))
    assert result.n_attempted == 64


def test_engine_configuration_roundtrip_replay_unbounded_constraints():
    import json

    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.pull_diagnostic import PullDistribution, make_pull_request, run_pull_request
    from asymmetry.core.pull_refit import EngineRefit

    refit = EngineRefit(_constant, ParameterSet([Parameter("a", 18)]))
    request = make_pull_request(
        _small_template(),
        _constant,
        {"a": 20},
        total_events=1e6,
        seed_start=71,
        n_seeds=2,
        refit_configuration=refit.configuration(),
    )
    first = run_pull_request(request, _constant, refit)
    restored = PullDistribution.from_dict(json.loads(json.dumps(first.to_dict(), allow_nan=False)))
    replay_refit = EngineRefit.from_configuration(_constant, restored.request.to_dict()["refit"])
    replay = run_pull_request(restored.request, _constant, replay_refit)
    assert first.to_dict() == restored.to_dict() == replay.to_dict()


def test_captured_packing_exclusions_and_integer_detector_timing():
    from asymmetry.core.pull_request import make_pull_request

    template = _small_template()
    template.histograms.append(Histogram(np.zeros(100), 0.1, t0_bin=3, good_bin_end=99))
    template.grouping.update(groups={1: [1, 3], 2: [2]}, excluded_detectors=[3], bunching_factor=3)
    request = make_pull_request(
        template, _constant, {"a": 20}, total_events=100000, time_range=(0.2, 5)
    )
    result = run_pull_request(
        request, _constant, lambda ds, cancel: RefitOutcome(True, {"a": 20}, {"a": 1})
    )
    assert request.to_dict()["source_context"]["selected_detector_indices"] == [0, 1]
    assert request.to_dict()["binning"]["factor"] == 3
    assert result.n_usable == 200


def test_nonreference_fractional_clock_and_fixed_tracking_refused():
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.pull_refit import EngineRefit

    template = _small_template()
    template.histograms[1].t0_time_us = 0.051
    with pytest.raises(ValueError, match="fractional"):
        run_pull_distribution(
            template, _constant, {"a": 20}, lambda ds, cancel: None, total_events=100000
        )
    refit = EngineRefit(_constant, ParameterSet([Parameter("a", 20, fixed=True)]))
    with pytest.raises(ValueError, match="free parameters"):
        run_pull_distribution(
            _small_template(),
            _constant,
            {"a": 20},
            refit,
            total_events=100000,
            refit_configuration=refit.configuration(),
        )


def test_selected_statistics_do_not_redistribute_excluded_detector_events():
    from asymmetry.core.pull_request import make_pull_request

    template = _small_template()
    template.histograms[0].counts[:] = 120
    template.histograms[1].counts[:] = 80
    template.histograms.append(Histogram(np.full(100, 1e7), 0.1, good_bin_end=99))
    template.grouping.update(groups={1: [1, 3], 2: [2]}, excluded_detectors=[3])
    request = make_pull_request(
        template, _constant, {"a": 20}, total_events=None, background_per_bin=None
    )
    assert request.to_dict()["total_events"] == 20000
    assert request.to_dict()["background_per_bin"] == 0


def test_extreme_finite_pull_statistics_remain_descriptive():
    pull = ParameterPull("a", 0, np.array([1e308, 1e308]))
    assert pull.mean == 1e308
    assert pull.width == 0
    pull = ParameterPull("a", 0, np.array([-1e308, 1e308]))
    assert pull.mean == 0
    assert pull.width == pytest.approx(np.sqrt(2) * 1e308)


def test_unavailable_uncertainty_distinct_from_bad_numeric_error():
    result = run_pull_distribution(
        _small_template(),
        _constant,
        {"a": 20},
        lambda ds, cancel: RefitOutcome(True, {"a": 20}, {"a": None}),
        total_events=100000,
        n_seeds=1,
    )
    assert result.attempts[0].reason == "unavailable_uncertainty:a"
    assert result.n_fit_success == 1 and result.n_usable == 0


@pytest.mark.parametrize(
    "timing",
    [{"effective_detector_t0_bins": [2, 2]}, {"t0_policy": {"mode": "manual", "offset_bins": 2}}],
)
def test_resolved_manual_origins_own_matched_background_and_generator(timing):
    from asymmetry.core.pull_request import make_pull_request

    template = _small_template()
    template.histograms[0].counts[:] = np.r_[2.0, 2.0, np.full(98, 102.0)]
    template.histograms[1].counts[:] = np.r_[2.0, 2.0, np.full(98, 82.0)]
    template.grouping.update(timing, first_good_bin=2)
    request = make_pull_request(
        template, _constant, {"a": 20}, total_events=None, background_per_bin=None, n_seeds=1
    )
    config = request.to_dict()
    assert config["background_per_bin"] == 2
    assert config["total_events"] == 17640
    assert [h["t0_bin"] for h in config["template"]["histograms"]] == [2, 2]


def test_synthetic_refit_attribution_uses_common_recipe_reference():
    import json

    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.pull_refit import EngineRefit

    result = run_pull_distribution(
        _small_template(),
        _constant,
        {"a": 20},
        EngineRefit(_constant, ParameterSet([Parameter("a", 18)])),
        total_events=1e6,
        n_seeds=2,
    )
    for attempt in result.attempts:
        attribution = json.loads(attempt.refit_provenance_json)
        assert attribution["input_attribution"]["n"] == 100
        assert "effective_recipe" not in attribution["input_attribution"]
        assert attribution["observation_report"]["used"] == 100
        assert attempt.recipe_index == 0
