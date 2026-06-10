"""Tests for the synthetic-run sampling core (asymmetry.core.simulate).

Verification-plan §1 (forward model) and §4 (degrade statistics) of
docs/porting/simulate-mode/verification-plan.md.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.simulate import (
    BUILTIN_TEMPLATES,
    GroupSignalSpec,
    build_builtin_template,
    build_group_signals,
    build_run_from_detector_asymmetries,
    degrade_run,
    expected_counts,
    group_specs_from_grouped_fit,
    poisson_asymmetry_errors,
    reduce_run_to_dataset,
    simulate_multi_group_run,
    simulate_run,
    simulate_run_from_group_signals,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US

N_BINS = 2000
BIN_WIDTH = 0.016
T0_BIN = 40


def _template(
    *,
    alpha: float = 1.0,
    n_det_per_group: int = 1,
    t0_bins: list[int] | None = None,
    n_bins: int = N_BINS,
) -> Run:
    """An F/B instrument template with empty histograms (counts unused)."""
    n_det = 2 * n_det_per_group
    if t0_bins is None:
        t0_bins = [T0_BIN] * n_det
    histograms = [
        Histogram(
            counts=np.zeros(n_bins),
            bin_width=BIN_WIDTH,
            t0_bin=t0,
            good_bin_start=max(t0_bins) + 5,
            good_bin_end=n_bins - 10,
        )
        for t0 in t0_bins
    ]
    forward_ids = list(range(1, n_det_per_group + 1))
    backward_ids = list(range(n_det_per_group + 1, n_det + 1))
    grouping = {
        "groups": {1: forward_ids, 2: backward_ids},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": alpha,
        "t0_bin": max(t0_bins),
        "first_good_bin": max(t0_bins) + 5,
        "last_good_bin": n_bins - 10,
        "bin_index_base": 0,
        "bunching_factor": 1,
        "deadtime_correction": True,
        "dead_time_us": [0.008] * n_det,
        "good_frames": 25000.0,
    }
    return Run(
        run_number=1234,
        histograms=histograms,
        metadata={
            "title": "Template",
            "temperature": 5.0,
            "field": 100.0,
            "instrument": "EMU",
        },
        grouping=grouping,
        source_file="/data/template.nxs",
    )


def _run_from_expected(template: Run, expected: list[np.ndarray]) -> Run:
    """Wrap noise-free expected counts in a Run for expectation-mode reduction."""
    histograms = [
        Histogram(
            counts=np.asarray(clean, dtype=float),
            bin_width=hist.bin_width,
            t0_bin=hist.t0_bin,
            good_bin_start=hist.good_bin_start,
            good_bin_end=hist.good_bin_end,
        )
        for hist, clean in zip(template.histograms, expected, strict=True)
    ]
    return Run(
        run_number=template.run_number,
        histograms=histograms,
        metadata=dict(template.metadata),
        grouping=copy.deepcopy(template.grouping),
    )


def _zero_model(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def _exp_model(t: np.ndarray, a0: float = 20.0, rate: float = 0.5) -> np.ndarray:
    return a0 * np.exp(-rate * t)


# ---------------------------------------------------------------------------
# §1 Forward-model correctness
# ---------------------------------------------------------------------------


class TestEnvelopeNormalisation:
    def test_expected_total_matches_event_budget(self) -> None:
        template = _template()
        total_events = 4.0e6
        expected = expected_counts(template, {}, total_events=total_events)

        window_us = (N_BINS - T0_BIN) * BIN_WIDTH
        budget = total_events * (1.0 - np.exp(-window_us / MUON_LIFETIME_US))
        assert np.isclose(sum(float(e.sum()) for e in expected), budget, rtol=1e-12)

    def test_per_bin_means_follow_lifetime_envelope(self) -> None:
        template = _template()
        expected = expected_counts(template, {}, total_events=1.0e6)[0]
        post = expected[T0_BIN:]
        t = np.arange(post.size) * BIN_WIDTH
        ratio = post / post[0]
        assert np.allclose(ratio, np.exp(-t / MUON_LIFETIME_US), rtol=1e-10)

    def test_sampled_total_within_counting_errors(self) -> None:
        template = _template()
        total_events = 2.0e6
        run = simulate_run(template, _zero_model, total_events=total_events, seed=7)
        sampled = sum(float(h.counts.sum()) for h in run.histograms)
        budget = total_events * (1.0 - np.exp(-(N_BINS - T0_BIN) * BIN_WIDTH / MUON_LIFETIME_US))
        assert abs(sampled - budget) < 5.0 * np.sqrt(budget)


class TestAlphaSplit:
    def test_group_rates_in_alpha_ratio(self) -> None:
        alpha = 1.6
        template = _template(alpha=alpha, n_det_per_group=2)
        run = simulate_run(template, _zero_model, total_events=8.0e6, seed=3)
        fwd = sum(float(run.histograms[i].counts.sum()) for i in (0, 1))
        bwd = sum(float(run.histograms[i].counts.sum()) for i in (2, 3))
        assert np.isclose(fwd / bwd, alpha, rtol=5e-3)

    def test_alpha_cancels_in_reduction(self) -> None:
        """Generation α and reduction α cancel: zero signal reduces to A ≈ 0."""
        alpha = 1.4
        template = _template(alpha=alpha)
        run = simulate_run(template, _zero_model, total_events=5.0e6, seed=11)
        dataset = reduce_run_to_dataset(run)
        # Weighted mean asymmetry consistent with zero.
        w = 1.0 / dataset.error**2
        mean = float(np.sum(w * dataset.asymmetry) / np.sum(w))
        sigma = float(1.0 / np.sqrt(np.sum(w)))
        assert abs(mean) < 4.0 * sigma

    def test_unequal_group_sizes_keep_alpha_ratio(self) -> None:
        """The α split fixes group TOTALS, so 1F vs 3B detectors still gives F/B = α."""
        alpha = 1.3
        n_bins = 1000
        histograms = [
            Histogram(counts=np.zeros(n_bins), bin_width=BIN_WIDTH, t0_bin=T0_BIN) for _ in range(4)
        ]
        template = Run(
            run_number=77,
            histograms=histograms,
            metadata={"title": "uneven"},
            grouping={
                "groups": {1: [1], 2: [2, 3, 4]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": alpha,
                "first_good_bin": T0_BIN,
                "last_good_bin": n_bins - 1,
            },
        )
        run = simulate_run(template, _zero_model, total_events=8.0e6, seed=13)
        fwd = float(run.histograms[0].counts.sum())
        bwd = sum(float(run.histograms[i].counts.sum()) for i in (1, 2, 3))
        assert np.isclose(fwd / bwd, alpha, rtol=1e-2)

        # And therefore zero signal still reduces to zero asymmetry.
        dataset = reduce_run_to_dataset(run)
        w = 1.0 / dataset.error**2
        mean = float(np.sum(w * dataset.asymmetry) / np.sum(w))
        sigma = float(1.0 / np.sqrt(np.sum(w)))
        assert abs(mean) < 4.0 * sigma

    def test_total_budget_independent_of_alpha(self) -> None:
        template_balanced = _template(alpha=1.0)
        template_skewed = _template(alpha=2.5)
        kwargs = {"total_events": 1.0e6}
        run_b = simulate_run(template_balanced, _zero_model, seed=1, **kwargs)
        run_s = simulate_run(template_skewed, _zero_model, seed=1, **kwargs)
        total_b = sum(float(h.counts.sum()) for h in run_b.histograms)
        total_s = sum(float(h.counts.sum()) for h in run_s.histograms)
        assert np.isclose(total_b, total_s, rtol=5e-3)


class TestSignalForwarding:
    def test_expectation_mode_recovers_model_exactly(self) -> None:
        """Reduction of the noise-free expectation reproduces A·P(t) bin-by-bin."""
        alpha = 1.3
        template = _template(alpha=alpha)

        def fractional(t: np.ndarray) -> np.ndarray:
            return _exp_model(t) / 100.0

        weights = {1: 2.0 * alpha / (1.0 + alpha), 2: 2.0 / (1.0 + alpha)}
        expected = expected_counts(
            template,
            {1: fractional, 2: lambda t: -fractional(t)},
            total_events=1.0e6,
            group_weights=weights,
        )
        dataset = reduce_run_to_dataset(_run_from_expected(template, expected))
        assert np.allclose(dataset.asymmetry, _exp_model(dataset.time), rtol=1e-9)

    def test_sampled_signal_recovers_model_within_errors(self) -> None:
        template = _template()
        run = simulate_run(template, _exp_model, total_events=5.0e7, seed=5)
        dataset = reduce_run_to_dataset(run)
        pulls = (dataset.asymmetry - _exp_model(dataset.time)) / dataset.error
        assert abs(float(pulls.mean())) < 4.0 / np.sqrt(pulls.size)
        assert 0.9 < float(pulls.std()) < 1.1

    def test_array_signal_accepted(self) -> None:
        template = _template()
        t_post = np.arange(N_BINS - T0_BIN) * BIN_WIDTH
        arr = 0.2 * np.exp(-0.5 * t_post)
        expected_arr = expected_counts(template, {1: arr}, total_events=1.0e6)
        expected_fn = expected_counts(
            template, {1: lambda t: 0.2 * np.exp(-0.5 * t)}, total_events=1.0e6
        )
        assert np.allclose(expected_arr[0], expected_fn[0], rtol=1e-12)


class TestPerDetectorT0:
    def test_staggered_t0_signals_align_in_time(self) -> None:
        template = _template(n_det_per_group=2, t0_bins=[40, 55, 47, 62])

        def fractional(t: np.ndarray) -> np.ndarray:
            return 0.2 * np.cos(2.0 * np.pi * 1.3 * t)

        weights = {1: 1.0, 2: 1.0}
        expected = expected_counts(
            template,
            {1: fractional, 2: lambda t: -fractional(t)},
            total_events=1.0e6,
            group_weights=weights,
        )
        # Each detector's signal starts at its own t0 bin.
        for det, hist in enumerate(template.histograms):
            assert np.all(expected[det][: hist.t0_bin] == 0.0)
        dataset = reduce_run_to_dataset(_run_from_expected(template, expected))
        # Aligned reduction reproduces the model on the shared time axis.
        assert np.allclose(
            dataset.asymmetry, 100.0 * fractional(dataset.time), rtol=1e-9, atol=1e-12
        )


class TestPreT0AndBackground:
    def test_pre_t0_bins_are_zero_without_background(self) -> None:
        template = _template()
        run = simulate_run(template, _exp_model, total_events=1.0e6, seed=2)
        for hist in run.histograms:
            assert np.all(hist.counts[:T0_BIN] == 0.0)

    def test_background_is_flat_everywhere(self) -> None:
        template = _template()
        bg = 7.5
        expected = expected_counts(template, {}, total_events=1.0e6, background_per_bin=bg)[0]
        assert np.allclose(expected[:T0_BIN], bg)
        post = expected[T0_BIN:]
        t = np.arange(post.size) * BIN_WIDTH
        envelope = (post[0] - bg) * np.exp(-t / MUON_LIFETIME_US)
        assert np.allclose(post, envelope + bg, rtol=1e-10)


class TestDeterminismAndProvenance:
    def test_same_seed_is_bit_identical(self) -> None:
        template = _template()
        run_a = simulate_run(template, _exp_model, total_events=1.0e6, seed=42)
        run_b = simulate_run(template, _exp_model, total_events=1.0e6, seed=42)
        for h_a, h_b in zip(run_a.histograms, run_b.histograms, strict=True):
            assert np.array_equal(h_a.counts, h_b.counts)

    def test_different_seed_differs(self) -> None:
        template = _template()
        run_a = simulate_run(template, _exp_model, total_events=1.0e6, seed=1)
        run_b = simulate_run(template, _exp_model, total_events=1.0e6, seed=2)
        assert any(
            not np.array_equal(h_a.counts, h_b.counts)
            for h_a, h_b in zip(run_a.histograms, run_b.histograms, strict=True)
        )

    def test_provenance_metadata(self) -> None:
        template = _template()
        run = simulate_run(
            template,
            _exp_model,
            {"a0": 18.0, "rate": 0.7},
            total_events=1.0e6,
            seed=9,
            run_number=90001,
        )
        assert run.metadata["synthetic"] is True
        assert run.run_number == 90001
        assert run.metadata["run_label"] == "SIM 90001"
        sim = run.metadata["simulation"]
        assert sim["seed"] == 9
        assert sim["total_events"] == 1.0e6
        assert sim["parameters"] == {"a0": 18.0, "rate": 0.7}
        assert sim["template_run_number"] == 1234
        assert sim["template_source_file"] == "/data/template.nxs"
        assert sim["alpha"] == 1.0
        # Instrument context survives; deadtimes are zeroed.
        assert run.metadata["instrument"] == "EMU"
        assert run.grouping["deadtime_correction"] is False
        assert run.grouping["dead_time_us"] == [0.0, 0.0]
        assert run.source_file == ""

    def test_composite_model_binding(self) -> None:
        """A CompositeModel evaluates through .function with bound parameters."""
        pytest.importorskip("iminuit")
        from asymmetry.core.fitting.composite import CompositeModel

        template = _template()
        model = CompositeModel(["Exponential"])
        params = dict(model.param_defaults)
        run = simulate_run(template, model, params, total_events=1.0e6, seed=4)
        assert run.metadata["simulation"]["model"] == model.formula_string()
        assert run.metadata["simulation"]["parameters"] == pytest.approx(params)


class TestValidation:
    def test_rejects_bad_event_budget(self) -> None:
        template = _template()
        with pytest.raises(ValueError, match="total_events"):
            simulate_run(template, _zero_model, total_events=0.0)

    def test_rejects_negative_background(self) -> None:
        template = _template()
        with pytest.raises(ValueError, match="background"):
            simulate_run(template, _zero_model, total_events=1e6, background_per_bin=-1.0)

    def test_rejects_bad_alpha(self) -> None:
        template = _template()
        with pytest.raises(ValueError, match="alpha"):
            simulate_run(template, _zero_model, total_events=1e6, alpha=-2.0)

    def test_rejects_template_without_grouping(self) -> None:
        template = _template()
        template.grouping = {}
        with pytest.raises(ValueError, match="grouping"):
            simulate_run(template, _zero_model, total_events=1e6)

    def test_rejects_template_without_histograms(self) -> None:
        template = _template()
        template.histograms = []
        with pytest.raises(ValueError, match="histograms"):
            simulate_run(template, _zero_model, total_events=1e6)

    def test_rejects_non_model(self) -> None:
        template = _template()
        with pytest.raises(TypeError, match="model"):
            simulate_run(template, object(), total_events=1e6)


# ---------------------------------------------------------------------------
# §4 Degrade statistics
# ---------------------------------------------------------------------------


def _flat_rate_run(rate: float = 100.0, n_bins: int = 4000, seed: int = 99) -> Run:
    """A run whose bins are iid Poisson(rate) — known λ for thinning checks."""
    template = _template(n_bins=n_bins)
    return simulate_run_from_group_signals(
        template,
        {},
        total_events=1.0,  # negligible envelope
        seed=seed,
        background_per_bin=rate,
    )


class TestDegrade:
    def test_mean_scales_with_factor(self) -> None:
        source = simulate_run(_template(), _zero_model, total_events=4.0e6, seed=1)
        thinned = degrade_run(source, 0.25, seed=2)
        total = sum(float(h.counts.sum()) for h in source.histograms)
        total_thinned = sum(float(h.counts.sum()) for h in thinned.histograms)
        assert abs(total_thinned - 0.25 * total) < 5.0 * np.sqrt(0.25 * total)

    def test_errors_scale_as_inverse_sqrt_factor(self) -> None:
        factor = 0.25
        source = simulate_run(_template(), _exp_model, total_events=2.0e7, seed=3)
        thinned = degrade_run(source, factor, seed=4)
        err_source = reduce_run_to_dataset(source).error
        err_thinned = reduce_run_to_dataset(thinned).error
        ratio = float(np.median(err_thinned / err_source))
        assert np.isclose(ratio, 1.0 / np.sqrt(factor), rtol=0.05)

    def test_thinning_is_exactly_poisson(self) -> None:
        """Binomial thinning of Poisson(λ) bins is Poisson(λf): variance ≈ mean."""
        factor = 0.25
        thinned = degrade_run(_flat_rate_run(rate=100.0), factor, seed=5)
        bins = np.concatenate([h.counts for h in thinned.histograms])
        fano = float(bins.var() / bins.mean())
        assert np.isclose(bins.mean(), 25.0, rtol=0.05)
        assert abs(fano - 1.0) < 5.0 * np.sqrt(2.0 / bins.size)

    def test_upscaling_is_overdispersed(self) -> None:
        """Poisson(k·f) given k ~ Poisson(λ) has variance/mean ≈ 1 + f."""
        factor = 4.0
        upscaled = degrade_run(_flat_rate_run(rate=100.0), factor, seed=6)
        bins = np.concatenate([h.counts for h in upscaled.histograms])
        fano = float(bins.var() / bins.mean())
        assert np.isclose(bins.mean(), 400.0, rtol=0.05)
        assert 1.0 + factor - 1.0 < fano < 1.0 + factor + 1.0

    def test_source_run_untouched_and_provenance(self) -> None:
        source = simulate_run(_template(), _zero_model, total_events=1.0e6, seed=7)
        before = [h.counts.copy() for h in source.histograms]
        derived = degrade_run(source, 0.5, seed=8, run_number=90002)
        for hist, original in zip(source.histograms, before, strict=True):
            assert np.array_equal(hist.counts, original)
        assert derived.run_number == 90002
        info = derived.metadata["degraded"]
        assert info["factor"] == 0.5
        assert info["seed"] == 8
        assert info["source_run_number"] == source.run_number
        assert "×0.5" in derived.metadata["run_label"]

    def test_determinism(self) -> None:
        source = _flat_rate_run()
        a = degrade_run(source, 0.5, seed=10)
        b = degrade_run(source, 0.5, seed=10)
        for h_a, h_b in zip(a.histograms, b.histograms, strict=True):
            assert np.array_equal(h_a.counts, h_b.counts)

    def test_identity_factor(self) -> None:
        source = _flat_rate_run()
        same = degrade_run(source, 1.0, seed=11)
        for h_a, h_b in zip(source.histograms, same.histograms, strict=True):
            assert np.array_equal(h_a.counts, h_b.counts)

    def test_rejects_bad_factor(self) -> None:
        source = _flat_rate_run()
        with pytest.raises(ValueError, match="factor"):
            degrade_run(source, 0.0)
        with pytest.raises(ValueError, match="factor"):
            degrade_run(source, float("nan"))

    def test_source_file_metadata_blanked(self) -> None:
        """The derived run must not masquerade as loaded from the source's file."""
        source = _flat_rate_run()
        source.metadata["source_file"] = "/data/REAL00001.nxs"
        derived = degrade_run(source, 0.5, seed=1)
        assert derived.source_file == ""
        assert derived.metadata["source_file"] == ""
        # Source untouched.
        assert source.metadata["source_file"] == "/data/REAL00001.nxs"

    def test_good_frames_scaled_by_factor(self) -> None:
        """Thinning by f is a measurement f times shorter — frames must scale."""
        source = _flat_rate_run()
        assert source.grouping["good_frames"] == 25000.0
        derived = degrade_run(source, 0.25, seed=2)
        assert derived.grouping["good_frames"] == pytest.approx(6250.0)
        # Inherited deadtimes stay (the instrument's own values), with the
        # correction now consistent: counts and frames scale together.
        assert derived.grouping["dead_time_us"] == source.grouping["dead_time_us"]


def _two_period_run() -> Run:
    """A combined-style two-period run: A = +60 % in red, −60 % in green."""
    n_bins = 400

    def _hists(forward_rate: float, backward_rate: float) -> list[Histogram]:
        rng = np.random.default_rng(101)
        return [
            Histogram(
                counts=rng.poisson(rate, n_bins).astype(float),
                bin_width=BIN_WIDTH,
                t0_bin=0,
                good_bin_start=0,
                good_bin_end=n_bins - 1,
            )
            for rate in (forward_rate, backward_rate)
        ]

    red = _hists(800.0, 200.0)
    green = _hists(200.0, 800.0)
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": n_bins - 1,
        "good_frames": 1000.0,
        "period_histograms": [red, green],
        "period_mode": "green",
        "period_good_frames": [1000.0, 1000.0],
    }
    histograms = [
        Histogram(
            counts=h.counts.copy(),
            bin_width=h.bin_width,
            t0_bin=h.t0_bin,
            good_bin_start=h.good_bin_start,
            good_bin_end=h.good_bin_end,
        )
        for h in red
    ]
    return Run(
        run_number=555,
        histograms=histograms,
        metadata={"title": "two period"},
        grouping=grouping,
    )


class TestTwoPeriodHandling:
    def test_reduce_honours_period_mode(self) -> None:
        run = _two_period_run()
        green_view = reduce_run_to_dataset(run)
        assert float(green_view.asymmetry.mean()) == pytest.approx(-60.0, abs=2.0)

        run.grouping["period_mode"] = "red"
        red_view = reduce_run_to_dataset(run)
        assert float(red_view.asymmetry.mean()) == pytest.approx(60.0, abs=2.0)

        run.grouping["period_mode"] = "green_minus_red"
        diff_view = reduce_run_to_dataset(run)
        assert float(diff_view.asymmetry.mean()) == pytest.approx(-120.0, abs=3.0)

    def test_degrade_thins_all_periods_and_keeps_payload(self) -> None:
        run = _two_period_run()
        derived = degrade_run(run, 0.25, seed=4)

        periods = derived.grouping["period_histograms"]
        assert len(periods) == 2
        # Both periods thinned to a quarter of their rates (800→200, 200→50).
        assert float(periods[0][0].counts.mean()) == pytest.approx(200.0, rel=0.05)
        assert float(periods[1][0].counts.mean()) == pytest.approx(50.0, rel=0.05)
        # run.histograms mirrors thinned period 0 (the loader convention).
        assert np.array_equal(derived.histograms[0].counts, periods[0][0].counts)
        # Frames scaled everywhere; mode preserved; reductions recomputed.
        assert derived.grouping["good_frames"] == pytest.approx(250.0)
        assert derived.grouping["period_good_frames"] == [
            pytest.approx(250.0),
            pytest.approx(250.0),
        ]
        assert derived.grouping["period_mode"] == "green"
        assert len(derived.grouping["period_reduced"]) == 2

        # The derived run still reduces as the period the user was viewing.
        green_view = reduce_run_to_dataset(derived)
        assert float(green_view.asymmetry.mean()) == pytest.approx(-60.0, abs=3.0)


class TestBunching:
    def test_reduce_applies_bunching_factor(self) -> None:
        template = _template()
        run = simulate_run(template, _exp_model, total_events=1.0e7, seed=21)
        plain = reduce_run_to_dataset(run)

        run.grouping["bunching_factor"] = 4
        bunched = reduce_run_to_dataset(run)
        assert bunched.n_points == plain.n_points // 4
        spacing = float(np.median(np.diff(bunched.time)))
        assert spacing == pytest.approx(4.0 * BIN_WIDTH, rel=1e-6)


# ---------------------------------------------------------------------------
# Promoted archetype helpers
# ---------------------------------------------------------------------------


class TestPromotedHelpers:
    def test_poisson_asymmetry_errors_shape_and_scale(self) -> None:
        asym = np.zeros(100)
        err = poisson_asymmetry_errors(asym, counts_per_bin=1.0e4)
        assert err.shape == asym.shape
        assert np.allclose(err, 100.0 / np.sqrt(1.0e4))

    def test_build_run_from_detector_asymmetries(self) -> None:
        rng = np.random.default_rng(1)
        t_post = np.arange(2300) * 0.005
        detectors = [
            {"label": "F", "asymmetry": +0.2 * np.exp(-0.3 * t_post)},
            {"label": "B", "asymmetry": -0.2 * np.exp(-0.3 * t_post)},
        ]
        run, time_axis, asym, err = build_run_from_detector_asymmetries(
            run_number=7777,
            detector_asymmetries=detectors,
            title="unit",
            temperature_k=1.0,
            field_g=0.0,
            rng=rng,
        )
        assert isinstance(run, Run)
        assert len(run.histograms) == 2
        # 1-based detector numbers, the convention resolve_group_indices decodes.
        assert run.grouping["groups"] == {1: [1], 2: [2]}
        assert time_axis.size == asym.size == err.size
        # Early-time asymmetry near the generating 20 %.
        assert abs(float(asym[:50].mean()) - 20.0) < 2.0
        # The grouping must reduce to the same signal (guards the 1-based
        # convention end-to-end through resolve_group_indices).
        reduced = reduce_run_to_dataset(run)
        assert abs(float(reduced.asymmetry[:50].mean()) - 20.0) < 2.0


# ---------------------------------------------------------------------------
# Reduction helper
# ---------------------------------------------------------------------------


class TestReduceRunToDataset:
    def test_units_axis_and_metadata(self) -> None:
        template = _template()
        run = simulate_run(template, _exp_model, total_events=1.0e7, seed=12)
        dataset = reduce_run_to_dataset(run)
        assert isinstance(dataset, MuonDataset)
        first_good = template.grouping["first_good_bin"]
        assert np.isclose(dataset.time[0], (first_good - T0_BIN) * BIN_WIDTH)
        assert dataset.run is run
        assert dataset.metadata["synthetic"] is True
        # Percent scale: early-time asymmetry near 20, not 0.2.
        assert 10.0 < float(dataset.asymmetry[:20].mean()) < 30.0


class TestBuiltinTemplates:
    """Built-in idealised instrument templates (no loaded run required)."""

    def test_registry_has_pulsed_and_continuous(self) -> None:
        assert "ideal_pulsed_fb" in BUILTIN_TEMPLATES
        assert "ideal_continuous_fb" in BUILTIN_TEMPLATES

    def test_pulsed_geometry(self) -> None:
        run = build_builtin_template("ideal_pulsed_fb")
        assert len(run.histograms) == 64
        assert run.histograms[0].n_bins == 2000
        assert np.isclose(run.histograms[0].bin_width, 0.016)
        assert run.grouping["groups"][1] == list(range(1, 33))
        assert run.grouping["groups"][2] == list(range(33, 65))
        # Counts are unused structure carriers — they must be all-zero.
        assert all(float(h.counts.sum()) == 0.0 for h in run.histograms)
        # Deadtimes are zero and correction is off (synthetic counts are clean).
        assert run.grouping["deadtime_correction"] is False
        assert run.grouping["dead_time_us"] == [0.0] * 64

    def test_continuous_geometry_and_window(self) -> None:
        run = build_builtin_template("ideal_continuous_fb")
        assert len(run.histograms) == 2
        assert run.histograms[0].n_bins == 10000
        assert np.isclose(run.histograms[0].bin_width, 0.001)
        # 10 μs window at 1 ns binning.
        assert np.isclose(run.histograms[0].n_bins * run.histograms[0].bin_width, 10.0)

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown built-in"):
            build_builtin_template("no_such_instrument")

    def test_simulate_from_builtin_round_trips_through_writer(self, tmp_path) -> None:
        """A built-in template simulates, writes NeXus and reloads bit-exact."""
        h5py = pytest.importorskip("h5py")  # noqa: F841
        from asymmetry.core.io import load
        from asymmetry.core.io.nexus_writer import write_nexus_v1

        template = build_builtin_template("ideal_pulsed_fb")
        run = simulate_run(
            template,
            _exp_model,
            {"a0": 20.0, "rate": 0.4},
            total_events=BUILTIN_TEMPLATES["ideal_pulsed_fb"].default_total_events,
            seed=3,
            run_number=90001,
        )
        path = tmp_path / "builtin.nxs"
        write_nexus_v1(run, path)
        reloaded = load(path).run
        assert len(reloaded.histograms) == len(run.histograms)
        for original, again in zip(run.histograms, reloaded.histograms, strict=True):
            assert np.array_equal(again.counts, original.counts)

    def test_builtin_refit_recovers_parameters(self) -> None:
        """Simulate from the pulsed built-in and refit to the generating values."""
        pytest.importorskip("iminuit")
        from asymmetry.core.fitting.engine import FitEngine
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet

        template = build_builtin_template("ideal_pulsed_fb")
        truth = {"a0": 22.0, "rate": 0.45}
        run = simulate_run(
            template,
            _exp_model,
            truth,
            total_events=BUILTIN_TEMPLATES["ideal_pulsed_fb"].default_total_events,
            seed=11,
            run_number=90001,
        )
        dataset = reduce_run_to_dataset(run)
        params = ParameterSet(
            [
                Parameter(name="a0", value=10.0, min=0.0, max=100.0),
                Parameter(name="rate", value=1.5, min=0.0, max=10.0),
            ]
        )
        # Window to the healthy-count region (early time), as a real fit would.
        result = FitEngine().fit(dataset, _exp_model, params, t_max=8.0)
        assert result.success
        fitted = {p.name: p.value for p in result.parameters}
        assert abs(fitted["a0"] - truth["a0"]) < 3.0 * result.uncertainties["a0"]
        assert abs(fitted["rate"] - truth["rate"]) < 3.0 * result.uncertainties["rate"]

    def test_continuous_background_seeds_flat_offset(self) -> None:
        """The continuous template's flat background lands in the pre-t0 bins."""
        spec = BUILTIN_TEMPLATES["ideal_continuous_fb"]
        assert spec.default_background_per_bin > 0.0
        template = spec.build()
        expected = expected_counts(
            template,
            {1: _zero_model, 2: _zero_model},
            total_events=spec.default_total_events,
            background_per_bin=spec.default_background_per_bin,
        )
        # Pre-t0 bins carry the flat background only.
        assert np.allclose(expected[0][: spec.t0_bin], spec.default_background_per_bin)


def _ring_template(n_groups: int = 4) -> Run:
    """A TF-ring template: one detector per group, all in one grouping."""
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=T0_BIN,
            good_bin_start=T0_BIN,
            good_bin_end=N_BINS - 1,
        )
        for _ in range(n_groups)
    ]
    grouping = {
        "groups": {gid: [gid] for gid in range(1, n_groups + 1)},
        "forward_group": 1,
        "backward_group": 3,
        "alpha": 1.0,
        "t0_bin": T0_BIN,
        "first_good_bin": T0_BIN,
        "last_good_bin": N_BINS - 1,
    }
    return Run(run_number=7, histograms=histograms, grouping=grouping)


def _poln(t: np.ndarray, frequency: float = 2.0, phase: float = 0.0) -> np.ndarray:
    return np.cos(2 * np.pi * frequency * t + phase)


class _FakeFitResult:
    def __init__(self, params: dict[str, float]) -> None:
        self.parameters = [type("P", (), {"name": k, "value": v}) for k, v in params.items()]


class _FakeGroupedResult:
    def __init__(self, group_results: dict) -> None:
        self.group_results = group_results


class TestMultiGroupSimulation:
    def test_build_group_signals_bin_exact_per_group(self) -> None:
        template = _ring_template(4)
        specs = [
            GroupSignalSpec(1, amplitude=0.20, relative_phase=0.0),
            GroupSignalSpec(2, amplitude=0.18, relative_phase=np.pi / 2),
            GroupSignalSpec(3, amplitude=0.22, relative_phase=np.pi),
            GroupSignalSpec(4, amplitude=0.19, relative_phase=3 * np.pi / 2),
        ]
        base = {"frequency": 2.0, "phase": 0.0}
        group_signals, group_weights = build_group_signals(_poln, specs, base_parameters=base)
        expected = expected_counts(
            template, group_signals, total_events=40e6, group_weights=group_weights
        )

        t = np.arange(N_BINS - T0_BIN) * BIN_WIDTH
        for index, spec in enumerate(specs):
            modulation = 1.0 + spec.amplitude * np.cos(2 * np.pi * 2.0 * t + spec.relative_phase)
            envelope = expected[index][T0_BIN:] / modulation
            decay = np.exp(-t / MUON_LIFETIME_US)
            ratio = envelope / decay
            # The de-modulated envelope is a clean lifetime exponential.
            assert np.allclose(ratio, ratio[0], rtol=1e-9), spec.group_id

    def test_non_zero_phase_without_phase_param_raises(self) -> None:
        def no_phase(t, frequency=1.0):
            return np.cos(2 * np.pi * frequency * t)

        with pytest.raises(ValueError, match="phase-capable"):
            build_group_signals(no_phase, [GroupSignalSpec(1, 0.2, relative_phase=0.5)])

    def test_simulate_multi_group_provenance_and_determinism(self) -> None:
        template = _ring_template(4)
        specs = [
            GroupSignalSpec(gid, amplitude=0.2, relative_phase=0.3 * gid) for gid in range(1, 5)
        ]
        base = {"frequency": 2.0, "phase": 0.0}
        run_a = simulate_multi_group_run(
            template, _poln, specs, total_events=20e6, seed=4, base_parameters=base
        )
        run_b = simulate_multi_group_run(
            template, _poln, specs, total_events=20e6, seed=4, base_parameters=base
        )
        assert run_a.metadata["synthetic"] is True
        assert run_a.metadata["simulation"]["multi_group"] is True
        assert len(run_a.metadata["simulation"]["group_specs"]) == 4
        for ha, hb in zip(run_a.histograms, run_b.histograms, strict=True):
            assert np.array_equal(ha.counts, hb.counts)

    def test_group_specs_from_grouped_fit(self) -> None:
        grouped = _FakeGroupedResult(
            {
                1: _FakeFitResult({"amplitude": 0.21, "relative_phase": 0.0, "N0": 1.0}),
                2: _FakeFitResult({"amplitude": 0.19, "relative_phase": 1.57, "N0": 1.2}),
            }
        )
        specs = group_specs_from_grouped_fit(grouped)
        assert {s.group_id for s in specs} == {1, 2}
        spec2 = next(s for s in specs if s.group_id == 2)
        assert spec2.amplitude == pytest.approx(0.19)
        assert spec2.relative_phase == pytest.approx(1.57)
        assert spec2.n0_weight == pytest.approx(1.2)


class TestRunStatistics:
    def test_total_events_of_sums_counts(self) -> None:
        run = simulate_run(_template(), _exp_model, total_events=3.0e6, seed=1)
        from asymmetry.core.simulate import total_events_of

        assert total_events_of(run) == sum(float(h.counts.sum()) for h in run.histograms)

    def test_pulsed_run_has_zero_background_estimate(self) -> None:
        from asymmetry.core.simulate import estimate_background_per_bin

        run = simulate_run(_template(), _exp_model, total_events=3.0e6, seed=1)
        # No background injected and a pre-t0 region present → estimate ~0.
        assert estimate_background_per_bin(run) == 0.0

    def test_matched_statistics_splits_background_off_signal(self) -> None:
        from asymmetry.core.simulate import (
            build_builtin_template,
            estimate_background_per_bin,
            matched_statistics,
            total_events_of,
        )

        template = build_builtin_template("ideal_continuous_fb")
        run = simulate_run(
            template, _zero_model, total_events=20.0e6, seed=2, background_per_bin=10.0
        )
        bg = estimate_background_per_bin(run)
        assert bg > 5.0  # recovers the ~10 counts/bin background from pre-t0
        signal_events, background_per_bin = matched_statistics(run)
        assert background_per_bin == bg
        # Signal budget excludes the background, so it is below the gross total.
        assert signal_events < total_events_of(run)
