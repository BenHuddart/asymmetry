"""Background modes: tail fit, reference-run subtraction, per-mode gating.

The WiMDA oracle is a transcription of ``Group.pas estBG``/``BGfit``: the
bin-integrated exponential + flat model fitted with √N weights over the late
half of the window, with bins of ≤ 4 counts deleted via σ = 10¹⁰ (study
divergence D4). On fine raw binning that deletion removes essentially every
late-time bin — WiMDA only functions on heavily bunched display bins — which
is pinned here alongside the Poisson-MLE replacement's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from asymmetry.core.transform import (
    apply_grouped_background_correction,
    available_background_modes,
    fit_tail_background,
    resolve_background_mode,
    subtract_scaled_counts,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US

N_BINS = 2000
BIN_WIDTH_US = 0.016
T0_BIN = 10


def _pulsed_counts(rate0_per_us: float, background_per_us: float, seed: int) -> np.ndarray:
    """Poisson histogram: muon exponential from t0 plus a flat rate."""
    rng = np.random.default_rng(seed)
    t = (np.arange(N_BINS) - T0_BIN) * BIN_WIDTH_US
    intensity = rate0_per_us * np.exp(-np.clip(t, 0.0, None) / MUON_LIFETIME_US) * (t >= 0)
    mu = np.clip((intensity + background_per_us) * BIN_WIDTH_US, 0.0, None)
    return rng.poisson(mu).astype(np.float64)


def _wimda_estbg(counts: np.ndarray, width_us: float, t0_bin: int, last: int) -> float:
    """Transcribed ``estBG``: √N weights, ≤ 4-count deletion, late-half window."""
    start = t0_bin + (last - t0_bin) // 2
    window = counts[start : last + 1]
    t = (np.arange(start, last + 1) - t0_bin) * width_us
    sigma = np.where(window > 4, np.sqrt(np.maximum(window, 1e-12)), 1e10)
    x = width_us / MUON_LIFETIME_US
    bin_factor = np.sinh(x / 2.0) / (x / 2.0)

    def residuals(params):
        mu = (params[0] * np.exp(-t / MUON_LIFETIME_US) * bin_factor + params[1]) * width_us
        return (mu - window) / sigma

    return float(least_squares(residuals, [1000.0, 0.001]).x[1])


# --- tail fit -------------------------------------------------------------------


def test_tail_fit_recovers_truth():
    counts = _pulsed_counts(5000.0, 0.8, seed=0)
    fit = fit_tail_background(
        counts, bin_width_us=BIN_WIDTH_US, t0_bin=T0_BIN, last_good_bin=N_BINS - 1
    )
    assert fit.ok
    assert fit.rate_error_per_us is not None
    assert fit.rate_per_us == pytest.approx(0.8, abs=3 * fit.rate_error_per_us)
    assert not fit.consistent_with_zero
    # Default window is the late half of (t0, last_good).
    assert fit.window[0] == T0_BIN + (N_BINS - 1 - T0_BIN) // 2


def test_tail_fit_unbiased_at_low_counts():
    """The D4 justification: Poisson MLE stays unbiased where bins hold ≪ 1
    count, which is the regime the mode exists for."""
    true_rate = 0.05  # 0.0008 counts per raw bin in the tail
    estimates = [
        fit_tail_background(
            _pulsed_counts(3000.0, true_rate, seed=seed),
            bin_width_us=BIN_WIDTH_US,
            t0_bin=T0_BIN,
            last_good_bin=N_BINS - 1,
        ).rate_per_us
        for seed in range(20)
    ]
    mean = float(np.mean(estimates))
    sem = float(np.std(estimates) / np.sqrt(len(estimates)))
    assert mean == pytest.approx(true_rate, abs=4 * max(sem, 1e-3))


def test_wimda_oracle_deletes_the_tail_on_raw_bins():
    """WiMDA's ≤ 4-count rule kills every late raw bin: the fit never moves
    from its start value. Documented failure mode behind divergence D4."""
    counts = _pulsed_counts(3000.0, 0.05, seed=1)
    oracle = _wimda_estbg(counts, BIN_WIDTH_US, T0_BIN, N_BINS - 1)
    assert oracle == pytest.approx(0.001, abs=1e-6)  # the FITE start value


def test_tail_fit_agrees_with_wimda_oracle_on_bunched_strong_background():
    """Where WiMDA's weighting functions (bunched bins, many counts), the two
    estimates agree within uncertainties."""
    bunch = 100
    true_rate = 20.0
    counts = _pulsed_counts(30000.0, true_rate, seed=3)
    usable = (counts.size - T0_BIN) // bunch * bunch
    bunched = counts[T0_BIN : T0_BIN + usable].reshape(-1, bunch).sum(axis=1)
    oracle = _wimda_estbg(bunched, BIN_WIDTH_US * bunch, 0, bunched.size - 1)
    fit = fit_tail_background(
        counts, bin_width_us=BIN_WIDTH_US, t0_bin=T0_BIN, last_good_bin=N_BINS - 1
    )
    assert fit.ok and fit.rate_error_per_us is not None
    assert fit.rate_per_us == pytest.approx(oracle, abs=3 * fit.rate_error_per_us)


def test_tail_fit_zero_background_is_consistent_with_zero():
    counts = _pulsed_counts(3000.0, 0.0, seed=1)
    fit = fit_tail_background(
        counts, bin_width_us=BIN_WIDTH_US, t0_bin=T0_BIN, last_good_bin=N_BINS - 1
    )
    assert fit.ok
    assert fit.consistent_with_zero
    assert fit.rate_per_us < 0.05


def test_tail_fit_failure_modes():
    short = fit_tail_background(np.ones(6), bin_width_us=0.016, t0_bin=0, last_good_bin=5)
    assert not short.ok and "5 bins" in short.message
    empty = fit_tail_background(np.zeros(100), bin_width_us=0.016, t0_bin=0, last_good_bin=99)
    assert not empty.ok and "no counts" in empty.message
    bad_width = fit_tail_background(np.ones(100), bin_width_us=0.0, t0_bin=0)
    assert not bad_width.ok


def test_tail_fit_explicit_window():
    counts = _pulsed_counts(5000.0, 0.8, seed=2)
    fit = fit_tail_background(
        counts,
        bin_width_us=BIN_WIDTH_US,
        t0_bin=T0_BIN,
        last_good_bin=N_BINS - 1,
        fit_start_bin=1500,
    )
    assert fit.ok
    assert fit.window == (1500, N_BINS - 1)


# --- reference-run subtraction --------------------------------------------------


def test_subtract_scaled_counts_algebra():
    counts = np.array([100.0, 50.0, 10.0])
    reference = np.array([20.0, 10.0, 4.0])
    corrected, errors = subtract_scaled_counts(counts, reference, 2.0)
    assert corrected == pytest.approx([60.0, 30.0, 2.0])
    assert errors == pytest.approx(np.sqrt([180.0, 90.0, 26.0]))


def test_subtract_scaled_counts_self_subtraction():
    counts = np.array([100.0, 50.0, 10.0])
    corrected, errors = subtract_scaled_counts(counts, counts, 1.0)
    assert corrected == pytest.approx([0.0, 0.0, 0.0])
    assert errors == pytest.approx(np.sqrt(2.0 * counts))


def test_subtract_scaled_counts_truncates_to_common_length():
    corrected, errors = subtract_scaled_counts(np.ones(5), np.ones(3), 1.0)
    assert corrected.shape == (3,)
    assert errors.shape == (3,)


# --- mode gating and resolution -------------------------------------------------


def test_available_modes_pulsed_vs_continuous():
    pulsed = available_background_modes(metadata={"facility": "ISIS"}, source_file="run.nxs")
    assert "range" not in pulsed
    assert {"fixed", "tail_fit", "reference_run"} <= set(pulsed)
    continuous = available_background_modes(metadata={"facility": "PSI"}, source_file="run.bin")
    assert "range" in continuous


def test_resolve_background_mode_back_compat():
    assert resolve_background_mode({"background_mode": "tail_fit"}) == "tail_fit"
    assert resolve_background_mode({"background_fixed_values": [1.0, 2.0]}) == "fixed"
    assert resolve_background_mode({}) == "range"  # historical single-flag behaviour
    assert resolve_background_mode(None) == "range"
    assert resolve_background_mode({"background_mode": "nonsense"}) == "range"


# --- dispatch through apply_grouped_background_correction ------------------------


def test_dispatch_tail_fit_populates_details():
    counts = _pulsed_counts(5000.0, 0.8, seed=4)
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "tail_fit"},
        t0_bin=T0_BIN,
        bin_width_us=BIN_WIDTH_US,
        last_good_bin=N_BINS - 1,
    )
    assert result.applied
    assert result.method == "tail_fit"
    assert result.forward_error is not None
    assert result.details is not None
    assert result.details["forward_rate_per_us"] == pytest.approx(
        result.details["backward_rate_per_us"]
    )
    # values are the per-bin counts subtracted
    assert result.values[0] == pytest.approx(result.details["forward_rate_per_us"] * BIN_WIDTH_US)


def test_dispatch_reference_run():
    counts = np.array([100.0, 50.0, 10.0])
    reference = np.array([20.0, 10.0, 4.0])
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "reference_run"},
        t0_bin=1,
        bin_width_us=BIN_WIDTH_US,
        reference_forward=reference,
        reference_backward=reference,
        reference_scale=2.0,
    )
    assert result.applied
    assert result.method == "reference_run"
    assert result.forward == pytest.approx([60.0, 30.0, 2.0])
    assert result.details == {"scale": 2.0}


def test_dispatch_reference_run_without_reference_reports_missing():
    counts = np.array([100.0, 50.0, 10.0])
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "reference_run"},
        t0_bin=1,
        bin_width_us=BIN_WIDTH_US,
    )
    assert not result.applied
    assert result.method == "missing_reference"


def test_dispatch_none_mode():
    counts = np.array([100.0, 50.0, 10.0])
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "none"},
        t0_bin=1,
        bin_width_us=BIN_WIDTH_US,
    )
    assert not result.applied


def test_legacy_fixed_and_range_paths_unchanged():
    counts = np.arange(100, dtype=np.float64) + 50.0
    fixed = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_fixed_values": [5.0, 7.0]},
        t0_bin=20,
        bin_width_us=BIN_WIDTH_US,
    )
    assert fixed.applied and fixed.method == "fixed"
    assert fixed.forward == pytest.approx(counts - 5.0)

    ranged = apply_grouped_background_correction(
        counts,
        counts,
        grouping={},
        t0_bin=20,
        bin_width_us=BIN_WIDTH_US,
    )
    assert ranged.applied and ranged.method == "estimated"


def test_explicit_range_mode_ignores_fixed_values():
    counts = np.arange(100, dtype=np.float64) + 50.0
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={
            "background_mode": "range",
            "background_fixed_values": [5.0, 7.0],
            "background_range": [0, 10],
        },
        t0_bin=20,
        bin_width_us=BIN_WIDTH_US,
    )
    assert result.applied
    assert result.method == "estimated"
