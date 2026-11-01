"""Background modes: tail fit, reference-run subtraction, per-mode gating.

WiMDA's tail estimate weights bins by √N and deletes those with few counts, so
it only functions on heavily bunched display bins (study divergence D4). The
Poisson-MLE replacement is pinned here against synthetic truth, and against one
fixed reference value for WiMDA's estimate on bunched bins.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_tail_fit_agrees_with_wimda_reference_on_bunched_strong_background():
    """Where WiMDA's weighting functions (bunched bins, many counts), the two
    estimates agree within uncertainties."""
    wimda_rate_per_us = 21.1688  # WiMDA's estimate for this seeded run in 100-bin bunches
    counts = _pulsed_counts(30000.0, 20.0, seed=3)
    fit = fit_tail_background(
        counts, bin_width_us=BIN_WIDTH_US, t0_bin=T0_BIN, last_good_bin=N_BINS - 1
    )
    assert fit.ok and fit.rate_error_per_us is not None
    assert fit.rate_per_us == pytest.approx(wimda_rate_per_us, abs=3 * fit.rate_error_per_us)


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
    corrected, variance = subtract_scaled_counts(counts, reference, 2.0)
    assert corrected == pytest.approx([60.0, 30.0, 2.0])
    assert variance == pytest.approx([180.0, 90.0, 26.0])


def test_subtract_scaled_counts_self_subtraction():
    counts = np.array([100.0, 50.0, 10.0])
    corrected, variance = subtract_scaled_counts(counts, counts, 1.0)
    assert corrected == pytest.approx([0.0, 0.0, 0.0])
    assert variance == pytest.approx(2.0 * counts)


def test_subtract_scaled_counts_rejects_different_shapes():
    with pytest.raises(ValueError, match="equal shape"):
        subtract_scaled_counts(np.ones(5), np.ones(3), 1.0)


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
    assert result.outcome.status == "applied"
    assert result.outcome.effective["method"] == "tail_fit"
    assert result.forward.shared
    assert result.outcome.effective["details"] is not None
    assert result.outcome.effective["details"]["forward_rate_per_us"] == pytest.approx(
        result.outcome.effective["details"]["backward_rate_per_us"]
    )
    # values are the per-bin counts subtracted
    assert result.outcome.effective["values"][0] == pytest.approx(
        result.outcome.effective["details"]["forward_rate_per_us"] * BIN_WIDTH_US
    )


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
    assert result.outcome.status == "applied"
    assert result.outcome.effective["method"] == "reference_run"
    assert result.forward.counts == pytest.approx([60.0, 30.0, 2.0])
    assert result.outcome.effective["details"] == {"scale": 2.0}


def test_dispatch_reference_run_without_reference_reports_missing():
    counts = np.array([100.0, 50.0, 10.0])
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "reference_run"},
        t0_bin=1,
        bin_width_us=BIN_WIDTH_US,
    )
    assert result.outcome.status != "applied"
    assert result.outcome.status == "unavailable"
    assert "reference" in result.outcome.reason


def test_dispatch_none_mode():
    counts = np.array([100.0, 50.0, 10.0])
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "none"},
        t0_bin=1,
        bin_width_us=BIN_WIDTH_US,
    )
    assert result.outcome.status != "applied"


def test_legacy_fixed_and_range_paths_unchanged():
    counts = np.arange(100, dtype=np.float64) + 50.0
    fixed = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_fixed_values": [5.0, 7.0]},
        t0_bin=20,
        bin_width_us=BIN_WIDTH_US,
    )
    assert fixed.outcome.status == "applied" and fixed.outcome.effective["method"] == "fixed"
    assert fixed.forward.counts == pytest.approx(counts - 5.0)

    ranged = apply_grouped_background_correction(
        counts,
        counts,
        grouping={},
        t0_bin=20,
        bin_width_us=BIN_WIDTH_US,
    )
    assert ranged.outcome.status == "applied" and ranged.outcome.effective["method"] == "estimated"


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
    assert result.outcome.status == "applied"
    assert result.outcome.effective["method"] == "estimated"


def test_explicit_fixed_mode_without_values_does_not_fall_through_to_range():
    """Review fix: an explicit fixed mode with missing values must not
    silently subtract a pre-t0 range mean instead."""
    counts = np.arange(100, dtype=np.float64) + 50.0
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "fixed"},
        t0_bin=20,
        bin_width_us=BIN_WIDTH_US,
    )
    assert result.outcome.status != "applied"
    assert result.outcome.status == "unavailable"
    assert "fixed" in result.outcome.reason
    np.testing.assert_array_equal(result.forward.counts, counts)


def test_fourier_group_path_honours_injected_values_under_explicit_range_mode():
    """Review fix: the grouped Fourier input subtracts the per-group values
    recorded by the time-domain reduction even though the persisted grouping
    carries background_mode='range'."""
    from asymmetry.core.fourier.grouped import _apply_group_background_correction

    counts = np.full(200, 80.0)
    grouping = {
        "background_correction": True,
        "background_mode": "range",
        "background_values": [7.0, 3.0],
        "forward_group": 1,
        "backward_group": 2,
    }
    corrected, applied, value = _apply_group_background_correction(
        counts,
        grouping=grouping,
        group_id=2,  # backward group must get ITS value, not the forward one
        t0_bin=20,
        bin_width=BIN_WIDTH_US,
        metadata={"facility": "PSI"},
        source_file="run.bin",
    )
    assert applied
    assert value == pytest.approx(3.0)
    np.testing.assert_allclose(corrected, counts - 3.0)


def test_fourier_group_path_applies_tail_fit_on_pulsed_data():
    """Review fix: pulsed data with tail-fit background corrects the Fourier
    input too (the old range-only gate skipped it)."""
    from asymmetry.core.fourier.grouped import _apply_group_background_correction

    counts = _pulsed_counts(5000.0, 0.8, seed=7)
    grouping = {
        "background_correction": True,
        "background_mode": "tail_fit",
        "last_good_bin": N_BINS - 1,
    }
    corrected, applied, value = _apply_group_background_correction(
        counts,
        grouping=grouping,
        group_id=1,
        t0_bin=T0_BIN,
        bin_width=BIN_WIDTH_US,
        metadata={"facility": "ISIS"},
        source_file="run.nxs",
    )
    assert applied
    assert value is not None and value > 0.0
    assert float(np.sum(corrected)) < float(np.sum(counts))


def test_fourier_group_path_refuses_missing_reference():
    from asymmetry.core.fourier.grouped import _apply_group_background_correction

    with pytest.raises(ValueError, match="reference run is not loaded"):
        _apply_group_background_correction(
            np.array([100.0, 50.0, 10.0]),
            grouping={"background_correction": True, "background_mode": "reference_run"},
            group_id=1,
            t0_bin=1,
            bin_width=BIN_WIDTH_US,
            metadata={},
            source_file="",
        )
