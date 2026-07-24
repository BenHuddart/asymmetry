"""Alpha estimation: diamagnetic / General / ratio methods with uncertainties.

WiMDA oracles are transcribed directly from ``Group.pas EstimateButtonClick``
(see docs/porting/data-reduction-parity/comparison.md §1): the coarse-to-fine
grid walk and both objectives. The production diamagnetic estimator must land
within the final grid step of the transcribed walk on identical input; the
General method intentionally diverges from WiMDA's scatter functional
(divergence D14) and is tested against synthetic truth plus the documented
WiMDA failure mode.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform import estimate_alpha, estimate_alpha_detailed
from asymmetry.core.transform.asymmetry import (
    SubtractedBackground,
    _alpha_window,
    _diamagnetic_objective,
    _pack_for_estimation,
    _positive_mask,
)
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.utils.constants import MUON_LIFETIME_US

ALPHA_TRUE = 1.37
A0 = 0.25
N_BINS = 2000
BIN_WIDTH_US = 0.016
TIME_US = np.arange(N_BINS) * BIN_WIDTH_US


def _synthetic_counts(rate_backward: float, polarization: np.ndarray, seed: int):
    """Poisson F/B histograms with known alpha and polarization."""
    rng = np.random.default_rng(seed)
    decay = np.exp(-TIME_US / MUON_LIFETIME_US)
    forward = rng.poisson(ALPHA_TRUE * rate_backward * decay * (1.0 + A0 * polarization))
    backward = rng.poisson(rate_backward * decay * (1.0 - A0 * polarization))
    return forward.astype(np.float64), backward.astype(np.float64)


def _tf_polarization() -> np.ndarray:
    return np.cos(2.0 * np.pi * 1.35 * TIME_US) * np.exp(-0.1 * TIME_US)


def _lf_polarization() -> np.ndarray:
    return np.exp(-0.3 * TIME_US)


# --- WiMDA oracle transcriptions (Group.pas EstimateButtonClick) -------------


def _wimda_general_objective(alpha, f, b, t):
    """Verbatim ``getdevn`` for method = general, including the α clamp."""
    alpha = min(max(alpha, 0.1), 10.0)
    atot = f / np.sqrt(alpha) + b * np.sqrt(alpha)
    aerr = np.sqrt(np.abs(atot))
    scale = np.exp(t / MUON_LIFETIME_US)
    atot = atot * scale
    aerr = aerr * scale
    m1 = np.sum(atot / aerr) / np.sum(1.0 / aerr)
    m2 = np.sum((atot / aerr) ** 2) / np.sum(1.0 / aerr**2)
    c = np.sum(atot / aerr**2) / np.sum(1.0 / aerr**2)
    return np.sqrt(max(m2 + m1 * m1 - 2.0 * m1 * c, 0.0)) / m1


def _wimda_grid_walk(objective, alpha0: float = 1.0) -> float:
    """Verbatim coarse-to-fine walk: steps 0.1 → 0.01 → 0.001, abort at α > 4."""
    alpha = alpha0
    for delta in (0.1, 0.01, 0.001):
        while True:
            if alpha > 4.0:
                return alpha
            t1 = objective(alpha)
            if objective(alpha + delta) < t1:
                alpha += delta
                continue
            if objective(alpha - delta) < t1:
                alpha -= delta
                continue
            break
    return alpha


def _prepared_bins(forward, backward, time_us=None):
    """The same window → pack → mask pipeline the production estimator uses."""
    f, b, t = _alpha_window(forward, backward, time_us, None, None)
    return _positive_mask(*_pack_for_estimation(f, b, t))


# --- diamagnetic --------------------------------------------------------------


def test_diamagnetic_matches_wimda_grid_walk_oracle():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=7)
    f, b, _ = _prepared_bins(forward, backward)
    oracle = _wimda_grid_walk(lambda a: _diamagnetic_objective(a, f, b))
    ours = estimate_alpha_detailed(forward, backward, method="diamagnetic", n_bootstrap=0)
    assert ours.ok
    # Continuous optimiser must land within the oracle's final grid step.
    assert abs(ours.alpha - oracle) < 1.5e-3


def test_diamagnetic_recovers_truth_on_tf_data():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=3)
    est = estimate_alpha_detailed(forward, backward, method="diamagnetic")
    assert est.ok
    assert est.alpha_error is not None
    # Known ~0.3% low bias inherited from WiMDA's σ(α) formula — allow for it.
    assert est.alpha == pytest.approx(ALPHA_TRUE, abs=max(4 * est.alpha_error, 0.02))
    assert est.method == "diamagnetic"
    assert est.objective_value is not None and est.objective_value > 0.0


def test_diamagnetic_profile_error_matches_bootstrap():
    """Δχ² = 1 profile width ≈ bootstrap σ (the O1 cross-check)."""
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=7)
    est = estimate_alpha_detailed(forward, backward, method="diamagnetic")
    assert est.ok and est.alpha_error is not None
    f, b, _ = _prepared_bins(forward, backward)
    s_min = _diamagnetic_objective(est.alpha, f, b)
    sigma_profile = brentq(
        lambda d: _diamagnetic_objective(est.alpha + d, f, b) - s_min - 1.0,
        1e-6,
        0.5,
    )
    assert sigma_profile == pytest.approx(est.alpha_error, rel=0.5)


def test_diamagnetic_independent_of_display_binning_choices():
    """Packing is internal: the estimate is set by the data, not bunching."""
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=11)
    base = estimate_alpha_detailed(forward, backward, method="diamagnetic", n_bootstrap=0)
    # Pre-bunching the input by small integer factors mimics WiMDA's coupling
    # to display bins; the estimate should barely move.
    for factor in (2, 5):
        m = (len(forward) // factor) * factor
        fb = forward[:m].reshape(-1, factor).sum(axis=1)
        bb = backward[:m].reshape(-1, factor).sum(axis=1)
        est = estimate_alpha_detailed(fb, bb, method="diamagnetic", n_bootstrap=0)
        assert est.alpha == pytest.approx(base.alpha, abs=0.01)


# --- general ------------------------------------------------------------------


def test_general_recovers_truth_on_relaxing_lf_data():
    forward, backward = _synthetic_counts(600.0, _lf_polarization(), seed=5)
    est = estimate_alpha_detailed(forward, backward, method="general", time_us=TIME_US)
    assert est.ok
    assert est.alpha_error is not None
    assert est.alpha == pytest.approx(ALPHA_TRUE, abs=3 * est.alpha_error)


def test_general_unbiased_across_seeds():
    estimates = []
    for seed in range(20):
        forward, backward = _synthetic_counts(600.0, _lf_polarization(), seed=seed)
        est = estimate_alpha_detailed(
            forward, backward, method="general", time_us=TIME_US, n_bootstrap=0
        )
        assert est.ok
        estimates.append(est.alpha)
    mean = float(np.mean(estimates))
    sem = float(np.std(estimates) / np.sqrt(len(estimates)))
    assert mean == pytest.approx(ALPHA_TRUE, abs=4 * sem)


def test_general_survives_where_wimda_scatter_walk_collapses():
    """Divergence D14: WiMDA's scatter functional has no interior minimum at
    realistic statistics and its grid walk runs to the clamp; the two-window
    flatness estimator recovers the truth on the same data."""
    forward, backward = _synthetic_counts(600.0, _lf_polarization(), seed=8)
    f, b, t = _prepared_bins(forward, backward, TIME_US)
    wimda = _wimda_grid_walk(lambda a: _wimda_general_objective(a, f, b, t))
    assert wimda < 0.2  # collapsed to (or through) WiMDA's α-clamp
    est = estimate_alpha_detailed(forward, backward, method="general", time_us=TIME_US)
    assert est.ok
    assert est.alpha == pytest.approx(ALPHA_TRUE, abs=3 * (est.alpha_error or 0.2))


def test_general_agrees_with_wimda_walk_at_high_statistics():
    forward, backward = _synthetic_counts(6000.0, _lf_polarization(), seed=9)
    f, b, t = _prepared_bins(forward, backward, TIME_US)
    wimda = _wimda_grid_walk(lambda a: _wimda_general_objective(a, f, b, t))
    est = estimate_alpha_detailed(
        forward, backward, method="general", time_us=TIME_US, n_bootstrap=0
    )
    assert 0.5 < wimda < 4.0  # interior minimum exists at these statistics
    assert est.alpha == pytest.approx(wimda, abs=0.15)


def test_general_fails_informatively_without_relaxation():
    forward, backward = _synthetic_counts(600.0, np.full(N_BINS, 0.8), seed=2)
    est = estimate_alpha_detailed(forward, backward, method="general", time_us=TIME_US)
    assert not est.ok
    assert "contrast" in est.message.lower() or "relax" in est.message.lower()


def test_general_requires_time_axis():
    forward, backward = _synthetic_counts(600.0, _lf_polarization(), seed=1)
    with pytest.raises(ValueError, match="time_us"):
        estimate_alpha_detailed(forward, backward, method="general")


# --- ratio --------------------------------------------------------------------


def test_ratio_matches_legacy_estimate_alpha():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=4)
    est = estimate_alpha_detailed(
        forward,
        backward,
        method="ratio",
        first_good_bin=10,
        last_good_bin=1500,
    )
    legacy = estimate_alpha(forward, backward, first_good_bin=10, last_good_bin=1500)
    assert est.alpha == pytest.approx(legacy, rel=1e-12)
    assert est.ok
    assert est.alpha_error is not None
    assert est.objective_value is None


def test_ratio_is_biased_on_relaxing_data_where_general_is_not():
    """The documented reason the General method exists: ΣF/ΣB absorbs a
    non-zero-mean polarization into alpha."""
    forward, backward = _synthetic_counts(600.0, _lf_polarization(), seed=6)
    ratio = estimate_alpha_detailed(forward, backward, method="ratio", n_bootstrap=0)
    general = estimate_alpha_detailed(
        forward, backward, method="general", time_us=TIME_US, n_bootstrap=0
    )
    assert abs(ratio.alpha - ALPHA_TRUE) > 0.3  # ≈ +a0·⟨P⟩ bias
    assert abs(general.alpha - ALPHA_TRUE) < 0.25


# --- uncertainties ------------------------------------------------------------


def test_bootstrap_sigma_is_calibrated():
    """Reported σ within a factor 2 of the empirical seed-to-seed scatter."""
    values, sigmas = [], []
    for seed in range(15):
        forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=seed)
        est = estimate_alpha_detailed(
            forward, backward, method="diamagnetic", n_bootstrap=100, seed=seed
        )
        assert est.ok and est.alpha_error is not None
        values.append(est.alpha)
        sigmas.append(est.alpha_error)
    empirical = float(np.std(values, ddof=1))
    reported = float(np.mean(sigmas))
    assert 0.5 < reported / empirical < 2.0


def test_bootstrap_is_seeded_and_reproducible():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=12)
    first = estimate_alpha_detailed(forward, backward, method="diamagnetic", seed=42)
    second = estimate_alpha_detailed(forward, backward, method="diamagnetic", seed=42)
    other = estimate_alpha_detailed(forward, backward, method="diamagnetic", seed=43)
    assert first.alpha_error == second.alpha_error
    assert first.alpha_error != other.alpha_error


def test_bootstrap_disabled_gives_no_error():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=12)
    est = estimate_alpha_detailed(forward, backward, method="diamagnetic", n_bootstrap=0)
    assert est.ok
    assert est.alpha_error is None


# --- degenerate input ---------------------------------------------------------


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        estimate_alpha_detailed(np.ones(10), np.ones(10), method="nonsense")


@pytest.mark.parametrize("method", ["diamagnetic", "general", "ratio"])
def test_empty_and_zero_inputs_fail_cleanly(method):
    for forward, backward in (
        (np.empty(0), np.empty(0)),
        (np.zeros(50), np.zeros(50)),
    ):
        est = estimate_alpha_detailed(
            forward,
            backward,
            method=method,
            time_us=np.arange(len(forward), dtype=float) * 0.016,
        )
        assert not est.ok
        assert est.alpha == 1.0
        assert est.alpha_error is None


def test_good_bin_window_is_respected():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=13)
    # Poison the data outside the window; the estimate must not change.
    forward_poisoned = forward.copy()
    forward_poisoned[:10] = 1e9
    forward_poisoned[1801:] = 1e9
    clean = estimate_alpha_detailed(
        forward,
        backward,
        method="diamagnetic",
        first_good_bin=10,
        last_good_bin=1800,
        n_bootstrap=0,
    )
    poisoned = estimate_alpha_detailed(
        forward_poisoned,
        backward,
        method="diamagnetic",
        first_good_bin=10,
        last_good_bin=1800,
        n_bootstrap=0,
    )
    assert poisoned.alpha == pytest.approx(clean.alpha, rel=1e-12)


# --- ratio uncertainty with a subtracted background ---------------------------
#
# Reported from a downstream continuous-source ZF/TF analysis: on a late,
# low-counts window of *background-subtracted* grouped counts the ratio method's
# reported σ_α covered the truth about half as often as it should. Resampling
# the already-subtracted counts loses two variance terms — the subtracted
# constant is missing from the Poisson variance, and the uncertainty on the
# single estimated baseline is fully correlated across every bin in the window.

RATIO_ALPHA_TRUE = 1.42


def _late_window_counts(seed: int, *, n_bkg_bins: int = 120):
    """Late-window F/B counts with a pre-t0 baseline estimated and subtracted.

    Both groups share one signal shape, so ΣF_signal/ΣB_signal is exactly
    ``RATIO_ALPHA_TRUE`` and the ratio estimator is unbiased by construction —
    any coverage failure is the *uncertainty*, not the value. Counts per bin are
    small and the flat background is a sizeable fraction of them, which is the
    regime where the two missing terms dominate.
    """
    rng = np.random.default_rng(seed)
    n_bins = 200
    signal_backward = np.full(n_bins, 9.0)
    level_forward, level_backward = 6.0, 5.0

    raw_forward = rng.poisson(RATIO_ALPHA_TRUE * signal_backward + level_forward)
    raw_backward = rng.poisson(signal_backward + level_backward)

    # The baseline is *estimated* from a separate pre-t0 window, exactly as the
    # reduction does — so it carries its own uncertainty.
    bkg_forward = rng.poisson(level_forward, n_bkg_bins).astype(np.float64)
    bkg_backward = rng.poisson(level_backward, n_bkg_bins).astype(np.float64)
    k_forward = float(np.mean(bkg_forward))
    k_backward = float(np.mean(bkg_backward))

    background = SubtractedBackground(
        forward=k_forward,
        backward=k_backward,
        forward_error=float(np.sqrt(np.sum(bkg_forward)) / n_bkg_bins),
        backward_error=float(np.sqrt(np.sum(bkg_backward)) / n_bkg_bins),
    )
    return (
        raw_forward.astype(np.float64) - k_forward,
        raw_backward.astype(np.float64) - k_backward,
        background,
    )


def _ratio_coverage(*, with_background: bool, realizations: int = 400) -> float:
    inside = 0
    for seed in range(realizations):
        forward, backward, background = _late_window_counts(seed)
        est = estimate_alpha_detailed(
            forward,
            backward,
            method="ratio",
            subtracted_background=background if with_background else None,
        )
        assert est.ok and est.alpha_error is not None
        if abs(est.alpha - RATIO_ALPHA_TRUE) <= est.alpha_error:
            inside += 1
    return inside / realizations


def test_ratio_error_ignoring_the_subtracted_background_under_covers():
    """Without the background terms the interval is far too narrow.

    This is the reported defect, kept as an executable statement of it: the
    nominal 68.3 % interval covers well under 60 % of realizations because both
    the subtracted counts' Poisson variance and the baseline's own (fully
    correlated) uncertainty are missing.
    """
    assert _ratio_coverage(with_background=False) < 0.58


def test_ratio_error_with_the_subtracted_background_is_calibrated():
    assert 0.63 < _ratio_coverage(with_background=True) < 0.74


def test_ratio_error_grows_when_the_background_is_declared():
    forward, backward, background = _late_window_counts(seed=99)
    naive = estimate_alpha_detailed(forward, backward, method="ratio")
    aware = estimate_alpha_detailed(
        forward, backward, method="ratio", subtracted_background=background
    )
    assert naive.alpha == pytest.approx(aware.alpha, rel=1e-12)
    assert aware.alpha_error is not None and naive.alpha_error is not None
    assert aware.alpha_error > 1.5 * naive.alpha_error


def test_ratio_error_matches_the_closed_form_propagation():
    """Pin the algebra: Poisson window sums plus the correlated baseline term."""
    forward, backward, background = _late_window_counts(seed=7)
    est = estimate_alpha_detailed(
        forward, backward, method="ratio", subtracted_background=background
    )

    n_window = float(forward.size)
    sum_f = float(np.sum(forward))
    sum_b = float(np.sum(backward))
    raw_f = sum_f + n_window * background.forward
    raw_b = sum_b + n_window * background.backward
    expected = est.alpha * np.sqrt(
        raw_f / sum_f**2
        + raw_b / sum_b**2
        + (n_window * background.forward_error / sum_f) ** 2
        + (n_window * background.backward_error / sum_b) ** 2
    )
    assert est.alpha_error == pytest.approx(expected, rel=1e-12)


def test_ratio_error_without_a_background_is_the_plain_poisson_form():
    """The no-background answer is unchanged (α·√(1/ΣF + 1/ΣB))."""
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=4)
    est = estimate_alpha_detailed(
        forward, backward, method="ratio", first_good_bin=10, last_good_bin=1500
    )
    window_f = forward[10:1501]
    window_b = backward[10:1501]
    expected = est.alpha * np.sqrt(1.0 / np.sum(window_f) + 1.0 / np.sum(window_b))
    assert est.alpha_error == pytest.approx(expected, rel=1e-12)


def test_subtracted_background_is_ratio_only():
    forward, backward = _synthetic_counts(600.0, _tf_polarization(), seed=4)
    background = SubtractedBackground(1.0, 1.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="ratio"):
        estimate_alpha_detailed(
            forward, backward, method="diamagnetic", subtracted_background=background
        )


def test_ratio_error_is_suppressed_when_bootstrap_is_disabled():
    forward, backward, background = _late_window_counts(seed=3)
    est = estimate_alpha_detailed(
        forward,
        backward,
        method="ratio",
        n_bootstrap=0,
        subtracted_background=background,
    )
    assert est.ok
    assert est.alpha_error is None


def test_corrected_counts_expose_the_subtracted_background() -> None:
    """The reduction hands the estimator what it cannot infer from the arrays."""
    n_bins = 400
    t0_bin = 150
    rng = np.random.default_rng(5)
    histograms = [
        Histogram(
            counts=rng.poisson(40.0, n_bins).astype(np.float64),
            bin_width=0.016,
            t0_bin=t0_bin,
        )
        for _index in range(2)
    ]
    grouping = {
        "background_correction": True,
        "background_mode": "range",
        "background_ranges": [[10, 120], [10, 120]],
    }

    corrected = corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )

    background = corrected.subtracted_background()
    assert background is not None
    assert background.forward == pytest.approx(35.0, abs=15.0)
    # σ on a mean of n Poisson bins: √(total)/n — small but not negligible.
    assert background.forward_error == pytest.approx(np.sqrt(background.forward / 111), rel=0.05)
    assert background.backward_error > 0.0


# --- diamagnetic systematics vs precession frequency --------------------------
#
# Investigated after a downstream report of the diamagnetic α drifting a few
# percent low as the precession frequency rose. Synthetically it does not: the
# estimator carries a small bias that is set by the *asymmetry amplitude* and
# the distance of α from 1, and is flat in frequency. See
# docs/reference/data_reduction/alpha_calibration for the write-up.


def _diamagnetic_bias(
    *, alpha_true: float, amplitude: float, frequency_mhz: float, seeds: int = 12
) -> float:
    """Mean fractional bias of the diamagnetic estimate on synthetic TF data."""
    n_bins = 6000
    bin_width = 0.0008
    times = (np.arange(n_bins) + 0.5) * bin_width
    decay = np.exp(-times / MUON_LIFETIME_US)
    polarization = np.cos(2.0 * np.pi * frequency_mhz * times) * np.exp(-0.15 * times)

    estimates = []
    for seed in range(seeds):
        rng = np.random.default_rng(4000 + seed)
        forward = rng.poisson(alpha_true * 30.0 * decay * (1.0 + amplitude * polarization))
        backward = rng.poisson(30.0 * decay * (1.0 - amplitude * polarization))
        est = estimate_alpha_detailed(
            forward.astype(np.float64),
            backward.astype(np.float64),
            method="diamagnetic",
            n_bootstrap=0,
        )
        assert est.ok
        estimates.append(est.alpha)
    return float(np.mean(estimates) - alpha_true) / alpha_true


def test_diamagnetic_bias_does_not_track_precession_frequency():
    """The reported symptom is absent: no drift from 5 MHz to 27 MHz.

    Each point is small and negative for the same reason (below); what this
    pins is that the *frequency* is not the driver, so a frequency-dependent
    dip in real calibration data points at the instrument, not the estimator.
    """
    biases = [
        _diamagnetic_bias(alpha_true=1.37, amplitude=0.22, frequency_mhz=frequency)
        for frequency in (5.0, 15.0, 27.0)
    ]
    assert all(abs(bias) < 0.012 for bias in biases)
    assert max(biases) - min(biases) < 0.006


def test_diamagnetic_bias_is_set_by_amplitude_and_the_distance_of_alpha_from_one():
    r"""The real systematic: an :math:`O(A^2)` term that changes sign at α = 1.

    Minimising WiMDA's Σ(A/σ)² has the closed-form stationary point
    :math:`α^4 = Σ F^3/(B(F+B)) \big/ Σ B^3/(F(F+B))`. Expanding it in the
    oscillation amplitude ``u = A·P(t)`` leaves the estimating equation
    ``Σ s·u = ((α−1)/(α+1))·Σ s·u²+ O(u³)``: the second-order term does **not**
    vanish for a perfectly zero-mean oscillation, so the estimate is displaced
    by roughly ``A²·(α−1)/(α+1)`` regardless of how much data is collected.
    That is why the residual is a fraction of a percent at a typical 20 %
    asymmetry, grows quadratically with it, and reverses for α < 1.
    """
    small = _diamagnetic_bias(alpha_true=1.37, amplitude=0.08, frequency_mhz=15.0)
    large = _diamagnetic_bias(alpha_true=1.37, amplitude=0.30, frequency_mhz=15.0)
    assert large < small < 0.0
    assert abs(large) > 2.0 * abs(small)

    below_one = _diamagnetic_bias(alpha_true=0.73, amplitude=0.25, frequency_mhz=15.0)
    above_one = _diamagnetic_bias(alpha_true=1.37, amplitude=0.25, frequency_mhz=15.0)
    assert below_one > 0.0 > above_one
