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

from asymmetry.core.transform import estimate_alpha, estimate_alpha_detailed
from asymmetry.core.transform.asymmetry import (
    _alpha_window,
    _diamagnetic_objective,
    _pack_for_estimation,
    _positive_mask,
)
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
