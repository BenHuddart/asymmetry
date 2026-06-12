"""Tests for :mod:`asymmetry.core.fourier.moments`.

Moments have closed-form values on analytic distributions, so the core is pinned
without any external corpus: a Gaussian (zero skew, known width), a skewed
two-Gaussian mixture (every moment analytic), a vortex-lattice-like lineshape for
the β sign, plus the transcribed WiMDA oracle. See
``docs/porting/spectral-moments/`` for the study.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.fourier.moments import (
    SpectrumMoments,
    moments_trend_row,
    spectrum_moments,
)
from asymmetry.core.fourier.units import FieldUnit, convert

# Load the hyphen-named porting oracle module by path.
_ORACLE_PATH = (
    Path(__file__).parent / "porting" / "spectral-moments" / "wimda_oracle.py"
)
_spec = importlib.util.spec_from_file_location("_wimda_moments_oracle", _ORACLE_PATH)
assert _spec and _spec.loader
wimda_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wimda_oracle)


# ── helpers ─────────────────────────────────────────────────────────────────


def _gaussian(x: np.ndarray, mu: float, sigma: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _mixture_analytic(
    weights: list[float], mus: list[float], sigmas: list[float]
) -> tuple[float, float, float, float]:
    """Analytic (b_ave, b_rms_mean, gamma1, m3) for a sum-of-Gaussians *amplitude*.

    The integration weight of each Gaussian is its **area** ``w·σ·√(2π)``, not its
    peak height — this is the subtlety that makes the mixture a real test.
    """
    w = np.asarray([wi * si for wi, si in zip(weights, sigmas)], float)  # area weights
    w = w / w.sum()
    mu = np.asarray(mus, float)
    sg = np.asarray(sigmas, float)
    mean = float((w * mu).sum())
    e_b2 = float((w * (sg**2 + mu**2)).sum())
    e_b3 = float((w * (mu**3 + 3 * mu * sg**2)).sum())
    var = e_b2 - mean**2
    m3 = e_b3 - 3 * mean * e_b2 + 2 * mean**3
    return mean, math.sqrt(var), m3 / var**1.5, m3


# ── 1. Gaussian: zero skew, known width ─────────────────────────────────────


def test_gaussian_is_symmetric_with_known_width():
    x = np.linspace(-60.0, 60.0, 6001)
    sigma = 8.0
    amp = _gaussian(x, 0.0, sigma)
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    assert m.b_ave == pytest.approx(0.0, abs=1e-6)
    assert m.b_pk == pytest.approx(0.0, abs=1e-3)
    assert m.b_diff == pytest.approx(0.0, abs=1e-3)
    assert m.b_rms_mean == pytest.approx(sigma, rel=1e-3)
    assert m.b_rms_peak == pytest.approx(sigma, rel=2e-3)
    assert m.skewness == pytest.approx(0.0, abs=1e-3)
    assert m.skewness_g1 == pytest.approx(0.0, abs=1e-3)
    assert m.beta == pytest.approx(0.0, abs=1e-3)
    assert m.peak_refined is True


# ── 2. Skewed mixture: every moment analytic ────────────────────────────────


def test_skewed_mixture_matches_closed_form():
    x = np.linspace(-60.0, 80.0, 14001)
    weights, mus, sigmas = [1.0, 0.35], [0.0, 25.0], [5.0, 6.0]
    amp = sum(_gaussian(x, mu, sg, w) for w, mu, sg in zip(weights, mus, sigmas))
    mean, rms, g1, m3 = _mixture_analytic(weights, mus, sigmas)

    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    assert m.b_ave == pytest.approx(mean, rel=1e-3)
    assert m.b_rms_mean == pytest.approx(rms, rel=1e-3)
    assert m.skewness_g1 == pytest.approx(g1, rel=2e-3)
    # Positive skew → high-field tail → mean above peak → beta > 0.
    assert m.skewness_g1 > 0.0
    assert m.beta > 0.0
    assert m.b_ave > m.b_pk


# ── 3. Vortex-lattice-like lineshape: beta sign convention ───────────────────


def test_vortex_lattice_lineshape_has_positive_beta():
    # Sharp low-field cutoff (saddle point) + long high-field tail (cores):
    # a one-sided, high-field-tailed profile proxying the mixed-state p(B).
    x = np.linspace(0.0, 200.0, 8001)
    b_min = 40.0
    amp = np.where(x >= b_min, np.exp(-(x - b_min) / 18.0), 0.0)
    # Soften the cutoff edge a touch so the discrete peak is well defined.
    amp = amp * (1.0 - np.exp(-np.clip(x - b_min, 0, None) / 1.5))
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.01, uncertainty="none")
    assert m.b_ave > m.b_pk  # mean pulled up by the tail
    assert m.beta > 0.0
    assert m.skewness > 0.0
    assert m.skewness_g1 > 0.0


# ── 4. Parabolic peak refinement + edge guard ───────────────────────────────


def test_parabolic_peak_refines_between_bins():
    # Peak deliberately off-grid (mu between samples) on a coarse grid.
    x = np.linspace(0.0, 100.0, 201)  # 0.5 G spacing
    mu = 50.27
    amp = _gaussian(x, mu, 6.0)
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    assert m.peak_refined is True
    # The refined peak should land nearer the true mu than the nearest bin.
    nearest_bin = x[int(np.argmax(amp))]
    assert abs(m.b_pk - mu) < abs(nearest_bin - mu) + 1e-9
    assert m.b_pk == pytest.approx(mu, abs=0.1)


def test_edge_peak_falls_back_to_discrete_bin():
    # Peak at the very first in-range bin → edge guard fires, no refinement.
    x = np.linspace(0.0, 100.0, 101)
    amp = np.exp(-x / 10.0)  # monotone decreasing → peak at bin 0
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    assert m.peak_refined is False
    assert m.b_pk == pytest.approx(0.0, abs=1e-9)


# ── 5. Cutoff / range sensitivity ───────────────────────────────────────────


def test_tight_range_excludes_satellite_and_symmetrises():
    x = np.linspace(-60.0, 80.0, 14001)
    amp = _gaussian(x, 0.0, 5.0, 1.0) + _gaussian(x, 25.0, 6.0, 0.35)
    full = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    # Restrict to the main line only: skew/beta collapse toward 0.
    tight = spectrum_moments(
        x, amp, x_range=(-20.0, 12.0), cutoff_fraction=0.0, uncertainty="none"
    )
    assert abs(tight.skewness_g1) < abs(full.skewness_g1)
    assert abs(tight.beta) < abs(full.beta)
    assert tight.b_rms_mean < full.b_rms_mean


def test_raising_cutoff_drops_points_and_narrows_width():
    x = np.linspace(-60.0, 60.0, 6001)
    amp = _gaussian(x, 0.0, 8.0)
    low = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    high = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.5, uncertainty="none")
    assert high.n_sample < low.n_sample
    assert high.b_rms_mean < low.b_rms_mean
    # Window recorded in provenance.
    assert high.recipe["cutoff_fraction"] == 0.5


# ── 6. Uncertainties ────────────────────────────────────────────────────────


def test_bootstrap_errors_finite_and_shrink_with_signal():
    x = np.linspace(-60.0, 60.0, 2001)
    amp = _gaussian(x, 0.0, 8.0)
    noisy = spectrum_moments(
        x, amp, x_range=None, cutoff_fraction=0.05,
        errors=0.05 * np.ones_like(x), uncertainty="bootstrap", n_bootstrap=128, seed=3,
    )
    clean = spectrum_moments(
        x, amp, x_range=None, cutoff_fraction=0.05,
        errors=0.005 * np.ones_like(x), uncertainty="bootstrap", n_bootstrap=128, seed=3,
    )
    assert math.isfinite(noisy.b_ave_err) and noisy.b_ave_err > 0
    assert clean.b_ave_err < noisy.b_ave_err  # less noise → tighter error


def test_bootstrap_exposes_b_pk_fragility():
    # A near-flat-topped, noisy spectrum: b_pk hops between bins → large b_pk_err,
    # while the integral b_ave stays well-determined.
    x = np.linspace(-60.0, 60.0, 1201)
    amp = _gaussian(x, 0.0, 30.0)  # very broad / flat top
    m = spectrum_moments(
        x, amp, x_range=None, cutoff_fraction=0.2,
        errors=0.1 * np.ones_like(x), uncertainty="bootstrap", n_bootstrap=128, seed=5,
    )
    assert m.b_pk_err > m.b_ave_err  # peak hops; mean is well-determined
    assert math.isfinite(m.beta_err) and m.beta_err > 0  # beta inherits the peak


def test_bootstrap_is_deterministic_under_seed():
    x = np.linspace(-60.0, 60.0, 1001)
    amp = _gaussian(x, 0.0, 8.0)
    kw = dict(
        x_range=None, cutoff_fraction=0.05, errors=0.05 * np.ones_like(x),
        uncertainty="bootstrap", n_bootstrap=64, seed=7,
    )
    a = spectrum_moments(x, amp, **kw)
    b = spectrum_moments(x, amp, **kw)
    assert a.b_ave_err == b.b_ave_err
    assert a.beta_err == b.beta_err


def test_propagate_gives_integral_errors_and_nan_for_peak():
    x = np.linspace(-60.0, 60.0, 2001)
    amp = _gaussian(x, 0.0, 8.0)
    m = spectrum_moments(
        x, amp, x_range=None, cutoff_fraction=0.05,
        errors=0.05 * np.ones_like(x), uncertainty="propagate",
    )
    assert math.isfinite(m.b_ave_err) and m.b_ave_err > 0
    assert math.isfinite(m.b_rms_mean_err)
    assert math.isnan(m.b_pk_err)  # parabolic peak not linearly propagated


def test_no_errors_yields_nan_uncertainties():
    x = np.linspace(-60.0, 60.0, 1001)
    amp = _gaussian(x, 0.0, 8.0)
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="bootstrap")
    assert math.isnan(m.b_ave_err)
    assert math.isnan(m.beta_err)


# ── 7. Empty window degrades gracefully ─────────────────────────────────────


def test_empty_window_returns_zero_sample_nan_moments():
    x = np.linspace(0.0, 100.0, 1001)
    amp = _gaussian(x, 50.0, 5.0)
    m = spectrum_moments(x, amp, x_range=(200.0, 300.0), cutoff_fraction=0.0, uncertainty="none")
    assert m.n_sample == 0
    assert math.isnan(m.b_ave)
    assert m.ok is False


def test_all_zero_amplitude_returns_empty():
    x = np.linspace(0.0, 100.0, 1001)
    m = spectrum_moments(x, np.zeros_like(x), x_range=None, cutoff_fraction=0.0, uncertainty="none")
    assert m.n_sample == 0


# ── 8. WiMDA transcribed oracle ─────────────────────────────────────────────


def test_matches_wimda_oracle_on_shared_spectrum():
    # One shared, mildly-skewed spectrum in field units (Gauss).
    x = np.linspace(2900.0, 3100.0, 2001)
    amp = _gaussian(x, 3000.0, 9.0, 1.0) + _gaussian(x, 3018.0, 7.0, 0.3)
    cutoff = 0.03
    ours = spectrum_moments(x, amp, x_range=None, cutoff_fraction=cutoff, uncertainty="none")
    ref = wimda_oracle.wimda_moments(list(x), list(amp), cutoff_fraction=cutoff, x_range=None)
    assert ours.b_pk == pytest.approx(ref["b_pk"], abs=1e-7)
    assert ours.b_ave == pytest.approx(ref["b_ave"], abs=1e-9)
    assert ours.b_diff == pytest.approx(ref["b_diff"], abs=1e-7)
    assert ours.b_rms_mean == pytest.approx(ref["b_rms_mean"], abs=1e-9)
    assert ours.b_rms_peak == pytest.approx(ref["b_rms_peak"], abs=1e-7)
    assert ours.skewness == pytest.approx(ref["skewness"], rel=1e-7)
    assert ours.beta == pytest.approx(ref["beta"], rel=1e-6)
    assert ours.n_sample == ref["n_sample"]


def test_matches_wimda_oracle_with_range_and_cutoff():
    x = np.linspace(2900.0, 3100.0, 2001)
    amp = _gaussian(x, 3000.0, 9.0) + _gaussian(x, 3030.0, 6.0, 0.4)
    rng = (2960.0, 3060.0)
    ours = spectrum_moments(x, amp, x_range=rng, cutoff_fraction=0.05, uncertainty="none")
    ref = wimda_oracle.wimda_moments(list(x), list(amp), cutoff_fraction=0.05, x_range=rng)
    assert ours.b_ave == pytest.approx(ref["b_ave"], abs=1e-9)
    assert ours.b_rms_mean == pytest.approx(ref["b_rms_mean"], abs=1e-9)
    assert ours.skewness == pytest.approx(ref["skewness"], rel=1e-7)
    assert ours.n_sample == ref["n_sample"]


# ── 9. Unit invariance (alpha/beta) vs scaling (B_*) ────────────────────────


def test_alpha_beta_invariant_under_field_frequency_rescaling():
    # Same spectrum, axis in Gauss vs MHz: alpha/beta invariant, B_* scale by gamma.
    b_gauss = np.linspace(2900.0, 3100.0, 2001)
    amp = _gaussian(b_gauss, 3000.0, 9.0) + _gaussian(b_gauss, 3022.0, 7.0, 0.3)
    f_mhz = convert(b_gauss, FieldUnit.GAUSS, FieldUnit.MHZ)
    mg = spectrum_moments(b_gauss, amp, x_range=None, cutoff_fraction=0.02, uncertainty="none")
    mf = spectrum_moments(f_mhz, amp, x_range=None, cutoff_fraction=0.02, uncertainty="none")
    assert mf.skewness == pytest.approx(mg.skewness, rel=1e-6)
    assert mf.beta == pytest.approx(mg.beta, rel=1e-6)
    scale = float(f_mhz[1] - f_mhz[0]) / float(b_gauss[1] - b_gauss[0])
    assert mf.b_rms_mean == pytest.approx(mg.b_rms_mean * scale, rel=1e-6)


# ── 10. trend-row shape + input validation ──────────────────────────────────


def test_trend_row_has_fit_summary_shape():
    x = np.linspace(2900.0, 3100.0, 1001)
    amp = _gaussian(x, 3000.0, 9.0)
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    row = moments_trend_row(m, run_number=1234, run_label="R1234", field=3000.0, temperature=5.0)
    assert set(row) == {
        "success", "parameters", "uncertainties", "run_number", "run_label",
        "field", "temperature",
    }
    assert row["success"] is True
    assert "B_rms_mean" in row["parameters"] and "beta" in row["parameters"]
    assert row["field"] == 3000.0


def test_trend_row_non_finite_coord_becomes_none():
    x = np.linspace(2900.0, 3100.0, 501)
    amp = _gaussian(x, 3000.0, 9.0)
    m = spectrum_moments(x, amp, x_range=None, cutoff_fraction=0.0, uncertainty="none")
    row = moments_trend_row(m, run_number=1, field=float("nan"), temperature=5.0)
    assert row["field"] is None
    assert row["temperature"] == 5.0


@pytest.mark.parametrize("cutoff", [-0.1, 1.0, 1.5, float("nan")])
def test_invalid_cutoff_rejected(cutoff):
    x = np.linspace(0.0, 10.0, 11)
    with pytest.raises(ValueError):
        spectrum_moments(x, x, x_range=None, cutoff_fraction=cutoff)


def test_invalid_uncertainty_method_rejected():
    x = np.linspace(0.0, 10.0, 11)
    with pytest.raises(ValueError):
        spectrum_moments(x, x, x_range=None, cutoff_fraction=0.0, uncertainty="jackknife")


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        spectrum_moments(np.arange(10.0), np.arange(9.0), x_range=None, cutoff_fraction=0.0)


def test_frozen_dataclass_and_columns_aligned():
    assert SpectrumMoments.__dataclass_params__.frozen is True
