"""Vortex-lattice field-distribution lineshape (``sc.lineshape``).

The line's *width* is calibrated to the validated Brandt second moment, so these
tests anchor against :mod:`asymmetry.core.fitting.sc.models`; the *shape* (skew)
and the time-domain relaxation are checked independently, and a synthetic
round-trip confirms a window-stable penetration depth is recoverable.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.sc.lineshape import (
    _TWO_PI_GAMMA,
    _field_offsets_calibrated,
    vortex_lattice_component,
    vortex_lattice_powder_component,
    vortex_lattice_relaxation,
)
from asymmetry.core.fitting.sc.models import (
    brandt_field_width_sigma,
    brandt_field_width_sigma_powder,
)


def _rate_from_offsets(offsets: np.ndarray) -> float:
    """Gaussian rate (µs⁻¹) from the second moment of the field offsets."""
    return float(_TWO_PI_GAMMA * np.sqrt(np.mean(offsets**2)))


@pytest.mark.parametrize(
    ("lam", "B0", "Bc2"),
    [(195.0, 400.0, 25.0), (240.0, 400.0, 25.0), (150.0, 1600.0, 30.0)],
)
def test_powder_second_moment_matches_brandt(lam: float, B0: float, Bc2: float) -> None:
    """The calibrated line reproduces brandt_field_width_sigma_powder exactly."""
    offsets = _field_offsets_calibrated(lam, B0, Bc2, powder=True, n_g=10, n_grid=96)
    assert offsets is not None
    rate = _rate_from_offsets(offsets)
    brandt = float(brandt_field_width_sigma_powder(B0, lambda_ab=lam, Bc2=Bc2, sigma_bg=0.0))
    assert rate == pytest.approx(brandt, rel=1e-3)


def test_single_crystal_second_moment_matches_brandt() -> None:
    offsets = _field_offsets_calibrated(195.0, 400.0, 25.0, powder=False, n_g=10, n_grid=96)
    assert offsets is not None
    rate = _rate_from_offsets(offsets)
    brandt = float(brandt_field_width_sigma(400.0, 195.0, 25.0, 0.0, powder=False))
    assert rate == pytest.approx(brandt, rel=1e-3)


def test_uncalibrated_modified_london_width_is_near_brandt() -> None:
    """Independent of the Brandt calibration: the RAW modified-London second
    moment is within a few percent of the Brandt rate, so the field-distribution
    computation itself (not just the rescale) produces the right width."""
    from asymmetry.core.fitting.sc.lineshape import _centered_field_offsets
    from asymmetry.core.fitting.sc.models import _POWDER_LAMBDA_FACTOR
    from asymmetry.core.utils.constants import GAUSS_TO_TESLA

    lam_eff = 195.0 * _POWDER_LAMBDA_FACTOR
    raw = _centered_field_offsets(
        round(lam_eff, 3), round(400.0 * GAUSS_TO_TESLA, 9), round(25.0, 6), 10, 96
    )
    raw_rate = _rate_from_offsets(raw)
    brandt = float(brandt_field_width_sigma_powder(400.0, lambda_ab=195.0, Bc2=25.0, sigma_bg=0.0))
    assert raw_rate == pytest.approx(brandt, rel=0.05)


def test_powder_line_is_narrower_than_single_crystal() -> None:
    """Powder average (3^{1/4} lambda) gives a narrower line than single crystal."""
    sc = _field_offsets_calibrated(195.0, 400.0, 25.0, powder=False, n_g=10, n_grid=96)
    powder = _field_offsets_calibrated(195.0, 400.0, 25.0, powder=True, n_g=10, n_grid=96)
    assert _rate_from_offsets(powder) < _rate_from_offsets(sc)
    # specifically the single-crystal value divided by sqrt(3)
    assert _rate_from_offsets(powder) == pytest.approx(
        _rate_from_offsets(sc) / np.sqrt(3.0), rel=2e-3
    )


def test_lineshape_is_positively_skewed() -> None:
    """Flux-line lattice: sharp low-field cutoff, long high-field tail -> skew > 0."""
    offsets = _field_offsets_calibrated(195.0, 400.0, 25.0, powder=True, n_g=10, n_grid=96)
    assert offsets is not None
    skew = float(np.mean(offsets**3) / np.mean(offsets**2) ** 1.5)
    # Ideal triangular FLL skewness is ~2.7 at these defaults (literature ~3.3
    # for the pure London limit); bracket it so a wrong/truncated shape fails.
    assert 2.4 < skew < 3.4


def test_relaxation_starts_at_unity_and_decays() -> None:
    t = np.linspace(0.0, 8.0, 400)
    r = vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)
    assert r[0] == pytest.approx(1.0)
    assert abs(r[-1]) < 0.5  # depolarised by late time


def test_scalar_time_does_not_crash() -> None:
    """Scalar t must not raise (the relaxation returns a length-1 array)."""
    r = vortex_lattice_relaxation(2.0, 195.0, 400.0, 25.0)
    assert r.shape == (1,)
    y = vortex_lattice_powder_component(
        0.0, A=20.0, field=400.0, phase=0.0, lambda_ab=195.0, Bc2=25.0
    )
    assert float(np.real(y).ravel()[0]) == pytest.approx(20.0, rel=1e-6)


def test_no_lattice_above_bc2_is_undamped() -> None:
    """B0 >= Bc2 (or non-physical inputs) -> no vortex lattice, R(t) == 1."""
    t = np.linspace(0.0, 8.0, 200)
    assert np.allclose(vortex_lattice_relaxation(t, 195.0, 400.0, 25.0, powder=True)[0], 1.0)
    # 40 T applied field above a 25 T Bc2:
    assert np.allclose(vortex_lattice_relaxation(t, 195.0, 400_000.0, 25.0), 1.0)
    assert np.allclose(vortex_lattice_relaxation(t, -5.0, 400.0, 25.0), 1.0)  # lambda<=0


def test_lambda_scaling_of_the_rate() -> None:
    """At fixed (B0, Bc2) the rate scales as lambda^-2 (London limit)."""
    r1 = _rate_from_offsets(
        _field_offsets_calibrated(150.0, 400.0, 25.0, powder=True, n_g=10, n_grid=96)
    )
    r2 = _rate_from_offsets(
        _field_offsets_calibrated(300.0, 400.0, 25.0, powder=True, n_g=10, n_grid=96)
    )
    assert r1 / r2 == pytest.approx(4.0, rel=2e-2)


def test_component_shape_and_amplitude() -> None:
    t = np.linspace(0.0, 6.0, 256)
    y = vortex_lattice_powder_component(
        t, A=20.0, field=400.0, phase=0.0, lambda_ab=195.0, Bc2=25.0
    )
    assert y.shape == t.shape
    assert y[0] == pytest.approx(20.0, rel=1e-6)  # R(0)=1, cos(0)=1
    sc = vortex_lattice_component(t, A=20.0, field=400.0, phase=0.0, lambda_ab=195.0, Bc2=25.0)
    assert sc.shape == t.shape


def test_registered_in_composite_registry() -> None:
    from asymmetry.core.fitting.composite import COMPONENTS

    for name in ("VortexLattice", "VortexLatticePowder"):
        assert name in COMPONENTS
        assert COMPONENTS[name].param_names == ["A", "field", "phase", "lambda_ab", "Bc2"]


def test_synthetic_round_trip_recovers_lambda() -> None:
    """Generate a powder VL + nuclear + background signal with known lambda_ab and
    recover it through scipy least squares. The lineshape (unlike a Gaussian)
    fixes the form, so lambda is identifiable given a constrained nuclear rate."""
    from scipy.optimize import curve_fit

    from asymmetry.core.utils.constants import (
        GAUSS_TO_TESLA,
        MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    )

    t = np.linspace(0.0, 8.0, 500)
    true_lam, B0, Bc2 = 195.0, 400.0, 25.0
    freq = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * B0
    rng = np.random.default_rng(1)
    nuclear = np.exp(-0.5 * (0.20 * t) ** 2)
    signal = (
        vortex_lattice_powder_component(t, 18.0, B0, 0.3, true_lam, Bc2) * nuclear
        + 3.0 * np.cos(2 * np.pi * freq * t + 0.3)
        + 0.4
    )
    y = signal + rng.normal(0.0, 0.15, t.size)
    err = np.full_like(t, 0.15)

    def model(tt, amp, lam, phi, sig_n, a_bg, c):
        nu = np.exp(-0.5 * (sig_n * tt) ** 2)
        vl = vortex_lattice_powder_component(tt, amp, B0, phi, lam, Bc2) * nu
        return vl + a_bg * np.cos(2 * np.pi * freq * tt + phi) + c

    popt, _ = curve_fit(
        model,
        t,
        y,
        p0=[15.0, 230.0, 0.0, 0.25, 2.0, 0.0],
        sigma=err,
        bounds=([0, 120, -np.pi, 0, 0, -2], [50, 360, np.pi, 1, 20, 2]),
        maxfev=40000,
    )
    assert popt[1] == pytest.approx(true_lam, abs=10.0)
