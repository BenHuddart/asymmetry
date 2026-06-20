"""Tests for WP4 — μ⁻SR optional polarisation multiplier.

Verification-plan §2 (Phase 4 acceptance):
- Synthesise a capture+precession histogram and recover the polarisation
  frequency within tolerance.
- With polarisation=None the model is bit-identical to the Phase-1 model
  (np.array_equal on a shared time axis).

All synthetic histograms use simulate_capture_run or direct numpy construction
as appropriate (the polarised histogram is built directly so that the exact
generating P_pol(t) is known with no Poisson noise, enabling a tight test of
model correctness separate from the fitting tolerance test).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.negmu.fit import (
    CaptureModelSpec,
    _default_polarisation_parameters,
    default_capture_parameters,
    fit_capture_histogram,
)
from asymmetry.core.negmu.model import build_capture_count_model
from asymmetry.core.negmu.polarisation import (
    POLARISATION_MODES,
    build_capture_count_model_with_polarisation,
    diamagnetic_polarisation,
    lorentzian_gaussian_polarisation,
)

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

N_BINS = 1024
BIN_WIDTH = 0.016  # µs
TIME = np.arange(N_BINS) * BIN_WIDTH  # 0 .. 16.368 µs

# Single-component spec (Carbon) for polarisation tests.
SPEC_C = CaptureModelSpec(elements=("C",), include_decay_background=False)


# ---------------------------------------------------------------------------
# Unit tests for the polarisation functions
# ---------------------------------------------------------------------------


class TestLorentzianGaussianPolarisation:
    def test_shape_matches_t(self):
        result = lorentzian_gaussian_polarisation(TIME, 0.2, 0.3, 0.5, 0.0)
        assert result.shape == TIME.shape

    def test_dtype_float64(self):
        result = lorentzian_gaussian_polarisation(TIME, 0.2, 0.3, 0.5, 0.0)
        assert result.dtype == np.float64

    def test_at_t0_equals_a0(self):
        """At t=0, exp(-λ·0)=1 and cos(phase)=1 when phase=0, so P(0)=a0."""
        a0 = 0.25
        result = lorentzian_gaussian_polarisation(np.array([0.0]), a0, 0.3, 0.5, 0.0)
        assert result[0] == pytest.approx(a0, rel=1e-12)

    def test_phase_pi_flips_sign_at_t0(self):
        """phase=π gives cos(π)=-1, so P(0) = -a0."""
        a0 = 0.2
        result = lorentzian_gaussian_polarisation(np.array([0.0]), a0, 0.0, 0.5, np.pi)
        assert result[0] == pytest.approx(-a0, rel=1e-12)

    def test_zero_lam_equals_diamagnetic(self):
        """lam=0 should equal diamagnetic_polarisation."""
        a0, freq, phase = 0.2, 0.5, 0.3
        lg = lorentzian_gaussian_polarisation(TIME, a0, 0.0, freq, phase)
        dm = diamagnetic_polarisation(TIME, a0, freq, phase)
        np.testing.assert_array_almost_equal(lg, dm, decimal=14)

    def test_large_lam_decays_to_zero(self):
        """Large λ rapidly damps the oscillation to near zero at late times."""
        result = lorentzian_gaussian_polarisation(TIME, 0.5, 100.0, 0.5, 0.0)
        assert abs(float(result[-1])) < 1e-10

    def test_frequency_scaling(self):
        """Doubling freq doubles the number of zero crossings in a fixed window."""
        a0, lam, phase = 0.3, 0.0, 0.0
        f1 = lorentzian_gaussian_polarisation(TIME, a0, lam, 0.25, phase)
        f2 = lorentzian_gaussian_polarisation(TIME, a0, lam, 0.50, phase)
        zc1 = int(np.sum(np.diff(np.sign(f1)) != 0))
        zc2 = int(np.sum(np.diff(np.sign(f2)) != 0))
        assert zc2 == pytest.approx(2 * zc1, abs=2)


class TestDiamagneticPolarisation:
    def test_shape_matches_t(self):
        result = diamagnetic_polarisation(TIME, 0.2, 0.5, 0.0)
        assert result.shape == TIME.shape

    def test_dtype_float64(self):
        result = diamagnetic_polarisation(TIME, 0.2, 0.5, 0.0)
        assert result.dtype == np.float64

    def test_at_t0_phase_zero(self):
        """At t=0, phase=0: P(0) = a0·cos(0) = a0."""
        a0 = 0.15
        result = diamagnetic_polarisation(np.array([0.0]), a0, 1.0, 0.0)
        assert result[0] == pytest.approx(a0, rel=1e-12)

    def test_constant_amplitude(self):
        """|P(t)| ≤ a0 everywhere (undamped oscillation)."""
        a0 = 0.25
        result = diamagnetic_polarisation(TIME, a0, 0.5, 0.3)
        assert np.all(np.abs(result) <= a0 + 1e-12)


# ---------------------------------------------------------------------------
# build_capture_count_model_with_polarisation
# ---------------------------------------------------------------------------


class TestBuildModelWithPolarisation:
    def test_unknown_polarisation_raises(self):
        comps = SPEC_C.components()
        with pytest.raises(ValueError, match="Unknown polarisation"):
            build_capture_count_model_with_polarisation(comps, "unknown_mode")

    def test_none_bit_identical_to_base_model(self):
        """polarisation=None is bit-identical to build_capture_count_model."""
        comps = SPEC_C.components()
        base = build_capture_count_model(comps)
        pol_none = build_capture_count_model_with_polarisation(comps, None)
        params = {"amp_C": 50000.0, "tau_C": 2.030, "background": 5.0}
        result_base = base(TIME, **params)
        result_pol = pol_none(TIME, **params)
        assert np.array_equal(result_base, result_pol), (
            "polarisation=None result differs from base model (not bit-identical)"
        )

    def test_lorgau_differs_from_base(self):
        """LorGau model with nonzero a0 should differ from the base model."""
        comps = SPEC_C.components()
        base = build_capture_count_model(comps)
        lg = build_capture_count_model_with_polarisation(comps, "lorgau")
        params_base = {"amp_C": 50000.0, "tau_C": 2.030, "background": 5.0}
        params_pol = {
            **params_base,
            "pol_a0": 0.3,
            "pol_lam": 0.2,
            "pol_freq": 0.5,
            "pol_phase": 0.0,
        }
        assert not np.array_equal(base(TIME, **params_base), lg(TIME, **params_pol))

    def test_lorgau_zero_a0_equals_base(self):
        """LorGau with pol_a0=0 → (1 + 0) = 1 → identical to base model."""
        comps = SPEC_C.components()
        base = build_capture_count_model(comps)
        lg = build_capture_count_model_with_polarisation(comps, "lorgau")
        params = {
            "amp_C": 50000.0,
            "tau_C": 2.030,
            "background": 5.0,
            "pol_a0": 0.0,
            "pol_lam": 0.5,
            "pol_freq": 1.0,
            "pol_phase": 0.0,
        }
        np.testing.assert_array_equal(base(TIME, **params), lg(TIME, **params))

    def test_diamagnetic_differs_from_base(self):
        """Diamagnetic model with nonzero a0 should differ from the base model."""
        comps = SPEC_C.components()
        base = build_capture_count_model(comps)
        dm = build_capture_count_model_with_polarisation(comps, "diamagnetic")
        params_base = {"amp_C": 50000.0, "tau_C": 2.030, "background": 5.0}
        params_pol = {**params_base, "pol_a0": 0.2, "pol_freq": 0.5, "pol_phase": 0.0}
        assert not np.array_equal(base(TIME, **params_base), dm(TIME, **params_pol))

    def test_background_not_modulated(self):
        """The flat background is NOT multiplied by (1 + P_pol(t)).

        With amp=0 (no exponential) and a nonzero background, the LorGau model
        should return just the flat background regardless of pol parameters.
        """
        comps = SPEC_C.components()
        lg = build_capture_count_model_with_polarisation(comps, "lorgau")
        bg = 10.0
        params = {
            "amp_C": 0.0,
            "tau_C": 2.030,
            "background": bg,
            "pol_a0": 0.5,
            "pol_lam": 0.1,
            "pol_freq": 1.0,
            "pol_phase": 0.0,
        }
        result = lg(TIME, **params)
        np.testing.assert_array_equal(result, np.full_like(TIME, bg))

    def test_lorgau_formula_matches_manual(self):
        """Model = exp_sum * (1 + a0*exp(-lam*t)*cos(2π*freq*t+phase)) + bg."""
        comps = SPEC_C.components()
        lg = build_capture_count_model_with_polarisation(comps, "lorgau")
        a0, lam, freq, phase = 0.3, 0.2, 0.5, 0.4
        amp_c, tau_c, bg = 40000.0, 2.030, 8.0
        params = {
            "amp_C": amp_c,
            "tau_C": tau_c,
            "background": bg,
            "pol_a0": a0,
            "pol_lam": lam,
            "pol_freq": freq,
            "pol_phase": phase,
        }
        result = lg(TIME, **params)
        exp_sum = amp_c * np.exp(-TIME / tau_c)
        p_t = lorentzian_gaussian_polarisation(TIME, a0, lam, freq, phase)
        expected = exp_sum * (1.0 + p_t) + bg
        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    def test_diamagnetic_formula_matches_manual(self):
        """Model = exp_sum * (1 + a0*cos(2π*freq*t+phase)) + bg."""
        comps = SPEC_C.components()
        dm = build_capture_count_model_with_polarisation(comps, "diamagnetic")
        a0, freq, phase = 0.2, 0.5, 0.3
        amp_c, tau_c, bg = 40000.0, 2.030, 8.0
        params = {
            "amp_C": amp_c,
            "tau_C": tau_c,
            "background": bg,
            "pol_a0": a0,
            "pol_freq": freq,
            "pol_phase": phase,
        }
        result = dm(TIME, **params)
        exp_sum = amp_c * np.exp(-TIME / tau_c)
        p_t = diamagnetic_polarisation(TIME, a0, freq, phase)
        expected = exp_sum * (1.0 + p_t) + bg
        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    @pytest.mark.parametrize("mode", POLARISATION_MODES)
    def test_output_dtype_float64(self, mode):
        comps = SPEC_C.components()
        fn = build_capture_count_model_with_polarisation(comps, mode)
        params = {
            "amp_C": 1000.0,
            "tau_C": 2.030,
            "background": 1.0,
            "pol_a0": 0.1,
            "pol_lam": 0.1,
            "pol_freq": 0.5,
            "pol_phase": 0.0,
        }
        assert fn(TIME, **params).dtype == np.float64


# ---------------------------------------------------------------------------
# Frequency recovery (fitting test)
# ---------------------------------------------------------------------------


def _make_lorgau_counts(
    amp_c: float,
    bg: float,
    a0: float,
    lam: float,
    freq: float,
    phase: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, counts) for a C-component capture with LorGau polarisation."""
    tau_c = 2.030
    exp_sum = amp_c * np.exp(-TIME / tau_c)
    p_t = lorentzian_gaussian_polarisation(TIME, a0, lam, freq, phase)
    expected = exp_sum * (1.0 + p_t) + bg
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.maximum(expected, 0.0)).astype(float)
    return TIME, counts


class TestPolarisedFitRecovery:
    """Fit a synthetic polarised histogram and check frequency recovery."""

    # Generating parameters
    AMP_C = 2.0e6
    BG = 5.0
    A0 = 0.25
    LAM = 0.15  # µs⁻¹
    FREQ = 0.5  # MHz  (period 2 µs, ~8 full periods in 16 µs window)
    PHASE = 0.0

    @pytest.fixture(scope="class")
    def lorgau_fit(self):
        t, cnts = _make_lorgau_counts(
            self.AMP_C, self.BG, self.A0, self.LAM, self.FREQ, self.PHASE, seed=7
        )
        # Build the full ParameterSet manually so we control pol seeds.
        params = default_capture_parameters(SPEC_C, time=t, counts=cnts)
        for p in _default_polarisation_parameters("lorgau", seeds={"pol_freq": 0.5, "pol_a0": 0.2}):
            params.add(p)
        return fit_capture_histogram(
            t, cnts, SPEC_C, cost="poisson", polarisation="lorgau", parameters=params
        )

    def test_fit_converges(self, lorgau_fit):
        assert lorgau_fit.success is True

    def test_frequency_recovered(self, lorgau_fit):
        """Recovered pol_freq within 10 % of truth (0.5 MHz)."""
        freq_fit = float(lorgau_fit.parameters["pol_freq"].value)
        assert abs(freq_fit - self.FREQ) / self.FREQ < 0.10, (
            f"pol_freq = {freq_fit:.4f} MHz; expected {self.FREQ} ± 10%"
        )

    def test_amplitude_positive(self, lorgau_fit):
        """Recovered amp_C must be positive."""
        amp_fit = float(lorgau_fit.parameters["amp_C"].value)
        assert amp_fit > 0.0, "amp_C must be positive"

    def test_none_polarisation_bit_identical_to_base(self):
        """fit_capture_histogram with polarisation=None gives bit-identical result to base."""
        comps = SPEC_C.components()
        base = build_capture_count_model(comps)
        pol_none = build_capture_count_model_with_polarisation(comps, None)
        params = {"amp_C": 45000.0, "tau_C": 2.030, "background": 6.0}
        assert np.array_equal(base(TIME, **params), pol_none(TIME, **params))


# ---------------------------------------------------------------------------
# Diamagnetic fit recovery
# ---------------------------------------------------------------------------


class TestDiamagneticFitRecovery:
    """Fit a synthetic undamped-polarisation histogram and recover frequency."""

    AMP_C = 2.0e6
    BG = 5.0
    A0 = 0.20
    FREQ = 0.4  # MHz — period 2.5 µs, ~6 full periods in 16 µs
    PHASE = 0.0

    @pytest.fixture(scope="class")
    def diamagnetic_fit(self):
        tau_c = 2.030
        exp_sum = self.AMP_C * np.exp(-TIME / tau_c)
        p_t = diamagnetic_polarisation(TIME, self.A0, self.FREQ, self.PHASE)
        expected = exp_sum * (1.0 + p_t) + self.BG
        rng = np.random.default_rng(13)
        counts = rng.poisson(np.maximum(expected, 0.0)).astype(float)

        params = default_capture_parameters(SPEC_C, time=TIME, counts=counts)
        for p in _default_polarisation_parameters(
            "diamagnetic", seeds={"pol_freq": 0.4, "pol_a0": 0.15}
        ):
            params.add(p)
        return fit_capture_histogram(
            TIME,
            counts,
            SPEC_C,
            cost="poisson",
            polarisation="diamagnetic",
            parameters=params,
        )

    def test_fit_converges(self, diamagnetic_fit):
        assert diamagnetic_fit.success is True

    def test_frequency_recovered(self, diamagnetic_fit):
        """Recovered pol_freq within 10 % of truth (0.4 MHz)."""
        freq_fit = float(diamagnetic_fit.parameters["pol_freq"].value)
        assert abs(freq_fit - self.FREQ) / self.FREQ < 0.10, (
            f"pol_freq = {freq_fit:.4f} MHz; expected {self.FREQ} ± 10%"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_bad_polarisation_mode_raises(self):
        t = TIME[:100]
        counts = np.ones(100)
        with pytest.raises(ValueError, match="Unknown polarisation"):
            fit_capture_histogram(t, counts, SPEC_C, polarisation="bad_mode")

    def test_valid_none_polarisation(self):
        """polarisation=None should not raise and should succeed."""
        t = TIME[:256]
        tau_c = 2.030
        counts = 50000.0 * np.exp(-t / tau_c) + 5.0
        result = fit_capture_histogram(t, counts, SPEC_C, polarisation=None)
        assert result is not None
