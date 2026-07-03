"""Basic tests for the longitudinal-field Kubo-Toyabe depolarization function.

Reference: Hayano et al., Phys. Rev. B 20, 850 (1979)
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.models import (
    MODELS,
    longitudinal_field_kubo_toyabe,
    static_gkt_zf,
)


class TestLFKuboToyabeBasic:
    """Core functionality tests for the LF-KT depolarization function."""

    def test_lf_kt_registered(self) -> None:
        """Verify the LF-KT model is registered in MODELS."""
        assert "LFKuboToyabe" in MODELS
        model = MODELS["LFKuboToyabe"]
        assert model.name == "LFKuboToyabe"
        assert set(model.param_names) == {"A0", "Delta", "B_L", "baseline"}

    def test_lf_kt_vectorized_array(self) -> None:
        """Verify LF-KT works with numpy arrays."""
        t = np.linspace(0, 10, 100)
        result = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=0.5, B_L=1000.0, baseline=0.0)

        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_lf_kt_scalar_input(self) -> None:
        """Verify LF-KT works with scalar input."""
        result = longitudinal_field_kubo_toyabe(1.0, A0=1.0, Delta=0.5, B_L=1000.0, baseline=0.0)

        assert np.isscalar(result) or result.ndim == 0
        assert np.isfinite(result)

    def test_lf_kt_zero_field_limit(self) -> None:
        """Verify LF-KT reduces to zero-field KT when B_L=0."""
        t = np.linspace(0, 5, 50)
        delta = 0.5

        # Zero-field KT result
        zf_result = static_gkt_zf(t, A0=1.0, Delta=delta, baseline=0.0)

        # LF-KT with B_L = 0
        lf_result = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=delta, B_L=0.0, baseline=0.0)

        # Should be very close
        np.testing.assert_allclose(lf_result, zf_result, rtol=1e-4, atol=1e-8)

    def test_lf_kt_initial_asymmetry(self) -> None:
        """Verify depolarization function starts at 1 (at t=0)."""
        t_zero = 0.0
        result = longitudinal_field_kubo_toyabe(t_zero, A0=1.0, Delta=0.5, B_L=1000.0, baseline=0.0)

        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_lf_kt_with_a0_scaling(self) -> None:
        """Verify A0 parameter scales the result."""
        t = np.linspace(0, 5, 50)
        a0_1 = 1.0
        a0_2 = 2.5

        result_1 = longitudinal_field_kubo_toyabe(t, A0=a0_1, Delta=0.5, B_L=1000.0, baseline=0.0)
        result_2 = longitudinal_field_kubo_toyabe(t, A0=a0_2, Delta=0.5, B_L=1000.0, baseline=0.0)

        # Result should scale linearly with A0
        np.testing.assert_allclose(result_2, result_1 * a0_2 / a0_1, rtol=1e-10)

    def test_lf_kt_with_baseline(self) -> None:
        """Verify baseline parameter is added correctly."""
        t = np.linspace(0, 5, 50)
        baseline_val = 0.5

        result_no_baseline = longitudinal_field_kubo_toyabe(
            t, A0=1.0, Delta=0.5, B_L=1000.0, baseline=0.0
        )
        result_with_baseline = longitudinal_field_kubo_toyabe(
            t, A0=1.0, Delta=0.5, B_L=1000.0, baseline=baseline_val
        )

        # Difference should be constant
        np.testing.assert_allclose(
            result_with_baseline - result_no_baseline, baseline_val, atol=1e-10
        )

    def test_lf_kt_field_decoupling(self) -> None:
        """Verify LF-KT shows field-induced decoupling effect."""
        t = np.linspace(0, 10, 100)
        delta = 0.5

        # Weak field
        result_weak = longitudinal_field_kubo_toyabe(
            t, A0=1.0, Delta=delta, B_L=100.0, baseline=0.0
        )

        # Strong field (decoupling regime)
        result_strong = longitudinal_field_kubo_toyabe(
            t, A0=1.0, Delta=delta, B_L=10000.0, baseline=0.0
        )

        # Strong field should show less depolarization (higher values) at large t
        assert np.mean(result_strong[50:]) > np.mean(result_weak[50:])

    def test_lf_kt_physically_reasonable_bounds(self) -> None:
        """Verify results are within physical bounds."""
        t = np.linspace(0, 100, 100)
        result = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=0.5, B_L=5000.0, baseline=0.0)

        assert np.all(np.isfinite(result))
        # Depolarization should not exceed initial amplitude
        assert np.all(result <= 1.0)
        # Should not go significantly negative
        assert np.all(result >= -0.5)

    def test_lf_kt_single_point(self) -> None:
        """Verify LF-KT works with single-point array."""
        t = np.array([1.0])
        result = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=0.5, B_L=1000.0, baseline=0.0)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_lf_kt_empty_array(self) -> None:
        """Verify LF-KT handles empty arrays."""
        t = np.array([])
        result = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=0.5, B_L=1000.0, baseline=0.0)
        assert result.shape == (0,)

    def test_lf_kt_available_in_model_registry(self) -> None:
        """Verify LF-KT can be accessed through the MODELS registry."""
        model_def = MODELS["LFKuboToyabe"]
        fn = model_def.function

        t = np.array([0.0, 1.0, 2.0])
        result = fn(t, A0=0.2, Delta=0.3, B_L=1000.0, baseline=0.01)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
