"""Tests for asymmetry calculation and transforms."""

import numpy as np
import pytest

from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
    estimate_alpha,
)
from asymmetry.core.transform.rebin import rebin


class TestComputeAsymmetry:
    def test_equal_counts(self):
        f = np.full(100, 1000.0)
        b = np.full(100, 1000.0)
        asym, err = compute_asymmetry(f, b, alpha=1.0)
        np.testing.assert_allclose(asym, 0.0)

    def test_known_asymmetry(self):
        f = np.array([150.0])
        b = np.array([100.0])
        asym, err = compute_asymmetry(f, b, alpha=1.0)
        expected = (150 - 100) / (150 + 100)
        assert asym[0] == pytest.approx(expected)

    def test_alpha_scaling(self):
        f = np.array([200.0])
        b = np.array([200.0])
        asym, _ = compute_asymmetry(f, b, alpha=0.5)
        # (200 - 0.5*200) / (200 + 0.5*200) = 100/300 = 1/3
        assert asym[0] == pytest.approx(1.0 / 3.0)

    def test_zero_denominator_uses_default_error(self):
        f = np.array([0.0])
        b = np.array([0.0])
        asym, err = compute_asymmetry(f, b, alpha=1.0)
        assert asym[0] == pytest.approx(0.0)
        assert err[0] == pytest.approx(1.0)

    def test_one_sided_counts_have_nonzero_error(self):
        f = np.array([1.0])
        b = np.array([0.0])
        asym, err = compute_asymmetry(f, b, alpha=1.0)
        assert asym[0] == pytest.approx(1.0)
        # Mantid-compatible formula gives sqrt(2) for this case.
        assert err[0] == pytest.approx(np.sqrt(2.0))

    def test_supplied_count_errors_use_musrfit_style_propagation(self):
        f = np.array([90.0])
        b = np.array([60.0])
        ef = np.array([np.sqrt(105.0)])
        eb = np.array([np.sqrt(90.0)])

        asym, err = compute_asymmetry_with_count_errors(f, b, ef, eb, alpha=1.0)

        assert asym[0] == pytest.approx(0.2)
        expected = 2.0 * np.sqrt((60.0 * ef[0]) ** 2 + (90.0 * eb[0]) ** 2) / (150.0**2)
        assert err[0] == pytest.approx(expected)


class TestEstimateAlpha:
    def test_estimate_alpha_full_range(self):
        forward = np.array([10.0, 20.0, 30.0])
        backward = np.array([5.0, 10.0, 15.0])
        alpha = estimate_alpha(forward, backward)
        assert alpha == pytest.approx(2.0)

    def test_estimate_alpha_good_bin_window(self):
        forward = np.array([100.0, 10.0, 10.0, 100.0])
        backward = np.array([50.0, 20.0, 20.0, 50.0])
        alpha = estimate_alpha(forward, backward, first_good_bin=1, last_good_bin=2)
        assert alpha == pytest.approx(0.5)


class TestRebin:
    def test_factor_2(self):
        t = np.arange(10, dtype=float)
        v = np.ones(10)
        e = np.full(10, 0.1)
        t2, v2, e2 = rebin(t, v, e, factor=2)
        assert len(t2) == 5

    def test_factor_1_is_noop(self):
        t = np.arange(5, dtype=float)
        v = np.ones(5)
        e = np.full(5, 0.1)
        t2, v2, e2 = rebin(t, v, e, factor=1)
        np.testing.assert_array_equal(t2, t)

    def test_invalid_factor(self):
        with pytest.raises(ValueError):
            rebin(np.zeros(5), np.zeros(5), np.zeros(5), factor=0)
