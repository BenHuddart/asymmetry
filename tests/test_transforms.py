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

    def test_one_sided_counts_use_default_error(self):
        f = np.array([1.0])
        b = np.array([0.0])
        asym, err = compute_asymmetry(f, b, alpha=1.0)
        assert asym[0] == pytest.approx(1.0)
        # One-sided bins (F*B == 0) are degenerate: A is pinned to +/-1 and the
        # exact first-order variance is zero, which is useless as a fit weight,
        # so they fall back to the 1.0 "no information" sentinel.
        assert err[0] == pytest.approx(1.0)

    def test_exact_poisson_error_alpha_one(self):
        # Exact propagation: var(A) = (1 - A^2)/(F + B) at alpha = 1.
        f = np.array([6000.0])
        b = np.array([4000.0])
        asym, err = compute_asymmetry(f, b, alpha=1.0)
        a = asym[0]
        assert err[0] ** 2 == pytest.approx((1.0 - a * a) / (f[0] + b[0]))

    def test_exact_error_general_alpha(self):
        # var(A) = 4 alpha^2 F B (F + B) / (F + alpha B)^4 for any alpha.
        f = np.array([5200.0])
        b = np.array([4100.0])
        alpha = 1.3
        _, err = compute_asymmetry(f, b, alpha=alpha)
        d = f[0] + alpha * b[0]
        expected = np.sqrt(4.0 * alpha**2 * f[0] * b[0] * (f[0] + b[0])) / d**2
        assert err[0] == pytest.approx(expected)

    def test_error_matches_count_error_path(self):
        # compute_asymmetry must agree with the explicit count-error form when
        # the count errors are the Poisson sqrt(N): both are exact propagation.
        f = np.array([90.0, 150.0, 5000.0])
        b = np.array([60.0, 150.0, 4800.0])
        alpha = 1.1
        _, err = compute_asymmetry(f, b, alpha=alpha)
        _, err_counts = compute_asymmetry_with_count_errors(
            f, b, np.sqrt(f), np.sqrt(b), alpha=alpha
        )
        np.testing.assert_allclose(err, err_counts)

    def test_negative_counts_do_not_produce_nan_error(self):
        # Out-of-contract negative (e.g. over-subtracted) counts must not yield
        # a NaN error from sqrt of a negative radicand; the clamp returns 0.0.
        f = np.array([-3.0])
        b = np.array([-2.0])
        _, err = compute_asymmetry(f, b, alpha=1.0)
        assert np.isfinite(err[0])
        assert err[0] == pytest.approx(0.0)

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
