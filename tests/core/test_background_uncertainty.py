"""Independent covariance and sampling checks for estimated count backgrounds."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.integral import integrate_asymmetry
from asymmetry.core.transform.rebin import binned_fb_asymmetry
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _background_counts(*, scale=1.0, background_range=(0, 1), bunching=4):
    forward = scale * np.array([8.0, 8.0, 100.0, 100.0, 100.0, 100.0])
    backward = scale * np.array([8.0, 8.0, 80.0, 80.0, 80.0, 80.0])
    grouping = {
        "background_mode": "range",
        "background_range": list(background_range),
        "first_good_bin": 2,
        "last_good_bin": 5,
        "bunching_factor": bunching,
    }
    counts = corrected_grouped_counts(
        histograms=[Histogram(array, bin_width=0.01, t0_bin=2) for array in (forward, backward)],
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        facility="",
    )
    return counts, grouping


def test_shared_background_binned_covariance() -> None:
    counts, grouping = _background_counts()
    time, asymmetry, error = binned_fb_asymmetry(counts, grouping=grouping, alpha=1.0)
    # Four raw Poisson bins minus four copies of one independent mean level.
    # var(F) = 400 + 4²·4 = 464; var(B) = 320 + 4²·4 = 384 count².
    np.testing.assert_allclose(time, [0.015], rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(asymmetry, [0.12195121951219512], rtol=1e-12)
    np.testing.assert_allclose(error, [0.04420992805650498], rtol=1e-12)


def test_overlapping_background_integral_uses_the_raw_linear_coefficients() -> None:
    counts, _ = _background_counts(background_range=(1, 2))
    value, error = integrate_asymmetry(
        counts.forward,
        counts.backward,
        time=np.arange(6, dtype=float),
        t_min=2.0,
        t_max=5.0,
    )
    # The sum is -2r₁-r₂+r₃+r₄+r₅: the squared coefficients give 432 and 352.
    expected = 2 * np.sqrt(144**2 * 432 + 184**2 * 352) / 328**2
    assert value == pytest.approx(40 / 328, rel=1e-12)
    assert error == pytest.approx(expected, rel=1e-12)


def test_shared_background_seeded_simulation() -> None:
    counts, grouping = _background_counts(scale=100.0)
    _, _, error = binned_fb_asymmetry(counts, grouping=grouping, alpha=1.0)
    assert error[0] == pytest.approx(0.004420992805650498, rel=1e-12)
    rng = np.random.default_rng(20261031)
    forward = rng.poisson([800, 800, 10000, 10000, 10000, 10000], size=(200_000, 6))
    backward = rng.poisson([800, 800, 8000, 8000, 8000, 8000], size=(200_000, 6))
    f_sum = forward[:, 2:].sum(axis=1) - 4 * forward[:, :2].mean(axis=1)
    b_sum = backward[:, 2:].sum(axis=1) - 4 * backward[:, :2].mean(axis=1)
    empirical = np.std((f_sum - b_sum) / (f_sum + b_sum), ddof=1)
    assert empirical == pytest.approx(error[0], rel=0.01)


@pytest.mark.parametrize("weights", [None, np.array([0.0, 1.0, -0.5, 2.0, 0.25, 1.0])])
@pytest.mark.parametrize("background_window", [(0, 1), (1, 2), (2, 5)])
def test_linear_variance_matches_independent_raw_coefficients(weights, background_window) -> None:
    from asymmetry.core.transform.fb_counts import (
        GroupCounts,
        SharedComponent,
        linear_combination_variance,
    )

    raw = np.array([8.0, 8.0, 100.0, 100.0, 100.0, 100.0])
    lo, hi = background_window
    width = hi - lo + 1
    coefficients = np.zeros(raw.size)
    coefficients[lo : hi + 1] = 1.0 / width
    level = coefficients @ raw
    group = GroupCounts(
        raw - level,
        raw,
        shared=(
            SharedComponent(
                "background",
                float((coefficients**2) @ raw),
                -np.ones(raw.size),
                coefficients * raw,
            ),
        ),
    )
    edges = np.array([0, 2, 3, 6])
    observed = linear_combination_variance(group, edges, weights)
    expected = []
    a = np.ones(raw.size) if weights is None else weights
    for first, last in zip(edges[:-1], edges[1:]):
        selection = np.zeros(raw.size)
        selection[first:last] = a[first:last]
        raw_coefficients = selection - selection.sum() * coefficients
        expected.append((raw_coefficients**2) @ raw)
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


def test_a_background_window_subtracted_from_itself_has_zero_sum_variance() -> None:
    from asymmetry.core.transform.fb_counts import linear_combination_variance

    counts, _ = _background_counts(background_range=(2, 5))
    assert np.sum(counts.forward.counts[2:6]) == 0.0
    assert linear_combination_variance(counts.forward, np.array([2, 6]))[0] == pytest.approx(
        0, abs=1e-12
    )


def test_sparse_tail_uses_one_poisson_basis_for_variance_and_influence() -> None:
    from asymmetry.core.transform.background import fit_tail_background
    from asymmetry.core.transform.fb_counts import (
        GroupCounts,
        SharedComponent,
        linear_combination_variance,
    )

    raw = np.random.default_rng(20261031).poisson(2.0, 300).astype(float)
    width = 0.05
    fitted = fit_tail_background(raw, bin_width_us=width, t0_bin=0, fit_start_bin=0)
    assert fitted.ok
    sigma_squared = (width * fitted.rate_error_per_us) ** 2
    covariance = width * fitted.count_covariance
    # A model-based influence row cannot use the observed zero-bin variances.
    assert np.count_nonzero(raw + sigma_squared - 2 * covariance < 0) == 31
    group = GroupCounts(
        raw - width * fitted.rate_per_us,
        fitted.expected_counts,
        shared=(SharedComponent("background", sigma_squared, -np.ones(raw.size), covariance),),
    )
    matrix = (
        np.diag(fitted.expected_counts) + sigma_squared - covariance[:, None] - covariance[None, :]
    )
    assert np.linalg.eigvalsh(matrix)[0] >= -1e-12
    marginal = linear_combination_variance(group, np.arange(raw.size + 1))
    np.testing.assert_allclose(marginal, np.diag(matrix), rtol=1e-13)
    assert np.all(marginal > 0)


def test_a_singular_tail_fit_fails_instead_of_declaring_zero_uncertainty() -> None:
    from asymmetry.core.transform.background import fit_tail_background

    fitted = fit_tail_background(np.full(10, 2.0), bin_width_us=1e-12, t0_bin=0)
    assert not fitted.ok
    assert "singular" in fitted.message
    assert fitted.rate_error_per_us is None


def test_zero_variance_is_not_replaced_in_the_count_domain() -> None:
    from asymmetry.core.transform.asymmetry import compute_asymmetry_with_count_errors
    from asymmetry.core.transform.background import subtract_scaled_counts

    corrected, variance = subtract_scaled_counts(np.zeros(3), np.zeros(3), 2.0)
    np.testing.assert_array_equal(corrected, 0)
    np.testing.assert_array_equal(variance, 0)
    _, error = compute_asymmetry_with_count_errors(
        np.array([10.0]), np.zeros(1), np.sqrt([10.0]), np.zeros(1)
    )
    np.testing.assert_array_equal(error, 1)


def test_integral_ratio_and_one_binned_point_share_the_same_variances() -> None:
    from asymmetry.core.transform.asymmetry import estimate_alpha_detailed

    counts, grouping = _background_counts()
    _, _, binned = binned_fb_asymmetry(counts, grouping=grouping, alpha=1.0)
    _, integral = integrate_asymmetry(
        counts.forward, counts.backward, time=np.arange(6), t_min=2, t_max=5
    )
    estimate = estimate_alpha_detailed(
        counts.forward, counts.backward, method="ratio", first_good_bin=2, last_good_bin=5
    )
    assert integral == pytest.approx(binned[0], rel=1e-12)
    assert estimate.alpha_error == pytest.approx(0.1146865571959643, rel=1e-12)


def test_variable_width_bins_match_direct_raw_count_coefficients() -> None:
    from asymmetry.core.transform.rebin import binning_slice_edges

    forward, backward = np.full(80, 100.0), np.full(80, 80.0)
    forward[:10], backward[:10] = np.arange(2, 12), np.arange(3, 13)
    grouping = {
        "background_mode": "range",
        "background_range": [0, 9],
        "first_good_bin": 8,
        "last_good_bin": 79,
        "binning_mode": "variable",
        "bin0_us": 0.1,
        "bin10_us": 0.4,
    }
    counts = corrected_grouped_counts(
        histograms=[Histogram(raw, bin_width=0.05) for raw in (forward, backward)],
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        facility="",
    )
    _, asymmetry, error = binned_fb_asymmetry(counts, grouping=grouping, alpha=1.0)
    edges = (
        binning_slice_edges(
            72, mode="variable", bin_width_us=0.05, t_start_us=0.4, bin0_us=0.1, bin10_us=0.4
        )
        + 8
    )
    expected_values, expected_errors = [], []
    for first, last in zip(edges[:-1], edges[1:]):
        coefficients = np.zeros(80)
        coefficients[first:last] = 1
        coefficients[:10] -= (last - first) / 10
        f, b = coefficients @ forward, coefficients @ backward
        vf, vb = (coefficients**2) @ forward, (coefficients**2) @ backward
        expected_values.append((f - b) / (f + b))
        expected_errors.append(2 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2)
    np.testing.assert_allclose(asymmetry, expected_values, rtol=1e-12)
    np.testing.assert_allclose(error, expected_errors, rtol=1e-12)


@pytest.mark.slow
def test_tail_fit_cross_covariance_matches_seeded_refits() -> None:
    from asymmetry.core.transform.background import apply_grouped_background_correction
    from asymmetry.core.transform.fb_counts import linear_combination_variance

    rng = np.random.default_rng(20261031)
    values, variances = [], []
    edges = np.array([150, 180, 270, 300])
    # Interior rate and decay amplitude satisfy the Fisher linearisation's
    # regularity conditions; the sparse zero-amplitude limit is checked above.
    x = 0.05 / MUON_LIFETIME_US
    mean = 2.0 + 250.0 * np.exp(-np.arange(300) * x) * np.sinh(x / 2) / (x / 2)
    for _ in range(3000):
        raw = rng.poisson(mean).astype(float)
        result = apply_grouped_background_correction(
            raw,
            raw,
            grouping={"background_mode": "tail_fit"},
            t0_bin=0,
            bin_width_us=0.05,
            last_good_bin=299,
        )
        assert result.outcome.status == "applied"
        values.append([result.forward.counts[150:180].sum(), result.forward.counts[270:300].sum()])
        variances.append(linear_combination_variance(result.forward, edges)[[0, 2]])
    empirical = np.var(values, axis=0, ddof=1)
    predicted = np.mean(variances, axis=0)
    np.testing.assert_allclose(predicted, empirical, rtol=0.12)


@pytest.mark.parametrize("variance,covariance", [(0.0, [1.0]), (1.0, [2.0])])
def test_shared_components_require_a_valid_joint_covariance(variance, covariance) -> None:
    from asymmetry.core.transform.fb_counts import GroupCounts, SharedComponent

    with pytest.raises(ValueError, match="covariance"):
        GroupCounts(
            [1.0], [1.0], shared=(SharedComponent("background", variance, [-1.0], covariance),)
        )


def test_mutually_independent_components_must_be_jointly_consistent() -> None:
    from asymmetry.core.transform.fb_counts import GroupCounts, SharedComponent

    with pytest.raises(ValueError, match="covariance"):
        GroupCounts(
            [1.0],
            [1.0],
            shared=tuple(SharedComponent(name, 1.0, [-1.0], [0.8]) for name in ("first", "second")),
        )
