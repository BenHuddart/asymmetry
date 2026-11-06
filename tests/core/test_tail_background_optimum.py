"""Independent likelihood, coordinate, and analytic checks of the tail estimator."""

import numpy as np
import pytest
from scipy.optimize import root
from scipy.special import xlogy

from asymmetry.core.transform.background import fit_tail_background
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def test_identical_observations_preserve_optimum_under_origin_change():
    width = 0.016
    indices = np.arange(2200)
    mean = width * (60000 * np.exp(-(indices - 10) * width / MUON_LIFETIME_US) + 8)
    counts = np.random.default_rng(90417).poisson(mean).astype(float)
    observed = counts[1100:]
    original = fit_tail_background(counts, bin_width_us=width, t0_bin=10, fit_start_bin=1100)
    anchored = fit_tail_background(observed, bin_width_us=width, t0_bin=0, fit_start_bin=0)
    assert original.ok and anchored.ok
    assert original.rate_per_us == pytest.approx(anchored.rate_per_us, rel=1e-10)
    np.testing.assert_allclose(original.expected_counts, anchored.expected_counts, rtol=1e-10)
    assert original.amplitude_per_us * np.exp(-1090 * width / MUON_LIFETIME_US) == pytest.approx(
        anchored.amplitude_per_us, rel=1e-10
    )
    # Independently solve the two unconstrained score equations in counts/bin.
    design = np.column_stack(
        (np.exp(-np.arange(observed.size) * width / MUON_LIFETIME_US), np.ones(observed.size))
    )
    solved = root(lambda p: design.T @ (1 - observed / (design @ p)), [0.25, 0.12], tol=1e-11)
    assert solved.success and np.all(solved.x > 0)
    expected = design @ solved.x
    assert original.rate_per_us == pytest.approx(solved.x[1] / width, rel=1e-9)

    def nll(mu):
        return np.sum(mu - xlogy(observed, mu))

    assert nll(original.expected_counts) == pytest.approx(nll(expected), abs=1e-10)
    assert original.expected_counts.sum() == pytest.approx(observed.sum(), rel=1e-12)


@pytest.mark.parametrize("width", [0.0002, 0.016, 0.2])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e6])
@pytest.mark.parametrize(
    "amplitude,background", [(120.0, 7.0), (0.0, 7.0), (120.0, 0.0), (1e-7, 7.0)]
)
def test_analytic_means_across_width_rate_and_boundary_scales(width, scale, amplitude, background):
    # Integrate exp(-t/tau) over bin edges independently of the sinh formula.
    centres = np.arange(200) * width
    integral = (
        MUON_LIFETIME_US
        * np.exp(-(centres - width / 2) / MUON_LIFETIME_US)
        * -np.expm1(-width / MUON_LIFETIME_US)
    )
    means = scale * (amplitude * integral + background * width)
    fitted = fit_tail_background(means, bin_width_us=width, t0_bin=0, fit_start_bin=0)
    assert fitted.ok, fitted.message
    # Weak contrasts amplify rounding; bound the flat error in rate units.
    assert fitted.rate_per_us == pytest.approx(scale * background, abs=scale * 2e-8)
    assert fitted.amplitude_per_us == pytest.approx(scale * amplitude, abs=scale * 2e-8)
    np.testing.assert_allclose(fitted.expected_counts, means, rtol=2e-12)
    # Saturated Poisson mean is an independent global lower bound on NLL.
    excess = np.sum(fitted.expected_counts - means + xlogy(means, means / fitted.expected_counts))
    assert abs(excess) < 1e-12 * means.sum()


@pytest.mark.parametrize(
    "observations",
    [
        np.arange(1, 201, dtype=float),
        np.r_[100.0, np.zeros(199)],
        np.random.default_rng(4152).poisson(0.3 + 4 * np.exp(-np.arange(200) * 0.03)),
    ],
)
def test_score_conditions_certify_constrained_optimum_including_zeros(observations):
    width = 0.07
    fitted = fit_tail_background(observations, bin_width_us=width, t0_bin=0, fit_start_bin=0)
    assert fitted.ok
    design = np.column_stack((np.exp(-np.arange(200) * width / MUON_LIFETIME_US), np.ones(200)))
    parameters = np.linalg.lstsq(design, fitted.expected_counts, rcond=None)[0]
    gradient = design.T @ (1 - observations / fitted.expected_counts)
    for parameter, derivative in zip(parameters, gradient):
        if parameter > 1e-10:
            assert abs(derivative) < 1e-10
        else:
            assert derivative >= -1e-10
    assert fitted.expected_counts.sum() == pytest.approx(observations.sum(), rel=1e-13)

    # Convexity makes the score conditions sufficient; also compare feasible points.
    def objective(mu):
        return np.sum(mu - xlogy(observations, mu))

    for candidate in (
        [0, observations.mean()],
        [observations.sum() / design[:, 0].sum(), 0],
        [1, 1],
        [20, 0.2],
    ):
        assert objective(fitted.expected_counts) <= objective(design @ candidate) + 1e-10


def test_supplied_variance_and_shared_covariance_match_independent_weighted_design():
    from asymmetry.core.transform.background import apply_grouped_background_correction
    from asymmetry.core.transform.fb_counts import GroupCounts, linear_combination_variance

    width = 0.04
    shape = np.exp(-np.arange(400) * width / MUON_LIFETIME_US)
    raw = 12 * shape + 0.7
    variance = np.linspace(0, 9, raw.size)
    raw_before, variance_before = raw.copy(), variance.copy()
    source = GroupCounts(raw, variance)
    fit = fit_tail_background(source, bin_width_us=width, t0_bin=-300, fit_start_bin=0)
    assert fit.ok
    # SVD pseudoinverse gives the expected-score influence without forming I^-1.
    design = np.column_stack((shape, np.ones(raw.size)))
    influence = np.linalg.pinv(design / np.sqrt(raw[:, None]))[1] / np.sqrt(raw)
    assert fit.rate_error_per_us == pytest.approx(
        np.linalg.norm(influence * np.sqrt(variance)) / width, rel=1e-12
    )
    np.testing.assert_allclose(fit.count_covariance, influence * variance / width, atol=1e-13)
    poisson = fit_tail_background(raw, bin_width_us=width, t0_bin=-300, fit_start_bin=0)
    assert poisson.ok
    assert poisson.rate_per_us == fit.rate_per_us
    assert poisson.rate_error_per_us == pytest.approx(
        np.linalg.norm(influence * np.sqrt(raw)) / width, rel=1e-12
    )
    # The default fit window is [200,399]; reconstruct that influence independently.
    result = apply_grouped_background_correction(
        source, source, grouping={"background_mode": "tail_fit"}, t0_bin=0, bin_width_us=width
    )
    assert result.outcome.status == "applied"
    start = 200
    local_design = design[start:]
    local_influence = np.linalg.pinv(local_design / np.sqrt(raw[start:, None]))[1] / np.sqrt(
        raw[start:]
    )
    weights = np.zeros(raw.size)
    weights[250:300] = 1
    coefficients = weights.copy()
    coefficients[start:] -= weights.sum() * local_influence
    actual = linear_combination_variance(result.forward, np.array([250, 300]))[0]
    assert actual == pytest.approx(np.sum(coefficients**2 * variance), rel=1e-12)
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(variance, variance_before)


def test_unusable_zero_supplied_uncertainty_stays_a_failure():
    from asymmetry.core.transform.fb_counts import GroupCounts

    fitted = fit_tail_background(
        GroupCounts(np.ones(100), np.zeros(100)), bin_width_us=0.1, t0_bin=0, fit_start_bin=0
    )
    assert not fitted.ok
    assert "variance" in fitted.message


@pytest.mark.parametrize("origin", [-100000, 100000])
def test_unrepresentable_positive_amplitude_is_not_reported_as_a_usable_fit(origin):
    counts = 12 * np.exp(-np.arange(100) * 0.1 / MUON_LIFETIME_US) + 0.7
    fitted = fit_tail_background(counts, bin_width_us=0.1, t0_bin=origin, fit_start_bin=0)
    assert not fitted.ok
    assert "amplitude at time zero is not representable" in fitted.message


@pytest.mark.parametrize("origin", [-1000, 1000])
def test_identical_bins_preserve_fitted_uncertainty_under_origin_change(origin):
    width = 0.04
    raw = (
        np.random.default_rng(1418)
        .poisson(12 * np.exp(-np.arange(400) * width / MUON_LIFETIME_US) + 0.7)
        .astype(float)
    )
    base = fit_tail_background(raw, bin_width_us=width, t0_bin=0, fit_start_bin=0)
    shifted = fit_tail_background(raw, bin_width_us=width, t0_bin=origin, fit_start_bin=0)
    assert base.ok and shifted.ok
    assert base.rate_per_us == shifted.rate_per_us
    assert base.rate_error_per_us == shifted.rate_error_per_us
    np.testing.assert_array_equal(base.expected_counts, shifted.expected_counts)
    np.testing.assert_array_equal(base.count_covariance, shifted.count_covariance)
    assert shifted.amplitude_per_us * np.exp(origin * width / MUON_LIFETIME_US) == pytest.approx(
        base.amplitude_per_us, rel=1e-13
    )


@pytest.mark.parametrize("width", [np.nan, np.inf, -np.inf, 0.0, -0.1])
def test_invalid_bin_width_returns_a_failed_fit_without_changing_observations(width):
    from asymmetry.core.transform.background import apply_grouped_background_correction

    counts = np.arange(1, 101, dtype=float)
    before = counts.copy()
    with np.errstate(all="raise"):
        fitted = fit_tail_background(counts, bin_width_us=width, t0_bin=0)
        correction = apply_grouped_background_correction(
            counts,
            counts,
            grouping={"background_mode": "tail_fit"},
            bin_width_us=width,
            t0_bin=0,
        )
    assert not fitted.ok
    assert fitted.message == "Bin width must be finite and positive"
    assert fitted.rate_error_per_us is None
    assert correction.outcome.status == "failed"
    assert "Bin width must be finite and positive" in correction.outcome.reason
    np.testing.assert_array_equal(counts, before)
    np.testing.assert_array_equal(correction.forward.counts, before)
    np.testing.assert_array_equal(correction.backward.counts, before)
