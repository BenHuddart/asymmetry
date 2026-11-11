"""Independent causal TF count targets through production calibration entry points."""

import numpy as np
import pytest
from scipy.integrate import quad

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.count_domain import fit_fb_alpha
from asymmetry.core.fitting.models import oscillatory
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, fit_tf_calibration
from asymmetry.core.fitting.time_response import TimeResponse
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _point_counts(time, beta, *, separation=0.324, envelope="exponential"):
    forward = np.full_like(time, 7.0)
    backward = np.full_like(time, 11.0)
    for birth in (-separation / 2, separation / 2):
        live = time >= birth
        age = time[live] - birth
        weight = 0.5 * np.exp(-age / MUON_LIFETIME_US)
        damping = {
            "undamped": np.ones_like(age),
            "exponential": np.exp(-0.15 * age),
            "gaussian": np.exp(-((0.15 * age) ** 2)),
            "stretched": np.exp(-((0.15 * age) ** 1.4)),
        }[envelope]
        p = 0.2 * damping * np.cos(2 * np.pi * 1.3 * age + 0.17)
        forward[live] += 20000 * weight * (1 + p)
        backward[live] += 16000 * weight * (1 - beta * p)
    return np.array([forward, backward])


def _acquisition(beta, envelope="exponential"):
    time = np.arange(600) * 0.016
    run = Run(
        7104,
        [Histogram(np.rint(y), 0.016) for y in _point_counts(time, beta, envelope=envelope)],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 4,
            "last_good_bin": 498,
            "deadtime_correction": False,
            "background_correction": False,
        },
    )
    return MuonDataset(np.array([]), np.array([]), np.array([]), run=run)


@pytest.mark.parametrize("envelope", ["undamped", "exponential", "gaussian", "stretched"])
@pytest.mark.parametrize("fit_beta,beta", [(False, 1.0), (True, 0.8)])
def test_public_fixed_pulse_tf_calibration_recovers_independent_point_truth(
    fit_beta, beta, envelope
):
    dataset = _acquisition(beta, envelope)
    request = TFCalibrationRequest(
        1, 2, fit_beta=fit_beta, frequency_mhz=1.3, pulse_separation_us=0.324
    )
    result = fit_tf_calibration(dataset, request, envelope=envelope)
    assert result.ok, result.message
    assert result.alpha == pytest.approx(1.25, abs=1e-3)
    assert result.beta == pytest.approx(beta, abs=3e-3)


def test_shared_public_fb_path_supports_fixed_beta_one_with_supplied_pulse():
    parameters = ParameterSet(
        [
            Parameter("alpha", 1.2, min=0.1, max=5),
            Parameter("N0", 18000, min=0),
            Parameter("background", 7, min=0),
            Parameter("background_b", 11, min=0),
            Parameter("A0", 20, min=0, max=100),
            Parameter("frequency", 1.3, min=0),
            Parameter("phase", 0.17),
            Parameter("Lambda", 0.15, min=0),
            Parameter("baseline", 0, fixed=True),
            Parameter("beta", 1, fixed=True),
            Parameter("dpsep", 0.324, fixed=True),
        ]
    )
    fit = fit_fb_alpha(_acquisition(1), 1, 2, oscillatory, parameters, estimate_beta=True)
    assert fit.success, fit.message


@pytest.mark.parametrize("beta", [1.0, 0.8])
def test_normalized_existing_response_matches_independent_count_ratio(beta):
    time = np.array([-0.1, 0.0, 0.1, 0.162, 0.2, 0.7, 4.0])
    counts = _point_counts(time, beta)
    ratio, _ = compute_asymmetry(counts[0] - 7, counts[1] - 11, alpha=1.25, beta=beta)
    expected = TimeResponse("double_delta", 0.324).evaluate(
        lambda age: 0.2 * np.exp(-0.15 * age) * np.cos(2 * np.pi * 1.3 * age + 0.17), time
    )
    np.testing.assert_allclose(ratio, expected, atol=2e-15, rtol=2e-13)


def test_raw_counts_before_birth_are_backgrounds_without_normalized_population():
    counts = _point_counts(np.array([-0.3]), 0.8)
    np.testing.assert_array_equal(counts[:, 0], [7.0, 11.0])


def test_count_bin_average_and_centre_are_distinct_at_arrival():
    left, right = 0.1, 0.2
    integrated = quad(
        lambda t: _point_counts(np.array([t]), 0.8)[0, 0], left, right, points=[0.162], epsabs=1e-9
    )[0] / (right - left)
    midpoint = float(_point_counts(np.array([(left + right) / 2]), 0.8)[0, 0])
    assert abs(midpoint - integrated) / integrated > 0.1


@pytest.mark.parametrize("separation", [0.0, 0.324])
@pytest.mark.parametrize("beta", [1.0, 0.8])
def test_shared_count_prediction_live_ages_background_once_and_absolute_scale(separation, beta):
    from asymmetry.core.fitting.count_response import double_pulse_fb_model

    time = np.array([-0.3, -separation / 2, 0, separation / 2, 0.2, 3, 2000.0])

    def source(age):
        assert np.all(age >= 0)
        return 0.2 * np.exp(-0.15 * age) * np.cos(2 * np.pi * 1.3 * age + 0.17)

    model = double_pulse_fb_model(source, with_beta=True)
    for sign, background, expected in zip(
        (1, -1), (7, 11), _point_counts(time, beta, separation=separation), strict=True
    ):
        actual = model(
            time,
            alpha=1.25,
            N0=np.sqrt(20000 * 16000),
            background=background,
            sign=sign,
            beta=beta,
            dpsep=separation,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-12)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_supplied_separation_is_rejected_at_request(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        TFCalibrationRequest(1, 2, pulse_separation_us=value)


def test_full_nuisance_poisson_fisher_information():
    """Independent point-count Jacobian includes all nine free nuisances.

    Poisson information J.T diag(1/mu) J uses fractional P, percent A0,
    MHz frequency, radians phase and inverse-us Lambda. HESSE is conditional
    on the fixed supplied d and the selected exponential envelope.
    """
    from scipy.optimize._numdiff import approx_derivative

    request = TFCalibrationRequest(
        1, 2, fit_beta=True, frequency_mhz=1.3, pulse_separation_us=0.324
    )
    fit = fit_tf_calibration(_acquisition(0.8), request).fit
    params = {p.name: p.value for p in fit.parameters}
    order = fit.covariance_parameters
    time = np.asarray(fit.domains[0].time_us)

    def counts(values):
        p = params | dict(zip(order, values, strict=True))
        f, b = np.full_like(time, p["background"]), np.full_like(time, p["background_b"])
        for birth in (-0.162, 0.162):
            live = time >= birth
            age = time[live] - birth
            survival = 0.5 * np.exp(-age / MUON_LIFETIME_US)
            polar = (
                p["A0"]
                / 100
                * np.exp(-p["Lambda"] * age)
                * np.cos(2 * np.pi * p["frequency"] * age + p["phase"])
            )
            f[live] += p["N0"] * np.sqrt(p["alpha"]) * survival * (1 + polar)
            b[live] += p["N0"] / np.sqrt(p["alpha"]) * survival * (1 - p["beta"] * polar)
        return np.r_[f, b]

    x = np.array([params[n] for n in order])
    j = approx_derivative(counts, x, method="3-point")
    information = j.T @ (j / counts(x)[:, None])
    independent = np.linalg.inv(information)
    covariance = np.asarray(fit.covariance)
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)), np.sqrt(np.diag(independent)), rtol=0.015
    )
    scale = np.sqrt(np.diag(independent))
    np.testing.assert_allclose(
        covariance / np.outer(scale, scale), independent / np.outer(scale, scale), atol=0.02
    )
    assert "dpsep" not in order and "baseline" not in order


@pytest.mark.parametrize("width", [0.016, 0.004])
def test_post_arrival_bin_average_converges_to_centre(width):
    errors = []
    for w in (width, width / 2):
        centre = 0.5
        integrated = (
            quad(lambda t: _point_counts(np.array([t]), 0.8)[0, 0], centre - w / 2, centre + w / 2)[
                0
            ]
            / w
        )
        point = _point_counts(np.array([centre]), 0.8)[0, 0]
        errors.append(abs(integrated - point))
    assert errors[1] < errors[0] * 0.251


@pytest.mark.parametrize("fit_beta", [False, True])
def test_all_candidates_share_fixed_pulse_domain_and_selection(fit_beta):
    from asymmetry.core.fitting.tf_calibration import select_tf_calibration

    request = TFCalibrationRequest(
        1, 2, fit_beta=fit_beta, frequency_mhz=1.3, pulse_separation_us=0.324
    )
    selection = select_tf_calibration(_acquisition(0.8 if fit_beta else 1), request)
    assert len(selection.candidates) == 4
    assert selection.selected.fit.statistic == min(
        c.fit.statistic for c in selection.candidates if c.ok
    )
    domains = selection.candidates[0].fit.domains
    for c in selection.candidates:
        assert c.request.pulse_separation_us == 0.324
        assert c.fit.domains == domains
        assert "dpsep" not in c.fit.covariance_parameters
        assert c.beta_fixed == (not fit_beta)


@pytest.mark.parametrize("fit_beta", [False, True])
@pytest.mark.parametrize("width", [0.016, 0.004])
def test_integrated_postarrival_acquisition_calibration_limit(width, fit_beta):
    """Independent quadrature is distinct from the fitted centre observation.

    Wholly post-arrival smooth bins converge quadratically. N0 is in counts
    per bin here, so this control averages a rate in the same N0 units.
    """
    time = np.arange(int(8 / width)) * width
    nodes, weights = np.polynomial.legendre.leggauss(16)
    sample = (time[:, None] + width / 2 * nodes).ravel()
    beta = 0.8 if fit_beta else 1
    means = _point_counts(sample, beta).reshape(2, len(time), len(nodes)) @ (weights / 2)
    run = Run(
        7104,
        [Histogram(np.rint(y), width) for y in means],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": int(np.ceil(0.3 / width)),
            "last_good_bin": len(time) - 1,
            "deadtime_correction": False,
            "background_correction": False,
        },
    )
    data = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    result = fit_tf_calibration(
        data,
        TFCalibrationRequest(1, 2, fit_beta=fit_beta, frequency_mhz=1.3, pulse_separation_us=0.324),
    )
    assert result.ok, result.message
    assert result.alpha == pytest.approx(1.25, abs=1e-4)
    assert result.beta == pytest.approx(beta, abs=1e-4)


@pytest.mark.parametrize("fit_beta", [False, True])
@pytest.mark.parametrize("factor", [1, 4])
def test_fixed_pulse_corrected_noise_packed_domains_and_no_mutation(fit_beta, factor):
    from asymmetry.core.fitting.likelihood import Objective

    ds = _acquisition(0.8 if fit_beta else 1)
    for histogram in ds.run.histograms:
        histogram.counts = np.rint(
            histogram.counts / (1 + 0.02 * histogram.counts / (0.016 * 100000))
        )
        histogram.count_variance = 4 * histogram.counts
    ds.run.grouping.update(
        deadtime_correction=True,
        deadtime_mode="manual",
        dead_time_us=[0.02, 0.02],
        good_frames=100000,
        bunching_factor=factor,
    )
    originals = [h.counts.copy() for h in ds.run.histograms]
    result = fit_tf_calibration(
        ds,
        TFCalibrationRequest(
            1,
            2,
            fit_beta=fit_beta,
            frequency_mhz=1.3,
            pulse_separation_us=0.324,
            t_min=0.3,
            exclude=(2, 3),
        ),
    )
    assert result.ok, result.message
    assert result.fit.objective is Objective.GAUSSIAN
    assert result.alpha == pytest.approx(1.25, abs=1e-3)
    assert result.beta == pytest.approx(0.8 if fit_beta else 1, abs=2e-3)
    for domain, histogram, original in zip(
        result.fit.domains, ds.run.histograms, originals, strict=True
    ):
        assert domain.deadtime.status == "applied"
        assert all(w.last - w.first + 1 == factor for w in domain.raw_windows)
        assert all(not 2 <= t <= 3 for t in domain.time_us)
        np.testing.assert_array_equal(histogram.counts, original)


def test_zero_separation_matches_off_on_the_causal_domain():
    from dataclasses import replace

    from tests.core.test_tf_calibration_fit import _raw_tf

    request = TFCalibrationRequest(1, 2, fit_beta=True, frequency_mhz=1.3)
    data = _raw_tf(beta=0.8)
    off = fit_tf_calibration(data, request)
    zero = fit_tf_calibration(data, replace(request, pulse_separation_us=0))
    assert zero.alpha == pytest.approx(off.alpha, abs=1e-6)
    assert zero.beta == pytest.approx(off.beta, abs=1e-6)
    assert zero.fit.statistic == pytest.approx(off.fit.statistic, abs=1e-6)


def test_wrong_supplied_separation_changes_early_count_fit_without_estimating_it():
    from dataclasses import replace

    data = _acquisition(0.8)
    request = TFCalibrationRequest(
        1, 2, fit_beta=True, frequency_mhz=1.3, pulse_separation_us=0.324
    )
    matched = fit_tf_calibration(data, request)
    wrong = fit_tf_calibration(data, replace(request, pulse_separation_us=0.2))
    assert matched.ok
    assert wrong.fit.statistic > matched.fit.statistic + 100
    assert next(p for p in wrong.fit.parameters if p.name == "dpsep").value == 0.2
    assert "dpsep" not in wrong.fit.covariance_parameters


def test_packed_onset_observations_remain_sums_with_centre_model_prediction():
    """A retained packed bin straddles a birth; point and bin means differ.

    This pins the current observation approximation, without a universal
    onset cut or an assertion of integrated-rate calibration accuracy.
    """
    from asymmetry.core.fitting.count_response import double_pulse_fb_model
    from asymmetry.core.fitting.grouped_time_domain import build_count_groups

    ds = _acquisition(0.8)
    ds.run.grouping["bunching_factor"] = 8
    group = build_count_groups(ds, [1, 2], lifetime_corrected=False)[0]
    first, last = group.metadata["count_fit_raw_windows"][0]
    assert first * 0.016 < 0.162 < last * 0.016
    assert group.counts[0] == sum(ds.run.histograms[0].counts[first : last + 1])

    def polar(age):
        return 0.2 * np.exp(-0.15 * age) * np.cos(2 * np.pi * 1.3 * age + 0.17)

    model = double_pulse_fb_model(polar, with_beta=True)
    predicted = model(
        np.array([group.time[0]]),
        N0=8 * np.sqrt(20000 * 16000),
        alpha=1.25,
        background=8 * 7,
        sign=1,
        beta=0.8,
        dpsep=0.324,
    )[0]
    point = 8 * _point_counts(np.array([group.time[0]]), 0.8)[0, 0]
    assert predicted == pytest.approx(point, rel=1e-14)
    assert abs(predicted - group.counts[0]) / group.counts[0] > 0.1
