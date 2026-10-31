"""Phase estimation against direct real sampled signals."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fourier.phase_fit import (
    PhaseAcquisition,
    PhaseEstimate,
    PhaseFitCancelled,
    PhaseFitConfig,
    PhaseRejection,
    fit_fft_phase,
)


def _observation(*, phase=0.73, scale=1.0, window="exponential"):
    n, dt, start = 512, 0.01, 0.037
    time = start + np.arange(n) * dt
    signal = sum(a * np.cos(2 * np.pi * f * time + phase) for a, f in [(1.0, 8.13), (0.35, 8.91)])
    weights = np.exp(-(time - start) / 2.0) if window == "exponential" else np.ones(n)
    error = np.full(n, 0.005) * scale
    mean_weights = 1 / error
    signal = scale * signal
    mean = np.average(signal, weights=mean_weights)
    return PhaseAcquisition(
        start, dt, weights, error, mean_weights, unwindowed_signal=signal - mean
    )


def test_phase_matches_direct_sampled_physical_cosine():
    a = _observation()
    result = fit_fft_phase(a, PhaseFitConfig(6.0, 11.0))
    assert isinstance(result, PhaseEstimate)
    assert abs(np.angle(np.exp(1j * (result.phase_radians - 0.73)))) < 0.025
    assert 0 <= result.standard_error_radians < 0.02
    assert result.to_dict()["method"] == "sampled_common_phase_v1"


def test_phase_and_uncertainty_ignore_units():
    results = []
    for scale in (1.0, 100.0, 1e-7, 1e180, 1e-180):
        a = _observation(scale=scale)
        results.append(fit_fft_phase(a, PhaseFitConfig(6.0, 11.0)))
    assert all(isinstance(r, PhaseEstimate) for r in results)
    assert max(r.phase_radians for r in results) - min(r.phase_radians for r in results) < 1e-5
    assert (
        max(r.standard_error_radians for r in results)
        - min(r.standard_error_radians for r in results)
        < 1e-5
    )


def test_noise_has_no_accepted_phase():
    a = _observation()
    rng = np.random.default_rng(172)
    y = rng.normal(0, 0.005, len(a.window_weights))
    a = PhaseAcquisition(
        a.time_start_us,
        a.sample_interval_us,
        a.window_weights,
        a.sample_errors,
        a.mean_weights,
        unwindowed_signal=y,
    )
    result = fit_fft_phase(a, PhaseFitConfig(6.0, 11.0))
    assert isinstance(result, PhaseRejection)
    assert result.to_dict()["status"] != "accepted"


def test_phase_fit_can_be_cancelled_before_work():
    a = _observation()
    with pytest.raises(PhaseFitCancelled):
        fit_fft_phase(a, PhaseFitConfig(6.0, 11.0), cancel_callback=lambda: True)


@pytest.mark.parametrize("bounds", [(4.0, 2.0), (-1.0, 3.0)])
def test_invalid_user_band_rejected_at_boundary(bounds):
    with pytest.raises(ValueError):
        PhaseFitConfig(*bounds)


def _sampled(*, window="none", phase=-1.24, noise=0.01, seed=9, components=None):
    n, dt, start = 256, 0.02, 0.083
    t = start + np.arange(n) * dt
    windows = {
        "none": np.ones(n),
        "exponential": np.exp(-np.arange(n) / n),
        "gaussian": np.exp(-0.5 * (np.arange(n) / (0.6 * n)) ** 2),
        "hann": np.hanning(n),
    }
    w = windows[window]
    if components is None:
        components = [(1.0, 30 / (n * dt), phase), (0.4, 34 / (n * dt), phase)]
    y = sum(
        amplitude * np.cos(2 * np.pi * frequency * t + p) for amplitude, frequency, p in components
    )
    errors = np.full(n, noise)
    y += np.random.default_rng(seed).normal(size=n) * errors
    u = np.linspace(1, 2, n)
    y -= np.average(y, weights=u)
    return PhaseAcquisition(start, dt, w, errors, u, unwindowed_signal=y)


@pytest.mark.parametrize("window", ["none", "exponential", "gaussian", "hann"])
def test_actual_window_and_nonzero_origin(window):
    a = _sampled(window=window, noise=0.002)
    result = fit_fft_phase(a, PhaseFitConfig(4.0, 9.0))
    assert isinstance(result, PhaseEstimate), result
    assert abs(np.angle(np.exp(1j * (result.phase_radians + 1.24)))) < 0.025


@pytest.mark.parametrize(
    "components",
    [
        [(1.0, 0.0, 0.0)],
        [(1.0, 5.859375, 0.3), (-0.7, 6.640625, 0.3)],
        [(1.0, 5.859375, 0.3), (0.7, 6.640625, 1.8)],
    ],
)
def test_unsuitable_components_have_no_phase(components):
    a = _sampled(components=components)
    report = fit_fft_phase(a, PhaseFitConfig(4.0, 9.0))
    assert isinstance(report, PhaseRejection), report
    assert "phase_radians" not in report.to_dict()


def test_monotone_decay_has_no_common_oscillatory_phase():
    a = _sampled()
    y = np.exp(-np.arange(len(a.window_weights)) / 40)
    y -= np.average(y, weights=a.mean_weights)
    a = PhaseAcquisition(
        a.time_start_us,
        a.sample_interval_us,
        a.window_weights,
        a.sample_errors,
        a.mean_weights,
        unwindowed_signal=y,
    )
    report = fit_fft_phase(a)
    assert isinstance(report, PhaseRejection), report


def test_exact_sampled_response_includes_both_sidebands_and_mean():
    from asymmetry.core.fourier.phase_fit import _ResponseTransforms, _sampled_response

    a = _sampled(window="hann")
    transforms = _ResponseTransforms(a.window_weights, a.mean_weights)
    plus, minus, centers, bins = _sampled_response(a, transforms, (4.0, 9.0), 3)
    t = a.time_start_us + np.arange(len(a.window_weights)) * a.sample_interval_us
    for j in (0, len(centers) // 2, len(centers) - 1):
        phase = 2.17
        y = np.cos(2 * np.pi * centers[j] * t + phase)
        y -= np.sum(a.mean_weights * y)
        direct = np.fft.rfft(a.window_weights * y)[bins]
        assert np.allclose(
            np.exp(1j * phase) * plus[:, j] + np.exp(-1j * phase) * minus[:, j],
            direct,
            rtol=1e-11,
            atol=1e-11,
        )


def test_exact_conditional_noise_operator_matches_direct_sample_perturbations():
    from asymmetry.core.fourier.phase_fit import (
        _conditional_standard_error,
        _phase_search,
        _ResponseTransforms,
    )

    a = _sampled(window="hann")
    z = np.fft.rfft(a.window_weights * a.unwindowed_signal)
    normalization = float(max(abs(z)))
    transforms = _ResponseTransforms(a.window_weights, a.mean_weights)
    solution = _phase_search(z / normalization, a, transforms, (4.0, 9.0), 3, None)
    active = solution.amplitudes > max(solution.amplitudes.max() * 1e-7, 1e-15)
    columns = solution.columns[:, active]
    derivative = (
        solution.derivative
        - columns @ np.linalg.lstsq(columns, solution.derivative, rcond=1e-10)[0]
    )
    tnoise = np.diag(a.sample_errors)
    tnoise -= np.ones((len(a.window_weights), 1)) @ (a.mean_weights @ tnoise)[None, :]
    response = (
        np.fft.rfft(a.window_weights[:, None] * tnoise, axis=0)[solution.bins] / normalization
    )
    jacobian = np.concatenate((response.real, response.imag), axis=0)
    expected = np.linalg.norm(derivative @ jacobian) / (derivative @ derivative)
    assert _conditional_standard_error(solution, a, normalization) == pytest.approx(
        expected, rel=1e-10
    )


def test_conditional_uncertainty_matches_repeated_independent_gaussian_samples():
    phases = []
    errors = []
    for seed in range(24):
        a = _sampled(noise=0.04, seed=seed, components=[(1.0, 5.859375, -1.24)])
        report = fit_fft_phase(a, PhaseFitConfig(4.0, 8.0))
        assert isinstance(report, PhaseEstimate), report
        phases.append(report.phase_radians)
        errors.append(report.standard_error_radians)
    assert 0.65 < np.std(phases, ddof=1) / np.mean(errors) < 1.5


def test_band_support_and_work_limits_are_explicit():
    a = _sampled()
    assert fit_fft_phase(a, PhaseFitConfig(5.7, 6.0)).reason == "insufficient_band"
    assert fit_fft_phase(a, PhaseFitConfig(0.0, 25.0)).reason == "band_too_wide"
    assert fit_fft_phase(a, PhaseFitConfig(4.0, 26.0)).reason == "invalid_band"


def test_cancellation_is_checked_during_phase_trials():
    a = _sampled()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 8

    with pytest.raises(PhaseFitCancelled):
        fit_fft_phase(a, PhaseFitConfig(4.0, 9.0), cancel_callback=cancelled)
    assert calls == 8


def test_asymmetric_positive_broad_distribution():
    centers = np.linspace(4.8, 7.2, 603)
    amplitudes = np.exp(-0.5 * ((centers - 5.7) / 0.16) ** 2) + 0.35 * np.exp(
        -0.5 * ((centers - 6.15) / 0.23) ** 2
    )
    amplitudes /= amplitudes.sum()
    components = list(zip(amplitudes, centers, np.full(len(centers), 0.57), strict=True))
    a = _sampled(noise=0.002, components=components)
    result = fit_fft_phase(a, PhaseFitConfig(4.0, 8.0))
    assert isinstance(result, PhaseEstimate), result
    assert abs(result.phase_radians - 0.57) < 0.035


def test_hann_with_weak_phase_information_rejects_or_reports_large_uncertainty():
    a = _sampled(window="hann", noise=0.04)
    result = fit_fft_phase(a, PhaseFitConfig(4.0, 9.0, tolerance_degrees=1.0))
    assert isinstance(result, PhaseRejection), result


def test_poisson_sample_simulation_checks_conditional_uncertainty():
    n, dt, start = 256, 0.02, 0.083
    time = start + np.arange(n) * dt
    expectation = 40000 * (1 + 0.1 * np.cos(2 * np.pi * 5.859375 * time - 0.62))
    errors = np.sqrt(expectation) / 40000
    phases = []
    standard_errors = []
    for seed in range(24):
        y = (np.random.default_rng(seed + 700).poisson(expectation) - 40000) / 40000
        y -= np.mean(y)
        a = PhaseAcquisition(start, dt, np.ones(n), errors, np.ones(n), unwindowed_signal=y)
        report = fit_fft_phase(a, PhaseFitConfig(4.0, 8.0))
        assert isinstance(report, PhaseEstimate), report
        phases.append(report.phase_radians)
        standard_errors.append(report.standard_error_radians)
    assert 0.6 < np.std(phases, ddof=1) / np.mean(standard_errors) < 1.6


def test_grid_sensitivity_is_a_real_rejection_gate():
    a = _observation()
    report = fit_fft_phase(a, PhaseFitConfig(6.0, 11.0, tolerance_degrees=0.5))
    assert isinstance(report, PhaseRejection), report
    assert report.reason == "unstable_phase"


def test_edge_mass_requires_a_wider_band():
    a = _sampled(noise=0.002, components=[(1.0, 5.859375, -1.24)])
    report = fit_fft_phase(a, PhaseFitConfig(5.7, 9.0))
    assert isinstance(report, PhaseRejection), report
    assert report.reason == "edge_supported"


def test_unsupported_neighbor_in_observation_guard_rejects():
    a = _sampled(noise=0.002, components=[(1.0, 5.859375, -1.24), (0.7, 7.421875, -1.24)])
    report = fit_fft_phase(a, PhaseFitConfig(4.0, 7.0))
    assert isinstance(report, PhaseRejection), report
    assert report.reason in {"incompatible_or_resolution_limited", "edge_supported"}
    assert report.observation_band_mhz[1] > 7.421875


def test_observation_guards_are_explicit_provenance():
    a = _sampled(noise=0.002)
    report = fit_fft_phase(a, PhaseFitConfig(4.0, 9.0))
    assert isinstance(report, PhaseEstimate), report
    provenance = report.to_dict()
    assert provenance["band_mhz"] == [4.0, 9.0]
    assert provenance["observation_band_mhz"][0] < 4.0
    assert provenance["observation_band_mhz"][1] > 9.0


def test_long_acquisition_uses_bounded_frequency_dictionary():
    n, dt, start = 16384, 0.001, 0.037
    t = start + np.arange(n) * dt
    frequency = 200 / (n * dt)
    signal = np.cos(2 * np.pi * frequency * t + 2.13)
    window = np.exp(-np.arange(n) / n)
    acquisition = PhaseAcquisition(
        start, dt, window, np.full(n, 0.01), None, unwindowed_signal=signal
    )
    result = fit_fft_phase(
        acquisition,
        PhaseFitConfig(frequency - 8 / (n * dt), frequency + 8 / (n * dt)),
    )
    assert isinstance(result, PhaseEstimate), result
    assert abs(result.phase_radians - 2.13) < 1e-6
    assert result.grid_size <= 1025


def test_physical_phase_survives_a_sample_crop_with_new_origin():
    n, dt, start = 512, 0.01, 0.037
    time = start + np.arange(n) * dt
    signal = np.cos(2 * np.pi * 7.8125 * time + 0.91)
    reports = []
    for offset in (0, 128):
        y = signal[offset:]
        weights = np.exp(-np.arange(len(y)) / len(y))
        acquisition = PhaseAcquisition(
            start + offset * dt, dt, weights, np.full(len(y), 0.005), None, unwindowed_signal=y
        )
        report = fit_fft_phase(
            acquisition,
            PhaseFitConfig(6.0, 10.0),
        )
        assert isinstance(report, PhaseEstimate), report
        reports.append(report)
    assert abs(reports[0].phase_radians - reports[1].phase_radians) < 1e-6
    assert reports[1].standard_error_radians > reports[0].standard_error_radians


def test_explicit_band_can_select_a_weaker_isolated_peak():
    a = _sampled(noise=0.002, components=[(0.5, 5.859375, -1.24), (1.0, 22.265625, -1.24)])
    report = fit_fft_phase(a, PhaseFitConfig(4.0, 8.0))
    assert isinstance(report, PhaseEstimate), report
    assert abs(report.phase_radians + 1.24) < 0.025
    assert report.observation_band_mhz[1] < 22.265625


def test_acquisition_normalizes_large_mean_weights_and_owns_immutable_arrays():
    values = np.ones(12)
    a = PhaseAcquisition(0.0, 0.01, values, values, np.full(12, 1e308), unwindowed_signal=values)
    values[0] = 9.0
    assert a.window_weights[0] == 1.0
    assert a.mean_weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        a.unwindowed_signal[0] = 2.0


def _known_cosine_solution(bins):
    from asymmetry.core.fourier.phase_fit import _PhaseSolution

    n, dt, start = 512, 0.01, 0.037
    phase, amplitude, error = 0.73, 0.1, 0.01
    frequency = 42 / (n * dt)
    angle = 2 * np.pi * frequency * (start + np.arange(n) * dt) + phase
    signal = amplitude * np.cos(angle)
    acquisition = PhaseAcquisition(
        start, dt, np.ones(n), np.full(n, error), None, unwindowed_signal=signal
    )
    response = np.fft.rfft(np.cos(angle))[bins]
    derivative = np.fft.rfft(-amplitude * np.sin(angle))[bins]
    columns = np.concatenate((response.real, response.imag))[:, None]
    solution = _PhaseSolution(
        phase,
        np.array([amplitude]),
        np.zeros(2 * len(bins)),
        columns,
        np.concatenate((derivative.real, derivative.imag)),
        np.array([frequency]),
        bins,
        float("inf"),
    )
    return acquisition, solution


def test_conditional_total_amplitude_error_matches_analytic_cosine():
    from asymmetry.core.fourier.phase_fit import _conditional_amplitude_standard_error

    acquisition, solution = _known_cosine_solution(np.arange(1, 257))
    error = _conditional_amplitude_standard_error(solution, acquisition, 1.0)
    assert error == pytest.approx(0.01 * np.sqrt(2 / 512), rel=1e-12)
    assert solution.amplitudes.sum() / error == pytest.approx(160.0, rel=1e-12)


def test_conditional_amplitude_significance_ignores_empty_observation_guards():
    from asymmetry.core.fourier.phase_fit import _conditional_amplitude_standard_error

    significances = []
    for radius in (1, 4, 8, 32, 128, 255):
        bins = np.arange(max(1, 42 - radius), min(256, 42 + radius) + 1)
        acquisition, solution = _known_cosine_solution(bins)
        error = _conditional_amplitude_standard_error(solution, acquisition, 1.0)
        significances.append(solution.amplitudes.sum() / error)
    assert significances == pytest.approx([160.0] * len(significances), rel=1e-12)


def test_unidentifiable_amplitude_functional_has_no_pseudoinverse_precision():
    from dataclasses import replace

    from asymmetry.core.fourier.phase_fit import _conditional_amplitude_standard_error

    acquisition, solution = _known_cosine_solution(np.arange(1, 257))
    solution = replace(
        solution,
        columns=np.column_stack((solution.columns, -solution.columns)),
        amplitudes=np.array([0.15, 0.05]),
    )
    assert np.isinf(_conditional_amplitude_standard_error(solution, acquisition, 1.0))


def test_strong_identifiable_phase_is_not_rejected_by_empty_guard_noise_energy():
    from asymmetry.core.fourier.phase_fit import _noise_power

    acquisition, _ = _known_cosine_solution(np.arange(1, 257))
    spectrum = np.fft.rfft(acquisition.unwindowed_signal)
    old_metric = np.linalg.norm(spectrum[1:]) / np.sqrt(
        _noise_power(acquisition, acquisition.window_weights, np.arange(1, 257))
    )
    assert old_metric < 8
    result = fit_fft_phase(acquisition, PhaseFitConfig(4.0, 20.0))
    assert isinstance(result, PhaseEstimate), result
    assert result.signal_to_noise == pytest.approx(160.0, rel=1e-5)
    assert result.standard_error_radians < np.deg2rad(5.0)
    assert abs(result.phase_radians - 0.73) < 1e-6
    assert (
        result.to_dict()["signal_to_noise_kind"]
        == "conditional_total_positive_amplitude_with_phase_nuisance"
    )


@pytest.mark.slow
@pytest.mark.parametrize("n", [128, 512])
@pytest.mark.parametrize("window", ["none", "exponential", "hann"])
@pytest.mark.parametrize("heteroscedastic", [False, True])
@pytest.mark.parametrize("band_kind", ["narrow", "maximum", "auto"])
def test_full_pipeline_noise_selection_has_no_accepted_phase(n, window, heteroscedastic, band_kind):
    dt, start = 0.01, 0.083
    duration = n * dt
    seed = (
        13000
        + n
        + {"none": 0, "exponential": 1, "hann": 2}[window] * 1000
        + int(heteroscedastic) * 10000
        + {"narrow": 0, "maximum": 1, "auto": 2}[band_kind] * 100
    )
    errors = 0.3 + 1.4 * (np.arange(n) / n) ** 2 if heteroscedastic else np.ones(n)
    signal = np.random.default_rng(seed).normal(size=n) * errors
    mean_weights = 1 / errors**2
    signal -= np.average(signal, weights=mean_weights)
    weights = {"none": np.ones(n), "exponential": np.exp(-np.arange(n) / n), "hann": np.hanning(n)}[
        window
    ]
    acquisition = PhaseAcquisition(
        start, dt, weights, errors, mean_weights, unwindowed_signal=signal
    )
    if band_kind == "auto":
        config = PhaseFitConfig()
    elif band_kind == "narrow":
        config = PhaseFitConfig(12 / duration, 20 / duration)
    else:
        low = 4 / duration
        high = min(0.5 / dt, low + 255 / (3 * duration))
        config = PhaseFitConfig(low, high)
    result = fit_fft_phase(acquisition, config)
    assert isinstance(result, PhaseRejection), result
    assert result.reason not in {"band_too_wide", "invalid_band", "insufficient_band"}
