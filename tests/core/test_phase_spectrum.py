"""Automatic phase integration preserves calibrated measured FFTs and provenance."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier import spectrum as spectrum_module
from asymmetry.core.fourier.fft import (
    fft_complex_prepared,
    finite_bin_mean_weights,
    prepare_fft_arrays,
    prepare_fft_time_signal,
)
from asymmetry.core.fourier.phase_fit import (
    PhaseEstimate,
    PhaseFitCancelled,
    PhaseFitRejectedError,
    PhaseRejection,
)
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
    config_differences,
)
from asymmetry.core.representation.factory import representation_from_dict
from asymmetry.core.representation.frequency import FrequencyFFT


def _phase_run(monkeypatch, *, count_scale=1.0):
    n, dt = 192, 0.04
    time = 0.23 + np.arange(n) * dt
    frequency = 30 / (n * dt)
    phases = (0.7, 0.7 - np.pi)
    signals = {
        gid: MuonDataset(
            time=time,
            asymmetry=level * (1.0 + 0.12 * np.cos(2 * np.pi * frequency * time + phase)),
            error=np.full(n, level * 0.0001),
        )
        for gid, level, phase in zip((1, 2), (1000 * count_scale, 7000 * count_scale), phases)
    }
    run = Run(
        run_number=9,
        histograms=[Histogram(counts=np.ones(n), bin_width=dt, t0_bin=0)],
        grouping={"groups": {1: [1], 2: [1]}, "group_names": {1: "F", 2: "B"}},
    )
    monkeypatch.setattr(spectrum_module, "precompute_group_fourier_inputs", lambda run: ([], 0))
    monkeypatch.setattr(
        spectrum_module, "build_group_signal_dataset", lambda run, gid, **kwargs: signals[gid]
    )
    return run, signals, phases, frequency


def _estimate(phase):
    return PhaseEstimate(phase, 0.001, (2.0, 5.0), 0.002, 100.0, 1.0, 91, 3, (0.0, 10.0))


def test_preparation_exposes_actual_mean_window_and_noise():
    time = 0.4 + np.arange(20) * 0.1
    signal = 200.0 + 5 * np.sin(np.arange(20))
    errors = np.arange(1.0, 21.0)
    prepared = prepare_fft_arrays(time, signal, errors, fractional=True, window="hann")
    weights = (1 / errors) / np.sum(1 / errors)
    baseline = np.dot(signal, weights)
    np.testing.assert_allclose(prepared.mean_weights, weights)
    np.testing.assert_allclose(prepared.sample_errors, errors / baseline)
    np.testing.assert_allclose(prepared.signal, prepared.unwindowed_signal * np.hanning(20))
    assert prepared.time_start_us == time[0]
    assert prepared.window_sum == pytest.approx(np.hanning(20).sum())
    invalid_errors = np.zeros(20)
    np.testing.assert_allclose(finite_bin_mean_weights(signal, invalid_errors), np.full(20, 0.05))


def test_per_group_phase_fit_rotates_measured_channels_before_average(monkeypatch):
    run, signals, phases, frequency = _phase_run(monkeypatch)
    config = GroupSpectrumConfig(display="phaseOptReal", padding=2)
    result = compute_average_group_spectrum(run, config)
    assert result is not None
    expected = []
    for gid, phase in zip((1, 2), phases):
        prepared = prepare_fft_time_signal(signals[gid], fractional=True)
        freq, observed = fft_complex_prepared(
            prepared, padding_factor=2, amplitude_calibration=True
        )
        expected.append((observed * np.exp(-1j * (phase + 2 * np.pi * freq * 0.23))).real)
        report = result.metadata["fourier_phase_fit"]["groups"][str(gid)]
        assert abs(np.angle(np.exp(1j * (report["phase_radians"] - phase)))) < 0.01
        assert report["status"] == "accepted"
    np.testing.assert_allclose(result.asymmetry, np.mean(expected, axis=0), atol=0.005)
    peak = np.argmin(abs(result.time - frequency))
    assert result.asymmetry[peak] == pytest.approx(12.0, abs=0.01)
    assert result.metadata["y_label"] == "FFT auto phase (%)"
    assert result.metadata["fourier_normalisation"] == "asymmetry_percent"


def test_auto_phase_is_intensity_independent_and_ignores_manual_controls(monkeypatch):
    run, _, phases, _ = _phase_run(monkeypatch)
    estimates = iter(_estimate(phase) for phase in phases * 2)
    monkeypatch.setattr(spectrum_module, "fit_fft_phase", lambda *args, **kwargs: next(estimates))

    def forbid_diamagnetic_fit(*args, **kwargs):
        pytest.fail("Auto Phase must preserve the measured signal and sample covariance")

    monkeypatch.setattr(spectrum_module, "fit_and_subtract_diamagnetic", forbid_diamagnetic_fit)
    config = GroupSpectrumConfig(display="phaseOptReal")
    first = compute_average_group_spectrum(run, config)
    run, _, _, _ = _phase_run(monkeypatch, count_scale=100)
    run.metadata["field"] = 100.0
    changed = replace(
        config,
        group_phase_degrees={1: 140, 2: -80},
        t0_offset_us=2.0,
        pulse_compensation=True,
        baseline_mode="sigma_clip",
        exclude_enabled=True,
        exclusion_ranges=[(3.9, 4.0)],
        remove_diamag=True,
    )
    second = compute_average_group_spectrum(run, changed)
    assert first is not None and second is not None
    np.testing.assert_allclose(first.asymmetry, second.asymmetry, atol=1e-12)
    assert not any(key.startswith("fourier_diamag") for key in second.metadata)
    assert config_differences(changed, config) == []


def test_phase_controls_round_trip_and_only_mark_auto_phase_stale():
    auto = GroupSpectrumConfig(
        display="phaseOptReal",
        phase_fit_min_mhz=1.2,
        phase_fit_max_mhz=4.3,
        phase_fit_tolerance_degrees=3.0,
    )
    assert GroupSpectrumConfig.from_dict(auto.to_dict()) == auto
    assert GroupSpectrumConfig.from_dict({}).phase_fit_tolerance_degrees == 5.0
    changed = replace(auto, phase_fit_max_mhz=4.5)
    assert "auto-phase maximum frequency" in config_differences(changed, auto)
    assert not config_differences(replace(changed, display="Cos"), replace(auto, display="Cos"))


def test_rejection_names_group_and_does_not_publish_partial_report(monkeypatch):
    run, _, phases, _ = _phase_run(monkeypatch)
    rejected = PhaseRejection("noise", "Increase counting statistics or select a stronger band.")
    responses = iter((_estimate(phases[0]), rejected))
    monkeypatch.setattr(spectrum_module, "fit_fft_phase", lambda *args, **kwargs: next(responses))
    rep = FrequencyFFT(
        recipe={"fourier_config": GroupSpectrumConfig(display="phaseOptReal").to_dict()}
    )
    rep.result_metadata = {"saved": "report"}
    with pytest.raises(PhaseFitRejectedError, match=r"Group 2 \(B\).*Increase") as caught:
        rep.compute(run)
    assert caught.value.report is rejected
    assert rep.result_metadata == {"saved": "report"}


@pytest.mark.parametrize(
    ("reason", "change"),
    [
        ("mean_not_subtracted", {"subtract_average_signal": False}),
        ("non_finite_samples", {}),
    ],
)
def test_unsuitable_samples_are_typed_group_rejections(monkeypatch, reason, change):
    run, signals, _, _ = _phase_run(monkeypatch)
    if reason == "non_finite_samples":
        signals[1].asymmetry[5] = np.nan
    monkeypatch.setattr(
        spectrum_module, "fit_fft_phase", lambda *args, **kwargs: pytest.fail("no fit expected")
    )
    config = GroupSpectrumConfig(display="phaseOptReal", **change)
    with pytest.raises(PhaseFitRejectedError, match=r"Group 1 \(F\)") as caught:
        compute_average_group_spectrum(run, config)
    assert caught.value.report.reason == reason


def test_cancellation_between_groups_and_context_forwarding(monkeypatch):
    run, _, phases, _ = _phase_run(monkeypatch)
    cancelled = False
    calls = []

    def fit(*args, cancel_callback, **kwargs):
        nonlocal cancelled
        assert cancel_callback is callback
        calls.append(1)
        cancelled = True
        return _estimate(phases[0])

    def callback():
        return cancelled

    monkeypatch.setattr(spectrum_module, "fit_fft_phase", fit)
    rep = FrequencyFFT(recipe={"fourier_config": {"display": "phaseOptReal"}})
    with pytest.raises(PhaseFitCancelled):
        rep.compute(run, context=callback)
    assert len(calls) == 1
    assert rep.result_metadata == {}


def test_recipe_matches_direct_and_preserves_phase_report(monkeypatch):
    run, _, phases, _ = _phase_run(monkeypatch)
    responses = iter(_estimate(phase) for phase in phases * 2)
    monkeypatch.setattr(spectrum_module, "fit_fft_phase", lambda *args, **kwargs: next(responses))
    config = GroupSpectrumConfig(display="phaseOptReal", t_min_us=0.3, t_max_us=6.0)
    direct = compute_average_group_spectrum(run, config)
    recipe = config.to_dict()
    recipe.pop("phase_fit_min_mhz")
    recipe["grouping_digest"] = "saved-grouping-digest"
    rep = FrequencyFFT(recipe={"fourier_config": recipe})
    assert not rep.recompute_on_load
    actual = rep.compute(run)[0]
    assert direct is not None
    np.testing.assert_array_equal(actual.asymmetry, direct.asymmetry)
    assert rep.result_metadata["fourier_phase_fit"] == direct.metadata["fourier_phase_fit"]
    restored = representation_from_dict(rep.to_dict())
    assert restored.result_metadata == rep.result_metadata
    for group_id in ("1", "2"):
        acquisition = restored.result_metadata["fourier_phase_fit"]["groups"][group_id][
            "acquisition"
        ]
        assert acquisition == {
            "sample_count": 143,
            "time_start_us": pytest.approx(0.31),
            "sample_interval_us": pytest.approx(0.04),
        }
    assert rep.recipe == {"fourier_config": recipe}
    assert FrequencyFFT(recipe={"fourier_config": {"display": "Cos"}}).recompute_on_load


def test_auto_fb_uses_percent_asymmetry_without_fractional_baseline(monkeypatch):
    run, signals, phases, _ = _phase_run(monkeypatch)
    sampled = signals[1]
    fractional = (sampled.asymmetry / 1000.0 - 1) * 100.0
    fb = MuonDataset(time=sampled.time, asymmetry=fractional, error=np.full(len(fractional), 0.01))
    import asymmetry.core.simulate as simulate

    monkeypatch.setattr(simulate, "reduce_run_to_dataset", lambda run: fb)
    monkeypatch.setattr(
        spectrum_module, "fit_fft_phase", lambda *args, **kwargs: _estimate(phases[0])
    )
    result = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="phaseOptReal", signal_source="fb_asymmetry")
    )
    assert result is not None
    assert "asymmetry" in result.metadata["fourier_phase_fit"]["groups"]
    assert result.asymmetry.max() == pytest.approx(12.0, abs=0.01)


@pytest.mark.parametrize(
    ("declared_t0", "exact_subbin"), [(124, None), (124, 0.0), (124, 0.25), (None, None)]
)
def test_auto_group_good_bins_preserve_declared_profile_t0_coordinates(
    monkeypatch, declared_t0, exact_subbin
):
    """Aligning detector t0s must preserve the profile's post-prompt good interval."""
    n, dt = 512, 0.01
    histograms = []
    for t0 in (130, 124, 124):
        relative_time = (np.arange(n) - t0) * dt
        counts = 1000.0 * (1.0 + 0.1 * np.cos(2 * np.pi * 4 * relative_time + 0.2))
        counts[t0 : t0 + 2] = 1.0e9
        histograms.append(Histogram(counts=counts, bin_width=dt, t0_bin=t0))
    grouping = {
        "groups": {1: [1], 2: [2], 3: [3]},
        "group_names": {1: "Forw", 2: "Up", 3: "Down"},
        "forward_group": 2,
        "backward_group": 3,
        "first_good_bin": 130,
        "last_good_bin": 500,
    }
    if declared_t0 is not None:
        grouping["t0_bin"] = declared_t0
    if exact_subbin is not None:
        grouping["t0_time_us"] = (declared_t0 + 0.5 + exact_subbin) * dt
    original_grouping = dict(grouping)
    run = Run(run_number=11, histograms=histograms, grouping=grouping)
    observed = []

    def capture_phase_input(acquisition, config, **kwargs):
        observed.append(acquisition)
        return _estimate(0.2)

    monkeypatch.setattr(spectrum_module, "fit_fft_phase", capture_phase_input)
    result = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="phaseOptReal", selected_group_ids=[2, 3])
    )
    assert result is not None
    assert len(observed) == 2
    expected_start = (6 - (exact_subbin or 0.0)) * dt if declared_t0 is not None else 0.0
    for acquisition, group_id in zip(observed, ("2", "3")):
        assert acquisition.time_start_us == pytest.approx(expected_start)
        assert len(acquisition.window_weights) == 371
        if declared_t0 is not None:
            assert np.max(np.abs(acquisition.unwindowed_signal)) < 2.0
        report = result.metadata["fourier_phase_fit"]["groups"][group_id]["acquisition"]
        assert report == {
            "sample_count": 371,
            "time_start_us": pytest.approx(expected_start),
            "sample_interval_us": pytest.approx(dt),
        }
    assert run.grouping is grouping
    assert grouping == original_grouping
    assert grouping["first_good_bin"] == 130
    assert grouping["last_good_bin"] == 500
    for histogram, t0 in zip(run.histograms, (130, 124, 124)):
        assert histogram.t0_bin == t0
        np.testing.assert_array_equal(histogram.counts[t0 : t0 + 2], [1e9, 1e9])
