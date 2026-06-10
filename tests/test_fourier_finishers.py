"""Tests for the frequency-domain finishers (Phase 1).

Field-axis verification, post-FFT conditioning (pulse compensation, robust
baseline, exclusions), the diamagnetic exclusion slot, the PSI harmonics preset,
and the Real+Imag display mode.  Conditioning is exercised both at the pure-core
level and end-to-end through :func:`compute_average_group_spectrum`.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.conditioning import (
    apply_spectrum_conditioning,
    pulse_compensation_gain,
    sigma_clip_baseline,
)
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)
from asymmetry.core.fourier.units import gauss_to_mhz, mhz_to_gauss
from asymmetry.core.maxent.pulse import pulse_amplitude_phase

GAMMA_MU_MHZ_PER_G = float(gauss_to_mhz(1.0))


def _tf_run(*, field_gauss: float, n: int = 512, bin_width: float = 0.04) -> Run:
    """A transverse-field run whose asymmetry precesses at γ_μ·B."""
    rng = np.random.default_rng(7)
    time = np.arange(n, dtype=float) * bin_width
    freq_mhz = field_gauss * GAMMA_MU_MHZ_PER_G
    amp = 0.20
    histograms: list[Histogram] = []
    for sign in (+1.0, -1.0):
        signal = 1.0 + sign * amp * np.cos(2.0 * np.pi * freq_mhz * time)
        counts = 4000.0 * np.exp(-time / 2.1969811) * signal
        counts = rng.poisson(np.clip(counts, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=bin_width, t0_bin=0))
    return Run(
        run_number=101,
        histograms=histograms,
        metadata={"field": float(field_gauss), "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Fwd", 2: "Bwd"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def _peak_frequency_mhz(freqs: np.ndarray, values: np.ndarray) -> float:
    """Return the frequency of the dominant non-DC bin."""
    if values.size <= 1:
        return 0.0
    idx = int(np.argmax(np.abs(values[1:]))) + 1
    return float(freqs[idx])


# ── field axis (verify-only) ────────────────────────────────────────────


def test_field_axis_fft_peak_at_gamma_b() -> None:
    field_gauss = 200.0
    run = _tf_run(field_gauss=field_gauss)
    ds = compute_average_group_spectrum(run, GroupSpectrumConfig(display="(Power)^1/2"))
    assert ds is not None
    peak_mhz = _peak_frequency_mhz(ds.time, ds.asymmetry)
    # One spectrum bin in Gauss sets the tolerance.
    bin_gauss = float(mhz_to_gauss(ds.time[1] - ds.time[0]))
    assert float(mhz_to_gauss(peak_mhz)) == pytest.approx(field_gauss, abs=1.5 * bin_gauss)


def test_fft_and_maxent_agree_on_peak_field() -> None:
    from asymmetry.core.maxent import MaxEntConfig, maxent

    field_gauss = 150.0
    run = _tf_run(field_gauss=field_gauss)
    fft_ds = compute_average_group_spectrum(run, GroupSpectrumConfig(display="(Power)^1/2"))
    fft_peak = _peak_frequency_mhz(fft_ds.time, fft_ds.asymmetry)

    freq = field_gauss * GAMMA_MU_MHZ_PER_G
    result = maxent(
        run,
        MaxEntConfig(
            n_spectrum_points=128,
            f_min_mhz=0.1,
            f_max_mhz=2.0 * freq,
            auto_window=False,
            outer_cycles=4,
            inner_iterations=4,
            fit_phases=False,
        ),
    )
    maxent_peak = float(result.frequencies_mhz[int(np.argmax(result.spectrum))])
    # Both spectra → the same applied field within their resolutions.
    assert float(mhz_to_gauss(fft_peak)) == pytest.approx(field_gauss, abs=8.0)
    assert float(mhz_to_gauss(maxent_peak)) == pytest.approx(field_gauss, abs=8.0)
    assert mhz_to_gauss(fft_peak) == pytest.approx(mhz_to_gauss(maxent_peak), abs=12.0)


# ── robust baseline ─────────────────────────────────────────────────────


def test_sigma_clip_removes_offset_preserves_peaks() -> None:
    rng = np.random.default_rng(0)
    values = 3.0 + rng.normal(0.0, 0.1, 1000)
    values[100] = 25.0
    values[400] = 18.0
    values[700] = 30.0
    baseline, sigma = sigma_clip_baseline(values, kappa=2.0)
    assert baseline == pytest.approx(3.0, abs=0.05)
    assert sigma == pytest.approx(0.1, abs=0.03)
    cleaned = values - baseline
    assert cleaned[100] == pytest.approx(22.0, abs=0.2)  # peak preserved


def test_sigma_clip_one_iteration_matches_wimda() -> None:
    rng = np.random.default_rng(1)
    values = 5.0 + rng.normal(0.0, 0.2, 500)
    values[::50] = 40.0  # sparse spikes
    # WiMDA: mean/std over all, then mean of points within 2σ — one reclip.
    mean = values.mean()
    std = values.std()
    inliers = values[np.abs(values - mean) <= 2.0 * std]
    wimda_expected = float(inliers.mean())
    baseline, _ = sigma_clip_baseline(values, kappa=2.0, max_iter=1, location="mean")
    assert baseline == pytest.approx(wimda_expected, abs=1e-9)


# ── pulse-response compensation ─────────────────────────────────────────


def test_pulse_compensation_flattens_amplitude() -> None:
    freqs = np.linspace(0.0, 30.0, 600)
    amplitude, _ = pulse_amplitude_phase(freqs, half_width_us=0.05, n_pulses=1)
    distorted = 1.0 * amplitude  # a flat-amplitude signal seen through the pulse
    result = apply_spectrum_conditioning(
        freqs, distorted, pulse_compensation=True, pulse_half_width_us=0.05
    )
    recovered = result.display
    below_cut = result.gain > 0.0
    assert np.allclose(recovered[below_cut], 1.0, atol=1e-9)


def test_pulse_compensation_guard_bounds_gain() -> None:
    freqs = np.linspace(0.0, 40.0, 800)
    gain = pulse_compensation_gain(freqs, half_width_us=0.07, max_gain=25.0)
    assert np.all(np.isfinite(gain))
    assert gain.max() <= 25.0 + 1e-9
    # Everything past the first node is cut off.
    cut = freqs[gain == 0.0]
    cut = cut[cut > 0.0]
    assert cut.size > 0
    cutoff = cut.min()
    assert np.all(gain[freqs >= cutoff] == 0.0)


def test_uncompensated_rolloff_is_monotone() -> None:
    # Below the first node the pulse amplitude falls monotonically: the
    # distortion compensation undoes.
    freqs = np.linspace(0.0, 12.0, 400)
    amplitude, _ = pulse_amplitude_phase(freqs, half_width_us=0.05, n_pulses=1)
    diffs = np.diff(amplitude)
    assert np.all(diffs <= 1e-9)


# ── exclusions + diamag slot ────────────────────────────────────────────


def test_exclusion_zeroes_requested_band() -> None:
    freqs = np.linspace(0.0, 20.0, 400)
    values = np.ones_like(freqs)
    result = apply_spectrum_conditioning(freqs, values, exclusion_ranges=[(10.0, 0.5)])
    inside = np.abs(freqs - 10.0) <= 0.5
    assert np.all(result.display[inside] == 0.0)
    assert np.all(result.display[~inside] == 1.0)


def test_diamag_slot_tracks_reference_field() -> None:
    field_gauss = 100.0
    run = _tf_run(field_gauss=field_gauss)
    config = GroupSpectrumConfig(
        display="(Power)^1/2",
        exclude_enabled=True,
        diamag_exclusion=True,
        diamag_half_width_mhz=0.3,
    )
    ds = compute_average_group_spectrum(run, config)
    ref_mhz = field_gauss * GAMMA_MU_MHZ_PER_G
    inside = np.abs(ds.time - ref_mhz) <= 0.3
    assert np.any(inside)  # the diamag band lies within the spectrum
    assert np.all(ds.asymmetry[inside] == 0.0)


def test_psi_preset_centres(qapp) -> None:
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    panel._apply_psi_harmonics_preset()
    ranges = panel.exclusion_ranges()
    centres = sorted(c for c, _ in ranges)
    assert centres == pytest.approx([0.0, 50.63, 101.26, 151.89, 202.52, 253.15], abs=1e-6)
    assert all(w == pytest.approx(0.5) for _, w in ranges)


# ── Real+Imag display mode ──────────────────────────────────────────────


def test_real_imag_returns_both_quadratures() -> None:
    run = _tf_run(field_gauss=200.0)
    ds = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Real+Imag"))
    assert ds is not None
    assert "fourier_imag" in ds.metadata
    imag = np.asarray(ds.metadata["fourier_imag"], dtype=float)
    assert imag.size == ds.asymmetry.size
    # Real and imaginary quadratures are genuinely different channels.
    assert not np.allclose(imag, ds.asymmetry)


# ── conditioning round-trip + recompute ─────────────────────────────────


def test_conditioning_config_roundtrip_and_recompute() -> None:
    run = _tf_run(field_gauss=200.0)
    config = GroupSpectrumConfig(
        display="(Power)^1/2",
        pulse_compensation=True,
        pulse_half_width_us=0.05,
        baseline_mode="sigma_clip",
        exclude_enabled=True,
        diamag_exclusion=True,
        exclusion_ranges=[(8.0, 0.4)],
    )
    restored = GroupSpectrumConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()

    first = compute_average_group_spectrum(run, config)
    second = compute_average_group_spectrum(run, restored)
    assert np.array_equal(first.asymmetry, second.asymmetry)
    assert np.array_equal(first.time, second.time)


def test_baseline_metadata_recorded() -> None:
    run = _tf_run(field_gauss=200.0)
    ds = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", baseline_mode="sigma_clip")
    )
    assert "fourier_baseline" in ds.metadata
    assert "fourier_baseline_noise" in ds.metadata
    assert ds.metadata["fourier_baseline_noise"] >= 0.0
