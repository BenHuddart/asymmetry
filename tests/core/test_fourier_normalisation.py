"""Calibrated FFT amplitude scale + unit-area field-distribution view.

The calibrated scale puts the grouped FFT on a fractional-asymmetry footing in
percent: a pure cosine of amplitude ``A`` peaks at ``100·A``, invariant to
counting statistics, window length, apodisation, and zero padding.  Each
invariant is pinned here.  The unit-area display normalisation presents a
magnitude spectrum as a field distribution ``p(ν)`` with ``∫ p dν = 1``.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.conditioning import unit_area_normalise
from asymmetry.core.fourier.fft import fft_complex_asymmetry
from asymmetry.core.fourier.spectrum import (
    UNIT_AREA_YLABEL,
    GroupSpectrumConfig,
    compute_average_group_spectrum,
    config_differences,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US

_A = 0.05  # fractional asymmetry amplitude → calibrated peak 100·A = 5.0
_EXPECT = 100.0 * _A


def _cosine_dataset(
    *,
    n0: float = 1.0e4,
    amplitude: float = _A,
    n: int = 256,
    dt: float = 0.01,
    cycles: int = 32,
) -> MuonDataset:
    """A flat-baseline cosine on a count scale: ``N₀·(1 + A·cos(2π f₀ t))``.

    ``cycles`` is an integer, so the tone sits exactly on an FFT bin (on-grid),
    and ``f₀ = cycles/(n·dt)`` is placed mid-band (well below Nyquist and above
    DC).  A constant error makes the error-weighted baseline exactly ``N₀`` so
    the calibration is isolated from the weighting.
    """
    t = np.arange(n, dtype=float) * dt
    f0 = cycles / (n * dt)
    counts = n0 * (1.0 + amplitude * np.cos(2.0 * np.pi * f0 * t))
    error = np.full(n, np.sqrt(n0), dtype=float)
    return MuonDataset(time=t, asymmetry=counts, error=error, metadata={"field": 100.0})


def _calibrated_peak(dataset: MuonDataset, **kwargs) -> float:
    freqs, spectrum = fft_complex_asymmetry(
        dataset, fractional=True, amplitude_calibration=True, **kwargs
    )
    magnitude = np.abs(spectrum)
    positive = freqs > 0.0
    return float(np.max(magnitude[positive]))


def test_pure_cosine_peaks_at_hundred_times_amplitude() -> None:
    peak = _calibrated_peak(_cosine_dataset())
    assert peak == pytest.approx(_EXPECT, abs=1.0e-3)


def test_invariant_to_count_level() -> None:
    low = _calibrated_peak(_cosine_dataset(n0=1.0e3))
    high = _calibrated_peak(_cosine_dataset(n0=1.0e4))
    assert low == pytest.approx(_EXPECT, abs=1.0e-3)
    assert high == pytest.approx(low, abs=1.0e-6)


def test_invariant_to_time_window_length() -> None:
    # Keep f0 and dt fixed while quadrupling the bin count → cycles scale ×4 so
    # the tone stays on-grid; the calibrated peak is unchanged.
    short = _calibrated_peak(_cosine_dataset(n=256, cycles=32))
    long = _calibrated_peak(_cosine_dataset(n=1024, cycles=128))
    assert short == pytest.approx(_EXPECT, abs=1.0e-3)
    assert long == pytest.approx(_EXPECT, abs=1.0e-3)


def test_invariant_to_zero_padding() -> None:
    base = _calibrated_peak(_cosine_dataset(), padding_factor=1)
    padded = _calibrated_peak(_cosine_dataset(), padding_factor=4)
    assert base == pytest.approx(_EXPECT, abs=1.0e-3)
    assert padded == pytest.approx(_EXPECT, abs=1.0e-3)


@pytest.mark.parametrize("window", ["hann", "cosine"])
def test_invariant_to_apodisation_window(window: str) -> None:
    # Symmetric apodisation windows on an on-grid mid-band tone: the coherent
    # gain correction (2/Σw) recovers the amplitude with negligible leakage.
    peak = _calibrated_peak(_cosine_dataset(), window=window)
    assert peak == pytest.approx(_EXPECT, abs=0.1)


@pytest.mark.parametrize("window", ["gaussian", "lorentzian"])
def test_invariant_to_apodisation_filter(window: str) -> None:
    # The gaussian/lorentzian FFT *filters* are time-domain decaying envelopes;
    # with a gentle (large) time constant they are near-flat, so Σw ≈ n and the
    # gain correction recovers the amplitude for an unrelaxed line.
    peak = _calibrated_peak(_cosine_dataset(), window=window, filter_time_constant_us=1000.0)
    assert peak == pytest.approx(_EXPECT, abs=0.1)


def _cosine_run(*, n0_fwd: float, n0_bwd: float, amplitude: float = _A) -> Run:
    """Two single-detector groups with different count levels, same amplitude.

    Counts carry the muon-decay envelope so the pipeline's lifetime correction
    restores a flat ``N₀·(1 + A·cos)`` before the FFT.
    """
    n = 512
    bw = 0.01
    cycles = 64
    t = np.arange(n, dtype=float) * bw
    f0 = cycles / (n * bw)
    envelope = np.exp(-t / MUON_LIFETIME_US)
    osc = 1.0 + amplitude * np.cos(2.0 * np.pi * f0 * t)
    counts_f = n0_fwd * envelope * osc
    counts_b = n0_bwd * envelope * osc
    return Run(
        run_number=7,
        histograms=[
            Histogram(counts=counts_f, bin_width=bw, t0_bin=0),
            Histogram(counts=counts_b, bin_width=bw, t0_bin=0),
        ],
        metadata={"field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Fwd", 2: "Bwd"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
        },
    )


def test_two_group_average_unequal_counts_calibrates() -> None:
    run = _cosine_run(n0_fwd=2.0e4, n0_bwd=5.0e3)
    spectrum = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Magnitude"))
    assert spectrum is not None
    peak = float(np.max(spectrum.asymmetry[spectrum.time > 0.0]))
    # Each group is independently calibrated, so unequal N₀ still averages to
    # the fractional amplitude in percent (small tolerance for the weighted
    # baseline under the decaying error).
    assert peak == pytest.approx(_EXPECT, abs=0.3)
    assert spectrum.metadata["fourier_normalisation"] == "asymmetry_percent"
    assert spectrum.metadata["y_label"] == "FFT Magnitude (%)"


def test_power_mode_is_square_of_calibrated_magnitude() -> None:
    run = _cosine_run(n0_fwd=1.0e4, n0_bwd=1.0e4)
    mag = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Magnitude"))
    power = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Power"))
    assert mag is not None and power is not None
    np.testing.assert_allclose(power.asymmetry, np.square(mag.asymmetry), rtol=1e-9, atol=1e-9)
    assert power.metadata["y_label"] == "FFT Power (%²)"


# ── unit-area (field distribution) ──────────────────────────────────────────


def _line_plus_floor(*, floor: float, n: int = 400, extra_noise: int = 0) -> tuple:
    """Frequency grid with a Gaussian line on a flat noise floor.

    ``extra_noise`` pads both ends with more floor-only bins to widen the
    computed range without adding signal.
    """
    rng = np.random.default_rng(1)
    dv = 0.05  # MHz per bin
    total = n + 2 * extra_noise
    freqs = np.arange(total, dtype=float) * dv
    centre = freqs[total // 2]
    line = 30.0 * np.exp(-0.5 * ((freqs - centre) / (5.0 * dv)) ** 2)
    values = floor + line + rng.normal(0.0, 0.05 * floor, total)
    return freqs, values, dv


def test_unit_area_integrates_to_one() -> None:
    freqs, values, dv = _line_plus_floor(floor=2.0)
    result = unit_area_normalise(freqs, values)
    assert result.applied
    integral = float(np.sum(result.display) * dv)
    assert integral == pytest.approx(1.0, abs=1e-6)


def test_unit_area_range_independent() -> None:
    freqs_a, values_a, dv = _line_plus_floor(floor=2.0, extra_noise=0)
    freqs_b, values_b, _ = _line_plus_floor(floor=2.0, extra_noise=200)
    area_a = unit_area_normalise(freqs_a, values_a).area
    area_b = unit_area_normalise(freqs_b, values_b).area
    # Doubling the noise-only range leaves the floor-subtracted integral stable.
    assert area_b == pytest.approx(area_a, rel=0.1)


def test_unit_area_floor_level_independent() -> None:
    freqs, values_low, dv = _line_plus_floor(floor=2.0)
    _, values_high, _ = _line_plus_floor(floor=4.0)
    low = unit_area_normalise(freqs, values_low)
    high = unit_area_normalise(freqs, values_high)
    assert low.applied and high.applied
    assert float(np.sum(low.display) * dv) == pytest.approx(1.0, abs=1e-6)
    assert float(np.sum(high.display) * dv) == pytest.approx(1.0, abs=1e-6)


def test_unit_area_guard_refuses_pure_noise() -> None:
    rng = np.random.default_rng(2)
    freqs = np.arange(400, dtype=float) * 0.05
    noise = 2.0 + rng.normal(0.0, 0.1, 400)
    result = unit_area_normalise(freqs, noise)
    assert not result.applied
    assert "not significant" in result.reason


def test_unit_area_skipped_note_for_real_display() -> None:
    run = _cosine_run(n0_fwd=1.0e4, n0_bwd=1.0e4)
    spectrum = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="Real", display_normalisation="unit_area")
    )
    assert spectrum is not None
    assert "magnitude" in spectrum.metadata["fourier_unit_area_skipped"]
    assert spectrum.metadata["y_label"] == "FFT Real (%)"


def test_unit_area_pipeline_labels_density() -> None:
    run = _cosine_run(n0_fwd=1.0e4, n0_bwd=1.0e4)
    spectrum = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="Magnitude", display_normalisation="unit_area")
    )
    assert spectrum is not None
    assert spectrum.metadata["fourier_display_normalisation"] == "unit_area"
    assert spectrum.metadata["y_label"] == UNIT_AREA_YLABEL


# ── config round-trip + staleness ───────────────────────────────────────────


def test_legacy_recipe_flags_amplitude_normalisation_stale() -> None:
    # A pre-calibration recipe lacks the normalisation key → "raw" sentinel,
    # which differs from a freshly-built config's "asymmetry_percent" default.
    recorded = GroupSpectrumConfig.from_dict({"display": "Magnitude"})
    assert recorded.normalisation == "raw"
    current = GroupSpectrumConfig(display="Magnitude")
    assert "amplitude normalisation" in config_differences(current, recorded)


def test_new_recipe_round_trips_without_false_staleness() -> None:
    config = GroupSpectrumConfig(display="Magnitude", display_normalisation="unit_area")
    restored = GroupSpectrumConfig.from_dict(config.to_dict())
    assert restored.normalisation == "asymmetry_percent"
    assert restored.display_normalisation == "unit_area"
    assert config_differences(config, restored) == []
