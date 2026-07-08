from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.spectral import (
    default_frequency_model,
    seed_peak_parameters_from_dataset,
)


def _dc_dominated_spectrum(
    *, peak_freq: float = 30.0, n: int = 2000, f_max: float = 100.0
) -> MuonDataset:
    """Synthetic (Power)^1/2 spectrum shaped like the recorded EuO 2960 fixture.

    A tall apodisation/DC spike near 0 MHz dwarfs the genuine ~30 MHz
    precession peak, as observed live (ratio ~3x).
    """
    freq = np.linspace(0.0, f_max, n)
    dc_spike = 30.0 * np.exp(-0.5 * (freq / 0.5) ** 2)
    physical_peak = 10.0 * np.exp(-0.5 * ((freq - peak_freq) / 1.5) ** 2)
    bg = 0.2
    values = bg + dc_spike + physical_peak
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 2960, "plot_domain": "frequency"},
    )


def test_seed_peak_excludes_dc_spike_and_finds_physical_peak() -> None:
    dataset = _dc_dominated_spectrum(peak_freq=30.0)
    model = default_frequency_model()

    # Sanity: unguarded argmax would land on the DC spike, not the physical peak.
    y = np.asarray(dataset.asymmetry)
    assert float(dataset.time[int(np.nanargmax(y))]) < 1.0

    seeds = seed_peak_parameters_from_dataset(dataset, model)

    assert seeds["nu0"] == pytest.approx(30.0, abs=1.0)


def test_seed_peak_falls_back_to_global_argmax_when_guard_empties_array() -> None:
    freq = np.linspace(0.0, 1.0, 50)
    values = 5.0 - freq  # monotonic decay entirely inside a wide guard band
    dataset = MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 1, "plot_domain": "frequency"},
    )
    model = default_frequency_model()

    seeds = seed_peak_parameters_from_dataset(dataset, model, guard_freq_mhz=2.0)

    assert seeds["nu0"] == 0.0


def test_seed_peak_bg_uses_guarded_median_not_dc_spike() -> None:
    dataset = _dc_dominated_spectrum(peak_freq=30.0)
    model = default_frequency_model()

    seeds = seed_peak_parameters_from_dataset(dataset, model)

    assert seeds["bg"] < 1.0


def test_seed_peak_fwhm_excludes_dc_spike_from_half_max_crossing() -> None:
    """The half-max crossing search must stay inside the same guarded region
    as the peak search, or a DC spike that also exceeds half_height drags the
    span out to the DC spike's edge and inflates fwhm by an order of
    magnitude even though nu0 correctly skipped it.
    """
    dataset = _dc_dominated_spectrum(peak_freq=30.0)  # physical peak sigma=1.5 MHz
    model = default_frequency_model()

    seeds = seed_peak_parameters_from_dataset(dataset, model)

    # True FWHM = 2.3548 * sigma ~= 3.53 MHz; a DC-contaminated crossing would
    # instead span from near the DC spike's edge to the peak's far edge (~30 MHz).
    assert seeds["fwhm"] < 10.0


def _two_peak_spectrum(
    *, first: float = 2.0, second: float = 3.5, second_height: float = 3.0
) -> MuonDataset:
    """A frequency spectrum with two Gaussian lines plus a flat background."""
    freq = np.linspace(1.0, 5.0, 401)
    values = (
        0.4
        + 6.0 * np.exp(-4.0 * np.log(2.0) * ((freq - first) / 0.2) ** 2)
        + second_height * np.exp(-4.0 * np.log(2.0) * ((freq - second) / 0.3) ** 2)
    )
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 7, "plot_domain": "frequency"},
    )


def _two_peak_model() -> CompositeModel:
    return CompositeModel(
        ["GaussianPeak", "GaussianPeak", "ConstantBackground"], operators=["+", "+"]
    )


def test_seed_two_peaks_assigns_strongest_first_to_each_component() -> None:
    dataset = _two_peak_spectrum(first=2.0, second=3.5, second_height=3.0)
    seeds = seed_peak_parameters_from_dataset(dataset, _two_peak_model())

    # Both peak components are seeded onto real lines (not the nu0=1.0 default).
    assert seeds["nu0_1"] == pytest.approx(2.0, abs=0.05)
    assert seeds["nu0_2"] == pytest.approx(3.5, abs=0.05)
    # Strongest-first: the taller line seeds component 1.
    assert seeds["height_1"] > seeds["height_2"]


def test_seed_two_peaks_seeds_a_weak_second_line() -> None:
    """A weak-but-real second line the user declared is still seeded (not gated)."""
    dataset = _two_peak_spectrum(first=2.0, second=3.5, second_height=0.6)
    seeds = seed_peak_parameters_from_dataset(dataset, _two_peak_model())

    assert seeds["nu0_2"] == pytest.approx(3.5, abs=0.1)


def test_seed_two_peaks_under_detection_stays_on_screen() -> None:
    """One real peak but two components: the extra nu0 must stay in the window."""
    freq = np.linspace(1.0, 5.0, 401)
    values = 0.4 + 6.0 * np.exp(-4.0 * np.log(2.0) * ((freq - 2.0) / 0.2) ** 2)
    dataset = MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 8, "plot_domain": "frequency"},
    )
    seeds = seed_peak_parameters_from_dataset(dataset, _two_peak_model())

    assert 1.0 <= seeds["nu0_1"] <= 5.0
    assert 1.0 <= seeds["nu0_2"] <= 5.0


def test_seed_two_peaks_with_linear_background_seeds_slope() -> None:
    model = CompositeModel(
        ["GaussianPeak", "GaussianPeak", "LinearBackground"], operators=["+", "+"]
    )
    seeds = seed_peak_parameters_from_dataset(_two_peak_spectrum(), model)

    assert "slope" in seeds
    assert "nu0_1" in seeds and "nu0_2" in seeds
