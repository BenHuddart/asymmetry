from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
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
