from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input


def _synthetic_run(*, field_gauss: float, frequency_mhz: float, n: int = 2048) -> Run:
    """A synthetic four-detector run precessing at *frequency_mhz*."""
    rng = np.random.default_rng(1234)
    bin_width = 0.01
    time = np.arange(n, dtype=float) * bin_width
    phases = [0.0, 90.0, 180.0, 270.0]
    histograms: list[Histogram] = []
    for phase in phases:
        signal = 1.0 + 0.18 * np.cos(2.0 * np.pi * frequency_mhz * time + np.deg2rad(phase))
        counts = 2500.0 * np.exp(-time / 2.1969811) * signal
        counts = rng.poisson(np.clip(counts, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=bin_width, t0_bin=0))
    return Run(
        run_number=2960,
        histograms=histograms,
        metadata={"field": field_gauss, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
            "group_names": {1: "G1", 2: "G2", 3: "G3", 4: "G4"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def test_zf_window_is_data_aware_and_contains_internal_field() -> None:
    """A ZF run's internal 30 MHz field must not collapse the window to (0, 10) MHz (D7/F19)."""
    run = _synthetic_run(field_gauss=0.0, frequency_mhz=30.0)
    config = MaxEntConfig(auto_window=True)
    prepared = build_maxent_input(run, config)

    assert prepared.f_min_mhz <= 30.0 <= prepared.f_max_mhz
    # Confirms the data-aware path engaged rather than the (0, 10) MHz
    # near-DC fallback that guaranteed the observed divergence (D7/F19).
    assert prepared.f_max_mhz > 10.0
    assert prepared.f_max_mhz < 100.0


def test_tf_window_stays_field_centred() -> None:
    """A field-scanning TF run keeps the existing field-centred auto window (regression)."""
    run = _synthetic_run(field_gauss=100.0, frequency_mhz=1.3554, n=256)
    config = MaxEntConfig(auto_window=True, window_half_width_gauss=50.0)
    prepared = build_maxent_input(run, config)

    from asymmetry.core.maxent.engine import _field_to_frequency_mhz

    center = _field_to_frequency_mhz(100.0)
    half_width = _field_to_frequency_mhz(50.0)
    assert prepared.f_min_mhz == max(0.0, center - half_width)
    assert prepared.f_max_mhz == center + half_width


def test_explicit_window_bounds_still_win_when_auto_window_off() -> None:
    """Manual Window entries must win regardless of the data-aware ZF path."""
    run = _synthetic_run(field_gauss=0.0, frequency_mhz=30.0)
    config = MaxEntConfig(auto_window=False, f_min_mhz=0.1, f_max_mhz=4.0)
    prepared = build_maxent_input(run, config)

    assert prepared.f_min_mhz == 0.1
    assert prepared.f_max_mhz == 4.0
