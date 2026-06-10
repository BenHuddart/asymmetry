"""Phase 3: ZF/LF mode, SpecBG, spectrum/log export."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.maxent import (
    MaxEntConfig,
    build_maxent_input,
    initialize_state,
    run_cycles,
    run_log_text,
    spectrum_to_text,
    subtract_zero_frequency,
)


def _kubo_toyabe_fb_run(*, delta: float = 0.3, alpha: float = 1.2) -> Run:
    """Two F/B groups carrying a static Gaussian Kubo–Toyabe relaxation."""
    rng = np.random.default_rng(4)
    bin_width = 0.04
    n = 300
    time = np.arange(n, dtype=float) * bin_width
    kubo_toyabe = 1.0 / 3.0 + (2.0 / 3.0) * (1.0 - delta**2 * time**2) * np.exp(
        -0.5 * delta**2 * time**2
    )
    histograms: list[Histogram] = []
    for phase, efficiency in ((0.0, 1.0), (180.0, alpha)):
        asym = 0.25 * kubo_toyabe * np.cos(np.deg2rad(phase))
        counts = efficiency * 4000.0 * np.exp(-time / 2.1969811) * (1.0 + asym)
        histograms.append(
            Histogram(
                counts=rng.poisson(np.clip(counts, 1.0, None)).astype(float),
                bin_width=bin_width,
                t0_bin=0,
            )
        )
    return Run(
        run_number=1,
        histograms=histograms,
        metadata={"field": 0.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "alpha": alpha,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def test_zf_lf_mode_pins_phases_ties_amplitudes_and_peaks_near_zero() -> None:
    alpha = 1.2
    run = _kubo_toyabe_fb_run(alpha=alpha)
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=8,
        inner_iterations=6,
        mode="zf_lf",
        selected_group_ids=[1, 2],
    )
    maxent_input = build_maxent_input(run, config)
    assert [g.phase_degrees for g in maxent_input.groups] == [0.0, 180.0]
    assert maxent_input.zf_lf_alpha == pytest.approx(alpha)

    result = run_cycles(maxent_input, config)
    # Phases stay pinned; amplitudes obey the α tie F = α·B.
    assert result.state.phases == {1: 0.0, 2: 180.0}
    assert result.state.amplitudes[1] / result.state.amplitudes[2] == pytest.approx(alpha, rel=1e-6)

    # A Kubo–Toyabe field distribution is broad and centred near zero — most of
    # the spectral weight sits below the line, not in a sharp high-frequency peak.
    frequencies = result.frequencies_mhz
    spectrum = result.spectrum
    near_zero = float(np.trapezoid(spectrum[frequencies < 0.5], frequencies[frequencies < 0.5]))
    away = float(np.trapezoid(spectrum[frequencies > 1.5], frequencies[frequencies > 1.5]))
    assert near_zero > away
    assert frequencies[int(np.argmax(spectrum))] < 0.5


def test_zf_lf_mode_requires_exactly_two_groups() -> None:
    run = _kubo_toyabe_fb_run()
    with pytest.raises(ValueError, match="two selected groups"):
        build_maxent_input(run, MaxEntConfig(mode="zf_lf", selected_group_ids=[1]))


def test_mode_round_trips_and_change_forces_restart() -> None:
    assert MaxEntConfig().mode == "general"
    assert MaxEntConfig.from_dict({"mode": "zf_lf"}).mode == "zf_lf"
    assert MaxEntConfig.from_dict({"mode": "nonsense"}).mode == "general"

    run = _kubo_toyabe_fb_run()
    base = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        selected_group_ids=[1, 2],
    )
    prepared = build_maxent_input(run, base)
    state = initialize_state(prepared, base)
    zf = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        selected_group_ids=[1, 2],
        mode="zf_lf",
    )
    with pytest.raises(ValueError, match="restart"):
        run_cycles(build_maxent_input(run, zf), zf, state=state, cycles=1)


def test_specbg_subtracts_central_peak_on_a_copy() -> None:
    frequencies = np.linspace(0.0, 3.0, 256)
    # A strong central peak plus a weak satellite at 1.5 MHz.
    central = np.exp(-((frequencies / 0.2) ** 2))
    satellite = 0.1 * np.exp(-(((frequencies - 1.5) / 0.1) ** 2))
    spectrum = central + satellite

    result = subtract_zero_frequency(
        frequencies,
        spectrum,
        gaussian_width=0.2,
        lorentzian_width=0.2,
        lorentzian_fraction=0.0,
    )
    # The central feature is strongly suppressed; the satellite survives.
    assert abs(result[0]) < 0.2 * spectrum[0]
    satellite_idx = int(np.argmin(np.abs(frequencies - 1.5)))
    assert result[satellite_idx] == pytest.approx(spectrum[satellite_idx], abs=0.02)
    # Display-only: the input array is untouched.
    assert spectrum[0] == pytest.approx(central[0] + satellite[0])


def test_spectrum_and_log_export_are_well_formed() -> None:
    run = _kubo_toyabe_fb_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=3,
        inner_iterations=3,
        selected_group_ids=[1, 2],
    )
    result = run_cycles(build_maxent_input(run, config), config)

    spectrum_text = spectrum_to_text(result, config)
    assert spectrum_text.startswith("# MaxEnt spectrum")
    assert "frequency_MHz" in spectrum_text
    data_lines = [ln for ln in spectrum_text.splitlines() if not ln.startswith("#")]
    assert len(data_lines) == result.frequencies_mhz.size
    assert len(data_lines[0].split("\t")) == 3

    log_text = run_log_text(result, config)
    assert "per-cycle convergence" in log_text
    assert "final group phases" in log_text
