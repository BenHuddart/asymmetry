"""Tests for the shared averaged grouped-FFT spectrum core (Phase 3)."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.fft import (
    average_fourier_display_values,
    fft_complex_asymmetry,
    fourier_display_values,
)
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
    precompute_group_fourier_inputs,
)


def _run() -> Run:
    rng = np.random.default_rng(0)
    counts_f = 1000.0 * np.exp(-np.arange(64) * 0.02) + rng.normal(0, 1, 64)
    counts_b = 950.0 * np.exp(-np.arange(64) * 0.02) + rng.normal(0, 1, 64)
    return Run(
        run_number=11,
        histograms=[
            Histogram(counts=np.abs(counts_f), bin_width=0.05, t0_bin=0),
            Histogram(counts=np.abs(counts_b), bin_width=0.05, t0_bin=0),
        ],
        metadata={"field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Fwd", 2: "Bwd"},
            "first_good_bin": 0,
            "last_good_bin": 63,
        },
    )


def _golden_average(run: Run, config: GroupSpectrumConfig) -> np.ndarray:
    """Replicate the GUI averaged-FFT pipeline inline (non-entropy modes)."""
    prepared, ref_t0 = precompute_group_fourier_inputs(run)
    values = []
    for gid in (1, 2):
        gds = build_group_signal_dataset(
            run, gid, center_signal=False, reference_t0_bin=ref_t0, prepared_histograms=prepared
        )
        freqs, spectrum = fft_complex_asymmetry(
            gds,
            window=config.window,
            padding_factor=config.padding,
            subtract_average_signal=config.subtract_average_signal,
            # "(Power)^1/2" is a canonical (non-derived) mode, so the shared core
            # applies the fractional footing + percent amplitude calibration; the
            # inline replica must do the same to stay a self-consistency pin.
            fractional=True,
            amplitude_calibration=True,
        )
        values.append(fourier_display_values(spectrum, display=config.display))
    averaged, _err = average_fourier_display_values(values, estimate_error=False)
    return averaged


def test_average_matches_inline_pipeline():
    run = _run()
    config = GroupSpectrumConfig(display="(Power)^1/2")
    spectrum = compute_average_group_spectrum(run, config)
    assert spectrum is not None
    np.testing.assert_allclose(spectrum.asymmetry, _golden_average(run, config))
    assert spectrum.metadata["fourier_group_output"] == "average"
    assert spectrum.metadata["group_ids"] == [1, 2]


def test_selected_group_subset_changes_label_and_ids():
    run = _run()
    config = GroupSpectrumConfig(selected_group_ids=[1])
    spectrum = compute_average_group_spectrum(run, config)
    assert spectrum is not None
    assert spectrum.metadata["group_ids"] == [1]
    assert "Average (" in spectrum.metadata["run_label"]


def test_no_groups_returns_none():
    run = Run(run_number=1, histograms=[Histogram(np.array([1.0, 2.0]), 0.1)], grouping={})
    assert compute_average_group_spectrum(run, GroupSpectrumConfig()) is None


def test_empty_selection_returns_none():
    run = _run()
    config = GroupSpectrumConfig(selected_group_ids=[])
    assert compute_average_group_spectrum(run, config) is None


def test_config_from_dict_round_trip_fields():
    config = GroupSpectrumConfig.from_dict(
        {
            "display": "Cos",
            "padding": 4,
            "window": "gaussian",
            "t_min_us": 0.1,
            "t_max_us": 8.0,
            "selected_group_ids": [2],
            "group_phase_degrees": {"1": 12.0, "2": -3.0},
        }
    )
    assert config.display == "Cos"
    assert config.padding == 4
    assert config.window == "gaussian"
    assert config.t_min_us == 0.1
    assert config.t_max_us == 8.0
    assert config.selected_group_ids == [2]
    assert config.group_phase_degrees == {1: 12.0, 2: -3.0}


def test_phase_modes_use_resolved_group_phases():
    run = _run()
    # "Phase" is the phase-corrected display mode that consumes per-group phases.
    base = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Phase"))
    phased = compute_average_group_spectrum(
        run,
        GroupSpectrumConfig(display="Phase", group_phase_degrees={1: 45.0, 2: 90.0}),
    )
    assert base is not None and phased is not None
    # A phase-correcting mode must respond to the resolved per-group phases.
    assert not np.allclose(base.asymmetry, phased.asymmetry)


def test_non_phase_mode_ignores_group_phases():
    run = _run()
    # Non-phase-correcting modes (e.g. magnitude) must ignore per-group phases,
    # matching the GUI's apply_phase_correction gating.
    base = compute_average_group_spectrum(run, GroupSpectrumConfig(display="(Power)^1/2"))
    phased = compute_average_group_spectrum(
        run,
        GroupSpectrumConfig(display="(Power)^1/2", group_phase_degrees={1: 45.0, 2: 90.0}),
    )
    assert base is not None and phased is not None
    np.testing.assert_allclose(base.asymmetry, phased.asymmetry)
