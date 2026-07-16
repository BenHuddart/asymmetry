"""Tests for the shared averaged grouped-FFT spectrum core (Phase 3)."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
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


def _fb_pair_run() -> Run:
    """A single F/B pair carrying a damped ~3 MHz oscillation with Poisson noise.

    Forward and backward counts share a common muon decay and carry the
    oscillation in antiphase (a real transverse-field asymmetry). The
    lifetime correction the grouped-average path applies amplifies the
    late-time Poisson noise, raising its spectral floor; the forward−backward
    asymmetry stays bounded, so it resolves the line with a cleaner floor.
    """
    rng = np.random.default_rng(42)
    n = 512
    dt = 0.016
    t = np.arange(n) * dt
    decay = np.exp(-t / 2.197)
    osc = 0.20 * np.cos(2 * np.pi * 3.0 * t) * np.exp(-t / 4.0)
    n0 = 3000.0
    counts_f = rng.poisson(n0 * decay * (1.0 + osc)).astype(float)
    counts_b = rng.poisson(n0 * decay * (1.0 - osc)).astype(float)
    return Run(
        run_number=7,
        histograms=[
            Histogram(counts=counts_f, bin_width=dt, t0_bin=0),
            Histogram(counts=counts_b, bin_width=dt, t0_bin=0),
        ],
        metadata={"field": 220.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "alpha": 1.0,
        },
    )


def _peak_to_floor_contrast(spectrum: MuonDataset) -> float:
    """Peak height over the median off-peak floor within the in-window band."""
    freqs = np.asarray(spectrum.time, dtype=float)
    values = np.abs(np.asarray(spectrum.asymmetry, dtype=float))
    band = (freqs > 0.2) & (freqs < 8.0)
    fb, yb = freqs[band], values[band]
    peak_idx = int(np.argmax(yb))
    off_peak = np.abs(fb - fb[peak_idx]) > 0.5
    floor = float(np.median(yb[off_peak]))
    return float(yb[peak_idx]) / floor


def test_fb_asymmetry_beats_grouped_average_contrast():
    run = _fb_pair_run()
    grouped = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", signal_source="grouped_average")
    )
    fb = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", signal_source="fb_asymmetry")
    )
    assert grouped is not None and fb is not None
    # Both resolve the same ~3 MHz line, but the F−B asymmetry keeps a cleaner
    # in-window floor than the lifetime-corrected grouped average.
    assert _peak_to_floor_contrast(fb) > _peak_to_floor_contrast(grouped)


def test_fb_asymmetry_metadata_and_label():
    run = _fb_pair_run()
    fb = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", signal_source="fb_asymmetry")
    )
    grouped = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", signal_source="grouped_average")
    )
    assert fb is not None and grouped is not None
    assert fb.metadata["fourier_signal_source"] == "fb_asymmetry"
    assert grouped.metadata["fourier_signal_source"] == "grouped_average"
    # The F−B label distinguishes the source from the grouped "… Average".
    assert fb.metadata["run_label"] == "7 F−B"
    assert grouped.metadata["run_label"].endswith("Average")
    # Labels and axes carry over unchanged from the grouped path.
    assert fb.metadata["x_label"] == grouped.metadata["x_label"]
    assert fb.metadata["y_label"] == grouped.metadata["y_label"]


def test_fb_asymmetry_ignores_group_selection_and_phase_table():
    run = _fb_pair_run()
    # selected_group_ids and the per-group phase table are inert in fb mode:
    # deselecting/rephasing groups must not change the spectrum.
    base = compute_average_group_spectrum(run, GroupSpectrumConfig(signal_source="fb_asymmetry"))
    perturbed = compute_average_group_spectrum(
        run,
        GroupSpectrumConfig(
            signal_source="fb_asymmetry",
            selected_group_ids=[1],
            group_phase_degrees={1: 33.0, 2: -12.0},
        ),
    )
    assert base is not None and perturbed is not None
    np.testing.assert_allclose(base.asymmetry, perturbed.asymmetry)


def test_signal_source_round_trip_and_missing_key_default():
    config = GroupSpectrumConfig(signal_source="fb_asymmetry")
    payload = config.to_dict()
    assert payload["signal_source"] == "fb_asymmetry"
    assert GroupSpectrumConfig.from_dict(payload).signal_source == "fb_asymmetry"
    # A pre-fb recipe (missing key) is a grouped-average recipe, not stale.
    legacy = {k: v for k, v in payload.items() if k != "signal_source"}
    assert GroupSpectrumConfig.from_dict(legacy).signal_source == "grouped_average"


def test_fb_mode_returns_none_without_fb_pair_grouped_still_computes():
    # A single detector group: the backward group (default id 2) references no
    # detectors, so the fb reduction is impossible. fb mode must return None
    # (a batch compute skips the run) while grouped mode still transforms the
    # one group.
    counts = 1000.0 * np.exp(-np.arange(64) * 0.02) + 1.0
    run = Run(
        run_number=12,
        histograms=[
            Histogram(counts=counts.copy(), bin_width=0.05, t0_bin=0),
            Histogram(counts=counts.copy(), bin_width=0.05, t0_bin=0),
        ],
        metadata={"field": 100.0},
        grouping={
            "groups": {1: [1, 2]},
            "group_names": {1: "All"},
            "first_good_bin": 0,
            "last_good_bin": 63,
        },
    )
    assert (
        compute_average_group_spectrum(run, GroupSpectrumConfig(signal_source="fb_asymmetry"))
        is None
    )
    grouped = compute_average_group_spectrum(run, GroupSpectrumConfig())
    assert grouped is not None
