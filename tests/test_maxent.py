from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.maxent import (
    MaxEntCancelledError,
    MaxEntConfig,
    build_maxent_input,
    estimate_maxent_workload,
    initialize_state,
    maxent,
    opus,
    run_cycles,
    tropus,
)
from asymmetry.core.representation import FrequencyMaxEnt, representation_from_dict


def _synthetic_run(*, frequency_mhz: float = 1.5) -> Run:
    rng = np.random.default_rng(1234)
    bin_width = 0.04
    n = 256
    time = np.arange(n, dtype=float) * bin_width
    phases = [0.0, 90.0, 180.0, 270.0]
    histograms: list[Histogram] = []
    for phase in phases:
        signal = 1.0 + 0.18 * np.cos(2.0 * np.pi * frequency_mhz * time + np.deg2rad(phase))
        counts = 2500.0 * np.exp(-time / 2.1969811) * signal
        counts = rng.poisson(np.clip(counts, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=bin_width, t0_bin=0))
    return Run(
        run_number=44,
        histograms=histograms,
        metadata={"field": 110.0, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
            "group_names": {1: "G1", 2: "G2", 3: "G3", 4: "G4"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def test_opus_tropus_are_adjoint() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(n_spectrum_points=64, f_min_mhz=0.1, f_max_mhz=4.0, auto_window=False)
    prepared = build_maxent_input(run, config)
    rng = np.random.default_rng(5)
    spectrum = rng.random(prepared.n_spectrum_points)
    group_values = {
        group.group_id: rng.normal(size=group.time_us.size) for group in prepared.groups
    }

    forward = opus(spectrum, prepared)
    lhs = sum(
        float(np.dot(forward[group.group_id], group_values[group.group_id]))
        for group in prepared.groups
    )
    rhs = float(np.dot(spectrum, tropus(group_values, prepared)))

    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


def test_maxent_returns_non_negative_finite_spectrum_with_peak() -> None:
    run = _synthetic_run(frequency_mhz=1.5)
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
        fit_phases=False,
    )

    result = maxent(run, config)

    assert result.state.cycle == 4
    assert np.all(np.isfinite(result.spectrum))
    assert np.all(result.spectrum >= 0.0)
    peak_frequency = result.frequencies_mhz[int(np.argmax(result.spectrum))]
    assert peak_frequency == pytest.approx(1.5, abs=0.2)
    assert result.diagnostics.cycles == [1, 2, 3, 4]


def test_maxent_cycles_are_resumable() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=1,
        inner_iterations=3,
        fit_phases=False,
    )
    prepared = build_maxent_input(run, config)
    state = initialize_state(prepared, config)

    first = run_cycles(prepared, config, state=state, cycles=1)
    second = run_cycles(prepared, config, state=first.state, cycles=2)

    assert second.state.cycle == 3
    assert second.diagnostics.cycles == [1, 2, 3]


def test_maxent_state_rejects_restart_required_changes() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False)
    prepared = build_maxent_input(run, config)
    state = initialize_state(prepared, config)
    changed = MaxEntConfig(n_spectrum_points=128, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False)
    changed_input = build_maxent_input(run, changed)

    with pytest.raises(ValueError, match="restart"):
        run_cycles(changed_input, changed, state=state, cycles=1)


def test_maxent_time_range_and_binning_reduce_input_points() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        t_min_us=1.0,
        t_max_us=5.0,
        time_binning_factor=4,
    )

    prepared = build_maxent_input(run, config)
    estimate = estimate_maxent_workload(run, config)

    assert all(np.count_nonzero(group.mask) == 25 for group in prepared.groups)
    assert estimate.max_time_points == 25
    assert estimate.peak_dense_matrix_bytes == 25 * 64 * 8
    assert prepared.metadata["time_binning_factor"] == 4


def test_maxent_progress_callback_reports_work() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=2,
        inner_iterations=1,
    )
    progress: list[tuple[int, int, str]] = []

    result = maxent(run, config, progress_callback=lambda *payload: progress.append(payload))

    assert result.state.cycle == 2
    assert progress


def test_maxent_cancel_callback_stops_run() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=2,
        inner_iterations=1,
    )

    with pytest.raises(MaxEntCancelledError):
        maxent(run, config, cancel_callback=lambda: True)


def test_frequency_maxent_representation_round_trip_has_metadata_only() -> None:
    run = _synthetic_run()
    rep = FrequencyMaxEnt(
        recipe={
            "maxent_config": MaxEntConfig(
                n_spectrum_points=64,
                f_min_mhz=0.2,
                f_max_mhz=3.0,
                auto_window=False,
                outer_cycles=2,
                inner_iterations=2,
            ).to_dict()
        }
    )

    curves = rep.compute(run)
    payload = rep.to_dict()
    restored = representation_from_dict(payload)

    assert len(curves) == 1
    assert payload["result_metadata"]["cycles"] == 2
    assert "diagnostics" in payload["result_metadata"]
    assert "_datasets" not in payload
    assert restored.result_metadata["cycles"] == 2


def test_maxent_fitted_amplitudes_converge_instead_of_oscillating() -> None:
    """Regression: regressing against the amp/bg-scaled prediction made the
    fitted amplitude alternate (amp_{n+1} ~ a / amp_n) every cycle."""
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=6,
        inner_iterations=4,
        fit_phases=False,
    )

    result = maxent(run, config)

    for series in result.diagnostics.amplitudes[-1]:
        last = [row[series] for row in result.diagnostics.amplitudes[-3:]]
        spread = max(last) - min(last)
        assert spread < 0.25 * max(abs(v) for v in last)


def test_maxent_auto_window_takes_precedence_over_stale_explicit_bounds() -> None:
    run = _synthetic_run()  # field = 110 G -> centre ~1.49 MHz
    auto = MaxEntConfig(f_min_mhz=0.0, f_max_mhz=5.0, auto_window=True)
    explicit = MaxEntConfig(f_min_mhz=0.0, f_max_mhz=5.0, auto_window=False)

    auto_input = build_maxent_input(run, auto)
    explicit_input = build_maxent_input(run, explicit)

    assert explicit_input.f_max_mhz == pytest.approx(5.0)
    assert auto_input.f_max_mhz != pytest.approx(5.0)
    # Field-derived window: centre ± half-width, clipped at zero.
    assert auto_input.f_min_mhz == pytest.approx(0.0)
    assert auto_input.f_min_mhz <= 1.49 <= auto_input.f_max_mhz


def test_maxent_use_deadtime_correction_flag_controls_preparation() -> None:
    run = _synthetic_run()
    grouping = dict(run.grouping)
    grouping["deadtime_correction"] = True
    grouping["dead_time_us"] = [0.5] * len(run.histograms)
    grouping["good_frames"] = 1000.0
    run_with_deadtime = Run(
        run_number=run.run_number,
        histograms=run.histograms,
        metadata=run.metadata,
        grouping=grouping,
    )
    config_off = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        use_deadtime_correction=False,
    )
    config_on = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        use_deadtime_correction=True,
    )

    off_input = build_maxent_input(run_with_deadtime, config_off)
    on_input = build_maxent_input(run_with_deadtime, config_on)
    plain_input = build_maxent_input(run, config_off)

    # Off must ignore the run's metadata flag and match the uncorrected run.
    for off_group, plain_group in zip(off_input.groups, plain_input.groups, strict=True):
        np.testing.assert_allclose(off_group.signal, plain_group.signal)
    # On must change the prepared signals.
    assert any(
        not np.allclose(on_group.signal, off_group.signal)
        for on_group, off_group in zip(on_input.groups, off_input.groups, strict=True)
    )


def test_maxent_state_rejects_time_range_and_binning_changes() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False)
    prepared = build_maxent_input(run, config)
    state = initialize_state(prepared, config)

    for changed in (
        MaxEntConfig(
            n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False, t_max_us=5.0
        ),
        MaxEntConfig(
            n_spectrum_points=64,
            f_min_mhz=0.2,
            f_max_mhz=3.0,
            auto_window=False,
            time_binning_factor=2,
        ),
        MaxEntConfig(
            n_spectrum_points=64,
            f_min_mhz=0.2,
            f_max_mhz=3.0,
            auto_window=False,
            use_deadtime_correction=False,
        ),
    ):
        changed_input = build_maxent_input(run, changed)
        with pytest.raises(ValueError, match="restart"):
            run_cycles(changed_input, changed, state=state, cycles=1)
