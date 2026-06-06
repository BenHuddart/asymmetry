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

    # Off must override the run's metadata flag and match the uncorrected run.
    for off_group, plain_group in zip(off_input.groups, plain_input.groups, strict=True):
        np.testing.assert_allclose(off_group.signal, plain_group.signal)
    # On + grouping flag on must change the prepared signals.
    assert any(
        not np.allclose(on_group.signal, off_group.signal)
        for on_group, off_group in zip(on_input.groups, off_input.groups, strict=True)
    )

    # "Use existing deadtime correction" honours the run-level setting: with
    # deadtime data present but the grouping flag OFF (the loader default),
    # config-on must NOT correct — keeping MaxEnt consistent with the FFT
    # path, which follows the grouping flag.
    grouping_flag_off = dict(grouping)
    grouping_flag_off["deadtime_correction"] = False
    run_flag_off = Run(
        run_number=run.run_number,
        histograms=run.histograms,
        metadata=run.metadata,
        grouping=grouping_flag_off,
    )
    flag_off_input = build_maxent_input(run_flag_off, config_on)
    for flag_off_group, plain_group in zip(flag_off_input.groups, plain_input.groups, strict=True):
        np.testing.assert_allclose(flag_off_group.signal, plain_group.signal)


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


def test_maxent_state_rejects_phase_seed_changes() -> None:
    """Edited phase seeds must force a restart: a resumed state keeps its own
    phases and would otherwise silently ignore the new seeds."""
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        fit_phases=False,
    )
    prepared = build_maxent_input(run, config)
    state = initialize_state(prepared, config)

    changed = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        fit_phases=False,
        group_phase_degrees={1: 90.0},
    )
    changed_input = build_maxent_input(run, changed)
    with pytest.raises(ValueError, match="restart"):
        run_cycles(changed_input, changed, state=state, cycles=1)


def test_fused_gradient_matches_opus_tropus_reference() -> None:
    """The fused production gradient must satisfy the same adjoint identity the
    opus/tropus pair does — protecting the path run_cycles actually executes."""
    from asymmetry.core.maxent.engine import _MIN_POSITIVE, _residual_gradient_payload

    run = _synthetic_run()
    config = MaxEntConfig(n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False)
    prepared = build_maxent_input(run, config)
    state = initialize_state(prepared, config)
    state.phases = {group.group_id: 3.0 * i for i, group in enumerate(prepared.groups)}
    state.amplitudes = {group.group_id: 0.8 + 0.1 * i for i, group in enumerate(prepared.groups)}
    state.backgrounds = {group.group_id: 0.01 * i for i, group in enumerate(prepared.groups)}

    predictions = opus(
        state.spectrum,
        prepared,
        phases=state.phases,
        amplitudes=state.amplitudes,
        backgrounds=state.backgrounds,
    )
    weighted: dict[int, np.ndarray] = {}
    chi2_ref = 0.0
    n_ref = 0
    for group in prepared.groups:
        mask = group.mask
        assert mask is not None
        sigma = np.maximum(np.asarray(group.sigma, dtype=float), _MIN_POSITIVE)
        residual = np.zeros_like(np.asarray(group.signal, dtype=float))
        residual[mask] = (
            np.asarray(group.signal, dtype=float)[mask] - predictions[group.group_id][mask]
        ) / sigma[mask]
        chi2_ref += float(np.sum(residual[mask] ** 2))
        n_ref += int(np.count_nonzero(mask))
        weighted[group.group_id] = np.where(mask, residual / sigma, 0.0)
    grad_ref = tropus(weighted, prepared, phases=state.phases, amplitudes=state.amplitudes)

    grad, chi2, n_obs = _residual_gradient_payload(state, prepared)

    np.testing.assert_allclose(grad, grad_ref, rtol=1e-10, atol=1e-12)
    assert chi2 == pytest.approx(chi2_ref, rel=1e-12)
    assert n_obs == n_ref


def test_maxent_config_from_dict_tolerates_malformed_recipe_entries() -> None:
    """Recipes cross the project-file boundary: malformed entries must degrade
    instead of raising out of whichever GUI slot touches the recipe."""
    config = MaxEntConfig.from_dict(
        {
            "n_spectrum_points": "not-a-number",
            "default_level": "bad",
            "window_half_width_gauss": None,
            "outer_cycles": "bad",
            "inner_iterations": None,
            "chi2_target_over_n": "bad",
            "selected_group_ids": [1, "abc", None, "2"],
            "group_phase_degrees": {"1": 10.0, "abc": 5.0, 2: "bad", "3": "inf"},
            "time_binning_factor": "bad",
        }
    )

    assert config.n_spectrum_points is None
    assert config.default_level == pytest.approx(0.01)
    assert config.outer_cycles == 10
    assert config.inner_iterations == 12
    assert config.selected_group_ids == [1, 2]
    assert config.group_phase_degrees == {1: 10.0}
    assert config.time_binning_factor == 1
