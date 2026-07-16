from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.units import gauss_to_mhz
from asymmetry.core.maxent import (
    MaxEntCancelledError,
    MaxEntConfig,
    build_maxent_input,
    estimate_maxent_workload,
    initialize_state,
    maxent,
    opus,
    reconstruct_group_signals,
    resolve_maxent_auto_steering,
    run_cycles,
    tropus,
)
from asymmetry.core.representation import (
    FrequencyMaxEnt,
    TimeMaxEntReconstruction,
    representation_from_dict,
)


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

    # This asserts amplitude behaviour over a fixed cycle count, so opt out of
    # the χ²-plateau early-stop rather than relying on staying under its gate.
    result = maxent(run, config, early_stop=False)

    for series in result.diagnostics.amplitudes[-1]:
        last = [row[series] for row in result.diagnostics.amplitudes[-3:]]
        spread = max(last) - min(last)
        assert spread < 0.25 * max(abs(v) for v in last)


def test_maxent_forced_cycles_stop_at_chi2_plateau_instead_of_diverging() -> None:
    """Early-stop (the default) treats the cycle count as a maximum and halts
    at the χ² plateau instead of burning the full budget.  (Historically the
    unguarded run also drifted off the line to a grid edge; that pathology was
    driven by the nuisance amplitude floor pinning the model oscillation far
    too large, which is fixed — a forced 60-cycle run now stays on the line,
    so the guard's job is purely to stop wasted post-plateau work.)
    """
    run = _synthetic_run(frequency_mhz=1.5)
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        inner_iterations=4,
        fit_phases=False,
    )

    unguarded = maxent(run, config, cycles=60, early_stop=False)
    guarded = maxent(run, config, cycles=60, early_stop=True)

    # Both resolve the true line — forcing cycles no longer loses the peak.
    unguarded_peak = unguarded.frequencies_mhz[int(np.argmax(unguarded.spectrum))]
    assert unguarded_peak == pytest.approx(1.5, abs=0.2)

    # The guard stops well short of the requested 60 cycles with the same
    # answer: the global argmax is the real line and χ² is comparable.
    assert guarded.early_stopped is True
    assert guarded.state.cycle < 60
    assert np.all(np.isfinite(guarded.spectrum))
    guarded_peak = guarded.frequencies_mhz[int(np.argmax(guarded.spectrum))]
    assert guarded_peak == pytest.approx(1.5, abs=0.2)
    assert guarded.diagnostics.chi2[-1] < 2.0 * unguarded.diagnostics.chi2[-1]


def test_maxent_stop_reason_flags_behave() -> None:
    """The converged/diverged/early_stopped flags report how the loop ended."""
    run = _synthetic_run(frequency_mhz=1.5)
    base = dict(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        fit_phases=False,
    )

    # A short run never reaches the early-stop gate: exact count, no flags.
    short = maxent(run, MaxEntConfig(inner_iterations=4, **base), cycles=4)
    assert short.state.cycle == 4
    assert short.stop_reason == "max_cycles"
    assert short.early_stopped is False
    assert short.converged is False
    assert short.diverged is False

    # Slow convergence (optimum past the gate): stops on the plateau, converged.
    converged = maxent(run, MaxEntConfig(inner_iterations=4, **base), cycles=60)
    assert converged.early_stopped is True
    assert converged.converged is True
    assert converged.diverged is False
    assert converged.converged is not converged.diverged
    assert converged.metadata["maxent_stop_reason"] == "converged"

    # Divergence: the healthy engine no longer reaches a genuinely rising-χ²
    # regime on this synthetic (the amplitude-floor defect that drove it is
    # fixed), so emulate a post-optimum resume: rewrite the resumed history's
    # last χ² well below anything the spectrum can actually achieve.  The next
    # cycle's "improvement" is then robustly negative and the guard must stop
    # immediately and flag divergence for the GUI to warn on.
    config = MaxEntConfig(inner_iterations=4, **base)
    prepared = build_maxent_input(run, config)
    past_optimum = run_cycles(prepared, config, cycles=8, early_stop=False)
    past_optimum.state.diagnostics.chi2[-1] *= 0.5
    diverged = run_cycles(prepared, config, state=past_optimum.state, cycles=3)
    assert diverged.early_stopped is True
    assert diverged.diverged is True
    assert diverged.stop_reason == "diverged"
    assert diverged.metadata["maxent_diverged"] is True


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
    phases and would otherwise silently ignore the new seeds.  (Data-derived
    seeding is off here — with it on, the table is only a fallback and an
    edit legitimately changes nothing.)"""
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        fit_phases=False,
        auto_phase_seed=False,
    )
    prepared = build_maxent_input(run, config)
    state = initialize_state(prepared, config)

    changed = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        fit_phases=False,
        auto_phase_seed=False,
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


def test_reconstruct_group_signals_chi2_equals_engine_chi2() -> None:
    """The overlay χ² is the engine χ² by construction (same residual path)."""
    run = _synthetic_run(frequency_mhz=1.5)
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=5,
        inner_iterations=4,
    )
    maxent_input = build_maxent_input(run, config)
    result = run_cycles(maxent_input, config)

    recon = reconstruct_group_signals(maxent_input, result.state)

    # One reconstruction per selected group.
    assert sorted(recon) == [group.group_id for group in maxent_input.groups]
    total_chi2 = sum(rg.chi2 for rg in recon.values())
    assert total_chi2 == pytest.approx(result.metadata["maxent_chi2"], rel=1e-9, abs=1e-9)

    # The model is exactly opus of the converged spectrum, masked to good points.
    predictions = opus(
        result.state.spectrum,
        maxent_input,
        phases=result.state.phases,
        amplitudes=result.state.amplitudes,
        backgrounds=result.state.backgrounds,
    )
    for group in maxent_input.groups:
        rg = recon[group.group_id]
        mask = group.mask
        np.testing.assert_allclose(rg.model, predictions[group.group_id][mask])
        np.testing.assert_allclose(rg.data, np.asarray(group.signal)[mask])
        np.testing.assert_allclose(rg.residual, (rg.data - rg.model) / rg.sigma)
        assert rg.n_obs == int(np.count_nonzero(mask))


def test_result_carries_prepared_input_for_zero_rebuild_reconstruction() -> None:
    """The result threads the exact prepared input the cycles iterated, so the
    GUI overlay can reconstruct without rebuilding it — and the χ² it gets is the
    engine's by identity (same object), not merely by deterministic rebuild."""
    run = _synthetic_run(frequency_mhz=1.5)
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
    )
    maxent_input = build_maxent_input(run, config)
    result = run_cycles(maxent_input, config)
    # run_cycles threads the very object it iterated through onto the result.
    assert result.maxent_input is maxent_input
    # Reconstructing from that carried input reproduces the engine χ² exactly.
    recon = reconstruct_group_signals(result.maxent_input, result.state)
    total_chi2 = sum(rg.chi2 for rg in recon.values())
    assert total_chi2 == pytest.approx(result.metadata["maxent_chi2"], rel=1e-12, abs=1e-12)

    # The maxent() convenience wrapper (build + run) attaches it too, and its
    # reconstruction matches the same-config rebuild path the GUI falls back to.
    wrapped = maxent(run, config)
    assert wrapped.maxent_input is not None
    carried = sum(
        rg.chi2 for rg in reconstruct_group_signals(wrapped.maxent_input, wrapped.state).values()
    )
    rebuilt = sum(
        rg.chi2
        for rg in reconstruct_group_signals(build_maxent_input(run, config), wrapped.state).values()
    )
    assert carried == pytest.approx(rebuilt, rel=1e-12, abs=1e-12)


def _fb_single_line_run(*, frequency_mhz: float = 1.5) -> Run:
    """Two opposed F/B groups carrying one clean TF line at known phase."""
    rng = np.random.default_rng(7)
    bin_width = 0.04
    n = 256
    time = np.arange(n, dtype=float) * bin_width
    histograms: list[Histogram] = []
    for phase in (0.0, 180.0):
        signal = 1.0 + 0.20 * np.cos(2.0 * np.pi * frequency_mhz * time + np.deg2rad(phase))
        counts = 3000.0 * np.exp(-time / 2.1969811) * signal
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
        metadata={"field": 110.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def test_reconstruct_matches_input_single_line() -> None:
    """Reconstruction of a clean single line tracks the input data and the
    spectrum peaks at the true frequency.

    The model positively reconstructs each group's oscillation (the overlay's
    purpose); the V1 engine clamps the joint amplitude to its floor, so the
    correct test of "matches the input" is that the reconstruction tracks the
    data shape and frequency, not an absolute weighted-residual bound (that
    would test the V1's convergence, deliberately out of scope here)."""
    run = _fb_single_line_run(frequency_mhz=1.5)
    config = MaxEntConfig(
        n_spectrum_points=256,
        f_min_mhz=0.5,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=10,
        inner_iterations=8,
        fit_phases=False,
        group_phase_degrees={1: 0.0, 2: 180.0},
    )
    maxent_input = build_maxent_input(run, config)
    result = run_cycles(maxent_input, config)

    peak = result.frequencies_mhz[int(np.argmax(result.spectrum))]
    assert peak == pytest.approx(1.5, abs=0.1)

    recon = reconstruct_group_signals(maxent_input, result.state)
    for rg in recon.values():
        correlation = float(np.corrcoef(rg.data, rg.model)[0, 1])
        assert correlation > 0.5


def test_maxent_reconstruction_representation_builds_overlay_datasets() -> None:
    run = _synthetic_run()
    rep = TimeMaxEntReconstruction(
        recipe={
            "maxent_config": MaxEntConfig(
                n_spectrum_points=64,
                f_min_mhz=0.2,
                f_max_mhz=3.0,
                auto_window=False,
                outer_cycles=3,
                inner_iterations=3,
            ).to_dict()
        }
    )

    datasets = rep.compute(run)

    # One overlay dataset per group, each carrying the model + residual arrays.
    assert len(datasets) == len(run.grouping["groups"])
    for dataset in datasets:
        assert dataset.metadata["maxent_reconstruction"] is True
        model = dataset.metadata["maxent_model"]
        residual = dataset.metadata["maxent_residual"]
        assert model.shape == dataset.asymmetry.shape
        assert residual.shape == dataset.asymmetry.shape
    # Recipe-only persistence: no arrays leak into the serialised payload.
    payload = rep.to_dict()
    assert payload["rep_type"] == "time_maxent_recon"
    assert "_datasets" not in payload
    assert "maxent_model" not in str(payload["result_metadata"])


def test_maxent_config_show_reconstruction_round_trips() -> None:
    assert MaxEntConfig().show_reconstruction is False
    assert MaxEntConfig.from_dict({}).show_reconstruction is False
    on = MaxEntConfig(show_reconstruction=True)
    assert MaxEntConfig.from_dict(on.to_dict()).show_reconstruction is True


# ── Long-window robustness, phase seeding, and workload auto-steering ────────


def _long_window_quadrature_run(
    *,
    frequency_mhz: float = 5.4,
    n_bins: int = 2000,
    bin_width_us: float = 0.016,
    field: float = 400.0,
) -> Run:
    """MUSR-shaped TF run: 4 quadrature groups, integer counts, ~14 lifetimes.

    The long window leaves the late tail with 0/1-count bins whose
    lifetime-corrected values explode (a 1-count bin at 30 µs becomes ~10⁶),
    which is exactly the regime that poisoned the plain-mean baseline and the
    unweighted nuisance regression on real MUSR data.
    """
    rng = np.random.default_rng(7)
    time = np.arange(n_bins, dtype=float) * bin_width_us
    histograms: list[Histogram] = []
    for phase in (0.0, 90.0, 180.0, 270.0):
        signal = 1.0 + 0.2 * np.cos(2.0 * np.pi * frequency_mhz * time + np.deg2rad(phase))
        expected = 15000.0 * np.exp(-time / 2.1969811) * signal
        counts = rng.poisson(expected).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=bin_width_us, t0_bin=0))
    return Run(
        run_number=77,
        histograms=histograms,
        metadata={"field": field},
        grouping={
            "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
            "group_names": {1: "G1", 2: "G2", 3: "G3", 4: "G4"},
            "first_good_bin": 0,
            "last_good_bin": n_bins - 1,
            "deadtime_correction": False,
        },
    )


def _phase_distance_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_maxent_baseline_survives_lifetime_amplified_tail() -> None:
    """The normalisation baseline is 1/σ²-weighted, so the exp-amplified
    late-time Poisson tail cannot inflate it: the well-measured early-time
    normalised signal stays centred on zero (a plain mean left it at ~−0.7 on
    real MUSR data, guaranteeing divergence from cycle 1)."""
    run = _long_window_quadrature_run()
    prepared = build_maxent_input(run, MaxEntConfig())

    for group in prepared.groups:
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        early = mask & (group.time_us < 4.0)
        assert abs(float(np.mean(group.signal[early]))) < 0.05


def test_maxent_auto_phase_seed_recovers_quadrature_geometry() -> None:
    """Data-derived phase seeds land near the groups' true 0/90/180/270 layout
    (up to a common rotation) — the ±4°/cycle refinement could never reach
    them from an all-zero start."""
    run = _long_window_quadrature_run()
    prepared = build_maxent_input(run, MaxEntConfig())

    seeded = {group.group_id: float(group.phase_degrees) for group in prepared.groups}
    reference = seeded[1]
    for gid, true_phase in ((1, 0.0), (2, 90.0), (3, 180.0), (4, 270.0)):
        assert _phase_distance_deg(seeded[gid] - reference, true_phase) < 20.0

    off = build_maxent_input(run, MaxEntConfig(auto_phase_seed=False))
    assert all(group.phase_degrees == 0.0 for group in off.groups)


def test_maxent_long_window_quadrature_converges_to_line() -> None:
    """End-to-end regression for the real-data divergence: an out-of-the-box
    config on a long-window quadrature run converges to the true line with
    χ²/N ≈ 1, and the fitted amplitudes settle well below the legacy 0.01
    clip floor (which forced a 50×-oversized oscillation the solver could
    only fight by decohering the spectrum into spikes)."""
    run = _long_window_quadrature_run()
    result = maxent(run, MaxEntConfig(), cycles=8)

    n_obs = sum(
        int(np.count_nonzero(group.mask)) if group.mask is not None else group.time_us.size
        for group in result.maxent_input.groups
    )
    chi2_per_n = result.diagnostics.chi2[-1] / n_obs
    assert not result.diverged
    assert chi2_per_n < 2.0
    peak = result.frequencies_mhz[int(np.argmax(result.spectrum))]
    assert peak == pytest.approx(5.4, abs=0.15)
    amplitudes = list(result.state.amplitudes.values())
    assert all(0.0 < amp < 0.01 for amp in amplitudes)
    assert max(amplitudes) / min(amplitudes) < 3.0


def _high_resolution_run(*, n_bins: int = 100_000, bin_width_us: float = 2.44e-5) -> Run:
    """HiFi/HAL-shaped run: ~0.02 ns bins, GHz bandwidth, two groups."""
    time = np.arange(n_bins, dtype=float) * bin_width_us
    histograms = []
    for phase in (0.0, 180.0):
        signal = 1.0 + 0.2 * np.cos(2.0 * np.pi * 813.0 * time + np.deg2rad(phase))
        counts = 50.0 * np.exp(-time / 2.1969811) * signal
        histograms.append(Histogram(counts=counts, bin_width=bin_width_us, t0_bin=0))
    return Run(
        run_number=88,
        histograms=histograms,
        metadata={"field": 60000.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "first_good_bin": 0,
            "last_good_bin": n_bins - 1,
            "deadtime_correction": False,
        },
    )


def test_auto_steering_is_noop_for_ordinary_resolution() -> None:
    run = _long_window_quadrature_run()
    config = MaxEntConfig()
    steered = resolve_maxent_auto_steering(run, config)

    assert steered.time_binning_factor == 1
    assert steered.t_max_us is None
    assert steered.n_spectrum_points is None


def test_auto_steering_sizes_high_resolution_workload() -> None:
    """Unset workload knobs are sized to the run: high-resolution data is
    binned toward the window's Nyquist need, the grid is capped, and the
    spectrum length stops defaulting to a runaway power of two."""
    run = _high_resolution_run()
    steered = resolve_maxent_auto_steering(run, MaxEntConfig())

    assert steered.time_binning_factor >= 2
    assert steered.t_max_us is not None
    assert steered.n_spectrum_points == 1024
    # Post-binning Nyquist still clears the top of the auto window.
    nyquist = 1.0 / (2.0 * 2.44e-5 * steered.time_binning_factor)
    f_max = gauss_to_mhz(60000.0) + gauss_to_mhz(300.0)
    assert nyquist > f_max

    estimate = estimate_maxent_workload(run, MaxEntConfig())
    assert estimate.max_time_points <= 8192
    assert estimate.n_spectrum_points == 1024


def test_auto_steering_respects_explicit_values_and_off_switch() -> None:
    run = _high_resolution_run()
    explicit = MaxEntConfig(time_binning_factor=3, t_max_us=1.0, n_spectrum_points=256)
    steered = resolve_maxent_auto_steering(run, explicit)
    assert steered.time_binning_factor == 3
    assert steered.t_max_us == pytest.approx(1.0)
    assert steered.n_spectrum_points == 256

    off = resolve_maxent_auto_steering(run, MaxEntConfig(auto_steer=False))
    assert off.time_binning_factor == 1
    assert off.t_max_us is None
    assert off.n_spectrum_points is None


def test_maxent_config_new_flags_round_trip() -> None:
    config = MaxEntConfig(auto_steer=False, auto_phase_seed=False)
    restored = MaxEntConfig.from_dict(config.to_dict())
    assert restored.auto_steer is False
    assert restored.auto_phase_seed is False
    # Old recipes without the keys default to the new behaviour.
    assert MaxEntConfig.from_dict({}).auto_steer is True
    assert MaxEntConfig.from_dict({}).auto_phase_seed is True
