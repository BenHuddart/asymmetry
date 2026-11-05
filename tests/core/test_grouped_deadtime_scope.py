"""Required-observation contracts at grouped count and spectral entry points."""

import warnings

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_count_groups
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.spectrum import GroupSpectrumConfig, compute_average_group_spectrum
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input


def scoped_run(kind):
    arrays = [np.full(64, 10.0), np.full(64, 5.0), np.full(64, 100.0)]
    coefficients = [0.1, 0.1, 1.0]
    if kind == "prefix":
        arrays[0][0] = 100.0
        arrays[2][:] = 4.0
        coefficients = [1.0, 0.1, 0.1]
    if kind == "missing":
        arrays[2][:] = 4.0
        coefficients = [0.1, 0.1, None]
    grouping = {
        "groups": {1: [1], 2: [2], 3: [3]},
        "group_names": {1: "F", 2: "B", 3: "unused"},
        "included_groups": {1: True, 2: True, 3: False},
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 1 if kind == "prefix" else 0,
        "last_good_bin": 63,
        "good_frames": 100.0,
        "deadtime_correction": True,
        "deadtime_mode": "manual",
        "deadtime_method": "manual",
        "dead_time_us": coefficients,
        "alpha": 1.0,
    }
    if kind == "missing":
        grouping.update(deadtime_file_us=coefficients, deadtime_mode="file", deadtime_method="file")
    return Run(
        histograms=[Histogram(a, bin_width=1.0, t0_bin=0) for a in arrays], grouping=grouping
    )


def consume(run, consumer, selected=(1,)):
    dataset = MuonDataset(time=np.arange(64.0), asymmetry=np.zeros(64), error=np.ones(64), run=run)
    if consumer == "counts":
        return build_count_groups(dataset, list(selected), lifetime_corrected=False)
    if consumer == "signal":
        return build_group_signal_dataset(
            run, selected[0], center_signal=False, apply_lifetime_correction=False
        )
    if consumer in {"fft", "fb_fft"}:
        return compute_average_group_spectrum(
            run,
            GroupSpectrumConfig(
                selected_group_ids=list(selected),
                signal_source="fb_asymmetry" if consumer == "fb_fft" else "grouped_average",
            ),
        )
    return build_maxent_input(
        run,
        MaxEntConfig(
            auto_steer=False,
            auto_phase_seed=False,
            auto_window=False,
            n_spectrum_points=16,
            f_min_mhz=0.0,
            f_max_mhz=0.4,
            selected_group_ids=list(selected),
            fit_phases=False,
        ),
    )


@pytest.mark.parametrize("kind", ["unused", "prefix", "missing"])
@pytest.mark.parametrize("consumer", ["counts", "signal", "fft", "fb_fft", "maxent"])
def test_unused_observations_do_not_block_selected_input(kind, consumer):
    run = scoped_run(kind)
    before = [h.counts.copy() for h in run.histograms]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = consume(run, consumer)
    assert result is not None
    assert not [w for w in caught if "deadtime" in str(w.message).lower()]
    expected = 10.0 / (1.0 - 10.0 * (1.0 if kind == "prefix" else 0.1) / 100.0)
    if consumer == "counts":
        np.testing.assert_allclose(result[0].counts, expected)
        np.testing.assert_allclose(result[0].error, np.sqrt(10.0))
        assert result[0].deadtime.status.value == "applied"
    if consumer == "signal":
        np.testing.assert_allclose(result.asymmetry, expected)
        np.testing.assert_allclose(result.error, np.sqrt(10.0))
        assert result.metadata["deadtime_outcome"]["status"] == "applied"
    for histogram, original in zip(run.histograms, before, strict=True):
        np.testing.assert_array_equal(histogram.counts, original)


@pytest.mark.parametrize("consumer", ["counts", "signal", "fft", "maxent"])
def test_required_pole_still_fails(consumer):
    with pytest.raises(ValueError, match="Deadtime correction rejected"):
        consume(scoped_run("unused"), consumer, (3,))


@pytest.mark.parametrize("consumer", ["counts", "fft", "maxent"])
def test_required_file_gap_skips_whole_cohort(consumer):
    with pytest.warns(UserWarning, match="deadtime"):
        result = consume(scoped_run("missing"), consumer, (1, 3))
    if consumer == "counts":
        np.testing.assert_array_equal(result[0].counts, 10.0)
        assert all(group.deadtime.status.value == "skipped" for group in result)


def test_count_masks_keep_disjoint_raw_windows_and_complete_merged_bins():
    run = scoped_run("prefix")
    run.grouping.update(first_good_bin=0, last_good_bin=6, bunching_factor=2)
    run.histograms[0].counts[0] = 10
    run.histograms[0].counts[2:4] = 100
    run.histograms[0].counts[6] = 100
    dataset = MuonDataset(time=np.arange(64.0), asymmetry=np.zeros(64), error=np.ones(64), run=run)
    group = build_count_groups(dataset, [1], lifetime_corrected=False, exclude=(2, 3))[0]
    np.testing.assert_array_equal(group.time, [0.5, 4.5])
    np.testing.assert_allclose(group.counts, 2 * 10 / 0.9)
    assert group.deadtime.to_dict()["effective"]["scope"]["raw_bin_ranges"] == {
        "0": [[0, 1], [4, 5]]
    }
    with pytest.raises(ValueError, match="occupancy"):
        build_count_groups(dataset, [1], lifetime_corrected=False)


def test_group_range_background_keeps_exact_dependency_gap():
    run = scoped_run("prefix")
    run.metadata["facility"] = "PSI"
    run.grouping.update(
        first_good_bin=8,
        last_good_bin=15,
        background_correction=True,
        background_mode="range",
        background_ranges={1: [2, 3]},
    )
    run.histograms[0].counts[0] = 10
    run.histograms[0].counts[5] = 100
    result = consume(run, "signal")
    np.testing.assert_allclose(result.asymmetry, 0, atol=1e-14)
    assert result.metadata["deadtime_outcome"]["effective"]["scope"]["raw_bin_ranges"] == {
        "0": [[2, 3], [8, 15]]
    }
    run.histograms[0].counts[2] = 100
    with pytest.raises(ValueError, match="occupancy"):
        consume(run, "signal")


def test_precompute_cohort_and_changed_inputs_cannot_leak():
    from asymmetry.core.fourier.spectrum import precompute_group_fourier_inputs

    run = scoped_run("missing")
    prepared, t0 = precompute_group_fourier_inputs(run, [1])
    result = build_group_signal_dataset(run, 1, prepared=prepared, reference_t0_bin=t0)
    assert result.metadata["deadtime_outcome"]["status"] == "applied"
    with pytest.raises(ValueError, match="scope"):
        build_group_signal_dataset(run, 2, prepared=prepared)
    with pytest.raises(ValueError, match="cohort"):
        compute_average_group_spectrum(
            run, GroupSpectrumConfig(selected_group_ids=[1, 3]), prepared=prepared
        )
    run.histograms[0].counts[2] += 1
    with pytest.raises(ValueError, match="source observations"):
        build_group_signal_dataset(run, 1, prepared=prepared)


def test_shared_detector_is_corrected_once_and_retains_supplied_variance(monkeypatch):
    from asymmetry.core.fourier.group_preparation import prepare_group_counts
    from asymmetry.core.transform import scoped_counts

    run = scoped_run("missing")
    run.grouping["groups"][2] = [1]
    run.histograms[0].count_variance = np.full(64, 7.0)
    inverse = scoped_counts.apply_deadtime_correction
    calls = []

    def recording_inverse(*args, **kwargs):
        calls.append(kwargs["raw_bin_indices"].copy())
        return inverse(*args, **kwargs)

    monkeypatch.setattr(scoped_counts, "apply_deadtime_correction", recording_inverse)
    prepared = prepare_group_counts(run, [1, 2])
    results = [
        build_group_signal_dataset(
            run, gid, prepared=prepared, center_signal=False, apply_lifetime_correction=False
        )
        for gid in (1, 2)
    ]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.arange(64))
    for result in results:
        np.testing.assert_allclose(result.asymmetry, 10 / 0.99)
        np.testing.assert_allclose(result.error, np.sqrt(7))


def test_maxent_time_limits_skip_prefix_but_interior_exclusion_keeps_baseline_input():
    run = scoped_run("prefix")
    run.grouping["first_good_bin"] = 0
    config = MaxEntConfig(
        auto_steer=False,
        auto_phase_seed=False,
        auto_window=False,
        selected_group_ids=[1],
        n_spectrum_points=16,
        f_max_mhz=0.4,
        fit_phases=False,
        t_min_us=1,
        t_max_us=15,
    )
    result = build_maxent_input(run, config)
    assert result.metadata["deadtime_outcome"]["effective"]["scope"]["raw_bin_ranges"] == {
        "0": [[1, 15]]
    }
    run.histograms[0].counts[5] = 100
    config.exclude_t_min_us, config.exclude_t_max_us = 4, 6
    with pytest.raises(ValueError, match="occupancy"):
        build_maxent_input(run, config)


def test_maxent_resume_identity_uses_actual_observations_and_correction_scope():
    from asymmetry.core.maxent.state_identity import state_signature

    run = scoped_run("missing")
    config = MaxEntConfig(
        auto_steer=False,
        auto_phase_seed=False,
        auto_window=False,
        selected_group_ids=[1],
        n_spectrum_points=16,
        f_max_mhz=0.4,
    )
    before = build_maxent_input(run, config)
    signature = state_signature(before, config)
    assert signature == state_signature(
        build_maxent_input(run, MaxEntConfig.from_dict(config.to_dict())), config
    )
    run.histograms[0].counts[4] *= 2
    assert signature != state_signature(build_maxent_input(run, config), config)


def test_maxent_accepts_matching_explicit_precompute():
    from asymmetry.core.fourier.spectrum import precompute_group_fourier_inputs

    run = scoped_run("missing")
    config = MaxEntConfig(
        auto_steer=False,
        auto_phase_seed=False,
        auto_window=False,
        selected_group_ids=[1],
        n_spectrum_points=16,
        f_max_mhz=0.4,
    )
    prepared, t0 = precompute_group_fourier_inputs(run, [1])
    actual = build_maxent_input(run, config, prepared=prepared, reference_t0_bin=t0)
    expected = build_maxent_input(run, config)
    np.testing.assert_array_equal(actual.groups[0].signal, expected.groups[0].signal)


def test_changed_actual_maxent_mask_invalidates_state():
    from dataclasses import replace

    from asymmetry.core.maxent import initialize_state, run_cycles

    config = MaxEntConfig(
        auto_steer=False,
        auto_phase_seed=False,
        auto_window=False,
        selected_group_ids=[1],
        n_spectrum_points=16,
        f_max_mhz=0.4,
    )
    prepared = build_maxent_input(scoped_run("missing"), config)
    state = initialize_state(prepared, config)
    mask = prepared.groups[0].mask.copy()
    mask[3] = False
    changed = replace(prepared, groups=(replace(prepared.groups[0], mask=mask),))
    with pytest.raises(ValueError, match="incompatible"):
        run_cycles(changed, config, state=state, cycles=1)


def test_fb_fft_preserves_required_inverse_failure():
    run = scoped_run("unused")
    run.histograms[0].counts[1] = 1000
    with pytest.raises(ValueError, match="occupancy"):
        consume(run, "fb_fft")


def test_fb_fft_uses_canonical_corrected_input_values():
    from asymmetry.core.fourier.fb_input import build_fb_signal_dataset

    run = scoped_run("missing")
    dataset = build_fb_signal_dataset(run)
    forward, backward = 10 / 0.99, 5 / 0.995
    np.testing.assert_allclose(dataset.asymmetry, 100 * (forward - backward) / (forward + backward))
    assert dataset.metadata["correction_outcomes"]["red"]["deadtime"]["status"] == "applied"


@pytest.mark.parametrize("mode", ["red", "green", "green_minus_red", "green_plus_red"])
def test_fb_fft_held_period_inputs_preserve_atomic_scope(mode):
    from asymmetry.core.fourier.fb_input import build_fb_signal_dataset

    run = scoped_run("missing")
    periods = [[Histogram(h.counts.copy(), h.bin_width) for h in run.histograms] for _ in range(2)]
    run.grouping.update(
        period_mode=mode,
        period_histograms=periods,
        period_dead_time_us=[[0.1, 0.1, None], [0.1, None, None]],
        period_good_frames=[100.0, 200.0],
    )
    if mode == "red":
        dataset = build_fb_signal_dataset(run)
        assert dataset.metadata["correction_outcomes"]["red"]["deadtime"]["status"] == "applied"
    else:
        with pytest.warns(UserWarning, match="deadtime"):
            dataset = build_fb_signal_dataset(run)
        assert all(
            outcomes["deadtime"]["status"] == "skipped"
            for outcomes in dataset.metadata["correction_outcomes"].values()
        )
        if mode == "green_minus_red":
            np.testing.assert_allclose(dataset.asymmetry, 0.0)


def test_unselected_timing_metadata_preserves_shared_grid_without_count_dependency():
    from asymmetry.core.fourier.group_preparation import prepare_group_counts

    run = scoped_run("unused")
    run.histograms[2].t0_bin = 4
    prepared = prepare_group_counts(run, [1])
    assert prepared.reference_t0_bin == 4
    assert prepared.sample.domain.detector_indices == (0,)
    signal = build_group_signal_dataset(
        run, 1, prepared=prepared, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_allclose(signal.asymmetry, 10 / 0.99)
    assert signal.metadata["deadtime_outcome"]["status"] == "applied"


def test_empty_time_selection_has_empty_domain_and_reports_no_usable_input():
    from asymmetry.core.fourier.group_preparation import prepare_group_counts

    run = scoped_run("unused")
    prepared = prepare_group_counts(run, [1], time_range=(100.0, 110.0))
    assert prepared.sample.domain.observations == ()
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    with pytest.raises(ValueError, match="produced no usable bins"):
        build_count_groups(dataset, [1], t_min=100, t_max=110)
    config = MaxEntConfig(
        auto_steer=False,
        auto_phase_seed=False,
        auto_window=False,
        selected_group_ids=[1],
        t_min_us=100,
        t_max_us=110,
        n_spectrum_points=16,
        f_max_mhz=0.4,
    )
    with pytest.raises(ValueError, match="no valid grouped signals"):
        build_maxent_input(run, config)
