"""Multi-period subset → red/green mapping (WiMDA PeriodMappingUnit)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import (
    combine_mapped_periods,
    normalise_period_mapping,
    select_period_histograms,
    sum_period_histograms,
)
from tests.corpus import corpus_path, requires_corpus

PHOTO_MUSR_FILE = corpus_path("Semiconductors/Photo-muSR in silicon/Data/HIFI00103277.nxs")


def _period_dataset(period: int, level: float, frames: float, n_periods: int = 4) -> MuonDataset:
    rng = np.random.default_rng(period)
    histograms = [
        Histogram(counts=rng.poisson(level, size=16).astype(float), bin_width=0.016, t0_bin=2)
        for _ in range(3)
    ]
    run = Run(
        run_number=9000 + period,
        histograms=histograms,
        metadata={
            "run_number": 9000 + period,
            "source_run_number": 900,
            "period_number": period,
            "period_count": n_periods,
        },
        grouping={
            "groups": {1: [1, 2], 2: [3]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "good_frames": frames,
            "dead_time_us": [0.008, 0.008, 0.008],
        },
    )
    z = np.zeros(16)
    return MuonDataset(
        time=z.copy(), asymmetry=z.copy(), error=z.copy(), metadata=dict(run.metadata), run=run
    )


def _four_periods() -> list[MuonDataset]:
    return [
        _period_dataset(1, 100.0, 1000.0),
        _period_dataset(2, 200.0, 2000.0),
        _period_dataset(3, 300.0, 3000.0),
        _period_dataset(4, 400.0, 4000.0),
    ]


# --- mapping validation ----------------------------------------------------------


def test_normalise_period_mapping_accepts_string_keys():
    assert normalise_period_mapping({"1": "red", "2": "GREEN", 3: "ignore"}, 3) == {
        1: "red",
        2: "green",
        3: "ignore",
    }


@pytest.mark.parametrize(
    "mapping, n, message",
    [
        ({}, 2, "non-empty"),
        ({1: "blue"}, 2, "not one of"),
        ({5: "red"}, 2, "outside"),
        ({1: "green"}, 2, "at least one period to red"),
        ({"x": "red"}, 2, "not a period number"),
    ],
)
def test_normalise_period_mapping_rejects(mapping, n, message):
    with pytest.raises(ValueError, match=message):
        normalise_period_mapping(mapping, n)


# --- count-level summation --------------------------------------------------------


def test_sum_period_histograms_is_exact_count_addition():
    periods = _four_periods()
    summed = sum_period_histograms([periods[0].run.histograms, periods[2].run.histograms])
    for det in range(3):
        np.testing.assert_array_equal(
            summed[det].counts,
            periods[0].run.histograms[det].counts + periods[2].run.histograms[det].counts,
        )
        assert summed[det].t0_bin == periods[0].run.histograms[det].t0_bin


def test_combine_mapped_periods_sums_counts_and_frames():
    mapped = combine_mapped_periods(_four_periods(), {1: "red", 3: "red", 2: "green", 4: "ignore"})
    grouping = mapped.run.grouping
    assert len(grouping["period_histograms"]) == 2
    periods = _four_periods()
    np.testing.assert_array_equal(
        grouping["period_histograms"][0][0].counts,
        periods[0].run.histograms[0].counts + periods[2].run.histograms[0].counts,
    )
    np.testing.assert_array_equal(
        grouping["period_histograms"][1][0].counts, periods[1].run.histograms[0].counts
    )
    assert grouping["period_good_frames"] == [4000.0, 2000.0]
    assert grouping["good_frames"] == 4000.0
    assert grouping["period_mapping"] == {"1": "red", "2": "green", "3": "red", "4": "ignore"}
    assert mapped.run_number == 900
    assert mapped.metadata["period_count"] == 2
    # Equal per-period deadtimes carry through unchanged.
    assert grouping["dead_time_us"] == [0.008, 0.008, 0.008]


def test_combine_mapped_periods_red_only_gives_single_set():
    mapped = combine_mapped_periods(
        _four_periods(), {1: "red", 2: "ignore", 3: "ignore", 4: "ignore"}
    )
    grouping = mapped.run.grouping
    assert len(grouping["period_histograms"]) == 1
    assert grouping["period_good_frames"] == [1000.0]


def test_trivial_mapping_matches_existing_two_period_structure():
    """{1→red, 2→green} must reproduce the loader's combined-run reduction
    path bit-for-bit through select_period_histograms."""
    periods = _four_periods()[:2]
    for ds in periods:
        ds.metadata["period_count"] = 2
        ds.run.metadata["period_count"] = 2
    mapped = combine_mapped_periods(periods, {1: "red", 2: "green"})
    grouping = mapped.run.grouping

    for index, source in ((0, periods[0]), (1, periods[1])):
        selected, effective = select_period_histograms(mapped.run.histograms, grouping, index)
        for det in range(3):
            np.testing.assert_array_equal(selected[det].counts, source.run.histograms[det].counts)
        assert effective["good_frames"] == source.run.grouping["good_frames"]


def test_mapped_dataset_reduces_through_standard_grouping():
    """The mapped run reduces exactly like the sum of its red periods."""
    from asymmetry.core.transform import group_forward_backward

    periods = _four_periods()
    mapped = combine_mapped_periods(periods, {1: "red", 2: "red", 3: "ignore", 4: "ignore"})
    fb = group_forward_backward(mapped.run.histograms, mapped.run.grouping)
    expected_forward = (
        periods[0].run.histograms[0].counts
        + periods[0].run.histograms[1].counts
        + periods[1].run.histograms[0].counts
        + periods[1].run.histograms[1].counts
    )
    np.testing.assert_array_equal(fb.forward, expected_forward)


def test_mismatched_deadtimes_keep_original_tables_without_effective_average():
    periods = _four_periods()[:2]
    periods[0].run.grouping["dead_time_us"] = [0.010, 0.010, 0.010]
    periods[1].run.grouping["dead_time_us"] = [0.020, 0.020, 0.020]
    mapped = combine_mapped_periods(periods, {1: "red", 2: "red"})
    assert mapped.run.grouping["dead_time_us"] == [None] * 3
    assert [r.grouping["dead_time_us"] for r in mapped.run.grouping["period_inputs"]] == [
        [0.010] * 3,
        [0.020] * 3,
    ]


# --- corpus -----------------------------------------------------------------------


@requires_corpus(PHOTO_MUSR_FILE)
def test_photo_musr_trivial_mapping_matches_loader_combination():
    """On the validated photo-µSR silicon run, mapping {1→red, 2→green} of the
    per-period datasets reproduces the loader's combined two-period dataset."""
    from asymmetry.core.io import load
    from asymmetry.core.io.periods import select_period

    combined = load(PHOTO_MUSR_FILE)
    assert isinstance(combined, MuonDataset)
    red = select_period(combined, "red")
    green = select_period(combined, "green")
    mapped = combine_mapped_periods([red, green], {1: "red", 2: "green"})
    grouping = mapped.run.grouping
    loader_grouping = combined.run.grouping
    for index in (0, 1):
        for det in range(len(loader_grouping["period_histograms"][index])):
            np.testing.assert_array_equal(
                grouping["period_histograms"][index][det].counts,
                loader_grouping["period_histograms"][index][det].counts,
            )
    assert grouping["period_good_frames"] == pytest.approx(loader_grouping["period_good_frames"])


def test_missing_contributing_period_prevents_a_combined_calibration():
    """Review fix: a period without a deadtime table must not shift the
    frame weights of the periods that have one."""
    periods = _four_periods()[:3]  # frames 1000, 2000, 3000
    periods[0].run.grouping["dead_time_us"] = [0.010, 0.010, 0.010]
    periods[1].run.grouping.pop("dead_time_us")  # middle period: no table
    periods[2].run.grouping["dead_time_us"] = [0.020, 0.020, 0.020]
    mapped = combine_mapped_periods(periods, {1: "red", 2: "red", 3: "red"})
    assert mapped.run.grouping["dead_time_us"] == [None] * 3


def test_ragged_contributing_tables_remain_unavailable():
    periods = _four_periods()[:2]
    periods[0].run.grouping["dead_time_us"] = [0.010, 0.010, 0.010]
    periods[1].run.grouping["dead_time_us"] = [0.020, 0.020]  # malformed: short
    mapped = combine_mapped_periods(periods, {1: "red", 2: "red"})
    table = mapped.run.grouping["dead_time_us"]
    assert table == [None] * 3


@pytest.mark.parametrize("conflicting_period", [0, 1])
def test_mapped_sets_keep_the_timing_source_of_their_histograms(conflicting_period):
    periods = _four_periods()
    for index, dataset in enumerate(periods):
        dataset.run.grouping["t0_source"] = "conflict" if index == conflicting_period else "file"
    mapped = combine_mapped_periods(periods, {1: "red", 2: "green", 3: "red"})
    for index in range(2):
        _, grouping = select_period_histograms(mapped.run.histograms, mapped.run.grouping, index)
        assert grouping["t0_source"] == periods[index].run.grouping["t0_source"]


@pytest.mark.parametrize(
    "mode", ["manual", "estimate", "load", "calibrate", "count_fit", "maxent_fit", "value"]
)
def test_mapping_keeps_authored_and_acquired_deadtime_tables_separate(mode):
    periods = _four_periods()[:2]
    for dataset in periods:
        dataset.run.grouping.update(
            deadtime_mode=mode,
            deadtime_method=mode,
            dead_time_us=[0.01, -0.02, 0.03],
            deadtime_file_us=[0.004] * 3,
        )
    mapped = combine_mapped_periods(periods, {1: "red", 2: "green"})
    assert mapped.run.grouping["dead_time_us"] == [0.01, -0.02, 0.03]
    assert mapped.run.grouping["deadtime_file_us"] == [0.004] * 3
    for index in (0, 1):
        _, grouping = select_period_histograms(mapped.run.histograms, mapped.run.grouping, index)
        assert grouping["dead_time_us"] == [0.01, -0.02, 0.03]
        assert grouping["deadtime_file_us"] == [0.004] * 3


@pytest.mark.parametrize("counts", [(10, 30), (-10, 50)])
def test_mapped_supplied_variance_reaches_normal_preparation(counts):
    from dataclasses import replace

    from asymmetry.core.transform.preparation import prepare_histogram_asymmetry

    periods = _four_periods()[:2]
    for dataset, count, variance in zip(periods, counts, [25, 49], strict=True):
        dataset.run.histograms = [
            replace(h, counts=np.full(16, count), count_variance=np.full(16, variance))
            for h in dataset.run.histograms
        ]
    mapped = combine_mapped_periods(periods, {1: "red", 2: "red"})
    assert mapped.run.histograms[0].count_variance is not None
    np.testing.assert_array_equal(mapped.run.histograms[0].count_variance, 74)
    result = prepare_histogram_asymmetry(mapped.run, merge=2, facility=None, reference=None)
    # Two forward detectors, one backward: F=160, B=80; VF=296, VB=148.
    np.testing.assert_allclose(result.asymmetry, 100 / 3)
    np.testing.assert_allclose(result.error, 200 * np.sqrt(80**2 * 296 + 160**2 * 148) / 240**2)


def test_red_only_mapping_prepares_and_requires_real_green():
    from dataclasses import replace

    from asymmetry.core.io.periods import select_period
    from asymmetry.core.transform.preparation import prepare_histogram_asymmetry

    mapped = combine_mapped_periods(_four_periods(), {1: "red", 3: "red", 4: "ignore"})
    reduced = prepare_histogram_asymmetry(mapped.run, merge=1, facility=None, reference=None)
    assert reduced.time.size
    selected = select_period(mapped, 1)
    np.testing.assert_allclose(selected.asymmetry, reduced.asymmetry)
    for mode in ["green", "green_plus_red", "green_minus_red"]:
        run = replace(mapped.run, grouping=mapped.run.grouping | {"period_mode": mode})
        with pytest.raises(ValueError, match="green|two held"):
            prepare_histogram_asymmetry(run, merge=1, facility=None, reference=None)


def test_mixed_raw_and_retained_variance_add_independently():
    periods = [
        [Histogram(np.array([10.0, 20.0]), 0.1)],
        [Histogram(np.array([-2.0, 30.0]), 0.1, count_variance=np.array([25.0, 49.0]))],
    ]
    summed = sum_period_histograms(periods)[0]
    np.testing.assert_array_equal(summed.counts, [8, 50])
    np.testing.assert_array_equal(summed.count_variance, [35, 69])
    raw = sum_period_histograms([periods[0], periods[0]])[0]
    assert raw.count_variance is None
    np.testing.assert_array_equal(raw.counts, [20, 40])


def test_mapping_retains_ignored_inputs_and_snapshot_worker_ownership():
    import pickle
    from dataclasses import replace

    from asymmetry.core.transform.preparation import histogram_observation, prepare_wizard_dataset

    periods = _four_periods()
    periods[2].run.histograms = [
        replace(h, count_variance=np.full(16, 49.0)) for h in periods[2].run.histograms
    ]
    before = [h.counts.copy() for ds in periods for h in ds.run.histograms]
    mapped = combine_mapped_periods(periods, {1: "red", 3: "red", 2: "green", 4: "ignore"})
    inputs = mapped.run.grouping["period_inputs"]
    assert len(inputs) == 4
    assert [r.metadata["period_number"] for r in inputs] == [1, 2, 3, 4]
    assert inputs[2].histograms[0].count_variance[0] == 49
    assert inputs[3].grouping["good_frames"] == 4000
    snapshot = histogram_observation(mapped.run)
    captured = snapshot.source.grouping["period_inputs"]
    assert not captured[3].histograms[0].counts.flags.writeable
    inputs[3].histograms[0].counts[0] += 1
    assert captured[3].histograms[0].counts[0] != inputs[3].histograms[0].counts[0]
    assert histogram_observation(mapped.run).identity != snapshot.identity
    inputs[3].histograms[0].counts[0] -= 1
    worker = pickle.loads(pickle.dumps(prepare_wizard_dataset(mapped, 2).fit_record()))
    assert "period_inputs" not in worker.run.grouping
    assert not worker.run.histograms
    for old, hist in zip(before, (h for ds in periods for h in ds.run.histograms), strict=True):
        np.testing.assert_array_equal(old, hist.counts)
    # Authored membership changes only the output; ignored observations remain usable.
    remapped = combine_mapped_periods(mapped, {4: "red", 1: "green"})
    np.testing.assert_array_equal(
        remapped.run.histograms[0].counts, periods[3].run.histograms[0].counts
    )
    np.testing.assert_array_equal(
        mapped.run.grouping["period_histograms"][0][0].counts,
        periods[0].run.histograms[0].counts + periods[2].run.histograms[0].counts,
    )


def test_signed_unknown_variance_cannot_become_poisson_by_pooling():
    with pytest.raises(ValueError, match="supplied variance"):
        sum_period_histograms(
            [[Histogram(np.array([-10.0]), 0.1)], [Histogram(np.array([50.0]), 0.1)]]
        )


def test_mapping_does_not_drop_unmatched_detectors():
    with pytest.raises(ValueError, match="corresponding detector"):
        sum_period_histograms(
            [[Histogram(np.ones(3), 0.1)], [Histogram(np.ones(3), 0.1), Histogram(np.ones(3), 0.1)]]
        )


def test_red_only_workdir_preserves_recorded_results_and_replay_changes_identity(tmp_path):
    from dataclasses import replace

    from asymmetry.core.io.reference_selection import resolve_loaded_reference
    from asymmetry.core.transform.preparation import prepare_histogram_asymmetry
    from asymmetry.core.workflow.reduction import ReductionSettings
    from asymmetry.core.workflow.workdir import ReducedEntry, WorkDir, reduction_digest

    periods = _four_periods()
    mapped = combine_mapped_periods(periods, {1: "red", 3: "red"})
    reduced = prepare_histogram_asymmetry(mapped.run, merge=2, facility=None, reference=None)
    dataset = replace(
        mapped,
        time=reduced.time,
        asymmetry=reduced.asymmetry,
        error=reduced.error,
        ratio_validity=reduced.ratio_validity,
    )
    settings = ReductionSettings()
    source = tmp_path / "source.nxs"
    source.write_bytes(b"synthetic identity")
    digest = reduction_digest(
        source_files=[source], grouping={"period_mapping": {1: "red", 3: "red"}}, settings=settings
    )
    entry = ReducedEntry(
        900,
        digest,
        str(source),
        dataset.n_points,
        settings,
        {"run_number": 900},
        1.0,
        "off",
        1,
        2,
        [],
    )
    workdir = WorkDir(tmp_path / "work")
    workdir.write_reduced(dataset, entry)
    restored = workdir.reduced(900)
    np.testing.assert_array_equal(restored.error, dataset.error)
    np.testing.assert_array_equal(restored.asymmetry, dataset.asymmetry)
    np.testing.assert_array_equal(restored.ratio_validity.reasons, dataset.ratio_validity.reasons)
    assert restored.run is None
    assert workdir.is_current(900, digest)
    # A pooled set cannot masquerade as one independent acquired reference.
    with pytest.raises(ValueError, match="mapped reference"):
        resolve_loaded_reference([mapped.run], {"run_number": 900, "period_number": 1})
    selected = resolve_loaded_reference([periods[2].run], {"run_number": 900, "period_number": 3})
    assert selected is periods[2].run


def test_mapped_and_selected_quicklook_replay_uses_its_actual_uncorrected_settings():
    from asymmetry.core.io.periods import select_period
    from asymmetry.core.transform.preparation import prepare_wizard_dataset

    periods = _four_periods()
    for ds in periods:
        ds.run.grouping.update(
            alpha=0.7,
            beta=1.2,
            bunching_factor=3,
            background_correction=True,
            background_mode="fixed",
            background_fixed_values=[2, 3, 4],
            good_window_policy={"mode": "manual", "first_offset_bins": 2, "last_offset_bins": 5},
        )
    mapped = combine_mapped_periods(periods, {1: "red", 3: "red"})
    for dataset in [mapped, select_period(mapped, 1)]:
        output = prepare_wizard_dataset(dataset, 2)
        h = dataset.run.histograms
        f = (h[0].counts + h[1].counts).reshape(-1, 2).sum(axis=1)
        b = h[2].counts.reshape(-1, 2).sum(axis=1)
        np.testing.assert_allclose(output.asymmetry, 100 * (f - 0.7 * b) / (f + 0.7 * b))
        raw_f, raw_b = h[0].counts + h[1].counts, h[2].counts
        np.testing.assert_allclose(
            dataset.asymmetry, 100 * (raw_f - 0.7 * raw_b) / (raw_f + 0.7 * raw_b)
        )
        assert output.n_points == 8
        assert dataset.run.grouping["alpha"] == 0.7


def test_held_mapped_inspection_borrows_existing_buffers_without_pooling(monkeypatch):
    import asymmetry.core.io.periods as period_io
    from asymmetry.core.io.reference_selection import select_reference_run

    mapped = combine_mapped_periods(_four_periods()[:2], {1: "red", 2: "green"})

    def unexpected_sum(*args, **kwargs):
        raise AssertionError("Inspection must not pool raw acquisition arrays")

    monkeypatch.setattr(period_io, "sum_period_histograms", unexpected_sum)
    selected = period_io.acquired_period_histograms(mapped.run.grouping, 1)
    assert selected is mapped.run.grouping["period_histograms"][1]
    reference = select_reference_run(mapped.run, {"run_number": 900, "period_number": 2})
    assert reference.histograms is selected


def test_mapping_polls_cancellation_before_capturing_ignored_inputs():
    calls = []

    def cancel():
        calls.append(True)
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        combine_mapped_periods(_four_periods(), {1: "red", 4: "ignore"}, cancel_check=cancel)
    assert calls == [True]


@pytest.mark.parametrize("count", [1, 2])
def test_public_period_sum_cancels_before_copying_or_pooling(count):
    def cancel():
        raise RuntimeError("cancelled before counts")

    with pytest.raises(RuntimeError, match="cancelled before counts"):
        sum_period_histograms([[Histogram(np.ones(4), 0.1)]] * count, cancel_check=cancel)
