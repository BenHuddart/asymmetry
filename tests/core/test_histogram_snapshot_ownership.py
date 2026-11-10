"""Captured acquisitions remain stable when callers write through source aliases."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.transform.preparation import (
    histogram_observation,
    prepare_wizard_dataset,
    wizard_observation_identity,
)
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.core.test_binning_entry_points import _run
from tests.core.test_wizard_preparation import observed


def test_workflow_capture_survives_source_edit_and_fresh_capture_uses_edit():
    run = _run([90, 1], [10, 9])
    dataset = reduce_run(run, ReductionSettings())
    identity = wizard_observation_identity(dataset)
    alias = run.histograms[0].counts.view()
    alias[0] = 10
    np.testing.assert_allclose(prepare_wizard_dataset(dataset, 2).asymmetry, [100 * 72 / 110])
    np.testing.assert_allclose(dataset.asymmetry, [80, -80])
    assert wizard_observation_identity(dataset) == identity
    fresh = reduce_run(run, ReductionSettings())
    np.testing.assert_allclose(prepare_wizard_dataset(fresh, 2).asymmetry, [-100 * 8 / 30])
    assert wizard_observation_identity(fresh) != identity
    run.grouping["alpha"] = 2
    np.testing.assert_allclose(prepare_wizard_dataset(dataset, 2).asymmetry, [100 * 72 / 110])
    assert wizard_observation_identity(dataset) == identity


@pytest.mark.parametrize("role", ["sample", "period", "reference"])
@pytest.mark.parametrize("quantity", ["counts", "count_variance"])
def test_alias_edits_preserve_cropped_repeated_merges(role, quantity):
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest

    run = _run(np.arange(200.0, 224), np.arange(150.0, 174))
    held = _run(np.arange(90.0, 114), np.arange(70.0, 94))
    reference_run = _run(np.arange(10.0, 36), np.arange(20.0, 46))
    for source in (run, held, reference_run):
        source.histograms = [
            replace(h, count_variance=np.full(h.n_bins, v, dtype=float))
            for h, v in zip(source.histograms, [5, 7])
        ]
    run.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        reference_exposure=ReferenceExposure(200, "file").to_dict(),
    )
    if role == "period":
        run.grouping.update(
            period_mode="green_minus_red", period_histograms=[run.histograms, held.histograms]
        )
    reference = BackgroundReference(
        [replace(h, t0_bin=1) for h in reference_run.histograms],
        ReferenceScaleRequest().resolve(
            ReferenceExposure(200, "file"), ReferenceExposure(100, "file")
        ),
        run_number=2,
    )
    source_histograms = {
        "sample": run.histograms,
        "period": held.histograms,
        "reference": reference.histograms,
    }[role]
    array = getattr(source_histograms[0], quantity)
    array.setflags(write=True)
    alias = array.view()
    dataset = observed(run, reference=reference)
    cropped = dataset.time_range(dataset.time[2], dataset.time[21])
    initial = prepare_wizard_dataset(prepare_wizard_dataset(cropped, 2), 2)
    identity = wizard_observation_identity(cropped)
    alias[:] *= 2
    replay = prepare_wizard_dataset(prepare_wizard_dataset(cropped, 2), 2)
    for name in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(replay, name), getattr(initial, name))
    assert wizard_observation_identity(cropped) == identity
    fresh = observed(run, reference=reference)
    assert fresh.histogram_observation.identity != dataset.histogram_observation.identity
    changed = prepare_wizard_dataset(
        prepare_wizard_dataset(fresh.time_range(dataset.time[2], dataset.time[21]), 2), 2
    )
    assert not np.array_equal(changed.error, initial.error)
    source_histograms[0].t0_time_us = 0.2
    source_histograms[0].bin_width = 0.2
    run.grouping["alpha"] = 2
    stable = prepare_wizard_dataset(prepare_wizard_dataset(cropped, 2), 2)
    for name in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(stable, name), getattr(initial, name))
    assert wizard_observation_identity(cropped) == identity
    assert stable.histogram_observation.source is dataset.histogram_observation.source
    assert stable.histogram_observation.reference is dataset.histogram_observation.reference
    record = stable.fit_record()
    assert record.histogram_observation is None
    assert record.run.histograms == []
    assert "period_histograms" not in record.run.grouping


@pytest.mark.parametrize("strided", [False, True])
def test_capture_owns_one_snapshot_per_array_across_roles(strided):
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest

    values = np.arange(1.0, 33)
    counts = values[::2] if strided else values
    run = _run(counts, counts)
    for h in run.histograms:
        h.count_variance = counts
    run.grouping["period_histograms"] = [run.histograms, run.histograms]
    scaling = ReferenceScaleRequest().resolve(
        ReferenceExposure(100, "file"), ReferenceExposure(100, "file")
    )
    observation = histogram_observation(run, reference=BackgroundReference(run.histograms, scaling))
    snapshot = observation.source.histograms[0].counts
    for histograms in (
        observation.source.histograms,
        *observation.source.grouping["period_histograms"],
        observation.reference.histograms,
    ):
        for h in histograms:
            assert h.counts is snapshot
            assert h.count_variance is snapshot
    assert snapshot.flags.owndata
    assert not snapshot.flags.writeable
    assert not np.shares_memory(snapshot, values)
    assert counts.flags.writeable
    values[:] = 0
    np.testing.assert_array_equal(
        snapshot, np.arange(1.0, 33)[::2] if strided else np.arange(1.0, 33)
    )


def test_identity_hashes_retained_values_when_source_changes_after_copy():
    run = _run([90, 1], [10, 9])
    original = histogram_observation(run)
    checks = 0

    def edit_after_copy():
        nonlocal checks
        checks += 1
        if checks == 3:
            run.histograms[0].counts[0] = 10

    captured = histogram_observation(run, cancel_check=edit_after_copy)
    np.testing.assert_array_equal(captured.source.histograms[0].counts, [90, 1])
    assert captured.identity == original.identity
    assert histogram_observation(run).identity != original.identity


@pytest.mark.parametrize("stop", [1, 2, 4])
def test_capture_cancels_before_or_during_large_array_hashing(stop):
    run = _run(np.full(300_000, 90.0), np.full(300_000, 10.0))
    counts = run.histograms[0].counts
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == stop:
            raise InterruptedError("cancelled capture")

    with pytest.raises(InterruptedError, match="cancelled capture"):
        histogram_observation(run, cancel_check=cancel)
    assert checks == stop
    assert counts.flags.writeable
    counts[0] = 1
    assert run.histograms[0].counts[0] == 1
