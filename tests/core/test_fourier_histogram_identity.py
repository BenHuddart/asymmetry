"""Mutable acquisition grids must invalidate spectral products without count scans."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.group_preparation import prepare_group_counts
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.spectrum import fourier_grouping_digest


def range_run():
    return Run(
        histograms=[
            Histogram(np.arange(100.0) + level, 0.1, t0)
            for level, t0 in ((10, 2), (20, 5), (30, 8))
        ],
        grouping={
            "groups": {1: [1], 2: [2], 3: [3]},
            "forward_group": 1,
            "backward_group": 2,
            "background_correction": True,
            "background_mode": "range",
            "background_range": [10, 20],
        },
    )


@pytest.mark.parametrize("signal_source", ["grouped_average", "fb_asymmetry"])
@pytest.mark.parametrize("field", ["t0_bin", "t0_time_us", "bin_width", "length"])
def test_histogram_grid_mutations_change_digest_without_grouping_edits(field, signal_source):
    run = range_run()
    grouping = deepcopy(run.grouping)
    before = fourier_grouping_digest(run, signal_source=signal_source)
    histogram = run.histograms[0]
    if field == "length":
        histogram.counts = histogram.counts[:-1]
    else:
        setattr(histogram, field, {"t0_bin": 3, "t0_time_us": 0.27, "bin_width": 0.2}[field])
    assert fourier_grouping_digest(run, signal_source=signal_source) != before
    assert run.grouping == grouping


def test_integer_timing_edit_changes_actual_range_observations_and_identity():
    run = range_run()
    original = prepare_group_counts(run, [1])
    digest = fourier_grouping_digest(run)
    assert original.backgrounds[1].outcome.effective["values"][0] == pytest.approx(22.0)
    run.histograms[0].t0_bin = 3
    changed = prepare_group_counts(run, [1])
    assert changed.backgrounds[1].outcome.effective["values"][0] == pytest.approx(23.0)
    assert fourier_grouping_digest(run) != digest
    with pytest.raises(ValueError, match="do not match"):
        original.require_request(run, 1, False, None, None)


def test_exact_timing_edit_changes_signal_time_and_identity():
    run = range_run()
    before = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    digest = fourier_grouping_digest(run)
    run.histograms[0].t0_time_us = 0.27
    after = build_group_signal_dataset(run, 1, center_signal=False, apply_lifetime_correction=False)
    np.testing.assert_allclose(after.time, before.time - 0.02, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(after.asymmetry, before.asymmetry)
    assert fourier_grouping_digest(run) != digest


def test_native_width_edit_changes_automatic_period_selection_and_identity():
    raw = np.arange(200.0) + 10
    run = Run(
        histograms=[Histogram(raw, 0.005, 100), Histogram(raw + 10, 0.005, 100)],
        metadata={"facility": "PSI"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "background_correction": True,
            "background_mode": "range",
        },
    )
    original = prepare_group_counts(run, [1])
    digest = fourier_grouping_digest(run)
    assert original.backgrounds[1].outcome.effective["ranges"] == ((10, 57), (10, 57))
    assert original.backgrounds[1].outcome.effective["values"][0] == pytest.approx(
        raw[10:58].mean()
    )
    for histogram in run.histograms:
        histogram.bin_width = 0.01
    changed = prepare_group_counts(run, [1])
    assert changed.backgrounds[1].outcome.effective["ranges"] == ((10, 59), (10, 59))
    assert changed.backgrounds[1].outcome.effective["values"][0] == pytest.approx(raw[10:60].mean())
    assert fourier_grouping_digest(run) != digest


def test_geometry_digest_does_not_load_or_iterate_count_values():
    class MetadataOnlyCounts:
        def __len__(self):
            return 100

        def __array__(self, *args, **kwargs):
            raise AssertionError("A grid digest must not load observations")

        def __iter__(self):
            raise AssertionError("A grid digest must not scan observations")

    run = range_run()
    run.histograms[0].counts = MetadataOnlyCounts()
    assert fourier_grouping_digest(run)
