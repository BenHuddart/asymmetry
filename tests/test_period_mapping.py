"""Multi-period subset → red/green mapping (WiMDA PeriodMappingUnit)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import (
    combine_mapped_periods,
    normalise_period_mapping,
    select_period_histograms,
    sum_period_histograms,
)

PHOTO_MUSR_FILE = os.path.expanduser(
    "~/Documents/WiMDA muon school/Semiconductors/Photo-muSR in silicon/Data_hdf5/HIFI00103277.nxs"
)


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


def test_mismatched_deadtimes_get_frame_weighted_mean():
    periods = _four_periods()[:2]
    periods[0].run.grouping["dead_time_us"] = [0.010, 0.010, 0.010]
    periods[1].run.grouping["dead_time_us"] = [0.020, 0.020, 0.020]
    mapped = combine_mapped_periods(periods, {1: "red", 2: "red"})
    # Frame weights 1000:2000 → (0.010·1000 + 0.020·2000)/3000.
    expected = (0.010 * 1000.0 + 0.020 * 2000.0) / 3000.0
    assert mapped.run.grouping["dead_time_us"] == pytest.approx([expected] * 3)


# --- corpus -----------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(PHOTO_MUSR_FILE), reason="photo-µSR corpus not available")
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
