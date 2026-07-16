"""Tests for the core period (red/green) selection API.

Covers ``asymmetry.core.io.periods`` and the ``load(period=...)`` kwarg:
correct period extraction, validation errors, label/scalar access, provenance
preservation, the green/red combination arithmetic, and (when a period-mode
NeXus file is available) a real round-trip plus GUI/core agreement.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io import load, period_count, period_labels, select_period
from asymmetry.core.io.periods import (
    GREEN_INDEX,
    RED_INDEX,
    combine_period_asymmetry,
    encode_period_run_number,
    resolve_period_index,
    select_period_histograms,
)
from asymmetry.core.utils.constants import PeriodMode

PHOTO_MUSR_FILE = os.path.expanduser(
    "~/Documents/WiMDA muon school/Semiconductors/Photo-muSR in silicon/Data_hdf5/HIFI00103277.nxs"
)


# --- fixtures ----------------------------------------------------------------


def _histograms(scale: float) -> list[Histogram]:
    """Two detector histograms with a recognisable per-period scale."""
    bins = np.arange(20, dtype=np.float64)
    return [
        Histogram(
            counts=scale * (100.0 - bins),
            bin_width=0.016,
            t0_bin=3,
            good_bin_start=3,
            good_bin_end=19,
        ),
        Histogram(
            counts=scale * (90.0 - 0.5 * bins),
            bin_width=0.016,
            t0_bin=3,
            good_bin_start=3,
            good_bin_end=19,
        ),
    ]


def _combined_two_period() -> MuonDataset:
    """Synthetic two-period dataset mirroring the NeXus loader's combined output."""
    red_hist = _histograms(1.0)
    green_hist = _histograms(2.0)

    red_time = np.linspace(0.0, 8.0, 17)
    red_asym = np.full(17, 5.0)
    red_err = np.full(17, 0.5)
    green_time = red_time.copy()
    green_asym = np.full(17, 12.0)
    green_err = np.full(17, 0.7)

    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 3,
        "last_good_bin": 19,
        "t0_bin": 3,
        "good_frames": 10.0,
        "dead_time_us": [0.01, 0.01],
        "period_histograms": [red_hist, green_hist],
        "period_reduced": [
            (red_time, red_asym, red_err),
            (green_time, green_asym, green_err),
        ],
        "period_good_frames": [10.0, 20.0],
        "period_dead_time_us": [[0.01, 0.01], [0.02, 0.02]],
        "period_mode": str(PeriodMode.RED),
    }
    metadata = {
        "run_number": 12345,
        "source_run_number": 12345,
        "run_label": "12345",
        "period_number": 1,
        "period_count": 2,
        "temperature": 291.0,
        "field": -100.0,
    }
    run = Run(
        run_number=12345,
        histograms=list(red_hist),
        metadata=dict(metadata),
        grouping=grouping,
        source_file="synthetic.nxs",
    )
    return MuonDataset(
        time=red_time.copy(),
        asymmetry=red_asym.copy(),
        error=red_err.copy(),
        metadata=dict(metadata),
        run=run,
    )


def _single_period() -> MuonDataset:
    run = Run(
        run_number=7,
        histograms=_histograms(1.0),
        metadata={"period_number": 1, "period_count": 1},
        grouping={},
    )
    return MuonDataset(
        time=np.linspace(0, 8, 17),
        asymmetry=np.zeros(17),
        error=np.ones(17),
        metadata={"period_number": 1, "period_count": 1},
        run=run,
    )


# --- resolve_period_index ----------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("red", RED_INDEX),
        ("Red", RED_INDEX),
        ("R", RED_INDEX),
        (1, RED_INDEX),
        ("1", RED_INDEX),
        (PeriodMode.RED, RED_INDEX),
        ("green", GREEN_INDEX),
        ("GREEN", GREEN_INDEX),
        ("g", GREEN_INDEX),
        (2, GREEN_INDEX),
        ("2", GREEN_INDEX),
        (PeriodMode.GREEN, GREEN_INDEX),
    ],
)
def test_resolve_period_index_two_period(selector, expected):
    assert resolve_period_index(selector, 2) == expected


def test_resolve_period_index_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_period_index(3, 2)
    with pytest.raises(ValueError, match="out of range"):
        resolve_period_index(0, 2)


def test_resolve_period_index_unknown_label():
    with pytest.raises(ValueError, match="Unknown period selector"):
        resolve_period_index("blue", 2)


def test_resolve_period_index_labels_rejected_for_many_periods():
    # red/green only make sense for a two-period run.
    with pytest.raises(ValueError, match="Unknown period selector"):
        resolve_period_index("red", 4)


def test_resolve_period_index_type_errors():
    with pytest.raises(TypeError):
        resolve_period_index(True, 2)  # bool is not a valid selector
    with pytest.raises(TypeError):
        resolve_period_index(1.5, 2)


# --- select_period on a combined two-period dataset --------------------------


def test_period_count_and_labels():
    combined = _combined_two_period()
    assert period_count(combined) == 2
    assert period_labels(combined) == ["red", "green"]


def test_select_period_extracts_correct_arrays():
    combined = _combined_two_period()
    red = select_period(combined, "red")
    green = select_period(combined, "green")

    assert np.allclose(red.asymmetry, 5.0)
    assert np.allclose(green.asymmetry, 12.0)
    assert np.allclose(green.error, 0.7)
    # default combined dataset shows the red (period 1) spectrum
    assert np.allclose(combined.asymmetry, red.asymmetry)
    # integer and PeriodMode selectors agree with labels
    assert np.allclose(select_period(combined, 2).asymmetry, green.asymmetry)
    assert np.allclose(select_period(combined, PeriodMode.GREEN).asymmetry, green.asymmetry)


def test_select_period_preserves_provenance():
    combined = _combined_two_period()
    green = select_period(combined, "green")

    assert green.metadata["period_number"] == 2
    assert green.metadata["period_count"] == 2
    assert green.run_label == "12345/2"
    assert green.metadata["period_label"] == "green"
    # temperature / field carried through
    assert green.run.temperature == 291.0
    assert green.run.field == -100.0
    # per-period good_frames / deadtime swapped in
    assert green.run.grouping["good_frames"] == 20.0
    assert green.run.grouping["dead_time_us"] == [0.02, 0.02]
    # t0 / good-bin window preserved on the histograms
    assert green.run.histograms[0].t0_bin == 3
    assert green.run.histograms[0].good_bin_end == 19
    # green histograms differ from red (distinct per-period scale)
    red = select_period(combined, "red")
    assert not np.allclose(green.run.histograms[0].counts, red.run.histograms[0].counts)
    # internal period bookkeeping stripped from the per-period grouping
    assert "period_histograms" not in green.run.grouping
    assert "period_reduced" not in green.run.grouping


def test_select_period_assigns_distinct_run_numbers():
    # Both periods share the source run's number, so a run-number-keyed data
    # browser would collapse them. select_period must hand each period a
    # distinct encoded key while preserving the true source run number.
    combined = _combined_two_period()
    red = select_period(combined, "red")
    green = select_period(combined, "green")

    assert red.run.run_number != green.run.run_number
    assert red.run.run_number == encode_period_run_number(12345, 1)
    assert green.run.run_number == encode_period_run_number(12345, 2)
    # the true source run is still recoverable, and the display label unchanged
    assert red.metadata["source_run_number"] == 12345
    assert green.metadata["source_run_number"] == 12345
    assert red.run_label == "12345/1"
    assert green.run_label == "12345/2"


def test_select_period_returns_copies():
    combined = _combined_two_period()
    green = select_period(combined, "green")
    green.asymmetry[:] = -1.0
    # mutating the result must not corrupt the parent's stored arrays
    again = select_period(combined, "green")
    assert np.allclose(again.asymmetry, 12.0)


# --- select_period on lists and single-period datasets -----------------------


def test_select_period_from_list():
    datasets = [_single_period() for _ in range(3)]
    for i, ds in enumerate(datasets):
        ds.metadata["period_number"] = i + 1
    assert select_period(datasets, 1) is datasets[0]
    assert select_period(datasets, 3) is datasets[2]
    with pytest.raises(ValueError, match="out of range"):
        select_period(datasets, 4)


def test_select_period_single_period():
    ds = _single_period()
    assert select_period(ds, 1) is ds
    with pytest.raises(ValueError, match="out of range"):
        select_period(ds, 2)


def test_select_period_bad_type():
    with pytest.raises(TypeError):
        select_period("not a dataset", 1)


# --- select_period_histograms (shared with the GUI) --------------------------


def test_select_period_histograms_picks_period_and_metadata():
    combined = _combined_two_period()
    grouping = combined.run.grouping
    red_hists, red_grouping = select_period_histograms(combined.run.histograms, grouping, RED_INDEX)
    green_hists, green_grouping = select_period_histograms(
        combined.run.histograms, grouping, GREEN_INDEX
    )

    assert np.allclose(red_hists[0].counts, grouping["period_histograms"][RED_INDEX][0].counts)
    assert np.allclose(green_hists[0].counts, grouping["period_histograms"][GREEN_INDEX][0].counts)
    assert red_grouping["good_frames"] == 10.0
    assert green_grouping["good_frames"] == 20.0
    assert green_grouping["dead_time_us"] == [0.02, 0.02]
    # returns clones, not the stored objects
    assert green_hists[0] is not grouping["period_histograms"][GREEN_INDEX][0]


def test_select_period_histograms_falls_back_without_period_data():
    base = _histograms(1.0)
    hists, grouping = select_period_histograms(base, {"good_frames": 5.0}, GREEN_INDEX)
    assert len(hists) == len(base)
    assert np.allclose(hists[0].counts, base[0].counts)
    assert hists[0] is not base[0]


# --- combine_period_asymmetry ------------------------------------------------


def test_combine_period_asymmetry_difference_and_sum():
    t = np.linspace(0, 1, 5)
    red_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    green_a = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    red_e = np.full(5, 0.3)
    green_e = np.full(5, 0.4)

    tm, diff, err = combine_period_asymmetry(
        t, red_a, red_e, t, green_a, green_e, PeriodMode.GREEN_MINUS_RED
    )
    assert np.allclose(diff, green_a - red_a)
    assert np.allclose(err, np.sqrt(0.3**2 + 0.4**2))
    assert np.allclose(tm, t)

    _, total, _ = combine_period_asymmetry(
        t, red_a, red_e, t, green_a, green_e, PeriodMode.GREEN_PLUS_RED
    )
    assert np.allclose(total, green_a + red_a)


def test_combine_period_asymmetry_truncates_to_common_length():
    tm, diff, err = combine_period_asymmetry(
        np.arange(5.0),
        np.arange(5.0),
        np.ones(5),
        np.arange(3.0),
        np.full(3, 10.0),
        np.ones(3),
        PeriodMode.GREEN_MINUS_RED,
    )
    assert len(tm) == len(diff) == len(err) == 3


def test_combine_period_asymmetry_empty_and_bad_mode():
    empty = np.array([])
    tm, a, e = combine_period_asymmetry(
        empty, empty, empty, empty, empty, empty, PeriodMode.GREEN_MINUS_RED
    )
    assert len(tm) == 0 and len(a) == 0 and len(e) == 0

    with pytest.raises(ValueError, match="GREEN_MINUS_RED or GREEN_PLUS_RED"):
        combine_period_asymmetry(
            np.arange(3.0),
            np.arange(3.0),
            np.ones(3),
            np.arange(3.0),
            np.arange(3.0),
            np.ones(3),
            PeriodMode.RED,
        )


# --- real NeXus round-trip (skipped without the WiMDA corpus) ----------------


@pytest.mark.skipif(not os.path.exists(PHOTO_MUSR_FILE), reason="photo-µSR corpus not available")
def test_nexus_period_roundtrip_and_load_kwarg():
    combined = load(PHOTO_MUSR_FILE)
    assert isinstance(combined, MuonDataset)
    assert period_count(combined) == 2

    red = select_period(combined, "red")
    green = select_period(combined, "green")
    # load(period=...) equals select_period on the full load result
    assert np.allclose(load(PHOTO_MUSR_FILE, period="green").asymmetry, green.asymmetry)
    assert np.allclose(load(PHOTO_MUSR_FILE, period=1).asymmetry, red.asymmetry)

    # shared time axis; light-ON (red) relaxes more than light-OFF (green)
    assert np.allclose(red.time, green.time)

    def drop(d: MuonDataset) -> float:
        early = d.asymmetry[d.time < 4.0]
        return float(early[0] - early[-1])

    assert drop(red) > drop(green)

    with pytest.raises(ValueError):
        load(PHOTO_MUSR_FILE, period=5)
