"""Validated count records keep array, window, variance and timing invariants."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    BinWindow,
    ForwardBackwardCounts,
    GroupCounts,
    SharedComponent,
    good_window,
)


def _counts(**overrides) -> ForwardBackwardCounts:
    values = dict(
        forward=GroupCounts(np.ones(10)),
        backward=GroupCounts(np.ones(10)),
        bin_width_us=0.01,
        t0_time_us=0.005,
        window=BinWindow(0, 9),
        alignment=AlignmentCrop(0, (), (10, 10), 10),
    )
    values.update(overrides)
    return ForwardBackwardCounts(**values)


@pytest.mark.parametrize(
    ("first", "last"), [(7, 3), (20, 30), (-5, -1), (3, 100), ("x", 4), (0, 1.5), (True, 4)]
)
def test_invalid_good_window_is_rejected(first, last) -> None:
    with pytest.raises(ValueError, match="good-bin window"):
        good_window({"first_good_bin": first, "last_good_bin": last}, 10)


def test_absent_good_window_selects_the_full_record() -> None:
    assert good_window({}, 10) == BinWindow(0, 9)
    assert good_window({"first_good_bin": 3}, 10) == BinWindow(3, 9)
    assert good_window({"last_good_bin": 5}, 10) == BinWindow(0, 5)


@pytest.mark.parametrize("data", [1.0, [[1.0, 2.0]], [1.0, np.nan], [np.inf], [1.0 + 2.0j]])
def test_group_counts_require_finite_real_one_dimensional_arrays(data) -> None:
    with pytest.raises(ValueError, match="count arrays"):
        GroupCounts(data)


@pytest.mark.parametrize("variance", [[1.0], [1.0, -1.0], [1.0, np.nan]])
def test_variance_has_the_count_shape_and_is_finite_non_negative(variance) -> None:
    with pytest.raises(ValueError):
        GroupCounts([10.0, 20.0], variance)


def test_group_counts_own_read_only_snapshots() -> None:
    source = np.array([10.0, 20.0])
    variance = np.array([2.0, 3.0])
    group = GroupCounts(source, variance)
    source[0] = np.nan
    variance[0] = -1
    np.testing.assert_array_equal(group.counts, [10.0, 20.0])
    np.testing.assert_array_equal(group.variance, [2.0, 3.0])
    with pytest.raises(ValueError, match="read-only"):
        group.counts[0] = 0


def test_forward_backward_shapes_and_variance_presence_must_match() -> None:
    with pytest.raises(ValueError, match="shape"):
        _counts(backward=GroupCounts(np.ones(9)))
    with pytest.raises(ValueError, match="both groups"):
        _counts(forward=GroupCounts(np.ones(10), np.ones(10)))
    with pytest.raises(ValueError, match="at least one bin"):
        _counts(forward=GroupCounts([]), backward=GroupCounts([]))


@pytest.mark.parametrize("width", [0.0, -0.01, np.nan, np.inf])
def test_count_grid_requires_a_positive_finite_bin_width(width: float) -> None:
    with pytest.raises(ValueError, match="bin width"):
        _counts(bin_width_us=width)


@pytest.mark.parametrize("t0", [np.nan, np.inf, -np.inf])
def test_exact_t0_must_be_finite(t0: float) -> None:
    with pytest.raises(ValueError, match="t0"):
        _counts(t0_time_us=t0)


def test_window_and_alignment_match_the_array_extent() -> None:
    with pytest.raises(ValueError, match="good-bin window"):
        _counts(window=BinWindow(0, 10))
    with pytest.raises(ValueError, match="alignment crop"):
        _counts(alignment=AlignmentCrop(0, (), (10, 10), 9))


def test_background_provenance_has_one_source_for_levels_and_errors() -> None:
    state = {"method": "range", "values": [10.0, 8.0]}
    groups = [
        GroupCounts(
            np.ones(10),
            np.ones(10),
            shared=(
                SharedComponent(
                    "background",
                    variance,
                    -np.ones(10),
                    np.zeros(10),
                ),
            ),
        )
        for variance in (1.0, 4.0)
    ]
    counts = _counts(forward=groups[0], backward=groups[1], background_state=state)
    state["values"][0] = 999.0
    assert counts.background_level == (10.0, 8.0)
    with pytest.raises(TypeError):
        counts.background_state["values"] = (999.0, 8.0)
    assert "value_errors" not in counts.background_state
    exported = counts.background_metadata()
    assert exported["values"] == [10.0, 8.0]
    assert exported["value_errors"] == [1.0, 2.0]
    exported["values"][0] = 999.0
    assert counts.background_level == (10.0, 8.0)


def test_ragged_detector_alignment_records_the_terminal_crop() -> None:
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.transform.grouping import align_forward_backward
    from asymmetry.core.transform.t0 import good_window_for_groups

    histograms = [Histogram(np.ones(1000), bin_width=0.01, t0_bin=t0) for t0 in (100, 110)]
    forward, backward, crop = align_forward_backward(histograms, [0], [1], [100, 110])
    assert crop == AlignmentCrop(110, ((0, 10), (1, 0)), (1010, 1000), 1000)
    np.testing.assert_array_equal(forward[:10], 0.0)
    np.testing.assert_array_equal(forward[10:], 1.0)
    np.testing.assert_array_equal(backward, 1.0)
    first, last = good_window_for_groups(
        [0, 1], [100, 110], [0, 0], [999, 999], common_t0_bin=110, n_bins=crop.length
    )
    assert good_window({"first_good_bin": first, "last_good_bin": last}, crop.length) == BinWindow(
        10, 999
    )


@pytest.mark.parametrize("window", [(7, 3), (20, 30), (-5, -1), (3, 100), ("x", 4)])
def test_reduction_rejects_invalid_good_windows(window) -> None:
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.transform.reduce import reduce_grouped_asymmetry

    with pytest.raises(ValueError, match="good-bin window"):
        reduce_grouped_asymmetry(
            histograms=[Histogram(np.ones(10), bin_width=0.01) for _ in range(2)],
            grouping={"first_good_bin": window[0], "last_good_bin": window[1]},
            forward_idx=[0],
            backward_idx=[1],
            alpha=1.0,
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )


@pytest.mark.parametrize("reference_length", [5, 10, 15])
def test_reference_must_cover_the_sample_aligned_grid(reference_length: int) -> None:
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.transform.reduce import corrected_grouped_counts

    sample = [Histogram(np.full(10, 20.0), bin_width=0.01) for _ in range(2)]
    reference = [Histogram(np.full(reference_length, 2.0), bin_width=0.01) for _ in range(2)]
    kwargs = dict(
        histograms=sample,
        grouping={"background_mode": "reference_run"},
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        reference_resolver=lambda _grouping: (reference, 2.0),
    )
    if reference_length < 10:
        with pytest.raises(
            ValueError, match="reference run has 5 aligned bins; sample requires 10"
        ):
            corrected_grouped_counts(**kwargs)
    else:
        result = corrected_grouped_counts(**kwargs)
        np.testing.assert_array_equal(result.forward.counts, np.full(10, 16.0))
        np.testing.assert_array_equal(result.backward.counts, np.full(10, 16.0))
        assert result.alignment.length == 10
        reference_alignment = result.background_state["reference_alignment"]
        assert reference_alignment["group_lengths"] == (reference_length, reference_length)
        assert reference_alignment["length"] == 10


@pytest.mark.io
@pytest.mark.parametrize("window", [(2, 7), (7, 3), (3, 100)])
def test_saved_window_loads_unchanged_and_is_checked_at_reduction(tmp_path, window) -> None:
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
    from asymmetry.core.transform.reduce import reduce_grouped_asymmetry

    grouping = {"first_good_bin": window[0], "last_good_bin": window[1]}
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {"run_number": 1, "source_file": "sample.nxs", "grouping_overrides": grouping}
        ],
    }
    path = tmp_path / "window.asymp"
    save_project(state, path)
    loaded_grouping = load_project(path)["datasets"][0]["grouping_overrides"]
    assert loaded_grouping == grouping
    kwargs = dict(
        histograms=[Histogram(np.full(10, 20.0), bin_width=0.01) for _ in range(2)],
        grouping=loaded_grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    if window == (2, 7):
        assert reduce_grouped_asymmetry(**kwargs).time.size == 6
    else:
        with pytest.raises(ValueError, match="good-bin window"):
            reduce_grouped_asymmetry(**kwargs)


@pytest.mark.parametrize(
    "state",
    [
        {"value_errors": [1.0, 2.0]},
        {"values": None},
        {"values": [1.0]},
        {"values": [1.0, np.nan]},
        {"values": [1.0, 2.0], "value_errors": [1.0, 2.0]},
    ],
)
def test_background_provenance_requires_finite_levels_and_derived_errors(state) -> None:
    with pytest.raises(ValueError):
        _counts(background_state=state)
