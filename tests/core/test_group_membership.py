"""Acquired channels select unique observations before count reduction."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.transform.grouping import (
    effective_group_indices,
    group_forward_backward,
    parse_group_detectors,
)


def histograms():
    return [
        Histogram(counts=np.full(20, c), count_variance=np.full(20, v), bin_width=0.1)
        for c, v in [(10.0, 2.0), (30.0, 7.0), (50.0, 11.0)]
    ]


@pytest.mark.parametrize("members", [[0], [-1], [True], [1.5], ["bad"], [1, 1], [1, 4], []])
@pytest.mark.parametrize("consumer", ["pair", "spectrum"])
def test_invalid_membership_fails_before_summing(members, consumer):
    hist = histograms()
    grouping = {"groups": {1: members, 2: [3]}, "first_good_bin": 0, "last_good_bin": 19}
    with pytest.raises(ValueError, match="[Gg]roup|[Dd]etector"):
        if consumer == "pair":
            group_forward_backward(hist, grouping)
        else:
            build_group_signal_dataset(Run(histograms=hist, grouping=grouping), 1)


def test_supplied_variance_and_raw_counts_survive_exclusion():
    hist = histograms()
    grouping = {"groups": {1: [1, 2], 2: [3]}, "excluded_detectors": [2]}
    fb = group_forward_backward(hist, grouping)
    np.testing.assert_array_equal(fb.forward, np.full(20, 10.0))
    np.testing.assert_array_equal(fb.counts.forward.variance, np.full(20, 2.0))
    assert effective_group_indices(grouping, 1, n_histograms=3) == [0]
    for h, c, v in zip(hist, [10.0, 30.0, 50.0], [2.0, 7.0, 11.0]):
        np.testing.assert_array_equal(h.counts, np.full(20, c))
        np.testing.assert_array_equal(h.count_variance, np.full(20, v))
    full = group_forward_backward(hist, {"groups": {1: [1, 2], 2: [3]}})
    np.testing.assert_array_equal(full.forward, np.full(20, 40.0))
    np.testing.assert_array_equal(full.counts.forward.variance, np.full(20, 9.0))


def test_typed_ranges_are_union_before_structured_validation():
    members = parse_group_detectors("1-2,2,2-1", 3)
    assert members == [1, 2]
    assert effective_group_indices({"groups": {1: members}}, 1, n_histograms=3) == [0, 1]


@pytest.mark.parametrize("excluded", [[0], [True], [1.5], ["bad"], [2, 2], [4], [1, 2]])
def test_invalid_or_empty_exclusion_is_unavailable(excluded):
    with pytest.raises(ValueError, match="[Dd]etector|[Gg]roup|exclusion"):
        group_forward_backward(
            histograms(), {"groups": {1: [1, 2], 2: [3]}, "excluded_detectors": excluded}
        )


def test_exclusion_cannot_hide_preset_data_mismatch():
    with pytest.raises(ValueError, match="4"):
        effective_group_indices(
            {"groups": {1: [1, 4]}, "excluded_detectors": [4]}, 1, n_histograms=3
        )


@pytest.mark.parametrize("consumer", ["all_groups", "one_group"])
def test_grouped_count_boundaries_reject_unknown_exclusions(consumer):
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.grouped_time_domain import (
        build_count_group,
        build_grouped_time_domain_datasets,
    )

    run = Run(
        histograms=histograms(),
        grouping={"groups": {1: [1, 2], 2: [3]}, "excluded_detectors": [99]},
    )
    dataset = MuonDataset(np.arange(20) * 0.1, np.zeros(20), np.ones(20), run=run)
    with pytest.raises(ValueError, match="99"):
        if consumer == "one_group":
            build_count_group(dataset, 1)
        else:
            build_grouped_time_domain_datasets(dataset)


def test_event_count_display_is_unavailable_for_malformed_window():
    from asymmetry.core.transform.grouping import good_event_count

    assert (
        good_event_count(
            histograms(), {"groups": {1: [1], 2: [2]}, "first_good_bin": "bad", "last_good_bin": 19}
        )
        is None
    )
