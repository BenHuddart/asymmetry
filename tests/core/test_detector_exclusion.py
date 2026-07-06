"""Per-detector exclusion at grouping time (WiMDA Group2.pas semantics).

Exclusion is applied where groups are summed — raw histograms stay intact
and no reload is needed (study divergence D10; WiMDA zeroes counts in its
file-read path instead).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets
from asymmetry.core.transform import (
    effective_group_indices,
    estimate_alpha_detailed,
    excluded_detector_indices,
    filter_excluded_indices,
    format_detector_list,
    group_detectors_outside_run,
    group_forward_backward,
    parse_detector_list,
)


def _histograms(values: list[float], n_bins: int = 20) -> list[Histogram]:
    return [Histogram(counts=np.full(n_bins, float(v)), bin_width=0.016) for v in values]


def _grouping(**extra) -> dict:
    return {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
    } | extra


# --- parser ---------------------------------------------------------------------


def test_parse_detector_list_wimda_forms():
    assert parse_detector_list("1,5,10-15") == [1, 5, 10, 11, 12, 13, 14, 15]
    assert parse_detector_list("15-10") == [10, 11, 12, 13, 14, 15]  # reversed range
    assert parse_detector_list(" 3 7 ") == [3, 7]  # whitespace separators
    assert parse_detector_list("2,2,2") == [2]  # duplicates collapse
    assert parse_detector_list("") == []


@pytest.mark.parametrize("bad", ["1,abc", "0", "1-2-3", "-3", "1,0-4"])
def test_parse_detector_list_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_detector_list(bad)


def test_format_detector_list_round_trips():
    ids = [1, 5, 10, 11, 12, 13, 14, 15]
    text = format_detector_list(ids)
    assert text == "1,5,10-15"
    assert parse_detector_list(text) == ids
    assert format_detector_list([]) == ""
    assert format_detector_list([4, 2, 3]) == "2-4"


# --- core application -----------------------------------------------------------


def test_excluded_detector_indices_one_based_to_zero_based():
    assert excluded_detector_indices({"excluded_detectors": [1, 5]}) == frozenset({0, 4})
    assert excluded_detector_indices({}) == frozenset()
    assert excluded_detector_indices(None) == frozenset()
    assert excluded_detector_indices({"excluded_detectors": ["2", None, 0]}) == frozenset({1})


def test_filter_excluded_indices():
    grouping = {"excluded_detectors": [2]}
    assert filter_excluded_indices([0, 1, 2], grouping) == [0, 2]
    assert filter_excluded_indices([0, 1, 2], {}) == [0, 1, 2]


def test_effective_group_indices_is_the_exclusion_aware_chokepoint():
    """The single resolver every reduction path shares: it decodes a group id
    to 0-based indices AND drops excluded detectors and absent detectors, so a
    new call site cannot resolve a group while forgetting exclusion."""
    grouping = _grouping(excluded_detectors=[2])
    # group 1 = detectors [1, 2] (0-based [0, 1]); detector 2 excluded -> [0]
    assert effective_group_indices(grouping, 1) == [0]
    assert effective_group_indices(grouping, 2) == [2, 3]
    # n_histograms drops indices a run does not contain
    assert effective_group_indices(_grouping(), 2, n_histograms=3) == [2]
    # absent / malformed grouping resolves to empty rather than raising
    assert effective_group_indices({}, 1) == []
    assert effective_group_indices(None, 1) == []


def test_group_detectors_outside_run_names_absent_detectors():
    """A HAL-9500 backward-ring preset applied to a forward-only 9-histogram
    ``.mdu`` names detectors 10-17 the run does not contain; the helper reports
    exactly those so the skip can be explained instead of silently dropped."""
    # Per-octant: group k pairs forward det (2+k-1) with backward det (10+k-1).
    per_octant = {k + 1: [2 + k, 10 + k] for k in range(8)}
    # Forward group 1 = [2, 10]; det 10 is absent from a 9-detector run.
    assert group_detectors_outside_run(per_octant, 1, 9) == [10]
    # Backward group 5 = [6, 14]; det 14 absent.
    assert group_detectors_outside_run(per_octant, 5, 9) == [14]
    # Longitudinal backward group = the whole backward ring 10-17.
    longitudinal = {1: list(range(2, 10)), 2: list(range(10, 18))}
    assert group_detectors_outside_run(longitudinal, 2, 9) == [10, 11, 12, 13, 14, 15, 16, 17]
    # A group wholly within the run is clean.
    assert group_detectors_outside_run(longitudinal, 1, 9) == []


def test_group_detectors_outside_run_edge_cases():
    groups = {1: [1, 2], "2": [3, 99], 3: [(4, 12), 5]}  # str key, (det, t0) pair
    assert group_detectors_outside_run(groups, 1, 4) == []
    assert group_detectors_outside_run(groups, 2, 4) == [99]  # str-keyed group resolves
    # pair (4, 12) decodes to det 4 (present); det 5 is absent from a 4-detector run
    assert group_detectors_outside_run(groups, 3, 4) == [5]
    assert group_detectors_outside_run(groups, 99, 4) == []  # missing group id -> empty
    assert group_detectors_outside_run(groups, 1, 0) == [1, 2]  # empty run: all absent
    assert group_detectors_outside_run({1: [0, 1]}, 1, 4) == [0]  # det number < 1 is invalid


def test_effective_group_indices_matches_manual_compose():
    """It is exactly filter_excluded_indices ∘ resolve_group_indices so the
    migrated call sites keep their previous behaviour."""
    from asymmetry.core.transform.grouping import _present_indices, resolve_group_indices

    grouping = _grouping(excluded_detectors=[3])
    groups = grouping["groups"]
    for gid in (1, 2):
        manual = filter_excluded_indices(
            _present_indices(resolve_group_indices(groups, gid), 4), grouping
        )
        assert effective_group_indices(grouping, gid, n_histograms=4) == manual


def test_exclusion_equals_manual_group_membership_removal():
    """The defining equivalence: excluding detector D gives the identical
    reduction to deleting D from its group definitions."""
    histograms = _histograms([10.0, 20.0, 30.0, 40.0])
    excluded = group_forward_backward(histograms, _grouping(excluded_detectors=[2]))
    manual = group_forward_backward(
        histograms,
        {"groups": {1: [1], 2: [3, 4]}, "forward_group": 1, "backward_group": 2, "alpha": 1.0},
    )
    np.testing.assert_array_equal(excluded.forward, manual.forward)
    np.testing.assert_array_equal(excluded.backward, manual.backward)


def test_exclusion_leaves_raw_histograms_intact():
    histograms = _histograms([10.0, 20.0, 30.0, 40.0])
    before = [h.counts.copy() for h in histograms]
    group_forward_backward(histograms, _grouping(excluded_detectors=[1, 3]))
    for hist, original in zip(histograms, before):
        np.testing.assert_array_equal(hist.counts, original)


def test_fully_excluded_group_raises():
    histograms = _histograms([10.0, 20.0, 30.0, 40.0])
    with pytest.raises(ValueError, match="exclusion"):
        group_forward_backward(histograms, _grouping(excluded_detectors=[1, 2]))


def test_alpha_estimate_sees_exclusion_through_grouped_counts():
    """Estimators consume group sums, so exclusion shifts alpha consistently."""
    histograms = _histograms([10.0, 20.0, 30.0, 40.0], n_bins=200)
    full = group_forward_backward(histograms, _grouping())
    excl = group_forward_backward(histograms, _grouping(excluded_detectors=[2]))
    alpha_full = estimate_alpha_detailed(full.forward, full.backward, method="ratio", n_bootstrap=0)
    alpha_excl = estimate_alpha_detailed(excl.forward, excl.backward, method="ratio", n_bootstrap=0)
    assert alpha_full.alpha == pytest.approx(30.0 / 70.0)
    assert alpha_excl.alpha == pytest.approx(10.0 / 70.0)


def test_grouped_time_domain_fit_input_respects_exclusion():
    rng = np.random.default_rng(0)
    histograms = [
        Histogram(counts=rng.poisson(100.0, size=50).astype(float), bin_width=0.016)
        for _ in range(4)
    ]
    run = Run(
        run_number=4242,
        histograms=histograms,
        metadata={"run_number": 4242},
        grouping=_grouping(excluded_detectors=[2], first_good_bin=0, last_good_bin=49),
    )
    dataset = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    grouped_excluded = build_grouped_time_domain_datasets(dataset)
    run.grouping.pop("excluded_detectors")
    grouped_full = build_grouped_time_domain_datasets(dataset)
    assert len(grouped_excluded) == len(grouped_full) == 2
    # Group 1 lost one of its two ~equal detectors: its count level halves.
    excluded_total = float(np.sum(grouped_excluded[0].asymmetry))
    full_total = float(np.sum(grouped_full[0].asymmetry))
    assert excluded_total == pytest.approx(0.5 * full_total, rel=0.05)
    # Group 2 is untouched.
    np.testing.assert_allclose(grouped_excluded[1].asymmetry, grouped_full[1].asymmetry)
