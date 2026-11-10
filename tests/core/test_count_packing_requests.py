"""Authored fixed packing and empty fit selections at public count boundaries."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.grouped_time_domain import (
    build_count_group,
    build_count_groups,
    build_grouped_time_domain_groups,
)
from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, select_tf_calibration
from tests.core.test_binning_entry_points import _run


@pytest.mark.parametrize("consumer", ["single", "pair", "grouped", "tf"])
@pytest.mark.parametrize("selection", ["packing", "limits", "exclude"])
@pytest.mark.parametrize("correction", ["off", "deadtime", "background"])
def test_empty_fit_precedes_unused_corrections(consumer, selection, correction):
    run = _run(
        np.arange(1, 7), np.arange(11, 17), bunching_factor=8 if selection == "packing" else 2
    )
    if correction == "deadtime":
        run.grouping.update(deadtime_correction=True, deadtime_mode="file", deadtime_method="file")
    elif correction == "background":
        run.grouping.update(
            background_correction=True, background_mode="range", background_ranges={1: [90, 100]}
        )
    before = deepcopy(run)
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    options = (
        {"t_min": 10.0}
        if selection == "limits"
        else {"exclude": (-1.0, 10.0)}
        if selection == "exclude"
        else {}
    )
    message = (
        "fewer than two usable groups"
        if consumer == "grouped"
        else "Count-domain group 1 produced no usable bins"
    )
    with pytest.raises(ValueError, match=message):
        if consumer == "tf":
            select_tf_calibration(dataset, TFCalibrationRequest(1, 2, **options))
        elif consumer == "single":
            build_count_group(dataset, 1, **options)
        elif consumer == "pair":
            build_count_groups(dataset, [1, 2], **options)
        else:
            build_grouped_time_domain_groups(dataset, **options)
    assert run.grouping == before.grouping
    for actual, original in zip(run.histograms, before.histograms, strict=True):
        np.testing.assert_array_equal(actual.counts, original.counts)
    assert dataset.run is run


@pytest.mark.parametrize("size", [7, 8, 9, 16, 19])
@pytest.mark.parametrize("start,offset", [(0, None), (2, 0.03)])
@pytest.mark.parametrize("lifetime", [False, True])
def test_count_packing_retains_complete_authored_groups(size, start, offset, lifetime):
    from asymmetry.core.utils.constants import MUON_LIFETIME_US

    values = np.arange(1.0, start + size + 1)
    run = _run(values, values + 10, first_good_bin=start, bunching_factor=8)
    run.histograms = [
        replace(h, t0_time_us=offset, count_variance=2 * h.counts) for h in run.histograms
    ]
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    if size < 8:
        with pytest.raises(ValueError, match="produced no usable bins"):
            build_count_group(dataset, 1, lifetime_corrected=lifetime)
        return
    group = build_count_group(dataset, 1, lifetime_corrected=lifetime)
    n = size // 8
    expected = values[start : start + n * 8].reshape(n, 8).sum(axis=1)
    centres = (start + 3.5 + 8 * np.arange(n)) * 0.1 + (
        0.05 - offset if offset is not None else 0.0
    )
    scale = np.exp(centres / MUON_LIFETIME_US) if lifetime else 1
    np.testing.assert_allclose(group.time, centres, atol=1e-15)
    np.testing.assert_allclose(group.counts, expected * scale)
    np.testing.assert_allclose(group.error, np.sqrt(2 * expected) * scale)
    assert group.metadata["grouped_time_domain_bunch_factor"] == 8
    assert group.metadata["count_fit_raw_windows"] == tuple(
        (start + 8 * i, start + 8 * i + 7) for i in range(n)
    )


@pytest.mark.parametrize("factor", [2, 8])
def test_packed_fractional_centres_scope_corrections_and_trim_tail(factor):
    from tests.core.test_grouped_deadtime_scope import scoped_run

    run = scoped_run("prefix")
    run.grouping.update(first_good_bin=1, last_good_bin=1 + 4 * factor, bunching_factor=factor)
    run.histograms = [replace(h, t0_time_us=0.53) for h in run.histograms]
    run.histograms[0].counts[:] = 10
    # Retain groups 0 and 2; filtered, excluded, and partial groups are unused.
    run.histograms[0].counts[1 + factor : 1 + 2 * factor] = 100
    run.histograms[0].counts[1 + 3 * factor :] = 100
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    centres = 1 + (factor - 1) / 2 + factor * np.arange(4) - 0.03
    group = build_count_group(
        dataset,
        1,
        lifetime_corrected=False,
        t_max=centres[2],
        exclude=(centres[1] - 0.1, centres[1] + 0.1),
    )
    np.testing.assert_allclose(group.time, centres[[0, 2]])
    np.testing.assert_allclose(group.counts, factor * 10 / 0.9)
    assert group.metadata["count_fit_raw_windows"] == ((1, factor), (1 + 2 * factor, 3 * factor))
    assert group.deadtime.effective["scope"]["raw_bin_ranges"] == {
        "0": ((1, factor), (1 + 2 * factor, 3 * factor))
    }
    with pytest.raises(ValueError, match="occupancy"):
        build_count_group(dataset, 1, lifetime_corrected=False)
