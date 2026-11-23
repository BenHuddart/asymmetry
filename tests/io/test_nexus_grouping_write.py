"""NeXus exports distinguish unauthored grouping from an explicit single group."""

import h5py
import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1

pytestmark = pytest.mark.io


@pytest.mark.parametrize("groups", [None, {7: [1, 2]}])
def test_write_read_preserves_grouping_presence_and_acquired_facts(tmp_path, groups):
    raw = [np.arange(24) + 30, np.arange(24) + 10]
    run = Run(
        run_number=42,
        histograms=[
            Histogram(counts, 0.02, t0_bin=2, good_bin_start=4, good_bin_end=19) for counts in raw
        ],
        metadata={"instrument": "ISIS EMU"},
        grouping={
            "first_good_bin": 4,
            "last_good_bin": 19,
            **({"groups": groups} if groups is not None else {}),
        },
    )
    path = tmp_path / "run.nxs"
    write_nexus_v1(run, path)
    with h5py.File(path) as handle:
        detector = handle["run/histogram_data_1"]
        assert ("grouping" in detector) is (groups is not None)
        if groups is not None:
            np.testing.assert_array_equal(detector["grouping"][()], [3, 3])
    dataset = load(path)
    assert dataset.n_points > 0 if groups is None else dataset.n_points == 0
    assert dataset.run.source_assessment.pulsed is True
    assert dataset.run.metadata["instrument"] == "ISIS EMU"
    if groups is not None:
        assert dataset.run.grouping["groups"] == {3: [1, 2]}
        assert "forward_group" not in dataset.run.grouping
    for histogram, counts in zip(dataset.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(histogram.counts, counts)
        assert (histogram.t0_bin, histogram.good_bin_start, histogram.good_bin_end) == (2, 4, 19)


@pytest.mark.parametrize("groups", [{"bad": [1]}, {1: []}, {1: [1], 2: [1]}, {0: [1]}, []])
def test_nonempty_invalid_grouping_is_not_exported_as_absence(tmp_path, groups):
    run = Run(42, histograms=[Histogram(np.ones(24), 0.02)], grouping={"groups": groups})
    with pytest.raises(ValueError):
        write_nexus_v1(run, tmp_path / "run.nxs")
