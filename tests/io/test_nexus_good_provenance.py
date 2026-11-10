"""Unavailable and mapped file windows keep their acquired timing and bounds."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.periods import (
    combine_mapped_periods,
    select_period,
    select_period_histograms,
)
from asymmetry.core.io.quick_asymmetry import quick_asymmetry
from asymmetry.core.project.profiles import (
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.simulate import reduce_run_to_dataset
from tests.io.test_nexus_good_times import write_good_times

h5py = pytest.importorskip("h5py")
pytestmark = pytest.mark.io


@pytest.mark.parametrize("exact", [False, True])
def test_empty_window_retains_selected_alignment_and_exact_origin(tmp_path, exact):
    path = tmp_path / "empty.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "grouping", "time_zero", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=np.vstack([raw, np.ones(32)])).attrs["t0_bin"] = [
            11,
            13,
            41,
        ]
        d.create_dataset("grouping", data=[1, 2, 3])
        if exact:
            d.create_dataset("time_zero", data=[0.2074, 0.247, 0.8074])
        d.create_dataset("first_good_time", data=[0.22, 0.40, 0.0])
        d.create_dataset("last_good_time", data=[0.30, 0.50, 0.62])
    ds = NexusLoader().load(path)
    assert ds.n_points == 0
    assert ds.run.grouping["t0_bin"] == 12
    assert ds.run.grouping.get("t0_time_us") == (0.247 if exact else None)
    request = dict(ds.run.grouping, groups={1: [1], 2: [2, 3]}, excluded_detectors=[3])
    request["effective_detector_t0_bins"] = [11, 14, 40]
    quick = quick_asymmetry(ds.run.histograms, request)
    assert "empty good-bin support" in quick.unavailable_reason
    assert quick.common_t0 == 14
    assert quick.alignment.first_bin == 3
    assert quick.file_grouping["t0_bin"] == 14
    if exact:
        assert quick.t0_time_us == pytest.approx(0.287, abs=1e-15)
    else:
        assert quick.t0_time_us is None
    with pytest.raises(ValueError, match="reversed|empty"):
        reduce_run_to_dataset(ds.run)
    np.testing.assert_array_equal(ds.run.histograms[0].counts, raw[0])


def test_unavailable_selected_period_retains_its_exact_origin(tmp_path):
    path = tmp_path / "period.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "time_zero", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=np.stack([raw, raw + 10]))
        d.create_dataset("time_zero", data=[[0.2074, 0.2074], [0.2074, 0.247]])
        d.create_dataset("first_good_time", data=[[0.22, 0.22], [0.22, 0.40]])
        d.create_dataset("last_good_time", data=[[0.30, 0.30], [0.30, 0.50]])
    selected = select_period(NexusLoader().load(path), "green")
    assert selected.n_points == 0
    assert selected.run.grouping["t0_bin"] == 12
    assert selected.run.grouping["t0_time_us"] == pytest.approx(0.247, abs=1e-15)
    np.testing.assert_array_equal(selected.run.histograms[0].counts, raw[0] + 10)


@pytest.mark.parametrize("held", ["present", "absent", "partial", "malformed"])
def test_mapped_and_fallback_detector_facts_follow_acquired_histograms(tmp_path, held):
    path = tmp_path / "mapped.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=np.stack([raw, raw + 10, raw + 20]))
        d.create_dataset("first_good_time", data=[[0.22] * 2, [0.24] * 2, [0.22] * 2])
        d.create_dataset("last_good_time", data=[[0.30] * 2, [0.32] * 2, [0.30] * 2])
    mapped = combine_mapped_periods(NexusLoader().load(path), {1: "red", 2: "red", 3: "green"})
    assert mapped.run.grouping["detector_first_good_bins"] == [12, 12]
    assert mapped.run.grouping["detector_last_good_bins"] == [14, 14]
    request = dict(mapped.run.grouping)
    index = 0
    if held != "present":
        # Legacy held-only fallback derives facts from its observed buffer.
        request.pop("period_inputs", None)
        request["detector_first_good_bins"] = [11, 11]
        if held == "absent":
            del request["period_histograms"]
        elif held == "partial":
            request["period_histograms"] = request["period_histograms"][:1]
            index = 1
        else:
            request["period_histograms"] = [request["period_histograms"][0], None]
            index = 1
    histograms, grouping = select_period_histograms(mapped.run.histograms, request, index)
    assert grouping["detector_first_good_bins"] == [12, 12]
    assert grouping["detector_last_good_bins"] == [14, 14]
    grouping.pop("period_histograms", None)
    run = replace(mapped.run, histograms=histograms, grouping=grouping)
    profile = GroupingProfile(
        name="file", fingerprint=ProfileFingerprint("", 2), groups={1: [1], 2: [2]}
    )
    effective = resolve_effective_grouping(profile, run)
    assert (effective["first_good_bin"], effective["last_good_bin"]) == (12, 14)
    reduced = reduce_run_to_dataset(replace(run, grouping=effective))
    np.testing.assert_allclose(reduced.time, [0.0426, 0.0626, 0.0826], atol=1e-15)
    np.testing.assert_array_equal(histograms[0].counts, 2 * raw[0] + 10)


@pytest.mark.parametrize("mode", ["green_minus_red", "green_plus_red"])
def test_period_combination_prepares_common_native_support(tmp_path, mode):
    path = tmp_path / "different-windows.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=np.stack([raw, raw + 10]))
        d.create_dataset("first_good_time", data=[[0.22] * 2, [0.24] * 2])
        d.create_dataset("last_good_time", data=[[0.30] * 2, [0.32] * 2])
    loaded = NexusLoader().load(path)
    for period in (1, 2):
        assert select_period(loaded, period).n_points == 4
    reduced = reduce_run_to_dataset(
        replace(loaded.run, grouping=dict(loaded.run.grouping, period_mode=mode))
    )
    np.testing.assert_allclose(reduced.time, [0.0426, 0.0626, 0.0826], atol=1e-15)
    f, b = raw[:, 12:15]
    if mode == "green_plus_red":
        expected = 100 * ((2 * f + 10) - (2 * b + 10)) / ((2 * f + 10) + (2 * b + 10))
    else:
        expected = 100 * ((f + 10) - (b + 10)) / ((f + 10) + (b + 10)) - 100 * (f - b) / (f + b)
    np.testing.assert_allclose(reduced.asymmetry, expected)
