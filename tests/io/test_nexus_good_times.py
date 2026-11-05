"""Raw-acquisition good-time selection through the production loading boundary."""

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.simulate import reduce_run_to_dataset

h5py = pytest.importorskip("h5py")
pytestmark = [pytest.mark.io]


def write_good_times(path, *, first=0.22, last=0.30):
    """Tag each acquired observation independently of timing and grouping."""
    counts = np.stack([100 + np.arange(32), 300 + 2 * np.arange(32)])
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data_1")
        entry.create_dataset("definition", data=b"muonTD")
        entry.create_dataset("IDF_version", data=2)
        detector = entry.create_group("instrument/detector_1")
        detector.create_dataset("counts", data=counts)
        detector.create_dataset("grouping", data=[1, 2])
        for name, value, unit in [
            ("raw_time", np.arange(33) * 0.02, "us"),
            ("resolution", 20.0, "ns"),
            ("time_zero", 0.2074, "us"),
            ("first_good_time", first, "us"),
            ("last_good_time", last, "us"),
        ]:
            if value is not None:
                detector.create_dataset(name, data=value).attrs["units"] = unit
    return counts


def test_raw_good_times_survive_rereduction(tmp_path):
    path = tmp_path / "good.nxs"
    raw = write_good_times(path)
    loaded = NexusLoader().load(path)
    expected_time = [0.0226, 0.0426, 0.0626, 0.0826]
    forward, backward = raw[:, 11:15]
    expected_asymmetry = 100 * (forward - backward) / (forward + backward)
    expected_error = 200 * np.sqrt(forward * backward / (forward + backward) ** 3)
    for dataset in (loaded, reduce_run_to_dataset(loaded.run)):
        np.testing.assert_allclose(dataset.time, expected_time, rtol=0, atol=1e-15)
        np.testing.assert_allclose(dataset.asymmetry, expected_asymmetry, rtol=1e-14)
        np.testing.assert_allclose(dataset.error, expected_error, rtol=1e-14)
    assert [(h.good_bin_start, h.good_bin_end) for h in loaded.run.histograms] == [(11, 14)] * 2
    for histogram, counts in zip(loaded.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(histogram.counts, counts)


@pytest.mark.parametrize(
    "unit,scale", [("us", 1), ("ns", 1000), ("ps", 1e6), ("s", 1e-6), (None, 1)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_good_time_units_and_stored_precision(tmp_path, unit, scale, dtype):
    path = tmp_path / "units.nxs"
    write_good_times(path, first=0.23, last=0.29)
    with h5py.File(path, "a") as f:
        detector = f["raw_data_1/instrument/detector_1"]
        for name, value in [("first_good_time", 0.23), ("last_good_time", 0.29)]:
            del detector[name]
            node = detector.create_dataset(name, data=dtype(value * scale))
            if unit is not None:
                node.attrs["units"] = unit
    loaded = NexusLoader().load(path)
    assert [(h.good_bin_start, h.good_bin_end) for h in loaded.run.histograms] == [(11, 14)] * 2
    np.testing.assert_allclose(loaded.time, [0.0226, 0.0426, 0.0626, 0.0826], atol=1e-15)


@pytest.mark.parametrize(
    "first,last,bounds",
    [
        (0.22, 0.30, (11, 14)),  # edge bounds
        (0.23, 0.29, (11, 14)),  # inclusive centres
        (0.230000001, 0.289999999, (12, 13)),
        (0.229999999, 0.290000001, (11, 14)),
        (None, 0.30, (0, 14)),
        (0.22, None, (11, 31)),
        (0, None, (0, 31)),
        (0.23, 0.23, (11, 11)),
        (0.230000001, 0.249999999, (32, 31)),
        (0.30, 0.22, (32, 31)),
        (1.0, None, (32, 31)),
        (None, -0.1, (32, 31)),
        (None, 0, (32, 31)),  # zero is a real bound, not unavailable
    ],
)
def test_physical_endpoint_membership(tmp_path, first, last, bounds):
    path = tmp_path / "endpoints.nxs"
    raw = write_good_times(path, first=first, last=last)
    dataset = NexusLoader().load(path)
    assert [(h.good_bin_start, h.good_bin_end) for h in dataset.run.histograms] == [bounds] * 2
    if bounds[0] > bounds[1]:
        assert dataset.n_points == 0
        assert "empty good-bin support" in dataset.metadata["correction_error"]
        with pytest.raises(ValueError, match="empty|reversed"):
            reduce_run_to_dataset(dataset.run)
    else:
        expected = (np.arange(bounds[0], bounds[1] + 1) + 0.5) * 0.02 - 0.2074
        np.testing.assert_allclose(dataset.time, expected, atol=1e-15)
        np.testing.assert_allclose(reduce_run_to_dataset(dataset.run).time, expected, atol=1e-15)
    np.testing.assert_array_equal(dataset.run.histograms[0].counts, raw[0])


@pytest.mark.parametrize(
    "endpoint,value,expected",
    [
        ("first_good_bin", b"13", (12, 14)),
        ("last_good_bin", 14, (11, 13)),
        ("first_good_bin", [12, 13], ((11, 14), (12, 14))),
    ],
)
def test_integer_precedence_is_endpoint_specific(tmp_path, endpoint, value, expected):
    path = tmp_path / "precedence.nxs"
    write_good_times(path)
    with h5py.File(path, "a") as f:
        f["raw_data_1/instrument/detector_1/counts"].attrs[endpoint] = value
    dataset = NexusLoader().load(path)
    bounds = [(h.good_bin_start, h.good_bin_end) for h in dataset.run.histograms]
    assert bounds == (list(expected) if isinstance(expected[0], tuple) else [expected] * 2)
    again = reduce_run_to_dataset(dataset.run)
    np.testing.assert_allclose(dataset.time, again.time, atol=1e-15)


@pytest.mark.parametrize("fault", ["units", "nan", "shape", "bad_integer"])
def test_unusable_optional_bound_retains_loading(tmp_path, fault):
    path = tmp_path / "optional.nxs"
    write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        if fault == "units":
            d["first_good_time"].attrs["units"] = "K"
        elif fault == "nan":
            d["first_good_time"][()] = np.nan
        elif fault == "bad_integer":
            d["counts"].attrs["first_good_bin"] = "unavailable"
        else:
            del d["first_good_time"]
            d.create_dataset("first_good_time", data=np.ones((2, 1)))
    dataset = NexusLoader().load(path)
    first = 11 if fault == "bad_integer" else 0
    assert dataset.run.histograms[0].good_bin_start == first
    assert dataset.run.histograms[0].good_bin_end == 14


@pytest.mark.parametrize("axis", ["resolution", "raw_time", "corrected_time"])
def test_good_times_use_same_zero_origin_acquisition_grid(tmp_path, axis):
    path = tmp_path / "axis.nxs"
    write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        if axis == "resolution":
            d["raw_time"][:] += 2  # auxiliary origin has no authority
        else:
            del d["resolution"]
            if axis == "corrected_time":
                del d["raw_time"]
                d.create_dataset("corrected_time", data=(np.arange(32) + 0.5) * 0.02 - 0.2074)
    dataset = NexusLoader().load(path)
    assert dataset.run.histograms[0].good_bin_start == 11
    assert dataset.run.histograms[0].good_bin_end == 14
    assert dataset.metadata["nexus_timing_source"] == axis


def test_detector_timing_windows_spectators_and_exclusions(tmp_path):
    from asymmetry.core.project.profiles import (
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )

    path = tmp_path / "detectors.nxs"
    write_good_times(path)
    raw = np.arange(96).reshape(3, 32) + 100
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "time_zero", "grouping", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=raw)
        d.create_dataset("time_zero", data=[0.2074, 0.227, 0.6])
        d.create_dataset("grouping", data=[1, 2, 3])
        d.create_dataset("first_good_time", data=[0.22, 0.26, 1.0])
        d.create_dataset("last_good_time", data=[0.30, 0.34, 1.1])
    dataset = NexusLoader().load(path)
    assert [(h.good_bin_start, h.good_bin_end) for h in dataset.run.histograms] == [
        (11, 14),
        (13, 16),
        (32, 31),
    ]
    # Anchor shifts: detector 1 raw 12..14 pairs with detector 2 raw 13..15.
    np.testing.assert_allclose(dataset.time, [0.043, 0.063, 0.083], atol=1e-15)
    np.testing.assert_allclose(
        dataset.asymmetry, 100 * (raw[0, 12:15] - raw[1, 13:16]) / (raw[0, 12:15] + raw[1, 13:16])
    )
    payload = dict(dataset.run.grouping, groups={1: [1], 2: [2, 3]}, excluded_detectors=[3])
    profile = profile_from_payload(payload, "selected", profile_fingerprint_for_run(dataset.run))
    effective = resolve_effective_grouping(profile, dataset.run)
    from dataclasses import replace

    reduced = reduce_run_to_dataset(replace(dataset.run, grouping=effective))
    np.testing.assert_allclose(reduced.time, dataset.time, atol=1e-15)
    np.testing.assert_allclose(reduced.asymmetry, dataset.asymmetry)


@pytest.mark.parametrize("n_periods", [2, 3])
def test_each_period_retains_physical_bounds(tmp_path, n_periods):
    from asymmetry.core.io.periods import combine_mapped_periods, select_period

    path = tmp_path / "periods.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=np.stack([raw + p * 10 for p in range(n_periods)]))
        d.create_dataset("first_good_time", data=[[0.22 + p * 0.02] * 2 for p in range(n_periods)])
        d.create_dataset("last_good_time", data=[[0.30 + p * 0.02] * 2 for p in range(n_periods)])
    loaded = NexusLoader().load(path)
    periods = [select_period(loaded, p + 1) for p in range(n_periods)]
    for p, ds in enumerate(periods):
        assert [(h.good_bin_start, h.good_bin_end) for h in ds.run.histograms] == [
            (11 + p, 14 + p)
        ] * 2
        np.testing.assert_allclose(ds.time, reduce_run_to_dataset(ds.run).time, atol=1e-15)
    mapped = combine_mapped_periods(periods, {1: "red", 2: "green"})
    for p in range(2):
        histograms = mapped.run.grouping["period_histograms"][p]
        assert [(h.good_bin_start, h.good_bin_end) for h in histograms] == [(11 + p, 14 + p)] * 2


def test_mapped_sum_intersects_contributing_good_support(tmp_path):
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.io.periods import sum_period_histograms

    periods = [
        [Histogram(np.ones(8) * n, 0.02, 1, first, last)]
        for n, first, last in [(10, 2, 6), (20, 3, 5)]
    ]
    result = sum_period_histograms(periods)
    assert (result[0].good_bin_start, result[0].good_bin_end) == (3, 5)
    np.testing.assert_array_equal(result[0].counts, np.full(8, 30))


@pytest.mark.parametrize("container", ["hdf5", "hdf4"])
def test_v1_keeps_integer_good_window_and_ignores_undefined_physical_fields(tmp_path, container):
    path = tmp_path / "v1.nxs"
    if container == "hdf4":
        pytest.importorskip("pyhdf")
        from tests.io.test_hdf4_loader import _write_hdf4_v1

        _write_hdf4_v1(
            path, counts=np.ones((2, 32)) * 10, first_good_bin=12, last_good_bin=15, t0_bin=11
        )
    else:
        write_good_times(path)
        with h5py.File(path, "a") as f:
            run = f.create_group("run")
            run.create_dataset("analysis", data=b"muonTD")
            f.move("raw_data_1/instrument/detector_1", "run/histogram_data_1")
            del f["raw_data_1"]
            d = run["histogram_data_1"]
            d["counts"].attrs["first_good_bin"] = 12
            d["counts"].attrs["last_good_bin"] = 15
            # Existing v1 child spelling cannot override attributes.
            d.create_dataset("first_good_bin", data=20)
    ds = NexusLoader().load(path)
    assert [(h.good_bin_start, h.good_bin_end) for h in ds.run.histograms] == [(11, 14)] * 2
    np.testing.assert_allclose(ds.time, reduce_run_to_dataset(ds.run).time, atol=1e-15)


def test_one_acquired_bin_uses_declared_resolution(tmp_path):
    path = tmp_path / "one.nxs"
    write_good_times(path, first=0.01, last=0.01)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["counts"]
        d.create_dataset("counts", data=[[100], [200]])
        d["time_zero"][()] = 0.0
    ds = NexusLoader().load(path)
    np.testing.assert_allclose(ds.time, [0.01])
    np.testing.assert_allclose(reduce_run_to_dataset(ds.run).time, ds.time)


def test_v1_child_integer_compatibility_is_retained(tmp_path):
    path = tmp_path / "child.nxs"
    write_good_times(path)
    with h5py.File(path, "a") as f:
        run = f.create_group("run")
        run.create_dataset("analysis", data=b"muonTD")
        f.move("raw_data_1/instrument/detector_1", "run/histogram_data_1")
        del f["raw_data_1"]
        d = run["histogram_data_1"]
        d.create_dataset("first_good_bin", data=12)
        d.create_dataset("last_good_bin", data=15)
    ds = NexusLoader().load(path)
    assert [(h.good_bin_start, h.good_bin_end) for h in ds.run.histograms] == [(11, 14)] * 2


@pytest.mark.parametrize("second", [(1.0, 1.1), (0.40, 0.50)])
def test_unavailable_selected_intersection_cannot_rereduce_first_detector(tmp_path, second):
    path = tmp_path / "empty-pair.nxs"
    write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name, values in [
            ("first_good_time", [0.22, second[0]]),
            ("last_good_time", [0.30, second[1]]),
        ]:
            del d[name]
            d.create_dataset(name, data=values)
    ds = NexusLoader().load(path)
    assert ds.n_points == 0
    with pytest.raises(ValueError, match="reversed|empty"):
        reduce_run_to_dataset(ds.run)
    assert ds.run.histograms[0].n_bins == 32


@pytest.mark.parametrize("with_green", [False, True])
def test_mapped_dataset_and_rereduction_share_intersected_support(tmp_path, with_green):
    from asymmetry.core.io.periods import combine_mapped_periods

    path = tmp_path / "mapped.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "first_good_time", "last_good_time"):
            del d[name]
        d.create_dataset("counts", data=np.stack([raw, raw + 10, raw + 20]))
        d.create_dataset("first_good_time", data=[[0.22] * 2, [0.24] * 2, [0.22] * 2])
        d.create_dataset("last_good_time", data=[[0.30] * 2, [0.32] * 2, [0.30] * 2])
    periods = NexusLoader().load(path)
    mapping = {1: "red", 2: "red", 3: "green" if with_green else "ignore"}
    mapped = combine_mapped_periods(periods, mapping)
    expected_time = [0.0426, 0.0626, 0.0826]
    f, b = (2 * raw + 10)[:, 12:15]
    expected_a = 100 * (f - b) / (f + b)
    expected_e = 200 * np.sqrt(f * b / (f + b) ** 3)
    for ds in (mapped, reduce_run_to_dataset(mapped.run)):
        np.testing.assert_allclose(ds.time, expected_time, atol=1e-15)
        np.testing.assert_allclose(ds.asymmetry, expected_a)
        np.testing.assert_allclose(ds.error, expected_e)


def test_period_selection_does_not_run_missing_t0_search(tmp_path):
    from asymmetry.core.io.periods import select_period

    path = tmp_path / "missing-t0.nxs"
    raw = write_good_times(path)
    raw[:, 10] = 10000
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["time_zero"]
        del d["counts"]
        d.create_dataset("counts", data=np.stack([raw, raw + 10]))
    loaded = NexusLoader().load(path)
    for period in (1, 2):
        selected = select_period(loaded, period)
        assert selected.run.grouping["t0_source"] == "missing"
        assert selected.n_points == 0
        assert "t0" in selected.metadata["correction_error"]
        with pytest.raises(ValueError, match="t0"):
            reduce_run_to_dataset(selected.run)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_unverified_integer_period_shapes_use_remaining_bounds(tmp_path, version):
    path = tmp_path / "integer-shape.nxs"
    raw = write_good_times(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["counts"]
        d.create_dataset("counts", data=np.stack([raw, raw + 10, raw + 20])).attrs[
            "first_good_bin"
        ] = [[16, 17]] * 3
        if version == "v1":
            run = f.create_group("run")
            run.create_dataset("analysis", data=b"muonTD")
            f.move(d.name, "run/histogram_data_1")
            del f["raw_data_1"]
    datasets = NexusLoader().load(path)
    expected = (11, 14) if version == "v2" else (0, 31)
    for ds in datasets:
        assert [(h.good_bin_start, h.good_bin_end) for h in ds.run.histograms] == [expected] * 2


@pytest.mark.parametrize("histogram_state", ["absent", "partial", "malformed"])
def test_cached_only_period_selection_keeps_released_fallback(histogram_state):
    from asymmetry.core.io.periods import select_period
    from tests.io.test_period_selection import _combined_two_period

    combined = _combined_two_period()
    if histogram_state == "absent":
        del combined.run.grouping["period_histograms"]
    elif histogram_state == "partial":
        combined.run.grouping["period_histograms"] = combined.run.grouping["period_histograms"][:1]
    else:
        combined.run.grouping["period_histograms"][1] = ["unavailable"]
    selected = select_period(combined, "green")
    np.testing.assert_array_equal(selected.time, combined.run.grouping["period_reduced"][1][0])
    assert selected.run.metadata["period_number"] == 2
    for original, restored in zip(combined.run.histograms, selected.run.histograms, strict=True):
        np.testing.assert_array_equal(restored.counts, original.counts)
        assert restored is not original
