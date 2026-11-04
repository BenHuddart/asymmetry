"""Declared acquisition timing remains usable across auxiliary axis variants."""

import h5py
import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader

pytestmark = pytest.mark.io


def write_timing(path, *, n_bins=64, exact=0.4, dtype=np.float32):
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data_1")
        entry.create_dataset("definition", data=b"muonTD")
        detector = entry.create_group("instrument/detector_1")
        counts = detector.create_dataset(
            "counts", data=np.arange(2 * n_bins).reshape(2, n_bins) + 10
        )
        counts.attrs["t0_bin"] = 26
        counts.attrs["first_good_bin"] = 1
        counts.attrs["last_good_bin"] = n_bins
        detector.create_dataset("grouping", data=[1, 2])
        detector.create_dataset("resolution", data=16000).attrs["units"] = "picoseconds"
        detector.create_dataset("raw_time", data=(np.arange(n_bins + 1) * 0.016).astype(dtype))
        detector.create_dataset(
            "corrected_time", data=((np.arange(n_bins) + 0.5) * 0.016 - dtype(0.4)).astype(dtype)
        )
        detector.create_dataset("time_zero", data=dtype(exact)).attrs["units"] = "microseconds"


@pytest.mark.parametrize("n_bins", [64, 4096])
def test_declared_resolution_preserves_float32_exact_timing(tmp_path, n_bins):
    path = tmp_path / "timing.nxs"
    write_timing(path, n_bins=n_bins)
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    for i, h in enumerate(ds.run.histograms):
        assert h.bin_width == 0.016
        assert h.t0_bin == 25
        assert h.t0_time_us == float(np.float32(0.4))
        np.testing.assert_array_equal(h.counts, np.arange(n_bins) + i * n_bins + 10)


def test_displaced_exact_time_remains_conflict(tmp_path):
    path = tmp_path / "conflict.nxs"
    write_timing(path, exact=0.5)
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "conflict"
    assert all(h.t0_time_us is None for h in ds.run.histograms)


@pytest.mark.parametrize(
    "unit,scale", [("us", 1), ("ns", 1000), ("ps", 1e6), ("s", 1e-6), (np.bytes_(b"\xb5s"), 1)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("edges", [False, True])
def test_equivalent_declarations_and_axis_representations(tmp_path, unit, scale, dtype, edges):
    path = tmp_path / "units.nxs"
    write_timing(path, dtype=dtype)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("raw_time", "corrected_time", "time_zero"):
            del d[name]
        raw = np.arange(65 if edges else 64) * 0.016 + (0 if edges else 0.008)
        corrected = np.arange(65 if edges else 64) * 0.016 + (0 if edges else 0.008) - 0.4
        for name, values in [("raw_time", raw), ("corrected_time", corrected), ("time_zero", 0.4)]:
            d.create_dataset(name, data=np.asarray(np.asarray(values) * scale, dtype=dtype)).attrs[
                "units"
            ] = unit
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    assert ds.run.histograms[0].bin_width == 0.016
    assert ds.run.histograms[0].t0_time_us == pytest.approx(0.4, abs=1e-7)


@pytest.mark.parametrize("version,value", [("v1", 16000), ("v2", 16)])
def test_resolution_only_uses_dialect_convention(tmp_path, version, value):
    path = tmp_path / "resolution.nxs"
    write_timing(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("raw_time", "corrected_time", "resolution"):
            del d[name]
        d.create_dataset("resolution", data=value)
        if version == "v1":
            run = f.create_group("run")
            run.create_dataset("analysis", data=b"muonTD")
            f.move(d.name, "/run/histogram_data_1")
            del f["raw_data_1"]
    ds = NexusLoader().load(path)
    assert ds.run.histograms[0].bin_width == 0.016
    assert ds.run.grouping["t0_source"] == "file"


@pytest.mark.parametrize(
    "fault", ["cumulative", "terminal", "displaced", "curved", "nan", "shape", "units"]
)
def test_declared_resolution_accepts_secondary_coordinate_variants(tmp_path, fault):
    from asymmetry.core.io.quick_asymmetry import quick_asymmetry

    path = tmp_path / "coordinates.nxs"
    write_timing(path, n_bins=2000)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        if fault == "cumulative":
            d["raw_time"][:] = np.concatenate(
                ([0], np.cumsum(np.full(2000, 0.016, dtype=np.float32)))
            )
        elif fault == "terminal":
            d["raw_time"][-1] -= 0.0025
            d["corrected_time"][-1] -= 0.0025
        elif fault == "displaced":
            d["corrected_time"][:] += 0.1
        elif fault == "curved":
            d["raw_time"][:] += 0.0005 * np.sin(np.pi * np.arange(2001) / 2000)
        elif fault == "nan":
            d["raw_time"][10] = np.nan
        elif fault == "shape":
            del d["raw_time"]
            d.create_dataset("raw_time", data=np.arange(12).reshape(3, 4))
        else:
            d["raw_time"].attrs["units"] = "K"
        original_raw = d["raw_time"][()]
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    assert ds.metadata["nexus_timing_source"] == "resolution"
    quick = quick_asymmetry(ds.run.histograms, ds.run.grouping)
    np.testing.assert_array_equal(ds.time, quick.time)
    np.testing.assert_array_equal(ds.asymmetry, quick.asymmetry)
    np.testing.assert_array_equal(ds.error, quick.error)
    assert ds.time.size == 2000
    for i, h in enumerate(ds.run.histograms):
        assert h.bin_width == 0.016
        assert h.t0_time_us == float(np.float32(0.4))
        np.testing.assert_array_equal(h.counts, np.arange(2000) + i * 2000 + 10)
    stored = ds.metadata["nexus_fields"]["instrument"]["detector_1"]["raw_time"]
    np.testing.assert_array_equal(stored["value"], original_raw)


@pytest.mark.parametrize("exact", [True, False])
def test_zero_is_valid_and_integer_only_is_sufficient(tmp_path, exact):
    path = tmp_path / "zero.nxs"
    write_timing(path, exact=0)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["corrected_time"]
        d["counts"].attrs["t0_bin"] = 1
        if not exact:
            del d["time_zero"]
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    assert ds.run.histograms[0].t0_time_us == (0 if exact else None)


def test_secondary_raw_origin_does_not_shift_declared_timing(tmp_path):
    path = tmp_path / "offset.nxs"
    write_timing(path, dtype=np.float64)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        d["raw_time"][:] += 2.0
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    assert ds.run.histograms[0].t0_time_us == 0.4
    np.testing.assert_allclose(ds.time, (np.arange(64) + 0.5) * 0.016 - 0.4, atol=1e-15)


def test_without_resolution_uses_validated_span(tmp_path):
    path = tmp_path / "axis-only.nxs"
    write_timing(path, n_bins=4096)
    with h5py.File(path, "a") as f:
        del f["raw_data_1/instrument/detector_1/resolution"]
    ds = NexusLoader().load(path)
    expected = float(np.float32(4096 * 0.016)) / 4096
    assert ds.run.histograms[0].bin_width == expected
    assert ds.run.grouping["t0_source"] == "file"


def test_period_detector_timing_retains_each_origin(tmp_path):
    path = tmp_path / "periods.nxs"
    write_timing(path)
    times = np.array([[0.4, 0.424], [0.504, 0.52], [0.616, 0.632]])
    bins = np.floor(times / 0.016).astype(int)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        counts = d["counts"][()]
        for name in ("counts", "time_zero", "corrected_time"):
            del d[name]
        d.create_dataset("counts", data=np.stack([counts + 1000 * i for i in range(3)])).attrs[
            "t0_bin"
        ] = bins + 1
        d.create_dataset("time_zero", data=times)
    datasets = NexusLoader().load(path)
    for period, ds in enumerate(datasets):
        assert ds.run.grouping["t0_source"] == "file"
        assert [h.t0_time_us for h in ds.run.histograms] == times[period].tolist()
        assert [h.t0_bin for h in ds.run.histograms] == bins[period].tolist()


def test_loaded_and_reduced_selected_pair_share_crop_and_exact_reference(tmp_path):
    from asymmetry.core.io.quick_asymmetry import quick_asymmetry
    from asymmetry.core.transform.grouping import group_forward_backward

    path = tmp_path / "selected.nxs"
    write_timing(path, dtype=np.float64)
    counts = np.arange(192).reshape(3, 64) + 10
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "grouping", "time_zero", "corrected_time"):
            del d[name]
        d.create_dataset("counts", data=counts).attrs["t0_bin"] = [26, 27, 57]
        d.create_dataset("time_zero", data=[0.4, 0.424, 0.9])
        d.create_dataset("grouping", data=[1, 2, 3])
    ds = NexusLoader().load(path)
    quick = quick_asymmetry(ds.run.histograms, ds.run.grouping)
    assert quick.alignment.first_bin == 1
    assert quick.alignment.common_t0_bin == 25
    np.testing.assert_allclose(ds.time, (np.arange(63) + 1.5) * 0.016 - 0.424, atol=1e-15)
    np.testing.assert_array_equal(ds.time, quick.time)
    np.testing.assert_array_equal(ds.asymmetry, quick.asymmetry)
    grouped = group_forward_backward(ds.run.histograms, ds.run.grouping)
    np.testing.assert_array_equal(grouped.forward, counts[0, :-1])
    np.testing.assert_array_equal(grouped.backward, counts[1, 1:])
    other = quick_asymmetry(ds.run.histograms, dict(ds.run.grouping, backward_group=3))
    assert other.alignment.first_bin == 31
    np.testing.assert_allclose(other.time, (np.arange(33) + 31.5) * 0.016 - 0.9, atol=1e-15)
    excluded = quick_asymmetry(ds.run.histograms, dict(ds.run.grouping, excluded_detectors=[3]))
    np.testing.assert_array_equal(excluded.time, quick.time)
    for i, h in enumerate(ds.run.histograms):
        np.testing.assert_array_equal(h.counts, counts[i])


def test_single_bin_centres_use_declared_width(tmp_path):
    path = tmp_path / "single.nxs"
    write_timing(path, n_bins=1, exact=0)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["raw_time"]
        d.create_dataset("raw_time", data=[0.008])
        d["corrected_time"][:] = [0.008]
        d["counts"].attrs["t0_bin"] = 1
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    assert ds.run.histograms[0].bin_width == 0.016
    np.testing.assert_array_equal(ds.time, [0.008])


@pytest.mark.parametrize("resolution", [0, -1, np.nan])
def test_unusable_resolution_uses_supported_axis(tmp_path, resolution):
    path = tmp_path / "fallback.nxs"
    write_timing(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["resolution"]
        d.create_dataset("resolution", data=resolution)
    ds = NexusLoader().load(path)
    assert ds.metadata["nexus_timing_source"] == "raw_time"
    assert ds.run.histograms[0].bin_width == float(np.float32(64 * 0.016)) / 64
    assert ds.time.size == 64


@pytest.mark.parametrize("fault", ["nonfinite", "shape", "units", "integer"])
def test_unusable_optional_t0_uses_remaining_calibration(tmp_path, fault, caplog):
    path = tmp_path / "optional.nxs"
    write_timing(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        if fault == "nonfinite":
            d["time_zero"][()] = np.nan
        elif fault == "shape":
            del d["time_zero"]
            d.create_dataset("time_zero", data=[0.4, 0.4, 0.4])
        elif fault == "units":
            d["time_zero"].attrs["units"] = "K"
        else:
            d["counts"].attrs["t0_bin"] = 25.5
    ds = NexusLoader().load(path)
    assert ds.run.grouping["t0_source"] == "file"
    assert ds.time.size == 64
    assert [h.t0_bin for h in ds.run.histograms] == [25, 25]
    if fault != "integer":
        assert all(h.t0_time_us is None for h in ds.run.histograms)
    assert "unusable" in caplog.text


@pytest.mark.parametrize(
    "fault,match",
    [
        ("shape", "no usable"),
        ("uniform", "no usable"),
        ("nonfinite", "no usable"),
        ("missing", "no usable"),
        ("t0", "no usable t0"),
    ],
)
def test_no_usable_required_timing_has_contextual_failure(tmp_path, fault, match):
    path = tmp_path / "unresolved.nxs"
    write_timing(path)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        if fault == "t0":
            d["counts"].attrs["t0_bin"] = -1
            d["time_zero"][()] = np.nan
        else:
            del d["resolution"]
            del d["corrected_time"]
            if fault == "shape":
                del d["raw_time"]
                d.create_dataset("raw_time", data=[0, 1, 2])
            elif fault == "uniform":
                d["raw_time"][10] += 0.003
            elif fault == "nonfinite":
                d["raw_time"][10] = np.nan
            else:
                del d["raw_time"]
    with pytest.raises(ValueError, match=match) as exc:
        NexusLoader().load(path)
    assert str(path) in str(exc.value)


def test_normal_writer_rounding_does_not_warn(tmp_path, caplog):
    path = tmp_path / "ordinary.nxs"
    write_timing(path, n_bins=4096)
    NexusLoader().load(path)
    assert not caplog.records


@pytest.mark.parametrize("fault", ["shape", "nonuniform"])
def test_axis_fallback_skips_unusable_candidate(tmp_path, fault, caplog):
    path = tmp_path / "alternate-axis.nxs"
    write_timing(path, dtype=np.float64)
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        del d["resolution"]
        if fault == "shape":
            del d["raw_time"]
            d.create_dataset("raw_time", data=[0, 1, 2])
        else:
            d["raw_time"][10] += 0.003
    ds = NexusLoader().load(path)
    assert ds.time.size == 64
    assert ds.metadata["nexus_timing_source"] == "corrected_time"
    assert ds.run.histograms[0].bin_width == pytest.approx(0.016, abs=1e-15)
    assert "raw_time" in caplog.text


@pytest.mark.parametrize(
    "field", ["resolution", "raw_time", "time_of_flight", "corrected_time", "time_zero"]
)
@pytest.mark.parametrize("tokens", [[], [b"us", b"ns"]])
def test_non_scalar_units_do_not_abort_timing_alternatives(tmp_path, field, tokens, caplog):
    path = tmp_path / "unit-array.nxs"
    write_timing(path)
    with h5py.File(path, "a") as file:
        detector = file["raw_data_1/instrument/detector_1"]
        if field in {"raw_time", "time_of_flight", "corrected_time"}:
            del detector["resolution"]
        if field == "time_of_flight":
            detector.move("raw_time", "time_of_flight")
        if field == "corrected_time":
            del detector["raw_time"]
        detector[field].attrs["units"] = np.asarray(tokens, dtype="S2")
    if field == "corrected_time":
        with pytest.raises(ValueError, match="no usable time axis or positive resolution"):
            NexusLoader().load(path)
    else:
        dataset = NexusLoader().load(path)
        assert dataset.time.size == 64
        assert dataset.run.grouping["t0_source"] == "file"
        expected_source = (
            "resolution"
            if field == "time_zero"
            else "raw_time"
            if field == "resolution"
            else "corrected_time"
        )
        assert dataset.metadata["nexus_timing_source"] == expected_source
        if field == "time_zero":
            assert all(hist.t0_time_us is None for hist in dataset.run.histograms)
        for i, hist in enumerate(dataset.run.histograms):
            np.testing.assert_array_equal(hist.counts, np.arange(64) + i * 64 + 10)
    assert f"/{field}: unusable timing declaration" in caplog.text
