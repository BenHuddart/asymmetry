"""Independent unit and timing oracles for raw-file boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader

h5py = pytest.importorskip("h5py")
pytestmark = pytest.mark.io


def _nexus(path, *, version=2, axis=None, axis_unit="us", t0=0.2074, t0_unit="us", n_bins=32):
    with h5py.File(path, "w") as h:
        entry = h.create_group("run" if version == 1 else "raw_data_1")
        entry.create_dataset("IDF_version", data=version)
        entry.create_dataset("number" if version == 1 else "run_number", data=123)
        detector = entry.create_group(
            "histogram_data_1" if version == 1 else "instrument/detector_1"
        )
        detector.create_dataset("counts", data=np.full((2, n_bins), 100))
        detector.create_dataset("grouping", data=[1, 2])
        if axis is None:
            axis = np.arange(n_bins + 1) * 0.02
        d = detector.create_dataset("corrected_time" if version == 1 else "raw_time", data=axis)
        if axis_unit is not None:
            d.attrs["units"] = axis_unit
        d = detector.create_dataset("time_zero", data=[t0, t0])
        d.attrs["units"] = t0_unit
        sample = entry.create_group("sample")
        for key, value, unit in (("temperature", 1.5, "K"), ("magnetic_field", 100, "G")):
            sample.create_dataset(key, data=value).attrs["units"] = unit


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "unit,scale",
    [("us", 1), ("microseconds", 1), (b"\xb5s", 1), ("μs", 1), ("ns", 1000), ("s", 1e-6)],
)
def test_nexus_time_axis_units_equivalent(tmp_path, version, unit, scale):
    path = tmp_path / "units.nxs"
    _nexus(
        path,
        version=version,
        axis=np.arange(33) * 0.02 * scale,
        axis_unit=unit,
        t0=0.2074 * scale,
        t0_unit=unit,
    )
    dataset = NexusLoader().load(path)
    histogram = dataset.run.histograms[0]
    assert histogram.bin_width == pytest.approx(0.02, rel=1e-12)
    assert histogram.t0_bin == 10
    assert histogram.t0_time_us == pytest.approx(0.2074)
    np.testing.assert_allclose(dataset.time, (np.arange(32) + 0.5) * 0.02 - 0.2074, atol=1e-14)


@pytest.mark.parametrize(
    "unit,value,expected",
    [
        ("G", 100, 100),
        ("T", 0.01, 100),
        ("mT", 10, 100),
        ("uT", 100, 1),
        ("kG", 0.5, 500),
        ("Gauss", 100, 100),
    ],
)
def test_nexus_field_units(tmp_path, unit, value, expected):
    path = tmp_path / "field.nxs"
    _nexus(path)
    with h5py.File(path, "r+") as h:
        sample = h["raw_data_1/sample"]
        del sample["magnetic_field"]
        sample.create_dataset("magnetic_field", data=value).attrs["units"] = unit
    assert NexusLoader().load(path).metadata["field"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "unit,value,expected", [("K", 5, 5), ("mK", 50, 0.05), ("degC", 20, 293.15), ("°C", 20, 293.15)]
)
def test_nexus_temperature_units(tmp_path, unit, value, expected):
    path = tmp_path / "temperature.nxs"
    _nexus(path)
    with h5py.File(path, "r+") as h:
        sample = h["raw_data_1/sample"]
        del sample["temperature"]
        sample.create_dataset("temperature", data=value).attrs["units"] = unit
    assert NexusLoader().load(path).metadata["temperature"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "key,unit",
    [
        ("magnetic_field", "MT"),
        ("magnetic_field", "Oe"),
        ("magnetic_field", "K"),
        ("magnetic_field", None),
        ("temperature", "F"),
        ("temperature", "G"),
        ("temperature", None),
    ],
)
def test_nexus_metadata_unit_rejections(tmp_path, key, unit):
    path = tmp_path / "bad.nxs"
    _nexus(path)
    with h5py.File(path, "r+") as h:
        d = h[f"raw_data_1/sample/{key}"]
        if unit is None:
            del d.attrs["units"]
        else:
            d.attrs["units"] = unit
    with pytest.raises(ValueError, match="supported") as error:
        NexusLoader().load(path)
    assert str(path) in str(error.value)
    assert key in str(error.value)
    assert str(unit) in str(error.value) if unit else "missing" in str(error.value)


@pytest.mark.parametrize(
    "axis",
    [
        np.r_[np.arange(32) * 0.02, np.nan],
        np.zeros(33),
        np.r_[0, 0.04, 0.02, np.arange(3, 33) * 0.02],
        np.r_[np.arange(32) * 0.02, 0.65],
        np.arange(30) * 0.02,
        np.array([]),
    ],
)
def test_nexus_invalid_axis_rejected(tmp_path, axis):
    path = tmp_path / "axis.nxs"
    _nexus(path, axis=axis)
    with pytest.raises(ValueError, match="time|axis"):
        NexusLoader().load(path)


def test_nexus_float32_axis_retains_stored_precision(tmp_path):
    path = tmp_path / "float32.nxs"
    axis = np.arange(2049, dtype=np.float32) * np.float32(0.016)
    _nexus(path, axis=axis, n_bins=2048, t0=0)
    dataset = NexusLoader().load(path)
    assert dataset.run.histograms[0].bin_width == pytest.approx(0.016, rel=1e-7)


@pytest.mark.parametrize(
    "version,value,unit", [(1, 16000, None), (1, 16, "ns"), (2, 16, None), (2, 16000, "ps")]
)
def test_nexus_resolution_fallback_records_versioned_convention(tmp_path, version, value, unit):
    path = tmp_path / "resolution.nxs"
    _nexus(path, version=version)
    prefix = "run/histogram_data_1" if version == 1 else "raw_data_1/instrument/detector_1"
    with h5py.File(path, "r+") as h:
        detector = h[prefix]
        del detector["corrected_time" if version == 1 else "raw_time"]
        d = detector.create_dataset("resolution", data=value)
        if unit:
            d.attrs["units"] = unit
    dataset = NexusLoader().load(path)
    assert dataset.run.histograms[0].bin_width == pytest.approx(0.016)
    declaration = dataset.metadata["declared_units"]["time"][f"/{prefix}/resolution"]
    assert declaration["source"] == ("file" if unit else "format")
    if not unit:
        assert declaration["reference"].startswith("https:")


def test_nexus_resolution_cross_check_accepts_half_unit_quantization(tmp_path):
    path = tmp_path / "quantized.nxs"
    _nexus(path, axis=np.arange(33) * 0.00390625, t0=0)
    with h5py.File(path, "r+") as h:
        h["raw_data_1/instrument/detector_1"].create_dataset("resolution", data=3906).attrs[
            "units"
        ] = "ps"
    assert NexusLoader().load(path).run.histograms[0].bin_width == pytest.approx(0.00390625)
    with h5py.File(path, "r+") as h:
        h["raw_data_1/instrument/detector_1/resolution"][()] = 3905
    with pytest.raises(ValueError, match="resolution disagrees"):
        NexusLoader().load(path)


def test_nexus_axes_must_agree_even_with_rounded_resolution(tmp_path):
    path = tmp_path / "axes.nxs"
    _nexus(path, t0=0)
    with h5py.File(path, "r+") as h:
        detector = h["raw_data_1/instrument/detector_1"]
        detector.create_dataset("corrected_time", data=np.arange(33) * 0.02000001).attrs[
            "units"
        ] = "us"
        detector.create_dataset("resolution", data=20000).attrs["units"] = "ps"
    with pytest.raises(ValueError, match="time axes disagree"):
        NexusLoader().load(path)


def test_nexus_one_centre_requires_and_uses_declared_resolution(tmp_path):
    path = tmp_path / "one.nxs"
    _nexus(path, n_bins=1, axis=np.array([0.01]), t0=0)
    with pytest.raises(ValueError, match="no usable time axis or resolution"):
        NexusLoader().load(path)
    with h5py.File(path, "r+") as h:
        h["raw_data_1/instrument/detector_1"].create_dataset("resolution", data=20).attrs[
            "units"
        ] = "ns"
    dataset = NexusLoader().load(path)
    assert dataset.run.histograms[0].bin_width == pytest.approx(0.02)
    assert dataset.n_points == 1


def test_nexus_good_times_and_deadtime_use_declared_time_units(tmp_path):
    path = tmp_path / "good.nxs"
    _nexus(path, t0=0)
    with h5py.File(path, "r+") as h:
        detector = h["raw_data_1/instrument/detector_1"]
        for name, value in (
            ("first_good_time", 40),
            ("last_good_time", 100),
            ("dead_time", [10, 20]),
        ):
            detector.create_dataset(name, data=value).attrs["units"] = "ns"
    dataset = NexusLoader().load(path)
    np.testing.assert_allclose(dataset.time, [0.05, 0.07, 0.09])
    assert dataset.run.grouping["dead_time_us"] == pytest.approx([0.01, 0.02])


def test_nexus_main_and_sweep_fields_are_added_in_gauss(tmp_path):
    path = tmp_path / "main.nxs"
    _nexus(path)
    with h5py.File(path, "r+") as h:
        entry = h["raw_data_1"]
        entry["sample/magnetic_field"][()] = 1
        for name, value, unit in (("Field_Main", 0.01, "T"), ("Field_Z", 1, "G")):
            log = entry.create_group(name)
            log.create_dataset("time", data=[0, 1]).attrs["units"] = "seconds"
            log.create_dataset("value", data=[value, value]).attrs["units"] = unit
    dataset = NexusLoader().load(path)
    assert dataset.metadata["field"] == pytest.approx(101)
    assert dataset.metadata["field_main_gauss"] == pytest.approx(100)
    assert dataset.metadata["nexus_time_series"]["Field_Main"]["units"] == "G"


@pytest.mark.parametrize(
    "name,unit", [("Field_Main", "K"), ("Temp_Sample", "min"), ("Temp_Sample", None)]
)
def test_nexus_incompatible_physical_log_declarations_raise(tmp_path, name, unit):
    path = tmp_path / "log.nxs"
    _nexus(path)
    with h5py.File(path, "r+") as h:
        log = h["raw_data_1"].create_group(name)
        log.create_dataset("time", data=[0, 1])
        d = log.create_dataset("value", data=[1.0, 1.0])
        if unit:
            d.attrs["units"] = unit
    with pytest.raises(ValueError, match="supported units") as error:
        NexusLoader().load(path)
    assert str(path) in str(error.value)
    assert name in str(error.value)


def test_nexus_log_clock_uses_its_unit_and_scaling_factor(tmp_path):
    path = tmp_path / "clock.nxs"
    _nexus(path)
    with h5py.File(path, "r+") as h:
        log = h["raw_data_1"].create_group("Temp_Sample")
        clock = log.create_dataset("time", data=[0, 500])
        clock.attrs["units"] = "ms"
        clock.attrs["scaling_factor"] = 2
        log.create_dataset("value", data=[500, 500]).attrs["units"] = "mK"
    dataset = NexusLoader().load(path)
    series = dataset.metadata["nexus_time_series"]["Temp_Sample"]
    np.testing.assert_allclose(series["time"], [0, 1])
    assert series["time_units"] == "s"
    assert series["mean"] == 0.5
    declaration = dataset.metadata["declared_units"]["time"]["/raw_data_1/Temp_Sample/time"]
    assert declaration["unit"] == "ms"
    assert declaration["scaling_factor"] == 2
    with h5py.File(path, "r+") as h:
        h["raw_data_1/Temp_Sample/time"].attrs["scaling_factor"] = np.nan
    with pytest.raises(ValueError, match="scaling_factor must be a finite scalar"):
        NexusLoader().load(path)
