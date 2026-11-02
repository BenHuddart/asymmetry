"""Independent unit-conversion and raw export oracles."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.io.physical_units import (
    UnitConvention,
    canonical,
    uniform_bin_width_us,
    unit_text,
)

pytestmark = pytest.mark.io


@pytest.mark.parametrize(
    "dimension,unit,value,expected",
    [
        ("time", "s", 1e-6, 1),
        ("time", "ms", 0.001, 1),
        ("time", "ns", 1000, 1),
        ("time", "ps", 1e6, 1),
        ("time", "MICROSECONDS", 1, 1),
        ("time", "micro.second", 1, 1),
        ("field", "G", 100, 100),
        ("field", "T", 0.01, 100),
        ("field", "mT", 10, 100),
        ("field", "uT", 100, 1),
        ("field", "kG", 0.5, 500),
        ("field", "Gauss", 100, 100),
        ("field", "milliTesla", 10, 100),
        ("temperature", "K", 5, 5),
        ("temperature", "mK", 50, 0.05),
        ("temperature", "degC", 20, 293.15),
        ("temperature", "°C", 20, 293.15),
        ("temperature", "Celsius", 20, 293.15),
    ],
)
def test_conversion_against_independent_si_values(dimension, unit, value, expected):
    converted, declaration = canonical(value, unit, dimension, context="synthetic: /value")
    assert float(converted) == pytest.approx(expected, rel=1e-14)
    assert declaration.source == "file"
    assert declaration.unit == unit


@pytest.mark.parametrize(
    "raw",
    ["us", "µs", "μs", b"\xb5s", "\udcb5s", " μs\x00 ", np.bytes_(b"\xb5s"), np.array([b"\xb5s"])],
)
def test_scalar_unit_decoding(raw):
    assert unit_text(raw) == "us"
    assert float(canonical(2, raw, "time", context="synthetic: /time")[0]) == 2


@pytest.mark.parametrize(
    "dimension,token",
    [
        ("field", "MT"),
        ("field", "Oe"),
        ("field", "K"),
        ("field", "A"),
        ("time", "parsec"),
        ("time", "K"),
        ("temperature", "F"),
        ("temperature", "G"),
        ("temperature", "k"),
        ("field", None),
    ],
)
def test_rejected_unit_names_file_dataset_token_and_supported_units(dimension, token):
    with pytest.raises(ValueError, match="synthetic: /value.*supported units") as error:
        canonical(1, token, dimension, context="synthetic: /value")
    assert (token or "<missing>") in str(error.value)


def test_explicit_units_override_recorded_format_convention():
    convention = UnitConvention("ps", "synthetic format specification")
    value, declaration = canonical(
        16000, None, "time", context="synthetic: /resolution", convention=convention
    )
    assert float(value) == pytest.approx(0.016)
    assert declaration.source == "format"
    assert declaration.reference == convention.reference
    value, declaration = canonical(
        16, "ns", "time", context="synthetic: /resolution", convention=convention
    )
    assert float(value) == pytest.approx(0.016)
    assert declaration.source == "file"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_uniform_axis_uses_storage_precision_and_end_to_end_span(dtype):
    axis = np.arange(2049, dtype=dtype) * dtype(0.016)
    expected = (float(axis[-1]) - float(axis[0])) / 2048
    width = uniform_bin_width_us(
        axis, kind="edges", n_bins=2048, stored_dtype=axis.dtype, context="synthetic: /raw_time"
    )
    assert width == expected
    assert width == pytest.approx(0.016, rel=1e-7)


@pytest.mark.parametrize(
    "axis,message",
    [
        (np.array([0, 0.02, np.nan]), "finite"),
        (np.array([0, 0, 0]), "increasing"),
        (np.array([0, 0.04, 0.02]), "increasing"),
        (np.array([0, 0.02, 0.05]), "uniform"),
        (np.array([0, 0.02]), "requires"),
        (np.array([[0, 0.02, 0.04]]), "requires"),
    ],
)
def test_invalid_time_axes_raise(axis, message):
    with pytest.raises(ValueError, match=message):
        uniform_bin_width_us(
            axis, kind="edges", n_bins=2, stored_dtype=axis.dtype, context="synthetic: /raw_time"
        )


def test_writer_declares_dimensioned_values_and_preserves_sub_bin_t0(tmp_path):
    h5py = pytest.importorskip("h5py")
    t0_values = [0.1234, 0.1271]
    run = Run(
        run_number=123,
        histograms=[
            Histogram(
                counts=np.full(64, 100.0),
                bin_width=0.02,
                t0_bin=6,
                t0_time_us=t0,
                good_bin_start=7,
                good_bin_end=63,
            )
            for t0 in t0_values
        ],
        metadata={"temperature": 1.5, "field": 100},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "dead_time_us": [0.01, 0.02],
        },
    )
    path = tmp_path / "units.nxs"
    write_nexus_v1(run, path)
    with h5py.File(path) as h:
        expected = {
            "sample/temperature": b"kelvin",
            "sample/magnetic_field": b"gauss",
            "histogram_data_1/corrected_time": b"microseconds",
            "histogram_data_1/time_zero": b"microseconds",
            "histogram_data_1/dead_time": b"microseconds",
            "histogram_data_1/resolution": b"picoseconds",
        }
        for key, unit in expected.items():
            assert h[f"run/{key}"].attrs["units"] == unit
    dataset = NexusLoader().load(path)
    assert dataset.metadata["temperature"] == 1.5
    assert dataset.metadata["field"] == 100
    assert [hist.t0_time_us for hist in dataset.run.histograms] == pytest.approx(t0_values)


def test_integer_axis_conversion_allows_only_floating_arithmetic_roundoff():
    axis = np.arange(33) * 20
    converted, _ = canonical(axis, "ns", "time", context="synthetic: /axis")
    width = uniform_bin_width_us(
        converted,
        kind="edges",
        n_bins=32,
        stored_dtype=axis.dtype,
        storage_scale=0.001,
        context="synthetic: /axis",
    )
    assert width == pytest.approx(0.02, rel=1e-14)
    converted[-1] += 0.001
    with pytest.raises(ValueError, match="uniform"):
        uniform_bin_width_us(
            converted,
            kind="edges",
            n_bins=32,
            stored_dtype=axis.dtype,
            storage_scale=0.001,
            context="synthetic: /axis",
        )


@pytest.mark.parametrize(
    "key,text,expected",
    [
        ("Time Resolution", "1.0e-6 s", 1.0),
        ("Time Resolution", "0.09765625 ns", 0.00009765625),
        ("Time Resolution", "97.65625 ps", 0.00009765625),
        ("Sample Magnetic Field", "10 mT", 100),
        ("Sample Magnetic Field", "0.5 kG", 500),
        ("Sample Magnetic Field", "100 uT", 1),
        ("Sample Temperature", "300 mK", 0.3),
        ("Sample Temperature", "20 +- 0.1 degC", 293.15),
    ],
)
def test_root_declared_quantity_units(tmp_path, key, text, expected):
    uproot = pytest.importorskip("uproot")
    from asymmetry.core.io.root import RootLoader
    from tests.io.test_root_loader import _write_musrroot_directory

    path = tmp_path / "units.root"
    _write_musrroot_directory(path)
    with uproot.update(path) as h:
        h[f"RunHeader/RunInfo/{key}"] = text
    dataset = RootLoader().load(path)
    if key == "Time Resolution":
        assert dataset.run.histograms[0].bin_width == pytest.approx(expected)
    else:
        assert dataset.metadata["field" if "Field" in key else "temperature"] == pytest.approx(
            expected
        )
    dimension = {
        "Time Resolution": "time",
        "Sample Magnetic Field": "field",
        "Sample Temperature": "temperature",
    }[key]
    assert dataset.metadata["declared_units"][dimension][f"RunInfo/{key}"]["source"] == "file"


@pytest.mark.parametrize("text", ["", "0.2 parsec", "0.2 K"])
def test_root_missing_or_invalid_resolution_raises(tmp_path, text):
    uproot = pytest.importorskip("uproot")
    from asymmetry.core.io.root import RootLoader
    from tests.io.test_root_loader import _write_musrroot_directory

    path = tmp_path / "units.root"
    _write_musrroot_directory(path)
    with uproot.update(path) as h:
        h["RunHeader/RunInfo/Time Resolution"] = text
    with pytest.raises(ValueError) as error:
        RootLoader().load(path)
    assert str(path) in str(error.value)
    assert "Time Resolution" in str(error.value)


@pytest.mark.parametrize(
    "field,temperature,expected_field,expected_temperature",
    [(b"10mT", b"300mK", 100, 0.3), (b"TF 100", b"5", 100, 5), (b"0.5kG", b"20degC", 500, 293.15)],
)
def test_psi_adjacent_units_and_unverified_missing_convention(
    tmp_path, field, temperature, expected_field, expected_temperature
):
    from asymmetry.core.io.psi import PsiLoader
    from tests.io.test_psi_loader import _write_psi_bin

    path = tmp_path / "units.bin"
    _write_psi_bin(path, field=field)
    content = bytearray(path.read_bytes())
    content[148:158] = temperature.ljust(10, b" ")
    path.write_bytes(content)
    dataset = PsiLoader().load(path)
    assert dataset.metadata["field"] == pytest.approx(expected_field)
    assert dataset.metadata["temperature"] == pytest.approx(expected_temperature)
    declaration = dataset.metadata["declared_units"]["field"]["header/field"]
    if field == b"TF 100":
        assert declaration["source"] == "format"
        assert "unverified" in declaration["reference"]
    else:
        assert declaration["source"] == "file"


@pytest.mark.parametrize(
    "key,text",
    [
        ("Sample Magnetic Field", "1 MT"),
        ("Sample Magnetic Field", "1 Oe"),
        ("Sample Magnetic Field", "1 K"),
        ("Sample Magnetic Field", "1"),
        ("Sample Temperature", "1 F"),
        ("Sample Temperature", "1"),
    ],
)
def test_root_invalid_metadata_unit_rejected(tmp_path, key, text):
    uproot = pytest.importorskip("uproot")
    from asymmetry.core.io.root import RootLoader
    from tests.io.test_root_loader import _write_musrroot_directory

    path = tmp_path / "bad.root"
    _write_musrroot_directory(path)
    with uproot.update(path) as h:
        h[f"RunHeader/RunInfo/{key}"] = text
    with pytest.raises(ValueError, match="supported units") as error:
        RootLoader().load(path)
    assert str(path) in str(error.value)
    assert key in str(error.value)


def test_root_sensor_conversion_and_primary_selection_use_the_same_kelvin_scale(tmp_path):
    uproot = pytest.importorskip("uproot")
    from asymmetry.core.io.root import RootLoader
    from tests.io.test_root_loader import _write_flame_root_directory

    path = tmp_path / "sensor.root"
    _write_flame_root_directory(path)
    with uproot.update(path) as h:
        h["RunHeader/RunInfo/Sample Temperature"] = "10.4 +- 0.1 mK; Sens=SAM_ts_value"
    dataset = RootLoader().load(path)
    series = dataset.metadata["nexus_time_series"]
    sample = series["musrroot_slow_control/flamesam0 SAM ts value"]
    other = series["musrroot_slow_control/flamedil0 DIL T mix value"]
    assert dataset.metadata["temperature"] == pytest.approx(0.0104)
    assert sample["mean"] == pytest.approx(0.0106)
    assert sample["units"] == "K"
    assert sample["unit_declaration"]["unit"] == "mK"
    assert sample["primary"] is True
    assert other["primary"] is False


def test_mdu_mean_temperature_precedence_records_its_actual_field():
    from asymmetry.core.io.psi_units import read_header_quantities

    quantities = read_header_quantities("synthetic", "25K", "10mT", mean_temperature="300mK")
    assert quantities.temperature == pytest.approx(0.3)
    assert quantities.field == 100
    assert quantities.declarations["temperature"]["header/mean_temp"]["unit"] == "mK"


@pytest.mark.parametrize("text", [b"10Oe", b"10MT", b"10K"])
def test_psi_incompatible_field_unit_raises(tmp_path, text):
    from asymmetry.core.io.psi import PsiLoader
    from tests.io.test_psi_loader import _write_psi_bin

    path = tmp_path / "bad.bin"
    _write_psi_bin(path, field=text)
    with pytest.raises(ValueError, match="supported units") as error:
        PsiLoader().load(path)
    assert str(path) in str(error.value)
    assert "header/field" in str(error.value)
