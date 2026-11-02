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
