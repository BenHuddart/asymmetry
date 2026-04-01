"""Tests for ISIS NeXus loader (V1/V2 and multiperiod support)."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from asymmetry.core.io.nexus import NexusLoader


@pytest.fixture()
def loader() -> NexusLoader:
    return NexusLoader()


def _write_v2_file(
    path,
    *,
    multiperiod: bool = False,
    include_corrected_time: bool = False,
    time_zero_us: float = 0.0,
    t0_bin_attr: int | None = None,
) -> None:
    """Create a synthetic V2 NeXus file used by loader unit tests.

    Parameters control whether the file carries a pre-corrected time axis and
    whether ``t0_bin`` is provided explicitly as an attribute.
    """
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data_1")
        entry.create_dataset("definition", data=np.bytes_("muonTD"))
        entry.create_dataset("IDF_version", data=2)
        entry.create_dataset("run_number", data=12345)
        entry.create_dataset("good_frames", data=np.array([200000], dtype=np.int32))
        entry.create_dataset("title", data=np.bytes_("V2 Test"))
        entry.create_dataset("start_time", data=np.bytes_("2026-03-15T10:00:00"))
        entry.create_dataset("end_time", data=np.bytes_("2026-03-15T11:00:00"))
        entry.create_dataset("name", data=np.bytes_("EMU"))

        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector_1")

        if multiperiod:
            counts = np.array(
                [
                    [
                        [100, 120, 140, 160],
                        [80, 95, 110, 130],
                    ],
                    [
                        [90, 110, 130, 150],
                        [70, 85, 95, 120],
                    ],
                ],
                dtype=np.float64,
            )
        else:
            counts = np.array(
                [
                    [100, 120, 140, 160],
                    [80, 95, 110, 130],
                ],
                dtype=np.float64,
            )

        counts_ds = detector.create_dataset("counts", data=counts)
        counts_ds.attrs["first_good_bin"] = np.bytes_("0")
        counts_ds.attrs["last_good_bin"] = np.bytes_("3")
        if t0_bin_attr is not None:
            counts_ds.attrs["t0_bin"] = np.int32(t0_bin_attr)

        raw_time = np.array([0.0, 0.02, 0.04, 0.06, 0.08], dtype=np.float64)
        detector.create_dataset("raw_time", data=raw_time)
        if include_corrected_time:
            corrected_time = 0.5 * (raw_time[:-1] + raw_time[1:]) - float(time_zero_us)
            detector.create_dataset("corrected_time", data=np.asarray(corrected_time, dtype=np.float64))
        detector.create_dataset("grouping", data=np.array([1, 2], dtype=np.int32))
        detector.create_dataset("dead_time", data=np.array([0.01, 0.02], dtype=np.float64))
        detector.create_dataset("time_zero", data=np.array([time_zero_us, time_zero_us], dtype=np.float64))
        detector.create_dataset("orientation", data=np.bytes_("L"))

        sample = entry.create_group("sample")
        sample.create_dataset("temperature", data=12.5)
        sample.create_dataset("magnetic_field", data=150.0)

        temp_log = sample.create_group("Temp_Sample")
        temp_log.create_dataset("time", data=np.array([0.0, 10.0, 20.0], dtype=np.float64))
        temp_log.create_dataset("value", data=np.array([12.0, 12.5, 13.0], dtype=np.float64))

        if multiperiod:
            periods = entry.create_group("periods")
            periods.create_dataset("number", data=2)


def _write_v1_file(path) -> None:
    with h5py.File(path, "w") as f:
        run = f.create_group("run")
        run.create_dataset("analysis", data=np.bytes_("muonTD"))
        run.create_dataset("IDF_version", data=1)
        run.create_dataset("number", data=2468)
        run.create_dataset("good_frames", data=np.array([100000], dtype=np.int32))
        run.create_dataset("title", data=np.bytes_("V1 Test"))
        run.create_dataset("start_time", data=np.bytes_("2026-03-15T10:00:00"))
        run.create_dataset("stop_time", data=np.bytes_("2026-03-15T11:00:00"))

        instrument = run.create_group("instrument")
        detector = instrument.create_group("detector")
        detector.create_dataset("orientation", data=np.bytes_("T"))

        sample = run.create_group("sample")
        sample.create_dataset("temperature", data=5.0)
        sample.create_dataset("magnetic_field", data=20.0)

        h_data = run.create_group("histogram_data_1")
        h_data.create_dataset(
            "counts",
            data=np.array(
                [
                    [120, 140, 160, 180],
                    [100, 120, 130, 150],
                ],
                dtype=np.float64,
            ),
        )
        h_data.create_dataset("corrected_time", data=np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64))
        h_data.create_dataset("grouping", data=np.array([1, 2], dtype=np.int32))
        h_data.create_dataset("dead_time", data=np.array([0.015, 0.025], dtype=np.float64))


def test_load_v2_single_period(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2.nxs"
    _write_v2_file(path, multiperiod=False)

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.metadata["nexus_version"] == "v2"
    assert ds.metadata["run_number"] == 12345
    assert ds.metadata["title"] == "V2 Test"
    assert ds.metadata["temperature"] == pytest.approx(12.5)
    assert ds.metadata["field"] == pytest.approx(150.0)
    assert "nexus_fields" in ds.metadata
    assert "nexus_time_series" in ds.metadata
    assert ds.n_points == 4
    # Values are percentages, not fractions.
    expected_first = 100.0 * (100.0 - 80.0) / (100.0 + 80.0)
    assert ds.asymmetry[0] == pytest.approx(expected_first)
    assert np.nanmax(np.abs(ds.asymmetry)) > 1.0
    assert np.nanmax(ds.error) > 0.1
    assert ds.run is not None
    assert ds.run.grouping.get("dead_time_us") == pytest.approx([0.01, 0.02])
    assert ds.run.grouping.get("good_frames") == pytest.approx(200000.0)


def test_load_v2_multiperiod(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_multi.nxs"
    _write_v2_file(path, multiperiod=True)

    result = loader.load(str(path))
    assert not isinstance(result, list)
    assert result.run is not None
    assert result.run_label == "12345"
    assert result.metadata["period_count"] == 2
    period_histograms = result.run.grouping.get("period_histograms")
    assert isinstance(period_histograms, list)
    assert len(period_histograms) == 2
    assert result.run.grouping.get("period_mode") == "red"


def test_load_v1_single_period(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v1.nxs"
    _write_v1_file(path)

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.metadata["nexus_version"] == "v1"
    assert ds.metadata["run_number"] == 2468
    assert ds.metadata["title"] == "V1 Test"
    assert ds.metadata["temperature"] == pytest.approx(5.0)
    assert ds.metadata["field"] == pytest.approx(20.0)
    assert ds.n_points == 4
    assert ds.run is not None
    assert ds.run.grouping.get("dead_time_us") == pytest.approx([0.015, 0.025])
    assert ds.run.grouping.get("good_frames") == pytest.approx(100000.0)


def test_load_v2_prefers_corrected_time_axis_when_present(tmp_path, loader: NexusLoader) -> None:
    """Use corrected_time directly when available instead of re-deriving from raw_time."""
    path = tmp_path / "run_v2_corrected_time.nxs"
    _write_v2_file(
        path,
        include_corrected_time=True,
        time_zero_us=0.04,
        t0_bin_attr=2,
    )

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.time == pytest.approx([-0.03, -0.01, 0.01, 0.03])
    assert ds.run is not None
    assert ds.run.histograms[0].t0_bin == 2
    assert ds.run.histograms[1].t0_bin == 2


def test_load_v2_raw_time_fallback_applies_time_zero_correction(tmp_path, loader: NexusLoader) -> None:
    """Fall back to raw_time centres and subtract t0 when corrected_time is absent."""
    path = tmp_path / "run_v2_raw_time_only.nxs"
    _write_v2_file(
        path,
        include_corrected_time=False,
        time_zero_us=0.04,
        t0_bin_attr=None,
    )

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.time == pytest.approx([-0.03, -0.01, 0.01, 0.03])
    assert ds.run is not None
    # With 0.02 us bins and time_zero=0.04 us, t0 maps to bin index 2.
    assert ds.run.histograms[0].t0_bin == 2
    assert ds.run.histograms[1].t0_bin == 2
