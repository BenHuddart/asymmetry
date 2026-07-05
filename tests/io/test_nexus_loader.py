"""Tests for ISIS NeXus loader (V1/V2 and multiperiod support)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.io]

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
    first_good_bin_attr: int | None = 0,
    last_good_bin_attr: int | None = 3,
    first_good_time_us: float | None = None,
    last_good_time_us: float | None = None,
    orientation: str = "L",
    field_state: str | None = None,
    field_vector: tuple[float, float, float] | None = None,
    field_vector_available: int | None = None,
    magnetic_field: float | None = 150.0,
    temp_setpoint: float = 12.5,
    temp_setpoint_units: str | None = None,
    temp_log_values: tuple[float, ...] = (12.0, 12.5, 13.0),
    temp_log_style: str = "flat",
    temp_log_units: str | None = None,
    temp_log_block: str = "Temp_Sample",
    temp_log_name: str | None = None,
    temp_log_times: tuple[float, ...] | None = None,
) -> None:
    """Create a synthetic V2 NeXus file used by loader unit tests.

    Parameters control whether the file carries a pre-corrected time axis and
    whether ``t0_bin`` is provided explicitly as an attribute. ``orientation``
    sets the detector-bank orientation; ``field_state`` writes
    ``sample/magnetic_field_state`` when not ``None``.
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
        if first_good_bin_attr is not None:
            counts_ds.attrs["first_good_bin"] = np.bytes_(str(first_good_bin_attr))
        if last_good_bin_attr is not None:
            counts_ds.attrs["last_good_bin"] = np.bytes_(str(last_good_bin_attr))
        if t0_bin_attr is not None:
            counts_ds.attrs["t0_bin"] = np.int32(t0_bin_attr)

        raw_time = np.array([0.0, 0.02, 0.04, 0.06, 0.08], dtype=np.float64)
        detector.create_dataset("raw_time", data=raw_time)
        if include_corrected_time:
            corrected_time = 0.5 * (raw_time[:-1] + raw_time[1:]) - float(time_zero_us)
            detector.create_dataset(
                "corrected_time", data=np.asarray(corrected_time, dtype=np.float64)
            )
        if first_good_time_us is not None:
            detector.create_dataset("first_good_time", data=float(first_good_time_us))
        if last_good_time_us is not None:
            detector.create_dataset("last_good_time", data=float(last_good_time_us))
        detector.create_dataset("grouping", data=np.array([1, 2], dtype=np.int32))
        detector.create_dataset("dead_time", data=np.array([0.01, 0.02], dtype=np.float64))
        detector.create_dataset(
            "time_zero", data=np.array([time_zero_us, time_zero_us], dtype=np.float64)
        )
        detector.create_dataset("orientation", data=np.bytes_(orientation))

        sample = entry.create_group("sample")
        temperature_ds = sample.create_dataset("temperature", data=temp_setpoint)
        if temp_setpoint_units is not None:
            temperature_ds.attrs["units"] = np.bytes_(temp_setpoint_units)
        if magnetic_field is not None:
            sample.create_dataset("magnetic_field", data=magnetic_field)
        if field_state is not None:
            sample.create_dataset("magnetic_field_state", data=np.bytes_(field_state))
        if field_vector is not None:
            vector_ds = sample.create_dataset(
                "magnetic_field_vector", data=np.asarray(field_vector, dtype=np.float32)
            )
            vector_ds.attrs["coordinate_system"] = np.bytes_("cartesian")
            vector_ds.attrs["units"] = np.bytes_("Gauss")
            if field_vector_available is not None:
                vector_ds.attrs["available"] = np.int32(field_vector_available)

        log_values = np.asarray(temp_log_values, dtype=np.float64)
        if temp_log_times is not None:
            log_times = np.asarray(temp_log_times, dtype=np.float64)
        else:
            log_times = np.arange(log_values.size, dtype=np.float64) * 10.0
        if temp_log_style == "selog":
            # ISIS selog convention: the NXlog lives in a ``value_log`` subgroup
            # of the block, e.g. ``selog/Temp_Sample/value_log``.
            selog = entry.create_group("selog")
            value_log = selog.create_group(temp_log_block).create_group("value_log")
            value_log.create_dataset("time", data=log_times)
            value_ds = value_log.create_dataset("value", data=log_values)
            if temp_log_name is not None:
                value_log.create_dataset("name", data=np.bytes_(temp_log_name))
        else:
            # Flat convention: the NXlog is the block group itself.
            temp_log = sample.create_group(temp_log_block)
            temp_log.create_dataset("time", data=log_times)
            value_ds = temp_log.create_dataset("value", data=log_values)
            if temp_log_name is not None:
                temp_log.create_dataset("name", data=np.bytes_(temp_log_name))
        if temp_log_units is not None:
            value_ds.attrs["units"] = np.bytes_(temp_log_units)

        if multiperiod:
            periods = entry.create_group("periods")
            periods.create_dataset("number", data=2)


def _write_v1_file(
    path,
    *,
    orientation: str = "T",
    field_state: str | None = None,
    magnetic_field: float | None = 20.0,
) -> None:
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
        detector.create_dataset("orientation", data=np.bytes_(orientation))

        sample = run.create_group("sample")
        sample.create_dataset("temperature", data=5.0)
        if magnetic_field is not None:
            sample.create_dataset("magnetic_field", data=magnetic_field)
        if field_state is not None:
            sample.create_dataset("magnetic_field_state", data=np.bytes_(field_state))

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
        h_data.create_dataset(
            "corrected_time", data=np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)
        )
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
    temp_series = ds.metadata["nexus_time_series"]["sample/Temp_Sample"]
    assert temp_series["mean"] == pytest.approx(12.5)
    assert temp_series["values"] == pytest.approx([12.0, 12.5, 13.0])
    assert ds.n_points == 4
    # Values are percentages, not fractions.
    expected_first = 100.0 * (100.0 - 80.0) / (100.0 + 80.0)
    assert ds.asymmetry[0] == pytest.approx(expected_first)
    assert np.nanmax(np.abs(ds.asymmetry)) > 1.0
    assert np.nanmax(ds.error) > 0.1
    assert ds.run is not None
    assert ds.run.grouping.get("dead_time_us") == pytest.approx([0.01, 0.02])
    assert ds.run.grouping.get("good_frames") == pytest.approx(200000.0)


def test_logged_sample_temperature_distinct_from_setpoint(tmp_path, loader: NexusLoader) -> None:
    # Setpoint parked at 1 K while the logged sample series sits near 5 K.
    path = tmp_path / "run_logged_t.nxs"
    _write_v2_file(
        path,
        temp_setpoint=1.0,
        temp_log_values=(4.8, 5.0, 5.2),
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)

    # Setpoint is unchanged (no behaviour change for existing callers).
    assert ds.metadata["temperature"] == pytest.approx(1.0)
    # Logged temperature is the mean of the Temp_Sample series, distinct from it.
    assert ds.metadata["sample_temperature_logged"] == pytest.approx(5.0)
    assert ds.sample_temperature_logged == pytest.approx(5.0)
    assert ds.sample_temperature_logged != ds.metadata["temperature"]
    assert ds.run is not None
    assert ds.run.sample_temperature_logged == pytest.approx(5.0)


def test_logged_sample_temperature_from_isis_selog_value_log(tmp_path, loader: NexusLoader) -> None:
    # Real ISIS files nest the NXlog as ``selog/Temp_Sample/value_log`` rather
    # than placing it flat on the block group. The logged value must still be
    # recovered (regression: previously only the flat layout matched).
    path = tmp_path / "run_selog_t.nxs"
    _write_v2_file(
        path,
        temp_setpoint=1.0,
        temp_log_values=(4.8, 5.0, 5.2),
        temp_log_style="selog",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)

    # The series is captured under the nested selog path...
    assert "selog/Temp_Sample/value_log" in ds.metadata["nexus_time_series"]
    # ...and the logged temperature is recovered, distinct from the 1 K setpoint.
    assert ds.metadata["temperature"] == pytest.approx(1.0)
    assert ds.metadata["sample_temperature_logged"] == pytest.approx(5.0)
    assert ds.sample_temperature_logged == pytest.approx(5.0)
    assert ds.run is not None
    assert ds.run.sample_temperature_logged == pytest.approx(5.0)


def test_logged_sample_temperature_matched_via_nxlog_name_child(
    tmp_path, loader: NexusLoader
) -> None:
    # Native HDF4 v1 logs name the Vgroup generically and carry the real sensor
    # name only in a ``name`` child (the converted HDF5 twin bakes it into the
    # selog path). The sample thermometer must be matched off that ``name`` so
    # both containers expose the same ``sample_temperature_logged``.
    path = tmp_path / "run_named_log.nxs"
    _write_v2_file(
        path,
        temp_setpoint=1.0,
        temp_log_values=(4.8, 5.0, 5.2),
        temp_log_style="selog",
        temp_log_block="log_1",  # generic, sensor-agnostic path segment
        temp_log_name="Temp_Sample",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)

    # The path is generic, so only the captured ``name`` identifies the sensor.
    assert "selog/log_1/value_log" in ds.metadata["nexus_time_series"]
    assert ds.metadata["nexus_time_series"]["selog/log_1/value_log"]["name"] == "Temp_Sample"
    assert ds.metadata["sample_temperature_logged"] == pytest.approx(5.0)
    assert ds.sample_temperature_logged == pytest.approx(5.0)


def test_logged_sample_temperature_gates_out_pre_run_samples(tmp_path, loader: NexusLoader) -> None:
    # The full-record NXlog mean includes the pre-run (t < 0) plateau parked at
    # the previous setpoint; sample_temperature_logged must average only the
    # run-active (t >= 0) samples so the first run of a block is not
    # mis-temperatured (the Sn 91516 case).
    path = tmp_path / "run_pre_run_plateau.nxs"
    _write_v2_file(
        path,
        temp_setpoint=1.6,
        temp_log_values=(4.62, 4.62, 1.60, 1.60),
        temp_log_times=(-20.0, -10.0, 0.0, 30.0),
        temp_log_style="selog",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    # Run-active mean = mean(1.60, 1.60) = 1.60, not the contaminated 3.11.
    assert ds.sample_temperature_logged == pytest.approx(1.60)


def test_no_logged_series_leaves_temperature_unset(tmp_path, loader: NexusLoader) -> None:
    # A file without any Temp_Sample log leaves the key absent and the accessor
    # returning None.
    path = tmp_path / "run_no_log.nxs"
    _write_v2_file(path, temp_setpoint=12.5, temp_log_values=())

    ds = loader.load(str(path))
    assert not isinstance(ds, list)

    assert "sample_temperature_logged" not in ds.metadata
    assert ds.sample_temperature_logged is None
    assert ds.run is not None
    assert ds.run.sample_temperature_logged is None


def test_empty_logged_series_emits_no_runtime_warning(tmp_path, loader: NexusLoader) -> None:
    # An all-NaN logged series must not trigger nanmean/nanmin/nanmax warnings.
    path = tmp_path / "run_empty_log.nxs"
    _write_v2_file(
        path,
        temp_log_values=(float("nan"), float("nan")),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ds = loader.load(str(path))

    assert not isinstance(ds, list)
    # No logged value could be derived, but the load still succeeds.
    assert ds.sample_temperature_logged is None
    assert "sample_temperature_logged" not in ds.metadata
    series = ds.metadata["nexus_time_series"]["sample/Temp_Sample"]
    assert series["mean"] is None


def test_temperature_setpoint_celsius_normalized_to_kelvin(tmp_path, loader: NexusLoader) -> None:
    # A setpoint field carrying a Celsius units attribute is normalized to K.
    path = tmp_path / "run_degc.nxs"
    _write_v2_file(path, temp_setpoint=380.0, temp_setpoint_units="degC")

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.metadata["temperature"] == pytest.approx(380.0 + 273.15)


@pytest.mark.parametrize("units", ["Kelvin", "K", None])
def test_temperature_setpoint_kelvin_or_unitless_unchanged(
    tmp_path, loader: NexusLoader, units
) -> None:
    # Kelvin, "K", or an absent units attribute pass the value through unchanged
    # — we never guess a conversion the file did not declare. (This is also why
    # a file that mislabels Celsius data as "Kelvin" is left as-is.)
    path = tmp_path / f"run_k_{units}.nxs"
    _write_v2_file(path, temp_setpoint=380.0, temp_setpoint_units=units)

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.metadata["temperature"] == pytest.approx(380.0)


def test_logged_temperature_celsius_normalized_to_kelvin(tmp_path, loader: NexusLoader) -> None:
    # The °C→K normalization also applies to the logged sample series.
    path = tmp_path / "run_logged_degc.nxs"
    _write_v2_file(
        path,
        temp_setpoint=1.0,
        temp_log_values=(4.8, 5.0, 5.2),
        temp_log_units="degC",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    # Mean of the logged series (5.0 °C) normalized to kelvin.
    assert ds.sample_temperature_logged == pytest.approx(5.0 + 273.15)


def test_logged_all_zero_series_returns_none(tmp_path, loader: NexusLoader) -> None:
    # An all-zero Temp_Sample log is a disconnected/unlogged sensor (seen on EMU
    # furnace runs), not a 0 K measurement: report None, not a misleading 0.0.
    path = tmp_path / "run_zero_log.nxs"
    _write_v2_file(path, temp_setpoint=350.0, temp_log_values=(0.0, 0.0, 0.0))

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.sample_temperature_logged is None
    assert "sample_temperature_logged" not in ds.metadata


def test_logged_furnace_controller_block_not_matched(tmp_path, loader: NexusLoader) -> None:
    # An EMU furnace run logs controller/cryostat readbacks (Temp_RBV, …) but no
    # sample thermometer. Those must NOT be reported as the logged sample T —
    # None is the honest answer rather than a guess from the wrong sensor.
    path = tmp_path / "run_furnace.nxs"
    _write_v2_file(
        path,
        temp_setpoint=100.0,
        temp_log_values=(95.0, 125.0, 210.0),
        temp_log_style="selog",
        temp_log_block="Temp_RBV",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.sample_temperature_logged is None
    assert "sample_temperature_logged" not in ds.metadata
    # The readback series is still captured for the advanced info view.
    assert "selog/Temp_RBV/value_log" in ds.metadata["nexus_time_series"]


def test_emu_furnace_temperature_unit_flagged_value_unchanged(
    tmp_path, loader: NexusLoader
) -> None:
    # EMU furnace run: a Celsius value stored under a "Kelvin" label with no
    # sample thermometer. SURFACE the suspicion (flag) but never convert — the
    # returned temperature must stay exactly as read.
    path = tmp_path / "run_furnace_hot.nxs"
    _write_v2_file(
        path,
        temp_setpoint=380.0,
        temp_setpoint_units="Kelvin",
        temp_log_style="selog",
        temp_log_block="Temp_RBV",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.metadata["temperature"] == pytest.approx(380.0)  # unchanged
    assert ds.sample_temperature_logged is None
    assert ds.metadata.get("temperature_unit_suspect") is True
    assert "°C" in ds.metadata["temperature_unit_suspect_reason"]


def test_emu_cold_kelvin_run_not_flagged(tmp_path, loader: NexusLoader) -> None:
    # A genuinely-cold Kelvin run (no furnace) must NOT be flagged, even with no
    # logged sample thermometer: below the cryostat ceiling the value is trusted.
    path = tmp_path / "run_cold.nxs"
    _write_v2_file(
        path,
        temp_setpoint=5.0,
        temp_setpoint_units="Kelvin",
        temp_log_style="selog",
        temp_log_block="Temp_RBV",
    )

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.metadata["temperature"] == pytest.approx(5.0)
    assert "temperature_unit_suspect" not in ds.metadata


def test_temperature_unit_suspect_heuristic_matrix(loader: NexusLoader) -> None:
    suspect = loader._temperature_unit_suspect
    # Canonical mislabel: EMU, hot, Kelvin-labelled, no logged thermometer.
    flagged, reason = suspect("EMU", 380.0, None, "Kelvin")
    assert flagged is True
    assert reason
    # A blank / unknown unit is equally at risk of being a disguised Celsius value.
    assert suspect("EMU", 380.0, None, "")[0] is True
    assert suspect("emu", 380.0, None, None)[0] is True
    # Below the ceiling the value is ambiguous (300 K vs 300 °C) — never flagged.
    assert suspect("EMU", 300.0, None, "Kelvin")[0] is False
    # A logged sample thermometer corroborates the setpoint — not flagged.
    assert suspect("EMU", 380.0, 379.0, "Kelvin")[0] is False
    # A declared Celsius unit is already converted and trustworthy — not flagged.
    assert suspect("EMU", 380.0 + 273.15, None, "degC")[0] is False
    # Out-of-scope instruments are never flagged by this EMU-specific heuristic.
    assert suspect("MuSR", 380.0, None, "Kelvin")[0] is False
    assert suspect("HIFI", 380.0, None, "Kelvin")[0] is False
    # Missing temperature: not flagged.
    assert suspect("EMU", None, None, "Kelvin")[0] is False


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


def _write_v1_file_detector_deadtimes(path) -> None:
    """V1 file whose dead times live at the legacy plural location only.

    Mirrors real ISIS muon NeXus v1 files (and the HDF4 originals read directly
    via the pyhdf path): the dead-time table is stored at
    ``/run/instrument/detector/deadtimes`` (plural, different group), with
    *nothing* under ``histogram_data_1``. The nxs4to5 converter maps
    ``instrument/detector/deadtimes`` -> ``detector_1/dead_time`` (see
    wimda-corpus/nxs4to5/README.md), so the HDF5 twins read fine; a directly
    loaded v1 file must use this fallback or it gets an all-zeros table.
    """
    with h5py.File(path, "w") as f:
        run = f.create_group("run")
        run.create_dataset("analysis", data=np.bytes_("muonTD"))
        run.create_dataset("IDF_version", data=1)
        run.create_dataset("number", data=2469)
        run.create_dataset("good_frames", data=np.array([100000], dtype=np.int32))

        instrument = run.create_group("instrument")
        detector = instrument.create_group("detector")
        detector.create_dataset("orientation", data=np.bytes_("T"))
        # Plural, in the detector group — NOT in histogram_data_1.
        detector.create_dataset("deadtimes", data=np.array([0.011, 0.021], dtype=np.float64))

        sample = run.create_group("sample")
        sample.create_dataset("temperature", data=5.0)
        sample.create_dataset("magnetic_field", data=20.0)

        h_data = run.create_group("histogram_data_1")
        h_data.create_dataset(
            "counts",
            data=np.array([[120, 140, 160, 180], [100, 120, 130, 150]], dtype=np.float64),
        )
        h_data.create_dataset(
            "corrected_time", data=np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)
        )
        h_data.create_dataset("grouping", data=np.array([1, 2], dtype=np.int32))


def test_load_v1_reads_deadtimes_from_detector_group(tmp_path, loader: NexusLoader) -> None:
    """Regression: v1 dead times stored only at instrument/detector/deadtimes.

    Without the fallback, the loader reads dead times solely from
    ``histogram_data_1`` and returns an all-zeros table for these (real) files.
    """
    path = tmp_path / "run_v1_detector_deadtimes.nxs"
    _write_v1_file_detector_deadtimes(path)

    ds = loader.load(str(path))
    assert not isinstance(ds, list)
    assert ds.metadata["nexus_version"] == "v1"
    dead_times = ds.run.grouping.get("dead_time_us")
    assert any(v != 0.0 for v in dead_times), "dead times should not be all-zeros"
    assert dead_times == pytest.approx([0.011, 0.021])


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


def test_load_v2_raw_time_fallback_applies_time_zero_correction(
    tmp_path, loader: NexusLoader
) -> None:
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


def test_load_v2_prefers_bin_attributes_over_explicit_good_times(
    tmp_path, loader: NexusLoader
) -> None:
    """Keep integer bin metadata canonical when both forms are present."""
    path = tmp_path / "run_v2_good_times.nxs"
    _write_v2_file(
        path,
        include_corrected_time=True,
        time_zero_us=0.04,
        t0_bin_attr=2,
        first_good_bin_attr=2,
        last_good_bin_attr=3,
        first_good_time_us=-0.01,
        last_good_time_us=0.03,
    )

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.time == pytest.approx([0.01, 0.03])
    assert ds.n_points == 2
    assert ds.run is not None
    assert ds.run.grouping["first_good_bin"] == 2


def test_load_v2_uses_good_times_when_bin_attributes_missing(tmp_path, loader: NexusLoader) -> None:
    """Use good-time metadata only when integer good-bin attributes are absent."""
    path = tmp_path / "run_v2_good_times_only.nxs"
    _write_v2_file(
        path,
        include_corrected_time=True,
        time_zero_us=0.04,
        t0_bin_attr=2,
        first_good_bin_attr=None,
        last_good_bin_attr=None,
        first_good_time_us=-0.01,
        last_good_time_us=0.03,
    )

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.time == pytest.approx([-0.01, 0.01, 0.03])
    assert ds.n_points == 3
    assert ds.run is not None
    assert ds.run.grouping["first_good_bin"] == 0


def test_load_v2_normalizes_one_based_bin_attributes(tmp_path, loader: NexusLoader) -> None:
    """Normalize 1-based V2 t0/first-good bin attributes to array indices."""
    path = tmp_path / "run_v2_one_based_bins.nxs"

    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data_1")
        entry.create_dataset("definition", data=np.bytes_("muonTD"))
        entry.create_dataset("IDF_version", data=2)
        entry.create_dataset("run_number", data=206601)
        entry.create_dataset("good_frames", data=np.array([50391], dtype=np.int32))
        entry.create_dataset("title", data=np.bytes_("HIFI Test"))
        entry.create_dataset("start_time", data=np.bytes_("2026-03-15T10:00:00"))
        entry.create_dataset("end_time", data=np.bytes_("2026-03-15T11:00:00"))
        entry.create_dataset("name", data=np.bytes_("HIFI"))

        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector_1")

        counts = np.array(
            [
                [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            ],
            dtype=np.float64,
        )
        counts_ds = detector.create_dataset("counts", data=counts)
        counts_ds.attrs["first_good_bin"] = np.int32(8)
        counts_ds.attrs["last_good_bin"] = np.int32(10)
        counts_ds.attrs["t0_bin"] = np.int32(2)

        raw_time = np.arange(11, dtype=np.float64) * 0.016
        corrected_time = 0.5 * (raw_time[:-1] + raw_time[1:]) - 0.024
        detector.create_dataset("raw_time", data=raw_time)
        detector.create_dataset("corrected_time", data=corrected_time)
        detector.create_dataset("grouping", data=np.array([1, 2], dtype=np.int32))
        detector.create_dataset("dead_time", data=np.array([0.01, 0.02], dtype=np.float64))
        detector.create_dataset("time_zero", data=np.array([0.024], dtype=np.float64))
        detector.create_dataset("orientation", data=np.bytes_("L"))

        sample = entry.create_group("sample")
        sample.create_dataset("temperature", data=12.5)
        sample.create_dataset("magnetic_field", data=150.0)

    result = loader.load(str(path))
    assert not isinstance(result, list)

    ds = result
    assert ds.time[0] == pytest.approx(0.096)
    assert ds.run is not None
    assert ds.run.grouping["first_good_bin"] == 7
    assert ds.run.grouping["bin_index_base"] == 1
    assert ds.run.histograms[0].t0_bin == 1


# --- Field geometry (TF/LF/ZF) vs detector orientation ---------------------
# The applied-field geometry comes from sample/magnetic_field_state, NOT from
# detector orientation. On EMU/MuSR the banks read 'L' even for TF runs, so
# orientation must never decide field_direction. See docs/porting/field-geometry/.


def test_v2_field_state_tf_overrides_l_orientation(tmp_path, loader: NexusLoader) -> None:
    """A TF run with L-oriented banks must read as Transverse, not Longitudinal."""
    path = tmp_path / "run_v2_tf.nxs"
    _write_v2_file(path, orientation="L", field_state="TF")

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == "TF"
    assert ds.metadata["field_direction"] == "Transverse"
    # The instrument-axis value is preserved separately, not lost.
    assert ds.metadata["detector_orientation"] == "Longitudinal"


def test_v2_field_state_lf(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_lf.nxs"
    _write_v2_file(path, orientation="L", field_state="LF")

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == "LF"
    assert ds.metadata["field_direction"] == "Longitudinal"
    assert ds.metadata["detector_orientation"] == "Longitudinal"


def test_v2_field_state_zf(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_zf.nxs"
    _write_v2_file(path, orientation="L", field_state="ZF")

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == "ZF"
    assert ds.metadata["field_direction"] == "Zero field"


def test_v2_field_state_absent_geometry_is_blank_no_orientation_fallback(
    tmp_path, loader: NexusLoader
) -> None:
    """Without magnetic_field_state the geometry is unknown (blank), NOT orientation."""
    path = tmp_path / "run_v2_no_state.nxs"
    _write_v2_file(path, orientation="L", field_state=None)

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == ""
    assert ds.metadata["field_direction"] == ""
    # Orientation is still captured for whoever needs the instrument axis.
    assert ds.metadata["detector_orientation"] == "Longitudinal"


def test_v2_field_state_blank_string_treated_as_unknown(tmp_path, loader: NexusLoader) -> None:
    """An empty/unrecognised state string is treated as unknown, not an error."""
    path = tmp_path / "run_v2_blank_state.nxs"
    _write_v2_file(path, orientation="T", field_state="n/a")

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == ""
    assert ds.metadata["field_direction"] == ""
    assert ds.metadata["detector_orientation"] == "Transverse"


def test_v1_field_state_tf_overrides_l_orientation(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v1_tf.nxs"
    _write_v1_file(path, orientation="L", field_state="TF")

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == "TF"
    assert ds.metadata["field_direction"] == "Transverse"
    assert ds.metadata["detector_orientation"] == "Longitudinal"


def test_v1_field_state_absent_geometry_is_blank(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v1_no_state.nxs"
    _write_v1_file(path, orientation="T", field_state=None)

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == ""
    assert ds.metadata["field_direction"] == ""
    assert ds.metadata["detector_orientation"] == "Transverse"


# --- sample/magnetic_field_vector extraction --------------------------------
#
# ISIS NeXus files carry a magnetic_field_vector dataset with an ``available``
# attribute; a real-corpus survey found every "unavailable" file held the same
# placeholder ([1, 1, 1], available=0). It is surfaced as raw provenance only —
# never used to infer TF/LF geometry (see NexusLoader._read_field_vector).


def test_v2_field_vector_available_is_surfaced(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_vector.nxs"
    _write_v2_file(
        path,
        field_state="TF",
        field_vector=(0.0, 0.0, 1.0),
        field_vector_available=1,
    )

    ds = loader.load(str(path))
    assert ds.metadata["field_vector"] == pytest.approx([0.0, 0.0, 1.0])


def test_v2_field_vector_unavailable_placeholder_is_not_surfaced(
    tmp_path, loader: NexusLoader
) -> None:
    """available=0 marks the [1,1,1] vector a placeholder; it must not appear."""
    path = tmp_path / "run_v2_vector_unavailable.nxs"
    _write_v2_file(
        path,
        field_state="TF",
        field_vector=(1.0, 1.0, 1.0),
        field_vector_available=0,
    )

    ds = loader.load(str(path))
    assert "field_vector" not in ds.metadata


def test_v2_field_vector_absent_is_not_surfaced(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_no_vector.nxs"
    _write_v2_file(path, field_state="TF", field_vector=None)

    ds = loader.load(str(path))
    assert "field_vector" not in ds.metadata


def test_v2_field_vector_never_overrides_field_direction(tmp_path, loader: NexusLoader) -> None:
    """A [0,0,1] TF vector must not make LF/ZF-labelled runs read differently."""
    path = tmp_path / "run_v2_vector_lf.nxs"
    _write_v2_file(
        path,
        field_state="LF",
        field_vector=(0.0, 0.0, 1.0),
        field_vector_available=1,
    )

    ds = loader.load(str(path))
    assert ds.metadata["field_direction"] == "Longitudinal"
    assert ds.metadata["field_vector"] == pytest.approx([0.0, 0.0, 1.0])


# --- ICP .log sidecar fill (only when NeXus metadata is missing) ------------
#
# See asymmetry.core.io.icp_log. The loader only fills field/field_direction
# from a sibling .log when the NeXus file itself carries no usable value —
# it must never override metadata the NeXus file actually recorded.

_ZF_LOG_TEXT = (
    "2012-03-10T09:14:33\tField_ZF_Magnitude\t0.00010\n"
    "2012-03-10T09:14:33\tField_Danfysik\t0\n"
    "2012-03-10T09:19:37\ta_selected_magnet\tActive ZF\n"
    "2012-03-10T09:19:37\tField_ZF_Magnitude\t0.00015\n"
)

_DANFYSIK_LOG_TEXT = (
    "2012-03-10T09:14:33\tField_Danfysik\t99.5\n"
    "2012-03-10T09:19:37\ta_selected_magnet\tDanfysik\n"
    "2012-03-10T09:19:40\tField_Danfysik\t100.0\n"
)


def test_v2_missing_field_metadata_filled_from_sidecar_log(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_no_state.nxs"
    _write_v2_file(path, field_state=None, magnetic_field=None)
    path.with_suffix(".log").write_text(_ZF_LOG_TEXT)

    ds = loader.load(str(path))
    assert ds.metadata["field_direction"] == "Zero field"
    assert ds.metadata["field_state"] == "ZF"
    assert ds.metadata["field"] == pytest.approx(0.00015)
    assert ds.metadata["field_source"] == "icp_log"


def test_v2_missing_field_metadata_filled_from_sidecar_log_nonzero(
    tmp_path, loader: NexusLoader
) -> None:
    path = tmp_path / "run_v2_no_state_nonzero.nxs"
    _write_v2_file(path, field_state=None, magnetic_field=None)
    path.with_suffix(".log").write_text(_DANFYSIK_LOG_TEXT)

    ds = loader.load(str(path))
    # Danfysik does not name a TF/LF direction; only magnitude is filled.
    assert ds.metadata["field_direction"] == ""
    assert ds.metadata["field"] == pytest.approx(100.0)
    assert ds.metadata["field_source"] == "icp_log"


def test_v2_present_field_state_is_never_overridden_by_log(tmp_path, loader: NexusLoader) -> None:
    """A NeXus file that already records TF must not be relabelled ZF by a log."""
    path = tmp_path / "run_v2_tf_with_log.nxs"
    _write_v2_file(path, field_state="TF", magnetic_field=0.0)
    path.with_suffix(".log").write_text(_ZF_LOG_TEXT)

    ds = loader.load(str(path))
    assert ds.metadata["field_state"] == "TF"
    assert ds.metadata["field_direction"] == "Transverse"
    assert ds.metadata["field"] == pytest.approx(0.0)
    assert "field_source" not in ds.metadata


def test_v2_no_sidecar_log_leaves_metadata_unchanged(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_no_state_no_log.nxs"
    _write_v2_file(path, field_state=None, magnetic_field=None)

    ds = loader.load(str(path))
    assert ds.metadata["field_direction"] == ""
    assert ds.metadata["field"] == 0.0  # _safe_float default, unchanged
    assert "field_source" not in ds.metadata


def test_v2_malformed_sidecar_log_does_not_raise(tmp_path, loader: NexusLoader) -> None:
    path = tmp_path / "run_v2_bad_log.nxs"
    _write_v2_file(path, field_state=None, magnetic_field=None)
    path.with_suffix(".log").write_text("this is not a valid ICP log\n\x00\xff garbage")

    ds = loader.load(str(path))  # must not raise
    assert ds.metadata["field_direction"] == ""
    assert "field_source" not in ds.metadata
