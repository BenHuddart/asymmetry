"""Tests for run information dialog include/log actions."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.run_info_dialog import RunInfoDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset() -> MuonDataset:
    t = np.linspace(0.0, 1.0, 8)
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={
            "run_number": 1001,
            "run_label": "1001",
            "title": "Test",
            "temperature": 10.0,
            "field": 50.0,
            "nexus_fields": {"sample": {"temperature": 10.0, "shape": "plate"}},
            "nexus_time_series": {
                "sample/Temp_Sample": {
                    "units": "K",
                    "time": [0.0, 1.0, 2.0],
                    "values": [9.0, 10.0, 11.0],
                    "mean": 10.0,
                    "min": 9.0,
                    "max": 11.0,
                }
            },
        },
    )


def _dataset_with_run() -> MuonDataset:
    h0 = Histogram(counts=np.array([100.0, 90.0, 80.0, 70.0]), bin_width=0.01)
    h1 = Histogram(counts=np.array([95.0, 85.0, 75.0, 65.0]), bin_width=0.01)
    run = Run(
        run_number=1001,
        histograms=[h0, h1],
        metadata={"run_number": 1001},
        grouping={"groups": {1: [1], 2: [2]}, "forward_group": 1, "backward_group": 2},
    )
    ds = _dataset()
    ds.run = run
    return ds


def _psi_dataset_with_temperature_log() -> MuonDataset:
    ds = _dataset()
    ds.metadata["facility"] = "PSI"
    ds.metadata["nexus_time_series"] = {
        "psi_temperature/Temp_Heater": {
            "units": "K",
            "time": [0.0, 10.0],
            "values": [4.9906, 5.0],
            "mean": 4.9953,
            "min": 4.9906,
            "max": 5.0,
            "source_file": "/tmp/run_4321_templs0.mon",
            "reader_provenance": "Mantid LoadPSIMuonBin-compatible",
        }
    }
    ds.metadata["psi_temperature_log"] = {
        "source_file": "/tmp/run_4321_templs0.mon",
        "source_format": "PSI .mon",
        "reader_provenance": "Mantid LoadPSIMuonBin-compatible",
        "channels": ["Temp_Heater"],
    }
    return ds


def _root_dataset_with_temperature_log() -> MuonDataset:
    ds = _dataset()
    ds.metadata["facility"] = "PSI"
    ds.metadata["root_format"] = "musr-root-directory"
    ds.metadata["temperature"] = 12.5
    ds.metadata["nexus_time_series"] = {
        "musrroot_slow_control/Sample Temperature": {
            "units": "K",
            "time": [10.0, 30.0],
            "values": [12.1, 12.3],
            "mean": 12.2,
            "min": 12.1,
            "max": 12.3,
            "source_file": "/tmp/lem.root",
            "reader_provenance": "MusrRoot slow-control histogram",
        }
    }
    ds.metadata["musrroot_slow_control_log"] = {
        "source_file": "/tmp/lem.root",
        "source_format": "MusrRoot SCAnaModule",
        "reader_provenance": "MusrRoot slow-control histogram",
        "channels": ["Sample Temperature"],
    }
    return ds


def _root_dataset_with_multiple_temperature_logs() -> MuonDataset:
    ds = _dataset()
    ds.metadata["facility"] = "PSI"
    ds.metadata["root_format"] = "musr-root-folder"
    ds.metadata["temperature"] = 45.0
    ds.metadata["nexus_time_series"] = {
        "musrroot_slow_control/Moderator Temperature Run lem15 2994": {
            "units": "K",
            "time": [0.0, 1.0],
            "values": [39.0, 41.0],
            "mean": 40.0,
            "min": 39.0,
            "max": 41.0,
        },
        "musrroot_slow_control/Sample Temperature Run lem15 2994": {
            "units": "K",
            "time": [0.0, 1.0],
            "values": [44.8, 45.2],
            "mean": 45.0,
            "min": 44.8,
            "max": 45.2,
        },
    }
    ds.metadata["musrroot_slow_control_log"] = {
        "source_file": "/tmp/lem15_his_2994.root",
        "source_format": "MusrRoot SCAnaModule",
        "reader_provenance": "MusrRoot slow-control histogram",
        "channels": [
            "Moderator Temperature Run lem15 2994",
            "Sample Temperature Run lem15 2994",
        ],
    }
    return ds


def _root_dataset_with_sensor_named_temperature_log() -> MuonDataset:
    ds = _dataset()
    ds.metadata["facility"] = "PSI"
    ds.metadata["root_format"] = "musr-root-folder"
    ds.metadata["temperature"] = 10.4
    ds.metadata["nexus_time_series"] = {
        "musrroot_slow_control/flamedil0 DIL T mix value": {
            "units": "K",
            "time": [0.0, 30.0],
            "values": [11.2, 11.0],
            "mean": 11.1,
            "min": 11.0,
            "max": 11.2,
            "role": "sample_temperature",
            "sensor": "DIL_T_mix_value",
            "primary": False,
        },
        "musrroot_slow_control/flamesam0 SAM ts value": {
            "units": "K",
            "time": [0.0, 30.0],
            "values": [10.5, 10.3],
            "mean": 10.4,
            "min": 10.3,
            "max": 10.5,
            "role": "sample_temperature",
            "sensor": "SAM_ts_value",
            "primary": True,
        },
    }
    ds.metadata["musrroot_slow_control_log"] = {
        "source_file": "/tmp/flame.root",
        "source_format": "MusrRoot SCAnaModule",
        "reader_provenance": "MusrRoot slow-control histogram",
        "channels": ["flamedil0 DIL T mix value", "flamesam0 SAM ts value"],
    }
    return ds


def _row_for_field(table, field_name: str) -> int:
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is not None and item.text() == field_name:
            return row
    return -1


def _row_containing_field(table, field_fragment: str) -> int:
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is not None and field_fragment in item.text():
            return row
    return -1


def test_run_info_dialog_uses_standard_button_box(qapp: QApplication) -> None:
    """P2-9: the action bar is a QDialogButtonBox (Close standard + Advanced action)."""
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = RunInfoDialog(_dataset())
    box = dialog.findChild(QDialogButtonBox)
    assert box is not None
    assert box.button(QDialogButtonBox.StandardButton.Close) is not None
    # Advanced remains, now as an ActionRole button on the same bar.
    assert box.buttonRole(dialog._advanced_button) == QDialogButtonBox.ButtonRole.ActionRole
    dialog.close()


def test_summary_checkbox_emits_include_signal(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())

    captured: list[str] = []
    dialog.add_to_browser_requested.connect(captured.append)
    inclusion_events: list[tuple[str, bool]] = []
    dialog.set_browser_field_inclusion_requested.connect(
        lambda key, include: inclusion_events.append((key, include))
    )

    temp_row = _row_for_field(dialog._summary_table, "Temperature (K)")
    assert temp_row >= 0
    checkbox = dialog._summary_table.cellWidget(temp_row, 0)
    assert checkbox is not None

    checkbox.setChecked(True)
    checkbox.setChecked(False)

    assert "temperature" in captured
    assert ("temperature", True) in inclusion_events
    assert ("temperature", False) in inclusion_events
    dialog.close()


def test_summary_table_shows_plot_button_for_series_backed_field(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())

    temp_row = _row_for_field(dialog._summary_table, "Temperature (K)")
    assert temp_row >= 0
    plot_button = dialog._summary_table.cellWidget(temp_row, 3)
    assert plot_button is not None
    dialog.close()


def test_summary_table_shows_plot_button_for_psi_temperature_log(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_psi_dataset_with_temperature_log())

    temp_row = _row_for_field(dialog._summary_table, "Temperature (K)")
    assert temp_row >= 0
    value_item = dialog._summary_table.item(temp_row, 2)
    assert value_item is not None
    assert float(value_item.text()) == pytest.approx(4.9953)
    plot_button = dialog._summary_table.cellWidget(temp_row, 3)
    assert plot_button is not None
    dialog.close()


def test_summary_table_shows_plot_button_for_musrroot_temperature_log(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_root_dataset_with_temperature_log())

    temp_row = _row_for_field(dialog._summary_table, "Temperature (K)")
    assert temp_row >= 0
    value_item = dialog._summary_table.item(temp_row, 2)
    assert value_item is not None
    assert float(value_item.text()) == pytest.approx(12.2)
    plot_button = dialog._summary_table.cellWidget(temp_row, 3)
    assert plot_button is not None
    dialog.close()


def test_summary_table_prefers_musrroot_sample_temperature_log(
    qapp: QApplication,
) -> None:
    dialog = RunInfoDialog(_root_dataset_with_multiple_temperature_logs())

    temp_row = _row_for_field(dialog._summary_table, "Temperature (K)")
    assert temp_row >= 0
    value_item = dialog._summary_table.item(temp_row, 2)
    assert value_item is not None
    assert float(value_item.text()) == pytest.approx(45.0)
    plot_button = dialog._summary_table.cellWidget(temp_row, 3)
    assert plot_button is not None
    dialog.close()


def test_summary_table_uses_musrroot_sensor_role_for_temperature_log(
    qapp: QApplication,
) -> None:
    dialog = RunInfoDialog(_root_dataset_with_sensor_named_temperature_log())

    temp_row = _row_for_field(dialog._summary_table, "Temperature (K)")
    assert temp_row >= 0
    value_item = dialog._summary_table.item(temp_row, 2)
    assert value_item is not None
    assert float(value_item.text()) == pytest.approx(10.4)
    plot_button = dialog._summary_table.cellWidget(temp_row, 3)
    assert plot_button is not None
    dialog.close()


def test_summary_derived_rows_have_enabled_include_checkboxes(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset_with_run())

    for field_name in (
        "Points",
        "Histograms",
        "Bins",
        "Bin Width (us)",
        "Counts (MEv)",
        "Counts per Detector",
    ):
        row = _row_for_field(dialog._summary_table, field_name)
        assert row >= 0
        checkbox = dialog._summary_table.cellWidget(row, 0)
        assert checkbox is not None
        assert checkbox.isEnabled()

    dialog.close()


def test_summary_total_counts_shows_placeholder_then_fills_after_event_pump(
    qapp: QApplication,
) -> None:
    """F3: the full-histogram sum is deferred via QTimer.singleShot(0, ...) so
    opening the dialog never blocks on it — the row shows a placeholder
    immediately and the real value once the event loop has run."""
    dialog = RunInfoDialog(_dataset_with_run())

    mev_row = _row_for_field(dialog._summary_table, "Counts (MEv)")
    per_detector_row = _row_for_field(dialog._summary_table, "Counts per Detector")
    assert mev_row >= 0
    assert per_detector_row >= 0
    assert dialog._summary_table.item(mev_row, 2).text() == "computing…"
    assert dialog._summary_table.item(per_detector_row, 2).text() == "computing…"

    # Pump the event loop once so the singleShot(0, ...) callback fires.
    qapp.processEvents()

    total = 100.0 + 90.0 + 80.0 + 70.0 + 95.0 + 85.0 + 75.0 + 65.0
    assert float(dialog._summary_table.item(mev_row, 2).text()) == pytest.approx(total / 1.0e6)
    assert float(dialog._summary_table.item(per_detector_row, 2).text()) == pytest.approx(total / 2)

    dialog.close()


def test_summary_shows_orientation_from_nexus_sample_shape(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())

    orientation_row = _row_for_field(dialog._summary_table, "Orientation")
    assert orientation_row >= 0
    value_item = dialog._summary_table.item(orientation_row, 2)
    assert value_item is not None
    assert value_item.text() == "plate"

    checkbox = dialog._summary_table.cellWidget(orientation_row, 0)
    assert checkbox is not None
    assert checkbox.isEnabled()
    dialog.close()


def test_advanced_dialog_emits_include_signal(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())

    inclusion_events: list[tuple[str, bool]] = []
    dialog.set_browser_field_inclusion_requested.connect(
        lambda key, include: inclusion_events.append((key, include))
    )

    dialog._open_advanced_dialog()
    advanced = dialog._advanced_dialog
    assert advanced is not None

    include_box = advanced._table.cellWidget(0, 0)
    assert include_box is not None
    include_box.setChecked(True)

    assert inclusion_events
    if dialog._advanced_dialog is not None:
        dialog._advanced_dialog.close()
    dialog.close()


def test_advanced_dialog_shows_plot_for_time_series_rows(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())
    dialog._open_advanced_dialog()
    advanced = dialog._advanced_dialog
    assert advanced is not None

    found_plot = False
    for row in range(advanced._table.rowCount()):
        label_item = advanced._table.item(row, 1)
        if label_item is None:
            continue
        if label_item.text().startswith("nexus_time_series."):
            if advanced._table.cellWidget(row, 3) is not None:
                found_plot = True
                break

    assert found_plot
    if dialog._advanced_dialog is not None:
        dialog._advanced_dialog.close()
    dialog.close()


def test_advanced_dialog_shows_psi_temperature_log_provenance(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_psi_dataset_with_temperature_log())
    dialog._open_advanced_dialog()
    advanced = dialog._advanced_dialog
    assert advanced is not None

    source_row = _row_containing_field(advanced._table, "psi_temperature_log.source_file")
    assert source_row >= 0
    value_item = advanced._table.item(source_row, 2)
    assert value_item is not None
    assert value_item.text() == "/tmp/run_4321_templs0.mon"

    series_row = _row_containing_field(
        advanced._table,
        "nexus_time_series.psi_temperature/Temp_Heater.mean",
    )
    assert series_row >= 0
    assert advanced._table.cellWidget(series_row, 3) is not None

    if dialog._advanced_dialog is not None:
        dialog._advanced_dialog.close()
    dialog.close()


def test_advanced_dialog_shows_musrroot_slow_control_log_provenance(
    qapp: QApplication,
) -> None:
    dialog = RunInfoDialog(_root_dataset_with_temperature_log())
    dialog._open_advanced_dialog()
    advanced = dialog._advanced_dialog
    assert advanced is not None

    source_row = _row_containing_field(advanced._table, "musrroot_slow_control_log.source_file")
    assert source_row >= 0
    value_item = advanced._table.item(source_row, 2)
    assert value_item is not None
    assert value_item.text() == "/tmp/lem.root"

    series_row = _row_containing_field(
        advanced._table,
        "nexus_time_series.musrroot_slow_control/Sample Temperature.mean",
    )
    assert series_row >= 0
    assert advanced._table.cellWidget(series_row, 3) is not None

    channel_row = _row_containing_field(
        advanced._table,
        "musrroot_slow_control_log.channel.Sample Temperature",
    )
    assert channel_row >= 0
    assert advanced._table.cellWidget(channel_row, 3) is not None

    if dialog._advanced_dialog is not None:
        dialog._advanced_dialog.close()
    dialog.close()


def test_advanced_dialog_search_bar_is_enabled(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())
    dialog._open_advanced_dialog()
    advanced = dialog._advanced_dialog
    assert advanced is not None

    assert advanced._search_bar.isEnabled()
    assert not advanced._search_bar.isReadOnly()

    if dialog._advanced_dialog is not None:
        dialog._advanced_dialog.close()
    dialog.close()


def test_advanced_dialog_search_filters_rows(qapp: QApplication) -> None:
    dialog = RunInfoDialog(_dataset())
    dialog._open_advanced_dialog()
    advanced = dialog._advanced_dialog
    assert advanced is not None

    advanced._filter_rows("sample/Temp_Sample")

    visible_rows = [
        row for row in range(advanced._table.rowCount()) if not advanced._table.isRowHidden(row)
    ]
    assert visible_rows
    assert len(visible_rows) < advanced._table.rowCount()

    for row in visible_rows:
        field_item = advanced._table.item(row, 1)
        value_item = advanced._table.item(row, 2)
        field_text = field_item.text() if field_item is not None else ""
        value_text = value_item.text() if value_item is not None else ""
        assert "sample/temp_sample" in (field_text + " " + value_text).lower()

    if dialog._advanced_dialog is not None:
        dialog._advanced_dialog.close()
    dialog.close()
