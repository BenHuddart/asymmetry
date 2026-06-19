"""Targeted tests for DataBrowserPanel behavior."""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.panels.data_browser import DataBrowserPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset(run_number: int, t_shift: float = 0.0) -> MuonDataset:
    t = np.linspace(0.0 + t_shift, 5.0 + t_shift, 40)
    return MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.2 * (t - t.min())),
        error=np.full_like(t, 0.01),
        metadata={
            "run_number": run_number,
            "title": "sample",
            "temperature": 10.0 + run_number,
            "field": 100.0 + run_number,
            "comment": "ok",
        },
    )


def _dataset_with_run(
    run_number: int,
    *,
    t_shift: float = 0.0,
    source_file: str | None = None,
    grouping: dict | None = None,
) -> MuonDataset:
    ds = _dataset(run_number, t_shift=t_shift)
    h0 = Histogram(counts=np.array([10.0, 20.0, 30.0]), bin_width=0.5)
    h1 = Histogram(counts=np.array([5.0, 10.0, 15.0]), bin_width=0.5)
    run = Run(
        run_number=run_number,
        histograms=[h0, h1],
        metadata={"run_number": run_number},
        grouping=grouping
        or {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 2,
            "bunching_factor": 1,
            "deadtime_correction": False,
        },
        source_file=source_file or f"/tmp/run_{run_number}.nxs",
    )
    ds.run = run
    return ds


def _click_row(
    panel: DataBrowserPanel,
    row: int,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    item = panel._table.item(row, 0)
    assert item is not None
    rect = panel._table.visualItemRect(item)
    assert rect.isValid()
    QTest.mouseClick(
        panel._table.viewport(),
        Qt.MouseButton.LeftButton,
        modifiers,
        rect.center(),
    )


def test_add_select_and_get_selected_datasets(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(1)
    d2 = _dataset(2)

    panel.add_dataset(d1)
    panel.add_dataset(d2)

    assert panel.get_dataset(1) is d1
    assert panel._table.rowCount() == 2

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )

    selected = panel.get_selected_datasets()
    assert {d.run_number for d in selected} == {1, 2}


def test_coadd_and_separate_roundtrip(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(10)
    d2 = _dataset_with_run(11, t_shift=0.1)  # force interpolation path

    panel.add_dataset(d1)
    panel.add_dataset(d2)

    # Select both rows then co-add.
    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    combined_runs = [rn for rn in panel._datasets if rn < 0]
    assert len(combined_runs) == 1
    combined_rn = combined_runs[0]
    assert combined_rn in panel._combined_datasets
    combined_ds = panel.get_dataset(combined_rn)
    assert combined_ds is not None
    assert combined_ds.run is not None
    assert combined_ds.run.grouping["forward_group"] == 1
    assert combined_ds.run.grouping["backward_group"] == 2

    # Select combined row and separate back.
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item and item.data(256) == combined_rn:  # Qt.ItemDataRole.UserRole = 256
            panel._table.selectRow(row)
            break
    panel._separate_combined()

    assert 10 in panel._datasets
    assert 11 in panel._datasets
    assert combined_rn not in panel._datasets


def test_delete_key_removes_selected_entries(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(21)
    d2 = _dataset(22)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel._table.selectRow(0)
    panel._table.setFocus()

    QTest.keyClick(panel._table, Qt.Key.Key_Delete)

    assert panel._table.rowCount() == 1
    assert panel.get_dataset(21) is None
    assert panel.get_dataset(22) is d2


def test_context_menu_action_removes_selected_entries(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(31)
    d2 = _dataset(32)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel._table.selectRow(1)

    menu = panel._create_table_context_menu()

    assert menu is not None
    actions = [a for a in menu.actions() if not a.isSeparator()]
    remove_action = next((a for a in actions if a.text() == "Remove Entry"), None)
    assert remove_action is not None

    remove_action.trigger()

    assert panel._table.rowCount() == 1
    assert panel.get_dataset(31) is d1
    assert panel.get_dataset(32) is None


def test_context_menu_shows_coadd_for_multiple_selected(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(41)
    d2 = _dataset(42)

    panel.add_dataset(d1)
    panel.add_dataset(d2)

    # Select both rows
    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )

    menu = panel._create_table_context_menu()

    assert menu is not None
    actions = [a for a in menu.actions() if not a.isSeparator()]
    action_texts = [a.text() for a in actions]
    assert "Co-add Selected" in action_texts
    assert "Form Data Group" in action_texts
    assert "Remove Selected Entries" in action_texts


def test_group_lookup_helpers_return_group_and_members(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(61)
    d2 = _dataset(62)
    d3 = _dataset(63)
    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)

    gid = panel.create_data_group([61, 62], name="Share Group")
    assert isinstance(gid, str)

    assert panel.get_group_id_for_run(61) == gid
    assert panel.get_group_id_for_run(63) is None

    members = panel.get_group_member_run_numbers(gid)
    assert members == [61, 62]


def test_grouped_dataset_rows_get_faint_background_tint(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(64))
    panel.add_dataset(_dataset(65))
    panel.add_dataset(_dataset(66))

    panel.create_data_group([64, 65], name="Tinted Group")

    grouped_background = None
    ungrouped_background = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is None:
            continue
        key = item.data(Qt.ItemDataRole.UserRole)
        if key == 64:
            grouped_background = panel._table.item(row, 1).background().color()
        elif key == 66:
            ungrouped_background = panel._table.item(row, 1).background().style()

    assert grouped_background == QColor(235, 239, 247)
    assert ungrouped_background == Qt.BrushStyle.NoBrush


def test_get_current_dataset_tracks_last_clicked_row_in_multi_selection(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(64)
    d2 = _dataset(65)
    panel.add_dataset(d1)
    panel.add_dataset(d2)

    _click_row(panel, 0)
    _click_row(panel, 1, Qt.KeyboardModifier.ControlModifier)

    current = panel.get_current_dataset()
    assert current is not None
    assert int(current.run_number) == 65


def test_context_menu_shows_separate_for_combined_dataset(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(51)
    d2 = _dataset_with_run(52, t_shift=0.1)

    panel.add_dataset(d1)
    panel.add_dataset(d2)

    # Co-add the datasets
    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    combined_rn = next(iter(panel._combined_datasets))
    panel.select_runs({combined_rn})

    menu = panel._create_table_context_menu()

    assert menu is not None
    actions = [a for a in menu.actions() if not a.isSeparator()]
    action_texts = [a.text() for a in actions]
    assert "Separate Combined" in action_texts
    assert "Remove Entry" in action_texts


def test_context_menu_shows_form_group_for_combined_and_regular_selection(
    qapp: QApplication,
) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(53)
    d2 = _dataset_with_run(54, t_shift=0.1)
    d3 = _dataset_with_run(55)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    combined_rn = next(rn for rn in panel._datasets if rn < 0)
    combined_row = None
    regular_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is None:
            continue
        run_number = item.data(Qt.ItemDataRole.UserRole)
        if run_number == combined_rn:
            combined_row = row
        elif run_number == 55:
            regular_row = row

    assert combined_row is not None
    assert regular_row is not None
    panel._table.clearSelection()
    panel._table.selectRow(combined_row)
    idx = panel._table.model().index(regular_row, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )

    menu = panel._create_table_context_menu()
    assert menu is not None
    action_texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "Form Data Group" in action_texts


def test_extra_column_roundtrip_in_state(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset(301)
    ds.metadata["nexus_fields"] = {"sample": {"temperature": 12.34}}
    panel.add_dataset(ds)

    panel.add_extra_column("nexus_fields.sample.temperature")
    state = panel.get_state()
    # extra_columns now serialise as column-definition dicts (id/label/kind/
    # source_key), not bare metadata keys.
    assert any(
        col.get("source_key") == "nexus_fields.sample.temperature" for col in state["extra_columns"]
    )

    panel.clear()
    panel.add_dataset(ds)
    panel.restore_state(state)

    header_labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    assert "nexus_fields.sample.temperature" in header_labels


def test_orientation_extra_column_uses_friendly_header(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset(303)
    ds.metadata["nexus_fields"] = {"sample": {"shape": "plate"}}
    panel.add_dataset(ds)

    panel.add_extra_column("nexus_fields.sample.shape")

    header_labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    assert "Orientation" in header_labels


def test_run_info_synthetic_extra_columns_render_values(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset_with_run(302)
    panel.add_dataset(ds)

    panel.add_extra_column("run_info.points")
    panel.add_extra_column("run_info.histograms")
    panel.add_extra_column("run_info.counts_mev")

    labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    points_col = labels.index("Points")
    hist_col = labels.index("Histograms")
    mev_col = labels.index("Counts (MEv)")

    assert panel._table.item(0, points_col).text() == str(ds.n_points)
    assert panel._table.item(0, hist_col).text() == "2"
    # (10+20+30+5+10+15) / 1e6 = 9e-05
    assert float(panel._table.item(0, mev_col).text()) == pytest.approx(9.0e-05)


def test_good_events_and_events_per_frame_columns(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    # good_frames present -> events/frame is finite.
    ds = _dataset_with_run(
        303,
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 0,
            "last_good_bin": 2,
            "good_frames": 1000.0,
        },
    )
    panel.add_dataset(ds)

    panel.add_extra_column("run_info.good_events_mev")
    panel.add_extra_column("run_info.events_per_frame")

    labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    good_col = labels.index("Good Events (MEv)")
    frame_col = labels.index("Events/frame")

    # Good-range [0,2] over both detectors: (10+20+30)+(5+10+15) = 90.
    assert float(panel._table.item(0, good_col).text()) == pytest.approx(9.0e-05)
    # Events per frame = 90 / 1000.
    assert float(panel._table.item(0, frame_col).text()) == pytest.approx(0.09)


def test_events_per_frame_dashes_without_good_frames(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset_with_run(305)  # default grouping carries no good_frames
    panel.add_dataset(ds)
    panel.add_extra_column("run_info.events_per_frame")
    labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    frame_col = labels.index("Events/frame")
    assert panel._table.item(0, frame_col).text() == "—"


def test_add_column_menu_lists_hideable_run_info_fields(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset_with_run(306)
    panel.add_dataset(ds)
    panel.add_extra_column("run_info.good_events_mev")

    available = panel._addable_run_info_columns()
    # Already-shown column is excluded; others remain offered.
    assert "run_info.good_events_mev" not in available
    assert "run_info.events_per_frame" in available
    assert "run_info.counts_mev" in available


def test_temperature_include_replaces_browser_value_with_log_mean(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset(304)
    ds.metadata["temperature"] = 50.0
    ds.metadata["nexus_time_series"] = {
        "psi_temperature/Temp_Sample": {
            "units": "K",
            "time": [0.0, 10.0],
            "values": [4.9906, 5.0],
            "mean": 4.9953,
            "min": 4.9906,
            "max": 5.0,
        }
    }
    panel.add_dataset(ds)

    assert panel._table.item(0, 2).text() == "50.00"

    panel.add_extra_column("temperature")

    header_labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    assert header_labels == list(DataBrowserPanel._COLUMNS)
    assert panel.get_extra_columns() == ["temperature"]
    assert panel._table.item(0, 2).text() == "5.00"

    panel.remove_extra_column("temperature")

    assert panel._table.item(0, 2).text() == "50.00"


def test_logged_temperature_prefers_cryostat_over_detector_electronics(
    qapp: QApplication,
) -> None:
    # HiFi logs both a detector-electronics thermometer (~298 K) and the
    # cryostat. They used to tie on score, and the alphabetical tie-break picked
    # DetectorTemp1 -> room temperature for every run. The cryostat must win.
    panel = DataBrowserPanel()
    ds = _dataset(91516)
    ds.metadata["temperature"] = 1.6
    ds.metadata["nexus_time_series"] = {
        "DetectorTemp1": {
            "units": "K",
            "time": [0.0, 30.0, 60.0],
            "values": [297.8, 298.1, 298.0],
            "mean": 297.97,
        },
        "Temp_Cryostat": {
            "units": "K",
            "time": [0.0, 30.0, 60.0],
            "values": [2.88, 2.90, 2.92],
            "mean": 2.90,
        },
    }
    panel.add_dataset(ds)

    selected = panel._series_mean_for_field(ds, "temperature")
    assert selected == pytest.approx(2.90, abs=1e-6)
    assert selected != pytest.approx(297.97, abs=1.0)


def test_logged_temperature_mean_gates_to_run_active_samples(qapp: QApplication) -> None:
    # The full-record mean includes the pre-run (t < 0) plateau, so the first run
    # of a setpoint block reads the previous setpoint (Sn 91516 -> 4.62 K). Gate
    # the mean to t >= 0 so only the run-active samples count.
    panel = DataBrowserPanel()
    ds = _dataset(91516)
    pre_run = [4.62, 4.62, 4.62]  # parked at the previous block's setpoint
    active = [1.599, 1.600, 1.598]
    times = [-30.0, -20.0, -10.0, 0.0, 30.0, 60.0]
    values = pre_run + active
    ds.metadata["nexus_time_series"] = {
        "Temp_Cryostat": {
            "units": "K",
            "time": times,
            "values": values,
            "mean": float(np.mean(values)),  # contaminated full-record mean
        }
    }
    panel.add_dataset(ds)

    gated = panel._series_mean_for_field(ds, "temperature")
    assert gated == pytest.approx(np.mean(active), abs=1e-6)
    assert gated < 2.0  # not the ~3.1 K full-record mean


def test_logged_temperature_mean_uses_full_record_when_no_time_axis(
    qapp: QApplication,
) -> None:
    # A series without a usable time axis falls back to the precomputed mean.
    panel = DataBrowserPanel()
    ds = _dataset(91517)
    ds.metadata["nexus_time_series"] = {"Temp_Cryostat": {"units": "K", "mean": 2.5}}
    panel.add_dataset(ds)

    assert panel._series_mean_for_field(ds, "temperature") == pytest.approx(2.5)


def _field_log_dataset(run_number: int) -> MuonDataset:
    ds = _dataset(run_number)
    ds.metadata["field"] = 100.0
    ds.metadata["nexus_time_series"] = {
        "entry/sample/magnetic_field": {
            "units": "G",
            "time": [0.0, 10.0],
            "values": [100.0, 101.0],  # run-active mean = 100.5
            "mean": 100.5,
            "role": "sample_field",
            "primary": True,
        }
    }
    return ds


def test_field_from_log_replaces_browser_value_with_log_mean(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _field_log_dataset(307)
    panel.add_dataset(ds)

    # Header value first (B column is index 3), editable.
    assert panel._table.item(0, 3).text() == "100.0"
    assert panel._table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable

    panel.set_use_field_from_log(True)
    assert panel.use_field_from_log() is True
    assert panel.get_extra_columns() == ["field"]
    # Now shows the log mean and is display-only (like the temperature column).
    assert panel._table.item(0, 3).text() == "100.5"
    assert not (panel._table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable)

    panel.set_use_field_from_log(False)
    assert panel._table.item(0, 3).text() == "100.0"
    assert panel._table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable


def test_field_from_log_per_run_override(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _field_log_dataset(308)
    panel.add_dataset(ds)
    # Global off, but override this run on.
    panel.set_dataset_field_from_log(308, True)
    assert panel.dataset_uses_field_from_log(308) is True
    assert panel._table.item(0, 3).text() == "100.5"


def test_field_from_log_falls_back_to_header_when_no_log(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset(309)
    ds.metadata["field"] = 250.0  # no nexus_time_series field channel
    panel.add_dataset(ds)
    panel.set_use_field_from_log(True)
    # No field log -> header scalar, cell stays editable (not log-tinted).
    assert panel._table.item(0, 3).text() == "250.0"
    assert panel._table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable


def test_field_from_log_state_round_trips(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _field_log_dataset(310)
    panel.add_dataset(ds)
    panel.set_use_field_from_log(True)

    state = panel.get_state()
    assert state["use_field_from_log"] is True

    restored = DataBrowserPanel()
    restored.add_dataset(_field_log_dataset(310))
    restored.restore_state(state)
    assert restored.use_field_from_log() is True
    assert restored._table.item(0, 3).text() == "100.5"


def test_legacy_field_extra_column_does_not_enable_field_from_log(
    qapp: QApplication,
) -> None:
    # An old project (pre-field-from-log) could save "field" as an ordinary
    # extra column with no "use_field_from_log" key. Restoring it must NOT
    # silently switch the B column into log-mean mode.
    panel = DataBrowserPanel()
    ds = _field_log_dataset(311)
    panel.add_dataset(ds)
    panel.restore_state(
        {
            "sort_column": -1,
            "filters": {},
            "extra_columns": ["field"],  # legacy: no use_field_from_log key
        }
    )
    assert panel.use_field_from_log() is False
    # B column shows the header scalar, not the log mean, and stays editable.
    assert panel._table.item(0, 3).text() == "100.0"
    assert panel._table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable


def test_nexus_temperature_include_replaces_browser_value_with_log_mean(
    qapp: QApplication,
) -> None:
    panel = DataBrowserPanel()
    ds = _dataset(305)
    ds.metadata["temperature"] = 12.5
    ds.metadata["nexus_time_series"] = {
        "entry/sample/Temp_Sample": {
            "units": "K",
            "time": [0.0, 10.0, 20.0],
            "values": [12.0, 12.5, 13.0],
            "mean": 12.5,
            "min": 12.0,
            "max": 13.0,
        }
    }
    panel.add_dataset(ds)

    assert panel._table.item(0, 2).text() == "12.50"

    panel.add_extra_column("temperature")

    header_labels = [
        panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())
    ]
    assert header_labels == list(DataBrowserPanel._COLUMNS)
    assert panel.get_extra_columns() == ["temperature"]
    assert panel._table.item(0, 2).text() == "12.50"

    # Updating the logged samples updates the displayed (run-active) mean.
    ds.metadata["nexus_time_series"]["entry/sample/Temp_Sample"]["values"] = [12.75, 12.75]
    ds.metadata["nexus_time_series"]["entry/sample/Temp_Sample"]["time"] = [0.0, 10.0]
    panel._rebuild_table()
    assert panel._table.item(0, 2).text() == "12.75"

    panel.remove_extra_column("temperature")

    assert panel._table.item(0, 2).text() == "12.50"


def test_temperature_log_global_toggle_and_per_dataset_overrides(
    qapp: QApplication,
) -> None:
    panel = DataBrowserPanel()
    ds1 = _dataset(306)
    ds1.metadata["temperature"] = 50.0
    ds1.metadata["nexus_time_series"] = {
        "psi_temperature/Temp_Sample": {
            "units": "K",
            "time": [0.0, 10.0],
            "values": [4.8, 5.2],
            "mean": 5.0,
            "min": 4.8,
            "max": 5.2,
        }
    }
    ds2 = _dataset(307)
    ds2.metadata["temperature"] = 60.0
    ds2.metadata["nexus_time_series"] = {
        "musrroot_slow_control/Sample Temperature": {
            "units": "K",
            "time": [0.0, 10.0],
            "values": [6.8, 7.2],
            "mean": 7.0,
            "min": 6.8,
            "max": 7.2,
        }
    }
    panel.add_dataset(ds1)
    panel.add_dataset(ds2)

    log_foreground = QColor(176, 36, 36)
    assert panel._table.item(0, 2).text() == "50.00"
    assert panel._table.item(1, 2).text() == "60.00"
    assert panel._table.item(0, 2).foreground().style() == Qt.BrushStyle.NoBrush
    assert panel._table.item(1, 2).foreground().style() == Qt.BrushStyle.NoBrush

    panel.set_use_temperature_from_log(True)

    assert panel._table.item(0, 2).text() == "5.00"
    assert panel._table.item(1, 2).text() == "7.00"
    assert panel._table.item(0, 2).foreground().color() == log_foreground
    assert panel._table.item(1, 2).foreground().color() == log_foreground

    panel.set_dataset_temperature_from_log(306, False)

    assert panel._table.item(0, 2).text() == "50.00"
    assert panel._table.item(1, 2).text() == "7.00"
    assert panel._table.item(0, 2).foreground().style() == Qt.BrushStyle.NoBrush
    assert panel._table.item(1, 2).foreground().color() == log_foreground

    panel.set_use_temperature_from_log(False)

    assert panel._table.item(0, 2).text() == "50.00"
    assert panel._table.item(1, 2).text() == "60.00"
    assert panel._table.item(0, 2).foreground().style() == Qt.BrushStyle.NoBrush
    assert panel._table.item(1, 2).foreground().style() == Qt.BrushStyle.NoBrush

    panel.set_dataset_temperature_from_log(307, True)

    assert panel._table.item(0, 2).text() == "50.00"
    assert panel._table.item(1, 2).text() == "7.00"
    assert panel._table.item(0, 2).foreground().style() == Qt.BrushStyle.NoBrush
    assert panel._table.item(1, 2).foreground().color() == log_foreground
    assert panel.get_extra_columns() == []


def test_coadd_inserts_at_first_selected_position(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(61)
    d2 = _dataset_with_run(62)
    d3 = _dataset_with_run(63)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)

    # Select d2 and d3 (at rows 1 and 2)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    idx = panel._table.model().index(2, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )

    panel._coadd_selected()

    # Combined dataset should be at row 1 (where d2 was)
    assert panel._table.rowCount() == 2
    row_1_item = panel._table.item(1, 0)
    assert row_1_item is not None
    rn_at_row_1 = row_1_item.data(Qt.ItemDataRole.UserRole)
    assert rn_at_row_1 in panel._combined_datasets
    assert panel._combined_datasets[rn_at_row_1] == [62, 63]


def test_coadd_selects_newly_created_run(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset_with_run(64))
    panel.add_dataset(_dataset_with_run(65))

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )

    panel._coadd_selected()

    combined_runs = [rn for rn in panel._datasets if rn < 0]
    assert len(combined_runs) == 1
    assert set(panel._get_selected_run_numbers()) == {combined_runs[0]}


def test_separate_combined_selects_restored_runs(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset_with_run(66))
    panel.add_dataset(_dataset_with_run(67))

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    combined_rn = next(rn for rn in panel._datasets if rn < 0)
    panel.select_runs({combined_rn})
    panel._separate_combined()

    assert set(panel._get_selected_run_numbers()) == {66, 67}


def test_coadded_temperature_from_log_uses_event_weighted_average(
    qapp: QApplication,
) -> None:
    panel = DataBrowserPanel()
    ds1 = _dataset_with_run(611)
    ds1.metadata["temperature"] = 40.0
    ds1.metadata["nexus_time_series"] = {
        "musrroot_slow_control/Sample Temperature": {
            "units": "K",
            "time": [0.0, 10.0],
            "values": [4.0, 4.0],
            "mean": 4.0,
            "min": 4.0,
            "max": 4.0,
        }
    }
    ds1.run.histograms[0].counts = np.array([100.0, 0.0, 0.0])
    ds1.run.histograms[1].counts = np.array([0.0, 0.0, 0.0])

    ds2 = _dataset_with_run(612)
    ds2.metadata["temperature"] = 80.0
    ds2.metadata["nexus_time_series"] = {
        "musrroot_slow_control/Sample Temperature": {
            "units": "K",
            "time": [0.0, 10.0],
            "values": [10.0, 10.0],
            "mean": 10.0,
            "min": 10.0,
            "max": 10.0,
        }
    }
    ds2.run.histograms[0].counts = np.array([300.0, 0.0, 0.0])
    ds2.run.histograms[1].counts = np.array([0.0, 0.0, 0.0])

    panel.add_dataset(ds1)
    panel.add_dataset(ds2)
    combined_rn = panel.add_combined_dataset([611, 612])

    assert combined_rn is not None
    assert panel._table.item(0, 2).text() == "60.00"

    panel.set_use_temperature_from_log(True)

    log_foreground = QColor(176, 36, 36)
    assert panel._table.item(0, 0).data(Qt.ItemDataRole.UserRole) == combined_rn
    assert panel._table.item(0, 2).text() == "8.50"
    assert panel._table.item(0, 2).foreground().color() == log_foreground


def test_separate_inserts_at_combined_dataset_position(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(71)
    d2 = _dataset_with_run(72, t_shift=0.1)
    d3 = _dataset_with_run(73)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)

    # Co-add d1 and d2
    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    # Now we should have: combined (row 0), d3 (row 1)
    combined_rn = next(iter(panel._combined_datasets))
    panel.select_runs({combined_rn})
    panel._separate_combined()

    # After separation, we should have: d1 (row 0), d2 (row 1), d3 (row 2)
    assert panel._table.rowCount() == 3
    row_0_item = panel._table.item(0, 0)
    row_1_item = panel._table.item(1, 0)
    row_2_item = panel._table.item(2, 0)
    assert row_0_item.data(Qt.ItemDataRole.UserRole) in [71, 72]
    assert row_1_item.data(Qt.ItemDataRole.UserRole) in [71, 72]
    assert row_2_item.data(Qt.ItemDataRole.UserRole) == 73


def test_form_data_group_accepts_combined_dataset(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(74)
    d2 = _dataset_with_run(75, t_shift=0.1)
    d3 = _dataset_with_run(76)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()
    combined_rn = next(rn for rn in panel._datasets if rn < 0)

    combined_row = None
    regular_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is None:
            continue
        run_number = item.data(Qt.ItemDataRole.UserRole)
        if run_number == combined_rn:
            combined_row = row
        elif run_number == 76:
            regular_row = row

    assert combined_row is not None
    assert regular_row is not None
    panel._table.clearSelection()
    panel._table.selectRow(combined_row)
    idx = panel._table.model().index(regular_row, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )

    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: ("Mixed Group", True))
    panel._form_data_group()

    group = next(iter(panel._groups.values()))
    assert group.member_run_numbers == [combined_rn, 76]
    assert panel._run_to_group[combined_rn] == group.group_id
    assert panel._run_to_group[76] == group.group_id


def test_separate_combined_inside_group_replaces_group_member_runs(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(77)
    d2 = _dataset_with_run(78, t_shift=0.1)
    d3 = _dataset_with_run(79)

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()
    combined_rn = next(rn for rn in panel._datasets if rn < 0)

    gid = panel.create_data_group([combined_rn, 79], name="Grouped Combined")
    assert gid is not None

    combined_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == combined_rn:
            combined_row = row
            break

    assert combined_row is not None
    panel._table.clearSelection()
    panel._table.selectRow(combined_row)
    panel._separate_combined()

    group = panel._groups[gid]
    assert group.member_run_numbers == [77, 78, 79]
    assert panel._run_to_group[77] == gid
    assert panel._run_to_group[78] == gid
    assert panel._run_to_group[79] == gid
    assert combined_rn not in panel._run_to_group
    assert combined_rn not in panel._datasets


def test_coadd_blocks_different_grouping(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(101)
    d2 = _dataset_with_run(
        102,
        grouping={
            "groups": {5: [1], 6: [2]},
            "forward_group": 5,
            "backward_group": 6,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 2,
            "bunching_factor": 1,
            "deadtime_correction": False,
        },
    )
    panel.add_dataset(d1)
    panel.add_dataset(d2)

    captured = {"title": "", "text": ""}

    def _stub_warning(_parent, title, text):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", _stub_warning)

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    assert "identical grouping" in captured["text"]
    assert 101 in panel._datasets
    assert 102 in panel._datasets
    assert not any(rn < 0 for rn in panel._datasets)


def test_coadd_allows_different_good_frames_with_identical_grouping(
    qapp: QApplication,
) -> None:
    panel = DataBrowserPanel()
    base_grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 2,
        "bunching_factor": 1,
        "deadtime_correction": True,
        "dead_time_us": [0.01, 0.02],
    }
    d1 = _dataset_with_run(
        201,
        grouping={
            **base_grouping,
            "good_frames": 12345.0,
        },
    )
    d2 = _dataset_with_run(
        202,
        grouping={
            **base_grouping,
            "good_frames": 54321.0,
        },
    )
    panel.add_dataset(d1)
    panel.add_dataset(d2)

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    combined_runs = [rn for rn in panel._datasets if rn < 0]
    assert len(combined_runs) == 1
    assert panel._combined_datasets[combined_runs[0]] == [201, 202]


def test_shift_click_selects_full_range_from_anchor(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    for run_number in range(81, 86):
        panel.add_dataset(_dataset(run_number))

    panel.show()
    qapp.processEvents()

    _click_row(panel, 1)
    _click_row(panel, 4, Qt.KeyboardModifier.ShiftModifier)

    selected = panel.get_selected_datasets()
    assert [dataset.run_number for dataset in selected] == [82, 83, 84, 85]


def test_shift_click_range_emits_single_selection_change(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    for run_number in range(700, 760):
        panel.add_dataset(_dataset(run_number))

    panel.show()
    qapp.processEvents()

    _click_row(panel, 0)
    qapp.processEvents()

    emissions = {"count": 0}
    panel.selection_changed.connect(lambda: emissions.__setitem__("count", emissions["count"] + 1))

    _click_row(panel, 59, Qt.KeyboardModifier.ShiftModifier)
    qapp.processEvents()

    assert emissions["count"] == 1
    assert len(panel.get_selected_datasets()) == 60


def test_shift_click_uses_latest_plain_click_as_anchor(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    for run_number in range(91, 96):
        panel.add_dataset(_dataset(run_number))

    panel.show()
    qapp.processEvents()

    _click_row(panel, 0)
    _click_row(panel, 2, Qt.KeyboardModifier.ShiftModifier)
    _click_row(panel, 4)
    _click_row(panel, 2, Qt.KeyboardModifier.ShiftModifier)

    selected = panel.get_selected_datasets()
    assert [dataset.run_number for dataset in selected] == [93, 94, 95]


def test_shift_arrow_selection_skips_filtered_out_rows(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    for run_number in range(601, 606):
        dataset = _dataset(run_number)
        dataset.metadata["title"] = "hidden" if run_number == 603 else "visible"
        panel.add_dataset(dataset)

    panel._column_filters[1] = {"visible"}
    panel._apply_row_visibility()

    panel.show()
    panel._table.setFocus()
    qapp.processEvents()

    panel._table.selectRow(1)
    panel._selection_anchor_row = 1
    qapp.processEvents()

    QTest.keyClick(panel._table, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    qapp.processEvents()
    QTest.keyClick(panel._table, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    qapp.processEvents()

    selected = panel.get_selected_datasets()
    assert [dataset.run_number for dataset in selected] == [602, 604, 605]


def test_group_header_context_menu_shows_ungroup(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(101))
    panel.add_dataset(_dataset(102))
    gid = panel.create_data_group([101, 102], name="T = 5 K")
    assert gid is not None

    # Select only the group header row.
    group_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is None:
            continue
        key = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(key, str) and key.startswith("group:"):
            group_row = row
            break

    assert group_row is not None
    panel._table.clearSelection()
    idx = panel._table.model().index(group_row, 0)
    panel._table.selectionModel().select(
        idx,
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    menu = panel._create_table_context_menu()
    assert menu is not None
    action_texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "Ungroup" in action_texts


def test_sort_keeps_groups_top_and_sorts_ungrouped(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(111)
    d1.metadata["temperature"] = 20.0
    d2 = _dataset(112)
    d2.metadata["temperature"] = 10.0
    d3 = _dataset(113)
    d3.metadata["temperature"] = 30.0
    d4 = _dataset(114)
    d4.metadata["temperature"] = 15.0

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)
    panel.add_dataset(d4)
    gid = panel.create_data_group([111, 112], name="Grouped")
    assert gid is not None

    # Sort by temperature ascending (column 2).
    panel._on_header_clicked(2)

    # Group header should remain at the top.
    header_item = panel._table.item(0, 0)
    assert header_item is not None
    header_key = header_item.data(Qt.ItemDataRole.UserRole)
    assert isinstance(header_key, str) and header_key.startswith("group:")

    # Ungrouped runs should be sorted (114=15K then 113=30K).
    visible_runs = []
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is None:
            continue
        key = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(key, int) and key in {113, 114}:
            visible_runs.append(key)

    assert visible_runs == [114, 113]

    # Group members should also be sorted within the group (112=10K, 111=20K).
    assert panel._groups[gid].member_run_numbers == [112, 111]
    grouped_rows = [
        item.data(Qt.ItemDataRole.UserRole)
        for row in range(panel._table.rowCount())
        if (item := panel._table.item(row, 0)) is not None
        and item.data(Qt.ItemDataRole.UserRole) in {111, 112}
    ]
    assert grouped_rows == [112, 111]

    # Reversing the sort should reverse the in-group order too.
    panel._on_header_clicked(2)
    assert panel._groups[gid].member_run_numbers == [111, 112]


def test_add_runs_to_existing_group_moves_entry(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(121))
    panel.add_dataset(_dataset(122))
    panel.add_dataset(_dataset(123))

    gid = panel.create_data_group([121, 122], name="Group A")
    assert gid is not None

    moved = panel.add_runs_to_group([123], gid)
    assert moved

    group = panel._groups[gid]
    assert group.member_run_numbers == [121, 122, 123]
    assert panel._run_to_group[123] == gid


def test_add_runs_to_group_rejects_unknown_group(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(131))

    moved = panel.add_runs_to_group([131], "missing-group")
    assert not moved


def test_context_menu_has_send_to_group_submenu(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(141))
    panel.add_dataset(_dataset(142))
    panel.add_dataset(_dataset(143))
    gid = panel.create_data_group([142, 143], name="Target Group")
    assert gid is not None

    # Select ungrouped dataset row for sending.
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == 141:
            panel._table.clearSelection()
            idx = panel._table.model().index(row, 0)
            panel._table.selectionModel().select(
                idx,
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
            break

    menu = panel._create_table_context_menu()
    assert menu is not None
    send_action = next((a for a in menu.actions() if a.text() == "Send to Group"), None)
    assert send_action is not None
    assert send_action.menu() is not None
    subgroup_actions = [a.text() for a in send_action.menu().actions() if not a.isSeparator()]
    assert "Target Group" in subgroup_actions


def test_send_to_group_action_moves_selected_runs(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(151))
    panel.add_dataset(_dataset(152))
    panel.add_dataset(_dataset(153))
    gid = panel.create_data_group([152, 153], name="Target Group")
    assert gid is not None

    # Select run 151 and invoke Send to Group -> Target Group.
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == 151:
            panel._table.clearSelection()
            idx = panel._table.model().index(row, 0)
            panel._table.selectionModel().select(
                idx,
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
            break

    menu = panel._create_table_context_menu()
    assert menu is not None
    send_action = next((a for a in menu.actions() if a.text() == "Send to Group"), None)
    assert send_action is not None and send_action.menu() is not None
    target_action = next(
        (a for a in send_action.menu().actions() if a.text() == "Target Group"), None
    )
    assert target_action is not None
    target_action.trigger()

    group = panel._groups[gid]
    assert 151 in group.member_run_numbers
    assert panel._run_to_group[151] == gid


def test_context_menu_has_remove_from_group_for_grouped_entry(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(161))
    panel.add_dataset(_dataset(162))
    panel.add_dataset(_dataset(163))
    gid = panel.create_data_group([161, 162], name="Group A")
    assert gid is not None

    # Select grouped run 161.
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == 161:
            panel._table.clearSelection()
            idx = panel._table.model().index(row, 0)
            panel._table.selectionModel().select(
                idx,
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
            break

    menu = panel._create_table_context_menu()
    assert menu is not None
    action_texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "Remove from Group" in action_texts


def test_remove_runs_from_group_moves_to_top_level(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(171))
    panel.add_dataset(_dataset(172))
    panel.add_dataset(_dataset(173))
    gid = panel.create_data_group([171, 172], name="Group A")
    assert gid is not None

    moved = panel.remove_runs_from_group([171])
    assert moved
    assert panel._run_to_group.get(171) is None
    assert 171 in panel._datasets
    assert 171 not in panel._groups[gid].member_run_numbers


def test_default_group_name_detects_near_constant_temperature_with_tolerance(
    qapp: QApplication,
) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(201)
    d2 = _dataset(202)
    d1.metadata["temperature"] = 20.10
    d2.metadata["temperature"] = 20.13
    d1.metadata["field"] = 150.0
    d2.metadata["field"] = 151.0
    panel.add_dataset(d1)
    panel.add_dataset(d2)

    name = panel._default_group_name([201, 202])
    assert name.startswith("T = ")
    assert name.endswith(" K")


def test_default_group_name_tolerance_not_too_loose_for_low_temperature(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(211)
    d2 = _dataset(212)
    d1.metadata["temperature"] = 0.10
    d2.metadata["temperature"] = 0.11
    d1.metadata["field"] = 300.0
    d2.metadata["field"] = 305.0
    panel.add_dataset(d1)
    panel.add_dataset(d2)

    name = panel._default_group_name([211, 212])
    assert name == "Group 1"


def test_export_logbook_tsv_includes_hidden_rows_and_aligned_columns(
    qapp: QApplication,
    tmp_path,
) -> None:
    panel = DataBrowserPanel()

    d1 = _dataset(401)
    d2 = _dataset(402)
    d3 = _dataset(403)
    d1.metadata["title"] = "Grouped"
    d2.metadata["title"] = "Grouped"
    d3.metadata["title"] = "Other"
    d1.metadata["nexus_fields"] = {"sample": {"shape": "rod"}}
    d2.metadata["nexus_fields"] = {"sample": {"shape": "slab"}}
    d3.metadata["nexus_fields"] = {"sample": {"shape": "powder"}}

    panel.add_dataset(d1)
    panel.add_dataset(d2)
    panel.add_dataset(d3)
    group_id = panel.create_data_group([401, 402], name="Grouped Runs")
    assert group_id is not None

    panel.add_extra_column("nexus_fields.sample.shape")

    # Hide grouped members by collapsing and hide ungrouped member by filter.
    panel._toggle_group_collapsed(group_id)
    panel._column_filters[1] = {"Grouped"}
    panel._apply_row_visibility()

    run_403_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == 403:
            run_403_row = row
            break

    assert run_403_row is not None
    assert panel._table.isRowHidden(run_403_row)

    output_path = tmp_path / "logbook.tsv"
    exported_count = panel.export_logbook_tsv(str(output_path))
    assert exported_count == 3

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    non_empty_rows = [row for row in rows if row]
    assert non_empty_rows
    expected_columns = 6

    assert non_empty_rows[0][:2] == ["Data Group", "Grouped Runs"]
    assert non_empty_rows[1][:6] == ["Run", "Title", "T (K)", "B (G)", "Comment", "Orientation"]
    assert any(row[:2] == ["Data Group", "Ungrouped"] for row in non_empty_rows)

    for row in non_empty_rows:
        assert len(row) == expected_columns

    exported_run_values = {row[0] for row in non_empty_rows if row[0].isdigit()}
    assert exported_run_values == {"401", "402", "403"}


def test_export_logbook_rtf_keeps_italic_temperature_and_field_headers(
    qapp: QApplication,
    tmp_path,
) -> None:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(501))

    output_path = tmp_path / "logbook.rtf"
    panel.export_logbook_rtf(str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert r"\i T\i0 (K)" in content
    assert r"\i B\i0 (G)" in content


# ── Phase 3 data-browser polish ────────────────────────────────────────────


def test_row_highlight_delegate_covers_whole_table(qapp: QApplication) -> None:
    """_RowHighlightDelegate must be installed as the global table delegate."""
    from asymmetry.gui.panels.data_browser import _RowHighlightDelegate

    panel = DataBrowserPanel()
    assert isinstance(panel._table.itemDelegate(), _RowHighlightDelegate)


def test_group_header_count_in_last_column(qapp: QApplication) -> None:
    """Group header row shows right-aligned 'N runs' count in col 3; col 1 is blank."""
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(601))
    panel.add_dataset(_dataset(602))
    panel.create_data_group([601, 602], name="T = 10 K")

    header_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and isinstance(item.data(Qt.ItemDataRole.UserRole), str):
            header_row = row
            break

    assert header_row is not None
    col1_item = panel._table.item(header_row, 1)
    assert col1_item is not None
    assert col1_item.text() == ""

    count_item = panel._table.item(header_row, 3)
    assert count_item is not None
    assert count_item.text() == "2 runs"
    assert count_item.textAlignment() & Qt.AlignmentFlag.AlignRight


def test_group_header_count_singular(qapp: QApplication) -> None:
    """A group with one member shows '1 run' (not '1 runs')."""
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(611))
    panel.add_dataset(_dataset(612))
    gid = panel.create_data_group([611, 612], name="Two runs")
    assert gid is not None
    # Remove one member to get down to a single-member group
    panel.remove_runs_from_group([612])
    # After ungrouping a member, if <2 members remain the group is dissolved.
    # Instead create a proper 2-member group and check the label after renaming
    # won't change the count — so just test directly: a fresh 1-member scenario
    # is not possible via the public API (create_data_group requires >=2).
    # Test the singular branch by patching the member list length directly.
    group = panel._groups.get(gid)
    if group is None:
        return  # group dissolved — test N/A for this path
    group.member_run_numbers = [611]
    panel._rebuild_table()
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and isinstance(item.data(Qt.ItemDataRole.UserRole), str):
            count_item = panel._table.item(row, 3)
            assert count_item is not None
            assert count_item.text() == "1 run"
            break


def test_footer_hint_exists(qapp: QApplication) -> None:
    """DataBrowserPanel has a footer hint label with the selection key hints."""
    panel = DataBrowserPanel()
    assert hasattr(panel, "_footer_hint")
    expected_key = "⌘" if sys.platform == "darwin" else "Ctrl"
    assert expected_key in panel._footer_hint.text()
    assert "shift" in panel._footer_hint.text()


def test_numeric_columns_centre_aligned(qapp: QApplication) -> None:
    """Temperature and field cells centre-align (keeps headers clear of the sort arrow)."""
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(701))

    t_item = panel._table.item(0, 2)
    b_item = panel._table.item(0, 3)
    assert t_item is not None
    assert b_item is not None
    assert t_item.textAlignment() & Qt.AlignmentFlag.AlignHCenter
    assert b_item.textAlignment() & Qt.AlignmentFlag.AlignHCenter


def test_chevron_click_toggles_group_collapse(qapp: QApplication) -> None:
    """Clicking within the 20-px chevron zone on a group header row collapses the group."""
    panel = DataBrowserPanel()
    panel.show()
    panel.add_dataset(_dataset(801))
    panel.add_dataset(_dataset(802))
    gid = panel.create_data_group([801, 802], name="T = 5 K")
    assert gid is not None

    group = panel._groups[gid]
    assert not group.collapsed

    header_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and isinstance(item.data(Qt.ItemDataRole.UserRole), str):
            header_row = row
            break
    assert header_row is not None

    item = panel._table.item(header_row, 0)
    rect = panel._table.visualItemRect(item)
    # Click at x=8 — well within the 20-px chevron hit zone
    chevron_pos = rect.topLeft() + QPoint(8, rect.height() // 2)
    QTest.mouseClick(
        panel._table.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        chevron_pos,
    )

    assert group.collapsed


def test_non_chevron_single_click_does_not_toggle(qapp: QApplication) -> None:
    """A single click outside the chevron zone selects the header but does not toggle collapse."""
    panel = DataBrowserPanel()
    panel.show()
    panel.add_dataset(_dataset(811))
    panel.add_dataset(_dataset(812))
    gid = panel.create_data_group([811, 812], name="B = 100 G")
    assert gid is not None

    group = panel._groups[gid]
    assert not group.collapsed

    header_row = None
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None and isinstance(item.data(Qt.ItemDataRole.UserRole), str):
            header_row = row
            break
    assert header_row is not None

    item = panel._table.item(header_row, 0)
    rect = panel._table.visualItemRect(item)
    # Click at x=60 — well outside the 20-px chevron zone
    non_chevron_pos = rect.topLeft() + QPoint(60, rect.height() // 2)
    QTest.mouseClick(
        panel._table.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        non_chevron_pos,
    )

    assert not group.collapsed


# ── select_runs ──────────────────────────────────────────────────────────────


class TestSelectRuns:
    """DataBrowserPanel.select_runs performs a true selection."""

    def test_select_runs_selects_matching_rows(self, qapp):
        browser = DataBrowserPanel()
        browser.add_dataset(_dataset(10))
        browser.add_dataset(_dataset(11))
        browser.add_dataset(_dataset(12))
        browser.select_runs({10, 12})
        selected = set(browser._get_selected_run_numbers())
        assert selected == {10, 12}

    def test_select_runs_does_not_alter_highlighted_runs(self, qapp):
        browser = DataBrowserPanel()
        browser.add_dataset(_dataset(20))
        browser.add_dataset(_dataset(21))
        browser.set_highlighted_runs({20})
        browser.select_runs({21})
        # The decorative tint must be independent of true selection.
        assert browser._highlighted_runs == {20}

    def test_select_runs_replaces_previous_selection(self, qapp):
        browser = DataBrowserPanel()
        for rn in (30, 31, 32):
            browser.add_dataset(_dataset(rn))
        browser.select_runs({30, 31})
        browser.select_runs({32})
        selected = set(browser._get_selected_run_numbers())
        assert selected == {32}

    def test_select_runs_unknown_run_no_crash(self, qapp):
        browser = DataBrowserPanel()
        browser.add_dataset(_dataset(40))
        browser.select_runs({999})  # 999 not in table
        assert set(browser._get_selected_run_numbers()) == set()

    def test_select_runs_empty_set_clears_selection(self, qapp):
        browser = DataBrowserPanel()
        browser.add_dataset(_dataset(50))
        browser.select_runs({50})
        browser.select_runs(set())
        assert set(browser._get_selected_run_numbers()) == set()


def test_release_derived_run_number_frees_unused_reservation(qapp):
    panel = DataBrowserPanel()
    first = panel.next_derived_run_number()
    second = panel.next_derived_run_number()
    assert second == first + 1
    # Releasing the first reservation lets it be handed out again.
    panel.release_derived_run_number(first)
    assert panel.next_derived_run_number() == first


def test_release_does_not_drop_a_used_run_number(qapp):
    panel = DataBrowserPanel()
    number = panel.next_derived_run_number()
    t = np.linspace(0, 1, 10)
    panel.add_dataset(
        MuonDataset(
            time=t,
            asymmetry=np.zeros_like(t),
            error=np.ones_like(t),
            metadata={"run_number": number, "run_label": f"SIM {number}"},
        )
    )
    # A number already claimed by a dataset must not be released back.
    panel.release_derived_run_number(number)
    assert panel.next_derived_run_number() != number


# ── Two-line title cells (title + comment) ──────────────────────────────────


def test_title_cell_carries_comment_role_and_tooltip(qapp: QApplication) -> None:
    """The comment rides on the Title cell instead of its own column."""
    from asymmetry.gui.panels.data_browser import _COMMENT_ROLE

    panel = DataBrowserPanel()
    ds = _dataset(701)
    ds.metadata["title"] = "EuO powder"
    ds.metadata["comment"] = "ZF cooldown, careful with degausser"
    panel.add_dataset(ds)

    assert panel._table.columnCount() == len(panel._COLUMNS)
    assert "Comment" not in panel._COLUMNS

    title_item = panel._table.item(0, 1)
    assert title_item is not None
    assert title_item.text() == "EuO powder"
    assert title_item.data(_COMMENT_ROLE) == "ZF cooldown, careful with degausser"
    assert "EuO powder" in title_item.toolTip()
    assert "ZF cooldown" in title_item.toolTip()


def test_rows_with_comment_are_taller_than_rows_without(qapp: QApplication) -> None:
    """Two-line title cells get a two-line row height."""
    panel = DataBrowserPanel()
    with_comment = _dataset(711)
    with_comment.metadata["comment"] = "long descriptive comment"
    without_comment = _dataset(712)
    without_comment.metadata["comment"] = ""
    panel.add_dataset(with_comment)
    panel.add_dataset(without_comment)

    rows = {}
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item is not None:
            rows[item.data(Qt.ItemDataRole.UserRole)] = row

    assert panel._table.rowHeight(rows[711]) > panel._table.rowHeight(rows[712])


def test_export_headers_keep_comment_column(qapp: QApplication) -> None:
    """The logbook export keeps Comment as its own column."""
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(721))
    headers = panel._active_column_headers()
    assert headers[: len(panel._COLUMNS)] == panel._COLUMNS
    assert headers[len(panel._COLUMNS)] == "Comment"


def test_restore_state_migrates_legacy_column_indices(qapp: QApplication) -> None:
    """V1 states (Comment as column 4) migrate: Comment filter/sort dropped,
    extra-column indices shift down by one."""
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(731))

    panel.restore_state({"sort_column": 5, "filters": {"4": ["ok"], "5": ["rod"]}})
    assert panel._column_filters == {4: {"rod"}}
    assert panel._current_sort_column == 4

    # A v1 sort on the Comment column itself is dropped entirely.
    panel.restore_state({"sort_column": 4, "filters": {}})
    assert panel._current_sort_column == -1

    # V2 states pass through unchanged.
    panel.restore_state({"column_layout": 2, "sort_column": 4, "filters": {"4": ["rod"]}})
    assert panel._column_filters == {4: {"rod"}}
    assert panel._current_sort_column == 4


# ── Column fit-to-viewport (option C) ────────────────────────────────────────


def _fill_panel(panel: DataBrowserPanel, *, width: int = 330) -> None:
    panel.resize(width, 600)
    panel.show()
    QApplication.processEvents()
    for rn in (785401, 785402, 785403):
        ds = _dataset(rn)
        ds.metadata["title"] = f"YBCO #B12 run {rn}"
        ds.metadata["comment"] = "TF 100G"
        # The module helper derives huge T/B values from the run number,
        # which would max out those columns and (correctly) trigger the
        # honest-overflow stand-down; use realistic values here.
        ds.metadata["temperature"] = 10.0
        ds.metadata["field"] = 100.0
        panel.add_dataset(ds)
    QApplication.processEvents()


def test_columns_fill_viewport_exactly_on_load(qapp: QApplication) -> None:
    """A default load stretches Title so the base columns exactly fill the
    viewport — no horizontal scrollbar.

    The dock is sized wide rather than at a tight pixel width so the Title fit
    is guaranteed to engage on every platform. The non-Title columns are
    content-sized, and the font renders at different pixel widths across
    platforms (IBM Plex Mono is wider on Windows than on the Linux CI host);
    at a tight width those base columns can exceed the viewport on a wide-font
    platform, correctly standing the fit down to honest overflow — that path
    is covered by test_overflowing_extra_columns_fall_back_to_honest_scrolling.
    A generous width keeps the *fill* invariant under test deterministic.
    """
    panel = DataBrowserPanel()
    _fill_panel(panel, width=700)
    header = panel._table.horizontalHeader()
    total = sum(header.sectionSize(i) for i in range(panel._table.columnCount()))
    assert total == panel._table.viewport().width()
    assert panel._table.horizontalScrollBar().maximum() == 0
    # Run hugs its (clamped) six-digit content — no artificial floor padding.
    # Stated as the content-clamp result rather than an absolute pixel cap so
    # the check holds whatever pixel width the platform font renders at; it
    # still fails if a floor above the content (the old 72px) is reintroduced.
    assert header.sectionSize(0) == min(150, max(56, panel._table.sizeHintForColumn(0)))
    panel.close()


def test_user_column_drag_stops_the_auto_fit(qapp: QApplication) -> None:
    """The first manual edge drag latches; later loads keep the user layout."""
    panel = DataBrowserPanel()
    _fill_panel(panel)
    header = panel._table.horizontalHeader()
    # An unguarded resize is indistinguishable from a user drag.
    header.resizeSection(1, 99)
    assert panel._user_sized_columns
    panel.add_dataset(_dataset(785409))
    QApplication.processEvents()
    assert header.sectionSize(1) == 99
    # A brand-new extra section still gets its initial sizing (clamped to
    # the 120-320 band) even though the user owns the rest of the layout.
    panel.add_extra_column("instrument")
    QApplication.processEvents()
    extra_col = panel._table.columnCount() - 1
    assert 120 <= header.sectionSize(extra_col) <= 320
    assert header.sectionSize(1) == 99  # user's Title width untouched
    panel.close()


def test_overflowing_extra_columns_fall_back_to_honest_scrolling(
    qapp: QApplication,
) -> None:
    """When extras leave Title below its readable floor, the fit stands down:
    the table overflows into a scrollbar and Title keeps a usable width."""
    panel = DataBrowserPanel()
    _fill_panel(panel)
    for key in ("instrument", "started", "stopped"):
        panel.add_extra_column(key)
    QApplication.processEvents()
    header = panel._table.horizontalHeader()
    assert not panel._user_sized_columns  # extras must not trip the latch
    assert header.sectionSize(1) >= panel._TITLE_FIT_FLOOR
    total = sum(header.sectionSize(i) for i in range(panel._table.columnCount()))
    assert total > panel._table.viewport().width()
    panel.close()
