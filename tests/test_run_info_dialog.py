"""Tests for run information dialog include/log actions."""

from __future__ import annotations

import os

import numpy as np
import pytest

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
    t = np.linspace(0.0, 1.0, 8)
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


def _row_for_field(table, field_name: str) -> int:
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is not None and item.text() == field_name:
            return row
    return -1


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
        row
        for row in range(advanced._table.rowCount())
        if not advanced._table.isRowHidden(row)
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
