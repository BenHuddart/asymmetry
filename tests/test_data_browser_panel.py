"""Targeted tests for DataBrowserPanel behavior."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
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
    d1 = _dataset(10)
    d2 = _dataset(11, t_shift=0.1)  # force interpolation path

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
    assert len(actions) == 1

    actions[0].trigger()

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
    # Should have "Co-add Selected" and "Remove Selected Entries"
    assert len(actions) == 2
    action_texts = [a.text() for a in actions]
    assert "Co-add Selected" in action_texts
    assert "Remove Selected Entries" in action_texts


def test_context_menu_shows_separate_for_combined_dataset(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(51)
    d2 = _dataset(52, t_shift=0.1)

    panel.add_dataset(d1)
    panel.add_dataset(d2)

    # Co-add the datasets
    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    # Select the combined dataset row
    for row in range(panel._table.rowCount()):
        item = panel._table.item(row, 0)
        if item:
            rn = item.data(Qt.ItemDataRole.UserRole)
            if rn in panel._combined_datasets:
                panel._table.selectRow(row)
                break

    menu = panel._create_table_context_menu()

    assert menu is not None
    actions = [a for a in menu.actions() if not a.isSeparator()]
    # Should have "Separate Combined" and "Remove Entry"
    assert len(actions) == 2
    action_texts = [a.text() for a in actions]
    assert "Separate Combined" in action_texts
    assert "Remove Entry" in action_texts


def test_coadd_inserts_at_first_selected_position(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(61)
    d2 = _dataset(62)
    d3 = _dataset(63)

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


def test_separate_inserts_at_combined_dataset_position(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset(71)
    d2 = _dataset(72, t_shift=0.1)
    d3 = _dataset(73)

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
    # Select the combined dataset
    panel._table.selectRow(0)
    panel._separate_combined()

    # After separation, we should have: d1 (row 0), d2 (row 1), d3 (row 2)
    assert panel._table.rowCount() == 3
    row_0_item = panel._table.item(0, 0)
    row_1_item = panel._table.item(1, 0)
    row_2_item = panel._table.item(2, 0)
    assert row_0_item.data(Qt.ItemDataRole.UserRole) in [71, 72]
    assert row_1_item.data(Qt.ItemDataRole.UserRole) in [71, 72]
    assert row_2_item.data(Qt.ItemDataRole.UserRole) == 73
