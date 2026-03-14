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


def _click_row(panel: DataBrowserPanel, row: int, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
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
    # Should have "Co-add Selected", "Form Data Group", and remove action.
    assert len(actions) == 3
    action_texts = [a.text() for a in actions]
    assert "Co-add Selected" in action_texts
    assert "Form Data Group" in action_texts
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
    target_action = next((a for a in send_action.menu().actions() if a.text() == "Target Group"), None)
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


def test_default_group_name_detects_near_constant_temperature_with_tolerance(qapp: QApplication) -> None:
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
