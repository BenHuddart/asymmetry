"""Targeted tests for DataBrowserPanel behavior."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel, Qt
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
        grouping=grouping or {
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
    action_texts = [a.text() for a in actions]
    assert "Separate Combined" in action_texts
    assert "Remove Entry" in action_texts


def test_context_menu_shows_form_group_for_combined_and_regular_selection(qapp: QApplication) -> None:
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
    assert "nexus_fields.sample.temperature" in state["extra_columns"]

    panel.clear()
    panel.add_dataset(ds)
    panel.restore_state(state)

    header_labels = [panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())]
    assert "nexus_fields.sample.temperature" in header_labels


def test_orientation_extra_column_uses_friendly_header(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset(303)
    ds.metadata["nexus_fields"] = {"sample": {"shape": "plate"}}
    panel.add_dataset(ds)

    panel.add_extra_column("nexus_fields.sample.shape")

    header_labels = [panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())]
    assert "Orientation" in header_labels


def test_run_info_synthetic_extra_columns_render_values(qapp: QApplication) -> None:
    panel = DataBrowserPanel()
    ds = _dataset_with_run(302)
    panel.add_dataset(ds)

    panel.add_extra_column("run_info.points")
    panel.add_extra_column("run_info.histograms")
    panel.add_extra_column("run_info.counts_mev")

    labels = [panel._table.horizontalHeaderItem(i).text() for i in range(panel._table.columnCount())]
    points_col = labels.index("Points")
    hist_col = labels.index("Histograms")
    mev_col = labels.index("Counts (MEv)")

    assert panel._table.item(0, points_col).text() == str(ds.n_points)
    assert panel._table.item(0, hist_col).text() == "2"
    # (10+20+30+5+10+15) / 1e6 = 9e-05
    assert float(panel._table.item(0, mev_col).text()) == pytest.approx(9.0e-05)


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


def test_form_data_group_accepts_combined_dataset(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_coadd_blocks_different_grouping(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_coadd_blocks_mixed_wim_and_non_wim(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = DataBrowserPanel()
    d1 = _dataset_with_run(111, source_file="/tmp/run_111.wim")
    d2 = _dataset_with_run(112, source_file="/tmp/run_112.nxs")
    panel.add_dataset(d1)
    panel.add_dataset(d2)

    captured = {"text": ""}

    def _stub_warning(_parent, _title, text):
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", _stub_warning)

    panel._table.selectRow(0)
    idx = panel._table.model().index(1, 0)
    panel._table.selectionModel().select(
        idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    )
    panel._coadd_selected()

    assert "Mixed .wim and non-WIM" in captured["text"]
    assert not any(rn < 0 for rn in panel._datasets)


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
    assert non_empty_rows[1][:6] == ["Run", "Title", "𝑇 (K)", "𝐵 (G)", "Comment", "Orientation"]
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
