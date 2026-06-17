"""Unified extra-column model: metadata + custom columns, persistence, rename.

Covers milestone M1 of the custom-columns feature — the column model behind the
data browser. The interactive '+' button, dialogs and context menu are exercised
separately; here we drive the panel API directly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.data_browser import (
    ANGLE_COLUMN_ID,
    ANGLE_COLUMN_LABEL,
    CUSTOM_FIELDS_METADATA_KEY,
    EXTRA_COLUMN_CUSTOM,
    EXTRA_COLUMN_METADATA,
    DataBrowserPanel,
    ExtraColumn,
)


def _trigger_add_field_action(panel, text):
    """Trigger an action from the rail "+" menu by its label, returning the action."""
    menu = panel._build_add_field_menu()
    for action in menu.actions():
        if action.text() == text:
            action.trigger()
            return action
    raise AssertionError(f"no add-field menu action labelled {text!r}")


def _angle_cell_item(panel, row=0):
    """Return the table item for the Angle column on a given row."""
    visible = [c.id for c in panel._visible_extra_columns()]
    col_idx = len(panel._COLUMNS) + visible.index(ANGLE_COLUMN_ID)
    return panel._table.item(row, col_idx)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dataset(run_number: int) -> MuonDataset:
    t = np.linspace(0.0, 5.0, 20)
    return MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.2 * t),
        error=np.full_like(t, 0.01),
        metadata={
            "run_number": run_number,
            "title": "sample",
            "temperature": 10.0,
            "field": 100.0,
        },
    )


def test_add_custom_column_is_empty_and_editable(qapp):
    panel = DataBrowserPanel()
    ds = _dataset(1)
    panel.add_dataset(ds)

    column = panel.add_custom_column("Anneal")
    assert column is not None
    assert column.kind == EXTRA_COLUMN_CUSTOM
    assert column.id.startswith("custom:")
    assert column.source_key is None

    # Empty by default; the cell exists and is user-editable.
    assert panel.custom_column_value(ds, column.id) == ""
    col_idx = len(panel._COLUMNS)  # first extra column
    item = panel._table.item(0, col_idx)
    assert item is not None
    assert item.flags() & Qt.ItemFlag.ItemIsEditable
    assert panel._table.horizontalHeaderItem(col_idx).text() == "Anneal"


def test_custom_value_edit_stored_in_dataset_metadata(qapp):
    panel = DataBrowserPanel()
    ds = _dataset(2)
    panel.add_dataset(ds)
    column = panel.add_custom_column("Anneal")

    # Simulate a user edit of the cell.
    col_idx = len(panel._COLUMNS)
    panel._table.item(0, col_idx).setText("annealed")

    assert panel.custom_column_value(ds, column.id) == "annealed"
    assert ds.metadata[CUSTOM_FIELDS_METADATA_KEY][column.id] == "annealed"

    # Clearing the cell removes the stored value rather than storing "".
    panel._table.item(0, col_idx).setText("")
    assert panel.custom_column_value(ds, column.id) == ""
    assert column.id not in ds.metadata.get(CUSTOM_FIELDS_METADATA_KEY, {})


def test_column_defs_round_trip_through_state(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(3))
    panel.add_extra_column("run_info.points")
    custom = panel.add_custom_column("Anneal")

    state = panel.get_state()
    # Defs serialise as dicts (not bare strings).
    assert all(isinstance(entry, dict) for entry in state["extra_columns"])

    restored = DataBrowserPanel()
    restored.add_dataset(_dataset(3))
    restored.restore_state(state)

    cols = restored.extra_columns()
    kinds = {c.kind for c in cols}
    assert kinds == {EXTRA_COLUMN_METADATA, EXTRA_COLUMN_CUSTOM}
    metadata_col = next(c for c in cols if c.kind == EXTRA_COLUMN_METADATA)
    assert metadata_col.source_key == "run_info.points"
    assert metadata_col.label == "Points"  # registry display label
    custom_col = next(c for c in cols if c.kind == EXTRA_COLUMN_CUSTOM)
    assert custom_col.id == custom.id
    assert custom_col.label == "Anneal"


def test_legacy_string_extra_columns_restore_as_metadata(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(4))
    panel.restore_state({"extra_columns": ["run_info.points"]})

    cols = panel.extra_columns()
    assert len(cols) == 1
    assert cols[0].kind == EXTRA_COLUMN_METADATA
    assert cols[0].source_key == "run_info.points"
    assert cols[0].label == "Points"


def test_rename_metadata_column_keeps_source_key(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(5))
    panel.add_extra_column("nexus_fields.sample.shape")
    column = panel.extra_columns()[0]

    assert panel.rename_extra_column(column.id, "Crystal orientation") is True
    renamed = panel.extra_columns()[0]
    assert renamed.label == "Crystal orientation"
    # The underlying NeXus field is retained, and inclusion tracking still keys
    # off the source path.
    assert renamed.source_key == "nexus_fields.sample.shape"
    assert "nexus_fields.sample.shape" in panel.get_extra_columns()
    header = panel._table.horizontalHeaderItem(len(panel._COLUMNS)).text()
    assert header == "Crystal orientation"


def test_add_field_rail_button_is_themed(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(6))

    # The rail "+" is present, carries a theme stylesheet and a helpful tooltip,
    # and there is no longer a footer add-column button.
    assert panel._add_field_btn.text() == "+"
    assert panel._add_field_btn.styleSheet()
    assert "custom field" in panel._add_field_btn.toolTip().lower()
    assert not hasattr(panel, "_add_column_btn")


def test_add_field_menu_offers_custom_and_angle(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(6))

    labels = [a.text() for a in panel._build_add_field_menu().actions()]
    assert labels == ["Custom column…", ANGLE_COLUMN_LABEL]

    # The Angle entry is enabled until the singleton field exists, then disabled.
    angle = next(
        a for a in panel._build_add_field_menu().actions() if a.text() == ANGLE_COLUMN_LABEL
    )
    assert angle.isEnabled()
    panel.add_angle_column()
    angle = next(
        a for a in panel._build_add_field_menu().actions() if a.text() == ANGLE_COLUMN_LABEL
    )
    assert not angle.isEnabled()


def test_add_field_menu_custom_action_prompts(qapp, monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(6))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Anneal", True))
    _trigger_add_field_action(panel, "Custom column…")
    cols = panel.extra_columns()
    assert len(cols) == 1
    assert cols[0].kind == EXTRA_COLUMN_CUSTOM
    assert cols[0].label == "Anneal"
    assert not cols[0].is_angle


def test_add_field_menu_custom_action_cancel_adds_nothing(qapp, monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(7))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    _trigger_add_field_action(panel, "Custom column…")
    assert panel.extra_columns() == []


def test_add_field_menu_angle_action_adds_singleton(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(8))
    assert not panel.has_angle_column()

    _trigger_add_field_action(panel, ANGLE_COLUMN_LABEL)
    angle_cols = [c for c in panel.extra_columns() if c.is_angle]
    assert len(angle_cols) == 1
    col = angle_cols[0]
    assert col.id == ANGLE_COLUMN_ID
    assert col.label == ANGLE_COLUMN_LABEL
    assert col.kind == EXTRA_COLUMN_CUSTOM and col.is_custom
    assert panel.has_angle_column()

    # Idempotent: a second add returns the existing field, never a duplicate.
    again = panel.add_angle_column()
    assert again is col
    assert sum(c.is_angle for c in panel.extra_columns()) == 1


def test_angle_column_persists_round_trip(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(8))
    panel.add_angle_column()

    restored = DataBrowserPanel()
    restored.add_dataset(_dataset(8))
    restored.restore_state(panel.get_state())

    angle_cols = [c for c in restored.extra_columns() if c.is_angle]
    assert len(angle_cols) == 1
    assert angle_cols[0].id == ANGLE_COLUMN_ID
    assert angle_cols[0].label == ANGLE_COLUMN_LABEL

    # A stray is_angle flag on a metadata column is ignored (degrees ride the
    # custom value plumbing, which a metadata column does not have).
    stray = ExtraColumn.from_dict(
        {
            "id": "x",
            "label": "x",
            "kind": EXTRA_COLUMN_METADATA,
            "source_key": "x",
            "is_angle": True,
        }
    )
    assert not stray.is_angle


def test_angle_column_validates_numeric(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(8))
    panel.add_angle_column()
    panel._rebuild_table()
    item = _angle_cell_item(panel)
    dataset = panel._datasets[8]

    item.setText("45.5")
    panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
    assert panel.custom_column_value(dataset, ANGLE_COLUMN_ID) == "45.5"

    # Non-numeric is rejected: stored value and cell text both revert.
    item.setText("abc")
    panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
    assert panel.custom_column_value(dataset, ANGLE_COLUMN_ID) == "45.5"
    assert item.text() == "45.5"

    # Negative degrees accepted; blank clears.
    item.setText("-30")
    panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
    assert panel.custom_column_value(dataset, ANGLE_COLUMN_ID) == "-30"
    item.setText("")
    panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
    assert panel.custom_column_value(dataset, ANGLE_COLUMN_ID) == ""

    # Non-finite floats parse but are not valid angles — rejected like non-numeric.
    item.setText("12")
    panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
    for bad in ("inf", "-inf", "nan", "1e999"):
        item.setText(bad)
        panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
        assert panel.custom_column_value(dataset, ANGLE_COLUMN_ID) == "12"
        assert item.text() == "12"


def test_deleting_angle_field_clears_stored_values(qapp):
    # Deleting the Angle field must purge its per-run values so they cannot
    # resurrect when the (fixed-id) field is re-added.
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(8))
    panel.add_angle_column()
    panel._rebuild_table()
    dataset = panel._datasets[8]

    item = _angle_cell_item(panel)
    item.setText("30")
    panel._on_custom_column_edited(item, ANGLE_COLUMN_ID)
    assert dataset.metadata[CUSTOM_FIELDS_METADATA_KEY] == {ANGLE_COLUMN_ID: "30"}

    panel.remove_extra_column(ANGLE_COLUMN_ID)
    assert ANGLE_COLUMN_ID not in dataset.metadata.get(CUSTOM_FIELDS_METADATA_KEY, {})

    # Re-adding the Angle field starts blank rather than inheriting "30".
    panel.add_angle_column()
    panel._rebuild_table()
    assert panel.custom_column_value(dataset, ANGLE_COLUMN_ID) == ""


def test_add_field_rail_button_is_keyboard_reachable(qapp, monkeypatch):
    # The rail "+" is the sole affordance for adding a field, so it must stay
    # keyboard-reachable (Tab) and activatable — not NoFocus. Activating it opens
    # the add-field menu (stubbed here to avoid a modal popup in the test).
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(11))
    panel.resize(320, 200)
    panel.show()
    qapp.processEvents()

    btn = panel._add_field_btn
    assert btn.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert btn.focusPolicy() & Qt.FocusPolicy.TabFocus
    btn.setFocus()
    qapp.processEvents()
    assert btn.hasFocus()

    # Activating via the keyboard pops the add-field menu. Stub the menu builder
    # (consulted by the connected _show_add_field_menu slot) so no real modal menu
    # is created mid-event, and capture that the menu was exec'd.
    class _StubMenu:
        def __init__(self):
            self.execed = False

        def exec(self, *args, **kwargs):
            self.execed = True

    stub = _StubMenu()
    monkeypatch.setattr(panel, "_build_add_field_menu", lambda: stub)
    QTest.keyClick(btn, Qt.Key.Key_Space)
    qapp.processEvents()
    assert stub.execed

    # The real menu still offers both entries.
    assert [a.text() for a in DataBrowserPanel._build_add_field_menu(panel).actions()] == [
        "Custom column…",
        ANGLE_COLUMN_LABEL,
    ]


def test_add_field_rail_strip_aligns_with_header(qapp):
    # The "+" strip spans the table's top frame plus the header height so its
    # top/bottom border lines land exactly on the header's, joining seamlessly.
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(9))
    panel.resize(320, 200)
    panel.show()
    qapp.processEvents()
    panel._sync_rail_header_height()
    header = panel._table.horizontalHeader()
    assert header.height() > 0
    frame_offset = header.geometry().top()
    assert panel._add_field_btn.height() == frame_offset + header.height()

    # The button's bottom edge coincides with the header's bottom edge.
    table_row = panel._table.parentWidget()
    header_bottom = panel._table.mapTo(table_row, header.geometry().bottomLeft()).y()
    btn = panel._add_field_btn
    btn_bottom = btn.parentWidget().mapTo(table_row, btn.geometry().bottomLeft()).y()
    assert btn_bottom == header_bottom

    # The rail does not add a table column — it is a sibling widget.
    assert panel._table.columnCount() == len(panel._COLUMNS)


def test_delete_custom_column_removes_it(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(8))
    column = panel.add_custom_column("Anneal")
    assert len(panel.extra_columns()) == 1

    panel.remove_extra_column(column.id)
    assert panel.extra_columns() == []
    # The table no longer carries the column.
    assert panel._table.columnCount() == len(panel._COLUMNS)


def test_sort_custom_column_with_mixed_values_does_not_crash(qapp):
    # Regression: a custom column is empty by default with the odd numeric entry,
    # so a sort key that returned float for some rows and str/"" for others would
    # raise TypeError comparing float to str. Sorting must stay type-safe.
    panel = DataBrowserPanel()
    for rn in (1, 2, 3):
        panel.add_dataset(_dataset(rn))
    panel.add_custom_column("Anneal")
    col_idx = len(panel._COLUMNS)
    # Run 1 numeric, run 2 text, run 3 left blank.
    panel._table.item(0, col_idx).setText("300")
    panel._table.item(1, col_idx).setText("as-grown")

    # Sort ascending then descending on the custom column — must not raise.
    panel._current_sort_column = col_idx
    panel._sort_table()
    panel._on_header_clicked(col_idx)  # toggles to descending and re-sorts
    assert panel._table.rowCount() == 3


def test_corrupt_metadata_column_without_source_key_is_safe(qapp):
    # A project that dropped source_key for a metadata column must not crash:
    # from_dict backfills source_key from the id and the cell renders blank.
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(9))
    panel.restore_state({"extra_columns": [{"id": "weird", "kind": "metadata"}]})
    col = panel.extra_columns()[0]
    assert col.source_key == "weird"
    # Rendering the (unknown) metadata column does not raise.
    panel._rebuild_table()
    assert panel._table.columnCount() == len(panel._COLUMNS) + 1
