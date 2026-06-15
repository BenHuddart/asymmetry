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
from PySide6.QtWidgets import QApplication, QInputDialog

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.data_browser import (
    CUSTOM_FIELDS_METADATA_KEY,
    EXTRA_COLUMN_CUSTOM,
    EXTRA_COLUMN_METADATA,
    DataBrowserPanel,
)


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


def test_add_column_button_is_themed_and_prompts(qapp, monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(6))

    # The button is present, carries a theme stylesheet and a helpful tooltip.
    assert panel._add_column_btn.styleSheet()
    assert "custom column" in panel._add_column_btn.toolTip().lower()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Anneal", True))
    panel._add_column_btn.click()

    cols = panel.extra_columns()
    assert len(cols) == 1
    assert cols[0].kind == EXTRA_COLUMN_CUSTOM
    assert cols[0].label == "Anneal"


def test_add_column_button_cancel_adds_nothing(qapp, monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(7))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    panel._add_column_btn.click()
    assert panel.extra_columns() == []


def test_delete_custom_column_removes_it(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(8))
    column = panel.add_custom_column("Anneal")
    assert len(panel.extra_columns()) == 1

    panel.remove_extra_column(column.id)
    assert panel.extra_columns() == []
    # The table no longer carries the column.
    assert panel._table.columnCount() == len(panel._COLUMNS)
