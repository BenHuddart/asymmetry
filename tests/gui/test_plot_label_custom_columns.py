"""Custom data-browser columns selectable as the plot legend label (M3)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.plot_panel import PlotPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dataset(run_number: int, custom: dict[str, str] | None = None) -> MuonDataset:
    t = np.linspace(0.0, 5.0, 40)
    metadata: dict = {"run_number": run_number}
    if custom is not None:
        metadata["custom_fields"] = dict(custom)
    return MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.3 * t),
        error=np.full_like(t, 0.01),
        metadata=metadata,
    )


def _select_label_field(panel: PlotPanel, key: str) -> None:
    idx = panel._label_field_combo.findData(key)
    assert idx >= 0, f"label field {key!r} not in combo"
    panel._label_field_combo.setCurrentIndex(idx)


def test_custom_column_offered_and_resolves_as_label(qapp):
    panel = PlotPanel()
    panel.set_custom_label_fields([("Anneal", "custom:abc123")])

    # The custom column joins the built-in label options.
    assert panel._label_field_combo.findData("custom:abc123") >= 0

    _select_label_field(panel, "custom:abc123")
    ds = _dataset(10, {"custom:abc123": "annealed"})
    assert panel._dataset_label_for(ds) == "annealed"


def test_custom_label_falls_back_to_run_when_empty(qapp):
    panel = PlotPanel()
    panel.set_custom_label_fields([("Anneal", "custom:abc123")])
    _select_label_field(panel, "custom:abc123")

    # A run with no value for this custom column falls back to its run label.
    ds = _dataset(11)
    assert panel._dataset_label_for(ds) == str(ds.run_label)


def test_pushing_columns_preserves_selection_and_tracks_rename(qapp):
    panel = PlotPanel()
    panel.set_custom_label_fields([("Anneal", "custom:abc123")])
    _select_label_field(panel, "custom:abc123")

    # Re-pushing with a renamed label keeps the same key selected and updates text.
    panel.set_custom_label_fields([("Annealing temp", "custom:abc123")])
    assert panel._label_field_combo.currentData() == "custom:abc123"
    assert panel._label_field_combo.currentText() == "Annealing temp"


def test_saved_custom_label_field_round_trips_through_state(qapp):
    panel = PlotPanel()
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib not available")
    panel.set_custom_label_fields([("Anneal", "custom:abc123")])
    _select_label_field(panel, "custom:abc123")
    state = panel.get_state()
    assert state["label_field"] == "custom:abc123"

    # A fresh panel restores the saved custom selection even before the column is
    # pushed back in (validity accepts any custom: key); once offered, it selects.
    restored = PlotPanel()
    restored.restore_state(state)
    assert restored._default_label_field == "custom:abc123"
    restored.set_custom_label_fields([("Anneal", "custom:abc123")])
    assert restored._label_field_combo.currentData() == "custom:abc123"
