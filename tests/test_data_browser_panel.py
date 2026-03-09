"""Targeted tests for DataBrowserPanel behavior."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel
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
