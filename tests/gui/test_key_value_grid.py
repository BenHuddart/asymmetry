"""Tests for the shared KeyValueGrid read-only results grid."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from asymmetry.gui.widgets.key_value_grid import KeyValueGrid  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_set_rows_populates_label_value_pairs(qapp) -> None:
    grid = KeyValueGrid()
    grid.set_rows([("χ²/ν", "1.04"), ("npar", "5")])
    # Two rows × two columns = four widgets.
    assert grid._grid.count() == 4
    label = grid._grid.itemAtPosition(0, 0).widget()
    value = grid._grid.itemAtPosition(0, 1).widget()
    assert isinstance(label, QLabel) and label.text() == "χ²/ν"
    assert isinstance(value, QLabel) and value.text() == "1.04"
    # Values are selectable so a user can copy a figure.
    assert value.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse


def test_value_supports_rich_text(qapp) -> None:
    grid = KeyValueGrid()
    grid.set_rows([("verdict", '<span style="color:#2a7a3f;">good</span>')])
    value = grid._grid.itemAtPosition(0, 1).widget()
    assert value.textFormat() == Qt.TextFormat.RichText
    assert "good" in value.text()


def test_repopulation_clears_previous_rows(qapp) -> None:
    grid = KeyValueGrid()
    grid.set_rows([("a", "1"), ("b", "2"), ("c", "3")])
    assert grid._grid.count() == 6
    grid.set_rows([("only", "1")])
    # Repopulating replaces, not appends — no leaked rows.
    assert grid._grid.count() == 2
    label = grid._grid.itemAtPosition(0, 0).widget()
    assert label.text() == "only"


def test_empty_rows_clears_grid(qapp) -> None:
    grid = KeyValueGrid()
    grid.set_rows([("a", "1")])
    grid.set_rows([])
    assert grid._grid.count() == 0
