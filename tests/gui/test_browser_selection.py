"""Rendered selection remains distinct from group membership and input focus."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]


def _row_paint(table, row: int) -> tuple:
    """Sample the fill and leading selection marker away from text and gridlines."""
    image = table.viewport().grab().toImage()
    rect = table.visualItemRect(table.item(row, 0))
    return (
        image.pixelColor(15, rect.top() + 3),
        image.pixelColor(1, rect.top() + 3),
        image.pixelColor(8, rect.center().y()),
    )


@pytest.mark.parametrize("kind", ["user", "auto", None])
@pytest.mark.parametrize("scale", [1.0, 1.2])
def test_selected_run_paint_survives_multiple_selection_and_focus_change(qapp, kind, scale):
    pytest.importorskip("PySide6")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.gui.panels.data_browser import DataBrowserPanel
    from asymmetry.gui.styles import tokens
    from asymmetry.gui.styles.fonts import set_ui_font_scale

    set_ui_font_scale(scale)
    host = QWidget()
    layout = QVBoxLayout(host)
    browser = DataBrowserPanel()
    other_input = QLineEdit()
    layout.addWidget(browser)
    layout.addWidget(other_input)
    for run_number in (1, 2, 3):
        time = np.linspace(0.0, 5.0, 20)
        browser.add_dataset(
            MuonDataset(
                time=time,
                asymmetry=np.zeros_like(time),
                error=np.full_like(time, 0.01),
                metadata={"run_number": run_number, "title": "Sample", "comment": "Comment"},
            )
        )
    if kind is not None:
        browser.create_data_group([1, 2, 3], name="Scan", kind=kind)
    host.resize(720, 360)
    host.show()
    host.activateWindow()
    table = browser._table
    table.setFocus()
    qapp.processEvents()
    offset = int(kind is not None)
    unselected_row, selected_row, current_row = (offset + i for i in range(3))
    for row, modifiers, expected in (
        (selected_row, Qt.KeyboardModifier.NoModifier, {selected_row}),
        (current_row, Qt.KeyboardModifier.ControlModifier, {selected_row, current_row}),
    ):
        QTest.mouseClick(
            table.viewport(),
            Qt.MouseButton.LeftButton,
            modifiers,
            table.visualItemRect(table.item(row, 0)).center(),
        )
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            QTest.keyRelease(table, Qt.Key.Key_Control)
        qapp.processEvents()
        assert table.hasFocus()
        assert {index.row() for index in table.selectionModel().selectedRows()} == expected
        before = [_row_paint(table, i) for i in range(table.rowCount())]
        other_input.setFocus()
        qapp.processEvents()
        assert not table.hasFocus()
        assert {index.row() for index in table.selectionModel().selectedRows()} == expected
        assert [_row_paint(table, i) for i in range(table.rowCount())] == before
        unselected_fill = before[unselected_row][0]
        for index in table.selectionModel().selectedRows():
            fill, marker, current_marker = before[index.row()]
            assert (current_marker == QColor(tokens.ACCENT)) == (
                index.row() == table.currentRow()
            ), "The current run needs its own visible pointer"
            assert marker == QColor(tokens.ACCENT), "Every selected run needs a solid marker"
            separation = sum(
                abs(a - b) for a, b in zip(fill.getRgb()[:3], unselected_fill.getRgb()[:3])
            )
            assert separation >= 40, "Selection must stand apart from the member tint"
        if kind is not None:
            assert before[0][0] == table.item(0, 0).background().color()
        table.setFocus()
    assert before[selected_row][0] != before[current_row][0]


@pytest.mark.parametrize("scale", [1.0, 1.2])
def test_current_marker_gutter_keeps_six_digit_phase_runs_readable(qapp, scale):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QStyleOptionViewItem

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.representation.group import PhaseSpec
    from asymmetry.gui.panels.data_browser import DataBrowserPanel
    from asymmetry.gui.styles.fonts import set_ui_font_scale

    set_ui_font_scale(scale)
    browser = DataBrowserPanel()
    for run_number in (123456, 123457):
        time = np.arange(2, dtype=float)
        browser.add_dataset(
            MuonDataset(time, np.zeros(2), np.ones(2), metadata={"run_number": run_number})
        )
    parent = browser.create_data_group([123456, 123457], name="Scan")
    browser._project_model.create_phase_groups(
        parent,
        [PhaseSpec(1, "Phase I", (123456, 123457), None, {}, None)],
    )
    browser.sync_groups_from_project_model()
    browser.resize(720, 360)
    browser.show()
    qapp.processEvents()
    table = browser._table
    phase_rows = [
        row
        for row in range(table.rowCount())
        if table.item(row, 0).text().strip() in {"123456", "123457"}
    ]
    assert len(phase_rows) == 2
    for row in phase_rows:
        index = table.model().index(row, 0)
        option = QStyleOptionViewItem()
        option.initFrom(table)
        required = table.itemDelegate().sizeHint(option, index).width()
        assert table.columnWidth(0) >= required, "Phase run identity must fit beside the pointer"
