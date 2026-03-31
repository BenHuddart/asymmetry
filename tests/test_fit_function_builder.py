"""Tests for fit-function builder dialog."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_dialog_builds_default_model(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.component_names == ["Exponential", "Constant"]
    assert model.operators == ["+"]


def test_dialog_add_component_updates_formula(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._add_component_row("Constant", "+")

    row_component = dialog._table.cellWidget(1, 2)
    assert isinstance(row_component, QComboBox)
    row_component.setCurrentText("Constant")

    row_op = dialog._table.cellWidget(1, 0)
    assert isinstance(row_op, QComboBox)
    row_op.setCurrentText("-")

    dialog._update_formula_preview()
    assert "A(t) =" in dialog._formula_label.text()
    assert "A_bg" in dialog._formula_label.text()


def test_dialog_prepopulate_model(qapp: QApplication) -> None:
    initial = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    dialog = FitFunctionBuilderDialog(initial_model=initial)

    assert dialog._table.rowCount() == 2
    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.component_names == ["Gaussian", "Constant"]
    assert model.operators == ["+"]


def test_dialog_builds_parenthesized_model(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._add_component_row("Constant", op="+", open_count=1, close_count=1)

    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.open_parentheses == [0, 0, 1]
    assert model.close_parentheses == [0, 0, 1]


def test_dialog_has_info_column_and_button(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()

    headers = [dialog._table.horizontalHeaderItem(i).text() for i in range(dialog._table.columnCount())]
    assert headers == ["Op", "(", "Component", "Info", ")", "Remove"]

    info_btn = dialog._table.cellWidget(0, 3)
    assert info_btn is not None
    assert info_btn.text() == "Info"
