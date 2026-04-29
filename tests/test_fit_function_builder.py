"""Tests for fit-function builder dialog."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.gui.panels.fit_function_builder import (
    FitFunctionBuilderDialog,
    _ComponentSelectorButton,
)


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
    assert isinstance(row_component, _ComponentSelectorButton)
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

    headers = [
        dialog._table.horizontalHeaderItem(i).text() for i in range(dialog._table.columnCount())
    ]
    assert headers == ["Op", "(", "Component", "Info", ")", "Remove"]

    info_btn = dialog._table.cellWidget(0, 3)
    assert info_btn is not None
    assert info_btn.text() == "Info"


def test_component_selector_includes_muon_fluorine_submenu(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    component_widget = dialog._table.cellWidget(0, 2)

    assert isinstance(component_widget, _ComponentSelectorButton)
    menu = component_widget._build_component_menu()
    assert menu is not None

    submenu_titles = [action.text() for action in menu.actions() if action.menu() is not None]
    assert "Muon-Fluorine" in submenu_titles

    muon_items: list[str] = []
    for action in menu.actions():
        submenu = action.menu()
        if submenu is None or action.text() != "Muon-Fluorine":
            continue
        muon_items = [
            sub_action.text() for sub_action in submenu.actions() if sub_action.isEnabled()
        ]
        break

    assert muon_items
    assert "MuF" in muon_items
    assert "FmuF_Linear" in muon_items
    assert "FmuF_General" in muon_items
