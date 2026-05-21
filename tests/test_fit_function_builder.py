"""Tests for fit-function builder dialog."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox

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
    assert dialog._expression_edit.text() == "Exponential + Constant"


def test_dialog_insert_function_updates_expression_and_preview(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._expression_edit.clear()
    dialog._component_selector.setCurrentText("Constant")
    dialog._insert_component_button.click()
    dialog._insert_token(" + ")
    dialog._component_selector.setCurrentText("Gaussian")
    dialog._insert_component_button.click()

    assert dialog._expression_edit.text() == "Constant + Gaussian"
    assert "Preview:" in dialog._preview_label.text()
    assert "A_bg" in dialog._preview_label.text()


def test_dialog_invalid_expression_disables_ok(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._expression_edit.setText("Exponential +")

    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok is not None
    assert ok.isEnabled() is False
    assert "operator" in dialog._status_label.text().lower()


def test_dialog_prepopulate_model(qapp: QApplication) -> None:
    initial = CompositeModel(
        ["Gaussian", "Constant", "Constant"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )
    dialog = FitFunctionBuilderDialog(initial_model=initial)

    assert dialog._expression_edit.text() == "Gaussian * (Constant + Constant)"
    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.component_names == initial.component_names
    assert model.operators == initial.operators
    assert model.open_parentheses == initial.open_parentheses
    assert model.close_parentheses == initial.close_parentheses


def test_dialog_builds_parenthesized_model(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._expression_edit.setText("Exponential * ( Constant + Constant )")

    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.open_parentheses == [0, 1, 0]
    assert model.close_parentheses == [0, 0, 1]


def test_dialog_builds_fraction_group_model(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._expression_edit.setText("Exponential + Gaussian")
    cursor = dialog._expression_edit.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(len("Exponential + Gaussian"), cursor.MoveMode.KeepAnchor)
    dialog._expression_edit.setTextCursor(cursor)
    dialog._group_fraction_button.click()

    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.fraction_groups == [(0, 1)]
    assert "fraction_1" in model.param_names
    assert "fraction_2" in model.param_names


def test_dialog_exposes_fraction_group_ui_controls(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()

    assert dialog._group_fraction_button.text() == "Fractions"
    assert dialog._separate_fraction_button.text() == "Separate"
    assert "matching colors" in dialog._syntax_help_label.text()


def test_dialog_normalizes_fraction_syntax_to_color_groups(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._expression_edit.setText("( Exponential + Gaussian ){frac}")

    assert dialog._expression_edit.text() == "Exponential + Gaussian"
    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.fraction_groups == [(0, 1)]


def test_fraction_button_groups_selected_components(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    text = "Exponential + Gaussian + Constant"
    dialog._expression_edit.setText(text)
    cursor = dialog._expression_edit.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(len("Exponential + Gaussian"), cursor.MoveMode.KeepAnchor)
    dialog._expression_edit.setTextCursor(cursor)

    dialog._group_fraction_button.click()

    assert dialog._expression_edit.text() == text
    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.fraction_groups == [(0, 1)]
    assert "Fractions:" in dialog._preview_label.text()


def test_separate_button_removes_fraction_group(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    text = "Exponential + Gaussian + Constant"
    dialog._expression_edit.setText(text)
    cursor = dialog._expression_edit.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(len("Exponential + Gaussian"), cursor.MoveMode.KeepAnchor)
    dialog._expression_edit.setTextCursor(cursor)

    dialog._group_fraction_button.click()
    dialog._expression_edit.setTextCursor(cursor)

    dialog._separate_fraction_button.click()

    assert dialog._expression_edit.text() == text
    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.fraction_groups == []


def test_backspace_removes_whole_function_token(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._expression_edit.setText("Exponential + Constant")
    dialog._expression_edit.setCursorPosition(len(dialog._expression_edit.text()))

    dialog._backspace_expression()

    assert dialog._expression_edit.text().strip() == "Exponential +"


def test_dialog_has_info_button_and_selector(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()

    assert isinstance(dialog._component_selector, _ComponentSelectorButton)
    assert dialog._info_button.text() == "Info"
    assert dialog._component_selector.text().endswith("\u25be")


def test_component_selector_includes_muon_fluorine_submenu(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    component_widget = dialog._component_selector

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
