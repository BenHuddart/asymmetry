"""Tests for composite parameter expression-builder dialog."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox

from asymmetry.core.fitting.composite_parameters import CompositeParameterDefinition
from asymmetry.gui.panels.composite_parameter_dialog import CompositeParameterDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _ok_enabled(dialog: CompositeParameterDialog) -> bool:
    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    return bool(ok is not None and ok.isEnabled())


def test_dialog_disables_ok_for_invalid_expression(qapp: QApplication) -> None:
    dialog = CompositeParameterDialog(
        available_parameters=["A0", "Lambda"],
        existing_parameter_names=["A0", "Lambda"],
    )

    dialog._name_edit.setText("Lambda_eff")
    dialog._expression_edit.setText("A0 +")

    assert _ok_enabled(dialog) is False
    assert (
        "invalid" in dialog._status_label.text().lower()
        or "unsupported" in dialog._status_label.text().lower()
    )


def test_dialog_rejects_duplicate_name(qapp: QApplication) -> None:
    dialog = CompositeParameterDialog(
        available_parameters=["A0", "Lambda"],
        existing_parameter_names=["A0", "Lambda"],
    )

    dialog._name_edit.setText("A0")
    dialog._expression_edit.setText("Lambda + 1")

    assert _ok_enabled(dialog) is False
    assert "already exists" in dialog._status_label.text().lower()


def test_dialog_accepts_valid_expression_and_returns_definition(qapp: QApplication) -> None:
    dialog = CompositeParameterDialog(
        available_parameters=["A0", "Lambda"],
        existing_parameter_names=["A0", "Lambda"],
        preview_values={"A0": 0.2, "Lambda": 0.4},
        preview_uncertainties={"A0": 0.01, "Lambda": 0.02},
    )

    dialog._name_edit.setText("Lambda_eff")
    dialog._expression_edit.setText("sqrt(A0^2 + Lambda^2)")

    assert _ok_enabled(dialog) is True
    assert "preview" in dialog._preview_label.text().lower()

    dialog._on_accept()
    definition = dialog.composite_definition()
    assert definition is not None
    assert definition.name == "Lambda_eff"
    assert definition.expression == "sqrt(A0^2 + Lambda^2)"


def test_insert_parameter_button_inserts_selected_parameter(qapp: QApplication) -> None:
    dialog = CompositeParameterDialog(
        available_parameters=["A0", "Lambda"],
        existing_parameter_names=["A0", "Lambda"],
    )
    dialog._parameter_combo.setCurrentText("Lambda")
    dialog._insert_parameter_button.click()

    assert dialog._expression_edit.text() == "Lambda"


def test_dialog_edit_mode_allows_original_name(qapp: QApplication) -> None:
    dialog = CompositeParameterDialog(
        available_parameters=["A0", "Lambda"],
        existing_parameter_names=["A0", "Lambda", "Lambda_eff"],
        initial_definition=CompositeParameterDefinition(
            name="Lambda_eff",
            expression="A0 + Lambda",
        ),
        preview_values={"A0": 0.2, "Lambda": 0.4},
        preview_uncertainties={"A0": 0.01, "Lambda": 0.02},
    )

    assert "Edit" in dialog.windowTitle()
    assert dialog._name_edit.text() == "Lambda_eff"
    assert _ok_enabled(dialog) is True
