"""Tests for the New User Function authoring dialog.

Offscreen GUI tests. Every test isolates the fit-function registries with the
shared ``registry_snapshot`` fixture and directs creation at ``tmp_path``, so a
draft registered or written here never leaks into another test. Validation is
driven directly through the dialog's ``_run_validation`` rather than pumping the
event loop for the debounce timer, so the tests are deterministic.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.core.fitting.user_function_authoring import generate_function_body
from asymmetry.gui.windows.new_user_function_dialog import NewUserFunctionDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def _registry_snapshot(registry_snapshot):
    """Alias of the shared conftest ``registry_snapshot`` fixture."""
    yield


def _ok_enabled(dialog: NewUserFunctionDialog) -> bool:
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def _set_param_row(dialog: NewUserFunctionDialog, row: int, name: str, value: float) -> None:
    dialog._param_table.item(row, 0).setText(name)
    spin = dialog._param_table.cellWidget(row, 1)
    assert isinstance(spin, QDoubleSpinBox)
    spin.setValue(value)


def _fill_valid_component(dialog: NewUserFunctionDialog, name: str = "UserGuiStretched") -> None:
    """Fill the pre-seeded component dialog with a valid stretched-exp draft."""
    dialog._name_edit.setText(name)
    dialog._description_edit.setText("A GUI-authored stretched exponential")
    dialog._formula_edit.setText("A*exp(-(x/tau)**alpha)")
    # Rename the pre-seeded A row and add tau/alpha.
    _set_param_row(dialog, 0, "A", 25.0)
    dialog._append_param_row("tau", 1.0)
    dialog._append_param_row("alpha", 1.0)


# ── initial state ───────────────────────────────────────────────────────────


def test_initial_state_ok_disabled_and_amplitude_seeded(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        assert not _ok_enabled(dialog)
        # Component kind pre-seeds a single amplitude row.
        assert dialog._param_table.rowCount() == 1
        assert dialog._param_table.item(0, 0).text() == "A"
        # Empty required fields show a neutral to-do hint, not the core's red
        # invalid-name error.
        from asymmetry.gui.styles import tokens

        text = dialog._status_label.text()
        assert text.startswith("To create a function, fill in:")
        assert "a name" in text
        assert "a formula" in text
        assert "a description" in text
        assert "Invalid" not in text
        assert tokens.TEXT_MUTED in dialog._status_label.styleSheet()
        assert tokens.ERROR not in dialog._status_label.styleSheet()
    finally:
        dialog.deleteLater()


def test_partial_fill_hint_lists_only_missing_fields(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        dialog._name_edit.setText("UserPartial")
        dialog._formula_edit.setText("A*exp(-x)")
        dialog._run_validation()

        text = dialog._status_label.text()
        assert "a description" in text
        assert "a name" not in text
        assert "a formula" not in text
        assert not _ok_enabled(dialog)
    finally:
        dialog.deleteLater()


def test_nonempty_invalid_name_still_shows_red_core_error(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        dialog._name_edit.setText("Bad Name!")  # non-empty but invalid grammar
        dialog._description_edit.setText("desc")
        dialog._formula_edit.setText("A*exp(-x)")
        _set_param_row(dialog, 0, "A", 1.0)
        dialog._run_validation()

        from asymmetry.gui.styles import tokens

        assert not _ok_enabled(dialog)
        assert "Bad Name!" in dialog._status_label.text()
        assert tokens.ERROR in dialog._status_label.styleSheet()
    finally:
        dialog.deleteLater()


def test_parameter_kind_starts_with_no_rows(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("parameter", directory=tmp_path)
    try:
        assert dialog._param_table.rowCount() == 0
        assert not _ok_enabled(dialog)
    finally:
        dialog.deleteLater()


# ── valid draft ─────────────────────────────────────────────────────────────


def test_valid_draft_enables_ok_and_plots_preview(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        _fill_valid_component(dialog)
        dialog._run_validation()

        assert _ok_enabled(dialog)
        assert "valid" in dialog._status_label.text().lower()
        # Status is green on success.
        from asymmetry.gui.styles import tokens

        assert tokens.OK in dialog._status_label.styleSheet()
        # A line was plotted on the preview axes.
        assert dialog._axes is not None
        assert len(dialog._axes.lines) >= 1
    finally:
        dialog.deleteLater()


# ── error surfacing ─────────────────────────────────────────────────────────


def test_unknown_name_surfaces_actionable_message(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        dialog._name_edit.setText("UserUnknown")
        dialog._description_edit.setText("desc")
        dialog._formula_edit.setText("A*exp(-x/taau)")  # typo: taau
        _set_param_row(dialog, 0, "A", 1.0)
        dialog._append_param_row("tau", 1.0)
        dialog._run_validation()

        assert not _ok_enabled(dialog)
        assert "taau" in dialog._status_label.text()
    finally:
        dialog.deleteLater()


def test_name_collision_with_builtin_rejected(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        dialog._name_edit.setText("Exponential")  # a built-in component
        dialog._description_edit.setText("desc")
        dialog._formula_edit.setText("A*exp(-x)")
        _set_param_row(dialog, 0, "A", 1.0)
        dialog._run_validation()

        assert not _ok_enabled(dialog)
        assert "already registered" in dialog._status_label.text()
    finally:
        dialog.deleteLater()


# ── detect parameters ───────────────────────────────────────────────────────


def test_detect_parameters_adds_missing_keeps_existing(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        # Only A present (pre-seeded); give it a distinctive value to check it
        # survives detection.
        _set_param_row(dialog, 0, "A", 0.7)
        dialog._formula_edit.setText("A*exp(-lam*x)")
        dialog._on_detect_parameters()

        names = dialog._current_param_names()
        assert names == ["A", "lam"]
        # A's start value was not touched.
        assert dialog._param_table.cellWidget(0, 1).value() == pytest.approx(0.7)
    finally:
        dialog.deleteLater()


# ── advanced toggle ─────────────────────────────────────────────────────────


def test_advanced_toggle_prefills_body_and_uses_edits(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        _fill_valid_component(dialog, name="UserAdvancedGui")
        dialog._run_validation()

        dialog._advanced_toggle.setChecked(True)
        # The formula field is hidden and the editor is shown, and the draft now
        # carries the editor text as its advanced body. (The dialog is never
        # shown in the test, so query the explicit hidden flag, not isVisible.)
        assert dialog._formula_edit.isHidden()
        assert not dialog._advanced_editor.isHidden()
        assert dialog._current_draft().advanced_body is not None

        # Editing the body with an extra harmless line still validates.
        body = dialog._advanced_editor.toPlainText()
        dialog._advanced_editor.setPlainText(
            "scale = 1.0\n" + body.replace("result =", "result = scale *")
        )
        dialog._run_validation()
        assert _ok_enabled(dialog)

        # Toggling back off under test mode reverts to formula mode (auto-discard).
        dialog._advanced_toggle.setChecked(False)
        assert not dialog._formula_edit.isHidden()
        assert dialog._advanced_editor.isHidden()
        assert dialog._current_draft().advanced_body is None
    finally:
        dialog.deleteLater()


def test_advanced_prefill_equals_generate_function_body(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        _fill_valid_component(dialog, name="UserPrefillGui")
        # Not yet advanced: build the reference body from the current draft.
        expected = generate_function_body(dialog._current_draft())
        dialog._advanced_toggle.setChecked(True)
        assert dialog._advanced_editor.toPlainText() == expected
    finally:
        dialog.deleteLater()


# ── accept ──────────────────────────────────────────────────────────────────


def test_accept_writes_file_and_registers(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        _fill_valid_component(dialog, name="UserAcceptStretched")
        dialog._run_validation()
        assert _ok_enabled(dialog)

        dialog._on_accept()

        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.created() is not None
        assert dialog.created_name() == "UserAcceptStretched"
        assert list(tmp_path.glob("*.py"))  # file written
        assert COMPONENTS["UserAcceptStretched"].user is True
    finally:
        dialog.deleteLater()


def test_accept_failure_keeps_dialog_open_and_registers_nothing(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("component", domain="time", directory=tmp_path)
    try:
        _fill_valid_component(dialog, name="UserLateCollision")
        dialog._run_validation()
        assert _ok_enabled(dialog)

        # Arrange a collision *after* the live validation passed: register the
        # same name, so create_user_function's re-validation fails on accept.
        import asymmetry

        asymmetry.register_component(
            "UserLateCollision",
            lambda t, A: A * np.asarray(t, dtype=float),
            ["A"],
            domain="time",
            description="pre-registered clash",
            formula_template="{A}*t",
        )

        dialog._on_accept()

        assert dialog.created() is None
        assert "already registered" in dialog._status_label.text()
        # Nothing half-registered / no stray file written by the failed accept.
        assert list(tmp_path.glob("*.py")) == []
    finally:
        dialog.deleteLater()


def test_accept_parameter_kind_registers_common_scope(qapp, _registry_snapshot, tmp_path):
    dialog = NewUserFunctionDialog("parameter", directory=tmp_path)
    try:
        dialog._name_edit.setText("UserGuiTrend")
        dialog._description_edit.setText("Linear GUI trend")
        dialog._formula_edit.setText("a*x+b")
        dialog._append_param_row("a", 1.0)
        dialog._append_param_row("b", 0.0)
        dialog._run_validation()
        assert _ok_enabled(dialog)

        dialog._on_accept()

        assert dialog.created_name() == "UserGuiTrend"
        definition = PARAMETER_MODEL_COMPONENTS["UserGuiTrend"]
        assert definition.user is True
        assert definition.scopes == ("common",)
    finally:
        dialog.deleteLater()
