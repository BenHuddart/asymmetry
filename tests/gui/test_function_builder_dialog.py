"""Tests for the structured two-panel function-builder dialog shell."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox

from asymmetry.core.fitting.composite import (
    COMPONENTS,
    QUADRATURE_OPERATOR,
    CompositeModel,
)
from asymmetry.gui.widgets.function_builder.dialog import (
    FunctionBuilderDialog,
    make_component_expression_parser,
    make_fit_expression_parser,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fit_dialog(initial_expression: str = "Exponential + Constant") -> FunctionBuilderDialog:
    return FunctionBuilderDialog(
        title="Build Fit Function",
        expression_prefix="A(t)",
        component_definitions=COMPONENTS,
        model_parser=CompositeModel.from_expression,
        expression_parser=make_fit_expression_parser(),
        initial_expression=initial_expression,
        enable_fraction_groups=True,
    )


def _ok(dialog: FunctionBuilderDialog):
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)


# ------------------------------------------------------------------ round-trip
def test_initial_expression_seeds_rows(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential * ( Gaussian + Constant )")
    names, ops, opens, closes, _f = dialog._rows.structure()
    assert names == ["Exponential", "Gaussian", "Constant"]
    assert ops == ["*", "+"]
    assert opens == [0, 1, 0]
    assert closes == [0, 0, 1]


def test_initial_fraction_expression_roundtrips(qapp: QApplication) -> None:
    dialog = _fit_dialog("( Exponential + Gaussian ){frac} + Constant")
    _n, _ops, _o, _c, fracs = dialog._rows.structure()
    assert fracs == [(0, 1)]
    assert "frac" in dialog._rows.expression()


# --------------------------------------------------------------------- gating
def test_ok_gating_empty_disabled(qapp: QApplication) -> None:
    dialog = _fit_dialog("")
    ok = _ok(dialog)
    assert ok is not None
    assert ok.isEnabled() is False


def test_ok_gating_valid_enabled(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    ok = _ok(dialog)
    assert ok is not None
    assert ok.isEnabled() is True


def test_preview_updates_on_valid(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    assert "Preview: A(t) =" in dialog._preview_label.text()


def test_preview_lists_fraction_weights(qapp: QApplication) -> None:
    dialog = _fit_dialog("( Exponential + Gaussian ){frac} + Constant")
    text = dialog._preview_label.text()
    assert "Preview: A(t) =" in text
    assert "Fraction group" in text


def test_built_model_is_composite(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    dialog._on_accept()
    model = dialog.built_model()
    assert isinstance(model, CompositeModel)
    assert model.component_names == ["Exponential", "Constant"]


# ------------------------------------------------------- unparseable seed
def test_unparseable_initial_expression_opens_in_text_mode_with_error(qapp: QApplication) -> None:
    # Regression: an initial_expression that fails to parse (e.g. a bad name
    # pasted/restored from elsewhere) must not silently open an empty
    # structured editor discarding the user's original text. It must switch to
    # text mode seeded with the raw string and surface the parse error.
    dialog = _fit_dialog("NotAComponent + Constant")
    assert dialog._stack.currentWidget() is dialog._text_edit
    assert dialog._text_edit.toPlainText() == "NotAComponent + Constant"
    assert dialog._status_label.text()  # error surfaced
    ok = _ok(dialog)
    assert ok is not None
    assert ok.isEnabled() is False


# --------------------------------------------------------------- library add
def test_library_activation_appends_component(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    dialog._library.component_activated.emit("Gaussian")
    names, _ops, _o, _c, _f = dialog._rows.structure()
    assert names == ["Exponential", "Constant", "Gaussian"]


# --------------------------------------------------------------- text mode
def test_text_mode_apply_valid_updates_rows(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    dialog._toggle_text_mode()
    assert dialog._stack.currentWidget() is dialog._text_edit
    dialog._text_edit.setPlainText("Gaussian + Exponential + Constant")
    dialog._apply_text()
    assert dialog._stack.currentWidget() is dialog._rows_page
    names, _ops, _o, _c, _f = dialog._rows.structure()
    assert names == ["Gaussian", "Exponential", "Constant"]


def test_text_mode_invalid_stays_with_error(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("Exponential +")
    result = dialog._apply_text()
    assert result is False
    assert dialog._stack.currentWidget() is dialog._text_edit
    assert dialog._status_label.text()  # error surfaced


def test_ok_from_text_mode_validates(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("Gaussian + Constant")
    dialog._on_accept()
    model = dialog.built_model()
    assert isinstance(model, CompositeModel)
    assert model.component_names == ["Gaussian", "Constant"]


def test_text_mode_preserves_frac_syntax(qapp: QApplication) -> None:
    dialog = _fit_dialog("( Exponential + Gaussian ){frac}")
    dialog._toggle_text_mode()
    assert "{frac}" in dialog._text_edit.toPlainText()


# ------------------------------------------------------ trending grammar (⊕)
def test_quadrature_operator_accepted_with_component_parser(qapp: QApplication) -> None:
    allowed = {"Exponential", "Gaussian", "Constant"}
    operators = ("+", "-", "*", "/", QUADRATURE_OPERATOR)
    parser = make_component_expression_parser(
        allowed_components=allowed,
        allowed_operators=set(operators),
    )
    names, ops, _o, _c, fracs = parser(f"Exponential {QUADRATURE_OPERATOR} Gaussian")
    assert names == ["Exponential", "Gaussian"]
    assert ops == [QUADRATURE_OPERATOR]
    assert fracs == []


def test_component_parser_dialog_roundtrips_quadrature(qapp: QApplication) -> None:
    allowed = {"Exponential", "Gaussian", "Constant"}
    operators = ("+", "-", "*", "/", QUADRATURE_OPERATOR)
    dialog = FunctionBuilderDialog(
        title="Trending",
        expression_prefix="y",
        component_definitions={k: COMPONENTS[k] for k in allowed},
        model_parser=lambda expr: expr,  # trivial: echo the expression
        expression_parser=make_component_expression_parser(
            allowed_components=allowed,
            allowed_operators=set(operators),
        ),
        initial_expression=f"Exponential {QUADRATURE_OPERATOR} Gaussian",
        operators=operators,
        enable_fraction_groups=False,
    )
    names, ops, _o, _c, _f = dialog._rows.structure()
    assert names == ["Exponential", "Gaussian"]
    assert ops == [QUADRATURE_OPERATOR]
    # Fraction actions are hidden when fractions are disabled.
    assert dialog._group_fraction_button.isVisible() is False


# ------------------------------------------------------------ grouping actions
def test_group_action_wraps_selection(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Gaussian + Constant")
    dialog._rows._selected_indices = {0, 1}
    dialog._update_action_buttons()
    dialog._group_selection()
    _n, _ops, opens, closes, _f = dialog._rows.structure()
    assert opens == [1, 0, 0]
    assert closes == [0, 1, 0]


def test_group_as_fractions_action(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Gaussian + Constant")
    dialog._rows._selected_indices = {0, 1}
    dialog._update_action_buttons()
    dialog._group_selection_as_fractions()
    _n, _ops, _o, _c, fracs = dialog._rows.structure()
    assert fracs == [(0, 1)]
