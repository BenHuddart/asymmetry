"""Tests for the structured two-panel function-builder dialog shell."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QFrame,
    QPushButton,
    QSplitter,
    QToolButton,
)

from asymmetry.core.fitting.composite import (
    COMPONENTS,
    QUADRATURE_OPERATOR,
    CompositeModel,
)
from asymmetry.gui.widgets.function_builder.dialog import (
    FunctionBuilderDialog,
    _prefix_to_mathtext,
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


# ------------------------------------------------ action-button enable states
def test_action_buttons_disabled_at_construction_with_no_selection(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    assert dialog._rows.selected_spans() == []
    assert dialog._group_button.isEnabled() is False
    assert dialog._group_fraction_button.isEnabled() is False
    assert dialog._ungroup_button.isEnabled() is False


def test_action_buttons_enabled_after_selecting_both_rows(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    dialog._rows._selected_indices = {0, 1}
    dialog._update_action_buttons()
    assert dialog._group_button.isEnabled() is True
    assert dialog._group_fraction_button.isEnabled() is True
    assert dialog._ungroup_button.isEnabled() is False


# ------------------------------------------------------------------ splitter
def test_library_and_rows_live_in_a_splitter(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    splitters = [w for w in dialog.findChildren(QSplitter)]
    assert len(splitters) == 1
    splitter = splitters[0]
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert splitter.indexOf(dialog._library) == 0
    assert splitter.childrenCollapsible() is False
    assert dialog._library.minimumWidth() >= 180


# ------------------------------------------------ fraction-group header button
def test_fraction_group_header_offers_absolute_amplitude_toggle(qapp: QApplication) -> None:
    dialog = _fit_dialog("( Exponential + Gaussian ){frac} + Constant")
    frames = [w for w in dialog.findChildren(QFrame) if w.objectName() == "groupFrame"]
    assert len(frames) == 1
    button_texts = [b.text() for b in frames[0].findChildren(QPushButton)]
    assert "Use absolute amplitudes" in button_texts
    assert "Ungroup" in button_texts
    assert "Use fractional amplitudes" not in button_texts


def test_plain_group_header_offers_fractional_toggle_not_absolute(qapp: QApplication) -> None:
    dialog = _fit_dialog("( Exponential + Gaussian ) + Constant")
    frames = [w for w in dialog.findChildren(QFrame) if w.objectName() == "groupFrame"]
    assert len(frames) == 1
    button_texts = [b.text() for b in frames[0].findChildren(QPushButton)]
    assert "Use fractional amplitudes" in button_texts
    assert "Use absolute amplitudes" not in button_texts


# ------------------------------------------------------------ preview card
def test_preview_card_exists_between_editor_and_status(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    root_layout = dialog.layout()
    widgets_in_order = [root_layout.itemAt(i).widget() for i in range(root_layout.count())]
    widgets_in_order = [w for w in widgets_in_order if w is not None]
    assert dialog._preview_card in widgets_in_order
    assert dialog._status_label in widgets_in_order
    assert widgets_in_order.index(dialog._preview_card) < widgets_in_order.index(
        dialog._status_label
    )


def test_preview_card_shows_composed_equation_for_real_model(qapp: QApplication) -> None:
    # CompositeModel now provides latex_terms()/latex_string(); a valid model
    # should render a real pixmap in the equation area (the composed-render
    # path), not fall back to plain text.
    dialog = _fit_dialog("Exponential + Constant")
    pixmap = dialog._equation_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert dialog._equation_label.text() == ""


def test_preview_card_fraction_weights_subline_preserved(qapp: QApplication) -> None:
    dialog = _fit_dialog("( Exponential + Gaussian ){frac} + Constant")
    assert "Fraction group" in dialog._preview_label.text()
    # The sub-line lives inside the card.
    assert dialog._preview_label.parent() is not None


def test_preview_card_disabled_when_expression_invalid(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    assert dialog._preview_card.isEnabled() is True
    # Drive the same path _on_structure_changed uses: a syntactically-parseable
    # expression that the model parser rejects (unknown component name), so
    # _validate_and_update reaches its "invalid model" branch directly.
    dialog._validate_and_update("NotAComponent")
    assert dialog._preview_card.isEnabled() is False
    assert dialog._status_label.text()  # error still surfaced below the card


def test_preview_card_empty_expression_clears_equation(qapp: QApplication) -> None:
    dialog = _fit_dialog("")
    assert dialog._preview_card.isEnabled() is False
    assert dialog._equation_label.pixmap().isNull()
    assert dialog._equation_label.text() == ""


def test_copy_button_copies_plain_formula(qapp: QApplication) -> None:
    dialog = _fit_dialog("Exponential + Constant")
    buttons = [b for b in dialog.findChildren(QToolButton) if b is dialog._copy_button]
    assert len(buttons) == 1
    dialog._copy_formula_to_clipboard()
    clipboard_text = QApplication.clipboard().text()
    assert clipboard_text == dialog._model.formula_string()
    assert clipboard_text  # non-empty for a valid model


def test_fallback_to_single_string_render_when_latex_terms_missing(qapp: QApplication) -> None:
    # Exercise the fallback chain's second rung: a model exposing
    # latex_string() but not latex_terms() (or latex_terms() returning
    # nothing usable) should render via the single-string mathtext path
    # rather than the composed-color path.
    dialog = _fit_dialog("Exponential + Constant")

    class _StubModel:
        def formula_string(self) -> str:
            return "stub formula"

        def latex_string(self) -> str:
            return r"\mathrm{stub}"

    dialog._set_equation_content(_StubModel())
    pixmap = dialog._equation_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert dialog._equation_label.text() == ""


def test_fallback_to_plain_text_when_no_latex_api_at_all(qapp: QApplication) -> None:
    # A model lacking both latex_terms() and latex_string() (e.g. the
    # trending ParameterCompositeModel before core support lands) falls all
    # the way back to the plain formula_string() text label.
    dialog = _fit_dialog("Exponential + Constant")

    class _PlainStubModel:
        def formula_string(self) -> str:
            return "plain stub formula"

    dialog._set_equation_content(_PlainStubModel())
    assert dialog._equation_label.pixmap().isNull()
    assert "plain stub formula" in dialog._equation_label.text()


def test_composed_render_colors_fraction_group_terms(qapp: QApplication) -> None:
    # A stubbed latex_terms() with a grouped term should reach the composed
    # colored-equation path (render_colored_equation_pixmap), producing a
    # non-null pixmap distinct from the ungrouped case. This exercises the
    # dialog's fragment-building/coloring logic independent of the real
    # CompositeModel.latex_terms() implementation.
    from asymmetry.core.fitting.composite import LatexTerm

    dialog = _fit_dialog("Exponential + Constant")

    class _StubGroupedModel:
        fraction_groups = [(0, 1)]

        def formula_string(self) -> str:
            return "stub"

        def latex_terms(self):
            return [
                LatexTerm(latex=r"A_1 f e^{-\lambda t}", separator="", group=(0, 1)),
                LatexTerm(latex=r"A_{bg}", separator=" + ", group=None),
            ]

    dialog._set_equation_content(_StubGroupedModel())
    pixmap = dialog._equation_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()


def test_prefix_to_mathtext_handles_greek_and_plain_prefixes() -> None:
    assert _prefix_to_mathtext("A(t)").startswith(r"\mathrm{A}(")
    assert r"\nu" in _prefix_to_mathtext("S(ν)")
    assert _prefix_to_mathtext("A(t)").endswith(" = ")
