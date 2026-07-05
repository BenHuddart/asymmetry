"""Tests for the fit-function builder dialog (structured-foundation rebuild)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from asymmetry.core.fitting.component_search import search_components
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _ok(dialog: FitFunctionBuilderDialog):
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)


# ------------------------------------------------------------------ defaults
def test_dialog_builds_default_model(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    assert dialog._domain == "time"
    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.component_names == ["Exponential", "Constant"]
    assert model.operators == ["+"]
    assert "Preview: A(t) =" in dialog._preview_label.text()


def test_dialog_prepopulate_model_roundtrips(qapp: QApplication) -> None:
    initial = CompositeModel(
        ["Gaussian", "Constant", "Constant"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )
    dialog = FitFunctionBuilderDialog(initial_model=initial)

    assert dialog._rows.expression() == "Gaussian * (Constant + Constant)"
    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.component_names == initial.component_names
    assert model.operators == initial.operators
    assert model.open_parentheses == initial.open_parentheses
    assert model.close_parentheses == initial.close_parentheses


def test_nested_paren_model_roundtrips_unchanged(qapp: QApplication) -> None:
    initial = CompositeModel(
        ["Exponential", "Gaussian", "Constant", "Constant"],
        operators=["*", "+", "+"],
        open_parentheses=[0, 1, 1, 0],
        close_parentheses=[0, 0, 0, 2],
    )
    dialog = FitFunctionBuilderDialog(initial_model=initial)
    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.component_names == initial.component_names
    assert model.operators == initial.operators
    assert model.open_parentheses == initial.open_parentheses
    assert model.close_parentheses == initial.close_parentheses


# ------------------------------------------------------------------- gating
def test_empty_structure_disables_ok(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._rows.clear()

    ok = _ok(dialog)
    assert ok is not None
    assert ok.isEnabled() is False


def test_invalid_text_reports_error_and_refuses_accept(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("Exponential +")

    # Applying an invalid expression stays in text mode with a surfaced error.
    assert dialog._apply_text() is False
    assert "operator" in dialog._status_label.text().lower()

    # Accepting an invalid expression does not fire acceptance.
    dialog._on_accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


# ----------------------------------------------------------- domain filtering
def test_time_domain_builder_excludes_frequency_components(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()

    assert "GaussianPeak" not in dialog._allowed_components
    assert "LorentzianPeak" not in dialog._allowed_components
    assert "Exponential" in dialog._allowed_components


def test_frequency_domain_builder_is_filtered(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog(domain="frequency")

    assert dialog._domain == "frequency"
    assert dialog._allowed_components == {
        "GaussianPeak",
        "LorentzianPeak",
        "ConstantBackground",
        "LinearBackground",
    }
    assert dialog._rows.expression() == "GaussianPeak + ConstantBackground"
    assert "Preview: S(ν) =" in dialog._preview_label.text()


def test_out_of_domain_component_gets_domain_hint_in_text_mode(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("GaussianPeak + Constant")

    applied = dialog._apply_text()
    assert applied is False  # stayed in text mode with an error

    status = dialog._status_label.text()
    assert "frequency-domain component" in status
    assert "time domain" in status

    # The invalid expression is not accepted.
    dialog._on_accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_frequency_builder_rejects_time_component(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog(domain="frequency")
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("Exponential + ConstantBackground")

    assert dialog._apply_text() is False
    status = dialog._status_label.text()
    assert "time-domain component" in status
    assert "frequency domain" in status


# -------------------------------------------------------------- fraction groups
def test_dialog_builds_fraction_group_model(qapp: QApplication) -> None:
    """Grouping two additive terms yields the new ``f_<Component>`` naming.

    The previously-failing legacy test asserted ``fraction_1``/``fraction_2``;
    the current model names free fractions ``f_<Component>`` (one fewer than the
    number of terms) with a derived remainder label from
    :meth:`CompositeModel.derived_fraction_names`.
    """
    dialog = FitFunctionBuilderDialog()
    dialog._rows.set_structure(["Exponential", "Gaussian"], ["+"], [0, 0], [0, 0], [])
    dialog._rows._selected_indices = {0, 1}
    dialog._update_action_buttons()
    dialog._group_selection_as_fractions()

    dialog._on_accept()
    model = dialog.get_composite_model()

    assert model is not None
    assert model.fraction_groups == [(0, 1)]
    # One free fraction parameter for two terms (n-1).
    free = model.fraction_parameter_groups()
    assert free == [["f_Exponential"]]
    assert "f_Exponential" in model.param_names
    # The remainder term is a derived, display-only label (no free parameter).
    assert model.derived_fraction_names() == ["f_Gaussian"]
    assert "f_Gaussian" not in model.param_names
    # Preview enumerates the fraction group.
    assert "Fraction group" in dialog._preview_label.text()


def test_initial_fraction_group_roundtrips(qapp: QApplication) -> None:
    initial = CompositeModel(
        ["Exponential", "Gaussian", "Constant"],
        operators=["+", "+"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
        fraction_groups=[(0, 1)],
    )
    dialog = FitFunctionBuilderDialog(initial_model=initial)
    assert "{frac}" in dialog._rows.expression()

    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.fraction_groups == [(0, 1)]


# ------------------------------------------- fraction group survives structural edits
# The old FunctionExpressionBuilderDialog's failure mode: any structural edit
# (append/duplicate/move a row) silently wiped previously-defined fraction
# groups because the row list was rebuilt from a flat text expression rather
# than shifting the group indices in place. These pins exercise the new
# ModelRowList's index-shifting logic directly through the dialog.


def _dialog_with_grouped_fraction(qapp: QApplication) -> FitFunctionBuilderDialog:
    """4-row dialog with terms 1-2 (Gaussian, Constant) grouped as a fraction."""
    dialog = FitFunctionBuilderDialog()
    dialog._rows.set_structure(
        ["Exponential", "Gaussian", "Constant", "Constant"],
        ["+", "+", "+"],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [],
    )
    span = (1, 2)
    dialog._rows._selected_indices = {1, 2}
    dialog._update_action_buttons()
    assert dialog._rows.set_fraction(span, True)
    assert dialog._rows.structure()[4] == [(1, 2)]
    return dialog


def test_fraction_group_survives_append_after_group(qapp: QApplication) -> None:
    dialog = _dialog_with_grouped_fraction(qapp)

    # Append a new component at the end (no row selected -> appends at top level).
    dialog._rows._selected_indices = set()
    dialog._rows.append_component("Constant")

    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.fraction_groups == [(1, 2)]


def test_fraction_group_survives_duplicate_row_before_group(qapp: QApplication) -> None:
    dialog = _dialog_with_grouped_fraction(qapp)

    # Duplicate the first row (index 0, before the group); the group shifts by one.
    dialog._rows.duplicate_row(0)

    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.fraction_groups == [(2, 3)]
    # The grouped components themselves are unchanged.
    names, *_rest = dialog._rows.structure()
    assert names[2] == "Gaussian"
    assert names[3] == "Constant"


def test_fraction_group_survives_moving_adjacent_row(qapp: QApplication) -> None:
    dialog = _dialog_with_grouped_fraction(qapp)

    # Move the trailing row (index 3, a sibling of the group) up past the group.
    dialog._rows.move_row(3, -1)

    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert len(model.fraction_groups) == 1
    group_start, group_end = model.fraction_groups[0]
    names, *_rest = dialog._rows.structure()
    assert names[group_start : group_end + 1] == ["Gaussian", "Constant"]


# ------------------------------------------------- text-mode honesty for groups
def test_text_mode_shows_parenthesized_frac_form_for_grouped_model(
    qapp: QApplication,
) -> None:
    """A fraction-grouped model's text view uses the parenthesized ``{frac}`` form.

    The old calculator-style builder could round-trip a plain, paren-less
    multiplicative expression that silently dropped group membership. The new
    text mode must always show the honest ``(...){frac}`` form for a grouped
    model, never a paren-less expression.
    """
    dialog = _dialog_with_grouped_fraction(qapp)

    dialog._toggle_text_mode()
    text = dialog._text_edit.toPlainText()

    assert "{frac}" in text
    assert "(Gaussian + Constant){frac}" in text

    # No paren-less multiplicative rendering of the grouped terms is shown.
    assert "Gaussian * Constant" not in text


# --------------------------------------------------------------- missing user fn
def test_missing_user_component_opens_with_placeholder(qapp: QApplication) -> None:
    model = CompositeModel(
        ["Exponential", "MyMissingFn", "Constant"],
        operators=["+", "+"],
        allow_missing=True,
    )
    assert model.missing_component_names == ("MyMissingFn",)

    dialog = FitFunctionBuilderDialog(initial_model=model)

    # The row list renders every component, including the missing one.
    names, *_rest = dialog._rows.structure()
    assert names == ["Exponential", "MyMissingFn", "Constant"]
    # A placeholder definition backs the missing name (warning-tinted row).
    assert "MyMissingFn" in dialog._placeholder_definitions
    # The missing name is never leaked into the global registry.
    from asymmetry.core.fitting.composite import COMPONENTS

    assert "MyMissingFn" not in COMPONENTS

    dialog._on_accept()
    built = dialog.get_composite_model()
    assert built is not None
    assert built.component_names == ["Exponential", "MyMissingFn", "Constant"]
    assert built.missing_component_names == ("MyMissingFn",)


# --------------------------------------------------------- simulate invocation
def test_simulate_style_invocation(qapp: QApplication) -> None:
    """A parent + initial_model construction (as SimulateDialog does) works."""
    parent = None  # SimulateDialog passes a real parent; None exercises the path
    initial = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dialog = FitFunctionBuilderDialog(parent, initial_model=initial, domain="time")

    dialog._on_accept()
    model = dialog.get_composite_model()
    assert model is not None
    assert model.component_names == ["Exponential", "Constant"]


# --------------------------------------------------------- placeholder exception safety
def test_parse_error_with_placeholders_injected_leaves_components_clean(
    qapp: QApplication,
) -> None:
    """A parse error mid-``_parse_model`` never leaves a placeholder in COMPONENTS.

    ``_parse_model`` temporarily injects the dialog's per-instance placeholder
    definitions into the global ``COMPONENTS`` registry via
    ``_placeholders_registered`` so an unregistered user function still parses,
    then removes them afterwards. That inject/restore must be exception-safe:
    a malformed expression (or any other parse failure) raised while the
    placeholder is registered must not leave it stuck in ``COMPONENTS``.
    """
    from asymmetry.core.fitting.composite import COMPONENTS

    model = CompositeModel(
        ["Exponential", "MyMissingFn", "Constant"],
        operators=["+", "+"],
        allow_missing=True,
    )
    dialog = FitFunctionBuilderDialog(initial_model=model)
    assert "MyMissingFn" in dialog._placeholder_definitions
    assert "MyMissingFn" not in COMPONENTS

    # A trailing operator is a ValueError raised deep inside
    # parse_composite_expression, while the placeholder is injected.
    with pytest.raises(ValueError):
        dialog._parse_model("Exponential + MyMissingFn +")

    assert "MyMissingFn" not in COMPONENTS

    # An unknown-name error (UnknownComponentError, re-raised as a domain-hint
    # ValueError) is the other exception path out of the same context manager.
    with pytest.raises(Exception):
        dialog._parse_model("Exponential + TotallyUnknownComponent")

    assert "MyMissingFn" not in COMPONENTS

    # The dialog still parses normally afterwards -- the registry was not left
    # in a broken state that would poison a subsequent successful parse.
    built = dialog._parse_model("Exponential + MyMissingFn + Constant")
    assert built.component_names == ["Exponential", "MyMissingFn", "Constant"]
    assert "MyMissingFn" not in COMPONENTS


# ----------------------------------------------------------------- library add
def test_library_activation_appends_component(qapp: QApplication) -> None:
    dialog = FitFunctionBuilderDialog()
    dialog._library.component_activated.emit("Gaussian")
    names, *_rest = dialog._rows.structure()
    assert names == ["Exponential", "Constant", "Gaussian"]


# ------------------------------------------------------ library category order
def test_library_empty_query_groups_by_domain_and_category() -> None:
    # The library panel renders ``search_components("")`` grouped by category,
    # so exercising that surface directly is what a user sees in the tree.
    time_names = [
        result.name for result in search_components("", components=COMPONENTS, domain="time")
    ]
    time_categories = {COMPONENTS[name].category for name in time_names}
    assert "Muonium" in time_categories
    # A frequency-domain component never leaks into the time catalogue.
    assert "GaussianPeak" not in time_names

    freq_names = [
        result.name for result in search_components("", components=COMPONENTS, domain="frequency")
    ]
    assert all(COMPONENTS[name].category == "Frequency Domain" for name in freq_names)
    assert "GaussianPeak" in freq_names
