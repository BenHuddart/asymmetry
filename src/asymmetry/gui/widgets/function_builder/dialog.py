"""Two-panel structured function-builder dialog shell.

``FunctionBuilderDialog`` combines the searchable
:class:`~asymmetry.gui.widgets.function_builder.library_panel.ComponentLibraryPanel`
(left) with the structured
:class:`~asymmetry.gui.widgets.function_builder.model_rows.ModelRowList` (right),
plus a Group/Fraction action row, a live preview + status, and a text-editing
fallback ("Edit as text") that speaks the honest canonical expression syntax
(including ``{frac}``).

The dialog is grammar-agnostic: callers inject an ``expression_parser`` (text →
five structure lists) and a ``model_parser`` (canonical expression → model). Two
ready-made wrapper factories are provided so downstream builders (milestone 3)
just import them:

- :func:`make_fit_expression_parser` — the fit grammar (``parse_composite_expression``).
- :func:`make_component_expression_parser` — the restricted parameter/trending
  grammar (``parse_component_expression`` + empty fraction list), with an
  injectable operator set (e.g. adding ``⊕``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.composite import (
    parse_component_expression,
    parse_composite_expression,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.utils.latex_renderer import (
    render_colored_equation_pixmap,
    render_latex_to_pixmap,
)
from asymmetry.gui.widgets.function_builder.library_panel import ComponentLibraryPanel
from asymmetry.gui.widgets.function_builder.model_rows import (
    FRACTION_GROUP_COLORS,
    ModelRowList,
)
from asymmetry.gui.widgets.screen_sizing import resize_to_available

#: Local button style for the preview card's copy button: opt out of the
#: global stylesheet's filled-button chrome so a small icon-only control fits
#: the card without dominating it (same pattern as the library row buttons in
#: ``library_panel.py``'s ``_ROW_BUTTON_QSS``, duplicated locally rather than
#: imported so the two widgets stay decoupled).
_COPY_BUTTON_QSS = (
    "QToolButton { border: none; background: transparent; padding: 2px 6px; margin: 0px; }"
    "QToolButton:hover { background: rgba(0, 0, 0, 28); border-radius: 4px; }"
    "QToolButton:pressed { background: rgba(0, 0, 0, 48); border-radius: 4px; }"
)

#: Dumb substring map for turning a plain-text expression prefix (e.g.
#: ``"A(t)"``, ``"S(ν)"``) into a mathtext-safe fragment. Anything not
#: covered here still renders via the ``\mathrm{}`` fallback wrapper applied
#: by :func:`_prefix_to_mathtext`, so an unrecognised prefix degrades to a
#: literal (if slightly less pretty) label rather than breaking mathtext.
_PREFIX_SYMBOL_MAP = {
    "ν": r"\nu",
    "λ": r"\lambda",
    "α": r"\alpha",
    "β": r"\beta",
    "ω": r"\omega",
}


def _prefix_to_mathtext(expression_prefix: str) -> str:
    """Turn a plain-text prefix like ``"A(t)"`` into a mathtext fragment.

    Wraps the letters/words in ``\\mathrm{}`` (so multi-letter names like "A"
    don't italicize as a product of single-letter variables) while leaving
    parentheses and known Greek-letter substitutions bare. Falls back to an
    escaped literal wrapped wholesale in ``\\mathrm{}`` if the dumb tokenizer
    below produces something mathtext can't parse (caller's render call still
    returns ``None`` gracefully in that case).
    """
    prefix = expression_prefix.strip()
    for literal, replacement in _PREFIX_SYMBOL_MAP.items():
        prefix = prefix.replace(literal, f"@@{replacement}@@")

    tokens_out: list[str] = []
    for chunk in re.split(r"(@@.*?@@|[()=])", prefix):
        if not chunk:
            continue
        if chunk.startswith("@@") and chunk.endswith("@@"):
            tokens_out.append(chunk[2:-2])
        elif chunk in "()=":
            tokens_out.append(chunk)
        else:
            tokens_out.append(rf"\mathrm{{{chunk}}}")
    return "".join(tokens_out) + " = "


#: Type of an expression parser: text → the five structure lists.
StructureParser = Callable[
    [str], tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]
]


def make_fit_expression_parser() -> StructureParser:
    """Return a structure parser for the full fit grammar (with ``{frac}``).

    Wraps :func:`asymmetry.core.fitting.composite.parse_composite_expression`,
    which already returns the five lists including fraction groups.
    """

    def _parse(
        expression: str,
    ) -> tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]:
        return parse_composite_expression(expression)

    return _parse


def make_component_expression_parser(
    *,
    allowed_components: set[str] | frozenset[str],
    allowed_operators: set[str] | frozenset[str] | None = None,
) -> StructureParser:
    """Return a structure parser for the restricted parameter/trending grammar.

    Wraps :func:`asymmetry.core.fitting.composite.parse_component_expression`
    (no ``{frac}`` decorator) and pads an empty fraction-group list so the
    five-tuple contract matches. Pass ``allowed_operators`` including ``⊕`` for
    the trending grammar.
    """

    def _parse(
        expression: str,
    ) -> tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]:
        if allowed_operators is None:
            names, operators, opens, closes = parse_component_expression(
                expression, allowed_components=allowed_components
            )
        else:
            names, operators, opens, closes = parse_component_expression(
                expression,
                allowed_components=allowed_components,
                allowed_operators=allowed_operators,
            )
        return names, operators, opens, closes, []

    return _parse


class FunctionBuilderDialog(QDialog):
    """Structured two-panel builder for composite functions."""

    def __init__(
        self,
        *,
        title: str,
        expression_prefix: str,
        component_definitions: Mapping[str, object],
        model_parser: Callable[[str], object],
        expression_parser: StructureParser,
        initial_expression: str,
        operators: Sequence[str] = ("+", "-", "*", "/"),
        enable_fraction_groups: bool = False,
        syntax_help_text: str | None = None,
        on_create_user_function: Callable[[], object | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._expression_prefix = expression_prefix
        self._component_definitions = dict(component_definitions)
        self._model_parser = model_parser
        self._expression_parser = expression_parser
        self._enable_fraction_groups = enable_fraction_groups
        self._on_create_user_function = on_create_user_function
        self._model: object | None = None

        root = QVBoxLayout(self)

        # -- top: library (left) + structured/text stack (right) ------------
        # A splitter (rather than a fixed-width library pane) lets a user with a
        # long search result list or a wide model give either side more room.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._library = ComponentLibraryPanel(self._component_definitions)
        self._library.setMinimumWidth(180)
        self._library.component_activated.connect(self._on_component_activated)
        if on_create_user_function is not None:
            self._library.set_creation_enabled(True)
            self._library.create_requested.connect(self._on_create_user_function_requested)
        splitter.addWidget(self._library)

        right_pane = QWidget()
        right = QVBoxLayout(right_pane)
        right.setContentsMargins(0, 0, 0, 0)

        self._action_row = QHBoxLayout()
        self._group_fraction_button = QPushButton("Group as fractions")
        self._group_fraction_button.clicked.connect(self._group_selection_as_fractions)
        self._group_button = QPushButton("Group")
        self._group_button.clicked.connect(self._group_selection)
        self._ungroup_button = QPushButton("Ungroup")
        self._ungroup_button.clicked.connect(self._ungroup_selection)
        if enable_fraction_groups:
            self._action_row.addWidget(self._group_fraction_button)
        else:
            self._group_fraction_button.setVisible(False)
        self._action_row.addWidget(self._group_button)
        self._action_row.addWidget(self._ungroup_button)
        self._action_row.addStretch(1)
        right.addLayout(self._action_row)

        self._stack = QStackedWidget()
        self._rows = ModelRowList(
            self._component_definitions,
            operators=operators,
            enable_fraction_groups=enable_fraction_groups,
        )
        self._rows.structure_changed.connect(self._on_structure_changed)
        self._rows.selection_changed.connect(self._update_action_buttons)
        # A large model must scroll rather than grow the dialog past the screen.
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows_scroll.setWidget(self._rows)
        self._rows_page = rows_scroll
        self._stack.addWidget(self._rows_page)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText(
            "Edit the expression directly, e.g. Exponential + Constant"
        )
        self._stack.addWidget(self._text_edit)
        right.addWidget(self._stack, 1)

        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 720])
        root.addWidget(splitter, 1)

        # -- syntax help / preview / status --------------------------------
        if syntax_help_text:
            help_label = QLabel(syntax_help_text)
            help_label.setWordWrap(True)
            help_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            root.addWidget(help_label)

        root.addWidget(self._build_preview_card())

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._status_label)

        # -- buttons -------------------------------------------------------
        button_row = QHBoxLayout()
        self._text_mode_button = QPushButton("Edit as text")
        self._text_mode_button.setCheckable(False)
        self._text_mode_button.clicked.connect(self._toggle_text_mode)
        button_row.addWidget(self._text_mode_button)

        self._apply_text_button = QPushButton("Apply")
        self._apply_text_button.clicked.connect(self._apply_text)
        self._apply_text_button.setVisible(False)
        button_row.addWidget(self._apply_text_button)

        button_row.addStretch(1)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        button_row.addWidget(self._buttons)
        root.addLayout(button_row)

        # Seed the structured editor from the initial expression. A parse
        # failure switches to text mode (see _seed_structure), in which case
        # re-validate the text buffer rather than the now-empty structured
        # rows, so the parse-error status _seed_structure just set is not
        # immediately overwritten by the generic "add a function" message.
        self._seed_structure(initial_expression)
        if self._stack.currentWidget() is self._text_edit:
            self._validate_and_update(self._text_edit.toPlainText())
        else:
            self._on_structure_changed()
        self._update_action_buttons()

        resize_to_available(self, 940, 640, min_width=760, min_height=520)

    # -------------------------------------------------------------- preview card
    def _build_preview_card(self) -> QFrame:
        """Build the prominent equation-preview card.

        Layout: a muted caption (the expression prefix), a horizontal-only
        scroll area holding the composed equation image (or plain-text
        fallback), a small muted sub-line (the legacy "Preview: ..." /
        fraction-weights text, kept as ``_preview_label`` for both display and
        backward-compatible test access), and a copy-to-clipboard button.
        """
        card = QFrame()
        card.setObjectName("equationPreviewCard")
        card.setStyleSheet(
            f"#equationPreviewCard {{ background: {tokens.SURFACE_ALT}; "
            f"border: 1px solid {tokens.BORDER}; border-radius: 6px; }}"
        )
        self._preview_card = card

        outer = QVBoxLayout(card)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)

        header = QHBoxLayout()
        caption = QLabel(self._expression_prefix or "Fit function")
        caption.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-weight: 600;")
        header.addWidget(caption)
        header.addStretch(1)

        self._copy_button = QToolButton()
        self._copy_button.setText("Copy")
        self._copy_button.setToolTip("Copy the formula as plain text")
        self._copy_button.setFixedHeight(20)
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.setStyleSheet(_COPY_BUTTON_QSS)
        self._copy_button.clicked.connect(self._copy_formula_to_clipboard)
        header.addWidget(self._copy_button)
        outer.addLayout(header)

        self._equation_label = QLabel("")
        self._equation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        equation_row = QHBoxLayout()
        equation_row.setContentsMargins(0, 0, 0, 0)
        equation_row.addWidget(self._equation_label)
        equation_row.addStretch(1)
        equation_container = QWidget()
        equation_container.setLayout(equation_row)

        self._equation_scroll = QScrollArea()
        self._equation_scroll.setWidgetResizable(True)
        self._equation_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._equation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._equation_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._equation_scroll.setWidget(equation_container)
        # Fixed height: the equation never wraps, only scrolls horizontally
        # when it is wider than the card.
        self._equation_scroll.setFixedHeight(48)
        outer.addWidget(self._equation_scroll)

        # The legacy plain-text preview line (formula + fraction-group
        # weights) becomes the card's small muted sub-line.
        self._preview_label = QLabel("")
        self._preview_label.setWordWrap(True)
        self._preview_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 11px;")
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self._preview_label)

        return card

    def _copy_formula_to_clipboard(self) -> None:
        formula = getattr(self._model, "formula_string", lambda: "")() if self._model else ""
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(formula)

    # ------------------------------------------------------------------ API
    def built_model(self) -> object | None:
        """Return the model captured when the dialog was accepted."""
        return self._model

    # ------------------------------------------------------------- seeding
    def _seed_structure(self, expression: str) -> None:
        expression = (expression or "").strip()
        if not expression:
            self._rows.set_structure([], [], [], [], [])
            return
        try:
            names, operators, opens, closes, fractions = self._expression_parser(expression)
        except Exception as exc:
            # Don't silently discard the caller's expression: switch to text
            # mode seeded with the original string so the user sees exactly
            # what they passed in and why it didn't parse, rather than an
            # unexplained empty dialog. OK stays gated by _validate_and_update
            # (called below), which will find the same parse failure.
            self._rows.set_structure([], [], [], [], [])
            self._text_edit.setPlainText(expression)
            self._stack.setCurrentWidget(self._text_edit)
            self._text_mode_button.setText("Back to structured")
            self._apply_text_button.setVisible(True)
            self._group_button.setEnabled(False)
            self._group_fraction_button.setEnabled(False)
            self._ungroup_button.setEnabled(False)
            self._set_status(str(exc), valid=False)
            return
        self._rows.set_structure(names, operators, opens, closes, fractions)

    # ------------------------------------------------------------ structured
    def _on_component_activated(self, name: str) -> None:
        if self._stack.currentWidget() is self._text_edit:
            self._text_edit.insertPlainText(name)
            return
        self._rows.append_component(name)

    def _on_create_user_function_requested(self) -> None:
        """Run the caller-supplied authoring hook and adopt the result.

        The hook (implemented by a subclass, e.g.
        :class:`~asymmetry.gui.panels.fit_function_builder.FitFunctionBuilderDialog`)
        shows the authoring dialog and returns the created definition (with a
        ``.name`` attribute) on success, or ``None`` on cancel/failure. On
        success the new component is folded into the library and row list and
        immediately inserted into the expression being edited, so the newly
        authored function is usable without reopening this dialog.
        """
        if self._on_create_user_function is None:
            return
        definition = self._on_create_user_function()
        if definition is None:
            return
        name = getattr(definition, "name", None)
        if not name:
            return

        self._component_definitions[name] = definition
        self._library.set_components(self._component_definitions)
        self._rows.set_component_definitions(self._component_definitions)
        self._on_component_activated(name)

    def _on_structure_changed(self) -> None:
        self._update_action_buttons()
        self._validate_and_update(self._rows.expression())

    def _validate_and_update(self, expression: str) -> None:
        expression = (expression or "").strip()
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if not expression:
            self._model = None
            if ok_button is not None:
                ok_button.setEnabled(False)
            self._preview_label.setText("")
            self._set_equation_content(None)
            self._preview_card.setEnabled(False)
            self._set_status("Add a function to build a model.", valid=False)
            return
        try:
            model = self._model_parser(expression)
        except Exception as exc:
            self._model = None
            if ok_button is not None:
                ok_button.setEnabled(False)
            self._preview_label.setText("")
            self._set_equation_content(None)
            self._preview_card.setEnabled(False)
            self._set_status(str(exc), valid=False)
            return

        self._model = model
        if ok_button is not None:
            ok_button.setEnabled(True)
        self._set_status("Expression is valid.", valid=True)
        self._preview_card.setEnabled(True)
        self._update_preview(model)

    def _update_preview(self, model: object) -> None:
        formula = getattr(model, "formula_string", lambda: "")()
        lines = [f"Preview: {self._expression_prefix} = {formula}"]

        fraction_groups = getattr(model, "fraction_groups", None)
        param_groups_fn = getattr(model, "fraction_parameter_groups", None)
        derived_fn = getattr(model, "derived_fraction_terms", None)
        amplitude_fn = getattr(model, "_fraction_group_amplitude_name", None)
        if fraction_groups and param_groups_fn and derived_fn and amplitude_fn:
            try:
                param_groups = param_groups_fn()
                derived = derived_fn()
                for group, free_names, (remainder_name, _grp) in zip(
                    fraction_groups, param_groups, derived, strict=True
                ):
                    amplitude = amplitude_fn(group)
                    weights = ", ".join([*free_names, f"{remainder_name} (derived)"])
                    lines.append(f"Fraction group {amplitude}: {weights}")
            except Exception:
                pass
        self._preview_label.setText("\n".join(lines))

        self._set_equation_content(model)

    # -------------------------------------------------------- equation render
    def _set_equation_content(self, model: object | None) -> None:
        """Render the equation area for *model* via the fallback chain.

        Chain: composed colored render (needs ``model.latex_terms()``) →
        single-string mathtext render (``model.latex_string()``) → plain
        ``formula_string()`` text. Each step is attempted only when the
        previous one is unavailable (missing API) or returns ``None``
        (render failure). ``model is None`` clears the equation area.
        """
        if model is None:
            self._equation_label.setPixmap(QPixmap())
            self._equation_label.setText("")
            return

        pixmap = self._render_composed_equation(model)
        if pixmap is None:
            latex_string_fn = getattr(model, "latex_string", None)
            if callable(latex_string_fn):
                try:
                    latex_string = latex_string_fn()
                except Exception:
                    latex_string = None
                if latex_string:
                    pixmap = render_latex_to_pixmap(latex_string)

        if pixmap is not None:
            self._equation_label.setPixmap(pixmap)
            self._equation_label.setText("")
            return

        # Final fallback: plain text (current pre-card behavior).
        formula = getattr(model, "formula_string", lambda: "")()
        self._equation_label.setPixmap(QPixmap())
        self._equation_label.setText(f"{self._expression_prefix} = {formula}")

    def _render_composed_equation(self, model: object) -> QPixmap | None:
        latex_terms_fn = getattr(model, "latex_terms", None)
        if not callable(latex_terms_fn):
            return None
        try:
            terms = latex_terms_fn()
        except Exception:
            return None
        if not terms:
            return None

        fraction_groups = list(getattr(model, "fraction_groups", None) or [])
        group_order = {tuple(g): i for i, g in enumerate(sorted(fraction_groups))}

        fragments: list[tuple[str, str]] = [
            (_prefix_to_mathtext(self._expression_prefix), tokens.TEXT)
        ]
        for term in terms:
            latex = getattr(term, "latex", None)
            separator = getattr(term, "separator", None)
            group = getattr(term, "group", None)
            if separator:
                fragments.append((str(separator), tokens.TEXT))
            if latex is None:
                continue
            if group is not None:
                index = group_order.get(tuple(group), 0)
                color = FRACTION_GROUP_COLORS[index % len(FRACTION_GROUP_COLORS)]
            else:
                color = tokens.TEXT
            fragments.append((str(latex), color))

        if len(fragments) <= 1:
            return None
        return render_colored_equation_pixmap(tuple(fragments))

    def _set_status(self, message: str, *, valid: bool) -> None:
        color = tokens.OK if valid else tokens.ERROR
        self._status_label.setText(f"<span style='color:{color};'>{message}</span>")

    # ------------------------------------------------------------- grouping
    def _single_selected_span(self) -> tuple[int, int] | None:
        spans = self._rows.selected_spans()
        if len(spans) != 1:
            return None
        return spans[0]

    def _group_selection(self) -> None:
        span = self._single_selected_span()
        if span is None or not self._rows.group_span(span):
            self._set_status(
                "Select two or more adjacent components joined by '+' to group.",
                valid=False,
            )

    def _group_selection_as_fractions(self) -> None:
        span = self._single_selected_span()
        if span is None or not self._rows.set_fraction(span, True):
            self._set_status(
                "Select two or more additive components joined by '+' for fractions.",
                valid=False,
            )

    def _ungroup_selection(self) -> None:
        span = self._single_selected_span()
        if span is None or not self._rows.ungroup_span(span):
            self._set_status("Select a grouped span to ungroup.", valid=False)

    def _update_action_buttons(self) -> None:
        span = self._single_selected_span()
        spans_ok = span is not None
        can_group = spans_ok and self._rows.can_group(span)
        is_paren = spans_ok and span in self._rows._parenthesized_spans()
        self._group_button.setEnabled(bool(can_group))
        if self._enable_fraction_groups:
            self._group_fraction_button.setEnabled(bool(can_group or is_paren))
        self._ungroup_button.setEnabled(bool(is_paren))

    # --------------------------------------------------------------- text mode
    def _toggle_text_mode(self) -> None:
        if self._stack.currentWidget() is self._rows_page:
            self._text_edit.setPlainText(self._rows.expression())
            self._stack.setCurrentWidget(self._text_edit)
            self._text_mode_button.setText("Back to structured")
            self._apply_text_button.setVisible(True)
            self._group_button.setEnabled(False)
            self._group_fraction_button.setEnabled(False)
            self._ungroup_button.setEnabled(False)
            self._validate_and_update(self._text_edit.toPlainText())
        else:
            if self._apply_text():
                return  # stayed valid; _apply_text switched us back

    def _apply_text(self) -> bool:
        """Parse the text buffer back into structured rows.

        Returns ``True`` when the text parsed and we returned to structured
        mode; ``False`` (staying in text mode with an error) otherwise.
        """
        expression = self._text_edit.toPlainText().strip()
        try:
            names, operators, opens, closes, fractions = self._expression_parser(expression)
        except Exception as exc:
            self._set_status(str(exc), valid=False)
            return False
        self._rows.set_structure(names, operators, opens, closes, fractions)
        self._stack.setCurrentWidget(self._rows_page)
        self._text_mode_button.setText("Edit as text")
        self._apply_text_button.setVisible(False)
        self._on_structure_changed()
        return True

    # ------------------------------------------------------------- accept
    def _current_expression(self) -> str:
        if self._stack.currentWidget() is self._text_edit:
            return self._text_edit.toPlainText().strip()
        return self._rows.expression()

    def _on_accept(self) -> None:
        expression = self._current_expression()
        if not expression:
            self._set_status("Add a function to build a model.", valid=False)
            return
        try:
            model = self._model_parser(expression)
        except Exception as exc:
            self._set_status(str(exc), valid=False)
            return
        self._model = model
        self.accept()
