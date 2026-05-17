"""Dialog for building composite fit functions."""

from __future__ import annotations

import html
import re
from collections import defaultdict

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from asymmetry.core.fitting.composite import (
    COMPONENTS,
    CompositeModel,
    build_component_expression,
    parse_component_expression,
    parse_composite_expression,
)
from asymmetry.gui.widgets.function_expression_builder import (
    ComponentSelectorButton as _ComponentSelectorButton,  # noqa: F401
)
from asymmetry.gui.widgets.function_expression_builder import (
    FunctionExpressionBuilderDialog,
)


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FRACTION_GROUP_COLORS = ["#005A9C", "#A44A00", "#0B6E4F", "#8A1C1C", "#6B4F00"]


def _build_components_by_category() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, definition in COMPONENTS.items():
        category = (definition.category or "General").strip() or "General"
        grouped[category].append(name)

    for names in grouped.values():
        names.sort()
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


class FitFunctionBuilderDialog(FunctionExpressionBuilderDialog):
    """Compose a custom fit function from predefined components."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_model: CompositeModel | None = None,
    ) -> None:
        self._allowed_components = set(COMPONENTS)
        self._fraction_groups = (
            sorted(initial_model.fraction_groups) if initial_model is not None else []
        )
        self._last_valid_structure: tuple[
            tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]
        ] | None = None
        self._preserve_fraction_groups_once = False
        self._normalizing_fraction_syntax = False
        initial_expression = (
            self._display_expression_for_model(initial_model)
            if initial_model is not None
            else "Exponential + Constant"
        )
        super().__init__(
            title="Build Fit Function",
            expression_prefix="A(t)",
            components_by_category=_build_components_by_category(),
            component_definitions=COMPONENTS,
            model_parser=CompositeModel.from_expression,
            initial_expression=initial_expression,
            expression_placeholder=(
                "e.g. Exponential + Gaussian or Exponential * ( Gaussian + Constant )"
            ),
            syntax_help_text=(
                "Select two or more additive components, press Fractions to link them, "
                "and use the matching colors in the editor and preview to track each group."
            ),
            parent=parent,
        )

        self._group_fraction_button = QPushButton("Fractions")
        self._group_fraction_button.clicked.connect(self._group_selection_as_fractions)
        self._separate_fraction_button = QPushButton("Separate")
        self._separate_fraction_button.clicked.connect(self._separate_fraction_selection)

        selection_row = QHBoxLayout()
        selection_row.addWidget(QLabel("Selection:"))
        selection_row.addWidget(self._group_fraction_button)
        selection_row.addWidget(self._separate_fraction_button)
        selection_row.addStretch()

        root_layout = self.layout()
        if isinstance(root_layout, QVBoxLayout):
            root_layout.insertLayout(2, selection_row)

        self._on_fields_changed()

    @staticmethod
    def _display_expression_for_model(model: CompositeModel | None) -> str:
        if model is None:
            return ""

        open_parentheses = list(model.open_parentheses)
        close_parentheses = list(model.close_parentheses)
        for start, end in model.fraction_groups:
            if open_parentheses[start] > 0:
                open_parentheses[start] -= 1
            if close_parentheses[end] > 0:
                close_parentheses[end] -= 1

        return build_component_expression(
            model.component_names,
            model.operators,
            open_parentheses,
            close_parentheses,
        )

    def _normalize_fraction_syntax_if_needed(self) -> bool:
        if self._normalizing_fraction_syntax:
            return False

        expression = self._expression_edit.text().strip()
        if "{frac}" not in expression:
            return False

        component_names, operators, open_parentheses, close_parentheses, fraction_groups = (
            parse_composite_expression(expression)
        )
        display_opens = list(open_parentheses)
        display_closes = list(close_parentheses)
        for start, end in fraction_groups:
            if display_opens[start] > 0:
                display_opens[start] -= 1
            if display_closes[end] > 0:
                display_closes[end] -= 1

        self._fraction_groups = sorted(fraction_groups)
        self._preserve_fraction_groups_once = True
        self._normalizing_fraction_syntax = True
        self._expression_edit.setText(
            build_component_expression(
                component_names,
                operators,
                display_opens,
                display_closes,
            )
        )
        self._normalizing_fraction_syntax = False
        return True

    @staticmethod
    def _structure_key(
        component_names: list[str],
        operators: list[str],
        open_parentheses: list[int],
        close_parentheses: list[int],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(component_names),
            tuple(operators),
            tuple(open_parentheses),
            tuple(close_parentheses),
        )

    @staticmethod
    def _component_spans(
        expression: str,
        component_names: list[str],
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        index = 0
        for match in _IDENTIFIER_RE.finditer(expression):
            if index >= len(component_names):
                break
            if match.group(0) != component_names[index]:
                continue
            spans.append((match.start(), match.end()))
            index += 1

        if len(spans) != len(component_names):
            raise ValueError("Could not map component positions in the expression editor")
        return spans

    @staticmethod
    def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] <= right[1] and right[0] <= left[1]

    def _fraction_group_colors(self) -> dict[tuple[int, int], str]:
        return {
            group: _FRACTION_GROUP_COLORS[index % len(_FRACTION_GROUP_COLORS)]
            for index, group in enumerate(sorted(self._fraction_groups))
        }

    def _sanitize_fraction_groups(
        self,
        component_names: list[str],
        operators: list[str],
    ) -> list[tuple[int, int]]:
        sanitized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for start, end in sorted(self._fraction_groups):
            group = (start, end)
            if group in seen:
                continue
            if start < 0 or end >= len(component_names) or start >= end:
                continue
            if any(operator != "+" for operator in operators[start:end]):
                continue
            if any(self._ranges_overlap(group, existing) for existing in sanitized):
                continue
            seen.add(group)
            sanitized.append(group)
        return sanitized

    def _build_model_from_parts(
        self,
        component_names: list[str],
        operators: list[str],
        open_parentheses: list[int],
        close_parentheses: list[int],
        fraction_groups: list[tuple[int, int]],
    ) -> CompositeModel:
        model_opens = list(open_parentheses)
        model_closes = list(close_parentheses)
        for start, end in fraction_groups:
            model_opens[start] += 1
            model_closes[end] += 1
        return CompositeModel(
            component_names,
            operators=operators,
            open_parentheses=model_opens,
            close_parentheses=model_closes,
            fraction_groups=fraction_groups,
        )

    def _selected_component_range(
        self,
    ) -> tuple[list[str], list[str], list[int], list[int], tuple[int, int]] | None:
        if not self._expression_edit.hasSelection():
            return None

        expression = self._expression_edit.text().strip()
        component_names, operators, open_parentheses, close_parentheses = parse_component_expression(
            expression,
            allowed_components=self._allowed_components,
        )
        spans = self._component_spans(expression, component_names)
        selection_start, selection_end = self._expression_edit.selectionRange()
        selected_indices = [
            index
            for index, (start, end) in enumerate(spans)
            if start < selection_end and end > selection_start
        ]
        if len(selected_indices) < 2:
            return None
        return (
            component_names,
            operators,
            open_parentheses,
            close_parentheses,
            (selected_indices[0], selected_indices[-1]),
        )

    @staticmethod
    def _render_colored_text(text: str, ranges: list[tuple[int, int, str]]) -> str:
        if not ranges:
            return html.escape(text)

        parts: list[str] = []
        cursor = 0
        for start, end, color in sorted(ranges, key=lambda entry: entry[0]):
            if cursor < start:
                parts.append(html.escape(text[cursor:start]))
            parts.append(
                f"<span style='color:{color}; font-weight:600;'>{html.escape(text[start:end])}</span>"
            )
            cursor = end
        if cursor < len(text):
            parts.append(html.escape(text[cursor:]))
        return "".join(parts)

    def _apply_fraction_visuals(
        self,
        component_names: list[str],
    ) -> list[tuple[int, int, str]]:
        expression = self._expression_edit.text().strip()
        spans = self._component_spans(expression, component_names)
        colors = self._fraction_group_colors()
        ranges = [
            (spans[start][0], spans[end][1], colors[(start, end)])
            for start, end in sorted(self._fraction_groups)
        ]
        self._expression_edit.set_highlight_ranges(ranges)
        return ranges

    def _validate_expression(self) -> tuple[bool, str | None, CompositeModel | None]:
        expression = self._expression_edit.text().strip()
        if not expression:
            return False, "Expression is required.", None

        component_names, operators, open_parentheses, close_parentheses = parse_component_expression(
            expression,
            allowed_components=self._allowed_components,
        )
        current_structure = self._structure_key(
            component_names,
            operators,
            open_parentheses,
            close_parentheses,
        )

        if (
            self._last_valid_structure is not None
            and current_structure != self._last_valid_structure
            and not self._preserve_fraction_groups_once
        ):
            self._fraction_groups = []

        self._preserve_fraction_groups_once = False
        self._fraction_groups = self._sanitize_fraction_groups(component_names, operators)
        model = self._build_model_from_parts(
            component_names,
            operators,
            open_parentheses,
            close_parentheses,
            self._fraction_groups,
        )
        self._last_valid_structure = current_structure
        return True, None, model

    def _on_fields_changed(self) -> None:
        if self._normalize_fraction_syntax_if_needed():
            return

        try:
            valid, error, model = self._validate_expression()
        except Exception as exc:
            valid, error, model = False, str(exc), None

        ok_button = self._buttons.button(self._buttons.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(valid)

        if not valid or model is None:
            self._model = None
            self._expression_edit.set_highlight_ranges([])
            self._preview_label.clear()
            self._set_status(error or "Expression is required.", valid=False)
            return

        self._model = model
        color_ranges = self._apply_fraction_visuals(model.component_names)
        expression_html = self._render_colored_text(self._expression_edit.text().strip(), color_ranges)
        preview = (
            f"Preview: <b>{html.escape(self._expression_prefix)}</b> = "
            f"{html.escape(model.formula_string())}<br>"
            f"Expression: {expression_html}"
        )
        if self._fraction_groups:
            preview += (
                "<br>Fractions: "
                + ", ".join(
                    html.escape(
                        self._expression_edit.text().strip()[start:end]
                    )
                    for start, end, _color in color_ranges
                )
            )
        self._preview_label.setText(preview)
        self._set_status("Expression is valid.", valid=True)

    def _group_selection_as_fractions(self) -> None:
        selected = self._selected_component_range()
        if selected is None:
            self._set_status(
                "Select two or more additive components before creating a fraction group.",
                valid=False,
            )
            return

        component_names, operators, open_parentheses, close_parentheses, group = selected
        if any(operator != "+" for operator in operators[group[0] : group[1]]):
            self._set_status(
                "Fraction groups can only contain additive components joined by '+'.",
                valid=False,
            )
            return

        for existing in self._fraction_groups:
            if existing == group:
                self._set_status("Selection is already a fraction group.", valid=False)
                return
            if self._ranges_overlap(existing, group):
                self._set_status(
                    "Fraction groups cannot overlap. Separate the existing group first.",
                    valid=False,
                )
                return

        candidate_groups = sorted(self._fraction_groups + [group])
        try:
            self._build_model_from_parts(
                component_names,
                operators,
                open_parentheses,
                close_parentheses,
                candidate_groups,
            )
        except Exception as exc:
            self._set_status(str(exc), valid=False)
            return

        self._fraction_groups = candidate_groups
        self._preserve_fraction_groups_once = True
        self._on_fields_changed()
        self._expression_edit.setFocus()

    def _separate_fraction_selection(self) -> None:
        selected = self._selected_component_range()
        if selected is None:
            self._set_status(
                "Select a fraction group before separating it.",
                valid=False,
            )
            return

        _component_names, _operators, _open_parentheses, _close_parentheses, group = selected
        if group not in self._fraction_groups:
            self._set_status(
                "Selection must cover an existing fraction group.",
                valid=False,
            )
            return

        self._fraction_groups = [existing for existing in self._fraction_groups if existing != group]
        self._preserve_fraction_groups_once = True
        self._on_fields_changed()
        self._expression_edit.setFocus()

    def get_composite_model(self) -> CompositeModel | None:
        """Return the model produced when the dialog is accepted."""
        model = self.built_model()
        return model if isinstance(model, CompositeModel) else None
