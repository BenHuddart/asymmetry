"""Shared calculator-style builder for composite function expressions."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog


class ComponentSelectorButton(QPushButton):
    """Menu-backed component selector with optional category submenus."""

    currentTextChanged = Signal(str)  # noqa: N815

    def __init__(
        self,
        component_pool: list[str],
        components_by_category: dict[str, list[str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._component_pool = sorted(component_pool)
        self._components_by_category = {
            key: list(value) for key, value in components_by_category.items()
        }
        self._current_text = self._component_pool[0] if self._component_pool else ""
        self._set_display_text(self._current_text or "Select function")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("text-align: left; padding: 2px 18px 2px 8px;")
        self.clicked.connect(self._open_component_menu)

    def currentText(self) -> str:  # noqa: N802
        return self._current_text

    def setCurrentText(self, name: str) -> None:  # noqa: N802
        if name not in self._component_pool:
            return
        changed = name != self._current_text
        self._current_text = name
        self._set_display_text(name)
        if changed:
            self.currentTextChanged.emit(name)

    def _set_display_text(self, text: str) -> None:
        # Keep a visible dropdown affordance even on styles where menus hide arrows.
        self.setText(f"{text}  \u25be")

    def _open_component_menu(self) -> None:
        menu = self._build_component_menu()
        if menu is None:
            return
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _build_component_menu(self) -> QMenu | None:
        if not self._component_pool:
            return None

        menu = QMenu(self)
        regular_components = self._components_by_category.get("General", [])
        for name in regular_components:
            action = menu.addAction(name)
            action.triggered.connect(lambda _checked=False, n=name: self.setCurrentText(n))

        # Preserve the caller-provided category order (the panel passes a dict
        # ordered by its canonical display order).
        submenu_categories = [
            category for category in self._components_by_category if category != "General"
        ]
        if regular_components and submenu_categories:
            menu.addSeparator()

        for category in submenu_categories:
            submenu = menu.addMenu(category)
            for name in self._components_by_category[category]:
                action = submenu.addAction(name)
                action.triggered.connect(lambda _checked=False, n=name: self.setCurrentText(n))
        return menu


class ExpressionTextEdit(QPlainTextEdit):
    """Multiline expression editor with token-aware backspace support."""

    _identifier_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, *, atomic_identifiers: set[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atomic_identifiers = set(atomic_identifiers)
        self.setMinimumHeight(86)
        self.setMaximumBlockCount(64)
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    # QLineEdit-compatible helpers used by tests and dialog methods.
    def setText(self, text: str) -> None:  # noqa: N802
        self.setPlainText(text)

    def text(self) -> str:
        return self.toPlainText()

    def insert(self, text: str) -> None:
        self.textCursor().insertText(text)

    def selectedText(self) -> str:  # noqa: N802
        return self.textCursor().selectedText().replace("\u2029", "\n")

    def hasSelection(self) -> bool:  # noqa: N802
        return self.textCursor().hasSelection()

    def selectionRange(self) -> tuple[int, int]:  # noqa: N802
        cursor = self.textCursor()
        return cursor.selectionStart(), cursor.selectionEnd()

    def replaceSelection(self, text: str) -> None:  # noqa: N802
        cursor = self.textCursor()
        cursor.insertText(text)
        self.setTextCursor(cursor)

    def cursorPosition(self) -> int:  # noqa: N802
        return self.textCursor().position()

    def setCursorPosition(self, pos: int) -> None:  # noqa: N802
        cursor = self.textCursor()
        cursor.setPosition(max(0, min(pos, len(self.text()))))
        self.setTextCursor(cursor)

    def set_highlight_ranges(self, ranges: list[tuple[int, int, str]]) -> None:
        selections: list[QTextEdit.ExtraSelection] = []
        for start, end, color in ranges:
            if end <= start:
                continue
            cursor = self.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)

            selection = QTextEdit.ExtraSelection()
            char_format = QTextCharFormat()
            char_format.setForeground(QColor(color))
            char_format.setFontWeight(600)
            selection.cursor = cursor
            selection.format = char_format
            selections.append(selection)

        self.setExtraSelections(selections)

    def remove_token_before_cursor(self) -> None:
        text = self.text()
        cursor = self.textCursor()
        position = cursor.position()
        if position <= 0:
            return

        end = position
        while end > 0 and text[end - 1].isspace():
            end -= 1
        if end <= 0:
            return

        start = end - 1
        if text[start].isalnum() or text[start] == "_":
            while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
                start -= 1

            token = text[start:end]
            if token in self._atomic_identifiers or self._identifier_pattern.match(token):
                removal_start = start
                removal_end = end
            else:
                removal_start = end - 1
                removal_end = end
        else:
            removal_start = end - 1
            removal_end = end

        new_text = text[:removal_start] + text[removal_end:]

        # Avoid doubled spaces introduced by token deletion.
        if (
            removal_start > 0
            and removal_start < len(new_text)
            and new_text[removal_start - 1].isspace()
            and new_text[removal_start].isspace()
        ):
            new_text = new_text[:removal_start] + new_text[removal_start + 1 :]

        self.setText(new_text)
        self.setCursorPosition(removal_start)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802
        if event.key() == Qt.Key.Key_Backspace:
            self.remove_token_before_cursor()
            return
        super().keyPressEvent(event)


class FunctionExpressionBuilderDialog(QDialog):
    """Generic scientific-calculator style builder for composite functions."""

    def __init__(
        self,
        *,
        title: str,
        expression_prefix: str,
        components_by_category: dict[str, list[str]],
        component_definitions: Mapping[str, object],
        model_parser: Callable[[str], object],
        initial_expression: str,
        expression_placeholder: str,
        extra_token_buttons: list[tuple[str, str]] | None = None,
        syntax_help_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(760)

        self._expression_prefix = expression_prefix
        self._components_by_category = {
            key: list(value) for key, value in components_by_category.items()
        }
        self._component_names = [
            name
            for category in sorted(self._components_by_category)
            for name in self._components_by_category[category]
        ]
        self._component_definitions = dict(component_definitions)
        self._model_parser = model_parser
        self._model: object | None = None

        root = QVBoxLayout(self)

        form = QFormLayout()
        self._expression_edit = ExpressionTextEdit(atomic_identifiers=set(self._component_names))
        self._expression_edit.setPlaceholderText(expression_placeholder)
        self._expression_edit.textChanged.connect(self._on_fields_changed)
        form.addRow("Expression:", self._expression_edit)
        root.addLayout(form)

        insertion_row = QHBoxLayout()
        self._component_selector = ComponentSelectorButton(
            self._component_names,
            self._components_by_category,
        )
        self._insert_component_button = QPushButton("Insert Function")
        self._insert_component_button.clicked.connect(self._insert_selected_component)
        self._info_button = QPushButton("Info")
        self._info_button.clicked.connect(self._show_selected_component_info)
        insertion_row.addWidget(QLabel("Functions:"))
        insertion_row.addWidget(self._component_selector, 1)
        insertion_row.addWidget(self._insert_component_button)
        insertion_row.addWidget(self._info_button)
        root.addLayout(insertion_row)

        keypad = QGridLayout()
        keypad_buttons: list[list[tuple[str, str] | None]] = [
            [("(", "("), (")", ")"), ("+", " + "), ("-", " - ")],
            [("*", " * "), ("/", " / "), ("Space", " "), None],
        ]
        for row_idx, row in enumerate(keypad_buttons):
            for col_idx, entry in enumerate(row):
                if entry is None:
                    continue
                label, token = entry
                button = QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, token_text=token: self._insert_token(token_text)
                )
                keypad.addWidget(button, row_idx, col_idx)

        self._extra_token_buttons: dict[str, QPushButton] = {}
        for offset, entry in enumerate(extra_token_buttons or []):
            label, token = entry
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, token_text=token: self._insert_token(token_text)
            )
            keypad.addWidget(button, 3, offset)
            self._extra_token_buttons[label] = button

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_expression)
        keypad.addWidget(clear_button, 2, 0, 1, 2)

        backspace_button = QPushButton("Backspace")
        backspace_button.clicked.connect(self._backspace_expression)
        keypad.addWidget(backspace_button, 2, 2)

        validate_button = QPushButton("Validate")
        validate_button.clicked.connect(self._on_fields_changed)
        keypad.addWidget(validate_button, 2, 3)

        keypad_container = QWidget()
        keypad_container.setLayout(keypad)
        root.addWidget(keypad_container)

        self._syntax_help_label = QLabel("")
        self._syntax_help_label.setWordWrap(True)
        self._syntax_help_label.setVisible(bool(syntax_help_text))
        if syntax_help_text:
            self._syntax_help_label.setText(syntax_help_text)
        root.addWidget(self._syntax_help_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._status_label)

        self._preview_label = QLabel("")
        self._preview_label.setWordWrap(True)
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._preview_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._expression_edit.setText(initial_expression)
        self._on_fields_changed()

    def built_model(self) -> object | None:
        """Return the model produced when the dialog is accepted."""
        return self._model

    def _insert_selected_component(self) -> None:
        component_name = self._component_selector.currentText().strip()
        if component_name:
            self._insert_token(component_name)

    def _insert_token(self, token: str) -> None:
        self._expression_edit.insert(token)
        self._expression_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _clear_expression(self) -> None:
        self._expression_edit.clear()

    def _backspace_expression(self) -> None:
        self._expression_edit.remove_token_before_cursor()

    def _show_selected_component_info(self) -> None:
        component_name = self._component_selector.currentText().strip()
        component = self._component_definitions.get(component_name)
        if component is None:
            return
        show_component_info_dialog(self, component)

    def _set_status(self, message: str, *, valid: bool) -> None:
        color = tokens.OK if valid else tokens.ERROR
        self._status_label.setText(f"<span style='color:{color};'>{message}</span>")

    def _validate_expression(self) -> tuple[bool, str | None, object | None]:
        expression = self._expression_edit.text().strip()
        if not expression:
            return False, "Expression is required.", None
        try:
            model = self._model_parser(expression)
        except Exception as exc:
            return False, str(exc), None
        return True, None, model

    def _on_fields_changed(self) -> None:
        valid, error, model = self._validate_expression()
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(valid)

        if not valid or model is None:
            self._set_status(error or "Expression is invalid.", valid=False)
            self._preview_label.setText("")
            return

        self._set_status("Expression is valid.", valid=True)
        formula = getattr(model, "formula_string", lambda: "")()
        self._preview_label.setText(f"Preview: {self._expression_prefix} = {formula}")

    def _on_accept(self) -> None:
        valid, error, model = self._validate_expression()
        if not valid or model is None:
            self._set_status(error or "Expression is invalid.", valid=False)
            return
        self._model = model
        self.accept()
