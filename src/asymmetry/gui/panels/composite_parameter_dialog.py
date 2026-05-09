"""Dialog for defining composite fit parameters from existing fitted parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.composite_parameters import (
    CompositeExpression,
    CompositeExpressionError,
    CompositeParameterDefinition,
    validate_composite_expression,
)


class CompositeParameterDialog(QDialog):
    """Scientific-calculator style expression builder for derived parameters."""

    def __init__(
        self,
        *,
        available_parameters: Sequence[str],
        existing_parameter_names: Sequence[str],
        initial_definition: CompositeParameterDefinition | None = None,
        preview_values: Mapping[str, float] | None = None,
        preview_uncertainties: Mapping[str, float] | None = None,
        preview_covariance: Mapping[str, Mapping[str, float]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._initial_name = initial_definition.name if initial_definition is not None else ""
        self.setWindowTitle(
            "Edit Composite Parameter"
            if initial_definition is not None
            else "Create Composite Parameter"
        )
        self.setMinimumWidth(760)

        self._available_parameters = sorted(
            {str(name) for name in available_parameters if str(name)}
        )
        self._existing_parameter_names = {
            str(name) for name in existing_parameter_names if str(name)
        }
        self._preview_values = dict(preview_values or {})
        self._preview_uncertainties = dict(preview_uncertainties or {})
        self._preview_covariance = {
            str(k): {str(subk): float(subv) for subk, subv in row.items()}
            for k, row in (preview_covariance or {}).items()
        }
        self._result: CompositeParameterDefinition | None = None

        root = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Lambda_eff")
        self._name_edit.textChanged.connect(self._on_fields_changed)
        form.addRow("Name:", self._name_edit)

        self._expression_edit = QLineEdit()
        self._expression_edit.setPlaceholderText("e.g. sqrt(Lambda^2 + A0^2)")
        self._expression_edit.textChanged.connect(self._on_fields_changed)
        form.addRow("Expression:", self._expression_edit)
        root.addLayout(form)

        insertion_row = QHBoxLayout()
        self._parameter_combo = QComboBox()
        self._parameter_combo.addItems(self._available_parameters)
        self._insert_parameter_button = QPushButton("Insert Parameter")
        self._insert_parameter_button.clicked.connect(self._insert_selected_parameter)
        insertion_row.addWidget(QLabel("Parameters:"))
        insertion_row.addWidget(self._parameter_combo, 1)
        insertion_row.addWidget(self._insert_parameter_button)
        root.addLayout(insertion_row)

        keypad = QGridLayout()
        keypad_buttons = [
            ["7", "8", "9", "/", "sin(", "cos(", "tan("],
            ["4", "5", "6", "*", "log(", "sqrt(", "("],
            ["1", "2", "3", "-", "exp(", "abs(", ")"],
            ["0", ".", "pi", "+", "^", "e", "atan("],
        ]
        for row_idx, row in enumerate(keypad_buttons):
            for col_idx, text in enumerate(row):
                button = QPushButton(text)
                button.clicked.connect(lambda _checked=False, token=text: self._insert_token(token))
                keypad.addWidget(button, row_idx, col_idx)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_expression)
        keypad.addWidget(clear_button, 4, 0, 1, 2)

        backspace_button = QPushButton("Backspace")
        backspace_button.clicked.connect(self._backspace_expression)
        keypad.addWidget(backspace_button, 4, 2, 1, 2)

        validate_button = QPushButton("Validate")
        validate_button.clicked.connect(self._on_fields_changed)
        keypad.addWidget(validate_button, 4, 4, 1, 3)

        keypad_container = QWidget()
        keypad_container.setLayout(keypad)
        root.addWidget(keypad_container)

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

        if initial_definition is not None:
            self._name_edit.setText(initial_definition.name)
            self._expression_edit.setText(initial_definition.expression)

        self._on_fields_changed()

    def composite_definition(self) -> CompositeParameterDefinition | None:
        """Return created definition when dialog was accepted."""
        return self._result

    def _insert_selected_parameter(self) -> None:
        parameter = self._parameter_combo.currentText().strip()
        if parameter:
            self._insert_token(parameter)

    def _insert_token(self, token: str) -> None:
        self._expression_edit.insert(token)
        self._expression_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _clear_expression(self) -> None:
        self._expression_edit.clear()

    def _backspace_expression(self) -> None:
        cursor_pos = self._expression_edit.cursorPosition()
        text = self._expression_edit.text()
        if cursor_pos <= 0 or not text:
            return
        new_text = text[: cursor_pos - 1] + text[cursor_pos:]
        self._expression_edit.setText(new_text)
        self._expression_edit.setCursorPosition(cursor_pos - 1)

    def _set_status(self, message: str, *, valid: bool) -> None:
        color = "#157347" if valid else "#B02A37"
        self._status_label.setText(f"<span style='color:{color};'>{message}</span>")

    def _validate_inputs(self) -> tuple[bool, str | None, CompositeExpression | None]:
        name = self._name_edit.text().strip()
        expression = self._expression_edit.text().strip()

        if not name:
            return False, "Name is required.", None

        if name in self._existing_parameter_names and name != self._initial_name:
            return False, f"Parameter name '{name}' already exists.", None

        if not expression:
            return False, "Expression is required.", None

        ok, message = validate_composite_expression(
            expression,
            allowed_symbols=self._available_parameters,
        )
        if not ok:
            return False, message or "Expression is invalid.", None

        try:
            parsed = CompositeExpression(expression)
        except CompositeExpressionError as exc:
            return False, str(exc), None

        return True, None, parsed

    def _on_fields_changed(self) -> None:
        valid, error, parsed = self._validate_inputs()
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

        if not valid:
            self._set_status(error or "Expression is invalid.", valid=False)
            self._preview_label.setText("")
            return

        self._set_status("Expression is valid.", valid=True)

        if parsed is None or not self._preview_values:
            self._preview_label.setText("")
            return

        try:
            evaluated = parsed.evaluate_with_uncertainty(
                self._preview_values,
                self._preview_uncertainties,
                covariance=self._preview_covariance,
            )
        except CompositeExpressionError as exc:
            self._preview_label.setText(f"Preview unavailable: {exc}")
            return

        self._preview_label.setText(
            f"Preview (representative row): {evaluated.value:.6g} +/- {evaluated.uncertainty:.3g}"
        )

    def _on_accept(self) -> None:
        valid, error, _parsed = self._validate_inputs()
        if not valid:
            self._set_status(error or "Expression is invalid.", valid=False)
            return

        name = self._name_edit.text().strip()
        expression = self._expression_edit.text().strip()
        self._result = CompositeParameterDefinition(name=name, expression=expression)
        self.accept()
