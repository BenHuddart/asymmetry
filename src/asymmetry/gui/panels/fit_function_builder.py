"""Dialog for building composite fit functions."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel

_OPERATOR_OPTIONS = ["+", "-", "*", "/"]


class FitFunctionBuilderDialog(QDialog):
    """Compose a custom fit function from predefined components."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_model: CompositeModel | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Fit Function")
        self.setMinimumWidth(720)

        self._component_names = sorted(COMPONENTS.keys())
        self._model: CompositeModel | None = None

        layout = QVBoxLayout(self)

        self._formula_label = QLabel("")
        self._formula_label.setWordWrap(True)
        self._formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._formula_label)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Op", "Component", "Remove"])
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 300)
        self._table.setColumnWidth(2, 90)
        layout.addWidget(self._table)

        button_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Component")
        self._add_btn.clicked.connect(self._add_component_row)
        button_row.addWidget(self._add_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self._dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._dialog_buttons.accepted.connect(self._on_accept)
        self._dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(self._dialog_buttons)

        if initial_model is not None:
            for i, cname in enumerate(initial_model.component_names):
                op = initial_model.operators[i - 1] if i > 0 else "+"
                self._add_component_row(cname, op)
        else:
            self._add_component_row("Exponential", "+")
            self._add_component_row("Constant", "+")

        self._update_formula_preview()

    def get_composite_model(self) -> CompositeModel | None:
        """Return the model produced when the dialog is accepted."""
        return self._model

    def _add_component_row(self, component_name: str = "Exponential", op: str = "+") -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        op_combo = QComboBox()
        op_combo.addItems(_OPERATOR_OPTIONS)
        op_combo.setCurrentText(op if op in _OPERATOR_OPTIONS else "+")
        op_combo.setEnabled(row > 0)
        op_combo.currentTextChanged.connect(self._update_formula_preview)
        self._table.setCellWidget(row, 0, op_combo)

        component_combo = QComboBox()
        component_combo.addItems(self._component_names)
        if component_name in self._component_names:
            component_combo.setCurrentText(component_name)
        component_combo.currentTextChanged.connect(self._update_formula_preview)
        self._table.setCellWidget(row, 1, component_combo)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self._table.setCellWidget(row, 2, remove_btn)

        self._refresh_row_controls()
        self._update_formula_preview()

    def _remove_row(self, row: int) -> None:
        if self._table.rowCount() <= 1:
            return
        if row < 0 or row >= self._table.rowCount():
            return
        self._table.removeRow(row)
        self._refresh_row_controls()
        self._update_formula_preview()

    def _refresh_row_controls(self) -> None:
        for row in range(self._table.rowCount()):
            op_combo = self._table.cellWidget(row, 0)
            if isinstance(op_combo, QComboBox):
                op_combo.setEnabled(row > 0)

            remove_btn = self._table.cellWidget(row, 2)
            if isinstance(remove_btn, QPushButton):
                try:
                    remove_btn.clicked.disconnect()
                except RuntimeError:
                    pass
                remove_btn.clicked.connect(lambda _checked=False, r=row: self._remove_row(r))
                remove_btn.setEnabled(self._table.rowCount() > 1)

    def _read_ui(self) -> tuple[list[str], list[str]]:
        component_names: list[str] = []
        operators: list[str] = []

        for row in range(self._table.rowCount()):
            component_combo = self._table.cellWidget(row, 1)
            if not isinstance(component_combo, QComboBox):
                continue
            component_names.append(component_combo.currentText())

            if row > 0:
                op_combo = self._table.cellWidget(row, 0)
                op = op_combo.currentText() if isinstance(op_combo, QComboBox) else "+"
                operators.append(op)

        return component_names, operators

    def _update_formula_preview(self) -> None:
        component_names, operators = self._read_ui()
        if not component_names:
            self._formula_label.setText("No components selected")
            return

        try:
            model = CompositeModel(component_names=component_names, operators=operators)
        except Exception as exc:
            self._formula_label.setText(f"Invalid function: {exc}")
            return

        self._formula_label.setText(f"A(t) = {model.formula_string()}")

    def _on_accept(self) -> None:
        component_names, operators = self._read_ui()
        if not component_names:
            return
        try:
            self._model = CompositeModel(component_names=component_names, operators=operators)
        except Exception as exc:
            self._formula_label.setText(f"Invalid function: {exc}")
            return
        self.accept()
