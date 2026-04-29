"""Dialog for building composite fit functions."""

from __future__ import annotations

from collections import defaultdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

_OPERATOR_OPTIONS = ["+", "-", "*", "/"]
_PARENTHESIS_COUNT_OPTIONS = ["0", "1", "2", "3"]


class _ComponentSelectorButton(QPushButton):
    """Menu-backed component selector with category submenus."""

    currentTextChanged = Signal(str)

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
        self.setText(self._current_text or "Select component")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("text-align: left; padding: 2px 8px;")
        self.clicked.connect(self._open_component_menu)

    def currentText(self) -> str:
        return self._current_text

    def setCurrentText(self, name: str) -> None:
        if name not in self._component_pool:
            return
        changed = name != self._current_text
        self._current_text = name
        self.setText(name)
        if changed:
            self.currentTextChanged.emit(name)

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

        submenu_categories = [
            cat for cat in sorted(self._components_by_category) if cat != "General"
        ]
        if submenu_categories and regular_components:
            menu.addSeparator()

        for category in submenu_categories:
            names = self._components_by_category[category]
            submenu = menu.addMenu(category)
            for name in names:
                action = submenu.addAction(name)
                action.triggered.connect(lambda _checked=False, n=name: self.setCurrentText(n))

        return menu


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

        self._components_by_category = self._build_components_by_category()
        self._component_names = [
            name
            for category in sorted(self._components_by_category)
            for name in self._components_by_category[category]
        ]
        self._model: CompositeModel | None = None

        layout = QVBoxLayout(self)

        self._formula_label = QLabel("")
        self._formula_label.setWordWrap(True)
        self._formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._formula_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Op", "(", "Component", "Info", ")", "Remove"])
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 60)
        self._table.setColumnWidth(2, 240)
        self._table.setColumnWidth(3, 80)
        self._table.setColumnWidth(4, 60)
        self._table.setColumnWidth(5, 90)
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
                open_count = (
                    initial_model.open_parentheses[i]
                    if i < len(initial_model.open_parentheses)
                    else 0
                )
                close_count = (
                    initial_model.close_parentheses[i]
                    if i < len(initial_model.close_parentheses)
                    else 0
                )
                self._add_component_row(cname, op, open_count=open_count, close_count=close_count)
        else:
            self._add_component_row("Exponential", "+")
            self._add_component_row("Constant", "+")

        self._update_formula_preview()

    def get_composite_model(self) -> CompositeModel | None:
        """Return the model produced when the dialog is accepted."""
        return self._model

    def _add_component_row(
        self,
        component_name: str = "Exponential",
        op: str = "+",
        *,
        open_count: int = 0,
        close_count: int = 0,
    ) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        op_combo = QComboBox()
        op_combo.addItems(_OPERATOR_OPTIONS)
        op_combo.setCurrentText(op if op in _OPERATOR_OPTIONS else "+")
        op_combo.setEnabled(row > 0)
        op_combo.currentTextChanged.connect(self._update_formula_preview)
        self._table.setCellWidget(row, 0, op_combo)

        open_combo = QComboBox()
        open_combo.addItems(_PARENTHESIS_COUNT_OPTIONS)
        open_combo.setCurrentText(
            str(open_count) if str(open_count) in _PARENTHESIS_COUNT_OPTIONS else "0"
        )
        open_combo.currentTextChanged.connect(self._update_formula_preview)
        self._table.setCellWidget(row, 1, open_combo)

        component_selector = _ComponentSelectorButton(
            self._component_names,
            self._components_by_category,
        )
        component_selector.setCurrentText(component_name)
        component_selector.currentTextChanged.connect(self._update_formula_preview)
        self._table.setCellWidget(row, 2, component_selector)

        info_btn = QPushButton("Info")
        info_btn.clicked.connect(lambda: self._show_component_info(row))
        self._table.setCellWidget(row, 3, info_btn)

        close_combo = QComboBox()
        close_combo.addItems(_PARENTHESIS_COUNT_OPTIONS)
        close_combo.setCurrentText(
            str(close_count) if str(close_count) in _PARENTHESIS_COUNT_OPTIONS else "0"
        )
        close_combo.currentTextChanged.connect(self._update_formula_preview)
        self._table.setCellWidget(row, 4, close_combo)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self._table.setCellWidget(row, 5, remove_btn)

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

            info_btn = self._table.cellWidget(row, 3)
            if isinstance(info_btn, QPushButton):
                try:
                    info_btn.clicked.disconnect()
                except RuntimeError:
                    pass
                info_btn.clicked.connect(lambda _checked=False, r=row: self._show_component_info(r))

            remove_btn = self._table.cellWidget(row, 5)
            if isinstance(remove_btn, QPushButton):
                try:
                    remove_btn.clicked.disconnect()
                except RuntimeError:
                    pass
                remove_btn.clicked.connect(lambda _checked=False, r=row: self._remove_row(r))
                remove_btn.setEnabled(self._table.rowCount() > 1)

    def _show_component_info(self, row: int) -> None:
        if row < 0 or row >= self._table.rowCount():
            return

        component_widget = self._table.cellWidget(row, 2)
        if not isinstance(component_widget, (QComboBox, _ComponentSelectorButton)):
            return

        component_name = component_widget.currentText().strip()
        component = COMPONENTS.get(component_name)
        if component is None:
            return
        show_component_info_dialog(self, component)

    def _build_components_by_category(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for name, definition in COMPONENTS.items():
            category = (definition.category or "General").strip() or "General"
            grouped[category].append(name)

        for names in grouped.values():
            names.sort()
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    def _read_ui(self) -> tuple[list[str], list[str], list[int], list[int]]:
        component_names: list[str] = []
        operators: list[str] = []
        open_parentheses: list[int] = []
        close_parentheses: list[int] = []

        for row in range(self._table.rowCount()):
            component_widget = self._table.cellWidget(row, 2)
            if not isinstance(component_widget, (QComboBox, _ComponentSelectorButton)):
                continue
            component_name = component_widget.currentText().strip()
            if not component_name:
                continue
            component_names.append(component_name)

            open_combo = self._table.cellWidget(row, 1)
            open_count = int(open_combo.currentText()) if isinstance(open_combo, QComboBox) else 0
            open_parentheses.append(open_count)

            close_combo = self._table.cellWidget(row, 4)
            close_count = (
                int(close_combo.currentText()) if isinstance(close_combo, QComboBox) else 0
            )
            close_parentheses.append(close_count)

            if row > 0:
                op_combo = self._table.cellWidget(row, 0)
                op = op_combo.currentText() if isinstance(op_combo, QComboBox) else "+"
                operators.append(op)

        return component_names, operators, open_parentheses, close_parentheses

    def _update_formula_preview(self) -> None:
        component_names, operators, open_parentheses, close_parentheses = self._read_ui()
        if not component_names:
            self._formula_label.setText("No components selected")
            return

        try:
            model = CompositeModel(
                component_names=component_names,
                operators=operators,
                open_parentheses=open_parentheses,
                close_parentheses=close_parentheses,
            )
        except Exception as exc:
            self._formula_label.setText(f"Invalid function: {exc}")
            return

        self._formula_label.setText(f"A(t) = {model.formula_string()}")

    def _on_accept(self) -> None:
        component_names, operators, open_parentheses, close_parentheses = self._read_ui()
        if not component_names:
            return
        try:
            self._model = CompositeModel(
                component_names=component_names,
                operators=operators,
                open_parentheses=open_parentheses,
                close_parentheses=close_parentheses,
            )
        except Exception as exc:
            self._formula_label.setText(f"Invalid function: {exc}")
            return
        self.accept()
