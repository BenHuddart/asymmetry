"""Dialog for fitting parameter trends vs field/temperature."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import traceback
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
    ParameterModelFit,
    component_names_for_x,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

_OPERATOR_OPTIONS = ["+", "-", "*", "/"]

_Y_PARAM_UNITS = {
    "A": "%",
    "A0": "%",
    "A_bg": "%",
    "baseline": "%",
    "Lambda": "us^-1",
    "sigma": "us^-1",
    "Delta": "us^-1",
    "frequency": "MHz",
    "phase": "rad",
}

_PARAM_UNITS = {
    "a": None,
    "b": None,
    "c": None,
    "f": "us^-1",
    "A": "MHz",
    "D": "MHz",
    "nu": "MHz",
    "m": None,
    "tau": "(x units)",
    "B0": "G",
    "Bwid": "G",
    "Tc": "K",
    "Ea": "meV",
    "C": "MHz",  # legacy alias used in older saved model-fit states
    "D_2D": "us^-1",
    "D_nD": "us^-1",
    "D_perp": "us^-1",
    "lambda_BG": "us^-1",
    "lambda_0D": "us^-1",
}

_NON_NEGATIVE_PARAMS = {"D", "D_2D", "D_nD", "D_perp", "lambda_BG", "lambda_0D", "f"}
_STRICTLY_POSITIVE_PARAMS = {"tau", "B0", "Bwid", "nu", "m"}
_POSITIVE_EPS = 1e-12
_SC_COMPONENT_MENU_TITLE = "Superconducting Gap Models"


class _ComponentSelectorButton(QPushButton):
    """Menu-backed component selector with a superconducting submenu."""

    currentTextChanged = Signal(str)

    def __init__(self, component_pool: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._component_pool = sorted(component_pool)
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
        regular_components = [name for name in self._component_pool if not name.startswith("SC_")]
        sc_components = [name for name in self._component_pool if name.startswith("SC_")]

        for name in regular_components:
            action = menu.addAction(name)
            action.triggered.connect(lambda _checked=False, n=name: self.setCurrentText(n))

        if sc_components:
            if regular_components:
                menu.addSeparator()
            sc_menu = menu.addMenu(_SC_COMPONENT_MENU_TITLE)
            for name in sc_components:
                action = sc_menu.addAction(name)
                action.triggered.connect(lambda _checked=False, n=name: self.setCurrentText(n))

        return menu


def _base_param_name(name: str) -> str:
    match = re.match(r"^(.+)_\d+$", name)
    return match.group(1) if match else name


def _x_unit(x_key: str) -> str | None:
    if x_key == "field":
        return "G"
    if x_key == "temperature":
        return "K"
    if x_key == "run":
        return "run"
    return None


def _y_unit(parameter_name: str) -> str | None:
    return _Y_PARAM_UNITS.get(_base_param_name(parameter_name))


def _format_param_label(name: str, x_key: str, parameter_name: str) -> str:
    """Return display label with units for range-parameter table."""
    base = _base_param_name(name)
    y_unit = _y_unit(parameter_name)
    x_unit = _x_unit(x_key)

    if base == "tau":
        unit = x_unit or "(x units)"
    elif base == "m":
        if y_unit and x_unit:
            unit = f"{y_unit} / {x_unit}"
        elif y_unit:
            unit = f"{y_unit} / x"
        else:
            unit = "(y units / x unit)"
    elif base in {"a", "b", "c"}:
        unit = y_unit or "(y units)"
    else:
        unit = _PARAM_UNITS.get(base)

    return f"{name} [{unit}]" if unit else name


def _format_model_param_label(
    model: ParameterCompositeModel,
    name: str,
    x_key: str,
    parameter_name: str,
) -> str:
    """Return display label for a specific model parameter.

    Keeps Redfield exponent ``m`` unitless while using unit-aware labels for
    all other parameters.
    """
    component_for_param: dict[str, str] = {}
    for mapping, component in zip(model._param_mappings, model.components, strict=True):
        for unique_name in mapping.values():
            component_for_param[unique_name] = component.name

    if _base_param_name(name) == "m" and component_for_param.get(name) == "Redfield":
        return name
    return _format_param_label(name, x_key, parameter_name)


def _component_name_for_param(model: ParameterCompositeModel, name: str) -> str | None:
    """Return component name that owns a unique model parameter name."""
    for mapping, component in zip(model._param_mappings, model.components, strict=True):
        for unique_name in mapping.values():
            if unique_name == name:
                return component.name
    return None


def _should_reset_param_on_model_change(model: ParameterCompositeModel, name: str) -> bool:
    """Return True when model changes should prefer defaults over name-based carryover."""
    return _base_param_name(name) == "m" and _component_name_for_param(model, name) == "Redfield"


def _component_pool_for_context(x_key: str, parameter_name: str) -> list[str]:
    """Return component pool with context-specific redundancy filtering."""
    available = component_names_for_x(x_key)
    if x_key != "field":
        return available

    base = _base_param_name(parameter_name).strip().lower()
    is_lambda_like = base.startswith("lambda") or base.startswith("λ")

    if is_lambda_like:
        return [name for name in available if name != "Constant"]
    return [name for name in available if name != "Lambda_bg"]


def _normalize_parameter_limits(
    name: str,
    value: float,
    p_min: float,
    p_max: float,
) -> tuple[float, float, float, list[str]]:
    """Normalize start/min/max based on model-domain expectations."""
    notes: list[str] = []
    base = _base_param_name(name)

    if p_min > p_max:
        p_min, p_max = p_max, p_min
        notes.append(f"{name}: swapped min/max")

    if base in _NON_NEGATIVE_PARAMS:
        if p_min < 0.0:
            p_min = 0.0
            notes.append(f"{name}: min clamped to 0")
        if p_max < p_min:
            p_max = p_min + 1.0
            notes.append(f"{name}: max raised above min")

    if base in _STRICTLY_POSITIVE_PARAMS:
        if p_min < _POSITIVE_EPS:
            p_min = _POSITIVE_EPS
            notes.append(f"{name}: min clamped to >0")
        if p_max < p_min:
            p_max = p_min * 10.0
            notes.append(f"{name}: max raised above positive min")

    clamped_value = min(max(value, p_min), p_max)
    if clamped_value != value:
        value = clamped_value
        notes.append(f"{name}: start value clamped to bounds")

    return value, p_min, p_max, notes


def _in_test_mode() -> bool:
    """Return True when running under pytest to avoid modal popups."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _show_info(parent: QWidget, title: str, text: str) -> None:
    """Show informational message in interactive mode only."""
    if _in_test_mode():
        return
    QMessageBox.information(parent, title, text)


def _show_warning(parent: QWidget, title: str, text: str) -> None:
    """Show warning message in interactive mode only."""
    if _in_test_mode():
        return
    QMessageBox.warning(parent, title, text)


class ParameterModelBuilderDialog(QDialog):
    """Compose a parameter model from basis components."""

    def __init__(
        self,
        component_pool: list[str],
        parent: QWidget | None = None,
        initial_model: ParameterCompositeModel | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Parameter Model")
        self.setMinimumWidth(700)

        self._component_pool = sorted(component_pool)
        self._model: ParameterCompositeModel | None = None

        layout = QVBoxLayout(self)
        self._formula_label = QLabel("")
        self._formula_label.setWordWrap(True)
        self._formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._formula_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Op", "Component", "Info", "Remove"])
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 300)
        self._table.setColumnWidth(2, 80)
        self._table.setColumnWidth(3, 100)
        layout.addWidget(self._table)

        row_btn = QPushButton("Add Component")
        row_btn.clicked.connect(self._add_row)
        layout.addWidget(row_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if initial_model is not None:
            for i, cname in enumerate(initial_model.component_names):
                op = initial_model.operators[i - 1] if i > 0 else "+"
                self._add_row(cname, op)
        else:
            default_name = self._component_pool[0] if self._component_pool else "Constant"
            self._add_row(default_name, "+")

        self._update_formula()

    def get_model(self) -> ParameterCompositeModel | None:
        return self._model

    def _add_row(self, component_name: str | None = None, op: str = "+") -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        op_combo = QComboBox()
        op_combo.addItems(_OPERATOR_OPTIONS)
        op_combo.setCurrentText(op if op in _OPERATOR_OPTIONS else "+")
        op_combo.setEnabled(row > 0)
        op_combo.currentTextChanged.connect(self._update_formula)
        self._table.setCellWidget(row, 0, op_combo)

        comp_selector = _ComponentSelectorButton(self._component_pool)
        if component_name and component_name in self._component_pool:
            comp_selector.setCurrentText(component_name)
        comp_selector.currentTextChanged.connect(self._update_formula)
        self._table.setCellWidget(row, 1, comp_selector)

        info_btn = QPushButton("Info")
        info_btn.clicked.connect(lambda: self._show_component_info(row))
        self._table.setCellWidget(row, 2, info_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self._table.setCellWidget(row, 3, remove_btn)

        self._refresh_row_bindings()
        self._update_formula()

    def _remove_row(self, row: int) -> None:
        if self._table.rowCount() <= 1:
            return
        if row < 0 or row >= self._table.rowCount():
            return
        self._table.removeRow(row)
        self._refresh_row_bindings()
        self._update_formula()

    def _refresh_row_bindings(self) -> None:
        for row in range(self._table.rowCount()):
            op_combo = self._table.cellWidget(row, 0)
            if isinstance(op_combo, QComboBox):
                op_combo.setEnabled(row > 0)

            info_btn = self._table.cellWidget(row, 2)
            if isinstance(info_btn, QPushButton):
                try:
                    info_btn.clicked.disconnect()
                except RuntimeError:
                    pass
                info_btn.clicked.connect(lambda _checked=False, r=row: self._show_component_info(r))

            remove_btn = self._table.cellWidget(row, 3)
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

        comp_widget = self._table.cellWidget(row, 1)
        if not isinstance(comp_widget, (QComboBox, _ComponentSelectorButton)):
            return

        component_name = comp_widget.currentText().strip()
        component = PARAMETER_MODEL_COMPONENTS.get(component_name)
        if component is None:
            return
        show_component_info_dialog(self, component)

    def _read_ui(self) -> tuple[list[str], list[str]]:
        component_names: list[str] = []
        operators: list[str] = []

        for row in range(self._table.rowCount()):
            comp_widget = self._table.cellWidget(row, 1)
            if not isinstance(comp_widget, (QComboBox, _ComponentSelectorButton)):
                continue
            component_names.append(comp_widget.currentText())
            if row > 0:
                op_combo = self._table.cellWidget(row, 0)
                op = op_combo.currentText() if isinstance(op_combo, QComboBox) else "+"
                operators.append(op)

        return component_names, operators

    def _update_formula(self) -> None:
        component_names, operators = self._read_ui()
        if not component_names:
            self._formula_label.setText("No components")
            return

        try:
            model = ParameterCompositeModel(component_names=component_names, operators=operators)
        except Exception as exc:
            self._formula_label.setText(f"Invalid model: {exc}")
            return

        self._formula_label.setText(f"y(x) = {model.formula_string()}")

    def _accept(self) -> None:
        component_names, operators = self._read_ui()
        try:
            self._model = ParameterCompositeModel(component_names=component_names, operators=operators)
        except Exception as exc:
            self._formula_label.setText(f"Invalid model: {exc}")
            return
        self.accept()


@dataclass
class _RangeWidgets:
    active: QCheckBox
    x_min: QDoubleSpinBox
    x_max: QDoubleSpinBox
    model_label: QLabel
    edit_button: QPushButton
    fit_button: QPushButton
    remove_button: QPushButton


class _FitWorker(QObject):
    """Run a fit task off the UI thread and return the result."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[[], object]) -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:
        try:
            self.finished.emit(self._task())
        except Exception:
            self.failed.emit(traceback.format_exc())


class ModelFitDialog(QDialog):
    """Configure and run model fits for one Y parameter vs selected X variable."""

    def __init__(
        self,
        parameter_name: str,
        x_key: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_errors: np.ndarray,
        existing_fit: ParameterModelFit | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(f"Model Fit: {parameter_name} vs {x_key}")
        self.resize(950, 650)

        self._parameter_name = parameter_name
        self._x_key = x_key
        self._component_pool = _component_pool_for_context(x_key, parameter_name)
        self._x = np.asarray(x_values, dtype=float)
        self._y = np.asarray(y_values, dtype=float)
        self._yerr = np.asarray(y_errors, dtype=float)
        self._removed = False
        self._range_widgets: list[_RangeWidgets] = []
        self._active_range_idx: int | None = None
        self._fit_in_progress = False
        self._fit_thread: QThread | None = None
        self._fit_worker: _FitWorker | None = None
        self._fit_done_callback: Callable[[object], None] | None = None

        if existing_fit is not None and existing_fit.ranges:
            self._fit = existing_fit
        else:
            self._fit = ParameterModelFit(parameter_name=parameter_name, x_key=x_key, ranges=[])
            self._fit.ranges.append(self._create_default_range())

        layout = QVBoxLayout(self)

        summary = QLabel(
            f"Y parameter: <b>{parameter_name}</b> | X variable: <b>{x_key}</b>"
        )
        layout.addWidget(summary)

        x_min_data = float(np.nanmin(self._x)) if np.any(np.isfinite(self._x)) else 0.0
        x_max_data = float(np.nanmax(self._x)) if np.any(np.isfinite(self._x)) else 1.0
        self._data_range_label = QLabel(f"Data range: {x_min_data:.6g} to {x_max_data:.6g}")
        layout.addWidget(self._data_range_label)

        ranges_group = QGroupBox("Model ranges")
        ranges_layout = QVBoxLayout(ranges_group)
        self._ranges_host = QVBoxLayout()
        ranges_layout.addLayout(self._ranges_host)

        add_row = QHBoxLayout()
        add_btn = QPushButton("Add Range")
        add_btn.clicked.connect(self._add_range)
        add_row.addWidget(add_btn)
        add_row.addStretch()
        ranges_layout.addLayout(add_row)

        layout.addWidget(ranges_group)

        params_group = QGroupBox("Range parameters")
        params_layout = QVBoxLayout(params_group)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Editing range:"))
        self._range_selector = QComboBox()
        self._range_selector.currentIndexChanged.connect(self._on_range_selector_changed)
        selector_row.addWidget(self._range_selector, 1)
        params_layout.addLayout(selector_row)

        self._range_hint_label = QLabel("Select a range above to edit its model parameters.")
        params_layout.addWidget(self._range_hint_label)

        self._formula_label = QLabel("")
        self._formula_label.setWordWrap(True)
        params_layout.addWidget(self._formula_label)

        self._chi2_label = QLabel("")
        self._chi2_label.setTextFormat(Qt.TextFormat.RichText)
        params_layout.addWidget(self._chi2_label)

        self._fit_progress_label = QLabel("")
        self._fit_progress_label.setStyleSheet("color: #9a6700;")
        self._fit_progress_label.setVisible(False)
        params_layout.addWidget(self._fit_progress_label)

        self._param_table = QTableWidget(0, 6)
        self._param_table.setHorizontalHeaderLabels(["Name", "Value", "Min", "Max", "Fixed", "Error"])
        self._param_table.itemChanged.connect(self._on_param_table_edited)
        params_layout.addWidget(self._param_table)

        layout.addWidget(params_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        remove_fit_btn = QPushButton("Remove Fit")
        remove_fit_btn.clicked.connect(self._on_remove_fit)
        buttons.addButton(remove_fit_btn, QDialogButtonBox.ButtonRole.DestructiveRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        self._remove_fit_btn = remove_fit_btn
        self._add_range_btn = add_btn
        layout.addWidget(buttons)

        self._rebuild_ranges_ui()
        self._select_range(0)

    def get_model_fit(self) -> ParameterModelFit | None:
        if self._removed:
            return None
        return self._fit

    def was_removed(self) -> bool:
        return self._removed

    def _create_default_range(self) -> ModelFitRange:
        x_min = float(np.nanmin(self._x)) if np.any(np.isfinite(self._x)) else 0.0
        x_max = float(np.nanmax(self._x)) if np.any(np.isfinite(self._x)) else 1.0

        available = self._component_pool
        default_component = "Linear" if "Linear" in available else (available[0] if available else "Constant")
        model = ParameterCompositeModel([default_component], [])

        params = ParameterSet()
        y_mean = float(np.nanmean(self._y)) if np.any(np.isfinite(self._y)) else 0.0
        y_span = float(np.nanmax(self._y) - np.nanmin(self._y)) if np.any(np.isfinite(self._y)) else 1.0

        for pname in model.param_names:
            default_val = model.param_defaults[pname]
            if pname in {"c", "b"}:
                default_val = y_mean
            elif pname in {"m", "a"}:
                default_val = y_span if y_span > 0 else default_val
            elif pname.startswith("B0") or pname.startswith("tau") or pname.startswith("nu"):
                default_val = max(1e-6, (x_max - x_min) / 2.0)
            elif pname.startswith("D_2D"):
                default_val = max(1e-6, default_val)
            elif pname.startswith("D"):
                default_val = max(1e-6, default_val)
            params.add(Parameter(name=pname, value=float(default_val), fixed=(pname == "shape_factor_a")))

        return ModelFitRange(x_min=x_min, x_max=x_max, model=model, parameters=params)

    def _add_range(self) -> None:
        self._fit.ranges.append(self._create_default_range())
        self._rebuild_ranges_ui()
        self._select_range(len(self._fit.ranges) - 1)

    def _remove_range(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        if len(self._fit.ranges) == 1:
            _show_info(self, "Range required", "At least one range must remain.")
            return
        del self._fit.ranges[idx]
        self._rebuild_ranges_ui()
        self._select_range(max(0, idx - 1))

    def _rebuild_ranges_ui(self) -> None:
        previous_idx = self._active_range_idx if self._active_range_idx is not None else 0

        while self._ranges_host.count():
            item = self._ranges_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._range_widgets = []

        for idx, fit_range in enumerate(self._fit.ranges):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)

            active = QCheckBox("active")
            active.setChecked(True)
            active.stateChanged.connect(lambda _state, i=idx: self._on_range_active_changed(i))
            row.addWidget(active)

            row.addWidget(QLabel(f"Range {idx + 1}"))

            xmin = QDoubleSpinBox()
            xmin.setRange(-1e12, 1e12)
            xmin.setDecimals(8)
            xmin.setValue(float(fit_range.x_min if fit_range.x_min is not None else 0.0))
            xmin.valueChanged.connect(lambda _v, i=idx: self._on_range_bounds_changed(i))
            row.addWidget(QLabel("x min"))
            row.addWidget(xmin)

            xmax = QDoubleSpinBox()
            xmax.setRange(-1e12, 1e12)
            xmax.setDecimals(8)
            xmax.setValue(float(fit_range.x_max if fit_range.x_max is not None else 0.0))
            xmax.valueChanged.connect(lambda _v, i=idx: self._on_range_bounds_changed(i))
            row.addWidget(QLabel("x max"))
            row.addWidget(xmax)

            model_label = QLabel(fit_range.model.formula_string())
            model_label.setMinimumWidth(220)
            row.addWidget(model_label)

            status_label = QLabel(self._status_text_for_range(fit_range))
            status_label.setTextFormat(Qt.TextFormat.RichText)
            row.addWidget(status_label)

            edit_btn = QPushButton("Edit Model")
            edit_btn.clicked.connect(lambda _checked=False, i=idx: self._edit_model(i))
            row.addWidget(edit_btn)

            fit_btn = QPushButton("Run Fit")
            fit_btn.clicked.connect(lambda _checked=False, i=idx: self._run_fit(i))
            row.addWidget(fit_btn)

            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda _checked=False, i=idx: self._remove_range(i))
            row.addWidget(remove_btn)

            select_btn = QPushButton("Edit Params")
            select_btn.clicked.connect(lambda _checked=False, i=idx: self._select_range(i))
            row.addWidget(select_btn)

            row.addStretch()
            self._ranges_host.addWidget(row_widget)

            self._range_widgets.append(
                _RangeWidgets(
                    active=active,
                    x_min=xmin,
                    x_max=xmax,
                    model_label=model_label,
                    edit_button=edit_btn,
                    fit_button=fit_btn,
                    remove_button=remove_btn,
                )
            )

        self._refresh_range_selector()
        if self._fit.ranges:
            self._select_range(max(0, min(previous_idx, len(self._fit.ranges) - 1)))

    def _refresh_range_selector(self) -> None:
        self._range_selector.blockSignals(True)
        self._range_selector.clear()
        for idx, fit_range in enumerate(self._fit.ranges, start=1):
            x_min = fit_range.x_min if fit_range.x_min is not None else float("nan")
            x_max = fit_range.x_max if fit_range.x_max is not None else float("nan")
            text = f"Range {idx}: [{x_min:.6g}, {x_max:.6g}]"
            self._range_selector.addItem(text)
        self._range_selector.blockSignals(False)

    def _on_range_selector_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self._select_range(idx)

    def _status_text_for_range(self, fit_range: ModelFitRange) -> str:
        if fit_range.result is None:
            return '<span style="color:#1f6feb;">Not run</span>'
        if fit_range.result.success:
            return '<span style="color:#22863a;">Success</span>'
        return '<span style="color:#d73a49;">Failed</span>'

    def _on_range_active_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return

    def _on_range_bounds_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        widgets = self._range_widgets[idx]
        fit_range = self._fit.ranges[idx]
        fit_range.x_min = float(widgets.x_min.value())
        fit_range.x_max = float(widgets.x_max.value())

    def _edit_model(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return

        fit_range = self._fit.ranges[idx]
        dlg = ParameterModelBuilderDialog(
            component_pool=self._component_pool,
            initial_model=fit_range.model,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        model = dlg.get_model()
        if model is None:
            return

        fit_range.model = model

        new_params = ParameterSet()
        for pname in model.param_names:
            if pname in fit_range.parameters and not _should_reset_param_on_model_change(model, pname):
                old = fit_range.parameters[pname]
                new_params.add(Parameter(name=pname, value=old.value, min=old.min, max=old.max, fixed=old.fixed))
            else:
                new_params.add(
                    Parameter(
                        name=pname,
                        value=float(model.param_defaults[pname]),
                        fixed=(pname == "shape_factor_a"),
                    )
                )
        fit_range.parameters = new_params
        fit_range.result = None

        self._rebuild_ranges_ui()
        self._select_range(idx)

    def _run_fit(self, idx: int) -> None:
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Please wait for the current fit to finish.")
            return
        if idx < 0 or idx >= len(self._fit.ranges):
            return

        self._commit_param_table(notify_adjustments=True)
        fit_range = self._fit.ranges[idx]

        if fit_range.x_max is not None and fit_range.x_min is not None and fit_range.x_max <= fit_range.x_min:
            _show_warning(self, "Invalid range", "x max must be greater than x min.")
            return

        model_snapshot = ParameterCompositeModel(
            component_names=list(fit_range.model.component_names),
            operators=list(fit_range.model.operators),
        )
        params_snapshot = ParameterSet(
            [
                Parameter(
                    name=p.name,
                    value=float(p.value),
                    min=float(p.min),
                    max=float(p.max),
                    fixed=bool(p.fixed),
                )
                for p in fit_range.parameters
            ]
        )
        x_vals = np.asarray(self._x, dtype=float).copy()
        y_vals = np.asarray(self._y, dtype=float).copy()
        y_errs = np.asarray(self._yerr, dtype=float).copy()
        x_min = fit_range.x_min
        x_max = fit_range.x_max

        self._fit_progress_label.setText(f"Fit in progress for Range {idx + 1}...")

        def _task():
            return fit_parameter_model(
                x=x_vals,
                y=y_vals,
                yerr=y_errs,
                model=model_snapshot,
                parameters=params_snapshot,
                x_min=x_min,
                x_max=x_max,
            )

        def _on_done(result: object) -> None:
            fit_result = result
            fit_range.result = fit_result
            if fit_result.success:
                fit_range.parameters = fit_result.parameters

            self._select_range(idx)

            if fit_result.success:
                _show_info(
                    self,
                    "Fit complete",
                    f"Range {idx + 1} fit succeeded. Reduced chi2 = {fit_result.reduced_chi_squared:.4g}",
                )
            else:
                _show_warning(self, "Fit failed", fit_result.message or "Model fit failed")

        self._start_fit_task(_task, _on_done)

    def _select_range(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        self._active_range_idx = idx
        fit_range = self._fit.ranges[idx]

        if self._range_selector.currentIndex() != idx:
            self._range_selector.blockSignals(True)
            self._range_selector.setCurrentIndex(idx)
            self._range_selector.blockSignals(False)

        self._formula_label.setText(f"y(x) = {fit_range.model.formula_string()}")

        self._range_hint_label.setText(
            f"Editing parameters for Range {idx + 1}. Run Fit to update result values/uncertainties."
        )

        if fit_range.result is not None:
            if fit_range.result.success:
                self._chi2_label.setText(
                    (
                        '<span style="color:#22863a;">'
                        f"Fit successful: chi2 = {fit_range.result.chi_squared:.6g}, "
                        f"reduced chi2 = {fit_range.result.reduced_chi_squared:.6g}"
                        "</span>"
                    )
                )
            else:
                self._chi2_label.setText(
                    (
                        '<span style="color:#d73a49;">'
                        f"Fit failed: {fit_range.result.message or 'No convergence'}"
                        "</span>"
                    )
                )
        else:
            self._chi2_label.setText(
                '<span style="color:#1f6feb;">Fitting not yet run for selected range</span>'
            )

        self._param_table.blockSignals(True)
        self._param_table.setRowCount(0)
        for row, param in enumerate(fit_range.parameters):
            self._param_table.insertRow(row)
            display_name = _format_model_param_label(
                fit_range.model,
                param.name,
                self._x_key,
                self._parameter_name,
            )
            name_item = QTableWidgetItem(
                display_name
            )
            name_item.setData(Qt.ItemDataRole.UserRole, param.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(row, 0, name_item)
            self._param_table.setItem(row, 1, QTableWidgetItem(f"{param.value:.8g}"))
            self._param_table.setItem(row, 2, QTableWidgetItem(f"{param.min:.8g}"))
            self._param_table.setItem(row, 3, QTableWidgetItem(f"{param.max:.8g}"))

            fixed = QCheckBox()
            fixed.setChecked(bool(param.fixed))
            fixed_container = QWidget()
            fixed_layout = QHBoxLayout(fixed_container)
            fixed_layout.setContentsMargins(0, 0, 0, 0)
            fixed_layout.addWidget(fixed)
            fixed_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._param_table.setCellWidget(row, 4, fixed_container)
            fixed.stateChanged.connect(lambda _state: self._on_param_table_edited())

            err = np.nan
            if fit_range.result is not None:
                err = fit_range.result.uncertainties.get(param.name, np.nan)
            self._param_table.setItem(row, 5, QTableWidgetItem(f"{err:.4g}" if np.isfinite(err) else ""))

        self._param_table.blockSignals(False)
        self._param_table.resizeColumnsToContents()

    def _on_param_table_edited(self, *_args: object) -> None:
        """Persist parameter edits immediately and invalidate stale fit results."""
        if self._active_range_idx is None:
            return

        fit_range = self._fit.ranges[self._active_range_idx]
        self._commit_param_table()
        fit_range.result = None
        self._chi2_label.setText(
            '<span style="color:#1f6feb;">Fitting not yet run for selected range</span>'
        )

    def _commit_param_table(self, *, notify_adjustments: bool = False) -> None:
        if self._active_range_idx is None:
            return

        fit_range = self._fit.ranges[self._active_range_idx]
        new_params = ParameterSet()
        adjustments: list[str] = []
        self._param_table.blockSignals(True)
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            value_item = self._param_table.item(row, 1)
            min_item = self._param_table.item(row, 2)
            max_item = self._param_table.item(row, 3)
            fixed_widget = self._param_table.cellWidget(row, 4)

            if name_item is None or value_item is None:
                continue

            name_data = name_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(name_data, str) and name_data.strip():
                name = name_data.strip()
            else:
                name = name_item.text().strip()
            if not name:
                continue

            try:
                value = float(value_item.text())
            except (TypeError, ValueError):
                value = 0.0

            try:
                p_min = float(min_item.text()) if min_item is not None else -float("inf")
            except (TypeError, ValueError):
                p_min = -float("inf")

            try:
                p_max = float(max_item.text()) if max_item is not None else float("inf")
            except (TypeError, ValueError):
                p_max = float("inf")

            value, p_min, p_max, notes = _normalize_parameter_limits(name, value, p_min, p_max)
            adjustments.extend(notes)

            if value_item is not None:
                value_item.setText(f"{value:.8g}")
            if min_item is not None:
                min_item.setText(f"{p_min:.8g}")
            if max_item is not None:
                max_item.setText(f"{p_max:.8g}")

            fixed = False
            if fixed_widget is not None and fixed_widget.layout() is not None and fixed_widget.layout().count() > 0:
                inner = fixed_widget.layout().itemAt(0).widget()
                if isinstance(inner, QCheckBox):
                    fixed = inner.isChecked()

            new_params.add(Parameter(name=name, value=value, min=p_min, max=p_max, fixed=fixed))

        self._param_table.blockSignals(False)
        fit_range.parameters = new_params

        if adjustments:
            self._range_hint_label.setText(
                "Adjusted parameter values to satisfy model-domain requirements "
                "(e.g. positive tau/B0/nu and non-negative diffusion rates)."
            )
            if notify_adjustments:
                _show_info(self, "Parameter limits adjusted", "; ".join(dict.fromkeys(adjustments)))

    def _on_remove_fit(self) -> None:
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Cannot remove fit while fitting is in progress.")
            return
        self._removed = True
        self.accept()

    def reject(self) -> None:
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Please wait for the current fit to finish.")
            return
        super().reject()

    def _start_fit_task(self, task: Callable[[], object], on_done: Callable[[object], None]) -> None:
        if self._fit_in_progress:
            return

        thread = QThread(self)
        worker = _FitWorker(task)
        worker.moveToThread(thread)

        self._fit_in_progress = True
        self._set_fit_ui_busy(True)
        self._fit_thread = thread
        self._fit_worker = worker
        self._fit_done_callback = on_done

        def _cleanup() -> None:
            self._fit_in_progress = False
            self._set_fit_ui_busy(False)
            self._fit_worker = None
            self._fit_thread = None

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_fit_worker_finished)
        worker.failed.connect(self._on_fit_worker_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(_cleanup)
        thread.start()

    def _on_fit_worker_finished(self, result: object) -> None:
        """Handle fit completion on the dialog (UI) thread."""
        callback = self._fit_done_callback
        thread = self._fit_thread
        if callback is not None:
            callback(result)
        if thread is not None:
            thread.quit()

    def _on_fit_worker_failed(self, trace: str) -> None:
        """Handle fit failure on the dialog (UI) thread."""
        _show_warning(self, "Fit failed", f"Unexpected error during fitting.\n\n{trace}")
        thread = self._fit_thread
        if thread is not None:
            thread.quit()

    def _set_fit_ui_busy(self, busy: bool) -> None:
        self._fit_progress_label.setVisible(busy)
        if not busy:
            self._fit_progress_label.setText("")

        self._range_selector.setEnabled(not busy)
        self._param_table.setEnabled(not busy)
        if hasattr(self, "_add_range_btn"):
            self._add_range_btn.setEnabled(not busy)
        if hasattr(self, "_remove_fit_btn"):
            self._remove_fit_btn.setEnabled(not busy)

        for button in self._buttons.buttons():
            button.setEnabled(not busy)

        for widgets in self._range_widgets:
            widgets.active.setEnabled(not busy)
            widgets.x_min.setEnabled(not busy)
            widgets.x_max.setEnabled(not busy)
            widgets.edit_button.setEnabled(not busy)
            widgets.fit_button.setEnabled(not busy)
            widgets.remove_button.setEnabled(not busy)
