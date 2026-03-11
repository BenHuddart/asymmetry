"""Dialog for fitting parameter trends vs field/temperature."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from PySide6.QtCore import Qt
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    component_names_for_x,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_OPERATOR_OPTIONS = ["+", "-", "*", "/"]


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

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Op", "Component", "Remove"])
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 320)
        self._table.setColumnWidth(2, 100)
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

        comp_combo = QComboBox()
        comp_combo.addItems(self._component_pool)
        if component_name and component_name in self._component_pool:
            comp_combo.setCurrentText(component_name)
        comp_combo.currentTextChanged.connect(self._update_formula)
        self._table.setCellWidget(row, 1, comp_combo)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self._table.setCellWidget(row, 2, remove_btn)

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
            comp_combo = self._table.cellWidget(row, 1)
            if not isinstance(comp_combo, QComboBox):
                continue
            component_names.append(comp_combo.currentText())
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
        self._x = np.asarray(x_values, dtype=float)
        self._y = np.asarray(y_values, dtype=float)
        self._yerr = np.asarray(y_errors, dtype=float)
        self._removed = False
        self._range_widgets: list[_RangeWidgets] = []
        self._active_range_idx: int | None = None

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

        available = component_names_for_x(self._x_key)
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
            elif pname.startswith("B0") or pname.startswith("tau"):
                default_val = max(1e-6, (x_max - x_min) / 2.0)
            params.add(Parameter(name=pname, value=float(default_val)))

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
            component_pool=component_names_for_x(self._x_key),
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
            if pname in fit_range.parameters:
                old = fit_range.parameters[pname]
                new_params.add(Parameter(name=pname, value=old.value, min=old.min, max=old.max, fixed=old.fixed))
            else:
                new_params.add(Parameter(name=pname, value=float(model.param_defaults[pname])))
        fit_range.parameters = new_params
        fit_range.result = None

        self._rebuild_ranges_ui()
        self._select_range(idx)

    def _run_fit(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return

        self._commit_param_table()
        fit_range = self._fit.ranges[idx]

        if fit_range.x_max is not None and fit_range.x_min is not None and fit_range.x_max <= fit_range.x_min:
            _show_warning(self, "Invalid range", "x max must be greater than x min.")
            return

        result = fit_parameter_model(
            x=self._x,
            y=self._y,
            yerr=self._yerr,
            model=fit_range.model,
            parameters=fit_range.parameters,
            x_min=fit_range.x_min,
            x_max=fit_range.x_max,
        )
        fit_range.result = result
        if result.success:
            fit_range.parameters = result.parameters

        self._select_range(idx)

        if result.success:
            _show_info(
                self,
                "Fit complete",
                f"Range {idx + 1} fit succeeded. Reduced chi2 = {result.reduced_chi_squared:.4g}",
            )
        else:
            _show_warning(self, "Fit failed", result.message or "Model fit failed")

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
            self._param_table.setItem(row, 0, QTableWidgetItem(param.name))
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

    def _commit_param_table(self) -> None:
        if self._active_range_idx is None:
            return

        fit_range = self._fit.ranges[self._active_range_idx]
        new_params = ParameterSet()
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            value_item = self._param_table.item(row, 1)
            min_item = self._param_table.item(row, 2)
            max_item = self._param_table.item(row, 3)
            fixed_widget = self._param_table.cellWidget(row, 4)

            if name_item is None or value_item is None:
                continue

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

            fixed = False
            if fixed_widget is not None and fixed_widget.layout() is not None and fixed_widget.layout().count() > 0:
                inner = fixed_widget.layout().itemAt(0).widget()
                if isinstance(inner, QCheckBox):
                    fixed = inner.isChecked()

            new_params.add(Parameter(name=name, value=value, min=p_min, max=p_max, fixed=fixed))

        fit_range.parameters = new_params

    def _on_remove_fit(self) -> None:
        self._removed = True
        self.accept()
