"""Fit panel — model selection, parameter table, and fit controls.

Mirrors WiMDA's Analyse → Fit dialog: choose a model, set initial
parameters, run the fit, and inspect results.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog


def _format_param_label(name: str) -> str:
    """Return a display label with Greek symbols and units where applicable."""
    return get_param_info(name).unicode_label()


def _configure_formula_label(label: QLabel) -> None:
    """Configure formula labels to wrap without clipping."""
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
    line_height = label.fontMetrics().lineSpacing()
    label.setMinimumHeight(line_height * 2 + 8)


class GlobalFitWorker(QObject):
    """Worker for running global fits in a background thread.

    Signals
    -------
    finished : Signal(object, object)
        Emitted with (results_dict, fitted_global) when fit completes successfully.
    error : Signal(str)
        Emitted with error message if fit fails.
    """
    # Use object/object for cross-thread payloads containing Python objects
    # (FitResult, ParameterSet, numpy arrays). Typed Qt containers can trigger
    # conversion errors when queued between threads.
    finished = Signal(object, object)  # results_dict, fitted_global
    error = Signal(str)

    def __init__(self, fit_engine, datasets, model_fn, global_params, local_params, initial_params):
        super().__init__()
        self.fit_engine = fit_engine
        self.datasets = datasets
        self.model_fn = model_fn
        self.global_params = global_params
        self.local_params = local_params
        self.initial_params = initial_params

    def run(self):
        """Execute the global fit."""
        try:
            results_dict, fitted_global = self.fit_engine.global_fit(
                self.datasets,
                self.model_fn,
                self.global_params,
                self.local_params,
                self.initial_params,
            )
            self.finished.emit(results_dict, fitted_global)
        except Exception as e:
            self.error.emit(str(e))


class SingleFitTab(QWidget):
    """Single dataset fitting interface.

    Provides model selection, parameter configuration, and fit execution for a
    single dataset. Emits signals when fit completes successfully.

    Attributes
    ----------
    fit_completed : Signal
        Emitted with (FitResult, tuple, list) when fit finishes successfully.
        The tuple contains (t_fit, y_fit) arrays for plotting the fit curve,
        and the list contains per-component additive curves as
        (component_name, y_component).

    Methods
    -------
    set_dataset(dataset)
        Set the current dataset to fit.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._current_dataset: MuonDataset | None = None
        self._fit_engine = FitEngine()
        self._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self._formula_label = QLabel()
        _configure_formula_label(self._formula_label)
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        model_layout.addRow("A(t):", self._formula_label)
        model_layout.addRow("", self._edit_model_btn)
        layout.addWidget(model_group)

        # Parameter table
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        self._param_table = QTableWidget(0, 5)
        self._param_table.setHorizontalHeaderLabels(["Name", "Value", "Fix", "Min", "Max"])
        self._param_table.horizontalHeader().setStretchLastSection(False)
        self._param_table.setColumnWidth(0, 80)   # Name
        self._param_table.setColumnWidth(1, 100)  # Value
        self._param_table.setColumnWidth(2, 40)   # Fix
        self._param_table.setColumnWidth(3, 80)   # Min
        self._param_table.setColumnWidth(4, 80)   # Max
        param_layout.addWidget(self._param_table)
        layout.addWidget(param_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.clicked.connect(self._run_fit)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._reset_parameters)
        btn_layout.addWidget(self._fit_btn)
        btn_layout.addWidget(self._reset_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Results
        results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(results_group)
        self._result_label = QLabel("No fit performed yet")
        self._result_label.setWordWrap(True)
        results_layout.addWidget(self._result_label)
        layout.addWidget(results_group)

        layout.addStretch()

        self._set_composite_model(self._composite_model)

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset to fit."""
        self._current_dataset = dataset
        self._fit_btn.setEnabled(dataset is not None)

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild the parameter table."""
        self._composite_model = model
        self._formula_label.setText(model.formula_string())

        self._param_table.setRowCount(len(model.param_names))
        for i, pname in enumerate(model.param_names):
            # Name column (read-only)
            name_item = QTableWidgetItem(_format_param_label(pname))
            name_item.setData(Qt.ItemDataRole.UserRole, pname)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(i, 0, name_item)

            # Value column
            value_item = QTableWidgetItem(str(model.param_defaults.get(pname, 0.0)))
            self._param_table.setItem(i, 1, value_item)

            # Fix checkbox column
            fix_widget = QWidget()
            fix_layout = QHBoxLayout(fix_widget)
            fix_layout.setContentsMargins(0, 0, 0, 0)
            fix_checkbox = QCheckBox()
            fix_layout.addWidget(fix_checkbox)
            fix_layout.setAlignment(fix_checkbox, Qt.AlignmentFlag.AlignCenter)
            self._param_table.setCellWidget(i, 2, fix_widget)

            # Min column
            min_item = QTableWidgetItem("-inf")
            self._param_table.setItem(i, 3, min_item)

            # Max column
            max_item = QTableWidgetItem("inf")
            self._param_table.setItem(i, 4, max_item)

    def _edit_function(self) -> None:
        """Launch the fit-function builder dialog."""
        dialog = FitFunctionBuilderDialog(self, initial_model=self._composite_model)
        if dialog.exec():
            new_model = dialog.get_composite_model()
            if new_model is not None:
                self._set_composite_model(new_model)

    def _reset_parameters(self) -> None:
        """Reset parameters to model defaults."""
        self._set_composite_model(self._composite_model)

    def _run_fit(self) -> None:
        """Execute the fit."""
        if self._current_dataset is None:
            self._result_label.setText("ERROR: No dataset selected")
            return

        if self._composite_model is None:
            self._result_label.setText("ERROR: No function defined")
            return

        # Build parameter set from table
        parameters = ParameterSet()
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            # Parse value
            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError):
                self._result_label.setText(
                    f"ERROR: Invalid value for {_format_param_label(param_name)}"
                )
                return

            # Check if fixed
            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox)
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            # Parse bounds
            try:
                min_text = self._param_table.item(i, 3).text()
                min_val = float(min_text) if min_text and min_text != "-inf" else -float("inf")
            except (ValueError, AttributeError):
                min_val = -float("inf")

            try:
                max_text = self._param_table.item(i, 4).text()
                max_val = float(max_text) if max_text and max_text != "inf" else float("inf")
            except (ValueError, AttributeError):
                max_val = float("inf")

            param = Parameter(
                name=param_name,
                value=value,
                min=min_val,
                max=max_val,
                fixed=fixed,
            )
            parameters.add(param)

        # Run the fit
        self._result_label.setText("Fitting...")
        try:
            result = self._fit_engine.fit(
                self._current_dataset,
                self._composite_model.function,
                parameters,
            )
        except Exception as e:
            self._result_label.setText(f"<b>Error during fit:</b><br>{str(e)}")
            return

        # Update results display
        if result.success:
            lines = [
                f"<b>χ² = {result.chi_squared:.4f}</b>",
                f"<b>χ²ᵣ = {result.reduced_chi_squared:.4f}</b>",
                "<br><b>Parameters:</b>",
            ]
            for param in result.parameters:
                unc = result.uncertainties.get(param.name, 0.0)
                lines.append(
                    f"  {_format_param_label(param.name)} = {param.value:.6f} ± {unc:.6f}"
                )
            self._result_label.setText("<br>".join(lines))

            # Update table with fit results
            for i in range(self._param_table.rowCount()):
                name_item = self._param_table.item(i, 0)
                param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(param_name, str):
                    param_name = name_item.text() if name_item else ""
                if param_name in result.parameters:
                    fitted_value = result.parameters[param_name].value
                    self._param_table.item(i, 1).setText(f"{fitted_value:.6f}")

            # Generate fitted curve for plotting
            import numpy as np
            t_fit = np.linspace(
                self._current_dataset.time.min(),
                self._current_dataset.time.max(),
                500,
            )
            param_dict = {p.name: p.value for p in result.parameters}
            y_fit = self._composite_model.function(t_fit, **param_dict)

            component_curves = self._composite_model.evaluate_components(
                t_fit,
                additive_only=True,
                **param_dict,
            )
            self.fit_completed.emit(result, (t_fit, y_fit), component_curves)
        else:
            self._result_label.setText(f"<b>Fit failed:</b> {result.message}")

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the single-fit tab state."""
        params = []
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = (
                name_item.data(Qt.ItemDataRole.UserRole)
                if name_item
                else f"param_{i}"
            )
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            value_item = self._param_table.item(i, 1)
            try:
                value = float(value_item.text()) if value_item else 0.0
            except ValueError:
                value = 0.0

            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            min_item = self._param_table.item(i, 3)
            max_item = self._param_table.item(i, 4)
            params.append({
                "name": param_name,
                "value": value,
                "fixed": fixed,
                "min": min_item.text() if min_item else "-inf",
                "max": max_item.text() if max_item else "inf",
            })

        return {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            "parameters": params,
            "result_html": self._result_label.text(),
        }

    def restore_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        composite_data = state.get("composite_model")
        if isinstance(composite_data, dict):
            try:
                self._set_composite_model(CompositeModel.from_dict(composite_data))
            except ValueError:
                self._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

        params_data = {p["name"]: p for p in state.get("parameters", [])}
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = (
                name_item.data(Qt.ItemDataRole.UserRole)
                if name_item
                else None
            )
            if not isinstance(param_name, str) and name_item:
                param_name = name_item.text()
            if param_name not in params_data:
                continue

            p_data = params_data[param_name]

            value_item = self._param_table.item(i, 1)
            if value_item:
                value_item.setText(str(p_data.get("value", 0.0)))

            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            if fix_checkbox:
                fix_checkbox.setChecked(bool(p_data.get("fixed", False)))

            min_item = self._param_table.item(i, 3)
            if min_item:
                min_item.setText(str(p_data.get("min", "-inf")))

            max_item = self._param_table.item(i, 4)
            if max_item:
                max_item.setText(str(p_data.get("max", "inf")))

        result_html = state.get("result_html")
        if isinstance(result_html, str) and result_html:
            self._result_label.setText(result_html)


class GlobalFitTab(QWidget):
    """Global fitting interface for simultaneous multi-dataset fitting.

    Allows user to specify which parameters are global (shared), local (vary per dataset),
    or fixed across all datasets in the workspace.

    Signals
    -------
    global_fit_completed : Signal(dict, ParameterSet)
        Emitted with (results_dict, global_params) when global fit completes.
        results_dict maps run_number -> (FitResult, fitted_curve_tuple).
    """

    # Use object/object to avoid Qt container coercion (which can alter key types).
    global_fit_completed = Signal(object, object)  # (results_dict, global_params)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._fit_engine = FitEngine()
        self._datasets = []  # Will be set by parent
        self._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self._formula_label = QLabel()
        _configure_formula_label(self._formula_label)
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        model_layout.addRow("A(t):", self._formula_label)
        model_layout.addRow("", self._edit_model_btn)
        layout.addWidget(model_group)

        # Parameter classification table
        param_group = QGroupBox("Parameter Classification")
        param_layout = QVBoxLayout(param_group)

        info_label = QLabel(
            "Specify how each parameter behaves across datasets:\n"
            "• Global: Same value for all datasets\n"
            "• Local: Different value for each dataset\n"
            "• Fixed: Held constant at the specified value"
        )
        info_label.setWordWrap(True)
        param_layout.addWidget(info_label)

        self._param_table = QTableWidget(0, 4)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Type", "Bounds"])
        self._param_table.horizontalHeader().setStretchLastSection(False)
        self._param_table.setColumnWidth(0, 80)   # Parameter name
        self._param_table.setColumnWidth(1, 80)   # Initial value
        self._param_table.setColumnWidth(2, 90)   # Type (dropdown)
        self._param_table.setColumnWidth(3, 150)  # Bounds
        param_layout.addWidget(self._param_table)
        layout.addWidget(param_group)

        # Fit button
        btn_layout = QHBoxLayout()
        self._fit_btn = QPushButton("Run Global Fit")
        self._fit_btn.clicked.connect(self._run_global_fit)
        self._fit_btn.setEnabled(False)
        btn_layout.addWidget(self._fit_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Results display
        results_group = QGroupBox("Global Fit Results")
        results_layout = QVBoxLayout(results_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(200)
        self._result_text.setText("No fit performed yet")
        results_layout.addWidget(self._result_text)
        layout.addWidget(results_group)

        layout.addStretch()

        # Thread management for non-blocking fits
        self._fit_thread: QThread | None = None
        self._fit_worker: GlobalFitWorker | None = None

        self._set_composite_model(self._composite_model)

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for global fitting."""
        self._datasets = datasets
        n = len(datasets)
        self._fit_btn.setEnabled(n > 1)
        if n == 0:
            self._result_text.setText(
                "No datasets selected.\n"
                "Select datasets in the browser to run a global fit."
            )
        elif n == 1:
            self._result_text.setText(
                "Global fitting requires at least 2 datasets.\n"
                "Currently have 1 selected dataset."
            )
        else:
            self._result_text.setText(
                f"{n} datasets selected. "
                "Configure parameters and click Run Global Fit."
            )

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild classification rows."""
        self._composite_model = model
        self._formula_label.setText(model.formula_string())

        self._param_table.setRowCount(len(model.param_names))
        for i, pname in enumerate(model.param_names):
            # Parameter name
            name_item = QTableWidgetItem(_format_param_label(pname))
            name_item.setData(Qt.ItemDataRole.UserRole, pname)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(i, 0, name_item)

            # Initial value
            value_item = QTableWidgetItem(str(model.param_defaults.get(pname, 0.0)))
            self._param_table.setItem(i, 1, value_item)

            # Type selection (Global/Local/Fixed dropdown)
            type_combo = QComboBox()
            type_combo.addItems(["Global", "Local", "Fixed"])
            # Set default: first parameter (usually amplitude) as Global, others as Local
            type_combo.setCurrentText("Global" if i == 0 else "Local")
            self._param_table.setCellWidget(i, 2, type_combo)

            # Bounds (min, max)
            bounds_item = QTableWidgetItem("-inf, inf")
            self._param_table.setItem(i, 3, bounds_item)

    def _edit_function(self) -> None:
        """Launch the fit-function builder dialog."""
        dialog = FitFunctionBuilderDialog(self, initial_model=self._composite_model)
        if dialog.exec():
            new_model = dialog.get_composite_model()
            if new_model is not None:
                self._set_composite_model(new_model)

    def _run_global_fit(self) -> None:
        """Execute global fit on all datasets."""
        if len(self._datasets) < 2:
            self._result_text.setText("Error: Need at least 2 datasets for global fitting")
            return

        if self._composite_model is None:
            self._result_text.setText("Error: No function defined")
            return
        model = self._composite_model

        # Parse parameter classification
        global_params = []
        local_params = []
        fixed_params = {}
        param_values = {}
        param_bounds = {}

        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else f"param_{i}"

            # Parse value
            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError):
                self._result_text.setText(
                    f"Error: Invalid value for {_format_param_label(pname)}"
                )
                return

            # Validate value is finite
            if not np.isfinite(value):
                self._result_text.setText(
                    f"Error: Parameter {_format_param_label(pname)} must be finite, got {value}"
                )
                return

            param_values[pname] = value

            # Parse bounds
            bounds_text = self._param_table.item(i, 3).text()
            try:
                parts = bounds_text.split(",")
                lo = parts[0].strip()
                hi = parts[1].strip()
                min_val = float(lo) if lo != "-inf" else -float("inf")
                max_val = float(hi) if hi != "inf" else float("inf")
            except (ValueError, IndexError):
                min_val, max_val = -float("inf"), float("inf")

            # Validate bounds
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val > max_val:
                self._result_text.setText(
                    f"Error: Parameter {_format_param_label(pname)}"
                    f" has invalid bounds: {min_val} > {max_val}"
                )
                return

            # Check value is within bounds
            if np.isfinite(min_val) and value < min_val:
                self._result_text.setText(
                    f"Error: Parameter {_format_param_label(pname)}"
                    f" value {value} is below minimum {min_val}"
                )
                return
            if np.isfinite(max_val) and value > max_val:
                self._result_text.setText(
                    f"Error: Parameter {_format_param_label(pname)}"
                    f" value {value} is above maximum {max_val}"
                )
                return

            param_bounds[pname] = (min_val, max_val)

            # Check which type is selected
            type_combo = self._param_table.cellWidget(i, 2)
            type_text = type_combo.currentText() if isinstance(type_combo, QComboBox) else "Local"

            if type_text == "Global":
                global_params.append(pname)
            elif type_text == "Local":
                local_params.append(pname)
            else:  # Fixed
                fixed_params[pname] = value

        # Build initial parameter sets for each dataset
        initial_params = {}
        for ds in self._datasets:
            params = ParameterSet()
            for pname in model.param_names:
                min_val, max_val = param_bounds[pname]
                value = param_values[pname]
                fixed = pname in fixed_params
                params.add(Parameter(
                    name=pname,
                    value=value,
                    min=min_val,
                    max=max_val,
                    fixed=fixed,
                ))
            initial_params[ds.run_number] = params

        # Run global fit in background thread
        self._result_text.setText("Fitting... This may take a moment for many datasets...")
        self._fit_btn.setEnabled(False)  # Disable button during fit

        # Clean up any existing thread
        if self._fit_thread is not None:
            self._fit_thread.quit()
            self._fit_thread.wait()

        # Create worker and thread
        self._fit_thread = QThread()
        self._fit_worker = GlobalFitWorker(
            self._fit_engine,
            self._datasets,
            self._composite_model.function,
            global_params,
            local_params,
            initial_params,
        )
        self._fit_worker.moveToThread(self._fit_thread)

        # Store model for later use in callbacks
        self._current_model = self._composite_model
        self._current_global_params = global_params

        # Connect signals
        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.finished.connect(self._fit_thread.quit)
        self._fit_worker.error.connect(self._fit_thread.quit)
        self._fit_thread.finished.connect(self._cleanup_thread)

        # Start the thread
        self._fit_thread.start()

    def _on_fit_finished(self, results_dict: dict, fitted_global: list) -> None:
        """Handle successful fit completion."""
        self._fit_btn.setEnabled(True)

        model = self._current_model
        global_params = self._current_global_params

        # Display results
        if all(r.success for r in results_dict.values()):
            # Build results summary
            lines = ["<b>Global Fit Successful!</b><br>"]

            # Global parameters
            lines.append("<b>Global Parameters:</b>")
            for p in fitted_global:
                if p.name in global_params:
                    # Find uncertainty from first dataset result
                    # (global params have same uncertainty)
                    first_result = next(iter(results_dict.values()))
                    unc = first_result.uncertainties.get(p.name, 0.0)
                    lines.append(
                        f"  {_format_param_label(p.name)} = {p.value:.6f} ± {unc:.6f}"
                    )

            # Summary statistics
            total_chi2 = sum(
                r.chi_squared for r in results_dict.values()
            )
            avg_red_chi2 = sum(
                r.reduced_chi_squared for r in results_dict.values()
            ) / len(results_dict)
            lines.append(f"<br><b>Total χ² = {total_chi2:.2f}</b>")
            lines.append(f"<b>Average χ²ᵣ = {avg_red_chi2:.3f}</b>")
            lines.append(f"<br>Fitted {len(results_dict)} datasets")

            self._result_text.setHtml("<br>".join(lines))

            # Generate fitted curves for all datasets
            import numpy as np
            results_with_curves = {}
            for ds in self._datasets:
                result = results_dict[ds.run_number]
                t_fit = np.linspace(ds.time.min(), ds.time.max(), 500)
                param_dict = {p.name: p.value for p in result.parameters}
                y_fit = model.function(t_fit, **param_dict)
                component_curves = model.evaluate_components(
                    t_fit,
                    additive_only=True,
                    **param_dict,
                )
                results_with_curves[ds.run_number] = (
                    result,
                    (t_fit, y_fit),
                    component_curves,
                )

            # Emit signal with all results
            self.global_fit_completed.emit(results_with_curves, fitted_global)
        else:
            failed = [run for run, r in results_dict.items() if not r.success]
            run_label_by_number = {ds.run_number: ds.run_label for ds in self._datasets}
            failed_labels = [run_label_by_number.get(run, str(run)) for run in failed]
            self._result_text.setText(
                f"<b>Global fit failed</b><br>"
                f"Failed datasets: {failed_labels}"
            )

    def _on_fit_error(self, error_msg: str) -> None:
        """Handle fit error."""
        self._fit_btn.setEnabled(True)
        self._result_text.setText(f"<b>Error during global fit:</b><br>{error_msg}")

    def _cleanup_thread(self) -> None:
        """Clean up thread resources."""
        if self._fit_thread is not None:
            self._fit_thread.deleteLater()
            self._fit_thread = None
        if self._fit_worker is not None:
            self._fit_worker.deleteLater()
            self._fit_worker = None

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the global-fit tab state."""
        params = []
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = (
                name_item.data(Qt.ItemDataRole.UserRole)
                if name_item
                else f"param_{i}"
            )
            if not isinstance(param_name, str) and name_item:
                param_name = name_item.text()

            value_item = self._param_table.item(i, 1)
            try:
                value = float(value_item.text()) if value_item else 0.0
            except ValueError:
                value = 0.0

            type_combo = self._param_table.cellWidget(i, 2)
            type_text = (
                type_combo.currentText()
                if isinstance(type_combo, QComboBox)
                else "Local"
            )

            bounds_item = self._param_table.item(i, 3)
            bounds_text = bounds_item.text() if bounds_item else "-inf, inf"

            params.append({
                "name": param_name,
                "value": value,
                "type": type_text,
                "bounds": bounds_text,
            })

        return {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            "parameters": params,
            "result_html": self._result_text.toHtml(),
        }

    def restore_state(self, state: dict) -> None:
        """Restore global-fit tab state from a saved dict."""
        composite_data = state.get("composite_model")
        if isinstance(composite_data, dict):
            try:
                self._set_composite_model(CompositeModel.from_dict(composite_data))
            except ValueError:
                self._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

        params_data = {p["name"]: p for p in state.get("parameters", [])}
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = (
                name_item.data(Qt.ItemDataRole.UserRole)
                if name_item
                else None
            )
            if not isinstance(param_name, str) and name_item:
                param_name = name_item.text()
            if param_name not in params_data:
                continue

            p_data = params_data[param_name]

            value_item = self._param_table.item(i, 1)
            if value_item:
                value_item.setText(str(p_data.get("value", 0.0)))

            type_combo = self._param_table.cellWidget(i, 2)
            if isinstance(type_combo, QComboBox):
                type_value = str(p_data.get("type") or "Local")
                idx = type_combo.findText(type_value)
                if idx >= 0:
                    type_combo.setCurrentIndex(idx)

            bounds_item = self._param_table.item(i, 3)
            if bounds_item:
                bounds_item.setText(p_data.get("bounds", "-inf, inf"))

        result_html = state.get("result_html")
        if isinstance(result_html, str) and result_html:
            self._result_text.setHtml(result_html)


class FitPanel(QWidget):
    """Fit setup and results panel with tabbed interface.

    Contains tabs for single dataset fitting and global (multi-dataset) fitting.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    # Keep payload generic to preserve Python dict key/value types end-to-end.
    global_fit_completed = Signal(object, object)  # (results_dict, global_params)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget
        self._tabs = QTabWidget()

        # Single fit tab
        self._single_tab = SingleFitTab()
        self._single_tab.fit_completed.connect(self.fit_completed.emit)
        self._tabs.addTab(self._single_tab, "Single")

        # Global fit tab
        self._global_tab = GlobalFitTab()
        self._global_tab.global_fit_completed.connect(self.global_fit_completed.emit)
        self._tabs.addTab(self._global_tab, "Global")

        layout.addWidget(self._tabs)

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset for single fitting tab."""
        self._single_tab.set_dataset(dataset)

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for global fitting tab."""
        self._global_tab.set_datasets(datasets)

    # ── project state helpers ──────────────────────────────────────────

    def get_single_state(self) -> dict:
        """Return serialisable state of the single-fit tab."""
        return self._single_tab.get_state()

    def restore_single_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        self._single_tab.restore_state(state)

    def get_global_state(self) -> dict:
        """Return serialisable state of the global-fit tab."""
        return self._global_tab.get_state()

    def restore_global_state(self, state: dict) -> None:
        """Restore global-fit tab state from a saved dict."""
        self._global_tab.restore_state(state)

    def get_ui_state(self) -> dict:
        """Return serialisable UI state for the fit panel container."""
        return {"active_tab_index": int(self._tabs.currentIndex())}

    def restore_ui_state(self, state: dict) -> None:
        """Restore serialisable UI state for the fit panel container."""
        index = state.get("active_tab_index")
        if isinstance(index, int) and 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)
