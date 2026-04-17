"""Fit panel — model selection, parameter table, and fit controls.

Mirrors WiMDA's Analyse → Fit dialog: choose a model, set initial
parameters, run the fit, and inspect results.
"""

from __future__ import annotations

import copy
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
    QMessageBox,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    deserialize_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.fit_wizard import CandidateAssessment, FitWizardRecommendation
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow


def _format_param_label(name: str) -> str:
    """Return a display label with Greek symbols and units where applicable."""
    return get_param_info(name).unicode_label()


def _field_value_overrides(model: CompositeModel, field_gauss: float) -> dict[str, float]:
    """Return a dict overriding ``field`` parameter defaults with *field_gauss*.

    Only overrides parameters whose base name is ``"field"`` and only when
    *field_gauss* is non-zero.
    """
    if field_gauss == 0.0:
        return {}
    overrides: dict[str, float] = {}
    for pname in model.param_names:
        base_name, _index = split_parameter_name(pname)
        if base_name == "field":
            overrides[pname] = field_gauss
    return overrides


def _configure_formula_label(label: QLabel) -> None:
    """Configure formula labels to wrap without clipping."""
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
    line_height = label.fontMetrics().lineSpacing()
    label.setMinimumHeight(line_height * 2 + 8)


def _format_bounds_pair(min_val: float, max_val: float) -> str:
    def _format(value: float) -> str:
        if value == float("inf"):
            return "inf"
        if value == -float("inf"):
            return "-inf"
        return f"{float(value):.6g}"

    return f"{_format(min_val)}, {_format(max_val)}"


def _fit_curve_sample_count(
    model: CompositeModel,
    param_values: dict[str, float],
    t_min: float,
    t_max: float,
    *,
    base_points: int = 500,
    points_per_cycle: int = 40,
    max_points: int = 20000,
) -> int:
    """Return a dense-enough sample count for plotting oscillatory models."""
    duration = max(float(t_max) - float(t_min), 0.0)
    if duration <= 0.0:
        return base_points

    max_frequency_mhz = 0.0
    for name, value in param_values.items():
        base_name, _index = split_parameter_name(name)
        try:
            numeric_value = abs(float(value))
        except (TypeError, ValueError):
            continue

        if base_name == "frequency":
            max_frequency_mhz = max(max_frequency_mhz, numeric_value)
        elif base_name == "field":
            field_frequency = (
                MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * numeric_value
            )
            max_frequency_mhz = max(max_frequency_mhz, field_frequency)

    if max_frequency_mhz <= 0.0:
        return base_points

    cycles = max_frequency_mhz * duration
    required_points = int(np.ceil(cycles * points_per_cycle)) + 1
    return int(max(base_points, min(max_points, required_points)))


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
    preview_requested = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    share_function_with_group_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._current_dataset: MuonDataset | None = None
        self._fit_blocked = False
        self._fit_block_reason = ""
        self._fit_engine = FitEngine()
        self._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
        self._fit_wizard_window: FitWizardWindow | None = None

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self._formula_label = QLabel()
        _configure_formula_label(self._formula_label)
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        self._fit_wizard_btn = QPushButton("Fit Wizard...")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)
        self._share_group_btn = QPushButton("Share Function With Data Group")
        self._share_group_btn.clicked.connect(self._on_share_function_with_group)
        self._share_group_btn.setEnabled(False)
        
        # Button row for Edit, Wizard, and Share
        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(self._edit_model_btn)
        model_button_layout.addWidget(self._fit_wizard_btn)
        model_button_layout.addWidget(self._share_group_btn)
        model_button_layout.addStretch()
        
        model_layout.addRow("A(t):", self._formula_label)
        model_layout.addRow("", model_button_layout)
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
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        self._preview_btn.setEnabled(False)
        btn_layout.addWidget(self._fit_btn)
        btn_layout.addWidget(self._reset_btn)
        btn_layout.addWidget(self._preview_btn)
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
        enabled = dataset is not None and (not self._fit_blocked)
        self._fit_btn.setEnabled(enabled)
        self._preview_btn.setEnabled(enabled)
        self._fit_wizard_btn.setEnabled(enabled)
        self._share_group_btn.setEnabled(dataset is not None)

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Enable/disable single-fit actions while preserving the active dataset."""
        self._fit_blocked = bool(blocked)
        self._fit_block_reason = str(reason)
        enabled = self._current_dataset is not None and (not self._fit_blocked)
        self._fit_btn.setEnabled(enabled)
        self._preview_btn.setEnabled(enabled)
        self._fit_wizard_btn.setEnabled(enabled)
        tooltip = self._fit_block_reason if self._fit_blocked else ""
        self._fit_btn.setToolTip(tooltip)
        self._preview_btn.setToolTip(tooltip)
        self._fit_wizard_btn.setToolTip(tooltip)

    def _on_share_function_with_group(self) -> None:
        """Request sharing the active single-fit function with the current data group."""
        if self._current_dataset is None:
            return
        try:
            run_number = int(self._current_dataset.run_number)
        except (TypeError, ValueError):
            return
        self.share_function_with_group_requested.emit(run_number)

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild the parameter table."""
        self._composite_model = model
        self._formula_label.setText(model.formula_string())

        dataset_field = (
            self._current_dataset.run.field
            if self._current_dataset is not None and self._current_dataset.run is not None
            else 0.0
        )
        field_overrides = _field_value_overrides(model, dataset_field)

        self._param_table.setRowCount(len(model.param_names))
        for i, pname in enumerate(model.param_names):
            # Name column (read-only)
            name_item = QTableWidgetItem(_format_param_label(pname))
            name_item.setData(Qt.ItemDataRole.UserRole, pname)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(i, 0, name_item)

            # Value column — use dataset field for 'field' parameters if available
            default_val = field_overrides.get(pname, model.param_defaults.get(pname, 0.0))
            value_item = QTableWidgetItem(str(default_val))
            self._param_table.setItem(i, 1, value_item)

            # Fix checkbox column
            fix_widget = QWidget()
            fix_layout = QHBoxLayout(fix_widget)
            fix_layout.setContentsMargins(0, 0, 0, 0)
            fix_checkbox = QCheckBox()
            if pname == "shape_factor_a":
                fix_checkbox.setChecked(True)
            fix_layout.addWidget(fix_checkbox)
            fix_layout.setAlignment(fix_checkbox, Qt.AlignmentFlag.AlignCenter)
            self._param_table.setCellWidget(i, 2, fix_widget)

            # Min column — default to 0 for physically positive-definite parameters
            default_min = get_param_info(pname).default_min
            min_text = str(default_min) if default_min is not None else "-inf"
            min_item = QTableWidgetItem(min_text)
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

    def _open_fit_wizard(self) -> None:
        """Launch or refresh the non-modal fit wizard window."""
        if self._current_dataset is None:
            QMessageBox.information(self, "Fit Wizard", "Select a dataset before opening the fit wizard.")
            return
        if self._fit_blocked:
            message = self._fit_block_reason or "Fit actions are unavailable for the current selection."
            QMessageBox.information(self, "Fit Wizard", message)
            return

        if self._fit_wizard_window is None:
            self._fit_wizard_window = FitWizardWindow(self)
            self._fit_wizard_window.apply_assessment_requested.connect(
                self._apply_fit_wizard_assessment
            )

        self._fit_wizard_window.set_analysis_context(
            self._current_dataset,
            current_model=self._composite_model,
        )
        self._fit_wizard_window.show()
        self._fit_wizard_window.raise_()
        self._fit_wizard_window.activateWindow()

    def _reset_parameters(self) -> None:
        """Reset parameters to model defaults."""
        self._set_composite_model(self._composite_model)

    def _apply_fit_wizard_assessment(
        self,
        assessment: CandidateAssessment,
        recommendation: FitWizardRecommendation,
    ) -> None:
        """Apply a fit-wizard assessment back into the single-fit tab."""
        if self._current_dataset is None:
            return
        if not isinstance(assessment, CandidateAssessment):
            return

        result = assessment.fit_result
        if not result.success:
            self._result_label.setText(f"<b>Fit Wizard failed:</b> {result.message}")
            return

        self._set_composite_model(assessment.template.model)
        fitted_by_name = {parameter.name: parameter for parameter in result.parameters}

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                continue
            fitted = fitted_by_name.get(param_name)
            if fitted is None:
                continue

            value_item = self._param_table.item(row, 1)
            if value_item is not None:
                value_item.setText(f"{fitted.value:.6f}")

            min_item = self._param_table.item(row, 3)
            if min_item is not None:
                min_item.setText("-inf" if not np.isfinite(fitted.min) else f"{fitted.min:g}")

            max_item = self._param_table.item(row, 4)
            if max_item is not None:
                max_item.setText("inf" if not np.isfinite(fitted.max) else f"{fitted.max:g}")

            fix_widget = self._param_table.cellWidget(row, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            if fix_checkbox is not None:
                fix_checkbox.setChecked(bool(fitted.fixed))

        lines = [
            f"<b>Fit Wizard — {assessment.template.title}</b>",
            f"<b>χ² = {result.chi_squared:.4f}</b>",
            f"<b>χ²ᵣ = {result.reduced_chi_squared:.4f}</b>",
            f"<b>{recommendation.metric.value} = {assessment.metric_value(recommendation.metric):.4f}</b>",
        ]
        if assessment.residual_gate_reasons:
            lines.append("<br><b>Residual warnings:</b>")
            for reason in assessment.residual_gate_reasons:
                lines.append(f"  {reason}")
        else:
            lines.append("<b>Residual gate passed</b>")
        lines.append("<br><b>Parameters:</b>")
        for parameter in result.parameters:
            unc = result.uncertainties.get(parameter.name, 0.0)
            lines.append(
                f"  {_format_param_label(parameter.name)} = {parameter.value:.6f} ± {unc:.6f}"
            )
        self._result_label.setText("<br>".join(lines))

        param_dict = {parameter.name: parameter.value for parameter in result.parameters}
        n_samples = _fit_curve_sample_count(
            self._composite_model,
            param_dict,
            float(self._current_dataset.time.min()),
            float(self._current_dataset.time.max()),
        )
        t_fit = np.linspace(
            self._current_dataset.time.min(),
            self._current_dataset.time.max(),
            n_samples,
        )
        y_fit = self._composite_model.function(t_fit, **param_dict)
        component_curves = self._composite_model.evaluate_components(
            t_fit,
            additive_only=True,
            **param_dict,
        )
        self.fit_completed.emit(result, (t_fit, y_fit), component_curves)

    def _on_preview(self) -> None:
        """Generate and emit a preview fit curve with current parameters."""
        if self._fit_blocked:
            return

        if self._current_dataset is None:
            return

        if self._composite_model is None:
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

        param_dict = {p.name: p.value for p in parameters}
        n_samples = _fit_curve_sample_count(
            self._composite_model,
            param_dict,
            float(self._current_dataset.time.min()),
            float(self._current_dataset.time.max()),
        )
        # Generate fitted curve for plotting
        t_fit = np.linspace(
            self._current_dataset.time.min(),
            self._current_dataset.time.max(),
            n_samples,
        )
        y_fit = self._composite_model.function(t_fit, **param_dict)

        component_curves = self._composite_model.evaluate_components(
            t_fit,
            additive_only=True,
            **param_dict,
        )

        # Create a dummy result object for preview (not a real fit)
        preview_result = object()
        self.preview_requested.emit(preview_result, (t_fit, y_fit), component_curves)

    def _run_fit(self) -> None:
        """Execute the fit."""
        if self._fit_blocked:
            message = self._fit_block_reason or "Fit is unavailable for the current selection."
            self._result_label.setText(f"ERROR: {message}")
            return

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

            param_dict = {p.name: p.value for p in result.parameters}
            n_samples = _fit_curve_sample_count(
                self._composite_model,
                param_dict,
                float(self._current_dataset.time.min()),
                float(self._current_dataset.time.max()),
            )

            # Generate fitted curve for plotting
            t_fit = np.linspace(
                self._current_dataset.time.min(),
                self._current_dataset.time.max(),
                n_samples,
            )
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
        self._fit_blocked = False
        self._fit_block_reason = ""
        self._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
        # Successful single-fit seeds keyed by run number.
        self._single_fit_seed_by_run: dict[int, dict[str, object]] = {}
        # Inherited seed cache for current dataset selection.
        self._inherited_seed_by_run: dict[int, dict[str, float]] = {}
        self._inherited_model_dict: dict[str, object] | None = None
        self._fit_wizard_window: GlobalFitWizardWindow | None = None
        self._cached_wizard_recommendation: GlobalFitWizardRecommendation | None = None
        self._cached_wizard_signature: dict[str, object] | None = None
        self._cached_wizard_log_text = ""
        self._fit_wizard_search_strategy = "legacy"

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self._formula_label = QLabel()
        _configure_formula_label(self._formula_label)
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        self._fit_wizard_btn = QPushButton("Global Fit Wizard...")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)
        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(self._edit_model_btn)
        model_button_layout.addWidget(self._fit_wizard_btn)
        model_button_layout.addStretch()
        model_layout.addRow("A(t):", self._formula_label)
        model_layout.addRow("", model_button_layout)
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

    def register_single_fit_seed(self, run_number: int, model: CompositeModel, fit_result: object) -> None:
        """Store successful single-fit results for later global-fit initialisation."""
        if getattr(fit_result, "success", False) is not True:
            return

        values_by_name: dict[str, float] = {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            value = getattr(param, "value", None)
            if isinstance(name, str):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    values_by_name[name] = numeric_value

        if not values_by_name:
            return

        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return

        self._single_fit_seed_by_run[run_key] = {
            "model": model.to_dict(),
            "values": values_by_name,
        }
        self._refresh_inherited_single_fit_defaults()

    def remove_single_fit_seeds(self, run_numbers: list[int] | set[int]) -> set[int]:
        """Remove stored single-fit seeds for the given runs."""
        removed: set[int] = set()
        for run_number in run_numbers:
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            if self._single_fit_seed_by_run.pop(run_key, None) is not None:
                removed.add(run_key)
        if removed:
            self._refresh_inherited_single_fit_defaults()
        return removed

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for global fitting."""
        self._datasets = datasets
        self._invalidate_wizard_cache_if_stale()
        n = len(datasets)
        self._fit_btn.setEnabled((n > 1) and (not self._fit_blocked))
        self._fit_btn.setToolTip(self._fit_block_reason if self._fit_blocked else "")
        self._fit_wizard_btn.setEnabled((n > 1) and (not self._fit_blocked))
        self._fit_wizard_btn.setToolTip(self._fit_block_reason if self._fit_blocked else "")
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
        self._refresh_inherited_single_fit_defaults()

    def _invalidate_wizard_cache_if_stale(self) -> None:
        signature = self._cached_wizard_signature
        if not isinstance(signature, dict):
            return
        cached_runs = signature.get("run_numbers")
        current_runs = [
            int(dataset.run_number)
            for dataset in self._datasets
            if getattr(dataset, "run_number", None) is not None
        ]
        if cached_runs != current_runs:
            self._cached_wizard_recommendation = None
            self._cached_wizard_signature = None
            self._cached_wizard_log_text = ""

    def _wizard_context_signature(self, parsed: dict[str, object]) -> dict[str, object]:
        search_strategy = self._fit_wizard_search_strategy
        if self._fit_wizard_window is not None and hasattr(
            self._fit_wizard_window,
            "current_search_strategy",
        ):
            search_strategy = self._fit_wizard_window.current_search_strategy()
        return {
            "run_numbers": [int(dataset.run_number) for dataset in self._datasets],
            "model": self._composite_model.to_dict(),
            "types": {str(key): str(value) for key, value in dict(parsed["types"]).items()},
            "values": {str(key): float(value) for key, value in dict(parsed["values"]).items()},
            "bounds": {
                str(key): [float(bounds[0]), float(bounds[1])]
                for key, bounds in dict(parsed["bounds"]).items()
            },
            "search_strategy": str(search_strategy),
        }

    def _cache_wizard_analysis(
        self,
        recommendation: GlobalFitWizardRecommendation,
        *,
        signature: dict[str, object],
        log_text: str = "",
    ) -> None:
        self._cached_wizard_recommendation = recommendation
        self._cached_wizard_signature = copy.deepcopy(signature)
        self._cached_wizard_log_text = str(log_text)
        self._fit_wizard_search_strategy = str(signature.get("search_strategy", "legacy"))

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Enable/disable global-fit execution while preserving selected datasets."""
        self._fit_blocked = bool(blocked)
        self._fit_block_reason = str(reason)
        self._fit_btn.setEnabled((len(self._datasets) > 1) and (not self._fit_blocked))
        self._fit_btn.setToolTip(self._fit_block_reason if self._fit_blocked else "")
        self._fit_wizard_btn.setEnabled((len(self._datasets) > 1) and (not self._fit_blocked))
        self._fit_wizard_btn.setToolTip(self._fit_block_reason if self._fit_blocked else "")

    def _refresh_inherited_single_fit_defaults(self) -> None:
        """Apply single-fit seeds when every selected dataset shares one model."""
        self._inherited_seed_by_run = {}
        self._inherited_model_dict = None

        if len(self._datasets) < 2:
            return

        run_numbers: list[int] = []
        for ds in self._datasets:
            try:
                run_numbers.append(int(ds.run_number))
            except (TypeError, ValueError):
                return

        seeds: list[dict[str, object]] = []
        for run_number in run_numbers:
            seed = self._single_fit_seed_by_run.get(run_number)
            if not isinstance(seed, dict):
                return
            seeds.append(seed)

        first_model = seeds[0].get("model")
        if not isinstance(first_model, dict):
            return
        for seed in seeds[1:]:
            if seed.get("model") != first_model:
                return

        try:
            inherited_model = CompositeModel.from_dict(first_model)
        except ValueError:
            return

        inherited_values_by_run: dict[int, dict[str, float]] = {}
        for run_number, seed in zip(run_numbers, seeds, strict=False):
            values = seed.get("values")
            if not isinstance(values, dict):
                return
            typed_values: dict[str, float] = {}
            for key, value in values.items():
                if not isinstance(key, str):
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    typed_values[key] = numeric_value
            if not typed_values:
                return
            inherited_values_by_run[run_number] = typed_values

        self._set_composite_model(inherited_model)

        averages = self._inherited_param_averages(
            inherited_values_by_run,
            inherited_model.param_names,
        )
        if averages:
            for row in range(self._param_table.rowCount()):
                name_item = self._param_table.item(row, 0)
                pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(pname, str):
                    pname = name_item.text() if name_item else ""
                if pname not in averages:
                    continue
                value_item = self._param_table.item(row, 1)
                if value_item is not None:
                    value_item.setText(f"{averages[pname]:.6g}")

        self._inherited_seed_by_run = inherited_values_by_run
        self._inherited_model_dict = inherited_model.to_dict()

    def _inherited_param_averages(
        self,
        values_by_run: dict[int, dict[str, float]],
        param_names: list[str],
    ) -> dict[str, float]:
        """Return finite means per parameter from inherited per-run seeds."""
        averages: dict[str, float] = {}
        for pname in param_names:
            vals: list[float] = []
            for values in values_by_run.values():
                value = values.get(pname)
                if value is None:
                    continue
                if np.isfinite(value):
                    vals.append(float(value))
            if vals:
                averages[pname] = float(np.mean(vals))
        return averages

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild classification rows."""
        self._composite_model = model
        self._formula_label.setText(model.formula_string())

        # Use the mean field across loaded datasets (if non-zero) as the default
        # for any 'field' parameters.
        dataset_fields = [
            ds.run.field
            for ds in self._datasets
            if ds.run is not None and ds.run.field != 0.0
        ]
        mean_field = float(np.mean(dataset_fields)) if dataset_fields else 0.0
        field_overrides = _field_value_overrides(model, mean_field)

        self._param_table.setRowCount(len(model.param_names))
        for i, pname in enumerate(model.param_names):
            # Parameter name
            name_item = QTableWidgetItem(_format_param_label(pname))
            name_item.setData(Qt.ItemDataRole.UserRole, pname)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(i, 0, name_item)

            # Initial value — use dataset field for 'field' parameters if available
            default_val = field_overrides.get(pname, model.param_defaults.get(pname, 0.0))
            value_item = QTableWidgetItem(str(default_val))
            self._param_table.setItem(i, 1, value_item)

            # Type selection (Global/Local/Fixed dropdown)
            type_combo = QComboBox()
            type_combo.addItems(["Global", "Local", "Fixed"])
            # Set default: first parameter (usually amplitude) as Global, others as Local
            type_combo.setCurrentText("Global" if i == 0 else "Local")
            self._param_table.setCellWidget(i, 2, type_combo)

            # Bounds (min, max) — default lower bound to 0 for positive-definite parameters
            default_min = get_param_info(pname).default_min
            min_text = str(default_min) if default_min is not None else "-inf"
            bounds_item = QTableWidgetItem(f"{min_text}, inf")
            self._param_table.setItem(i, 3, bounds_item)

    def _edit_function(self) -> None:
        """Launch the fit-function builder dialog."""
        dialog = FitFunctionBuilderDialog(self, initial_model=self._composite_model)
        if dialog.exec():
            new_model = dialog.get_composite_model()
            if new_model is not None:
                self._set_composite_model(new_model)

    def _open_fit_wizard(self) -> None:
        """Launch or refresh the non-modal global fit wizard window."""
        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason or "Global fit is unavailable for the current selection."
            )
            return
        if len(self._datasets) < 2:
            self._result_text.setText("Global fit wizard requires at least 2 datasets.")
            return

        try:
            parsed = self._parse_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        if self._fit_wizard_window is None:
            self._fit_wizard_window = GlobalFitWizardWindow(self)
            self._fit_wizard_window.apply_assessment_requested.connect(
                self._apply_fit_wizard_assessment
            )
            self._fit_wizard_window.analysis_cached.connect(self._on_fit_wizard_analysis_cached)
            self._fit_wizard_window.parameter_setup_applied.connect(
                self._on_fit_wizard_parameter_setup_applied
            )
        self._fit_wizard_window.set_search_strategy(self._fit_wizard_search_strategy)

        signature = self._wizard_context_signature(parsed)

        self._fit_wizard_window.set_analysis_context(
            self._datasets,
            current_model=self._composite_model,
            current_parameter_types=parsed["types"],
            current_values=parsed["values"],
            parameter_bounds=parsed["bounds"],
        )
        if (
            self._cached_wizard_recommendation is not None
            and self._wizard_base_signature_matches(
                self._cached_wizard_signature,
                signature,
            )
        ):
            self._fit_wizard_window.set_cached_recommendation(
                self._cached_wizard_recommendation,
                signature=self._cached_wizard_signature,
                log_text=self._cached_wizard_log_text,
            )
        self._fit_wizard_window.show()
        self._fit_wizard_window.raise_()
        self._fit_wizard_window.activateWindow()

    def _on_fit_wizard_analysis_cached(
        self,
        recommendation: GlobalFitWizardRecommendation,
        log_text: str,
        signature: object,
    ) -> None:
        if not isinstance(signature, dict):
            return
        self._cache_wizard_analysis(
            recommendation,
            signature=signature,
            log_text=log_text,
        )

    def _on_fit_wizard_parameter_setup_applied(self, config: object) -> None:
        if not isinstance(config, dict):
            return
        types = config.get("types")
        bounds = config.get("bounds")
        if not isinstance(types, dict) or not isinstance(bounds, dict):
            return

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else ""

            if pname in types:
                type_combo = self._param_table.cellWidget(row, 2)
                if isinstance(type_combo, QComboBox):
                    idx = type_combo.findText(str(types[pname]))
                    if idx >= 0:
                        type_combo.setCurrentIndex(idx)

            raw_bounds = bounds.get(pname)
            if not isinstance(raw_bounds, tuple | list) or len(raw_bounds) != 2:
                continue
            try:
                min_val = float(raw_bounds[0])
                max_val = float(raw_bounds[1])
            except (TypeError, ValueError):
                continue

            bounds_item = self._param_table.item(row, 3)
            if bounds_item is not None:
                bounds_item.setText(_format_bounds_pair(min_val, max_val))

            value_item = self._param_table.item(row, 1)
            if value_item is None:
                continue
            try:
                value = float(value_item.text())
            except (TypeError, ValueError):
                continue
            clipped = float(np.clip(value, min_val, max_val))
            if clipped != value:
                value_item.setText(f"{clipped:.6g}")

    def _parse_parameter_configuration(self) -> dict[str, object]:
        """Return validated parameter values, roles, and bounds from the table."""
        global_params: list[str] = []
        local_params: list[str] = []
        fixed_params: dict[str, float] = {}
        param_values: dict[str, float] = {}
        param_bounds: dict[str, tuple[float, float]] = {}
        param_types: dict[str, str] = {}

        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else f"param_{i}"

            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError):
                raise ValueError(f"Error: Invalid value for {_format_param_label(pname)}") from None

            if not np.isfinite(value):
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} must be finite, got {value}"
                )
            param_values[pname] = value

            bounds_text = self._param_table.item(i, 3).text()
            try:
                parts = bounds_text.split(",")
                lo = parts[0].strip()
                hi = parts[1].strip()
                min_val = float(lo) if lo != "-inf" else -float("inf")
                max_val = float(hi) if hi != "inf" else float("inf")
            except (ValueError, IndexError):
                min_val, max_val = -float("inf"), float("inf")

            if np.isfinite(min_val) and np.isfinite(max_val) and min_val > max_val:
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} has invalid bounds: {min_val} > {max_val}"
                )
            if np.isfinite(min_val) and value < min_val:
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} value {value} is below minimum {min_val}"
                )
            if np.isfinite(max_val) and value > max_val:
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} value {value} is above maximum {max_val}"
                )

            param_bounds[pname] = (min_val, max_val)
            type_combo = self._param_table.cellWidget(i, 2)
            type_text = type_combo.currentText() if isinstance(type_combo, QComboBox) else "Local"
            param_types[pname] = type_text

            if type_text == "Global":
                global_params.append(pname)
            elif type_text == "Local":
                local_params.append(pname)
            else:
                fixed_params[pname] = value

        return {
            "global": global_params,
            "local": local_params,
            "fixed": fixed_params,
            "values": param_values,
            "bounds": param_bounds,
            "types": param_types,
        }

    def _wizard_base_signature_matches(
        self,
        cached_signature: dict[str, object] | None,
        base_signature: dict[str, object],
    ) -> bool:
        if not isinstance(cached_signature, dict):
            return False
        for key in ("run_numbers", "model", "values", "search_strategy"):
            if cached_signature.get(key) != base_signature.get(key):
                return False
        cached_types = cached_signature.get("types")
        base_types = base_signature.get("types")
        if not isinstance(cached_types, dict) or cached_types != base_types:
            return False
        cached_bounds = cached_signature.get("bounds")
        base_bounds = base_signature.get("bounds")
        if not isinstance(cached_bounds, dict):
            return False
        for name, bounds in base_bounds.items():
            if cached_bounds.get(name) != bounds:
                return False
        return True

    def _run_global_fit(self) -> None:
        """Execute global fit on all datasets."""
        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason or "Global fit is unavailable for the current selection."
            )
            return

        if len(self._datasets) < 2:
            self._result_text.setText("Error: Need at least 2 datasets for global fitting")
            return

        if self._composite_model is None:
            self._result_text.setText("Error: No function defined")
            return
        model = self._composite_model

        try:
            parsed = self._parse_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        inherited_seed_by_run: dict[int, dict[str, float]] = {}
        inherited_averages: dict[str, float] = {}
        if self._inherited_model_dict == model.to_dict() and self._inherited_seed_by_run:
            selected_runs = {int(ds.run_number) for ds in self._datasets}
            if selected_runs.issubset(self._inherited_seed_by_run):
                inherited_seed_by_run = {
                    run_number: self._inherited_seed_by_run[run_number]
                    for run_number in selected_runs
                }
                inherited_averages = self._inherited_param_averages(
                    inherited_seed_by_run,
                    model.param_names,
                )
        global_params = list(parsed["global"])
        local_params = list(parsed["local"])
        fixed_params = dict(parsed["fixed"])
        param_values = dict(parsed["values"])
        param_bounds = dict(parsed["bounds"])

        # Build initial parameter sets for each dataset
        initial_params = {}
        for ds in self._datasets:
            run_number = int(ds.run_number)
            local_seed_values = inherited_seed_by_run.get(run_number, {})
            params = ParameterSet()
            for pname in model.param_names:
                min_val, max_val = param_bounds[pname]
                value = param_values[pname]
                if inherited_seed_by_run:
                    if pname in local_params and pname in local_seed_values:
                        value = local_seed_values[pname]
                    elif pname in inherited_averages and (pname in global_params or pname in fixed_params):
                        value = inherited_averages[pname]
                fixed = pname in fixed_params
                params.add(Parameter(
                    name=pname,
                    value=value,
                    min=min_val,
                    max=max_val,
                    fixed=fixed,
                ))
            initial_params[run_number] = params

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

    def _render_global_fit_success(
        self,
        *,
        results_dict: dict[int, FitResult],
        fitted_global: ParameterSet,
        global_param_names: list[str],
    ) -> None:
        lines = ["<b>Global Fit Successful!</b><br>"]
        lines.append("<b>Global Parameters:</b>")
        for parameter in fitted_global:
            if parameter.name not in global_param_names:
                continue
            first_result = next(iter(results_dict.values()))
            unc = first_result.uncertainties.get(parameter.name, 0.0)
            lines.append(
                f"  {_format_param_label(parameter.name)} = {parameter.value:.6f} ± {unc:.6f}"
            )

        total_chi2 = sum(result.chi_squared for result in results_dict.values())
        avg_red_chi2 = sum(result.reduced_chi_squared for result in results_dict.values()) / len(results_dict)
        lines.append(f"<br><b>Total χ² = {total_chi2:.2f}</b>")
        lines.append(f"<b>Average χ²ᵣ = {avg_red_chi2:.3f}</b>")
        lines.append(f"<br>Fitted {len(results_dict)} datasets")
        self._result_text.setHtml("<br>".join(lines))

    def _results_with_curves(
        self,
        model: CompositeModel,
        results_dict: dict[int, FitResult],
    ) -> dict[int, tuple[FitResult, tuple[np.ndarray, np.ndarray], tuple[tuple[str, np.ndarray], ...]]]:
        results_with_curves = {}
        for dataset in self._datasets:
            result = results_dict[int(dataset.run_number)]
            param_dict = {parameter.name: parameter.value for parameter in result.parameters}
            n_samples = _fit_curve_sample_count(
                model,
                param_dict,
                float(dataset.time.min()),
                float(dataset.time.max()),
            )
            t_fit = np.linspace(dataset.time.min(), dataset.time.max(), n_samples)
            y_fit = model.function(t_fit, **param_dict)
            component_curves = tuple(
                model.evaluate_components(
                    t_fit,
                    additive_only=True,
                    **param_dict,
                )
            )
            results_with_curves[int(dataset.run_number)] = (
                result,
                (t_fit, y_fit),
                component_curves,
            )
        return results_with_curves

    def _emit_global_fit_success(
        self,
        *,
        model: CompositeModel,
        results_dict: dict[int, FitResult],
        fitted_global: ParameterSet,
        global_param_names: list[str],
    ) -> None:
        self._render_global_fit_success(
            results_dict=results_dict,
            fitted_global=fitted_global,
            global_param_names=global_param_names,
        )
        self.global_fit_completed.emit(
            self._results_with_curves(model, results_dict),
            fitted_global,
        )

    def _apply_fit_wizard_assessment(
        self,
        assessment: GlobalCandidateAssessment,
        recommendation: GlobalFitWizardRecommendation,
    ) -> None:
        """Apply a global-fit wizard assessment back into the global tab."""
        if not isinstance(assessment, GlobalCandidateAssessment):
            return
        if not assessment.is_successful:
            self._result_text.setText("<b>Global Fit Wizard failed</b>")
            return
        try:
            parsed = self._parse_parameter_configuration()
        except ValueError:
            parsed = None
        if parsed is not None:
            log_text = (
                self._fit_wizard_window.current_log_text()
                if self._fit_wizard_window is not None
                else self._cached_wizard_log_text
            )
            self._cache_wizard_analysis(
                recommendation,
                signature=self._wizard_context_signature(parsed),
                log_text=log_text,
            )

        self._set_composite_model(assessment.template.model)
        role_by_name = {name: "Global" for name in assessment.global_param_names}
        role_by_name.update({name: "Local" for name in assessment.local_param_names})
        role_by_name.update(
            {
                parameter.name: parameter.recommended_role
                for parameter in assessment.parameter_recommendations
            }
        )

        representative_run = self._datasets[0].run_number if self._datasets else None
        representative_result = (
            assessment.fit_results_by_run.get(int(representative_run))
            if representative_run is not None
            else None
        )
        fitted_by_name = {
            parameter.name: parameter
            for parameter in (representative_result.parameters if representative_result is not None else [])
        }

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                continue

            type_combo = self._param_table.cellWidget(row, 2)
            if isinstance(type_combo, QComboBox):
                if pname in assessment.fixed_param_names:
                    type_combo.setCurrentText("Fixed")
                else:
                    type_combo.setCurrentText(role_by_name.get(pname, "Global"))

            value_item = self._param_table.item(row, 1)
            fitted = fitted_by_name.get(pname)
            if value_item is not None and fitted is not None:
                value_item.setText(f"{fitted.value:.6g}")

            bounds_item = self._param_table.item(row, 3)
            if bounds_item is not None and fitted is not None:
                min_text = "-inf" if not np.isfinite(fitted.min) else f"{float(fitted.min):g}"
                max_text = "inf" if not np.isfinite(fitted.max) else f"{float(fitted.max):g}"
                bounds_item.setText(f"{min_text}, {max_text}")

        self._current_model = assessment.template.model
        self._current_global_params = list(assessment.global_param_names)
        self._status_text_from_global_wizard(assessment, recommendation)
        self.global_fit_completed.emit(
            {
                run_number: (
                    result,
                    assessment.fitted_curves_by_run[run_number],
                    assessment.component_curves_by_run[run_number],
                )
                for run_number, result in assessment.fit_results_by_run.items()
            },
            assessment.global_parameters,
        )

    def _status_text_from_global_wizard(
        self,
        assessment: GlobalCandidateAssessment,
        recommendation: GlobalFitWizardRecommendation,
    ) -> None:
        lines = [
            f"<b>Global Fit Wizard — {assessment.template.title}</b>",
            f"<b>{recommendation.metric.value} = {assessment.metric_value(recommendation.metric):.4f}</b>",
            f"<b>Global:</b> {', '.join(assessment.global_param_names) or 'None'}",
            f"<b>Local:</b> {', '.join(assessment.local_param_names) or 'None'}",
        ]
        if assessment.series_warnings:
            lines.append("<br><b>Warnings:</b>")
            lines.extend(f"  {warning}" for warning in assessment.series_warnings)
        self._result_text.setHtml("<br>".join(lines))

    def _on_fit_finished(self, results_dict: dict, fitted_global: list) -> None:
        """Handle successful fit completion."""
        self._fit_btn.setEnabled(True)

        model = self._current_model
        global_params = self._current_global_params

        # Display results
        if all(r.success for r in results_dict.values()):
            self._emit_global_fit_success(
                model=model,
                results_dict=results_dict,
                fitted_global=fitted_global,
                global_param_names=global_params,
            )
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
        if self._fit_wizard_window is not None:
            recommendation = self._fit_wizard_window.current_recommendation()
            signature = self._cached_wizard_signature
            if recommendation is not None and signature is None:
                try:
                    parsed = self._parse_parameter_configuration()
                except ValueError:
                    parsed = None
                if parsed is not None:
                    signature = self._wizard_context_signature(parsed)
            if recommendation is not None and signature is not None:
                self._cache_wizard_analysis(
                    recommendation,
                    signature=signature,
                    log_text=self._fit_wizard_window.current_log_text(),
                )
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

        state = {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            "parameters": params,
            "result_html": self._result_text.toHtml(),
        }
        if (
            self._cached_wizard_recommendation is not None
            and self._cached_wizard_signature is not None
        ):
            state["wizard_state"] = {
                "signature": copy.deepcopy(self._cached_wizard_signature),
                "recommendation": serialize_global_fit_wizard_recommendation(
                    self._cached_wizard_recommendation
                ),
                "log_text": self._cached_wizard_log_text,
            }
        return state

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

        wizard_state = state.get("wizard_state")
        if isinstance(wizard_state, dict):
            recommendation = deserialize_global_fit_wizard_recommendation(
                wizard_state.get("recommendation")
            )
            signature = wizard_state.get("signature")
            if recommendation is not None and isinstance(signature, dict):
                self._cached_wizard_recommendation = recommendation
                self._cached_wizard_signature = copy.deepcopy(signature)
                self._cached_wizard_log_text = str(wizard_state.get("log_text", ""))
                self._fit_wizard_search_strategy = str(signature.get("search_strategy", "legacy"))


class FitPanel(QWidget):
    """Fit setup and results panel with tabbed interface.

    Contains tabs for single dataset fitting and global (multi-dataset) fitting.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    preview_requested = Signal(object, object, object)  # (preview_result, fitted_curve, component_curves)
    # Keep payload generic to preserve Python dict key/value types end-to-end.
    global_fit_completed = Signal(object, object)  # (results_dict, global_params)
    share_function_with_group_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._single_state_by_run: dict[int, dict] = {}
        self._active_single_run_number: int | None = None

        # Create tab widget
        self._tabs = QTabWidget()

        # Single fit tab
        self._single_tab = SingleFitTab()
        self._single_tab.fit_completed.connect(self._on_single_fit_completed)
        self._single_tab.preview_requested.connect(self.preview_requested.emit)
        self._single_tab.share_function_with_group_requested.connect(
            self.share_function_with_group_requested.emit
        )
        self._tabs.addTab(self._single_tab, "Single")

        # Global fit tab
        self._global_tab = GlobalFitTab()
        self._global_tab.global_fit_completed.connect(self.global_fit_completed.emit)
        self._tabs.addTab(self._global_tab, "Global")

        layout.addWidget(self._tabs)

    def _on_single_fit_completed(self, fit_result, fitted_curve, component_curves) -> None:
        """Forward single-fit completion and cache seeds for global fitting."""
        dataset = self._single_tab._current_dataset
        if dataset is not None:
            run_number = int(dataset.run_number)
            self._global_tab.register_single_fit_seed(
                run_number,
                self._single_tab._composite_model,
                fit_result,
            )
            # Keep most recent tab state per run (parameters, function, and result text).
            self._single_state_by_run[run_number] = self._single_tab.get_state()
        self.fit_completed.emit(fit_result, fitted_curve, component_curves)

    def _run_number_from_dataset(self, dataset: MuonDataset | None) -> int | None:
        if dataset is None:
            return None
        try:
            return int(dataset.run_number)
        except (TypeError, ValueError):
            return None

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset for single fitting tab."""
        if self._active_single_run_number is not None:
            self._single_state_by_run[self._active_single_run_number] = self._single_tab.get_state()

        self._single_tab.set_dataset(dataset)

        run_number = self._run_number_from_dataset(dataset)
        self._active_single_run_number = run_number

        if run_number is None:
            return

        if run_number in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[run_number])
        else:
            # Unseen datasets should not inherit another run's fit UI/result state.
            self._single_tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))
            self._single_tab._result_label.setText("No fit performed yet")

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for global fitting tab."""
        self._global_tab.set_datasets(datasets)

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Apply fit-action blocking to both single and global tabs."""
        self._single_tab.set_fit_blocked(blocked, reason)
        self._global_tab.set_fit_blocked(blocked, reason)

    def single_fit_formula_string(self) -> str | None:
        """Return the active single-fit formula string, if available."""
        model = getattr(self._single_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def global_fit_formula_string(self) -> str | None:
        """Return the active global-fit formula string, if available."""
        model = getattr(self._global_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def clear_fits_for_runs(self, run_numbers: list[int]) -> int:
        """Clear cached single/global fit state for specific dataset runs."""
        normalized_runs: set[int] = set()
        for run_number in run_numbers:
            try:
                normalized_runs.add(int(run_number))
            except (TypeError, ValueError):
                continue

        if not normalized_runs:
            return 0

        changed_runs: set[int] = set()
        for run_number in normalized_runs:
            if self._single_state_by_run.pop(run_number, None) is not None:
                changed_runs.add(run_number)

        changed_runs |= self._global_tab.remove_single_fit_seeds(normalized_runs)

        active_run = self._active_single_run_number
        if active_run is not None and active_run in normalized_runs:
            self._single_tab._result_label.setText("No fit performed yet")

        return len(changed_runs)

    def get_single_state_for_run(self, run_number: int) -> dict | None:
        """Return current single-fit state for one run, if available."""
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return None

        if self._active_single_run_number == run_key:
            state = self._single_tab.get_state()
            self._single_state_by_run[run_key] = state
            return copy.deepcopy(state)

        state = self._single_state_by_run.get(run_key)
        if isinstance(state, dict):
            return copy.deepcopy(state)
        return None

    def share_single_function_state(self, source_run_number: int, target_run_numbers: list[int]) -> int:
        """Copy source single-fit function/parameter state to target runs.

        The copied state intentionally clears fit-result text for targets because
        no fit has been run for those datasets yet.
        """
        source_state = self.get_single_state_for_run(source_run_number)
        if not isinstance(source_state, dict):
            return 0

        shared_state = copy.deepcopy(source_state)
        shared_state["result_html"] = "No fit performed yet"

        updated = 0
        active_run = self._active_single_run_number
        for run_number in target_run_numbers:
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            if run_key == int(source_run_number):
                continue
            self._single_state_by_run[run_key] = copy.deepcopy(shared_state)
            if active_run is not None and run_key == active_run:
                self._single_tab.restore_state(self._single_state_by_run[run_key])
            updated += 1
        return updated

    def _result_html_from_fit(self, fit_result: object, source: str) -> str:
        """Build single-fit result HTML from a completed fit result object."""
        if getattr(fit_result, "success", False) is not True:
            message = str(getattr(fit_result, "message", "Fit failed"))
            return f"<b>{source} failed:</b> {message}"

        reduced = float(getattr(fit_result, "reduced_chi_squared", float("nan")))
        chi2 = float(getattr(fit_result, "chi_squared", float("nan")))
        lines = [
            f"<b>{source}</b>",
            f"<b>χ² = {chi2:.4f}</b>",
            f"<b>χ²ᵣ = {reduced:.4f}</b>",
            "<br><b>Parameters:</b>",
        ]

        uncertainties = getattr(fit_result, "uncertainties", {}) or {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            if not isinstance(name, str):
                continue
            value = float(getattr(param, "value", 0.0))
            unc = float(uncertainties.get(name, 0.0))
            lines.append(f"  {_format_param_label(name)} = {value:.6f} ± {unc:.6f}")
        return "<br>".join(lines)

    def _single_state_from_fit_result(
        self,
        model: CompositeModel,
        fit_result: object,
        source: str,
    ) -> dict:
        """Return single-tab state populated from a fitted model result."""
        values_by_name: dict[str, object] = {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            if isinstance(name, str):
                values_by_name[name] = param

        params: list[dict[str, object]] = []
        for pname in model.param_names:
            param = values_by_name.get(pname)
            if param is None:
                value = float(model.param_defaults.get(pname, 0.0))
                fixed = False
                default_min = get_param_info(pname).default_min
                min_text = str(default_min) if default_min is not None else "-inf"
                max_text = "inf"
            else:
                try:
                    value = float(getattr(param, "value", model.param_defaults.get(pname, 0.0)))
                except (TypeError, ValueError):
                    value = float(model.param_defaults.get(pname, 0.0))
                fixed = bool(getattr(param, "fixed", False))

                min_val = getattr(param, "min", -float("inf"))
                max_val = getattr(param, "max", float("inf"))
                min_text = "-inf" if min_val is None or not np.isfinite(float(min_val)) else str(float(min_val))
                max_text = "inf" if max_val is None or not np.isfinite(float(max_val)) else str(float(max_val))

            params.append({
                "name": pname,
                "value": value,
                "fixed": fixed,
                "min": min_text,
                "max": max_text,
            })

        return {
            "model_name": "Composite",
            "composite_model": model.to_dict(),
            "parameters": params,
            "result_html": self._result_html_from_fit(fit_result, source),
        }

    def register_global_fit_results(self, results_by_run: dict[int, tuple[object, object, object]]) -> None:
        """Persist per-run single-tab state using the latest successful global fit."""
        model = self._global_tab._composite_model
        active_run = self._active_single_run_number

        for run_number, payload in results_by_run.items():
            if not isinstance(payload, tuple) or not payload:
                continue
            fit_result = payload[0]
            if getattr(fit_result, "success", False) is not True:
                continue
            self._global_tab.register_single_fit_seed(run_number, model, fit_result)
            run_state = self._single_state_from_fit_result(model, fit_result, source="Global fit")
            self._single_state_by_run[int(run_number)] = run_state

            if active_run is not None and int(run_number) == int(active_run):
                self._single_tab.restore_state(run_state)

    # ── project state helpers ──────────────────────────────────────────

    def get_single_state(self) -> dict:
        """Return serialisable state of the single-fit tab."""
        if self._active_single_run_number is not None:
            self._single_state_by_run[self._active_single_run_number] = self._single_tab.get_state()

        active_state = self._single_tab.get_state()
        states_by_run = {
            str(run_number): dict(state)
            for run_number, state in self._single_state_by_run.items()
            if isinstance(state, dict)
        }
        combined_state = dict(active_state)
        combined_state["states_by_run"] = states_by_run
        combined_state["active_run_number"] = self._active_single_run_number
        return combined_state

    def restore_single_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        states_by_run: dict[int, dict] = {}
        raw_states = state.get("states_by_run") if isinstance(state, dict) else None
        if isinstance(raw_states, dict):
            for run_key, run_state in raw_states.items():
                if not isinstance(run_state, dict):
                    continue
                try:
                    run_number = int(run_key)
                except (TypeError, ValueError):
                    continue
                states_by_run[run_number] = dict(run_state)

        self._single_state_by_run = states_by_run

        active_run = self._active_single_run_number
        if active_run is not None and active_run in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[active_run])
            return

        # Backward-compatible legacy payloads (single shared state).
        if isinstance(state, dict):
            self._single_tab.restore_state(state)
            if active_run is not None:
                self._single_state_by_run[active_run] = self._single_tab.get_state()

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
