"""Undocked window for cross-group global parameter fit results."""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
    evaluate_parameter_model_fit,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
from asymmetry.gui.styles.widgets import apply_param_table_style

_PARAMETER_FIT_CURVE_SAMPLE_COUNT = 800


class GlobalParameterFitWindow(QMainWindow):
    """Display cross-group fit data, fitted model curves, and global/local values."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Parameter Fit")
        self.resize(1200, 800)

        self._result: CrossGroupFitResult | None = None
        self._groups: list[ParameterGroupData] = []
        self._model = None
        self._parameter_name: str | None = None
        self._x_key: str = "run"
        self._fit_x_min: float = float("nan")
        self._fit_x_max: float = float("nan")

        self._axes_tag_map: dict[int, str] = {}
        self._local_axes_tag_map: dict[int, str] = {}
        self._plot_annotations: list[dict[str, object]] = []
        self._local_plot_annotations: list[dict[str, object]] = []
        self._suppressed_group_label_tags: set[str] = set()
        self._local_y_controls: dict[str, dict[str, object]] = {}
        self._local_model_fits: dict[str, ParameterModelFit] = {}
        self._local_param_log_y: dict[str, bool] = {}
        self._local_selected_y_names: list[str] = []
        self._restored_local_selected_y: list[str] = []
        self._add_label_mode = False
        self._dragging_annotation: dict[str, object] | None = None

        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._left_canvas = None
        self._left_figure = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._left_figure = Figure(tight_layout=True)
            self._left_canvas = FigureCanvasQTAgg(self._left_figure)
            left_layout.addWidget(self._left_canvas)

            self._left_canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._left_canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self._left_canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        except ImportError:
            pass

        fit_controls_row = QHBoxLayout()
        self._show_components_check = QCheckBox("Show components")
        self._show_components_check.toggled.connect(self._on_show_components_toggled)
        fit_controls_row.addWidget(self._show_components_check)

        self._fit_log_x_check = QCheckBox("Log X")
        self._fit_log_x_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_log_x_check)

        self._fit_log_y_check = QCheckBox("Log Y")
        self._fit_log_y_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_log_y_check)

        self._fit_share_x_check = QCheckBox("Share X Axis")
        self._fit_share_x_check.setChecked(False)
        self._fit_share_x_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_share_x_check)

        fit_controls_row.addStretch()
        left_layout.addLayout(fit_controls_row)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        controls_row = QHBoxLayout()
        self._local_plot_mode_combo = QComboBox()
        self._local_plot_mode_combo.addItems(["Single Axes", "Subplots"])
        self._local_plot_mode_combo.currentTextChanged.connect(
            lambda _text: self._refresh_local_parameter_plots()
        )
        controls_row.addWidget(self._local_plot_mode_combo)

        self._local_log_x_check = QCheckBox("Log X")
        self._local_log_x_check.toggled.connect(self._refresh_local_parameter_plots)
        controls_row.addWidget(self._local_log_x_check)

        self._local_log_y_check = QCheckBox("Log Y")
        self._local_log_y_check.toggled.connect(self._refresh_local_parameter_plots)

        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._add_label_btn.toggled.connect(self._set_add_label_mode)
        controls_row.addWidget(self._add_label_btn)
        controls_row.addStretch()
        right_layout.addLayout(controls_row)

        self._params_table = QTableWidget(0, 3)
        self._params_table.setHorizontalHeaderLabels(["Parameter", "Value", "Uncertainty"])
        apply_param_table_style(self._params_table)
        right_layout.addWidget(self._params_table)

        self._local_y_selector_table = QTableWidget(0, 3)
        self._local_y_selector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._local_y_selector_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._local_y_selector_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._local_y_selector_table.horizontalHeader().setVisible(False)
        self._local_y_selector_table.verticalHeader().setVisible(False)
        self._local_y_selector_table.itemSelectionChanged.connect(
            self._on_local_y_selection_changed
        )
        right_layout.addWidget(self._local_y_selector_table)

        self._local_canvas = None
        self._local_figure = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._local_figure = Figure(tight_layout=True)
            self._local_canvas = FigureCanvasQTAgg(self._local_figure)
            right_layout.addWidget(self._local_canvas)

            self._local_canvas.mpl_connect("button_press_event", self._on_local_canvas_button_press)
            self._local_canvas.mpl_connect("motion_notify_event", self._on_local_canvas_motion)
            self._local_canvas.mpl_connect(
                "button_release_event", self._on_local_canvas_button_release
            )
        except ImportError:
            pass

        self._export_btn = QPushButton("Export fits to GLE")
        self._export_btn.clicked.connect(self._export_fit_subplot_gle)

        self._fit_gle_format_combo = QComboBox()
        self._fit_gle_format_combo.addItems(["PDF", "EPS"])
        self._fit_subplot_aspect_spin = QDoubleSpinBox()
        self._fit_subplot_aspect_spin.setRange(1.0, 5.0)
        self._fit_subplot_aspect_spin.setSingleStep(0.1)
        self._fit_subplot_aspect_spin.setDecimals(2)
        self._fit_subplot_aspect_spin.setValue(2.61)
        self._fit_subplot_aspect_spin.setSuffix(" : 1")
        fit_export_row = QHBoxLayout()
        fit_export_row.addWidget(self._export_btn)
        fit_export_row.addWidget(QLabel("Aspect (W:H):"))
        fit_export_row.addWidget(self._fit_subplot_aspect_spin)
        fit_export_row.addWidget(QLabel("Format:"))
        fit_export_row.addWidget(self._fit_gle_format_combo)
        fit_export_row.addStretch()
        left_layout.addLayout(fit_export_row)

        self._export_local_btn = QPushButton("Export plot(s) to GLE")
        self._export_local_btn.clicked.connect(self._export_local_parameters_gle)

        self._local_gle_format_combo = QComboBox()
        self._local_gle_format_combo.addItems(["PDF", "EPS"])
        local_export_row = QHBoxLayout()
        local_export_row.addWidget(self._export_local_btn)
        local_export_row.addWidget(QLabel("Format:"))
        local_export_row.addWidget(self._local_gle_format_combo)
        local_export_row.addStretch()
        right_layout.addLayout(local_export_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 500])
        self._sync_fit_scale_controls()

    def _sync_fit_scale_controls(self) -> None:
        components_on = self._show_components_check.isChecked()
        if components_on and self._fit_log_y_check.isChecked():
            self._fit_log_y_check.setChecked(False)
        self._fit_log_y_check.setEnabled(not components_on)

    def _on_show_components_toggled(self, _checked: bool) -> None:
        self._sync_fit_scale_controls()
        self._refresh_plot()

    def has_result(self) -> bool:
        return self._result is not None

    def _serialize_annotations(
        self, annotations: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for ann in annotations:
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(ann.get("text", "")),
                    "axis_tag": str(ann.get("axis_tag", "")),
                    "is_group_label": bool(ann.get("is_group_label", False)),
                }
            )
        return out

    def _deserialize_annotations(self, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, list):
            return []
        out: list[dict[str, object]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                x = float(entry.get("x", 0.0))
                y = float(entry.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(entry.get("text", "")),
                    "axis_tag": str(entry.get("axis_tag", "")),
                    "is_group_label": bool(entry.get("is_group_label", False)),
                    "artist": None,
                }
            )
        return out

    def _serialize_local_model_fits(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for param_name, model_fit in self._local_model_fits.items():
            ranges_data: list[dict[str, object]] = []
            for fit_range in model_fit.ranges:
                range_item: dict[str, object] = {
                    "x_min": fit_range.x_min,
                    "x_max": fit_range.x_max,
                    "model": fit_range.model.to_dict(),
                    "parameters": [
                        {
                            "name": p.name,
                            "value": p.value,
                            "min": p.min,
                            "max": p.max,
                            "fixed": p.fixed,
                        }
                        for p in fit_range.parameters
                    ],
                }
                if fit_range.result is not None:
                    range_item["result"] = {
                        "success": fit_range.result.success,
                        "chi_squared": fit_range.result.chi_squared,
                        "reduced_chi_squared": fit_range.result.reduced_chi_squared,
                        "message": fit_range.result.message,
                        "parameters": [
                            {
                                "name": p.name,
                                "value": p.value,
                                "min": p.min,
                                "max": p.max,
                                "fixed": p.fixed,
                            }
                            for p in fit_range.result.parameters
                        ],
                        "uncertainties": dict(fit_range.result.uncertainties),
                    }
                ranges_data.append(range_item)

            payload[param_name] = {
                "parameter_name": model_fit.parameter_name,
                "x_key": model_fit.x_key,
                "active": model_fit.active,
                "ranges": ranges_data,
            }
        return payload

    def _deserialize_local_model_fits(self, state: object) -> dict[str, ParameterModelFit]:
        if not isinstance(state, dict):
            return {}

        restored: dict[str, ParameterModelFit] = {}
        for key, entry in state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue

            x_key = str(entry.get("x_key", "run"))
            ranges_state = entry.get("ranges", [])
            if not isinstance(ranges_state, list):
                continue

            ranges: list[ModelFitRange] = []
            for range_state in ranges_state:
                if not isinstance(range_state, dict):
                    continue
                try:
                    model = ParameterCompositeModel.from_dict(
                        dict(range_state.get("model", range_state))
                    )
                except Exception:
                    continue

                params = ParameterSet()
                for p in range_state.get("parameters", []):
                    if not isinstance(p, dict):
                        continue
                    try:
                        params.add(
                            Parameter(
                                name=str(p.get("name", "")),
                                value=float(p.get("value", 0.0)),
                                min=float(p.get("min", -float("inf"))),
                                max=float(p.get("max", float("inf"))),
                                fixed=bool(p.get("fixed", False)),
                            )
                        )
                    except Exception:
                        continue

                result_obj = None
                result_state = range_state.get("result")
                if isinstance(result_state, dict):
                    result_params = ParameterSet()
                    for p in result_state.get("parameters", []):
                        if not isinstance(p, dict):
                            continue
                        try:
                            result_params.add(
                                Parameter(
                                    name=str(p.get("name", "")),
                                    value=float(p.get("value", 0.0)),
                                    min=float(p.get("min", -float("inf"))),
                                    max=float(p.get("max", float("inf"))),
                                    fixed=bool(p.get("fixed", False)),
                                )
                            )
                        except Exception:
                            continue
                    result_obj = ParameterModelFitResult(
                        success=bool(result_state.get("success", False)),
                        chi_squared=float(result_state.get("chi_squared", 0.0)),
                        reduced_chi_squared=float(result_state.get("reduced_chi_squared", 0.0)),
                        parameters=result_params,
                        uncertainties={
                            str(k): float(v)
                            for k, v in dict(result_state.get("uncertainties", {})).items()
                        },
                        message=str(result_state.get("message", "")),
                    )

                ranges.append(
                    ModelFitRange(
                        x_min=float(range_state.get("x_min"))
                        if range_state.get("x_min") is not None
                        else None,
                        x_max=float(range_state.get("x_max"))
                        if range_state.get("x_max") is not None
                        else None,
                        model=model,
                        parameters=params,
                        result=result_obj,
                    )
                )

            if not ranges:
                continue

            restored[key] = ParameterModelFit(
                parameter_name=str(entry.get("parameter_name", key)),
                x_key=x_key,
                active=bool(entry.get("active", True)),
                ranges=ranges,
            )

        return restored

    def get_state(self) -> dict[str, object]:
        return {
            "show_components": bool(self._show_components_check.isChecked()),
            "fit_log_x": bool(self._fit_log_x_check.isChecked()),
            "fit_log_y": bool(self._fit_log_y_check.isChecked()),
            "fit_share_x": bool(self._fit_share_x_check.isChecked()),
            "fit_subplot_aspect": float(self._fit_subplot_aspect_spin.value()),
            "local_log_x": bool(self._local_log_x_check.isChecked()),
            "local_log_y": bool(self._local_log_y_check.isChecked()),
            "local_param_log_y": dict(self._local_param_log_y),
            "local_model_fits": self._serialize_local_model_fits(),
            "local_selected_y": list(self._local_selected_y_names),
            "local_plot_mode": str(self._local_plot_mode_combo.currentText()),
            "plot_annotations": self._serialize_annotations(self._plot_annotations),
            "local_plot_annotations": self._serialize_annotations(self._local_plot_annotations),
            "suppressed_group_label_tags": sorted(self._suppressed_group_label_tags),
        }

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return

        self._show_components_check.setChecked(bool(state.get("show_components", False)))
        self._fit_log_x_check.setChecked(bool(state.get("fit_log_x", False)))
        self._fit_log_y_check.setChecked(bool(state.get("fit_log_y", False)))
        self._fit_share_x_check.setChecked(bool(state.get("fit_share_x", False)))
        self._local_log_x_check.setChecked(bool(state.get("local_log_x", False)))
        self._local_log_y_check.setChecked(bool(state.get("local_log_y", False)))
        raw_local_param_log = state.get("local_param_log_y", {})
        if isinstance(raw_local_param_log, dict):
            self._local_param_log_y = {str(k): bool(v) for k, v in raw_local_param_log.items()}
        else:
            self._local_param_log_y = {}
        self._local_model_fits = self._deserialize_local_model_fits(
            state.get("local_model_fits", {})
        )
        fit_subplot_aspect = state.get("fit_subplot_aspect")
        try:
            if fit_subplot_aspect is not None:
                self._fit_subplot_aspect_spin.setValue(float(fit_subplot_aspect))
            else:
                # Backward compatibility: convert older saved width setting to aspect ratio.
                fit_plot_width = state.get("fit_plot_width")
                if fit_plot_width is not None:
                    self._fit_subplot_aspect_spin.setValue(float(fit_plot_width) / 3.1)
        except (TypeError, ValueError):
            pass

        local_plot_mode = state.get("local_plot_mode")
        if isinstance(local_plot_mode, str) and local_plot_mode in {"Single Axes", "Subplots"}:
            self._local_plot_mode_combo.setCurrentText(local_plot_mode)

        selected_y = state.get("local_selected_y", [])
        if isinstance(selected_y, list):
            self._local_selected_y_names = [str(p) for p in selected_y]
            self._restored_local_selected_y = list(self._local_selected_y_names)
        else:
            self._local_selected_y_names = []
            self._restored_local_selected_y = []

        self._plot_annotations = self._deserialize_annotations(state.get("plot_annotations", []))
        self._local_plot_annotations = self._deserialize_annotations(
            state.get("local_plot_annotations", [])
        )

        suppressed = state.get("suppressed_group_label_tags", [])
        if isinstance(suppressed, list):
            self._suppressed_group_label_tags = {str(tag) for tag in suppressed}
        else:
            self._suppressed_group_label_tags = set()

        self._sync_fit_scale_controls()
        if self.has_result():
            self._refresh_plot()
            self._refresh_local_parameter_plots()

    def _selected_local_y_parameters(self) -> list[str]:
        params: list[str] = []
        for row in sorted(
            {index.row() for index in self._local_y_selector_table.selectedIndexes()}
        ):
            item = self._local_y_selector_table.item(row, 0)
            if item is None:
                continue
            pname = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(pname, str) and pname:
                params.append(pname)
        return params

    def _on_local_y_selection_changed(self) -> None:
        self._local_selected_y_names = self._selected_local_y_parameters()
        self._refresh_local_parameter_plots()

    def _rebuild_local_y_controls(
        self, parameter_names: list[str], preferred_selected: list[str] | None = None
    ) -> None:
        self._local_y_selector_table.blockSignals(True)
        self._local_y_selector_table.clearContents()
        self._local_y_selector_table.setRowCount(0)
        self._local_y_controls = {}

        if not parameter_names:
            self._local_y_selector_table.blockSignals(False)
            return

        self._local_y_selector_table.setRowCount(len(parameter_names))
        for idx, name in enumerate(parameter_names):
            item = QTableWidgetItem(get_param_info(name).unicode_label())
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._local_y_selector_table.setItem(idx, 0, item)

            fit_button = QPushButton("Model Fit")
            fit_button.clicked.connect(
                lambda _checked=False, p=name: self._open_local_model_fit_dialog(p)
            )
            self._local_y_selector_table.setCellWidget(idx, 1, fit_button)

            log_check = QCheckBox("log")
            log_check.setChecked(bool(self._local_param_log_y.get(name, False)))
            log_check.toggled.connect(
                lambda checked, p=name: self._on_local_param_log_toggled(p, checked)
            )
            log_container = QWidget()
            log_layout = QHBoxLayout(log_container)
            log_layout.setContentsMargins(0, 0, 0, 0)
            log_layout.addWidget(log_check)
            log_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._local_y_selector_table.setCellWidget(idx, 2, log_container)

            self._local_y_controls[name] = {"fit_button": fit_button, "log": log_check}

            fit = self._local_model_fits.get(name)
            if fit is not None and fit.active and self._has_successful_local_fit_curve(fit):
                fit_button.setText("Model Fit*")

        preferred = [p for p in (preferred_selected or []) if p in parameter_names]
        if preferred:
            for idx, name in enumerate(parameter_names):
                item = self._local_y_selector_table.item(idx, 0)
                if item is not None and name in preferred:
                    item.setSelected(True)
        elif parameter_names:
            item = self._local_y_selector_table.item(0, 0)
            if item is not None:
                item.setSelected(True)

        self._local_y_selector_table.resizeColumnsToContents()
        self._local_selected_y_names = self._selected_local_y_parameters()
        self._local_y_selector_table.blockSignals(False)

    def _has_successful_local_fit_curve(self, fit: ParameterModelFit) -> bool:
        return any(r.result is not None and r.result.success for r in fit.ranges)

    def _on_local_param_log_toggled(self, param_name: str, checked: bool) -> None:
        self._local_param_log_y[param_name] = bool(checked)
        self._refresh_local_parameter_plots()

    def _is_local_param_log_enabled(self, param_name: str) -> bool:
        return bool(self._local_param_log_y.get(param_name, False))

    def _local_group_x_key(self) -> str:
        if self._x_key == "field":
            return "temperature"
        if self._x_key == "temperature":
            return "field"
        return "run"

    def _open_local_model_fit_dialog(self, param_name: str) -> None:
        if self._result is None:
            return

        xs: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        for group in self._groups:
            pset = self._result.local_parameters.get(group.group_id)
            if pset is None or param_name not in pset:
                continue
            xs.append(float(group.group_variable_value))
            ys.append(float(pset[param_name].value))
            err = self._result.local_uncertainties.get(group.group_id, {}).get(param_name)
            es.append(float(err) if err is not None and np.isfinite(err) and err > 0 else np.nan)

        if len(xs) < 2:
            QMessageBox.information(self, "Model Fit", "Need at least two points for a model fit.")
            return

        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        e_arr = np.asarray(es, dtype=float)
        invalid_err = ~np.isfinite(e_arr) | (e_arr <= 0)
        if np.any(invalid_err):
            finite = np.abs(y_arr[np.isfinite(y_arr)])
            fallback = max(float(np.nanmedian(finite)) * 0.02, 1e-9) if finite.size else 1e-3
            e_arr = e_arr.copy()
            e_arr[invalid_err] = fallback

        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]
        e_arr = e_arr[order]

        dialog = ModelFitDialog(
            parameter_name=param_name,
            x_key=self._local_group_x_key(),
            x_values=x_arr,
            y_values=y_arr,
            y_errors=e_arr,
            existing_fit=self._local_model_fits.get(param_name),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.was_removed():
            self._local_model_fits.pop(param_name, None)
        else:
            fit = dialog.get_model_fit()
            if fit is not None:
                self._local_model_fits[param_name] = fit

        controls = self._local_y_controls.get(param_name)
        if isinstance(controls, dict):
            fit_button = controls.get("fit_button")
            if isinstance(fit_button, QPushButton):
                active_fit = self._local_model_fits.get(param_name)
                fit_button.setText(
                    "Model Fit*" if active_fit is not None and active_fit.active else "Model Fit"
                )

        self._refresh_local_parameter_plots()

    def set_results(
        self,
        *,
        parameter_name: str,
        x_key: str,
        groups: list[ParameterGroupData],
        model,
        result: CrossGroupFitResult,
        fit_x_min: float = float("nan"),
        fit_x_max: float = float("nan"),
    ) -> None:
        self._parameter_name = parameter_name
        self._x_key = x_key
        self._groups = groups
        self._model = model
        self._result = result
        self._fit_x_min = float(fit_x_min)
        self._fit_x_max = float(fit_x_max)
        self._refresh_table()
        self._refresh_plot()
        self._refresh_local_parameter_plots()

    def _x_label(self) -> str:
        return {
            "field": "$B$ (G)",
            "temperature": "$T$ (K)",
            "run": "Run Number",
        }.get(self._x_key, "x")

    def _x_label_gle(self) -> str:
        return {
            "field": "{\\it B} (G)",
            "temperature": "{\\it T} (K)",
            "run": "Run Number",
        }.get(self._x_key, "x")

    def _local_group_axis_label(self) -> str:
        return {
            "field": "$T$ (K)",
            "temperature": "$B$ (G)",
            "run": "Group variable",
        }.get(self._x_key, "Group variable")

    def _local_group_axis_label_gle(self) -> str:
        return {
            "field": "{\\it T} (K)",
            "temperature": "{\\it B} (G)",
            "run": "Group variable",
        }.get(self._x_key, "Group variable")

    def _local_group_axis_label_plain(self) -> str:
        return {
            "field": "T (K)",
            "temperature": "B (G)",
            "run": "Group variable",
        }.get(self._x_key, "Group variable")

    def _parameter_label(self, name: str | None) -> str:
        if not name:
            return "y"
        return get_param_info(name).latex_label()

    def _parameter_label_gle(self, name: str | None) -> str:
        if not name:
            return "y"
        return get_param_info(name).gle_label()

    def _safe_file_token(self, value: str) -> str:
        token = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip()
        )
        token = "_".join(part for part in token.split("_") if part)
        return token or "group"

    def _safe_data_name(self, value: str) -> str:
        token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).strip())
        token = "_".join(part for part in token.split("_") if part)
        return (token or "series").lower()

    def _model_kwargs_for_group(self, group_id: str) -> dict[str, float]:
        if self._result is None:
            return {}
        kwargs = {p.name: p.value for p in self._result.global_parameters}
        for p in self._result.fixed_parameters:
            kwargs[p.name] = p.value
        local = self._result.local_parameters.get(group_id)
        if local is not None:
            for p in local:
                kwargs[p.name] = p.value

        missing = [name for name in getattr(self._model, "param_names", []) if name not in kwargs]
        defaults = getattr(self._model, "param_defaults", {})
        for name in missing:
            if isinstance(defaults, dict) and name in defaults:
                kwargs[name] = float(defaults[name])
        return kwargs

    def _sample_group_fit_curve(
        self, group: ParameterGroupData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self._model is None:
            return None
        kwargs = self._model_kwargs_for_group(group.group_id)
        if not kwargs:
            return None

        xx = np.asarray(group.x, dtype=float)
        finite_x = xx[np.isfinite(xx)]
        if finite_x.size < 2:
            return None

        xx = np.sort(finite_x)
        if (
            np.isfinite(self._fit_x_min)
            and np.isfinite(self._fit_x_max)
            and self._fit_x_max > self._fit_x_min
        ):
            mask = (xx >= self._fit_x_min) & (xx <= self._fit_x_max)
            xx = xx[mask]
        if xx.size < 2:
            return None

        x_min = float(np.nanmin(xx))
        x_max = float(np.nanmax(xx))
        if self._local_group_x_key() == "field" and x_min > 0.0 and x_max > 0.0:
            xs = np.geomspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
        else:
            xs = np.linspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
        try:
            ys = np.asarray(self._model.function(xs, **kwargs), dtype=float)
        except KeyError:
            return None
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return None
        return xs[mask], ys[mask]

    def _write_fit_subplot_files(self, gle_path: Path) -> dict[str, dict[str, object]]:
        """Write explicit per-group data and fit files for GLE export."""
        written: dict[str, dict[str, object]] = {}
        for group in self._groups:
            token = self._safe_file_token(group.group_name or group.group_id)
            data_path = gle_path.with_name(f"{gle_path.stem}_{token}_data.dat")
            fit_path = gle_path.with_name(f"{gle_path.stem}_{token}_fit.fit")

            x = np.asarray(group.x, dtype=float)
            y = np.asarray(group.y, dtype=float)
            yerr = np.asarray(group.yerr, dtype=float)
            if yerr.shape != y.shape:
                yerr = np.full_like(y, np.nan, dtype=float)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask] if yerr.size else np.full_like(y, np.nan, dtype=float)
            if x.size == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            yerr = yerr[order]

            with open(data_path, "w", encoding="utf-8") as f:
                f.write("! Cross-group fit data\n")
                f.write("! columns: x y yerr\n")
                for xv, yv, ev in zip(x, y, yerr, strict=True):
                    ev_out = ev if np.isfinite(ev) and ev > 0 else np.nan
                    f.write(f"{xv:.10g} {yv:.10g} {ev_out:.10g}\n")

            sampled = self._sample_group_fit_curve(group)
            fit_written = False
            if sampled is not None:
                xs, ys = sampled
                with open(fit_path, "w", encoding="utf-8") as f:
                    f.write("! Cross-group fit model curve\n")
                    f.write("! columns: x y\n")
                    for xv, yv in zip(xs, ys, strict=True):
                        f.write(f"{xv:.10g} {yv:.10g}\n")
                fit_written = True

            written[group.group_id] = {
                "data_path": data_path,
                "fit_path": fit_path,
                "has_fit": fit_written,
                "has_err": bool(np.any(np.isfinite(yerr) & (yerr > 0))),
            }

        return written

    def _apply_fit_axis_scales(self, ax) -> None:
        if self._fit_log_x_check.isChecked():
            try:
                ax.set_xscale("log")
            except Exception:
                pass
        if self._fit_log_y_check.isChecked() and not self._show_components_check.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass

    def _refresh_table(self) -> None:
        self._params_table.setRowCount(0)
        if self._result is None:
            return

        for p in self._result.global_parameters:
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(
                row, 0, QTableWidgetItem(get_param_info(p.name).unicode_label())
            )
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            err = self._result.global_uncertainties.get(p.name)
            self._params_table.setItem(
                row, 2, QTableWidgetItem("" if err is None else f"{err:.3g}")
            )

        for p in self._result.fixed_parameters:
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(
                row, 0, QTableWidgetItem(f"{get_param_info(p.name).unicode_label()} (fixed)")
            )
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            self._params_table.setItem(row, 2, QTableWidgetItem(""))

        self._params_table.resizeColumnsToContents()

    def _refresh_plot(self) -> None:
        if self._left_canvas is None or self._left_figure is None:
            return
        self._left_figure.clear()
        self._axes_tag_map = {}
        if self._result is None or self._model is None:
            self._left_canvas.draw()
            return

        n = max(1, len(self._groups))
        x_label = self._x_label()
        y_label = self._parameter_label(self._parameter_name)
        share_x_axis = self._fit_share_x_check.isChecked()
        valid_group_tags = {group.group_id for group in self._groups}
        self._prune_stale_group_label_annotations(valid_group_tags)
        shared_x_ax = None
        for idx, group in enumerate(self._groups):
            if share_x_axis and shared_x_ax is not None:
                ax = self._left_figure.add_subplot(n, 1, idx + 1, sharex=shared_x_ax)
            else:
                ax = self._left_figure.add_subplot(n, 1, idx + 1)
                if share_x_axis and shared_x_ax is None:
                    shared_x_ax = ax
            self._axes_tag_map[id(ax)] = group.group_id
            x = group.x
            y = group.y
            e = group.yerr
            ax.errorbar(x, y, yerr=e, fmt="o", linestyle="none", color="black", capsize=2)

            kwargs = {p.name: p.value for p in self._result.global_parameters}
            for p in self._result.fixed_parameters:
                kwargs[p.name] = p.value
            local = self._result.local_parameters.get(group.group_id)
            if local is not None:
                for p in local:
                    kwargs[p.name] = p.value

            # Backward-compatible restore support: older saved cross-group
            # states may miss newer model parameters (e.g. D_perp). Fill from
            # model defaults before evaluating.
            missing = [
                name for name in getattr(self._model, "param_names", []) if name not in kwargs
            ]
            defaults = getattr(self._model, "param_defaults", {})
            for name in missing:
                if isinstance(defaults, dict) and name in defaults:
                    kwargs[name] = float(defaults[name])

            if kwargs:
                xx = x
                if xx.size >= 2:
                    xx = xx.copy()
                    xx.sort()

                if (
                    np.isfinite(self._fit_x_min)
                    and np.isfinite(self._fit_x_max)
                    and self._fit_x_max > self._fit_x_min
                ):
                    mask = (xx >= self._fit_x_min) & (xx <= self._fit_x_max)
                    xx = xx[mask]

                if xx.size >= 2:
                    x_min = float(np.nanmin(xx))
                    x_max = float(np.nanmax(xx))
                    if self._x_key == "field" and x_min > 0.0 and x_max > 0.0:
                        xx = np.geomspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
                    else:
                        xx = np.linspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)

                try:
                    if self._show_components_check.isChecked():
                        components = self._model.evaluate_components(
                            xx, additive_only=True, **kwargs
                        )
                        ordered = self._ordered_components_for_stacking(components)
                        cumulative = np.zeros_like(xx, dtype=float)
                        component_colors = [
                            "#8ecae6",
                            "#90be6d",
                            "#f4a261",
                            "#e5989b",
                            "#bdb2ff",
                            "#ffd166",
                        ]
                        for cidx, (_name, comp_y) in enumerate(ordered):
                            fill_color = component_colors[cidx % len(component_colors)]
                            comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                            lower = cumulative
                            upper = cumulative + comp_fill
                            ax.fill_between(xx, lower, upper, color=fill_color, alpha=0.3, zorder=1)
                            ax.plot(
                                xx,
                                upper,
                                linestyle="--",
                                linewidth=0.8,
                                color=fill_color,
                                alpha=0.9,
                                zorder=2,
                            )
                            cumulative = upper

                    yy = self._model.function(xx, **kwargs)
                    ax.plot(xx, yy, color="red", linewidth=1.5)
                except KeyError as exc:
                    ax.text(
                        0.02,
                        0.95,
                        f"Fit curve unavailable: {exc}",
                        transform=ax.transAxes,
                        fontsize=9,
                        va="top",
                        color="tab:red",
                    )
            if share_x_axis:
                self._ensure_group_label_annotation(group.group_id, group.group_name, ax)
                ax.set_title("")
            else:
                ax.set_title(group.group_name, pad=10)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            self._apply_fit_axis_scales(ax)
            if (not share_x_axis) or idx == n - 1:
                ax.set_xlabel(x_label)

        self._draw_plot_annotations(local=False)

        self._left_figure.tight_layout(h_pad=1.8)
        self._left_figure.subplots_adjust(left=0.16, hspace=0.5)
        self._left_canvas.draw()

    def _refresh_local_parameter_plots(self) -> None:
        if self._local_canvas is None or self._local_figure is None:
            return

        self._local_figure.clear()
        self._local_axes_tag_map = {}
        if self._result is None:
            self._local_canvas.draw()
            return

        local_param_names = sorted(
            {p.name for pset in self._result.local_parameters.values() for p in pset}
        )

        previous_selected = list(self._local_selected_y_names)
        preferred_selected = previous_selected or self._restored_local_selected_y
        self._rebuild_local_y_controls(local_param_names, preferred_selected=preferred_selected)
        self._restored_local_selected_y = []
        selected_names = self._selected_local_y_parameters()
        self._local_selected_y_names = list(selected_names)
        if selected_names:
            local_param_names = [p for p in local_param_names if p in selected_names]

        if not local_param_names:
            ax = self._local_figure.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            ax.grid(True, alpha=0.3)
            self._local_canvas.draw()
            return

        x_label = self._local_group_axis_label()

        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for pname in local_param_names:
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []

            for group in self._groups:
                pset = self._result.local_parameters.get(group.group_id)
                if pset is None:
                    continue
                param = pset[pname] if pname in pset else None
                if param is None:
                    continue
                xs.append(float(group.group_variable_value))
                ys.append(float(param.value))
                err = self._result.local_uncertainties.get(group.group_id, {}).get(pname)
                es.append(
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )

            if not xs:
                continue
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            e_arr = np.asarray(es, dtype=float)
            order = np.argsort(x_arr)
            traces.append((pname, x_arr[order], y_arr[order], e_arr[order]))

        if not traces:
            ax = self._local_figure.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            ax.grid(True, alpha=0.3)
            self._local_canvas.draw()
            return

        plot_mode = self._local_plot_mode_combo.currentText()
        if plot_mode == "Subplots" and len(traces) > 1:
            num_cols = 1
            num_rows = (len(traces) + num_cols - 1) // num_cols
            shared_x_ax = None
            for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                if shared_x_ax is not None:
                    ax = self._local_figure.add_subplot(
                        num_rows, num_cols, idx + 1, sharex=shared_x_ax
                    )
                else:
                    ax = self._local_figure.add_subplot(num_rows, num_cols, idx + 1)
                    shared_x_ax = ax
                self._local_axes_tag_map[id(ax)] = pname
                ax.errorbar(
                    x_arr,
                    y_arr,
                    yerr=e_arr,
                    fmt="o",
                    linestyle="none",
                    color="C0",
                    capsize=2,
                    zorder=6,
                )
                finite_err = np.isfinite(e_arr) & (e_arr > 0)
                if np.any(finite_err):
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr,
                        fmt="none",
                        ecolor="gray",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )
                ax.set_ylabel(self._parameter_label(pname))
                fit = self._local_model_fits.get(pname)
                if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                    for curve in evaluate_parameter_model_fit(fit, num_points=200):
                        ax.plot(curve.x, curve.y, color="red", linewidth=1.5, zorder=7)
                row = idx // num_cols
                if row == num_rows - 1:
                    ax.set_xlabel(x_label)
                else:
                    ax.tick_params(labelbottom=False)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                param_log = self._is_local_param_log_enabled(pname)
                if param_log:
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
        else:
            ax = self._local_figure.add_subplot(111)
            self._local_axes_tag_map[id(ax)] = "main"
            if len(traces) == 2:
                pname_l, x_l, y_l, e_l = traces[0]
                pname_r, x_r, y_r, e_r = traces[1]
                ax2 = ax.twinx()
                self._local_axes_tag_map[id(ax)] = pname_l
                self._local_axes_tag_map[id(ax2)] = pname_r

                ax.errorbar(
                    x_l, y_l, yerr=e_l, fmt="o", linestyle="none", color="C0", capsize=2, zorder=6
                )
                finite_l = np.isfinite(e_l) & (e_l > 0)
                if np.any(finite_l):
                    ax.errorbar(
                        x_l,
                        y_l,
                        yerr=e_l,
                        fmt="none",
                        ecolor="C0",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax2.errorbar(
                    x_r, y_r, yerr=e_r, fmt="o", linestyle="none", color="C1", capsize=2, zorder=6
                )
                finite_r = np.isfinite(e_r) & (e_r > 0)
                if np.any(finite_r):
                    ax2.errorbar(
                        x_r,
                        y_r,
                        yerr=e_r,
                        fmt="none",
                        ecolor="C1",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax.set_ylabel(self._parameter_label(pname_l), color="C0")
                ax2.set_ylabel(self._parameter_label(pname_r), color="C1")
                ax.tick_params(axis="y", colors="C0")
                ax2.tick_params(axis="y", colors="C1")
                ax.set_xlabel(x_label)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                log_l = self._is_local_param_log_enabled(pname_l)
                log_r = self._is_local_param_log_enabled(pname_r)
                if log_l:
                    ax.set_yscale("log")
                if log_r:
                    ax2.set_yscale("log")
                ax.grid(True, alpha=0.3)
            else:
                for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                    color = f"C{idx % 10}"
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr,
                        fmt="o",
                        linestyle="none",
                        color=color,
                        capsize=2,
                        zorder=6,
                        label=self._parameter_label(pname),
                    )
                    finite_err = np.isfinite(e_arr) & (e_arr > 0)
                    if np.any(finite_err):
                        ax.errorbar(
                            x_arr,
                            y_arr,
                            yerr=e_arr,
                            fmt="none",
                            ecolor=color,
                            capsize=2,
                            elinewidth=1,
                            zorder=5,
                        )
                    fit = self._local_model_fits.get(pname)
                    if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                        for curve in evaluate_parameter_model_fit(fit, num_points=200):
                            ax.plot(curve.x, curve.y, color=color, linewidth=1.2, zorder=7)
                if len(traces) == 1:
                    ax.set_ylabel(self._parameter_label(traces[0][0]))
                else:
                    ax.set_ylabel("Parameter Value")
                    ax.legend(loc="best")
                ax.set_xlabel(x_label)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                if len(traces) == 1:
                    param_log = self._is_local_param_log_enabled(traces[0][0])
                else:
                    param_log = False
                if param_log:
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

        self._draw_plot_annotations(local=True)

        self._local_figure.tight_layout()
        self._local_canvas.draw()

    def _write_local_parameter_data_file(
        self,
        gle_path: Path,
        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[Path, dict[str, tuple[int, int]]]:
        data_path = gle_path.with_name(f"{gle_path.stem}_local_parameters.dat")

        x_keys: dict[str, float] = {}
        for _pname, x_arr, _y_arr, _e_arr in traces:
            for xv in x_arr:
                x_keys[f"{float(xv):.12g}"] = float(xv)
        x_values = np.array(sorted(x_keys.values()), dtype=float)

        col_map: dict[str, tuple[int, int]] = {}
        columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
            y_out = np.full_like(x_values, np.nan, dtype=float)
            e_out = np.full_like(x_values, np.nan, dtype=float)
            local = {
                f"{float(x):.12g}": (float(y), float(e))
                for x, y, e in zip(x_arr, y_arr, e_arr, strict=True)
            }
            for i, xv in enumerate(x_values):
                key = f"{float(xv):.12g}"
                if key in local:
                    yv, ev = local[key]
                    y_out[i] = yv
                    e_out[i] = ev if np.isfinite(ev) and ev >= 0 else np.nan
            columns[pname] = (y_out, e_out)
            y_col = 2 + idx * 2
            e_col = y_col + 1
            col_map[pname] = (y_col, e_col)

        global_cols: list[tuple[str, float, float]] = []
        if self._result is not None:
            for p in self._result.global_parameters:
                err = self._result.global_uncertainties.get(p.name, np.nan)
                err_val = (
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )
                global_cols.append((p.name, float(p.value), err_val))

        with open(data_path, "w", encoding="utf-8") as f:
            f.write("! Local parameter trends\n")

            if self._result is not None:
                f.write("! Global parameter table:\n")
                f.write("!   Parameter                     Value                Uncertainty\n")
                for p in self._result.global_parameters:
                    info = get_param_info(p.name)
                    unit = f" ({info.unit})" if info.unit else ""
                    err = self._result.global_uncertainties.get(p.name)
                    err_text = "nan"
                    if err is not None and np.isfinite(err):
                        err_text = f"{float(err):.10g}"
                    f.write(f"!   {p.name}{unit:<24} {float(p.value):>16.10g} {err_text:>24}\n")
                for p in self._result.fixed_parameters:
                    info = get_param_info(p.name)
                    unit = f" ({info.unit})" if info.unit else ""
                    f.write(f"!   {p.name}{unit} [fixed] = {float(p.value):.10g}\n")
                f.write("!\n")

            f.write(f"! columns: {self._local_group_axis_label_plain()}")
            for pname, _x, _y, _e in traces:
                info = get_param_info(pname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f" {pname}{unit} err_{pname}{unit}")
            for gname, _gval, _gerr in global_cols:
                info = get_param_info(gname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f" {gname}{unit} err_{gname}{unit}")
            f.write("\n")

            col_idx = 1
            f.write("! Column map:\n")
            f.write(f"!   c{col_idx:>2} = {self._local_group_axis_label_plain()}\n")
            col_idx += 1
            for pname, _x, _y, _e in traces:
                info = get_param_info(pname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f"!   c{col_idx:>2} = {pname}{unit}\n")
                col_idx += 1
                f.write(f"!   c{col_idx:>2} = err_{pname}{unit}\n")
                col_idx += 1
            for gname, _gval, _gerr in global_cols:
                info = get_param_info(gname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f"!   c{col_idx:>2} = {gname}{unit}\n")
                col_idx += 1
                f.write(f"!   c{col_idx:>2} = err_{gname}{unit}\n")
                col_idx += 1

            for row_idx, xv in enumerate(x_values):
                parts = [f"{xv:.10g}"]
                for pname, _x, _y, _e in traces:
                    yv, ev = columns[pname]
                    parts.append(f"{yv[row_idx]:.10g}")
                    parts.append(f"{ev[row_idx]:.10g}")
                for _gname, gval, gerr in global_cols:
                    parts.append(f"{gval:.10g}")
                    parts.append(f"{gerr:.10g}")
                f.write(" ".join(parts) + "\n")

        return data_path, col_map

    def _write_local_parameter_fit_files(
        self,
        gle_path: Path,
        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, Path]:
        fit_files: dict[str, Path] = {}
        for pname, _x_arr, _y_arr, _e_arr in traces:
            fit = self._local_model_fits.get(pname)
            if fit is None or not fit.active or fit.x_key != self._local_group_x_key():
                continue

            curves = evaluate_parameter_model_fit(fit, num_points=200)
            if not curves:
                continue

            fit_path = gle_path.with_name(
                f"{gle_path.stem}_local_{self._safe_data_name(pname)}.fit"
            )
            info = get_param_info(pname)
            with open(fit_path, "w", encoding="utf-8") as f:
                f.write("! Local parameter model fit curve\n")
                f.write(f"! parameter: {info.plain_label()}\n")
                f.write(f"! x-axis: {self._local_group_axis_label_plain()}\n")
                f.write("! columns: x y\n")
                for cidx, curve in enumerate(curves):
                    if cidx > 0:
                        f.write("nan nan\n")
                    for xv, yv in zip(curve.x, curve.y, strict=True):
                        f.write(f"{float(xv):.10g} {float(yv):.10g}\n")

            fit_files[pname] = fit_path

        return fit_files

    def _ordered_components_for_stacking(
        self, components: list[tuple[str, np.ndarray]]
    ) -> list[tuple[str, np.ndarray]]:
        if not components:
            return []

        def _priority(name: str) -> int:
            lname = name.lower()
            if "bg" in lname or "background" in lname or "constant" in lname:
                return 0
            return 1

        scored: list[tuple[int, float, float, int, tuple[str, np.ndarray]]] = []
        for idx, item in enumerate(components):
            name, values = item
            arr = np.maximum(np.asarray(values, dtype=float), 0.0)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                mean_val = 0.0
                variability = 0.0
            else:
                mean_val = float(np.mean(finite))
                variability = float(np.std(finite) / max(mean_val, 1e-12))
            scored.append((_priority(name), variability, mean_val, idx, item))

        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [item for *_meta, item in scored]

    def _set_add_label_mode(self, enabled: bool) -> None:
        self._add_label_mode = bool(enabled)

    def _default_group_label_position(self, ax) -> tuple[float, float]:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if not (
            np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max)
        ):
            return 0.0, 0.0
        x = float(x_min + 0.02 * (x_max - x_min))
        y = float(y_max - 0.05 * (y_max - y_min))
        return x, y

    def _ensure_group_label_annotation(self, axis_tag: str, label: str, ax) -> None:
        if axis_tag in self._suppressed_group_label_tags:
            return
        for ann in self._plot_annotations:
            if bool(ann.get("is_group_label")) and str(ann.get("axis_tag", "")) == axis_tag:
                return
        x_pos, y_pos = self._default_group_label_position(ax)
        self._plot_annotations.append(
            {
                "x": x_pos,
                "y": y_pos,
                "text": label,
                "axis_tag": axis_tag,
                "is_group_label": True,
                "artist": None,
            }
        )

    def _prune_stale_group_label_annotations(self, valid_tags: set[str]) -> None:
        self._plot_annotations = [
            ann
            for ann in self._plot_annotations
            if not (
                bool(ann.get("is_group_label")) and str(ann.get("axis_tag", "")) not in valid_tags
            )
        ]
        self._suppressed_group_label_tags &= valid_tags

    def _draw_plot_annotations(self, *, local: bool) -> None:
        annotations = self._local_plot_annotations if local else self._plot_annotations
        for ann in annotations:
            ann["artist"] = None
            ax = self._axis_for_tag(str(ann.get("axis_tag", "")), local=local)
            if ax is None:
                continue
            is_group_label = bool(ann.get("is_group_label"))
            if is_group_label and (not local) and (not self._fit_share_x_check.isChecked()):
                continue
            text_artist = ax.text(
                float(ann.get("x", 0.0)),
                float(ann.get("y", 0.0)),
                str(ann.get("text", "")),
                fontsize=9,
                ha="left",
                va="top" if is_group_label else "bottom",
                zorder=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5}
                if is_group_label
                else None,
            )
            text_artist.set_picker(True)
            ann["artist"] = text_artist

    def _axis_for_tag(self, tag: str, *, local: bool):
        figure = self._local_figure if local else self._left_figure
        tag_map = self._local_axes_tag_map if local else self._axes_tag_map
        if figure is None:
            return None
        for ax in figure.axes:
            if tag_map.get(id(ax)) == tag:
                return ax
        return None

    def _annotation_at_event(self, event, *, local: bool) -> dict[str, object] | None:
        annotations = self._local_plot_annotations if local else self._plot_annotations
        for ann in annotations:
            artist = ann.get("artist")
            if artist is None:
                continue
            contains, _ = artist.contains(event)
            if contains:
                return ann
        return None

    def _on_canvas_button_press(self, event) -> None:
        self._handle_canvas_button_press(event, local=False)

    def _on_local_canvas_button_press(self, event) -> None:
        self._handle_canvas_button_press(event, local=True)

    def _handle_canvas_button_press(self, event, *, local: bool) -> None:
        if event.inaxes is None:
            return

        ann = self._annotation_at_event(event, local=local)
        if event.button == 3 and ann is not None:
            target = self._local_plot_annotations if local else self._plot_annotations
            if not local and bool(ann.get("is_group_label")):
                self._suppressed_group_label_tags.add(str(ann.get("axis_tag", "")))
            target.remove(ann)
            if local:
                self._refresh_local_parameter_plots()
            else:
                self._refresh_plot()
            return

        if event.button == 1 and event.dblclick and ann is not None:
            current = str(ann.get("text", ""))
            text, ok = QInputDialog.getText(self, "Edit Label", "Text:", text=current)
            if ok:
                ann["text"] = text
                if local:
                    self._refresh_local_parameter_plots()
                else:
                    self._refresh_plot()
            return

        if event.button == 1 and self._add_label_mode:
            text, ok = QInputDialog.getText(self, "Add Label", "Text:")
            if ok and text.strip():
                axis_tag = (
                    self._local_axes_tag_map.get(id(event.inaxes), "")
                    if local
                    else self._axes_tag_map.get(id(event.inaxes), "")
                )
                target = self._local_plot_annotations if local else self._plot_annotations
                target.append(
                    {
                        "x": float(event.xdata),
                        "y": float(event.ydata),
                        "text": text.strip(),
                        "axis_tag": axis_tag,
                        "artist": None,
                    }
                )
                if local:
                    self._refresh_local_parameter_plots()
                else:
                    self._refresh_plot()
                self._add_label_btn.setChecked(False)
            return

        if event.button == 1 and ann is not None:
            self._dragging_annotation = ann

    def _on_canvas_motion(self, event) -> None:
        self._handle_canvas_motion(event, local=False)

    def _on_local_canvas_motion(self, event) -> None:
        self._handle_canvas_motion(event, local=True)

    def _handle_canvas_motion(self, event, *, local: bool) -> None:
        if self._dragging_annotation is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._dragging_annotation["x"] = float(event.xdata)
        self._dragging_annotation["y"] = float(event.ydata)
        artist = self._dragging_annotation.get("artist")
        if artist is not None:
            artist.set_position((float(event.xdata), float(event.ydata)))
            canvas = self._local_canvas if local else self._left_canvas
            if canvas is not None:
                canvas.draw_idle()

    def _on_canvas_button_release(self, _event) -> None:
        self._dragging_annotation = None

    def _on_local_canvas_button_release(self, _event) -> None:
        self._dragging_annotation = None

    def _fit_subplot_layout_params(self, subplot_aspect: float) -> dict[str, float]:
        """Return aspect-aware subplot spacing to reduce clipping across shapes."""
        # Narrower plots need more side margins for y-labels/ticks; wider plots can use tighter margins.
        t = max(0.0, min(1.0, (2.8 - float(subplot_aspect)) / 1.8))
        left = 0.11 + 0.09 * t
        right = 0.992 - 0.045 * t
        return {
            "left": left,
            "right": right,
            "top": 0.968,
            "bottom": 0.065,
            "hspace": 0.50,
        }

    def _build_fit_subplot_gle_figure(self, glp, gle_path: Path):
        share_x_axis = self._fit_share_x_check.isChecked()
        subplot_height = 3.1
        subplot_aspect = float(self._fit_subplot_aspect_spin.value())
        plot_width = subplot_aspect * subplot_height
        fig, axes = glp.subplots(
            nrows=max(1, len(self._groups)),
            ncols=1,
            figsize=(plot_width, max(5.2, subplot_height * max(1, len(self._groups)))),
            sharex=share_x_axis,
        )
        subplot_axes = axes if isinstance(axes, list) else [axes]
        file_map = self._write_fit_subplot_files(gle_path)

        x_label = self._x_label_gle()
        y_label = self._parameter_label_gle(self._parameter_name)
        for idx, group in enumerate(self._groups):
            ax = subplot_axes[idx]
            file_info = file_map.get(group.group_id)
            data_path = file_info.get("data_path") if isinstance(file_info, dict) else None
            if (
                isinstance(file_info, dict)
                and data_path is not None
                and hasattr(ax, "errorbar_from_file")
            ):
                ax.errorbar_from_file(
                    data_path.name,
                    x_col=1,
                    y_col=2,
                    yerr_col=3 if bool(file_info.get("has_err", False)) else None,
                    color="black",
                    marker="o",
                    markersize=4,
                    capsize=2,
                )
            else:
                yerr = np.asarray(group.yerr, dtype=float)
                has_err = bool(np.any(np.isfinite(yerr) & (yerr > 0)))
                ax.errorbar(
                    group.x,
                    group.y,
                    yerr=group.yerr if has_err else None,
                    fmt="o",
                    linestyle="none",
                    marker="o",
                    color="black",
                    capsize=2,
                )

            fit_path = file_info.get("fit_path") if isinstance(file_info, dict) else None
            sampled = self._sample_group_fit_curve(group)
            if sampled is not None:
                xx, yy = sampled
            else:
                xx, yy = None, None

            if (
                xx is not None
                and self._show_components_check.isChecked()
                and self._model is not None
            ):
                try:
                    kwargs = self._model_kwargs_for_group(group.group_id)
                    components = self._model.evaluate_components(xx, additive_only=True, **kwargs)
                    ordered = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xx, dtype=float)
                    component_colors = [
                        "lightblue",
                        "lightgreen",
                        "pink",
                        "lightgray",
                        "cyan",
                        "yellow",
                    ]
                    group_token = self._safe_data_name(group.group_name or group.group_id)
                    for cidx, (_name, comp_y) in enumerate(ordered):
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_token = self._safe_data_name(_name)
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(
                            xx,
                            lower,
                            upper,
                            color=fill_color,
                            alpha=0.3,
                            data_name=f"component_{group_token}_{comp_token}_fill",
                        )
                        ax.plot(
                            xx,
                            upper,
                            linestyle="--",
                            linewidth=0.8,
                            color=fill_color,
                            data_name=f"component_{group_token}_{comp_token}_edge",
                        )
                        cumulative = upper
                except KeyError:
                    pass

            if (
                isinstance(file_info, dict)
                and bool(file_info.get("has_fit", False))
                and fit_path is not None
                and hasattr(ax, "line_from_file")
            ):
                ax.line_from_file(
                    fit_path.name,
                    x_col=1,
                    y_col=2,
                    color="red",
                    linestyle="-",
                    linewidth=1.5,
                )
            else:
                if xx is not None and yy is not None:
                    group_token = self._safe_data_name(group.group_name or group.group_id)
                    ax.plot(xx, yy, color="red", linewidth=1.5, data_name=f"model_{group_token}")
                else:
                    ax.text(0.02, 0.95, "Fit curve unavailable", color="red", ha="left")

            for ann in self._plot_annotations:
                if str(ann.get("axis_tag", "")) != group.group_id:
                    continue
                is_group_label = bool(ann.get("is_group_label"))
                if is_group_label and not share_x_axis:
                    continue
                try:
                    x_ann = float(ann.get("x", 0.0))
                    y_ann = float(ann.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                text = str(ann.get("text", "")).strip()
                if text:
                    ax.text(
                        x_ann,
                        y_ann,
                        text,
                        ha="left",
                        va="top" if is_group_label else "bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5}
                        if is_group_label
                        else None,
                    )

            if share_x_axis:
                self._ensure_group_label_annotation(group.group_id, group.group_name, ax)
                ax.set_title("")
            else:
                ax.set_title(group.group_name)
            ax.set_ylabel(y_label)
            self._apply_fit_axis_scales(ax)
            if (not share_x_axis) or idx == len(self._groups) - 1:
                ax.set_xlabel(x_label)

        if hasattr(fig, "subplots_adjust"):
            fig.subplots_adjust(**self._fit_subplot_layout_params(subplot_aspect))

        return fig

    def _build_local_parameter_gle_figure(self, glp, gle_path: Path):
        local_param_names = sorted(
            {p.name for pset in self._result.local_parameters.values() for p in pset}
        )
        if not local_param_names:
            fig = glp.figure(figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            return fig

        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for pname in local_param_names:
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []
            for group in self._groups:
                pset = self._result.local_parameters.get(group.group_id)
                if pset is None or pname not in pset:
                    continue
                xs.append(float(group.group_variable_value))
                ys.append(float(pset[pname].value))
                err = self._result.local_uncertainties.get(group.group_id, {}).get(pname)
                es.append(
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )
            if not xs:
                continue
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            e_arr = np.asarray(es, dtype=float)
            order = np.argsort(x_arr)
            traces.append((pname, x_arr[order], y_arr[order], e_arr[order]))

        all_traces = list(traces)
        selected_names = self._selected_local_y_parameters()
        if selected_names:
            traces = [t for t in traces if t[0] in selected_names]
        if not traces:
            fig = glp.figure(figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title("No local parameters selected")
            return fig

        data_path, col_map = self._write_local_parameter_data_file(gle_path, all_traces)
        fit_file_map = self._write_local_parameter_fit_files(gle_path, all_traces)

        x_label = self._local_group_axis_label_gle()
        plot_mode = self._local_plot_mode_combo.currentText()
        if plot_mode == "Subplots" and len(traces) > 1:
            num_cols = 1
            num_rows = (len(traces) + num_cols - 1) // num_cols
            fig, axes = glp.subplots(
                nrows=num_rows, ncols=num_cols, figsize=(7.2, max(4.8, 2.8 * num_rows)), sharex=True
            )
            flat_axes = axes if isinstance(axes, list) else [axes]
            for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                ax = flat_axes[idx]
                has_err = bool(np.any(np.isfinite(e_arr) & (e_arr >= 0)))
                cols = col_map[pname]
                if hasattr(ax, "errorbar_from_file"):
                    ax.errorbar_from_file(
                        data_path.name,
                        x_col=1,
                        y_col=cols[0],
                        yerr_col=cols[1] if has_err else None,
                        marker="o",
                        color="black",
                        capsize=2,
                    )
                else:
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr if has_err else None,
                        marker="o",
                        color="black",
                        linestyle="none",
                        capsize=2,
                    )
                ax.set_ylabel(self._parameter_label_gle(pname))
                row = idx // num_cols
                if row == num_rows - 1:
                    ax.set_xlabel(x_label)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                param_log = self._is_local_param_log_enabled(pname)
                if param_log:
                    ax.set_yscale("log")
                fit = self._local_model_fits.get(pname)
                if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                    fit_path = fit_file_map.get(pname)
                    if fit_path is not None and hasattr(ax, "line_from_file"):
                        ax.line_from_file(
                            fit_path.name,
                            x_col=1,
                            y_col=2,
                            color="red",
                            linestyle="-",
                            linewidth=1.5,
                        )
                    else:
                        for curve in evaluate_parameter_model_fit(fit, num_points=200):
                            ax.plot(
                                curve.x,
                                curve.y,
                                color="red",
                                linewidth=1.5,
                                data_name=f"model_local_{self._safe_data_name(pname)}",
                            )
                for ann in self._local_plot_annotations:
                    if str(ann.get("axis_tag", "")) != pname:
                        continue
                    text = str(ann.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        x_ann = float(ann.get("x", 0.0))
                        y_ann = float(ann.get("y", 0.0))
                    except (TypeError, ValueError):
                        continue
                    ax.text(x_ann, y_ann, text, ha="left")
            return fig

        fig = glp.figure(figsize=(7.0, 4.5))
        ax = fig.add_subplot(111)
        for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
            color = ["black", "blue", "red"][idx % 3]
            has_err = bool(np.any(np.isfinite(e_arr) & (e_arr >= 0)))
            cols = col_map[pname]
            if hasattr(ax, "errorbar_from_file"):
                ax.errorbar_from_file(
                    data_path.name,
                    x_col=1,
                    y_col=cols[0],
                    yerr_col=cols[1] if has_err else None,
                    marker="o",
                    color=color,
                    capsize=2,
                    label=self._parameter_label_gle(pname),
                )
            else:
                ax.errorbar(
                    x_arr,
                    y_arr,
                    yerr=e_arr if has_err else None,
                    marker="o",
                    linestyle="none",
                    color=color,
                    capsize=2,
                    label=self._parameter_label_gle(pname),
                )
            fit = self._local_model_fits.get(pname)
            if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                fit_path = fit_file_map.get(pname)
                if fit_path is not None and hasattr(ax, "line_from_file"):
                    ax.line_from_file(
                        fit_path.name,
                        x_col=1,
                        y_col=2,
                        color=color,
                        linestyle="-",
                        linewidth=1.2,
                    )
                else:
                    for curve in evaluate_parameter_model_fit(fit, num_points=200):
                        ax.plot(
                            curve.x,
                            curve.y,
                            color=color,
                            linewidth=1.2,
                            data_name=f"model_local_{self._safe_data_name(pname)}",
                        )
        ax.set_xlabel(x_label)
        if self._local_log_x_check.isChecked():
            ax.set_xscale("log")
        if len(traces) == 1:
            param_log = self._is_local_param_log_enabled(traces[0][0])
            if param_log:
                ax.set_yscale("log")
        if len(traces) == 1:
            ax.set_ylabel(self._parameter_label_gle(traces[0][0]))
        else:
            ax.set_ylabel("Parameter Value")
            ax.legend(loc="best")

        for ann in self._local_plot_annotations:
            axis_tag = str(ann.get("axis_tag", ""))
            if axis_tag not in {"main"} and not any(name == axis_tag for name, *_rest in traces):
                continue
            text = str(ann.get("text", "")).strip()
            if not text:
                continue
            try:
                x_ann = float(ann.get("x", 0.0))
                y_ann = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            ax.text(x_ann, y_ann, text, ha="left")

        return fig

    def _show_gle_preview(self, output_path: Path) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if not output_path.exists():
            return

        try:
            import tempfile

            from PySide6.QtGui import QPixmap

            dialog = QDialog(self)
            dialog.setWindowTitle("GLE Plot Preview")
            dialog.resize(850, 620)
            layout = QVBoxLayout(dialog)

            image_label = QLabel("Preview unavailable")
            layout.addWidget(image_label)

            if shutil.which("gle") is not None:
                gle_path = output_path.with_suffix(".gle")
                if gle_path.exists():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir_path = Path(tmpdir)
                        tmp_gle = tmpdir_path / gle_path.name
                        preview_png = tmp_gle.with_suffix(".png")

                        shutil.copy2(gle_path, tmp_gle)
                        for dep_name in self._extract_gle_data_dependencies(gle_path):
                            src = gle_path.parent / dep_name
                            if src.exists() and src.is_file():
                                shutil.copy2(src, tmpdir_path / dep_name)

                        subprocess.run(
                            ["gle", "-d", "png", str(tmp_gle)],
                            capture_output=True,
                            check=True,
                            cwd=str(tmpdir_path),
                        )

                        pixmap = QPixmap(str(preview_png))
                        if not pixmap.isNull():
                            image_label.setPixmap(pixmap)
                            image_label.setText("")

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec()
        except Exception as exc:
            QMessageBox.warning(self, "Preview error", f"Failed to show preview: {exc}")

    def _extract_gle_data_dependencies(self, gle_path: Path) -> list[str]:
        """Return data-file names referenced by `data <file>` commands in a GLE script."""
        try:
            text = gle_path.read_text(encoding="utf-8")
        except OSError:
            return []

        seen: set[str] = set()
        deps: list[str] = []
        pattern = r"^\s*data\s+(?:\"([^\"]+)\"|(\S+))"
        for match in re.finditer(pattern, text, flags=re.MULTILINE):
            token = (match.group(1) or match.group(2) or "").strip()
            name = Path(token).name
            if name and name not in seen:
                seen.add(name)
                deps.append(name)
        return deps

    def _compile_and_preview_gle(self, gle_path: Path, output_format: str) -> None:
        if shutil.which("gle") is None:
            QMessageBox.information(
                self,
                "GLE Not Installed",
                f"GLE script saved to {gle_path}. Install GLE to compile to {output_format.upper()}.",
            )
            return

        output_path = gle_path.with_suffix(f".{output_format}")
        try:
            subprocess.run(
                ["gle", "-d", output_format, str(gle_path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(gle_path.parent),
            )
            QMessageBox.information(
                self,
                "Export Successful",
                f"GLE plot exported:\n\nGLE script: {gle_path}\nOutput: {output_path}",
            )
            self._show_gle_preview(output_path)
        except subprocess.CalledProcessError as exc:
            QMessageBox.warning(self, "GLE compilation failed", exc.stderr or str(exc))
            self._show_gle_preview(output_path)

    def _export_plot_gle(
        self, *, title: str, default_name: str, builder, output_format: str
    ) -> None:
        if self._result is None or self._parameter_name is None:
            QMessageBox.information(self, "No result", "Run a cross-group fit first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_export_path(default_name),
            "GLE export folders (*.gleplot)",
        )
        if not path:
            return
        remember_export_path(path)

        try:
            glp = importlib.import_module("gleplot")
            requested_gle_path = Path(path)
            gle_path, export_dir = resolve_gle_export_paths(requested_gle_path, folder=True)
            export_dir.mkdir(parents=True, exist_ok=True)
            fig = builder(glp, gle_path)
            fig.savefig(str(gle_path))
            self._compile_and_preview_gle(gle_path, output_format)
        except ImportError:
            QMessageBox.warning(
                self, "gleplot not available", "Install gleplot to export GLE plots."
            )
        except TypeError as exc:
            if "folder" in str(exc):
                QMessageBox.warning(
                    self, "gleplot update required", "Please update gleplot to a newer version."
                )
                return
            QMessageBox.warning(self, "Export failed", f"Could not export GLE: {exc}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", f"Could not export GLE: {exc}")

    def _export_fit_subplot_gle(self) -> None:
        self._export_plot_gle(
            title="Export Fit Subplots to GLE",
            default_name="global_parameter_fit_subplots.gleplot",
            builder=self._build_fit_subplot_gle_figure,
            output_format=self._fit_gle_format_combo.currentText().lower(),
        )

    def _export_local_parameters_gle(self) -> None:
        self._export_plot_gle(
            title="Export Local Parameters Plot to GLE",
            default_name="global_parameter_fit_local_parameters.gleplot",
            builder=self._build_local_parameter_gle_figure,
            output_format=self._local_gle_format_combo.currentText().lower(),
        )
