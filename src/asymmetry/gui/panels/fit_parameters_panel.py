"""Panel for inspecting fitted parameters across multiple datasets."""

from __future__ import annotations

import csv
import importlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

_PARAM_SYMBOLS = {
    "A0": "A₀",
    "A_bg": "A_bg",
    "Lambda": "λ",
    "sigma": "σ",
    "Delta": "Δ",
    "beta": "β",
    "phase": "φ",
    "frequency": "f",
}

_PARAM_UNITS = {
    "A0": "%",
    "A_bg": "%",
    "baseline": "%",
    "Lambda": "μs⁻¹",
    "sigma": "μs⁻¹",
    "Delta": "μs⁻¹",
    "frequency": "MHz",
    "phase": "rad",
}


def _format_param_label(name: str) -> str:
    symbol = _PARAM_SYMBOLS.get(name, name)
    unit = _PARAM_UNITS.get(name)
    return f"{symbol} ({unit})" if unit else symbol


def _format_plot_label(name: str) -> str:
    if name == "A_bg":
        return "$A_{bg}$ (%)"
    return _format_param_label(name)


def _format_plot_legend_label(name: str) -> str:
    if name == "A_bg":
        return "$A_{bg}$"
    return _PARAM_SYMBOLS.get(name, name)


def _format_gle_label(name: str) -> str:
    gle_labels = {
        "A0": "{\\it{A}}_{0} (%)",
        "A_bg": "{\\it{A}}_{bg} (%)",
        "Lambda": "{\\it{λ}} (μs^{-1})",
        "sigma": "{\\it{σ}} (μs^{-1})",
        "Delta": "{\\it{Δ}} (μs^{-1})",
        "beta": "{\\it{β}}",
        "phase": "{\\it{φ}} (rad)",
        "frequency": "{\\it{f}} (MHz)",
        "baseline": "baseline (%)",
    }
    return gle_labels.get(name, name)


def _format_gle_legend_label(name: str) -> str:
    gle_labels = {
        "A0": "{\\it{A}}_{0}",
        "A_bg": "{\\it{A}}_{bg}",
        "Lambda": "{\\it{λ}}",
        "sigma": "{\\it{σ}}",
        "Delta": "{\\it{Δ}}",
        "beta": "{\\it{β}}",
        "phase": "{\\it{φ}}",
        "frequency": "{\\it{f}}",
        "baseline": "baseline",
    }
    return gle_labels.get(name, name)


def _format_x_label_gle(x_key: str) -> str:
    if x_key == "field":
        return "{\\it{B}} (G)"
    if x_key == "temperature":
        return "{\\it{T}} (K)"
    return "Run Number"


def _gle_series_color(index: int) -> str:
    primary = ["black", "blue", "red"]
    fallback = ["green", "orange", "purple", "brown", "magenta", "cyan", "olive"]
    if index < len(primary):
        return primary[index]
    return fallback[(index - len(primary)) % len(fallback)]


def _fit_overlay_color(index: int) -> str:
    colors = ["red", "green", "orange", "purple", "brown", "magenta", "cyan"]
    return colors[index % len(colors)]


def _fit_overlay_label(param_name: str, index: int, total: int, *, gle: bool) -> str:
    base = _format_gle_legend_label(param_name) if gle else _format_plot_legend_label(param_name)
    suffix = "" if index == 0 else f" #{index + 1}"
    if total <= 1:
        suffix = ""
    return f"fit {base}{suffix}"


def _normalize_x_key(value: object) -> str:
    """Normalize persisted x-axis key to an internal identifier."""
    text = str(value or "").strip()
    if text == "field":
        return "field"
    if text == "temperature":
        return "temperature"
    if text == "run":
        return "run"
    return "run"


@dataclass
class _FitRow:
    run_number: int
    run_label: str
    field: float
    temperature: float
    values: dict[str, float]
    errors: dict[str, float]
    combined_from: list[int] | None = None


@dataclass
class _YParamControls:
    fit_button: QPushButton
    log: QCheckBox


class FitParametersPanel(QWidget):
    """Table + plot view for parameter trends from global fits."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._rows: list[_FitRow] = []
        self._varying_params: list[str] = []
        self._global_params: ParameterSet | None = None
        self._table_dialog: QDialog | None = None
        self._inferred_x_key = "field"
        self._y_controls: dict[str, _YParamControls] = {}
        self._model_fits: dict[str, ParameterModelFit] = {}

        layout = QVBoxLayout(self)

        controls_group = QGroupBox("Parameter settings")
        controls_form = QFormLayout(controls_group)

        self._show_table_btn = QPushButton("Show fitted parameter table")
        self._show_table_btn.setEnabled(False)
        self._show_table_btn.clicked.connect(self._show_table_dialog)
        controls_form.addRow(self._show_table_btn)

        self._x_combo = QComboBox()
        self._x_combo.addItems(["Auto", "𝐵 (G)", "𝑇 (K)", "Run"])
        self._x_combo.currentTextChanged.connect(self._on_x_axis_changed)
        self._x_auto_hint = QLabel("")
        self._log_x_check = QCheckBox("log")
        self._log_x_check.stateChanged.connect(self._refresh_plot)
        x_row = QHBoxLayout()
        x_row.addWidget(self._x_combo)
        x_row.addWidget(self._x_auto_hint)
        x_row.addStretch()
        x_row.addWidget(self._log_x_check)
        x_container = QWidget()
        x_container.setLayout(x_row)
        controls_form.addRow("X axis:", x_container)

        self._y_selector_table = QTableWidget(0, 3)
        self._y_selector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._y_selector_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._y_selector_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._y_selector_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._y_selector_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._y_selector_table.horizontalHeader().setVisible(False)
        self._y_selector_table.verticalHeader().setVisible(False)
        self._y_selector_table.itemSelectionChanged.connect(self._refresh_plot)

        y_header = self._y_selector_table.horizontalHeader()
        y_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        y_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        y_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._y_selector_table.setMinimumWidth(460)

        controls_form.addRow("Y parameters:", self._y_selector_table)

        self._plot_mode_combo = QComboBox()
        self._plot_mode_combo.addItems(["Single Axes", "Subplots"])
        self._plot_mode_combo.currentTextChanged.connect(self._refresh_plot)
        controls_form.addRow("Plot mode:", self._plot_mode_combo)

        # Hidden global log-y toggle used to mirror selected-series log state.
        self._log_y_check = QCheckBox("log")
        self._log_y_check.setVisible(False)
        self._log_y_check.stateChanged.connect(self._on_global_log_y_changed)

        self._export_csv_btn = QPushButton("Export CSV")
        self._export_csv_btn.setEnabled(False)
        self._export_csv_btn.clicked.connect(self._export_csv)

        self._export_gle_btn = QPushButton("Export to GLE")
        self._export_gle_btn.setEnabled(False)
        self._export_gle_btn.clicked.connect(self._export_gle)

        self._gle_format_combo = QComboBox()
        self._gle_format_combo.addItems(["PDF", "EPS"])
        self._gle_format_combo.setEnabled(False)

        export_row = QHBoxLayout()
        export_row.addWidget(self._export_csv_btn)
        export_row.addWidget(self._export_gle_btn)
        export_row.addWidget(QLabel("Format:"))
        export_row.addWidget(self._gle_format_combo)
        export_row.addStretch()
        export_container = QWidget()
        export_container.setLayout(export_row)
        controls_form.addRow("", export_container)

        layout.addWidget(controls_group)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        plot_group = QGroupBox("Parameter Plot")
        plot_layout = QVBoxLayout(plot_group)
        self._has_mpl = False
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._figure = Figure(tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            plot_layout.addWidget(self._canvas)
            self._has_mpl = True
        except ImportError:
            plot_layout.addWidget(QLabel("matplotlib not installed - plotting disabled"))
        layout.addWidget(plot_group)

        self._update_x_axis_auto_hint()

    def clear(self) -> None:
        self._rows = []
        self._varying_params = []
        self._global_params = None
        self._model_fits = {}
        self._show_table_btn.setEnabled(False)
        self._rebuild_y_controls()
        self._refresh_plot()

    def get_state(self) -> dict:
        rows = [
            {
                "run_number": int(row.run_number),
                "run_label": str(row.run_label),
                "field": float(row.field),
                "temperature": float(row.temperature),
                "values": {k: float(v) for k, v in row.values.items()},
                "errors": {k: float(v) for k, v in row.errors.items()},
                "combined_from": [int(v) for v in row.combined_from] if row.combined_from else None,
            }
            for row in self._rows
        ]

        selected_y = self._selected_y_parameters()
        log_y = [name for name, c in self._y_controls.items() if c.log.isChecked()]

        return {
            "rows": rows,
            "varying_params": list(self._varying_params),
            "inferred_x_key": self._inferred_x_key,
            "x_axis": self._x_combo.currentText(),
            "selected_y_params": selected_y,
            "log_x": bool(self._log_x_check.isChecked()),
            "log_y_params": log_y,
            "plot_mode": self._plot_mode_combo.currentText(),
            "model_fits": self._serialize_model_fits(),
        }

    def restore_state(self, state: dict) -> None:
        rows_data = state.get("rows", [])
        restored_rows: list[_FitRow] = []
        if isinstance(rows_data, list):
            for entry in rows_data:
                if not isinstance(entry, dict):
                    continue
                try:
                    restored_rows.append(
                        _FitRow(
                            run_number=int(entry.get("run_number", 0)),
                            run_label=str(entry.get("run_label") or entry.get("run_number", 0)),
                            field=float(entry.get("field", 0.0)),
                            temperature=float(entry.get("temperature", 0.0)),
                            values={str(k): float(v) for k, v in dict(entry.get("values", {})).items()},
                            errors={str(k): float(v) for k, v in dict(entry.get("errors", {})).items()},
                            combined_from=[int(v) for v in entry.get("combined_from", [])] if entry.get("combined_from") else None,
                        )
                    )
                except Exception:
                    continue

        self._rows = restored_rows
        self._show_table_btn.setEnabled(bool(self._rows))
        self._export_csv_btn.setEnabled(bool(self._rows))
        self._export_gle_btn.setEnabled(bool(self._rows))
        self._gle_format_combo.setEnabled(bool(self._rows))

        varying = state.get("varying_params", [])
        if isinstance(varying, list) and all(isinstance(v, str) for v in varying):
            self._varying_params = list(varying)
        else:
            self._varying_params = self._detect_varying_parameters(self._rows)

        inferred_x = state.get("inferred_x_key", "field")
        self._inferred_x_key = inferred_x if inferred_x in {"field", "temperature", "run"} else "field"

        self._rebuild_y_controls()

        selected_y = set(state.get("selected_y_params", []))
        for i in range(self._y_selector_table.rowCount()):
            item = self._y_selector_table.item(i, 0)
            if item is None:
                continue
            pname = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(pname, str):
                continue
            item.setSelected(pname in selected_y if selected_y else i == 0)

        log_y_state = state.get("log_y_params", [])
        log_y = set(log_y_state if isinstance(log_y_state, list) else [])
        for name, controls in self._y_controls.items():
            controls.log.setChecked(name in log_y)

        self._log_y_check.setChecked(bool(log_y))

        self._model_fits = self._deserialize_model_fits(state.get("model_fits", {}))
        self._refresh_model_fit_button_labels()

        x_axis = state.get("x_axis")
        if isinstance(x_axis, str):
            idx = self._x_combo.findText(x_axis)
            if idx >= 0:
                self._x_combo.setCurrentIndex(idx)

        self._log_x_check.setChecked(bool(state.get("log_x", False)))

        plot_mode = state.get("plot_mode")
        if isinstance(plot_mode, str):
            idx = self._plot_mode_combo.findText(plot_mode)
            if idx >= 0:
                self._plot_mode_combo.setCurrentIndex(idx)

        self._update_x_axis_auto_hint()
        self._refresh_views()

    def set_fit_results(
        self,
        results_dict: dict[int, tuple[FitResult, tuple[np.ndarray, np.ndarray]]],
        datasets_by_run: dict[int, MuonDataset],
        global_params: ParameterSet | None = None,
    ) -> None:
        self._rows = []
        self._global_params = global_params

        for run_number, (fit_result, _) in results_dict.items():
            try:
                run_number = int(run_number)
            except (TypeError, ValueError):
                continue
            if not fit_result.success:
                continue

            dataset = datasets_by_run.get(run_number)
            if dataset is None:
                continue

            meta = dataset.metadata
            values = {p.name: p.value for p in fit_result.parameters}
            errors = dict(fit_result.uncertainties)
            self._rows.append(
                _FitRow(
                    run_number=run_number,
                    run_label=str(dataset.metadata.get("run_label") or run_number),
                    field=float(meta.get("field", 0.0)),
                    temperature=float(meta.get("temperature", 0.0)),
                    values=values,
                    errors=errors,
                    combined_from=[int(v) for v in meta.get("combined_from", [])] if meta.get("combined_from") else None,
                )
            )

        self._rows.sort(key=lambda r: r.run_number)
        has_rows = bool(self._rows)
        self._show_table_btn.setEnabled(has_rows)
        self._export_csv_btn.setEnabled(has_rows)
        self._export_gle_btn.setEnabled(has_rows)
        self._gle_format_combo.setEnabled(has_rows)

        self._varying_params = self._detect_varying_parameters(self._rows)
        self._inferred_x_key = self._infer_x_key(self._rows)
        self._model_fits = {k: v for k, v in self._model_fits.items() if k in self._varying_params}

        self._rebuild_y_controls()
        self._refresh_model_fit_button_labels()
        self._update_x_axis_auto_hint()
        self._refresh_views()

    def _detect_varying_parameters(self, rows: list[_FitRow]) -> list[str]:
        if not rows:
            return []

        all_names = sorted(rows[0].values.keys())
        varying: list[str] = []
        for name in all_names:
            vals = [r.values.get(name, np.nan) for r in rows]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) < 2:
                continue
            span = max(vals) - min(vals)
            scale = max(1.0, max(abs(v) for v in vals))
            if span > 1e-9 * scale:
                varying.append(name)
        return varying

    def _infer_x_key(self, rows: list[_FitRow]) -> str:
        if len(rows) < 2:
            return "run"
        fields = np.array([r.field for r in rows], dtype=float)
        temps = np.array([r.temperature for r in rows], dtype=float)
        field_unique = len(np.unique(np.round(fields, 9)))
        temp_unique = len(np.unique(np.round(temps, 9)))
        field_span = float(np.nanmax(fields) - np.nanmin(fields))
        temp_span = float(np.nanmax(temps) - np.nanmin(temps))
        if field_unique > 1 and (field_span > temp_span or temp_unique <= 1):
            return "field"
        if temp_unique > 1:
            return "temperature"
        return "run"

    def _effective_x_key(self) -> str:
        selected = self._x_combo.currentText()
        if selected in {"B (G)", "𝐵 (G)"}:
            return "field"
        if selected in {"T (K)", "𝑇 (K)"}:
            return "temperature"
        if selected == "Run":
            return "run"
        return self._inferred_x_key

    def _on_global_log_y_changed(self, _state: int) -> None:
        enabled = self._log_y_check.isChecked()
        selected = set(self._selected_y_parameters())
        for name, controls in self._y_controls.items():
            if name in selected:
                controls.log.setChecked(enabled)
        self._refresh_plot()

    def _on_x_axis_changed(self, *_args: object) -> None:
        self._update_x_axis_auto_hint()
        self._refresh_views()

    def _update_x_axis_auto_hint(self) -> None:
        if self._x_combo.currentText() != "Auto":
            self._x_auto_hint.setText("")
            return
        inferred_label = {"field": "(B)", "temperature": "(T)", "run": "(Run)"}
        self._x_auto_hint.setText(inferred_label.get(self._inferred_x_key, "(Run)"))

    def _rebuild_y_controls(self) -> None:
        self._y_selector_table.blockSignals(True)
        self._y_selector_table.clearContents()
        self._y_selector_table.setRowCount(0)

        self._y_controls = {}

        if not self._varying_params:
            self._set_y_table_visible_rows(3)
            self._y_selector_table.blockSignals(False)
            return

        self._y_selector_table.setRowCount(len(self._varying_params))

        for idx, name in enumerate(self._varying_params):
            name_item = QTableWidgetItem(_format_param_label(name))
            name_item.setData(Qt.ItemDataRole.UserRole, name)
            self._y_selector_table.setItem(idx, 0, name_item)

            fit_button = QPushButton("Model Fit")
            fit_button.clicked.connect(lambda _checked=False, p=name: self._open_model_fit_dialog(p))
            self._y_selector_table.setCellWidget(idx, 1, fit_button)

            log_check = QCheckBox("log")
            log_check.stateChanged.connect(self._refresh_plot)

            log_container = QWidget()
            log_layout = QHBoxLayout(log_container)
            log_layout.setContentsMargins(0, 0, 0, 0)
            log_layout.addWidget(log_check)
            log_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._y_selector_table.setCellWidget(idx, 2, log_container)

            self._y_controls[name] = _YParamControls(
                fit_button=fit_button,
                log=log_check,
            )

        self._y_selector_table.resizeColumnsToContents()
        self._set_y_table_visible_rows(3)

        if self._y_selector_table.rowCount() > 0:
            item = self._y_selector_table.item(0, 0)
            if item is not None:
                item.setSelected(True)

        self._y_selector_table.blockSignals(False)

    def _set_y_table_visible_rows(self, visible_rows: int = 3) -> None:
        """Set selector table height to show at most ``visible_rows`` rows."""
        row_count = self._y_selector_table.rowCount()
        if row_count <= 0:
            rows = 1
        else:
            rows = min(max(1, visible_rows), row_count)
        row_height = self._y_selector_table.verticalHeader().defaultSectionSize()
        if self._y_selector_table.rowCount() > 0:
            row_height = max(row_height, self._y_selector_table.rowHeight(0))
        frame = 2 * self._y_selector_table.frameWidth()
        height = row_height * rows + frame + 2
        self._y_selector_table.setMinimumHeight(height)
        self._y_selector_table.setMaximumHeight(height)

    def _refresh_model_fit_button_labels(self) -> None:
        for name, controls in self._y_controls.items():
            fit = self._model_fits.get(name)
            if fit is not None and fit.active and self._has_successful_fit_curve(fit):
                controls.fit_button.setText("Model Fit*")
                controls.fit_button.setToolTip("Model fit active")
            else:
                controls.fit_button.setText("Model Fit")
                controls.fit_button.setToolTip("")

    def _has_successful_fit_curve(self, fit: ParameterModelFit) -> bool:
        for fit_range in fit.ranges:
            if fit_range.result is not None and fit_range.result.success:
                return True
        return False

    def _open_model_fit_dialog(self, param_name: str) -> None:
        if not self._rows:
            return

        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))

        x_vals = np.array([self._x_value(r, x_key) for r in rows], dtype=float)
        y_vals = np.array([r.values.get(param_name, np.nan) for r in rows], dtype=float)
        y_err = np.array([r.errors.get(param_name, np.nan) for r in rows], dtype=float)

        invalid_err = ~np.isfinite(y_err) | (y_err <= 0)
        if np.any(invalid_err):
            finite = np.abs(y_vals[np.isfinite(y_vals)])
            fallback = max(float(np.nanmedian(finite)) * 0.02, 1e-9) if finite.size else 1e-3
            y_err = y_err.copy()
            y_err[invalid_err] = fallback

        dialog = ModelFitDialog(
            parameter_name=param_name,
            x_key=x_key,
            x_values=x_vals,
            y_values=y_vals,
            y_errors=y_err,
            existing_fit=self._model_fits.get(param_name),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.was_removed():
            self._model_fits.pop(param_name, None)
        else:
            fit = dialog.get_model_fit()
            if fit is not None:
                self._model_fits[param_name] = fit

        self._refresh_model_fit_button_labels()
        self._refresh_plot()

    def _refresh_views(self) -> None:
        self._refresh_table()
        self._refresh_plot()

    def _refresh_table(self) -> None:
        if not self._rows:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))

        columns = ["Run", "𝐵 (G)", "𝑇 (K)"]
        for name in self._varying_params:
            label = _format_param_label(name)
            columns.extend([label, f"err {label}"])

        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(str(row.run_label)))
            self._table.setItem(i, 1, QTableWidgetItem(f"{row.field:.6g}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{row.temperature:.6g}"))
            col = 3
            for name in self._varying_params:
                val = row.values.get(name, np.nan)
                err = row.errors.get(name, np.nan)
                self._table.setItem(i, col, QTableWidgetItem(f"{val:.6g}"))
                self._table.setItem(i, col + 1, QTableWidgetItem(f"{err:.3g}"))
                col += 2

        self._table.resizeColumnsToContents()

    def _selected_y_parameters(self) -> list[str]:
        params: list[str] = []
        for row in sorted({index.row() for index in self._y_selector_table.selectedIndexes()}):
            item = self._y_selector_table.item(row, 0)
            if item is None:
                continue
            pname = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(pname, str) and pname:
                params.append(pname)
        return [p for p in self._varying_params if p in params]

    def _is_log_y_for(self, name: str) -> bool:
        controls = self._y_controls.get(name)
        if controls is None:
            return bool(self._log_y_check.isChecked())
        return bool(controls.log.isChecked() or self._log_y_check.isChecked())

    def _draw_model_overlay_mpl(self, ax, param_name: str, color: str = "red") -> None:
        fit = self._model_fits.get(param_name)
        if fit is None or not fit.active:
            return
        if fit.x_key != self._effective_x_key():
            return

        curves = self._sampled_fit_curves(param_name, x_key=fit.x_key, num_points=200)
        for idx, (_, xs, ys) in enumerate(curves):
            line_color = _fit_overlay_color(idx) if len(curves) > 1 else color
            ax.plot(xs, ys, linestyle="-", linewidth=1.5, color=line_color, alpha=0.9)

    def _refresh_plot(self) -> None:
        if not self._has_mpl:
            return

        y_params = self._selected_y_parameters()
        if not self._rows or not y_params:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.set_title("No varying fit parameters")
            self._canvas.draw()
            return

        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        x_vals = np.array([self._x_value(r, x_key) for r in rows], dtype=float)
        x_label = {"field": "$B$ (G)", "temperature": "$T$ (K)", "run": "Run Number"}[x_key]

        self._figure.clear()
        plot_mode = self._plot_mode_combo.currentText()

        if plot_mode == "Subplots" and len(y_params) > 1:
            num_params = len(y_params)
            num_cols = 2
            num_rows = (num_params + num_cols - 1) // num_cols

            for idx, y_name in enumerate(y_params):
                ax = self._figure.add_subplot(num_rows, num_cols, idx + 1)
                y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)

                ax.scatter(x_vals, y_vals, s=16, zorder=3, color="C0")
                finite_err = np.isfinite(y_err) & (y_err > 0)
                if np.any(finite_err):
                    ax.errorbar(x_vals, y_vals, yerr=y_err, fmt="none", ecolor="gray", capsize=2, elinewidth=1, zorder=2)

                self._draw_model_overlay_mpl(ax, y_name)

                ax.set_xlabel(x_label)
                ax.set_ylabel(_format_plot_label(y_name))
                ax.set_title(_format_plot_label(y_name))
                ax.set_xscale("log" if self._log_x_check.isChecked() else "linear")
                ax.set_yscale("log" if self._is_log_y_for(y_name) else "linear")
                ax.grid(True, alpha=0.3)
        else:
            ax = self._figure.add_subplot(111)
            ax.set_xlabel(x_label)

            if len(y_params) == 2:
                left_name, right_name = y_params
                left_vals = np.array([r.values.get(left_name, np.nan) for r in rows], dtype=float)
                left_err = np.array([r.errors.get(left_name, np.nan) for r in rows], dtype=float)
                right_vals = np.array([r.values.get(right_name, np.nan) for r in rows], dtype=float)
                right_err = np.array([r.errors.get(right_name, np.nan) for r in rows], dtype=float)

                ax2 = ax.twinx()
                left_color = "C0"
                right_color = "C1"

                ax.scatter(x_vals, left_vals, s=16, zorder=3, color=left_color)
                if np.any(np.isfinite(left_err) & (left_err > 0)):
                    ax.errorbar(x_vals, left_vals, yerr=left_err, fmt="none", ecolor=left_color, capsize=2, elinewidth=1, zorder=2)

                ax2.scatter(x_vals, right_vals, s=16, zorder=3, color=right_color)
                if np.any(np.isfinite(right_err) & (right_err > 0)):
                    ax2.errorbar(x_vals, right_vals, yerr=right_err, fmt="none", ecolor=right_color, capsize=2, elinewidth=1, zorder=2)

                self._draw_model_overlay_mpl(ax, left_name, color=left_color)
                self._draw_model_overlay_mpl(ax2, right_name, color=right_color)

                ax.set_ylabel(_format_plot_label(left_name), color=left_color)
                ax2.set_ylabel(_format_plot_label(right_name), color=right_color)
                ax.tick_params(axis="y", colors=left_color)
                ax2.tick_params(axis="y", colors=right_color)
                ax.set_yscale("log" if self._is_log_y_for(left_name) else "linear")
                ax2.set_yscale("log" if self._is_log_y_for(right_name) else "linear")
                ax.set_xscale("log" if self._log_x_check.isChecked() else "linear")
                ax.grid(True, alpha=0.3)
            else:
                for idx, y_name in enumerate(y_params):
                    y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                    y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                    color = f"C{idx % 10}"
                    label = _format_plot_legend_label(y_name) if len(y_params) > 1 else None

                    ax.scatter(x_vals, y_vals, s=16, zorder=3, label=label, color=color)
                    if np.any(np.isfinite(y_err) & (y_err > 0)):
                        ax.errorbar(x_vals, y_vals, yerr=y_err, fmt="none", ecolor=color, capsize=2, elinewidth=1, zorder=2)

                    self._draw_model_overlay_mpl(ax, y_name, color=color)

                if len(y_params) == 1:
                    ax.set_ylabel(_format_plot_label(y_params[0]))
                    ax.set_yscale("log" if self._is_log_y_for(y_params[0]) else "linear")
                else:
                    ax.set_ylabel("Parameter Value")
                    if len(y_params) > 2:
                        ax.legend(loc="best")
                    ax.set_yscale("log" if any(self._is_log_y_for(name) for name in y_params) else "linear")

                ax.set_xscale("log" if self._log_x_check.isChecked() else "linear")
                ax.grid(True, alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw()

    def _x_value(self, row: _FitRow, x_key: str) -> float:
        if x_key == "field":
            return row.field
        if x_key == "temperature":
            return row.temperature
        return float(row.run_number)

    def _show_table_dialog(self) -> None:
        if self._table.rowCount() == 0 or self._table.columnCount() == 0:
            return

        if self._table_dialog is not None:
            self._table_dialog.close()
            self._table_dialog = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Fitted Variable Parameters")
        dialog.resize(1000, 600)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        header_title = QLabel("Global fitting parameters")
        layout.addWidget(header_title)

        if self._global_params is not None:
            lines = []
            for param in self._global_params:
                unit = _PARAM_UNITS.get(param.name)
                unit_text = f" {unit}" if unit else ""
                lines.append(f"{param.name} = {param.value:.6g}{unit_text}")
            header_text = "\n".join(lines) if lines else "None"
        else:
            header_text = "None"

        header_label = QLabel(header_text)
        header_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        table_view = QTableWidget(self._table.rowCount(), self._table.columnCount(), dialog)
        table_view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        headers = [self._table.horizontalHeaderItem(col).text() for col in range(self._table.columnCount())]
        table_view.setHorizontalHeaderLabels(headers)

        for row in range(self._table.rowCount()):
            for col in range(self._table.columnCount()):
                source_item = self._table.item(row, col)
                text = source_item.text() if source_item is not None else ""
                table_view.setItem(row, col, QTableWidgetItem(text))

        table_view.resizeColumnsToContents()
        layout.addWidget(table_view)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        self._table_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _export_csv(self) -> None:
        if self._table.columnCount() == 0 or self._table.rowCount() == 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Parameter Table",
            "fit_parameters.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        headers = ["Run", "B (G)", "T (K)"]
        for name in self._varying_params:
            unit = _PARAM_UNITS.get(name)
            if unit:
                headers.extend([f"{name} ({unit})", f"err_{name} ({unit})"])
            else:
                headers.extend([name, f"err_{name}"])

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in range(self._table.rowCount()):
                values: list[str] = []
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    values.append(item.text() if item is not None else "")
                writer.writerow(values)

    def _serialize_model_fits(self) -> dict:
        payload: dict[str, dict] = {}
        for param_name, model_fit in self._model_fits.items():
            ranges_data = []
            for fit_range in model_fit.ranges:
                range_item = {
                    "x_min": fit_range.x_min,
                    "x_max": fit_range.x_max,
                    "component_names": list(fit_range.model.component_names),
                    "operators": list(fit_range.model.operators),
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

    def _deserialize_model_fits(self, state: object) -> dict[str, ParameterModelFit]:
        if not isinstance(state, dict):
            return {}

        restored: dict[str, ParameterModelFit] = {}
        for key, entry in state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue

            x_key = _normalize_x_key(entry.get("x_key", "run"))
            ranges_state = entry.get("ranges", [])
            if not isinstance(ranges_state, list):
                continue

            ranges: list[ModelFitRange] = []
            for range_state in ranges_state:
                if not isinstance(range_state, dict):
                    continue
                try:
                    model = ParameterCompositeModel(
                        component_names=list(range_state.get("component_names", [])),
                        operators=list(range_state.get("operators", [])),
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
                        x_min=float(range_state.get("x_min")) if range_state.get("x_min") is not None else None,
                        x_max=float(range_state.get("x_max")) if range_state.get("x_max") is not None else None,
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

    def _iter_active_fit_ranges(self, x_key: str, y_params: list[str] | None = None):
        """Yield (parameter_name, range_index, fit_range) for successful active fits."""
        allowed = set(y_params or self._varying_params)
        for pname, fit in self._model_fits.items():
            if pname not in allowed:
                continue
            if not fit.active or fit.x_key != x_key:
                continue
            for idx, fit_range in enumerate(fit.ranges):
                if fit_range.result is None or not fit_range.result.success:
                    continue
                yield pname, idx, fit_range

    def _count_fit_curves(self, x_key: str, y_params: list[str]) -> int:
        count = 0
        for pname in y_params:
            count += len(self._sampled_fit_curves(pname, x_key, num_points=200))
        return count

    def _count_fit_curves_for_param(self, x_key: str, param_name: str) -> int:
        return len(self._sampled_fit_curves(param_name, x_key, num_points=200))

    def _x_domain_for_sampling(self, x_key: str) -> tuple[float, float] | None:
        if not self._rows:
            return None
        x_vals = np.array([self._x_value(r, x_key) for r in self._rows], dtype=float)
        finite = x_vals[np.isfinite(x_vals)]
        if finite.size == 0:
            return None
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def _sample_fit_range_curve(
        self,
        fit_range: ModelFitRange,
        x_key: str,
        num_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        result = fit_range.result
        if result is None or not result.success:
            return None

        x_min = fit_range.x_min
        x_max = fit_range.x_max
        if x_min is None or x_max is None:
            domain = self._x_domain_for_sampling(x_key)
            if domain is None:
                return None
            if x_min is None:
                x_min = domain[0]
            if x_max is None:
                x_max = domain[1]

        if x_max <= x_min:
            return None

        xs = np.linspace(float(x_min), float(x_max), num=max(2, int(num_points)), dtype=float)
        kwargs = {p.name: p.value for p in result.parameters}
        try:
            ys = fit_range.model.function(xs, **kwargs)
        except KeyError:
            return None
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return None

        xs = xs[mask]
        ys = ys[mask]
        order = np.argsort(xs)
        return xs[order], ys[order]

    def _sampled_fit_curves(
        self,
        param_name: str,
        x_key: str,
        num_points: int = 200,
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        fit = self._model_fits.get(param_name)
        if fit is None or not fit.active or fit.x_key != x_key:
            return []

        curves: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, fit_range in enumerate(fit.ranges):
            sampled = self._sample_fit_range_curve(fit_range, x_key=x_key, num_points=num_points)
            if sampled is None:
                continue
            xs, ys = sampled
            curves.append((idx, xs, ys))
        return curves

    def _write_fit_files(self, gle_path: Path, x_key: str, y_params: list[str]) -> dict[tuple[str, int], Path]:
        """Write model-fit sidecar files with metadata and sampled curve points."""
        entries = list(self._iter_active_fit_ranges(x_key, y_params))
        if not entries:
            return {}

        unique_params = sorted({pname for pname, _, _ in entries})
        include_param_name = len(unique_params) > 1

        def _safe_token(name: str) -> str:
            token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
            return token or "fit"

        base_names: list[str] = []
        for pname, _, _fit_range in entries:
            if include_param_name:
                base_names.append(f"{gle_path.stem}_{_safe_token(pname)}")
            else:
                base_names.append(gle_path.stem)

        totals: dict[str, int] = {}
        for base in base_names:
            totals[base] = totals.get(base, 0) + 1

        seen: dict[str, int] = {}
        written: dict[tuple[str, int], Path] = {}
        for (pname, r_idx, fit_range), base in zip(entries, base_names, strict=True):
            seen[base] = seen.get(base, 0) + 1
            occurrence = seen[base]
            stem = base if occurrence == 1 else f"{base}_{occurrence}"
            fit_path = gle_path.with_name(f"{stem}.fit")

            if totals[base] == 1:
                fit_path = gle_path.with_name(f"{base}.fit")

            result = fit_range.result
            assert result is not None

            sampled = self._sample_fit_range_curve(fit_range, x_key=x_key, num_points=200)
            if sampled is None:
                continue
            x_vals, y_vals = sampled

            with open(fit_path, "w", encoding="utf-8") as f:
                f.write("! Parameter model fit curve\n")
                f.write("! Generated by Asymmetry (GLE-readable data file)\n")
                f.write(f"! parameter: {pname}\n")
                f.write(f"! x_variable: {x_key}\n")
                f.write(f"! x_min: {fit_range.x_min}\n")
                f.write(f"! x_max: {fit_range.x_max}\n")
                f.write(f"! model_function: {fit_range.model.formula_string()}\n")
                f.write(f"! chi_squared: {result.chi_squared:.8g}\n")
                f.write(f"! reduced_chi_squared: {result.reduced_chi_squared:.8g}\n")
                f.write("! fitted_parameters:\n")
                for p in result.parameters:
                    err = result.uncertainties.get(p.name, np.nan)
                    if np.isfinite(err):
                        f.write(f"!   {p.name} = {p.value:.8g} +/- {err:.4g}\n")
                    else:
                        f.write(f"!   {p.name} = {p.value:.8g}\n")
                for xv, yv in zip(x_vals, y_vals, strict=True):
                    f.write(f"{xv:.10g} {yv:.10g}\n")

            written[(pname, r_idx)] = fit_path

        return written

    def _write_gle_data_file(self, data_path: Path) -> None:
        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))

        with open(data_path, "w", encoding="utf-8") as f:
            f.write("! Fit parameter data for GLE export\n")

            if self._global_params is not None:
                f.write("! Global fitting parameters:\n")
                for param in self._global_params:
                    unit = _PARAM_UNITS.get(param.name)
                    label = f"{param.name} ({unit})" if unit else param.name
                    f.write(f"!   {label} = {param.value:.6g}\n")

            f.write("!\n")

            headers = ["Run", "B_field(G)", "Temperature(K)"]
            for name in self._varying_params:
                unit = _PARAM_UNITS.get(name)
                if unit:
                    headers.extend([f"{name} ({unit})", f"err_{name} ({unit})"])
                else:
                    headers.extend([name, f"err_{name}"])

            f.write("! Column map:\n")
            for col_idx, name in enumerate(headers, start=1):
                f.write(f"!   c{col_idx:>2} = {name}\n")
            combined_rows = [row for row in rows if row.combined_from]
            if combined_rows:
                f.write("!\n")
                f.write("! Combined run mapping:\n")
                for row in combined_rows:
                    combined_label = " + ".join(str(v) for v in row.combined_from or [])
                    f.write(f"!   {row.run_number} = {combined_label}\n")
            f.write("!\n")
            f.write("! " + " ".join(f"{h:>16}" for h in headers) + "\n")

            for row in rows:
                values: list[float] = [float(row.run_number), float(row.field), float(row.temperature)]
                for name in self._varying_params:
                    values.append(row.values.get(name, np.nan))
                    values.append(row.errors.get(name, np.nan))
                f.write(" ".join(f"{v:>16.8g}" for v in values) + "\n")

    def _gle_x_column(self, x_key: str) -> int:
        if x_key == "run":
            return 1
        if x_key == "field":
            return 2
        return 3

    def _gle_columns_for_param(self, name: str) -> tuple[int, int] | None:
        if name not in self._varying_params:
            return None
        idx = self._varying_params.index(name)
        value_col = 4 + idx * 2
        error_col = value_col + 1
        return value_col, error_col

    def _add_gle_model_overlay(
        self,
        ax,
        param_name: str,
        color: str,
        yaxis: str = "y",
        include_labels: bool = False,
        fit_file_map: dict[tuple[str, int], Path] | None = None,
    ) -> None:
        fit = self._model_fits.get(param_name)
        if fit is None or not fit.active:
            return
        if fit.x_key != self._effective_x_key():
            return

        curves = self._sampled_fit_curves(param_name, x_key=fit.x_key, num_points=200)
        for idx, (range_index, xs, ys) in enumerate(curves):
            line_color = _fit_overlay_color(idx) if len(curves) > 1 else color
            line_label = _fit_overlay_label(param_name, idx, len(curves), gle=True) if include_labels else None
            fit_path = None
            if fit_file_map is not None:
                fit_path = fit_file_map.get((param_name, int(range_index)))
            if fit_path is not None and hasattr(ax, "line_from_file"):
                ax.line_from_file(
                    fit_path.name,
                    x_col=1,
                    y_col=2,
                    color=line_color,
                    linestyle="-",
                    linewidth=1,
                    yaxis=yaxis,
                    label=line_label,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    linestyle="-",
                    color=line_color,
                    linewidth=1,
                    yaxis=yaxis,
                    label=line_label,
                )

    def _export_gle(self) -> None:
        if not self._rows:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to GLE",
            "fit_parameters.gle",
            "GLE files (*.gle);;All files (*)",
        )
        if not path:
            return

        gle_path = Path(path)
        data_path = gle_path.with_suffix(".dat")
        self._write_gle_data_file(data_path)
        x_key = self._effective_x_key()
        y_params = self._selected_y_parameters() or ([self._varying_params[0]] if self._varying_params else [])

        # Export fit sidecars for all active fits on this x-axis, regardless of
        # current y-selection, so restored projects always emit .fit files.
        all_active_fit_params = [
            pname
            for pname, fit in self._model_fits.items()
            if fit.active and _normalize_x_key(fit.x_key) == x_key
        ]
        fit_file_map = self._write_fit_files(gle_path, x_key, all_active_fit_params)

        # Make available to preview/notifications invoked during _generate_gle_plot.
        self._last_export_fit_files = list(fit_file_map.values())
        output_format = self._gle_format_combo.currentText().lower()
        self._generate_gle_plot(gle_path, data_path, output_format, fit_file_map)

    def _generate_gle_plot(
        self,
        gle_path: Path,
        data_path: Path,
        output_format: str,
        fit_file_map: dict[tuple[str, int], Path] | None = None,
    ) -> None:
        is_test_mode = bool(os.environ.get("PYTEST_CURRENT_TEST"))

        try:
            glp = importlib.import_module("gleplot")
        except ImportError:
            QMessageBox.warning(self, "gleplot not available", "Install gleplot to export GLE plots.")
            return

        if not hasattr(glp, "Axes") or not hasattr(glp.Axes, "errorbar_from_file"):
            QMessageBox.warning(self, "gleplot update required", "Please update gleplot to a newer version.")
            return

        if fit_file_map and (not hasattr(glp.Axes, "line_from_file")):
            QMessageBox.warning(self, "gleplot update required", "Please update gleplot to a newer version.")
            return

        x_key = self._effective_x_key()
        y_params = self._selected_y_parameters() or ([self._varying_params[0]] if self._varying_params else [])
        if not y_params:
            return

        x_label = _format_x_label_gle(x_key)
        data_file_ref = data_path.name
        x_col = self._gle_x_column(x_key)
        plot_mode = self._plot_mode_combo.currentText()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        show_fit_legend = self._count_fit_curves(x_key, y_params) > 1

        if plot_mode == "Subplots" and len(y_params) > 1:
            fig, axes = glp.subplots(nrows=len(y_params), ncols=1, figsize=(5.8, 3.0 * len(y_params)), sharex=True)
            subplot_axes = axes if isinstance(axes, list) else [axes]
            for idx, y_name in enumerate(y_params):
                cols = self._gle_columns_for_param(y_name)
                if cols is None:
                    continue
                y_col, yerr_col = cols
                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))
                ax = subplot_axes[idx]
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=x_col,
                    y_col=y_col,
                    yerr_col=yerr_col if has_err else None,
                    color="black",
                    marker="o",
                    markersize=5,
                    capsize=2,
                )
                show_subplot_fit_legend = self._count_fit_curves_for_param(x_key, y_name) > 1
                self._add_gle_model_overlay(
                    ax,
                    y_name,
                    color=_fit_overlay_color(0),
                    yaxis="y",
                    include_labels=show_subplot_fit_legend,
                    fit_file_map=fit_file_map,
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(_format_gle_label(y_name))
                if self._log_x_check.isChecked():
                    ax.set_xscale("log")
                if self._is_log_y_for(y_name):
                    ax.set_yscale("log")
                if show_subplot_fit_legend:
                    ax.legend(loc="best")
        else:
            fig = glp.figure(figsize=(5.8, 4.2))
            ax = fig.add_subplot(111)

            if len(y_params) == 2:
                left_name, right_name = y_params
                left_cols = self._gle_columns_for_param(left_name)
                right_cols = self._gle_columns_for_param(right_name)
                if left_cols is None or right_cols is None:
                    return
                left_y_col, left_err_col = left_cols
                right_y_col, right_err_col = right_cols

                left_err = np.array([r.errors.get(left_name, np.nan) for r in rows], dtype=float)
                right_err = np.array([r.errors.get(right_name, np.nan) for r in rows], dtype=float)
                has_left_err = bool(np.any(np.isfinite(left_err) & (left_err > 0)))
                has_right_err = bool(np.any(np.isfinite(right_err) & (right_err > 0)))

                ax.errorbar_from_file(data_file_ref, x_col=x_col, y_col=left_y_col, yerr_col=left_err_col if has_left_err else None, color=_gle_series_color(0), marker="o", markersize=5, capsize=2, yaxis="y")
                ax.errorbar_from_file(data_file_ref, x_col=x_col, y_col=right_y_col, yerr_col=right_err_col if has_right_err else None, color=_gle_series_color(1), marker="o", markersize=5, capsize=2, yaxis="y2")
                self._add_gle_model_overlay(
                    ax,
                    left_name,
                    color=_fit_overlay_color(0),
                    yaxis="y",
                    include_labels=show_fit_legend,
                    fit_file_map=fit_file_map,
                )
                self._add_gle_model_overlay(
                    ax,
                    right_name,
                    color=_fit_overlay_color(1),
                    yaxis="y2",
                    include_labels=show_fit_legend,
                    fit_file_map=fit_file_map,
                )
                ax.set_ylabel(_format_gle_label(left_name), axis="y")
                ax.set_ylabel(_format_gle_label(right_name), axis="y2")
                if show_fit_legend:
                    ax.legend(loc="best")
            else:
                for idx, y_name in enumerate(y_params):
                    cols = self._gle_columns_for_param(y_name)
                    if cols is None:
                        continue
                    y_col, yerr_col = cols
                    y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                    has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))
                    ax.errorbar_from_file(
                        data_file_ref,
                        x_col=x_col,
                        y_col=y_col,
                        yerr_col=yerr_col if has_err else None,
                        color=_gle_series_color(idx),
                        marker="o",
                        markersize=5,
                        capsize=2,
                    )
                    self._add_gle_model_overlay(
                        ax,
                        y_name,
                        color=_fit_overlay_color(idx),
                        yaxis="y",
                        include_labels=show_fit_legend,
                        fit_file_map=fit_file_map,
                    )

                if len(y_params) == 1:
                    ax.set_ylabel(_format_gle_label(y_params[0]))
                else:
                    ax.set_ylabel("Parameter Value")
                    if show_fit_legend:
                        ax.legend(loc="best")

                if len(y_params) == 1 and self._is_log_y_for(y_params[0]):
                    ax.set_yscale("log")
                elif len(y_params) > 1 and any(self._is_log_y_for(name) for name in y_params):
                    ax.set_yscale("log")

            ax.set_xlabel(x_label)
            if self._log_x_check.isChecked():
                ax.set_xscale("log")

        fig.savefig(str(gle_path))

        if shutil.which("gle") is not None:
            output_path = gle_path.with_suffix(f".{output_format}")
            try:
                subprocess.run(["gle", "-d", output_format, str(gle_path)], capture_output=True, text=True, check=True)
                if not is_test_mode:
                    fit_files = getattr(self, "_last_export_fit_files", [])
                    fit_files_text = "\n".join(str(p) for p in fit_files) if fit_files else "(none)"
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        (
                            f"GLE plot exported:\n\n"
                            f"GLE script: {gle_path}\n"
                            f"Data file: {data_path}\n"
                            f"Fit files:\n{fit_files_text}\n"
                            f"Output: {output_path}"
                        ),
                    )
                    self._show_gle_preview(fig, data_path, list(fit_file_map.values()) if fit_file_map else [])
            except subprocess.CalledProcessError as exc:
                if not is_test_mode:
                    QMessageBox.warning(self, "GLE compilation failed", exc.stderr or str(exc))
                    self._show_gle_preview(fig, data_path, list(fit_file_map.values()) if fit_file_map else [])
        else:
            if not is_test_mode:
                QMessageBox.information(
                    self,
                    "GLE Not Installed",
                    f"GLE script saved to {gle_path}. Install GLE to compile to {output_format.upper()}.",
                )
                self._show_gle_preview(fig, data_path, list(fit_file_map.values()) if fit_file_map else [])

    def _show_gle_preview(self, fig, data_path: Path, fit_files: list[Path] | None = None) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
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
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    gle_file = tmpdir_path / "preview.gle"
                    data_file = tmpdir_path / data_path.name
                    png_file = tmpdir_path / "preview.png"

                    shutil.copy2(data_path, data_file)
                    for fit_file in fit_files or []:
                        src = Path(fit_file)
                        if src.exists():
                            shutil.copy2(src, tmpdir_path / src.name)
                    fig.savefig(str(gle_file))
                    subprocess.run(["gle", "-d", "png", str(gle_file)], capture_output=True, check=True, cwd=str(tmpdir_path))

                    pixmap = QPixmap(str(png_file))
                    if not pixmap.isNull():
                        image_label.setPixmap(pixmap)
                        image_label.setText("")

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec()
        except Exception as exc:
            QMessageBox.warning(self, "Preview error", f"Failed to show preview: {exc}")
