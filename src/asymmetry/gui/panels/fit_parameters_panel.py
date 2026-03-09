"""Panel for inspecting fitted parameters across multiple datasets.

Shows a table of varying fit parameters and a plot of one fitted parameter versus
an inferred sweep variable (usually magnetic field or temperature).
"""

from __future__ import annotations

import csv
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
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import ParameterSet

_PARAM_SYMBOLS = {
    "A0": "A₀",
    "Lambda": "λ",
    "sigma": "σ",
    "Delta": "Δ",
    "beta": "β",
    "phase": "φ",
    "frequency": "f",
}

_PARAM_UNITS = {
    "A0": "%",
    "baseline": "%",
    "Lambda": "μs⁻¹",
    "sigma": "μs⁻¹",
    "Delta": "μs⁻¹",
    "frequency": "MHz",
    "phase": "rad",
}


def _format_param_label(name: str) -> str:
    """Return a display label with Greek symbols and units where applicable."""
    symbol = _PARAM_SYMBOLS.get(name, name)
    unit = _PARAM_UNITS.get(name)
    if unit:
        return f"{symbol} ({unit})"
    return symbol


def _format_gle_label(name: str) -> str:
    """Format a parameter label for GLE plots with proper italics and superscripts."""
    # Only italicize physical symbols, not full words.
    gle_labels = {
        "A0": "{\\it{A}}_{0} (%)",
        "Lambda": "{\\it{λ}} (μs^{-1})",  # λ printed as is, superscript -1
        "sigma": "{\\it{σ}} (μs^{-1})",
        "Delta": "{\\it{Δ}} (μs^{-1})",
        "beta": "{\\it{β}}",
        "phase": "{\\it{φ}} (rad)",
        "frequency": "{\\it{f}} (MHz)",
        "baseline": "baseline (%)",
    }

    if name in gle_labels:
        return gle_labels[name]

    # For unknown parameters, keep words plain and only italicize single-char symbols.
    unit = _PARAM_UNITS.get(name)
    display_name = f"{{\\it{{{name}}}}}" if len(name) == 1 else name
    if unit:
        unit_gle = unit.replace("μs⁻¹", "μs^{-1}")
        return f"{display_name} ({unit_gle})"
    return display_name


def _format_x_label_gle(x_key: str) -> str:
    """Format x-axis label for GLE plots."""
    if x_key == "field":
        return "{\\it{B}} (G)"
    elif x_key == "temperature":
        return "{\\it{T}} (K)"
    else:
        return "Run Number"


def _gle_series_color(index: int) -> str:
    """Return a deterministic color for GLE multi-series plots."""
    primary = ["black", "blue", "red"]
    fallback = ["green", "orange", "purple", "brown", "magenta", "cyan", "olive"]

    if index < len(primary):
        return primary[index]
    return fallback[(index - len(primary)) % len(fallback)]


@dataclass
class _FitRow:
    run_number: int
    field: float
    temperature: float
    values: dict[str, float]
    errors: dict[str, float]


class FitParametersPanel(QWidget):
    """Table + plot view for parameter trends from global fits."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        temp_label = "𝑇 (K)"
        field_label = "𝐵 (G)"  # Unicode italic B (U+1D435)

        self._rows: list[_FitRow] = []
        self._varying_params: list[str] = []
        self._global_params: ParameterSet | None = None
        self._table_dialog: QDialog | None = None
        self._inferred_x_key = "field"

        layout = QVBoxLayout(self)

        controls_group = QGroupBox("Parameter settings")
        controls_form = QFormLayout(controls_group)

        self._show_table_btn = QPushButton("Show fitted parameter table")
        self._show_table_btn.setEnabled(False)
        self._show_table_btn.clicked.connect(self._show_table_dialog)
        controls_form.addRow(self._show_table_btn)

        self._x_combo = QComboBox()
        self._x_combo.addItems(["Auto", field_label, temp_label, "Run"])
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

        self._y_combo = QListWidget()
        self._y_combo.itemSelectionChanged.connect(self._refresh_plot)
        self._y_combo.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._y_combo.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._log_y_check = QCheckBox("log")
        self._log_y_check.stateChanged.connect(self._refresh_plot)
        y_row = QHBoxLayout()
        y_row.addWidget(self._y_combo, 1)
        y_row.addWidget(self._log_y_check)
        y_container = QWidget()
        y_container.setLayout(y_row)
        controls_form.addRow("Y parameters:", y_container)

        # Plot mode selector (Single Axes vs Subplots)
        self._plot_mode_combo = QComboBox()
        self._plot_mode_combo.addItems(["Single Axes", "Subplots"])
        self._plot_mode_combo.currentTextChanged.connect(self._refresh_plot)
        controls_form.addRow("Plot mode:", self._plot_mode_combo)

        # Export buttons row
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

        # Keep table data model internal; display in a dialog on demand.
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

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

    def set_fit_results(
        self,
        results_dict: dict[int, tuple[FitResult, tuple[np.ndarray, np.ndarray]]],
        datasets_by_run: dict[int, MuonDataset],
        global_params: ParameterSet | None = None,
    ) -> None:
        """Load global-fit results and rebuild table/plot views.

        Parameters
        ----------
        results_dict : dict
            Dictionary mapping run_number to (FitResult, fitted_curve_tuple).
        datasets_by_run : dict
            Dictionary mapping run_number to MuonDataset.
        global_params : ParameterSet, optional
            The fitted global parameters shared across all datasets.
        """
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
            field = float(meta.get("field", 0.0))
            temperature = float(meta.get("temperature", 0.0))
            values = {p.name: p.value for p in fit_result.parameters}
            errors = dict(fit_result.uncertainties)
            self._rows.append(
                _FitRow(
                    run_number=run_number,
                    field=field,
                    temperature=temperature,
                    values=values,
                    errors=errors,
                )
            )

        self._rows.sort(key=lambda r: r.run_number)
        has_rows = bool(self._rows)
        self._show_table_btn.setEnabled(has_rows)
        self._export_csv_btn.setEnabled(has_rows)
        self._export_gle_btn.setEnabled(has_rows)
        self._gle_format_combo.setEnabled(has_rows)
        self._varying_params = self._detect_varying_parameters(self._rows)
        self._rebuild_y_parameter_combo()
        self._inferred_x_key = self._infer_x_key(self._rows)
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
        if selected == "𝐵 (G)":
            return "field"
        if selected == "𝑇 (K)":
            return "temperature"
        if selected == "Run":
            return "run"
        return self._inferred_x_key

    def _on_x_axis_changed(self, *_args: object) -> None:
        self._update_x_axis_auto_hint()
        self._refresh_views()

    def _update_x_axis_auto_hint(self) -> None:
        if self._x_combo.currentText() != "Auto":
            self._x_auto_hint.setText("")
            return

        inferred_label = {
            "field": "(𝐵)",
            "temperature": "(𝑇)",
            "run": "(Run)",
        }
        self._x_auto_hint.setText(inferred_label.get(self._inferred_x_key, "(Run)"))

    def _set_y_list_visible_rows(self, visible_rows: int = 3) -> None:
        visible_rows = max(1, visible_rows)
        row_height = self._y_combo.sizeHintForRow(0)
        if row_height <= 0:
            row_height = 20
        frame = 2 * self._y_combo.frameWidth()
        self._y_combo.setMinimumHeight(row_height * visible_rows + frame + 2)
        self._y_combo.setMaximumHeight(row_height * visible_rows + frame + 2)

    def _rebuild_y_parameter_combo(self) -> None:
        self._y_combo.blockSignals(True)
        self._y_combo.clear()
        for name in self._varying_params:
            item = QListWidgetItem(_format_param_label(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._y_combo.addItem(item)

        # Keep behavior intuitive: preselect first parameter when available.
        if self._y_combo.count() > 0:
            self._y_combo.item(0).setSelected(True)
        self._set_y_list_visible_rows(3)
        self._y_combo.blockSignals(False)

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
                err_candidates = [
                    row.errors.get(param.name, np.nan)
                    for row in self._rows
                ]
                finite_errs = [e for e in err_candidates if np.isfinite(e) and e > 0]
                if finite_errs:
                    value_text = f"({param.value:.6g} +/- {finite_errs[0]:.3g})"
                else:
                    value_text = f"{param.value:.6g}"

                unit = _PARAM_UNITS.get(param.name)
                unit_text = f" {unit}" if unit else ""
                lines.append(f"{param.name} = {value_text}{unit_text}")
            header_text = "\n".join(lines) if lines else "None"
        else:
            header_text = "None"

        header_label = QLabel(header_text)
        header_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        table_view = QTableWidget(self._table.rowCount(), self._table.columnCount(), dialog)
        table_view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_view.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_view.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        headers = [
            self._table.horizontalHeaderItem(col).text()
            for col in range(self._table.columnCount())
        ]
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
            display = _format_param_label(name)
            columns.append(display)
            columns.append(f"err {display}")

        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(str(row.run_number)))
            self._table.setItem(i, 1, QTableWidgetItem(f"{row.field:.6g}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{row.temperature:.6g}"))

            col = 3
            for name in self._varying_params:
                value = row.values.get(name, np.nan)
                err = row.errors.get(name, np.nan)
                self._table.setItem(i, col, QTableWidgetItem(f"{value:.6g}"))
                self._table.setItem(i, col + 1, QTableWidgetItem(f"{err:.3g}"))
                col += 2

        self._table.resizeColumnsToContents()

    def _refresh_plot(self) -> None:
        if not self._has_mpl:
            return

        if not self._rows or not self._varying_params:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.set_title("No varying fit parameters")
            self._canvas.draw()
            return

        y_params = self._get_selected_y_parameters()
        if not y_params:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.set_title("No varying fit parameters")
            self._canvas.draw()
            return

        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        x_vals = np.array([self._x_value(r, x_key) for r in rows], dtype=float)

        x_label = {
            "field": "$B$ (G)",
            "temperature": "$T$ (K)",
            "run": "Run Number",
        }[x_key]

        plot_mode = self._plot_mode_combo.currentText()

        if plot_mode == "Single Axes":
            # Single axes mode: all parameters on one plot
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.set_xlabel(x_label)

            if len(y_params) == 2:
                # For two selected parameters, use left/right y-axes for readability.
                left_name, right_name = y_params
                left_vals = np.array([r.values.get(left_name, np.nan) for r in rows], dtype=float)
                left_err = np.array([r.errors.get(left_name, np.nan) for r in rows], dtype=float)
                right_vals = np.array([r.values.get(right_name, np.nan) for r in rows], dtype=float)
                right_err = np.array([r.errors.get(right_name, np.nan) for r in rows], dtype=float)

                ax2 = ax.twinx()
                left_color = "C0"
                right_color = "C1"

                finite_left_err = np.isfinite(left_err) & (left_err > 0)
                ax.scatter(x_vals, left_vals, s=16, zorder=3, color=left_color)
                if np.any(finite_left_err):
                    ax.errorbar(
                        x_vals, left_vals, yerr=left_err, fmt="none", ecolor=left_color,
                        capsize=2, elinewidth=1, zorder=2,
                    )

                finite_right_err = np.isfinite(right_err) & (right_err > 0)
                ax2.scatter(x_vals, right_vals, s=16, zorder=3, color=right_color)
                if np.any(finite_right_err):
                    ax2.errorbar(
                        x_vals, right_vals, yerr=right_err, fmt="none", ecolor=right_color,
                        capsize=2, elinewidth=1, zorder=2,
                    )

                ax.set_ylabel(_format_param_label(left_name), color=left_color)
                ax2.set_ylabel(_format_param_label(right_name), color=right_color)
                ax.tick_params(axis="y", colors=left_color)
                ax2.tick_params(axis="y", colors=right_color)

                if self._log_y_check.isChecked():
                    ax.set_yscale("log")
                    ax2.set_yscale("log")
                else:
                    ax.set_yscale("linear")
                    ax2.set_yscale("linear")
            else:
                # One or 3+ parameters share a single y-axis.
                for y_name in y_params:
                    y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                    y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                    label = _format_param_label(y_name) if len(y_params) > 1 else None

                    finite_err = np.isfinite(y_err) & (y_err > 0)
                    if np.any(finite_err):
                        ax.scatter(x_vals, y_vals, s=16, zorder=3, label=label)
                        ax.errorbar(
                            x_vals, y_vals, yerr=y_err, fmt="none", ecolor="gray",
                            capsize=2, elinewidth=1, zorder=2,
                        )
                    else:
                        ax.scatter(x_vals, y_vals, s=16, zorder=3, label=label)

                if len(y_params) == 1:
                    ax.set_ylabel(_format_param_label(y_params[0]))
                else:
                    ax.set_ylabel("Parameter Value")

                if len(y_params) > 2:
                    ax.legend(loc="best")

                if self._log_y_check.isChecked():
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("linear")

            if self._log_x_check.isChecked():
                ax.set_xscale("log")
            else:
                ax.set_xscale("linear")

            ax.grid(True, alpha=0.3)

        else:
            # Subplots mode: one subplot per parameter
            self._figure.clear()

            # Calculate grid layout (prefer 2 columns)
            num_params = len(y_params)
            num_cols = 2
            num_rows = (num_params + num_cols - 1) // num_cols

            axes = []
            for idx, y_name in enumerate(y_params):
                ax = self._figure.add_subplot(num_rows, num_cols, idx + 1)
                axes.append(ax)

                # Plot this parameter on its own axis
                y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)

                finite_err = np.isfinite(y_err) & (y_err > 0)
                if np.any(finite_err):
                    ax.scatter(x_vals, y_vals, s=16, zorder=3, color='C0')
                    ax.errorbar(
                        x_vals, y_vals, yerr=y_err, fmt="none", ecolor="gray",
                        capsize=2, elinewidth=1, zorder=2,
                    )
                else:
                    ax.scatter(x_vals, y_vals, s=16, zorder=3, color='C0')

                ax.set_xlabel(x_label)
                ax.set_ylabel(_format_param_label(y_name))
                ax.set_title(_format_param_label(y_name))

                # Apply log scale if requested
                if self._log_x_check.isChecked():
                    ax.set_xscale('log')
                else:
                    ax.set_xscale('linear')

                if self._log_y_check.isChecked():
                    ax.set_yscale('log')
                else:
                    ax.set_yscale('linear')

                ax.grid(True, alpha=0.3)
                # No legend in subplots mode since each plot has only one series

        self._figure.tight_layout()
        self._canvas.draw()


    def _x_value(self, row: _FitRow, x_key: str) -> float:
        if x_key == "field":
            return row.field
        if x_key == "temperature":
            return row.temperature
        return float(row.run_number)

    def _get_selected_y_parameters(self) -> list[str]:
        """Get the list of parameters selected for plotting on y-axis.

        Returns all selected items from the Y parameters list widget.
        """
        selected_items = []
        for i in range(self._y_combo.count()):
            item = self._y_combo.item(i)
            if item.isSelected():
                param_name = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(param_name, str) and param_name:
                    selected_items.append(param_name)

        # If nothing is selected, return empty list
        if not selected_items:
            return []

        # Return in the order they appear in _varying_params
        return [p for p in self._varying_params if p in selected_items]

    def _needs_secondary_axis(self, y_values_list: list[np.ndarray]) -> bool:
        """Check if a secondary y-axis is needed for multiple series.

        Returns True if the value ranges differ significantly (ratio > 5).
        """
        if len(y_values_list) < 2:
            return False

        ranges = []
        for y_vals in y_values_list:
            finite_vals = y_vals[np.isfinite(y_vals)]
            if len(finite_vals) > 0:
                y_min, y_max = float(np.min(finite_vals)), float(np.max(finite_vals))
                if y_max > y_min:
                    ranges.append(y_max - y_min)

        if len(ranges) < 2:
            return False

        ranges.sort()
        return ranges[-1] / ranges[0] > 5 if ranges[0] > 0 else False

    def _export_csv(self) -> None:
        """Export the current table view to CSV."""
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

        headers = [
            self._table.horizontalHeaderItem(col).text()
            for col in range(self._table.columnCount())
        ]

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in range(self._table.rowCount()):
                values: list[str] = []
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    values.append(item.text() if item is not None else "")
                writer.writerow(values)

    def _export_gle(self) -> None:
        """Export the fit parameters as GLE plot with preview."""
        if not self._rows:
            return

        # Get the save path
        default_name = "fit_parameters.gle"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to GLE",
            default_name,
            "GLE files (*.gle);;All files (*)",
        )
        if not path:
            return

        gle_path = Path(path)
        data_path = gle_path.with_suffix(".dat")

        # Write data file with headers and comments
        self._write_gle_data_file(data_path)

        # Generate GLE plot
        output_format = self._gle_format_combo.currentText().lower()
        self._generate_gle_plot(gle_path, data_path, output_format)

    def _write_gle_data_file(self, data_path: Path) -> None:
        """Write a single well-documented data file consumed directly by GLE."""
        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))

        with open(data_path, "w", encoding="utf-8") as f:
            f.write("! Fit parameter data for GLE export\n")

            if self._global_params is not None:
                f.write("! Global Fitting Parameters (shared across all datasets):\n")
                for param in self._global_params:
                    f.write(f"!   {param.name} = {param.value:.6g}\n")
            f.write("!\n")

            x_label = {
                "field": "B_field(G)",
                "temperature": "Temperature(K)",
                "run": "Run",
            }[x_key]

            headers = [x_label]
            for name in self._varying_params:
                headers.append(name)
                headers.append(f"err_{name}")

            f.write("! Column map:\n")
            for col_idx, name in enumerate(headers, start=1):
                f.write(f"!   c{col_idx:>2} = {name}\n")
            f.write("!\n")
            f.write("! " + " ".join(f"{h:>16}" for h in headers) + "\n")

            for row in rows:
                values: list[float] = [self._x_value(row, x_key)]
                for name in self._varying_params:
                    values.append(row.values.get(name, np.nan))
                    values.append(row.errors.get(name, np.nan))
                f.write(" ".join(f"{v:>16.8g}" for v in values) + "\n")

    def _gle_columns_for_param(self, name: str) -> tuple[int, int] | None:
        """Return (value_col, error_col) for a varying parameter in fit_parameters.dat."""
        if name not in self._varying_params:
            return None
        idx = self._varying_params.index(name)
        value_col = 2 + idx * 2
        error_col = value_col + 1
        return value_col, error_col

    def _generate_gle_plot(self, gle_path: Path, data_path: Path, output_format: str) -> None:
        """Generate GLE plot and optionally compile it."""
        try:
            import gleplot as glp
        except ImportError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "gleplot not available",
                "gleplot is not installed. Please install it with:\n\npip install gleplot",
            )
            return

        # Plot by referencing columns from the generated fit_parameters.dat file.
        if not hasattr(glp, "Axes") or not hasattr(glp.Axes, "errorbar_from_file"):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "gleplot update required",
                "This export mode requires a newer gleplot with column-reference support.\n"
                "Please reinstall gleplot from the updated source.",
            )
            return

        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        y_params = self._get_selected_y_parameters()
        if not y_params:
            if self._varying_params:
                y_params = [self._varying_params[0]]
            else:
                return

        plot_mode = self._plot_mode_combo.currentText()
        x_label = _format_x_label_gle(x_key)
        data_file_ref = data_path.name

        if plot_mode == "Subplots" and len(y_params) > 1:
            num_rows = len(y_params)
            fig, axes = glp.subplots(
                nrows=num_rows,
                ncols=1,
                figsize=(5.5, 3.0 * num_rows),
                sharex=True,
            )
            subplot_axes = axes if isinstance(axes, list) else [axes]

            for idx, y_name in enumerate(y_params):
                ax = subplot_axes[idx]
                cols = self._gle_columns_for_param(y_name)
                if cols is None:
                    continue
                y_col, yerr_col = cols

                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))

                # One series per subplot: use black markers and error bars.
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=1,
                    y_col=y_col,
                    yerr_col=yerr_col if has_err else None,
                    color="black",
                    marker="o",
                    markersize=5,
                    capsize=2,
                )

                ax.set_xlabel(x_label)
                ax.set_ylabel(_format_gle_label(y_name))

                if self._log_x_check.isChecked():
                    ax.set_xscale("log")
                if self._log_y_check.isChecked():
                    ax.set_yscale("log")
        else:
            fig = glp.figure(figsize=(5.5, 4))
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

                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=1,
                    y_col=left_y_col,
                    yerr_col=left_err_col if has_left_err else None,
                    color=_gle_series_color(0),
                    marker="o",
                    markersize=5,
                    label=_format_gle_label(left_name),
                    capsize=2,
                    yaxis="y",
                )
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=1,
                    y_col=right_y_col,
                    yerr_col=right_err_col if has_right_err else None,
                    color=_gle_series_color(1),
                    marker="o",
                    markersize=5,
                    label=_format_gle_label(right_name),
                    capsize=2,
                    yaxis="y2",
                )

                ax.set_ylabel(_format_gle_label(left_name), axis="y")
                ax.set_ylabel(_format_gle_label(right_name), axis="y2")
                ax.legend(loc="best")
            else:
                for idx, y_name in enumerate(y_params):
                    cols = self._gle_columns_for_param(y_name)
                    if cols is None:
                        continue
                    y_col, yerr_col = cols

                    y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                    has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))
                    label = _format_gle_label(y_name) if len(y_params) > 1 else None

                    ax.errorbar_from_file(
                        data_file_ref,
                        x_col=1,
                        y_col=y_col,
                        yerr_col=yerr_col if has_err else None,
                        color=_gle_series_color(idx),
                        marker="o",
                        markersize=5,
                        label=label,
                        capsize=2,
                        yaxis="y",
                    )

                if len(y_params) == 1:
                    ax.set_ylabel(_format_gle_label(y_params[0]))
                else:
                    ax.set_ylabel("Parameter Value")
                    ax.legend(loc="best")

            ax.set_xlabel(x_label)

            if self._log_x_check.isChecked():
                ax.set_xscale("log")
            if self._log_y_check.isChecked():
                ax.set_yscale("log")

        # Save GLE script
        fig.savefig(str(gle_path))

        # Check if GLE is installed
        gle_installed = shutil.which("gle") is not None

        if gle_installed:
            # Compile to PDF or EPS
            output_path = gle_path.with_suffix(f".{output_format}")
            try:
                subprocess.run(
                    ["gle", "-d", output_format, str(gle_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Show success message
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"GLE plot exported successfully:\n\n"
                    f"GLE script: {gle_path}\n"
                    f"Data file: {data_path}\n"
                    f"Output: {output_path}",
                )

                # Show preview
                self._show_gle_preview(fig, data_path)

            except subprocess.CalledProcessError as e:
                from PySide6.QtWidgets import QMessageBox
                error_msg = e.stderr if e.stderr else str(e)

                # Check for common library issues
                if "libpoppler" in error_msg or "dyld" in error_msg:
                    msg_text = (
                        f"GLE compilation failed due to a library dependency issue.\n\n"
                        f"This is a common problem on macOS when GLE was built against\n"
                        f"an older version of poppler.\n\n"
                        f"To fix this, try reinstalling GLE:\n"
                        f"  brew reinstall gle\n\n"
                        f"The GLE script has been saved to:\n{gle_path}\n"
                        f"You can compile it manually after fixing GLE."
                    )
                else:
                    msg_text = f"Failed to compile GLE file:\n\n{error_msg}"

                QMessageBox.warning(
                    self,
                    "GLE Compilation Failed",
                    msg_text,
                )

                # Still show preview if possible
                try:
                    self._show_gle_preview(fig, data_path)
                except Exception:
                    pass
        else:
            # GLE not installed, just show preview
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("GLE Not Installed")
            msg.setText(
                f"GLE script saved to {gle_path}\n\n"
                "GLE is not installed, so the output file cannot be generated.\n"
                "Install GLE from http://glx.sourceforge.net/ to compile the script."
            )
            msg.exec()

            # Still show preview
            self._show_gle_preview(fig, data_path)

    def _show_gle_preview(self, fig, data_path: Path) -> None:
        """Show preview window using gleplot's view function."""
        try:
            # Create a dialog to show the preview
            dialog = QDialog(self)
            dialog.setWindowTitle("GLE Plot Preview")
            dialog.resize(800, 600)

            layout = QVBoxLayout(dialog)

            # Try to display the figure
            try:
                import tempfile

                from PySide6.QtGui import QPixmap
                from PySide6.QtWidgets import QLabel as QLabelWidget

                # Create a temporary directory for GLE compilation
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    gle_file = tmpdir_path / "preview.gle"
                    data_file = tmpdir_path / data_path.name
                    png_file = tmpdir_path / "preview.png"

                    # Copy data file to temp directory
                    shutil.copy2(data_path, data_file)

                    # Save GLE script to temp directory
                    fig.savefig(str(gle_file))

                    # Compile with GLE if available
                    if shutil.which("gle") is not None:
                        subprocess.run(
                            ["gle", "-d", "png", str(gle_file)],
                            capture_output=True,
                            check=True,
                            cwd=str(tmpdir_path),
                        )

                        # Load and display the image
                        pixmap = QPixmap(str(png_file))
                        if not pixmap.isNull():
                            label = QLabelWidget()
                            label.setPixmap(pixmap)
                            label.setScaledContents(False)
                            layout.addWidget(label)
                        else:
                            layout.addWidget(QLabelWidget("Failed to load preview image"))
                    else:
                        layout.addWidget(QLabelWidget("GLE not installed - preview not available"))

            except Exception as e:
                layout.addWidget(QLabelWidget(f"Preview error: {e}"))

            # Add close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Preview Error",
                f"Failed to show preview: {e}",
            )
