"""Central plot panel using Matplotlib embedded in Qt.

Displays time-domain asymmetry with error bars and optional fit overlay,
similar to WiMDA's main plot area.

The bunch-factor control rebins the displayed data and also defines the
dataset passed to fitting in the GUI. The original MuonDataset is preserved,
so changing the bunch factor after fitting only changes the plotted data while
the existing fit curve remains overlaid.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.rebin import rebin
from asymmetry.gui.export_paths import default_export_path, remember_export_path

# Metadata fields available for dataset labelling in the legend.
_LABEL_FIELDS: list[tuple[str, str]] = [
    ("Run", "run"),
    ("Field (G)", "field"),
    ("Temperature (K)", "temperature"),
    ("Comment", "comment"),
]


class PlotPanel(QWidget):
    """Matplotlib canvas for time- and frequency-domain plots.

    Notes
    -----
    The bunch factor controls both the plotted representation and the dataset
    prepared for fitting in the GUI. The stored source dataset remains
    unchanged, and any rebinned fit dataset is produced as a temporary copy.
    """

    bunch_factor_changed = Signal(int)
    fit_range_changed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._figure = Figure(tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._ax = self._figure.add_subplot(111)
            self._ax.set_xlabel("Time (μs)")
            self._ax.set_ylabel("Asymmetry (%)")

            # Fit-range interaction state.
            self._fit_x_min: float | None = None
            self._fit_x_max: float | None = None
            self._fit_span_artist = None
            self._fit_min_handle = None
            self._fit_max_handle = None
            self._active_fit_handle: str | None = None
            self._drag_started = False

            # Add plot limit controls toolbar
            self._create_limit_controls()
            layout.addLayout(self._limit_toolbar)

            layout.addWidget(self._canvas)
            self._has_mpl = True

            # Store current dataset for rebunching
            self._current_dataset = None
            self._current_datasets: list[MuonDataset] = []
            self._limits_initialized = False

            # Legend label field preferences can be scoped per Data Group.
            self._active_label_group_id: str | None = None
            self._default_label_field: str = "run"
            self._label_field_by_group: dict[str, str] = {}

            # Store fit curve data to persist across redraws
            self._fit_curve = None  # (t_fit, y_fit, label) for single fits
            self._fit_curve_run_number = None
            self._fit_curves = {}   # {run_number: (t_fit, y_fit, label)} for global fits

            # Per-fit additive component curves for shading.
            self._fit_components = None  # list[(name, y_component)] for single fit
            self._fit_components_by_run = {}  # {run_number: list[(name, y_component)]}

            # Per-run fit metadata for export headers.
            self._fit_metadata: dict[int, dict] = {}  # {run_number: {formula, chi2, ...}}

            # Interactive plot labels (text annotations).
            self._default_annotations: list[dict] = []
            self._annotations_by_group: dict[str, list[dict]] = {}
            self._annotations: list[dict] = self._default_annotations
            self._active_annotation_idx: int | None = None
            self._annotation_drag_started = False

            # Cached arrays from the most recently plotted analysis dataset.
            self._last_plot_time = None
            self._last_plot_asymmetry = None
            self._last_plot_error = None
            self._last_low_count_mask = None

            self._canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion_notify)
            self._canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        except ImportError:
            from PySide6.QtWidgets import QLabel

            layout.addWidget(QLabel("matplotlib not installed — plotting disabled"))
            self._has_mpl = False

    def _create_limit_controls(self) -> None:
        """Create toolbar for adjusting plot limits.

        Uses a grid layout for compactness.
        """
        self._limit_toolbar = QGridLayout()
        self._limit_toolbar.setSpacing(4)  # Tight spacing
        self._limit_toolbar.setContentsMargins(4, 4, 4, 4)  # Minimal margins

        # X-axis limits
        self._limit_toolbar.addWidget(QLabel("X:"), 0, 0)
        self._x_min = QDoubleSpinBox()
        self._x_min.setRange(-1e6, 1e6)
        self._x_min.setDecimals(3)
        self._x_min.setValue(0.0)
        self._x_min.setSuffix(" μs")
        self._x_min.setMaximumWidth(80)  # Constrain width
        self._limit_toolbar.addWidget(self._x_min, 0, 1)

        self._limit_toolbar.addWidget(QLabel("–"), 0, 2)  # Dash separator

        self._x_max = QDoubleSpinBox()
        self._x_max.setRange(-1e6, 1e6)
        self._x_max.setDecimals(3)
        self._x_max.setValue(10.0)
        self._x_max.setSuffix(" μs")
        self._x_max.setMaximumWidth(80)
        self._limit_toolbar.addWidget(self._x_max, 0, 3)

        # Y-axis limits
        self._limit_toolbar.addWidget(QLabel("Y:"), 0, 4)
        self._y_min = QDoubleSpinBox()
        self._y_min.setRange(-1e6, 1e6)
        self._y_min.setDecimals(3)
        self._y_min.setValue(-30.0)
        self._y_min.setMaximumWidth(80)
        self._limit_toolbar.addWidget(self._y_min, 0, 5)

        self._limit_toolbar.addWidget(QLabel("–"), 0, 6)  # Dash separator

        self._y_max = QDoubleSpinBox()
        self._y_max.setRange(-1e6, 1e6)
        self._y_max.setDecimals(3)
        self._y_max.setValue(30.0)
        self._y_max.setMaximumWidth(80)
        self._limit_toolbar.addWidget(self._y_max, 0, 7)

        # Separate axis auto-scale controls (restored behavior).
        auto_x_btn = QPushButton("Auto X")
        auto_x_btn.clicked.connect(self._auto_x_limits)
        auto_x_btn.setMaximumWidth(65)
        self._limit_toolbar.addWidget(auto_x_btn, 0, 8)

        auto_y_btn = QPushButton("Auto Y")
        auto_y_btn.clicked.connect(self._auto_y_limits)
        auto_y_btn.setMaximumWidth(65)
        self._limit_toolbar.addWidget(auto_y_btn, 0, 9)

        # Apply limit changes immediately from spinbox edits.
        self._x_min.editingFinished.connect(self._apply_limits)
        self._x_max.editingFinished.connect(self._apply_limits)
        self._y_min.editingFinished.connect(self._apply_limits)
        self._y_max.editingFinished.connect(self._apply_limits)

        # Stretch to fill remaining space
        self._limit_toolbar.setColumnStretch(10, 1)
        self._limit_toolbar.addWidget(QWidget(), 0, 10)

        # Keep bunching control internal (hidden) for backward compatibility
        # with project state and tests; it is intentionally not shown in UI.
        self._bunch_factor = QSpinBox()
        self._bunch_factor.setRange(1, 1000)
        self._bunch_factor.setValue(1)
        self._bunch_factor.setMaximumWidth(60)
        self._bunch_factor.valueChanged.connect(self._on_bunch_changed)
        self._bunch_factor.hide()

        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._add_label_btn.setMaximumWidth(90)
        self._limit_toolbar.addWidget(self._add_label_btn, 1, 0)

        self._limit_toolbar.addWidget(QLabel("Label:"), 1, 1)
        self._label_field_combo = QComboBox()
        for display, key in _LABEL_FIELDS:
            self._label_field_combo.addItem(display, userData=key)
        self._label_field_combo.setMaximumWidth(140)
        self._label_field_combo.currentIndexChanged.connect(self._on_label_field_changed)
        self._limit_toolbar.addWidget(self._label_field_combo, 1, 2)

        # Export controls (row 1, right side)
        self._export_gle_btn = QPushButton("Export Plot(s) to GLE")
        self._export_gle_btn.setEnabled(False)
        self._export_gle_btn.clicked.connect(self.export_plots_to_gle)
        self._gle_format_combo = QComboBox()
        self._gle_format_combo.addItems(["PDF", "EPS"])
        self._gle_format_combo.setEnabled(False)

        export_row = QHBoxLayout()
        export_row.addWidget(self._export_gle_btn)
        export_row.addWidget(QLabel("Format:"))
        export_row.addWidget(self._gle_format_combo)
        export_row.addStretch()
        export_container = QWidget()
        export_container.setLayout(export_row)
        self._limit_toolbar.addWidget(export_container, 1, 4, 1, 7)

    def _dataset_label_for(self, dataset: MuonDataset) -> str:
        """Return the legend label for *dataset* using the selected label field."""
        field = self._label_field_combo.currentData()
        if field == "run":
            return str(dataset.run_label)
        run = dataset.run
        val = dataset.metadata.get(field)
        if val is None and run is not None:
            val = run.metadata.get(field)
        if val is None:
            return str(dataset.run_label)
        if field == "field":
            try:
                return f"{float(val):.1f} G"
            except (ValueError, TypeError):
                pass
        elif field == "temperature":
            try:
                return f"{float(val):.2f} K"
            except (ValueError, TypeError):
                pass
        return str(val)

    def _on_label_field_changed(self) -> None:
        """Re-draw the current plot using the newly selected label field."""
        field = self._label_field_combo.currentData()
        if field is None:
            field = "run"
        if self._active_label_group_id is None:
            self._default_label_field = str(field)
        else:
            self._label_field_by_group[str(self._active_label_group_id)] = str(field)

        if not self._has_mpl or not self._current_datasets:
            return
        self.plot_datasets(self._current_datasets)

    def set_active_label_group(self, group_id: str | None) -> None:
        """Switch legend label-field context between ungrouped and Data Group views."""
        if not self._has_mpl:
            return

        normalized_group_id = None if group_id is None else str(group_id)
        if normalized_group_id == self._active_label_group_id:
            return

        current_field = self._label_field_combo.currentData()
        if current_field is None:
            current_field = "run"

        # Persist the outgoing context before switching.
        if self._active_label_group_id is None:
            self._default_label_field = str(current_field)
        else:
            self._label_field_by_group[str(self._active_label_group_id)] = str(current_field)

        self._active_label_group_id = normalized_group_id
        if normalized_group_id is None:
            self._annotations = self._default_annotations
        else:
            self._annotations = self._annotations_by_group.setdefault(normalized_group_id, [])
        self._active_annotation_idx = None
        self._annotation_drag_started = False

        target_field = self._default_label_field
        if normalized_group_id is not None:
            target_field = self._label_field_by_group.get(normalized_group_id, self._default_label_field)

        idx = self._label_field_combo.findData(target_field)
        if idx < 0:
            idx = self._label_field_combo.findData("run")
            target_field = "run"
        if idx < 0:
            return

        self._label_field_combo.blockSignals(True)
        self._label_field_combo.setCurrentIndex(idx)
        self._label_field_combo.blockSignals(False)

        if self._active_label_group_id is None:
            self._default_label_field = str(target_field)
        else:
            self._label_field_by_group[str(self._active_label_group_id)] = str(target_field)

        if self._current_datasets:
            self._redraw_current_view()

    def _serialize_annotations(self, annotations: list[dict]) -> list[dict[str, object]]:
        """Return serializable annotation payload from in-memory annotation dicts."""
        return [
            {"x": ann["x"], "y": ann["y"], "text": ann["text"]}
            for ann in annotations
        ]

    def _deserialize_annotations(self, payload: object) -> list[dict]:
        """Return in-memory annotation dicts from serialized annotation payload."""
        restored: list[dict] = []
        if not isinstance(payload, list):
            return restored
        for ann in payload:
            if not isinstance(ann, dict):
                continue
            try:
                restored.append(
                    {
                        "x": float(ann.get("x", 0.0)),
                        "y": float(ann.get("y", 0.0)),
                        "text": str(ann.get("text", "")),
                        "artist": None,
                    }
                )
            except (TypeError, ValueError):
                continue
        return restored

    def get_analysis_dataset(self, dataset: MuonDataset | None) -> MuonDataset | None:
        """Return the dataset that should be used for plotting and fitting.

        When the bunch factor is 1, the original dataset is returned. For a
        larger bunch factor, a rebinned MuonDataset copy is returned with the
        same metadata and run association.
        """
        if dataset is None:
            return None

        bunch_factor = self._bunch_factor.value()
        if bunch_factor <= 1:
            return dataset

        time, asymmetry, error = rebin(
            dataset.time,
            dataset.asymmetry,
            dataset.error,
            bunch_factor,
        )
        return MuonDataset(
            time=time,
            asymmetry=asymmetry,
            error=error,
            metadata=dict(dataset.metadata),
            run=dataset.run,
        )

    def get_fit_dataset(self, dataset: MuonDataset | None) -> MuonDataset | None:
        """Return *dataset* restricted to the currently selected fit range."""
        if dataset is None:
            return None

        t_min, t_max = self.get_fit_range()
        if t_min is None or t_max is None:
            return dataset
        return dataset.time_range(t_min, t_max)

    def get_fit_range(self) -> tuple[float | None, float | None]:
        """Return the active fit range as (x_min, x_max)."""
        if self._fit_x_min is None or self._fit_x_max is None:
            return None, None
        return float(self._fit_x_min), float(self._fit_x_max)

    def _redraw_current_view(self) -> None:
        """Redraw using the active single- or multi-dataset context."""
        if len(self._current_datasets) > 1:
            self.plot_datasets(self._current_datasets)
        elif self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def set_fit_range(self, x_min: float, x_max: float) -> None:
        """Set fit range limits and refresh visual handles."""
        self._set_fit_range(x_min, x_max, emit_signal=True, redraw=True)

    def plot_datasets(self, datasets: list[MuonDataset]) -> None:
        """Plot multiple datasets on the same axes with per-dataset colours.

        Each dataset is assigned a colour from matplotlib's default cycle
        (C0, C1, …).  Any stored fit curve for a run is drawn in the same
        colour.  Low-count (grey) points are still drawn at reduced opacity.
        The axes limits are initialised from the combined data extent on the
        first call, then held fixed on subsequent redraws.

        Delegates to :meth:`plot_dataset` when *datasets* has exactly one
        entry so that the single-dataset code path (limit initialisation,
        fit-range clamping, etc.) is exercised unchanged.
        """
        if not self._has_mpl or not datasets:
            return
        if len(datasets) == 1:
            self.plot_dataset(datasets[0])
            return

        self._current_dataset = datasets[-1]
        self._current_datasets = list(datasets)
        self._ax.clear()

        all_times: list[np.ndarray] = []
        all_asym: list[np.ndarray] = []
        all_err: list[np.ndarray] = []
        all_low: list[np.ndarray] = []

        for i, dataset in enumerate(datasets):
            color = f"C{i % 10}"
            analysis_dataset = self.get_analysis_dataset(dataset)
            if analysis_dataset is None:
                continue

            time = analysis_dataset.time
            asymmetry = analysis_dataset.asymmetry
            error = analysis_dataset.error
            low_count_mask = self._low_count_mask_for_dataset(analysis_dataset)

            finite_mask = np.isfinite(time) & np.isfinite(asymmetry) & np.isfinite(error)
            valid_low = finite_mask & low_count_mask
            valid_main = finite_mask & ~low_count_mask

            if np.any(valid_low):
                self._ax.errorbar(
                    time[valid_low],
                    asymmetry[valid_low],
                    yerr=error[valid_low],
                    fmt=".",
                    markersize=3,
                    color="0.6",
                    ecolor="0.6",
                    label="_nolegend_",
                )

            draw_mask = valid_main if np.any(valid_main) else finite_mask
            self._ax.errorbar(
                time[draw_mask],
                asymmetry[draw_mask],
                yerr=error[draw_mask],
                fmt=".",
                markersize=3,
                color=color,
                label=self._dataset_label_for(dataset),
            )

            # Overlay fit curve in same colour; excluded from legend by "_" prefix.
            fit_to_plot = self._fit_curves.get(dataset.run_number)
            if fit_to_plot is None:
                if self._fit_curve is not None and self._fit_curve_run_number == dataset.run_number:
                    fit_to_plot = self._fit_curve
            if fit_to_plot is not None:
                t_fit, y_fit, fit_label = fit_to_plot
                self._ax.plot(t_fit, y_fit, '-', color=color, linewidth=2,
                              label="_nolegend_")

            if np.any(finite_mask):
                all_times.append(time[finite_mask])
                all_asym.append(asymmetry[finite_mask])
                all_err.append(error[finite_mask])
                all_low.append(low_count_mask[finite_mask])

        self._ax.set_xlabel("Time (μs)")
        self._ax.set_ylabel("Asymmetry (%)")
        self._draw_annotations()
        self._ax.legend()

        if all_times:
            self._last_plot_time = np.concatenate(all_times)
            self._last_plot_asymmetry = np.concatenate(all_asym)
            self._last_plot_error = np.concatenate(all_err)
            self._last_low_count_mask = np.concatenate(all_low)

            if not self._limits_initialized:
                t_all = self._last_plot_time
                a_all = self._last_plot_asymmetry
                e_all = self._last_plot_error
                x_min, x_max = float(t_all.min()), float(t_all.max())
                y_min = float((a_all - e_all).min())
                y_max = float((a_all + e_all).max())
                xpad = (x_max - x_min) * 0.05
                ypad = (y_max - y_min) * 0.05
                self._x_min.setValue(x_min - xpad)
                self._x_max.setValue(x_max + xpad)
                self._y_min.setValue(y_min - ypad)
                self._y_max.setValue(y_max + ypad)
                self._limits_initialized = True

            # Set fit range to span all datasets.
            all_t_min = float(self._last_plot_time.min())
            all_t_max = float(self._last_plot_time.max())
            if self._fit_x_min is None or self._fit_x_max is None:
                self._fit_x_min = all_t_min
                self._fit_x_max = all_t_max

        self._draw_fit_range_artists()
        self._apply_limits()

    def plot_dataset(self, dataset: MuonDataset) -> None:
        """Plot a dataset, optionally rebinned according to the bunch factor.

        The input dataset is stored unchanged as the current dataset. If the
        bunch factor is greater than 1, temporary rebinned arrays are created
        for plotting. The source dataset itself is never mutated.
        """
        if not self._has_mpl:
            return

        # Store the original dataset
        self._current_dataset = dataset
        self._current_datasets = [dataset]

        analysis_dataset = self.get_analysis_dataset(dataset)
        if analysis_dataset is None:
            return
        time = analysis_dataset.time
        asymmetry = analysis_dataset.asymmetry
        error = analysis_dataset.error
        low_count_mask = self._low_count_mask_for_dataset(analysis_dataset)

        self._last_plot_time = time
        self._last_plot_asymmetry = asymmetry
        self._last_plot_error = error
        self._last_low_count_mask = low_count_mask

        self._ax.clear()

        finite_mask = np.isfinite(time) & np.isfinite(asymmetry) & np.isfinite(error)
        valid_low = finite_mask & low_count_mask
        valid_main = finite_mask & ~low_count_mask

        if np.any(valid_low):
            self._ax.errorbar(
                time[valid_low],
                asymmetry[valid_low],
                yerr=error[valid_low],
                fmt=".",
                markersize=3,
                color="0.6",
                ecolor="0.6",
                label="_nolegend_",
            )

        draw_mask = valid_main if np.any(valid_main) else finite_mask
        self._ax.errorbar(
            time[draw_mask],
            asymmetry[draw_mask],
            yerr=error[draw_mask],
            fmt=".",
            markersize=3,
            label=self._dataset_label_for(dataset),
        )
        self._ax.set_xlabel("Time (μs)")
        self._ax.set_ylabel("Asymmetry (%)")

        # Re-plot fit curve if it exists (check both single and global fits)
        fit_to_plot = None
        if dataset.run_number in self._fit_curves:
            fit_to_plot = self._fit_curves[dataset.run_number]
        elif self._fit_curve is not None and self._fit_curve_run_number == dataset.run_number:
            fit_to_plot = self._fit_curve

        if fit_to_plot is not None:
            t_fit, y_fit, fit_label = fit_to_plot
            self._ax.plot(t_fit, y_fit, 'r-', linewidth=2, label=fit_label)

        self._draw_annotations()

        self._ax.legend()

        # Initialize limits once; preserve user-set limits on redraw.
        if not self._limits_initialized:
            x_min, x_max = float(time.min()), float(time.max())
            y_min = float((asymmetry - error).min())
            y_max = float((asymmetry + error).max())

            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05

            self._x_min.setValue(x_min - x_padding)
            self._x_max.setValue(x_max + x_padding)
            self._y_min.setValue(y_min - y_padding)
            self._y_max.setValue(y_max + y_padding)
            self._limits_initialized = True

        data_x_min = float(time.min())
        data_x_max = float(time.max())
        if self._fit_x_min is None or self._fit_x_max is None:
            self._fit_x_min = data_x_min
            self._fit_x_max = data_x_max
        else:
            self._fit_x_min = min(max(self._fit_x_min, data_x_min), data_x_max)
            self._fit_x_max = min(max(self._fit_x_max, data_x_min), data_x_max)
            if self._fit_x_min >= self._fit_x_max:
                self._fit_x_min = data_x_min
                self._fit_x_max = data_x_max

        self._draw_fit_range_artists()

        # Apply the limits
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Apply the specified axis limits to the plot."""
        if not self._has_mpl:
            return

        self._ax.set_xlim(self._x_min.value(), self._x_max.value())

        self._ax.set_ylim(self._y_min.value(), self._y_max.value())
        self._draw_fit_range_artists()
        self._canvas.draw()

    def _draw_annotations(self) -> None:
        """Recreate annotation artists on the active axis."""
        for ann in self._annotations:
            artist = self._ax.text(
                ann["x"],
                ann["y"],
                ann["text"],
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85},
                zorder=5,
            )
            ann["artist"] = artist

    def _auto_x_limits(self) -> None:
        """Auto-scale x-axis and update x-limit controls."""
        if not self._has_mpl:
            return

        self._ax.relim()
        self._ax.autoscale(enable=True, axis="x", tight=False)
        x_lim = self._ax.get_xlim()
        self._x_min.setValue(x_lim[0])
        self._x_max.setValue(x_lim[1])

        self._draw_fit_range_artists()
        self._canvas.draw()

    def _auto_y_limits(self) -> None:
        """Auto-scale y-axis from visible, non-low-count points only."""
        if not self._has_mpl:
            return

        if self._last_plot_time is None or self._last_plot_asymmetry is None or self._last_plot_error is None:
            return

        x_lo = float(self._x_min.value())
        x_hi = float(self._x_max.value())
        lo, hi = (x_lo, x_hi) if x_lo <= x_hi else (x_hi, x_lo)

        time = self._last_plot_time
        asymmetry = self._last_plot_asymmetry
        error = self._last_plot_error
        low_mask = self._last_low_count_mask
        if low_mask is None:
            low_mask = np.zeros_like(time, dtype=bool)

        mask = (
            np.isfinite(time)
            & np.isfinite(asymmetry)
            & np.isfinite(error)
            & (time >= lo)
            & (time <= hi)
            & (~low_mask)
        )

        if not np.any(mask):
            mask = np.isfinite(asymmetry) & np.isfinite(error) & (~low_mask)
        if not np.any(mask):
            mask = np.isfinite(asymmetry) & np.isfinite(error)
        if not np.any(mask):
            return

        y_min = float(np.min(asymmetry[mask] - error[mask]))
        y_max = float(np.max(asymmetry[mask] + error[mask]))
        if y_max <= y_min:
            delta = max(abs(y_min) * 0.05, 1e-6)
            y_min -= delta
            y_max += delta
        else:
            padding = (y_max - y_min) * 0.05
            y_min -= padding
            y_max += padding

        self._y_min.setValue(y_min)
        self._y_max.setValue(y_max)

        self._apply_limits()

    def _low_count_mask_for_dataset(self, dataset: MuonDataset) -> np.ndarray:
        """Return mask of low-count bins (plotted gray) for *dataset* points."""
        time = np.asarray(dataset.time, dtype=float)
        mask = np.zeros_like(time, dtype=bool)

        run = dataset.run
        if run is None or not isinstance(getattr(run, "grouping", None), dict):
            return mask

        grouping = run.grouping
        first_good = grouping.get("first_good_bin")
        last_good = grouping.get("last_good_bin")
        if first_good is None or last_good is None:
            return mask

        histograms = getattr(run, "histograms", None)
        if not histograms:
            return mask

        hist0 = histograms[0]
        axis = np.asarray(hist0.time_axis, dtype=float)
        if axis.size == 0:
            return mask

        try:
            lo_idx = max(0, int(first_good))
            hi_idx = min(int(last_good), axis.size - 1)
        except (TypeError, ValueError):
            return mask

        if lo_idx > hi_idx:
            return mask

        good_t_min = float(axis[lo_idx])
        good_t_max = float(axis[hi_idx])
        return (time < good_t_min) | (time > good_t_max)

    def _on_bunch_changed(self) -> None:
        """Re-plot and refresh fit inputs when the bunch factor changes."""
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)
        self.bunch_factor_changed.emit(self._bunch_factor.value())

    def _set_fit_range(
        self,
        x_min: float,
        x_max: float,
        *,
        emit_signal: bool,
        redraw: bool,
    ) -> None:
        """Set fit range with ordering, clamping, and optional signaling."""
        lo = float(min(x_min, x_max))
        hi = float(max(x_min, x_max))

        if self._current_dataset is not None:
            analysis_dataset = self.get_analysis_dataset(self._current_dataset)
            if analysis_dataset is not None and analysis_dataset.n_points > 0:
                data_x_min = float(analysis_dataset.time.min())
                data_x_max = float(analysis_dataset.time.max())
                lo = min(max(lo, data_x_min), data_x_max)
                hi = min(max(hi, data_x_min), data_x_max)

                if lo >= hi:
                    step = max((data_x_max - data_x_min) * 0.001, 1e-6)
                    if lo >= data_x_max:
                        lo = max(data_x_min, data_x_max - step)
                        hi = data_x_max
                    else:
                        hi = min(data_x_max, lo + step)

        self._fit_x_min = lo
        self._fit_x_max = hi

        if redraw:
            self._draw_fit_range_artists()
            self._canvas.draw_idle()

        if emit_signal:
            self.fit_range_changed.emit(self._fit_x_min, self._fit_x_max)

    def _draw_fit_range_artists(self) -> None:
        """Draw highlight and edge handles for the selected fit range."""
        if not self._has_mpl:
            return

        if self._fit_span_artist is not None:
            try:
                self._fit_span_artist.remove()
            except NotImplementedError:
                pass
            self._fit_span_artist = None
        if self._fit_min_handle is not None:
            try:
                self._fit_min_handle.remove()
            except NotImplementedError:
                pass
            self._fit_min_handle = None
        if self._fit_max_handle is not None:
            try:
                self._fit_max_handle.remove()
            except NotImplementedError:
                pass
            self._fit_max_handle = None

        if self._fit_x_min is None or self._fit_x_max is None:
            return

        self._fit_span_artist = self._ax.axvspan(
            self._fit_x_min,
            self._fit_x_max,
            color="gold",
            alpha=0.18,
            zorder=1,
        )
        self._fit_min_handle = self._ax.axvline(
            self._fit_x_min,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            zorder=4,
        )
        self._fit_max_handle = self._ax.axvline(
            self._fit_x_max,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            zorder=4,
        )

    def _detect_handle_hit(self, event) -> str | None:
        """Return which fit handle (min/max) was clicked, if any."""
        if (
            self._fit_x_min is None
            or self._fit_x_max is None
            or event.inaxes != self._ax
            or event.x is None
            or event.y is None
        ):
            return None

        min_px = self._ax.transData.transform((self._fit_x_min, 0.0))[0]
        max_px = self._ax.transData.transform((self._fit_x_max, 0.0))[0]
        tolerance_px = 8.0

        if abs(event.x - min_px) <= tolerance_px:
            return "min"
        if abs(event.x - max_px) <= tolerance_px:
            return "max"
        return None

    def _detect_annotation_hit(self, event) -> int | None:
        """Return annotation index hit by the mouse event, if any."""
        if event.inaxes != self._ax:
            return None
        for idx, ann in enumerate(self._annotations):
            artist = ann.get("artist")
            if artist is None:
                continue
            contains, _ = artist.contains(event)
            if contains:
                return idx
        return None

    def _add_annotation_at_event(self, event) -> None:
        """Prompt for label text and place an annotation at the click location."""
        if event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return

        text, ok = QInputDialog.getText(self, "Add Label", "Label text:")
        if not ok or not text.strip():
            return

        annotation = {
            "x": float(event.xdata),
            "y": float(event.ydata),
            "text": text.strip(),
            "artist": None,
        }
        self._annotations.append(annotation)
        self._add_label_btn.setChecked(False)
        self._redraw_current_view()

    def _edit_annotation(self, idx: int) -> None:
        """Edit an existing annotation label."""
        current = self._annotations[idx]["text"]
        text, ok = QInputDialog.getText(self, "Edit Label", "Label text:", text=current)
        if not ok or not text.strip():
            return
        self._annotations[idx]["text"] = text.strip()
        self._redraw_current_view()

    def _delete_annotation(self, idx: int) -> None:
        """Delete an annotation by index."""
        self._annotations.pop(idx)
        self._redraw_current_view()

    def _on_canvas_button_press(self, event) -> None:
        """Capture left-clicks on fit-range handles for drag/edit."""
        if not self._has_mpl:
            return

        if event.button == 3:
            ann_idx = self._detect_annotation_hit(event)
            if ann_idx is not None:
                self._delete_annotation(ann_idx)
            return

        if event.button != 1:
            return

        if self._add_label_btn.isChecked():
            self._add_annotation_at_event(event)
            return

        handle = self._detect_handle_hit(event)
        if handle is not None:
            self._active_fit_handle = handle
            self._drag_started = False
            return

        ann_idx = self._detect_annotation_hit(event)
        if ann_idx is not None:
            self._active_annotation_idx = ann_idx
            self._annotation_drag_started = False

    def _on_canvas_motion_notify(self, event) -> None:
        """Drag the active fit-range handle while the mouse moves."""
        if (
            self._active_fit_handle is not None
            and event.inaxes == self._ax
            and event.xdata is not None
        ):
            self._drag_started = True
            if self._active_fit_handle == "min":
                self._set_fit_range(event.xdata, self._fit_x_max, emit_signal=True, redraw=True)
            else:
                self._set_fit_range(self._fit_x_min, event.xdata, emit_signal=True, redraw=True)

        if (
            self._active_annotation_idx is not None
            and event.inaxes == self._ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            self._annotation_drag_started = True
            ann = self._annotations[self._active_annotation_idx]
            ann["x"] = float(event.xdata)
            ann["y"] = float(event.ydata)
            artist = ann.get("artist")
            if artist is not None:
                artist.set_position((ann["x"], ann["y"]))
                self._canvas.draw_idle()

    def _on_canvas_button_release(self, event) -> None:
        """End drag and open numeric editor on click without drag."""
        if self._active_fit_handle is not None:
            handle = self._active_fit_handle
            was_drag = self._drag_started

            self._active_fit_handle = None
            self._drag_started = False

            if not was_drag and event.button == 1:
                self._prompt_handle_value_edit(handle)

        if self._active_annotation_idx is None:
            return

        ann_idx = self._active_annotation_idx
        was_ann_drag = self._annotation_drag_started
        self._active_annotation_idx = None
        self._annotation_drag_started = False

        if not was_ann_drag and event.button == 1 and getattr(event, "dblclick", False):
            self._edit_annotation(ann_idx)

    def _prompt_handle_value_edit(self, handle: str) -> None:
        """Prompt for an exact fit-handle x-value."""
        if self._fit_x_min is None or self._fit_x_max is None:
            return

        current = self._fit_x_min if handle == "min" else self._fit_x_max
        value, ok = QInputDialog.getDouble(
            self,
            "Set Fit Range",
            "Fit x-value (μs):",
            float(current),
            -1e6,
            1e6,
            6,
        )
        if not ok:
            return

        if handle == "min":
            self._set_fit_range(value, self._fit_x_max, emit_signal=True, redraw=True)
        else:
            self._set_fit_range(self._fit_x_min, value, emit_signal=True, redraw=True)

    def plot_fit(
        self,
        t_fit,
        y_fit,
        label: str = "Fit",
        component_curves: list[tuple[str, object]] | None = None,
        fit_result: object | None = None,
        fit_function: str | None = None,
    ) -> None:
        """Overlay a fit curve on the current plot.

        The fit curve will be retained even when bunching or limits change.

        Parameters
        ----------
        t_fit : array
            Time points for the fit curve.
        y_fit : array
            Fitted asymmetry values.
        label : str, optional
            Label for the fit curve in the legend.
        fit_result : object, optional
            FitResult containing chi_squared, parameters, uncertainties, etc.
        """
        if not self._has_mpl:
            return

        # Store fit curve data for persistence across redraws (single fit)
        self._fit_curve = (t_fit, y_fit, label)
        run_number = None
        if self._current_dataset is not None:
            try:
                run_number = int(self._current_dataset.run_number)
            except (TypeError, ValueError):
                run_number = None
        self._fit_curve_run_number = run_number

        if run_number is not None:
            self._fit_curves[run_number] = (t_fit, y_fit, label)
            self._fit_components_by_run[run_number] = list(component_curves or [])
            if fit_result is not None or fit_function:
                self._store_fit_metadata(run_number, fit_result, fit_function=fit_function)

        self._fit_components = list(component_curves or [])

        self._update_export_enabled()

        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)
        else:
            self._ax.plot(t_fit, y_fit, 'r-', linewidth=2, label=label)
            self._ax.legend()
            self._canvas.draw()

    def set_global_fits(self, fit_curves_dict: dict) -> None:
        """Set fit curves from global fitting.

        Parameters
        ----------
        fit_curves_dict : dict
            Dictionary mapping run_number -> (t_fit, y_fit, label, component_curves),
            (t_fit, y_fit, label, component_curves, fit_result), or
            (t_fit, y_fit, label, component_curves, fit_result, fit_function).
        """
        if not self._has_mpl:
            return

        # Update fit curves, preserving results from other groups
        for run_number, payload in fit_curves_dict.items():
            if len(payload) >= 6:
                t_fit, y_fit, label, component_curves, fit_result, fit_function = payload[:6]
            elif len(payload) >= 5:
                t_fit, y_fit, label, component_curves, fit_result = payload[:5]
                fit_function = None
            elif len(payload) == 4:
                t_fit, y_fit, label, component_curves = payload
                fit_result = None
                fit_function = None
            else:
                t_fit, y_fit, label = payload
                component_curves = []
                fit_result = None
                fit_function = None
            self._fit_curves[run_number] = (t_fit, y_fit, label)
            self._fit_components_by_run[run_number] = list(component_curves or [])
            if fit_result is not None or fit_function:
                self._store_fit_metadata(run_number, fit_result, fit_function=fit_function)
        # Clear single fit curve
        self._fit_curve = None
        self._fit_curve_run_number = None
        self._fit_components = None

        self._update_export_enabled()

        # Redraw current view while preserving multi-selection overlays.
        if len(self._current_datasets) > 1:
            self.plot_datasets(self._current_datasets)
        elif self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def clear(self) -> None:
        """Clear the plot and reset stored data."""
        if self._has_mpl:
            self._ax.clear()
            self._canvas.draw()
            self._current_dataset = None
            self._current_datasets = []
            self._fit_curve = None
            self._fit_curve_run_number = None
            self._fit_curves = {}
            self._fit_components = None
            self._fit_components_by_run = {}
            self._fit_metadata = {}
            self._limits_initialized = False
            self._last_plot_time = None
            self._last_plot_asymmetry = None
            self._last_plot_error = None
            self._last_low_count_mask = None
            self._default_annotations = []
            self._annotations_by_group = {}
            self._annotations = self._default_annotations
            self._active_annotation_idx = None
            self._annotation_drag_started = False
            self._fit_x_min = None
            self._fit_x_max = None
            self._fit_span_artist = None
            self._fit_min_handle = None
            self._fit_max_handle = None
            self._update_export_enabled()

    def clear_fit(self) -> None:
        """Clear all fit curves and redraw the plot."""
        if not self._has_mpl:
            return

        self._fit_curve = None
        self._fit_curve_run_number = None
        self._fit_curves = {}
        self._fit_components = None
        self._fit_components_by_run = {}
        self._fit_metadata = {}
        self._update_export_enabled()
        self._redraw_current_view()

    def clear_fits_for_runs(self, run_numbers: list[int]) -> int:
        """Clear stored fit overlays for the provided run numbers."""
        if not self._has_mpl:
            return 0

        normalized_runs: set[int] = set()
        for run_number in run_numbers:
            try:
                normalized_runs.add(int(run_number))
            except (TypeError, ValueError):
                continue

        if not normalized_runs:
            return 0

        removed = 0
        for run_number in normalized_runs:
            if self._fit_curves.pop(run_number, None) is not None:
                removed += 1
            self._fit_components_by_run.pop(run_number, None)
            self._fit_metadata.pop(run_number, None)

        if self._fit_curve_run_number in normalized_runs:
            self._fit_curve = None
            self._fit_curve_run_number = None
            self._fit_components = None
            removed += 1

        if removed > 0:
            self._update_export_enabled()
            if len(self._current_datasets) > 1:
                self.plot_datasets(self._current_datasets)
            elif self._current_dataset is not None:
                self.plot_dataset(self._current_dataset)

        return removed

    def _store_fit_metadata(
        self,
        run_number: int,
        fit_result: object | None,
        fit_function: str | None = None,
    ) -> None:
        """Extract and store fit metadata from a FitResult for export headers."""
        meta: dict = {}
        if fit_result is not None:
            chi2 = getattr(fit_result, "chi_squared", None)
            red_chi2 = getattr(fit_result, "reduced_chi_squared", None)
            if chi2 is not None:
                meta["chi_squared"] = float(chi2)
            if red_chi2 is not None:
                meta["reduced_chi_squared"] = float(red_chi2)

            params = getattr(fit_result, "parameters", None)
            uncertainties = getattr(fit_result, "uncertainties", {})
            if params is not None:
                meta["parameters"] = [
                    {
                        "name": p.name,
                        "value": float(p.value),
                        "error": float(uncertainties.get(p.name, float("nan"))),
                    }
                    for p in params
                ]

        fit_function_value = fit_function
        if not fit_function_value:
            fit_function_value = getattr(fit_result, "fit_function", None)
        if fit_function_value:
            meta["fit_function"] = str(fit_function_value)
        self._fit_metadata[int(run_number)] = meta

    def _update_export_enabled(self) -> None:
        """Enable the export button when any displayed dataset has a fit curve."""
        has_fit = bool(self._fit_curves) or self._fit_curve is not None
        self._export_gle_btn.setEnabled(has_fit)
        self._gle_format_combo.setEnabled(has_fit)

    @staticmethod
    def _safe_file_token(value: str) -> str:
        """Sanitize a string for use in a filename."""
        token = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(value).strip()
        )
        token = "_".join(part for part in token.split("_") if part)
        return token or "dataset"

    @staticmethod
    def _sanitize_gle_text(value: object, *, fallback: str = "") -> str:
        """Return text that is safe for GLE string rendering."""
        text = str(value)
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        text = text.replace("\r", " ").replace("\n", " ")
        text = text.replace("μ", "u").replace("µ", "u")
        text = text.replace("χ", "chi").replace("²", "^2")
        text = "".join(ch for ch in text if ch.isprintable())
        text = " ".join(text.split())
        text = text.encode("ascii", "ignore").decode("ascii")
        return text or fallback

    def get_current_plot_export_data(self) -> list[dict] | None:
        """Return export payloads for all displayed datasets with fits.

        Returns a list of dicts (one per dataset with a fit), or *None* when
        nothing is available to export.
        """
        if not self._current_datasets:
            return None

        payloads: list[dict] = []
        for dataset in self._current_datasets:
            analysis = self.get_analysis_dataset(dataset)
            if analysis is None:
                continue

            rn = dataset.run_number
            fit_data = self._fit_curves.get(rn)
            if fit_data is None:
                if self._fit_curve is not None and self._fit_curve_run_number == rn:
                    fit_data = self._fit_curve
            if fit_data is None:
                continue

            t_fit, y_fit, fit_label = fit_data
            component_data = self._fit_components_by_run.get(rn) or []
            if not component_data and self._fit_components and self._fit_curve_run_number == rn:
                component_data = self._fit_components or []

            label_text = self._dataset_label_for(dataset)

            payloads.append({
                "run_number": rn,
                "label": label_text,
                "data": {
                    "t": analysis.time,
                    "y": analysis.asymmetry,
                    "err": analysis.error,
                },
                "fit": {"t": t_fit, "y": y_fit, "label": fit_label},
                "components": [
                    {"name": name, "y": y_vals} for name, y_vals in component_data
                ],
                "fit_metadata": self._fit_metadata.get(rn, {}),
            })

        if not payloads:
            return None

        # Append annotations (shared across all datasets in this view)
        annotations = [
            {"x": ann["x"], "y": ann["y"], "text": ann["text"]}
            for ann in self._annotations
        ]
        for p in payloads:
            p["annotations"] = annotations

        return payloads

    def _write_fit_file(self, fit_path: Path, payload: dict) -> None:
        """Write a .fit file with fit-curve data and metadata header."""
        fit = payload.get("fit") or {}
        t_fit = fit.get("t")
        y_fit = fit.get("y")
        if t_fit is None or y_fit is None:
            return

        meta = payload.get("fit_metadata") or {}
        with open(fit_path, "w", encoding="utf-8") as f:
            f.write(f"! Fit curve for {payload.get('label', 'dataset')}\n")
            f.write(f"! run_number: {payload.get('run_number', '')}\n")
            fit_function = meta.get("fit_function") or fit.get("label") or "Fit"
            f.write(f"! fit_function: {fit_function}\n")
            chi2 = meta.get("chi_squared")
            red_chi2 = meta.get("reduced_chi_squared")
            if chi2 is not None:
                f.write(f"! chi_squared: {chi2:.8g}\n")
            if red_chi2 is not None:
                f.write(f"! reduced_chi_squared: {red_chi2:.8g}\n")
            params = meta.get("parameters")
            if params:
                f.write("! fitted_parameters:\n")
                for p in params:
                    err = p.get("error", float("nan"))
                    if np.isfinite(err):
                        f.write(f"!   {p['name']} = {p['value']:.8g} +/- {err:.4g}\n")
                    else:
                        f.write(f"!   {p['name']} = {p['value']:.8g}\n")
            f.write("!\n")
            f.write("! time  asymmetry_fit\n")
            for t_val, y_val in zip(t_fit, y_fit):
                f.write(f"{float(t_val):.10g} {float(y_val):.10g}\n")

    def _show_export_result_dialog(self, title: str, summary: str, details: str) -> None:
        """Show export results with scrollable details and fixed bottom button."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.resize(760, 460)

        layout = QVBoxLayout(dialog)
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        details_view = QTextEdit()
        details_view.setReadOnly(True)
        details_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        details_view.setPlainText(details)
        details_view.setMinimumHeight(180)
        details_view.setMaximumHeight(280)
        layout.addWidget(details_view)

        button_row = QHBoxLayout()
        button_row.addStretch()
        close_btn = QPushButton("OK")
        close_btn.clicked.connect(dialog.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        dialog.exec()

    @staticmethod
    def _export_figure_size(series_count: int) -> tuple[float, float]:
        """Return a figure size that grows taller for crowded multi-series exports."""
        width = 6.0
        if series_count <= 1:
            return width, 4.2

        # Increase height with number of overlaid spectra while capping growth.
        # This keeps multi-series plots closer to square for readability.
        height = 4.2 + min(3.0, 0.55 * float(series_count - 1))
        height = max(height, width * 0.85)
        height = min(height, 7.2)
        return width, height

    def _extract_gle_data_dependencies(self, gle_path: Path) -> list[str]:
        """Return data-file names referenced by `data <file>` commands."""
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

    def _show_gle_preview(self, gle_path: Path) -> None:
        """Show an in-app preview dialog for an exported GLE plot."""
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if not gle_path.exists():
            return
        if shutil.which("gle") is None:
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
        except Exception:
            # Preview is best-effort only; export should still succeed.
            return

    def export_plots_to_gle(self) -> None:
        """Export current main-plot view as GLE using gleplot.

        Data is plotted with error bars (no connecting lines), fit curves
        with lines (no markers).  File names are derived from the Label
        dropdown value for each dataset.
        """
        payloads = self.get_current_plot_export_data()
        if not payloads:
            QMessageBox.warning(
                self, "Export unavailable",
                "No fitted curve is available to export.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot(s) to GLE",
            default_export_path("asymmetry_plot.gle"),
            "GLE files (*.gle)",
        )
        if not path:
            return
        remember_export_path(path)

        try:
            glp = importlib.import_module("gleplot")
        except ImportError:
            QMessageBox.warning(
                self, "gleplot not available",
                "Install gleplot to export GLE plots.",
            )
            return

        gle_path = Path(path)
        output_format = self._gle_format_combo.currentText().lower()
        is_multi = len(payloads) > 1
        colors = [
            "black", "red", "blue", "green", "orange", "purple",
            "cyan", "magenta", "brown", "gray",
        ]

        fig = glp.figure(figsize=self._export_figure_size(len(payloads)))
        ax = fig.add_subplot(111)
        written_files: list[Path] = []

        for i, payload in enumerate(payloads):
            label_text = payload.get("label", f"dataset_{i}")
            safe_label = self._sanitize_gle_text(
                label_text,
                fallback=f"Run {payload.get('run_number', i)}",
            )
            token = self._safe_file_token(label_text)
            data_color = colors[i % len(colors)] if is_multi else "black"
            fit_color = data_color if is_multi else "red"

            data = payload.get("data") or {}
            fit = payload.get("fit") or {}
            t_data = data.get("t")
            y_data = data.get("y")
            y_err = data.get("err")
            t_fit = fit.get("t")
            y_fit = fit.get("y")

            # Write .dat data file
            dat_path = gle_path.parent / f"{token}.dat"
            if t_data is not None and y_data is not None:
                with open(dat_path, "w", encoding="utf-8") as f:
                    f.write(f"! Data for {label_text}\n")
                    f.write(f"! run_number: {payload.get('run_number', '')}\n")
                    f.write("! time  asymmetry  error\n")
                    err_arr = y_err if y_err is not None else np.zeros_like(y_data)
                    for t_val, y_val, e_val in zip(t_data, y_data, err_arr):
                        f.write(f"{float(t_val):.10g} {float(y_val):.10g} {float(e_val):.10g}\n")
                written_files.append(dat_path)

                ax.errorbar(
                    t_data,
                    y_data,
                    yerr=y_err,
                    fmt="none",
                    marker="o",
                    color=data_color,
                    markersize=4,
                    capsize=2,
                    label=safe_label,
                    data_name=token,
                )

            # Write .fit file with header info
            fit_path = gle_path.parent / f"{token}.fit"
            if t_fit is not None and y_fit is not None:
                self._write_fit_file(fit_path, payload)
                written_files.append(fit_path)

                fit_label = fit.get("label", "Fit")
                if is_multi:
                    fit_label = None
                else:
                    fit_label = self._sanitize_gle_text(fit_label, fallback="Fit")
                ax.plot(
                    t_fit,
                    y_fit,
                    color=fit_color,
                    linewidth=1.6,
                    label=fit_label,
                    data_name=f"{token}_fit",
                )

        # Add annotations
        annotations = payloads[0].get("annotations") or []
        for ann in annotations:
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            text = str(ann.get("text", "")).strip()
            text = self._sanitize_gle_text(text)
            if text:
                ax.text(x, y, text, color="black", ha="left")

        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Asymmetry (%)")
        ax.legend(loc="best")

        # Preserve the visible GUI window limits in the exported GLE plot.
        if hasattr(ax, "set_xlim"):
            ax.set_xlim(float(self._x_min.value()), float(self._x_max.value()))
        if hasattr(ax, "set_ylim"):
            ax.set_ylim(float(self._y_min.value()), float(self._y_max.value()))

        fig.savefig(str(gle_path))

        # Compile using gleplot / GLE
        if shutil.which("gle") is not None:
            output_path = gle_path.with_suffix(f".{output_format}")
            try:
                subprocess.run(
                    ["gle", "-d", output_format, str(gle_path)],
                    capture_output=True, text=True, check=True,
                )
                files_text = "\n".join(str(p) for p in written_files)
                self._show_export_result_dialog(
                    "Export Successful",
                    "GLE plot exported successfully.",
                    (
                        f"GLE script: {gle_path}\n"
                        f"Output: {output_path}\n\n"
                        f"Data/fit files:\n{files_text}"
                    ),
                )
                self._show_gle_preview(gle_path)
            except subprocess.CalledProcessError as exc:
                QMessageBox.warning(
                    self, "GLE compilation failed",
                    exc.stderr or str(exc),
                )
        else:
            QMessageBox.information(
                self,
                "GLE Not Installed",
                f"GLE script saved to {gle_path}.\nInstall GLE to compile to {output_format.upper()}.",
            )

    # Keep old name as alias for backward compatibility with tests.
    def export_current_plot(self) -> None:
        """Export current main-plot view as GLE (with optional compiled output)."""
        self.export_plots_to_gle()

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the plot panel state.

        This captures the bunch factor, axis limits, the currently displayed
        run number, and any stored fit curves.  Fit curve arrays are
        serialised as plain Python lists for JSON compatibility.

        Returns
        -------
        dict
            Plot state suitable for inclusion in a project file.
        """
        state: dict = {
            "current_run_number": (
                self._current_dataset.run_number
                if self._current_dataset is not None
                else None
            ),
            "label_field": self._label_field_combo.currentData() if self._has_mpl else "run",
            "default_label_field": self._default_label_field,
            "label_field_by_group": dict(self._label_field_by_group),
            "bunch_factor": self._bunch_factor.value() if self._has_mpl else 1,
            "x_min": self._x_min.value() if self._has_mpl else 0.0,
            "x_max": self._x_max.value() if self._has_mpl else 10.0,
            "y_min": self._y_min.value() if self._has_mpl else -30.0,
            "y_max": self._y_max.value() if self._has_mpl else 30.0,
            "fit_curve": None,
            "fit_curve_run_number": self._fit_curve_run_number,
            "fit_curves": {},
            "fit_components": None,
            "fit_components_by_run": {},
            "annotations": [],
            "annotations_by_group": {},
            "fit_x_min": self._fit_x_min,
            "fit_x_max": self._fit_x_max,
        }

        if self._has_mpl:
            if self._fit_curve is not None:
                t_fit, y_fit, label = self._fit_curve
                state["fit_curve"] = {
                    "t": list(t_fit),
                    "y": list(y_fit),
                    "label": label,
                }
            for run_number, (t_fit, y_fit, label) in self._fit_curves.items():
                state["fit_curves"][str(run_number)] = {
                    "t": list(t_fit),
                    "y": list(y_fit),
                    "label": label,
                }
            if self._fit_components is not None:
                state["fit_components"] = [
                    {"name": name, "y": list(y_vals)}
                    for name, y_vals in self._fit_components
                ]
            for run_number, curves in self._fit_components_by_run.items():
                state["fit_components_by_run"][str(run_number)] = [
                    {"name": name, "y": list(y_vals)}
                    for name, y_vals in curves
                ]
            state["annotations"] = self._serialize_annotations(self._default_annotations)
            state["annotations_by_group"] = {
                str(group_id): self._serialize_annotations(annotations)
                for group_id, annotations in self._annotations_by_group.items()
            }
            state["fit_metadata"] = {
                str(rn): meta for rn, meta in self._fit_metadata.items()
            }

        return state

    def restore_state(
        self,
        state: dict,
        dataset: "MuonDataset | None" = None,
    ) -> None:
        """Restore plot panel state from a saved dict.

        Parameters
        ----------
        state : dict
            Plot state as returned by :meth:`get_state`.
        dataset : MuonDataset, optional
            Dataset to re-plot after restoring limits.  If *None* no plot is
            drawn, but all other state (limits, bunch factor, fit curves) is
            still applied.
        """
        if not self._has_mpl:
            return

        import numpy as np

        valid_label_fields = {key for _, key in _LABEL_FIELDS}
        default_label_field = state.get("default_label_field", state.get("label_field", "run"))
        if default_label_field not in valid_label_fields:
            default_label_field = "run"
        self._default_label_field = str(default_label_field)

        raw_group_label_fields = state.get("label_field_by_group", {})
        self._label_field_by_group = {}
        if isinstance(raw_group_label_fields, dict):
            for group_id, field in raw_group_label_fields.items():
                if field in valid_label_fields:
                    self._label_field_by_group[str(group_id)] = str(field)

        self._active_label_group_id = None

        label_field = state.get("label_field", self._default_label_field)
        if label_field not in valid_label_fields:
            label_field = "run"
        idx = self._label_field_combo.findData(label_field)
        if idx < 0:
            idx = self._label_field_combo.findData("run")
        if idx >= 0:
            self._label_field_combo.blockSignals(True)
            self._label_field_combo.setCurrentIndex(idx)
            self._label_field_combo.blockSignals(False)
            selected_field = self._label_field_combo.currentData()
            if selected_field in valid_label_fields:
                self._default_label_field = str(selected_field)

        # Keep bunch factor at default in restored projects; control is hidden.
        self._bunch_factor.blockSignals(True)
        self._bunch_factor.setValue(1)
        self._bunch_factor.blockSignals(False)

        # Restore axis limit spinboxes (will be applied after optional re-plot).
        for spin, key, default in (
            (self._x_min, "x_min", 0.0),
            (self._x_max, "x_max", 10.0),
            (self._y_min, "y_min", -30.0),
            (self._y_max, "y_max", 30.0),
        ):
            spin.blockSignals(True)
            spin.setValue(state.get(key, default))
            spin.blockSignals(False)

        # Treat restored limits as user-defined so later dataset additions do
        # not overwrite them with auto-derived bounds.
        self._limits_initialized = True

        fit_x_min = state.get("fit_x_min")
        fit_x_max = state.get("fit_x_max")
        if fit_x_min is not None and fit_x_max is not None:
            self._fit_x_min = float(fit_x_min)
            self._fit_x_max = float(fit_x_max)

        # Restore fit curves.
        self._fit_curve = None
        self._fit_curve_run_number = None
        self._fit_curves = {}
        self._fit_components = None
        self._fit_components_by_run = {}

        fit_curve_data = state.get("fit_curve")
        if fit_curve_data:
            self._fit_curve = (
                np.array(fit_curve_data["t"]),
                np.array(fit_curve_data["y"]),
                fit_curve_data.get("label", "Fit"),
            )
            fit_curve_run_number = state.get("fit_curve_run_number")
            if fit_curve_run_number is not None:
                try:
                    self._fit_curve_run_number = int(fit_curve_run_number)
                except (TypeError, ValueError):
                    self._fit_curve_run_number = None

        for run_str, curve_data in state.get("fit_curves", {}).items():
            self._fit_curves[int(run_str)] = (
                np.array(curve_data["t"]),
                np.array(curve_data["y"]),
                curve_data.get("label", "Global Fit"),
            )

        fit_components = state.get("fit_components")
        if isinstance(fit_components, list):
            self._fit_components = [
                (entry.get("name", "Component"), np.array(entry.get("y", []), dtype=float))
                for entry in fit_components
                if isinstance(entry, dict)
            ]

        for run_str, entries in state.get("fit_components_by_run", {}).items():
            if not isinstance(entries, list):
                continue
            self._fit_components_by_run[int(run_str)] = [
                (entry.get("name", "Component"), np.array(entry.get("y", []), dtype=float))
                for entry in entries
                if isinstance(entry, dict)
            ]

        self._default_annotations = self._deserialize_annotations(state.get("annotations", []))
        raw_annotations_by_group = state.get("annotations_by_group", {})
        self._annotations_by_group = {}
        if isinstance(raw_annotations_by_group, dict):
            for group_id, payload in raw_annotations_by_group.items():
                self._annotations_by_group[str(group_id)] = self._deserialize_annotations(payload)
        self._annotations = self._default_annotations
        self._active_annotation_idx = None
        self._annotation_drag_started = False

        # Restore fit metadata.
        self._fit_metadata = {}
        raw_fit_metadata = state.get("fit_metadata", {})
        if isinstance(raw_fit_metadata, dict):
            for rn_str, meta in raw_fit_metadata.items():
                if isinstance(meta, dict):
                    try:
                        self._fit_metadata[int(rn_str)] = meta
                    except (TypeError, ValueError):
                        pass

        self._update_export_enabled()

        # Re-plot the current dataset if one was provided.
        if dataset is not None:
            self._current_dataset = dataset
            self.plot_dataset(dataset)

        # Re-apply saved axis limits after dataset redraw, which may reset
        # spinbox values to data-derived defaults.
        for spin, key, default in (
            (self._x_min, "x_min", 0.0),
            (self._x_max, "x_max", 10.0),
            (self._y_min, "y_min", -30.0),
            (self._y_max, "y_max", 30.0),
        ):
            spin.blockSignals(True)
            spin.setValue(state.get(key, default))
            spin.blockSignals(False)

        if fit_x_min is not None and fit_x_max is not None:
            self._set_fit_range(
                float(fit_x_min),
                float(fit_x_max),
                emit_signal=False,
                redraw=True,
            )

        # Always apply the restored limits.
        self._apply_limits()
