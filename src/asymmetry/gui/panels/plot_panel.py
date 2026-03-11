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
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.rebin import rebin


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

            # Store fit curve data to persist across redraws
            self._fit_curve = None  # (t_fit, y_fit, label) for single fits
            self._fit_curves = {}   # {run_number: (t_fit, y_fit, label)} for global fits

            # Per-fit additive component curves for shading.
            self._fit_components = None  # list[(name, y_component)] for single fit
            self._fit_components_by_run = {}  # {run_number: list[(name, y_component)]}

            # Interactive plot labels (text annotations).
            self._annotations: list[dict] = []
            self._active_annotation_idx: int | None = None
            self._annotation_drag_started = False

            self._canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion_notify)
            self._canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        except ImportError:
            from PySide6.QtWidgets import QLabel

            layout.addWidget(QLabel("matplotlib not installed — plotting disabled"))
            self._has_mpl = False

    def _create_limit_controls(self) -> None:
        """Create toolbar for adjusting plot limits.

        Uses a grid layout for compactness:
        Row 1: X min/max  Y min/max  Apply Auto
        Row 2: Bunch factor
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

        # Apply and Auto buttons
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_limits)
        apply_btn.setMaximumWidth(60)
        self._limit_toolbar.addWidget(apply_btn, 0, 8)

        auto_btn = QPushButton("Auto")
        auto_btn.clicked.connect(self._auto_limits)
        auto_btn.setMaximumWidth(50)
        self._limit_toolbar.addWidget(auto_btn, 0, 9)

        # Stretch to fill remaining space
        self._limit_toolbar.setColumnStretch(10, 1)
        self._limit_toolbar.addWidget(QWidget(), 0, 10)

        # Bunch factor on second row
        self._limit_toolbar.addWidget(QLabel("Bunch:"), 1, 0)
        self._bunch_factor = QSpinBox()
        self._bunch_factor.setRange(1, 1000)
        self._bunch_factor.setValue(1)
        self._bunch_factor.setMaximumWidth(60)
        self._bunch_factor.valueChanged.connect(self._on_bunch_changed)
        self._limit_toolbar.addWidget(self._bunch_factor, 1, 1)

        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._add_label_btn.setMaximumWidth(90)
        self._limit_toolbar.addWidget(self._add_label_btn, 1, 2)

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

    def set_fit_range(self, x_min: float, x_max: float) -> None:
        """Set fit range limits and refresh visual handles."""
        self._set_fit_range(x_min, x_max, emit_signal=True, redraw=True)

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

        analysis_dataset = self.get_analysis_dataset(dataset)
        if analysis_dataset is None:
            return
        time = analysis_dataset.time
        asymmetry = analysis_dataset.asymmetry
        error = analysis_dataset.error

        self._ax.clear()
        self._ax.errorbar(
            time,
            asymmetry,
            yerr=error,
            fmt=".",
            markersize=3,
            label=f"Run {dataset.run_label}",
        )
        self._ax.set_xlabel("Time (μs)")
        self._ax.set_ylabel("Asymmetry (%)")

        # Re-plot fit curve if it exists (check both single and global fits)
        fit_to_plot = None
        if self._fit_curve is not None:
            fit_to_plot = self._fit_curve
        elif dataset.run_number in self._fit_curves:
            fit_to_plot = self._fit_curves[dataset.run_number]

        if fit_to_plot is not None:
            t_fit, y_fit, fit_label = fit_to_plot
            self._ax.plot(t_fit, y_fit, 'r-', linewidth=2, label=fit_label)

        self._draw_annotations()

        self._ax.legend()

        # Set default limits based on data range (including error bars)
        x_min, x_max = time.min(), time.max()
        y_min = (asymmetry - error).min()
        y_max = (asymmetry + error).max()

        # Add 5% padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05

        self._x_min.setValue(x_min - x_padding)
        self._x_max.setValue(x_max + x_padding)
        self._y_min.setValue(y_min - y_padding)
        self._y_max.setValue(y_max + y_padding)

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

    def _auto_limits(self) -> None:
        """Auto-scale the plot to fit all data."""
        if not self._has_mpl:
            return

        self._ax.relim()
        self._ax.autoscale()

        # Update spinboxes to reflect the new limits
        x_lim = self._ax.get_xlim()
        y_lim = self._ax.get_ylim()

        self._x_min.setValue(x_lim[0])
        self._x_max.setValue(x_lim[1])
        self._y_min.setValue(y_lim[0])
        self._y_max.setValue(y_lim[1])

        self._draw_fit_range_artists()
        self._canvas.draw()

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
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def _edit_annotation(self, idx: int) -> None:
        """Edit an existing annotation label."""
        current = self._annotations[idx]["text"]
        text, ok = QInputDialog.getText(self, "Edit Label", "Label text:", text=current)
        if not ok or not text.strip():
            return
        self._annotations[idx]["text"] = text.strip()
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def _delete_annotation(self, idx: int) -> None:
        """Delete an annotation by index."""
        self._annotations.pop(idx)
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

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
        """
        if not self._has_mpl:
            return

        # Store fit curve data for persistence across redraws (single fit)
        self._fit_curve = (t_fit, y_fit, label)
        # Clear global fits when doing a single fit
        self._fit_curves = {}
        self._fit_components_by_run = {}

        self._fit_components = list(component_curves or [])

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
            Dictionary mapping run_number -> (t_fit, y_fit, label, component_curves).
        """
        if not self._has_mpl:
            return

        # Store all fit curves
        self._fit_curves = {}
        self._fit_components_by_run = {}
        for run_number, payload in fit_curves_dict.items():
            if len(payload) == 4:
                t_fit, y_fit, label, component_curves = payload
            else:
                t_fit, y_fit, label = payload
                component_curves = []
            self._fit_curves[run_number] = (t_fit, y_fit, label)
            self._fit_components_by_run[run_number] = list(component_curves or [])
        # Clear single fit curve
        self._fit_curve = None
        self._fit_components = None

        # Redraw current dataset with its fit
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def clear(self) -> None:
        """Clear the plot and reset stored data."""
        if self._has_mpl:
            self._ax.clear()
            self._canvas.draw()
            self._current_dataset = None
            self._fit_curve = None
            self._fit_curves = {}
            self._fit_components = None
            self._fit_components_by_run = {}
            self._annotations = []
            self._active_annotation_idx = None
            self._annotation_drag_started = False
            self._fit_x_min = None
            self._fit_x_max = None
            self._fit_span_artist = None
            self._fit_min_handle = None
            self._fit_max_handle = None

    def clear_fit(self) -> None:
        """Clear all fit curves and redraw the plot."""
        if not self._has_mpl:
            return

        self._fit_curve = None
        self._fit_curves = {}
        self._fit_components = None
        self._fit_components_by_run = {}
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def get_current_plot_export_data(self) -> dict | None:
        """Return current fit/components/annotation payload for export."""
        if self._current_dataset is None:
            return None

        analysis_dataset = self.get_analysis_dataset(self._current_dataset)
        if analysis_dataset is None:
            return None

        fit_data = None
        component_data = None
        run_number = self._current_dataset.run_number
        if self._fit_curve is not None:
            fit_data = self._fit_curve
            component_data = self._fit_components or []
        elif run_number in self._fit_curves:
            fit_data = self._fit_curves[run_number]
            component_data = self._fit_components_by_run.get(run_number, [])

        if fit_data is None:
            return None

        t_fit, y_fit, fit_label = fit_data
        return {
            "run_number": run_number,
            "data": {
                "t": analysis_dataset.time,
                "y": analysis_dataset.asymmetry,
                "err": analysis_dataset.error,
            },
            "fit": {"t": t_fit, "y": y_fit, "label": fit_label},
            "components": [
                {"name": name, "y": y_vals} for name, y_vals in (component_data or [])
            ],
            "annotations": [
                {"x": ann["x"], "y": ann["y"], "text": ann["text"]}
                for ann in self._annotations
            ],
        }

    def export_current_plot(self) -> None:
        """Export current main-plot view as GLE (with optional compiled output)."""
        payload = self.get_current_plot_export_data()
        if not payload:
            QMessageBox.warning(self, "Export unavailable", "No fitted curve is available to export.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Current Plot",
            "current_plot.gle",
            "GLE files (*.gle);;PDF files (*.pdf);;EPS files (*.eps)",
        )
        if not path:
            return

        target = Path(path)
        suffix = target.suffix.lower()
        if "PDF" in selected_filter and suffix not in {".pdf", ".gle", ".eps"}:
            target = target.with_suffix(".pdf")
            suffix = ".pdf"
        elif "EPS" in selected_filter and suffix not in {".eps", ".gle", ".pdf"}:
            target = target.with_suffix(".eps")
            suffix = ".eps"
        elif suffix not in {".gle", ".pdf", ".eps"}:
            target = target.with_suffix(".gle")
            suffix = ".gle"

        try:
            glp = importlib.import_module("gleplot")
        except ImportError:
            QMessageBox.warning(self, "gleplot not available", "Install gleplot to export GLE plots.")
            return

        data = payload.get("data") or {}
        fit = payload.get("fit") or {}
        annotations = payload.get("annotations") or []
        t_data = data.get("t")
        y_data = data.get("y")
        y_err = data.get("err")
        t_fit = fit.get("t")
        y_fit = fit.get("y")

        fig = glp.figure(figsize=(6.0, 4.2))
        ax = fig.add_subplot(111)

        if t_data is not None and y_data is not None:
            has_err = y_err is not None
            ax.errorbar(
                t_data,
                y_data,
                yerr=y_err if has_err else None,
                fmt='none',
                marker='o',
                color='black',
                markersize=4,
                capsize=2,
                label=f"Run {payload.get('run_number', '')}",
            )

        if t_fit is not None and y_fit is not None:
            ax.plot(t_fit, y_fit, color='red', linewidth=1.6, label=fit.get("label", "Fit"))

        for ann in annotations:
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            text = str(ann.get("text", "")).strip()
            if text:
                ax.text(x, y, text, color='black', ha='left')

        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Asymmetry (%)")
        ax.legend(loc="best")

        gle_path = target if suffix == ".gle" else target.with_suffix(".gle")
        fig.savefig(str(gle_path))

        if suffix in {".pdf", ".eps"}:
            if shutil.which("gle") is None:
                QMessageBox.information(
                    self,
                    "GLE Not Installed",
                    f"GLE script saved to {gle_path}. Install GLE to compile to {suffix[1:].upper()}.",
                )
                return
            fmt = "pdf" if suffix == ".pdf" else "eps"
            try:
                subprocess.run(["gle", "-d", fmt, str(gle_path)], capture_output=True, text=True, check=True)
                QMessageBox.information(self, "Export Successful", f"Plot exported:\n\n{gle_path}\n{target}")
            except subprocess.CalledProcessError as exc:
                QMessageBox.warning(self, "GLE compilation failed", exc.stderr or str(exc))
            return

        QMessageBox.information(self, "Export Successful", f"GLE plot exported:\n\n{gle_path}")

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
            "bunch_factor": self._bunch_factor.value() if self._has_mpl else 1,
            "x_min": self._x_min.value() if self._has_mpl else 0.0,
            "x_max": self._x_max.value() if self._has_mpl else 10.0,
            "y_min": self._y_min.value() if self._has_mpl else -30.0,
            "y_max": self._y_max.value() if self._has_mpl else 30.0,
            "fit_curve": None,
            "fit_curves": {},
            "fit_components": None,
            "fit_components_by_run": {},
            "annotations": [],
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
            state["annotations"] = [
                {"x": ann["x"], "y": ann["y"], "text": ann["text"]}
                for ann in self._annotations
            ]

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

        # Restore bunch factor without triggering bunch_factor_changed signal.
        self._bunch_factor.blockSignals(True)
        self._bunch_factor.setValue(state.get("bunch_factor", 1))
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

        fit_x_min = state.get("fit_x_min")
        fit_x_max = state.get("fit_x_max")
        if fit_x_min is not None and fit_x_max is not None:
            self._fit_x_min = float(fit_x_min)
            self._fit_x_max = float(fit_x_max)

        # Restore fit curves.
        self._fit_curve = None
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

        self._annotations = []
        for ann in state.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            try:
                self._annotations.append(
                    {
                        "x": float(ann.get("x", 0.0)),
                        "y": float(ann.get("y", 0.0)),
                        "text": str(ann.get("text", "")),
                        "artist": None,
                    }
                )
            except (TypeError, ValueError):
                continue

        # Re-plot the current dataset if one was provided.
        if dataset is not None:
            self._current_dataset = dataset
            self.plot_dataset(dataset)

        if fit_x_min is not None and fit_x_max is not None:
            self._set_fit_range(
                float(fit_x_min),
                float(fit_x_max),
                emit_signal=False,
                redraw=True,
            )

        # Always apply the restored limits.
        self._apply_limits()
