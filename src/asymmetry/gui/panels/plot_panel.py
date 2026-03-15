"""Central plot panel using Matplotlib embedded in Qt.

Displays time-domain asymmetry with error bars and optional fit overlay,
similar to WiMDA's main plot area.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform import apply_grouping


class PlotPanel(QWidget):
    """Matplotlib canvas for time- and frequency-domain plots.

    Notes
    -----
    The plot panel renders the dataset currently selected in the data browser.
    Grouping/bunching choices are controlled in the grouping workflow.
    """

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
            self._limits_initialized = False

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

        Uses a compact grid layout:
        Row 1: X min/max  Y min/max  AutoX AutoY  Add Label
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

        for spin in (self._x_min, self._x_max, self._y_min, self._y_max):
            spin.editingFinished.connect(self._apply_limits)

        # Independent auto buttons
        auto_x_btn = QPushButton("Auto X")
        auto_x_btn.clicked.connect(self._auto_x_limits)
        auto_x_btn.setFixedWidth(72)
        self._limit_toolbar.addWidget(auto_x_btn, 0, 8)

        auto_y_btn = QPushButton("Auto Y")
        auto_y_btn.clicked.connect(self._auto_y_limits)
        auto_y_btn.setFixedWidth(72)
        self._limit_toolbar.addWidget(auto_y_btn, 0, 9)

        # Add-label action
        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._add_label_btn.setMaximumWidth(90)
        self._limit_toolbar.addWidget(self._add_label_btn, 0, 10)

        # Stretch to fill remaining space
        self._limit_toolbar.setColumnStretch(11, 1)
        self._limit_toolbar.addWidget(QWidget(), 0, 11)

    def get_analysis_dataset(self, dataset: MuonDataset | None) -> MuonDataset | None:
        """Return the dataset used for plotting and fitting.

        Grouping/rebin choices are now applied upstream, so this returns the
        original dataset unchanged.
        """
        return dataset

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
        """Plot the selected dataset with error bars and fit overlays."""
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

        plot_asymmetry = asymmetry.astype(float, copy=True)
        plot_error = error.astype(float, copy=True)
        background_asymmetry = np.full_like(plot_asymmetry, np.nan, dtype=float)
        background_error = np.full_like(plot_error, np.nan, dtype=float)
        valid = self._compute_plot_valid_mask(analysis_dataset)
        if valid is not None and valid.shape == plot_asymmetry.shape:
            invalid = ~valid
            background_asymmetry[invalid] = plot_asymmetry[invalid]
            background_error[invalid] = plot_error[invalid]
            plot_asymmetry[invalid] = float("nan")
            plot_error[invalid] = float("nan")

        self._ax.clear()
        self._ax.errorbar(
            time,
            background_asymmetry,
            yerr=background_error,
            fmt=".",
            markersize=3,
            color="lightgray",
            ecolor="lightgray",
            elinewidth=1.0,
            capsize=0,
            zorder=1,
            label="_nolegend_",
        )
        self._ax.errorbar(
            time,
            plot_asymmetry,
            yerr=plot_error,
            fmt=".",
            markersize=3,
            label=f"Run {dataset.run_label}",
            zorder=2,
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

        # Only initialize limits once; do not reset user limits on replot.
        if not self._limits_initialized:
            x_min, x_max = self._compute_auto_x_limits(time)
            y_min, y_max = self._compute_auto_y_limits(
                asymmetry,
                error,
                plot_asymmetry,
                plot_error,
            )
            self._x_min.setValue(x_min)
            self._x_max.setValue(x_max)
            self._y_min.setValue(y_min)
            self._y_max.setValue(y_max)
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

    def _compute_auto_x_limits(self, time: np.ndarray) -> tuple[float, float]:
        """Compute padded X limits from the displayed time array."""
        x_min = float(np.nanmin(time))
        x_max = float(np.nanmax(time))
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return 0.0, 1.0
        if x_max <= x_min:
            return x_min - 0.5, x_max + 0.5
        x_padding = (x_max - x_min) * 0.05
        return x_min - x_padding, x_max + x_padding

    def _compute_auto_y_limits(
        self,
        asymmetry: np.ndarray,
        error: np.ndarray,
        reliable_asymmetry: np.ndarray,
        reliable_error: np.ndarray,
    ) -> tuple[float, float]:
        """Compute padded Y limits using reliable points only when available."""
        y_low = reliable_asymmetry - reliable_error
        y_high = reliable_asymmetry + reliable_error

        if np.all(~np.isfinite(y_low)) or np.all(~np.isfinite(y_high)):
            y_low = asymmetry - error
            y_high = asymmetry + error

        y_min = float(np.nanmin(y_low))
        y_max = float(np.nanmax(y_high))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return -1.0, 1.0
        if y_max <= y_min:
            return y_min - 1.0, y_max + 1.0
        y_padding = (y_max - y_min) * 0.05
        return y_min - y_padding, y_max + y_padding

    def _auto_x_limits(self) -> None:
        """Auto-scale X limits only."""
        if not self._has_mpl:
            return

        analysis_dataset = self.get_analysis_dataset(self._current_dataset)
        if analysis_dataset is None or analysis_dataset.n_points == 0:
            return
        x_min, x_max = self._compute_auto_x_limits(analysis_dataset.time)
        self._x_min.setValue(x_min)
        self._x_max.setValue(x_max)
        self._apply_limits()

    def _auto_y_limits(self) -> None:
        """Auto-scale Y limits only using reliable points inside the current X range."""
        if not self._has_mpl:
            return

        analysis_dataset = self.get_analysis_dataset(self._current_dataset)
        if analysis_dataset is None or analysis_dataset.n_points == 0:
            return

        asymmetry = analysis_dataset.asymmetry.astype(float, copy=True)
        error = analysis_dataset.error.astype(float, copy=True)
        reliable_asymmetry = asymmetry.copy()
        reliable_error = error.copy()
        valid = self._compute_plot_valid_mask(analysis_dataset)
        if valid is not None and valid.shape == reliable_asymmetry.shape:
            invalid = ~valid
            reliable_asymmetry[invalid] = float("nan")
            reliable_error[invalid] = float("nan")

        x_min = float(min(self._x_min.value(), self._x_max.value()))
        x_max = float(max(self._x_min.value(), self._x_max.value()))
        in_x_range = (analysis_dataset.time >= x_min) & (analysis_dataset.time <= x_max)

        if np.any(in_x_range):
            asymmetry = asymmetry[in_x_range]
            error = error[in_x_range]
            reliable_asymmetry = reliable_asymmetry[in_x_range]
            reliable_error = reliable_error[in_x_range]

        y_min, y_max = self._compute_auto_y_limits(
            asymmetry,
            error,
            reliable_asymmetry,
            reliable_error,
        )
        self._y_min.setValue(y_min)
        self._y_max.setValue(y_max)
        self._apply_limits()

    def _auto_limits(self) -> None:
        """Auto-scale both X and Y limits."""
        if not self._has_mpl:
            return

        self._auto_x_limits()
        self._auto_y_limits()

    def _compute_plot_valid_mask(self, dataset: MuonDataset):
        """Return a mask of bins to plot as reliable foreground points.

        Bins with non-positive grouped denominator are treated as undefined.
        Bins saturated at ±100% are shown as low-confidence background points.
        """
        run = dataset.run
        if run is None or not run.histograms:
            return None

        grouping = run.grouping or {}
        groups = grouping.get("groups")
        try:
            forward_gid = int(grouping.get("forward_group", 1))
            backward_gid = int(grouping.get("backward_group", 2))
        except (TypeError, ValueError):
            return None
        if not isinstance(groups, dict):
            return None

        def _to_indices(values):
            out = []
            for v in values:
                try:
                    out.append(max(0, int(v) - 1))
                except (TypeError, ValueError):
                    continue
            return out

        forward_idx = _to_indices(groups.get(forward_gid, []))
        backward_idx = _to_indices(groups.get(backward_gid, []))
        if not forward_idx or not backward_idx:
            return None
        if max(forward_idx, default=-1) >= len(run.histograms):
            return None
        if max(backward_idx, default=-1) >= len(run.histograms):
            return None

        forward = apply_grouping(run.histograms, forward_idx)
        backward = apply_grouping(run.histograms, backward_idx)
        alpha = float(grouping.get("alpha", 1.0))
        denominator = forward + alpha * backward

        lo = int(grouping.get("first_good_bin", 0))
        hi = int(grouping.get("last_good_bin", len(denominator) - 1))
        lo = max(0, lo)
        hi = min(len(denominator) - 1, hi)
        if lo > hi:
            return None

        denominator = denominator[lo : hi + 1]
        if denominator.shape[0] != dataset.time.shape[0]:
            return None
        reliable = denominator > 0.0
        saturated = np.isclose(np.abs(dataset.asymmetry), 100.0, atol=1e-12)
        if saturated.shape == reliable.shape:
            reliable = reliable & (~saturated)
        return reliable

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

        The fit curve will be retained when limits change.

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
            self._limits_initialized = False

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

        This captures axis limits, the currently displayed
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
            # Kept for backward compatibility with existing project files.
            "bunch_factor": 1,
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
            drawn, but all other state (limits, fit curves) is
            still applied.
        """
        if not self._has_mpl:
            return

        import numpy as np

        # ``bunch_factor`` is intentionally ignored (legacy key).

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

        # Keep restored limits when plot_dataset is called during restore.
        self._limits_initialized = True

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
