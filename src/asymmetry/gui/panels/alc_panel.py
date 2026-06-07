"""Bespoke ALC-mode panels: build an integral-asymmetry field scan and view it.

ALC mode is toggled from the main toolbar when the Forward-Backward asymmetry
representation is active. Enabling it swaps the Fit and Parameters docks for
these focused widgets:

* :class:`ALCFitPanel` — the build controls (integration window + Build Scan).
  ALC mode *integrates* the asymmetry over the fit-range window for each
  selected run (one value per run) rather than fitting a model.
* :class:`ALCScanView` — the resulting field scan (integral asymmetry vs the
  run variable) as a plot plus a table.

The scan is recorded as a model-less ``FitSeries`` internally (see
``MainWindow._on_scan_requested``); these widgets are pure presentation and hold
no analysis logic of their own.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles.fonts import mono_font


class ALCFitPanel(QWidget):
    """Build controls for an integral-asymmetry field scan (ALC mode).

    The integration window IS the time-spectrum fit-range, mirroring the regular
    fitting machinery: the user can drag the shaded range on the plot, or set it
    precisely with the spinboxes here (the two stay in sync). Editing a spinbox
    emits :attr:`fit_range_edit_committed`; clicking *Build Scan* emits
    :attr:`build_requested`. The main window does the work.
    """

    build_requested = Signal()
    #: (x_min, x_max) committed from the spinboxes — pushed to the plot fit-range.
    fit_range_edit_committed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Integral scan (ALC)")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        intro = QLabel(
            "Integrate the asymmetry over the window for each selected run to "
            "build a field scan (ALC / repolarisation / QLCR)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        window_box = QGroupBox("Integration window")
        window_layout = QVBoxLayout(window_box)
        range_row = QHBoxLayout()
        range_row.setContentsMargins(6, 4, 6, 4)
        range_row.setSpacing(4)
        self._min_spin = self._make_time_spin()
        mid_label = QLabel("≤ <i>t</i> ≤")
        mid_label.setTextFormat(Qt.TextFormat.RichText)
        self._max_spin = self._make_time_spin()
        range_row.addWidget(self._min_spin)
        range_row.addWidget(mid_label)
        range_row.addWidget(self._max_spin)
        range_row.addWidget(QLabel("μs"))
        range_row.addStretch()
        window_layout.addLayout(range_row)
        hint = QLabel("Drag the shaded range on the time plot, or set it here.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        window_layout.addWidget(hint)
        layout.addWidget(window_box)

        self._min_spin.editingFinished.connect(self._on_spin_committed)
        self._max_spin.editingFinished.connect(self._on_spin_committed)

        self._build_btn = QPushButton("Build Scan")
        self._build_btn.clicked.connect(self.build_requested.emit)
        layout.addWidget(self._build_btn)

        layout.addStretch()

    @staticmethod
    def _make_time_spin() -> QDoubleSpinBox:
        """A μs fit-range spinbox configured like the regular fit panel's."""
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1000.0, 1000.0)
        spin.setSingleStep(0.1)
        spin.setMinimumWidth(90)
        spin.setFont(mono_font(11.0))
        spin.setEnabled(False)  # enabled once the plot has a fit-range
        return spin

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Update the spinboxes from the plot fit-range without re-emitting.

        Mirrors ``SingleFitTab.set_fit_range_display``; the spinboxes are
        disabled until the plot has a fit-range.
        """
        have_range = x_min is not None and x_max is not None
        self._min_spin.setEnabled(have_range)
        self._max_spin.setEnabled(have_range)
        if not have_range:
            return
        with QSignalBlocker(self._min_spin):
            self._min_spin.setValue(float(x_min))
        with QSignalBlocker(self._max_spin):
            self._max_spin.setValue(float(x_max))

    def _on_spin_committed(self) -> None:
        self.fit_range_edit_committed.emit(self._min_spin.value(), self._max_spin.value())


class ALCScanView(QWidget):
    """Plot + table view of one integral-asymmetry field scan.

    Carries the view controls — an x-axis selector (field / temperature / run)
    and a dA/dB derivative toggle — and emits :attr:`options_changed` when they
    change. The main window owns the scan data and re-feeds arrays via
    :meth:`show_scan`; this widget holds no analysis logic.
    """

    #: Emitted when the x-axis or derivative toggle changes.
    options_changed = Signal()
    #: Emitted when the user requests a baseline fit (Fit baseline button).
    baseline_fit_requested = Signal()

    #: x-combo index → ordering key.
    _X_KEYS = ("field", "temperature", "run")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_plot: dict | None = None
        self._baseline_curve: NDArray[np.float64] | None = None
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("x:"))
        self._x_combo = QComboBox()
        self._x_combo.addItems(["B (G)", "T (K)", "Run"])
        self._x_combo.currentIndexChanged.connect(self._emit_options_changed)
        controls.addWidget(self._x_combo)
        self._derivative_check = QCheckBox("dA/dB")
        self._derivative_check.setToolTip(
            "Show the derivative of the scan (WiMDA 'differential ALC')."
        )
        self._derivative_check.toggled.connect(self._emit_options_changed)
        controls.addWidget(self._derivative_check)
        controls.addStretch()
        layout.addLayout(controls)
        self._update_derivative_label()

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout.addWidget(self._canvas, 1)

        layout.addWidget(self._build_baseline_group())

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Run", "x", "A (%)", "± err"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(160)
        layout.addWidget(self._table)

        self.clear()

    def _build_baseline_group(self) -> QGroupBox:
        """Baseline controls: model + non-resonant regions table + Fit button."""
        group = QGroupBox("Baseline")
        outer = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self._baseline_model_combo = QComboBox()
        self._baseline_model_combo.addItems(["Linear", "Constant"])
        row.addWidget(self._baseline_model_combo)
        row.addStretch()
        self._fit_baseline_btn = QPushButton("Fit baseline")
        self._fit_baseline_btn.clicked.connect(self.baseline_fit_requested.emit)
        row.addWidget(self._fit_baseline_btn)
        outer.addLayout(row)

        self._regions_table = QTableWidget(0, 2)
        self._regions_table.setHorizontalHeaderLabels(["start", "end"])
        self._regions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._regions_table.setMaximumHeight(110)
        self._regions_table.itemChanged.connect(self._render_plot)
        outer.addWidget(self._regions_table)

        btns = QHBoxLayout()
        add_btn = QPushButton("+ region")
        add_btn.clicked.connect(self._add_region)
        remove_btn = QPushButton("− region")
        remove_btn.clicked.connect(self._remove_region)
        btns.addWidget(add_btn)
        btns.addWidget(remove_btn)
        btns.addStretch()
        outer.addLayout(btns)
        return group

    #: Derivative-checkbox label per x-axis (the y-quantity is dA/dx).
    _DERIV_LABELS = {"field": "dA/dB", "temperature": "dA/dT", "run": "dA/d(run)"}

    def _emit_options_changed(self, *_: object) -> None:
        """Re-emit the combo/checkbox change as the 0-arg ``options_changed``."""
        self._update_derivative_label()
        self.options_changed.emit()

    def _update_derivative_label(self) -> None:
        """Keep the derivative-toggle label in step with the x-axis."""
        self._derivative_check.setText(self._DERIV_LABELS[self.x_key()])

    def x_key(self) -> str:
        """Return the selected ordering key: ``"field"``/``"temperature"``/``"run"``."""
        return self._X_KEYS[self._x_combo.currentIndex()]

    def derivative_enabled(self) -> bool:
        """Return True when the derivative toggle is on."""
        return self._derivative_check.isChecked()

    # --- baseline controls ---------------------------------------------------

    def baseline_model(self) -> str:
        """Return the selected baseline model name (``"Linear"``/``"Constant"``)."""
        return self._baseline_model_combo.currentText()

    def baseline_regions(self) -> list[tuple[float, float]]:
        """Parse the regions table into ``(start, end)`` pairs (skipping bad rows)."""
        regions: list[tuple[float, float]] = []
        for row in range(self._regions_table.rowCount()):
            start_item = self._regions_table.item(row, 0)
            end_item = self._regions_table.item(row, 1)
            try:
                lo = float(start_item.text())
                hi = float(end_item.text())
            except (AttributeError, ValueError):
                continue
            if lo < hi:
                regions.append((lo, hi))
        return regions

    def _add_region(self) -> None:
        """Append a region row defaulting to the left edge of the current x-range."""
        lo, hi = 0.0, 1.0
        if self._last_plot is not None and self._last_plot["x"].size:
            xs = self._last_plot["x"]
            span = float(xs.max() - xs.min()) or 1.0
            lo = float(xs.min())
            hi = lo + 0.1 * span
        row = self._regions_table.rowCount()
        self._regions_table.insertRow(row)
        with QSignalBlocker(self._regions_table):
            for col, val in ((0, lo), (1, hi)):
                self._regions_table.setItem(row, col, QTableWidgetItem(f"{val:.4g}"))
        self._render_plot()

    def _remove_region(self) -> None:
        """Remove the selected region row (or the last row)."""
        row = self._regions_table.currentRow()
        if row < 0:
            row = self._regions_table.rowCount() - 1
        if row >= 0:
            self._regions_table.removeRow(row)
            self._render_plot()

    def show_baseline_overlay(self, baseline: NDArray[np.float64]) -> None:
        """Overlay a fitted baseline curve on the current scan plot."""
        self._baseline_curve = np.asarray(baseline, dtype=float)
        self._render_plot()

    # --- plotting ------------------------------------------------------------

    def clear(self, message: str = "Build a scan to see the ALC curve") -> None:
        """Show an empty-state placeholder with *message*."""
        self._last_plot = None
        self._baseline_curve = None
        self._ax.clear()
        self._ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self._ax.transAxes,
            color="gray",
            wrap=True,
        )
        self._ax.set_axis_off()
        self._canvas.draw_idle()
        self._table.setRowCount(0)

    def show_scan(
        self,
        x: NDArray[np.float64],
        value: NDArray[np.float64],
        error: NDArray[np.float64],
        run_numbers: list[int],
        *,
        x_label: str,
        y_label: str,
        value_header: str = "value",
    ) -> None:
        """Render a scan from parallel arrays (already in display units)."""
        self._last_plot = {
            "x": np.asarray(x, dtype=float),
            "value": np.asarray(value, dtype=float),
            "error": np.asarray(error, dtype=float),
            "x_label": x_label,
            "y_label": y_label,
        }
        # A fresh scan (rebuild / x-axis change) invalidates any baseline overlay.
        self._baseline_curve = None
        self._render_plot()

        self._table.setHorizontalHeaderLabels(["Run", x_label, value_header, "± err"])
        n = int(self._last_plot["x"].size)
        self._table.setRowCount(n)
        for row in range(n):
            run = run_numbers[row] if row < len(run_numbers) else ""
            cells = [
                str(run),
                f"{self._last_plot['x'][row]:.4g}",
                f"{self._last_plot['value'][row]:.4f}",
                f"{self._last_plot['error'][row]:.4f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)

    def _render_plot(self, *_: object) -> None:
        """Draw the stored scan plus shaded baseline regions and any baseline fit."""
        plot = self._last_plot
        if plot is None or plot["x"].size == 0:
            return
        self._ax.clear()
        self._ax.set_axis_on()
        for lo, hi in self.baseline_regions():
            self._ax.axvspan(lo, hi, color="0.85", alpha=0.6, zorder=0)
        self._ax.errorbar(
            plot["x"],
            plot["value"],
            yerr=plot["error"],
            fmt="o-",
            markersize=4,
            capsize=2,
            linewidth=1.0,
        )
        if self._baseline_curve is not None and self._baseline_curve.shape == plot["x"].shape:
            self._ax.plot(
                plot["x"], self._baseline_curve, color="#a8332a", linewidth=1.2, label="baseline"
            )
            self._ax.legend(loc="best", fontsize=8)
        self._ax.set_xlabel(plot["x_label"])
        self._ax.set_ylabel(plot["y_label"])
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()
