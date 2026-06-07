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
    QDialog,
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

from asymmetry.gui.panels.draggable_handles import nearest_handle
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
    #: Emitted when the user requests a peak fit (Fit peaks button).
    peaks_fit_requested = Signal()
    #: Emitted when a region edit/drag invalidates the baseline-corrected scan.
    baseline_invalidated = Signal()

    #: x-combo index → ordering key.
    _X_KEYS = ("field", "temperature", "run")
    #: Peak-row "Type" → core component name.
    _PEAK_COMPONENTS = {"Gaussian": "GaussianLCR", "Lorentzian": "LorentzianLCR"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_plot: dict | None = None
        self._baseline_curve: NDArray[np.float64] | None = None
        self._fit_curve: NDArray[np.float64] | None = None
        self._data_dialog: QDialog | None = None
        self._data_table: QTableWidget | None = None
        #: Active plot drag, as ("region", row, col) or ("peak", row, 1).
        self._drag: tuple[str, int, int] | None = None
        #: The Line2D being dragged (moved in place; full re-render on release).
        self._drag_artist = None
        #: Handle key → drawn edge/centre Line2D, rebuilt by each _render_plot.
        self._handle_artists: dict[tuple[str, int, int], object] = {}
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("x:"))
        self._x_combo = QComboBox()
        self._x_combo.addItems(["B (G)", "T (K)", "Run"])
        self._x_combo.currentIndexChanged.connect(self._on_x_changed)
        controls.addWidget(self._x_combo)
        self._derivative_check = QCheckBox("dA/dB")
        self._derivative_check.setToolTip(
            "Show the derivative of the scan (WiMDA 'differential ALC')."
        )
        self._derivative_check.toggled.connect(self._emit_options_changed)
        controls.addWidget(self._derivative_check)
        controls.addStretch()
        data_btn = QPushButton("Data table…")
        data_btn.setToolTip("Show the scan's per-point values in a separate window.")
        data_btn.clicked.connect(self._on_show_data_table)
        controls.addWidget(data_btn)
        layout.addLayout(controls)
        self._update_derivative_label()

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(260)
        self._ax = self._figure.add_subplot(111)
        # Drag baseline-region edges and peak centres directly on the plot.
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        # The plot takes all spare vertical space; the analysis sections below
        # are collapsible so the plot keeps the room when they are not in use.
        layout.addWidget(self._canvas, 1)

        layout.addWidget(self._build_baseline_group())
        layout.addWidget(self._build_peaks_group())
        layout.addStretch(0)

        self.clear()

    @staticmethod
    def _collapsible_group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """A checkable group box that hides its content when unchecked."""
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        content = QWidget()
        shell = QVBoxLayout(group)
        shell.setContentsMargins(6, 2, 6, 6)
        shell.addWidget(content)
        group.toggled.connect(content.setVisible)
        return group, QVBoxLayout(content)

    def _build_baseline_group(self) -> QGroupBox:
        """Baseline controls: model + non-resonant regions table + Fit button."""
        group, outer = self._collapsible_group("Baseline")

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
        self._regions_table.itemChanged.connect(self._invalidate_baseline)
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

    def _build_peaks_group(self) -> QGroupBox:
        """Peak controls: add/remove Gaussian/Lorentzian peaks + Fit peaks."""
        group, outer = self._collapsible_group("Peaks")

        row = QHBoxLayout()
        add_g = QPushButton("+ Gaussian")
        add_g.clicked.connect(lambda: self._add_peak("Gaussian"))
        add_l = QPushButton("+ Lorentzian")
        add_l.clicked.connect(lambda: self._add_peak("Lorentzian"))
        remove_peak = QPushButton("− peak")
        remove_peak.clicked.connect(self._remove_peak)
        row.addWidget(add_g)
        row.addWidget(add_l)
        row.addWidget(remove_peak)
        row.addStretch()
        self._fit_peaks_btn = QPushButton("Fit peaks")
        self._fit_peaks_btn.clicked.connect(self.peaks_fit_requested.emit)
        row.addWidget(self._fit_peaks_btn)
        outer.addLayout(row)

        self._peaks_table = QTableWidget(0, 4)
        self._peaks_table.setHorizontalHeaderLabels(["Type", "B0 (G)", "Width (G)", "Amp (%)"])
        self._peaks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._peaks_table.setMaximumHeight(110)
        self._peaks_table.itemChanged.connect(self._invalidate_peaks)
        outer.addWidget(self._peaks_table)

        self._peaks_results = QLabel("")
        self._peaks_results.setWordWrap(True)
        self._peaks_results.setStyleSheet("color: #1f4d8a;")
        outer.addWidget(self._peaks_results)
        return group

    #: Derivative-checkbox label per x-axis (the y-quantity is dA/dx).
    _DERIV_LABELS = {"field": "dA/dB", "temperature": "dA/dT", "run": "dA/d(run)"}

    def _emit_options_changed(self, *_: object) -> None:
        """Re-emit the combo/checkbox change as the 0-arg ``options_changed``."""
        self._update_derivative_label()
        self.options_changed.emit()

    def _on_x_changed(self, *_: object) -> None:
        """X-axis changed: the regions/peaks (in x units) no longer apply."""
        self.clear_analysis()
        self._emit_options_changed()

    def clear_analysis(self) -> None:
        """Drop the baseline regions, peaks, and any fitted overlays."""
        with QSignalBlocker(self._regions_table), QSignalBlocker(self._peaks_table):
            self._regions_table.setRowCount(0)
            self._peaks_table.setRowCount(0)
        self._baseline_curve = None
        self._fit_curve = None
        self._peaks_results.setText("")

    def _invalidate_baseline(self, *_: object) -> None:
        """A region change makes the baseline (and the peak fit on it) stale.

        Drops the baseline + total-fit overlays and tells the main window to
        discard the stored baseline-corrected scan, so a later peak fit cannot
        run against a baseline that no longer matches the regions on screen.
        """
        self._baseline_curve = None
        self._fit_curve = None
        self._peaks_results.setText("")
        self.baseline_invalidated.emit()
        self._render_plot()

    def _invalidate_peaks(self, *_: object) -> None:
        """A peak change makes the peak fit stale (the baseline still holds)."""
        self._fit_curve = None
        self._peaks_results.setText("")
        self._render_plot()

    def _update_derivative_label(self) -> None:
        """Keep the derivative-toggle label in step with the x-axis."""
        self._derivative_check.setText(self._DERIV_LABELS[self.x_key()])

    def x_key(self) -> str:
        """Return the selected ordering key: ``"field"``/``"temperature"``/``"run"``."""
        return self._X_KEYS[self._x_combo.currentIndex()]

    def derivative_enabled(self) -> bool:
        """Return True when the derivative toggle is on."""
        return self._derivative_check.isChecked()

    # --- plot drag (region edges + peak centres) -----------------------------

    @staticmethod
    def _cell_float(table: QTableWidget, row: int, col: int) -> float | None:
        """Parse a table cell to float, or ``None`` if empty/unparseable."""
        item = table.item(row, col)
        try:
            return float(item.text())
        except (AttributeError, ValueError):
            return None

    def _peak_centres(self) -> list[tuple[int, float]]:
        """Return ``(row, B0)`` for each peak row with a parseable centre."""
        centres = []
        for row in range(self._peaks_table.rowCount()):
            b0 = self._cell_float(self._peaks_table, row, 1)
            if b0 is not None:
                centres.append((row, b0))
        return centres

    def _drag_handles(self) -> list[tuple[float, tuple[str, int, int]]]:
        """Return draggable handles as ``(x, (kind, row, col))`` from the tables.

        This is the single source of truth for which handles exist; the renderer
        draws the same set so every drawn handle is grabbable and vice versa.
        """
        handles: list[tuple[float, tuple[str, int, int]]] = []
        for row, lo, hi in self._raw_regions():
            handles.append((lo, ("region", row, 0)))
            handles.append((hi, ("region", row, 1)))
        for row, b0 in self._peak_centres():
            handles.append((b0, ("peak", row, 1)))
        return handles

    def _on_canvas_press(self, event: object) -> None:
        if event.inaxes is not self._ax or event.button != 1 or self._last_plot is None:
            return
        # 12 device px ≈ 6 logical px on a 2× display.
        self._drag = nearest_handle(self._ax, self._drag_handles(), event.x, tolerance_px=12.0)
        if self._drag is None:
            return
        # Grabbing a handle starts an edit: the existing fit is now stale.
        if self._drag[0] == "region":
            self._invalidate_baseline()
        else:
            self._invalidate_peaks()
        # Grab the drawn line for this handle so the drag moves only it (the
        # full re-render — shaded span included — is deferred to release).
        self._drag_artist = self._handle_artists.get(self._drag)

    def _on_canvas_motion(self, event: object) -> None:
        if self._drag is None or event.inaxes is not self._ax or event.xdata is None:
            return
        kind, row, col = self._drag
        x = float(event.xdata)
        table = self._regions_table if kind == "region" else self._peaks_table
        with QSignalBlocker(table):
            item = table.item(row, col)
            if item is not None:
                item.setText(f"{x:.4g}")
        if self._drag_artist is not None:
            self._drag_artist.set_xdata([x, x])
            self._canvas.draw_idle()
        else:
            self._render_plot()

    def _on_canvas_release(self, _event: object) -> None:
        was_dragging = self._drag is not None
        self._drag = None
        self._drag_artist = None
        if was_dragging:
            self._render_plot()  # restore the shaded span at the final edges

    # --- baseline controls ---------------------------------------------------

    def baseline_model(self) -> str:
        """Return the selected baseline model name (``"Linear"``/``"Constant"``)."""
        return self._baseline_model_combo.currentText()

    def baseline_regions(self) -> list[tuple[float, float]]:
        """Parse the regions table into valid ``(lo, hi)`` pairs.

        A region is defined by its two edges regardless of which was typed/dragged
        first, so the pair is normalised to ``(min, max)``; only a zero-width
        region (``lo == hi``) is dropped.
        """
        regions = []
        for _row, a, b in self._raw_regions():
            lo, hi = (a, b) if a <= b else (b, a)
            if lo < hi:
                regions.append((lo, hi))
        return regions

    def _raw_regions(self) -> list[tuple[int, float, float]]:
        """Parse the regions table into ``(row, lo, hi)`` (any parseable values).

        Unlike :meth:`baseline_regions`, this keeps rows where ``lo >= hi`` so a
        handle stays visible/draggable mid-drag past the other edge.
        """
        rows: list[tuple[int, float, float]] = []
        for row in range(self._regions_table.rowCount()):
            lo = self._cell_float(self._regions_table, row, 0)
            hi = self._cell_float(self._regions_table, row, 1)
            if lo is not None and hi is not None:
                rows.append((row, lo, hi))
        return rows

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
        self._invalidate_baseline()

    def _remove_region(self) -> None:
        """Remove the selected region row (or the last row)."""
        row = self._regions_table.currentRow()
        if row < 0:
            row = self._regions_table.rowCount() - 1
        if row >= 0:
            self._regions_table.removeRow(row)
            self._invalidate_baseline()

    def show_baseline_overlay(self, baseline: NDArray[np.float64]) -> None:
        """Overlay a fitted baseline curve; invalidate any prior peak fit."""
        self._baseline_curve = np.asarray(baseline, dtype=float)
        self._fit_curve = None
        self._peaks_results.setText("")
        self._render_plot()

    # --- peak controls -------------------------------------------------------

    def _add_peak(self, peak_type: str) -> None:
        """Append a peak row (Type read-only) with guesses from the x-range."""
        b0, wid, amp = 0.0, 1.0, -1.0
        if self._last_plot is not None and self._last_plot["x"].size:
            xs = self._last_plot["x"]
            span = float(xs.max() - xs.min()) or 1.0
            b0 = float(0.5 * (xs.min() + xs.max()))
            wid = 0.1 * span
        row = self._peaks_table.rowCount()
        self._peaks_table.insertRow(row)
        with QSignalBlocker(self._peaks_table):
            type_item = QTableWidgetItem(peak_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._peaks_table.setItem(row, 0, type_item)
            for col, val in ((1, b0), (2, wid), (3, amp)):
                self._peaks_table.setItem(row, col, QTableWidgetItem(f"{val:.4g}"))
        self._invalidate_peaks()  # new peak invalidates the prior fit; draw its marker

    def _remove_peak(self) -> None:
        """Remove the selected peak row (or the last row)."""
        row = self._peaks_table.currentRow()
        if row < 0:
            row = self._peaks_table.rowCount() - 1
        if row >= 0:
            self._peaks_table.removeRow(row)
            self._invalidate_peaks()

    def peak_specs(self) -> list[dict]:
        """Return the peaks as ``{component, label, f, B0, Bwid}`` initial guesses.

        Every peak row must be valid: a row with an unknown type or an
        unparseable B0/Width/Amp raises ``ValueError`` (naming the row) rather
        than being silently skipped, so the returned specs stay aligned 1:1 with
        the table rows (the fitted values are written straight back by row).
        ``label`` is the row's display type (e.g. ``"Gaussian"``).
        """
        specs: list[dict] = []
        for row in range(self._peaks_table.rowCount()):
            type_item = self._peaks_table.item(row, 0)
            label = type_item.text() if type_item else ""
            component = self._PEAK_COMPONENTS.get(label)
            if component is None:
                raise ValueError(f"Peak {row + 1}: unknown peak type.")
            b0 = self._cell_float(self._peaks_table, row, 1)
            bwid = self._cell_float(self._peaks_table, row, 2)
            amp = self._cell_float(self._peaks_table, row, 3)
            if b0 is None or bwid is None or amp is None:
                raise ValueError(f"Peak {row + 1}: B0, Width and Amp must be numbers.")
            specs.append({"component": component, "label": label, "f": amp, "B0": b0, "Bwid": bwid})
        return specs

    def set_peak_results(self, results: list[dict], summary: str = "") -> None:
        """Update the peaks table to the fitted values and show *summary*."""
        with QSignalBlocker(self._peaks_table):
            for row, res in enumerate(results):
                if row >= self._peaks_table.rowCount():
                    break
                for col, key in ((1, "B0"), (2, "Bwid"), (3, "f")):
                    item = self._peaks_table.item(row, col)
                    if item is not None:
                        item.setText(f"{res[key]:.4g}")
        self._peaks_results.setText(summary)

    def show_fit_overlay(self, fit_curve: NDArray[np.float64]) -> None:
        """Overlay the total (baseline + peaks) fit curve on the scan plot."""
        self._fit_curve = np.asarray(fit_curve, dtype=float)
        self._render_plot()

    # --- data-table dialog ---------------------------------------------------

    def point_count(self) -> int:
        """Number of points in the current scan view (0 when empty)."""
        return int(self._last_plot["x"].size) if self._last_plot is not None else 0

    def _on_show_data_table(self) -> None:
        """Open (or raise) the per-point data table in a separate dialog."""
        if self._data_dialog is None:
            self._data_dialog = QDialog(self)
            self._data_dialog.setWindowTitle("ALC scan data")
            self._data_dialog.resize(440, 360)
            dialog_layout = QVBoxLayout(self._data_dialog)
            self._data_table = QTableWidget(0, 4)
            self._data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            dialog_layout.addWidget(self._data_table)
        self._fill_data_table()
        self._data_dialog.show()
        self._data_dialog.raise_()

    def _fill_data_table(self) -> None:
        """Populate the data-table dialog from the current scan (if both exist)."""
        table = self._data_table
        plot = self._last_plot
        if table is None:
            return
        if plot is None:
            table.setRowCount(0)
            return
        table.setHorizontalHeaderLabels(["Run", plot["x_label"], plot["value_header"], "± err"])
        run_numbers = plot["run_numbers"]
        n = int(plot["x"].size)
        table.setRowCount(n)
        for row in range(n):
            run = run_numbers[row] if row < len(run_numbers) else ""
            cells = [
                str(run),
                f"{plot['x'][row]:.4g}",
                f"{plot['value'][row]:.4f}",
                f"{plot['error'][row]:.4f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)

    # --- plotting ------------------------------------------------------------

    def clear(self, message: str = "Build a scan to see the ALC curve") -> None:
        """Show an empty-state placeholder with *message*.

        The scan is gone, so its analysis goes with it: the region/peak tables
        and the results read-out are emptied (otherwise stale rows, markers, and
        draggable handles would persist onto the next, unrelated scan).
        """
        self._last_plot = None
        self._baseline_curve = None
        self._fit_curve = None
        with QSignalBlocker(self._regions_table), QSignalBlocker(self._peaks_table):
            self._regions_table.setRowCount(0)
            self._peaks_table.setRowCount(0)
        self._peaks_results.setText("")
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
        self._fill_data_table()

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
            "run_numbers": list(run_numbers),
            "x_label": x_label,
            "value_header": value_header,
            "y_label": y_label,
        }
        # A fresh scan (rebuild / x-axis change) invalidates any fit overlays.
        self._baseline_curve = None
        self._fit_curve = None
        self._peaks_results.setText("")
        self._render_plot()
        # Keep the data-table dialog in sync if it is open.
        if self._data_dialog is not None and self._data_dialog.isVisible():
            self._fill_data_table()

    def _render_plot(self, *_: object) -> None:
        """Draw the stored scan plus shaded regions, the baseline, and the fit."""
        plot = self._last_plot
        if plot is None or plot["x"].size == 0:
            return
        self._ax.clear()
        self._ax.set_axis_on()
        # Draw the draggable handles from the same source the hit-test uses, and
        # remember each line so a drag can move just it (keys: region edge cols
        # 0/1; peak centre col 1).
        self._handle_artists = {}
        # Baseline regions: shaded spans with draggable edge lines (the span is
        # the same whichever edge was dragged first, so shade min..max).
        for row, a, b in self._raw_regions():
            lo, hi = (a, b) if a <= b else (b, a)
            if lo < hi:
                self._ax.axvspan(lo, hi, color="0.85", alpha=0.6, zorder=0)
            for col, edge in ((0, a), (1, b)):
                line = self._ax.axvline(edge, color="0.55", linewidth=0.8, zorder=1)
                self._handle_artists[("region", row, col)] = line
        # Peak centres: draggable dashed markers.
        for row, b0 in self._peak_centres():
            line = self._ax.axvline(b0, color="#2a7a3f", linestyle="--", linewidth=0.9, zorder=1)
            self._handle_artists[("peak", row, 1)] = line
        # Data as markers only (no joining line).
        self._ax.errorbar(
            plot["x"], plot["value"], yerr=plot["error"], fmt="o", markersize=4, capsize=2
        )
        has_overlay = False
        if self._baseline_curve is not None and self._baseline_curve.shape == plot["x"].shape:
            self._ax.plot(
                plot["x"], self._baseline_curve, color="#a8332a", linewidth=1.2, label="baseline"
            )
            has_overlay = True
        if self._fit_curve is not None and self._fit_curve.shape == plot["x"].shape:
            self._ax.plot(plot["x"], self._fit_curve, color="#1f4d8a", linewidth=1.4, label="fit")
            has_overlay = True
        if has_overlay:
            self._ax.legend(loc="best", fontsize=8)
        self._ax.set_xlabel(plot["x_label"])
        self._ax.set_ylabel(plot["y_label"])
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()
