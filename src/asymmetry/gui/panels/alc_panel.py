"""Bespoke ALC-mode panels: build an integral-asymmetry field scan and view it.

The "Integral scan" representation *integrates* the asymmetry over the
fit-range window for each selected run (one value per run) rather than fitting
a model. Its surfaces:

* :class:`ALCFitPanel` — the build controls (integration window + Build Scan),
  swapped into the Fit dock.
* :class:`ALCScanView` — the field scan (integral asymmetry vs the run
  variable) plus its Baseline/Peaks/RF analysis controls. Its two sections are
  relocatable: the plot section goes to the central
  :class:`IntegralScanPanel`, the analysis section to the Parameters dock.
* :class:`IntegralScanPanel` — the central workspace page: the scan plot above
  an :class:`IntegralTimeStrip`, the slim time-domain preview carrying the
  draggable integration window.

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
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.export_paths import default_export_path, remember_export_path
from asymmetry.gui.panels.draggable_handles import nearest_handle
from asymmetry.gui.panels.plot_panel import _FloatLimitField
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.plots import draw_empty_state_message, draw_fit_range_span, style_axes
from asymmetry.gui.styles.widgets import build_nav_button_qss, build_primary_button_qss


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
        range_row.addWidget(QLabel("µs"))
        range_row.addStretch()
        window_layout.addLayout(range_row)
        hint = QLabel("Drag the shaded range on the time plot, or set it here.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        window_layout.addWidget(hint)
        layout.addWidget(window_box)

        self._min_spin.editingFinished.connect(self._on_spin_committed)
        self._max_spin.editingFinished.connect(self._on_spin_committed)

        self._rf_difference_check = QCheckBox("RF resonance (Green − Red)")
        self._rf_difference_check.setToolTip(
            "Build the RF-µSR observable: integrate the (Green − Red) period "
            "difference per run (Green = RF-off, Red = RF-on), giving the "
            "resonance curve vs field. Requires two-period (red/green) runs."
        )
        layout.addWidget(self._rf_difference_check)

        self._build_btn = QPushButton("Build Scan")
        self._build_btn.setStyleSheet(build_primary_button_qss())
        self._build_btn.clicked.connect(self.build_requested.emit)
        layout.addWidget(self._build_btn)

        layout.addStretch()

    def rf_difference_enabled(self) -> bool:
        """Return True when the scan should be built as a (Green − Red) RF scan."""
        return self._rf_difference_check.isChecked()

    @staticmethod
    def _make_time_spin() -> QDoubleSpinBox:
        """A µs fit-range spinbox configured like the regular fit panel's."""
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

    The view is the single owner of the scan canvas, the drag machinery, and
    the Baseline/Peaks/RF tables, but its two sections are physically
    relocatable (see :meth:`plot_widget` / :meth:`analysis_widget`): in the
    application the plot section lives in the central integral-scan panel and
    the analysis section in the Parameters dock, while a standalone
    ``ALCScanView`` still lays out both (tests build it directly).
    """

    #: Emitted when the x-axis or derivative toggle changes.
    options_changed = Signal()
    #: Emitted when the user requests a baseline fit (Fit baseline button).
    baseline_fit_requested = Signal()
    #: Emitted when the user requests a peak fit (Fit peaks button).
    peaks_fit_requested = Signal()
    #: Emitted when the user requests an RF-resonance fit (Fit RF resonance button).
    rf_fit_requested = Signal()
    #: Emitted when a region edit/drag invalidates the baseline-corrected scan.
    baseline_invalidated = Signal()
    #: Emitted when the user clicks a scan point: exclude/restore that run.
    point_toggled = Signal(int)

    #: x-combo index → ordering key.
    _X_KEYS = ("field", "temperature", "run")
    #: Peak-row "Type" → core component name.
    _PEAK_COMPONENTS = {"Gaussian": "GaussianLCR", "Lorentzian": "LorentzianLCR"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_plot: dict | None = None
        #: Fitted overlays as (x, y) pairs — fits run on included points only,
        #: so an overlay's x can be a subset of the rendered scan's.
        self._baseline_curve: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._fit_curve: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._data_dialog: QDialog | None = None
        self._data_table: QTableWidget | None = None
        #: Active plot drag, as ("region", row, col) or ("peak", row, 1).
        self._drag: tuple[str, int, int] | None = None
        #: The Line2D being dragged (moved in place; full re-render on release).
        self._drag_artist = None
        #: Handle key → drawn edge/centre Line2D, rebuilt by each _render_plot.
        self._handle_artists: dict[tuple[str, int, int], object] = {}
        #: Pending click-to-toggle candidate: (run_number, press x/y in px).
        self._toggle_candidate: tuple[int, float, float] | None = None
        #: Auto-scale state per axis (matches the PlotPanel limit toolbar): when
        #: on, the axis frames the data and its fields mirror the result; when
        #: off, the axis is pinned to the typed field values.
        self._auto_x = True
        self._auto_y = True
        layout = QVBoxLayout(self)

        # The view is two relocatable sections: the plot section (view
        # controls + canvas + provenance line) and the analysis section
        # (Baseline/Peaks/RF groups). Both are laid out here so a standalone
        # ALCScanView still shows everything; the main window reparents the
        # plot section into the central integral-scan panel and the analysis
        # section into the Parameters dock (see plot_widget/analysis_widget).
        self._plot_section = QWidget()
        plot_layout = QVBoxLayout(self._plot_section)
        plot_layout.setContentsMargins(0, 0, 0, 0)

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
        plot_layout.addLayout(controls)
        self._update_derivative_label()

        plot_layout.addLayout(self._build_limit_controls())

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(200)
        self._ax = self._figure.add_subplot(111)
        # Drag baseline-region edges and peak centres directly on the plot.
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        plot_layout.addWidget(self._canvas, 1)

        # Scan provenance: which runs contribute, which were dropped and why
        # (previously only in the log). The full drop list rides the tooltip.
        self._provenance_label = QLabel("")
        self._provenance_label.setWordWrap(True)
        self._provenance_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._provenance_label.hide()
        plot_layout.addWidget(self._provenance_label)
        layout.addWidget(self._plot_section, 3)

        # The plot is the dominant pane, but the fitted-parameter tables below
        # must stay reachable: rather than let the canvas grab every spare pixel
        # (pushing the Baseline/Peaks sections below the fold), the analysis
        # sections live in their own scroll area with a guaranteed minimum
        # height. On a tall dock everything is visible; on a short one the
        # analysis area scrolls internally instead of vanishing.

        analysis = QWidget()
        analysis_layout = QVBoxLayout(analysis)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.addWidget(self._build_baseline_group())
        analysis_layout.addWidget(self._build_peaks_group())
        analysis_layout.addWidget(self._build_rf_group())
        analysis_layout.addStretch(0)
        self._analysis_scroll = QScrollArea()
        self._analysis_scroll.setWidgetResizable(True)
        self._analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._analysis_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._analysis_scroll.setMinimumHeight(170)
        self._analysis_scroll.setWidget(analysis)
        layout.addWidget(self._analysis_scroll, 2)

        self.clear()

    # --- relocatable sections -------------------------------------------------

    def plot_widget(self) -> QWidget:
        """The plot section (view controls + canvas + provenance line).

        The main window reparents this into the central integral-scan panel;
        all logic (drag handles, tables) stays on this view.
        """
        return self._plot_section

    def analysis_widget(self) -> QWidget:
        """The analysis section (Baseline/Peaks/RF groups in a scroll area).

        The main window reparents this into the Parameters dock.
        """
        return self._analysis_scroll

    def figure(self) -> Figure:
        """The scan plot's matplotlib figure (for export)."""
        return self._figure

    def set_provenance(self, text: str, tooltip: str = "") -> None:
        """Show which runs contribute to the scan (empty *text* hides the line)."""
        self._provenance_label.setText(text)
        self._provenance_label.setToolTip(tooltip)
        self._provenance_label.setVisible(bool(text))

    # --- axis-limit controls --------------------------------------------------

    def _build_limit_controls(self) -> QHBoxLayout:
        """Build the X/Y range fields + Auto toggles (as on the main plot panels)."""
        row = QHBoxLayout()
        row.setSpacing(4)

        row.addWidget(QLabel("X:"))
        self._x_min = _FloatLimitField(0.0, minimum_width=64)
        self._x_max = _FloatLimitField(1.0, minimum_width=64)
        row.addWidget(self._x_min)
        row.addWidget(QLabel("–"))
        row.addWidget(self._x_max)

        row.addWidget(QLabel("Y:"))
        self._y_min = _FloatLimitField(0.0, minimum_width=64)
        self._y_max = _FloatLimitField(1.0, minimum_width=64)
        row.addWidget(self._y_min)
        row.addWidget(QLabel("–"))
        row.addWidget(self._y_max)

        nav_qss = build_nav_button_qss()
        self._auto_x_btn = QPushButton("Auto X")
        self._auto_x_btn.setCheckable(True)
        self._auto_x_btn.setChecked(True)
        self._auto_x_btn.setStyleSheet(nav_qss)
        self._auto_x_btn.setMaximumWidth(65)
        self._auto_x_btn.toggled.connect(self._on_auto_x_toggled)
        row.addWidget(self._auto_x_btn)

        self._auto_y_btn = QPushButton("Auto Y")
        self._auto_y_btn.setCheckable(True)
        self._auto_y_btn.setChecked(True)
        self._auto_y_btn.setStyleSheet(nav_qss)
        self._auto_y_btn.setMaximumWidth(65)
        self._auto_y_btn.toggled.connect(self._on_auto_y_toggled)
        row.addWidget(self._auto_y_btn)

        row.addStretch()

        # A manual edit is an explicit override: it turns that axis's Auto off so
        # the next render does not reframe the typed value back to the extent.
        self._x_min.editingFinished.connect(self._on_x_limit_edited)
        self._x_max.editingFinished.connect(self._on_x_limit_edited)
        self._y_min.editingFinished.connect(self._on_y_limit_edited)
        self._y_max.editingFinished.connect(self._on_y_limit_edited)
        return row

    def _on_auto_x_toggled(self, checked: bool) -> None:
        self._auto_x = checked
        self._render_plot()

    def _on_auto_y_toggled(self, checked: bool) -> None:
        self._auto_y = checked
        self._render_plot()

    def _on_x_limit_edited(self) -> None:
        self._auto_x = False
        with QSignalBlocker(self._auto_x_btn):
            self._auto_x_btn.setChecked(False)
        self._render_plot()

    def _on_y_limit_edited(self) -> None:
        self._auto_y = False
        with QSignalBlocker(self._auto_y_btn):
            self._auto_y_btn.setChecked(False)
        self._render_plot()

    def _auto_data_bounds(self) -> tuple[float, float, float, float] | None:
        """Return ``(xmin, xmax, ymin, ymax)`` framing the scan data + overlays.

        Computed from the plotted data, **not** from the axes' data limits: the
        shaded baseline regions and the peak / region handle lines span the whole
        axes, so ``autoscale_view`` would frame to them and squash the data.
        Two deliberate asymmetries:

        * The **y** extent uses the *included* points only (value ± error) plus
          the fit overlays — a run the user click-excluded should not stretch
          the frame; that is the whole point of excluding an outlier.
        * The **x** extent keeps *every* point (excluded ones stay horizontally
          visible, so a mildly-excluded point can still be clicked to restore)
          and also the region edges / peak centres, so a handle dragged past the
          data stays on-screen and grabbable.

        Returns ``None`` when there is nothing finite to frame.
        """
        plot = self._last_plot
        if plot is None or plot["x"].size == 0:
            return None
        included = ~plot["excluded"]
        if not included.any():  # everything excluded: fall back to all points
            included = np.ones(plot["x"].size, dtype=bool)
        xs = [plot["x"]]  # x-frame keeps every point (see docstring)
        y_lo = [plot["value"][included] - plot["error"][included]]
        y_hi = [plot["value"][included] + plot["error"][included]]
        for curve in (self._baseline_curve, self._fit_curve):
            if curve is not None:
                cx, cy = curve
                if cx.size and cx.shape == cy.shape:
                    xs.append(cx)
                    y_lo.append(cy)
                    y_hi.append(cy)
        # Region edges and peak centres are meaningful x-locations the user
        # drags; keep them in the x-frame (x only — their full-height artists
        # must never enter the y-frame, which was the squash bug).
        handle_x = [edge for _row, lo, hi in self._raw_regions() for edge in (lo, hi)]
        handle_x += [b0 for _row, b0 in self._peak_centres()]
        if handle_x:
            xs.append(np.asarray(handle_x, dtype=float))
        x = np.concatenate(xs)
        lo = np.concatenate(y_lo)
        hi = np.concatenate(y_hi)
        x = x[np.isfinite(x)]
        lo = lo[np.isfinite(lo)]
        hi = hi[np.isfinite(hi)]
        if x.size == 0 or lo.size == 0 or hi.size == 0:
            return None
        return float(x.min()), float(x.max()), float(lo.min()), float(hi.max())

    @staticmethod
    def _padded(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
        """Return ``(lo, hi)`` widened by *frac* of the span (both sides).

        A degenerate range (``hi <= lo``, e.g. a single point or a manual
        min==max) is expanded to a small window around the value so the axis is
        valid and the point/line is visible rather than collapsed to a hairline.
        """
        if hi <= lo:
            pad = max(abs(lo), 1.0) * frac  # unit-scale floor for lo≈0
            return lo - pad, hi + pad
        pad = (hi - lo) * frac
        return lo - pad, hi + pad

    def _axis_target(
        self, auto: bool, data_range: tuple[float, float] | None, lo_field, hi_field
    ) -> tuple[float, float] | None:
        """Target ``(lo, hi)`` for one axis, or ``None`` to leave it unchanged.

        Auto frames the padded data range (``None`` when there is no data, so the
        axis is left as-is). Manual uses the typed fields, sorted so an inverted
        entry is read as a range, and expanded when degenerate so a min==max
        entry still produces a valid axis instead of being silently dropped.
        """
        if auto:
            return None if data_range is None else self._padded(*data_range)
        lo, hi = sorted((lo_field.value(), hi_field.value()))
        return self._padded(lo, hi) if lo == hi else (lo, hi)

    def _apply_axis_limits(self) -> None:
        """Pin manual axes to their fields, frame auto axes from the data, and
        sync the fields to the applied limits.

        Called at the end of :meth:`_render_plot` once every artist is drawn, so
        the limits it writes back are the ones the canvas will show.
        """
        bounds = self._auto_data_bounds()
        x_range = None if bounds is None else (bounds[0], bounds[1])
        y_range = None if bounds is None else (bounds[2], bounds[3])
        xlim = self._axis_target(self._auto_x, x_range, self._x_min, self._x_max)
        ylim = self._axis_target(self._auto_y, y_range, self._y_min, self._y_max)
        if xlim is not None:
            self._ax.set_xlim(*xlim)
        if ylim is not None:
            self._ax.set_ylim(*ylim)
        # Reflect the applied limits (fall back to the current view only when we
        # left an axis unchanged — auto with no data — never clobbering a typed
        # value with a stale axis extent).
        x0, x1 = xlim if xlim is not None else self._ax.get_xlim()
        y0, y1 = ylim if ylim is not None else self._ax.get_ylim()
        for field, value in (
            (self._x_min, x0),
            (self._x_max, x1),
            (self._y_min, y0),
            (self._y_max, y1),
        ):
            with QSignalBlocker(field):
                field.setValue(float(value))

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
        """Baseline controls: model + non-resonant regions table + Fit button.

        The "Fit baseline" button sits on its own row at the bottom rather than at
        the right end of the Model row: ``Model:`` + combo + the button together
        need ~386px, wider than the ~360px inspector deck, and the analysis area
        scrolls vertically only — so an over-wide row clips the button off the
        right edge with no way to reach it (test_rf_fit_row_reachable; mirrors the
        RF group's wrapping in ``_build_rf_group``).
        """
        group, outer = self._collapsible_group("Baseline")

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self._baseline_model_combo = QComboBox()
        # Cubic is the WiMDA/Mantid-prescribed ALC background (a curved/sloping
        # baseline Linear cannot match); fitted over the non-resonant regions.
        # Quartic→Sextic add the higher orders a steep 0–3 T muonium-
        # repolarisation envelope needs to flatten cleanly (Cubic tops out too
        # low and leaves the radical dips on a curved residual).
        self._baseline_model_combo.addItems(
            ["Linear", "Constant", "Cubic", "Quartic", "Quintic", "Sextic"]
        )
        row.addWidget(self._baseline_model_combo)
        row.addStretch()
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

        fit_row = QHBoxLayout()
        fit_row.addStretch()
        self._fit_baseline_btn = QPushButton("Fit baseline")
        self._fit_baseline_btn.clicked.connect(self.baseline_fit_requested.emit)
        fit_row.addWidget(self._fit_baseline_btn)
        outer.addLayout(fit_row)
        return group

    def _build_peaks_group(self) -> QGroupBox:
        """Peak controls: add/remove Gaussian/Lorentzian peaks + Fit peaks.

        The buttons are wrapped to fit the ~360px inspector deck (the analysis
        area scrolls vertically only, so an over-wide row clips off the right
        edge unreachably — test_rf_fit_row_reachable; mirrors ``_build_rf_group``).
        A single row of all three add/remove buttons plus "Fit peaks" needs
        ~536px, and even the three add/remove buttons alone need ~390px, so the
        add pair and "− peak" sit on separate rows and "Fit peaks" gets its own
        row beneath the table.
        """
        group, outer = self._collapsible_group("Peaks")

        add_row = QHBoxLayout()
        add_g = QPushButton("+ Gaussian")
        add_g.clicked.connect(lambda: self._add_peak("Gaussian"))
        add_l = QPushButton("+ Lorentzian")
        add_l.clicked.connect(lambda: self._add_peak("Lorentzian"))
        add_row.addWidget(add_g)
        add_row.addWidget(add_l)
        add_row.addStretch()
        outer.addLayout(add_row)

        remove_row = QHBoxLayout()
        remove_peak = QPushButton("− peak")
        remove_peak.clicked.connect(self._remove_peak)
        remove_row.addWidget(remove_peak)
        remove_row.addStretch()
        outer.addLayout(remove_row)

        self._peaks_table = QTableWidget(0, 4)
        self._peaks_table.setHorizontalHeaderLabels(["Type", "B0 (G)", "Width (G)", "Amp (%)"])
        self._peaks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._peaks_table.setMaximumHeight(110)
        self._peaks_table.itemChanged.connect(self._invalidate_peaks)
        outer.addWidget(self._peaks_table)

        self._peaks_results = QLabel("")
        self._peaks_results.setWordWrap(True)
        self._peaks_results.setStyleSheet(f"color: {tokens.ACCENT};")
        outer.addWidget(self._peaks_results)

        fit_row = QHBoxLayout()
        fit_row.addStretch()
        self._fit_peaks_btn = QPushButton("Fit peaks")
        self._fit_peaks_btn.clicked.connect(self.peaks_fit_requested.emit)
        fit_row.addWidget(self._fit_peaks_btn)
        outer.addLayout(fit_row)
        return group

    def _build_rf_group(self) -> QGroupBox:
        """RF-resonance fit: exact muon+proton model → A_µ, A_p (collapsed by default).

        Fits the (Green − Red) field scan with the ``RFResonanceMuP`` component:
        ν_RF is a known acquisition constant (seeded 218.5 MHz, held fixed), and
        A_µ / A_p seed the resonance position and splitting and are read off the
        fit. The amplitudes/widths/background are seeded from the data by the core
        helper, so only these three physics inputs are exposed here.
        """
        group, outer = self._collapsible_group("RF resonance (A_µ, A_p)")
        group.setChecked(False)  # advanced; collapsed until needed

        # One labeled row per seed. A single horizontal row of three MHz
        # spinboxes needs ~510px — wider than the ~360px inspector deck — and
        # the analysis area scrolls vertically only, so it would clip the Fit
        # button off-screen with no way to reach it (test_rf_fit_row_reachable).
        def _seed_row(label: str, spin: QDoubleSpinBox) -> None:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(spin)
            row.addWidget(QLabel("MHz"))
            row.addStretch()
            outer.addLayout(row)

        self._rf_nu_spin = self._make_mhz_spin(218.5)
        self._rf_nu_spin.setToolTip("RF frequency (MHz); held fixed during the fit.")
        _seed_row("ν_RF:", self._rf_nu_spin)
        self._rf_a_mu_spin = self._make_mhz_spin(515.0)
        self._rf_a_mu_spin.setToolTip("Starting guess for the muon hyperfine coupling A_µ (MHz).")
        _seed_row("A_µ₀:", self._rf_a_mu_spin)
        self._rf_a_p_spin = self._make_mhz_spin(124.0)
        self._rf_a_p_spin.setToolTip("Starting guess for the proton hyperfine coupling A_p (MHz).")
        _seed_row("A_p₀:", self._rf_a_p_spin)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._fit_rf_btn = QPushButton("Fit RF resonance")
        self._fit_rf_btn.clicked.connect(self.rf_fit_requested.emit)
        btn_row.addWidget(self._fit_rf_btn)
        outer.addLayout(btn_row)

        self._rf_results = QLabel("")
        self._rf_results.setWordWrap(True)
        self._rf_results.setStyleSheet(f"color: {tokens.ACCENT};")
        outer.addWidget(self._rf_results)
        return group

    @staticmethod
    def _make_mhz_spin(default: float) -> QDoubleSpinBox:
        """A compact MHz spinbox for the RF seed inputs."""
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(0.0, 100000.0)
        spin.setSingleStep(1.0)
        spin.setValue(float(default))
        spin.setMinimumWidth(80)
        spin.setFont(mono_font(11.0))
        return spin

    def rf_nu(self) -> float:
        """Return the user's RF frequency ν_RF (MHz)."""
        return float(self._rf_nu_spin.value())

    def rf_a_mu_seed(self) -> float:
        """Return the starting guess for A_µ (MHz)."""
        return float(self._rf_a_mu_spin.value())

    def rf_a_p_seed(self) -> float:
        """Return the starting guess for A_p (MHz)."""
        return float(self._rf_a_p_spin.value())

    def set_rf_results(self, summary: str) -> None:
        """Show the RF-resonance fit read-out (A_µ / A_p)."""
        self._rf_results.setText(summary)

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
        self._rf_results.setText("")

    # --- persistence ---------------------------------------------------------

    def analysis_state(self) -> dict:
        """Serialise the view options, baseline regions, and peaks to a dict."""
        # Persist only valid (normalised lo<hi) regions — the ones the fit uses —
        # so a reopened scan can't show a phantom region with no effect.
        regions = [[lo, hi] for lo, hi in self.baseline_regions()]
        peaks: list[list] = []
        for row in range(self._peaks_table.rowCount()):
            type_item = self._peaks_table.item(row, 0)
            vals = [self._cell_float(self._peaks_table, row, col) for col in (1, 2, 3)]
            if type_item is None or any(v is None for v in vals):
                continue
            peaks.append([type_item.text(), *vals])
        return {
            "x_key": self.x_key(),
            "derivative": self.derivative_enabled(),
            "baseline_model": self.baseline_model(),
            "regions": regions,
            "peaks": peaks,
            "baseline_fitted": self._baseline_curve is not None,
            "peaks_fitted": self._fit_curve is not None,
            "rf_nu": self.rf_nu(),
            "rf_a_mu": self.rf_a_mu_seed(),
            "rf_a_p": self.rf_a_p_seed(),
        }

    def restore_analysis_state(self, state: dict) -> None:
        """Restore view options + region/peak tables from :meth:`analysis_state`.

        Does not re-render or re-fit (the caller drives that); signals are
        blocked so setting the x-axis does not wipe the tables it just filled.
        """
        x_key = state.get("x_key", "field")
        with QSignalBlocker(self._x_combo):
            if x_key in self._X_KEYS:
                self._x_combo.setCurrentIndex(self._X_KEYS.index(x_key))
        self._update_derivative_label()
        with QSignalBlocker(self._derivative_check):
            self._derivative_check.setChecked(bool(state.get("derivative", False)))
        model_idx = self._baseline_model_combo.findText(str(state.get("baseline_model", "Linear")))
        if model_idx >= 0:
            self._baseline_model_combo.setCurrentIndex(model_idx)

        for key, spin in (
            ("rf_nu", self._rf_nu_spin),
            ("rf_a_mu", self._rf_a_mu_spin),
            ("rf_a_p", self._rf_a_p_spin),
        ):
            if key in state:
                try:
                    spin.setValue(float(state[key]))
                except (TypeError, ValueError):
                    pass

        with QSignalBlocker(self._regions_table):
            self._regions_table.setRowCount(0)
            for region in self._coerce_rows(state.get("regions"), 2):
                row = self._regions_table.rowCount()
                self._regions_table.insertRow(row)
                for col, val in enumerate(region):
                    self._regions_table.setItem(row, col, QTableWidgetItem(f"{val:.4g}"))

        with QSignalBlocker(self._peaks_table):
            self._peaks_table.setRowCount(0)
            for peak in state.get("peaks", []):
                if not isinstance(peak, (list, tuple)) or len(peak) != 4:
                    continue
                if str(peak[0]) not in self._PEAK_COMPONENTS:
                    continue
                vals = self._coerce_rows([peak[1:]], 3)
                if not vals:
                    continue
                row = self._peaks_table.rowCount()
                self._peaks_table.insertRow(row)
                type_item = QTableWidgetItem(str(peak[0]))
                type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._peaks_table.setItem(row, 0, type_item)
                for col, val in zip((1, 2, 3), vals[0], strict=True):
                    self._peaks_table.setItem(row, col, QTableWidgetItem(f"{val:.4g}"))

    @staticmethod
    def _coerce_rows(rows: object, width: int) -> list[list[float]]:
        """Coerce *rows* (untrusted on-disk data) to ``width``-float lists, skipping bad ones."""
        out: list[list[float]] = []
        if not isinstance(rows, (list, tuple)):
            return out
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != width:
                continue
            try:
                out.append([float(v) for v in row])
            except (TypeError, ValueError):
                continue
        return out

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

    def _point_run_at(self, event: object) -> int | None:
        """Run number of the scan point within click tolerance of *event*, else None."""
        plot = self._last_plot
        if plot is None or not plot["x"].size:
            return None
        pts = self._ax.transData.transform(np.column_stack([plot["x"], plot["value"]]))
        d2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
        # A non-finite point transforms to NaN; NaN would win argmin and pass
        # the tolerance test (NaN > tol is False), making every click toggle
        # that run — treat it as infinitely far instead.
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        idx = int(np.argmin(d2))
        # 12 device px ≈ 6 logical px on a 2× display (same as the handle grab).
        if d2[idx] > 12.0**2:
            return None
        run_numbers = plot["run_numbers"]
        return int(run_numbers[idx]) if idx < len(run_numbers) else None

    def _on_canvas_press(self, event: object) -> None:
        if event.inaxes is not self._ax or event.button != 1 or self._last_plot is None:
            return
        # 12 device px ≈ 6 logical px on a 2× display.
        self._drag = nearest_handle(self._ax, self._drag_handles(), event.x, tolerance_px=12.0)
        if self._drag is None:
            # Not on a drag handle: a stationary click on a data point toggles
            # that run's exclusion (resolved on release, so a drag that starts
            # near a point does not mis-fire). Renders that mark themselves
            # non-toggleable (the derivative view, where excluded points have
            # no greyed marker to click back) never arm the candidate.
            if not self._last_plot.get("toggleable", True):
                return
            run = self._point_run_at(event)
            if run is not None:
                self._toggle_candidate = (run, float(event.x), float(event.y))
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
        if self._toggle_candidate is not None:
            _run, x0, y0 = self._toggle_candidate
            if abs(event.x - x0) > 3.0 or abs(event.y - y0) > 3.0:
                self._toggle_candidate = None  # moved: it's a pan/drag, not a click
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

    def _on_canvas_release(self, event: object) -> None:
        was_dragging = self._drag is not None
        self._drag = None
        self._drag_artist = None
        # Consume the pending click candidate on *any* release so it cannot
        # linger and fire on a later, unrelated left-release.
        candidate = self._toggle_candidate
        self._toggle_candidate = None
        if was_dragging:
            self._render_plot()  # restore the shaded span at the final edges
        elif candidate is not None and getattr(event, "button", None) == 1:
            # Only a released *left* button completes the click — a right/middle
            # release mid-gesture must not mutate scan membership.
            self.point_toggled.emit(candidate[0])

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

    def show_baseline_overlay(self, x: NDArray[np.float64], baseline: NDArray[np.float64]) -> None:
        """Overlay a fitted baseline curve; invalidate any prior peak fit.

        The overlay carries its own *x*: fits run on the included points only,
        so their curves can be shorter than the rendered scan (which also shows
        excluded points, greyed).
        """
        self._baseline_curve = (np.asarray(x, dtype=float), np.asarray(baseline, dtype=float))
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

    def show_fit_overlay(self, x: NDArray[np.float64], fit_curve: NDArray[np.float64]) -> None:
        """Overlay the total (baseline + peaks) fit curve on the scan plot.

        Like :meth:`show_baseline_overlay`, the overlay carries its own *x*.
        """
        self._fit_curve = (np.asarray(x, dtype=float), np.asarray(fit_curve, dtype=float))
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
            excluded = bool(plot["excluded"][row]) if row < plot["excluded"].size else False
            cells = [
                str(run),
                f"{plot['x'][row]:.4g}",
                f"{plot['value'][row]:.4f}",
                f"{plot['error'][row]:.4f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if excluded:
                    item.setForeground(Qt.GlobalColor.gray)
                    if col == 0:
                        item.setToolTip("Excluded from the scan (click the point to restore).")
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
        self._rf_results.setText("")
        self.set_provenance("")
        self._ax.clear()
        draw_empty_state_message(self._ax, message)
        self._canvas.draw_idle()
        self._fill_data_table()

    def reset(self) -> None:
        """Reset to the pristine initial state for a new project.

        Beyond :meth:`clear` (scan, regions, peaks, overlays, results), this also
        restores the default baseline model so a new project does not inherit the
        previous scan's baseline choice. ``clear`` deliberately leaves the model
        alone because the empty-axis re-render paths reuse it; only New Project
        should wipe it.
        """
        self.clear()
        with QSignalBlocker(self._baseline_model_combo):
            self._baseline_model_combo.setCurrentIndex(0)

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
        excluded_mask: NDArray[np.bool_] | None = None,
        toggleable: bool = True,
        reset_view: bool = True,
    ) -> None:
        """Render a scan from parallel arrays (already in display units).

        *excluded_mask* marks user-excluded points: they are drawn greyed
        (click a point to toggle its exclusion) and skipped by the fits.
        *toggleable* arms click-to-exclude; the derivative view passes False
        because its points are pair-midpoints with no greyed marker to click
        back, so a stray click would exclude a run with no visible undo.
        *reset_view* re-enables auto-scaling on both axes; pass False for a
        same-scan re-render (e.g. click-exclude) so a manual zoom is kept.
        """
        n = np.asarray(x, dtype=float).size
        mask = (
            np.zeros(n, dtype=bool)
            if excluded_mask is None
            else np.asarray(excluded_mask, dtype=bool)
        )
        self._last_plot = {
            "x": np.asarray(x, dtype=float),
            "value": np.asarray(value, dtype=float),
            "error": np.asarray(error, dtype=float),
            "run_numbers": list(run_numbers),
            "x_label": x_label,
            "value_header": value_header,
            "y_label": y_label,
            "excluded": mask,
            "toggleable": bool(toggleable),
        }
        # A fresh scan (rebuild / x-axis change) invalidates any fit overlays.
        self._baseline_curve = None
        self._fit_curve = None
        self._peaks_results.setText("")
        self._rf_results.setText("")
        # New data (build, x-axis or derivative change) reframes both axes: a
        # stale manual range in the old units/scale would hide the new scan.
        # A same-scan re-render (click-exclude) passes reset_view=False to keep
        # a manual zoom; annotation edits (region/peak drags) call _render_plot
        # directly and likewise keep the user's manual view.
        if reset_view:
            self._reset_auto_scale()
        self._render_plot()
        # Keep the data-table dialog in sync if it is open.
        if self._data_dialog is not None and self._data_dialog.isVisible():
            self._fill_data_table()

    def _reset_auto_scale(self) -> None:
        """Re-enable auto-scaling on both axes (and the Auto toggles)."""
        self._auto_x = True
        self._auto_y = True
        for btn in (self._auto_x_btn, self._auto_y_btn):
            with QSignalBlocker(btn):
                btn.setChecked(True)

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
            line = self._ax.axvline(b0, color=tokens.OK, linestyle="--", linewidth=0.9, zorder=1)
            self._handle_artists[("peak", row, 1)] = line
        # Data as markers only (no joining line); user-excluded points greyed
        # and hollow so they stay visible (and clickable to restore).
        excluded = plot["excluded"]
        included = ~excluded
        if np.any(included):
            self._ax.errorbar(
                plot["x"][included],
                plot["value"][included],
                yerr=plot["error"][included],
                fmt="o",
                markersize=4,
                capsize=2,
            )
        if np.any(excluded):
            self._ax.errorbar(
                plot["x"][excluded],
                plot["value"][excluded],
                yerr=plot["error"][excluded],
                fmt="o",
                markersize=4,
                capsize=2,
                color=tokens.PLOT_LOW_COUNT,
                markerfacecolor="none",
                alpha=0.8,
                label="excluded",
            )
        has_overlay = bool(np.any(excluded))
        if self._baseline_curve is not None:
            bx, by = self._baseline_curve
            if bx.size and bx.shape == by.shape:
                self._ax.plot(bx, by, color=tokens.ACCENT_RED, linewidth=1.2, label="baseline")
                has_overlay = True
        if self._fit_curve is not None:
            fx, fy = self._fit_curve
            if fx.size and fx.shape == fy.shape:
                self._ax.plot(fx, fy, color=tokens.ACCENT, linewidth=1.4, label="fit")
                has_overlay = True
        if has_overlay:
            self._ax.legend(loc="best", fontsize=8)
        self._ax.set_xlabel(plot["x_label"])
        self._ax.set_ylabel(plot["y_label"])
        self._ax.grid(True, alpha=0.3)
        self._apply_axis_limits()
        self._canvas.draw_idle()


class IntegralTimeStrip(QWidget):
    """Slim, collapsible time-domain strip carrying the integration window.

    In the integral-scan view the per-run time spectra leave the centre; this
    strip keeps the one interaction they hosted — dragging the shaded
    integration window — next to the scan plot. A committed drag emits
    :attr:`window_edited` (µs); the main window pushes it into the time plot
    panel (which stays the canonical fit-range owner) and echoes changes back
    via :meth:`set_window`, so the strip, the time plot, and the build panel's
    spinboxes can never disagree.
    """

    #: (t_min, t_max) committed by dragging a window edge, already normalised.
    window_edited = Signal(float, float)

    #: Cap on preview points — the strip is a context view, not the analysis
    #: surface, so heavy spectra are strided down before plotting.
    _MAX_PREVIEW_POINTS = 4000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._time: NDArray[np.float64] | None = None
        self._asym: NDArray[np.float64] | None = None
        self._run_label = ""
        self._window: tuple[float | None, float | None] = (None, None)
        #: Active drag: 0 = min edge, 1 = max edge.
        self._drag_edge: int | None = None
        #: Edge index → drawn Line2D, rebuilt by each _render.
        self._edge_artists: dict[int, object] = {}
        #: A render was requested while the canvas was hidden (collapsed strip
        #: or inactive view); flushed when the canvas shows again.
        self._render_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.setAutoRaise(True)
        self._toggle_btn.setToolTip("Collapse/expand the time-spectrum preview.")
        self._toggle_btn.toggled.connect(self._on_toggled)
        header.addWidget(self._toggle_btn)
        title = QLabel("Integration window")
        title.setStyleSheet("font-weight: 600;")
        header.addWidget(title)
        self._window_label = QLabel("")
        self._window_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        header.addWidget(self._window_label)
        header.addStretch()
        layout.addLayout(header)

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(110)
        self._canvas.setMaximumHeight(160)
        self._ax = self._figure.add_subplot(111)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        layout.addWidget(self._canvas)
        self._render()

    def _on_toggled(self, checked: bool) -> None:
        """Collapse to just the header row (the window read-out stays visible)."""
        self._canvas.setVisible(checked)
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        if checked and self._render_pending:
            self._render()

    def showEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Flush a render skipped while the strip's canvas was hidden."""
        super().showEvent(event)
        if self._render_pending and self._canvas.isVisible():
            self._render()

    # --- data / window state --------------------------------------------------

    def show_dataset(self, time, asymmetry, label: str = "") -> None:
        """Show *time* vs *asymmetry* as the preview spectrum (strided if heavy)."""
        t = np.asarray(time, dtype=float)
        a = np.asarray(asymmetry, dtype=float)
        if t.size > self._MAX_PREVIEW_POINTS:
            step = t.size // self._MAX_PREVIEW_POINTS
            idx = np.arange(0, t.size, step)
            if idx[-1] != t.size - 1:
                idx = np.append(idx, t.size - 1)  # keep the spectrum's true end
            t, a = t[idx], a[idx]
        self._time, self._asym = t, a
        self._run_label = str(label)
        self._update_window_label()
        self._render()

    def clear(self) -> None:
        """Drop the preview spectrum (the window itself mirrors the plot panel)."""
        self._time = None
        self._asym = None
        self._run_label = ""
        self._update_window_label()
        self._render()

    def set_window(self, t_min: float | None, t_max: float | None) -> None:
        """Echo the canonical integration window into the strip (no re-emit).

        A no-op when unchanged: the time plot emits ``fit_range_changed`` on
        every motion event of its own fit-range drag, and this echo must not
        pay a full strip re-render per event.
        """
        window = (None, None) if t_min is None or t_max is None else (float(t_min), float(t_max))
        if window == self._window:
            return
        self._window = window
        self._update_window_label()
        if self._drag_edge is None:  # don't fight an in-flight drag
            self._render()

    def window(self) -> tuple[float | None, float | None]:
        """Return the currently displayed (t_min, t_max) window."""
        return self._window

    def _update_window_label(self) -> None:
        lo, hi = self._window
        if lo is not None and hi is not None:
            mn, mx = (lo, hi) if lo <= hi else (hi, lo)  # readable mid-drag too
            window_text = f"{mn:.4g} ≤ t ≤ {mx:.4g} µs"
        else:
            window_text = ""
        parts = [p for p in (self._run_label, window_text) if p]
        self._window_label.setText("   ·   ".join(parts))

    # --- window-edge drag -------------------------------------------------------

    def _on_press(self, event: object) -> None:
        lo, hi = self._window
        if event.inaxes is not self._ax or event.button != 1 or lo is None or hi is None:
            return
        handles = [(float(lo), 0), (float(hi), 1)]
        # 12 device px ≈ 6 logical px on a 2× display.
        self._drag_edge = nearest_handle(self._ax, handles, event.x, tolerance_px=12.0)

    def _on_motion(self, event: object) -> None:
        if self._drag_edge is None or event.inaxes is not self._ax or event.xdata is None:
            return
        x = float(event.xdata)
        lo, hi = self._window
        self._window = (x, hi) if self._drag_edge == 0 else (lo, x)
        self._update_window_label()
        line = self._edge_artists.get(self._drag_edge)
        if line is not None:
            line.set_xdata([x, x])
            self._canvas.draw_idle()
        else:
            self._render()

    def _on_release(self, event: object) -> None:
        # Only a left-button release ends the drag and commits the window; a
        # right/middle release mid-gesture (left still held) is ignored so it
        # cannot push a stray window into the canonical time plot.
        if self._drag_edge is None or getattr(event, "button", None) != 1:
            return
        self._drag_edge = None
        lo, hi = self._window
        if lo is None or hi is None:
            return
        mn, mx = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
        self._window = (mn, mx)
        self._update_window_label()
        self._render()  # restore the shaded span at the final edges
        self.window_edited.emit(mn, mx)

    # --- rendering ---------------------------------------------------------------

    def _render(self) -> None:
        if not self._canvas.isVisible():
            # Collapsed strip / inactive view: defer to the next show. State
            # (data, window, label) is already up to date, so nothing is lost.
            self._render_pending = True
            return
        self._render_pending = False
        self._ax.clear()
        self._edge_artists = {}
        if self._time is None or self._time.size == 0:
            draw_empty_state_message(self._ax, "Select a run to preview the integration window")
            self._canvas.draw_idle()
            return
        self._ax.plot(self._time, self._asym, ".", markersize=2, color=tokens.PLOT_LOW_COUNT)
        lo, hi = self._window
        if lo is not None and hi is not None:
            # Same styled span as the time plot's fit range — it IS that range.
            _span, left_line, right_line = draw_fit_range_span(self._ax, lo, hi)
            self._edge_artists = {0: left_line, 1: right_line}
        style_axes(self._ax)
        self._ax.set_xlabel("t (µs)", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._canvas.draw_idle()


class IntegralScanPanel(QWidget):
    """Central workspace page for the integral-scan (ALC) representation.

    Hosts the scan view's plot section (the ALC curve with its view controls
    and draggable baseline/peak handles) above a slim, collapsible time strip
    carrying the draggable integration window. The scan view's analysis
    section (Baseline/Peaks/RF) stays in the Parameters dock; ALCScanView
    remains the single owner of all scan state and logic.
    """

    def __init__(self, scan_view: ALCScanView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scan_view = scan_view
        self._time_strip = IntegralTimeStrip()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(scan_view.plot_widget(), 1)
        layout.addWidget(self._time_strip)

    def time_strip(self) -> IntegralTimeStrip:
        """The integration-window strip (the main window wires and feeds it)."""
        return self._time_strip

    def clear(self) -> None:
        """Clear the scan and the strip (PlotWorkspacePanel.clear calls this)."""
        self._scan_view.clear()
        self._time_strip.clear()

    def export_current_plot(self) -> None:
        """Save the scan figure to an image file."""
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Export scan plot",
            default_export_path("alc_scan.png"),
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)",
        )
        if not path:
            return
        try:
            # savefig raises ValueError (not OSError) for an unsupported
            # extension typed into the free-text dialog field.
            self._scan_view.figure().savefig(path, dpi=200)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export scan plot", f"Could not save the plot: {exc}")
            return
        remember_export_path(path)
