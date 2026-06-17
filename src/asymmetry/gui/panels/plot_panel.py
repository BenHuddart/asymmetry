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
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.spectrum import reference_field_gauss
from asymmetry.core.fourier.units import convert as convert_field_unit
from asymmetry.core.fourier.units import gauss_to_mhz
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    available_background_modes,
    resolve_background_mode,
)
from asymmetry.core.transform.grouping import good_event_count, group_forward_backward
from asymmetry.core.transform.integral import integrate_curve
from asymmetry.core.transform.peakfit import parabolic_peak
from asymmetry.core.transform.rebin import rebin, resolve_binning_mode
from asymmetry.core.utils.constants import (
    PeriodMode,
)
from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)
from asymmetry.gui.gle_settings import get_gle_executable
from asymmetry.gui.panels.draggable_handles import nearest_handle
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.plots import (
    draw_empty_state_message,
    draw_fit_range_span,
    draw_zero_line,
    style_axes,
    style_figure,
    style_legend,
)
from asymmetry.gui.styles.widgets import build_nav_button_qss
from asymmetry.gui.widgets.projection_chip_bar import ProjectionChipBar
from asymmetry.gui.widgets.rrf_controls import (
    install_rrf_controls,
    rrf_display_dataset,
    rrf_display_fit_curve,
    rrf_draw_badge,
)

# Metadata fields available for dataset labelling in the legend.
_LABEL_FIELDS: list[tuple[str, str]] = [
    ("Run", "run"),
    ("Field (G)", "field"),
    ("Temperature (K)", "temperature"),
    ("Comment", "comment"),
]

_TIME_VIEW_FIELDS: list[tuple[str, str]] = [
    ("FB Asymmetry", "fb_asymmetry"),
    ("Individual Groups", "groups"),
    ("Raw Counts", "raw_counts"),
]

#: Map the frequency plot panel's x-unit tokens to ``core.fourier.units`` units.
_FREQUENCY_X_UNIT_FIELD = {
    "frequency_mhz": "mhz",
    "field_gauss": "gauss",
    "field_tesla": "tesla",
}

# Neutral y-range for a stacked projection subplot whose asymmetry is entirely
# non-finite (no data). Without it matplotlib keeps its default (0, 1) box,
# which reads as "counts" rather than "asymmetry, no data".
_EMPTY_PROJECTION_YLIM = (-0.3, 0.3)


class _FloatLimitField(QLineEdit):
    """Plain text field that stores a floating-point axis limit."""

    def __init__(
        self,
        value: float,
        *,
        decimals: int = 3,
        minimum_width: int = 76,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._decimals = max(0, int(decimals))
        self._last_value = float(value)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setClearButtonEnabled(False)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(minimum_width)
        self.setFont(mono_font(11.0))
        validator = QDoubleValidator(-1e6, 1e6, self._decimals, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        self.editingFinished.connect(self._normalize_text)
        self.setValue(value)

    def value(self) -> float:
        """Return the current numeric value, falling back to the last valid value."""
        text = self.text().strip()
        if not text:
            return self._last_value
        try:
            value = float(text)
        except ValueError:
            return self._last_value
        self._last_value = value
        return value

    def setValue(self, value: float) -> None:  # noqa: N802
        """Set the displayed text from a numeric value."""
        numeric_value = float(value)
        self._last_value = numeric_value
        self.setText(f"{numeric_value:.{self._decimals}f}")

    def _normalize_text(self) -> None:
        """Rewrite the field using the canonical numeric format."""
        self.setValue(self.value())


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
    view_limits_changed = Signal(float, float, float, float)
    polarization_axis_changed = Signal(str)
    #: Emitted with the projection label when a stacked subplot is clicked to
    #: become the single-fit target (multi-subplot / ALL view only).
    fit_target_projection_changed = Signal(str)
    overlay_toggled = Signal(bool)
    time_view_changed = Signal(str)
    # Payload dict for the status-bar cursor readout, or None to clear. Keys:
    # x, y (snapped where possible), err, snr, peak (x,y)|None,
    # window (mean,err,n)|None. See _build_cursor_readout.
    cursor_coords_changed = Signal(object)
    #: Spectral-moments overlay drags (frequency panel): window in canonical MHz.
    moments_window_changed = Signal(float, float)
    moments_cutoff_changed = Signal(float)  # new cutoff fraction in [0, 1)

    def __init__(self, parent: QWidget | None = None, *, domain: str = "time") -> None:
        super().__init__(parent)
        self._domain = "frequency" if str(domain).strip().lower() == "frequency" else "time"
        self._current_time_view_mode = "fb_asymmetry"
        self._available_time_view_modes = ["fb_asymmetry"]
        self._default_canvas_min_height = 360
        self._subplot_canvas_height_per_axis = 220
        self._current_frequency_x_unit = "frequency_mhz"
        self._frequency_axis_relative_to_reference = False
        self._frequency_reference_mhz: float | None = None
        #: True when the active frequency dataset is a muoniated-radical
        #: correlation spectrum, whose x-axis is a hyperfine coupling (A_µ, MHz)
        #: and must not be field-converted by the MHz/G/T selector.
        self._frequency_axis_is_correlation = False
        self._frequency_correlation_x_label = "Muon hyperfine coupling Aμ (MHz)"
        #: Field-unit view stashed on entry to a correlation spectrum and
        #: restored when leaving it, so the user's MHz/G/T + relative choice
        #: survives the excursion.
        self._frequency_axis_unit_before_correlation = "frequency_mhz"
        self._frequency_axis_relative_before_correlation = False
        self._frequency_x_limits_by_unit: dict[str, tuple[float, float]] = {}
        #: Optional ``(run_number, time_us, signal)`` diamagnetic-fit overlay for
        #: the time view; only drawn when the displayed run matches ``run_number``.
        self._diamagnetic_overlay: tuple[int | None, np.ndarray, np.ndarray] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._figure = Figure(tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            self._canvas.setMinimumHeight(self._default_canvas_min_height)
            self._canvas_host = QWidget(self)
            self._canvas_host.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            self._canvas_host_layout = QVBoxLayout(self._canvas_host)
            self._canvas_host_layout.setContentsMargins(0, 0, 0, 0)
            self._canvas_host_layout.setSpacing(0)
            self._canvas_host_layout.addWidget(self._canvas)
            self._canvas_scroll_area = QScrollArea(self)
            self._canvas_scroll_area.setWidget(self._canvas_host)
            self._canvas_scroll_area.setWidgetResizable(True)
            self._canvas_scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
            self._canvas_scroll_area.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._canvas_scroll_area.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._canvas_scroll_area.verticalScrollBar().setObjectName("plotScroll")
            self._ax = self._figure.add_subplot(111)
            style_figure(self._figure)
            style_axes(self._ax)
            self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)
            self._nav_toolbar.hide()
            self._axis_limit_callback_ids: list[tuple[object, int, int]] = []
            self._syncing_limits_from_axes = False
            self._connect_axis_limit_callbacks([self._ax])
            default_x_label, default_y_label = self._default_axis_labels()
            self._ax.set_xlabel(default_x_label)
            self._ax.set_ylabel(default_y_label)
            self._default_x_axis_label_color = self._x_axis_label_color(self._ax)
            self._default_x_axis_tick_color = self._x_axis_tick_color(self._ax)

            # Fit-range interaction state.
            self._fit_x_min: float | None = None
            self._fit_x_max: float | None = None
            self._fit_span_artists: list[object] = []
            self._fit_min_handles: list[object] = []
            self._fit_max_handles: list[object] = []
            self._active_fit_handle: str | None = None
            self._active_fit_axis = None
            self._drag_started = False

            # Spectral-moments overlay state (frequency panel only). The window
            # is held in canonical absolute MHz; the cutoff line sits at
            # ``peak · cutoff_fraction`` in amplitude.
            self._moments_window_mhz: tuple[float, float] | None = None
            self._moments_cutoff_fraction: float = 0.0
            self._moments_peak_amp: float | None = None
            self._moments_overlay_visible: bool = False
            self._moments_span_artists: list[object] = []
            self._moments_cutoff_artists: list[object] = []
            self._active_moments_handle: str | None = None

            # Add plot limit controls toolbar
            self._create_limit_controls()
            self._sync_navigation_buttons()
            layout.addLayout(self._limit_toolbar)

            self._projection_bar = ProjectionChipBar()
            self._projection_bar.selection_changed.connect(self._on_projection_selection_changed)

            nav_row = QHBoxLayout()
            nav_row.setContentsMargins(4, 0, 4, 0)
            nav_row.setSpacing(4)
            # The label combo applies to both panels: it labels overlaid traces
            # (multiple runs in the time panel; multiple runs' spectra in the
            # frequency panel) by run / temperature / field / custom column.
            nav_row.addWidget(QLabel("Label:"))
            nav_row.addWidget(self._label_field_combo)
            nav_row.addWidget(self._time_view_label)
            nav_row.addWidget(self._time_view_combo)
            nav_row.addWidget(self._log_counts_checkbox)
            nav_row.addWidget(self._overlay_checkbox)
            nav_row.addStretch()
            nav_row.addWidget(self._projection_bar)
            nav_row.addSpacing(4)

            _nav_qss = build_nav_button_qss()
            self._pan_btn = QPushButton("Pan")
            self._pan_btn.setCheckable(True)
            self._pan_btn.setMaximumWidth(60)
            self._pan_btn.setStyleSheet(_nav_qss)
            self._pan_btn.clicked.connect(self._on_pan_button_clicked)
            nav_row.addWidget(self._pan_btn)

            self._zoom_btn = QPushButton("Zoom")
            self._zoom_btn.setCheckable(True)
            self._zoom_btn.setMaximumWidth(60)
            self._zoom_btn.setStyleSheet(_nav_qss)
            self._zoom_btn.clicked.connect(self._on_zoom_button_clicked)
            nav_row.addWidget(self._zoom_btn)

            layout.addLayout(nav_row)

            layout.addWidget(install_rrf_controls(self))

            self._plot_header = self._create_plot_header()
            layout.addWidget(self._plot_header)
            layout.addWidget(self._canvas_scroll_area)
            self._plot_footer = self._create_plot_footer()
            layout.addWidget(self._plot_footer)
            self._has_mpl = True

            # Store current dataset for rebunching
            self._current_dataset = None
            self._current_datasets: list[MuonDataset] = []
            self._limits_initialized = False
            self._current_polarization_axis: str | None = None
            # Ordered projection specs ({"label", "tint"}) and the per-label tint
            # lookup used for frame-tinting subplots; driven by the chip bar.
            self._projection_specs: list[dict] = []
            self._tint_by_label: dict[str, str] = {}
            self._selected_projection_labels: list[str] = []
            # Which stacked subplot is the active single-fit target (multi-view).
            self._fit_target_projection: str | None = None
            self._fit_target_artists: list = []
            self._y_limits_by_polarization: dict[str, tuple[float, float]] = {}
            self._subplot_axes_by_polarization: dict[str, object] = {}
            self._vector_subplot_datasets: dict[str, list[MuonDataset]] = {}
            self._grouped_time_subplot_datasets: list[MuonDataset] = []
            # Tracks the intended stacked-subplot count independently of the
            # _subplot_axes_by_polarization dict, which is briefly empty while
            # plot_vector_subplots/plot_grouped_time_domain_subplots rebuild axes.
            # resizeEvent reads this so it doesn't reset the scroll area during
            # that window.
            self._stacked_axis_count: int = 1

            # Store fit curve data to persist across redraws
            self._fit_curve = None  # (t_fit, y_fit, label) for single fits
            self._fit_curve_run_number = None
            self._fit_curves = {}  # {run_number: (t_fit, y_fit, label)} for global fits
            self._fit_curves_by_key: dict[tuple[int, str | None], tuple] = {}

            # Per-fit additive component curves for shading.
            self._fit_components = None  # list[(name, y_component)] for single fit
            self._fit_components_by_run = {}  # {run_number: list[(name, y_component)]}
            self._fit_components_by_key: dict[tuple[int, str | None], list[tuple[str, object]]] = {}

            # Per-run fit metadata for export headers.
            self._fit_metadata: dict[int, dict] = {}  # {run_number: {formula, chi2, ...}}
            self._fit_metadata_by_key: dict[tuple[int, str | None], dict] = {}

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
            self._decimation_enabled = True
            self._decimation_applied_for_current_view = False
            self._decimation_points_shown = 0
            self._decimation_points_total = 0
            #: Chip Text artist per axis id; cleared on every view reset.
            self._decimation_chip_artists: dict[int, object] = {}
            self._max_render_points_per_trace = 4000
            self._viewport_refresh_pending = False
            self._viewport_refresh_in_progress = False

            self._canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion_notify)
            self._canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
            self._canvas.mpl_connect("draw_event", self._on_canvas_draw_event)
        except ImportError:
            layout.addWidget(QLabel("matplotlib not installed — plotting disabled"))
            self._has_mpl = False

    def _create_limit_controls(self) -> None:
        """Create inline controls for adjusting plot limits and plot options."""
        self._limit_toolbar = QVBoxLayout()
        self._limit_toolbar.setSpacing(2)
        self._limit_toolbar.setContentsMargins(4, 4, 4, 4)

        # ── Row 0: axis range fields + auto-scale buttons ─────────────────
        row0 = QHBoxLayout()
        row0.setSpacing(4)

        # X-axis limits
        row0.addWidget(QLabel("X:"))
        self._x_min = _FloatLimitField(0.0, minimum_width=76)
        row0.addWidget(self._x_min)
        row0.addWidget(QLabel("–"))
        self._x_max = _FloatLimitField(10.0, minimum_width=76)
        row0.addWidget(self._x_max)
        self._x_unit_label = QLabel("MHz" if self._is_frequency_plot_panel() else "μs")
        row0.addWidget(self._x_unit_label)

        # Y-axis limits
        row0.addWidget(QLabel("Y:"))
        self._y_min = _FloatLimitField(-30.0, minimum_width=76)
        row0.addWidget(self._y_min)
        row0.addWidget(QLabel("–"))
        self._y_max = _FloatLimitField(30.0, minimum_width=76)
        row0.addWidget(self._y_max)
        self._y_unit_label = QLabel("a.u." if self._is_frequency_plot_panel() else "%")
        row0.addWidget(self._y_unit_label)

        # Separate axis auto-scale controls.
        _nav_qss = build_nav_button_qss()
        self._auto_x_btn = QPushButton("Auto X")
        self._auto_x_btn.setCheckable(True)
        self._auto_x_btn.setStyleSheet(_nav_qss)
        self._auto_x_btn.clicked.connect(self._on_auto_x_button_clicked)
        self._auto_x_btn.setMaximumWidth(65)
        row0.addWidget(self._auto_x_btn)

        self._auto_y_btn = QPushButton("Auto Y")
        self._auto_y_btn.setCheckable(True)
        self._auto_y_btn.setStyleSheet(_nav_qss)
        self._auto_y_btn.clicked.connect(self._on_auto_y_button_clicked)
        self._auto_y_btn.setMaximumWidth(65)
        row0.addWidget(self._auto_y_btn)

        row0.addStretch()
        self._limit_toolbar.addLayout(row0)

        # Apply limit changes immediately from text field edits.
        self._x_min.editingFinished.connect(self._on_limit_fields_edited)
        self._x_max.editingFinished.connect(self._on_limit_fields_edited)
        self._y_min.editingFinished.connect(self._on_limit_fields_edited)
        self._y_max.editingFinished.connect(self._on_limit_fields_edited)

        # Keep bunching control internal (hidden) for backward compatibility
        # with project state and tests; it is intentionally not shown in UI.
        self._bunch_factor = QSpinBox()
        self._bunch_factor.setRange(1, 1000)
        self._bunch_factor.setValue(1)
        self._bunch_factor.setMaximumWidth(60)
        self._bunch_factor.valueChanged.connect(self._on_bunch_changed)
        self._bunch_factor.hide()

        # Legend label-field preferences. Initialised unconditionally (not only on
        # the matplotlib-available path) so the label combo can be built even in a
        # headless/no-mpl panel. Preferences can be scoped per Data Group.
        self._active_label_group_id: str | None = None
        self._default_label_field: str = "run"
        self._label_field_by_group: dict[str, str] = {}
        #: User-defined data-browser custom columns offered as legend labels, as
        #: ``(display_label, "custom:<id>")`` pairs pushed in by the host (see
        #: :meth:`set_custom_label_fields`). Their per-run values are read straight
        #: from ``dataset.metadata["custom_fields"]``.
        self._custom_label_fields: list[tuple[str, str]] = []

        # Label and Overlay widgets are created here but placed in the nav row below.
        self._label_field_combo = QComboBox()
        self._rebuild_label_field_combo()
        self._label_field_combo.setMaximumWidth(140)
        self._label_field_combo.currentIndexChanged.connect(self._on_label_field_changed)

        self._time_view_label = QLabel("Time View:")
        self._time_view_combo = QComboBox()
        self._time_view_combo.setMaximumWidth(160)
        self._time_view_combo.currentIndexChanged.connect(self._on_time_view_mode_changed)
        self.set_time_view_modes(self._available_time_view_modes, self._current_time_view_mode)
        self._time_view_label.hide()
        self._time_view_combo.hide()

        # Log-count diagnostic: a log-y toggle that applies only on the
        # raw-counts view. A pure muon-decay histogram is a straight line on a
        # log count axis, so a mis-placed t0, a wrong background level, or
        # high-rate deadtime curvature jump out without a fit.
        self._log_counts_enabled = False
        self._log_counts_checkbox = QCheckBox("Log scale")
        self._log_counts_checkbox.setChecked(False)
        self._log_counts_checkbox.setToolTip(
            "Plot raw counts on a logarithmic y-axis — a pure decay is a "
            "straight line, so t0, background and deadtime deviations are "
            "obvious. Non-positive bins are dropped (log undefined)."
        )
        self._log_counts_checkbox.toggled.connect(self._on_log_counts_toggled)
        self._log_counts_checkbox.hide()

        self._overlay_checkbox = QCheckBox("Overlay")
        self._overlay_checkbox.setChecked(False)
        self._overlay_checkbox.toggled.connect(self.overlay_toggled.emit)

        # ── Row 1: frequency-specific controls (surfaceAlt tinted second row) ──
        if self._is_frequency_plot_panel():
            row1_widget = QWidget()
            row1_widget.setObjectName("plotFrequencyRow")
            # Scope to the container: a bare QWidget selector would cascade
            # the border/background onto every child label and combo.
            row1_widget.setStyleSheet(
                f"QWidget#plotFrequencyRow {{ background-color: {tokens.SURFACE_ALT};"
                f" border-top: 1px solid {tokens.BORDER}; }}"
            )
            row1 = QHBoxLayout(row1_widget)
            row1.setContentsMargins(4, 3, 4, 3)
            row1.setSpacing(8)

            row1.addWidget(QLabel("X Units:"))
            self._frequency_x_unit_combo = QComboBox()
            self._frequency_x_unit_combo.addItem("Frequency (MHz)", userData="frequency_mhz")
            self._frequency_x_unit_combo.addItem("Field (G)", userData="field_gauss")
            self._frequency_x_unit_combo.addItem("Field (T)", userData="field_tesla")
            self._frequency_x_unit_combo.currentIndexChanged.connect(
                self._on_frequency_x_unit_changed
            )
            row1.addWidget(self._frequency_x_unit_combo)

            self._frequency_axis_relative_check = QCheckBox("X relative to ref. field")
            self._frequency_axis_relative_check.setChecked(
                self._frequency_axis_relative_to_reference
            )
            self._frequency_axis_relative_check.toggled.connect(
                self._on_frequency_relative_check_toggled
            )
            row1.addWidget(self._frequency_axis_relative_check)

            row1.addWidget(QLabel("Reference:"))
            self._frequency_reference_spin = QDoubleSpinBox()
            self._frequency_reference_spin.setRange(0.0, 1_000_000.0)
            self._frequency_reference_spin.setDecimals(2)
            self._frequency_reference_spin.setSuffix(" G")
            self._frequency_reference_spin.setValue(0.0)
            self._frequency_reference_spin.setMinimumWidth(90)
            self._frequency_reference_spin.setFont(mono_font(11.0))
            self._frequency_reference_spin.setEnabled(False)
            self._frequency_reference_spin.editingFinished.connect(
                self._on_frequency_reference_spin_committed
            )
            row1.addWidget(self._frequency_reference_spin)

            row1.addStretch()
            self._limit_toolbar.addWidget(row1_widget)

    def _is_frequency_plot_panel(self) -> bool:
        """Return True when this panel is dedicated to frequency-domain viewing."""
        return self._domain == "frequency"

    def _create_plot_header(self) -> QWidget:
        """Return the title strip shown above the canvas."""
        widget = QWidget()
        widget.setObjectName("plotHeaderStrip")
        widget.setStyleSheet(
            f"QWidget#plotHeaderStrip {{ background-color: {tokens.SURFACE_ALT};"
            f" border-bottom: 1px solid {tokens.BORDER}; }}"
        )
        row = QHBoxLayout(widget)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        self._header_title_label = QLabel()
        title_font = self._header_title_label.font()
        title_font.setPointSizeF(11.0)
        title_font.setWeight(title_font.Weight.DemiBold)
        self._header_title_label.setFont(title_font)
        self._header_title_label.setStyleSheet(f"color: {tokens.TEXT};")
        row.addWidget(self._header_title_label, 1)

        self._header_meta_label = QLabel()
        self._header_meta_label.setFont(mono_font(10.5))
        self._header_meta_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._header_meta_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        row.addWidget(self._header_meta_label)

        self._update_plot_header()
        return widget

    def _create_plot_footer(self) -> QWidget:
        """Return the control bar shown below the canvas."""
        widget = QWidget()
        widget.setObjectName("plotFooterStrip")
        widget.setStyleSheet(
            f"QWidget#plotFooterStrip {{ background-color: {tokens.SURFACE_ALT};"
            f" border-top: 1px solid {tokens.BORDER}; }}"
        )
        row = QHBoxLayout(widget)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        self._add_label_btn = QPushButton("Add Annotation")
        self._add_label_btn.setCheckable(True)
        min_btn_width = self._add_label_btn.fontMetrics().horizontalAdvance("Add Annotation") + 32
        self._add_label_btn.setMinimumWidth(min_btn_width)
        row.addWidget(self._add_label_btn)
        row.addStretch()

        # One export button hosting a menu (no second export button, per the
        # workflow-visualisation brief): GLE export and a plain-text data
        # export that skips the GLE folder/script/compile.
        self._export_gle_btn = QPushButton("Export…")
        self._export_gle_btn.setEnabled(False)
        self._export_menu = QMenu(self._export_gle_btn)
        self._export_menu.addAction("Export to GLE…", self.export_plots_to_gle)
        self._export_menu.addAction("Export plotted data (text)…", self.export_plotted_data_as_text)
        self._export_gle_btn.setMenu(self._export_menu)
        row.addWidget(self._export_gle_btn)

        row.addWidget(QLabel("Format:"))

        self._gle_format_combo = QComboBox()
        self._gle_format_combo.addItems(["PDF", "EPS"])
        self._gle_format_combo.setEnabled(False)
        row.addWidget(self._gle_format_combo)

        return widget

    def _update_plot_header(self) -> None:
        """Refresh the title-strip labels from the current plot state."""
        if not hasattr(self, "_header_title_label"):
            return
        self._header_title_label.setText(self._header_title_text())

    def _header_title_text(self) -> str:
        """Return the left-side text for the plot title strip."""
        if self._domain == "frequency":
            domain = "Fourier spectrum"
            multi_suffix = "runs overlaid"
        elif getattr(self, "_current_time_view_mode", "fb_asymmetry") == "groups":
            domain = "Grouped time-domain"
            multi_suffix = "runs"
        elif getattr(self, "_current_time_view_mode", "fb_asymmetry") == "raw_counts":
            domain = "Raw counts"
            multi_suffix = "runs"
        else:
            domain = "Time-domain F-B asymmetry"
            multi_suffix = "runs highlighted"

        datasets = getattr(self, "_current_datasets", [])
        if not datasets:
            return domain
        if len(datasets) == 1:
            return f"{domain} — {datasets[0].run_label}"
        return f"{domain} — {len(datasets)} {multi_suffix}"

    def _set_canvas_minimum_height_for_axes(self, axis_count: int) -> None:
        """Scale the canvas height so stacked subplot views scroll vertically."""
        if not hasattr(self, "_canvas"):
            return
        count = max(1, int(axis_count))
        # Set _stacked_axis_count before clearing _subplot_axes_by_polarization
        # so that any resize events that fire while the dict is temporarily empty
        # still see the correct intended count.
        self._stacked_axis_count = count
        height = max(self._default_canvas_min_height, self._subplot_canvas_height_per_axis * count)
        self._canvas.setMinimumHeight(height)
        self._sync_canvas_scroll_geometry(axis_count=count, target_height=height)
        self._canvas.updateGeometry()

    def _sync_canvas_scroll_geometry(
        self,
        *,
        axis_count: int | None = None,
        target_height: int | None = None,
    ) -> None:
        """Keep the canvas sized correctly for single-axis fill vs stacked-axis scrolling."""
        if (
            not hasattr(self, "_canvas")
            or not hasattr(self, "_canvas_host")
            or not hasattr(self, "_canvas_scroll_area")
        ):
            return

        count = max(
            1,
            int(axis_count)
            if axis_count is not None
            else max(1, len(getattr(self, "_subplot_axes_by_polarization", {}))),
        )
        height = (
            int(target_height)
            if target_height is not None
            else max(self._default_canvas_min_height, int(self._canvas.minimumHeight()))
        )
        stacked_mode = count > 1
        self._canvas_scroll_area.setWidgetResizable(not stacked_mode)
        self._canvas_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
            if stacked_mode
            else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        if stacked_mode:
            viewport = self._canvas_scroll_area.viewport()
            viewport_width = viewport.width() if viewport is not None else self.width()
            viewport_height = viewport.height() if viewport is not None else 0
            effective_height = max(height, int(viewport_height))
            self._canvas_host.setFixedWidth(max(1, int(viewport_width)))
            self._canvas_host.setFixedHeight(effective_height)
        else:
            self._canvas_host.setMinimumHeight(height)
            self._canvas_host.setMaximumHeight(16777215)
            self._canvas_host.setMinimumWidth(0)
            self._canvas_host.setMaximumWidth(16777215)
        self._canvas_host.updateGeometry()

    def _default_axis_labels(self) -> tuple[str, str]:
        """Return fallback axis labels for this panel domain."""
        if self._is_frequency_plot_panel():
            return self._display_x_label(), r"$|F|$ (arb.)"
        return "Time (μs)", "Asymmetry (%)"

    def _mhz_per_gauss(self) -> float:
        """Return the frequency equivalent of one Gauss in MHz.

        Routed through the shared ``core.fourier.units`` converter so the
        Gauss↔MHz constant lives in exactly one place.
        """
        return float(gauss_to_mhz(1.0))

    def _frequency_limit_mode_key(
        self,
        *,
        unit: str | None = None,
        relative: bool | None = None,
    ) -> str:
        """Return a stable storage key for frequency x-limit display modes."""
        resolved_unit = self._current_frequency_x_unit if unit is None else str(unit)
        resolved_relative = (
            self._frequency_axis_relative_to_reference if relative is None else bool(relative)
        )
        return f"{resolved_unit}:{'relative' if resolved_relative else 'absolute'}"

    def _frequency_reference_for_dataset(self, dataset: MuonDataset | None) -> float | None:
        """Return the applied-field reference frequency in MHz for *dataset*.

        Delegates the dataset-metadata-before-run-metadata field lookup to the
        shared core resolver and converts the resulting Gauss value to MHz; the
        unit conversion is the GUI-only part of this method.
        """
        if dataset is None:
            return None
        field_gauss = reference_field_gauss(getattr(dataset, "run", None), dataset)
        if field_gauss is None:
            return None
        return field_gauss * self._mhz_per_gauss()

    def _display_frequency_reference(self, *, unit: str | None = None) -> float:
        """Return the current frequency reference in the requested display unit."""
        reference_mhz = self._frequency_reference_mhz
        if reference_mhz is None:
            return 0.0
        resolved_unit = self._current_frequency_x_unit if unit is None else str(unit)
        field_unit = _FREQUENCY_X_UNIT_FIELD.get(resolved_unit, "mhz")
        return float(convert_field_unit(reference_mhz, "mhz", field_unit))

    def _display_x_label(self) -> str:
        """Return the x-axis label for the current display unit."""
        if not self._is_frequency_plot_panel():
            return "Time (μs)"
        if self._frequency_axis_is_correlation:
            return self._frequency_correlation_x_label
        return {
            "field_gauss": "Field (G)",
            "field_tesla": "Field (T)",
        }.get(self._current_frequency_x_unit, "Frequency (MHz)")

    def _display_x_unit_suffix(self) -> str:
        """Return the compact unit suffix for the x-limit controls."""
        if not self._is_frequency_plot_panel():
            return "μs"
        return {
            "field_gauss": "G",
            "field_tesla": "T",
        }.get(self._current_frequency_x_unit, "MHz")

    def _display_y_unit_suffix(self, y_label: str | None = None) -> str:
        """Return the compact unit suffix for the y-limit controls."""
        if not self._is_frequency_plot_panel():
            return "%"
        text = str(y_label or "").strip().lower()
        return "deg" if "deg" in text else "a.u."

    def _convert_frequency_axis_for_display(self, x_values) -> np.ndarray:
        """Convert canonical MHz axis data into the selected absolute display unit."""
        arr = np.asarray(x_values, dtype=float)
        if not self._is_frequency_plot_panel() or self._frequency_axis_is_correlation:
            return arr
        field_unit = _FREQUENCY_X_UNIT_FIELD.get(self._current_frequency_x_unit, "mhz")
        if field_unit == "mhz":
            return arr
        return np.asarray(convert_field_unit(arr, "mhz", field_unit), dtype=float)

    def _convert_frequency_axis_limit_to_control_value(self, value: float) -> float:
        """Convert one absolute axis x-limit into the toolbar control value."""
        if not self._is_frequency_plot_panel():
            return float(value)
        control_value = float(value)
        if self._frequency_axis_relative_to_reference:
            control_value -= self._display_frequency_reference(unit=self._current_frequency_x_unit)
        return control_value

    def _convert_frequency_control_value_to_axis_limit(self, value: float) -> float:
        """Convert one toolbar x-limit value into the absolute plotted axis value."""
        if not self._is_frequency_plot_panel():
            return float(value)
        axis_value = float(value)
        if self._frequency_axis_relative_to_reference:
            axis_value += self._display_frequency_reference(unit=self._current_frequency_x_unit)
        return axis_value

    def _convert_display_limit_between_units(
        self,
        value: float,
        *,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert one x-limit value between frequency-view display units."""
        if from_unit == to_unit:
            return float(value)
        source = _FREQUENCY_X_UNIT_FIELD.get(from_unit)
        target = _FREQUENCY_X_UNIT_FIELD.get(to_unit)
        if source is None or target is None:
            return float(value)
        return float(convert_field_unit(value, source, target))

    def _convert_display_limit_to_canonical_mhz(
        self,
        value: float,
        *,
        unit: str,
        relative: bool,
    ) -> float:
        """Convert one displayed x value back into canonical absolute MHz."""
        canonical = self._convert_display_limit_between_units(
            value, from_unit=unit, to_unit="frequency_mhz"
        )
        if relative:
            canonical += self._display_frequency_reference(unit="frequency_mhz")
        return canonical

    def _convert_canonical_mhz_to_display_limit(
        self,
        value: float,
        *,
        unit: str,
        relative: bool,
    ) -> float:
        """Convert one canonical absolute MHz value into the current display mode."""
        display_value = float(value)
        if relative:
            display_value -= self._display_frequency_reference(unit="frequency_mhz")
        return self._convert_display_limit_between_units(
            display_value,
            from_unit="frequency_mhz",
            to_unit=unit,
        )

    def _set_view_limits_fields(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        """Update toolbar-backed view-limit fields without drawing."""
        self._set_limit_field_value(self._x_min, float(x_min))
        self._set_limit_field_value(self._x_max, float(x_max))
        self._set_limit_field_value(self._y_min, float(y_min))
        self._set_limit_field_value(self._y_max, float(y_max))

    def set_frequency_axis_relative_to_reference(self, enabled: bool) -> None:
        """Toggle between absolute and reference-relative frequency axes."""
        if not self._is_frequency_plot_panel():
            return
        self._switch_frequency_axis_display(relative=bool(enabled))
        if hasattr(self, "_frequency_axis_relative_check"):
            prev = self._frequency_axis_relative_check.blockSignals(True)
            self._frequency_axis_relative_check.setChecked(
                bool(self._frequency_axis_relative_to_reference)
            )
            self._frequency_axis_relative_check.blockSignals(prev)

    def is_frequency_axis_relative_to_reference(self) -> bool:
        """Return whether the frequency x axis is shown relative to the field."""
        return bool(self._frequency_axis_relative_to_reference)

    def _on_frequency_relative_check_toggled(self, checked: bool) -> None:
        """Enable/disable the reference spin and apply the axis change."""
        if hasattr(self, "_frequency_reference_spin"):
            self._frequency_reference_spin.setEnabled(checked)
        self.set_frequency_axis_relative_to_reference(checked)

    def _on_frequency_reference_spin_committed(self) -> None:
        """Apply a user-supplied reference field override on editing commit."""
        if not hasattr(self, "_frequency_reference_spin"):
            return
        value_gauss = self._frequency_reference_spin.value()
        self._frequency_reference_mhz = float(value_gauss) * self._mhz_per_gauss()
        if self._frequency_axis_relative_to_reference:
            self._switch_frequency_axis_display(force=True)

    def get_frequency_view_window_mhz(
        self,
        *,
        reference_dataset: MuonDataset | None = None,
    ) -> tuple[float, float] | None:
        """Return the current frequency-view x window in canonical absolute MHz."""
        if not self._is_frequency_plot_panel() or not self._has_mpl:
            return None

        x_min, x_max, _y_min, _y_max = self.get_view_limits()
        reference_mhz = self._frequency_reference_mhz
        if reference_dataset is not None:
            dataset_reference = self._frequency_reference_for_dataset(reference_dataset)
            if dataset_reference is not None:
                reference_mhz = dataset_reference
        if reference_mhz is None:
            reference_mhz = 0.0

        def _to_absolute_mhz(value: float) -> float:
            absolute = self._convert_display_limit_between_units(
                value,
                from_unit=self._current_frequency_x_unit,
                to_unit="frequency_mhz",
            )
            if self._frequency_axis_relative_to_reference:
                absolute += float(reference_mhz)
            return float(absolute)

        lo = _to_absolute_mhz(float(x_min))
        hi = _to_absolute_mhz(float(x_max))
        return (lo, hi) if lo <= hi else (hi, lo)

    def _switch_frequency_axis_display(
        self,
        *,
        unit: str | None = None,
        relative: bool | None = None,
        force: bool = False,
    ) -> None:
        """Switch frequency-axis display mode with a single redraw."""
        if not self._is_frequency_plot_panel():
            return

        old_unit = self._current_frequency_x_unit
        old_relative = self._frequency_axis_relative_to_reference
        new_unit = old_unit if unit is None else str(unit)
        new_relative = old_relative if relative is None else bool(relative)
        if not force and new_unit == old_unit and new_relative == old_relative:
            return

        current_x_min, current_x_max, current_y_min, current_y_max = self.get_view_limits()
        old_key = self._frequency_limit_mode_key(unit=old_unit, relative=old_relative)
        self._frequency_x_limits_by_unit[old_key] = (float(current_x_min), float(current_x_max))

        new_key = self._frequency_limit_mode_key(unit=new_unit, relative=new_relative)
        if new_key in self._frequency_x_limits_by_unit:
            new_x_min, new_x_max = self._frequency_x_limits_by_unit[new_key]
        else:
            canonical_min = self._convert_display_limit_to_canonical_mhz(
                current_x_min,
                unit=old_unit,
                relative=old_relative,
            )
            canonical_max = self._convert_display_limit_to_canonical_mhz(
                current_x_max,
                unit=old_unit,
                relative=old_relative,
            )
            new_x_min = self._convert_canonical_mhz_to_display_limit(
                canonical_min,
                unit=new_unit,
                relative=new_relative,
            )
            new_x_max = self._convert_canonical_mhz_to_display_limit(
                canonical_max,
                unit=new_unit,
                relative=new_relative,
            )

        self._current_frequency_x_unit = new_unit
        self._frequency_axis_relative_to_reference = new_relative
        self._set_view_limits_fields(new_x_min, new_x_max, current_y_min, current_y_max)

        if self.has_plot_content():
            self._redraw_current_view()
        else:
            self._apply_axis_labels(*self._default_axis_labels())
            self._apply_limits()

    def _current_axis_labels(self) -> tuple[str, str]:
        """Return the axis labels for the currently plotted datasets."""
        dataset = self._current_datasets[0] if self._current_datasets else self._current_dataset
        return self._axis_labels_for_dataset(dataset, self._current_polarization_axis)

    def _apply_axis_labels(self, x_label: str, y_label: str) -> None:
        """Apply axis labels and keep the limit-unit helpers in sync."""
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)
        self._apply_x_axis_decimation_indicator(self._ax)
        if hasattr(self, "_x_unit_label"):
            self._x_unit_label.setText(self._display_x_unit_suffix())
        if hasattr(self, "_y_unit_label"):
            self._y_unit_label.setText(self._display_y_unit_suffix(y_label))

    def _x_axis_label_color(self, ax) -> str:
        """Return the current x-axis label color for ``ax``."""
        xaxis = getattr(ax, "xaxis", None)
        label = getattr(xaxis, "label", None)
        color_getter = getattr(label, "get_color", None)
        if callable(color_getter):
            return str(color_getter())
        return "black"

    def _x_axis_tick_color(self, ax) -> str:
        """Return the current x-axis tick-label color for ``ax``."""
        xaxis = getattr(ax, "xaxis", None)
        ticklabels_getter = getattr(xaxis, "get_ticklabels", None)
        if callable(ticklabels_getter):
            for tick in ticklabels_getter():
                color_getter = getattr(tick, "get_color", None)
                if callable(color_getter):
                    return str(color_getter())
        return self._x_axis_label_color(ax)

    def _reset_decimation_view_state(self) -> None:
        """Reset per-view decimation tracking: flag, chip counters, chip artists.

        Removing and clearing the chip artists here (every plot path resets
        before drawing) keeps the per-axis dict from accumulating entries for
        axes that ``clf()`` has already destroyed.
        """
        self._decimation_applied_for_current_view = False
        self._decimation_points_shown = 0
        self._decimation_points_total = 0
        artists = getattr(self, "_decimation_chip_artists", None)
        if artists:
            for artist in artists.values():
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):
                    pass  # the axis was already cleared/destroyed
            artists.clear()

    @staticmethod
    def _format_point_count(count: int) -> str:
        """Human-readable point count for the decimation chip (1.2M, 4.0k, 312)."""
        value = float(count)
        if value >= 1e6:
            return f"{value / 1e6:.1f}M"
        if value >= 1e3:
            return f"{value / 1e3:.1f}k"
        return str(int(value))

    _DECIMATION_TOOLTIP = (
        "Display decimated for responsiveness — zoom in for full resolution. "
        "Fits, transforms and exports always use every point."
    )

    def _apply_x_axis_decimation_indicator(self, ax) -> None:
        """Maintain the corner chip that flags a display-decimated view.

        A small translucent badge ("4.0k of 1.2M pts") in the plot corner —
        it disappears once the user zooms in far enough that every visible
        point is rendered, which itself teaches how the decimation behaves.
        The canvas tooltip carries the full explanation, including that only
        the *display* is decimated.
        """
        artists = getattr(self, "_decimation_chip_artists", None)
        if artists is None:
            artists = {}
            self._decimation_chip_artists = artists
        existing = artists.pop(id(ax), None)
        if existing is not None:
            try:
                existing.remove()
            except (ValueError, NotImplementedError):
                pass  # axis was cleared; the artist is already gone

        active = bool(getattr(self, "_decimation_applied_for_current_view", False))
        canvas = getattr(self, "_canvas", None)
        if not active:
            if canvas is not None and hasattr(canvas, "setToolTip"):
                canvas.setToolTip("")
            return

        shown = int(getattr(self, "_decimation_points_shown", 0))
        total = int(getattr(self, "_decimation_points_total", 0))
        if total > 0 and shown > 0:
            chip_text = (
                f"{self._format_point_count(shown)} of {self._format_point_count(total)} pts"
            )
        else:
            chip_text = "decimated display"
        text_fn = getattr(ax, "text", None)
        if callable(text_fn):
            artists[id(ax)] = text_fn(
                0.99,
                0.015,
                chip_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="white",
                zorder=20,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "0.25",
                    "alpha": 0.65,
                    "edgecolor": "none",
                },
            )
        if canvas is not None and hasattr(canvas, "setToolTip"):
            canvas.setToolTip(self._DECIMATION_TOOLTIP)

    def decimation_chip_text(self, ax=None) -> str | None:
        """Return the decimation chip text on *ax* (None when not displayed)."""
        artists = getattr(self, "_decimation_chip_artists", {})
        artist = artists.get(id(ax if ax is not None else self._ax))
        return str(artist.get_text()) if artist is not None else None

    def _on_frequency_x_unit_changed(self, _index: int) -> None:
        """Switch the displayed frequency x-axis between MHz and Gauss."""
        if not self._is_frequency_plot_panel() or not hasattr(self, "_frequency_x_unit_combo"):
            return

        new_unit = str(self._frequency_x_unit_combo.currentData() or "frequency_mhz")
        self._switch_frequency_axis_display(unit=new_unit)

    def active_domain(self) -> str:
        """Return the fixed domain represented by this panel."""
        return self._domain

    def has_plot_content(self) -> bool:
        """Return True when the panel currently holds plotted datasets."""
        return bool(self._current_dataset is not None or self._current_datasets)

    def _current_navigation_mode(self) -> str:
        """Return active Matplotlib nav mode as ``none``, ``pan``, or ``zoom``."""
        toolbar = getattr(self, "_nav_toolbar", None)
        if toolbar is None:
            return "none"

        mode = getattr(toolbar, "mode", None)
        if mode is None:
            return "none"

        name = getattr(mode, "name", None)
        if isinstance(name, str):
            lowered = name.lower()
            if "pan" in lowered:
                return "pan"
            if "zoom" in lowered:
                return "zoom"
            return "none"

        mode_text = str(mode).strip().lower()
        if "pan" in mode_text:
            return "pan"
        if "zoom" in mode_text:
            return "zoom"
        return "none"

    def _sync_navigation_buttons(self) -> None:
        """Mirror current Matplotlib nav mode on Pan/Zoom toggle buttons."""
        if not hasattr(self, "_pan_btn") or not hasattr(self, "_zoom_btn"):
            return

        mode = self._current_navigation_mode()
        self._pan_btn.blockSignals(True)
        self._zoom_btn.blockSignals(True)
        self._pan_btn.setChecked(mode == "pan")
        self._zoom_btn.setChecked(mode == "zoom")
        self._pan_btn.blockSignals(False)
        self._zoom_btn.blockSignals(False)

    def _set_navigation_mode(self, mode: str) -> None:
        """Activate/deactivate Matplotlib pan/zoom mode."""
        toolbar = getattr(self, "_nav_toolbar", None)
        if toolbar is None:
            self._sync_navigation_buttons()
            return

        target = mode.lower().strip()
        current = self._current_navigation_mode()

        if target == "pan":
            if current == "zoom":
                toolbar.zoom()
                current = self._current_navigation_mode()
            if current != "pan":
                toolbar.pan()
        elif target == "zoom":
            if current == "pan":
                toolbar.pan()
                current = self._current_navigation_mode()
            if current != "zoom":
                toolbar.zoom()
        else:
            if current == "pan":
                toolbar.pan()
            elif current == "zoom":
                toolbar.zoom()

        self._sync_navigation_buttons()

    def _on_pan_button_clicked(self, checked: bool) -> None:
        """Toggle Matplotlib pan mode from toolbar button."""
        self._set_navigation_mode("pan" if checked else "none")

    def _on_zoom_button_clicked(self, checked: bool) -> None:
        """Toggle Matplotlib zoom mode from toolbar button."""
        self._set_navigation_mode("zoom" if checked else "none")

    def _connect_axis_limit_callbacks(self, axes: list[object]) -> None:
        """Attach x/y-limit listeners used to mirror interactive nav updates."""
        self._disconnect_axis_limit_callbacks()
        self._axis_limit_callback_ids = []

        for axis_obj in axes:
            callbacks = getattr(axis_obj, "callbacks", None)
            if callbacks is None:
                continue
            connect = getattr(callbacks, "connect", None)
            if connect is None:
                continue
            x_cid = connect("xlim_changed", self._on_axis_limits_changed)
            y_cid = connect("ylim_changed", self._on_axis_limits_changed)
            self._axis_limit_callback_ids.append((axis_obj, x_cid, y_cid))

    def _disconnect_axis_limit_callbacks(self) -> None:
        """Disconnect previously attached x/y-limit listeners."""
        callback_ids = getattr(self, "_axis_limit_callback_ids", None)
        if not callback_ids:
            return

        for axis_obj, x_cid, y_cid in callback_ids:
            callbacks = getattr(axis_obj, "callbacks", None)
            disconnect = getattr(callbacks, "disconnect", None)
            if disconnect is None:
                continue
            for cid in (x_cid, y_cid):
                try:
                    disconnect(cid)
                except Exception:
                    continue

    def _set_limit_field_value(self, field: _FloatLimitField, value: float) -> None:
        """Set a limit field value without signal churn."""
        field.blockSignals(True)
        field.setValue(float(value))
        field.blockSignals(False)

    def _sync_limits_from_axes(self, source_axis: object | None = None) -> bool:
        """Update x/y limit fields from current Matplotlib axis limits."""
        if not self._has_mpl or self._syncing_limits_from_axes:
            return False

        self._syncing_limits_from_axes = True
        try:
            if self._subplot_axes_by_polarization:
                subplot_axes = list(self._subplot_axes_by_polarization.values())
                if source_axis is not None and not any(
                    source_axis is axis for axis in subplot_axes
                ):
                    return False

                axis_obj = None
                if source_axis in self._subplot_axes_by_polarization.values():
                    axis_obj = source_axis
                else:
                    ordered = self._all_mode_axes_order()
                    if ordered:
                        axis_obj = self._subplot_axes_by_polarization.get(ordered[0])
                    if axis_obj is None:
                        axis_obj = next(iter(self._subplot_axes_by_polarization.values()), None)

                if axis_obj is None or not hasattr(axis_obj, "get_xlim"):
                    return False

                x0, x1 = axis_obj.get_xlim()
                self._set_limit_field_value(
                    self._x_min,
                    self._convert_frequency_axis_limit_to_control_value(x0),
                )
                self._set_limit_field_value(
                    self._x_max,
                    self._convert_frequency_axis_limit_to_control_value(x1),
                )

                for axis_key, subplot_axis in self._subplot_axes_by_polarization.items():
                    if not hasattr(subplot_axis, "get_ylim"):
                        continue
                    y0, y1 = subplot_axis.get_ylim()
                    lo, hi = (float(y0), float(y1)) if y0 <= y1 else (float(y1), float(y0))
                    self._y_limits_by_polarization[axis_key] = (lo, hi)

                current_axis = self._current_polarization_axis
                if current_axis in self._subplot_axes_by_polarization:
                    current_obj = self._subplot_axes_by_polarization[current_axis]
                    if hasattr(current_obj, "get_ylim"):
                        y0, y1 = current_obj.get_ylim()
                        self._set_limit_field_value(self._y_min, y0)
                        self._set_limit_field_value(self._y_max, y1)
                else:
                    self._sync_y_controls_with_visible_axis()
                return True

            if not hasattr(self._ax, "get_xlim") or not hasattr(self._ax, "get_ylim"):
                return False
            if source_axis is not None and source_axis is not self._ax:
                return False

            x0, x1 = self._ax.get_xlim()
            y0, y1 = self._ax.get_ylim()
            self._set_limit_field_value(
                self._x_min,
                self._convert_frequency_axis_limit_to_control_value(x0),
            )
            self._set_limit_field_value(
                self._x_max,
                self._convert_frequency_axis_limit_to_control_value(x1),
            )
            self._set_limit_field_value(self._y_min, y0)
            self._set_limit_field_value(self._y_max, y1)
            self._cache_current_y_limits_for_axis()
            self._emit_view_limits_changed()
            return True
        finally:
            self._syncing_limits_from_axes = False

    def _is_reconstruction_view(self) -> bool:
        """True when the panel shows a MaxEnt reconstruction layout.

        Reconstruction subplots are keyed ``recon:<run>:<idx>`` /
        ``recon:combined:<run>``. They are drawn with plain ``ax.plot`` (never
        decimated) and ``_redraw_current_view`` cannot rebuild them — it would
        fall through to a plain dataset plot — so they must be excluded from the
        viewport-refresh path.
        """
        return any(
            isinstance(key, str) and key.startswith("recon:")
            for key in self._subplot_axes_by_polarization
        )

    def _schedule_viewport_refresh(self) -> None:
        """Coalesce viewport-triggered density refreshes onto the next event loop turn."""
        if (
            not self._has_mpl
            or not self._current_datasets
            or self._viewport_refresh_in_progress
            or self._viewport_refresh_pending
            # MaxEnt reconstructions are not decimated and cannot be rebuilt by
            # _redraw_current_view; refreshing one would replace it with a plain
            # dataset plot. (Pre-existing: limit-field edits also route here.)
            or self._is_reconstruction_view()
        ):
            return
        self._viewport_refresh_pending = True
        QTimer.singleShot(0, self._apply_viewport_refresh)

    def _apply_viewport_refresh(self) -> None:
        """Re-render the current view using the latest visible-axis limits."""
        if not self._viewport_refresh_pending:
            return
        self._viewport_refresh_pending = False
        if not self._current_datasets or self._viewport_refresh_in_progress:
            return

        self._viewport_refresh_in_progress = True
        try:
            self._redraw_current_view()
        finally:
            self._viewport_refresh_in_progress = False

    def _on_axis_limits_changed(self, axis_obj) -> None:
        """Sync limit controls when Matplotlib axes change via pan/zoom."""
        synced = self._sync_limits_from_axes(source_axis=axis_obj)
        if synced and not self._viewport_refresh_in_progress:
            self._schedule_viewport_refresh()

    def _on_canvas_draw_event(self, _event) -> None:
        """Keep nav buttons and limit controls aligned after Matplotlib redraws."""
        self._sync_navigation_buttons()
        if self._current_navigation_mode() != "none":
            self._sync_limits_from_axes()

    def _on_limit_fields_edited(self) -> None:
        """Apply edited limits and refresh display density for the new viewport."""
        self._apply_limits(schedule_viewport_refresh=True)

    def _active_label_field_key(self) -> str:
        """Return the label field that should currently be shown/used.

        The per-group preference when a Data Group is active, otherwise the
        default. This is the stored *intent*, which the combo targets on every
        rebuild — so a saved custom column that is not yet offered (e.g. just
        after project load) is selected the moment the host pushes it in.
        """
        if self._active_label_group_id is None:
            return self._default_label_field
        return self._label_field_by_group.get(
            str(self._active_label_group_id), self._default_label_field
        )

    def _rebuild_label_field_combo(self) -> None:
        """Populate the label-field combo with the built-ins plus custom columns.

        Targets the active label-field intent (not the transient combo selection)
        so adding/removing/renaming custom columns — or restoring a project whose
        saved label is a not-yet-offered custom column — lands on the right entry.
        """
        combo = self._label_field_combo
        target = self._active_label_field_key()
        blocker = QSignalBlocker(combo)
        combo.clear()
        for display, key in _LABEL_FIELDS:
            combo.addItem(display, userData=key)
        for display, key in self._custom_label_fields:
            combo.addItem(display, userData=key)
        idx = combo.findData(target)
        if idx < 0:
            idx = combo.findData("run")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        del blocker

    def set_custom_label_fields(self, fields: list[tuple[str, str]]) -> None:
        """Set the data-browser custom columns offered as legend-label options.

        ``fields`` is a list of ``(display_label, "custom:<id>")`` pairs. The
        combo is rebuilt (selection preserved) and, if the active label field is
        a custom column whose label changed, the plot is redrawn so the legend
        tracks the rename.
        """
        normalized = [(str(label), str(key)) for label, key in fields]
        if normalized == self._custom_label_fields:
            return
        active_is_custom = str(self._label_field_combo.currentData() or "").startswith("custom:")
        self._custom_label_fields = normalized
        self._rebuild_label_field_combo()
        if active_is_custom and self._has_mpl and self._current_datasets:
            self._redraw_current_view()

    def _is_valid_label_field(self, field: object) -> bool:
        """Whether ``field`` is a selectable label key (built-in or custom column).

        Any ``custom:`` key is accepted even before its column is pushed in, so a
        saved selection survives project load regardless of restore ordering; an
        unknown custom column simply falls back to the run label until resolved.
        """
        if not isinstance(field, str):
            return False
        if field.startswith("custom:"):
            return True
        return field in {key for _, key in _LABEL_FIELDS}

    def _custom_field_value(self, dataset: MuonDataset, field: str) -> str | None:
        """Resolve a ``custom:<id>`` label key to a dataset's stored text."""
        fields = dataset.metadata.get("custom_fields")
        if isinstance(fields, dict):
            value = fields.get(field)
            if value is not None and str(value) != "":
                return str(value)
        return None

    def _dataset_label_for(self, dataset: MuonDataset) -> str:
        """Return the legend label for *dataset* using the selected label field."""
        field = self._label_field_combo.currentData()
        if field == "run":
            return str(dataset.run_label)
        if isinstance(field, str) and field.startswith("custom:"):
            value = self._custom_field_value(dataset, field)
            return value if value is not None else str(dataset.run_label)
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
        self._redraw_current_view()

    def _normalize_time_view_mode(self, mode: object) -> str:
        """Normalize a stored time-view token to a supported internal key.

        The integral-scan representation plots the F-B asymmetry (with the
        fit-range acting as the integration window), so it maps onto the
        ``fb_asymmetry`` plot mode.
        """
        token = str(mode or "").strip().lower().replace(" ", "_")
        if token in {"groups", "group", "individual_groups", "grouped", "grouped_counts"}:
            return "groups"
        if token in {"raw_counts", "raw", "raw_count"}:
            return "raw_counts"
        return "fb_asymmetry"

    def _time_view_display_text(self, mode: str) -> str:
        """Return user-facing label for a time-view mode."""
        normalized = self._normalize_time_view_mode(mode)
        for label, key in _TIME_VIEW_FIELDS:
            if key == normalized:
                return label
        return "FB Asymmetry"

    def _on_time_view_mode_changed(self, _index: int) -> None:
        """Propagate explicit time-view changes from the main plot toolbar."""
        if not hasattr(self, "_time_view_combo"):
            return
        mode = self._normalize_time_view_mode(self._time_view_combo.currentData())
        if mode == self._current_time_view_mode:
            return
        self._current_time_view_mode = mode
        self._refresh_log_counts_visibility()
        self.time_view_changed.emit(mode)

    def current_time_view_mode(self) -> str:
        """Return current main time-domain plot mode."""
        return self._current_time_view_mode

    def set_time_view_modes(
        self,
        modes: list[str],
        current_mode: str | None = None,
    ) -> None:
        """Update the explicit time-domain view selector."""
        if not hasattr(self, "_time_view_combo"):
            return

        cleaned: list[str] = []
        for mode in modes:
            normalized = self._normalize_time_view_mode(mode)
            if normalized not in cleaned:
                cleaned.append(normalized)
        if not cleaned:
            cleaned = ["fb_asymmetry"]

        selected = self._normalize_time_view_mode(
            self._current_time_view_mode if current_mode is None else current_mode
        )
        if selected not in cleaned:
            selected = cleaned[0]

        self._available_time_view_modes = list(cleaned)
        self._current_time_view_mode = selected

        self._time_view_combo.blockSignals(True)
        self._time_view_combo.clear()
        for mode in cleaned:
            self._time_view_combo.addItem(self._time_view_display_text(mode), mode)
        idx = self._time_view_combo.findData(selected)
        if idx < 0:
            idx = 0
        self._time_view_combo.setCurrentIndex(idx)
        self._time_view_combo.blockSignals(False)
        self._time_view_combo.setEnabled(len(cleaned) > 1)
        self._refresh_log_counts_visibility()

    def set_current_time_view_mode(self, mode: str, *, emit_signal: bool = False) -> None:
        """Select the active time-domain view mode."""
        if not hasattr(self, "_time_view_combo"):
            return

        normalized = self._normalize_time_view_mode(mode)
        if normalized not in self._available_time_view_modes:
            normalized = self._available_time_view_modes[0]
        idx = self._time_view_combo.findData(normalized)
        if idx < 0:
            idx = 0
        previous = self._time_view_combo.blockSignals(not emit_signal)
        self._time_view_combo.setCurrentIndex(idx)
        self._time_view_combo.blockSignals(previous)
        self._current_time_view_mode = normalized
        self._refresh_log_counts_visibility()

    def _refresh_log_counts_visibility(self) -> None:
        """Show the log-scale toggle only on the raw-counts view."""
        checkbox = getattr(self, "_log_counts_checkbox", None)
        if checkbox is None:
            return
        checkbox.setVisible(self._current_time_view_mode == "raw_counts")

    def _log_counts_active(self) -> bool:
        """True when the log-count diagnostic should apply to the render."""
        return bool(
            getattr(self, "_log_counts_enabled", False)
            and self._current_time_view_mode == "raw_counts"
        )

    def _on_log_counts_toggled(self, checked: bool) -> None:
        """Persist and apply the log-count diagnostic scale."""
        self._log_counts_enabled = bool(checked)
        self._redraw_current_view()

    def _apply_log_counts_scale(self) -> None:
        """Switch the raw-count subplots to a log y-axis (diagnostic view).

        Applied after the linear limits so the axis is converted from already
        positive count limits — matplotlib then autoscales to the positive data
        and silently drops the non-positive (e.g. empty) bins.
        """
        if not self._has_mpl or not self._log_counts_active():
            return
        axes = list(getattr(self, "_subplot_axes_by_polarization", {}).values())
        if not axes:
            return
        for ax in axes:
            ax.set_yscale("log")
            ax.relim()
            ax.autoscale(axis="y")
        self._canvas.draw_idle()

    def is_overlay_enabled(self) -> bool:
        """Return whether multi-selection overlays are currently enabled."""
        if not self._has_mpl:
            return True
        return bool(getattr(self, "_overlay_checkbox", None).isChecked())

    def set_overlay_enabled(self, enabled: bool, *, emit_signal: bool = False) -> None:
        """Set overlay mode from state restore or external UI events."""
        if not self._has_mpl or not hasattr(self, "_overlay_checkbox"):
            return

        checkbox = self._overlay_checkbox
        previous = checkbox.blockSignals(not emit_signal)
        checkbox.setChecked(bool(enabled))
        checkbox.blockSignals(previous)

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
            target_field = self._label_field_by_group.get(
                normalized_group_id, self._default_label_field
            )

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
        return [{"x": ann["x"], "y": ann["y"], "text": ann["text"]} for ann in annotations]

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

        if self._is_frequency_plot_panel() or self._is_frequency_domain_dataset(dataset):
            return dataset

        bunch_factor = self._bunch_factor.value()
        if bunch_factor <= 1:
            return dataset

        time, asymmetry, error = rebin(
            dataset.time,
            dataset.asymmetry,
            dataset.error,
            bunch_factor,
        )
        if time.size == 0 or asymmetry.size == 0 or error.size == 0:
            return dataset
        return MuonDataset(
            time=time,
            asymmetry=asymmetry,
            error=error,
            metadata=dict(dataset.metadata),
            run=dataset.run,
        )

    def _raw_fit_seed_range(self, datasets: list[MuonDataset | None]) -> tuple[float, float] | None:
        """Default fit-range seed from the *untransformed* analysis time axes.

        The RRF display trims the filter-edge region off the drawn arrays;
        seeding the fit range from those would silently exclude raw early-time
        bins from fits that always consume raw data.
        """
        lows: list[float] = []
        highs: list[float] = []
        for dataset in datasets:
            base = self.get_analysis_dataset(dataset)
            if base is None:
                continue
            tt = self._convert_frequency_axis_for_display(np.asarray(base.time, dtype=float))
            tt = tt[np.isfinite(tt)]
            if tt.size:
                lows.append(float(tt.min()))
                highs.append(float(tt.max()))
        if not lows:
            return None
        return min(lows), max(highs)

    def _is_frequency_domain_dataset(self, dataset: MuonDataset | None) -> bool:
        """Return True when *dataset* carries frequency-domain plot metadata."""
        if dataset is None:
            return False
        return str(dataset.metadata.get("plot_domain", "")).strip().lower() == "frequency"

    def _axis_labels_for_dataset(
        self,
        dataset: MuonDataset | None,
        axis_key: str | None,
    ) -> tuple[str, str]:
        """Return axis labels, preferring explicit metadata for special plots."""
        if self._is_frequency_plot_panel():
            y_label = r"$|F|$ (arb.)"
            if dataset is not None and isinstance(dataset.metadata, dict):
                raw_y_label = dataset.metadata.get("y_label")
                if isinstance(raw_y_label, str) and raw_y_label.strip():
                    y_label = raw_y_label
            return self._display_x_label(), y_label
        if dataset is not None and isinstance(dataset.metadata, dict):
            x_label = dataset.metadata.get("x_label")
            y_label = dataset.metadata.get("y_label")
            if (
                isinstance(x_label, str)
                and x_label.strip()
                and isinstance(y_label, str)
                and y_label.strip()
            ):
                return x_label, y_label
        return "Time (μs)", self._polarization_ylabel(axis_key)

    def _has_plottable_samples(self, dataset: MuonDataset | None) -> bool:
        """Return True when *dataset* contains at least one aligned sample."""
        if dataset is None:
            return False
        return (
            np.asarray(dataset.time).size > 0
            and np.asarray(dataset.asymmetry).size > 0
            and np.asarray(dataset.error).size > 0
        )

    def _visible_plot_indices(self, time: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return indices inside the current viewport before any density reduction."""
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if indices.size <= 0:
            return indices

        visible_indices = indices
        if self._limits_initialized:
            x_lo = float(self._x_min.value())
            x_hi = float(self._x_max.value())
            if self._is_frequency_plot_panel():
                x_lo = self._convert_frequency_control_value_to_axis_limit(x_lo)
                x_hi = self._convert_frequency_control_value_to_axis_limit(x_hi)
            lo, hi = (x_lo, x_hi) if x_lo <= x_hi else (x_hi, x_lo)
            visible_mask = (
                np.asarray(mask, dtype=bool)
                & np.isfinite(time)
                & (np.asarray(time, dtype=float) >= lo)
                & (np.asarray(time, dtype=float) <= hi)
            )
            candidate = np.flatnonzero(visible_mask)
            if candidate.size > 0:
                visible_indices = candidate
        return visible_indices

    def _decimated_plot_indices(
        self,
        time: np.ndarray,
        mask: np.ndarray,
        values: np.ndarray | None = None,
        *,
        visible_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return bounded sample indices for display-only errorbar rendering.

        Time-domain scatter uses a uniform stride — an unbiased visual sample
        of noisy data (a min-max envelope would exaggerate the noise).
        Frequency-domain spectra use min-max bucketing on *values* instead:
        a stride can drop a narrow spectral peak entirely, and the peaks are
        the physics. Callers that already computed the viewport's
        ``visible_indices`` pass them in to avoid a second full-array scan.
        """
        if visible_indices is None:
            visible_indices = self._visible_plot_indices(time, mask)
        if visible_indices.size <= 0:
            return visible_indices
        if not self.decimation_enabled():
            return visible_indices

        max_points = max(1, int(getattr(self, "_max_render_points_per_trace", 4000)))
        if visible_indices.size <= max_points:
            return visible_indices

        if values is not None and self._is_frequency_plot_panel():
            return self._minmax_bucket_indices(visible_indices, np.asarray(values), max_points)

        stride = max(1, int(np.ceil(visible_indices.size / float(max_points))))
        sampled = visible_indices[::stride]
        if sampled.size == 0:
            return visible_indices[:1]
        if sampled[-1] != visible_indices[-1]:
            sampled = np.append(sampled, visible_indices[-1])
        return sampled

    @staticmethod
    def _minmax_bucket_indices(
        visible_indices: np.ndarray,
        values: np.ndarray,
        max_points: int,
    ) -> np.ndarray:
        """Keep the min and max sample of each display bucket (extrema survive).

        Vectorised: the visible values are NaN-padded into a
        ``(n_buckets, bucket_len)`` grid and reduced per row — a Python loop
        over ~2000 buckets would cost tens of ms per trace per pan tick.
        """
        n = int(visible_indices.size)
        n_buckets = max(1, max_points // 2)
        bucket_len = -(-n // n_buckets)  # ceil; >= 2 because n > max_points
        pad = n_buckets * bucket_len - n
        y = np.asarray(values, dtype=float)[visible_indices]
        padded = np.concatenate([y, np.full(pad, np.nan)]) if pad else y
        grid = padded.reshape(n_buckets, bucket_len)
        nan_mask = np.isnan(grid)
        # Real values always beat the ±inf sentinels, so argmin/argmax can
        # never select padding except in all-NaN rows, which are dropped.
        mins = np.argmin(np.where(nan_mask, np.inf, grid), axis=1)
        maxs = np.argmax(np.where(nan_mask, -np.inf, grid), axis=1)
        valid = ~nan_mask.all(axis=1)
        offsets = np.arange(n_buckets, dtype=np.int64) * bucket_len
        keep = np.concatenate(
            [
                np.asarray([0, n - 1], dtype=np.int64),
                (offsets + mins)[valid],
                (offsets + maxs)[valid],
            ]
        )
        return visible_indices[np.unique(keep)]

    def _plot_errorbar_masked(
        self,
        ax,
        time: np.ndarray,
        asymmetry: np.ndarray,
        error: np.ndarray,
        mask: np.ndarray,
        **kwargs,
    ) -> int:
        """Plot a masked errorbar series using bounded display density."""
        visible_indices = self._visible_plot_indices(time, mask)
        indices = self._decimated_plot_indices(
            time, mask, values=asymmetry, visible_indices=visible_indices
        )
        if indices.size <= 0:
            return 0
        if self.decimation_enabled() and indices.size < visible_indices.size:
            self._decimation_applied_for_current_view = True
        # Per-view totals feed the corner chip ("4.0k of 1.2M pts").
        self._decimation_points_shown += int(indices.size)
        self._decimation_points_total += int(visible_indices.size)

        ax.errorbar(
            time[indices],
            asymmetry[indices],
            yerr=error[indices],
            **kwargs,
        )
        return int(indices.size)

    def _set_frequency_reference_from_dataset(self, dataset: MuonDataset | None) -> None:
        """Update the reference frequency used by relative frequency displays."""
        if not self._is_frequency_plot_panel():
            return
        self._frequency_reference_mhz = self._frequency_reference_for_dataset(dataset)
        self._update_correlation_axis_state(dataset)
        if hasattr(self, "_frequency_reference_spin"):
            gauss = (
                self._frequency_reference_mhz / self._mhz_per_gauss()
                if self._frequency_reference_mhz is not None
                else 0.0
            )
            with QSignalBlocker(self._frequency_reference_spin):
                self._frequency_reference_spin.setValue(gauss)

    def _update_correlation_axis_state(self, dataset: MuonDataset | None) -> None:
        """Lock the field-unit selector when *dataset* is a correlation spectrum.

        A muoniated-radical correlation spectrum's x-axis is the muon hyperfine
        coupling A_µ (MHz), not γ_µ·B, so the MHz/G/T field selector and the
        applied-field reference are meaningless: they are forced to a plain MHz
        absolute axis and disabled, and the axis keeps its own label from the
        dataset metadata. On entry the user's unit + relative-reference choice is
        stashed and on exit it is restored, so toggling the correlation mode does
        not silently discard (or persist) the wrong field-unit view.

        (Note: the frequency panel shows a single averaged spectrum at a time, so
        keying off the active dataset is sufficient; mixed correlation/FFT
        overlays are not produced by the current pipeline.)
        """
        is_correlation = bool(
            dataset is not None
            and isinstance(dataset.metadata, dict)
            and dataset.metadata.get("correlation_axis") is True
        )
        if is_correlation:
            raw_label = dataset.metadata.get("x_label")
            if isinstance(raw_label, str) and raw_label.strip():
                self._frequency_correlation_x_label = raw_label

        was_correlation = self._frequency_axis_is_correlation
        if is_correlation == was_correlation:
            return  # No transition; nothing to stash, restore, or re-sync.

        if is_correlation:
            # Entering: stash the field-unit view and force a plain MHz axis.
            self._frequency_axis_unit_before_correlation = self._current_frequency_x_unit
            self._frequency_axis_relative_before_correlation = (
                self._frequency_axis_relative_to_reference
            )
            self._frequency_axis_is_correlation = True
            self._current_frequency_x_unit = "frequency_mhz"
            self._frequency_axis_relative_to_reference = False
        else:
            # Leaving: restore the stashed field-unit view.
            self._frequency_axis_is_correlation = False
            self._current_frequency_x_unit = self._frequency_axis_unit_before_correlation
            self._frequency_axis_relative_to_reference = (
                self._frequency_axis_relative_before_correlation
            )
        self._sync_frequency_axis_controls()

    def _sync_frequency_axis_controls(self) -> None:
        """Push the current unit / relative state onto the frequency-view widgets.

        Keeps the unit combo, the relative-reference checkbox, and the reference
        spin coherent with the backing state, and disables them while a
        correlation (coupling) axis is shown.
        """
        is_correlation = self._frequency_axis_is_correlation
        if hasattr(self, "_frequency_x_unit_combo"):
            with QSignalBlocker(self._frequency_x_unit_combo):
                idx = self._frequency_x_unit_combo.findData(self._current_frequency_x_unit)
                if idx >= 0:
                    self._frequency_x_unit_combo.setCurrentIndex(idx)
            self._frequency_x_unit_combo.setEnabled(not is_correlation)
        if hasattr(self, "_frequency_axis_relative_check"):
            with QSignalBlocker(self._frequency_axis_relative_check):
                self._frequency_axis_relative_check.setChecked(
                    self._frequency_axis_relative_to_reference
                )
            self._frequency_axis_relative_check.setEnabled(not is_correlation)
        if hasattr(self, "_frequency_reference_spin"):
            self._frequency_reference_spin.setEnabled(
                not is_correlation and self._frequency_axis_relative_to_reference
            )

    def update_frequency_reference(self, dataset: MuonDataset | None) -> None:
        """Public entry-point: sync the reference field from *dataset* metadata.

        Call this whenever the active dataset changes so the Reference spinbox
        shows the correct value even before an FFT has been computed.
        """
        self._set_frequency_reference_from_dataset(dataset)

    def _render_empty_plot_state(self, *, alpha_text: str | None = None) -> None:
        """Render an empty but valid plot state when no plottable data is available."""
        self._last_plot_time = None
        self._last_plot_asymmetry = None
        self._last_plot_error = None
        self._last_low_count_mask = None
        self._reset_decimation_view_state()
        self._fit_x_min = None
        self._fit_x_max = None
        if self._is_frequency_plot_panel():
            self._frequency_reference_mhz = None

        self._ax.clear()
        style_axes(self._ax)
        self._apply_axis_labels(
            *self._axis_labels_for_dataset(None, self._current_polarization_axis)
        )
        self._set_alpha_label(alpha_text)
        self._draw_annotations()
        self._draw_fit_range_artists()
        self._update_export_enabled()
        self._canvas.draw_idle()

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

    def get_view_limits(self) -> tuple[float, float, float, float]:
        """Return the currently displayed x/y limits from the toolbar fields."""
        if not self._has_mpl:
            return 0.0, 10.0, -30.0, 30.0
        return (
            float(self._x_min.value()),
            float(self._x_max.value()),
            float(self._y_min.value()),
            float(self._y_max.value()),
        )

    def set_view_limits(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Apply x/y limits through the existing toolbar-backed controls."""
        if not self._has_mpl:
            return
        self._set_view_limits_fields(x_min, x_max, y_min, y_max)
        self._apply_limits(schedule_viewport_refresh=True)

    def get_bunch_factor(self) -> int:
        """Return the currently configured plot-panel bunch factor."""
        if not self._has_mpl:
            return 1
        return int(self._bunch_factor.value())

    def decimation_enabled(self) -> bool:
        """Return whether display-only plot decimation is enabled."""
        return bool(getattr(self, "_decimation_enabled", True))

    def set_decimation_enabled(self, enabled: bool, *, redraw: bool = True) -> None:
        """Enable or disable display-only plot decimation."""
        self._decimation_enabled = bool(enabled)
        if redraw and getattr(self, "_current_datasets", None):
            self._redraw_current_view()

    def set_bunch_factor(self, value: int, *, emit_signal: bool = True) -> None:
        """Set the plot-panel bunch factor and optionally emit the normal signal."""
        if not self._has_mpl:
            return
        bunch_factor = max(1, int(value))
        if emit_signal:
            self._bunch_factor.setValue(bunch_factor)
            return

        previous = self._bunch_factor.blockSignals(True)
        self._bunch_factor.setValue(bunch_factor)
        self._bunch_factor.blockSignals(previous)
        self._redraw_current_view()

    def _redraw_current_view(self) -> None:
        """Redraw using the active single- or multi-dataset context."""
        if self._grouped_time_subplot_datasets:
            self.plot_grouped_time_domain_subplots(self._grouped_time_subplot_datasets)
            return
        if self._current_polarization_axis == "ALL" and self._vector_subplot_datasets:
            self.plot_vector_subplots(self._vector_subplot_datasets)
            return
        # Preserve explicit multi-dataset views (for example, after
        # plot_datasets()) even if the overlay checkbox is currently off.
        if len(self._current_datasets) > 1:
            self.plot_datasets(self._current_datasets)
        elif self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)
        elif self._current_datasets:
            self.plot_dataset(self._current_datasets[-1])

    def _emit_view_limits_changed(self) -> None:
        """Broadcast the current view limits to external UI owners."""
        if not self._has_mpl:
            return
        self.view_limits_changed.emit(*self.get_view_limits())

    def _set_alpha_label(self, text: str | None) -> None:
        """Update the header metadata label with alpha / unit info."""
        if hasattr(self, "_header_meta_label"):
            self._header_meta_label.setText(text or "")

    def _axis_canonical_key(self, axis_text: str | None) -> str | None:
        """Normalize display/projection axis text to its identity key.

        The canonical EMU vector aliases (``px``/``x`` → ``P_x`` …) and the
        ``ALL`` sentinel are recognised; any other non-empty label passes
        through unchanged so transverse-field projection labels survive.
        Returns ``None`` only for an empty/absent axis.
        """
        if axis_text is None:
            return None
        raw = str(axis_text).strip()
        if not raw:
            return None
        token = raw.lower().replace(" ", "")
        token = token.replace("ₓ", "x").replace("ᵧ", "y").replace("ᶻ", "z")
        token = token.replace("_", "")
        if token in {"all", "pall"}:
            return "ALL"
        if token in {"px", "x"}:
            return "P_x"
        if token in {"py", "y"}:
            return "P_y"
        if token in {"pz", "z"}:
            return "P_z"
        return raw

    def get_current_polarization_axis(self) -> str | None:
        """Return current polarization selector key (``P_*`` or ``ALL``)."""
        return self._current_polarization_axis

    def _axis_key_for_dataset(
        self,
        dataset: MuonDataset | None,
        *,
        axis_override: str | None = None,
    ) -> str | None:
        """Return canonical polarization axis for *dataset* when available."""
        if axis_override is not None:
            return self._axis_canonical_key(axis_override)
        if dataset is None:
            return None

        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if isinstance(grouping, dict):
            axis = self._axis_canonical_key(grouping.get("vector_axis"))
            if axis is not None and axis != "ALL":
                return axis

        grouping_meta = dataset.metadata.get("grouping")
        if isinstance(grouping_meta, dict):
            axis = self._axis_canonical_key(grouping_meta.get("vector_axis"))
            if axis is not None and axis != "ALL":
                return axis

        return None

    @staticmethod
    def _encode_fit_storage_key(run_number: int, axis_key: str | None) -> str:
        """Encode ``(run_number, axis_key)`` to a serialisable key string."""
        return f"{int(run_number)}|{axis_key or ''}"

    def _decode_fit_storage_key(self, value: object) -> tuple[int, str | None] | None:
        """Decode serialised fit key values."""
        if not isinstance(value, str) or "|" not in value:
            return None
        run_token, axis_token = value.split("|", 1)
        try:
            run_number = int(run_token)
        except (TypeError, ValueError):
            return None
        axis_key = self._axis_canonical_key(axis_token) if axis_token else None
        if axis_key == "ALL":
            axis_key = None
        return run_number, axis_key

    def _fit_storage_key_for_dataset(
        self,
        dataset: MuonDataset | None,
        *,
        axis_override: str | None = None,
    ) -> tuple[int, str | None] | None:
        """Return axis-aware fit storage key for *dataset*."""
        if dataset is None:
            return None
        try:
            run_number = int(dataset.run_number)
        except (TypeError, ValueError):
            return None
        axis_key = self._axis_key_for_dataset(dataset, axis_override=axis_override)
        return run_number, axis_key

    @staticmethod
    def _is_raw_counts_dataset(dataset: MuonDataset | None) -> bool:
        """True for grouped datasets built without lifetime correction.

        Stored grouped/count fit overlays are on the lifetime-corrected scale,
        so they must not be drawn against raw-count curves (they diverge by
        e^(t/τ), ~38× at 8 μs).
        """
        return (
            dataset is not None
            and dataset.metadata.get("grouped_time_domain_lifetime_corrected") is False
        )

    def _fit_curve_for_dataset(
        self,
        dataset: MuonDataset | None,
        *,
        axis_override: str | None = None,
    ) -> tuple | None:
        """Return best-matching fit curve payload for *dataset*."""
        if self._is_raw_counts_dataset(dataset):
            return None
        storage_key = self._fit_storage_key_for_dataset(dataset, axis_override=axis_override)
        if storage_key is not None:
            fit_data = self._fit_curves_by_key.get(storage_key)
            if fit_data is not None:
                return fit_data

            run_number, axis_key = storage_key
            if axis_key is not None:
                has_axis_specific_fit = any(
                    key_run == run_number and key_axis is not None
                    for key_run, key_axis in self._fit_curves_by_key
                )
                if has_axis_specific_fit:
                    return None

                fit_data = self._fit_curves_by_key.get((run_number, None))
                if fit_data is not None:
                    return fit_data

            fit_data = self._fit_curves.get(run_number)
            if fit_data is not None:
                return fit_data

            if self._fit_curve is not None and self._fit_curve_run_number == run_number:
                return self._fit_curve

        return None

    def _fit_components_for_dataset(
        self,
        dataset: MuonDataset | None,
        *,
        axis_override: str | None = None,
    ) -> list[tuple[str, object]]:
        """Return best-matching additive component curves for *dataset*."""
        if self._is_raw_counts_dataset(dataset):
            return []
        storage_key = self._fit_storage_key_for_dataset(dataset, axis_override=axis_override)
        if storage_key is not None:
            components = self._fit_components_by_key.get(storage_key)
            if components:
                return list(components)

            run_number, axis_key = storage_key
            if axis_key is not None:
                has_axis_specific_components = any(
                    key_run == run_number and key_axis is not None
                    for key_run, key_axis in self._fit_components_by_key
                )
                if has_axis_specific_components:
                    return []

                components = self._fit_components_by_key.get((run_number, None))
                if components:
                    return list(components)

            components = self._fit_components_by_run.get(run_number)
            if components:
                return list(components)

            if self._fit_components and self._fit_curve_run_number == run_number:
                return list(self._fit_components)

        return []

    def _fit_metadata_for_dataset(
        self,
        dataset: MuonDataset | None,
        *,
        axis_override: str | None = None,
    ) -> dict:
        """Return best-matching fit metadata for *dataset*."""
        storage_key = self._fit_storage_key_for_dataset(dataset, axis_override=axis_override)
        if storage_key is None:
            return {}

        meta = self._fit_metadata_by_key.get(storage_key)
        if isinstance(meta, dict):
            return meta

        run_number, axis_key = storage_key
        if axis_key is not None:
            has_axis_specific_meta = any(
                key_run == run_number and key_axis is not None
                for key_run, key_axis in self._fit_metadata_by_key
            )
            if has_axis_specific_meta:
                return {}

            meta = self._fit_metadata_by_key.get((run_number, None))
            if isinstance(meta, dict):
                return meta

        meta = self._fit_metadata.get(run_number)
        return meta if isinstance(meta, dict) else {}

    def _active_y_axis(self) -> str | None:
        """The axis whose y-limits the manual controls reflect and drive.

        In the stacked multi-subplot view that is the selected (fit-target)
        subplot; otherwise it is the single visible polarization axis.
        """
        if self._subplot_axes_by_polarization:
            return self.fit_target_projection()
        return self._current_polarization_axis

    def _cache_current_y_limits_for_axis(self) -> None:
        """Store current y-limits under the active (focused) axis, if any."""
        axis = self._active_y_axis()
        if axis is None or axis == "ALL":
            return
        y0 = float(self._y_min.value())
        y1 = float(self._y_max.value())
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        self._y_limits_by_polarization[axis] = (lo, hi)

    def _restore_y_limits_for_axis(self, axis: str | None) -> None:
        """Restore the y-limit controls to *axis*'s cached (or live) limits."""
        if axis is None or axis == "ALL":
            return
        limits = self._y_limits_by_polarization.get(axis)
        if limits is None:
            # No cached value yet — reflect the subplot's actual current limits.
            ax = self._subplot_axes_by_polarization.get(axis)
            if ax is not None and hasattr(ax, "get_ylim"):
                try:
                    limits = ax.get_ylim()
                except Exception:
                    limits = None
        if limits is None:
            return
        self._y_min.setValue(float(limits[0]))
        self._y_max.setValue(float(limits[1]))

    def _axis_for_selection(self, labels: list[str]) -> str | None:
        """Map a chip selection onto the internal polarization-axis token.

        One projection selected → that label (single-subplot mode); more than
        one → the ``"ALL"`` sentinel (stacked-subplot mode); none → ``None``.
        """
        if len(labels) > 1:
            return "ALL"
        return labels[0] if labels else None

    def _on_projection_selection_changed(self, labels: list[str]) -> None:
        """Handle a projection chip-selection change from the header chip bar."""
        self._selected_projection_labels = list(labels)
        axis = self._axis_for_selection(list(labels))
        if axis is None:
            return
        previous_axis = self._current_polarization_axis
        if previous_axis != axis:
            self._cache_current_y_limits_for_axis()
            self._current_polarization_axis = str(axis)
            self._restore_y_limits_for_axis(self._current_polarization_axis)
            self._sync_y_controls_with_visible_axis()
            self._update_y_limit_controls_for_axis(self._current_polarization_axis)
            self._apply_limits()
        self.polarization_axis_changed.emit(str(axis))

    def _update_y_limit_controls_for_axis(self, axis: str | None) -> None:
        """Keep the Y controls enabled, driving the focused (selected) subplot.

        In the stacked view, manual Y now applies to the selected (fit-target)
        subplot rather than being disabled; auto Y still rescales every
        projection subplot.
        """
        in_subplots = bool(self._subplot_axes_by_polarization)
        manual_tooltip = (
            "Y limits apply to the selected subplot — click another subplot to switch."
            if in_subplots
            else ""
        )
        auto_tooltip = "Auto Y rescales every projection subplot." if in_subplots else ""
        self._y_min.setEnabled(True)
        self._y_max.setEnabled(True)
        self._y_min.setToolTip(manual_tooltip)
        self._y_max.setToolTip(manual_tooltip)
        if hasattr(self, "_auto_y_btn"):
            self._auto_y_btn.setEnabled(True)
            self._auto_y_btn.setToolTip(auto_tooltip)

    def _all_mode_axes_order(self) -> list[str]:
        """Return the axis order currently visible in ALL mode.

        Prefers the declared projection order, then the canonical vector order,
        and finally the subplot dict's own order — so transverse-field
        projections order by their preset declaration, not a canonical filter.
        """
        if not self._subplot_axes_by_polarization:
            return []
        spec_order = [str(p["label"]) for p in self._projection_specs]
        ordered = [a for a in spec_order if a in self._subplot_axes_by_polarization]
        if ordered:
            return ordered
        ordered = [
            axis for axis in ("P_x", "P_y", "P_z") if axis in self._subplot_axes_by_polarization
        ]
        if ordered:
            return ordered
        return list(self._subplot_axes_by_polarization)

    def _sync_y_controls_with_visible_axis(self) -> None:
        """Align the Y controls with the focused (selected) subplot's limits."""
        if not self._subplot_axes_by_polarization:
            return

        # The Y fields reflect the focused (fit-target) subplot, so a global
        # auto-Y leaves them showing the selected projection's own range rather
        # than a span across all of them.
        axis = self._active_y_axis()
        if axis in self._subplot_axes_by_polarization:
            limits = self._y_limits_by_polarization.get(axis)
            if limits is not None:
                self._y_min.setValue(float(limits[0]))
                self._y_max.setValue(float(limits[1]))
            return

        # No focused subplot (e.g. nothing selected yet): show a global y-range
        # spanning all visible subplot axes.
        ranges: list[tuple[float, float]] = []
        for axis_key in self._all_mode_axes_order():
            limits = self._y_limits_by_polarization.get(axis_key)
            if limits is not None:
                ranges.append((float(limits[0]), float(limits[1])))
        if not ranges:
            return
        y_lo = min(lo for lo, _ in ranges)
        y_hi = max(hi for _, hi in ranges)
        self._y_min.setValue(y_lo)
        self._y_max.setValue(y_hi)

    def set_projections(
        self,
        projections: list[dict],
        selected: list[str] | None = None,
    ) -> None:
        """Show/update the projection chip bar, or hide it when unavailable.

        ``projections`` is an ordered list of ``{"label", "tint"?}`` dicts;
        ``selected`` is the subset of labels to show as subplots (defaults to
        all). The bar (and any multi-projection behaviour) is suppressed when
        fewer than two projections exist.
        """
        if not hasattr(self, "_projection_bar"):
            return

        specs = [dict(p) for p in (projections or []) if p.get("label")]
        if len(specs) < 2:
            self._cache_current_y_limits_for_axis()
            # Was a vector multi-pane (EMU ``ALL`` or stacked transverse
            # projections) on screen? ``_vector_subplot_datasets`` is populated
            # only by plot_vector_subplots and is the precise indicator — unlike
            # ``_current_polarization_axis``, which grouped-time subplots also set
            # to a run key, so keying on it would spuriously fire a full grouped
            # rebuild from clear(). Capture before reset so we can collapse after.
            had_vector_view = bool(self._vector_subplot_datasets)
            self._current_polarization_axis = None
            self._projection_specs = []
            self._tint_by_label = {}
            self._selected_projection_labels = []
            self._vector_subplot_datasets = {}
            self._projection_bar.set_projections([])
            self._update_y_limit_controls_for_axis(None)
            # Replot only on the vector → non-vector transition. With the vector
            # state cleared, _redraw_current_view falls through to the single-pane
            # path, collapsing the dual-pane back to one Axes. Skipping this when
            # no vector view was active avoids a spurious redraw on the common
            # non-vector refresh path (set_projections([]) fires routinely).
            if had_vector_view:
                self._redraw_current_view()
            return

        self._projection_specs = specs
        self._tint_by_label = {str(p["label"]): str(p["tint"]) for p in specs if p.get("tint")}
        labels = [str(p["label"]) for p in specs]
        wanted = set(selected) if selected else set(labels)
        chosen = [lbl for lbl in labels if lbl in wanted] or list(labels)

        self._projection_bar.set_projections(specs, chosen)
        self._selected_projection_labels = self._projection_bar.selected_labels()

        new_axis = self._axis_for_selection(self._selected_projection_labels)
        previous_axis = self._current_polarization_axis
        if previous_axis != new_axis:
            self._cache_current_y_limits_for_axis()
        self._current_polarization_axis = new_axis
        if previous_axis != new_axis:
            self._restore_y_limits_for_axis(new_axis)
            self._sync_y_controls_with_visible_axis()
            self._update_y_limit_controls_for_axis(new_axis)
            self._apply_limits()
        else:
            self._update_y_limit_controls_for_axis(new_axis)

    def selected_projection_labels(self) -> list[str]:
        """Return the projection labels currently selected.

        The chip bar is the source of truth once populated; the stored copy is
        the fallback during project restore, before the bar has been rebuilt.
        """
        bar_selection = self._projection_bar.selected_labels()
        return bar_selection or list(self._selected_projection_labels)

    # ── stacked-subplot fit target (Step 4) ────────────────────────────────

    def projection_tint(self, label: str | None) -> str | None:
        """Return the frame/identity tint for a projection label, if any."""
        if label is None:
            return None
        return self._tint_by_label.get(str(label))

    def fit_target_projection(self) -> str | None:
        """Return the projection whose subplot is the active single-fit target.

        Only meaningful in the stacked multi-subplot view; ``None`` otherwise.
        """
        if self._fit_target_projection in self._subplot_axes_by_polarization:
            return self._fit_target_projection
        return None

    def set_fit_target_projection(self, label: str | None, *, emit: bool = True) -> None:
        """Mark *label*'s subplot as the single-fit target and redraw its box.

        The fit target is also the focus for the manual Y-limit controls, so a
        change caches the outgoing subplot's limits and surfaces the incoming
        subplot's into the Y fields.
        """
        if label is not None and label not in self._subplot_axes_by_polarization:
            return
        changed = label != self._fit_target_projection
        if not changed:
            return  # no-op re-click: skip the tight-bbox recompute and redraw
        in_subplots = bool(self._subplot_axes_by_polarization)
        if in_subplots:
            self._cache_current_y_limits_for_axis()  # caches the outgoing target
        self._fit_target_projection = label
        if in_subplots:
            self._restore_y_limits_for_axis(label)
            self._update_y_limit_controls_for_axis(self._current_polarization_axis)
        self._refresh_fit_target_decoration()
        if emit and label is not None:
            self.fit_target_projection_changed.emit(str(label))

    def _subplot_projection_at_event(self, event) -> str | None:
        """Return the projection whose subplot axes contains the click event."""
        inaxes = getattr(event, "inaxes", None)
        if inaxes is None:
            return None
        for label, axis in self._subplot_axes_by_polarization.items():
            if axis is inaxes:
                return label
        return None

    def _default_fit_target(self) -> str | None:
        """Pick a sensible default target: the active single axis, else the first."""
        order = self._all_mode_axes_order()
        if not order:
            return None
        axis = self._current_polarization_axis
        if axis in order:
            return axis
        return order[0]

    def _clear_fit_target_decoration(self) -> None:
        for artist in self._fit_target_artists:
            try:
                artist.remove()
            except Exception:
                continue
        self._fit_target_artists = []

    def _refresh_fit_target_decoration(self) -> None:
        """Draw a neutral focus ring + 'fit target' pill on the active subplot.

        The ring/pill is *selection* state and is deliberately distinct from the
        per-projection frame tint (which encodes projection identity).
        """
        if not self._has_mpl:
            return
        self._clear_fit_target_decoration()
        target = self.fit_target_projection()
        # Only decorate when there is a genuine choice (two or more subplots).
        if target is None or len(self._subplot_axes_by_polarization) < 2:
            self._canvas.draw_idle()
            return

        ax = self._subplot_axes_by_polarization[target]
        # Decoration is best-effort chrome — a non-standard axis (e.g. a test
        # double) must never break the click/selection path.
        if not hasattr(ax, "get_position"):
            self._canvas.draw_idle()
            return

        from matplotlib.patches import FancyBboxPatch

        # Enclose the WHOLE subplot — tick labels and y-axis label included — by
        # using the axes' tight bbox (display coords) rather than just the data
        # area; fall back to the data-area position if no renderer is available.
        pos = ax.get_position()
        try:
            renderer = self._figure.canvas.get_renderer()
            tight = ax.get_tightbbox(renderer)
            if tight is not None:
                pos = tight.transformed(self._figure.transFigure.inverted())
        except Exception:
            pass
        margin = 0.006  # small gap so the ring doesn't touch the labels
        ring = FancyBboxPatch(
            (pos.x0 - margin, pos.y0 - margin),
            pos.width + 2 * margin,
            pos.height + 2 * margin,
            boxstyle="round,pad=0.002",
            transform=self._figure.transFigure,
            fill=False,
            edgecolor=tokens.TEXT,
            linewidth=1.6,
            zorder=10,
        )
        self._figure.add_artist(ring)
        self._fit_target_artists.append(ring)

        pill = ax.text(
            0.985,
            0.94,
            "fit target",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=tokens.SURFACE,
            zorder=11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": tokens.TEXT, "edgecolor": "none"},
        )
        self._fit_target_artists.append(pill)
        self._canvas.draw_idle()

    def _polarization_ylabel(self, axis_key: str | None) -> str:
        """Return y-axis label for the provided polarization component."""
        if axis_key in {"P_x", "P_y", "P_z"}:
            suffix = axis_key.split("_", 1)[1]
            return rf"$a_0 P_{{{suffix}}}(t)$ (%)"
        return "Asymmetry (%)"

    def _ensure_single_axis_mode(self) -> None:
        """Recreate a single-axis figure when leaving any multi-axis mode.

        The single-dataset path always renders onto exactly one ``self._ax``, so
        the postcondition is one tracked axis. Rebuild whenever that does not
        already hold: tracked vector subplots, *any* lingering extra axes (e.g. a
        dual-pane left behind when a projection was cleared without replotting),
        or a stale ``self._ax`` detached from the figure. Relying solely on
        ``_subplot_axes_by_polarization`` was fragile — clearing the vector state
        without clearing that dict still left the panes on the canvas.
        """
        already_single = (
            not self._subplot_axes_by_polarization
            and len(self._figure.axes) == 1
            and getattr(self, "_ax", None) in self._figure.axes
        )
        if already_single:
            return
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        self._ax = self._figure.add_subplot(111)
        style_figure(self._figure)
        style_axes(self._ax)
        self._subplot_axes_by_polarization = {}
        self._vector_subplot_datasets = {}
        self._sync_canvas_scroll_geometry(
            axis_count=1, target_height=self._default_canvas_min_height
        )

    def _fit_range_axes(self) -> list[object]:
        """Return axes that should show fit-range handles for the current view."""
        if self._subplot_axes_by_polarization:
            return list(self._subplot_axes_by_polarization.values())
        return [self._ax] if hasattr(self, "_ax") else []

    def _clear_fit_range_artists(self) -> None:
        """Remove existing fit-range span and handle artists from all axes."""
        for artists in (
            getattr(self, "_fit_span_artists", []),
            getattr(self, "_fit_min_handles", []),
            getattr(self, "_fit_max_handles", []),
        ):
            for artist in artists:
                try:
                    artist.remove()
                except NotImplementedError:
                    continue
                except Exception:
                    continue
        self._fit_span_artists = []
        self._fit_min_handles = []
        self._fit_max_handles = []

    def _plot_datasets_on_axis(
        self, ax, datasets: list[MuonDataset], axis_key: str | None
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Plot one or more datasets on ``ax`` and return flattened arrays for auto-y."""
        # Handoff plot grammar: y = 0 reference line under the data (it is
        # excluded from autoscaling, so positive-only data never stretches).
        draw_zero_line(ax)
        self._rrf_frame_drawn = None
        all_times: list[np.ndarray] = []
        all_asym: list[np.ndarray] = []
        all_err: list[np.ndarray] = []
        all_low: list[np.ndarray] = []
        period_color_counts: dict[str, int] = {}

        for i, dataset in enumerate(datasets):
            color = f"C{i % 10}"
            period_color = self._period_mode_color_for_dataset(dataset)
            if period_color is not None:
                variant_idx = period_color_counts.get(period_color, 0)
                color = self._period_mode_color_variant(period_color, variant_idx)
                period_color_counts[period_color] = variant_idx + 1
            analysis_dataset = rrf_display_dataset(self, self.get_analysis_dataset(dataset))
            if analysis_dataset is None:
                continue

            time = self._convert_frequency_axis_for_display(analysis_dataset.time)
            asymmetry = analysis_dataset.asymmetry
            error = analysis_dataset.error
            low_count_mask = self._low_count_mask_for_dataset(
                analysis_dataset,
                source_dataset=dataset,
            )

            finite_mask = np.isfinite(time) & np.isfinite(asymmetry) & np.isfinite(error)
            valid_low = finite_mask & low_count_mask
            valid_main = finite_mask & ~low_count_mask

            if np.any(valid_low):
                self._plot_errorbar_masked(
                    ax,
                    time,
                    asymmetry,
                    error,
                    valid_low,
                    fmt=".",
                    markersize=3,
                    color="0.6",
                    ecolor="0.6",
                    label="_nolegend_",
                )

            draw_mask = valid_main if np.any(valid_main) else finite_mask
            self._plot_errorbar_masked(
                ax,
                time,
                asymmetry,
                error,
                draw_mask,
                fmt=".",
                markersize=3,
                color=color,
                label=self._dataset_label_for(dataset),
            )

            fit_to_plot = self._fit_curve_for_dataset(dataset, axis_override=axis_key)
            fit_to_plot = rrf_display_fit_curve(self, fit_to_plot, analysis_dataset)
            if fit_to_plot is not None:
                t_fit, y_fit, fit_label = fit_to_plot
                fit_color = self._fit_line_color_for_dataset(
                    dataset,
                    default_color=color,
                    variant_index=i,
                    fit_label=fit_label,
                )
                ax.plot(t_fit, y_fit, "-", color=fit_color, linewidth=2, label="_nolegend_")

            if np.any(finite_mask):
                all_times.append(time[finite_mask])
                all_asym.append(asymmetry[finite_mask])
                all_err.append(error[finite_mask])
                all_low.append(low_count_mask[finite_mask])

        _, y_label = self._axis_labels_for_dataset(datasets[0] if datasets else None, axis_key)
        ax.set_ylabel(y_label)
        # Decimation chip: applied by the caller once every axis is drawn —
        # the chip counters are still accumulating while subplots render.
        rrf_draw_badge(self, ax)
        if all_times:
            return (
                np.concatenate(all_times),
                np.concatenate(all_asym),
                np.concatenate(all_err),
                np.concatenate(all_low),
            )
        return None, None, None, None

    def _projection_subplot_order(
        self, datasets_by_axis: dict[str, list[MuonDataset]]
    ) -> list[str]:
        """Return the subplot order for the projections that have datasets.

        Prefers the declared projection order, falling back to the canonical
        vector order and finally to the dict's own order.
        """
        spec_order = [str(p["label"]) for p in self._projection_specs]
        order = [a for a in spec_order if datasets_by_axis.get(a)]
        if not order:
            order = [a for a in ("P_x", "P_y", "P_z") if datasets_by_axis.get(a)]
        if not order:
            order = [a for a in datasets_by_axis if datasets_by_axis.get(a)]
        return order

    def _apply_projection_frame_tint(self, ax, axis_key: str) -> None:
        """Tint a subplot's left rail and y-label with the projection's colour.

        This is *projection identity* (chip ↔ subplot), deliberately distinct
        from the data trace colour, which encodes run identity in RG mode.
        """
        tint = self._tint_by_label.get(str(axis_key))
        if not tint:
            return
        ax.yaxis.label.set_color(tint)
        left = ax.spines.get("left") if hasattr(ax.spines, "get") else ax.spines["left"]
        if left is not None:
            left.set_color(tint)
            left.set_linewidth(2.0)

    def plot_vector_subplots(self, datasets_by_axis: dict[str, list[MuonDataset]]) -> None:
        """Render the selected projections as stacked subplots."""
        if not self._has_mpl:
            return

        order = self._projection_subplot_order(datasets_by_axis)
        if not order:
            return
        self._set_canvas_minimum_height_for_axes(len(order))

        self._set_alpha_label(None)
        self._grouped_time_subplot_datasets = []
        self._reset_decimation_view_state()
        self._vector_subplot_datasets = {k: list(v) for k, v in datasets_by_axis.items() if v}
        self._current_datasets = list(self._vector_subplot_datasets.get(order[0], []))
        self._current_dataset = self._current_datasets[-1] if self._current_datasets else None
        self._update_plot_header()

        # Stop listening to old axes before clearing the figure; stale callbacks
        # can otherwise push default [0, 1] limits into the limit fields.
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        style_figure(self._figure)
        self._subplot_axes_by_polarization = {}
        shared_ax = None
        last_arrays = (None, None, None, None)
        vector_x_ranges: list[tuple[float, float]] = []
        for idx, axis_key in enumerate(order):
            ax = self._figure.add_subplot(len(order), 1, idx + 1, sharex=shared_ax)
            style_axes(ax)
            if shared_ax is None:
                shared_ax = ax
            self._subplot_axes_by_polarization[axis_key] = ax
            self._ax = ax if idx == 0 else self._ax

            t, a, e, low = self._plot_datasets_on_axis(
                ax, self._vector_subplot_datasets.get(axis_key, []), axis_key
            )
            self._apply_projection_frame_tint(ax, axis_key)
            if axis_key in self._y_limits_by_polarization:
                y0, y1 = self._y_limits_by_polarization[axis_key]
                ax.set_ylim(y0, y1)
            elif t is None:
                # All-NaN projection: give it a neutral asymmetry range instead
                # of matplotlib's default (0, 1). Not cached in
                # _y_limits_by_polarization, so a later render with real data
                # auto-scales normally.
                ax.set_ylim(*_EMPTY_PROJECTION_YLIM)
            if idx == len(order) - 1:
                x_label, _ = self._axis_labels_for_dataset(
                    self._vector_subplot_datasets.get(axis_key, [None])[0],
                    axis_key,
                )
                ax.set_xlabel(x_label)
            else:
                ax.tick_params(labelbottom=False)
            if idx == 0:
                style_legend(ax.legend())
            if t is not None:
                last_arrays = (t, a, e, low)
                vector_x_ranges.append((float(np.min(t)), float(np.max(t))))

        # One chip on the bottom axis, after every subplot's points are
        # counted — applying per axis mid-loop would show partial totals.
        self._apply_x_axis_decimation_indicator(ax)

        self._last_plot_time = last_arrays[0]
        self._last_plot_asymmetry = last_arrays[1]
        self._last_plot_error = last_arrays[2]
        self._last_low_count_mask = last_arrays[3]

        if (not self._limits_initialized) and self._last_plot_time is not None:
            x_min = float(np.min(self._last_plot_time))
            x_max = float(np.max(self._last_plot_time))
            xpad = (x_max - x_min) * 0.05
            self._x_min.setValue(self._convert_frequency_axis_limit_to_control_value(x_min - xpad))
            self._x_max.setValue(self._convert_frequency_axis_limit_to_control_value(x_max + xpad))
            self._limits_initialized = True

        if vector_x_ranges and (self._fit_x_min is None or self._fit_x_max is None):
            seed = self._raw_fit_seed_range(
                [ds for axis_datasets in datasets_by_axis.values() for ds in axis_datasets]
            )
            if seed is None:
                seed = (
                    min(lo for lo, _ in vector_x_ranges),
                    max(hi for _, hi in vector_x_ranges),
                )
            self._fit_x_min, self._fit_x_max = seed

        if self._current_polarization_axis in self._subplot_axes_by_polarization:
            y_limits = self._y_limits_by_polarization.get(self._current_polarization_axis)
            if y_limits is not None:
                self._y_min.setValue(float(y_limits[0]))
                self._y_max.setValue(float(y_limits[1]))
        else:
            self._sync_y_controls_with_visible_axis()

        self._update_y_limit_controls_for_axis(self._current_polarization_axis)

        self._apply_limits(schedule_viewport_refresh=True)
        self._apply_auto_limits_if_enabled()
        self._connect_axis_limit_callbacks(list(self._subplot_axes_by_polarization.values()))

        # Auto-select a fit target so fitting is never dead-on-arrival; keep the
        # prior target when it is still visible. Emit so the fit panel rebinds.
        if self._fit_target_projection not in self._subplot_axes_by_polarization:
            self.set_fit_target_projection(self._default_fit_target())
        else:
            self._refresh_fit_target_decoration()

    def plot_grouped_time_domain_subplots(self, datasets: list[MuonDataset]) -> None:
        """Render grouped time-domain traces as stacked subplots."""
        if not self._has_mpl or not datasets:
            return
        self._set_canvas_minimum_height_for_axes(len(datasets))

        self._set_alpha_label(None)
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        self._subplot_axes_by_polarization = {}
        self._vector_subplot_datasets = {}
        self._reset_decimation_view_state()
        self._grouped_time_subplot_datasets = list(datasets)
        self._current_datasets = list(datasets)
        self._current_dataset = datasets[-1]
        self._update_plot_header()
        self._current_polarization_axis = None
        if hasattr(self, "_projection_bar"):
            self._projection_bar.hide()

        shared_ax = None
        last_arrays = (None, None, None, None)
        ordered_keys: list[str] = []
        grouped_x_ranges: list[tuple[float, float]] = []
        for idx, dataset in enumerate(datasets):
            # Raw and corrected builds share synthetic run numbers but live on
            # very different y scales; qualify the key so pinned y-limits from
            # one mode are not applied to the other.
            axis_key = str(dataset.run_number)
            if self._is_raw_counts_dataset(dataset):
                axis_key += ":raw"
            ordered_keys.append(axis_key)
            ax = self._figure.add_subplot(len(datasets), 1, idx + 1, sharex=shared_ax)
            style_axes(ax)
            if shared_ax is None:
                shared_ax = ax
            self._subplot_axes_by_polarization[axis_key] = ax
            if idx == 0:
                self._ax = ax

            t, a, e, low = self._plot_datasets_on_axis(ax, [dataset], axis_key)
            ax.set_title(str(dataset.run_label), loc="left", fontsize=10)
            if axis_key in self._y_limits_by_polarization:
                y0, y1 = self._y_limits_by_polarization[axis_key]
                ax.set_ylim(y0, y1)
            if idx == len(datasets) - 1:
                x_label, _ = self._axis_labels_for_dataset(dataset, axis_key)
                ax.set_xlabel(x_label)
            else:
                ax.tick_params(labelbottom=False)
            style_legend(ax.legend(loc="upper right"))
            if t is not None:
                last_arrays = (t, a, e, low)
                grouped_x_ranges.append((float(np.min(t)), float(np.max(t))))

        # One chip on the bottom axis, after every subplot's points are
        # counted — applying per axis mid-loop would show partial totals.
        self._apply_x_axis_decimation_indicator(ax)

        self._current_polarization_axis = ordered_keys[0] if ordered_keys else None
        self._last_plot_time = last_arrays[0]
        self._last_plot_asymmetry = last_arrays[1]
        self._last_plot_error = last_arrays[2]
        self._last_low_count_mask = last_arrays[3]

        if (not self._limits_initialized) and self._last_plot_time is not None:
            x_min = float(np.min(self._last_plot_time))
            x_max = float(np.max(self._last_plot_time))
            xpad = (x_max - x_min) * 0.05
            self._x_min.setValue(self._convert_frequency_axis_limit_to_control_value(x_min - xpad))
            self._x_max.setValue(self._convert_frequency_axis_limit_to_control_value(x_max + xpad))
            self._limits_initialized = True

        if grouped_x_ranges and (self._fit_x_min is None or self._fit_x_max is None):
            seed = self._raw_fit_seed_range(list(datasets))
            if seed is None:
                seed = (
                    min(lo for lo, _ in grouped_x_ranges),
                    max(hi for _, hi in grouped_x_ranges),
                )
            self._fit_x_min, self._fit_x_max = seed

        self._sync_y_controls_with_visible_axis()
        self._update_y_limit_controls_for_axis(self._current_polarization_axis)
        self._apply_limits(schedule_viewport_refresh=True)
        self._apply_auto_limits_if_enabled()
        self._connect_axis_limit_callbacks(list(self._subplot_axes_by_polarization.values()))
        self._apply_log_counts_scale()

    def plot_maxent_reconstruction(
        self, datasets: list[MuonDataset], *, combined: bool = False
    ) -> None:
        """Overlay the MaxEnt time-domain reconstruction on the measured data.

        Each dataset carries the observed signal in ``asymmetry`` and the
        forward-model reconstruction + weighted residual in
        ``metadata['maxent_model']`` / ``['maxent_residual']``.  Two layouts
        share the same data and χ²:

        - *per-group* (default): one main axis (data points + model line) above
          a residuals strip per group, stacked vertically;
        - *combined* (``combined=True``): every group's data+model overlaid on a
          single colour-coded axis above one shared residuals strip.

        The displayed χ² is the engine's by construction (the residuals are
        ``(data − model)/σ``), so both layouts report the same total.
        """
        if not self._has_mpl or not datasets:
            return
        if combined:
            self._plot_maxent_reconstruction_combined(datasets)
        else:
            self._plot_maxent_reconstruction_per_group(datasets)

    def _plot_maxent_reconstruction_per_group(self, datasets: list[MuonDataset]) -> None:
        """Stacked per-group data+model axes, each above its own residuals strip."""
        self._set_canvas_minimum_height_for_axes(len(datasets))
        self._set_alpha_label(None)
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        self._subplot_axes_by_polarization = {}
        self._vector_subplot_datasets = {}
        self._grouped_time_subplot_datasets = []
        self._current_datasets = list(datasets)
        self._current_dataset = datasets[-1]
        self._update_plot_header()
        self._current_polarization_axis = None
        if hasattr(self, "_projection_bar"):
            self._projection_bar.hide()

        n = len(datasets)
        gridspec = self._figure.add_gridspec(2 * n, 1, height_ratios=[3, 1] * n, hspace=0.45)
        shared_ax = None
        total_chi2 = 0.0
        total_obs = 0
        last_time = None
        for idx, dataset in enumerate(datasets):
            time = np.asarray(dataset.time, dtype=float)
            data = np.asarray(dataset.asymmetry, dtype=float)
            model = np.asarray(dataset.metadata.get("maxent_model", data), dtype=float)
            residual = np.asarray(
                dataset.metadata.get("maxent_residual", data - model), dtype=float
            )
            total_chi2 += float(
                dataset.metadata.get("maxent_group_chi2", float(np.sum(residual**2)))
            )
            total_obs += int(dataset.metadata.get("maxent_group_n_obs", residual.size))

            ax_main = self._figure.add_subplot(gridspec[2 * idx], sharex=shared_ax)
            ax_res = self._figure.add_subplot(gridspec[2 * idx + 1], sharex=ax_main)
            if shared_ax is None:
                shared_ax = ax_main
            if idx == 0:
                self._ax = ax_main
            style_axes(ax_main)
            style_axes(ax_res)
            axis_key = f"recon:{dataset.run_number}:{idx}"
            self._subplot_axes_by_polarization[axis_key] = ax_main

            ax_main.plot(
                time, data, ".", markersize=3, color=tokens.PLOT_DATA, label="Data", alpha=0.7
            )
            ax_main.plot(
                time, model, "-", linewidth=1.4, color=tokens.PLOT_FIT, label="Reconstruction"
            )
            ax_main.set_title(str(dataset.run_label), loc="left", fontsize=10)
            ax_main.set_ylabel("Recon. (a.u.)")
            ax_main.tick_params(labelbottom=False)
            style_legend(ax_main.legend(loc="upper right"))

            ax_res.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=0.8)
            ax_res.plot(time, residual, "-", linewidth=0.9, color=tokens.PLOT_DATA)
            ax_res.set_ylabel("(d−m)/σ")
            if idx == n - 1:
                ax_res.set_xlabel("Time (μs)")
            else:
                ax_res.tick_params(labelbottom=False)
            last_time = time

        self._current_polarization_axis = next(iter(self._subplot_axes_by_polarization), None)
        if total_obs:
            chi2_per_n = total_chi2 / float(total_obs)
            self._figure.suptitle(
                f"MaxEnt reconstruction — χ² = {total_chi2:.1f} ({chi2_per_n:.2f} per point)",
                fontsize=10,
            )
        if last_time is not None and last_time.size:
            self._last_plot_time = last_time
        self._figure.canvas.draw_idle()

    def _plot_maxent_reconstruction_combined(self, datasets: list[MuonDataset]) -> None:
        """All groups' data+model on one colour-coded axis + a shared residuals strip.

        Each group keeps a distinct colour (matplotlib's default cycle) shared
        between its data points, model line and residual trace, so the combined
        fit quality is legible at a glance.  The total χ² equals the per-group
        layout's by construction.
        """
        self._set_canvas_minimum_height_for_axes(2)
        self._set_alpha_label(None)
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        self._subplot_axes_by_polarization = {}
        self._vector_subplot_datasets = {}
        self._grouped_time_subplot_datasets = []
        self._current_datasets = list(datasets)
        self._current_dataset = datasets[-1]
        self._update_plot_header()
        self._current_polarization_axis = None
        if hasattr(self, "_projection_bar"):
            self._projection_bar.hide()

        gridspec = self._figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_main = self._figure.add_subplot(gridspec[0])
        ax_res = self._figure.add_subplot(gridspec[1], sharex=ax_main)
        style_axes(ax_main)
        style_axes(ax_res)
        self._ax = ax_main
        axis_key = f"recon:combined:{datasets[0].run_number}"
        self._subplot_axes_by_polarization[axis_key] = ax_main
        self._current_polarization_axis = axis_key

        total_chi2 = 0.0
        total_obs = 0
        last_time = None
        for idx, dataset in enumerate(datasets):
            time = np.asarray(dataset.time, dtype=float)
            data = np.asarray(dataset.asymmetry, dtype=float)
            model = np.asarray(dataset.metadata.get("maxent_model", data), dtype=float)
            residual = np.asarray(
                dataset.metadata.get("maxent_residual", data - model), dtype=float
            )
            total_chi2 += float(
                dataset.metadata.get("maxent_group_chi2", float(np.sum(residual**2)))
            )
            total_obs += int(dataset.metadata.get("maxent_group_n_obs", residual.size))
            label = str(dataset.metadata.get("group_name", dataset.run_label))
            color = f"C{idx % 10}"
            ax_main.plot(time, data, ".", markersize=3, color=color, alpha=0.55)
            ax_main.plot(time, model, "-", linewidth=1.4, color=color, label=label)
            ax_res.plot(time, residual, "-", linewidth=0.8, color=color, alpha=0.85)
            last_time = time

        ax_main.set_ylabel("Recon. (a.u.)")
        ax_main.tick_params(labelbottom=False)
        style_legend(ax_main.legend(loc="upper right", title="Group"))
        ax_res.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=0.8)
        ax_res.set_ylabel("(d−m)/σ")
        ax_res.set_xlabel("Time (μs)")

        if total_obs:
            chi2_per_n = total_chi2 / float(total_obs)
            self._figure.suptitle(
                f"MaxEnt reconstruction (combined) — χ² = {total_chi2:.1f} "
                f"({chi2_per_n:.2f} per point)",
                fontsize=10,
            )
        if last_time is not None and last_time.size:
            self._last_plot_time = last_time
        self._figure.canvas.draw_idle()

    def _alpha_value_for_dataset(self, dataset: MuonDataset) -> float | None:
        """Return the asymmetry alpha value used for *dataset*, if available."""
        run = dataset.run
        if run is not None and isinstance(run.grouping, dict):
            axis = self._axis_canonical_key(run.grouping.get("vector_axis"))
            axis_key = {
                "P_x": "alpha_x",
                "P_y": "alpha_y",
                "P_z": "alpha_z",
            }.get(axis)
            legacy_axis_key = {
                "P_x": "alpha_px",
                "P_y": "alpha_py",
                "P_z": "alpha_pz",
            }.get(axis)
            if axis_key is not None:
                alpha = run.grouping.get(axis_key, run.grouping.get(legacy_axis_key))
                try:
                    return float(alpha)
                except (TypeError, ValueError):
                    pass

            alpha = run.grouping.get("alpha")
            try:
                return float(alpha)
            except (TypeError, ValueError):
                pass

        alpha = dataset.metadata.get("alpha")
        try:
            return float(alpha)
        except (TypeError, ValueError):
            return None

    def _single_dataset_alpha_label_text(self, dataset: MuonDataset) -> str | None:
        """Return alpha label text for single-dataset views when available."""
        if self._is_frequency_domain_dataset(dataset):
            return None
        alpha = self._alpha_value_for_dataset(dataset)
        if alpha is None:
            return None
        return f"(alpha = {alpha:.6g})"

    def _period_mode_color_for_dataset(self, dataset: MuonDataset) -> str | None:
        """Return WiMDA-like color for selected two-period mode, else None."""
        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if not isinstance(grouping, dict):
            return None
        period_hist = grouping.get("period_histograms")
        if not isinstance(period_hist, list) or len(period_hist) != 2:
            return None

        mode = str(grouping.get("period_mode", PeriodMode.RED))
        color_map = {
            str(PeriodMode.RED): "#c00000",
            str(PeriodMode.GREEN): "#008000",
            str(PeriodMode.GREEN_MINUS_RED): "#0000c0",
            str(PeriodMode.GREEN_PLUS_RED): "#800080",
        }
        return color_map.get(mode)

    def _period_mode_color_variant(self, base_color: str, index: int) -> str:
        """Return a deterministic, high-contrast color variant for RG mode.

        The first trace uses the mode's base color. Additional traces rotate
        through a contrasting palette so selected runs are clearly separable.
        """
        if index <= 0:
            return base_color

        # Okabe-Ito style high-contrast palette with mode-specific exclusions
        # so overlays remain visually distinct from the selected RG base color.
        palette_by_base = {
            "#c00000": [  # red mode
                "#0072b2",  # blue
                "#56b4e9",  # sky blue
                "#009e73",  # bluish green
                "#f0e442",  # yellow
                "#cc79a7",  # magenta
                "#000000",  # black
                "#e69f00",  # orange
            ],
            "#008000": [  # green mode
                "#0072b2",  # blue
                "#56b4e9",  # sky blue
                "#f0e442",  # yellow
                "#cc79a7",  # magenta
                "#000000",  # black
                "#e69f00",  # orange
                "#d55e00",  # vermillion
            ],
            "#0000c0": [  # G-R mode (blue)
                "#e69f00",  # orange
                "#f0e442",  # yellow
                "#009e73",  # bluish green
                "#cc79a7",  # magenta
                "#000000",  # black
                "#d55e00",  # vermillion
            ],
            "#800080": [  # G+R mode (purple)
                "#0072b2",  # blue
                "#56b4e9",  # sky blue
                "#009e73",  # bluish green
                "#f0e442",  # yellow
                "#e69f00",  # orange
                "#000000",  # black
            ],
        }
        distinct_palette = palette_by_base.get(
            base_color.lower(),
            [
                "#0072b2",  # blue
                "#e69f00",  # orange
                "#56b4e9",  # sky blue
                "#f0e442",  # yellow
                "#009e73",  # bluish green
                "#cc79a7",  # magenta
                "#000000",  # black
                "#d55e00",  # vermillion
            ],
        )
        return distinct_palette[(index - 1) % len(distinct_palette)]

    def _fit_line_color_for_dataset(
        self,
        dataset: MuonDataset,
        *,
        default_color: str,
        variant_index: int = 0,
        fit_label: str | None = None,
    ) -> str:
        """Return the fit-line colour for a dataset.

        Preview curves get a fixed accent colour. In grouped time-domain mode each
        detector group sits on its own subplot, so there is no fit↔dataset overlay
        to disambiguate — the fit uses the canonical red fit colour, which resolves
        far better against the (blue) data points than matching the marker colour.
        Every other fit matches its data markers so that in overlay mode each fit
        visually belongs to its dataset.
        """
        if isinstance(fit_label, str) and "preview" in fit_label.lower():
            return "#d73a49"
        if self._grouped_time_subplot_datasets:
            return tokens.PLOT_FIT
        return default_color

    def set_fit_range(self, x_min: float, x_max: float) -> None:
        """Set fit range limits and refresh visual handles."""
        self._set_fit_range(x_min, x_max, emit_signal=True, redraw=True)

    def plot_datasets(self, datasets: list[MuonDataset]) -> None:
        """Plot multiple datasets on the same axes with per-dataset colours.

        Each dataset is assigned a colour from matplotlib's default cycle
        (C0, C1, …). Any stored fit curve for a run is drawn in a matching
        style, with high-contrast overrides for period-mode datasets.
        Low-count (grey) points are still drawn at reduced opacity.
        The axes limits are initialised from the combined data extent on the
        first call, then held fixed on subsequent redraws.

        Delegates to :meth:`plot_dataset` when *datasets* has exactly one
        entry so that the single-dataset code path (limit initialisation,
        fit-range handling, etc.) is exercised unchanged.
        """
        if not self._has_mpl or not datasets:
            return
        if len(datasets) == 1:
            self.plot_dataset(datasets[0])
            return

        self._set_canvas_minimum_height_for_axes(1)

        self._ensure_single_axis_mode()
        self._grouped_time_subplot_datasets = []
        self._set_alpha_label(None)
        self._reset_decimation_view_state()
        self._rrf_frame_drawn = None
        self._current_dataset = datasets[-1]
        self._current_datasets = list(datasets)
        self._update_plot_header()
        self._set_frequency_reference_from_dataset(datasets[0])
        self._ax.clear()
        style_axes(self._ax)
        draw_zero_line(self._ax)

        all_times: list[np.ndarray] = []
        all_asym: list[np.ndarray] = []
        all_err: list[np.ndarray] = []
        all_low: list[np.ndarray] = []
        period_color_counts: dict[str, int] = {}

        for i, dataset in enumerate(datasets):
            color = f"C{i % 10}"
            period_color = self._period_mode_color_for_dataset(dataset)
            if period_color is not None:
                variant_idx = period_color_counts.get(period_color, 0)
                color = self._period_mode_color_variant(period_color, variant_idx)
                period_color_counts[period_color] = variant_idx + 1
            analysis_dataset = rrf_display_dataset(self, self.get_analysis_dataset(dataset))
            if not self._has_plottable_samples(analysis_dataset):
                continue

            time = self._convert_frequency_axis_for_display(analysis_dataset.time)
            asymmetry = analysis_dataset.asymmetry
            error = analysis_dataset.error
            low_count_mask = self._low_count_mask_for_dataset(
                analysis_dataset,
                source_dataset=dataset,
            )

            finite_mask = np.isfinite(time) & np.isfinite(asymmetry) & np.isfinite(error)
            valid_low = finite_mask & low_count_mask
            valid_main = finite_mask & ~low_count_mask

            if np.any(valid_low):
                self._plot_errorbar_masked(
                    self._ax,
                    time,
                    asymmetry,
                    error,
                    valid_low,
                    fmt=".",
                    markersize=3,
                    color="0.6",
                    ecolor="0.6",
                    label="_nolegend_",
                )

            draw_mask = valid_main if np.any(valid_main) else finite_mask
            self._plot_errorbar_masked(
                self._ax,
                time,
                asymmetry,
                error,
                draw_mask,
                fmt=".",
                markersize=3,
                color=color,
                label=self._dataset_label_for(dataset),
            )

            # Overlay fit curve in same colour; excluded from legend by "_" prefix.
            fit_to_plot = self._fit_curve_for_dataset(dataset)
            fit_to_plot = rrf_display_fit_curve(self, fit_to_plot, analysis_dataset)
            if fit_to_plot is not None:
                t_fit, y_fit, fit_label = fit_to_plot
                fit_color = self._fit_line_color_for_dataset(
                    dataset,
                    default_color=color,
                    variant_index=i,
                    fit_label=fit_label,
                )
                self._ax.plot(t_fit, y_fit, "-", color=fit_color, linewidth=2, label="_nolegend_")

            if np.any(finite_mask):
                all_times.append(time[finite_mask])
                all_asym.append(asymmetry[finite_mask])
                all_err.append(error[finite_mask])
                all_low.append(low_count_mask[finite_mask])

        x_label, y_label = self._axis_labels_for_dataset(
            datasets[0], self._current_polarization_axis
        )
        self._apply_axis_labels(x_label, y_label)
        self._draw_annotations()

        if all_times:
            self._last_plot_time = np.concatenate(all_times)
            self._last_plot_asymmetry = np.concatenate(all_asym)
            self._last_plot_error = np.concatenate(all_err)
            self._last_low_count_mask = np.concatenate(all_low)
            style_legend(self._ax.legend())

            if not self._limits_initialized:
                t_all = self._last_plot_time
                a_all = self._last_plot_asymmetry
                e_all = self._last_plot_error
                x_min, x_max = float(t_all.min()), float(t_all.max())
                y_min = float((a_all - e_all).min())
                y_max = float((a_all + e_all).max())
                xpad = (x_max - x_min) * 0.05
                ypad = (y_max - y_min) * 0.05
                self._x_min.setValue(
                    self._convert_frequency_axis_limit_to_control_value(x_min - xpad)
                )
                self._x_max.setValue(
                    self._convert_frequency_axis_limit_to_control_value(x_max + xpad)
                )
                self._y_min.setValue(y_min - ypad)
                self._y_max.setValue(y_max + ypad)
                self._limits_initialized = True

            # Set fit range to span all datasets (raw axes — see _raw_fit_seed_range).
            if self._fit_x_min is None or self._fit_x_max is None:
                seed = self._raw_fit_seed_range(list(datasets))
                if seed is None:
                    seed = (
                        float(self._last_plot_time.min()),
                        float(self._last_plot_time.max()),
                    )
                self._fit_x_min, self._fit_x_max = seed
        else:
            self._last_plot_time = None
            self._last_plot_asymmetry = None
            self._last_plot_error = None
            self._last_low_count_mask = None
            self._fit_x_min = None
            self._fit_x_max = None

        self._draw_fit_range_artists()
        self._apply_limits()
        self._apply_auto_limits_if_enabled()
        self._update_export_enabled()
        self._connect_axis_limit_callbacks([self._ax])

    def set_diamagnetic_overlay(
        self,
        time_us: np.ndarray | None,
        signal: np.ndarray | None,
        *,
        run_number: int | None = None,
    ) -> None:
        """Set (or clear) the diamagnetic-fit curve drawn on the time-domain view.

        Passing ``None`` for the arrays clears it. The overlay is tagged with
        *run_number* and only drawn while that run is displayed, so it never
        lingers on an unrelated run; it is redrawn on the next
        :meth:`plot_dataset` call for a time-domain panel.
        """
        if time_us is None or signal is None:
            self._diamagnetic_overlay = None
        else:
            self._diamagnetic_overlay = (
                None if run_number is None else int(run_number),
                np.asarray(time_us, dtype=float),
                np.asarray(signal, dtype=float),
            )
        if (
            self._has_mpl
            and self._current_dataset is not None
            and not self._is_frequency_plot_panel()
        ):
            self.plot_dataset(self._current_dataset)

    def _overlay_diamagnetic_fit(self, analysis_dataset: MuonDataset | None = None) -> None:
        """Draw the stored diamagnetic-fit curve on the time-domain axes.

        Only drawn when the overlay's run matches the displayed dataset, so a
        fit from a previous run never shows on an unrelated one.  When the
        display is in the rotating frame the curve passes through the same
        demodulation pipeline as the data.
        """
        if (
            not self._has_mpl
            or self._diamagnetic_overlay is None
            or self._is_frequency_plot_panel()
        ):
            return
        overlay_run, time_us, signal = self._diamagnetic_overlay
        shown = rrf_display_fit_curve(self, (time_us, signal, "Diamagnetic fit"), analysis_dataset)
        if shown is not None:
            time_us, signal, _ = shown
        current_run = (
            int(self._current_dataset.run_number)
            if self._current_dataset is not None and self._current_dataset.run_number is not None
            else None
        )
        if overlay_run is not None and overlay_run != current_run:
            return
        finite = np.isfinite(time_us) & np.isfinite(signal)
        if not np.any(finite):
            return
        self._ax.plot(
            time_us[finite],
            signal[finite],
            "-",
            color=tokens.WARN,
            linewidth=1.5,
            alpha=0.9,
            label="Diamagnetic fit",
        )

    def _overlay_fourier_imag(self, dataset: MuonDataset, time: np.ndarray) -> None:
        """Overlay the imaginary quadrature for the Real+Imag display mode.

        The averaged spectrum carries the imag channel in
        ``metadata["fourier_imag"]``; the primary trace is the real part.
        """
        if not self._has_mpl or not isinstance(dataset.metadata, dict):
            return
        imag = dataset.metadata.get("fourier_imag")
        if imag is None:
            return
        imag_arr = np.asarray(imag, dtype=float)
        if imag_arr.size != np.asarray(time).size:
            return
        finite = np.isfinite(time) & np.isfinite(imag_arr)
        if not np.any(finite):
            return
        self._ax.plot(
            np.asarray(time)[finite],
            imag_arr[finite],
            "-",
            color="C1",
            linewidth=1.2,
            alpha=0.85,
            label="Imag",
        )

    def plot_dataset(self, dataset: MuonDataset) -> None:
        """Plot a dataset, optionally rebinned according to the bunch factor.

        The input dataset is stored unchanged as the current dataset. If the
        bunch factor is greater than 1, temporary rebinned arrays are created
        for plotting. The source dataset itself is never mutated.
        """
        if not self._has_mpl:
            return

        self._set_canvas_minimum_height_for_axes(1)

        self._ensure_single_axis_mode()
        self._grouped_time_subplot_datasets = []
        self._reset_decimation_view_state()
        self._rrf_frame_drawn = None
        # Store the original dataset
        self._current_dataset = dataset
        self._current_datasets = [dataset]
        self._update_plot_header()
        self._set_frequency_reference_from_dataset(dataset)

        analysis_dataset = rrf_display_dataset(self, self.get_analysis_dataset(dataset))
        if not self._has_plottable_samples(analysis_dataset):
            self._render_empty_plot_state(alpha_text=self._single_dataset_alpha_label_text(dataset))
            return
        time = self._convert_frequency_axis_for_display(analysis_dataset.time)
        asymmetry = analysis_dataset.asymmetry
        error = analysis_dataset.error
        low_count_mask = self._low_count_mask_for_dataset(
            analysis_dataset,
            source_dataset=dataset,
        )

        self._last_plot_time = time
        self._last_plot_asymmetry = asymmetry
        self._last_plot_error = error
        self._last_low_count_mask = low_count_mask

        self._ax.clear()
        style_axes(self._ax)
        draw_zero_line(self._ax)

        finite_mask = np.isfinite(time) & np.isfinite(asymmetry) & np.isfinite(error)
        valid_low = finite_mask & low_count_mask
        valid_main = finite_mask & ~low_count_mask

        if np.any(valid_low):
            self._plot_errorbar_masked(
                self._ax,
                time,
                asymmetry,
                error,
                valid_low,
                fmt=".",
                markersize=3,
                color="0.6",
                ecolor="0.6",
                label="_nolegend_",
            )

        draw_mask = valid_main if np.any(valid_main) else finite_mask
        point_color = self._period_mode_color_for_dataset(dataset)
        self._plot_errorbar_masked(
            self._ax,
            time,
            asymmetry,
            error,
            draw_mask,
            fmt=".",
            markersize=3,
            color=point_color,
            ecolor=point_color,
            label=self._dataset_label_for(dataset),
        )
        self._overlay_fourier_imag(dataset, time)
        self._overlay_diamagnetic_fit(analysis_dataset)
        x_label, y_label = self._axis_labels_for_dataset(dataset, self._current_polarization_axis)
        self._apply_axis_labels(x_label, y_label)
        self._set_alpha_label(self._single_dataset_alpha_label_text(dataset))

        # Re-plot fit curve if it exists (check both single and global fits)
        fit_to_plot = self._fit_curve_for_dataset(dataset)
        fit_to_plot = rrf_display_fit_curve(self, fit_to_plot, analysis_dataset)

        if fit_to_plot is not None:
            t_fit, y_fit, fit_label = fit_to_plot
            fit_color = self._fit_line_color_for_dataset(
                dataset,
                default_color=tokens.PLOT_FIT,
                fit_label=fit_label,
            )
            self._ax.plot(
                t_fit,
                y_fit,
                "-",
                color=fit_color,
                linewidth=2,
                label=fit_label,
            )

        self._draw_annotations()

        style_legend(self._ax.legend())

        # Initialize limits once; preserve user-set limits on redraw.
        if not self._limits_initialized:
            x_min, x_max = float(time.min()), float(time.max())
            y_min = float((asymmetry - error).min())
            y_max = float((asymmetry + error).max())

            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05

            self._x_min.setValue(
                self._convert_frequency_axis_limit_to_control_value(x_min - x_padding)
            )
            self._x_max.setValue(
                self._convert_frequency_axis_limit_to_control_value(x_max + x_padding)
            )
            self._y_min.setValue(y_min - y_padding)
            self._y_max.setValue(y_max + y_padding)
            self._limits_initialized = True

        if self._fit_x_min is None or self._fit_x_max is None:
            seed = self._raw_fit_seed_range([dataset])
            if seed is None:
                seed = (float(time.min()), float(time.max()))
            self._fit_x_min, self._fit_x_max = seed

        self._draw_fit_range_artists()

        # Apply the limits
        self._apply_limits()
        self._apply_auto_limits_if_enabled()
        self._update_export_enabled()
        self._connect_axis_limit_callbacks([self._ax])

    def _apply_limits(self, *, schedule_viewport_refresh: bool = False) -> None:
        """Apply the specified axis limits to the plot."""
        if not self._has_mpl:
            return

        x0 = float(self._x_min.value())
        x1 = float(self._x_max.value())
        if self._is_frequency_plot_panel():
            x0 = self._convert_frequency_control_value_to_axis_limit(x0)
            x1 = self._convert_frequency_control_value_to_axis_limit(x1)
        y0 = float(self._y_min.value())
        y1 = float(self._y_max.value())

        # Matplotlib warns on identical x-limits; expand degenerate ranges slightly.
        if np.isclose(x0, x1):
            pad = max(1e-9, abs(x0) * 1e-6)
            x0 -= pad
            x1 += pad

        if self._subplot_axes_by_polarization:
            self._draw_fit_range_artists()
            self._syncing_limits_from_axes = True
            try:
                focused_axis = self._active_y_axis()
                for axis_key, axis_obj in self._subplot_axes_by_polarization.items():
                    axis_obj.set_xlim(x0, x1)
                    # Manual Y applies only to the focused (selected) subplot.
                    if focused_axis == axis_key:
                        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
                        self._y_limits_by_polarization[axis_key] = (lo, hi)
                    limits = self._y_limits_by_polarization.get(axis_key)
                    if limits is not None:
                        axis_obj.set_ylim(float(limits[0]), float(limits[1]))
                    # A non-focused subplot with no cached limit keeps its own
                    # (auto-scaled) y-range — manual Y never bleeds across
                    # subplots.
            finally:
                self._syncing_limits_from_axes = False
            self._canvas.draw()
            self._emit_view_limits_changed()
            if schedule_viewport_refresh and not self._viewport_refresh_in_progress:
                self._schedule_viewport_refresh()
            return

        self._ax.set_xlim(x0, x1)
        self._ax.set_ylim(y0, y1)
        self._cache_current_y_limits_for_axis()
        self._draw_fit_range_artists()
        self._canvas.draw()
        self._emit_view_limits_changed()
        if schedule_viewport_refresh and not self._viewport_refresh_in_progress:
            self._schedule_viewport_refresh()

    def _apply_auto_limits_if_enabled(self) -> None:
        """Re-apply persistent auto-limit toggles after a dataset redraw."""
        if not self._has_mpl:
            return

        if self._auto_x_btn.isChecked():
            self._auto_x_limits()
        if self._auto_y_btn.isChecked():
            self._auto_y_limits()

    def _on_auto_x_button_clicked(self, checked: bool) -> None:
        """Apply auto X immediately when the toggle is enabled."""
        if checked:
            self._auto_x_limits()

    def _on_auto_y_button_clicked(self, checked: bool) -> None:
        """Apply auto Y immediately when the toggle is enabled."""
        if checked:
            self._auto_y_limits()

    def _draw_annotations(self) -> None:
        """Recreate annotation artists on the active axis."""
        rrf_draw_badge(self, self._ax)
        for ann in self._annotations:
            artist = self._ax.text(
                ann["x"],
                ann["y"],
                ann["text"],
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": tokens.SURFACE,
                    "edgecolor": tokens.BORDER,
                    "alpha": 0.95,
                },
                zorder=5,
            )
            ann["artist"] = artist

    def _auto_x_limits(self) -> None:
        """Auto-scale x-axis and update x-limit controls."""
        if not self._has_mpl:
            return

        if self._last_plot_time is None:
            return

        finite_mask = np.isfinite(self._last_plot_time)
        if not np.any(finite_mask):
            return

        time = self._last_plot_time[finite_mask]
        x_min = float(np.min(time))
        x_max = float(np.max(time))
        if x_max <= x_min:
            delta = max(abs(x_min) * 0.05, 1e-6)
            x_min -= delta
            x_max += delta
        else:
            padding = (x_max - x_min) * 0.05
            x_min -= padding
            x_max += padding

        self._x_min.setValue(self._convert_frequency_axis_limit_to_control_value(x_min))
        self._x_max.setValue(self._convert_frequency_axis_limit_to_control_value(x_max))
        # Widening the x-window past the previous (possibly zoomed-in) view leaves
        # the rendered points decimated for the *old* narrow viewport — the data
        # outside it stays missing until a redraw recomputes decimation. Schedule a
        # viewport refresh so Auto X re-decimates over the full range it just set.
        # (_last_plot_time is already full-resolution, so the computed range is
        # correct; only the on-screen sample was stale.) The refresh is coalesced
        # and self-suppresses while one is already in progress, so the render-path
        # call via _apply_auto_limits_if_enabled cannot recurse.
        self._apply_limits(schedule_viewport_refresh=True)

    def _auto_y_limits(self) -> None:
        """Auto-scale y-axis from visible, non-low-count points only."""
        if not self._has_mpl:
            return

        if self._grouped_time_subplot_datasets and self._subplot_axes_by_polarization:
            updated = False
            for dataset in self._grouped_time_subplot_datasets:
                axis_key = str(dataset.run_number)
                if self._is_raw_counts_dataset(dataset):
                    axis_key += ":raw"
                if axis_key not in self._subplot_axes_by_polarization:
                    continue
                limits = self._auto_y_limits_for_datasets([dataset])
                if limits is None:
                    continue
                self._y_limits_by_polarization[axis_key] = limits
                updated = True
            if not updated:
                return
            self._sync_y_controls_with_visible_axis()
            self._apply_limits()
            return

        if self._subplot_axes_by_polarization and self._current_polarization_axis == "ALL":
            updated = False
            for axis_key in self._all_mode_axes_order():
                limits = self._auto_y_limits_for_datasets(
                    self._vector_subplot_datasets.get(axis_key, [])
                )
                if limits is None:
                    continue
                self._y_limits_by_polarization[axis_key] = limits
                updated = True
            if not updated:
                return
            self._sync_y_controls_with_visible_axis()
            self._apply_limits()
            return

        if (
            self._last_plot_time is None
            or self._last_plot_asymmetry is None
            or self._last_plot_error is None
        ):
            return

        x_lo = float(self._x_min.value())
        x_hi = float(self._x_max.value())
        if self._is_frequency_plot_panel():
            x_lo = self._convert_frequency_control_value_to_axis_limit(x_lo)
            x_hi = self._convert_frequency_control_value_to_axis_limit(x_hi)
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

        limits = self._auto_y_limits_from_arrays(asymmetry, error, mask)
        if limits is None:
            return
        y_min, y_max = limits

        self._y_min.setValue(y_min)
        self._y_max.setValue(y_max)

        self._apply_limits()

    def _auto_y_limits_from_arrays(
        self,
        asymmetry: np.ndarray,
        error: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, float] | None:
        """Return padded y-limits for the masked asymmetry arrays."""
        if not np.any(mask):
            return None

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
        return y_min, y_max

    def _auto_y_limits_for_datasets(
        self,
        datasets: list[MuonDataset],
    ) -> tuple[float, float] | None:
        """Return auto y-limits for one polarization's datasets."""
        all_times: list[np.ndarray] = []
        all_asymmetry: list[np.ndarray] = []
        all_error: list[np.ndarray] = []
        all_low_masks: list[np.ndarray] = []

        for dataset in datasets:
            analysis_dataset = rrf_display_dataset(self, self.get_analysis_dataset(dataset))
            if analysis_dataset is None:
                continue

            time = self._convert_frequency_axis_for_display(analysis_dataset.time)
            asymmetry = analysis_dataset.asymmetry
            error = analysis_dataset.error
            low_mask = self._low_count_mask_for_dataset(
                analysis_dataset,
                source_dataset=dataset,
            )

            finite_mask = np.isfinite(time) & np.isfinite(asymmetry) & np.isfinite(error)
            if not np.any(finite_mask):
                continue

            all_times.append(time[finite_mask])
            all_asymmetry.append(asymmetry[finite_mask])
            all_error.append(error[finite_mask])
            all_low_masks.append(low_mask[finite_mask])

        if not all_times:
            return None

        time = np.concatenate(all_times)
        asymmetry = np.concatenate(all_asymmetry)
        error = np.concatenate(all_error)
        low_mask = np.concatenate(all_low_masks)

        x_lo = float(self._x_min.value())
        x_hi = float(self._x_max.value())
        if self._is_frequency_plot_panel():
            x_lo = self._convert_frequency_control_value_to_axis_limit(x_lo)
            x_hi = self._convert_frequency_control_value_to_axis_limit(x_hi)
        lo, hi = (x_lo, x_hi) if x_lo <= x_hi else (x_hi, x_lo)

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

        return self._auto_y_limits_from_arrays(asymmetry, error, mask)

    def _low_count_mask_for_dataset(
        self,
        dataset: MuonDataset,
        *,
        source_dataset: MuonDataset | None = None,
    ) -> np.ndarray:
        """Return mask of low-count bins (plotted gray) for *dataset* points."""
        time = np.asarray(dataset.time, dtype=float)
        mask = np.zeros_like(time, dtype=bool)
        if (
            self._is_frequency_plot_panel()
            or self._is_frequency_domain_dataset(dataset)
            or self._is_frequency_domain_dataset(source_dataset)
        ):
            return mask

        low_confidence = self._low_confidence_mask_for_dataset(
            dataset,
            source_dataset=source_dataset,
        )
        if low_confidence is not None and low_confidence.shape == mask.shape:
            mask |= low_confidence

        run = dataset.run
        if run is None or not isinstance(getattr(run, "grouping", None), dict):
            return mask

        grouping = run.grouping
        first_good = grouping.get("first_good_bin")
        last_good = grouping.get("last_good_bin")
        if first_good is None or last_good is None:
            return mask

        histograms = getattr(run, "histograms", None)
        axis = np.asarray([], dtype=float)
        if histograms:
            hist0 = histograms[0]
            axis = np.asarray(hist0.time_axis, dtype=float)

        try:
            lo_idx = max(0, int(first_good))
            hi_idx = int(last_good)
        except (TypeError, ValueError):
            return mask

        if lo_idx > hi_idx:
            return mask

        # Prefer histogram time-axis boundaries when available. This keeps the
        # masking consistent for rebinned analysis datasets.
        if axis.size > 0:
            hi_axis_idx = min(hi_idx, axis.size - 1)
            if lo_idx > hi_axis_idx:
                return mask

            good_t_min = float(axis[lo_idx])
            good_t_max = float(axis[hi_axis_idx])
            if good_t_min > good_t_max:
                good_t_min, good_t_max = good_t_max, good_t_min
            tol = 1e-12
            if axis.size >= 2:
                step = float(np.nanmedian(np.diff(axis)))
                if np.isfinite(step) and step != 0.0:
                    tol = max(1e-12, abs(step) * 1e-6)
            mask |= (time < (good_t_min - tol)) | (time > (good_t_max + tol))
            return mask

        # Fallback for datasets that carry grouping metadata but no raw
        # histograms. Use the source, pre-rebinning time axis when available.
        reference_time = np.asarray(
            source_dataset.time if source_dataset is not None else dataset.time,
            dtype=float,
        )
        if reference_time.size == 0:
            return mask

        max_idx = reference_time.size - 1
        if lo_idx > max_idx:
            return mask
        hi_ref_idx = min(hi_idx, max_idx)
        if lo_idx > hi_ref_idx:
            return mask

        good_t_min = float(reference_time[lo_idx])
        good_t_max = float(reference_time[hi_ref_idx])
        if good_t_min > good_t_max:
            good_t_min, good_t_max = good_t_max, good_t_min

        tol = 1e-12
        if reference_time.size >= 2:
            step = float(np.nanmedian(np.diff(reference_time)))
            if np.isfinite(step) and step != 0.0:
                tol = max(1e-12, abs(step) * 1e-6)

        mask |= (time < (good_t_min - tol)) | (time > (good_t_max + tol))

        return mask

    def _low_confidence_mask_for_dataset(
        self,
        dataset: MuonDataset,
        *,
        source_dataset: MuonDataset | None = None,
    ) -> np.ndarray | None:
        """Return low-confidence mask from grouped-count reliability checks.

        Historical behavior marks saturated ±100% bins and bins with
        non-positive grouped denominator as low-confidence, rendered in gray.
        """
        reference_dataset = source_dataset if source_dataset is not None else dataset
        reference_metadata = getattr(reference_dataset, "metadata", None)
        analysis_metadata = getattr(dataset, "metadata", None)
        is_grouped_time_domain = bool(
            isinstance(reference_metadata, dict)
            and reference_metadata.get("grouped_time_domain")
            or isinstance(analysis_metadata, dict)
            and analysis_metadata.get("grouped_time_domain")
        )
        reference_asym = np.asarray(reference_dataset.asymmetry, dtype=float)
        if reference_asym.size == 0:
            return np.zeros_like(dataset.time, dtype=bool)

        saturated = np.zeros_like(reference_asym, dtype=bool)
        if not is_grouped_time_domain:
            saturated = (np.abs(reference_asym) > 100.0) | np.isclose(
                np.abs(reference_asym),
                100.0,
                atol=1e-12,
            )

        run = reference_dataset.run
        if (
            run is None
            or not run.histograms
            or not isinstance(getattr(run, "grouping", None), dict)
        ):
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        grouping = run.grouping
        groups = grouping.get("groups")
        if not isinstance(groups, dict):
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        # The denominator-reliability reduction below sums RAW grouped counts at
        # fixed binning. Under variable / constant-error binning the displayed
        # asymmetry is summed onto wider edges, so a per-raw-bin denominator no
        # longer maps to the plotted bins; the saturation check above (computed
        # on the displayed asymmetry) already flags unreliable bins there. Skip
        # the raw-bin reduction explicitly rather than letting it self-disable
        # via a silent array-shape mismatch.
        binning_mode, _, _ = resolve_binning_mode(grouping)
        if binning_mode != "fixed":
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        # Reduce the forward/backward groups through the SAME core chokepoint the
        # reduction uses (group_forward_backward: resolves groups via the
        # exclusion-aware resolver, aligns to a common t0, sums, clamps alpha),
        # so the mask cannot drift from the engine's grouping/exclusion/alpha
        # handling. It raises when groups are undefined or empty after exclusion.
        try:
            fb = group_forward_backward(run.histograms, grouping)
        except ValueError:
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )
        n_grouped = min(len(fb.forward), len(fb.backward))
        if n_grouped == 0:
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )
        forward = fb.forward[:n_grouped]
        backward = fb.backward[:n_grouped]
        alpha = fb.alpha
        common_t0 = fb.common_t0

        if bool(grouping.get("background_correction", False)):
            run_metadata = getattr(run, "metadata", None)
            metadata = run_metadata if isinstance(run_metadata, dict) else {}
            source_file = str(
                getattr(run, "source_file", "") or reference_dataset.metadata.get("source_file", "")
            )
            # Mode-aware gate (matches the real reduction): apply whichever
            # background mode the grouping selects when it is available for this
            # source, instead of the old PSI/LEM-only range gate. (reference_run
            # needs externally resolved counts that this display-only mask does
            # not load, so apply_grouped_background_correction no-ops it here —
            # a minor difference in which bins render gray, never in the data.)
            mode = resolve_background_mode(grouping)
            available = available_background_modes(metadata=metadata, source_file=source_file)
            if mode in available:
                bin_width = float(run.histograms[0].bin_width) if run.histograms else 1.0
                facility = str(
                    metadata.get(
                        "facility",
                        reference_dataset.metadata.get(
                            "facility",
                            reference_dataset.metadata.get("instrument", ""),
                        ),
                    )
                )
                bkg_result = apply_grouped_background_correction(
                    forward,
                    backward,
                    grouping=grouping,
                    t0_bin=common_t0,
                    bin_width_us=bin_width,
                    facility=facility,
                )
                if bkg_result.applied:
                    forward = bkg_result.forward
                    backward = bkg_result.backward

        denominator = np.asarray(forward + alpha * backward, dtype=float)
        if denominator.size == 0:
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        lo_default = 0
        hi_default = denominator.size - 1
        try:
            lo = int(grouping.get("first_good_bin", lo_default))
        except (TypeError, ValueError):
            lo = lo_default
        try:
            hi = int(grouping.get("last_good_bin", hi_default))
        except (TypeError, ValueError):
            hi = hi_default

        lo = max(0, min(lo, hi_default))
        hi = max(0, min(hi, hi_default))
        if lo > hi:
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        denominator = denominator[lo : hi + 1]
        reliable = denominator > 0.0
        source_low = saturated.copy()
        if reliable.shape == source_low.shape:
            source_low |= ~reliable

        return self._project_source_mask_to_analysis_dataset(
            source_mask=source_low,
            source_dataset=reference_dataset,
            analysis_dataset=dataset,
        )

    def _project_source_mask_to_analysis_dataset(
        self,
        *,
        source_mask: np.ndarray,
        source_dataset: MuonDataset,
        analysis_dataset: MuonDataset,
    ) -> np.ndarray:
        """Project source-bin boolean mask onto the plotted analysis dataset."""
        source_mask = np.asarray(source_mask, dtype=bool)
        source_time = np.asarray(source_dataset.time, dtype=float)
        target_time = np.asarray(analysis_dataset.time, dtype=float)

        if target_time.size == 0:
            return np.zeros(0, dtype=bool)
        if source_mask.size == 0 or source_time.size == 0:
            return np.zeros_like(target_time, dtype=bool)

        if source_mask.size != source_time.size:
            n = min(source_mask.size, source_time.size)
            source_mask = source_mask[:n]
            source_time = source_time[:n]
            if n == 0:
                return np.zeros_like(target_time, dtype=bool)

        if source_time.size == target_time.size:
            return source_mask.copy()

        # The RRF display slices the filter-edge region off the analysis
        # dataset; the recorded trim restores the cheap projection paths.
        trim = (
            analysis_dataset.metadata.get("rrf_trim")
            if isinstance(analysis_dataset.metadata, dict)
            else None
        )
        if isinstance(trim, (list, tuple)) and len(trim) == 3:
            start, stop, pre_size = (int(v) for v in trim)
            if stop - start == target_time.size and 0 <= start <= stop <= pre_size:
                if source_mask.size == pre_size:
                    return source_mask[start:stop].copy()
                pre_bunch = int(self._bunch_factor.value()) if hasattr(self, "_bunch_factor") else 1
                if pre_bunch > 1 and source_mask.size >= pre_size * pre_bunch:
                    folded = (
                        source_mask[: pre_size * pre_bunch].reshape(pre_size, pre_bunch).any(axis=1)
                    )
                    return folded[start:stop]

        bunch_factor = int(self._bunch_factor.value()) if hasattr(self, "_bunch_factor") else 1
        if bunch_factor > 1 and source_time.size >= target_time.size:
            trimmed = (source_time.size // bunch_factor) * bunch_factor
            if trimmed > 0 and (trimmed // bunch_factor) == target_time.size:
                return source_mask[:trimmed].reshape(target_time.size, bunch_factor).any(axis=1)

        if source_time.size == 1:
            return np.full(target_time.size, bool(source_mask[0]), dtype=bool)

        edges = np.empty(target_time.size + 1, dtype=float)
        if target_time.size == 1:
            edges[0] = -np.inf
            edges[1] = np.inf
        else:
            edges[1:-1] = 0.5 * (target_time[:-1] + target_time[1:])
            edges[0] = -np.inf
            edges[-1] = np.inf

        projected = np.zeros(target_time.size, dtype=bool)
        for i in range(target_time.size):
            if i == target_time.size - 1:
                in_bucket = (source_time >= edges[i]) & (source_time <= edges[i + 1])
            else:
                in_bucket = (source_time >= edges[i]) & (source_time < edges[i + 1])
            if np.any(in_bucket):
                projected[i] = bool(np.any(source_mask[in_bucket]))

        return projected

    def _on_bunch_changed(self) -> None:
        """Re-plot and refresh fit inputs when the bunch factor changes."""
        self._redraw_current_view()
        self.bunch_factor_changed.emit(self._bunch_factor.value())

    def _set_fit_range(
        self,
        x_min: float,
        x_max: float,
        *,
        emit_signal: bool,
        redraw: bool,
    ) -> None:
        """Set fit range with ordering and optional signaling."""
        if self._is_frequency_plot_panel():
            return
        lo = float(min(x_min, x_max))
        hi = float(max(x_min, x_max))

        self._fit_x_min = lo
        self._fit_x_max = hi

        # Widen the X-axis view to keep fit bounds that fall outside the plotted
        # data extent visible. Only ever expand the view (never shrink it), so an
        # in-data fit range leaves the user's current limits untouched.
        view_min = min(float(self._x_min.value()), lo)
        view_max = max(float(self._x_max.value()), hi)
        if view_min != float(self._x_min.value()) or view_max != float(self._x_max.value()):
            self._set_limit_field_value(self._x_min, view_min)
            self._set_limit_field_value(self._x_max, view_max)
            if redraw:
                self._apply_limits()

        if redraw:
            self._draw_fit_range_artists()
            self._canvas.draw_idle()

        if emit_signal:
            self.fit_range_changed.emit(self._fit_x_min, self._fit_x_max)

    # ── spectral-moments overlay (frequency panel) ──────────────────────────

    def current_frequency_dataset(self) -> MuonDataset | None:
        """Return the active frequency-domain dataset, or ``None``."""
        # ``_current_datasets`` and the overlay state only exist when matplotlib
        # is available (they are set up in the canvas try-block); guard so a
        # headless / no-mpl panel degrades to "no spectrum" instead of raising.
        if not self._has_mpl or not self._is_frequency_plot_panel():
            return None
        if self._current_datasets:
            return self._current_datasets[0]
        return self._current_dataset

    def active_spectrum_for_moments(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str] | None:
        """W15 accessor: ``(x, amplitude, errors, x_unit)`` for the active spectrum.

        ``x`` is the canonical **absolute MHz** axis (the unit-invariant the
        moments feature works in) and ``x_unit`` is ``"mhz"``; the caller converts
        to its chosen field unit. Returns ``None`` when no spectrum is shown or the
        active spectrum is a correlation axis (a hyperfine coupling, not a field).
        """
        if not self._is_frequency_plot_panel() or self._frequency_axis_is_correlation:
            return None
        ds = self.current_frequency_dataset()
        if ds is None:
            return None
        x = np.asarray(getattr(ds, "time", []), dtype=float)
        y = np.asarray(getattr(ds, "asymmetry", []), dtype=float)
        if x.size == 0 or x.shape != y.shape:
            return None
        err = getattr(ds, "error", None)
        err_arr = None if err is None else np.asarray(err, dtype=float)
        if err_arr is not None and err_arr.shape != x.shape:
            err_arr = None
        return x, y, err_arr, "mhz"

    def set_moments_overlay(
        self,
        *,
        window_mhz: tuple[float, float] | None,
        cutoff_fraction: float,
        peak_amplitude: float | None,
        visible: bool,
    ) -> None:
        """Show/update the moment window + cutoff line on the spectrum plot.

        *window_mhz* is the analysis window in canonical absolute MHz; the cutoff
        line is drawn at ``peak_amplitude · cutoff_fraction``. Drawn only on the
        frequency panel.
        """
        if not self._has_mpl or not self._is_frequency_plot_panel():
            return
        self._moments_window_mhz = (
            None if window_mhz is None else (float(window_mhz[0]), float(window_mhz[1]))
        )
        self._moments_cutoff_fraction = float(cutoff_fraction)
        self._moments_peak_amp = None if peak_amplitude is None else float(peak_amplitude)
        self._moments_overlay_visible = bool(visible) and self._moments_window_mhz is not None
        self._draw_moments_artists()
        if self._has_mpl:
            self._canvas.draw_idle()

    def clear_moments_overlay(self) -> None:
        """Hide the moment window/cutoff overlay."""
        if not self._has_mpl:
            return
        self._moments_overlay_visible = False
        self._clear_moments_artists()
        self._canvas.draw_idle()

    def _clear_moments_artists(self) -> None:
        for artist in (*self._moments_span_artists, *self._moments_cutoff_artists):
            try:
                artist.remove()
            except NotImplementedError:
                continue
            except Exception:
                continue
        self._moments_span_artists = []
        self._moments_cutoff_artists = []

    def _moments_window_display(self) -> tuple[float, float] | None:
        """Return the moment window converted to the current display axis."""
        if self._moments_window_mhz is None:
            return None
        unit = self._current_frequency_x_unit
        relative = self._frequency_axis_relative_to_reference
        lo = self._convert_canonical_mhz_to_display_limit(
            self._moments_window_mhz[0], unit=unit, relative=relative
        )
        hi = self._convert_canonical_mhz_to_display_limit(
            self._moments_window_mhz[1], unit=unit, relative=relative
        )
        return (min(lo, hi), max(lo, hi))

    def _draw_moments_artists(self) -> None:
        """Draw the moment window span + handles and the cutoff line."""
        if not self._has_mpl or not self._is_frequency_plot_panel():
            return
        self._clear_moments_artists()
        if not self._moments_overlay_visible or self._frequency_axis_is_correlation:
            return
        window = self._moments_window_display()
        if window is None:
            return
        # Reuse the established fit-range span grammar for visual consistency.
        span, left_line, right_line = draw_fit_range_span(self._ax, window[0], window[1])
        self._moments_span_artists.extend([span, left_line, right_line])
        if self._moments_peak_amp is not None and self._moments_cutoff_fraction > 0.0:
            level = self._moments_peak_amp * self._moments_cutoff_fraction
            cutoff_line = self._ax.axhline(
                level, color=right_line.get_color(), alpha=0.45, linestyle=":", linewidth=1.2
            )
            self._moments_cutoff_artists.append(cutoff_line)

    def _detect_moments_handle_hit(self, event) -> str | None:
        """Return ``"min"``/``"max"``/``"cutoff"`` for a moment handle near the cursor."""
        if (
            not self._moments_overlay_visible
            or self._moments_window_mhz is None
            or event.inaxes is not self._ax
            or event.x is None
            or event.y is None
        ):
            return None
        window = self._moments_window_display()
        if window is None:
            return None
        handle = nearest_handle(
            self._ax,
            [(window[0], "min"), (window[1], "max")],
            event.x,
            tolerance_px=8.0,
        )
        if handle is not None:
            return handle
        if self._moments_peak_amp is not None and self._moments_cutoff_fraction > 0.0:
            level = self._moments_peak_amp * self._moments_cutoff_fraction
            level_px = self._ax.transData.transform((0.0, level))[1]
            if abs(event.y - level_px) <= 6.0:
                return "cutoff"
        return None

    def _drag_moments_handle(self, event) -> None:
        """Apply a moment-handle drag, updating state and emitting the change."""
        if self._active_moments_handle == "cutoff":
            if (
                event.ydata is None
                or self._moments_peak_amp is None
                or self._moments_peak_amp <= 0.0
            ):
                return
            fraction = float(min(max(event.ydata / self._moments_peak_amp, 0.0), 0.99))
            self._moments_cutoff_fraction = fraction
            self._draw_moments_artists()
            self._canvas.draw_idle()
            self.moments_cutoff_changed.emit(fraction)
            return
        if event.xdata is None or self._moments_window_mhz is None:
            return
        canonical = self._convert_display_limit_to_canonical_mhz(
            float(event.xdata),
            unit=self._current_frequency_x_unit,
            relative=self._frequency_axis_relative_to_reference,
        )
        lo, hi = self._moments_window_mhz
        if self._active_moments_handle == "min":
            lo = canonical
        else:
            hi = canonical
        self._moments_window_mhz = (lo, hi)
        self._draw_moments_artists()
        self._canvas.draw_idle()
        self.moments_window_changed.emit(min(lo, hi), max(lo, hi))

    def _draw_fit_range_artists(self) -> None:
        """Draw highlight and edge handles for the selected fit range."""
        if not self._has_mpl:
            return
        # The frequency panel has no time-domain fit range, but it does carry the
        # moments overlay; redraw it here so it survives every plot rebuild.
        self._draw_moments_artists()
        if self._is_frequency_plot_panel():
            return
        self._clear_fit_range_artists()

        if self._fit_x_min is None or self._fit_x_max is None:
            return

        axes = self._fit_range_axes()
        if not axes:
            return

        for axis in axes:
            span, left_line, right_line = draw_fit_range_span(
                axis, self._fit_x_min, self._fit_x_max
            )
            self._fit_span_artists.append(span)
            self._fit_min_handles.append(left_line)
            self._fit_max_handles.append(right_line)

    def _detect_handle_hit(self, event) -> str | None:
        """Return which fit handle (min/max) was clicked, if any."""
        if (
            self._fit_x_min is None
            or self._fit_x_max is None
            or event.inaxes is None
            or event.x is None
            or event.y is None
        ):
            return None

        hit_axis = None
        for axis in self._fit_range_axes():
            if event.inaxes is axis:
                hit_axis = axis
                break
        if hit_axis is None:
            return None

        return nearest_handle(
            hit_axis,
            [(self._fit_x_min, "min"), (self._fit_x_max, "max")],
            event.x,
            tolerance_px=8.0,
        )

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
        if self._current_navigation_mode() != "none":
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
            self._active_fit_axis = event.inaxes
            self._drag_started = False
            return

        moments_handle = self._detect_moments_handle_hit(event)
        if moments_handle is not None:
            self._active_moments_handle = moments_handle
            self._drag_started = False
            return

        ann_idx = self._detect_annotation_hit(event)
        if ann_idx is not None:
            self._active_annotation_idx = ann_idx
            self._annotation_drag_started = False
        elif self._subplot_axes_by_polarization:
            # A plain click inside a stacked subplot makes it the fit target.
            projection = self._subplot_projection_at_event(event)
            if projection is not None:
                self.set_fit_target_projection(projection)

    def _on_canvas_motion_notify(self, event) -> None:
        """Drag the active fit-range handle while the mouse moves."""
        if self._current_navigation_mode() != "none":
            return

        if self._active_fit_handle is not None and event.xdata is not None:
            if not any(event.inaxes is axis for axis in self._fit_range_axes()):
                return
            self._drag_started = True
            if self._active_fit_handle == "min":
                self._set_fit_range(event.xdata, self._fit_x_max, emit_signal=True, redraw=True)
            else:
                self._set_fit_range(self._fit_x_min, event.xdata, emit_signal=True, redraw=True)

        if self._active_moments_handle is not None and event.inaxes is self._ax:
            self._drag_started = True
            self._drag_moments_handle(event)

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

        # Emit cursor readout for the status bar (no-op during drags).
        if (
            self._active_fit_handle is None
            and self._active_annotation_idx is None
            and self._active_moments_handle is None
            and event.inaxes == self._ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            self.cursor_coords_changed.emit(
                self._build_cursor_readout(float(event.xdata), float(event.ydata))
            )
        else:
            self.cursor_coords_changed.emit(None)

    def _cursor_snap_arrays(self):
        """Cached (t, y, err) for snapping, or None on multi-curve views.

        Snapping is only well-defined when the cached arrays correspond to the
        main axis. On stacked grouped/vector subplots the cache holds the last
        subplot's data, so we decline to snap and report the raw cursor instead.
        """
        if getattr(self, "_grouped_time_subplot_datasets", None) or getattr(
            self, "_vector_subplot_datasets", None
        ):
            return None
        t = self._last_plot_time
        y = self._last_plot_asymmetry
        if t is None or y is None or len(t) == 0:
            return None
        return t, y, self._last_plot_error

    def _build_cursor_readout(self, xdata: float, ydata: float) -> dict:
        """Build the status-bar cursor payload, snapping to the nearest point.

        Adds the spectrum-reading readouts (S/N, a 3-point parabolic peak, and
        the windowed average over the visible x-range) when a single curve is in
        view; falls back to the raw coordinate otherwise.
        """
        arrays = self._cursor_snap_arrays()
        if arrays is None:
            return {"x": xdata, "y": ydata, "snapped": False}

        t, y, e = arrays
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        idx = int(np.argmin(np.abs(t - xdata)))
        tx = float(t[idx])
        ty = float(y[idx])
        err = None
        if e is not None and idx < len(e):
            try:
                err = float(e[idx])
            except (TypeError, ValueError):
                err = None
        snr = None
        if err is not None and np.isfinite(err) and err != 0.0:
            snr = abs(ty / err)

        peak = None
        if 1 <= idx < len(t) - 1:
            peak = parabolic_peak(t[idx - 1 : idx + 2], y[idx - 1 : idx + 2])

        window = None
        if not self._is_frequency_plot_panel():
            lo = float(self._x_min.value())
            hi = float(self._x_max.value())
            err_arr = np.asarray(e, dtype=float) if e is not None else np.zeros_like(y)
            try:
                mean, mean_err = integrate_curve(
                    t, y, err_arr, t_min=min(lo, hi), t_max=max(lo, hi)
                )
                n = int(np.count_nonzero((t >= min(lo, hi)) & (t <= max(lo, hi))))
                window = (float(mean), float(mean_err), n)
            except (ValueError, ZeroDivisionError):
                window = None

        return {
            "x": tx,
            "y": ty,
            "err": err,
            "snr": snr,
            "peak": peak,
            "window": window,
            "snapped": True,
        }

    def _on_canvas_button_release(self, event) -> None:
        """End drag and open numeric editor on click without drag."""
        if self._current_navigation_mode() != "none":
            return

        if self._active_fit_handle is not None:
            handle = self._active_fit_handle
            was_drag = self._drag_started

            self._active_fit_handle = None
            self._active_fit_axis = None
            self._drag_started = False

            if not was_drag and event.button == 1:
                self._prompt_handle_value_edit(handle)

        if self._active_moments_handle is not None:
            self._active_moments_handle = None
            self._drag_started = False
            return

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
        run_number: int | None = None,
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
        run_number : int, optional
            Run the fit was computed for. In a multi-run overlay stacked view
            ``_current_dataset`` points at the first projection's *last* run, not
            the fitted (selected) run, so the caller passes the fitted run
            explicitly to keep the overlay key and the persisted single-fit slot
            on the same run. Defaults to inferring it from ``_current_dataset``.
        """
        if not self._has_mpl:
            return

        # Store fit curve data for persistence across redraws (single fit)
        self._fit_curve = (t_fit, y_fit, label)
        explicit_run = run_number
        run_number = None
        if self._current_dataset is not None:
            try:
                run_number = int(self._current_dataset.run_number)
            except (TypeError, ValueError):
                run_number = None
        # The fitted run is authoritative when supplied: source it from the one
        # place that knows it (the caller's selected dataset) so the curve is
        # keyed under the same run the slot is recorded against.
        if explicit_run is not None:
            try:
                run_number = int(explicit_run)
            except (TypeError, ValueError):
                pass
        # Derive the axis from the dataset that actually matches the fitted run,
        # not whichever one _current_dataset points at — in a multi-run overlay
        # that is the last overlaid run, possibly a different projection — so the
        # (run, axis) key is always self-consistent.
        axis_source = self._current_dataset
        if run_number is not None:
            for dataset in self._current_datasets or ():
                try:
                    if int(dataset.run_number) == run_number:
                        axis_source = dataset
                        break
                except (TypeError, ValueError):
                    continue
        axis_key = self._axis_key_for_dataset(axis_source) if axis_source is not None else None
        # In the stacked view the fit belongs to the SELECTED subplot, not the
        # first projection that _current_dataset happens to point at — otherwise
        # every fit would draw on the first subplot regardless of the target.
        if self._subplot_axes_by_polarization:
            target = self.fit_target_projection()
            if target is not None:
                axis_key = target
        self._fit_curve_run_number = run_number

        if run_number is not None:
            self._fit_curves[run_number] = (t_fit, y_fit, label)
            self._fit_curves_by_key[(run_number, axis_key)] = (t_fit, y_fit, label)
            self._fit_components_by_run[run_number] = list(component_curves or [])
            self._fit_components_by_key[(run_number, axis_key)] = list(component_curves or [])
            if fit_result is not None or fit_function:
                self._store_fit_metadata(
                    run_number,
                    fit_result,
                    fit_function=fit_function,
                    axis_key=axis_key,
                )

        self._fit_components = list(component_curves or [])

        self._update_export_enabled()

        if self._subplot_axes_by_polarization and self._vector_subplot_datasets:
            # Stacked multi-subplot view: re-render the subplots so the fit
            # overlays the target projection's subplot (drawn per-axis from
            # _fit_curves_by_key) without dropping the other projections.
            self.plot_vector_subplots(self._vector_subplot_datasets)
        elif self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)
        else:
            self._ax.plot(t_fit, y_fit, "-", color=tokens.PLOT_FIT, linewidth=2, label=label)
            style_legend(self._ax.legend())
            self._canvas.draw()

    def set_global_fits(self, fit_curves_dict: dict) -> None:
        """Set fit curves from global fitting.

        Parameters
        ----------
        fit_curves_dict : dict
            Dictionary mapping run_number -> (t_fit, y_fit, label, component_curves),
            (t_fit, y_fit, label, component_curves, fit_result), or
            (t_fit, y_fit, label, component_curves, fit_result, fit_function), or
            (t_fit, y_fit, label, component_curves, fit_result, fit_function, axis_key).
        """
        if not self._has_mpl:
            return

        # Update fit curves, preserving results from other groups
        for run_number, payload in fit_curves_dict.items():
            axis_key = None
            if len(payload) >= 6:
                t_fit, y_fit, label, component_curves, fit_result, fit_function = payload[:6]
                if len(payload) >= 7:
                    axis_key = self._axis_canonical_key(payload[6])
                    if axis_key == "ALL":
                        axis_key = None
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
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            self._fit_curves[run_key] = (t_fit, y_fit, label)
            self._fit_curves_by_key[(run_key, axis_key)] = (t_fit, y_fit, label)
            self._fit_components_by_run[run_key] = list(component_curves or [])
            self._fit_components_by_key[(run_key, axis_key)] = list(component_curves or [])
            if fit_result is not None or fit_function:
                self._store_fit_metadata(
                    run_key,
                    fit_result,
                    fit_function=fit_function,
                    axis_key=axis_key,
                )
        # Clear single fit curve
        self._fit_curve = None
        self._fit_curve_run_number = None
        self._fit_components = None

        self._update_export_enabled()

        # Redraw current view while preserving multi-selection overlays.
        self._redraw_current_view()

    def clear(self, *, message: str | None = None) -> None:
        """Clear the plot and reset stored data.

        When *message* is given, draw it as a centred grey placeholder over the
        (axis-off) plot area — the empty-state pattern used by
        :meth:`asymmetry.gui.panels.alc_panel.AlcPanel.clear`. The frequency
        panel passes a message to prompt the user to compute a spectrum (an FFT
        is computed on demand, never automatically); every other caller leaves
        *message* ``None`` and gets an unchanged blank plot.
        """
        if self._has_mpl:
            self._set_canvas_minimum_height_for_axes(1)
            self._set_navigation_mode("none")
            self._set_alpha_label(None)
            self.set_projections([])
            self._ax.clear()
            style_axes(self._ax)
            if message:
                draw_empty_state_message(self._ax, message)
            self._canvas.draw()
            self._current_dataset = None
            self._current_datasets = []
            self._update_plot_header()
            self._fit_curve = None
            self._fit_curve_run_number = None
            self._fit_curves = {}
            self._fit_curves_by_key = {}
            self._fit_components = None
            self._fit_components_by_run = {}
            self._fit_components_by_key = {}
            self._fit_metadata = {}
            self._fit_metadata_by_key = {}
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
            self._fit_span_artists = []
            self._fit_min_handles = []
            self._fit_max_handles = []
            self._active_fit_axis = None
            self._current_polarization_axis = None
            self._y_limits_by_polarization = {}
            self._subplot_axes_by_polarization = {}
            self._vector_subplot_datasets = {}
            if self._is_frequency_plot_panel():
                self._frequency_reference_mhz = None
                self._apply_axis_labels(*self._default_axis_labels())
            self._update_export_enabled()

    def resizeEvent(self, event) -> None:
        """Keep the canvas width aligned with the viewport during grouped scrolling."""
        super().resizeEvent(event)
        if not getattr(self, "_has_mpl", False):
            return
        # Use _stacked_axis_count rather than len(_subplot_axes_by_polarization):
        # the dict is briefly empty while plot_vector_subplots/plot_grouped_time_domain_subplots
        # rebuild axes, and a resize event in that window would otherwise reset the
        # scroll area to single-axis mode, which is the root cause of the scrollbar
        # disappearing on first render in the frozen macOS app.
        axis_count = max(1, getattr(self, "_stacked_axis_count", 1))
        target_height = max(self._default_canvas_min_height, int(self._canvas.minimumHeight()))
        self._sync_canvas_scroll_geometry(axis_count=axis_count, target_height=target_height)
        self._canvas.draw_idle()

    def clear_fit(self) -> None:
        """Clear all fit curves and redraw the plot."""
        if not self._has_mpl:
            return

        self._fit_curve = None
        self._fit_curve_run_number = None
        self._fit_curves = {}
        self._fit_curves_by_key = {}
        self._fit_components = None
        self._fit_components_by_run = {}
        self._fit_components_by_key = {}
        self._fit_metadata = {}
        self._fit_metadata_by_key = {}
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

        removed_curve_keys = [key for key in self._fit_curves_by_key if key[0] in normalized_runs]
        removed_component_keys = [
            key for key in self._fit_components_by_key if key[0] in normalized_runs
        ]
        removed_metadata_keys = [
            key for key in self._fit_metadata_by_key if key[0] in normalized_runs
        ]

        for key in removed_curve_keys:
            self._fit_curves_by_key.pop(key, None)
        for key in removed_component_keys:
            self._fit_components_by_key.pop(key, None)
        for key in removed_metadata_keys:
            self._fit_metadata_by_key.pop(key, None)

        if self._fit_curve_run_number in normalized_runs:
            self._fit_curve = None
            self._fit_curve_run_number = None
            self._fit_components = None
            removed += 1

        if removed > 0:
            self._update_export_enabled()
            self._redraw_current_view()

        return removed

    def _store_fit_metadata(
        self,
        run_number: int,
        fit_result: object | None,
        fit_function: str | None = None,
        axis_key: str | None = None,
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
        self._fit_metadata_by_key[(int(run_number), axis_key)] = meta

    def _update_export_enabled(self) -> None:
        """Enable export controls when there is plotted data to export."""
        has_data = bool(self._current_datasets)
        self._export_gle_btn.setEnabled(has_data)
        self._gle_format_combo.setEnabled(has_data)

    @staticmethod
    def _safe_file_token(value: str) -> str:
        """Sanitize a string for use in a filename."""
        token = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip()
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

    def get_current_plot_export_data(
        self, datasets: list[MuonDataset] | None = None
    ) -> list[dict] | None:
        """Return export payloads for all displayed datasets.

        Returns a list of dicts (one per displayed dataset), or *None* when
        nothing is available to export.
        """
        target_datasets = list(datasets) if datasets is not None else list(self._current_datasets)
        if not target_datasets:
            return None

        payloads: list[dict] = []
        for dataset in target_datasets:
            analysis = rrf_display_dataset(self, self.get_analysis_dataset(dataset))
            if analysis is None:
                continue

            grouping = None
            run = getattr(dataset, "run", None)
            if run is not None and isinstance(getattr(run, "grouping", None), dict):
                grouping = dict(run.grouping)
            elif isinstance(dataset.metadata.get("grouping"), dict):
                grouping = dict(dataset.metadata["grouping"])

            run_metadata: dict[str, object] = {}
            if run is not None and isinstance(getattr(run, "metadata", None), dict):
                run_metadata.update(run.metadata)
            if isinstance(dataset.metadata, dict):
                run_metadata.update(dataset.metadata)
            if isinstance(analysis.metadata, dict) and analysis.metadata.get("rrf_frame"):
                run_metadata["rrf_frame"] = analysis.metadata["rrf_frame"]

            histogram_info: dict[str, object] | None = None
            if run is not None and getattr(run, "histograms", None):
                histograms = list(run.histograms)
                if histograms:
                    h0 = histograms[0]
                    total_events = float(np.sum([np.sum(h.counts) for h in histograms]))
                    histogram_info = {
                        "count": len(histograms),
                        "n_bins": int(h0.n_bins),
                        "bin_width_us": float(h0.bin_width),
                        "t0_bins": [int(h.t0_bin) for h in histograms],
                        "events_total": total_events,
                    }

                    grouped_total = good_event_count(histograms, grouping)
                    if grouped_total is not None and grouped_total > 0:
                        histogram_info["events_grouped"] = grouped_total

            rn = dataset.run_number
            fit_data = self._fit_curve_for_dataset(dataset)
            fit_data = rrf_display_fit_curve(self, fit_data, analysis)

            t_fit = None
            y_fit = None
            fit_label = "Fit"
            if fit_data is not None:
                t_fit, y_fit, fit_label = fit_data
            component_data = self._fit_components_for_dataset(dataset)
            if t_fit is not None and component_data:
                wrapped_components = []
                for name, y_vals in component_data:
                    shown = rrf_display_fit_curve(self, (t_fit, y_vals, name), analysis)
                    wrapped_components.append((name, shown[1] if shown is not None else y_vals))
                component_data = wrapped_components

            label_text = self._dataset_label_for(dataset)

            payloads.append(
                {
                    "run_number": rn,
                    "label": label_text,
                    "data": {
                        "t": analysis.time,
                        "y": analysis.asymmetry,
                        "err": analysis.error,
                    },
                    "fit": {"t": t_fit, "y": y_fit, "label": fit_label}
                    if t_fit is not None and y_fit is not None
                    else None,
                    "components": [{"name": name, "y": y_vals} for name, y_vals in component_data],
                    "fit_metadata": self._fit_metadata_for_dataset(dataset),
                    "grouping": grouping,
                    "run_metadata": run_metadata,
                    "histogram_info": histogram_info,
                }
            )

        if not payloads:
            return None

        # Append annotations (shared across all datasets in this view)
        annotations = [
            {"x": ann["x"], "y": ann["y"], "text": ann["text"]} for ann in self._annotations
        ]
        for p in payloads:
            p["annotations"] = annotations

        return payloads

    def _export_axis_suffix(self, axis_key: str | None) -> str:
        """Return filename-safe suffix for per-axis exports."""
        if axis_key in {"P_x", "P_y", "P_z"}:
            return axis_key.lower().replace("_", "")
        return "main"

    def _export_axis_ylabel(self, axis_key: str | None) -> str:
        """Return plain-text y-axis label for plot exports."""
        if axis_key in {"P_x", "P_y", "P_z"}:
            suffix = axis_key.split("_", 1)[1]
            return f"a_0 P_{{{suffix}}}(t) (%)"
        return "Asymmetry (%)"

    def _plot_export_payloads_on_axis(
        self,
        ax,
        payloads: list[dict],
        *,
        axis_key: str | None,
        written_files: list[Path],
        dat_writes: list[tuple[Path, dict, object]],
        gle_path: Path,
        colors: list[str],
        show_legend: bool,
    ) -> None:
        """Draw export payloads on a provided axis with axis-specific naming."""
        is_multi = len(payloads) > 1
        axis_suffix = self._export_axis_suffix(axis_key)

        for i, payload in enumerate(payloads):
            label_text = payload.get("label", f"dataset_{i}")
            safe_label = self._sanitize_gle_text(
                label_text,
                fallback=f"Run {payload.get('run_number', i)}",
            )
            token = self._safe_file_token(f"{label_text}_{axis_suffix}")
            data_color = colors[i % len(colors)] if is_multi else "black"
            fit_color = data_color if is_multi else "red"
            suppress_subplot_labels = axis_key in {"P_x", "P_y", "P_z"} and not show_legend
            data_label = None if suppress_subplot_labels else safe_label

            data = payload.get("data") or {}
            fit = payload.get("fit") or {}
            t_data = data.get("t")
            y_data = data.get("y")
            y_err = data.get("err")
            t_fit = fit.get("t")
            y_fit = fit.get("y")

            dat_path = gle_path.parent / f"{token}.dat"
            if t_data is not None and y_data is not None:
                dat_writes.append((dat_path, payload, label_text))
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
                    label=data_label,
                    data_name=token,
                )

            fit_path = gle_path.parent / f"{token}.fit"
            if t_fit is not None and y_fit is not None:
                self._write_fit_file(fit_path, payload)
                written_files.append(fit_path)

                fit_label = fit.get("label", "Fit")
                if is_multi or suppress_subplot_labels:
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

        ax.set_ylabel(self._export_axis_ylabel(axis_key))
        if hasattr(ax, "set_xlim"):
            ax.set_xlim(float(self._x_min.value()), float(self._x_max.value()))

        if axis_key in self._y_limits_by_polarization and hasattr(ax, "set_ylim"):
            y0, y1 = self._y_limits_by_polarization[axis_key]
            ax.set_ylim(float(y0), float(y1))
        elif hasattr(ax, "set_ylim"):
            ax.set_ylim(float(self._y_min.value()), float(self._y_max.value()))

        if show_legend:
            style_legend(ax.legend(loc="best"))

    def _write_fit_file(
        self, fit_path: Path, payload: dict, *, x_range: tuple[float, float] | None = None
    ) -> None:
        """Write a .fit file with fit-curve data and metadata header.

        ``x_range`` optionally restricts the written rows to ``[lo, hi]``;
        ``None`` (the GLE-export default) writes the whole curve.
        """
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
            lo, hi = (None, None) if x_range is None else (min(x_range), max(x_range))
            for t_val, y_val in zip(t_fit, y_fit):
                tf = float(t_val)
                if lo is not None and (tf < lo or tf > hi):
                    continue
                f.write(f"{tf:.10g} {float(y_val):.10g}\n")

    def _write_data_file(
        self,
        dat_path: Path,
        payload: dict,
        *,
        label_text: object | None = None,
        x_range: tuple[float, float] | None = None,
    ) -> None:
        """Write a .dat file with spectra data and metadata header.

        ``x_range`` optionally restricts the written rows to ``[lo, hi]``
        (inclusive); ``None`` (the default, used by the GLE export) writes every
        point.
        """
        data = payload.get("data") or {}
        t_data = data.get("t")
        y_data = data.get("y")
        y_err = data.get("err")
        if t_data is None or y_data is None:
            return

        display_label = label_text if label_text is not None else payload.get("label", "dataset")
        run_metadata = (
            payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
        )
        histogram_info = (
            payload.get("histogram_info") if isinstance(payload.get("histogram_info"), dict) else {}
        )
        grouping = payload.get("grouping") if isinstance(payload.get("grouping"), dict) else {}

        def _fmt_float(value: object, decimals: int) -> str:
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return ""

        def _fmt_mev(value: object) -> str:
            try:
                return f"{float(value) / 1.0e6:.2f}"
            except (TypeError, ValueError):
                return ""

        def _safe_int(value: object) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        title = str(run_metadata.get("title", "") or "")
        comment = str(run_metadata.get("comment", "") or "")
        started = str(run_metadata.get("started", "") or "")
        stopped = str(run_metadata.get("stopped", "") or "")
        temperature = _fmt_float(run_metadata.get("temperature"), 3)
        temperature_label = str(
            run_metadata.get("temperature_label") or run_metadata.get("temperature_setpoint") or ""
        ).strip()
        field = _fmt_float(run_metadata.get("field"), 2)

        n_hist = _safe_int(histogram_info.get("count"))
        n_bins = _safe_int(histogram_info.get("n_bins"))
        bin_width_us = histogram_info.get("bin_width_us")
        bin_width_ps_text = ""
        bin_width_us_text = ""
        if bin_width_us is not None:
            bin_width_ps_text = _fmt_float(float(bin_width_us) * 1.0e6, 6)
            bin_width_us_text = _fmt_float(bin_width_us, 2)

        events_grouped = _fmt_mev(histogram_info.get("events_grouped"))
        events_raw = _fmt_mev(histogram_info.get("events_total"))

        groups_raw = grouping.get("groups")
        groups: dict[int, list[int]] = {}
        if isinstance(groups_raw, dict):
            for key, values in groups_raw.items():
                gid = _safe_int(key)
                if gid is None or not isinstance(values, list):
                    continue
                detectors: list[int] = []
                for v in values:
                    vv = _safe_int(v)
                    if vv is not None:
                        detectors.append(vv)
                if detectors:
                    groups[gid] = sorted(set(detectors))

        # Backward-compatible fallback for payloads that store detectors as
        # direct forward/backward lists rather than a groups mapping.
        if not groups:
            f_list = grouping.get("forward")
            b_list = grouping.get("backward")
            if isinstance(f_list, list):
                f_det = [v for v in (_safe_int(x) for x in f_list) if v is not None]
                if f_det:
                    groups[1] = sorted(set(f_det))
            if isinstance(b_list, list):
                b_det = [v for v in (_safe_int(x) for x in b_list) if v is not None]
                if b_det:
                    groups[2] = sorted(set(b_det))

        t0_bins = (
            histogram_info.get("t0_bins") if isinstance(histogram_info.get("t0_bins"), list) else []
        )
        forward_group = _safe_int(grouping.get("forward_group"))
        backward_group = _safe_int(grouping.get("backward_group"))
        alpha = _fmt_float(grouping.get("alpha"), 4)
        t0_bin = _safe_int(grouping.get("t0_bin"))
        if t0_bin is None:
            t0_bin = 0
        t_good_offset = _safe_int(grouping.get("t_good_offset"))
        first_good_bin = _safe_int(grouping.get("first_good_bin"))
        if t_good_offset is None and first_good_bin is not None:
            t_good_offset = max(0, first_good_bin - t0_bin)
        last_good_bin = _safe_int(grouping.get("last_good_bin"))
        bunching_factor = _safe_int(grouping.get("bunching_factor"))
        deadtime_correction = bool(grouping.get("deadtime_correction", False))

        with open(dat_path, "w", encoding="utf-8") as f:
            f.write("! START OF RUN INFORMATION\n")
            f.write(f"!  Run number  : {payload.get('run_number', '')}\n")
            if title:
                f.write(f"!  Title       : {title}\n")
            if temperature:
                if temperature_label:
                    f.write(f"!  Temperature : {temperature} K (label {temperature_label})\n")
                else:
                    f.write(f"!  Temperature : {temperature} K\n")
            if field:
                f.write(f"!  Field       : {field} G\n")
            if comment:
                f.write(f"!  Comment     : {comment}\n")
            if started:
                f.write(f"!  Started     : {started}\n")
            if stopped:
                f.write(f"!  Stopped     : {stopped}\n")
            if (
                n_hist is not None
                and n_bins is not None
                and bin_width_ps_text
                and bin_width_us_text
            ):
                f.write(
                    "!  Histograms  : "
                    f"{n_hist} ({n_bins} bins of {bin_width_ps_text} ps = {bin_width_us_text} us)\n"
                )
            if events_grouped and events_raw:
                f.write(
                    f"!  Events      : {events_grouped} MEv grouped in range (raw = {events_raw})\n"
                )
            elif events_raw:
                f.write(f"!  Events      : {events_raw} MEv\n")
            f.write("! END OF RUN INFORMATION\n")
            f.write("!\n")

            f.write("! START OF GROUPING INFORMATION\n")
            for gid in sorted(groups):
                det_tokens: list[str] = []
                for det in groups[gid]:
                    idx = det - 1
                    if 0 <= idx < len(t0_bins):
                        det_tokens.append(f"{det:02d}({int(t0_bins[idx])})")
                    else:
                        det_tokens.append(f"{det:02d}")
                f.write(f"!  Group#{gid:02d}  Hist(t0): {', '.join(det_tokens)}\n")

            if forward_group is not None or backward_group is not None:
                fwd_text = "" if forward_group is None else str(forward_group)
                bwd_text = "" if backward_group is None else str(backward_group)
                f.write(
                    f"!  Forward Group = {fwd_text}, Backward Group = {bwd_text}, Alpha = {alpha}\n"
                )
            elif isinstance(grouping.get("forward"), list) or isinstance(
                grouping.get("backward"), list
            ):
                f.write(f"!  Forward Group = forward, Backward Group = backward, Alpha = {alpha}\n")
            if t_good_offset is not None or last_good_bin is not None:
                fg_text = "" if t_good_offset is None else str(t_good_offset)
                lg_text = "" if last_good_bin is None else str(last_good_bin)
                f.write(f"!  Offset to first good bin = {fg_text}, Last good bin = {lg_text}\n")
            binning_mode = str(grouping.get("binning_mode", "fixed"))
            if binning_mode == "variable":
                f.write(
                    f"!  Variable binning, initial bin = {grouping.get('bin0_us')} us, "
                    f"bin at 10 us = {grouping.get('bin10_us')} us\n"
                )
            elif binning_mode == "constant_error":
                f.write(f"!  Constant-error binning, initial bin = {grouping.get('bin0_us')} us\n")
            elif bunching_factor is not None:
                f.write(f"!  Fixed binning, bunching factor = {bunching_factor}\n")
            f.write(f"!  Deadtime correction {'on' if deadtime_correction else 'off'}\n")
            f.write("! END OF GROUPING INFORMATION\n")
            f.write("!\n")

            f.write("! START OF DATA SET INFORMATION\n")
            f.write("!  Datarow: (Time) (FB asymmetry) (Error)\n")
            f.write(f"!  Title: {display_label}\n")
            f.write("!  Xlabel: Time in microseconds\n")
            f.write("!  Ylabel: % Asymmetry\n")
            f.write("! END OF DATA SET INFORMATION\n")
            f.write("! time  asymmetry  error\n")
            err_arr = y_err if y_err is not None else np.zeros_like(y_data)
            lo, hi = (None, None) if x_range is None else (min(x_range), max(x_range))
            for t_val, y_val, e_val in zip(t_data, y_data, err_arr):
                tf = float(t_val)
                if lo is not None and (tf < lo or tf > hi):
                    continue
                f.write(f"{tf:.10g} {float(y_val):.10g} {float(e_val):.10g}\n")

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
        _gle = get_gle_executable()
        if _gle is None:
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
                    [_gle, "-d", "png", str(tmp_gle)],
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

    def _collect_export_payloads(self) -> list[dict] | None:
        """Assemble the current plot's export payloads (with the ALL fallback).

        Shared by the GLE export and the plain-text data export so both see the
        same per-dataset payloads.
        """
        payloads = self.get_current_plot_export_data()
        if (
            payloads is None
            and self._current_polarization_axis == "ALL"
            and self._vector_subplot_datasets
        ):
            order = self._projection_subplot_order(self._vector_subplot_datasets)
            first_axis = order[0] if order else None
            payloads = self.get_current_plot_export_data(
                self._vector_subplot_datasets.get(first_axis) if first_axis else []
            )
        return payloads

    def _prompt_text_export_options(self) -> tuple[str, bool] | None:
        """Ask for the data-only export content and x-range option.

        Returns ``(content, limit_to_range)`` where content is one of
        ``"data"`` / ``"data_fit"`` / ``"fit"``, or ``None`` if cancelled.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Export plotted data (text)")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Content to export:"))
        content_combo = QComboBox(dialog)
        content_combo.addItem("Data only", "data")
        content_combo.addItem("Data + fit", "data_fit")
        content_combo.addItem("Fit only", "fit")
        layout.addWidget(content_combo)
        range_box = QCheckBox("Limit to current x-range", dialog)
        range_box.setChecked(False)
        layout.addWidget(range_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return str(content_combo.currentData()), bool(range_box.isChecked())

    def export_plotted_data_as_text(self) -> None:
        """Export the plotted curve(s) as plain text, without the GLE machinery.

        Reuses the same ``.dat``/``.fit`` writers (and their provenance header)
        as the GLE export, but writes only the text files — no ``.gleplot``
        folder, GLE script, or compile. The content switch (data only / data +
        fit / fit only) mirrors WiMDA's stData/stBoth/stFit.
        """
        payloads = self._collect_export_payloads()
        if not payloads:
            QMessageBox.warning(
                self,
                "Export unavailable",
                "No plotted data is available to export.",
            )
            return

        options = self._prompt_text_export_options()
        if options is None:
            return
        content, limit_to_range = options
        x_range = None
        if limit_to_range:
            x0 = float(self._x_min.value())
            x1 = float(self._x_max.value())
            if self._is_frequency_plot_panel():
                x0 = self._convert_frequency_control_value_to_axis_limit(x0)
                x1 = self._convert_frequency_control_value_to_axis_limit(x1)
            x_range = (x0, x1)

        if len(payloads) == 1:
            token = self._safe_file_token(str(payloads[0].get("label", "dataset")))
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export plotted data (text)",
                default_export_path(f"{token}.dat"),
                "Data files (*.dat);;Text files (*.txt);;All files (*)",
            )
            if not path:
                return
            remember_export_path(path)
            base = Path(path).with_suffix("")
            written = self._write_payload_text_files(base, payloads[0], content, x_range)
        else:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Export plotted data (text) — choose a folder",
                default_export_path(""),
            )
            if not directory:
                return
            remember_export_path(directory)
            export_dir = Path(directory)
            written = []
            for i, payload in enumerate(payloads):
                token = self._safe_file_token(str(payload.get("label", f"dataset_{i}")))
                written.extend(
                    self._write_payload_text_files(export_dir / token, payload, content, x_range)
                )

        if not written:
            QMessageBox.warning(
                self,
                "Nothing written",
                "The selected content produced no files (for example 'fit only' "
                "with no fit on the plot).",
            )
            return
        files_text = "\n".join(str(p) for p in written)
        self._show_export_result_dialog(
            "Export Successful",
            f"Wrote {len(written)} text file(s).",
            files_text,
        )

    def _write_payload_text_files(
        self,
        base: Path,
        payload: dict,
        content: str,
        x_range: tuple[float, float] | None,
    ) -> list[Path]:
        """Write the .dat and/or .fit text files for one payload."""
        written: list[Path] = []
        if content in ("data", "data_fit"):
            dat_path = base.with_suffix(".dat")
            self._write_data_file(
                dat_path, payload, label_text=payload.get("label"), x_range=x_range
            )
            if dat_path.exists():
                written.append(dat_path)
        if content in ("data_fit", "fit"):
            fit = payload.get("fit") or {}
            if fit.get("t") is not None and fit.get("y") is not None:
                fit_path = base.with_suffix(".fit")
                self._write_fit_file(fit_path, payload, x_range=x_range)
                if fit_path.exists():
                    written.append(fit_path)
        return written

    def export_plots_to_gle(self) -> None:
        """Export current main-plot view as GLE using gleplot.

        Data is plotted with error bars (no connecting lines), fit curves
        with lines (no markers).  File names are derived from the Label
        dropdown value for each dataset.
        """
        payloads = self._collect_export_payloads()
        if not payloads:
            QMessageBox.warning(
                self,
                "Export unavailable",
                "No plotted data is available to export.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot(s) to GLE",
            default_export_path("asymmetry_plot.gleplot"),
            "GLE export folders (*.gleplot)",
        )
        if not path:
            return
        remember_export_path(path)

        try:
            glp = importlib.import_module("gleplot")
        except ImportError:
            QMessageBox.warning(
                self,
                "gleplot not available",
                "Install gleplot to export GLE plots.",
            )
            return

        requested_gle_path = Path(path)
        gle_path, export_dir = resolve_gle_export_paths(requested_gle_path, folder=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        output_format = self._gle_format_combo.currentText().lower()
        colors = [
            "black",
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "brown",
            "gray",
        ]

        fig = glp.figure(figsize=self._export_figure_size(len(payloads)))
        written_files: list[Path] = []
        dat_writes: list[tuple[Path, dict, object]] = []

        if self._current_polarization_axis == "ALL" and self._vector_subplot_datasets:
            axis_order = self._projection_subplot_order(self._vector_subplot_datasets)
            if axis_order:
                axes_objs = None
                if hasattr(glp, "subplots"):
                    _fig_candidate, axes_candidate = glp.subplots(
                        nrows=len(axis_order),
                        ncols=1,
                        figsize=(6.4, 2.8 * len(axis_order)),
                        sharex=True,
                    )
                    fig = _fig_candidate
                    if isinstance(axes_candidate, list):
                        axes_objs = list(axes_candidate)
                    else:
                        axes_objs = [axes_candidate]

                if axes_objs is None:
                    fig = glp.figure(figsize=(6.4, 2.8 * len(axis_order)))
                    axes_objs = []
                    shared_ax = None
                    for idx in range(len(axis_order)):
                        ax = fig.add_subplot(len(axis_order), 1, idx + 1, sharex=shared_ax)
                        if shared_ax is None:
                            shared_ax = ax
                        axes_objs.append(ax)

                for idx, axis_key in enumerate(axis_order):
                    ax = axes_objs[idx]
                    axis_payloads = self.get_current_plot_export_data(
                        self._vector_subplot_datasets.get(axis_key, [])
                    )
                    if not axis_payloads:
                        continue
                    show_legend = idx == 0 and len(axis_payloads) > 1
                    self._plot_export_payloads_on_axis(
                        ax,
                        axis_payloads,
                        axis_key=axis_key,
                        written_files=written_files,
                        dat_writes=dat_writes,
                        gle_path=gle_path,
                        colors=colors,
                        show_legend=show_legend,
                    )
                    if idx == len(axis_order) - 1:
                        ax.set_xlabel("Time (µs)")

                if hasattr(fig, "subplots_adjust"):
                    # Keep y-axis labels and bottom x-label clear of clipping.
                    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.12, hspace=0.08)
            else:
                ax = fig.add_subplot(111)
                self._plot_export_payloads_on_axis(
                    ax,
                    payloads,
                    axis_key=None,
                    written_files=written_files,
                    dat_writes=dat_writes,
                    gle_path=gle_path,
                    colors=colors,
                    show_legend=len(payloads) > 1,
                )
                ax.set_xlabel("Time (µs)")
        else:
            ax = fig.add_subplot(111)
            self._plot_export_payloads_on_axis(
                ax,
                payloads,
                axis_key=None,
                written_files=written_files,
                dat_writes=dat_writes,
                gle_path=gle_path,
                colors=colors,
                show_legend=len(payloads) > 1,
            )
            ax.set_xlabel("Time (µs)")

        try:
            fig.savefig(str(gle_path))
        except TypeError as exc:
            if "folder" in str(exc):
                QMessageBox.warning(
                    self,
                    "gleplot update required",
                    "Please update gleplot to a newer version.",
                )
                return
            raise

        # Ensure our sidecar .dat files retain metadata headers even when
        # gleplot generates/overwrites data_name-matched data files on save.
        for dat_path, payload, label_text in dat_writes:
            self._write_data_file(dat_path, payload, label_text=label_text)

        # Compile using gleplot / GLE
        _gle = get_gle_executable()
        if _gle is not None:
            output_path = gle_path.with_suffix(f".{output_format}")
            try:
                subprocess.run(
                    [_gle, "-d", output_format, str(gle_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=str(gle_path.parent),
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
                    self,
                    "GLE compilation failed",
                    exc.stderr or str(exc),
                )
                self._show_gle_preview(gle_path)
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

    # ── rotating reference frame (Options → Advanced) ───────────────────

    def set_rrf_feature_enabled(self, enabled: bool) -> None:
        """Enable/disable the whole RRF surface (the Advanced toggle, app-level).

        Forwards to the RRF controls (which appear/disappear under the usual
        view condition) and redraws so the display reverts to or from the
        rotating frame immediately.
        """
        controls = getattr(self, "_rrf_controls", None)
        if controls is None:
            return
        controls.set_feature_enabled(bool(enabled))
        self._redraw_current_view()

    def rrf_has_active_parameters(self) -> bool:
        """True when the RRF controls carry an active frame (enabled + ν₀ > 0).

        Independent of the feature toggle: used on project open to decide
        whether a stored RRF configuration should auto-enable the toggle so the
        user's analysis is not silently hidden.
        """
        controls = getattr(self, "_rrf_controls", None)
        if controls is None:
            return False
        return controls.has_active_frame()

    def rrf_fit_frequency_mhz(self) -> float | None:
        """Frame frequency ν₀ (MHz) when an RRF fit should be performed, else None.

        The fit auto-couples to the plot's RRF controls: a single composite fit
        runs in the rotating frame exactly when the RRF display is active (the
        feature is on, the controls are enabled with ν₀ > 0, and the FB-asymmetry
        time view is showing). The fit consumes raw data with this offset and
        reports lab-frame frequencies (δν + ν₀).
        """
        controls = getattr(self, "_rrf_controls", None)
        if controls is None or not controls.is_active() or not controls.applies_to_current_view():
            return None
        return controls.frequency_mhz()

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
            "plot_panel_domain": self._domain,
            "current_run_number": (
                self._current_dataset.run_number if self._current_dataset is not None else None
            ),
            "time_view_mode": self.current_time_view_mode(),
            "log_counts_scale": bool(getattr(self, "_log_counts_enabled", False)),
            "label_field": self._label_field_combo.currentData() if self._has_mpl else "run",
            "default_label_field": self._default_label_field,
            "label_field_by_group": dict(self._label_field_by_group),
            "overlay_enabled": self.is_overlay_enabled(),
            "bunch_factor": self._bunch_factor.value() if self._has_mpl else 1,
            "auto_x_enabled": self._auto_x_btn.isChecked() if self._has_mpl else False,
            "auto_y_enabled": self._auto_y_btn.isChecked() if self._has_mpl else False,
            "x_min": self._x_min.value() if self._has_mpl else 0.0,
            "x_max": self._x_max.value() if self._has_mpl else 10.0,
            "y_min": self._y_min.value() if self._has_mpl else -30.0,
            "y_max": self._y_max.value() if self._has_mpl else 30.0,
            "polarization_axis": self._current_polarization_axis,
            "projection_selection": list(self._selected_projection_labels),
            "y_limits_by_polarization": {
                axis: [float(lim[0]), float(lim[1])]
                for axis, lim in self._y_limits_by_polarization.items()
            },
            "fit_curve": None,
            "fit_curve_run_number": self._fit_curve_run_number,
            "fit_curves": {},
            "fit_curves_by_key": {},
            "fit_components": None,
            "fit_components_by_run": {},
            "fit_components_by_key": {},
            "annotations": [],
            "annotations_by_group": {},
            "fit_x_min": self._fit_x_min,
            "fit_x_max": self._fit_x_max,
        }
        if hasattr(self, "_rrf_controls"):
            state["rrf"] = self._rrf_controls.get_state()

        if self._is_frequency_plot_panel():
            state["frequency_x_unit"] = self._current_frequency_x_unit
            state["frequency_axis_relative_to_reference"] = bool(
                self._frequency_axis_relative_to_reference
            )
            state["frequency_x_limits_by_unit"] = {
                unit: [float(limits[0]), float(limits[1])]
                for unit, limits in self._frequency_x_limits_by_unit.items()
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
            for (run_number, axis_key), (t_fit, y_fit, label) in self._fit_curves_by_key.items():
                state["fit_curves_by_key"][self._encode_fit_storage_key(run_number, axis_key)] = {
                    "t": list(t_fit),
                    "y": list(y_fit),
                    "label": label,
                }
            if self._fit_components is not None:
                state["fit_components"] = [
                    {"name": name, "y": list(y_vals)} for name, y_vals in self._fit_components
                ]
            for run_number, curves in self._fit_components_by_run.items():
                state["fit_components_by_run"][str(run_number)] = [
                    {"name": name, "y": list(y_vals)} for name, y_vals in curves
                ]
            for (run_number, axis_key), curves in self._fit_components_by_key.items():
                state["fit_components_by_key"][
                    self._encode_fit_storage_key(run_number, axis_key)
                ] = [{"name": name, "y": list(y_vals)} for name, y_vals in curves]
            state["annotations"] = self._serialize_annotations(self._default_annotations)
            state["annotations_by_group"] = {
                str(group_id): self._serialize_annotations(annotations)
                for group_id, annotations in self._annotations_by_group.items()
            }
            state["fit_metadata"] = {str(rn): meta for rn, meta in self._fit_metadata.items()}
            state["fit_metadata_by_key"] = {
                self._encode_fit_storage_key(run_number, axis_key): meta
                for (run_number, axis_key), meta in self._fit_metadata_by_key.items()
            }

        return state

    def restore_state(
        self,
        state: dict,
        dataset: MuonDataset | None = None,
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

        if hasattr(self, "_rrf_controls"):
            self._rrf_controls.set_state(state.get("rrf"))

        self._auto_x_btn.setChecked(bool(state.get("auto_x_enabled", False)))
        self._auto_y_btn.setChecked(bool(state.get("auto_y_enabled", False)))

        default_label_field = state.get("default_label_field", state.get("label_field", "run"))
        if not self._is_valid_label_field(default_label_field):
            default_label_field = "run"
        self._default_label_field = str(default_label_field)

        raw_group_label_fields = state.get("label_field_by_group", {})
        self._label_field_by_group = {}
        if isinstance(raw_group_label_fields, dict):
            for group_id, field in raw_group_label_fields.items():
                if self._is_valid_label_field(field):
                    self._label_field_by_group[str(group_id)] = str(field)

        self._active_label_group_id = None
        self._current_polarization_axis = self._axis_canonical_key(state.get("polarization_axis"))
        # Seed the selected subset so a restored stacked-subplot view reopens
        # with the exact projections it was saved with (the chip bar is rebuilt
        # later by _refresh_vector_axis_selector, which reads this).
        raw_selection = state.get("projection_selection")
        self._selected_projection_labels = (
            [str(label) for label in raw_selection if label]
            if isinstance(raw_selection, list)
            else []
        )
        self._y_limits_by_polarization = {}
        raw_y_limits_by_axis = state.get("y_limits_by_polarization", {})
        if isinstance(raw_y_limits_by_axis, dict):
            for raw_axis, raw_limits in raw_y_limits_by_axis.items():
                axis = self._axis_canonical_key(raw_axis)
                if (
                    axis is None
                    or not isinstance(raw_limits, (list, tuple))
                    or len(raw_limits) != 2
                ):
                    continue
                try:
                    lo = float(raw_limits[0])
                    hi = float(raw_limits[1])
                except (TypeError, ValueError):
                    continue
                self._y_limits_by_polarization[axis] = (lo, hi)

        # Adopt the saved current selection as the default when valid, then let the
        # combo target that intent. A saved *custom* column that the host has not
        # pushed back yet is kept as the default (not clobbered to "run") and gets
        # selected automatically once set_custom_label_fields offers it.
        label_field = state.get("label_field", self._default_label_field)
        if self._is_valid_label_field(label_field) and self._active_label_group_id is None:
            self._default_label_field = str(label_field)
        self._rebuild_label_field_combo()

        self.set_overlay_enabled(bool(state.get("overlay_enabled", True)), emit_signal=False)
        self.set_time_view_modes(
            self._available_time_view_modes,
            current_mode=state.get("time_view_mode", self._current_time_view_mode),
        )
        self._log_counts_enabled = bool(state.get("log_counts_scale", False))
        if hasattr(self, "_log_counts_checkbox"):
            self._log_counts_checkbox.blockSignals(True)
            self._log_counts_checkbox.setChecked(self._log_counts_enabled)
            self._log_counts_checkbox.blockSignals(False)
        self._refresh_log_counts_visibility()

        if self._is_frequency_plot_panel():
            self._frequency_x_limits_by_unit = {}
            raw_limits_by_unit = state.get("frequency_x_limits_by_unit", {})
            if isinstance(raw_limits_by_unit, dict):
                for raw_unit, raw_limits in raw_limits_by_unit.items():
                    if (
                        raw_unit
                        not in {
                            "frequency_mhz",
                            "field_gauss",
                            "field_tesla",
                            "frequency_mhz:absolute",
                            "frequency_mhz:relative",
                            "field_gauss:absolute",
                            "field_gauss:relative",
                            "field_tesla:absolute",
                            "field_tesla:relative",
                        }
                        or not isinstance(raw_limits, (list, tuple))
                        or len(raw_limits) != 2
                    ):
                        continue
                    try:
                        lo = float(raw_limits[0])
                        hi = float(raw_limits[1])
                    except (TypeError, ValueError):
                        continue
                    self._frequency_x_limits_by_unit[str(raw_unit)] = (lo, hi)

            restored_unit = str(state.get("frequency_x_unit", "frequency_mhz"))
            if restored_unit not in {"frequency_mhz", "field_gauss", "field_tesla"}:
                restored_unit = "frequency_mhz"
            self._current_frequency_x_unit = restored_unit
            self._frequency_axis_relative_to_reference = bool(
                state.get("frequency_axis_relative_to_reference", False)
            )
            if hasattr(self, "_frequency_x_unit_combo"):
                idx = self._frequency_x_unit_combo.findData(restored_unit)
                if idx >= 0:
                    previous = self._frequency_x_unit_combo.blockSignals(True)
                    self._frequency_x_unit_combo.setCurrentIndex(idx)
                    self._frequency_x_unit_combo.blockSignals(previous)
            if hasattr(self, "_frequency_axis_relative_check"):
                prev = self._frequency_axis_relative_check.blockSignals(True)
                self._frequency_axis_relative_check.setChecked(
                    self._frequency_axis_relative_to_reference
                )
                self._frequency_axis_relative_check.blockSignals(prev)
            self._apply_axis_labels(
                *self._axis_labels_for_dataset(dataset, self._current_polarization_axis)
            )

        # Keep bunch factor at default in restored projects; control is hidden.
        self._bunch_factor.blockSignals(True)
        self._bunch_factor.setValue(1)
        self._bunch_factor.blockSignals(False)

        # Restore axis limit fields (will be applied after optional re-plot).
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
        self._fit_curves_by_key = {}
        self._fit_components = None
        self._fit_components_by_run = {}
        self._fit_components_by_key = {}

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

        raw_fit_curves_by_key = state.get("fit_curves_by_key", {})
        if isinstance(raw_fit_curves_by_key, dict):
            for storage_key, curve_data in raw_fit_curves_by_key.items():
                decoded = self._decode_fit_storage_key(storage_key)
                if decoded is None or not isinstance(curve_data, dict):
                    continue
                self._fit_curves_by_key[decoded] = (
                    np.array(curve_data.get("t", [])),
                    np.array(curve_data.get("y", [])),
                    curve_data.get("label", "Global Fit"),
                )
            for (run_number, _axis_key), curve_data in self._fit_curves_by_key.items():
                self._fit_curves.setdefault(run_number, curve_data)
        else:
            for run_number, curve_data in self._fit_curves.items():
                self._fit_curves_by_key[(run_number, None)] = curve_data

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

        raw_fit_components_by_key = state.get("fit_components_by_key", {})
        if isinstance(raw_fit_components_by_key, dict):
            for storage_key, entries in raw_fit_components_by_key.items():
                decoded = self._decode_fit_storage_key(storage_key)
                if decoded is None or not isinstance(entries, list):
                    continue
                self._fit_components_by_key[decoded] = [
                    (entry.get("name", "Component"), np.array(entry.get("y", []), dtype=float))
                    for entry in entries
                    if isinstance(entry, dict)
                ]
            for (run_number, _axis_key), curves in self._fit_components_by_key.items():
                self._fit_components_by_run.setdefault(run_number, list(curves))
        else:
            for run_number, curves in self._fit_components_by_run.items():
                self._fit_components_by_key[(run_number, None)] = list(curves)

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
        self._fit_metadata_by_key = {}
        raw_fit_metadata = state.get("fit_metadata", {})
        if isinstance(raw_fit_metadata, dict):
            for rn_str, meta in raw_fit_metadata.items():
                if isinstance(meta, dict):
                    try:
                        self._fit_metadata[int(rn_str)] = meta
                    except (TypeError, ValueError):
                        pass

        raw_fit_metadata_by_key = state.get("fit_metadata_by_key", {})
        if isinstance(raw_fit_metadata_by_key, dict):
            for storage_key, meta in raw_fit_metadata_by_key.items():
                decoded = self._decode_fit_storage_key(storage_key)
                if decoded is None or not isinstance(meta, dict):
                    continue
                self._fit_metadata_by_key[decoded] = meta
            for (run_number, _axis_key), meta in self._fit_metadata_by_key.items():
                self._fit_metadata.setdefault(run_number, meta)
        else:
            for run_number, meta in self._fit_metadata.items():
                self._fit_metadata_by_key[(run_number, None)] = meta

        self._update_export_enabled()

        # Re-plot the current dataset if one was provided.
        if dataset is not None:
            self._current_dataset = dataset
            self.plot_dataset(dataset)

        # Re-apply saved axis limits after dataset redraw, which may reset
        # field values to data-derived defaults.
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
