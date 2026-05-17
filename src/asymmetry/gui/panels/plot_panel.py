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
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    supports_background_correction,
)
from asymmetry.core.transform.grouping import (
    apply_grouping_aligned,
    common_t0_for_groups,
)
from asymmetry.core.transform.rebin import rebin
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    PeriodMode,
)
from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)

# Metadata fields available for dataset labelling in the legend.
_LABEL_FIELDS: list[tuple[str, str]] = [
    ("Run", "run"),
    ("Field (G)", "field"),
    ("Temperature (K)", "temperature"),
    ("Comment", "comment"),
]

_NAV_BUTTON_STYLE = """
QPushButton {
    min-width: 60px;
    border: 1px solid #9aa4b2;
    border-radius: 4px;
}
QPushButton:checked {
    font-weight: 600;
    border: 2px solid #1f6feb;
    background-color: #dbeafe;
    color: #0f3d91;
}
""".strip()


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
    overlay_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None, *, domain: str = "time") -> None:
        super().__init__(parent)
        self._domain = "frequency" if str(domain).strip().lower() == "frequency" else "time"
        self._current_frequency_x_unit = "frequency_mhz"
        self._frequency_axis_relative_to_reference = False
        self._frequency_reference_mhz: float | None = None
        self._frequency_x_limits_by_unit: dict[str, tuple[float, float]] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._figure = Figure(tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._ax = self._figure.add_subplot(111)
            self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)
            self._nav_toolbar.hide()
            self._axis_limit_callback_ids: list[tuple[object, int, int]] = []
            self._syncing_limits_from_axes = False
            self._connect_axis_limit_callbacks([self._ax])
            default_x_label, default_y_label = self._default_axis_labels()
            self._ax.set_xlabel(default_x_label)
            self._ax.set_ylabel(default_y_label)

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
            self._sync_navigation_buttons()
            layout.addLayout(self._limit_toolbar)

            self._alpha_label = QLabel("")
            self._alpha_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._alpha_label.hide()

            self._polarization_label = QLabel("Polarization:")
            self._polarization_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._polarization_label.hide()

            self._polarization_combo = QComboBox()
            self._polarization_combo.setMinimumWidth(90)
            self._polarization_combo.currentIndexChanged.connect(self._on_polarization_axis_changed)
            self._polarization_combo.hide()

            alpha_row = QHBoxLayout()
            alpha_row.setContentsMargins(2, 0, 2, 0)
            alpha_row.setSpacing(0)
            alpha_row.addStretch()
            alpha_row.addWidget(self._polarization_label)
            alpha_row.addWidget(self._polarization_combo)
            alpha_row.addSpacing(8)
            alpha_row.addWidget(self._alpha_label)
            layout.addLayout(alpha_row)

            nav_row = QHBoxLayout()
            nav_row.setContentsMargins(4, 0, 4, 0)
            nav_row.setSpacing(4)
            nav_row.addWidget(QLabel("Label:"))
            nav_row.addWidget(self._label_field_combo)
            nav_row.addWidget(self._overlay_checkbox)
            nav_row.addStretch()

            self._pan_btn = QPushButton("Pan")
            self._pan_btn.setCheckable(True)
            self._pan_btn.setMaximumWidth(60)
            self._pan_btn.setStyleSheet(_NAV_BUTTON_STYLE)
            self._pan_btn.clicked.connect(self._on_pan_button_clicked)
            nav_row.addWidget(self._pan_btn)

            self._zoom_btn = QPushButton("Zoom")
            self._zoom_btn.setCheckable(True)
            self._zoom_btn.setMaximumWidth(60)
            self._zoom_btn.setStyleSheet(_NAV_BUTTON_STYLE)
            self._zoom_btn.clicked.connect(self._on_zoom_button_clicked)
            nav_row.addWidget(self._zoom_btn)

            layout.addLayout(nav_row)

            layout.addWidget(self._canvas)
            self._has_mpl = True

            # Store current dataset for rebunching
            self._current_dataset = None
            self._current_datasets: list[MuonDataset] = []
            self._limits_initialized = False
            self._current_polarization_axis: str | None = None
            self._y_limits_by_polarization: dict[str, tuple[float, float]] = {}
            self._subplot_axes_by_polarization: dict[str, object] = {}
            self._vector_subplot_datasets: dict[str, list[MuonDataset]] = {}

            # Legend label field preferences can be scoped per Data Group.
            self._active_label_group_id: str | None = None
            self._default_label_field: str = "run"
            self._label_field_by_group: dict[str, str] = {}

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
        self._auto_x_btn = QPushButton("Auto X")
        self._auto_x_btn.setCheckable(True)
        self._auto_x_btn.setStyleSheet(_NAV_BUTTON_STYLE)
        self._auto_x_btn.clicked.connect(self._on_auto_x_button_clicked)
        self._auto_x_btn.setMaximumWidth(65)
        row0.addWidget(self._auto_x_btn)

        self._auto_y_btn = QPushButton("Auto Y")
        self._auto_y_btn.setCheckable(True)
        self._auto_y_btn.setStyleSheet(_NAV_BUTTON_STYLE)
        self._auto_y_btn.clicked.connect(self._on_auto_y_button_clicked)
        self._auto_y_btn.setMaximumWidth(65)
        row0.addWidget(self._auto_y_btn)

        row0.addStretch()
        self._limit_toolbar.addLayout(row0)

        # Apply limit changes immediately from text field edits.
        self._x_min.editingFinished.connect(self._apply_limits)
        self._x_max.editingFinished.connect(self._apply_limits)
        self._y_min.editingFinished.connect(self._apply_limits)
        self._y_max.editingFinished.connect(self._apply_limits)

        # Keep bunching control internal (hidden) for backward compatibility
        # with project state and tests; it is intentionally not shown in UI.
        self._bunch_factor = QSpinBox()
        self._bunch_factor.setRange(1, 1000)
        self._bunch_factor.setValue(1)
        self._bunch_factor.setMaximumWidth(60)
        self._bunch_factor.valueChanged.connect(self._on_bunch_changed)
        self._bunch_factor.hide()

        # Label and Overlay widgets are created here but placed in the nav row below.
        self._label_field_combo = QComboBox()
        for display, key in _LABEL_FIELDS:
            self._label_field_combo.addItem(display, userData=key)
        self._label_field_combo.setMaximumWidth(140)
        self._label_field_combo.currentIndexChanged.connect(self._on_label_field_changed)

        self._overlay_checkbox = QCheckBox("Overlay")
        self._overlay_checkbox.setChecked(False)
        self._overlay_checkbox.toggled.connect(self.overlay_toggled.emit)

        # ── Row 1: frequency-specific controls (X units and relative axis) ──
        if self._is_frequency_plot_panel():
            row1 = QHBoxLayout()
            row1.setSpacing(4)
            row1.addWidget(QLabel("X Units:"))
            self._frequency_x_unit_combo = QComboBox()
            self._frequency_x_unit_combo.addItem("Frequency (MHz)", userData="frequency_mhz")
            self._frequency_x_unit_combo.addItem("Field (G)", userData="field_gauss")
            self._frequency_x_unit_combo.currentIndexChanged.connect(
                self._on_frequency_x_unit_changed
            )
            row1.addWidget(self._frequency_x_unit_combo)

            self._frequency_axis_relative_check = QCheckBox("X relative to field")
            self._frequency_axis_relative_check.setChecked(
                self._frequency_axis_relative_to_reference
            )
            self._frequency_axis_relative_check.toggled.connect(
                self.set_frequency_axis_relative_to_reference
            )
            row1.addWidget(self._frequency_axis_relative_check)

            row1.addStretch()
            self._limit_toolbar.addLayout(row1)

        # ── Row 2: annotation + export controls ────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addStretch()

        self._add_label_btn = QPushButton("Add Annotation")
        self._add_label_btn.setCheckable(True)
        # Avoid clipping on platforms/themes where checkable button chrome
        # adds extra horizontal padding around the label text.
        min_btn_width = self._add_label_btn.fontMetrics().horizontalAdvance("Add Annotation") + 32
        self._add_label_btn.setMinimumWidth(min_btn_width)
        self._add_label_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        row2.addWidget(self._add_label_btn)

        # Export controls (right side of row 2)
        self._export_gle_btn = QPushButton("Export Plot(s) to GLE")
        self._export_gle_btn.setEnabled(False)
        self._export_gle_btn.clicked.connect(self.export_plots_to_gle)
        self._gle_format_combo = QComboBox()
        self._gle_format_combo.addItems(["PDF", "EPS"])
        self._gle_format_combo.setEnabled(False)

        row2.addWidget(self._export_gle_btn)
        row2.addWidget(QLabel("Format:"))
        row2.addWidget(self._gle_format_combo)

        self._limit_toolbar.addLayout(row2)

    def _is_frequency_plot_panel(self) -> bool:
        """Return True when this panel is dedicated to frequency-domain viewing."""
        return self._domain == "frequency"

    def _default_axis_labels(self) -> tuple[str, str]:
        """Return fallback axis labels for this panel domain."""
        if self._is_frequency_plot_panel():
            return self._display_x_label(), "FFT (a.u.)"
        return "Time (μs)", "Asymmetry (%)"

    def _mhz_per_gauss(self) -> float:
        """Return the frequency equivalent of one Gauss in MHz."""
        return MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA

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
        """Return the applied-field reference frequency in MHz for *dataset*."""
        if dataset is None:
            return None
        field_value = dataset.metadata.get("field")
        try:
            field_gauss = float(field_value)
        except (TypeError, ValueError):
            run = getattr(dataset, "run", None)
            metadata = getattr(run, "metadata", {}) if run is not None else {}
            try:
                field_gauss = float(metadata.get("field"))
            except (TypeError, ValueError):
                return None
        return field_gauss * self._mhz_per_gauss()

    def _display_frequency_reference(self, *, unit: str | None = None) -> float:
        """Return the current frequency reference in the requested display unit."""
        reference_mhz = self._frequency_reference_mhz
        if reference_mhz is None:
            return 0.0
        resolved_unit = self._current_frequency_x_unit if unit is None else str(unit)
        if resolved_unit == "field_gauss":
            return reference_mhz / self._mhz_per_gauss()
        return reference_mhz

    def _display_x_label(self) -> str:
        """Return the x-axis label for the current display unit."""
        if not self._is_frequency_plot_panel():
            return "Time (μs)"
        if self._current_frequency_x_unit == "field_gauss":
            return "Field (G)"
        return "Frequency (MHz)"

    def _display_x_unit_suffix(self) -> str:
        """Return the compact unit suffix for the x-limit controls."""
        if not self._is_frequency_plot_panel():
            return "μs"
        return "G" if self._current_frequency_x_unit == "field_gauss" else "MHz"

    def _display_y_unit_suffix(self, y_label: str | None = None) -> str:
        """Return the compact unit suffix for the y-limit controls."""
        if not self._is_frequency_plot_panel():
            return "%"
        text = str(y_label or "").strip().lower()
        return "deg" if "deg" in text else "a.u."

    def _convert_frequency_axis_for_display(self, x_values) -> np.ndarray:
        """Convert canonical MHz axis data into the selected absolute display unit."""
        arr = np.asarray(x_values, dtype=float)
        if not self._is_frequency_plot_panel():
            return arr
        if self._current_frequency_x_unit != "field_gauss":
            return arr
        return arr / self._mhz_per_gauss()

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
        if from_unit == "frequency_mhz" and to_unit == "field_gauss":
            return float(value) / self._mhz_per_gauss()
        if from_unit == "field_gauss" and to_unit == "frequency_mhz":
            return float(value) * self._mhz_per_gauss()
        return float(value)

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
    ) -> None:
        """Switch frequency-axis display mode with a single redraw."""
        if not self._is_frequency_plot_panel():
            return

        old_unit = self._current_frequency_x_unit
        old_relative = self._frequency_axis_relative_to_reference
        new_unit = old_unit if unit is None else str(unit)
        new_relative = old_relative if relative is None else bool(relative)
        if new_unit == old_unit and new_relative == old_relative:
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
        if hasattr(self, "_x_unit_label"):
            self._x_unit_label.setText(self._display_x_unit_suffix())
        if hasattr(self, "_y_unit_label"):
            self._y_unit_label.setText(self._display_y_unit_suffix(y_label))

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

    def _sync_limits_from_axes(self, source_axis: object | None = None) -> None:
        """Update x/y limit fields from current Matplotlib axis limits."""
        if not self._has_mpl or self._syncing_limits_from_axes:
            return

        self._syncing_limits_from_axes = True
        try:
            if self._subplot_axes_by_polarization:
                subplot_axes = list(self._subplot_axes_by_polarization.values())
                if source_axis is not None and not any(
                    source_axis is axis for axis in subplot_axes
                ):
                    return

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
                    return

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
                return

            if not hasattr(self._ax, "get_xlim") or not hasattr(self._ax, "get_ylim"):
                return
            if source_axis is not None and source_axis is not self._ax:
                return

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
        finally:
            self._syncing_limits_from_axes = False

    def _on_axis_limits_changed(self, axis_obj) -> None:
        """Sync limit controls when Matplotlib axes change via pan/zoom."""
        self._sync_limits_from_axes(source_axis=axis_obj)

    def _on_canvas_draw_event(self, _event) -> None:
        """Keep nav buttons and limit controls aligned after Matplotlib redraws."""
        self._sync_navigation_buttons()
        if self._current_navigation_mode() != "none":
            self._sync_limits_from_axes()

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
        self._redraw_current_view()

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
            y_label = "FFT (a.u.)"
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

    def _set_frequency_reference_from_dataset(self, dataset: MuonDataset | None) -> None:
        """Update the reference frequency used by relative frequency displays."""
        if not self._is_frequency_plot_panel():
            return
        self._frequency_reference_mhz = self._frequency_reference_for_dataset(dataset)

    def _render_empty_plot_state(self, *, alpha_text: str | None = None) -> None:
        """Render an empty but valid plot state when no plottable data is available."""
        self._last_plot_time = None
        self._last_plot_asymmetry = None
        self._last_plot_error = None
        self._last_low_count_mask = None
        self._fit_x_min = None
        self._fit_x_max = None
        if self._is_frequency_plot_panel():
            self._frequency_reference_mhz = None

        self._ax.clear()
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
        self._apply_limits()

    def get_bunch_factor(self) -> int:
        """Return the currently configured plot-panel bunch factor."""
        if not self._has_mpl:
            return 1
        return int(self._bunch_factor.value())

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
        """Show or hide the alpha label above the plot."""
        if not hasattr(self, "_alpha_label"):
            return
        if text:
            self._alpha_label.setText(text)
            self._alpha_label.show()
        else:
            self._alpha_label.clear()
            self._alpha_label.hide()

    def _axis_display_text(self, axis_key: str) -> str:
        """Return UI label for canonical polarization keys."""
        return {
            "ALL": "All",
            "P_x": "x",
            "P_y": "y",
            "P_z": "z",
        }.get(str(axis_key), str(axis_key))

    def _axis_canonical_key(self, axis_text: str | None) -> str | None:
        """Normalize display/canonical axis text to canonical ``P_x`` form."""
        if axis_text is None:
            return None
        token = str(axis_text).strip().lower().replace(" ", "")
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
        return None

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
            if axis in {"P_x", "P_y", "P_z"}:
                return axis

        grouping_meta = dataset.metadata.get("grouping")
        if isinstance(grouping_meta, dict):
            axis = self._axis_canonical_key(grouping_meta.get("vector_axis"))
            if axis in {"P_x", "P_y", "P_z"}:
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

    def _fit_curve_for_dataset(
        self,
        dataset: MuonDataset | None,
        *,
        axis_override: str | None = None,
    ) -> tuple | None:
        """Return best-matching fit curve payload for *dataset*."""
        storage_key = self._fit_storage_key_for_dataset(dataset, axis_override=axis_override)
        if storage_key is not None:
            fit_data = self._fit_curves_by_key.get(storage_key)
            if fit_data is not None:
                return fit_data

            run_number, axis_key = storage_key
            if axis_key is not None:
                has_axis_specific_fit = any(
                    key_run == run_number and key_axis in {"P_x", "P_y", "P_z"}
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
        storage_key = self._fit_storage_key_for_dataset(dataset, axis_override=axis_override)
        if storage_key is not None:
            components = self._fit_components_by_key.get(storage_key)
            if components:
                return list(components)

            run_number, axis_key = storage_key
            if axis_key is not None:
                has_axis_specific_components = any(
                    key_run == run_number and key_axis in {"P_x", "P_y", "P_z"}
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
                key_run == run_number and key_axis in {"P_x", "P_y", "P_z"}
                for key_run, key_axis in self._fit_metadata_by_key
            )
            if has_axis_specific_meta:
                return {}

            meta = self._fit_metadata_by_key.get((run_number, None))
            if isinstance(meta, dict):
                return meta

        meta = self._fit_metadata.get(run_number)
        return meta if isinstance(meta, dict) else {}

    def _cache_current_y_limits_for_axis(self) -> None:
        """Store current y-limits under the active polarization axis, if any."""
        axis = self._current_polarization_axis
        if axis is None or axis == "ALL":
            return
        y0 = float(self._y_min.value())
        y1 = float(self._y_max.value())
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        self._y_limits_by_polarization[axis] = (lo, hi)

    def _restore_y_limits_for_axis(self, axis: str | None) -> None:
        """Restore cached y-limits for the selected polarization axis."""
        if axis is None or axis == "ALL":
            return
        limits = self._y_limits_by_polarization.get(axis)
        if limits is None:
            return
        self._y_min.setValue(float(limits[0]))
        self._y_max.setValue(float(limits[1]))

    def _on_polarization_axis_changed(self, _index: int) -> None:
        """Emit polarization-axis changes from the plot header selector."""
        axis = self._polarization_combo.currentData()
        if axis is None:
            axis = self._axis_canonical_key(self._polarization_combo.currentText())
        if axis:
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
        """Enable per-axis Y editing except when in vector ALL mode."""
        disable_y_edit = bool(self._subplot_axes_by_polarization) and axis == "ALL"
        tooltip = (
            "In All mode, Y limits are inherited from x, y, and z. "
            "Select each polarization to set limits."
            if disable_y_edit
            else ""
        )
        self._y_min.setEnabled(not disable_y_edit)
        self._y_max.setEnabled(not disable_y_edit)
        self._y_min.setToolTip(tooltip)
        self._y_max.setToolTip(tooltip)
        if hasattr(self, "_auto_y_btn"):
            self._auto_y_btn.setEnabled(not disable_y_edit)
            self._auto_y_btn.setToolTip(tooltip)

    def _all_mode_axes_order(self) -> list[str]:
        """Return the axis order currently visible in ALL mode."""
        if not self._subplot_axes_by_polarization:
            return []
        return [
            axis for axis in ("P_x", "P_y", "P_z") if axis in self._subplot_axes_by_polarization
        ]

    def _sync_y_controls_with_visible_axis(self) -> None:
        """Keep Y controls aligned with currently visible polarization context."""
        if not self._subplot_axes_by_polarization:
            return

        axis = self._current_polarization_axis
        if axis in self._subplot_axes_by_polarization:
            limits = self._y_limits_by_polarization.get(axis)
            if limits is not None:
                self._y_min.setValue(float(limits[0]))
                self._y_max.setValue(float(limits[1]))
            return

        # For ALL, show a global y-range spanning all visible subplot axes.
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

    def set_polarization_axes(
        self,
        axes: list[str],
        current_axis: str | None = None,
    ) -> None:
        """Show/update the polarization selector or hide it when unavailable."""
        if not hasattr(self, "_polarization_combo"):
            return

        cleaned = [str(a) for a in axes if str(a).strip()]
        if not cleaned:
            self._cache_current_y_limits_for_axis()
            self._current_polarization_axis = None
            self._vector_subplot_datasets = {}
            self._polarization_combo.blockSignals(True)
            self._polarization_combo.clear()
            self._polarization_combo.blockSignals(False)
            self._polarization_label.hide()
            self._polarization_combo.hide()
            self._update_y_limit_controls_for_axis(None)
            return

        selected = str(current_axis) if current_axis in cleaned else cleaned[0]
        previous_axis = self._current_polarization_axis
        if previous_axis != selected:
            self._cache_current_y_limits_for_axis()

        self._polarization_combo.blockSignals(True)
        self._polarization_combo.clear()
        for axis in cleaned:
            self._polarization_combo.addItem(self._axis_display_text(axis), axis)
        idx = self._polarization_combo.findData(selected)
        if idx < 0:
            idx = 0
        self._polarization_combo.setCurrentIndex(idx)
        self._polarization_combo.blockSignals(False)
        self._polarization_label.show()
        self._polarization_combo.show()
        self._current_polarization_axis = selected

        if previous_axis != selected:
            self._restore_y_limits_for_axis(selected)
            self._sync_y_controls_with_visible_axis()
            self._update_y_limit_controls_for_axis(selected)
            self._apply_limits()
        else:
            self._update_y_limit_controls_for_axis(selected)

    def _polarization_ylabel(self, axis_key: str | None) -> str:
        """Return y-axis label for the provided polarization component."""
        if axis_key in {"P_x", "P_y", "P_z"}:
            suffix = axis_key.split("_", 1)[1]
            return rf"$a_0 P_{{{suffix}}}(t)$ (%)"
        return "Asymmetry (%)"

    def _ensure_single_axis_mode(self) -> None:
        """Recreate a single-axis figure when leaving vector-subplot mode."""
        if not self._subplot_axes_by_polarization:
            return
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        self._ax = self._figure.add_subplot(111)
        self._subplot_axes_by_polarization = {}
        self._vector_subplot_datasets = {}

    def _plot_datasets_on_axis(
        self, ax, datasets: list[MuonDataset], axis_key: str | None
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Plot one or more datasets on ``ax`` and return flattened arrays for auto-y."""
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
            analysis_dataset = self.get_analysis_dataset(dataset)
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
                ax.errorbar(
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
            ax.errorbar(
                time[draw_mask],
                asymmetry[draw_mask],
                yerr=error[draw_mask],
                fmt=".",
                markersize=3,
                color=color,
                label=self._dataset_label_for(dataset),
            )

            fit_to_plot = self._fit_curve_for_dataset(dataset, axis_override=axis_key)
            if fit_to_plot is not None:
                t_fit, y_fit, _fit_label = fit_to_plot
                fit_color = self._fit_line_color_for_dataset(
                    dataset,
                    default_color=color,
                    variant_index=i,
                )
                ax.plot(t_fit, y_fit, "-", color=fit_color, linewidth=2, label="_nolegend_")

            if np.any(finite_mask):
                all_times.append(time[finite_mask])
                all_asym.append(asymmetry[finite_mask])
                all_err.append(error[finite_mask])
                all_low.append(low_count_mask[finite_mask])

        _, y_label = self._axis_labels_for_dataset(datasets[0] if datasets else None, axis_key)
        ax.set_ylabel(y_label)
        if all_times:
            return (
                np.concatenate(all_times),
                np.concatenate(all_asym),
                np.concatenate(all_err),
                np.concatenate(all_low),
            )
        return None, None, None, None

    def plot_vector_subplots(self, datasets_by_axis: dict[str, list[MuonDataset]]) -> None:
        """Render P_x/P_y/P_z as stacked subplots for vector ``ALL`` mode."""
        if not self._has_mpl:
            return

        order = [axis for axis in ("P_x", "P_y", "P_z") if datasets_by_axis.get(axis)]
        if not order:
            return

        self._set_alpha_label(None)
        self._vector_subplot_datasets = {k: list(v) for k, v in datasets_by_axis.items() if v}
        self._current_datasets = list(self._vector_subplot_datasets.get(order[0], []))
        self._current_dataset = self._current_datasets[-1] if self._current_datasets else None

        # Stop listening to old axes before clearing the figure; stale callbacks
        # can otherwise push default [0, 1] limits into the limit fields.
        self._disconnect_axis_limit_callbacks()
        self._figure.clf()
        self._subplot_axes_by_polarization = {}
        shared_ax = None
        last_arrays = (None, None, None, None)
        for idx, axis_key in enumerate(order):
            ax = self._figure.add_subplot(len(order), 1, idx + 1, sharex=shared_ax)
            if shared_ax is None:
                shared_ax = ax
            self._subplot_axes_by_polarization[axis_key] = ax
            self._ax = ax if idx == 0 else self._ax

            t, a, e, low = self._plot_datasets_on_axis(
                ax, self._vector_subplot_datasets.get(axis_key, []), axis_key
            )
            if axis_key in self._y_limits_by_polarization:
                y0, y1 = self._y_limits_by_polarization[axis_key]
                ax.set_ylim(y0, y1)
            if idx == len(order) - 1:
                x_label, _ = self._axis_labels_for_dataset(
                    self._vector_subplot_datasets.get(axis_key, [None])[0],
                    axis_key,
                )
                ax.set_xlabel(x_label)
            else:
                ax.tick_params(labelbottom=False)
            if idx == 0:
                ax.legend()
            if t is not None:
                last_arrays = (t, a, e, low)

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

        if self._current_polarization_axis in self._subplot_axes_by_polarization:
            y_limits = self._y_limits_by_polarization.get(self._current_polarization_axis)
            if y_limits is not None:
                self._y_min.setValue(float(y_limits[0]))
                self._y_max.setValue(float(y_limits[1]))
        else:
            self._sync_y_controls_with_visible_axis()

        self._update_y_limit_controls_for_axis(self._current_polarization_axis)

        self._apply_limits()
        self._connect_axis_limit_callbacks(list(self._subplot_axes_by_polarization.values()))

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
    ) -> str:
        """Return a fit-line color with improved visibility in period mode."""
        if self._period_mode_color_for_dataset(dataset) is None:
            return default_color

        # Neutral dark tones remain visible against strong period-mode colors,
        # especially when red points are active.
        fit_palette = ["#111111", "#3f3f3f", "#636363", "#2a2a2a"]
        return fit_palette[int(variant_index) % len(fit_palette)]

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

        self._ensure_single_axis_mode()
        self._set_alpha_label(None)
        self._current_dataset = datasets[-1]
        self._current_datasets = list(datasets)
        self._set_frequency_reference_from_dataset(datasets[0])
        self._ax.clear()

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
            analysis_dataset = self.get_analysis_dataset(dataset)
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
            fit_to_plot = self._fit_curve_for_dataset(dataset)
            if fit_to_plot is not None:
                t_fit, y_fit, fit_label = fit_to_plot
                fit_color = self._fit_line_color_for_dataset(
                    dataset,
                    default_color=color,
                    variant_index=i,
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
            self._ax.legend()

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

            # Set fit range to span all datasets.
            all_t_min = float(self._last_plot_time.min())
            all_t_max = float(self._last_plot_time.max())
            if self._fit_x_min is None or self._fit_x_max is None:
                self._fit_x_min = all_t_min
                self._fit_x_max = all_t_max
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

    def plot_dataset(self, dataset: MuonDataset) -> None:
        """Plot a dataset, optionally rebinned according to the bunch factor.

        The input dataset is stored unchanged as the current dataset. If the
        bunch factor is greater than 1, temporary rebinned arrays are created
        for plotting. The source dataset itself is never mutated.
        """
        if not self._has_mpl:
            return

        self._ensure_single_axis_mode()
        # Store the original dataset
        self._current_dataset = dataset
        self._current_datasets = [dataset]
        self._set_frequency_reference_from_dataset(dataset)

        analysis_dataset = self.get_analysis_dataset(dataset)
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
        point_color = self._period_mode_color_for_dataset(dataset)
        self._ax.errorbar(
            time[draw_mask],
            asymmetry[draw_mask],
            yerr=error[draw_mask],
            fmt=".",
            markersize=3,
            color=point_color,
            ecolor=point_color,
            label=self._dataset_label_for(dataset),
        )
        x_label, y_label = self._axis_labels_for_dataset(dataset, self._current_polarization_axis)
        self._apply_axis_labels(x_label, y_label)
        self._set_alpha_label(self._single_dataset_alpha_label_text(dataset))

        # Re-plot fit curve if it exists (check both single and global fits)
        fit_to_plot = self._fit_curve_for_dataset(dataset)

        if fit_to_plot is not None:
            t_fit, y_fit, fit_label = fit_to_plot
            fit_color = self._fit_line_color_for_dataset(
                dataset,
                default_color="r",
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

        self._ax.legend()

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

        data_x_min = float(time.min())
        data_x_max = float(time.max())
        if self._fit_x_min is None or self._fit_x_max is None:
            self._fit_x_min = data_x_min
            self._fit_x_max = data_x_max

        self._draw_fit_range_artists()

        # Apply the limits
        self._apply_limits()
        self._apply_auto_limits_if_enabled()
        self._update_export_enabled()
        self._connect_axis_limit_callbacks([self._ax])

    def _apply_limits(self) -> None:
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
            for axis_key, axis_obj in self._subplot_axes_by_polarization.items():
                axis_obj.set_xlim(x0, x1)
                if self._current_polarization_axis == axis_key:
                    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
                    self._y_limits_by_polarization[axis_key] = (lo, hi)
                limits = self._y_limits_by_polarization.get(axis_key)
                if limits is not None:
                    axis_obj.set_ylim(float(limits[0]), float(limits[1]))
                elif self._current_polarization_axis == "ALL":
                    # Fallback for axes without cached limits yet.
                    axis_obj.set_ylim(y0, y1)
            self._canvas.draw()
            self._emit_view_limits_changed()
            return

        self._ax.set_xlim(x0, x1)
        self._ax.set_ylim(y0, y1)
        self._cache_current_y_limits_for_axis()
        self._draw_fit_range_artists()
        self._canvas.draw()
        self._emit_view_limits_changed()

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
        self._apply_limits()

    def _auto_y_limits(self) -> None:
        """Auto-scale y-axis from visible, non-low-count points only."""
        if not self._has_mpl:
            return

        if self._subplot_axes_by_polarization and self._current_polarization_axis == "ALL":
            # ALL mode inherits per-axis limits from individual polarization views.
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
        reference_asym = np.asarray(reference_dataset.asymmetry, dtype=float)
        if reference_asym.size == 0:
            return np.zeros_like(dataset.time, dtype=bool)

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

        try:
            forward_gid = int(grouping.get("forward_group", 1))
            backward_gid = int(grouping.get("backward_group", 2))
        except (TypeError, ValueError):
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        def _to_indices(values) -> list[int]:
            out: list[int] = []
            for val in values:
                try:
                    out.append(max(0, int(val) - 1))
                except (TypeError, ValueError):
                    continue
            return out

        forward_idx = _to_indices(groups.get(forward_gid, []))
        backward_idx = _to_indices(groups.get(backward_gid, []))
        if not forward_idx or not backward_idx:
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        if max(forward_idx, default=-1) >= len(run.histograms):
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )
        if max(backward_idx, default=-1) >= len(run.histograms):
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )

        common_t0 = common_t0_for_groups(run.histograms, forward_idx, backward_idx)
        forward = apply_grouping_aligned(run.histograms, forward_idx, common_t0_bin=common_t0)
        backward = apply_grouping_aligned(run.histograms, backward_idx, common_t0_bin=common_t0)
        n_grouped = min(len(forward), len(backward))
        forward = forward[:n_grouped]
        backward = backward[:n_grouped]
        if n_grouped == 0:
            return self._project_source_mask_to_analysis_dataset(
                source_mask=saturated,
                source_dataset=reference_dataset,
                analysis_dataset=dataset,
            )
        try:
            alpha = float(grouping.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha = 1.0

        if bool(grouping.get("background_correction", False)):
            run_metadata = getattr(run, "metadata", None)
            metadata = run_metadata if isinstance(run_metadata, dict) else {}
            source_file = str(
                getattr(run, "source_file", "") or reference_dataset.metadata.get("source_file", "")
            )
            if supports_background_correction(metadata=metadata, source_file=source_file):
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

        limits_changed = False
        current_x_min = float(self._x_min.value())
        current_x_max = float(self._x_max.value())
        if lo < current_x_min:
            self._set_limit_field_value(self._x_min, lo)
            limits_changed = True
        if hi > current_x_max:
            self._set_limit_field_value(self._x_max, hi)
            limits_changed = True

        if redraw:
            if limits_changed:
                self._apply_limits()
            else:
                self._draw_fit_range_artists()
                self._canvas.draw_idle()

        if emit_signal:
            self.fit_range_changed.emit(self._fit_x_min, self._fit_x_max)

    def _draw_fit_range_artists(self) -> None:
        """Draw highlight and edge handles for the selected fit range."""
        if not self._has_mpl:
            return
        if self._is_frequency_plot_panel():
            return
        if self._subplot_axes_by_polarization:
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
            self._drag_started = False
            return

        ann_idx = self._detect_annotation_hit(event)
        if ann_idx is not None:
            self._active_annotation_idx = ann_idx
            self._annotation_drag_started = False

    def _on_canvas_motion_notify(self, event) -> None:
        """Drag the active fit-range handle while the mouse moves."""
        if self._current_navigation_mode() != "none":
            return

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
        if self._current_navigation_mode() != "none":
            return

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
        axis_key = None
        if self._current_dataset is not None:
            try:
                run_number = int(self._current_dataset.run_number)
            except (TypeError, ValueError):
                run_number = None
            axis_key = self._axis_key_for_dataset(self._current_dataset)
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

        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)
        else:
            self._ax.plot(t_fit, y_fit, "r-", linewidth=2, label=label)
            self._ax.legend()
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

    def clear(self) -> None:
        """Clear the plot and reset stored data."""
        if self._has_mpl:
            self._set_navigation_mode("none")
            self._set_alpha_label(None)
            self.set_polarization_axes([])
            self._ax.clear()
            self._canvas.draw()
            self._current_dataset = None
            self._current_datasets = []
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
            self._fit_span_artist = None
            self._fit_min_handle = None
            self._fit_max_handle = None
            self._current_polarization_axis = None
            self._y_limits_by_polarization = {}
            self._subplot_axes_by_polarization = {}
            self._vector_subplot_datasets = {}
            if self._is_frequency_plot_panel():
                self._frequency_reference_mhz = None
                self._apply_axis_labels(*self._default_axis_labels())
            self._update_export_enabled()

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
            analysis = self.get_analysis_dataset(dataset)
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

                    def _safe_int(raw: object) -> int | None:
                        try:
                            return int(raw)
                        except (TypeError, ValueError):
                            return None

                    if isinstance(grouping, dict):
                        first_good = _safe_int(grouping.get("first_good_bin"))
                        last_good = _safe_int(grouping.get("last_good_bin"))
                        if first_good is not None and last_good is not None:
                            lo = max(0, min(first_good, last_good))
                            hi = max(first_good, last_good)

                            grouped_total = 0.0
                            n_hist = len(histograms)
                            groups_raw = grouping.get("groups")
                            if isinstance(groups_raw, dict):
                                f_gid = _safe_int(grouping.get("forward_group"))
                                b_gid = _safe_int(grouping.get("backward_group"))
                                selected = []
                                if f_gid is not None and f_gid in groups_raw:
                                    selected.extend(groups_raw[f_gid])
                                if b_gid is not None and b_gid in groups_raw:
                                    selected.extend(groups_raw[b_gid])
                                for det in selected:
                                    det_idx = _safe_int(det)
                                    if det_idx is None:
                                        continue
                                    hist_idx = det_idx - 1
                                    if hist_idx < 0 or hist_idx >= n_hist:
                                        continue
                                    counts = np.asarray(histograms[hist_idx].counts, dtype=float)
                                    if counts.size == 0:
                                        continue
                                    hi_clamped = min(hi, counts.size - 1)
                                    if hi_clamped >= lo:
                                        grouped_total += float(np.sum(counts[lo : hi_clamped + 1]))

                            if grouped_total > 0:
                                histogram_info["events_grouped"] = grouped_total

            rn = dataset.run_number
            fit_data = self._fit_curve_for_dataset(dataset)

            t_fit = None
            y_fit = None
            fit_label = "Fit"
            if fit_data is not None:
                t_fit, y_fit, fit_label = fit_data
            component_data = self._fit_components_for_dataset(dataset)

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
            ax.legend(loc="best")

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

    def _write_data_file(
        self, dat_path: Path, payload: dict, *, label_text: object | None = None
    ) -> None:
        """Write a .dat file with spectra data and metadata header."""
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
            if bunching_factor is not None:
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
            for t_val, y_val, e_val in zip(t_data, y_data, err_arr):
                f.write(f"{float(t_val):.10g} {float(y_val):.10g} {float(e_val):.10g}\n")

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
        if (
            payloads is None
            and self._current_polarization_axis == "ALL"
            and self._vector_subplot_datasets
        ):
            first_axis_payloads = self.get_current_plot_export_data(
                self._vector_subplot_datasets.get("P_x")
                or self._vector_subplot_datasets.get("P_y")
                or self._vector_subplot_datasets.get("P_z")
                or []
            )
            payloads = first_axis_payloads
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
            axis_order = [a for a in ("P_x", "P_y", "P_z") if self._vector_subplot_datasets.get(a)]
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
        if shutil.which("gle") is not None:
            output_path = gle_path.with_suffix(f".{output_format}")
            try:
                subprocess.run(
                    ["gle", "-d", output_format, str(gle_path)],
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

        self._auto_x_btn.setChecked(bool(state.get("auto_x_enabled", False)))
        self._auto_y_btn.setChecked(bool(state.get("auto_y_enabled", False)))

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
        self._current_polarization_axis = self._axis_canonical_key(state.get("polarization_axis"))
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

        self.set_overlay_enabled(bool(state.get("overlay_enabled", True)), emit_signal=False)

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
                            "frequency_mhz:absolute",
                            "frequency_mhz:relative",
                            "field_gauss:absolute",
                            "field_gauss:relative",
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
            if restored_unit not in {"frequency_mhz", "field_gauss"}:
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
