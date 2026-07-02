"""Panel for inspecting fitted parameters across multiple datasets."""

from __future__ import annotations

import csv
import importlib
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.angular_assignment import (
    AngularAssignmentResult,
    fit_assigned_angular_curves,
)
from asymmetry.core.fitting.component_tracking import (
    Component,
    CrossingEvent,
    ScanPoint,
    detect_crossings,
)
from asymmetry.core.fitting.composite_parameters import (
    CompositeExpression,
    CompositeExpressionError,
    CompositeParameterDefinition,
)
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    KnightShiftConfig,
    concrete_unit,
    knight_shift,
    label_for_unit,
    larmor_frequency_mhz,
    scale_for_unit,
)
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
    effective_range_bounds,
    parse_fit_windows,
)
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    get_param_info,
    register_derived_param_info,
    split_parameter_name,
    unregister_derived_param_info,
)
from asymmetry.core.utils.angles import wrap_angle_deg
from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)
from asymmetry.gui.gle_settings import get_gle_executable
from asymmetry.gui.panels.composite_parameter_dialog import CompositeParameterDialog
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog
from asymmetry.gui.panels.knight_joint_fit_dialog import KnightJointFitDialog
from asymmetry.gui.panels.knight_shift_dialog import KnightShiftDialog
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    apply_param_table_style,
    clear_layout,
    make_section,
    style_group_state_button,
)
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.collapsible_section import CollapsibleSection
from asymmetry.gui.widgets.loading_overlay import LoadingOverlay

_PARAMETER_FIT_CURVE_SAMPLE_COUNT = 800

#: Sentinel distinguishing "x_domain not provided" (GUI callers — read it live
#: from the rows) from an explicitly-passed snapshot that may itself be ``None``
#: (the off-thread worker — must never read ``self`` for it).
_UNSET = object()


def _format_param_label(name: str) -> str:
    return get_param_info(name).unicode_label()


def _format_plot_label(name: str) -> str:
    return get_param_info(name).latex_label()


def _format_plot_legend_label(name: str) -> str:
    return get_param_info(name).latex


def _format_gle_label(name: str) -> str:
    return get_param_info(name).gle_label()


def _format_gle_legend_label(name: str) -> str:
    return get_param_info(name).gle


def _format_x_label_gle(x_key: str, custom_labels: dict[str, str] | None = None) -> str:
    if x_key == "field":
        return "{\\it{B}} (G)"
    if x_key == "temperature":
        return "{\\it{T}} (K)"
    name = _x_param_name(x_key)
    if name is not None:
        return _format_gle_label(name)
    custom_id = _x_custom_id(x_key)
    if custom_id is not None:
        return str((custom_labels or {}).get(custom_id, custom_id))
    # First-class Angle (and any non-"custom:" keyed label) resolves by direct
    # key lookup in the supplied label map.
    if custom_labels and x_key in custom_labels:
        return str(custom_labels[x_key])
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


def _safe_data_name(value: object) -> str:
    """Build a filesystem-safe stem for generated GLE data files."""
    text = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(value))
    text = "_".join(part for part in text.split("_") if part)
    return text or "series"


def _normalize_x_key(value: object) -> str:
    """Normalize persisted x-axis key to an internal identifier.

    Recognises the reserved run-level axes (``field``/``temperature``/``run``),
    the first-class ``angle`` axis, the ``param:<name>`` namespace used for
    parameter-vs-parameter trending (item 1), and the ``custom:<id>`` namespace
    for data-browser custom columns. Anything else collapses to ``run``.
    """
    text = str(value or "").strip()
    if text in ("field", "temperature", "run", "angle"):
        return text
    if text.startswith("param:") and len(text) > len("param:"):
        return text
    if text.startswith("custom:") and len(text) > len("custom:"):
        return text
    return "run"


def _x_param_name(x_key: str) -> str | None:
    """Return the fitted-parameter name for a ``param:<name>`` x-key, else None."""
    if x_key.startswith("param:"):
        return x_key[len("param:") :]
    return None


@dataclass
class _FitRow:
    run_number: int
    run_label: str
    field: float
    temperature: float
    values: dict[str, float]
    errors: dict[str, float]
    combined_from: list[int] | None = None
    covariance: dict[str, dict[str, float]] | None = None
    #: Per-run data-browser custom-column values, keyed by column id
    #: (``custom:<hex>``). Stored as raw text — they are free-form and may be
    #: empty or non-numeric for some runs; coercion to a float abscissa (and the
    #: dropping of invalid points) happens lazily in :meth:`_x_value`.
    custom_values: dict[str, str] = field(default_factory=dict)
    #: Originating time-domain fit provenance, surfaced so parameter-plot exports
    #: record which model produced the values and each run's goodness of fit.
    #: ``model_name`` is the model formula/label; the χ² fields are ``None`` when
    #: the source path did not supply them (e.g. legacy projects).
    model_name: str = ""
    chi_squared: float | None = None
    reduced_chi_squared: float | None = None


def _x_custom_id(x_key: str) -> str | None:
    """Return the custom-column id for a ``custom:<id>`` x-key, else None."""
    return x_key if isinstance(x_key, str) and x_key.startswith("custom:") else None


def _coerce_abscissa(raw: object) -> float:
    """Coerce a free-text custom/Angle cell value to a numeric abscissa.

    Empty, non-numeric, and non-finite (``inf``/``nan``) values all map to NaN so
    the point is dropped (and counted in the skip note) rather than plotted at 0
    or corrupting the axis/fit.
    """
    text = str(raw).strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _optional_float(value: object) -> float | None:
    """Coerce to a finite float, or ``None`` when missing/non-numeric.

    Used for the optional χ² provenance fields, which legacy projects and
    computed (model-less) series may not carry.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _custom_values_from_metadata(meta: object) -> dict[str, str]:
    """Extract a dataset's custom-column values (``custom:<id>`` → text)."""
    raw = meta.get("custom_fields") if isinstance(meta, dict) else None
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {}


def _custom_values_from_row_dict(entry: object) -> dict[str, str]:
    """Extract persisted/serialised custom-column values from a row dict."""
    raw = entry.get("custom_values") if isinstance(entry, dict) else None
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {}


@dataclass
class _YParamControls:
    fit_button: QPushButton
    log: QCheckBox


@dataclass
class _GroupFitData:
    group_id: str
    group_name: str
    rows: list[_FitRow]
    global_params: ParameterSet | None
    varying_params: list[str]
    inferred_x_key: str
    model_fits: dict[str, ParameterModelFit]
    plot_annotations: list[dict[str, object]]
    global_param_uncertainties: dict[str, float] = field(default_factory=dict)
    composite_parameters: list[CompositeParameterDefinition] = field(default_factory=list)
    #: Fitted-param → Knight-shift kind for this series' convertible components,
    #: derived from its model (empty for computed/model-less series).
    knight_observables: dict[str, str] = field(default_factory=dict)


class FitParametersPanel(QWidget):
    """Table + plot view for parameter trends from global fits."""

    cross_group_fit_completed = Signal(object, object, object)
    #: Emitted when a single-series model fit completes (parameter_name, x_key,
    #: ParameterModelFit) so its per-range outputs become a trendable series.
    model_fit_completed = Signal(object, object, object)
    delete_group_fits_requested = Signal(str, object)
    #: Emitted when the user activates a different fit series (batch_id).
    series_selection_changed = Signal(str)
    #: Emitted when the user renames a series via the context menu (batch_id, new_label).
    series_rename_requested = Signal(str, str)
    #: Emitted when the user chooses "Select members in browser" (batch_id).
    series_select_members_requested = Signal(str)
    #: Emitted when the user confirms "Delete series…" (batch_id).
    series_delete_requested = Signal(str)

    @property
    def last_cross_group_fit(self) -> dict[str, object] | None:
        """Return the most recent cross-group fit payload, if available."""
        return self._last_cross_group_fit

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._rows: list[_FitRow] = []
        self._varying_params: list[str] = []
        self._global_params: ParameterSet | None = None
        self._global_param_uncertainties: dict[str, float] = {}
        self._table_dialog: QDialog | None = None
        self._inferred_x_key = "field"
        #: Data-browser custom columns offered as the trend x-axis, as
        #: ``(display_label, "custom:<id>")`` pairs pushed in by the host. Their
        #: per-run values ride on each row's ``custom_values`` (see _FitRow).
        self._custom_x_fields: list[tuple[str, str]] = []
        #: The special Angle field promoted to a first-class x-axis, as a
        #: ``(display_label, key)`` pair (or None). Its key is the column id, and
        #: each row's per-run value rides on ``custom_values`` under that same key.
        self._angle_x_field: tuple[str, str] | None = None
        #: Period (degrees) to fold the Angle abscissa into, or None for no folding.
        self._angle_wrap_period: float | None = None
        self._y_controls: dict[str, _YParamControls] = {}
        self._selected_y_param_names: list[str] = []
        self._model_fits: dict[str, ParameterModelFit] = {}
        self._composite_parameters: list[CompositeParameterDefinition] = []
        #: Knight-shift conversion (panel-global; applies to the active series).
        self._knight_shift_config = KnightShiftConfig()
        #: Generated Knight-shift Y-quantity name → source frequency parameter,
        #: for the active series (rebuilt whenever derived quantities are applied).
        self._knight_shift_names: dict[str, str] = {}
        #: Active series' fitted-param → Knight-shift kind, from its model (empty
        #: → fall back to the name-based heuristic in _oscillation_components).
        self._knight_observables: dict[str, str] = {}
        #: Derived-param labels this panel registered globally, so they can be
        #: unregistered when the conversion changes / the panel is cleared.
        self._registered_knight_labels: set[str] = set()
        #: Per-series normalised fraction weights (batch_id → {fraction: weight}),
        #: supplied by the host so the table dialog can show the physical amplitude
        #: partition alongside the raw (un-normalised) fitted fractions.
        self._fraction_weights_by_id: dict[str, dict[str, float]] = {}
        #: Active joint K(θ) fit: reorders the existing K traces in place (no extra
        #: traces) so each follows one physical curve through crossings. Holds the
        #: source trace names and the per-run component→curve permutation; the raw
        #: (un-reordered) traces are regenerated by re-running the conversion, which
        #: clears this. None when no joint fit is active.
        self._joint_fit: dict | None = None
        #: True while the off-thread joint K(θ) fit is running (gates its button).
        self._joint_fit_compute_active = False
        #: Crossing/degeneracy events flagged on the active series (for annotation).
        self._knight_shift_crossings: list[object] = []
        #: The x-axis key the crossings were computed against (so stale markers are
        #: not drawn after the x-axis changes).
        self._knight_shift_crossing_x_key: str | None = None
        self._plot_annotations: list[dict[str, object]] = []
        self._axes_tag_map: dict[int, str] = {}
        self._active_annotation_idx: int | None = None
        self._annotation_drag_started = False
        self._group_fit_results: dict[str, _GroupFitData] = {}
        self._group_button_map: dict[str, QPushButton] = {}
        self._active_group_id: str | None = None
        self._last_cross_group_fit: dict[str, object] | None = None
        self._cross_group_fit_configs: dict[str, dict[str, object]] = {}
        self._group_button_style_scale = 1.0
        self._ui_scale_sync_connected = False
        #: Run numbers to highlight in the browser for the active series (used by
        #: :meth:`load_representation_series` + ``series_selection_changed``).
        self._series_run_numbers: dict[str, list[int]] = {}

        # Background machinery for the trend-overlay model evaluation, which runs
        # model.function (and optional components) per fit range over an 800-pt
        # axis and would otherwise block the GUI thread when a saved project's
        # trend fits are drawn on open.
        self._tasks = TaskRunner(self)
        self._trend_curve_compute_active = False
        #: The (active, sig) of a recompute requested while one was already in
        #: flight; a burst of refreshes collapses to this single rerun, dispatched
        #: once the running compute finishes. ``None`` means nothing pending.
        self._pending_trend_request: tuple[list[str], tuple] | None = None
        #: Cache of computed overlay curves —
        #: ``{param_name: [(range_index, xs, ys, components_or_None), ...]}`` —
        #: reused for pure-render redraws (log/scale/plot-mode toggles) and
        #: recomputed off-thread only when :attr:`_trend_cache_sig` changes.
        self._precomputed_trend_curves: dict[str, list] | None = None
        #: Signature (x-key, show-components, rows identity, per-param fit
        #: identity) the cache was computed for; a mismatch triggers an async
        #: recompute. ``None`` forces the first compute.
        self._trend_cache_sig: tuple | None = None
        #: Suppresses synchronous plot draws while a bulk state change (project
        #: restore) is in progress — every intermediate trigger (checkbox
        #: setChecked signals, group-selection sync) would otherwise evaluate the
        #: trend overlay inline on the GUI thread. The bulk operation issues a
        #: single async recompute when it finishes.
        self._suspend_plot_refresh = False

        layout = QVBoxLayout(self)

        controls_group, controls_layout = make_section("Parameter settings")
        controls_form = QFormLayout()
        controls_layout.addLayout(controls_form)

        self._group_tabs_widget = QWidget()
        self._group_tabs_layout = QHBoxLayout(self._group_tabs_widget)
        self._group_tabs_layout.setContentsMargins(0, 0, 0, 0)
        self._group_tabs_layout.setSpacing(6)
        self._group_tabs_widget.setVisible(False)
        controls_form.addRow(self._group_tabs_widget)

        self._show_table_btn = QPushButton("Show table")
        self._show_table_btn.setToolTip("Show the fitted parameter table.")
        self._show_table_btn.setEnabled(False)
        self._show_table_btn.clicked.connect(self._show_table_dialog)
        controls_form.addRow(self._show_table_btn)

        self._x_combo = QComboBox()
        self._x_combo.addItems(["Auto", "𝐵 (G)", "𝑇 (K)", "Run"])
        self._x_combo.currentTextChanged.connect(self._on_x_axis_changed)
        self._x_auto_hint = QLabel("")
        # Fold a periodic Angle abscissa into one period so equivalent crystal
        # orientations overlay (visible only when Angle is the x-axis).
        self._angle_fold_label = QLabel("Fold:")
        self._angle_fold_combo = QComboBox()
        for text, period in (("Off", None), ("180°", 180.0), ("360°", 360.0)):
            self._angle_fold_combo.addItem(text, userData=period)
        self._angle_fold_combo.currentIndexChanged.connect(self._on_angle_fold_changed)
        self._log_x_check = QCheckBox("log")
        log_x_width = self._log_x_check.fontMetrics().horizontalAdvance("log") + 28
        self._log_x_check.setMinimumWidth(log_x_width)
        self._log_x_check.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._log_x_check.stateChanged.connect(self._refresh_plot)
        x_row = QHBoxLayout()
        x_row.setContentsMargins(0, 0, 6, 0)
        x_row.setSpacing(6)
        x_row.addWidget(self._x_combo)
        x_row.addWidget(self._x_auto_hint)
        x_row.addStretch()
        x_row.addWidget(self._angle_fold_label)
        x_row.addWidget(self._angle_fold_combo)
        x_row.addWidget(self._log_x_check)
        x_container = QWidget()
        x_container.setLayout(x_row)
        controls_form.addRow("X axis:", x_container)
        self._update_angle_fold_visibility()

        self._y_selector_table = QTableWidget(0, 3)
        self._y_selector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._y_selector_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._y_selector_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # The name column (0) stretches and elides; the Model Fit (1) and log (2)
        # columns are fixed and pinned to the right. Horizontal scrolling is OFF
        # so a long parameter name can never push the action columns into an
        # off-screen scroll region (the round-10 finding: at wider inspector
        # widths the Model Fit buttons hid behind a horizontal scrollbar that
        # would not scroll to them). The name elides with "…" instead, with the
        # full label on the row tooltip.
        self._y_selector_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._y_selector_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._y_selector_table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._y_selector_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._y_selector_table.horizontalHeader().setVisible(False)
        apply_param_table_style(self._y_selector_table)
        self._y_selector_table.itemSelectionChanged.connect(self._on_y_selection_changed)
        # Selection-driven redraws are debounced: clicking through parameters
        # or drag-selecting fires per row, and each full-figure redraw costs
        # 200-500 ms in Subplots mode. One redraw after the last change wins.
        self._plot_refresh_timer = QTimer(self)
        self._plot_refresh_timer.setSingleShot(True)
        self._plot_refresh_timer.setInterval(120)
        self._plot_refresh_timer.timeout.connect(self._refresh_plot)

        y_header = self._y_selector_table.horizontalHeader()
        y_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        y_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        y_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._y_selector_table.setMinimumWidth(0)

        # Give the selector its own full-width rows rather than the form's
        # narrower field column: the label sits above and the table spans both
        # columns, so the Model Fit / log action columns keep room to stay
        # visible even at the inspector's minimum width.
        controls_form.addRow(QLabel("Y parameters:"))
        controls_form.addRow(self._y_selector_table)

        # Hint surfaced when the batch classified a parameter as Global (shared):
        # that parameter takes one value across every run, so it is held flat and
        # excluded from the trendable Y list. A user trending an amplitude curve
        # (where A_1 defaults to Global) would otherwise find their curve missing
        # with no explanation — point them to set it Local and re-fit.
        self._global_param_hint = QLabel("")
        self._global_param_hint.setWordWrap(True)
        self._global_param_hint.setObjectName("trendGlobalParamHint")
        self._global_param_hint.setStyleSheet("color: palette(mid); font-style: italic;")
        self._global_param_hint.setVisible(False)
        controls_form.addRow("", self._global_param_hint)

        self._create_composite_btn = QPushButton("New composite")
        self._create_composite_btn.setToolTip("Create a composite (derived) parameter.")
        self._create_composite_btn.setEnabled(False)
        self._create_composite_btn.clicked.connect(self._open_composite_parameter_dialog)

        self._edit_composite_btn = QPushButton("Edit composite")
        self._edit_composite_btn.setToolTip("Edit the selected composite parameter.")
        self._edit_composite_btn.setEnabled(False)
        self._edit_composite_btn.clicked.connect(self._edit_selected_composite_parameter)

        self._remove_composite_btn = QPushButton("Remove")
        self._remove_composite_btn.setToolTip(
            "Remove the selected composite parameter(s) or Knight-shift K trace(s)."
        )
        self._remove_composite_btn.setEnabled(False)
        self._remove_composite_btn.clicked.connect(self._remove_selected_composite_parameters)

        self._knight_shift_btn = QPushButton("Knight shift…")
        self._knight_shift_btn.setToolTip(
            "Convert fitted precession frequencies to the Knight shift K = (ν − ν_ref)/ν_ref."
        )
        self._knight_shift_btn.setEnabled(False)
        self._knight_shift_btn.clicked.connect(self._open_knight_shift_dialog)

        self._joint_knight_btn = QPushButton("Joint K(θ) fit…")
        self._joint_knight_btn.setToolTip(
            "Fit several K(θ) curves at once, assigning each angle's components to the "
            "curve they fit best (resolves crossings). Select ≥2 Knight-shift traces "
            "with Angle as the x-axis."
        )
        self._joint_knight_btn.setEnabled(False)
        self._joint_knight_btn.clicked.connect(self._open_joint_knight_fit_dialog)

        self._derived_section = CollapsibleSection("Derived parameters", expanded=False)
        self._derived_section.setObjectName("fit-parameters-derived-section")
        composite_row = QGridLayout()
        composite_row.setContentsMargins(0, 0, 0, 0)
        composite_row.setHorizontalSpacing(6)
        composite_row.setVerticalSpacing(6)
        composite_row.addWidget(self._create_composite_btn, 0, 0)
        composite_row.addWidget(self._edit_composite_btn, 0, 1)
        composite_row.addWidget(self._remove_composite_btn, 1, 0, 1, 2)
        composite_row.addWidget(self._knight_shift_btn, 2, 0, 1, 2)
        composite_row.addWidget(self._joint_knight_btn, 3, 0, 1, 2)
        composite_row.setColumnStretch(2, 1)
        self._derived_section.addLayout(composite_row)
        controls_layout.addWidget(self._derived_section)

        self._plot_mode_combo = QComboBox()
        self._plot_mode_combo.addItems(["Single Axes", "Subplots"])
        self._plot_mode_combo.currentTextChanged.connect(self._refresh_plot)
        controls_form.addRow("Plot mode:", self._plot_mode_combo)

        self._show_components_check = QCheckBox("Show components")
        self._show_components_check.setChecked(False)
        self._show_components_check.stateChanged.connect(self._on_show_components_changed)
        controls_form.addRow("Model components:", self._show_components_check)
        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._clear_labels_btn = QPushButton("Clear Labels")
        self._clear_labels_btn.clicked.connect(self._clear_plot_labels)

        # Hidden global log-y toggle used to mirror selected-series log state.
        self._log_y_check = QCheckBox("log")
        self._log_y_check.setVisible(False)
        self._log_y_check.stateChanged.connect(self._on_global_log_y_changed)

        self._export_tsv_btn = QPushButton("Export TSV")
        self._export_tsv_btn.setEnabled(False)
        self._export_tsv_btn.clicked.connect(self._export_tsv)

        self._export_gle_btn = QPushButton("Export to GLE")
        self._export_gle_btn.setEnabled(False)
        self._export_gle_btn.clicked.connect(self._export_gle)

        self._gle_format_combo = QComboBox()
        self._gle_format_combo.addItems(["PDF", "EPS"])
        self._gle_format_combo.setEnabled(False)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        apply_param_table_style(self._table)

        self._plot_group, plot_layout = make_section("Parameter plot")
        plot_layout.setSpacing(8)

        # Two-column grids (not a single wide row) so the plot toolbar does not
        # set the Parameters dock's minimum width past the other tabs on a 13"
        # screen; the buttons wrap to a second line instead of growing the dock.
        self._plot_labels_bar = QWidget(self._plot_group)
        labels_row = QGridLayout(self._plot_labels_bar)
        labels_row.setContentsMargins(0, 0, 0, 0)
        labels_row.setHorizontalSpacing(6)
        labels_row.setVerticalSpacing(4)
        labels_row.addWidget(QLabel("Plot labels:"), 0, 0, 1, 2)
        labels_row.addWidget(self._add_label_btn, 1, 0)
        labels_row.addWidget(self._clear_labels_btn, 1, 1)
        labels_row.setColumnStretch(2, 1)
        plot_layout.addWidget(self._plot_labels_bar)

        self._has_mpl = False
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._figure = Figure(constrained_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            plot_layout.addWidget(self._canvas, 1)
            self._has_mpl = True
            self._canvas.mpl_connect("button_press_event", self._on_plot_button_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
            self._canvas.mpl_connect("button_release_event", self._on_plot_button_release)
            # Covers the trend plot while its overlay curves recompute off-thread.
            self._trend_overlay: LoadingOverlay | None = LoadingOverlay(self._canvas)
        except ImportError:
            plot_layout.addWidget(QLabel("matplotlib not installed - plotting disabled"), 1)
            self._trend_overlay = None

        self._plot_export_bar = QWidget(self._plot_group)
        export_row = QGridLayout(self._plot_export_bar)
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setHorizontalSpacing(6)
        export_row.setVerticalSpacing(4)
        export_row.addWidget(self._export_tsv_btn, 0, 0)
        export_row.addWidget(self._export_gle_btn, 0, 1)
        export_row.addWidget(QLabel("Format:"), 1, 0)
        export_row.addWidget(self._gle_format_combo, 1, 1)
        export_row.setColumnStretch(2, 1)
        plot_layout.addWidget(self._plot_export_bar)

        controls_group.setMinimumHeight(0)
        controls_group.setMinimumWidth(0)

        self._controls_scroll = QScrollArea(self)
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._controls_scroll.setMinimumHeight(0)
        self._controls_scroll.setWidget(controls_group)

        # Empty-state hint: until a batch series is loaded the whole panel is a
        # wall of greyed controls with no cue as to where the data comes from.
        # This one-liner sits above the controls and hides itself the moment any
        # fitted rows arrive (P3-3).
        self._empty_state_hint = QLabel(
            "No fitted parameters yet — run a batch fit from the Batch tab "
            "(or open a project with batch results) to populate this trend view."
        )
        self._empty_state_hint.setObjectName("trendEmptyStateHint")
        self._empty_state_hint.setWordWrap(True)
        self._empty_state_hint.setContentsMargins(2, 2, 2, 6)
        self._empty_state_hint.setStyleSheet("color: palette(mid); font-style: italic;")
        layout.addWidget(self._empty_state_hint)

        self._content_splitter = QSplitter(Qt.Orientation.Vertical)
        self._content_splitter.setObjectName("fit-parameters-splitter")
        self._content_splitter.addWidget(self._controls_scroll)
        self._content_splitter.addWidget(self._plot_group)
        self._content_splitter.setStretchFactor(0, 0)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes([240, 600])
        layout.addWidget(self._content_splitter)

        self._update_x_axis_auto_hint()
        self._refresh_group_button_styles()
        self._update_empty_state_hint()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._ensure_ui_scale_sync()

    def _ensure_ui_scale_sync(self) -> None:
        if self._ui_scale_sync_connected:
            return
        manager = getattr(self.window(), "_ui_manager", None)
        if manager is None:
            return
        manager.ui_scale_changed.connect(self._on_ui_scale_changed)
        self._ui_scale_sync_connected = True
        self._on_ui_scale_changed(manager.ui_scale, manager.effective_scale)

    def _on_ui_scale_changed(self, _ui_scale: float, effective_scale: float) -> None:
        self._group_button_style_scale = max(0.8, float(effective_scale))
        self._refresh_group_button_styles()

    def clear(self) -> None:
        self._rows = []
        self._varying_params = []
        self._global_params = None
        self._global_param_uncertainties = {}
        self._model_fits = {}
        self._composite_parameters = []
        self._knight_shift_config = KnightShiftConfig()
        self._knight_shift_names = {}
        self._knight_observables = {}
        self._unregister_knight_labels()
        self._reset_angle_fold()
        self._knight_shift_crossings = []
        self._knight_shift_crossing_x_key = None
        self._fraction_weights_by_id = {}
        self._joint_fit = None
        self._group_fit_results = {}
        self._group_button_map = {}
        self._active_group_id = None
        self._last_cross_group_fit = None
        self._cross_group_fit_configs = {}
        self._selected_y_param_names = []
        self._series_run_numbers = {}
        self._rebuild_group_buttons()
        self._show_table_btn.setEnabled(False)
        self._create_composite_btn.setEnabled(False)
        self._edit_composite_btn.setEnabled(False)
        self._remove_composite_btn.setEnabled(False)
        self._knight_shift_btn.setEnabled(False)
        self._joint_knight_btn.setEnabled(False)
        self._rebuild_y_controls()
        self._refresh_plot()

    def get_state(self) -> dict:
        self._sync_active_group_state()
        rows = [
            {
                "run_number": int(row.run_number),
                "run_label": str(row.run_label),
                "field": float(row.field),
                "temperature": float(row.temperature),
                "values": {k: float(v) for k, v in row.values.items()},
                "errors": {k: float(v) for k, v in row.errors.items()},
                "combined_from": [int(v) for v in row.combined_from] if row.combined_from else None,
                "covariance": self._serialize_row_covariance(row.covariance),
                "custom_values": {k: str(v) for k, v in row.custom_values.items()},
                "model_name": row.model_name,
                "chi_squared": row.chi_squared,
                "reduced_chi_squared": row.reduced_chi_squared,
            }
            for row in self._rows
        ]

        selected_y = list(self._selected_y_param_names) or self._selected_y_parameters()
        log_y = [name for name, c in self._y_controls.items() if c.log.isChecked()]

        return {
            "rows": rows,
            "varying_params": list(self._varying_params),
            "composite_parameters": self._serialize_composite_parameters(
                self._composite_parameters
            ),
            "knight_shift": self._knight_shift_config.to_dict(),
            "joint_fit": self._serialize_joint_fit(),
            "inferred_x_key": self._inferred_x_key,
            "x_axis": self._x_combo.currentText(),
            "x_axis_key": self._effective_x_key(),
            "angle_wrap_period": self._angle_wrap_period,
            "selected_y_params": selected_y,
            "log_x": bool(self._log_x_check.isChecked()),
            "log_y_params": log_y,
            "show_components": bool(self._show_components_check.isChecked()),
            "plot_mode": self._plot_mode_combo.currentText(),
            "plot_annotations": [
                {
                    "x": float(ann.get("x", 0.0)),
                    "y": float(ann.get("y", 0.0)),
                    "text": str(ann.get("text", "")),
                    "axis_tag": str(ann.get("axis_tag", "main")),
                }
                for ann in self._plot_annotations
            ],
            "model_fits": self._serialize_model_fits(),
            "group_fit_results": self._serialize_group_fit_results(),
            "active_group_id": self._active_group_id,
            "selected_group_ids": self._selected_group_ids_from_buttons(),
            "last_cross_group_fit": self._serialize_last_cross_group_fit(),
            "cross_group_fit_configs": self._serialize_cross_group_fit_configs(),
        }

    @classmethod
    def _migrate_legacy_state(cls, state: dict) -> dict:
        """Strip obsolete ``K⟨n⟩`` joint-fit track artefacts from a saved state.

        Projects saved before the joint K(θ) fit reordered traces in place carry
        standalone ``K⟨1⟩…`` columns, Y-selections and overlays that the current
        code can neither regenerate nor remove via the UI. Drop them on load so
        they don't linger as un-removable stale traces.
        """
        if not isinstance(state, dict):
            return state
        is_track = cls._is_legacy_joint_track_name
        cleaned = dict(state)

        rows = state.get("rows")
        if isinstance(rows, list):
            new_rows = []
            for entry in rows:
                if isinstance(entry, dict):
                    entry = dict(entry)
                    for key in ("values", "errors"):
                        mapping = entry.get(key)
                        if isinstance(mapping, dict):
                            entry[key] = {k: v for k, v in mapping.items() if not is_track(k)}
                new_rows.append(entry)
            cleaned["rows"] = new_rows

        for list_key in ("varying_params", "selected_y_params", "log_y_params"):
            values = state.get(list_key)
            if isinstance(values, list):
                cleaned[list_key] = [v for v in values if not is_track(v)]

        model_fits = state.get("model_fits")
        if isinstance(model_fits, dict):
            cleaned["model_fits"] = {k: v for k, v in model_fits.items() if not is_track(k)}

        return cleaned

    def restore_state(self, state: dict, *, defer_refresh: bool = False) -> None:
        # Suppress the heavy synchronous plot draws each intermediate restore step
        # would otherwise trigger (checkbox setChecked signals, group-selection
        # sync); a single off-thread recompute runs at the end. try/finally
        # guarantees the guard clears even if a malformed project raises mid-way,
        # so the plot is never left permanently suspended.
        self._suspend_plot_refresh = True
        try:
            self._restore_state_locked(state)
        finally:
            self._suspend_plot_refresh = False
        # Project restore immediately re-derives the panel from the project model
        # (MainWindow._refresh_trend_panel), which rebuilds _group_fit_results and
        # runs its own table+plot refresh. Drawing here too would build the panel
        # — and re-run the heavy trend-curve compute — a second time. The
        # deserialisation above already populated the state that re-derivation
        # reads (its ``preserved`` model-fit/annotation carry-forward), so the
        # caller passes defer_refresh=True and triggers the single draw itself.
        if defer_refresh:
            return
        # The table is cheap; _refresh_plot routes the heavy overlay curves
        # (model eval per fit range over an 800-pt axis — e.g. DiffusionLF_2D runs
        # scipy quadrature per sample) onto a worker behind the overlay, so a
        # saved project's trend fits don't block the GUI thread on open.
        self.refresh_display()

    def refresh_display(self) -> None:
        """Redraw the parameter table and trend plot from the current state.

        Public entry for callers that deferred :meth:`restore_state`'s refresh
        and then need to draw (e.g. project restore where no representation-level
        re-derivation ran)."""
        self._refresh_table()
        self._refresh_plot()

    def _restore_state_locked(self, state: dict) -> None:
        state = self._migrate_legacy_state(state)
        rows_data = state.get("rows", [])
        self._composite_parameters = self._deserialize_composite_parameters(
            state.get("composite_parameters", [])
        )
        self._knight_shift_config = KnightShiftConfig.from_dict(state.get("knight_shift"))
        self._joint_fit = self._deserialize_joint_fit(state.get("joint_fit"))
        # A restored joint fit re-enables its assignment-swap markers; the reorder
        # itself is re-applied when the K traces are regenerated below.
        self._DRAW_CROSSING_MARKERS = self._joint_fit is not None
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
                            values={
                                str(k): float(v) for k, v in dict(entry.get("values", {})).items()
                            },
                            errors={
                                str(k): float(v) for k, v in dict(entry.get("errors", {})).items()
                            },
                            combined_from=[int(v) for v in entry.get("combined_from", [])]
                            if entry.get("combined_from")
                            else None,
                            covariance=self._deserialize_row_covariance(entry.get("covariance")),
                            custom_values=_custom_values_from_row_dict(entry),
                            model_name=str(entry.get("model_name") or ""),
                            chi_squared=_optional_float(entry.get("chi_squared")),
                            reduced_chi_squared=_optional_float(entry.get("reduced_chi_squared")),
                        )
                    )
                except Exception:
                    continue

        self._rows = restored_rows
        self._show_table_btn.setEnabled(bool(self._rows))
        self._export_tsv_btn.setEnabled(bool(self._rows))
        self._export_gle_btn.setEnabled(bool(self._rows))
        self._gle_format_combo.setEnabled(bool(self._rows))
        self._create_composite_btn.setEnabled(bool(self._rows))
        self._edit_composite_btn.setEnabled(False)
        self._remove_composite_btn.setEnabled(False)
        self._knight_shift_btn.setEnabled(bool(self._rows))

        varying = state.get("varying_params", [])
        if isinstance(varying, list) and all(isinstance(v, str) for v in varying):
            # Drop any persisted K[...] names: live Knight-shift columns are
            # re-added via _knight_shift_names, and stale ones must not survive as
            # frozen Y parameters.
            self._varying_params = [v for v in varying if not self._is_knight_shift_name(v)]
        else:
            self._varying_params = self._detect_varying_parameters(self._rows)

        self._apply_composite_parameters_to_rows(
            self._rows,
            self._composite_parameters,
            self._global_param_uncertainties,
        )

        inferred_x = state.get("inferred_x_key", "field")
        self._inferred_x_key = (
            inferred_x if inferred_x in {"field", "temperature", "run"} else "field"
        )

        selected_y_state = state.get("selected_y_params", [])
        if isinstance(selected_y_state, list):
            self._selected_y_param_names = [str(v) for v in selected_y_state]
        else:
            self._selected_y_param_names = []

        self._rebuild_y_controls(preferred_selected=self._selected_y_param_names)

        selected_y = set(self._selected_y_param_names)
        for i in range(self._y_selector_table.rowCount()):
            item = self._y_selector_table.item(i, 0)
            if item is None:
                continue
            pname = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(pname, str):
                continue
            item.setSelected(pname in selected_y if selected_y else i == 0)
        self._selected_y_param_names = self._selected_y_parameters()

        log_y_state = state.get("log_y_params", [])
        log_y = set(log_y_state if isinstance(log_y_state, list) else [])
        for name, controls in self._y_controls.items():
            controls.log.setChecked(name in log_y)

        self._log_y_check.setChecked(bool(log_y))

        self._show_components_check.setChecked(bool(state.get("show_components", False)))

        ann_state = state.get("plot_annotations", [])
        restored_annotations: list[dict[str, object]] = []
        if isinstance(ann_state, list):
            for entry in ann_state:
                if not isinstance(entry, dict):
                    continue
                try:
                    restored_annotations.append(
                        {
                            "x": float(entry.get("x", 0.0)),
                            "y": float(entry.get("y", 0.0)),
                            "text": str(entry.get("text", "")),
                            "axis_tag": str(entry.get("axis_tag", "main")),
                            "artist": None,
                        }
                    )
                except Exception:
                    continue
        self._plot_annotations = restored_annotations

        self._model_fits = self._deserialize_model_fits(state.get("model_fits", {}))
        self._group_fit_results = self._deserialize_group_fit_results(
            state.get("group_fit_results", {})
        )
        self._active_group_id = (
            state.get("active_group_id") if isinstance(state.get("active_group_id"), str) else None
        )
        self._last_cross_group_fit = self._deserialize_last_cross_group_fit(
            state.get("last_cross_group_fit")
        )
        self._cross_group_fit_configs = self._deserialize_cross_group_fit_configs(
            state.get("cross_group_fit_configs", {})
        )
        self._rebuild_group_buttons()

        selected_group_ids = state.get("selected_group_ids", [])
        if isinstance(selected_group_ids, list) and selected_group_ids:
            self._set_selected_group_ids([str(v) for v in selected_group_ids], emit=False)
            self._apply_group_selection_to_view(sync_active=False)
        elif self._active_group_id and self._active_group_id in self._group_fit_results:
            self._set_selected_group_ids([self._active_group_id], emit=False)
            self._apply_group_selection_to_view(sync_active=False)
        self._refresh_model_fit_button_labels()

        # Prefer the resolved x-axis key so a param:<name> or custom:<id> selection
        # survives label collisions / renames; fall back to the legacy combo-text
        # match for the fixed run-level axes.
        restored_x = False
        x_axis_key = state.get("x_axis_key")
        if isinstance(x_axis_key, str) and (
            x_axis_key.startswith("param:") or x_axis_key.startswith("custom:")
        ):
            idx = self._x_combo.findData(x_axis_key)
            if idx >= 0:
                self._x_combo.setCurrentIndex(idx)
                restored_x = True
        if not restored_x:
            x_axis = state.get("x_axis")
            if isinstance(x_axis, str):
                idx = self._x_combo.findText(x_axis)
                if idx >= 0:
                    self._x_combo.setCurrentIndex(idx)

        self._log_x_check.setChecked(bool(state.get("log_x", False)))
        self._restore_angle_fold(state.get("angle_wrap_period"))

        plot_mode = state.get("plot_mode")
        if isinstance(plot_mode, str):
            idx = self._plot_mode_combo.findText(plot_mode)
            if idx >= 0:
                self._plot_mode_combo.setCurrentIndex(idx)

        self._update_x_axis_auto_hint()

    def set_fit_results(
        self,
        results_dict: dict[int, tuple[FitResult, tuple[np.ndarray, np.ndarray]]],
        datasets_by_run: dict[int, MuonDataset],
        global_params: ParameterSet | None = None,
        *,
        group_id: str | None = None,
        group_name: str | None = None,
    ) -> None:
        self._sync_active_group_state()
        rows: list[_FitRow] = []

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
            rows.append(
                _FitRow(
                    run_number=run_number,
                    run_label=str(dataset.metadata.get("run_label") or run_number),
                    field=float(meta.get("field", 0.0)),
                    temperature=float(meta.get("temperature", 0.0)),
                    values=values,
                    errors=errors,
                    combined_from=[int(v) for v in meta.get("combined_from", [])]
                    if meta.get("combined_from")
                    else None,
                    covariance=self._fit_result_covariance_map(fit_result),
                    custom_values=_custom_values_from_metadata(meta),
                    chi_squared=_optional_float(getattr(fit_result, "chi_squared", None)),
                    reduced_chi_squared=_optional_float(
                        getattr(fit_result, "reduced_chi_squared", None)
                    ),
                )
            )

        rows.sort(key=lambda r: r.run_number)

        # Extract global parameter uncertainties from any successful per-run result.
        # The engine embeds global param errors in each FitResult.uncertainties.
        global_param_uncertainties: dict[str, float] = {}
        if global_params is not None:
            global_param_names = {p.name for p in global_params if not p.fixed}
            for fit_result, _ in results_dict.values():
                if fit_result.success and fit_result.uncertainties:
                    for pname in global_param_names:
                        if (
                            pname in fit_result.uncertainties
                            and pname not in global_param_uncertainties
                        ):
                            global_param_uncertainties[pname] = fit_result.uncertainties[pname]
                    if global_param_uncertainties.keys() >= global_param_names:
                        break

        gid = str(group_id).strip() if group_id else "__ungrouped__"
        gname = str(group_name).strip() if group_name else "Ungrouped"
        existing_group = self._group_fit_results.get(gid)
        composite_parameters = (
            list(existing_group.composite_parameters) if existing_group is not None else []
        )

        varying = self._detect_varying_parameters(rows)
        self._apply_composite_parameters_to_rows(
            rows,
            composite_parameters,
            global_param_uncertainties,
        )
        inferred_x = self._infer_x_key(rows)
        # A newly completed asymmetry fit replaces prior per-group trend/model-fit
        # state for this group. Keep only the fresh fit-parameter rows.
        model_fits: dict[str, ParameterModelFit] = {}
        plot_annotations: list[dict[str, Any]] = []

        self._group_fit_results[gid] = _GroupFitData(
            group_id=gid,
            group_name=gname,
            rows=rows,
            global_params=self._copy_parameter_set(global_params)
            if global_params is not None
            else None,
            varying_params=varying,
            inferred_x_key=inferred_x,
            model_fits=model_fits,
            plot_annotations=plot_annotations,
            global_param_uncertainties=global_param_uncertainties,
            composite_parameters=composite_parameters,
        )
        self._active_group_id = gid
        self._rebuild_group_buttons()
        self._set_selected_group_ids([gid], emit=False)
        self._apply_group_selection_to_view(sync_active=False)

    def load_representation_series(
        self,
        series_entries: list[tuple[str, str, list[dict]]],
        *,
        highlight_runs_by_id: dict[str, list[int]] | None = None,
        select_id: str | None = None,
        global_params_by_id: dict[str, dict[str, dict[str, float]]] | None = None,
        knight_observables_by_id: dict[str, dict[str, str]] | None = None,
        fraction_weights_by_id: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Reload the panel to show all series for one representation.

        This is the *pull* entry point: the caller (usually ``MainWindow``)
        fetches data from ``ProjectModel`` and passes it here whenever the
        active representation changes or a new fit is recorded.

        Parameters
        ----------
        series_entries:
            List of ``(batch_id, series_name, row_dicts)`` tuples.
            Each ``row_dict`` must have keys: ``run_number``, ``run_label``,
            ``field``, ``temperature``, ``values`` (dict), ``errors`` (dict).
            An optional ``combined_from`` list may also be present.
        highlight_runs_by_id:
            Optional mapping of ``batch_id → [run_numbers]`` used by the main
            window to drive data-browser highlighting via
            :signal:`series_selection_changed`.  Pass ``None`` to leave the
            stored map unchanged.
        select_id:
            Optional ``batch_id`` to make the active selection (e.g. the
            just-computed batch series). When present and still in the reloaded
            set it overrides the "keep prior selection / fall back to newest"
            default, so a freshly-recorded series is surfaced immediately.
        """
        self._sync_active_group_state()

        # Preserve any model-fit / composite-param / annotation state for
        # series that survive this reload.
        preserved: dict[str, dict] = {}
        for gid, gdata in self._group_fit_results.items():
            preserved[gid] = {
                "model_fits": dict(gdata.model_fits),
                "composite_parameters": list(gdata.composite_parameters),
                "plot_annotations": list(gdata.plot_annotations),
                "global_param_uncertainties": dict(gdata.global_param_uncertainties),
            }

        self._group_fit_results = {}

        for batch_id, series_name, row_dicts in series_entries:
            rows: list[_FitRow] = []
            for rd in row_dicts:
                try:
                    rows.append(
                        _FitRow(
                            run_number=int(rd["run_number"]),
                            run_label=str(rd.get("run_label") or rd["run_number"]),
                            field=float(rd.get("field", 0.0)),
                            temperature=float(rd.get("temperature", 0.0)),
                            values=dict(rd.get("values") or {}),
                            errors=dict(rd.get("errors") or {}),
                            combined_from=rd.get("combined_from"),
                            custom_values=_custom_values_from_row_dict(rd),
                            model_name=str(rd.get("model_name") or ""),
                            chi_squared=_optional_float(rd.get("chi_squared")),
                            reduced_chi_squared=_optional_float(rd.get("reduced_chi_squared")),
                        )
                    )
                except Exception:
                    continue
            if not rows:
                continue

            prev = preserved.get(batch_id, {})
            composite_params = list(prev.get("composite_parameters", []))
            global_uncert = dict(prev.get("global_param_uncertainties", {}))
            # Build the shared ("Global fitting parameters") set from the model-supplied
            # values (FitSeries.shared_parameters); without it the header shows "None".
            series_global_params = self._build_global_params(
                (global_params_by_id or {}).get(batch_id, {}), global_uncert
            )
            varying = self._detect_varying_parameters(rows)
            inferred_x = self._infer_x_key(rows)
            observables = dict((knight_observables_by_id or {}).get(batch_id, {}))
            # Apply the derived quantities with *this* series' observable map so its
            # K columns are computed from its own model (activation re-applies for
            # the active series).
            self._knight_observables = observables
            self._apply_composite_parameters_to_rows(rows, composite_params, global_uncert)

            self._group_fit_results[batch_id] = _GroupFitData(
                group_id=batch_id,
                group_name=series_name,
                rows=rows,
                global_params=series_global_params,
                varying_params=varying,
                inferred_x_key=inferred_x,
                model_fits=dict(prev.get("model_fits", {})),
                plot_annotations=list(prev.get("plot_annotations", [])),
                global_param_uncertainties=global_uncert,
                composite_parameters=composite_params,
                knight_observables=observables,
            )

        # Update per-series run-number map for browser highlighting.
        if highlight_runs_by_id is not None:
            self._series_run_numbers = dict(highlight_runs_by_id)
        # Normalised fraction weights for the table dialog (keyed by series id, so
        # they survive group switches without per-group plumbing).
        self._fraction_weights_by_id = {
            str(k): {str(n): float(w) for n, w in v.items()}
            for k, v in (fraction_weights_by_id or {}).items()
        }

        # Caller-requested selection (e.g. a just-computed batch) wins when it
        # survived the reload; otherwise activate the most-recently-added series
        # (last entry) if the prior selection is gone, else keep it.
        if select_id is not None and select_id in self._group_fit_results:
            self._active_group_id = select_id
        elif series_entries and self._active_group_id not in self._group_fit_results:
            self._active_group_id = series_entries[-1][0]
        elif not self._group_fit_results:
            self._active_group_id = None

        self._rebuild_group_buttons()
        if self._active_group_id:
            self._set_selected_group_ids([self._active_group_id], emit=False)
        self._apply_group_selection_to_view(sync_active=False)
        # A custom/Angle x-axis left selected from a previous dataset has no values
        # for these rows; fall back to Auto so the new data gets a sensible axis.
        self._reset_stale_custom_x_axis()
        # Notify listeners of the initial active series so data-browser highlights
        # fire immediately rather than waiting for the user to click a button.
        if self._active_group_id is not None:
            self.series_selection_changed.emit(self._active_group_id)

    @staticmethod
    def _build_global_params(
        shared: dict[str, dict[str, float]],
        global_uncert: dict[str, float],
    ) -> ParameterSet | None:
        """Build the shared-parameter set from model-supplied ``{name: {value, error}}``.

        ``shared`` comes from :meth:`FitSeries.shared_parameters`, so the GUI does not
        re-derive which parameters are global or harvest their values from the rows.
        Errors are copied into ``global_uncert`` so the header can show ``v ± e``.
        Returns ``None`` when there are no shared parameters (e.g. a pure batch fit).
        """
        if not shared:
            return None
        params = ParameterSet()
        for name, info in shared.items():
            try:
                params.add(Parameter(name=str(name), value=float(info["value"])))
            except (TypeError, ValueError, KeyError):
                continue
            error = info.get("error")
            if error is not None:
                try:
                    global_uncert[str(name)] = float(error)
                except (TypeError, ValueError):
                    pass
        return params if len(params) else None

    def _sync_active_group_state(self) -> None:
        """Persist current view state into the active group snapshot."""
        gid = self._active_group_id
        if gid is None or gid not in self._group_fit_results:
            return

        current = self._group_fit_results[gid]
        self._group_fit_results[gid] = _GroupFitData(
            group_id=current.group_id,
            group_name=current.group_name,
            rows=list(self._rows),
            global_params=self._copy_parameter_set(self._global_params)
            if self._global_params is not None
            else None,
            varying_params=list(self._varying_params),
            inferred_x_key=self._inferred_x_key,
            model_fits=dict(self._model_fits),
            plot_annotations=list(self._plot_annotations),
            global_param_uncertainties=dict(self._global_param_uncertainties),
            composite_parameters=list(self._composite_parameters),
            knight_observables=dict(self._knight_observables),
        )

    def _selected_group_ids_from_buttons(self) -> list[str]:
        selected: list[str] = []
        for gid, button in self._group_button_map.items():
            if button.isChecked():
                selected.append(gid)
        return selected

    def _set_selected_group_ids(self, group_ids: list[str], *, emit: bool) -> None:
        selected = set(group_ids)
        for gid, button in self._group_button_map.items():
            button.blockSignals(not emit)
            button.setChecked(gid in selected)
            button.blockSignals(False)

    def _rebuild_group_buttons(self) -> None:
        clear_layout(self._group_tabs_layout)

        self._group_button_map = {}
        groups = sorted(self._group_fit_results.values(), key=lambda g: g.group_name.lower())
        for group in groups:
            button = QPushButton(group.group_name)
            button.setCheckable(True)
            button.clicked.connect(self._on_group_button_clicked)
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda pos, gid=group.group_id, b=button: self._show_group_button_context_menu(
                    gid, b, pos
                )
            )
            self._group_tabs_layout.addWidget(button)
            self._group_button_map[group.group_id] = button
        self._group_tabs_layout.addStretch()
        self._group_tabs_widget.setVisible(bool(groups))
        self._refresh_group_button_styles()

    def _exec_menu(self, menu: QMenu, pos) -> object:
        return menu.exec(pos)

    def _show_group_button_context_menu(self, group_id: str, button: QPushButton, pos) -> None:
        if group_id not in self._group_fit_results:
            return

        group = self._group_fit_results[group_id]
        menu = QMenu(self)
        rename_action = menu.addAction("Rename…")
        select_action = menu.addAction("Select members in browser")
        menu.addSeparator()
        delete_action = menu.addAction("Delete series…")
        selected_action = self._exec_menu(menu, button.mapToGlobal(pos))

        if selected_action is rename_action:
            new_name, ok = QInputDialog.getText(
                self,
                "Rename series",
                "Series name:",
                text=group.group_name,
            )
            if ok:
                self.series_rename_requested.emit(group_id, new_name.strip())
        elif selected_action is select_action:
            self.series_select_members_requested.emit(group_id)
        elif selected_action is delete_action:
            self._delete_group_fits(group_id)

    def _group_run_numbers(self, group: _GroupFitData) -> list[int]:
        run_numbers: set[int] = set()
        for row in group.rows:
            try:
                run_numbers.add(int(row.run_number))
            except (TypeError, ValueError):
                continue
        return sorted(run_numbers)

    def _delete_group_fits(self, group_id: str) -> None:
        group = self._group_fit_results.get(group_id)
        if group is None:
            return

        reply = QMessageBox.question(
            self,
            "Delete series",
            (
                f'Delete series "{group.group_name}"?\n'
                "This removes it from the project and clears its dataset fits."
            ),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return

        run_numbers = self._group_run_numbers(group)
        self._sync_active_group_state()
        self._group_fit_results.pop(group_id, None)

        if self._active_group_id == group_id:
            self._active_group_id = None

        remaining_ids = sorted(
            self._group_fit_results,
            key=lambda gid: self._group_fit_results[gid].group_name.lower(),
        )
        if self._active_group_id not in self._group_fit_results:
            self._active_group_id = remaining_ids[0] if remaining_ids else None

        self._rebuild_group_buttons()
        selected_ids = [self._active_group_id] if self._active_group_id is not None else []
        self._set_selected_group_ids(selected_ids, emit=False)
        self._apply_group_selection_to_view(sync_active=False)
        self.delete_group_fits_requested.emit(group_id, run_numbers)
        self.series_delete_requested.emit(group_id)

    def _refresh_group_button_styles(self) -> None:
        selected_ids = set(self._selected_group_ids_from_buttons())
        active_gid: str | None = None
        if self._active_group_id in selected_ids:
            active_gid = self._active_group_id
        elif selected_ids:
            active_gid = next(iter(selected_ids))

        scale = max(0.8, float(self._group_button_style_scale))
        border_radius = max(10, round(12 * scale))
        padding_v = max(4, round(6 * scale))
        padding_h = max(10, round(14 * scale))

        base = (
            "QPushButton {"
            f" border-radius: {border_radius}px;"
            f" padding: {padding_v}px {padding_h}px;"
            " }"
            "QPushButton:checked {"
            f" border-radius: {border_radius}px;"
            f" padding: {padding_v}px {padding_h}px;"
            " }"
        )

        for gid, button in self._group_button_map.items():
            if gid == active_gid:
                state = "active"
            elif gid in selected_ids:
                state = "selected"
            else:
                state = "unselected"
            style_group_state_button(button, state, base=base, palette="red")

    def _on_group_button_clicked(self) -> None:
        self._sync_active_group_state()

        clicked_gid: str | None = None
        sender = self.sender()
        for gid, button in self._group_button_map.items():
            if button is sender:
                clicked_gid = gid
                break

        modifiers = QApplication.keyboardModifiers()
        shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if clicked_gid is not None:
            clicked_button = self._group_button_map[clicked_gid]
            # Shift+click toggles whether a group is included for global fit,
            # but does not change which group is currently selected.
            if (
                shift_pressed
                and not clicked_button.isChecked()
                and clicked_gid == self._active_group_id
            ):
                clicked_button.setChecked(True)
            elif clicked_button.isChecked() and not shift_pressed:
                self._active_group_id = clicked_gid

        selected_ids = self._selected_group_ids_from_buttons()
        if not selected_ids and self._active_group_id in self._group_button_map:
            self._group_button_map[self._active_group_id].setChecked(True)
        elif self._active_group_id not in selected_ids and selected_ids:
            self._active_group_id = selected_ids[0]

        # We already synced the previous active group at the top of this
        # handler. Avoid a second sync here because _active_group_id may now
        # refer to the newly clicked group while self._rows still contains the
        # previous group's data.
        self._apply_group_selection_to_view(sync_active=False)
        # Notify listeners (e.g. main window → data-browser highlight).
        if self._active_group_id is not None:
            self.series_selection_changed.emit(self._active_group_id)

    def set_highlight_active(self, active: bool) -> None:
        """Ask the browser to restore (or leave) the series-member highlight.

        Called by the main window when the Parameters dock becomes visible or
        hidden.  When *active* is ``True`` and there is a current series, the
        ``series_selection_changed`` signal is re-emitted so the browser tint
        is restored via the existing funnel.  When *active* is ``False`` this
        method is a no-op — the main window clears the browser highlight
        directly.
        """
        if active and self._active_group_id is not None:
            self.series_selection_changed.emit(self._active_group_id)

    def _apply_group_selection_to_view(self, *, sync_active: bool = True) -> None:
        if sync_active:
            self._sync_active_group_state()
        previous_selected_y = list(self._selected_y_param_names) or self._selected_y_parameters()
        selected_group_ids = self._selected_group_ids_from_buttons()
        if (
            not selected_group_ids
            and self._active_group_id
            and self._active_group_id in self._group_fit_results
        ):
            selected_group_ids = [self._active_group_id]
            self._set_selected_group_ids(selected_group_ids, emit=False)

        selected_groups = [
            self._group_fit_results[gid]
            for gid in selected_group_ids
            if gid in self._group_fit_results
        ]
        if not selected_groups:
            self._rows = []
            self._varying_params = []
            self._composite_parameters = []
            self._global_params = None
            self._global_param_uncertainties = {}
            self._inferred_x_key = "run"
            self._model_fits = {}
            self._plot_annotations = []
            self._show_table_btn.setEnabled(False)
            self._export_tsv_btn.setEnabled(False)
            self._export_gle_btn.setEnabled(False)
            self._gle_format_combo.setEnabled(False)
            self._create_composite_btn.setEnabled(False)
            self._edit_composite_btn.setEnabled(False)
            self._remove_composite_btn.setEnabled(False)
            self._knight_shift_btn.setEnabled(False)
            self._rebuild_y_controls(preferred_selected=previous_selected_y)
            self._refresh_model_fit_button_labels()
            self._update_x_axis_auto_hint()
            self._refresh_views()
            return

        if len(selected_groups) == 1:
            group = selected_groups[0]
            self._active_group_id = group.group_id
            self._rows = list(group.rows)
            self._varying_params = list(group.varying_params)
            self._global_params = (
                self._copy_parameter_set(group.global_params)
                if group.global_params is not None
                else None
            )
            self._global_param_uncertainties = dict(group.global_param_uncertainties)
            self._inferred_x_key = group.inferred_x_key
            self._model_fits = dict(group.model_fits)
            self._composite_parameters = list(group.composite_parameters)
            self._knight_observables = dict(group.knight_observables)
            self._plot_annotations = list(group.plot_annotations)
        else:
            active_gid = (
                self._active_group_id
                if self._active_group_id in selected_group_ids
                else selected_group_ids[0]
            )
            active_group = self._group_fit_results.get(active_gid)
            if active_group is None:
                active_group = selected_groups[0]
                active_gid = active_group.group_id

            # Keep group data distinct: with multi-selection we still display
            # the active group's parameter table/plot only.
            self._active_group_id = active_gid
            self._rows = list(active_group.rows)
            self._varying_params = list(active_group.varying_params)
            self._global_params = (
                self._copy_parameter_set(active_group.global_params)
                if active_group.global_params is not None
                else None
            )
            self._global_param_uncertainties = dict(active_group.global_param_uncertainties)
            self._inferred_x_key = active_group.inferred_x_key
            self._model_fits = dict(active_group.model_fits)
            self._composite_parameters = list(active_group.composite_parameters)
            self._knight_observables = dict(active_group.knight_observables)
            self._plot_annotations = list(active_group.plot_annotations)

        has_rows = bool(self._rows)

        self._apply_composite_parameters_to_rows(
            self._rows,
            self._composite_parameters,
            self._global_param_uncertainties,
        )

        self._show_table_btn.setEnabled(has_rows)
        self._export_tsv_btn.setEnabled(has_rows)
        self._export_gle_btn.setEnabled(has_rows)
        self._gle_format_combo.setEnabled(has_rows)
        self._create_composite_btn.setEnabled(has_rows)
        self._knight_shift_btn.setEnabled(has_rows)

        display_params = set(self._display_y_parameters())
        self._model_fits = {k: v for k, v in self._model_fits.items() if k in display_params}

        self._rebuild_y_controls(preferred_selected=previous_selected_y)
        self._refresh_model_fit_button_labels()
        self._update_x_axis_auto_hint()
        self._refresh_group_button_styles()
        self._refresh_views()

    def _on_y_selection_changed(self) -> None:
        self._selected_y_param_names = self._selected_y_parameters()
        self._update_composite_action_buttons()
        self._update_joint_fit_button()
        self._plot_refresh_timer.start()

    def _selected_knight_traces(self) -> list[str]:
        """Currently-selected Knight-shift Y traces (eligible for the joint fit)."""
        return [name for name in self._selected_y_parameters() if name in self._knight_shift_names]

    def _update_joint_fit_button(self) -> None:
        """Enable the joint K(θ) fit only for Angle x-axis + ≥2 selected K traces."""
        ready = (
            not self._joint_fit_compute_active
            and self._angle_axis_active()
            and len(self._selected_knight_traces()) >= 2
        )
        self._joint_knight_btn.setEnabled(ready)

    def _copy_parameter_set(self, source: ParameterSet) -> ParameterSet:
        copied = ParameterSet()
        for p in source:
            copied.add(
                Parameter(
                    name=p.name,
                    value=float(p.value),
                    min=float(p.min),
                    max=float(p.max),
                    fixed=bool(p.fixed),
                )
            )
        return copied

    def _build_cross_group_group_model_fit(
        self,
        *,
        parameter_name: str,
        x_key: str,
        group_id: str,
        model: ParameterCompositeModel,
        fit_result: CrossGroupFitResult,
        fit_x_min: float,
        fit_x_max: float,
    ) -> ParameterModelFit:
        param_result = ParameterSet()
        fit_params = ParameterSet()
        uncertainties: dict[str, float] = {}

        global_params = fit_result.global_parameters
        local_params = fit_result.local_parameters.get(group_id, ParameterSet())
        fixed_params = fit_result.fixed_parameters

        for pname in model.param_names:
            if pname in local_params:
                src = local_params[pname]
                err = fit_result.local_uncertainties.get(group_id, {}).get(pname)
            elif pname in global_params:
                src = global_params[pname]
                err = fit_result.global_uncertainties.get(pname)
            elif pname in fixed_params:
                src = fixed_params[pname]
                err = None
            else:
                src = Parameter(
                    name=pname,
                    value=float(model.param_defaults.get(pname, 0.0)),
                    fixed=(pname == "shape_factor_a"),
                )
                err = None

            fit_param = Parameter(
                name=src.name,
                value=float(src.value),
                min=float(src.min),
                max=float(src.max),
                fixed=bool(src.fixed),
            )
            fit_params.add(fit_param)
            param_result.add(
                Parameter(
                    name=fit_param.name,
                    value=fit_param.value,
                    min=fit_param.min,
                    max=fit_param.max,
                    fixed=fit_param.fixed,
                )
            )
            if isinstance(err, (int, float)) and np.isfinite(float(err)):
                uncertainties[pname] = float(err)

        result = ParameterModelFitResult(
            success=fit_result.success,
            chi_squared=float(fit_result.chi_squared),
            reduced_chi_squared=float(fit_result.reduced_chi_squared),
            parameters=param_result,
            uncertainties=uncertainties,
            message=fit_result.message,
        )

        x_min_value: float | None = float(fit_x_min) if np.isfinite(fit_x_min) else None
        x_max_value: float | None = float(fit_x_max) if np.isfinite(fit_x_max) else None

        model_snapshot = ParameterCompositeModel(
            component_names=list(model.component_names),
            operators=list(model.operators),
        )
        fit_range = ModelFitRange(
            x_min=x_min_value,
            x_max=x_max_value,
            model=model_snapshot,
            parameters=fit_params,
            result=result,
        )
        return ParameterModelFit(
            parameter_name=parameter_name,
            x_key=x_key,
            ranges=[fit_range],
            active=True,
        )

    def _apply_cross_group_fit_to_groups(
        self,
        *,
        parameter_name: str,
        x_key: str,
        selected_groups: list[_GroupFitData],
        output: object,
    ) -> None:
        fit_result = getattr(output, "fit_result", None)
        model = getattr(output, "model", None)
        fit_x_min = getattr(output, "fit_x_min", float("nan"))
        fit_x_max = getattr(output, "fit_x_max", float("nan"))
        if not isinstance(fit_result, CrossGroupFitResult):
            return
        if not isinstance(model, ParameterCompositeModel):
            return
        if not fit_result.success:
            return

        for group in selected_groups:
            existing = self._group_fit_results.get(group.group_id)
            if existing is None:
                continue

            group_fit = self._build_cross_group_group_model_fit(
                parameter_name=parameter_name,
                x_key=x_key,
                group_id=group.group_id,
                model=model,
                fit_result=fit_result,
                fit_x_min=float(fit_x_min),
                fit_x_max=float(fit_x_max),
            )
            next_model_fits = dict(existing.model_fits)
            next_model_fits[parameter_name] = group_fit
            self._group_fit_results[group.group_id] = _GroupFitData(
                group_id=existing.group_id,
                group_name=existing.group_name,
                rows=list(existing.rows),
                global_params=existing.global_params,
                varying_params=list(existing.varying_params),
                inferred_x_key=existing.inferred_x_key,
                model_fits=next_model_fits,
                plot_annotations=list(existing.plot_annotations),
                global_param_uncertainties=dict(existing.global_param_uncertainties),
                composite_parameters=list(existing.composite_parameters),
            )

        self._refresh_model_fit_button_labels()
        self._refresh_plot()

    def _serialize_group_fit_results(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for gid, group in self._group_fit_results.items():
            out[gid] = {
                "group_id": group.group_id,
                "group_name": group.group_name,
                "global_params": [
                    {
                        "name": p.name,
                        "value": float(p.value),
                        "min": float(p.min),
                        "max": float(p.max),
                        "fixed": bool(p.fixed),
                    }
                    for p in group.global_params
                ]
                if group.global_params is not None
                else None,
                "rows": [
                    {
                        "run_number": int(row.run_number),
                        "run_label": str(row.run_label),
                        "field": float(row.field),
                        "temperature": float(row.temperature),
                        "values": {k: float(v) for k, v in row.values.items()},
                        "errors": {k: float(v) for k, v in row.errors.items()},
                        "combined_from": [int(v) for v in row.combined_from]
                        if row.combined_from
                        else None,
                        "covariance": self._serialize_row_covariance(row.covariance),
                        "custom_values": {k: str(v) for k, v in row.custom_values.items()},
                    }
                    for row in group.rows
                ],
                "varying_params": list(group.varying_params),
                "composite_parameters": self._serialize_composite_parameters(
                    group.composite_parameters
                ),
                "inferred_x_key": group.inferred_x_key,
                "model_fits": self._serialize_specific_model_fits(group.model_fits),
                "plot_annotations": [
                    {
                        "x": float(ann.get("x", 0.0)),
                        "y": float(ann.get("y", 0.0)),
                        "text": str(ann.get("text", "")),
                        "axis_tag": str(ann.get("axis_tag", "main")),
                    }
                    for ann in group.plot_annotations
                ],
                "global_param_uncertainties": {
                    k: float(v) for k, v in group.global_param_uncertainties.items()
                },
            }
        return out

    def _serialize_specific_model_fits(self, model_fits: dict[str, ParameterModelFit]) -> dict:
        original = self._model_fits
        try:
            self._model_fits = model_fits
            return self._serialize_model_fits()
        finally:
            self._model_fits = original

    def _deserialize_group_fit_results(self, payload: object) -> dict[str, _GroupFitData]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, _GroupFitData] = {}
        for gid, entry in payload.items():
            if not isinstance(gid, str) or not isinstance(entry, dict):
                continue
            rows: list[_FitRow] = []
            for row_entry in entry.get("rows", []):
                if not isinstance(row_entry, dict):
                    continue
                try:
                    rows.append(
                        _FitRow(
                            run_number=int(row_entry.get("run_number", 0)),
                            run_label=str(
                                row_entry.get("run_label", row_entry.get("run_number", ""))
                            ),
                            field=float(row_entry.get("field", 0.0)),
                            temperature=float(row_entry.get("temperature", 0.0)),
                            values={
                                str(k): float(v)
                                for k, v in dict(row_entry.get("values", {})).items()
                            },
                            errors={
                                str(k): float(v)
                                for k, v in dict(row_entry.get("errors", {})).items()
                            },
                            combined_from=[int(v) for v in row_entry.get("combined_from", [])]
                            if row_entry.get("combined_from")
                            else None,
                            covariance=self._deserialize_row_covariance(
                                row_entry.get("covariance")
                            ),
                            custom_values=_custom_values_from_row_dict(row_entry),
                        )
                    )
                except Exception:
                    continue

            model_fits = self._deserialize_model_fits(entry.get("model_fits", {}))
            plot_annotations = []
            for ann in entry.get("plot_annotations", []):
                if not isinstance(ann, dict):
                    continue
                plot_annotations.append(
                    {
                        "x": float(ann.get("x", 0.0)),
                        "y": float(ann.get("y", 0.0)),
                        "text": str(ann.get("text", "")),
                        "axis_tag": str(ann.get("axis_tag", "main")),
                        "artist": None,
                    }
                )

            global_params_state = entry.get("global_params")
            global_params: ParameterSet | None = None
            if isinstance(global_params_state, list):
                restored = ParameterSet()
                for p in global_params_state:
                    if not isinstance(p, dict):
                        continue
                    try:
                        restored.add(
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
                global_params = restored

            gpu_state = entry.get("global_param_uncertainties", {})
            global_param_uncertainties: dict[str, float] = {}
            if isinstance(gpu_state, dict):
                for k, v in gpu_state.items():
                    try:
                        global_param_uncertainties[str(k)] = float(v)
                    except (TypeError, ValueError):
                        pass

            out[gid] = _GroupFitData(
                group_id=gid,
                group_name=str(entry.get("group_name", gid)),
                rows=rows,
                global_params=global_params,
                varying_params=[
                    str(v) for v in entry.get("varying_params", []) if isinstance(v, str)
                ],
                inferred_x_key=_normalize_x_key(entry.get("inferred_x_key", "run")),
                model_fits=model_fits,
                plot_annotations=plot_annotations,
                global_param_uncertainties=global_param_uncertainties,
                composite_parameters=self._deserialize_composite_parameters(
                    entry.get("composite_parameters", [])
                ),
            )
        return out

    def _serialize_composite_parameters(
        self,
        definitions: list[CompositeParameterDefinition],
    ) -> list[dict[str, str]]:
        return [
            {
                "name": str(definition.name),
                "expression": str(definition.expression),
            }
            for definition in definitions
            if definition.name and definition.expression
        ]

    def _deserialize_composite_parameters(
        self,
        payload: object,
    ) -> list[CompositeParameterDefinition]:
        if not isinstance(payload, list):
            return []

        definitions: list[CompositeParameterDefinition] = []
        seen: set[str] = set()
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            expression = str(entry.get("expression", "")).strip()
            if not name or not expression or name in seen:
                continue
            definitions.append(CompositeParameterDefinition(name=name, expression=expression))
            seen.add(name)
        return definitions

    def _serialize_row_covariance(
        self,
        covariance: dict[str, dict[str, float]] | None,
    ) -> dict[str, dict[str, float]] | None:
        if not covariance:
            return None
        out: dict[str, dict[str, float]] = {}
        for name, row in covariance.items():
            if not isinstance(name, str) or not isinstance(row, dict):
                continue
            out[name] = {}
            for sub_name, value in row.items():
                if not isinstance(sub_name, str):
                    continue
                try:
                    out[name][sub_name] = float(value)
                except (TypeError, ValueError):
                    continue
        return out or None

    def _deserialize_row_covariance(
        self,
        payload: object,
    ) -> dict[str, dict[str, float]] | None:
        if not isinstance(payload, dict):
            return None
        out: dict[str, dict[str, float]] = {}
        for name, row in payload.items():
            if not isinstance(name, str) or not isinstance(row, dict):
                continue
            parsed_row: dict[str, float] = {}
            for sub_name, value in row.items():
                if not isinstance(sub_name, str):
                    continue
                try:
                    parsed_row[sub_name] = float(value)
                except (TypeError, ValueError):
                    continue
            if parsed_row:
                out[name] = parsed_row
        return out or None

    def _fit_result_covariance_map(
        self, fit_result: FitResult
    ) -> dict[str, dict[str, float]] | None:
        if fit_result.covariance is None:
            return None
        cov = np.asarray(fit_result.covariance, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            return None

        order = list(fit_result.covariance_parameters)
        if not order:
            order = [p.name for p in fit_result.parameters if p.name in fit_result.uncertainties]
        if len(order) != cov.shape[0]:
            return None

        out: dict[str, dict[str, float]] = {}
        for i, name_i in enumerate(order):
            row: dict[str, float] = {}
            for j, name_j in enumerate(order):
                val = float(cov[i, j])
                if np.isfinite(val):
                    row[name_j] = val
            if row:
                out[name_i] = row
        return out or None

    def _display_y_parameters(self) -> list[str]:
        params = list(self._varying_params)
        for definition in self._composite_parameters:
            if definition.name and definition.name not in params:
                params.append(definition.name)
        for kname in self._knight_shift_names:
            if kname not in params:
                params.append(kname)
        return params

    def _selected_composite_parameter_names(self) -> list[str]:
        selected = set(self._selected_y_parameters())
        composite_names = {definition.name for definition in self._composite_parameters}
        return [
            name
            for name in self._display_y_parameters()
            if name in selected and name in composite_names
        ]

    def _selected_knight_trace_names(self) -> list[str]:
        """Currently-selected Knight-shift K traces (removable derived quantities)."""
        selected = set(self._selected_y_parameters())
        return [
            name
            for name in self._display_y_parameters()
            if name in selected and name in self._knight_shift_names
        ]

    def _update_composite_action_buttons(self) -> None:
        has_rows = bool(self._rows)
        selected_composites = self._selected_composite_parameter_names()
        selected_knight = self._selected_knight_trace_names()
        # Edit only applies to composite (expression) parameters.
        self._edit_composite_btn.setEnabled(has_rows and len(selected_composites) == 1)
        # Remove deletes composites and/or Knight-shift K traces.
        self._remove_composite_btn.setEnabled(
            has_rows and bool(selected_composites or selected_knight)
        )

    def _available_composite_source_parameters(self) -> list[str]:
        composite_names = {definition.name for definition in self._composite_parameters}
        names: set[str] = set()
        for row in self._rows:
            names.update(name for name in row.values if name not in composite_names)
        if self._global_params is not None:
            names.update(
                p.name for p in self._global_params if p.name and p.name not in composite_names
            )
        return sorted(names)

    def _apply_composite_parameters_to_rows(
        self,
        rows: list[_FitRow],
        definitions: list[CompositeParameterDefinition],
        global_uncertainties: dict[str, float] | None = None,
        *,
        drop_names: set[str] | None = None,
    ) -> None:
        if not rows:
            return

        names_to_clear = set(drop_names or set())
        names_to_clear.update(definition.name for definition in definitions)
        if names_to_clear:
            for row in rows:
                for name in names_to_clear:
                    row.values.pop(name, None)
                    row.errors.pop(name, None)

        if definitions:
            for row in rows:
                symbol_values = dict(row.values)
                symbol_uncertainties = dict(row.errors)
                if global_uncertainties:
                    for name, value in global_uncertainties.items():
                        symbol_uncertainties.setdefault(name, value)

                for definition in definitions:
                    try:
                        parsed = CompositeExpression(definition.expression)
                        evaluation = parsed.evaluate_with_uncertainty(
                            symbol_values,
                            symbol_uncertainties,
                            covariance=row.covariance,
                        )
                        row.values[definition.name] = float(evaluation.value)
                        row.errors[definition.name] = float(evaluation.uncertainty)
                        symbol_values[definition.name] = float(evaluation.value)
                        symbol_uncertainties[definition.name] = float(evaluation.uncertainty)
                    except (CompositeExpressionError, ValueError):
                        row.values[definition.name] = float("nan")
                        row.errors[definition.name] = float("nan")

        # Knight-shift quantities are derived from the (just-applied) frequencies
        # the same way composites are, so apply them through the same chokepoint.
        self._apply_knight_shift_to_rows(rows)

    # ── Knight-shift conversion (Phase 3) ──────────────────────────────────
    #: Oscillation components can be parameterised by precession *frequency*
    #: (MHz; reference ν_ref = γ_µ·B) or directly by the local *field* (Gauss;
    #: reference is the applied field B itself). Both are converted to a Knight
    #: shift; the base parameter name distinguishes them.
    _KNIGHT_COMPONENT_KINDS = ("frequency", "field")

    def _oscillation_components(self, rows: list[_FitRow]) -> list[tuple[str, str]]:
        """``(parameter_name, kind)`` Knight-convertible components in the rows.

        ``kind`` is ``"frequency"`` or ``"field"``. When the active series carries
        a model-derived observable map (``self._knight_observables``) it is used —
        this excludes components whose ``field`` is the *applied* field (muonium),
        which a bare name match cannot tell apart. For computed / model-less series
        the map is empty and we fall back to matching the base parameter name.
        Ordered by kind then component index so the order matches the model's
        component order — the basis for stable per-component identity.
        """
        present: set[str] = set()
        for row in rows:
            present.update(row.values)

        found: dict[str, str] = {}
        if self._knight_observables:
            for name, kind in self._knight_observables.items():
                if name in present and kind in self._KNIGHT_COMPONENT_KINDS:
                    found[name] = kind
        else:
            for name in present:
                base, _index = split_parameter_name(name)
                if base in self._KNIGHT_COMPONENT_KINDS:
                    found[name] = base

        def _order(item: tuple[str, str]) -> tuple[int, int, str]:
            name, kind = item
            _b, index = split_parameter_name(name)
            kind_rank = self._KNIGHT_COMPONENT_KINDS.index(kind)
            return (kind_rank, int(index) if index is not None else 1, name)

        return sorted(found.items(), key=_order)

    def _oscillation_component_names(self, rows: list[_FitRow]) -> list[str]:
        """Just the component parameter names (see :meth:`_oscillation_components`)."""
        return [name for name, _kind in self._oscillation_components(rows)]

    def _knight_shift_subscript(self, component_name: str) -> str:
        """1-based component ordinal for a component name (for the K symbol)."""
        _, index = split_parameter_name(component_name)
        return index if index is not None else "1"

    @staticmethod
    def _is_knight_shift_name(name: str) -> bool:
        """True for a generated Knight-shift column name (``K[...]``)."""
        return name.startswith("K[") and name.endswith("]")

    @staticmethod
    def _is_legacy_joint_track_name(name: str) -> bool:
        """True for an obsolete joint K(θ) fit track column (``K⟨1⟩``, ``K⟨2⟩``, …).

        Earlier joint fits added these as standalone traces; the joint fit now
        reorders the ``K[...]`` traces in place, so any such column found in a
        saved project is stale and is migrated away on load / re-conversion.
        """
        return name.startswith("K⟨") and name.endswith("⟩")

    def _strip_legacy_joint_tracks(self, rows: list[_FitRow]) -> None:
        """Remove obsolete ``K⟨n⟩`` track columns left by older joint fits."""
        for row in rows:
            for name in [n for n in row.values if self._is_legacy_joint_track_name(n)]:
                row.values.pop(name, None)
                row.errors.pop(name, None)
        if rows is self._rows:
            stale = [n for n in self._model_fits if self._is_legacy_joint_track_name(n)]
            for name in stale:
                self._model_fits.pop(name, None)
                unregister_derived_param_info(name)
                self._registered_knight_labels.discard(name)
            self._varying_params = [
                v for v in self._varying_params if not self._is_legacy_joint_track_name(v)
            ]
            self._selected_y_param_names = [
                v for v in self._selected_y_param_names if not self._is_legacy_joint_track_name(v)
            ]

    def _apply_knight_shift_to_rows(self, rows: list[_FitRow]) -> None:
        """Compute Knight-shift Y-quantities for the rows from the active config.

        Stores ``K[<component>]`` columns (scaled to the resolved display unit) on
        each row, registers display metadata so labels render as ``K_n (ppm)``,
        and (for the active series) records detected component crossings.
        Idempotent: *all* prior ``K[...]`` columns are stripped first, so columns
        persisted in a saved project (or left by a previous config) never linger
        as stale trend parameters.
        """
        for row in rows:
            for name in [n for n in row.values if self._is_knight_shift_name(n)]:
                row.values.pop(name, None)
                row.errors.pop(name, None)
        # Migrate away obsolete K⟨n⟩ track columns from older joint fits.
        self._strip_legacy_joint_tracks(rows)
        self._knight_shift_names = {}
        self._knight_shift_crossings = []
        self._unregister_knight_labels()

        config = self._knight_shift_config
        if not rows or config is None or not config.enabled:
            return
        components = self._oscillation_components(rows)
        if not components:
            return
        kind_by_name = dict(components)

        selected = [
            (name, kind)
            for name, kind in components
            if (not config.components or name in config.components)
        ]
        ref_name = config.reference_component
        if config.reference_mode != REFERENCE_APPLIED_FIELD:
            ref_kind = kind_by_name.get(ref_name)
            if ref_kind is None:
                return  # designated reference is gone; emit nothing rather than guess
            # Only convert components of the *same kind* as the reference: dividing
            # a frequency (MHz) by a field (Gauss), or vice versa, is meaningless.
            selected = [
                (name, kind) for name, kind in selected if name != ref_name and kind == ref_kind
            ]
        if not selected:
            return

        # First pass: compute the dimensionless shifts so AUTO can pick a unit.
        shifts: dict[str, list[tuple[_FitRow, float, float]]] = {}
        for comp, kind in selected:
            per_row: list[tuple[_FitRow, float, float]] = []
            for row in rows:
                k, sigma_k = self._row_knight_shift(row, comp, kind, ref_name)
                per_row.append((row, k, sigma_k))
            shifts[comp] = per_row

        all_fractions = [k for per_row in shifts.values() for _, k, _ in per_row]
        unit = concrete_unit(config.unit, all_fractions)
        scale = scale_for_unit(unit)
        unit_label = label_for_unit(unit)

        for comp, _kind in selected:
            kname = f"K[{comp}]"
            self._knight_shift_names[kname] = comp
            self._register_knight_label(kname, self._knight_shift_subscript(comp), unit_label)
            for row, k, sigma_k in shifts[comp]:
                row.values[kname] = k * scale
                row.errors[kname] = sigma_k * scale

        # Flag component crossings/degeneracies across the scan (detection only;
        # the K traces still follow the raw component labels). Only the active
        # series feeds the panel-global crossing state — running it for every
        # series in the load loop would be wasted work (only the last survives).
        if rows is self._rows:
            self._knight_shift_crossings = self._detect_component_crossings(components)

        # The raw K traces were just (re)generated in component order; if a joint
        # fit is active, reorder them in place so each follows its physical curve
        # (durable across refreshes) and refresh the assignment-swap markers.
        self._apply_joint_reorder(rows)

    def _detect_component_crossings(self, components: list[tuple[str, str]]) -> list[object]:
        """Detect oscillation-component crossings along the active x-axis.

        Components are grouped by kind so the frequency-continuity matcher never
        compares a frequency (MHz) against a field (Gauss); a mixed-unit gap would
        otherwise dominate the tolerance and mask real crossings.
        """
        self._knight_shift_crossing_x_key = None
        x_key = self._effective_x_key()
        events: list[object] = []
        for kind in self._KNIGHT_COMPONENT_KINDS:
            names = [name for name, k in components if k == kind]
            if len(names) < 2:
                continue
            points: list[ScanPoint] = []
            for row in self._rows:
                comps = tuple(
                    Component(frequency=float(row.values[c])) for c in names if c in row.values
                )
                if len(comps) == len(names):
                    # Use the raw scan coordinate (fold=False): crossings follow the
                    # true rotation order; a folded axis would collapse distinct
                    # orientations onto one x and manufacture zero-width crossings.
                    x = self._x_value(row, x_key, fold=False)
                    points.append(ScanPoint(x=float(x), components=comps))
            events.extend(detect_crossings(points))
        self._knight_shift_crossing_x_key = x_key
        return events

    def _row_knight_shift(
        self, row: _FitRow, component_name: str, kind: str, reference_name: str | None
    ) -> tuple[float, float]:
        """Dimensionless Knight shift (and σ) for one component on one row.

        ``kind`` selects the applied-field reference: a *frequency* component
        (MHz) is referenced to the Larmor frequency γ_µ·B, a *field* component
        (Gauss; the fitted local field B_µ) directly to the applied field B —
        i.e. K = (B_µ − B)/B, the most direct form of the shift.
        """
        nu = row.values.get(component_name)
        if nu is None:
            return float("nan"), float("nan")
        sigma_nu = float(row.errors.get(component_name, 0.0) or 0.0)
        if self._knight_shift_config.reference_mode == REFERENCE_APPLIED_FIELD:
            nu_ref = row.field if kind == "field" else larmor_frequency_mhz(row.field)
            return knight_shift(nu, nu_ref, sigma_nu=sigma_nu)
        nu_ref = row.values.get(reference_name)
        if nu_ref is None:
            return float("nan"), float("nan")
        sigma_ref = float(row.errors.get(reference_name, 0.0) or 0.0)
        cov = 0.0
        if row.covariance is not None:
            cov = float(row.covariance.get(component_name, {}).get(reference_name, 0.0))
        return knight_shift(nu, nu_ref, sigma_nu=sigma_nu, sigma_ref=sigma_ref, cov=cov)

    def _register_knight_label(self, kname: str, subscript: str, unit_label: str) -> None:
        """Register display metadata so a Knight-shift quantity renders as K_n (unit).

        The registered name is tracked so it can be removed again
        (:meth:`_unregister_knight_labels`), bounding the global registry's growth
        and clearing stale metadata when the conversion changes or the panel closes.
        """
        sub_unicode = subscript.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
        register_derived_param_info(
            kname,
            plain=f"K{subscript}",
            unicode=f"K{sub_unicode}",
            latex=f"$K_{{{subscript}}}$",
            gle=f"K_{{{subscript}}}",
            unit=unit_label or None,
        )
        self._registered_knight_labels.add(kname)

    def _unregister_knight_labels(self) -> None:
        """Drop every derived-label this panel registered for Knight-shift columns."""
        for kname in self._registered_knight_labels:
            unregister_derived_param_info(kname)
        self._registered_knight_labels = set()

    def _detect_varying_parameters(self, rows: list[_FitRow]) -> list[str]:
        if not rows:
            return []

        composite_names = {definition.name for definition in self._composite_parameters}
        # Knight-shift columns surface only via _knight_shift_names; never let a
        # K[...] column (incl. one persisted in a saved project) be picked up as a
        # generic varying parameter.
        all_names = sorted(
            name
            for name in rows[0].values.keys()
            if name not in composite_names and not self._is_knight_shift_name(name)
        )
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
        # Count distinct values over finite coordinates only: a computed series
        # can mix real-axis rows with off-axis ones (NaN — e.g. the cross-group
        # 'globals' row or the Global summary series), and a phantom NaN bucket
        # would otherwise inflate the unique count and mis-infer the axis.
        field_unique = len(np.unique(np.round(fields[np.isfinite(fields)], 9)))
        temp_unique = len(np.unique(np.round(temps[np.isfinite(temps)], 9)))

        def _finite_span(arr: np.ndarray) -> float:
            # A computed series can sit entirely off an axis (every coordinate
            # NaN — e.g. the cross-fit Global summary). Span over finite values
            # only, defaulting to 0.0, so np.nanmax/min never warn on an all-NaN
            # slice.
            finite = arr[np.isfinite(arr)]
            return float(np.max(finite) - np.min(finite)) if finite.size else 0.0

        field_span = _finite_span(fields)
        temp_span = _finite_span(temps)
        if field_unique > 1 and (field_span > temp_span or temp_unique <= 1):
            return "field"
        if temp_unique > 1:
            return "temperature"
        return "run"

    def _effective_x_key(self) -> str:
        data = self._x_combo.currentData()
        # The parameter-vs-parameter (param:<name>), data-browser custom-column
        # (custom:<id>), and first-class Angle axes all carry their key as item
        # data; the fixed run-level axes carry none and are matched by display
        # text below. The Angle key has no "param:"/"custom:" prefix, so match it
        # explicitly — otherwise selecting Angle silently falls through to the
        # inferred field/temperature/run axis (the whole axis becomes a no-op).
        if isinstance(data, str) and (
            data.startswith("param:") or data.startswith("custom:") or data == self._angle_x_key()
        ):
            return data
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
        if enabled and self._show_components_check.isChecked():
            self._log_y_check.setChecked(False)
            enabled = False
        selected = set(self._selected_y_parameters())
        for name, controls in self._y_controls.items():
            if name in selected:
                controls.log.setChecked(enabled)
        self._refresh_plot()

    def _on_show_components_changed(self, _state: int) -> None:
        """Enable/disable component shading for parameter-model overlays."""
        show_components = self._show_components_check.isChecked()
        if show_components:
            self._log_y_check.blockSignals(True)
            self._log_y_check.setChecked(False)
            self._log_y_check.blockSignals(False)
            for controls in self._y_controls.values():
                controls.log.blockSignals(True)
                controls.log.setChecked(False)
                controls.log.blockSignals(False)

        for controls in self._y_controls.values():
            controls.log.setEnabled(not show_components)

        self._refresh_plot()

    def _clear_plot_labels(self) -> None:
        """Remove all user-placed labels from the parameter plot."""
        self._plot_annotations = []
        self._active_annotation_idx = None
        self._annotation_drag_started = False
        self._refresh_plot()

    def _on_plot_button_press(self, event) -> None:
        """Handle click interactions for parameter-plot labels."""
        if not self._has_mpl:
            return

        if event.button == 3:
            idx = self._detect_annotation_hit(event)
            if idx is not None:
                self._plot_annotations.pop(idx)
                self._refresh_plot()
            return

        if event.button != 1:
            return

        if self._add_label_btn.isChecked():
            self._add_annotation_at_event(event)
            return

        idx = self._detect_annotation_hit(event)
        if idx is not None:
            self._active_annotation_idx = idx
            self._annotation_drag_started = False

    def _on_plot_motion(self, event) -> None:
        """Drag labels on the parameter plot."""
        if (
            self._active_annotation_idx is None
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            return

        ann = self._plot_annotations[self._active_annotation_idx]
        axis_tag = str(ann.get("axis_tag", "main"))
        current_tag = self._axes_tag_map.get(id(event.inaxes), "main")
        if axis_tag != current_tag:
            return

        ann["x"] = float(event.xdata)
        ann["y"] = float(event.ydata)
        self._annotation_drag_started = True
        artist = ann.get("artist")
        if artist is not None:
            artist.set_position((ann["x"], ann["y"]))
            self._canvas.draw_idle()

    def _on_plot_button_release(self, event) -> None:
        """Finish drag, edit label on double click."""
        if self._active_annotation_idx is None:
            return

        idx = self._active_annotation_idx
        was_drag = self._annotation_drag_started
        self._active_annotation_idx = None
        self._annotation_drag_started = False

        if not was_drag and event.button == 1 and getattr(event, "dblclick", False):
            current = str(self._plot_annotations[idx].get("text", ""))
            text, ok = QInputDialog.getText(self, "Edit Label", "Label text:", text=current)
            if ok and text.strip():
                self._plot_annotations[idx]["text"] = text.strip()
                self._refresh_plot()

    def _detect_annotation_hit(self, event) -> int | None:
        """Return annotation index under cursor, if any."""
        if event.inaxes is None:
            return None
        for idx, ann in enumerate(self._plot_annotations):
            artist = ann.get("artist")
            if artist is None:
                continue
            contains, _ = artist.contains(event)
            if contains:
                return idx
        return None

    def _add_annotation_at_event(self, event) -> None:
        """Prompt for text and place annotation at click location."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        text, ok = QInputDialog.getText(self, "Add Label", "Label text:")
        if not ok or not text.strip():
            return

        axis_tag = self._axes_tag_map.get(id(event.inaxes), "main")
        self._plot_annotations.append(
            {
                "x": float(event.xdata),
                "y": float(event.ydata),
                "text": text.strip(),
                "axis_tag": axis_tag,
                "artist": None,
            }
        )
        self._add_label_btn.setChecked(False)
        self._refresh_plot()

    #: On-plot crossing markers are suppressed for now: when components run close
    #: together throughout a scan the near-degeneracy flag fires at almost every
    #: angle and the markers swamp the plot. Crossings are still *detected* (the
    #: dialog reports the count and the events feed the future realignment step);
    #: re-enable the markers once that lands and presents them more robustly.
    _DRAW_CROSSING_MARKERS = False

    @staticmethod
    def _cluster_crossing_bands(events: list[object]) -> list[tuple[float, float]]:
        """Merge neighbouring crossing transitions into ``(lo, hi)`` bands.

        Each crossing spans one scan step ``[x_left, x_right]``; runs of adjacent
        crossings (e.g. repeated near-degeneracies around one physical crossing)
        are merged into a single band so each crossing shows once instead of as
        several near-identical lines. An isolated crossing is padded to a thin,
        visible band.
        """
        intervals = sorted(
            (float(min(e.x_left, e.x_right)), float(max(e.x_left, e.x_right))) for e in events
        )
        if not intervals:
            return []
        widths = [hi - lo for lo, hi in intervals if hi > lo]
        step = float(np.median(widths)) if widths else 0.0
        tol = 1.5 * step  # merge crossings within ~1.5 scan steps of each other
        pad = 0.25 * step  # so a lone crossing is a visible band, not a hairline
        bands: list[tuple[float, float]] = []
        lo, hi = intervals[0]
        for next_lo, next_hi in intervals[1:]:
            if next_lo - hi <= tol:
                hi = max(hi, next_hi)
            else:
                bands.append((lo, hi))
                lo, hi = next_lo, next_hi
        bands.append((lo, hi))
        return [(low - pad, high + pad) for low, high in bands]

    def _draw_knight_shift_crossings(self, axes_by_tag: dict[str, object], x_key: str) -> None:
        """Shade angle bands where oscillation components cross.

        Drawn only while a Knight-shift conversion is active and the plotted
        x-axis matches the one the crossings were computed against (so changing
        the x-axis doesn't leave stale markers). Adjacent crossings are merged
        into one band (see :meth:`_cluster_crossing_bands`).
        """
        if not self._DRAW_CROSSING_MARKERS:
            return
        if not self._knight_shift_crossings or not self._knight_shift_names:
            return
        if x_key != self._knight_shift_crossing_x_key:
            return
        bands = self._cluster_crossing_bands(self._knight_shift_crossings)
        if not bands:
            return
        for ax in axes_by_tag.values():
            for first, (lo, hi) in enumerate(bands):
                ax.axvspan(
                    lo,
                    hi,
                    color="#b5651d",
                    alpha=0.12,
                    zorder=1,
                    label="component crossing" if first == 0 else None,
                )

    def _draw_plot_annotations(self, axes_by_tag: dict[str, object]) -> None:
        """Draw stored annotations on currently visible parameter axes."""
        for ann in self._plot_annotations:
            axis_tag = str(ann.get("axis_tag", "main"))
            ax = axes_by_tag.get(axis_tag)
            if ax is None:
                ann["artist"] = None
                continue
            artist = ax.text(
                float(ann.get("x", 0.0)),
                float(ann.get("y", 0.0)),
                str(ann.get("text", "")),
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85},
                zorder=6,
            )
            ann["artist"] = artist

    def _on_x_axis_changed(self, *_args: object) -> None:
        self._update_x_axis_auto_hint()
        self._update_angle_fold_visibility()
        self._update_joint_fit_button()
        self._refresh_views()

    def _angle_axis_active(self) -> bool:
        """Whether the Angle field is the current trend x-axis."""
        angle_key = self._angle_x_key()
        return angle_key is not None and self._effective_x_key() == angle_key

    def _update_angle_fold_visibility(self) -> None:
        """Show the fold control only while the Angle axis is selected."""
        visible = self._angle_axis_active()
        self._angle_fold_label.setVisible(visible)
        self._angle_fold_combo.setVisible(visible)

    def _on_angle_fold_changed(self, *_args: object) -> None:
        period = self._angle_fold_combo.currentData()
        self._angle_wrap_period = float(period) if period is not None else None
        if self._angle_axis_active():
            self._refresh_views()

    def _reset_angle_fold(self) -> None:
        """Clear angle folding (combo + attribute) — used on New Project."""
        self._restore_angle_fold(None)

    def _restore_angle_fold(self, period: object) -> None:
        """Restore the angle-fold period from saved state (combo + attribute)."""
        value = float(period) if isinstance(period, (int, float)) else None
        self._angle_wrap_period = value
        idx = self._angle_fold_combo.findData(value)
        if idx >= 0:
            with QSignalBlocker(self._angle_fold_combo):
                self._angle_fold_combo.setCurrentIndex(idx)
        self._update_angle_fold_visibility()

    def _update_x_axis_auto_hint(self) -> None:
        if self._x_combo.currentText() != "Auto":
            self._x_auto_hint.setText("")
            return
        inferred_label = {"field": "(B)", "temperature": "(T)", "run": "(Run)"}
        self._x_auto_hint.setText(inferred_label.get(self._inferred_x_key, "(Run)"))

    def _update_custom_x_skip_note(self, x_key: str, x_vals: np.ndarray) -> None:
        """Note how many runs a custom x-axis drops (empty/non-numeric values).

        Only custom columns can carry non-numeric/empty abscissae, so the note is
        scoped to them; matplotlib already omits the NaN points, this just tells
        the user *why* some runs are missing. Cleared when nothing is dropped.
        """
        if _x_custom_id(x_key) is None and x_key != self._angle_x_key():
            return
        values = np.asarray(x_vals, dtype=float)
        total = int(values.size)
        dropped = int(np.count_nonzero(~np.isfinite(values))) if total else 0
        if dropped:
            self._x_auto_hint.setText(f"⚠ {dropped}/{total} skipped (empty/non-numeric)")
        else:
            self._x_auto_hint.setText("")

    def _reset_stale_custom_x_axis(self) -> None:
        """Fall back to Auto when the selected custom/Angle axis has no values here.

        A custom column (e.g. "Current (A)") created for one project/batch
        persists in the x-axis combo and stays selected when an unrelated dataset
        is trended; every new run is then "skipped (empty/non-numeric)" and the
        plot is empty. When the active axis is such a free-text column and *none*
        of the freshly loaded rows carry a finite value for it, revert to Auto
        (which infers Run/T/B) so a new dataset gets a sensible default instead
        of a stale, empty abscissa.
        """
        x_key = self._effective_x_key()
        is_free_text = _x_custom_id(x_key) is not None or x_key == self._angle_x_key()
        if not is_free_text or not self._rows:
            return
        values = np.array(
            [self._x_value(row, x_key, fold=False) for row in self._rows], dtype=float
        )
        if values.size and np.any(np.isfinite(values)):
            return
        idx = self._x_combo.findText("Auto")
        if idx >= 0 and self._x_combo.currentIndex() != idx:
            self._x_combo.setCurrentIndex(idx)

    def _rebuild_x_axis_combo(self) -> None:
        """Re-populate the X-axis combo: the fixed run-level axes plus every
        currently-trendable fitted parameter (param-vs-param trending, item 1).

        Parameter entries carry ``param:<name>`` as their item data so the
        selection survives label collisions; the current selection is preserved
        across the rebuild by data (param) or text (run-level axis).
        """
        combo = self._x_combo
        prev_data = combo.currentData()
        prev_text = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for label in ("Auto", "𝐵 (G)", "𝑇 (K)", "Run"):
            combo.addItem(label)
        # The Angle field is a first-class run-level axis, listed with the fixed
        # axes (it carries its column id as item data, like the custom columns).
        if self._angle_x_field is not None:
            combo.addItem(self._angle_x_field[0], userData=self._angle_x_field[1])
        for name in self._display_y_parameters():
            combo.addItem(_format_param_label(name), userData=f"param:{name}")
        # Data-browser custom columns (param:<…> and custom:<…> both carry their
        # key as item data so the selection survives label collisions / renames).
        for label, key in self._custom_x_fields:
            combo.addItem(label, userData=key)
        restored = False
        if isinstance(prev_data, str) and prev_data:
            # Restore by item data (param:/custom:/angle keys) so the selection
            # survives label collisions and renames.
            idx = combo.findData(prev_data)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                restored = True
        if not restored:
            idx = combo.findText(prev_text)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def set_custom_x_fields(self, fields: list[tuple[str, str]]) -> None:
        """Set the data-browser custom columns offered as the trend x-axis.

        ``fields`` is a list of ``(display_label, "custom:<id>")`` pairs. The combo
        is rebuilt (selection preserved); if a custom column is the active x-axis,
        the plot is refreshed so a rename or value change is reflected.
        """
        normalized = [(str(label), str(key)) for label, key in fields]
        if normalized == self._custom_x_fields:
            return
        active_is_custom = _x_custom_id(self._effective_x_key()) is not None
        self._custom_x_fields = normalized
        self._rebuild_x_axis_combo()
        if active_is_custom:
            self._refresh_plot()

    def relink_custom_values(self, values_by_run: dict[int, dict[str, str]]) -> None:
        """Refresh every trend row's custom-column values from the live browser.

        A completed batch fit snapshots each run's custom-column text when it
        records the series. If the user adds or populates a custom column
        *afterwards*, those snapshots are stale and the column trends as all-NaN
        ("N/N skipped — empty/non-numeric") with no recovery short of re-running
        the batch. Re-pushing the current per-run values here re-links existing
        results to the new/edited column live, so the ordering trap dissolves.

        ``values_by_run`` maps run number → the run's full ``custom_fields`` dict
        (``custom:<id>``/``angle`` → text). Rows whose run has a live dataset are
        re-synced wholesale (so cleared values propagate too); rows with no live
        dataset keep their snapshot.
        """
        if not values_by_run:
            return
        changed = False
        for row in self._iter_all_fit_rows():
            live = values_by_run.get(row.run_number)
            if live is None:
                continue
            if row.custom_values != live:
                row.custom_values = dict(live)
                changed = True
        if not changed:
            return
        # Keep the active group's snapshot in step with the mutated rows so a
        # later view rebuild (group switch / re-plot) doesn't resurrect stale
        # values, then redraw if a custom/Angle column is the live abscissa.
        self._sync_active_group_state()
        x_key = self._effective_x_key()
        if _x_custom_id(x_key) is not None or x_key == self._angle_x_key():
            self._refresh_plot()

    def _iter_all_fit_rows(self) -> Iterator[_FitRow]:
        """Yield every distinct :class:`_FitRow` held by the panel.

        Covers the active view (``_rows``) and every stored series group, so a
        custom-column re-link reaches inactive series too. Rows are de-duplicated
        by identity because the active ``_rows`` share objects with their group.
        """
        seen: set[int] = set()
        for group in self._group_fit_results.values():
            for row in group.rows:
                if id(row) not in seen:
                    seen.add(id(row))
                    yield row
        for row in self._rows:
            if id(row) not in seen:
                seen.add(id(row))
                yield row

    def set_angle_x_field(self, field: tuple[str, str] | None) -> None:
        """Set (or clear) the special Angle field offered as a first-class x-axis.

        ``field`` is a ``(display_label, key)`` pair or None. Rebuilds the combo
        (selection preserved); if Angle is the active x-axis, the plot is refreshed
        so a value change or the field's removal is reflected.
        """
        normalized = (str(field[0]), str(field[1])) if field else None
        if normalized == self._angle_x_field:
            return
        active_is_angle = self._effective_x_key() == self._angle_x_key()
        self._angle_x_field = normalized
        self._rebuild_x_axis_combo()
        self._update_angle_fold_visibility()
        if active_is_angle:
            self._refresh_plot()

    def _angle_x_key(self) -> str | None:
        """Return the key identifying the first-class Angle x-axis, or None."""
        return self._angle_x_field[1] if self._angle_x_field is not None else None

    def _angle_fold_suffix(self) -> str:
        """Axis-label suffix noting the active angle fold (empty when off)."""
        if self._angle_wrap_period is None:
            return ""
        return f" (folded {self._angle_wrap_period:g}°)"

    def _custom_x_labels(self) -> dict[str, str]:
        """Map each free-text x-axis key to its display label (custom + Angle).

        The Angle label carries the fold note so every consumer (GLE export,
        export column headers) stays consistent with the on-screen plot.
        """
        labels = {key: label for label, key in self._custom_x_fields}
        if self._angle_x_field is not None:
            labels[self._angle_x_field[1]] = self._angle_x_field[0] + self._angle_fold_suffix()
        return labels

    def _export_abscissa_key(self) -> str | None:
        """Active x-axis key when it is a free-text column (Angle or custom).

        These are the only x-axes not already emitted as a fixed Run/B/T/param
        column, so an export must add them explicitly. Returns None otherwise.
        """
        x_key = self._effective_x_key()
        if x_key == self._angle_x_key() or _x_custom_id(x_key) is not None:
            return x_key
        return None

    def _export_abscissa_column(self) -> tuple[str, str] | None:
        """``(x_key, header_label)`` for the Angle/custom abscissa export column.

        The single source of the trailing free-text x-axis column added to the
        table, TSV, and GLE data file (label folded as displayed). ``None`` when
        the x-axis is already a fixed Run/B/T/param column.
        """
        key = self._export_abscissa_key()
        if key is None:
            return None
        return key, self._custom_x_labels().get(key, key)

    def _shared_held_constant_params(self) -> list[str]:
        """Names of Global-classified (shared) params held constant, hence flat.

        A batch fit shares one value across every run for each ``global``-role
        parameter, so it never varies and is dropped from the trendable Y list.
        These are exactly the names worth flagging: the user who classified an
        amplitude as Global (the Batch-tab default for the leading amplitude) and
        then tries to trend it would otherwise find it silently absent.
        """
        if self._global_params is None:
            return []
        varying = set(self._varying_params)
        names: list[str] = []
        for param in self._global_params:
            if getattr(param, "fixed", False):
                continue
            name = str(param.name)
            if name in varying or name in names:
                continue
            names.append(name)
        return names

    def _update_global_param_hint(self) -> None:
        """Show/hide the hint pointing at Global params that won't trend."""
        names = self._shared_held_constant_params()
        if not names:
            self._global_param_hint.setText("")
            self._global_param_hint.setVisible(False)
            return
        labels = ", ".join(_format_param_label(name) for name in names)
        subject = "it is" if len(names) == 1 else "they are"
        self._global_param_hint.setText(
            f"{labels} fitted as Global (one shared value), so {subject} held "
            "constant and not shown as a trend. To trend across runs, set to "
            "Local in the Batch tab and re-fit."
        )
        self._global_param_hint.setVisible(True)

    def _rebuild_y_controls(self, *, preferred_selected: list[str] | None = None) -> None:
        self._y_selector_table.blockSignals(True)
        self._y_selector_table.clearContents()
        self._y_selector_table.setRowCount(0)

        self._y_controls = {}

        display_params = self._display_y_parameters()

        # Keep the X-axis selector's parameter entries in sync with the
        # trendable parameters (param-vs-param trending, item 1).
        self._rebuild_x_axis_combo()
        # Refresh the "shared param held constant" hint whenever the trendable
        # set changes (group switch, new fit, restore) — both exit paths below.
        self._update_global_param_hint()

        if not display_params:
            self._set_y_table_visible_rows(3)
            self._y_selector_table.blockSignals(False)
            return

        self._y_selector_table.setRowCount(len(display_params))

        for idx, name in enumerate(display_params):
            name_item = QTableWidgetItem(_format_param_label(name))
            name_item.setData(Qt.ItemDataRole.UserRole, name)
            # The name column elides; back the truncated text with the full label
            # on hover so nothing is lost when the inspector is narrow.
            name_item.setToolTip(_format_param_label(name))
            self._y_selector_table.setItem(idx, 0, name_item)

            fit_button = QPushButton("Model Fit")
            fit_button.setMinimumWidth(
                fit_button.fontMetrics().horizontalAdvance("Model Fit*") + 36
            )
            # Keep keyboard focus on the table's selection model: a focusable cell
            # widget steals focus on interaction and can collapse a multi-row
            # selection built with Shift+Arrow.
            fit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            fit_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
            fit_button.clicked.connect(
                lambda _checked=False, p=name: self._open_model_fit_dialog(p)
            )
            self._y_selector_table.setCellWidget(idx, 1, fit_button)

            log_check = QCheckBox("log")
            log_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            log_check.stateChanged.connect(self._refresh_plot)
            log_check.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            log_control_width = log_check.fontMetrics().horizontalAdvance("log") + 28
            log_check.setMinimumWidth(log_control_width)

            log_container = QWidget()
            log_container.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            log_container.setMinimumWidth(log_control_width + 8)
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
        fit_column_width = max(
            (
                max(
                    controls.fit_button.minimumWidth(),
                    controls.fit_button.minimumSizeHint().width(),
                )
                for controls in self._y_controls.values()
            ),
            default=120,
        )
        log_column_width = max(
            (
                max(
                    controls.log.minimumWidth(),
                    controls.log.sizeHint().width(),
                )
                + 12
                for controls in self._y_controls.values()
            ),
            default=56,
        )
        self._y_selector_table.setColumnWidth(1, fit_column_width)
        self._y_selector_table.setColumnWidth(2, log_column_width)
        # Floor the table just wide enough for the two fixed action columns, a
        # short name stub, and the vertical scrollbar — NOT the full longest
        # name. The name column stretches and elides, so a long parameter name
        # must not force the table (and panel) wider than the dock; that is what
        # used to spawn the horizontal scrollbar that hid the action columns.
        frame = 2 * self._y_selector_table.frameWidth()
        vscroll_width = self._y_selector_table.style().pixelMetric(
            self._y_selector_table.style().PixelMetric.PM_ScrollBarExtent,
        )
        name_stub_width = 48
        minimum_width = (
            name_stub_width + fit_column_width + log_column_width + frame + vscroll_width + 8
        )
        self._y_selector_table.setMinimumWidth(minimum_width)
        self._set_y_table_visible_rows(3)

        preferred = [name for name in (preferred_selected or []) if name in display_params]
        if preferred:
            for idx, name in enumerate(display_params):
                item = self._y_selector_table.item(idx, 0)
                if item is not None and name in preferred:
                    item.setSelected(True)
        elif self._y_selector_table.rowCount() > 0:
            item = self._y_selector_table.item(0, 0)
            if item is not None:
                item.setSelected(True)

        self._y_selector_table.blockSignals(False)
        self._selected_y_param_names = self._selected_y_parameters()
        self._update_composite_action_buttons()

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
        self._y_selector_table.setMinimumHeight(0)
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

    def _show_composite_parameter_dialog(
        self,
        *,
        initial_definition: CompositeParameterDefinition | None = None,
    ) -> CompositeParameterDefinition | None:
        if not self._rows:
            return None

        available_parameters = self._available_composite_source_parameters()
        if not available_parameters:
            QMessageBox.information(
                self,
                "Create Composite Parameter",
                "No fitted parameters are available for building a composite expression.",
            )
            return None

        rows = sorted(self._rows, key=lambda row: row.run_number)
        preview_row = rows[0]
        preview_uncertainties = dict(preview_row.errors)
        for name, value in self._global_param_uncertainties.items():
            preview_uncertainties.setdefault(name, value)

        dialog = CompositeParameterDialog(
            available_parameters=available_parameters,
            existing_parameter_names=self._display_y_parameters(),
            initial_definition=initial_definition,
            preview_values=dict(preview_row.values),
            preview_uncertainties=preview_uncertainties,
            preview_covariance=preview_row.covariance,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return dialog.composite_definition()

    def _refresh_after_composite_change(self, *, preferred_selected: list[str]) -> None:
        display_params = set(self._display_y_parameters())
        self._model_fits = {k: v for k, v in self._model_fits.items() if k in display_params}
        self._selected_y_param_names = [
            name for name in preferred_selected if name in display_params
        ]
        self._sync_active_group_state()
        self._rebuild_y_controls(preferred_selected=self._selected_y_param_names)
        self._refresh_model_fit_button_labels()
        self._refresh_views()

    def _open_composite_parameter_dialog(self) -> None:
        definition = self._show_composite_parameter_dialog()
        if definition is None:
            return

        self._composite_parameters = [
            existing for existing in self._composite_parameters if existing.name != definition.name
        ]
        self._composite_parameters.append(definition)
        self._apply_composite_parameters_to_rows(
            self._rows,
            self._composite_parameters,
            self._global_param_uncertainties,
        )

        preferred_selected = list(self._selected_y_param_names)
        if definition.name not in preferred_selected:
            preferred_selected.append(definition.name)
        self._refresh_after_composite_change(preferred_selected=preferred_selected)

    # ── Knight-shift public API (driven by the dedicated dialog, Phase 3d) ──
    def knight_shift_config(self) -> KnightShiftConfig:
        """Return a copy of the active Knight-shift configuration."""
        return KnightShiftConfig.from_dict(self._knight_shift_config.to_dict())

    def available_oscillation_components(self) -> list[str]:
        """Frequency parameter names available to convert, in component order."""
        return self._oscillation_component_names(self._rows)

    def knight_shift_crossings(self) -> list[object]:
        """Crossing events flagged on the active series (for annotation/reporting)."""
        return list(self._knight_shift_crossings)

    def set_knight_shift_config(self, config: KnightShiftConfig) -> None:
        """Apply a new Knight-shift configuration and refresh the trend.

        Newly-generated K quantities are auto-selected (and the previous ones
        de-selected) so enabling the conversion immediately shows the result.
        """
        previous_k = set(self._knight_shift_names)
        # Re-running the conversion regenerates the raw, per-component K traces, so
        # any prior joint-fit reorder is dropped (and its markers turned off). Drop
        # the joint fit's per-curve overlays too: they are keyed by the K[...] trace
        # names, which survive the conversion, so otherwise a stale K(θ) curve would
        # be drawn over the regenerated raw data.
        if self._joint_fit:
            for name in self._joint_fit.get("curves", {}):
                self._model_fits.pop(name, None)
        self._joint_fit = None
        self._DRAW_CROSSING_MARKERS = False
        self._knight_shift_config = config
        self._apply_composite_parameters_to_rows(
            self._rows,
            self._composite_parameters,
            self._global_param_uncertainties,
        )
        preferred = [n for n in self._selected_y_param_names if n not in previous_k]
        preferred.extend(n for n in self._knight_shift_names if n not in preferred)
        self._refresh_after_composite_change(preferred_selected=preferred)

    def _open_knight_shift_dialog(self) -> None:
        components = self.available_oscillation_components()
        if not components:
            QMessageBox.information(
                self,
                "Knight Shift",
                "No oscillation-frequency components were found in this series.",
            )
            return
        dialog = KnightShiftDialog(
            available_components=components,
            config=self._knight_shift_config,
            crossing_count=len(self._knight_shift_crossings),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        config = dialog.knight_shift_config()
        if config is not None:
            self.set_knight_shift_config(config)

    # ── Joint K(θ) fit with per-angle assignment (Phase 6) ─────────────────
    def _open_joint_knight_fit_dialog(self) -> None:
        traces = self._selected_knight_traces()
        if len(traces) < 2 or not self._angle_axis_active():
            QMessageBox.information(
                self,
                "Joint K(θ) Fit",
                "Select at least two Knight-shift traces with Angle as the x-axis.",
            )
            return
        dialog = KnightJointFitDialog(n_curves=len(traces), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        config = dialog.joint_fit_config()
        if config is not None:
            model_name, max_iter = config
            self._run_joint_knight_fit(traces, model_name, max_iter)

    def _run_joint_knight_fit(self, traces: list[str], model_name: str, max_iter: int) -> None:
        """Fit the selected Knight-shift traces jointly, off the GUI thread.

        The per-angle assignment must follow the *true rotation order*, so the
        abscissa is taken with ``fold=False`` (the raw scan coordinate), exactly
        as crossing detection does: folding would collapse distinct orientations
        onto one angle and scramble the continuity / crossing-swap seeds. The
        classification-EM fit runs many least-squares fits across several seeds,
        so it is dispatched to a worker rather than blocking the GUI thread.
        """
        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key, fold=False))
        angles = [self._x_value(r, x_key, fold=False) for r in rows]
        values = [[r.values.get(name, float("nan")) for name in traces] for r in rows]
        errors = [[r.errors.get(name, float("nan")) for name in traces] for r in rows]

        self._joint_fit_compute_active = True
        self._joint_knight_btn.setEnabled(False)
        if self._trend_overlay is not None:
            self._trend_overlay.show_message("Fitting K(θ)…")
        self._tasks.start(
            lambda _worker: fit_assigned_angular_curves(
                angles, values, errors, model_name=model_name, max_iter=max_iter
            ),
            on_finished=lambda result: self._on_joint_knight_fit_ready(
                result, rows, traces, model_name
            ),
            on_error=self._on_joint_knight_fit_error,
        )

    def _on_joint_knight_fit_ready(
        self,
        result: AngularAssignmentResult,
        rows: list[_FitRow],
        traces: list[str],
        model_name: str,
    ) -> None:
        self._joint_fit_compute_active = False
        if self._trend_overlay is not None:
            self._trend_overlay.hide()
        self._update_joint_fit_button()
        if not result.curves:
            QMessageBox.information(
                self, "Joint K(θ) Fit", result.message or "The joint fit produced no curves."
            )
            return
        self._apply_joint_knight_fit(result, rows, traces, model_name)

    def _on_joint_knight_fit_error(self, message: str) -> None:
        self._joint_fit_compute_active = False
        if self._trend_overlay is not None:
            self._trend_overlay.hide()
        self._update_joint_fit_button()
        QMessageBox.warning(self, "Joint K(θ) Fit", f"The joint fit failed: {message}")

    def _apply_joint_knight_fit(
        self,
        result: AngularAssignmentResult,
        rows: list[_FitRow],
        traces: list[str],
        model_name: str,
    ) -> None:
        """Reorder the existing K traces in place and overlay the per-curve fits.

        No new traces are created: each selected ``K[...]`` trace is replaced by
        the continuous physical curve assigned to it (the raw, per-component
        ordering is regenerated by re-running the conversion). The component→curve
        permutation is stored per run so the reorder survives trend refreshes.
        """
        # Permutation per run (stable across refreshes / reload), built from the
        # angle-sorted assignment the core returned.
        assignment = {
            int(row.run_number): tuple(int(c) for c in result.assignment[i])
            for i, row in enumerate(rows)
        }
        # Per-curve overlays, keyed on the trace each curve now occupies. They are
        # held inside ``_joint_fit`` (and serialised with it) so the joint fit is
        # reconstructed deterministically on every reload / group switch, rather
        # than relying on the per-group ``model_fits`` round-trip surviving.
        angles = [self._x_value(r, self._effective_x_key()) for r in rows]
        finite = [a for a in angles if np.isfinite(a)]
        x_min = min(finite) if finite else None
        x_max = max(finite) if finite else None
        curves: dict[str, ParameterModelFit] = {}
        for curve, name in enumerate(traces):
            fit_result = result.curves[curve]
            curves[name] = ParameterModelFit(
                parameter_name=name,
                x_key=self._effective_x_key(),
                ranges=[
                    ModelFitRange(
                        x_min=x_min,
                        x_max=x_max,
                        model=ParameterCompositeModel([model_name]),
                        parameters=fit_result.parameters,
                        result=fit_result,
                    )
                ],
                active=True,
            )

        self._joint_fit = {
            "traces": list(traces),
            "assignment": assignment,
            "model_name": model_name,
            "curves": curves,
        }

        self._apply_joint_reorder(self._rows)
        # The suppressed markers are exactly what the joint fit now justifies: show
        # them at the angles where the assignment swaps.
        self._DRAW_CROSSING_MARKERS = True

        self._sync_active_group_state()
        # Keep the user's existing K-trace selection (no new traces were added).
        self._rebuild_y_controls(preferred_selected=traces)
        self._refresh_model_fit_button_labels()
        self._refresh_views()

    def _apply_joint_reorder(self, rows: list[_FitRow]) -> None:
        """Reorder the joint-fit K traces in place per the stored per-run permutation.

        Re-applied after every Knight-shift (re)generation so the reorder is durable
        across refreshes. Also refreshes the assignment-swap crossing markers. A
        no-op when no joint fit is active or its traces are absent from the rows.
        """
        if not self._joint_fit:
            return
        traces = list(self._joint_fit["traces"])
        assignment: dict[int, tuple[int, ...]] = self._joint_fit["assignment"]
        present = {name for row in rows for name in row.values}
        if not all(name in present for name in traces):
            return
        n = len(traces)
        for row in rows:
            perm = assignment.get(int(row.run_number))
            if perm is None or len(perm) != n:
                continue
            # perm[component] = curve; place each component's raw value on its curve's trace.
            old_v = {name: row.values.get(name, float("nan")) for name in traces}
            old_e = {name: row.errors.get(name, float("nan")) for name in traces}
            for component, curve in enumerate(perm):
                row.values[traces[curve]] = old_v[traces[component]]
                row.errors[traces[curve]] = old_e[traces[component]]
        if rows is self._rows:
            # Re-inject the per-curve overlays and markers from the joint-fit state
            # so they survive a group switch / reload that rebuilt _model_fits from
            # the (possibly lossy) per-group snapshot.
            for name, model_fit in self._joint_fit.get("curves", {}).items():
                self._model_fits[name] = model_fit
            self._DRAW_CROSSING_MARKERS = True
        self._refresh_joint_fit_markers(rows)

    def _refresh_joint_fit_markers(self, rows: list[_FitRow]) -> None:
        """Mark angles where the joint-fit assignment swaps between adjacent points."""
        if not self._joint_fit:
            return
        x_key = self._effective_x_key()
        assignment: dict[int, tuple[int, ...]] = self._joint_fit["assignment"]
        ordered = sorted(rows, key=lambda r: self._x_value(r, x_key))
        events: list[object] = []
        for k in range(len(ordered) - 1):
            a = assignment.get(int(ordered[k].run_number))
            b = assignment.get(int(ordered[k + 1].run_number))
            if a is None or b is None or a == b:
                continue
            changed = [c for c in range(len(a)) if a[c] != b[c]]
            pair = (changed[0], changed[1]) if len(changed) >= 2 else (changed[0], changed[0])
            events.append(
                CrossingEvent(
                    k,
                    float(self._x_value(ordered[k], x_key)),
                    float(self._x_value(ordered[k + 1], x_key)),
                    pair,
                    "order_swap",
                )
            )
        self._knight_shift_crossings = events
        self._knight_shift_crossing_x_key = x_key

    def _serialize_joint_fit(self) -> dict | None:
        """Serialise the active joint-fit reorder (permutation, traces, overlays)."""
        if not self._joint_fit:
            return None
        return {
            "traces": list(self._joint_fit["traces"]),
            "model_name": self._joint_fit["model_name"],
            "assignment": {
                str(run): list(perm) for run, perm in self._joint_fit["assignment"].items()
            },
            "curves": self._serialize_model_fits_mapping(self._joint_fit.get("curves", {})),
        }

    def _deserialize_joint_fit(self, data: object) -> dict | None:
        if not isinstance(data, dict):
            return None
        try:
            traces = [str(t) for t in data.get("traces", [])]
            assignment = {
                int(run): tuple(int(c) for c in perm)
                for run, perm in (data.get("assignment") or {}).items()
            }
        except (TypeError, ValueError):
            return None
        if not traces or not assignment:
            return None
        return {
            "traces": traces,
            "assignment": assignment,
            "model_name": str(data.get("model_name", "")),
            "curves": self._deserialize_model_fits(data.get("curves", {})),
        }

    def _edit_selected_composite_parameter(self) -> None:
        selected = self._selected_composite_parameter_names()
        if len(selected) != 1:
            QMessageBox.information(
                self,
                "Edit Composite Parameter",
                "Select exactly one composite parameter to edit.",
            )
            return

        selected_name = selected[0]
        initial_definition = next(
            (
                definition
                for definition in self._composite_parameters
                if definition.name == selected_name
            ),
            None,
        )
        if initial_definition is None:
            return

        updated_definition = self._show_composite_parameter_dialog(
            initial_definition=initial_definition,
        )
        if updated_definition is None:
            return

        self._composite_parameters = [
            definition
            for definition in self._composite_parameters
            if definition.name not in {selected_name, updated_definition.name}
        ]
        self._composite_parameters.append(updated_definition)

        self._apply_composite_parameters_to_rows(
            self._rows,
            self._composite_parameters,
            self._global_param_uncertainties,
            drop_names={selected_name},
        )

        self._model_fits.pop(selected_name, None)
        if updated_definition.name != selected_name:
            self._model_fits.pop(updated_definition.name, None)

        preferred_selected = [
            updated_definition.name if name == selected_name else name
            for name in self._selected_y_param_names
        ]
        if updated_definition.name not in preferred_selected:
            preferred_selected.append(updated_definition.name)

        self._refresh_after_composite_change(preferred_selected=preferred_selected)

    def _remove_selected_composite_parameters(self) -> None:
        composites = self._selected_composite_parameter_names()
        knight = self._selected_knight_trace_names()
        selected = composites + knight
        if not selected:
            return

        if len(selected) == 1:
            message = f"Remove '{selected[0]}'?"
        else:
            message = f"Remove selected quantities ({', '.join(selected)})?"
        confirm = QMessageBox.question(self, "Remove", message)
        if confirm != QMessageBox.StandardButton.Yes:
            return

        preferred_selected = [
            name for name in self._selected_y_param_names if name not in set(selected)
        ]

        if composites:
            names_to_remove = set(composites)
            self._composite_parameters = [
                definition
                for definition in self._composite_parameters
                if definition.name not in names_to_remove
            ]
            for name in names_to_remove:
                self._model_fits.pop(name, None)
            self._apply_composite_parameters_to_rows(
                self._rows,
                self._composite_parameters,
                self._global_param_uncertainties,
                drop_names=names_to_remove,
            )

        if knight:
            # Drop the components backing the selected K traces from the conversion
            # (and any joint fit that used them); set_knight_shift_config re-applies
            # and refreshes, so it is the last step.
            self._remove_knight_traces(knight, preferred_selected=preferred_selected)
            return

        self._refresh_after_composite_change(preferred_selected=preferred_selected)

    def _remove_knight_traces(self, knames: list[str], *, preferred_selected: list[str]) -> None:
        """Delete the selected Knight-shift K traces by excluding their components.

        The K traces are derived by the Knight-shift conversion, so removing one
        means dropping its component from the conversion's component list (an
        explicit allow-list). When the last component goes the conversion is
        disabled. ``set_knight_shift_config`` re-applies and refreshes (and clears
        any joint fit, which spans all components).
        """
        comps_to_remove = {
            self._knight_shift_names[name] for name in knames if name in self._knight_shift_names
        }
        if not comps_to_remove:
            self._refresh_after_composite_change(preferred_selected=preferred_selected)
            return

        config = self._knight_shift_config
        if config.components:
            current = list(config.components)
        else:
            current = [name for name, _ in self._oscillation_components(self._rows)]
        remaining = [c for c in current if c not in comps_to_remove]

        if not remaining:
            new_config = replace(config, enabled=False, components=())
        else:
            reference_component = config.reference_component
            reference_mode = config.reference_mode
            if reference_component in comps_to_remove:
                reference_component = None
                reference_mode = REFERENCE_APPLIED_FIELD
            new_config = replace(
                config,
                components=tuple(remaining),
                reference_component=reference_component,
                reference_mode=reference_mode,
            )
        self.set_knight_shift_config(new_config)

    def _open_model_fit_dialog(self, param_name: str) -> None:
        selected_group_ids = self._selected_group_ids_from_buttons()
        selected_groups = [
            self._group_fit_results[gid]
            for gid in selected_group_ids
            if gid in self._group_fit_results
        ]

        if len(selected_groups) >= 2:
            payload = self._run_cross_group_model_fit(param_name, selected_groups)
            if payload is not None:
                self._last_cross_group_fit = payload
            return

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

        # Per-point x-uncertainty for the effective-variance option — only
        # meaningful when the abscissa is itself a fitted parameter.
        x_err = self._x_error_array(rows, x_key)

        dialog = ModelFitDialog(
            parameter_name=param_name,
            x_key=x_key,
            x_values=x_vals,
            y_values=y_vals,
            y_errors=y_err,
            existing_fit=self._model_fits.get(param_name),
            parent=self,
            x_errors=x_err,
            x_label=self._x_axis_display_label(x_key),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.was_removed():
            self._model_fits.pop(param_name, None)
        else:
            fit = dialog.get_model_fit()
            if fit is not None:
                self._model_fits[param_name] = fit
                # Item B: surface this single fit's per-range outputs as a
                # trendable results series (one row per range), so a single fit's
                # outputs can themselves be trended.
                self.model_fit_completed.emit(param_name, x_key, fit)

        self._sync_active_group_state()

        self._refresh_model_fit_button_labels()
        self._refresh_plot()

    def _run_cross_group_model_fit(
        self,
        param_name: str,
        selected_groups: list[_GroupFitData],
    ) -> dict[str, object] | None:
        if not selected_groups:
            return None

        x_key = self._effective_x_key()
        group_payload: list[ParameterGroupData] = []
        for group in selected_groups:
            rows = sorted(group.rows, key=lambda r: self._x_value(r, x_key))
            if not rows:
                continue
            if any(param_name not in row.values for row in rows):
                QMessageBox.information(
                    self,
                    "Cross-group fit",
                    f"Selected groups do not all contain fitted values for '{param_name}'.",
                )
                return None
            x_vals = np.array([self._x_value(r, x_key) for r in rows], dtype=float)
            y_vals = np.array([row.values.get(param_name, np.nan) for row in rows], dtype=float)
            y_err = np.array([row.errors.get(param_name, np.nan) for row in rows], dtype=float)
            invalid_err = ~np.isfinite(y_err) | (y_err <= 0)
            if np.any(invalid_err):
                finite = np.abs(y_vals[np.isfinite(y_vals)])
                fallback = max(float(np.nanmedian(finite)) * 0.02, 1e-9) if finite.size else 1e-3
                y_err = y_err.copy()
                y_err[invalid_err] = fallback
            group_payload.append(
                ParameterGroupData(
                    group_id=group.group_id,
                    group_name=group.group_name,
                    x=x_vals,
                    y=y_vals,
                    yerr=y_err,
                    group_variable_value=self._group_variable_value_for_rows(rows, x_key),
                    # Per-point x-uncertainty for the effective-variance option —
                    # only present (non-None) when the abscissa is a fitted param.
                    xerr=self._x_error_array(rows, x_key),
                )
            )

        if len(group_payload) < 2:
            QMessageBox.information(
                self,
                "Cross-group fit",
                "Need at least two selected groups with valid points for cross-group fitting.",
            )
            return None

        config_key = self._cross_group_config_key(param_name, x_key, selected_groups)
        existing_config = self._cross_group_fit_configs.get(config_key)
        if existing_config is None:
            existing_config = self._build_inherited_cross_group_config(
                param_name,
                x_key,
                selected_groups,
            )

        dialog = CrossGroupFitDialog(
            parameter_name=param_name,
            x_key=x_key,
            groups=group_payload,
            existing_config=existing_config,
            parent=self,
            x_label=self._x_axis_display_label(x_key),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        output = dialog.output()
        if output is None:
            return None

        self._apply_cross_group_fit_to_groups(
            parameter_name=param_name,
            x_key=x_key,
            selected_groups=selected_groups,
            output=output,
        )

        fitted_groups = output.groups if output.groups else group_payload

        payload: dict[str, object] = {
            "parameter_name": param_name,
            "x_key": x_key,
            "groups": fitted_groups,
            "model": output.model,
            "fit_result": output.fit_result,
            "fit_x_min": output.fit_x_min,
            "fit_x_max": output.fit_x_max,
            "config": output.config,
            "config_key": config_key,
        }
        self._cross_group_fit_configs[config_key] = dict(output.config)
        self.cross_group_fit_completed.emit(param_name, fitted_groups, output)
        return payload

    def _group_variable_value_for_rows(self, rows: list[_FitRow], x_key: str) -> float:
        """Return per-group coordinate used for local-parameter trend plots.

        For cross-group fits over field, local parameters should be plotted
        against group temperature; for fits over temperature, against group
        field. This keeps group-level trends physically meaningful.
        """
        if not rows:
            return 0.0

        if x_key == "field":
            vals = np.asarray([row.temperature for row in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                return float(np.nanmedian(finite))

            fallback = np.asarray([row.field for row in rows], dtype=float)
            finite_fb = fallback[np.isfinite(fallback)]
            return float(np.nanmedian(finite_fb)) if finite_fb.size else 0.0

        if x_key == "temperature":
            vals = np.asarray([row.field for row in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                return float(np.nanmedian(finite))

            fallback = np.asarray([row.temperature for row in rows], dtype=float)
            finite_fb = fallback[np.isfinite(fallback)]
            return float(np.nanmedian(finite_fb)) if finite_fb.size else 0.0

        # For run-index and parameter-vs-parameter fits, prefer temperature if
        # available, otherwise field, as the orthogonal group coordinate.
        temps = np.asarray([row.temperature for row in rows], dtype=float)
        finite_t = temps[np.isfinite(temps)]
        if finite_t.size:
            return float(np.nanmedian(finite_t))

        fields = np.asarray([row.field for row in rows], dtype=float)
        finite_f = fields[np.isfinite(fields)]
        if finite_f.size:
            return float(np.nanmedian(finite_f))

        return float(rows[0].run_number)

    def _build_inherited_cross_group_config(
        self,
        param_name: str,
        x_key: str,
        selected_groups: list[_GroupFitData],
    ) -> dict[str, object] | None:
        """Build cross-group defaults from the best successful single-group fit."""
        best_group: _GroupFitData | None = None
        best_range: ModelFitRange | None = None
        best_result: ParameterModelFitResult | None = None
        best_chi2 = float("inf")

        for group in selected_groups:
            model_fit = group.model_fits.get(param_name)
            if model_fit is None or model_fit.x_key != x_key:
                continue
            for fit_range in model_fit.ranges:
                result = fit_range.result
                if result is None or not result.success:
                    continue
                chi2r = result.reduced_chi_squared
                if not np.isfinite(chi2r):
                    chi2r = float("inf")
                if chi2r < best_chi2:
                    best_chi2 = float(chi2r)
                    best_group = group
                    best_range = fit_range
                    best_result = result

        if best_group is None or best_range is None or best_result is None:
            return None

        rows: list[dict[str, object]] = []
        for pname in best_range.model.param_names:
            if pname in best_result.parameters:
                fitted = best_result.parameters[pname]
                initial = float(fitted.value)
                pmin = float(fitted.min)
                pmax = float(fitted.max)
                role = "Fixed" if fitted.fixed else "Global"
            elif pname in best_range.parameters:
                fallback = best_range.parameters[pname]
                initial = float(fallback.value)
                pmin = float(fallback.min)
                pmax = float(fallback.max)
                role = "Fixed" if fallback.fixed else "Global"
            else:
                initial = float(best_range.model.param_defaults.get(pname, 0.0))
                pmin = -float("inf")
                pmax = float("inf")
                role = "Global"

            rows.append(
                {
                    "name": pname,
                    "initial": initial,
                    "min": pmin,
                    "max": pmax,
                    "type": role,
                }
            )

        config: dict[str, object] = {
            "model": best_range.model.to_dict(),
            "fit_x_min": float(best_range.x_min) if best_range.x_min is not None else None,
            "fit_x_max": float(best_range.x_max) if best_range.x_max is not None else None,
            "parameter_rows": rows,
            "source_group_id": best_group.group_id,
            "source_group_name": best_group.group_name,
            "source_reduced_chi_squared": float(best_result.reduced_chi_squared),
        }
        return config

    def _cross_group_config_key(
        self,
        param_name: str,
        x_key: str,
        selected_groups: list[_GroupFitData],
    ) -> str:
        group_ids = sorted(str(group.group_id) for group in selected_groups)
        return f"{param_name}::{x_key}::{'|'.join(group_ids)}"

    def _serialize_cross_group_fit_configs(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for key, config in self._cross_group_fit_configs.items():
            if not isinstance(key, str) or not isinstance(config, dict):
                continue
            out[key] = {
                "model": dict(config.get("model", {}))
                if isinstance(config.get("model"), dict)
                else {},
                "fit_x_min": float(config.get("fit_x_min"))
                if isinstance(config.get("fit_x_min"), (int, float))
                else None,
                "fit_x_max": float(config.get("fit_x_max"))
                if isinstance(config.get("fit_x_max"), (int, float))
                else None,
                "parameter_rows": [
                    {
                        "name": str(row.get("name", "")),
                        "initial": float(row.get("initial", 0.0))
                        if isinstance(row.get("initial"), (int, float))
                        else 0.0,
                        "min": float(row.get("min", -float("inf")))
                        if isinstance(row.get("min"), (int, float))
                        else -float("inf"),
                        "max": float(row.get("max", float("inf")))
                        if isinstance(row.get("max"), (int, float))
                        else float("inf"),
                        "type": str(row.get("type", "Global")),
                    }
                    for row in config.get("parameter_rows", [])
                    if isinstance(row, dict)
                ],
                "error_mode": str(config.get("error_mode", "column")),
                "error_value": float(config.get("error_value"))
                if isinstance(config.get("error_value"), (int, float))
                else None,
                "windows": (
                    [
                        [float(lo), float(hi)]
                        for lo, hi in (parse_fit_windows(config.get("windows")) or [])
                    ]
                    or None
                ),
                "use_x_errors": bool(config.get("use_x_errors", False)),
            }
        return out

    def _deserialize_cross_group_fit_configs(self, state: object) -> dict[str, dict[str, object]]:
        if not isinstance(state, dict):
            return {}
        out: dict[str, dict[str, object]] = {}
        for key, config in state.items():
            if not isinstance(key, str) or not isinstance(config, dict):
                continue
            model = config.get("model")
            rows = config.get("parameter_rows")
            out[key] = {
                "model": dict(model) if isinstance(model, dict) else {},
                "fit_x_min": float(config.get("fit_x_min"))
                if isinstance(config.get("fit_x_min"), (int, float))
                else None,
                "fit_x_max": float(config.get("fit_x_max"))
                if isinstance(config.get("fit_x_max"), (int, float))
                else None,
                "parameter_rows": [dict(row) for row in rows if isinstance(row, dict)]
                if isinstance(rows, list)
                else [],
                "error_mode": str(config.get("error_mode", "column")),
                "error_value": float(config.get("error_value"))
                if isinstance(config.get("error_value"), (int, float))
                else None,
                "windows": (
                    [
                        [float(lo), float(hi)]
                        for lo, hi in (parse_fit_windows(config.get("windows")) or [])
                    ]
                    or None
                ),
                "use_x_errors": bool(config.get("use_x_errors", False)),
            }
        return out

    def _serialize_last_cross_group_fit(self) -> dict | None:
        payload = self._last_cross_group_fit
        if payload is None:
            return None
        fit_result = payload.get("fit_result")
        model = payload.get("model")
        groups = payload.get("groups")
        if not isinstance(fit_result, CrossGroupFitResult):
            return None
        if not isinstance(model, ParameterCompositeModel):
            return None
        if not isinstance(groups, list):
            return None

        fit_x_min_raw = payload.get("fit_x_min", float("nan"))
        fit_x_max_raw = payload.get("fit_x_max", float("nan"))
        fit_x_min = (
            float(fit_x_min_raw) if isinstance(fit_x_min_raw, (int, float)) else float("nan")
        )
        fit_x_max = (
            float(fit_x_max_raw) if isinstance(fit_x_max_raw, (int, float)) else float("nan")
        )

        return {
            "parameter_name": str(payload.get("parameter_name", "")),
            "x_key": str(payload.get("x_key", "run")),
            "fit_x_min": fit_x_min if np.isfinite(fit_x_min) else None,
            "fit_x_max": fit_x_max if np.isfinite(fit_x_max) else None,
            "config": dict(payload.get("config", {}))
            if isinstance(payload.get("config"), dict)
            else {},
            "config_key": str(payload.get("config_key", "")),
            "groups": [
                {
                    "group_id": g.group_id,
                    "group_name": g.group_name,
                    "x": np.asarray(g.x, dtype=float).tolist(),
                    "y": np.asarray(g.y, dtype=float).tolist(),
                    "yerr": np.asarray(g.yerr, dtype=float).tolist(),
                    "group_variable_value": float(g.group_variable_value),
                }
                for g in groups
                if isinstance(g, ParameterGroupData)
            ],
            "model": model.to_dict(),
            "fit_result": {
                "success": bool(fit_result.success),
                "chi_squared": float(fit_result.chi_squared),
                "reduced_chi_squared": float(fit_result.reduced_chi_squared),
                "message": str(fit_result.message),
                "global_parameters": [
                    {
                        "name": p.name,
                        "value": p.value,
                        "min": p.min,
                        "max": p.max,
                        "fixed": p.fixed,
                    }
                    for p in fit_result.global_parameters
                ],
                "global_uncertainties": dict(fit_result.global_uncertainties),
                "local_parameters": {
                    gid: [
                        {
                            "name": p.name,
                            "value": p.value,
                            "min": p.min,
                            "max": p.max,
                            "fixed": p.fixed,
                        }
                        for p in pset
                    ]
                    for gid, pset in fit_result.local_parameters.items()
                },
                "fixed_parameters": [
                    {
                        "name": p.name,
                        "value": p.value,
                        "min": p.min,
                        "max": p.max,
                        "fixed": p.fixed,
                    }
                    for p in fit_result.fixed_parameters
                ],
                "local_uncertainties": {
                    gid: dict(vals) for gid, vals in fit_result.local_uncertainties.items()
                },
            },
        }

    def _deserialize_last_cross_group_fit(self, state: object) -> dict[str, object] | None:
        if not isinstance(state, dict):
            return None
        model_state = state.get("model")
        fit_state = state.get("fit_result")
        groups_state = state.get("groups")
        if (
            not isinstance(model_state, dict)
            or not isinstance(fit_state, dict)
            or not isinstance(groups_state, list)
        ):
            return None

        try:
            model = ParameterCompositeModel.from_dict(model_state)
        except Exception:
            return None

        groups: list[ParameterGroupData] = []
        for entry in groups_state:
            if not isinstance(entry, dict):
                continue
            groups.append(
                ParameterGroupData(
                    group_id=str(entry.get("group_id", "")),
                    group_name=str(entry.get("group_name", "")),
                    x=np.asarray(entry.get("x", []), dtype=float),
                    y=np.asarray(entry.get("y", []), dtype=float),
                    yerr=np.asarray(entry.get("yerr", []), dtype=float),
                    group_variable_value=float(entry.get("group_variable_value", 0.0)),
                )
            )

        global_params = ParameterSet()
        for p in fit_state.get("global_parameters", []):
            if isinstance(p, dict):
                global_params.add(
                    Parameter(
                        name=str(p.get("name", "")),
                        value=float(p.get("value", 0.0)),
                        min=float(p.get("min", -float("inf"))),
                        max=float(p.get("max", float("inf"))),
                        fixed=bool(p.get("fixed", False)),
                    )
                )

        local_params: dict[str, ParameterSet] = {}
        for gid, plist in dict(fit_state.get("local_parameters", {})).items():
            pset = ParameterSet()
            for p in plist:
                if isinstance(p, dict):
                    pset.add(
                        Parameter(
                            name=str(p.get("name", "")),
                            value=float(p.get("value", 0.0)),
                            min=float(p.get("min", -float("inf"))),
                            max=float(p.get("max", float("inf"))),
                            fixed=bool(p.get("fixed", False)),
                        )
                    )
            local_params[str(gid)] = pset

        fixed_params = ParameterSet()
        for p in fit_state.get("fixed_parameters", []):
            if isinstance(p, dict):
                fixed_params.add(
                    Parameter(
                        name=str(p.get("name", "")),
                        value=float(p.get("value", 0.0)),
                        min=float(p.get("min", -float("inf"))),
                        max=float(p.get("max", float("inf"))),
                        fixed=bool(p.get("fixed", True)),
                    )
                )

        fit_result = CrossGroupFitResult(
            success=bool(fit_state.get("success", False)),
            chi_squared=float(fit_state.get("chi_squared", 0.0)),
            reduced_chi_squared=float(fit_state.get("reduced_chi_squared", 0.0)),
            global_parameters=global_params,
            local_parameters=local_params,
            fixed_parameters=fixed_params,
            global_uncertainties={
                str(k): float(v) for k, v in dict(fit_state.get("global_uncertainties", {})).items()
            },
            local_uncertainties={
                str(gid): {str(k): float(v) for k, v in dict(vals).items()}
                for gid, vals in dict(fit_state.get("local_uncertainties", {})).items()
            },
            message=str(fit_state.get("message", "")),
        )

        return {
            "parameter_name": str(state.get("parameter_name", "")),
            "x_key": str(state.get("x_key", "run")),
            "fit_x_min": float(state.get("fit_x_min"))
            if isinstance(state.get("fit_x_min"), (int, float))
            else float("nan"),
            "fit_x_max": float(state.get("fit_x_max"))
            if isinstance(state.get("fit_x_max"), (int, float))
            else float("nan"),
            "config": dict(state.get("config", {}))
            if isinstance(state.get("config"), dict)
            else {},
            "config_key": str(state.get("config_key", "")),
            "groups": groups,
            "model": model,
            "fit_result": fit_result,
        }

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

        display_params = self._display_y_parameters()
        columns = ["Run", "𝐵 (G)", "𝑇 (K)"]
        for name in display_params:
            label = _format_param_label(name)
            columns.extend([label, f"err {label}"])
        # A free-text x-axis (Angle / custom column) is not one of the fixed
        # columns, so add it explicitly (folded as displayed) — otherwise the
        # table/TSV would not show the abscissa the plot is drawn against.
        abscissa = self._export_abscissa_column()
        abscissa_key = abscissa[0] if abscissa is not None else None
        if abscissa is not None:
            columns.append(abscissa[1])

        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(str(row.run_label)))
            self._table.setItem(i, 1, QTableWidgetItem(f"{row.field:.6g}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{row.temperature:.6g}"))
            col = 3
            for name in display_params:
                val = row.values.get(name, np.nan)
                err = row.errors.get(name, np.nan)
                self._table.setItem(i, col, QTableWidgetItem(f"{val:.6g}"))
                self._table.setItem(i, col + 1, QTableWidgetItem(f"{err:.3g}"))
                col += 2
            if abscissa_key is not None:
                self._table.setItem(
                    i, col, QTableWidgetItem(f"{self._x_value(row, abscissa_key):.6g}")
                )

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
        display_params = self._display_y_parameters()
        return [p for p in display_params if p in params]

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

        show_components = self._show_components_check.isChecked()
        component_colors = ["#8ecae6", "#90be6d", "#f4a261", "#e5989b", "#bdb2ff", "#ffd166"]

        # Consume the off-thread cache when present. A missing key means the
        # worker produced no curves for this param — draw nothing rather than
        # re-evaluating the (possibly very slow) model on the GUI thread, which
        # would defeat the off-thread design and re-run on every redraw.
        if self._precomputed_trend_curves is not None:
            precomputed = self._precomputed_trend_curves.get(param_name) or []
            curves = [(ri, xs, ys) for (ri, xs, ys, _comp) in precomputed]
            components_by_range = {ri: comp for (ri, _xs, _ys, comp) in precomputed}
        else:
            # No cache yet (defensive: the _refresh_plot router only reaches
            # _draw_plot with active overlays once the cache is populated).
            curves = self._sampled_fit_curves(
                param_name,
                x_key=fit.x_key,
                num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
            )
            components_by_range = None

        for idx, (range_index, xs, ys) in enumerate(curves):
            line_color = _fit_overlay_color(idx) if len(curves) > 1 else color

            if show_components:
                if components_by_range is not None:
                    # Precompute path: components already evaluated off-thread;
                    # use them directly and never read the (possibly-changed)
                    # live fit.ranges, so a stale index can't mismatch.
                    components = components_by_range.get(range_index)
                elif range_index < len(fit.ranges):
                    components = None
                    fit_range = fit.ranges[range_index]
                    result = fit_range.result
                    if result is not None and result.success:
                        kwargs = {p.name: p.value for p in result.parameters}
                        try:
                            components = fit_range.model.evaluate_components(
                                xs, additive_only=True, **kwargs
                            )
                        except Exception:  # noqa: BLE001 - bad model → skip components, keep curve
                            components = None
                else:
                    components = None
                if components is not None:
                    ordered_components = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xs, dtype=float)
                    for cidx, (_cname, comp_y) in enumerate(ordered_components):
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(xs, lower, upper, color=fill_color, alpha=0.3, zorder=1)
                        ax.plot(
                            xs,
                            upper,
                            linestyle="--",
                            linewidth=0.8,
                            color=fill_color,
                            alpha=0.9,
                            zorder=2,
                        )
                        cumulative = upper

            ax.plot(xs, ys, linestyle="-", linewidth=1.5, color=line_color, alpha=0.9, zorder=3)

    def _ordered_components_for_stacking(
        self,
        components: list[tuple[str, np.ndarray]],
    ) -> list[tuple[str, np.ndarray]]:
        """Return a stable bottom-to-top stacking order for additive components.

        Heuristic:
        1. Put background-like components (bg/constant) at the bottom.
        2. For remaining components, place smoother/lower-variance traces lower.
        3. Break ties by mean magnitude (smaller first).
        """
        if not components:
            return []

        def _priority(name: str) -> int:
            lname = name.lower()
            if "bg" in lname or "background" in lname or "constant" in lname:
                return 0
            return 1

        scored: list[tuple[int, float, float, int, tuple[str, np.ndarray]]] = []
        for idx, item in enumerate(components):
            name, values = item
            arr = np.maximum(np.asarray(values, dtype=float), 0.0)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                mean_val = 0.0
                variability = 0.0
            else:
                mean_val = float(np.mean(finite))
                variability = float(np.std(finite) / max(mean_val, 1e-12))
            scored.append((_priority(name), variability, mean_val, idx, item))

        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [item for *_meta, item in scored]

    def _active_overlay_params(self) -> list[str]:
        """Selected y-parameters that have an active model-fit overlay to draw."""
        x_key = self._effective_x_key()
        return [
            name
            for name in self._selected_y_parameters()
            if (fit := self._model_fits.get(name)) is not None and fit.active and fit.x_key == x_key
        ]

    def _trend_overlay_signature(self, active: list[str]) -> tuple:
        """Inputs the overlay curves depend on; a change invalidates the cache.

        Captures the x-axis key, the show-components flag, the data sampling
        domain (the span for open-ended fit ranges — taken as a value, not as
        ``id(self._rows)``, because composite-parameter edits mutate row values
        in place without replacing the rows list), and each active fit's object
        identity (fits are *replaced*, not mutated, on edit/re-fit). Pure-render
        toggles (log scale, plot mode, share-x) leave this unchanged, so they
        redraw from the cache instead of re-evaluating the model.
        """
        x_key = self._effective_x_key()
        return (
            x_key,
            bool(self._show_components_check.isChecked()),
            self._x_domain_for_sampling(x_key),
            tuple((name, id(self._model_fits.get(name))) for name in active),
        )

    def _update_empty_state_hint(self) -> None:
        """Show the 'load a batch series' hint only while no rows are loaded."""
        hint = getattr(self, "_empty_state_hint", None)
        if hint is not None:
            hint.setVisible(not self._rows)

    def _refresh_plot(self) -> None:
        """Redraw the trend plot, recomputing overlay curves off-thread if stale.

        Routing entry point: the scatter is cheap, but the model-fit overlays can
        be very slow (e.g. DiffusionLF_2D runs scipy quadrature per sample). When
        the overlay inputs changed since the cache was built, recompute them on a
        worker behind the overlay; otherwise (pure-render toggles, or no active
        overlays) draw synchronously now.
        """
        self._update_empty_state_hint()
        if not self._has_mpl:
            return
        if self._suspend_plot_refresh:
            # A bulk state change (project restore) is in progress; it issues a
            # single recompute when done. Skip the intermediate draw.
            return
        active = self._active_overlay_params()
        sig = self._trend_overlay_signature(active)
        if active and sig != self._trend_cache_sig:
            self._start_trend_curve_compute(active, sig)
            return
        self._draw_plot()

    def _draw_plot(self) -> None:
        if not self._has_mpl:
            return

        self._axes_tag_map = {}
        axes_by_tag: dict[str, object] = {}

        y_params = self._selected_y_parameters()
        if not self._rows or not y_params:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.set_title("No varying fit parameters")
            self._axes_tag_map[id(ax)] = "main"
            axes_by_tag["main"] = ax
            self._draw_plot_annotations(axes_by_tag)
            self._canvas.draw()
            return

        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        x_vals = np.array([self._x_value(r, x_key) for r in rows], dtype=float)
        x_err = self._x_error_array(rows, x_key)
        x_label = self._x_axis_label_mpl(x_key)
        self._update_custom_x_skip_note(x_key, x_vals)

        self._figure.clear()
        plot_mode = self._plot_mode_combo.currentText()

        if plot_mode == "Subplots" and len(y_params) > 1:
            num_params = len(y_params)
            num_cols = 2
            num_rows = (num_params + num_cols - 1) // num_cols

            for idx, y_name in enumerate(y_params):
                ax = self._figure.add_subplot(num_rows, num_cols, idx + 1)
                self._axes_tag_map[id(ax)] = y_name
                axes_by_tag[y_name] = ax
                y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)

                self._draw_model_overlay_mpl(ax, y_name)

                ax.scatter(x_vals, y_vals, s=16, zorder=6, color="C0")
                ye = y_err if np.any(np.isfinite(y_err) & (y_err > 0)) else None
                if ye is not None or x_err is not None:
                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=ye,
                        xerr=x_err,
                        fmt="none",
                        ecolor="gray",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax.set_xlabel(x_label)
                ax.set_ylabel(_format_plot_label(y_name))
                ax.set_title(_format_plot_label(y_name))
                ax.set_xscale("log" if self._log_x_check.isChecked() else "linear")
                if self._show_components_check.isChecked():
                    ax.set_yscale("linear")
                    ax.set_ylim(bottom=0.0)
                else:
                    ax.set_yscale("log" if self._is_log_y_for(y_name) else "linear")
                ax.grid(True, alpha=0.3)
        else:
            ax = self._figure.add_subplot(111)
            ax.set_xlabel(x_label)
            self._axes_tag_map[id(ax)] = "main"
            axes_by_tag["main"] = ax

            if len(y_params) == 2:
                left_name, right_name = y_params
                left_vals = np.array([r.values.get(left_name, np.nan) for r in rows], dtype=float)
                left_err = np.array([r.errors.get(left_name, np.nan) for r in rows], dtype=float)
                right_vals = np.array([r.values.get(right_name, np.nan) for r in rows], dtype=float)
                right_err = np.array([r.errors.get(right_name, np.nan) for r in rows], dtype=float)

                ax2 = ax.twinx()
                self._axes_tag_map[id(ax)] = left_name
                self._axes_tag_map[id(ax2)] = right_name
                axes_by_tag[left_name] = ax
                axes_by_tag[right_name] = ax2
                left_color = "C0"
                right_color = "C1"

                self._draw_model_overlay_mpl(ax, left_name, color=left_color)
                self._draw_model_overlay_mpl(ax2, right_name, color=right_color)

                ax.scatter(x_vals, left_vals, s=16, zorder=6, color=left_color)
                ye_left = left_err if np.any(np.isfinite(left_err) & (left_err > 0)) else None
                if ye_left is not None or x_err is not None:
                    ax.errorbar(
                        x_vals,
                        left_vals,
                        yerr=ye_left,
                        xerr=x_err,
                        fmt="none",
                        ecolor=left_color,
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax2.scatter(x_vals, right_vals, s=16, zorder=6, color=right_color)
                ye_right = right_err if np.any(np.isfinite(right_err) & (right_err > 0)) else None
                if ye_right is not None or x_err is not None:
                    ax2.errorbar(
                        x_vals,
                        right_vals,
                        yerr=ye_right,
                        xerr=x_err,
                        fmt="none",
                        ecolor=right_color,
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax.set_ylabel(_format_plot_label(left_name), color=left_color)
                ax2.set_ylabel(_format_plot_label(right_name), color=right_color)
                ax.tick_params(axis="y", colors=left_color)
                ax2.tick_params(axis="y", colors=right_color)
                if self._show_components_check.isChecked():
                    ax.set_yscale("linear")
                    ax2.set_yscale("linear")
                    ax.set_ylim(bottom=0.0)
                    ax2.set_ylim(bottom=0.0)
                else:
                    ax.set_yscale("log" if self._is_log_y_for(left_name) else "linear")
                    ax2.set_yscale("log" if self._is_log_y_for(right_name) else "linear")
                ax.set_xscale("log" if self._log_x_check.isChecked() else "linear")
                ax.grid(True, alpha=0.3)
            else:
                axes_by_tag["main"] = ax
                for idx, y_name in enumerate(y_params):
                    y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                    y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                    color = f"C{idx % 10}"
                    label = _format_plot_legend_label(y_name) if len(y_params) > 1 else None

                    self._draw_model_overlay_mpl(ax, y_name, color=color)

                    ax.scatter(x_vals, y_vals, s=16, zorder=6, label=label, color=color)
                    ye = y_err if np.any(np.isfinite(y_err) & (y_err > 0)) else None
                    if ye is not None or x_err is not None:
                        ax.errorbar(
                            x_vals,
                            y_vals,
                            yerr=ye,
                            xerr=x_err,
                            fmt="none",
                            ecolor=color,
                            capsize=2,
                            elinewidth=1,
                            zorder=5,
                        )

                if len(y_params) == 1:
                    ax.set_ylabel(_format_plot_label(y_params[0]))
                    if self._show_components_check.isChecked():
                        ax.set_yscale("linear")
                        ax.set_ylim(bottom=0.0)
                    else:
                        ax.set_yscale("log" if self._is_log_y_for(y_params[0]) else "linear")
                else:
                    ax.set_ylabel("Parameter Value")
                    if len(y_params) > 2:
                        ax.legend(loc="best")
                    if self._show_components_check.isChecked():
                        ax.set_yscale("linear")
                        ax.set_ylim(bottom=0.0)
                    else:
                        ax.set_yscale(
                            "log"
                            if any(self._is_log_y_for(name) for name in y_params)
                            else "linear"
                        )

                ax.set_xscale("log" if self._log_x_check.isChecked() else "linear")
                ax.grid(True, alpha=0.3)

        self._draw_knight_shift_crossings(axes_by_tag, x_key)
        self._draw_plot_annotations(axes_by_tag)

        if getattr(self._figure, "get_constrained_layout", lambda: False)():
            layout_engine = getattr(self._figure, "get_layout_engine", lambda: None)()
            if layout_engine is not None and hasattr(layout_engine, "set"):
                layout_engine.set(w_pad=0.04, h_pad=0.04, hspace=0.05, wspace=0.05)
        else:
            self._figure.tight_layout(pad=1.2)
        self._canvas.draw()

    def _x_axis_display_label(self, x_key: str) -> str:
        """Friendly, plain-text label for an x-axis key (for dialog titles).

        Unlike :meth:`_x_axis_label_mpl` this returns no mathtext, so a
        ``custom:<id>`` key reads as its column name ("Current (A)") rather than
        the raw internal id in the Model-Fit / cross-group dialogs.
        """
        name = _x_param_name(x_key)
        if name is not None:
            return _format_param_label(name)
        labels = self._custom_x_labels()
        if x_key in labels:
            return labels[x_key]
        return {"field": "B (G)", "temperature": "T (K)", "run": "Run"}.get(x_key, "Run")

    def _x_axis_label_mpl(self, x_key: str) -> str:
        name = _x_param_name(x_key)
        if name is not None:
            return _format_plot_label(name)
        if self._angle_x_field is not None and x_key == self._angle_x_field[1]:
            return self._angle_x_field[0] + self._angle_fold_suffix()
        custom_id = _x_custom_id(x_key)
        if custom_id is not None:
            return self._custom_x_labels().get(custom_id, custom_id)
        return {"field": "$B$ (G)", "temperature": "$T$ (K)", "run": "Run Number"}.get(
            x_key, "Run Number"
        )

    def _x_value(self, row: _FitRow, x_key: str, *, fold: bool = True) -> float:
        name = _x_param_name(x_key)
        if name is not None:
            return float(row.values.get(name, float("nan")))
        # The Angle field and the generic custom columns both store free text per
        # run under their key in ``custom_values``; coerce to a numeric abscissa
        # (empty/non-numeric/non-finite → NaN so the point is dropped and counted
        # rather than plotted at 0 or corrupting the axis/fit).
        value_key = x_key if x_key == self._angle_x_key() else _x_custom_id(x_key)
        if value_key is not None:
            value = _coerce_abscissa(row.custom_values.get(value_key, ""))
            # Fold the Angle axis into its chosen period so equivalent orientations
            # overlay (no-op for non-angle custom columns or when folding is off).
            # ``fold=False`` recovers the raw scan coordinate — used by crossing
            # detection, which must follow the true rotation order, not the folded
            # display (folding would collapse distinct orientations onto one x).
            if fold and x_key == self._angle_x_key() and self._angle_wrap_period is not None:
                return wrap_angle_deg(value, self._angle_wrap_period)
            return value
        if x_key == "field":
            return row.field
        if x_key == "temperature":
            return row.temperature
        return float(row.run_number)

    def _x_error(self, row: _FitRow, x_key: str) -> float:
        """Per-point x-uncertainty for a ``param:<name>`` x-key (NaN otherwise).

        Run-level axes (field/temperature/run) carry no uncertainty, so the
        fit treats them as exact and no horizontal error bars are drawn.
        """
        name = _x_param_name(x_key)
        if name is not None:
            return float(row.errors.get(name, float("nan")))
        return float("nan")

    def _x_error_array(self, rows: list[_FitRow], x_key: str) -> np.ndarray | None:
        """Horizontal-error array for the plot, or None when x is exact."""
        if _x_param_name(x_key) is None:
            return None
        errs = np.array([self._x_error(r, x_key) for r in rows], dtype=float)
        if np.any(np.isfinite(errs) & (errs > 0)):
            return errs
        return None

    def _fraction_weights_note(self) -> str:
        """Footnote giving the normalised fraction weights for the active series.

        The raw fitted fractions shown in the header are un-normalised relative
        weights (the model divides each by its group sum), so they need not add to
        1; this line reports the physical partition fraction_i / Σ — including the
        usually-hidden fixed last fraction — which does sum to 1 per group.
        """
        weights = self._fraction_weights_by_id.get(self._active_group_id or "")
        if not weights:
            return ""

        def _order(name: str) -> tuple[int, str]:
            _, index = split_parameter_name(name)
            return (int(index) if index is not None else 0, name)

        parts = [f"{name} = {weights[name]:.3g}" for name in sorted(weights, key=_order)]
        return "Normalised fraction weights (relative; each group sums to 1): " + ", ".join(parts)

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
                unit = get_param_info(param.name).unit
                unit_text = f" {unit}" if unit else ""
                err = self._global_param_uncertainties.get(param.name)
                if err is not None:
                    lines.append(f"{param.name} = {param.value:.6g} \u00b1 {err:.6g}{unit_text}")
                else:
                    lines.append(f"{param.name} = {param.value:.6g}{unit_text}")
            header_text = "\n".join(lines) if lines else "None"
        else:
            header_text = "None"

        header_label = QLabel(header_text)
        header_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        fraction_note = self._fraction_weights_note()
        if fraction_note:
            note_label = QLabel(fraction_note)
            note_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            note_label.setWordWrap(True)
            note_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            layout.addWidget(note_label)

        table_view = QTableWidget(self._table.rowCount(), self._table.columnCount(), dialog)
        table_view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        headers = [
            self._table.horizontalHeaderItem(col).text() for col in range(self._table.columnCount())
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

    def _export_model_label(self) -> str:
        """Return the originating fit model common to the displayed rows.

        The trend panel plots one series fit with a single time-domain model, so
        the first non-empty per-row model label is representative. Empty when no
        row recorded a model (computed/model-less series, or legacy projects).
        """
        for row in self._rows:
            if row.model_name:
                return row.model_name
        return ""

    def _export_tsv(self) -> None:
        if self._table.columnCount() == 0 or self._table.rowCount() == 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Parameter Table",
            default_export_path("fit_parameters.tsv"),
            "TSV files (*.tsv);;All files (*)",
        )
        if not path:
            return
        remember_export_path(path)

        headers = ["Run", "B (G)", "T (K)"]
        for name in self._display_y_parameters():
            unit = get_param_info(name).unit
            if unit:
                headers.extend([f"{name} ({unit})", f"err_{name} ({unit})"])
            else:
                headers.extend([name, f"err_{name}"])
        # The Angle/custom abscissa is appended by _refresh_table as a trailing
        # column; mirror it here so the header row matches the data cells read
        # from the table below.
        abscissa = self._export_abscissa_column()
        if abscissa is not None:
            headers.append(abscissa[1])
        # Per-run goodness of fit, appended after the parameter columns so the
        # TSV records each run's fit quality alongside its values.
        headers.extend(["reduced_chi2", "chi2"])

        model_label = self._export_model_label()

        # Map each displayed table row back to its source _FitRow positionally,
        # replaying _refresh_table's sort, so the χ² columns track the right run
        # even when two rows share a run_label (keying on the label would
        # silently misattribute a shadowed row's goodness of fit).
        sorted_rows = sorted(self._rows, key=lambda r: self._x_value(r, self._effective_x_key()))

        with open(path, "w", newline="", encoding="utf-8") as tsvfile:
            # Provenance header (comment lines) so the TSV is self-describing and
            # at parity with the GLE .dat export: the originating fit model and
            # the shared global-parameter values with uncertainties.
            if model_label:
                tsvfile.write(f"# Model: {model_label}\n")
            if self._global_params is not None:
                tsvfile.write("# Global fitting parameters:\n")
                for param in self._global_params:
                    unit = get_param_info(param.name).unit
                    label = f"{param.name} ({unit})" if unit else param.name
                    err = self._global_param_uncertainties.get(param.name)
                    if err is not None:
                        tsvfile.write(f"#   {label} = {param.value:.6g} +/- {err:.6g}\n")
                    else:
                        tsvfile.write(f"#   {label} = {param.value:.6g}\n")

            writer = csv.writer(tsvfile, delimiter="\t")
            writer.writerow(headers)
            for row in range(self._table.rowCount()):
                values: list[str] = []
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    values.append(item.text() if item is not None else "")
                src = sorted_rows[row] if row < len(sorted_rows) else None
                rchi = None if src is None else src.reduced_chi_squared
                chi = None if src is None else src.chi_squared
                values.append("" if rchi is None else f"{rchi:.6g}")
                values.append("" if chi is None else f"{chi:.6g}")
                writer.writerow(values)

    def _serialize_model_fits(self) -> dict:
        return self._serialize_model_fits_mapping(self._model_fits)

    def _serialize_model_fits_mapping(self, mapping: dict[str, ParameterModelFit]) -> dict:
        payload: dict[str, dict] = {}
        for param_name, model_fit in mapping.items():
            ranges_data = []
            for fit_range in model_fit.ranges:
                range_item = {
                    "x_min": fit_range.x_min,
                    "x_max": fit_range.x_max,
                    "windows": (
                        [[float(lo), float(hi)] for lo, hi in fit_range.windows]
                        if fit_range.windows
                        else None
                    ),
                    "model": fit_range.model.to_dict(),
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
                        "error_mode": fit_range.result.error_mode,
                        "n_points": fit_range.result.n_points,
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
                "use_x_errors": bool(model_fit.use_x_errors),
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
                    model = ParameterCompositeModel.from_dict(
                        dict(range_state.get("model", range_state))
                    )
                except Exception:
                    continue

                # Backward compatibility: older serialized diffusion models used
                # C instead of A for the diffusion coupling parameter.
                uses_diffusion_component = any(
                    name in {"DiffusionLF_1D", "DiffusionLF_2D", "DiffusionLF_3D"}
                    for name in model.component_names
                )

                params = ParameterSet()
                for p in range_state.get("parameters", []):
                    if not isinstance(p, dict):
                        continue
                    try:
                        pname = str(p.get("name", ""))
                        if uses_diffusion_component and pname == "C":
                            pname = "A"
                        params.add(
                            Parameter(
                                name=pname,
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
                            pname = str(p.get("name", ""))
                            if uses_diffusion_component and pname == "C":
                                pname = "A"
                            result_params.add(
                                Parameter(
                                    name=pname,
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
                        error_mode=str(result_state.get("error_mode", "column")),
                        n_points=int(result_state.get("n_points", 0)),
                    )

                ranges.append(
                    ModelFitRange(
                        x_min=float(range_state.get("x_min"))
                        if range_state.get("x_min") is not None
                        else None,
                        x_max=float(range_state.get("x_max"))
                        if range_state.get("x_max") is not None
                        else None,
                        model=model,
                        parameters=params,
                        result=result_obj,
                        windows=parse_fit_windows(range_state.get("windows")),
                    )
                )

            if not ranges:
                continue

            restored[key] = ParameterModelFit(
                parameter_name=str(entry.get("parameter_name", key)),
                x_key=x_key,
                active=bool(entry.get("active", True)),
                use_x_errors=bool(entry.get("use_x_errors", False)),
                ranges=ranges,
            )

        return restored

    def _iter_active_fit_ranges(self, x_key: str, y_params: list[str] | None = None):
        """Yield (parameter_name, range_index, fit_range) for successful active fits."""
        allowed = set(y_params or self._display_y_parameters())
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
            count += len(
                self._sampled_fit_curves(
                    pname,
                    x_key,
                    num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
                )
            )
        return count

    def _count_fit_curves_for_param(self, x_key: str, param_name: str) -> int:
        return len(
            self._sampled_fit_curves(
                param_name,
                x_key,
                num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
            )
        )

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
        num_points: int = _PARAMETER_FIT_CURVE_SAMPLE_COUNT,
        *,
        x_domain: object = _UNSET,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample one fit range's overlay curve.

        ``x_domain`` lets the off-thread trend recompute pass a GUI-thread
        snapshot of the data x-range. When it is omitted (``_UNSET``) the
        GUI-thread callers read it live from the rows; when passed explicitly
        (even as ``None``) the snapshot is used as-is, so the worker thread never
        touches ``self._rows``.
        """
        result = fit_range.result
        if result is None or not result.success:
            return None

        try:
            # Window-union envelope when windows exist, else the plain bounds
            # — keeps the overlay span consistent with the fitted mask.
            x_min, x_max = effective_range_bounds(fit_range)
        except ValueError:
            return None
        if x_min is None or x_max is None:
            domain = self._x_domain_for_sampling(x_key) if x_domain is _UNSET else x_domain
            if domain is None:
                return None
            if x_min is None:
                x_min = domain[0]
            if x_max is None:
                x_max = domain[1]

        if x_max <= x_min:
            return None

        sample_count = max(2, int(num_points))
        if x_key == "field" and float(x_min) > 0.0 and float(x_max) > 0.0:
            xs = np.geomspace(float(x_min), float(x_max), num=sample_count, dtype=float)
        else:
            xs = np.linspace(float(x_min), float(x_max), num=sample_count, dtype=float)
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
        num_points: int = _PARAMETER_FIT_CURVE_SAMPLE_COUNT,
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

    def _compute_trend_curves(
        self,
        model_fits: dict,
        x_key: str,
        x_domain: tuple[float, float] | None,
        y_params: list[str],
        show_components: bool,
        num_points: int = _PARAMETER_FIT_CURVE_SAMPLE_COUNT,
    ) -> dict[str, list]:
        """Sample the model-fit overlay curves for *y_params* (worker-thread safe).

        Pure: operates only on the passed snapshots (model_fits / x_key / x_domain)
        — no widget reads — so the trend recompute can run off the GUI thread.
        Returns ``{param_name: [(range_index, xs, ys, components_or_None), ...]}``
        for :meth:`_draw_model_overlay_mpl` to consume.
        """
        result: dict[str, list] = {}
        for param_name in y_params:
            # Isolate per-parameter failures: a single bad model must not raise
            # out of the worker (which would blank *every* overlay via on_error).
            try:
                fit = model_fits.get(param_name)
                if fit is None or not fit.active or fit.x_key != x_key:
                    continue
                ranges_out: list = []
                for idx, fit_range in enumerate(fit.ranges):
                    sampled = self._sample_fit_range_curve(
                        fit_range, x_key=x_key, num_points=num_points, x_domain=x_domain
                    )
                    if sampled is None:
                        continue
                    xs, ys = sampled
                    components = None
                    fit_result = fit_range.result
                    if show_components and fit_result is not None and fit_result.success:
                        kwargs = {p.name: p.value for p in fit_result.parameters}
                        try:
                            components = fit_range.model.evaluate_components(
                                xs, additive_only=True, **kwargs
                            )
                        except Exception:  # noqa: BLE001 - bad model → no components, keep curve
                            components = None
                    ranges_out.append((idx, xs, ys, components))
            except Exception:  # noqa: BLE001 - skip this parameter, keep the others
                continue
            if ranges_out:
                result[param_name] = ranges_out
        return result

    def _start_trend_curve_compute(
        self, active: list[str] | None = None, sig: tuple | None = None
    ) -> None:
        """Recompute the *active* overlay curves off-thread behind the overlay.

        Called by :meth:`_refresh_plot` when the overlay cache is stale. ``active``
        / ``sig`` are passed in to avoid recomputing them; when omitted (a direct
        force-recompute) they are derived here.
        """
        if not self._has_mpl:
            return
        if active is None:
            active = self._active_overlay_params()
        if sig is None:
            sig = self._trend_overlay_signature(active)
        if not active:
            # Nothing heavy to draw. Drop any stale overlay from an earlier
            # in-flight compute, mark the (empty) cache current, and draw now.
            if self._trend_overlay is not None:
                self._trend_overlay.hide()
            self._precomputed_trend_curves = None
            self._trend_cache_sig = sig
            self._draw_plot()
            return
        if self._trend_curve_compute_active:
            # A compute is already running; fold this request into a single
            # rerun carrying the latest (active, sig).
            self._pending_trend_request = (active, sig)
            return
        # Snapshot everything the worker needs so it never reads GUI state that a
        # concurrent edit could mutate.
        model_fits = dict(self._model_fits)
        x_key = self._effective_x_key()
        x_domain = self._x_domain_for_sampling(x_key)
        show_components = self._show_components_check.isChecked()
        if self._trend_overlay is not None:
            self._trend_overlay.show_message("Computing trend curves…")
        self._trend_curve_compute_active = True
        self._tasks.start(
            lambda _worker: self._compute_trend_curves(
                model_fits, x_key, x_domain, active, show_components
            ),
            on_finished=lambda curves: self._on_trend_curves_ready(curves, sig),
            on_error=lambda message: self._on_trend_curves_error(message, sig),
        )

    def _redispatch_pending_trend_compute(self) -> bool:
        """If a recompute was requested mid-flight, start it now. Returns True then."""
        request = self._pending_trend_request
        if request is None:
            return False
        self._pending_trend_request = None
        self._start_trend_curve_compute(*request)
        return True

    def _on_trend_curves_ready(self, curves: object, sig: tuple) -> None:
        self._trend_curve_compute_active = False
        if self._redispatch_pending_trend_compute():
            # Inputs changed mid-compute; this result is stale — skip drawing it.
            return
        if self._trend_overlay is not None:
            self._trend_overlay.hide()
        # Cache the curves (keyed by the signature they were computed for) so
        # pure-render redraws reuse them instead of re-evaluating the model.
        self._precomputed_trend_curves = curves if isinstance(curves, dict) else {}
        self._trend_cache_sig = sig
        self._draw_plot()

    def _on_trend_curves_error(self, message: str, sig: tuple) -> None:
        self._trend_curve_compute_active = False
        if self._redispatch_pending_trend_compute():
            return
        if self._trend_overlay is not None:
            self._trend_overlay.hide()
        # Draw the data points without overlay curves; mark the (empty) cache
        # current so the failure isn't retried on every redraw for this signature.
        self._precomputed_trend_curves = {}
        self._trend_cache_sig = sig
        self._draw_plot()

    def shutdown_workers(self) -> None:
        """Cancel and join the trend-curve recompute worker (called on close)."""
        self._tasks.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown_workers()
        self._unregister_knight_labels()
        super().closeEvent(event)

    def _write_fit_files(
        self, gle_path: Path, x_key: str, y_params: list[str]
    ) -> dict[tuple[str, int], Path]:
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

            sampled = self._sample_fit_range_curve(
                fit_range,
                x_key=x_key,
                num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
            )
            if sampled is None:
                continue
            x_vals, y_vals = sampled

            with open(fit_path, "w", encoding="utf-8") as f:
                f.write("! Parameter model fit curve\n")
                f.write("! Generated by Asymmetry (GLE-readable data file)\n")
                f.write(f"! parameter: {pname}\n")
                f.write(f"! x_variable: {x_key}\n")
                if fit_range.windows:
                    windows_text = " ".join(f"[{lo:g}, {hi:g}]" for lo, hi in fit_range.windows)
                    f.write(f"! fit_windows: {windows_text}\n")
                    f.write(f"! x_min: {x_vals[0]:.10g}\n")
                    f.write(f"! x_max: {x_vals[-1]:.10g}\n")
                else:
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

            model_label = self._export_model_label()
            if model_label:
                f.write(f"! Model: {model_label}\n")

            if self._global_params is not None:
                f.write("! Global fitting parameters:\n")
                for param in self._global_params:
                    unit = get_param_info(param.name).unit
                    label = f"{param.name} ({unit})" if unit else param.name
                    err = self._global_param_uncertainties.get(param.name)
                    if err is not None:
                        f.write(f"!   {label} = {param.value:.6g} +/- {err:.6g}\n")
                    else:
                        f.write(f"!   {label} = {param.value:.6g}\n")

            f.write("!\n")

            headers = ["Run", "B_field(G)", "Temperature(K)"]
            for name in self._display_y_parameters():
                unit = get_param_info(name).unit
                if unit:
                    headers.extend([f"{name} ({unit})", f"err_{name} ({unit})"])
                else:
                    headers.extend([name, f"err_{name}"])
            # A free-text x-axis (Angle/custom) is appended as a trailing column so
            # GLE can plot against it; param columns keep their fixed indices.
            abscissa = self._export_abscissa_column()
            abscissa_key = abscissa[0] if abscissa is not None else None
            if abscissa is not None:
                headers.append(abscissa[1])
            # Per-run goodness of fit as trailing columns. Appended *after* the
            # abscissa so the fixed parameter/abscissa column indices that
            # _gle_columns_for_param / _gle_x_column compute are unaffected.
            headers.extend(["reduced_chi2", "chi2"])

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
                values: list[float] = [
                    float(row.run_number),
                    float(row.field),
                    float(row.temperature),
                ]
                for name in self._display_y_parameters():
                    values.append(row.values.get(name, np.nan))
                    values.append(row.errors.get(name, np.nan))
                if abscissa_key is not None:
                    values.append(self._x_value(row, abscissa_key))
                values.append(
                    row.reduced_chi_squared if row.reduced_chi_squared is not None else np.nan
                )
                values.append(row.chi_squared if row.chi_squared is not None else np.nan)
                f.write(" ".join(f"{v:>16.8g}" for v in values) + "\n")

    def _gle_x_column(self, x_key: str) -> int:
        name = _x_param_name(x_key)
        if name is not None:
            # Parameter-vs-parameter: x is the fitted parameter's value column,
            # which the data file already emits (with its error column alongside).
            cols = self._gle_columns_for_param(name)
            return cols[0] if cols is not None else 1
        if x_key == "run":
            return 1
        if x_key == "field":
            return 2
        if x_key == "temperature":
            return 3
        # Angle / custom column: the trailing column appended by _write_gle_data_file
        # (after Run/B/T and the 2-per-parameter columns).
        return 4 + 2 * len(self._display_y_parameters())

    def _gle_columns_for_param(self, name: str) -> tuple[int, int] | None:
        display_params = self._display_y_parameters()
        if name not in display_params:
            return None
        idx = display_params.index(name)
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

        show_components = self._show_components_check.isChecked()
        component_colors = ["lightblue", "lightgreen", "pink", "lightgray", "cyan", "yellow"]

        curves = self._sampled_fit_curves(
            param_name,
            x_key=fit.x_key,
            num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
        )
        for idx, (range_index, xs, ys) in enumerate(curves):
            line_color = _fit_overlay_color(idx) if len(curves) > 1 else color
            line_label = (
                _fit_overlay_label(param_name, idx, len(curves), gle=True)
                if include_labels
                else None
            )
            base_name = f"{_safe_data_name(param_name)}_range_{int(range_index)}"

            if show_components and range_index < len(fit.ranges):
                fit_range = fit.ranges[range_index]
                result = fit_range.result
                if result is not None and result.success:
                    kwargs = {p.name: p.value for p in result.parameters}
                    components = fit_range.model.evaluate_components(
                        xs, additive_only=True, **kwargs
                    )
                    ordered_components = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xs, dtype=float)
                    for cidx, (_cname, comp_y) in enumerate(ordered_components):
                        cname = _safe_data_name(_cname)
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(
                            xs,
                            lower,
                            upper,
                            color=fill_color,
                            alpha=0.3,
                            data_name=f"component_{base_name}_{cname}_fill",
                        )
                        ax.plot(
                            xs,
                            upper,
                            linestyle="--",
                            color=fill_color,
                            linewidth=1,
                            yaxis=yaxis,
                            data_name=f"component_{base_name}_{cname}_edge",
                        )
                        cumulative = upper

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
                    data_name=f"model_{base_name}",
                )

    def _add_gle_annotations(self, ax, axis_tag: str) -> None:
        """Add user plot annotations to a specific exported axis."""
        for ann in self._plot_annotations:
            if str(ann.get("axis_tag", "main")) != axis_tag:
                continue
            text = str(ann.get("text", "")).strip()
            if not text:
                continue
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            ax.text(x, y, text, color="black", ha="left")

    def _export_gle(self) -> None:
        if not self._rows:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to GLE",
            default_export_path("fit_parameters.gleplot"),
            "GLE export folders (*.gleplot);;All files (*)",
        )
        if not path:
            return
        remember_export_path(path)

        requested_gle_path = Path(path)
        gle_path, export_dir = resolve_gle_export_paths(requested_gle_path, folder=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        data_path = gle_path.with_suffix(".dat")
        self._write_gle_data_file(data_path)
        x_key = self._effective_x_key()

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
        self._generate_gle_plot(
            requested_gle_path,
            gle_path,
            data_path,
            output_format,
            fit_file_map,
        )

    def _generate_gle_plot(
        self,
        requested_gle_path: Path,
        gle_path: Path,
        data_path: Path,
        output_format: str,
        fit_file_map: dict[tuple[str, int], Path] | None = None,
    ) -> None:
        is_test_mode = bool(os.environ.get("PYTEST_CURRENT_TEST"))

        try:
            glp = importlib.import_module("gleplot")
        except ImportError:
            QMessageBox.warning(
                self, "gleplot not available", "Install gleplot to export GLE plots."
            )
            return

        if not hasattr(glp, "Axes") or not hasattr(glp.Axes, "errorbar_from_file"):
            QMessageBox.warning(
                self, "gleplot update required", "Please update gleplot to a newer version."
            )
            return

        if fit_file_map and (not hasattr(glp.Axes, "line_from_file")):
            QMessageBox.warning(
                self, "gleplot update required", "Please update gleplot to a newer version."
            )
            return

        x_key = self._effective_x_key()
        display_params = self._display_y_parameters()
        y_params = self._selected_y_parameters() or ([display_params[0]] if display_params else [])
        if not y_params:
            return

        x_label = _format_x_label_gle(x_key, self._custom_x_labels())
        data_file_ref = data_path.name
        x_col = self._gle_x_column(x_key)
        plot_mode = self._plot_mode_combo.currentText()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        show_fit_legend = self._count_fit_curves(x_key, y_params) > 1

        if plot_mode == "Subplots" and len(y_params) > 1:
            fig, axes = glp.subplots(
                nrows=len(y_params), ncols=1, figsize=(5.8, 3.0 * len(y_params)), sharex=True
            )
            subplot_axes = axes if isinstance(axes, list) else [axes]
            for idx, y_name in enumerate(y_params):
                cols = self._gle_columns_for_param(y_name)
                if cols is None:
                    continue
                y_col, yerr_col = cols
                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))
                ax = subplot_axes[idx]
                show_subplot_fit_legend = self._count_fit_curves_for_param(x_key, y_name) > 1
                self._add_gle_model_overlay(
                    ax,
                    y_name,
                    color=_fit_overlay_color(0),
                    yaxis="y",
                    include_labels=show_subplot_fit_legend,
                    fit_file_map=fit_file_map,
                )
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
                ax.set_xlabel(x_label)
                ax.set_ylabel(_format_gle_label(y_name))
                if self._log_x_check.isChecked():
                    ax.set_xscale("log")
                if not self._show_components_check.isChecked() and self._is_log_y_for(y_name):
                    ax.set_yscale("log")
                if self._show_components_check.isChecked():
                    ax.set_ylim(0.0, None)
                self._add_gle_annotations(ax, y_name)
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
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=x_col,
                    y_col=left_y_col,
                    yerr_col=left_err_col if has_left_err else None,
                    color=_gle_series_color(0),
                    marker="o",
                    markersize=5,
                    capsize=2,
                    yaxis="y",
                )
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=x_col,
                    y_col=right_y_col,
                    yerr_col=right_err_col if has_right_err else None,
                    color=_gle_series_color(1),
                    marker="o",
                    markersize=5,
                    capsize=2,
                    yaxis="y2",
                )
                ax.set_ylabel(_format_gle_label(left_name), axis="y")
                ax.set_ylabel(_format_gle_label(right_name), axis="y2")
                if self._show_components_check.isChecked():
                    ax.set_ylim(0.0, None, axis="y")
                    ax.set_ylim(0.0, None, axis="y2")
                self._add_gle_annotations(ax, left_name)
                self._add_gle_annotations(ax, right_name)
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
                    self._add_gle_model_overlay(
                        ax,
                        y_name,
                        color=_fit_overlay_color(idx),
                        yaxis="y",
                        include_labels=show_fit_legend,
                        fit_file_map=fit_file_map,
                    )
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

                if len(y_params) == 1:
                    ax.set_ylabel(_format_gle_label(y_params[0]))
                else:
                    ax.set_ylabel("Parameter Value")
                    if show_fit_legend:
                        ax.legend(loc="best")

                if self._show_components_check.isChecked():
                    ax.set_ylim(0.0, None)
                elif len(y_params) == 1 and self._is_log_y_for(y_params[0]):
                    ax.set_yscale("log")
                elif len(y_params) > 1 and any(self._is_log_y_for(name) for name in y_params):
                    ax.set_yscale("log")

                self._add_gle_annotations(ax, "main")

            ax.set_xlabel(x_label)
            if self._log_x_check.isChecked():
                ax.set_xscale("log")

        try:
            fig.savefig(str(gle_path))
        except TypeError as exc:
            if "folder" in str(exc):
                QMessageBox.warning(
                    self, "gleplot update required", "Please update gleplot to a newer version."
                )
                return
            raise

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
                    self._show_gle_preview(
                        fig, data_path, list(fit_file_map.values()) if fit_file_map else []
                    )
            except subprocess.CalledProcessError as exc:
                if not is_test_mode:
                    QMessageBox.warning(self, "GLE compilation failed", exc.stderr or str(exc))
                    self._show_gle_preview(
                        fig, data_path, list(fit_file_map.values()) if fit_file_map else []
                    )
        else:
            if not is_test_mode:
                QMessageBox.information(
                    self,
                    "GLE Not Installed",
                    f"GLE script saved to {gle_path}. Install GLE to compile to {output_format.upper()}.",
                )
                self._show_gle_preview(
                    fig, data_path, list(fit_file_map.values()) if fit_file_map else []
                )

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

            _gle = get_gle_executable()
            if _gle is not None:
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
                    subprocess.run(
                        [_gle, "-d", "png", str(gle_file)],
                        capture_output=True,
                        check=True,
                        cwd=str(tmpdir_path),
                    )

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
