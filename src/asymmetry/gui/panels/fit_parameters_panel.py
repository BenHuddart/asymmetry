"""Panel for inspecting fitted parameters across multiple datasets."""

from __future__ import annotations

import csv
import importlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
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
from asymmetry.core.fitting.composite_parameters import (
    CompositeExpression,
    CompositeExpressionError,
    CompositeParameterDefinition,
)
from asymmetry.core.fitting.engine import FitResult
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
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)
from asymmetry.gui.gle_settings import get_gle_executable
from asymmetry.gui.panels.composite_parameter_dialog import CompositeParameterDialog
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
from asymmetry.gui.styles.widgets import (
    apply_param_table_style,
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


def _format_x_label_gle(x_key: str) -> str:
    if x_key == "field":
        return "{\\it{B}} (G)"
    if x_key == "temperature":
        return "{\\it{T}} (K)"
    name = _x_param_name(x_key)
    if name is not None:
        return _format_gle_label(name)
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

    Recognises the three reserved run-level axes (``field``/``temperature``/
    ``run``) and the ``param:<name>`` namespace used for parameter-vs-parameter
    trending (item 1). Anything else collapses to ``run``.
    """
    text = str(value or "").strip()
    if text in ("field", "temperature", "run"):
        return text
    if text.startswith("param:") and len(text) > len("param:"):
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
        self._y_controls: dict[str, _YParamControls] = {}
        self._selected_y_param_names: list[str] = []
        self._model_fits: dict[str, ParameterModelFit] = {}
        self._composite_parameters: list[CompositeParameterDefinition] = []
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
        x_row.addWidget(self._log_x_check)
        x_container = QWidget()
        x_container.setLayout(x_row)
        controls_form.addRow("X axis:", x_container)

        self._y_selector_table = QTableWidget(0, 3)
        self._y_selector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._y_selector_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._y_selector_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._y_selector_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._y_selector_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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

        controls_form.addRow("Y parameters:", self._y_selector_table)

        self._create_composite_btn = QPushButton("New composite")
        self._create_composite_btn.setToolTip("Create a composite (derived) parameter.")
        self._create_composite_btn.setEnabled(False)
        self._create_composite_btn.clicked.connect(self._open_composite_parameter_dialog)

        self._edit_composite_btn = QPushButton("Edit composite")
        self._edit_composite_btn.setToolTip("Edit the selected composite parameter.")
        self._edit_composite_btn.setEnabled(False)
        self._edit_composite_btn.clicked.connect(self._edit_selected_composite_parameter)

        self._remove_composite_btn = QPushButton("Remove composite")
        self._remove_composite_btn.setToolTip("Remove the selected composite parameter.")
        self._remove_composite_btn.setEnabled(False)
        self._remove_composite_btn.clicked.connect(self._remove_selected_composite_parameters)

        self._derived_section = CollapsibleSection("Derived parameters", expanded=False)
        self._derived_section.setObjectName("fit-parameters-derived-section")
        composite_row = QGridLayout()
        composite_row.setContentsMargins(0, 0, 0, 0)
        composite_row.setHorizontalSpacing(6)
        composite_row.setVerticalSpacing(6)
        composite_row.addWidget(self._create_composite_btn, 0, 0)
        composite_row.addWidget(self._edit_composite_btn, 0, 1)
        composite_row.addWidget(self._remove_composite_btn, 1, 0, 1, 2)
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

        self._export_csv_btn = QPushButton("Export CSV")
        self._export_csv_btn.setEnabled(False)
        self._export_csv_btn.clicked.connect(self._export_csv)

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

        self._plot_labels_bar = QWidget(self._plot_group)
        labels_row = QHBoxLayout(self._plot_labels_bar)
        labels_row.setContentsMargins(0, 0, 0, 0)
        labels_row.setSpacing(6)
        labels_row.addWidget(QLabel("Plot labels:"))
        labels_row.addWidget(self._add_label_btn)
        labels_row.addWidget(self._clear_labels_btn)
        labels_row.addStretch()
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
        export_row = QHBoxLayout(self._plot_export_bar)
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(6)
        export_row.addWidget(QLabel("Export:"))
        export_row.addWidget(self._export_csv_btn)
        export_row.addWidget(self._export_gle_btn)
        export_row.addWidget(QLabel("Format:"))
        export_row.addWidget(self._gle_format_combo)
        export_row.addStretch()
        plot_layout.addWidget(self._plot_export_bar)

        controls_group.setMinimumHeight(0)
        controls_group.setMinimumWidth(0)

        self._controls_scroll = QScrollArea(self)
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._controls_scroll.setMinimumHeight(0)
        self._controls_scroll.setWidget(controls_group)

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
            "inferred_x_key": self._inferred_x_key,
            "x_axis": self._x_combo.currentText(),
            "x_axis_key": self._effective_x_key(),
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

    def restore_state(self, state: dict) -> None:
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
        # The table is cheap; _refresh_plot routes the heavy overlay curves
        # (model eval per fit range over an 800-pt axis — e.g. DiffusionLF_2D runs
        # scipy quadrature per sample) onto a worker behind the overlay, so a
        # saved project's trend fits don't block the GUI thread on open.
        self._refresh_table()
        self._refresh_plot()

    def _restore_state_locked(self, state: dict) -> None:
        rows_data = state.get("rows", [])
        self._composite_parameters = self._deserialize_composite_parameters(
            state.get("composite_parameters", [])
        )
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
                        )
                    )
                except Exception:
                    continue

        self._rows = restored_rows
        self._show_table_btn.setEnabled(bool(self._rows))
        self._export_csv_btn.setEnabled(bool(self._rows))
        self._export_gle_btn.setEnabled(bool(self._rows))
        self._gle_format_combo.setEnabled(bool(self._rows))
        self._create_composite_btn.setEnabled(bool(self._rows))
        self._edit_composite_btn.setEnabled(False)
        self._remove_composite_btn.setEnabled(False)

        varying = state.get("varying_params", [])
        if isinstance(varying, list) and all(isinstance(v, str) for v in varying):
            self._varying_params = list(varying)
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

        # Prefer the resolved x-axis key so a param:<name> selection survives
        # label collisions; fall back to the legacy combo-text match.
        restored_x = False
        x_axis_key = state.get("x_axis_key")
        if isinstance(x_axis_key, str) and x_axis_key.startswith("param:"):
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
                        )
                    )
                except Exception:
                    continue
            if not rows:
                continue

            prev = preserved.get(batch_id, {})
            composite_params = list(prev.get("composite_parameters", []))
            global_uncert = dict(prev.get("global_param_uncertainties", {}))
            varying = self._detect_varying_parameters(rows)
            inferred_x = self._infer_x_key(rows)
            self._apply_composite_parameters_to_rows(rows, composite_params, global_uncert)

            self._group_fit_results[batch_id] = _GroupFitData(
                group_id=batch_id,
                group_name=series_name,
                rows=rows,
                global_params=None,
                varying_params=varying,
                inferred_x_key=inferred_x,
                model_fits=dict(prev.get("model_fits", {})),
                plot_annotations=list(prev.get("plot_annotations", [])),
                global_param_uncertainties=global_uncert,
                composite_parameters=composite_params,
            )

        # Update per-series run-number map for browser highlighting.
        if highlight_runs_by_id is not None:
            self._series_run_numbers = dict(highlight_runs_by_id)

        # Activate the most-recently-added series (last entry) if possible;
        # otherwise keep the existing active group if it survived the reload.
        if series_entries and self._active_group_id not in self._group_fit_results:
            self._active_group_id = series_entries[-1][0]
        elif not self._group_fit_results:
            self._active_group_id = None

        self._rebuild_group_buttons()
        if self._active_group_id:
            self._set_selected_group_ids([self._active_group_id], emit=False)
        self._apply_group_selection_to_view(sync_active=False)
        # Notify listeners of the initial active series so data-browser highlights
        # fire immediately rather than waiting for the user to click a button.
        if self._active_group_id is not None:
            self.series_selection_changed.emit(self._active_group_id)

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
        while self._group_tabs_layout.count() > 0:
            item = self._group_tabs_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

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
            self._export_csv_btn.setEnabled(False)
            self._export_gle_btn.setEnabled(False)
            self._gle_format_combo.setEnabled(False)
            self._create_composite_btn.setEnabled(False)
            self._edit_composite_btn.setEnabled(False)
            self._remove_composite_btn.setEnabled(False)
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
            self._plot_annotations = list(active_group.plot_annotations)

        has_rows = bool(self._rows)

        self._apply_composite_parameters_to_rows(
            self._rows,
            self._composite_parameters,
            self._global_param_uncertainties,
        )

        self._show_table_btn.setEnabled(has_rows)
        self._export_csv_btn.setEnabled(has_rows)
        self._export_gle_btn.setEnabled(has_rows)
        self._gle_format_combo.setEnabled(has_rows)
        self._create_composite_btn.setEnabled(has_rows)

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
        self._plot_refresh_timer.start()

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
        return params

    def _selected_composite_parameter_names(self) -> list[str]:
        selected = set(self._selected_y_parameters())
        composite_names = {definition.name for definition in self._composite_parameters}
        return [
            name
            for name in self._display_y_parameters()
            if name in selected and name in composite_names
        ]

    def _update_composite_action_buttons(self) -> None:
        has_rows = bool(self._rows)
        selected_composites = self._selected_composite_parameter_names()
        self._edit_composite_btn.setEnabled(has_rows and len(selected_composites) == 1)
        self._remove_composite_btn.setEnabled(has_rows and len(selected_composites) >= 1)

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

        if not definitions:
            return

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

    def _detect_varying_parameters(self, rows: list[_FitRow]) -> list[str]:
        if not rows:
            return []

        composite_names = {definition.name for definition in self._composite_parameters}
        all_names = sorted(name for name in rows[0].values.keys() if name not in composite_names)
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
        if isinstance(data, str) and data.startswith("param:"):
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
        self._refresh_views()

    def _update_x_axis_auto_hint(self) -> None:
        if self._x_combo.currentText() != "Auto":
            self._x_auto_hint.setText("")
            return
        inferred_label = {"field": "(B)", "temperature": "(T)", "run": "(Run)"}
        self._x_auto_hint.setText(inferred_label.get(self._inferred_x_key, "(Run)"))

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
        for name in self._display_y_parameters():
            combo.addItem(_format_param_label(name), userData=f"param:{name}")
        restored = False
        if isinstance(prev_data, str) and prev_data.startswith("param:"):
            idx = combo.findData(prev_data)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                restored = True
        if not restored:
            idx = combo.findText(prev_text)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _rebuild_y_controls(self, *, preferred_selected: list[str] | None = None) -> None:
        self._y_selector_table.blockSignals(True)
        self._y_selector_table.clearContents()
        self._y_selector_table.setRowCount(0)

        self._y_controls = {}

        display_params = self._display_y_parameters()

        # Keep the X-axis selector's parameter entries in sync with the
        # trendable parameters (param-vs-param trending, item 1).
        self._rebuild_x_axis_combo()

        if not display_params:
            self._set_y_table_visible_rows(3)
            self._y_selector_table.blockSignals(False)
            return

        self._y_selector_table.setRowCount(len(display_params))

        for idx, name in enumerate(display_params):
            name_item = QTableWidgetItem(_format_param_label(name))
            name_item.setData(Qt.ItemDataRole.UserRole, name)
            self._y_selector_table.setItem(idx, 0, name_item)

            fit_button = QPushButton("Model Fit")
            fit_button.setMinimumWidth(
                fit_button.fontMetrics().horizontalAdvance("Model Fit*") + 36
            )
            fit_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
            fit_button.clicked.connect(
                lambda _checked=False, p=name: self._open_model_fit_dialog(p)
            )
            self._y_selector_table.setCellWidget(idx, 1, fit_button)

            log_check = QCheckBox("log")
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
        name_column_width = max(
            (
                self._y_selector_table.fontMetrics().horizontalAdvance(_format_param_label(name))
                for name in display_params
            ),
            default=120,
        )
        frame = 2 * self._y_selector_table.frameWidth()
        scroll_width = self._y_selector_table.style().pixelMetric(
            self._y_selector_table.style().PixelMetric.PM_ScrollBarExtent,
        )
        minimum_width = (
            name_column_width + fit_column_width + log_column_width + frame + scroll_width + 28
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
        selected = self._selected_composite_parameter_names()
        if not selected:
            return

        if len(selected) == 1:
            message = f"Remove composite parameter '{selected[0]}'?"
        else:
            names = ", ".join(selected)
            message = f"Remove selected composite parameters ({names})?"

        confirm = QMessageBox.question(
            self,
            "Remove Composite Parameter",
            message,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        names_to_remove = set(selected)
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

        preferred_selected = [
            name for name in self._selected_y_param_names if name not in names_to_remove
        ]
        self._refresh_after_composite_change(preferred_selected=preferred_selected)

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

    def _refresh_plot(self) -> None:
        """Redraw the trend plot, recomputing overlay curves off-thread if stale.

        Routing entry point: the scatter is cheap, but the model-fit overlays can
        be very slow (e.g. DiffusionLF_2D runs scipy quadrature per sample). When
        the overlay inputs changed since the cache was built, recompute them on a
        worker behind the overlay; otherwise (pure-render toggles, or no active
        overlays) draw synchronously now.
        """
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

        self._draw_plot_annotations(axes_by_tag)

        if getattr(self._figure, "get_constrained_layout", lambda: False)():
            layout_engine = getattr(self._figure, "get_layout_engine", lambda: None)()
            if layout_engine is not None and hasattr(layout_engine, "set"):
                layout_engine.set(w_pad=0.04, h_pad=0.04, hspace=0.05, wspace=0.05)
        else:
            self._figure.tight_layout(pad=1.2)
        self._canvas.draw()

    def _x_axis_label_mpl(self, x_key: str) -> str:
        name = _x_param_name(x_key)
        if name is not None:
            return _format_plot_label(name)
        return {"field": "$B$ (G)", "temperature": "$T$ (K)", "run": "Run Number"}.get(
            x_key, "Run Number"
        )

    def _x_value(self, row: _FitRow, x_key: str) -> float:
        name = _x_param_name(x_key)
        if name is not None:
            return float(row.values.get(name, float("nan")))
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

    def _export_csv(self) -> None:
        if self._table.columnCount() == 0 or self._table.rowCount() == 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Parameter Table",
            default_export_path("fit_parameters.csv"),
            "CSV files (*.csv);;All files (*)",
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

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in range(self._table.rowCount()):
                values: list[str] = []
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    values.append(item.text() if item is not None else "")
                writer.writerow(values)

    def _serialize_model_fits(self) -> dict:
        payload: dict[str, dict] = {}
        for param_name, model_fit in self._model_fits.items():
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
        return 3

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
        display_params = self._display_y_parameters()
        self._selected_y_parameters() or ([display_params[0]] if display_params else [])

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

        x_label = _format_x_label_gle(x_key)
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
