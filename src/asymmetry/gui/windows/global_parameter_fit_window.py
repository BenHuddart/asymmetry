"""Undocked window for cross-group global parameter fit results."""

from __future__ import annotations

import importlib
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
    evaluate_parameter_model_fit,
    parse_fit_windows,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)
from asymmetry.gui.gle_settings import get_gle_executable
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import apply_param_table_style
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.utils.export import compile_gle
from asymmetry.gui.widgets.loading_overlay import LoadingOverlay
from asymmetry.gui.widgets.mpl_canvas import create_canvas
from asymmetry.gui.windows.global_fit_window_helpers import (
    CorrelationMatrixDialog,
    _label_indicates_no_verdict,
    global_table_csv,
    global_table_latex,
    global_table_tsv,
)

#: Stable stacked-component fill colours, indexed by component order (after
#: ``_ordered_components_for_stacking``). One legend entry per component name.
_COMPONENT_COLORS = [
    "#8ecae6",
    "#90be6d",
    "#f4a261",
    "#e5989b",
    "#bdb2ff",
    "#ffd166",
]

_PARAMETER_FIT_CURVE_SAMPLE_COUNT = 800

#: Absolute floor (guards the ``err > 3×|value|`` test when ``value`` is ~0) for
#: the uncertainty trust flag. An uncertainty above ``max(3×|value|, this)`` — or
#: a non-finite uncertainty — marks the parameter as effectively unconstrained.
_UNCERTAINTY_FLAG_ABS_FLOOR = 1e-300


def _uncertainty_is_untrustworthy(value: float, err: float | None) -> bool:
    """Return ``True`` when *err* signals a degenerate/unconstrained parameter.

    The parameter is flagged when the reported uncertainty is present but
    non-finite, or exceeds ``max(3×|value|, _UNCERTAINTY_FLAG_ABS_FLOOR)`` — i.e.
    the error bar is larger than (three times) the value itself, so the fit did
    not meaningfully pin the parameter down. ``err is None`` (fixed / no
    covariance) is *not* flagged.
    """
    if err is None:
        return False
    try:
        e = float(err)
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(e):
        return True
    threshold = max(3.0 * abs(v), _UNCERTAINTY_FLAG_ABS_FLOOR)
    return e > threshold


#: Minimum on-screen height (px) for one group panel in the Fig-3 grid. The left
#: canvas lives inside a :class:`QScrollArea` and is given a minimum height of
#: ``nrows × _MIN_PANEL_PX + _GRID_MARGIN_PX`` so panels never squeeze below
#: legibility; the grid scrolls vertically instead (horizontal always fits).
_MIN_PANEL_PX = 230
#: Fixed vertical slack added to the per-row panel budget for the figure legend
#: band, inter-panel padding, and axis labels.
_GRID_MARGIN_PX = 60


@dataclass(frozen=True)
class StudySidebarEntry:
    """One row of the studies sidebar, driven by the MainWindow.

    Carries enough to render the row (``name``/``stale``) and to build the
    "Compare with…" submenu: two studies can be compared only when they share a
    ``parameter_name`` *and* an ``x_key`` (so the per-group overlay is
    apples-to-apples). Replaces the earlier bare ``(study_id, name, stale)``
    tuple that :meth:`GlobalParameterFitWindow.set_studies_list` used to take.
    """

    study_id: str
    name: str
    stale: bool
    parameter_name: str
    x_key: str


#: Keys in :meth:`GlobalParameterFitWindow.get_state` that are *decorations* —
#: trend-attached state (local model fits, plot annotations) that persists inside
#: the owning ``FitSeries.extra`` rather than under the window-state project key.
#: Everything else in the state dict is a view preference and stays in the key.
_DECORATION_STATE_KEYS = (
    "local_model_fits",
    "plot_annotations",
    "local_plot_annotations",
    "suppressed_group_label_tags",
)


class GlobalParameterFitWindow(QMainWindow):
    """Display cross-group fit data, fitted model curves, and global/local values."""

    #: Emitted (study_id) when the user clicks Refit on the stale banner. The
    #: MainWindow re-assembles the study's groups from the live trend data, re-runs
    #: the fit off-thread, and updates the study.
    refit_requested = Signal(str)
    #: Emitted (study_id) when a sidebar row is selected — the MainWindow displays
    #: that study.
    study_selected = Signal(str)
    #: Emitted (study_id, new_name) from the sidebar Rename… context action.
    study_rename_requested = Signal(str, str)
    #: Emitted (study_id) from the sidebar Duplicate context action.
    study_duplicate_requested = Signal(str)
    #: Emitted (study_id) from the sidebar Delete… context action.
    study_delete_requested = Signal(str)
    #: Emitted (id_a, id_b) from the sidebar "Compare with…" submenu: ``id_a`` is
    #: the right-clicked study, ``id_b`` the chosen comparison target (same
    #: parameter_name and x_key). The MainWindow opens the comparison dialog.
    study_compare_requested = Signal(str, str)
    #: Emitted (study_id) from the "Edit fit…" button; the MainWindow reopens the
    #: cross-group fit dialog seeded from the study config against current data.
    edit_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Parameter Fit")
        self.resize(1200, 800)
        #: The id of the study currently displayed (== batch id), for the Refit
        #: signal. ``None`` until :meth:`set_study`/:meth:`set_results` runs.
        self._study_id: str | None = None

        self._result: CrossGroupFitResult | None = None
        self._groups: list[ParameterGroupData] = []
        self._model = None
        #: The ``modelfit-<digest>`` batch id of the fit currently displayed. The
        #: window's *decorations* (local model fits, plot annotations) belong to
        #: this series' ``extra`` rather than to a standalone project key, so they
        #: cannot orphan when the backing fit is re-run or removed.
        self._batch_id: str | None = None
        self._parameter_name: str | None = None
        self._x_key: str = "run"
        self._fit_x_min: float = float("nan")
        self._fit_x_max: float = float("nan")
        #: Study-carried axis labels (Phase 3): when set they override the
        #: hardcoded field/T/run maps in ``_x_label*`` / ``_local_group_axis_*``.
        #: ``None`` falls back to the legacy hardcoded maps (older results).
        self._x_label_override: str | None = None
        self._group_variable_label_override: str | None = None
        #: Per-group show/hide state for the Fig-3 grid (group_id -> shown). A
        #: hidden group is excluded from the grid but NOT from the fit/result.
        self._group_visibility: dict[str, bool] = {}

        self._axes_tag_map: dict[int, str] = {}
        self._local_axes_tag_map: dict[int, str] = {}
        self._plot_annotations: list[dict[str, object]] = []
        self._local_plot_annotations: list[dict[str, object]] = []
        self._suppressed_group_label_tags: set[str] = set()
        self._local_y_controls: dict[str, dict[str, object]] = {}
        self._local_model_fits: dict[str, ParameterModelFit] = {}
        self._local_param_log_y: dict[str, bool] = {}
        self._local_selected_y_names: list[str] = []
        self._restored_local_selected_y: list[str] = []
        self._add_label_mode = False
        self._dragging_annotation: dict[str, object] | None = None

        # Background machinery for the cross-group fit-curve evaluation, which
        # runs per group over an 800-point axis and would otherwise block the
        # GUI thread when a saved fit is restored on project open.
        self._tasks = TaskRunner(self)
        self._fit_curve_compute_active = False
        # Set when a fresh recompute is requested while one is in flight (the
        # set_results + restore_state + restore_decorations burst on project
        # open), so the burst collapses to a single rerun instead of a thread
        # per call.
        self._fit_curve_recompute_pending = False
        #: Window-lifetime curve cache: ``{show_components_flag: {group_id: curve}}``.
        #: The per-group curve dict carries ``xx``/``yy``/``components`` (the 800-pt
        #: model sample, optional stacked components) *and* ``pulls`` (the
        #: data-x residual ``(x, pull)`` arrays), all computed off-thread in one
        #: worker payload. Curves depend only on (result, model, groups, x-range,
        #: x_key, show_components) — every one of those changes solely via
        #: :meth:`set_results`, which is the only place the cache is cleared. Every
        #: user-driven control (Log X/Y, Share X/Y, Columns, Groups show/hide,
        #: Residuals, Show-components) is therefore a cache-backed redraw:
        #: :meth:`_refresh_plot` draws from the cache on a hit and kicks an
        #: off-thread compute (then returns) on a miss — it NEVER evaluates the
        #: model inline.
        self._curve_cache: dict[bool, dict[str, dict]] = {}

        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        # Stale banner: a highlighted row above the splitter, hidden unless the
        # displayed study's inputs have drifted from the live trend data. Its
        # Refit button asks the MainWindow to re-run the fit on current data.
        self._stale_banner = QWidget()
        stale_layout = QHBoxLayout(self._stale_banner)
        stale_layout.setContentsMargins(8, 4, 8, 4)
        self._stale_label = QLabel("")
        self._stale_label.setWordWrap(True)
        self._refit_btn = QPushButton("Refit")
        self._refit_btn.clicked.connect(self._on_refit_clicked)
        stale_layout.addWidget(self._stale_label, 1)
        stale_layout.addWidget(self._refit_btn, 0)
        self._stale_banner.setStyleSheet(
            f"QWidget {{ background-color: {tokens.WARN_BANNER_BG}; }}"
            f"QLabel {{ color: {tokens.WARN_BANNER_TEXT}; font-weight: bold; }}"
        )
        self._stale_banner.setVisible(False)
        root_layout.addWidget(self._stale_banner)

        # Outer splitter: [studies sidebar | plots+tables]. The sidebar is a
        # compact list of studies driven by the MainWindow; selecting a row asks
        # to display it (context menu: rename/duplicate/delete).
        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(outer_splitter)

        self._studies_list = QListWidget()
        self._studies_list.setMaximumWidth(220)
        self._studies_list.setMinimumWidth(140)
        self._studies_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._studies_list.customContextMenuRequested.connect(self._on_studies_context_menu)
        self._studies_list.currentRowChanged.connect(self._on_studies_row_changed)
        #: Guards :meth:`set_studies_list`/:meth:`set_active_study_id` from
        #: emitting ``study_selected`` while the MainWindow is repopulating.
        self._suppress_study_selection = False
        #: The most recent :class:`StudySidebarEntry` list, keyed by study_id, so
        #: the context menu can offer same-parameter/x_key comparison candidates.
        self._sidebar_entries: dict[str, StudySidebarEntry] = {}
        outer_splitter.addWidget(self._studies_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_splitter.addWidget(splitter)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([180, 1020])

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._left_canvas = None
        self._left_figure = None
        #: Vertical scroll host for the Fig-3 grid. ``widgetResizable`` lets the
        #: canvas fill the viewport width (so the grid never needs a horizontal
        #: scrollbar) while the canvas' minimum height — set to
        #: ``nrows × _MIN_PANEL_PX + _GRID_MARGIN_PX`` in
        #: :meth:`_update_grid_canvas_min_height` — forces a vertical scrollbar
        #: rather than squeezing panels below legibility.
        self._left_scroll: QScrollArea | None = None
        try:
            self._left_figure, self._left_canvas = create_canvas(layout="tight")
            self._left_scroll = QScrollArea()
            self._left_scroll.setWidgetResizable(True)
            self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._left_scroll.setWidget(self._left_canvas)
            left_layout.addWidget(self._left_scroll, 1)

            self._left_canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._left_canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self._left_canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        except ImportError:
            pass

        # Covers the fit plot while its per-group curves are recomputed off the
        # GUI thread (created only when the canvas exists).
        self._fit_overlay = (
            LoadingOverlay(self._left_canvas) if self._left_canvas is not None else None
        )

        fit_controls_row = QHBoxLayout()
        self._show_components_check = QCheckBox("Show components")
        self._show_components_check.toggled.connect(self._on_show_components_toggled)
        fit_controls_row.addWidget(self._show_components_check)

        self._fit_log_x_check = QCheckBox("Log X")
        self._fit_log_x_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_log_x_check)

        self._fit_log_y_check = QCheckBox("Log Y")
        self._fit_log_y_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_log_y_check)

        self._fit_share_x_check = QCheckBox("Share X Axis")
        self._fit_share_x_check.setChecked(False)
        self._fit_share_x_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_share_x_check)

        self._fit_share_y_check = QCheckBox("Share Y Axis")
        self._fit_share_y_check.setChecked(False)
        self._fit_share_y_check.toggled.connect(self._refresh_plot)
        fit_controls_row.addWidget(self._fit_share_y_check)

        self._fit_residuals_check = QCheckBox("Residuals")
        self._fit_residuals_check.setToolTip(
            "Show (data − model)/σ pulls per group instead of the fit curves."
        )
        self._fit_residuals_check.toggled.connect(self._on_residuals_toggled)
        fit_controls_row.addWidget(self._fit_residuals_check)

        fit_controls_row.addWidget(QLabel("Columns:"))
        self._fit_columns_combo = QComboBox()
        self._fit_columns_combo.addItems(["Auto", "1", "2", "3", "4"])
        self._fit_columns_combo.currentTextChanged.connect(lambda _t: self._refresh_plot())
        fit_controls_row.addWidget(self._fit_columns_combo)

        self._groups_popup_btn = QPushButton("Groups…")
        self._groups_popup_btn.setToolTip("Show or hide individual group panels.")
        self._groups_popup_btn.clicked.connect(self._show_groups_popup)
        fit_controls_row.addWidget(self._groups_popup_btn)

        fit_controls_row.addStretch()
        left_layout.addLayout(fit_controls_row)

        # The right pane is a vertical splitter of three blocks so the user can
        # trade space between the global parameter table, the local Y-selector,
        # and the local trend canvas: [params-table block] / [Y-selector] /
        # [local canvas]. Each block carries a minimum height so none collapses
        # to nothing when another is dragged large.
        right = QWidget()
        right_outer_layout = QVBoxLayout(right)
        right_outer_layout.setContentsMargins(0, 0, 0, 0)
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_outer_layout.addWidget(self._right_splitter)

        # ── Block 1: global parameter table (quality bar + table + actions) ──
        params_block = QWidget()
        right_layout = QVBoxLayout(params_block)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Quality bar: χ², χ²ᵣ, n, error-mode chip, and (on failure) the message.
        quality_row = QHBoxLayout()
        quality_row.setContentsMargins(0, 0, 0, 0)
        self._quality_label = QLabel("")
        self._quality_label.setTextFormat(Qt.TextFormat.RichText)
        quality_row.addWidget(self._quality_label, 1)
        self._edit_fit_btn = QPushButton("Edit fit…")
        self._edit_fit_btn.setToolTip("Reopen the cross-group fit dialog seeded from this fit.")
        self._edit_fit_btn.clicked.connect(self._on_edit_fit_clicked)
        quality_row.addWidget(self._edit_fit_btn, 0)
        right_layout.addLayout(quality_row)

        self._params_table = QTableWidget(0, 4)
        self._params_table.setHorizontalHeaderLabels(["Parameter", "Value", "Uncertainty", "Units"])
        apply_param_table_style(self._params_table)
        self._configure_params_table_header()
        right_layout.addWidget(self._params_table, 1)

        # Table actions: Copy (TSV to clipboard), Export (CSV/LaTeX), and the
        # correlation-matrix dialog (enabled only when correlations exist).
        table_actions_row = QHBoxLayout()
        table_actions_row.setContentsMargins(0, 0, 0, 0)
        self._copy_table_btn = QPushButton("Copy")
        self._copy_table_btn.setToolTip("Copy the global parameter table (TSV) to the clipboard.")
        self._copy_table_btn.clicked.connect(self._on_copy_global_table)
        table_actions_row.addWidget(self._copy_table_btn)
        self._export_table_btn = QPushButton("Export…")
        self._export_table_btn.setToolTip("Export the global parameter table as CSV or LaTeX.")
        self._export_table_btn.clicked.connect(self._on_export_global_table)
        table_actions_row.addWidget(self._export_table_btn)
        self._correlations_btn = QPushButton("Correlations…")
        self._correlations_btn.setToolTip("Show the free-global-parameter correlation matrix.")
        self._correlations_btn.clicked.connect(self._on_show_correlations)
        self._correlations_btn.setEnabled(False)
        table_actions_row.addWidget(self._correlations_btn)
        table_actions_row.addStretch()
        right_layout.addLayout(table_actions_row)
        params_block.setMinimumHeight(160)
        self._right_splitter.addWidget(params_block)

        # ── Block 2: local Y-parameter selector table ───────────────────────
        self._local_y_selector_table = QTableWidget(0, 3)
        self._local_y_selector_table.setHorizontalHeaderLabels(["Parameter", "Fit", "Log"])
        self._local_y_selector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._local_y_selector_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._local_y_selector_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._local_y_selector_table.horizontalHeader().setVisible(False)
        self._local_y_selector_table.verticalHeader().setVisible(False)
        self._local_y_selector_table.itemSelectionChanged.connect(
            self._on_local_y_selection_changed
        )
        self._configure_y_selector_header()
        self._local_y_selector_table.setMinimumHeight(80)
        self._right_splitter.addWidget(self._local_y_selector_table)

        # ── Block 3: local trend canvas (controls + canvas) ─────────────────
        canvas_block = QWidget()
        canvas_layout = QVBoxLayout(canvas_block)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        controls_row = QHBoxLayout()
        self._local_plot_mode_combo = QComboBox()
        self._local_plot_mode_combo.addItems(["Single Axes", "Subplots"])
        self._local_plot_mode_combo.currentTextChanged.connect(
            lambda _text: self._refresh_local_parameter_plots()
        )
        controls_row.addWidget(self._local_plot_mode_combo)

        self._local_log_x_check = QCheckBox("Log X")
        self._local_log_x_check.toggled.connect(self._refresh_local_parameter_plots)
        controls_row.addWidget(self._local_log_x_check)

        self._local_log_y_check = QCheckBox("Log Y")
        self._local_log_y_check.toggled.connect(self._refresh_local_parameter_plots)

        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._add_label_btn.toggled.connect(self._set_add_label_mode)
        controls_row.addWidget(self._add_label_btn)
        controls_row.addStretch()
        canvas_layout.addLayout(controls_row)

        self._local_canvas = None
        self._local_figure = None
        try:
            self._local_figure, self._local_canvas = create_canvas(layout="tight")
            canvas_layout.addWidget(self._local_canvas, 1)

            self._local_canvas.mpl_connect("button_press_event", self._on_local_canvas_button_press)
            self._local_canvas.mpl_connect("motion_notify_event", self._on_local_canvas_motion)
            self._local_canvas.mpl_connect(
                "button_release_event", self._on_local_canvas_button_release
            )
        except ImportError:
            pass
        canvas_block.setMinimumHeight(180)
        self._right_splitter.addWidget(canvas_block)

        self._right_splitter.setStretchFactor(0, 0)
        self._right_splitter.setStretchFactor(1, 0)
        self._right_splitter.setStretchFactor(2, 1)
        self._right_splitter.setSizes([260, 140, 360])

        self._export_btn = QPushButton("Export fits to GLE")
        self._export_btn.clicked.connect(self._export_fit_subplot_gle)

        self._fit_gle_format_combo = QComboBox()
        self._fit_gle_format_combo.addItems(["PDF", "EPS"])
        self._fit_subplot_aspect_spin = QDoubleSpinBox()
        self._fit_subplot_aspect_spin.setRange(1.0, 5.0)
        self._fit_subplot_aspect_spin.setSingleStep(0.1)
        self._fit_subplot_aspect_spin.setDecimals(2)
        self._fit_subplot_aspect_spin.setValue(2.61)
        self._fit_subplot_aspect_spin.setSuffix(" : 1")
        fit_export_row = QHBoxLayout()
        fit_export_row.addWidget(self._export_btn)
        fit_export_row.addWidget(QLabel("Aspect (W:H):"))
        fit_export_row.addWidget(self._fit_subplot_aspect_spin)
        fit_export_row.addWidget(QLabel("Format:"))
        fit_export_row.addWidget(self._fit_gle_format_combo)
        fit_export_row.addStretch()
        left_layout.addLayout(fit_export_row)

        self._export_local_btn = QPushButton("Export plot(s) to GLE")
        self._export_local_btn.clicked.connect(self._export_local_parameters_gle)

        self._local_gle_format_combo = QComboBox()
        self._local_gle_format_combo.addItems(["PDF", "EPS"])
        local_export_row = QHBoxLayout()
        local_export_row.addWidget(self._export_local_btn)
        local_export_row.addWidget(QLabel("Format:"))
        local_export_row.addWidget(self._local_gle_format_combo)
        local_export_row.addStretch()
        canvas_layout.addLayout(local_export_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 500])
        self._sync_fit_scale_controls()

    def _configure_params_table_header(self) -> None:
        """Apply per-section resize policies to the global-parameter table.

        Symbol/Units size to their content; Value/Uncertainty are user-resizable
        but never shrink below a content-derived minimum (so the "Uncertainty"
        header and wide numeric strings are never truncated); the last section
        stretches to fill. Reapplied on every populate via :meth:`_refresh_table`.
        """
        header = self._params_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        # Content-based minimums for the interactive numeric columns so neither
        # the header text nor a wide value/error string can be clipped.
        fm = header.fontMetrics()
        header.setMinimumSectionSize(
            max(header.minimumSectionSize(), fm.horizontalAdvance("Uncertainty") + 24)
        )

    def _configure_y_selector_header(self) -> None:
        """Apply per-section resize policies to the local Y-selector table.

        The parameter-name column stretches; the "Model Fit" button column and
        the "log" column size to their content so neither label truncates.
        Reapplied on every populate via :meth:`_rebuild_local_y_controls`.
        """
        header = self._local_y_selector_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)

    def _update_grid_canvas_min_height(self, nrows: int) -> None:
        """Size the scrolled Fig-3 canvas so each panel keeps a legible height.

        The canvas' minimum height is ``nrows × _MIN_PANEL_PX + _GRID_MARGIN_PX``.
        Inside the ``widgetResizable`` :class:`QScrollArea` this forces a vertical
        scrollbar (rather than squeezing panels) when the grid is taller than the
        viewport, while the width still tracks the viewport (no horizontal
        scrollbar). Called from :meth:`_refresh_plot` whenever the grid shape
        (row count, columns, show/hide, residual toggle) changes.
        """
        if self._left_canvas is None or self._left_scroll is None:
            return
        rows = max(1, int(nrows))
        min_h = rows * _MIN_PANEL_PX + _GRID_MARGIN_PX
        self._left_canvas.setMinimumHeight(min_h)

    def _sync_fit_scale_controls(self) -> None:
        components_on = self._show_components_check.isChecked()
        residuals_on = self._fit_residuals_check.isChecked()
        # Log-Y makes no sense for stacked components (which fill from 0) nor for
        # pull residuals (signed values around zero).
        if (components_on or residuals_on) and self._fit_log_y_check.isChecked():
            self._fit_log_y_check.setChecked(False)
        self._fit_log_y_check.setEnabled(not components_on and not residuals_on)
        # Components and the fit curve are meaningless in residual mode.
        self._show_components_check.setEnabled(not residuals_on)

    def _on_show_components_toggled(self, _checked: bool) -> None:
        self._sync_fit_scale_controls()
        # Cache-backed: _refresh_plot draws from the flag's cache entry on a hit,
        # or kicks the off-thread compute for that flag on the first miss.
        self._refresh_plot()

    def _on_residuals_toggled(self, _checked: bool) -> None:
        self._sync_fit_scale_controls()
        self._refresh_plot()

    def has_result(self) -> bool:
        return self._result is not None

    def _serialize_annotations(
        self, annotations: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for ann in annotations:
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(ann.get("text", "")),
                    "axis_tag": str(ann.get("axis_tag", "")),
                    "is_group_label": bool(ann.get("is_group_label", False)),
                }
            )
        return out

    def _deserialize_annotations(self, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, list):
            return []
        out: list[dict[str, object]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                x = float(entry.get("x", 0.0))
                y = float(entry.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(entry.get("text", "")),
                    "axis_tag": str(entry.get("axis_tag", "")),
                    "is_group_label": bool(entry.get("is_group_label", False)),
                    "artist": None,
                }
            )
        return out

    def _serialize_local_model_fits(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for param_name, model_fit in self._local_model_fits.items():
            ranges_data: list[dict[str, object]] = []
            for fit_range in model_fit.ranges:
                range_item: dict[str, object] = {
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

    def _deserialize_local_model_fits(self, state: object) -> dict[str, ParameterModelFit]:
        if not isinstance(state, dict):
            return {}

        restored: dict[str, ParameterModelFit] = {}
        for key, entry in state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue

            x_key = str(entry.get("x_key", "run"))
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

                params = ParameterSet()
                for p in range_state.get("parameters", []):
                    if not isinstance(p, dict):
                        continue
                    try:
                        params.add(
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

                result_obj = None
                result_state = range_state.get("result")
                if isinstance(result_state, dict):
                    result_params = ParameterSet()
                    for p in result_state.get("parameters", []):
                        if not isinstance(p, dict):
                            continue
                        try:
                            result_params.add(
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

    def get_state(self) -> dict[str, object]:
        return {
            "show_components": bool(self._show_components_check.isChecked()),
            "fit_log_x": bool(self._fit_log_x_check.isChecked()),
            "fit_log_y": bool(self._fit_log_y_check.isChecked()),
            "fit_share_x": bool(self._fit_share_x_check.isChecked()),
            "fit_share_y": bool(self._fit_share_y_check.isChecked()),
            "fit_residuals": bool(self._fit_residuals_check.isChecked()),
            "fit_columns": str(self._fit_columns_combo.currentText()),
            "fit_subplot_aspect": float(self._fit_subplot_aspect_spin.value()),
            "local_log_x": bool(self._local_log_x_check.isChecked()),
            "local_log_y": bool(self._local_log_y_check.isChecked()),
            "local_param_log_y": dict(self._local_param_log_y),
            "local_model_fits": self._serialize_local_model_fits(),
            "local_selected_y": list(self._local_selected_y_names),
            "local_plot_mode": str(self._local_plot_mode_combo.currentText()),
            "plot_annotations": self._serialize_annotations(self._plot_annotations),
            "local_plot_annotations": self._serialize_annotations(self._local_plot_annotations),
            "suppressed_group_label_tags": sorted(self._suppressed_group_label_tags),
        }

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return

        self._show_components_check.setChecked(bool(state.get("show_components", False)))
        self._fit_log_x_check.setChecked(bool(state.get("fit_log_x", False)))
        self._fit_log_y_check.setChecked(bool(state.get("fit_log_y", False)))
        self._fit_share_x_check.setChecked(bool(state.get("fit_share_x", False)))
        self._fit_share_y_check.setChecked(bool(state.get("fit_share_y", False)))
        self._fit_residuals_check.setChecked(bool(state.get("fit_residuals", False)))
        fit_columns = state.get("fit_columns")
        if isinstance(fit_columns, str) and fit_columns in {"Auto", "1", "2", "3", "4"}:
            self._fit_columns_combo.setCurrentText(fit_columns)
        self._local_log_x_check.setChecked(bool(state.get("local_log_x", False)))
        self._local_log_y_check.setChecked(bool(state.get("local_log_y", False)))
        raw_local_param_log = state.get("local_param_log_y", {})
        if isinstance(raw_local_param_log, dict):
            self._local_param_log_y = {str(k): bool(v) for k, v in raw_local_param_log.items()}
        else:
            self._local_param_log_y = {}
        fit_subplot_aspect = state.get("fit_subplot_aspect")
        try:
            if fit_subplot_aspect is not None:
                self._fit_subplot_aspect_spin.setValue(float(fit_subplot_aspect))
            else:
                # Backward compatibility: convert older saved width setting to aspect ratio.
                fit_plot_width = state.get("fit_plot_width")
                if fit_plot_width is not None:
                    self._fit_subplot_aspect_spin.setValue(float(fit_plot_width) / 3.1)
        except (TypeError, ValueError):
            pass

        local_plot_mode = state.get("local_plot_mode")
        if isinstance(local_plot_mode, str) and local_plot_mode in {"Single Axes", "Subplots"}:
            self._local_plot_mode_combo.setCurrentText(local_plot_mode)

        selected_y = state.get("local_selected_y", [])
        if isinstance(selected_y, list):
            self._local_selected_y_names = [str(p) for p in selected_y]
            self._restored_local_selected_y = list(self._local_selected_y_names)
        else:
            self._local_selected_y_names = []
            self._restored_local_selected_y = []

        # Decorations may travel inline in a legacy window-state dict (older
        # projects) or be applied separately from the owning series' ``extra``
        # (current projects, via :meth:`restore_decorations`). Applying any keys
        # present here keeps legacy projects loading unchanged.
        self._apply_decoration_state(state)

        self._sync_fit_scale_controls()
        if self.has_result():
            # Recompute the fit curves off-thread (controls like Show Components
            # change what is evaluated); coalesced with the set_results compute
            # already in flight during project restore.
            self._start_fit_curve_compute()
            self._refresh_local_parameter_plots()

    def _apply_decoration_state(self, state: dict) -> None:
        """Restore the decoration keys (local model fits, annotations) from *state*.

        Shared by :meth:`restore_state` (legacy inline dict) and
        :meth:`restore_decorations` (the series-attached home). A missing key
        resets that decoration to empty, matching the historical restore_state
        behaviour, so callers pass a complete decoration dict.
        """
        self._local_model_fits = self._deserialize_local_model_fits(
            state.get("local_model_fits", {})
        )
        self._plot_annotations = self._deserialize_annotations(state.get("plot_annotations", []))
        self._local_plot_annotations = self._deserialize_annotations(
            state.get("local_plot_annotations", [])
        )
        suppressed = state.get("suppressed_group_label_tags", [])
        if isinstance(suppressed, list):
            self._suppressed_group_label_tags = {str(tag) for tag in suppressed}
        else:
            self._suppressed_group_label_tags = set()

    def get_view_state(self) -> dict[str, object]:
        """Return the view-preference subset of :meth:`get_state`.

        This is what persists under the ``global_parameter_fit_window_state``
        project key. Decorations are excluded — they live in the owning series'
        ``extra`` (see :meth:`get_decorations`).
        """
        return {
            key: value
            for key, value in self.get_state().items()
            if key not in _DECORATION_STATE_KEYS
        }

    def get_decorations(self) -> dict[str, object]:
        """Return the decoration subset of :meth:`get_state`.

        These persist inside the displayed fit's ``FitSeries.extra`` keyed by
        batch id, so they cannot orphan when the fit is re-run or removed.
        """
        full = self.get_state()
        return {key: full[key] for key in _DECORATION_STATE_KEYS if key in full}

    def has_decorations(self) -> bool:
        """Return ``True`` when the window carries any user decorations."""
        return bool(
            self._local_model_fits
            or self._plot_annotations
            or self._local_plot_annotations
            or self._suppressed_group_label_tags
        )

    def restore_decorations(self, state: object) -> None:
        """Apply decorations loaded from the owning series' ``extra``."""
        if not isinstance(state, dict):
            return
        self._apply_decoration_state(state)
        if self.has_result():
            self._start_fit_curve_compute()
            self._refresh_local_parameter_plots()

    def _selected_local_y_parameters(self) -> list[str]:
        params: list[str] = []
        for row in sorted(
            {index.row() for index in self._local_y_selector_table.selectedIndexes()}
        ):
            item = self._local_y_selector_table.item(row, 0)
            if item is None:
                continue
            pname = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(pname, str) and pname:
                params.append(pname)
        return params

    def _on_local_y_selection_changed(self) -> None:
        self._local_selected_y_names = self._selected_local_y_parameters()
        self._refresh_local_parameter_plots()

    def _rebuild_local_y_controls(
        self, parameter_names: list[str], preferred_selected: list[str] | None = None
    ) -> None:
        self._local_y_selector_table.blockSignals(True)
        self._local_y_selector_table.clearContents()
        self._local_y_selector_table.setRowCount(0)
        self._local_y_controls = {}

        if not parameter_names:
            self._local_y_selector_table.blockSignals(False)
            return

        self._local_y_selector_table.setRowCount(len(parameter_names))
        for idx, name in enumerate(parameter_names):
            item = QTableWidgetItem(get_param_info(name).unicode_label())
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._local_y_selector_table.setItem(idx, 0, item)

            fit_button = QPushButton("Model Fit")
            fit_button.clicked.connect(
                lambda _checked=False, p=name: self._open_local_model_fit_dialog(p)
            )
            self._local_y_selector_table.setCellWidget(idx, 1, fit_button)

            log_check = QCheckBox("log")
            log_check.setChecked(bool(self._local_param_log_y.get(name, False)))
            log_check.toggled.connect(
                lambda checked, p=name: self._on_local_param_log_toggled(p, checked)
            )
            log_container = QWidget()
            log_layout = QHBoxLayout(log_container)
            log_layout.setContentsMargins(0, 0, 0, 0)
            log_layout.addWidget(log_check)
            log_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._local_y_selector_table.setCellWidget(idx, 2, log_container)

            self._local_y_controls[name] = {"fit_button": fit_button, "log": log_check}

            fit = self._local_model_fits.get(name)
            if fit is not None and fit.active and self._has_successful_local_fit_curve(fit):
                fit_button.setText("Model Fit*")

        preferred = [p for p in (preferred_selected or []) if p in parameter_names]
        if preferred:
            for idx, name in enumerate(parameter_names):
                item = self._local_y_selector_table.item(idx, 0)
                if item is not None and name in preferred:
                    item.setSelected(True)
        elif parameter_names:
            item = self._local_y_selector_table.item(0, 0)
            if item is not None:
                item.setSelected(True)

        # Reapply the section policies (Parameter stretches; Fit-button and log
        # columns size to content so "Model Fit"/"log" never truncate).
        self._configure_y_selector_header()
        self._local_selected_y_names = self._selected_local_y_parameters()
        self._local_y_selector_table.blockSignals(False)

    def _has_successful_local_fit_curve(self, fit: ParameterModelFit) -> bool:
        return any(r.result is not None and r.result.success for r in fit.ranges)

    def _on_local_param_log_toggled(self, param_name: str, checked: bool) -> None:
        self._local_param_log_y[param_name] = bool(checked)
        self._refresh_local_parameter_plots()

    def _is_local_param_log_enabled(self, param_name: str) -> bool:
        return bool(self._local_param_log_y.get(param_name, False))

    def _local_group_x_key(self) -> str:
        if self._x_key == "field":
            return "temperature"
        if self._x_key == "temperature":
            return "field"
        return "run"

    def _open_local_model_fit_dialog(self, param_name: str) -> None:
        if self._result is None:
            return

        xs: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        for group in self._groups:
            pset = self._result.local_parameters.get(group.group_id)
            if pset is None or param_name not in pset:
                continue
            xs.append(float(group.group_variable_value))
            ys.append(float(pset[param_name].value))
            err = self._result.local_uncertainties.get(group.group_id, {}).get(param_name)
            es.append(float(err) if err is not None and np.isfinite(err) and err > 0 else np.nan)

        if len(xs) < 2:
            QMessageBox.information(self, "Model Fit", "Need at least two points for a model fit.")
            return

        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        e_arr = np.asarray(es, dtype=float)
        invalid_err = ~np.isfinite(e_arr) | (e_arr <= 0)
        if np.any(invalid_err):
            finite = np.abs(y_arr[np.isfinite(y_arr)])
            fallback = max(float(np.nanmedian(finite)) * 0.02, 1e-9) if finite.size else 1e-3
            e_arr = e_arr.copy()
            e_arr[invalid_err] = fallback

        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]
        e_arr = e_arr[order]

        dialog = ModelFitDialog(
            parameter_name=param_name,
            x_key=self._local_group_x_key(),
            x_values=x_arr,
            y_values=y_arr,
            y_errors=e_arr,
            existing_fit=self._local_model_fits.get(param_name),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.was_removed():
            self._local_model_fits.pop(param_name, None)
        else:
            fit = dialog.get_model_fit()
            if fit is not None:
                self._local_model_fits[param_name] = fit

        controls = self._local_y_controls.get(param_name)
        if isinstance(controls, dict):
            fit_button = controls.get("fit_button")
            if isinstance(fit_button, QPushButton):
                active_fit = self._local_model_fits.get(param_name)
                fit_button.setText(
                    "Model Fit*" if active_fit is not None and active_fit.active else "Model Fit"
                )

        self._refresh_local_parameter_plots()

    def batch_id(self) -> str | None:
        """Return the batch id of the displayed fit (decorations' owning series)."""
        return self._batch_id

    def set_results(
        self,
        *,
        parameter_name: str,
        x_key: str,
        groups: list[ParameterGroupData],
        model,
        result: CrossGroupFitResult,
        fit_x_min: float = float("nan"),
        fit_x_max: float = float("nan"),
        batch_id: str | None = None,
        x_label: str | None = None,
        group_variable_label: str | None = None,
    ) -> None:
        # Showing a *different* fit replaces the decoration context: the previous
        # fit's local model fits and annotations belong to its own series and
        # must not bleed onto the new one. A re-run of the *same* fit (matching
        # batch id) keeps the live decorations so they follow the replacement.
        if batch_id is not None and batch_id != self._batch_id:
            self._local_model_fits = {}
            self._plot_annotations = []
            self._local_plot_annotations = []
            self._suppressed_group_label_tags = set()
        self._batch_id = batch_id
        self._study_id = batch_id
        self._parameter_name = parameter_name
        self._x_key = x_key
        self._groups = groups
        self._model = model
        self._result = result
        self._fit_x_min = float(fit_x_min)
        self._fit_x_max = float(fit_x_max)
        self._x_label_override = x_label or None
        self._group_variable_label_override = group_variable_label or None
        # New fit ⇒ show every group by default; drop visibility flags for groups
        # that are no longer present.
        valid_ids = {g.group_id for g in groups}
        self._group_visibility = {gid: self._group_visibility.get(gid, True) for gid in valid_ids}
        # Curves depend only on the inputs just rebound above, so the previous
        # fit's cached curves are now invalid — clear them. Both components-flag
        # entries are dropped; the display recomputes the one it needs off-thread.
        self._curve_cache = {}
        self._refresh_table()
        self._refresh_quality_bar()
        self._refresh_local_parameter_plots()
        # The cross-group fit curves are the heavy part (per-group model eval
        # over an 800-point axis); compute them off-thread behind the overlay so
        # restoring a saved fit on project open does not block the GUI.
        self._start_fit_curve_compute()

    def set_study(self, study, *, stale: bool, stale_reason: str = "") -> None:
        """Display *study* (a ``GlobalFitStudy``) with a stale banner if needed.

        Thin adapter over :meth:`set_results`: the study is the persisted entity,
        this window is the view. The study id is the batch id (they share the
        digest scheme), so decorations keep keying by it. Custom x/group-variable
        labels are carried so the plots read the study's stored labels rather than
        this window's hardcoded field/T/run maps (Phase 3 consumes them fully).
        """
        self.set_results(
            parameter_name=study.parameter_name,
            x_key=study.x_key,
            groups=list(study.groups),
            model=study.model,
            result=study.result,
            fit_x_min=study.fit_x_min,
            fit_x_max=study.fit_x_max,
            batch_id=study.study_id,
            x_label=getattr(study, "x_label", None),
            group_variable_label=getattr(study, "group_variable_label", None),
        )
        self.setWindowTitle(f"Global Parameter Fit — {study.name}")
        self.set_stale(stale, stale_reason)

    def set_stale(self, stale: bool, reason: str = "") -> None:
        """Show or hide the stale banner (with *reason*) above the plot."""
        self._stale_label.setText(
            f"This fit is out of date: {reason}." if reason else "This fit is out of date."
        )
        # A study whose source series are gone cannot be refitted from live data.
        self._refit_btn.setEnabled(reason != "source series missing")
        self._stale_banner.setVisible(bool(stale))

    def _on_refit_clicked(self) -> None:
        if self._study_id:
            self.refit_requested.emit(self._study_id)

    # ── Studies sidebar ─────────────────────────────────────────────────────

    def set_studies_list(self, entries: list[StudySidebarEntry]) -> None:
        """Populate the sidebar from :class:`StudySidebarEntry` records.

        Driven by the MainWindow whenever the registry, staleness, or displayed
        study changes. Repopulating does not emit :attr:`study_selected` (the
        selection is set separately via :meth:`set_active_study_id`). The full
        entry list is retained so the "Compare with…" context submenu can offer
        the studies that share a parameter and x-axis with the right-clicked one.
        """
        self._sidebar_entries = {entry.study_id: entry for entry in entries}
        self._suppress_study_selection = True
        try:
            self._studies_list.clear()
            for entry in entries:
                label = f"⚠ {entry.name}" if entry.stale else entry.name
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, str(entry.study_id))
                if entry.stale:
                    item.setToolTip("This fit is out of date — refit to update.")
                self._studies_list.addItem(item)
        finally:
            self._suppress_study_selection = False

    def set_active_study_id(self, study_id: str | None) -> None:
        """Select the sidebar row for *study_id* without emitting a signal."""
        self._suppress_study_selection = True
        try:
            target = -1
            for row in range(self._studies_list.count()):
                item = self._studies_list.item(row)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) == study_id:
                    target = row
                    break
            self._studies_list.setCurrentRow(target)
        finally:
            self._suppress_study_selection = False

    def _study_id_at_row(self, row: int) -> str | None:
        item = self._studies_list.item(row)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if isinstance(value, str) and value else None

    def _on_studies_row_changed(self, row: int) -> None:
        if self._suppress_study_selection:
            return
        study_id = self._study_id_at_row(row)
        if study_id:
            self.study_selected.emit(study_id)

    def _on_studies_context_menu(self, pos) -> None:
        item = self._studies_list.itemAt(pos)
        if item is None:
            return
        study_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not study_id:
            return
        menu = QMenu(self._studies_list)
        rename_action = menu.addAction("Rename…")
        duplicate_action = menu.addAction("Duplicate")

        # "Compare with…" submenu: other studies sharing this one's parameter
        # and x-axis (an apples-to-apples model-selection comparison). Disabled
        # when there are no such candidates.
        compare_menu = menu.addMenu("Compare with…")
        compare_actions: dict[object, str] = {}
        this_entry = self._sidebar_entries.get(study_id)
        candidates = []
        if this_entry is not None:
            for other in self._sidebar_entries.values():
                if other.study_id == study_id:
                    continue
                if (
                    other.parameter_name == this_entry.parameter_name
                    and other.x_key == this_entry.x_key
                ):
                    candidates.append(other)
        if candidates:
            for other in candidates:
                action = compare_menu.addAction(other.name)
                compare_actions[action] = other.study_id
        else:
            compare_menu.setEnabled(False)

        menu.addSeparator()
        delete_action = menu.addAction("Delete…")
        chosen = menu.exec(self._studies_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen in compare_actions:
            self.study_compare_requested.emit(study_id, compare_actions[chosen])
            return
        if chosen == rename_action:
            current = item.text().lstrip("⚠ ").strip()
            new_name, ok = QInputDialog.getText(self, "Rename study", "Name:", text=current)
            if ok and new_name.strip():
                self.study_rename_requested.emit(study_id, new_name.strip())
        elif chosen == duplicate_action:
            self.study_duplicate_requested.emit(study_id)
        elif chosen == delete_action:
            confirm = QMessageBox.question(
                self,
                "Delete study",
                "Delete this global parameter fit? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.study_delete_requested.emit(study_id)

    def clear_display(self) -> None:
        """Empty the window (plots, tables, banner, title) — no study shown."""
        self._batch_id = None
        self._study_id = None
        self._result = None
        self._model = None
        self._groups = []
        self._parameter_name = None
        self._x_label_override = None
        self._group_variable_label_override = None
        self._group_visibility = {}
        self._curve_cache = {}
        self.setWindowTitle("Global Parameter Fit")
        self.set_stale(False, "")
        self._refresh_table()
        self._refresh_quality_bar()
        self._refresh_local_parameter_plots()
        if self._left_canvas is not None and self._left_figure is not None:
            self._left_figure.clear()
            self._axes_tag_map = {}
            self._left_canvas.draw()

    # ── Edit fit ────────────────────────────────────────────────────────────

    def _on_edit_fit_clicked(self) -> None:
        if self._study_id:
            self.edit_requested.emit(self._study_id)

    def _start_fit_curve_compute(self) -> None:
        """Compute the current-flag cross-group fit curves off-thread, into the cache.

        Evaluates the per-group 800-pt curves (plus components when
        Show-components is on, plus the data-x residual pulls) for the *current*
        components flag, stores them under ``self._curve_cache[flag]``, and
        redraws from the cache. Coalesces concurrent requests (the set_results +
        restore burst) into a single rerun.
        """
        if self._left_canvas is None or self._left_figure is None:
            return
        if self._result is None or self._model is None:
            # Nothing to evaluate; _refresh_plot just clears the canvas (cheap).
            self._refresh_plot()
            return
        if self._fit_curve_compute_active:
            # A compute is already running; fold this request into a single
            # rerun rather than spawning another worker.
            self._fit_curve_recompute_pending = True
            return
        groups = list(self._groups)
        show_components = self._show_components_check.isChecked()
        #: The components flag the in-flight worker is computing for, so
        #: :meth:`_on_fit_curves_ready` files the result under the right cache key.
        self._fit_curve_compute_flag = show_components
        # Snapshot the model/result/x-range references so the worker never reads
        # state that a concurrent set_results may rebind on the GUI thread.
        result = self._result
        model = self._model
        fit_x_min = self._fit_x_min
        fit_x_max = self._fit_x_max
        x_key = self._x_key
        if self._fit_overlay is not None:
            self._fit_overlay.show_message("Computing fit curves…")
        self._fit_curve_compute_active = True
        self._tasks.start(
            lambda _worker: {
                group.group_id: self._compute_group_fit_curve(
                    group,
                    show_components,
                    result=result,
                    model=model,
                    fit_x_min=fit_x_min,
                    fit_x_max=fit_x_max,
                    x_key=x_key,
                )
                for group in groups
            },
            on_finished=self._on_fit_curves_ready,
            on_error=self._on_fit_curves_error,
        )

    def _on_fit_curves_ready(self, curves: object) -> None:
        self._fit_curve_compute_active = False
        if self._fit_curve_recompute_pending:
            # Controls changed mid-compute (restore burst); recompute with the
            # latest state and skip drawing this now-stale result.
            self._fit_curve_recompute_pending = False
            self._start_fit_curve_compute()
            return
        if self._fit_overlay is not None:
            self._fit_overlay.hide()
        # File the computed curves under the flag they were computed for, so
        # later redraws (control toggles) draw from the cache without re-eval.
        flag = getattr(self, "_fit_curve_compute_flag", self._show_components_check.isChecked())
        self._curve_cache[bool(flag)] = curves if isinstance(curves, dict) else {}
        self._refresh_plot()

    def _on_fit_curves_error(self, message: str) -> None:
        self._fit_curve_compute_active = False
        if self._fit_curve_recompute_pending:
            self._fit_curve_recompute_pending = False
            self._start_fit_curve_compute()
            return
        if self._fit_overlay is not None:
            self._fit_overlay.hide()
        # Cache an empty result for this flag so the plot draws the data points
        # without curves rather than re-evaluating inline (which would re-raise
        # the same error on the GUI thread) on every later redraw.
        flag = getattr(self, "_fit_curve_compute_flag", self._show_components_check.isChecked())
        self._curve_cache[bool(flag)] = {}
        self._refresh_plot()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Cancel and join any in-flight fit-curve recompute before closing."""
        self._tasks.shutdown()
        super().closeEvent(event)

    def _x_label(self) -> str:
        if self._x_label_override:
            return self._x_label_override
        return {
            "field": "$B$ (G)",
            "temperature": "$T$ (K)",
            "run": "Run Number",
        }.get(self._x_key, "x")

    def _x_label_gle(self) -> str:
        # Known keys keep the {\it B}/{\it T} classics; a custom study label is
        # rendered as plain text (GLE-safe) since it carries no math markup.
        known = {
            "field": "{\\it B} (G)",
            "temperature": "{\\it T} (K)",
            "run": "Run Number",
        }
        if self._x_key in known:
            return known[self._x_key]
        if self._x_label_override:
            return self._x_label_override
        return "x"

    def _local_group_axis_label(self) -> str:
        if self._group_variable_label_override:
            return self._group_variable_label_override
        return {
            "field": "$T$ (K)",
            "temperature": "$B$ (G)",
            "run": "Group variable",
        }.get(self._x_key, "Group variable")

    def _local_group_axis_label_gle(self) -> str:
        known = {
            "field": "{\\it T} (K)",
            "temperature": "{\\it B} (G)",
            "run": "Group variable",
        }
        if self._x_key in known:
            return known[self._x_key]
        if self._group_variable_label_override:
            return self._group_variable_label_override
        return "Group variable"

    def _local_group_axis_label_plain(self) -> str:
        known = {
            "field": "T (K)",
            "temperature": "B (G)",
            "run": "Group variable",
        }
        if self._x_key in known:
            return known[self._x_key]
        if self._group_variable_label_override:
            return self._group_variable_label_override
        return "Group variable"

    def _parameter_label(self, name: str | None) -> str:
        if not name:
            return "y"
        return get_param_info(name).latex_label()

    def _parameter_label_gle(self, name: str | None) -> str:
        if not name:
            return "y"
        return get_param_info(name).gle_label()

    def _safe_file_token(self, value: str) -> str:
        token = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip()
        )
        token = "_".join(part for part in token.split("_") if part)
        return token or "group"

    def _safe_data_name(self, value: str) -> str:
        token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).strip())
        token = "_".join(part for part in token.split("_") if part)
        return (token or "series").lower()

    @staticmethod
    def _model_kwargs(result, model, group_id: str) -> dict[str, float]:
        """Build a group's model kwargs (global + fixed + local, defaults-filled).

        Pure — takes ``result``/``model`` explicitly so it is safe to call from
        the off-thread fit-curve worker as well as the GUI thread.
        """
        kwargs = {p.name: p.value for p in result.global_parameters}
        for p in result.fixed_parameters:
            kwargs[p.name] = p.value
        local = result.local_parameters.get(group_id)
        if local is not None:
            for p in local:
                kwargs[p.name] = p.value
        # Backward-compatible restore support: older saved cross-group states may
        # miss newer model parameters (e.g. D_perp). Fill from model defaults.
        missing = [name for name in getattr(model, "param_names", []) if name not in kwargs]
        defaults = getattr(model, "param_defaults", {})
        for name in missing:
            if isinstance(defaults, dict) and name in defaults:
                kwargs[name] = float(defaults[name])
        return kwargs

    def _model_kwargs_for_group(self, group_id: str) -> dict[str, float]:
        if self._result is None:
            return {}
        return self._model_kwargs(self._result, self._model, group_id)

    def _cached_group_total_curve(self, group_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a group's cached total ``(xx, yy)`` curve, or ``None`` if cold.

        The total fit curve is identical across both components flags, so either
        cache entry serves the GLE export. Returns ``None`` when neither flag has
        been computed yet (cold cache) or the entry carries an error.
        """
        for flag in (False, True):
            entry = self._curve_cache.get(flag)
            if not entry:
                continue
            curve = entry.get(group_id)
            if isinstance(curve, dict) and "error" not in curve and curve.get("xx") is not None:
                return curve["xx"], curve["yy"]
        return None

    def _sample_group_fit_curve(
        self, group: ParameterGroupData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        # Reuse the warm curve cache when present (post-display it always is);
        # only fall back to an inline evaluation when the cache is cold.
        cached = self._cached_group_total_curve(group.group_id)
        if cached is not None:
            return cached
        if self._model is None:
            return None
        kwargs = self._model_kwargs_for_group(group.group_id)
        if not kwargs:
            return None

        xx = np.asarray(group.x, dtype=float)
        finite_x = xx[np.isfinite(xx)]
        if finite_x.size < 2:
            return None

        xx = np.sort(finite_x)
        if (
            np.isfinite(self._fit_x_min)
            and np.isfinite(self._fit_x_max)
            and self._fit_x_max > self._fit_x_min
        ):
            mask = (xx >= self._fit_x_min) & (xx <= self._fit_x_max)
            xx = xx[mask]
        if xx.size < 2:
            return None

        x_min = float(np.nanmin(xx))
        x_max = float(np.nanmax(xx))
        if self._local_group_x_key() == "field" and x_min > 0.0 and x_max > 0.0:
            xs = np.geomspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
        else:
            xs = np.linspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
        try:
            ys = np.asarray(self._model.function(xs, **kwargs), dtype=float)
        except KeyError:
            return None
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return None
        return xs[mask], ys[mask]

    @staticmethod
    def _global_parameter_table_lines(result) -> list[str]:
        """Return the ``! Global parameter table`` provenance lines for a result.

        Shared by the fit-subplot and local-parameter exports so the global /
        fixed parameter formatting (units, precision, ``nan`` sentinel, ``[fixed]``
        flag) stays identical across both. Lines have no trailing newline.
        """
        lines: list[str] = []
        if not (len(result.global_parameters) or len(result.fixed_parameters)):
            return lines
        lines.append("! Global parameter table:")
        lines.append("!   Parameter                     Value                Uncertainty")
        for p in result.global_parameters:
            info = get_param_info(p.name)
            unit = f" ({info.unit})" if info.unit else ""
            err = result.global_uncertainties.get(p.name)
            err_text = "nan"
            if err is not None and np.isfinite(err):
                err_text = f"{float(err):.10g}"
            lines.append(f"!   {p.name}{unit:<24} {float(p.value):>16.10g} {err_text:>24}")
        for p in result.fixed_parameters:
            info = get_param_info(p.name)
            unit = f" ({info.unit})" if info.unit else ""
            lines.append(f"!   {p.name}{unit} [fixed] = {float(p.value):.10g}")
        return lines

    def _fit_metadata_header_lines(self) -> list[str]:
        """Return shared provenance lines for GLE fit-subplot exports.

        Mirrors the comprehensive local-parameter export header so the fit-data
        and model-curve files record *which* model was fitted, the values (and
        uncertainties) of the shared global parameters, and the fit quality —
        not just bare ``x y yerr`` columns.
        """
        lines: list[str] = []
        if self._model is not None:
            try:
                formula = self._model.formula_string()
            except Exception:
                formula = ""
            if formula:
                lines.append(f"! model: {formula}")

        result = self._result
        if result is not None:
            lines.extend(self._global_parameter_table_lines(result))
            if np.isfinite(result.chi_squared):
                lines.append(f"! chi_squared: {float(result.chi_squared):.8g}")
            if np.isfinite(result.reduced_chi_squared):
                lines.append(f"! reduced_chi_squared: {float(result.reduced_chi_squared):.8g}")
            lines.append(f"! error_mode: {result.error_mode}")
            if result.n_points:
                lines.append(f"! n_points: {int(result.n_points)}")
            if result.per_group_chi_squared:
                lines.append("! Per-group chi-squared:")
                for gid in sorted(result.per_group_chi_squared):
                    chi = result.per_group_chi_squared[gid]
                    npts = result.per_group_n_points.get(gid)
                    npts_txt = f" (n={int(npts)})" if npts is not None else ""
                    lines.append(f"!   {gid}: {float(chi):.8g}{npts_txt}")
            if result.global_correlations is not None:
                names, matrix = result.global_correlations
                lines.append("! Global parameter correlations:")
                lines.append("!   " + " ".join(str(n) for n in names))
                for i, name in enumerate(names):
                    row = matrix[i] if i < len(matrix) else []
                    row_txt = " ".join(f"{float(v):+.3f}" for v in row)
                    lines.append(f"!   {name}: {row_txt}")
        return lines

    def _write_fit_subplot_files(self, gle_path: Path) -> dict[str, dict[str, object]]:
        """Write explicit per-group data and fit files for GLE export."""
        written: dict[str, dict[str, object]] = {}
        header_lines = self._fit_metadata_header_lines()
        x_axis_label = self._local_group_axis_label_plain()
        for group in self._groups:
            token = self._safe_file_token(group.group_name or group.group_id)
            data_path = gle_path.with_name(f"{gle_path.stem}_{token}_data.dat")
            fit_path = gle_path.with_name(f"{gle_path.stem}_{token}_fit.fit")
            group_label = str(group.group_name or group.group_id)

            x = np.asarray(group.x, dtype=float)
            y = np.asarray(group.y, dtype=float)
            yerr = np.asarray(group.yerr, dtype=float)
            if yerr.shape != y.shape:
                yerr = np.full_like(y, np.nan, dtype=float)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask] if yerr.size else np.full_like(y, np.nan, dtype=float)
            if x.size == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            yerr = yerr[order]

            with open(data_path, "w", encoding="utf-8") as f:
                f.write("! Cross-group fit data\n")
                f.write(f"! group: {group_label}\n")
                for line in header_lines:
                    f.write(line + "\n")
                f.write(f"! x-axis: {x_axis_label}\n")
                f.write("! columns: x y yerr\n")
                for xv, yv, ev in zip(x, y, yerr, strict=True):
                    ev_out = ev if np.isfinite(ev) and ev > 0 else np.nan
                    f.write(f"{xv:.10g} {yv:.10g} {ev_out:.10g}\n")

            sampled = self._sample_group_fit_curve(group)
            fit_written = False
            if sampled is not None:
                xs, ys = sampled
                with open(fit_path, "w", encoding="utf-8") as f:
                    f.write("! Cross-group fit model curve\n")
                    f.write(f"! group: {group_label}\n")
                    for line in header_lines:
                        f.write(line + "\n")
                    f.write(f"! x-axis: {x_axis_label}\n")
                    f.write("! columns: x y\n")
                    for xv, yv in zip(xs, ys, strict=True):
                        f.write(f"{xv:.10g} {yv:.10g}\n")
                fit_written = True

            written[group.group_id] = {
                "data_path": data_path,
                "fit_path": fit_path,
                "has_fit": fit_written,
                "has_err": bool(np.any(np.isfinite(yerr) & (yerr > 0))),
            }

        return written

    def _apply_fit_axis_scales(self, ax) -> None:
        if self._fit_log_x_check.isChecked():
            try:
                ax.set_xscale("log")
            except Exception:
                pass
        if self._fit_log_y_check.isChecked() and not self._show_components_check.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass

    def _refresh_table(self) -> None:
        self._params_table.setRowCount(0)
        has_corr = bool(self._result is not None and self._result.global_correlations)
        if hasattr(self, "_correlations_btn"):
            self._correlations_btn.setEnabled(has_corr)
            self._correlations_btn.setToolTip(
                "Show the free-global-parameter correlation matrix."
                if has_corr
                else "correlation matrix unavailable — fit covariance is missing or singular"
            )
        if self._result is None:
            return

        for p in self._result.global_parameters:
            info = get_param_info(p.name)
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(
                row, 0, QTableWidgetItem(info.unicode_label(include_unit=False))
            )
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            err = self._result.global_uncertainties.get(p.name)
            err_item = QTableWidgetItem("" if err is None else f"{err:.3g}")
            self._apply_uncertainty_trust_flag(err_item, float(p.value), err)
            self._params_table.setItem(row, 2, err_item)
            self._params_table.setItem(row, 3, QTableWidgetItem(info.unit or ""))

        for p in self._result.fixed_parameters:
            info = get_param_info(p.name)
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(
                row, 0, QTableWidgetItem(f"{info.unicode_label(include_unit=False)} (fixed)")
            )
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            self._params_table.setItem(row, 2, QTableWidgetItem(""))
            self._params_table.setItem(row, 3, QTableWidgetItem(info.unit or ""))

        # Reapply the section resize policies (row inserts can reset them) and
        # snap the interactive numeric columns to their content width.
        self._configure_params_table_header()
        self._params_table.resizeColumnToContents(1)
        self._params_table.resizeColumnToContents(2)

    def _apply_uncertainty_trust_flag(
        self, item: QTableWidgetItem, value: float, err: float | None
    ) -> None:
        """Colour + tooltip an Uncertainty cell when the error bar is untrustworthy.

        Flags (warning background + explanatory tooltip) a parameter whose
        reported uncertainty exceeds ``max(3×|value|, tiny-floor)`` or is
        non-finite — the fit did not meaningfully constrain it. Healthy cells are
        left with the default background and no tooltip.
        """
        if _uncertainty_is_untrustworthy(value, err):
            item.setBackground(QColor(tokens.WARN_BANNER_BG))
            item.setToolTip(
                "uncertainty exceeds |value| — parameter likely degenerate/unconstrained"
            )

    def _refresh_quality_bar(self) -> None:
        """Update the compact χ²/χ²ᵣ/n/error-mode row above the global table.

        For NONE/SCATTER error modes χ²ᵣ carries no goodness verdict (the σ used
        is not a measured column error), so the value is shown greyed with a
        tooltip rather than as a quality figure. A failed fit surfaces its
        message in red.
        """
        result = self._result
        if result is None:
            self._quality_label.setText("")
            self._quality_label.setToolTip("")
            return

        parts: list[str] = []
        if np.isfinite(result.chi_squared):
            parts.append(f"χ²={result.chi_squared:.4g}")

        no_verdict = _label_indicates_no_verdict(result.error_mode)
        if np.isfinite(result.reduced_chi_squared):
            chi_r = f"χ²ᵣ={result.reduced_chi_squared:.4g}"
            if no_verdict:
                parts.append(f"<span style='color:{tokens.TEXT_DIM}'>{chi_r}</span>")
            else:
                parts.append(chi_r)
        if result.n_points:
            parts.append(f"n={int(result.n_points)}")
        # Error-mode chip.
        mode_text = {
            "column": "column σ",
            "percent": "percent σ",
            "absolute": "absolute σ",
            "none": "no errors",
            "scatter": "scatter σ",
        }.get(str(result.error_mode).lower(), str(result.error_mode))
        parts.append(
            f"<span style='background:{tokens.SURFACE_HI};padding:1px 5px;"
            f"border-radius:3px'>{mode_text}</span>"
        )

        text = " &nbsp; ".join(parts)
        if not result.success and result.message:
            text += (
                f" &nbsp; <span style='color:{tokens.ERROR};font-weight:bold'>"
                f"{result.message}</span>"
            )
        self._quality_label.setText(text)
        if no_verdict:
            self._quality_label.setToolTip(
                "χ²ᵣ carries no goodness-of-fit verdict under this error mode: the "
                "per-point σ is not a measured uncertainty."
            )
        else:
            self._quality_label.setToolTip("")

    # ── Global table: copy / export / correlations ──────────────────────────

    def _on_copy_global_table(self) -> None:
        if self._result is None:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(global_table_tsv(self._result))

    def _on_export_global_table(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "No result", "Run a cross-group fit first.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export global parameter table",
            default_export_path("global_parameter_table.csv"),
            "CSV (*.csv);;LaTeX (*.tex)",
        )
        if not path:
            return
        remember_export_path(path)
        is_latex = path.lower().endswith(".tex") or "LaTeX" in selected_filter
        try:
            content = (
                global_table_latex(self._result, parameter_name=self._parameter_name)
                if is_latex
                else global_table_csv(self._result)
            )
            Path(path).write_text(content, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", f"Could not write file: {exc}")

    def _on_show_correlations(self) -> None:
        if self._result is None or not self._result.global_correlations:
            return
        names, matrix = self._result.global_correlations
        dialog = CorrelationMatrixDialog(list(names), [list(row) for row in matrix], parent=self)
        dialog.exec()

    # ── Fig-3 grid: columns / show-hide ─────────────────────────────────────

    def _shown_groups(self) -> list[ParameterGroupData]:
        """Return the groups whose panel is shown (visibility flag True)."""
        return [g for g in self._groups if self._group_visibility.get(g.group_id, True)]

    def _grid_columns(self, n: int) -> int:
        """Return the column count for *n* shown panels honouring the combo.

        Auto = ``ceil(sqrt(n))`` capped at 4 (so 8 groups → 3 columns); an
        explicit "1"/"2"/"3"/"4" pins that many columns (never more than *n*).
        """
        if n <= 0:
            return 1
        text = self._fit_columns_combo.currentText()
        if text == "Auto":
            cols = int(np.ceil(np.sqrt(n)))
            return max(1, min(cols, 4))
        try:
            cols = int(text)
        except (TypeError, ValueError):
            cols = 1
        return max(1, min(cols, n))

    def _show_groups_popup(self) -> None:
        """Pop up a checkable list to show/hide individual group panels."""
        if not self._groups:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Show groups")
        layout = QVBoxLayout(dialog)
        list_widget = QListWidget()
        for group in self._groups:
            item = QListWidgetItem(group.group_name or group.group_id)
            item.setData(Qt.ItemDataRole.UserRole, group.group_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if self._group_visibility.get(group.group_id, True)
                else Qt.CheckState.Unchecked
            )
            list_widget.addItem(item)
        layout.addWidget(list_widget)
        close_btn = QPushButton("Done")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            gid = str(item.data(Qt.ItemDataRole.UserRole))
            self._group_visibility[gid] = item.checkState() == Qt.CheckState.Checked
        self._refresh_plot()

    def _per_group_reduced_chi_squared(self, group_id: str) -> float | None:
        """Return a per-group reduced χ² for the panel chip, or ``None``.

        dof convention: per-group ``n_points`` minus the number of *local*
        parameters for that group. The shared global parameters' dof cost is
        borne jointly across all groups, so it is deliberately NOT subtracted
        here — this makes the chip a per-panel data-vs-model figure, not a
        rigorous joint χ²ᵣ (documented so it is not mistaken for one).
        """
        result = self._result
        if result is None:
            return None
        chi = result.per_group_chi_squared.get(group_id)
        n = result.per_group_n_points.get(group_id)
        if chi is None or n is None or not np.isfinite(chi):
            return None
        local = result.local_parameters.get(group_id)
        local_count = len(local) if local is not None else 0
        dof = max(int(n) - local_count, 1)
        return float(chi) / dof

    def _group_chip_visible(self) -> bool:
        """Chips are shown only when per-group data exists and σ is meaningful."""
        result = self._result
        if result is None:
            return False
        if not result.per_group_chi_squared or not result.per_group_n_points:
            return False
        return not _label_indicates_no_verdict(result.error_mode)

    def _compute_group_fit_curve(
        self,
        group: ParameterGroupData,
        show_components: bool,
        *,
        result,
        model,
        fit_x_min: float,
        fit_x_max: float,
        x_key: str,
    ) -> dict | None:
        """Evaluate one group's cross-group fit curve (and optional components).

        Pure: operates only on the explicitly-passed ``result``/``model``/x-range
        and the ``show_components`` flag — never ``self`` GUI state, no widget or
        matplotlib access — so the fit-curve worker can call it off the GUI
        thread without racing ``set_results`` (which rebinds ``self._result`` /
        ``self._model`` on the GUI thread). Returns
        ``{"xx", "yy", "components", "pulls"}`` — where ``pulls`` is the
        ``(x, pull)`` residual arrays evaluated at the *data* x (always computed,
        so the Residuals toggle is a cache-backed redraw, never an inline eval) —
        ``{"error": msg}`` when the model rejects the parameters, or ``None``
        when there is nothing to draw.
        """
        if model is None or result is None:
            return None
        kwargs = self._model_kwargs(result, model, group.group_id)
        if not kwargs:
            return None

        xx = np.asarray(group.x, dtype=float)
        if xx.size >= 2:
            xx = xx.copy()
            xx.sort()
        if np.isfinite(fit_x_min) and np.isfinite(fit_x_max) and fit_x_max > fit_x_min:
            mask = (xx >= fit_x_min) & (xx <= fit_x_max)
            xx = xx[mask]
        if xx.size >= 2:
            x_min = float(np.nanmin(xx))
            x_max = float(np.nanmax(xx))
            if x_key == "field" and x_min > 0.0 and x_max > 0.0:
                xx = np.geomspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
            else:
                xx = np.linspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)

        try:
            components = (
                model.evaluate_components(xx, additive_only=True, **kwargs)
                if show_components
                else None
            )
            yy = np.asarray(model.function(xx, **kwargs), dtype=float)
            pulls = self._compute_group_pulls(group, model=model, kwargs=kwargs)
        except KeyError as exc:
            return {"error": str(exc)}
        return {"xx": xx, "yy": yy, "components": components, "pulls": pulls}

    @staticmethod
    def _compute_group_pulls(
        group: ParameterGroupData, *, model, kwargs: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(xs, pulls)`` for ``(data − model)/σ`` at the *data* x, or None.

        Pure — evaluates the model at the group's data x (N points, cheap next to
        the 800-sample curve) so the residual pulls ride along in the same
        off-thread worker payload and the Residuals toggle never re-evaluates the
        model on the GUI thread. May raise ``KeyError`` (caught by the caller,
        which turns the whole group into an ``{"error": …}`` entry).
        """
        x = np.asarray(group.x, dtype=float)
        y = np.asarray(group.y, dtype=float)
        e = np.asarray(group.yerr, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return None
        xm = x[mask]
        ym = y[mask]
        em = e[mask] if e.shape == y.shape else np.full_like(ym, np.nan)
        model_y = np.asarray(model.function(xm, **kwargs), dtype=float)
        sigma = np.where(np.isfinite(em) & (em > 0), em, np.nan)
        pulls = (ym - model_y) / sigma
        order = np.argsort(xm)
        return xm[order], pulls[order]

    def _current_curve_cache(self) -> dict[str, dict] | None:
        """Return the cached per-group curves for the current components flag.

        ``None`` signals a cache miss — the caller (:meth:`_refresh_plot`) then
        kicks an off-thread compute rather than evaluating inline. The pulls used
        by residual mode ride on the same cached entries, so residual redraws are
        cache-backed too.
        """
        return self._curve_cache.get(self._show_components_check.isChecked())

    def _refresh_plot(self) -> None:
        if self._left_canvas is None or self._left_figure is None:
            return
        if self._result is not None and self._model is not None:
            # The heavy per-group curves (and residual pulls) are never evaluated
            # inline here: a cache hit draws from the cache; a miss kicks the
            # off-thread compute (overlay + coalescing) and returns — its ready
            # callback re-enters _refresh_plot once the cache is warm.
            if self._current_curve_cache() is None:
                self._start_fit_curve_compute()
                return
        self._left_figure.clear()
        self._axes_tag_map = {}
        if self._result is None or self._model is None:
            self._left_canvas.draw()
            return

        curves = self._current_curve_cache() or {}
        residuals = self._fit_residuals_check.isChecked()
        shown_groups = self._shown_groups()
        # Group-label annotations are keyed by group_id; prune stale ones against
        # the full group set (a hidden group's annotation is kept for when it is
        # shown again).
        valid_group_tags = {group.group_id for group in self._groups}
        self._prune_stale_group_label_annotations(valid_group_tags)

        if not shown_groups:
            ax = self._left_figure.add_subplot(111)
            ax.set_title("No group panels shown")
            ax.grid(True, alpha=0.3)
            self._left_canvas.draw()
            return

        x_label = self._x_label()
        y_label = self._parameter_label(self._parameter_name)
        share_x_axis = self._fit_share_x_check.isChecked()
        share_y_axis = self._fit_share_y_check.isChecked()

        n = len(shown_groups)
        ncols = self._grid_columns(n)
        nrows = (n + ncols - 1) // ncols
        # Size the scrolled canvas so each panel row keeps a legible height; the
        # QScrollArea grows a vertical scrollbar rather than squeezing panels.
        self._update_grid_canvas_min_height(nrows)

        # Track component names+colors seen so a single figure-level legend can
        # be assembled after the loop.
        legend_entries: dict[str, str] = {}
        shared_x_ax = None
        shared_y_ax = None
        for idx, group in enumerate(shown_groups):
            share_kwargs = {}
            if share_x_axis and shared_x_ax is not None:
                share_kwargs["sharex"] = shared_x_ax
            if share_y_axis and shared_y_ax is not None:
                share_kwargs["sharey"] = shared_y_ax
            ax = self._left_figure.add_subplot(nrows, ncols, idx + 1, **share_kwargs)
            if share_x_axis and shared_x_ax is None:
                shared_x_ax = ax
            if share_y_axis and shared_y_ax is None:
                shared_y_ax = ax
            self._axes_tag_map[id(ax)] = group.group_id

            curve = curves.get(group.group_id)
            if residuals:
                self._draw_group_residuals(ax, group, curve)
            else:
                self._draw_group_fit_panel(ax, group, legend_entries, curve)

            row = idx // ncols
            col = idx % ncols
            if share_x_axis:
                self._ensure_group_label_annotation(group.group_id, group.group_name, ax)
                ax.set_title("")
            else:
                ax.set_title(group.group_name, pad=10)
            # Only the left column carries the y-label to reduce clutter in wide
            # grids; single-column grids always label.
            if col == 0 or ncols == 1:
                ax.set_ylabel("(data − model)/σ" if residuals else y_label)
            ax.grid(True, alpha=0.3)
            if not residuals:
                self._apply_fit_axis_scales(ax)
            elif self._fit_log_x_check.isChecked():
                try:
                    ax.set_xscale("log")
                except Exception:
                    pass
            # X-label on the bottom row of each column (or the last panel when
            # sharing x reduces to one axis).
            is_bottom = row == nrows - 1 or (idx + ncols) >= n
            if (not share_x_axis) or is_bottom:
                ax.set_xlabel(x_label)

            if (not residuals) and self._group_chip_visible():
                self._draw_group_chi_chip(ax, group.group_id)

        # One figure-level component legend (names in stacking order, stable
        # colours), only in component mode with entries collected. The legend is
        # anchored in the top figure margin, and that margin is *reserved* below
        # so the first-row panel titles never overlap it (see below).
        legend = None
        if legend_entries and not residuals:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=color, alpha=0.5, label=name)
                for name, color in legend_entries.items()
            ]
            legend = self._left_figure.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=min(len(handles), 4),
                fontsize=8,
                frameon=False,
            )

        self._draw_plot_annotations(local=False)

        # Lay the grid out, reserving a top band for the legend so first-row
        # titles never collide with it. We measure the legend's rendered height
        # in figure coordinates and pass ``tight_layout`` a ``rect`` whose top
        # edge sits just below the legend; ``tight_layout`` then packs every
        # panel (titles included) into the remaining area.
        top_rect = 1.0
        if legend is not None:
            try:
                renderer = self._left_canvas.get_renderer()
                legend_bbox = legend.get_window_extent(renderer)
                fig_h = float(self._left_figure.bbox.height) or 1.0
                legend_frac = float(legend_bbox.height) / fig_h
                # Reserve the legend band plus a small gap; clamp so the axes
                # never collapse on a very tall legend.
                top_rect = float(np.clip(1.0 - (legend_frac + 0.02), 0.5, 0.98))
            except Exception:
                top_rect = 0.94
        import warnings

        # tight_layout emits a benign UserWarning when the (scrolled) canvas is
        # temporarily smaller than the packed decorations need — the QScrollArea
        # gives the canvas its full minimum height once shown, so suppress the
        # transient here rather than let it clutter logs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tight layout not applied")
            try:
                self._left_figure.tight_layout(h_pad=1.8, rect=(0.0, 0.0, 1.0, top_rect))
            except Exception:
                self._left_figure.tight_layout(h_pad=1.8)
            self._left_figure.subplots_adjust(left=0.12, hspace=0.5)
        self._left_canvas.draw()

    def _draw_group_fit_panel(
        self, ax, group: ParameterGroupData, legend_entries: dict[str, str], curve: dict | None
    ) -> None:
        """Draw one group's data + stacked components + total curve on *ax*.

        The heavy per-group curve (800-pt model eval) is supplied pre-computed
        from the window curve cache (:meth:`_current_curve_cache`); this method
        never evaluates the model itself.
        """
        ax.errorbar(
            group.x, group.y, yerr=group.yerr, fmt="o", linestyle="none", color="black", capsize=2
        )

        if curve is not None and "error" in curve:
            ax.text(
                0.02,
                0.95,
                f"Fit curve unavailable: {curve['error']}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                color="tab:red",
            )
            return
        if curve is None:
            return

        xx = curve["xx"]
        components = curve.get("components")
        if components is not None:
            ordered = self._ordered_components_for_stacking(components)
            cumulative = np.zeros_like(xx, dtype=float)
            for cidx, (name, comp_y) in enumerate(ordered):
                fill_color = _COMPONENT_COLORS[cidx % len(_COMPONENT_COLORS)]
                legend_entries.setdefault(str(name), fill_color)
                comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                lower = cumulative
                upper = cumulative + comp_fill
                ax.fill_between(xx, lower, upper, color=fill_color, alpha=0.3, zorder=1)
                ax.plot(
                    xx,
                    upper,
                    linestyle="--",
                    linewidth=0.8,
                    color=fill_color,
                    alpha=0.9,
                    zorder=2,
                )
                cumulative = upper
        ax.plot(xx, curve["yy"], color="red", linewidth=1.5)

    def _draw_group_residuals(self, ax, group: ParameterGroupData, curve: dict | None) -> None:
        """Draw ``(data − model)/σ`` pulls for one group around a zero line.

        The pulls are read from the cached ``curve`` (computed off-thread in the
        same worker payload as the fit curves); this method never evaluates the
        model. Components and the fit curve are omitted in residual mode.
        """
        ax.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=1.0, zorder=1)
        if not isinstance(curve, dict):
            return
        pulls = curve.get("pulls")
        if pulls is None:
            return
        xs, ps = pulls
        xs = np.asarray(xs, dtype=float)
        ps = np.asarray(ps, dtype=float)
        finite = np.isfinite(ps)
        if np.any(finite):
            ax.stem(
                xs[finite],
                ps[finite],
                linefmt="C0-",
                markerfmt="C0o",
                basefmt=" ",
            )

    def _draw_group_chi_chip(self, ax, group_id: str) -> None:
        """Annotate a panel with a subtle ``χ²ᵣ=<val>`` chip in its corner."""
        chi_r = self._per_group_reduced_chi_squared(group_id)
        if chi_r is None or not np.isfinite(chi_r):
            return
        ax.text(
            0.97,
            0.95,
            f"χ²ᵣ={chi_r:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            color=tokens.TEXT_MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 1.5},
            zorder=8,
        )

    def _refresh_local_parameter_plots(self) -> None:
        if self._local_canvas is None or self._local_figure is None:
            return

        self._local_figure.clear()
        self._local_axes_tag_map = {}
        if self._result is None:
            self._local_canvas.draw()
            return

        local_param_names = sorted(
            {p.name for pset in self._result.local_parameters.values() for p in pset}
        )

        previous_selected = list(self._local_selected_y_names)
        preferred_selected = previous_selected or self._restored_local_selected_y
        self._rebuild_local_y_controls(local_param_names, preferred_selected=preferred_selected)
        self._restored_local_selected_y = []
        selected_names = self._selected_local_y_parameters()
        self._local_selected_y_names = list(selected_names)
        if selected_names:
            local_param_names = [p for p in local_param_names if p in selected_names]

        if not local_param_names:
            ax = self._local_figure.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            ax.grid(True, alpha=0.3)
            self._local_canvas.draw()
            return

        x_label = self._local_group_axis_label()

        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for pname in local_param_names:
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []

            for group in self._groups:
                pset = self._result.local_parameters.get(group.group_id)
                if pset is None:
                    continue
                param = pset[pname] if pname in pset else None
                if param is None:
                    continue
                xs.append(float(group.group_variable_value))
                ys.append(float(param.value))
                err = self._result.local_uncertainties.get(group.group_id, {}).get(pname)
                es.append(
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )

            if not xs:
                continue
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            e_arr = np.asarray(es, dtype=float)
            order = np.argsort(x_arr)
            traces.append((pname, x_arr[order], y_arr[order], e_arr[order]))

        if not traces:
            ax = self._local_figure.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            ax.grid(True, alpha=0.3)
            self._local_canvas.draw()
            return

        plot_mode = self._local_plot_mode_combo.currentText()
        if plot_mode == "Subplots" and len(traces) > 1:
            num_cols = 1
            num_rows = (len(traces) + num_cols - 1) // num_cols
            shared_x_ax = None
            for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                if shared_x_ax is not None:
                    ax = self._local_figure.add_subplot(
                        num_rows, num_cols, idx + 1, sharex=shared_x_ax
                    )
                else:
                    ax = self._local_figure.add_subplot(num_rows, num_cols, idx + 1)
                    shared_x_ax = ax
                self._local_axes_tag_map[id(ax)] = pname
                ax.errorbar(
                    x_arr,
                    y_arr,
                    yerr=e_arr,
                    fmt="o",
                    linestyle="none",
                    color="C0",
                    capsize=2,
                    zorder=6,
                )
                finite_err = np.isfinite(e_arr) & (e_arr > 0)
                if np.any(finite_err):
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr,
                        fmt="none",
                        ecolor="gray",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )
                ax.set_ylabel(self._parameter_label(pname))
                fit = self._local_model_fits.get(pname)
                if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                    for curve in evaluate_parameter_model_fit(fit, num_points=200):
                        ax.plot(curve.x, curve.y, color="red", linewidth=1.5, zorder=7)
                row = idx // num_cols
                if row == num_rows - 1:
                    ax.set_xlabel(x_label)
                else:
                    ax.tick_params(labelbottom=False)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                param_log = self._is_local_param_log_enabled(pname)
                if param_log:
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
        else:
            ax = self._local_figure.add_subplot(111)
            self._local_axes_tag_map[id(ax)] = "main"
            if len(traces) == 2:
                pname_l, x_l, y_l, e_l = traces[0]
                pname_r, x_r, y_r, e_r = traces[1]
                ax2 = ax.twinx()
                self._local_axes_tag_map[id(ax)] = pname_l
                self._local_axes_tag_map[id(ax2)] = pname_r

                ax.errorbar(
                    x_l, y_l, yerr=e_l, fmt="o", linestyle="none", color="C0", capsize=2, zorder=6
                )
                finite_l = np.isfinite(e_l) & (e_l > 0)
                if np.any(finite_l):
                    ax.errorbar(
                        x_l,
                        y_l,
                        yerr=e_l,
                        fmt="none",
                        ecolor="C0",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax2.errorbar(
                    x_r, y_r, yerr=e_r, fmt="o", linestyle="none", color="C1", capsize=2, zorder=6
                )
                finite_r = np.isfinite(e_r) & (e_r > 0)
                if np.any(finite_r):
                    ax2.errorbar(
                        x_r,
                        y_r,
                        yerr=e_r,
                        fmt="none",
                        ecolor="C1",
                        capsize=2,
                        elinewidth=1,
                        zorder=5,
                    )

                ax.set_ylabel(self._parameter_label(pname_l), color="C0")
                ax2.set_ylabel(self._parameter_label(pname_r), color="C1")
                ax.tick_params(axis="y", colors="C0")
                ax2.tick_params(axis="y", colors="C1")
                ax.set_xlabel(x_label)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                log_l = self._is_local_param_log_enabled(pname_l)
                log_r = self._is_local_param_log_enabled(pname_r)
                if log_l:
                    ax.set_yscale("log")
                if log_r:
                    ax2.set_yscale("log")
                ax.grid(True, alpha=0.3)
            else:
                for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                    color = f"C{idx % 10}"
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr,
                        fmt="o",
                        linestyle="none",
                        color=color,
                        capsize=2,
                        zorder=6,
                        label=self._parameter_label(pname),
                    )
                    finite_err = np.isfinite(e_arr) & (e_arr > 0)
                    if np.any(finite_err):
                        ax.errorbar(
                            x_arr,
                            y_arr,
                            yerr=e_arr,
                            fmt="none",
                            ecolor=color,
                            capsize=2,
                            elinewidth=1,
                            zorder=5,
                        )
                    fit = self._local_model_fits.get(pname)
                    if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                        for curve in evaluate_parameter_model_fit(fit, num_points=200):
                            ax.plot(curve.x, curve.y, color=color, linewidth=1.2, zorder=7)
                if len(traces) == 1:
                    ax.set_ylabel(self._parameter_label(traces[0][0]))
                else:
                    ax.set_ylabel("Parameter Value")
                    ax.legend(loc="best")
                ax.set_xlabel(x_label)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                if len(traces) == 1:
                    param_log = self._is_local_param_log_enabled(traces[0][0])
                else:
                    param_log = False
                if param_log:
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

        self._draw_plot_annotations(local=True)

        self._local_figure.tight_layout()
        self._local_canvas.draw()

    def _write_local_parameter_data_file(
        self,
        gle_path: Path,
        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[Path, dict[str, tuple[int, int]]]:
        data_path = gle_path.with_name(f"{gle_path.stem}_local_parameters.dat")

        x_keys: dict[str, float] = {}
        for _pname, x_arr, _y_arr, _e_arr in traces:
            for xv in x_arr:
                x_keys[f"{float(xv):.12g}"] = float(xv)
        x_values = np.array(sorted(x_keys.values()), dtype=float)

        col_map: dict[str, tuple[int, int]] = {}
        columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
            y_out = np.full_like(x_values, np.nan, dtype=float)
            e_out = np.full_like(x_values, np.nan, dtype=float)
            local = {
                f"{float(x):.12g}": (float(y), float(e))
                for x, y, e in zip(x_arr, y_arr, e_arr, strict=True)
            }
            for i, xv in enumerate(x_values):
                key = f"{float(xv):.12g}"
                if key in local:
                    yv, ev = local[key]
                    y_out[i] = yv
                    e_out[i] = ev if np.isfinite(ev) and ev >= 0 else np.nan
            columns[pname] = (y_out, e_out)
            y_col = 2 + idx * 2
            e_col = y_col + 1
            col_map[pname] = (y_col, e_col)

        global_cols: list[tuple[str, float, float]] = []
        if self._result is not None:
            for p in self._result.global_parameters:
                err = self._result.global_uncertainties.get(p.name, np.nan)
                err_val = (
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )
                global_cols.append((p.name, float(p.value), err_val))

        with open(data_path, "w", encoding="utf-8") as f:
            f.write("! Local parameter trends\n")

            if self._result is not None:
                for line in self._global_parameter_table_lines(self._result):
                    f.write(line + "\n")
                f.write("!\n")

            f.write(f"! columns: {self._local_group_axis_label_plain()}")
            for pname, _x, _y, _e in traces:
                info = get_param_info(pname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f" {pname}{unit} err_{pname}{unit}")
            for gname, _gval, _gerr in global_cols:
                info = get_param_info(gname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f" {gname}{unit} err_{gname}{unit}")
            f.write("\n")

            col_idx = 1
            f.write("! Column map:\n")
            f.write(f"!   c{col_idx:>2} = {self._local_group_axis_label_plain()}\n")
            col_idx += 1
            for pname, _x, _y, _e in traces:
                info = get_param_info(pname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f"!   c{col_idx:>2} = {pname}{unit}\n")
                col_idx += 1
                f.write(f"!   c{col_idx:>2} = err_{pname}{unit}\n")
                col_idx += 1
            for gname, _gval, _gerr in global_cols:
                info = get_param_info(gname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f"!   c{col_idx:>2} = {gname}{unit}\n")
                col_idx += 1
                f.write(f"!   c{col_idx:>2} = err_{gname}{unit}\n")
                col_idx += 1

            for row_idx, xv in enumerate(x_values):
                parts = [f"{xv:.10g}"]
                for pname, _x, _y, _e in traces:
                    yv, ev = columns[pname]
                    parts.append(f"{yv[row_idx]:.10g}")
                    parts.append(f"{ev[row_idx]:.10g}")
                for _gname, gval, gerr in global_cols:
                    parts.append(f"{gval:.10g}")
                    parts.append(f"{gerr:.10g}")
                f.write(" ".join(parts) + "\n")

        return data_path, col_map

    def _write_local_parameter_fit_files(
        self,
        gle_path: Path,
        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, Path]:
        fit_files: dict[str, Path] = {}
        for pname, _x_arr, _y_arr, _e_arr in traces:
            fit = self._local_model_fits.get(pname)
            if fit is None or not fit.active or fit.x_key != self._local_group_x_key():
                continue

            curves = evaluate_parameter_model_fit(fit, num_points=200)
            if not curves:
                continue

            fit_path = gle_path.with_name(
                f"{gle_path.stem}_local_{self._safe_data_name(pname)}.fit"
            )
            info = get_param_info(pname)
            with open(fit_path, "w", encoding="utf-8") as f:
                f.write("! Local parameter model fit curve\n")
                f.write(f"! parameter: {info.plain_label()}\n")
                f.write(f"! x-axis: {self._local_group_axis_label_plain()}\n")
                f.write("! columns: x y\n")
                for cidx, curve in enumerate(curves):
                    if cidx > 0:
                        f.write("nan nan\n")
                    for xv, yv in zip(curve.x, curve.y, strict=True):
                        f.write(f"{float(xv):.10g} {float(yv):.10g}\n")

            fit_files[pname] = fit_path

        return fit_files

    def _ordered_components_for_stacking(
        self, components: list[tuple[str, np.ndarray]]
    ) -> list[tuple[str, np.ndarray]]:
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

    def _set_add_label_mode(self, enabled: bool) -> None:
        self._add_label_mode = bool(enabled)

    def _default_group_label_position(self, ax) -> tuple[float, float]:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if not (
            np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max)
        ):
            return 0.0, 0.0
        x = float(x_min + 0.02 * (x_max - x_min))
        y = float(y_max - 0.05 * (y_max - y_min))
        return x, y

    def _ensure_group_label_annotation(self, axis_tag: str, label: str, ax) -> None:
        if axis_tag in self._suppressed_group_label_tags:
            return
        for ann in self._plot_annotations:
            if bool(ann.get("is_group_label")) and str(ann.get("axis_tag", "")) == axis_tag:
                return
        x_pos, y_pos = self._default_group_label_position(ax)
        self._plot_annotations.append(
            {
                "x": x_pos,
                "y": y_pos,
                "text": label,
                "axis_tag": axis_tag,
                "is_group_label": True,
                "artist": None,
            }
        )

    def _prune_stale_group_label_annotations(self, valid_tags: set[str]) -> None:
        self._plot_annotations = [
            ann
            for ann in self._plot_annotations
            if not (
                bool(ann.get("is_group_label")) and str(ann.get("axis_tag", "")) not in valid_tags
            )
        ]
        self._suppressed_group_label_tags &= valid_tags

    def _draw_plot_annotations(self, *, local: bool) -> None:
        annotations = self._local_plot_annotations if local else self._plot_annotations
        for ann in annotations:
            ann["artist"] = None
            ax = self._axis_for_tag(str(ann.get("axis_tag", "")), local=local)
            if ax is None:
                continue
            is_group_label = bool(ann.get("is_group_label"))
            if is_group_label and (not local) and (not self._fit_share_x_check.isChecked()):
                continue
            text_artist = ax.text(
                float(ann.get("x", 0.0)),
                float(ann.get("y", 0.0)),
                str(ann.get("text", "")),
                fontsize=9,
                ha="left",
                va="top" if is_group_label else "bottom",
                zorder=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5}
                if is_group_label
                else None,
            )
            text_artist.set_picker(True)
            ann["artist"] = text_artist

    def _axis_for_tag(self, tag: str, *, local: bool):
        figure = self._local_figure if local else self._left_figure
        tag_map = self._local_axes_tag_map if local else self._axes_tag_map
        if figure is None:
            return None
        for ax in figure.axes:
            if tag_map.get(id(ax)) == tag:
                return ax
        return None

    def _annotation_at_event(self, event, *, local: bool) -> dict[str, object] | None:
        annotations = self._local_plot_annotations if local else self._plot_annotations
        for ann in annotations:
            artist = ann.get("artist")
            if artist is None:
                continue
            contains, _ = artist.contains(event)
            if contains:
                return ann
        return None

    def _on_canvas_button_press(self, event) -> None:
        self._handle_canvas_button_press(event, local=False)

    def _on_local_canvas_button_press(self, event) -> None:
        self._handle_canvas_button_press(event, local=True)

    def _handle_canvas_button_press(self, event, *, local: bool) -> None:
        if event.inaxes is None:
            return

        ann = self._annotation_at_event(event, local=local)
        if event.button == 3 and ann is not None:
            target = self._local_plot_annotations if local else self._plot_annotations
            if not local and bool(ann.get("is_group_label")):
                self._suppressed_group_label_tags.add(str(ann.get("axis_tag", "")))
            target.remove(ann)
            if local:
                self._refresh_local_parameter_plots()
            else:
                self._refresh_plot()
            return

        if event.button == 1 and event.dblclick and ann is not None:
            current = str(ann.get("text", ""))
            text, ok = QInputDialog.getText(self, "Edit Label", "Text:", text=current)
            if ok:
                ann["text"] = text
                if local:
                    self._refresh_local_parameter_plots()
                else:
                    self._refresh_plot()
            return

        if event.button == 1 and self._add_label_mode:
            text, ok = QInputDialog.getText(self, "Add Label", "Text:")
            if ok and text.strip():
                axis_tag = (
                    self._local_axes_tag_map.get(id(event.inaxes), "")
                    if local
                    else self._axes_tag_map.get(id(event.inaxes), "")
                )
                target = self._local_plot_annotations if local else self._plot_annotations
                target.append(
                    {
                        "x": float(event.xdata),
                        "y": float(event.ydata),
                        "text": text.strip(),
                        "axis_tag": axis_tag,
                        "artist": None,
                    }
                )
                if local:
                    self._refresh_local_parameter_plots()
                else:
                    self._refresh_plot()
                self._add_label_btn.setChecked(False)
            return

        if event.button == 1 and ann is not None:
            self._dragging_annotation = ann

    def _on_canvas_motion(self, event) -> None:
        self._handle_canvas_motion(event, local=False)

    def _on_local_canvas_motion(self, event) -> None:
        self._handle_canvas_motion(event, local=True)

    def _handle_canvas_motion(self, event, *, local: bool) -> None:
        if self._dragging_annotation is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._dragging_annotation["x"] = float(event.xdata)
        self._dragging_annotation["y"] = float(event.ydata)
        artist = self._dragging_annotation.get("artist")
        if artist is not None:
            artist.set_position((float(event.xdata), float(event.ydata)))
            canvas = self._local_canvas if local else self._left_canvas
            if canvas is not None:
                canvas.draw_idle()

    def _on_canvas_button_release(self, _event) -> None:
        self._dragging_annotation = None

    def _on_local_canvas_button_release(self, _event) -> None:
        self._dragging_annotation = None

    def _fit_subplot_layout_params(self, subplot_aspect: float) -> dict[str, float]:
        """Return aspect-aware subplot spacing to reduce clipping across shapes."""
        # Narrower plots need more side margins for y-labels/ticks; wider plots can use tighter margins.
        t = max(0.0, min(1.0, (2.8 - float(subplot_aspect)) / 1.8))
        left = 0.11 + 0.09 * t
        right = 0.992 - 0.045 * t
        return {
            "left": left,
            "right": right,
            "top": 0.968,
            "bottom": 0.065,
            "hspace": 0.50,
        }

    def _build_fit_subplot_gle_figure(self, glp, gle_path: Path):
        share_x_axis = self._fit_share_x_check.isChecked()
        subplot_height = 3.1
        subplot_aspect = float(self._fit_subplot_aspect_spin.value())
        plot_width = subplot_aspect * subplot_height
        fig, axes = glp.subplots(
            nrows=max(1, len(self._groups)),
            ncols=1,
            figsize=(plot_width, max(5.2, subplot_height * max(1, len(self._groups)))),
            sharex=share_x_axis,
        )
        subplot_axes = axes if isinstance(axes, list) else [axes]
        file_map = self._write_fit_subplot_files(gle_path)

        x_label = self._x_label_gle()
        y_label = self._parameter_label_gle(self._parameter_name)
        for idx, group in enumerate(self._groups):
            ax = subplot_axes[idx]
            file_info = file_map.get(group.group_id)
            data_path = file_info.get("data_path") if isinstance(file_info, dict) else None
            if (
                isinstance(file_info, dict)
                and data_path is not None
                and hasattr(ax, "errorbar_from_file")
            ):
                ax.errorbar_from_file(
                    data_path.name,
                    x_col=1,
                    y_col=2,
                    yerr_col=3 if bool(file_info.get("has_err", False)) else None,
                    color="black",
                    marker="o",
                    markersize=4,
                    capsize=2,
                )
            else:
                yerr = np.asarray(group.yerr, dtype=float)
                has_err = bool(np.any(np.isfinite(yerr) & (yerr > 0)))
                ax.errorbar(
                    group.x,
                    group.y,
                    yerr=group.yerr if has_err else None,
                    fmt="o",
                    linestyle="none",
                    marker="o",
                    color="black",
                    capsize=2,
                )

            fit_path = file_info.get("fit_path") if isinstance(file_info, dict) else None
            sampled = self._sample_group_fit_curve(group)
            if sampled is not None:
                xx, yy = sampled
            else:
                xx, yy = None, None

            if (
                xx is not None
                and self._show_components_check.isChecked()
                and self._model is not None
            ):
                try:
                    kwargs = self._model_kwargs_for_group(group.group_id)
                    components = self._model.evaluate_components(xx, additive_only=True, **kwargs)
                    ordered = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xx, dtype=float)
                    component_colors = [
                        "lightblue",
                        "lightgreen",
                        "pink",
                        "lightgray",
                        "cyan",
                        "yellow",
                    ]
                    group_token = self._safe_data_name(group.group_name or group.group_id)
                    for cidx, (_name, comp_y) in enumerate(ordered):
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_token = self._safe_data_name(_name)
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(
                            xx,
                            lower,
                            upper,
                            color=fill_color,
                            alpha=0.3,
                            data_name=f"component_{group_token}_{comp_token}_fill",
                        )
                        ax.plot(
                            xx,
                            upper,
                            linestyle="--",
                            linewidth=0.8,
                            color=fill_color,
                            data_name=f"component_{group_token}_{comp_token}_edge",
                        )
                        cumulative = upper
                except KeyError:
                    pass

            if (
                isinstance(file_info, dict)
                and bool(file_info.get("has_fit", False))
                and fit_path is not None
                and hasattr(ax, "line_from_file")
            ):
                ax.line_from_file(
                    fit_path.name,
                    x_col=1,
                    y_col=2,
                    color="red",
                    linestyle="-",
                    linewidth=1.5,
                )
            else:
                if xx is not None and yy is not None:
                    group_token = self._safe_data_name(group.group_name or group.group_id)
                    ax.plot(xx, yy, color="red", linewidth=1.5, data_name=f"model_{group_token}")
                else:
                    ax.text(0.02, 0.95, "Fit curve unavailable", color="red", ha="left")

            for ann in self._plot_annotations:
                if str(ann.get("axis_tag", "")) != group.group_id:
                    continue
                is_group_label = bool(ann.get("is_group_label"))
                if is_group_label and not share_x_axis:
                    continue
                try:
                    x_ann = float(ann.get("x", 0.0))
                    y_ann = float(ann.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                text = str(ann.get("text", "")).strip()
                if text:
                    ax.text(
                        x_ann,
                        y_ann,
                        text,
                        ha="left",
                        va="top" if is_group_label else "bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5}
                        if is_group_label
                        else None,
                    )

            if share_x_axis:
                self._ensure_group_label_annotation(group.group_id, group.group_name, ax)
                ax.set_title("")
            else:
                ax.set_title(group.group_name)
            ax.set_ylabel(y_label)
            self._apply_fit_axis_scales(ax)
            if (not share_x_axis) or idx == len(self._groups) - 1:
                ax.set_xlabel(x_label)

        if hasattr(fig, "subplots_adjust"):
            fig.subplots_adjust(**self._fit_subplot_layout_params(subplot_aspect))

        return fig

    def _build_local_parameter_gle_figure(self, glp, gle_path: Path):
        local_param_names = sorted(
            {p.name for pset in self._result.local_parameters.values() for p in pset}
        )
        if not local_param_names:
            fig = glp.figure(figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            return fig

        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for pname in local_param_names:
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []
            for group in self._groups:
                pset = self._result.local_parameters.get(group.group_id)
                if pset is None or pname not in pset:
                    continue
                xs.append(float(group.group_variable_value))
                ys.append(float(pset[pname].value))
                err = self._result.local_uncertainties.get(group.group_id, {}).get(pname)
                es.append(
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )
            if not xs:
                continue
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            e_arr = np.asarray(es, dtype=float)
            order = np.argsort(x_arr)
            traces.append((pname, x_arr[order], y_arr[order], e_arr[order]))

        all_traces = list(traces)
        selected_names = self._selected_local_y_parameters()
        if selected_names:
            traces = [t for t in traces if t[0] in selected_names]
        if not traces:
            fig = glp.figure(figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title("No local parameters selected")
            return fig

        data_path, col_map = self._write_local_parameter_data_file(gle_path, all_traces)
        fit_file_map = self._write_local_parameter_fit_files(gle_path, all_traces)

        x_label = self._local_group_axis_label_gle()
        plot_mode = self._local_plot_mode_combo.currentText()
        if plot_mode == "Subplots" and len(traces) > 1:
            num_cols = 1
            num_rows = (len(traces) + num_cols - 1) // num_cols
            fig, axes = glp.subplots(
                nrows=num_rows, ncols=num_cols, figsize=(7.2, max(4.8, 2.8 * num_rows)), sharex=True
            )
            flat_axes = axes if isinstance(axes, list) else [axes]
            for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                ax = flat_axes[idx]
                has_err = bool(np.any(np.isfinite(e_arr) & (e_arr >= 0)))
                cols = col_map[pname]
                if hasattr(ax, "errorbar_from_file"):
                    ax.errorbar_from_file(
                        data_path.name,
                        x_col=1,
                        y_col=cols[0],
                        yerr_col=cols[1] if has_err else None,
                        marker="o",
                        color="black",
                        capsize=2,
                    )
                else:
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr if has_err else None,
                        marker="o",
                        color="black",
                        linestyle="none",
                        capsize=2,
                    )
                ax.set_ylabel(self._parameter_label_gle(pname))
                row = idx // num_cols
                if row == num_rows - 1:
                    ax.set_xlabel(x_label)
                if self._local_log_x_check.isChecked():
                    ax.set_xscale("log")
                param_log = self._is_local_param_log_enabled(pname)
                if param_log:
                    ax.set_yscale("log")
                fit = self._local_model_fits.get(pname)
                if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                    fit_path = fit_file_map.get(pname)
                    if fit_path is not None and hasattr(ax, "line_from_file"):
                        ax.line_from_file(
                            fit_path.name,
                            x_col=1,
                            y_col=2,
                            color="red",
                            linestyle="-",
                            linewidth=1.5,
                        )
                    else:
                        for curve in evaluate_parameter_model_fit(fit, num_points=200):
                            ax.plot(
                                curve.x,
                                curve.y,
                                color="red",
                                linewidth=1.5,
                                data_name=f"model_local_{self._safe_data_name(pname)}",
                            )
                for ann in self._local_plot_annotations:
                    if str(ann.get("axis_tag", "")) != pname:
                        continue
                    text = str(ann.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        x_ann = float(ann.get("x", 0.0))
                        y_ann = float(ann.get("y", 0.0))
                    except (TypeError, ValueError):
                        continue
                    ax.text(x_ann, y_ann, text, ha="left")
            return fig

        fig = glp.figure(figsize=(7.0, 4.5))
        ax = fig.add_subplot(111)
        for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
            color = ["black", "blue", "red"][idx % 3]
            has_err = bool(np.any(np.isfinite(e_arr) & (e_arr >= 0)))
            cols = col_map[pname]
            if hasattr(ax, "errorbar_from_file"):
                ax.errorbar_from_file(
                    data_path.name,
                    x_col=1,
                    y_col=cols[0],
                    yerr_col=cols[1] if has_err else None,
                    marker="o",
                    color=color,
                    capsize=2,
                    label=self._parameter_label_gle(pname),
                )
            else:
                ax.errorbar(
                    x_arr,
                    y_arr,
                    yerr=e_arr if has_err else None,
                    marker="o",
                    linestyle="none",
                    color=color,
                    capsize=2,
                    label=self._parameter_label_gle(pname),
                )
            fit = self._local_model_fits.get(pname)
            if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                fit_path = fit_file_map.get(pname)
                if fit_path is not None and hasattr(ax, "line_from_file"):
                    ax.line_from_file(
                        fit_path.name,
                        x_col=1,
                        y_col=2,
                        color=color,
                        linestyle="-",
                        linewidth=1.2,
                    )
                else:
                    for curve in evaluate_parameter_model_fit(fit, num_points=200):
                        ax.plot(
                            curve.x,
                            curve.y,
                            color=color,
                            linewidth=1.2,
                            data_name=f"model_local_{self._safe_data_name(pname)}",
                        )
        ax.set_xlabel(x_label)
        if self._local_log_x_check.isChecked():
            ax.set_xscale("log")
        if len(traces) == 1:
            param_log = self._is_local_param_log_enabled(traces[0][0])
            if param_log:
                ax.set_yscale("log")
        if len(traces) == 1:
            ax.set_ylabel(self._parameter_label_gle(traces[0][0]))
        else:
            ax.set_ylabel("Parameter Value")
            ax.legend(loc="best")

        for ann in self._local_plot_annotations:
            axis_tag = str(ann.get("axis_tag", ""))
            if axis_tag not in {"main"} and not any(name == axis_tag for name, *_rest in traces):
                continue
            text = str(ann.get("text", "")).strip()
            if not text:
                continue
            try:
                x_ann = float(ann.get("x", 0.0))
                y_ann = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            ax.text(x_ann, y_ann, text, ha="left")

        return fig

    def _show_gle_preview(self, output_path: Path) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if not output_path.exists():
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
                gle_path = output_path.with_suffix(".gle")
                if gle_path.exists():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir_path = Path(tmpdir)
                        tmp_gle = tmpdir_path / gle_path.name
                        preview_png = tmp_gle.with_suffix(".png")

                        shutil.copy2(gle_path, tmp_gle)
                        for dep_name in self._extract_gle_data_dependencies(gle_path):
                            src = gle_path.parent / dep_name
                            if src.exists() and src.is_file():
                                shutil.copy2(src, tmpdir_path / dep_name)

                        compile_gle(_gle, tmp_gle, "png", cwd=tmpdir_path)

                        pixmap = QPixmap(str(preview_png))
                        if not pixmap.isNull():
                            image_label.setPixmap(pixmap)
                            image_label.setText("")

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec()
        except Exception as exc:
            QMessageBox.warning(self, "Preview error", f"Failed to show preview: {exc}")

    def _extract_gle_data_dependencies(self, gle_path: Path) -> list[str]:
        """Return data-file names referenced by `data <file>` commands in a GLE script."""
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

    def _compile_and_preview_gle(self, gle_path: Path, output_format: str) -> None:
        _gle = get_gle_executable()
        if _gle is None:
            QMessageBox.information(
                self,
                "GLE Not Installed",
                f"GLE script saved to {gle_path}. Install GLE to compile to {output_format.upper()}.",
            )
            return

        output_path = gle_path.with_suffix(f".{output_format}")
        try:
            compile_gle(_gle, gle_path, output_format, cwd=gle_path.parent)
            QMessageBox.information(
                self,
                "Export Successful",
                f"GLE plot exported:\n\nGLE script: {gle_path}\nOutput: {output_path}",
            )
            self._show_gle_preview(output_path)
        except subprocess.CalledProcessError as exc:
            QMessageBox.warning(self, "GLE compilation failed", exc.stderr or str(exc))
            self._show_gle_preview(output_path)

    def _export_plot_gle(
        self, *, title: str, default_name: str, builder, output_format: str
    ) -> None:
        if self._result is None or self._parameter_name is None:
            QMessageBox.information(self, "No result", "Run a cross-group fit first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_export_path(default_name),
            "GLE export folders (*.gleplot)",
        )
        if not path:
            return
        remember_export_path(path)

        # The fit-subplot export reuses the warm curve cache; if it is cold
        # (export before the plot has drawn) the builder falls back to an inline
        # 800-pt evaluation per group — show a busy cursor for that stretch.
        cold = self._fit_curve_cache_is_cold()
        if cold:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            glp = importlib.import_module("gleplot")
            requested_gle_path = Path(path)
            gle_path, export_dir = resolve_gle_export_paths(requested_gle_path, folder=True)
            export_dir.mkdir(parents=True, exist_ok=True)
            fig = builder(glp, gle_path)
            fig.savefig(str(gle_path))
            self._compile_and_preview_gle(gle_path, output_format)
        except ImportError:
            QMessageBox.warning(
                self, "gleplot not available", "Install gleplot to export GLE plots."
            )
        except TypeError as exc:
            if "folder" in str(exc):
                QMessageBox.warning(
                    self, "gleplot update required", "Please update gleplot to a newer version."
                )
                return
            QMessageBox.warning(self, "Export failed", f"Could not export GLE: {exc}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", f"Could not export GLE: {exc}")
        finally:
            if cold:
                QGuiApplication.restoreOverrideCursor()

    def _fit_curve_cache_is_cold(self) -> bool:
        """Return ``True`` when any shown group lacks a cached total fit curve.

        A warm cache (the normal post-display state) lets the GLE export reuse
        the off-thread curves; a cold cache means the export will evaluate the
        800-pt curve inline, so the caller shows a busy cursor.
        """
        for group in self._groups:
            if self._cached_group_total_curve(group.group_id) is None:
                return True
        return False

    def _export_fit_subplot_gle(self) -> None:
        self._export_plot_gle(
            title="Export Fit Subplots to GLE",
            default_name="global_parameter_fit_subplots.gleplot",
            builder=self._build_fit_subplot_gle_figure,
            output_format=self._fit_gle_format_combo.currentText().lower(),
        )

    def _export_local_parameters_gle(self) -> None:
        self._export_plot_gle(
            title="Export Local Parameters Plot to GLE",
            default_name="global_parameter_fit_local_parameters.gleplot",
            builder=self._build_local_parameter_gle_figure,
            output_format=self._local_gle_format_combo.currentText().lower(),
        )
