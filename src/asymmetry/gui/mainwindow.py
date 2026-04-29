"""Main window shell — WiMDA-inspired layout.

Layout overview (mirroring WiMDA):

    ┌──────────────────────────────────────────────────┐
    │  Menu bar  ·  Toolbar                            │
    ├────────────┬─────────────────────┬───────────────┤
    │            │                     │               │
    │  Data      │    Plot canvas      │  Fit /        │
    │  browser   │    (central)        │  Fourier /    │
    │  / logbook │                     │  Analysis     │
    │  (left     │                     │  panels       │
    │   dock)    │                     │  (right dock) │
    │            │                     │               │
    ├────────────┴─────────────────────┴───────────────┤
    │  Log / message panel  (bottom dock)              │
    └──────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QToolBar,
)

from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    load_project,
    save_project,
)
from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform import (
    apply_deadtime_correction,
    apply_grouped_background_correction,
    apply_grouping_aligned,
    common_t0_for_groups,
    compute_asymmetry,
    has_file_deadtime,
    prepare_histograms_with_deadtime,
    supports_background_correction,
)
from asymmetry.core.transform.rebin import rebin
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.export_paths import default_export_path, remember_export_path

_MAX_RECENT_PROJECTS = 10
_PROJECT_FILE_FILTER = "Asymmetry projects (*.asymp);;All files (*)"

from asymmetry.gui.panels.data_browser import DataBrowserPanel
from asymmetry.gui.panels.fit_panel import FitPanel
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from asymmetry.gui.windows.grouping_dialog import GroupingDialog, WimGroupingDialog
from asymmetry.gui.windows.run_info_dialog import RunInfoDialog


def _normalise_source_path(path: str) -> str:
    """Return a canonical string for source-file path comparisons."""
    return os.path.normcase(os.path.abspath(os.path.realpath(path)))


def _load_window_icon() -> QIcon | None:
    """Load window icon from package resources.

    Returns None if icon cannot be loaded.
    """
    # Try importlib.resources (preferred for installed packages)
    try:
        from importlib.resources import files

        logo = files("asymmetry.resources").joinpath("logo_256x256.png")
        if logo.is_file():
            pixmap = QPixmap()
            if pixmap.loadFromData(logo.read_bytes(), "PNG"):
                icon = QIcon(pixmap)
                if not icon.isNull():
                    return icon
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError, OSError):
        pass

    # Fallback: try direct path (for development)
    try:
        resources_dir = Path(__file__).parent.parent / "resources"
        icon_path = resources_dir / "logo_256x256.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon
    except (OSError, ValueError):
        pass

    return None


class MainWindow(QMainWindow):
    """Top-level application window for the Asymmetry μSR analysis GUI.

    Orchestrates all panels (data browser, plot, fit, Fourier, log) and
    manages the central data flow between them.  Also owns project-file
    save/load and the recent-projects list stored in :class:`QSettings`.

    Project files (``.asymp``)
    --------------------------
    The ``collect_project_state`` / ``restore_project_state`` pair serialises
    and deserialises the full application state as a versioned JSON file.
    Source data files are *referenced* rather than embedded, so each relevant
    ``.wim`` file must remain accessible at its original path (or at the same
    relative path from the ``.asymp`` file) for the project to reopen cleanly.
    Missing files are skipped with a ``WARNING`` log entry; the rest of the
    session restores normally.

    Saved state includes:

    * List of loaded datasets with source-file paths and field overrides
    * Co-added (combined) dataset group definitions
    * Data browser sort column, active filters, and selected runs
    * Plot panel axis limits and fit-curve overlay
    * Single-fit and global-fit model selection and parameter table contents
    * Fourier panel window, zero-pad factor, and display mode

    Not saved:

    * Raw asymmetry / time / error arrays (reloaded from source files)
    * Fit *result* statistics (χ², uncertainties) — only the fitted
      parameter values that were written back to the parameter table
    * Fourier transform output (settings only)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Asymmetry — μSR Data Analysis")

        # Set window icon from package resources
        icon = _load_window_icon()
        if icon is not None:
            self.setWindowIcon(icon)

        self.resize(1400, 900)

        self._settings = QSettings()
        self._last_open_dir = self._settings.value("io/last_open_dir", "", str)
        self._current_dataset = None  # Track currently selected dataset
        self._current_project_path: str | None = None  # Path of currently open project
        self._active_group_context: tuple[str, str] | None = None
        self._global_parameter_fit_window: GlobalParameterFitWindow | None = None

        self._setup_menus()
        self._setup_toolbar()
        self._setup_panels()

        # Check for SciPy availability and warn if using fallback
        from asymmetry.core.fitting.diffusion import is_scipy_available

        if not is_scipy_available():
            self._log_panel.log(
                "⚠️  WARNING: SciPy is unavailable or broken. "
                "Diffusion model will use slower NumPy fallback for numerical integration. "
                "Please repair SciPy in your Python environment."
            )

        self.statusBar().showMessage("Ready")

    # ── menus ──────────────────────────────────────────────────────────

    def _setup_menus(self) -> None:
        """Build the application menu bar with File, Analysis, View, and Help menus."""
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        file_menu.addAction("Open Data File(s)\u2026", self._on_open)
        file_menu.addSeparator()
        file_menu.addAction("&New Project", self._on_new_project)
        file_menu.addAction("Open Project\u2026", self._on_open_project)
        file_menu.addAction("&Save Project", self._on_save_project)
        file_menu.addAction("Save Project &As\u2026", self._on_save_project_as)
        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._update_recent_projects_menu()
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        # Analysis
        analysis_menu = mb.addMenu("&Analysis")
        analysis_menu.addAction("&Fit", self._on_fit)
        analysis_menu.addAction("F&ourier", self._on_fourier)
        analysis_menu.addAction("Fit &Parameters", self._on_fit_parameters)
        analysis_menu.addAction("Grouping...", self._on_grouping_current)
        self._global_parameter_fit_action = analysis_menu.addAction(
            "Global Parameter Fit",
            self._on_global_parameter_fit,
        )
        self._update_global_parameter_fit_menu_style(False)

        # View
        view_menu = mb.addMenu("&View")
        view_menu.addAction("Reset layout", self._reset_layout)

        # Help
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("&About…", self._on_about)

    # ── toolbar ────────────────────────────────────────────────────────

    def _setup_toolbar(self) -> None:
        """Add a non-movable toolbar with shortcuts for the most common actions."""
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("Open", self._on_open)
        tb.addAction("Export logbook", self._on_export_logbook)
        tb.addAction("Grouping", self._on_grouping_current)
        tb.addAction("Fit", self._on_fit)
        tb.addAction("FFT", self._on_fourier)
        tb.addAction("Params", self._on_fit_parameters)
        self._global_parameter_fit_toolbar_action = tb.addAction("Global Fit", self._on_global_parameter_fit)
        self._global_parameter_fit_toolbar_action.setEnabled(False)

    # ── panels / docks ─────────────────────────────────────────────────

    def _setup_panels(self) -> None:
        """Create and dock all child panels, then connect inter-panel signals."""
        # Enable dock nesting for proper splitter behavior
        self.setDockNestingEnabled(True)

        # Central plot
        self._plot_panel = PlotPanel()
        self.setCentralWidget(self._plot_panel)

        # Left dock — data browser / logbook
        self._data_browser = DataBrowserPanel()
        dock_left = QDockWidget("Data Browser", self)
        dock_left.setWidget(self._data_browser)
        dock_left.setMinimumWidth(250)  # Can be shrunk by user, but defaults to larger size below
        dock_left.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_left)

        # Right dock — fit controls
        self._fit_panel = FitPanel()
        self._dock_fit = QDockWidget("Fit", self)
        self._dock_fit.setWidget(self._fit_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fit)

        # Right dock — Fourier controls (tabbed with fit)
        self._fourier_panel = FourierPanel()
        self._dock_fourier = QDockWidget("Fourier", self)
        self._dock_fourier.setWidget(self._fourier_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fourier)
        self.tabifyDockWidget(self._dock_fit, self._dock_fourier)

        # Right dock — fitted parameter trends (tabbed with fit/fourier)
        self._fit_parameters_panel = FitParametersPanel()
        self._dock_fit_parameters = QDockWidget("Fit Parameters", self)
        self._dock_fit_parameters.setWidget(self._fit_parameters_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fit_parameters)
        self.tabifyDockWidget(self._dock_fit, self._dock_fit_parameters)

        # Analysis docks are opened on demand from toolbar/menu actions.
        self._dock_fit.hide()
        self._dock_fourier.hide()
        self._dock_fit_parameters.hide()

        # Bottom dock — log panel
        self._log_panel = LogPanel()
        dock_log = QDockWidget("Log", self)
        dock_log.setWidget(self._log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_log)

        # Set initial dock widths: data browser gets ~480px by default
        self.resizeDocks(
            [dock_left],
            [480],
            Qt.Orientation.Horizontal
        )

        # Connect signals
        self._data_browser.dataset_selected.connect(self._on_dataset_selected)
        if hasattr(self._data_browser, "get_info_requested"):
            self._data_browser.get_info_requested.connect(self._on_get_info_requested)
        if hasattr(self._data_browser, "grouping_requested"):
            self._data_browser.grouping_requested.connect(self._on_grouping_requested)
        if hasattr(self._data_browser, "group_selected"):
            self._data_browser.group_selected.connect(self._on_group_selected)
        self._data_browser.selection_changed.connect(self._update_selected_datasets)
        self._plot_panel.fit_range_changed.connect(self._on_fit_range_changed)
        if hasattr(self._plot_panel, "bunch_factor_changed"):
            self._plot_panel.bunch_factor_changed.connect(self._update_selected_datasets)
        if hasattr(self._plot_panel, "overlay_toggled"):
            self._plot_panel.overlay_toggled.connect(self._on_overlay_toggled)
        if hasattr(self._plot_panel, "polarization_axis_changed"):
            self._plot_panel.polarization_axis_changed.connect(
                self._on_plot_polarization_axis_changed
            )
        self._fit_panel.fit_completed.connect(self._on_fit_completed)
        self._fit_panel.global_fit_completed.connect(self._on_global_fit_completed)
        if hasattr(self._fit_panel, "preview_requested"):
            self._fit_panel.preview_requested.connect(self._on_preview_requested)
        if hasattr(self._fit_panel, "share_function_with_group_requested"):
            self._fit_panel.share_function_with_group_requested.connect(
                self._on_share_single_function_with_group
            )
        if hasattr(self._fit_parameters_panel, "cross_group_fit_completed"):
            self._fit_parameters_panel.cross_group_fit_completed.connect(
                self._on_cross_group_fit_completed
            )
        if hasattr(self._fit_parameters_panel, "delete_group_fits_requested"):
            self._fit_parameters_panel.delete_group_fits_requested.connect(
                self._on_fit_parameters_group_fits_deleted
            )

        # Update selected datasets for global fitting whenever selection changes
        self._update_selected_datasets()

    def _normalize_vector_axis(self, axis: object) -> str | None:
        """Normalize polarization-axis labels to one of ``P_x``, ``P_y``, ``P_z``."""
        if axis is None:
            return None
        token = str(axis).strip().lower().replace(" ", "").replace("_", "")
        if token in {"all", "pall"}:
            return "ALL"
        if token in {"px", "x"}:
            return "P_x"
        if token in {"py", "y"}:
            return "P_y"
        if token in {"pz", "z"}:
            return "P_z"
        return None

    def _vector_alpha_key(self, axis: str | None) -> str | None:
        """Return grouping alpha key for a canonical vector axis."""
        return {
            "P_x": "alpha_x",
            "P_y": "alpha_y",
            "P_z": "alpha_z",
        }.get(str(axis) if axis is not None else "")

    def _legacy_vector_alpha_key(self, axis: str | None) -> str | None:
        """Return legacy vector-alpha key for backward compatibility."""
        return {
            "P_x": "alpha_px",
            "P_y": "alpha_py",
            "P_z": "alpha_pz",
        }.get(str(axis) if axis is not None else "")

    def _resolve_vector_alpha_values(
        self,
        grouping_result: dict,
        existing_grouping: dict | None,
    ) -> dict[str, float]:
        """Resolve per-axis alpha values with backward-compatible fallback."""
        existing = existing_grouping if isinstance(existing_grouping, dict) else {}
        try:
            base_alpha = float(grouping_result.get("alpha", existing.get("alpha", 1.0)))
        except (TypeError, ValueError):
            base_alpha = 1.0

        resolved: dict[str, float] = {}
        for axis in ("P_x", "P_y", "P_z"):
            key = self._vector_alpha_key(axis)
            legacy_key = self._legacy_vector_alpha_key(axis)
            raw = grouping_result.get(
                key,
                grouping_result.get(
                    legacy_key,
                    existing.get(key, existing.get(legacy_key, base_alpha)),
                ),
            )
            try:
                resolved[axis] = float(raw)
            except (TypeError, ValueError):
                resolved[axis] = base_alpha
        return resolved

    def _vector_axis_pairs_for_grouping(
        self,
        groups: dict[int, list[int]],
        group_names: dict[int, str] | None,
    ) -> dict[str, tuple[int, int]]:
        """Return vector-axis pair mapping for EMU-style group names when present."""
        if not isinstance(groups, dict) or not groups:
            return {}

        names = group_names if isinstance(group_names, dict) else {}
        by_name: dict[str, int] = {}
        for gid, name in names.items():
            try:
                gid_int = int(gid)
            except (TypeError, ValueError):
                continue
            by_name[str(name).strip().lower()] = gid_int

        def _find(*candidates: str) -> int | None:
            for cand in candidates:
                gid = by_name.get(cand)
                if gid in groups and groups.get(gid):
                    return gid
            return None

        pz_f = _find("pz forward")
        pz_b = _find("pz backward")
        py_a = _find("py top", "py up")
        py_b = _find("py bottom", "py down")
        px_a = _find("px left")
        px_b = _find("px right")

        if None in {pz_f, pz_b, py_a, py_b, px_a, px_b}:
            return {}

        return {
            "P_z": (int(pz_f), int(pz_b)),
            "P_y": (int(py_a), int(py_b)),
            "P_x": (int(px_a), int(px_b)),
        }

    def _vector_axis_state_for_dataset(self, dataset) -> tuple[dict[str, tuple[int, int]], str | None]:
        """Return vector-axis pair mapping and currently selected axis for a dataset."""
        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if not isinstance(grouping, dict):
            return {}, None

        groups_raw = grouping.get("groups")
        if not isinstance(groups_raw, dict):
            return {}, None
        groups: dict[int, list[int]] = {}
        for k, vals in groups_raw.items():
            try:
                gid = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(vals, list) and vals:
                det_ids: list[int] = []
                for v in vals:
                    try:
                        det_ids.append(int(v))
                    except (TypeError, ValueError):
                        continue
                if det_ids:
                    groups[gid] = det_ids

        pairs = self._vector_axis_pairs_for_grouping(groups, grouping.get("group_names"))
        if not pairs:
            return {}, None

        axis = self._normalize_vector_axis(grouping.get("vector_axis"))
        if axis not in pairs:
            axis = "P_z"
        return pairs, axis

    def _refresh_vector_axis_selector(self) -> None:
        """Show/hide and synchronize the plot polarization selector."""
        if not hasattr(self._plot_panel, "set_polarization_axes"):
            return

        selected = list(self._data_browser.get_selected_datasets())
        if len(selected) > 1 and self._overlay_enabled():
            targets = selected
        else:
            targets = [self._current_dataset] if self._current_dataset else []
        targets = [ds for ds in targets if ds is not None]
        if not targets:
            self._plot_panel.set_polarization_axes([])
            return

        first_pairs, first_axis = self._vector_axis_state_for_dataset(targets[0])
        if not first_pairs:
            self._plot_panel.set_polarization_axes([])
            return

        for dataset in targets[1:]:
            pairs, _axis = self._vector_axis_state_for_dataset(dataset)
            if pairs != first_pairs:
                self._plot_panel.set_polarization_axes([])
                return

        axis_order = ["P_x", "P_y", "P_z"]
        available = [axis for axis in axis_order if axis in first_pairs]
        if available:
            available = ["ALL", *available]

        current = None
        if hasattr(self._plot_panel, "get_current_polarization_axis"):
            current = self._normalize_vector_axis(self._plot_panel.get_current_polarization_axis())
        if current not in available:
            current = first_axis if first_axis in available else (available[0] if available else None)
        self._plot_panel.set_polarization_axes(available, current)

    def _selected_or_current_datasets(self) -> list[MuonDataset]:
        """Return selected datasets, or the current dataset when none are selected."""
        selected = list(self._data_browser.get_selected_datasets())
        if selected:
            return selected
        if self._current_dataset is not None:
            return [self._current_dataset]
        return []

    def _overlay_enabled(self) -> bool:
        """Return whether multi-selection overlays should be shown."""
        if hasattr(self._plot_panel, "is_overlay_enabled"):
            return bool(self._plot_panel.is_overlay_enabled())
        return True

    @staticmethod
    def _run_numbers_match(dataset_a: MuonDataset | None, dataset_b: MuonDataset | None) -> bool:
        """Return True when both datasets represent the same run number."""
        if dataset_a is None or dataset_b is None:
            return False
        try:
            return int(dataset_a.run_number) == int(dataset_b.run_number)
        except (TypeError, ValueError):
            return False

    def _select_non_overlay_target(self, targets: list[MuonDataset]) -> MuonDataset | None:
        """Return the dataset that should remain visible when overlay is disabled."""
        if not targets:
            return None

        selected_group_ids = (
            self._data_browser.get_selected_group_ids()
            if hasattr(self._data_browser, "get_selected_group_ids")
            else []
        )
        single_group_selected = bool(
            hasattr(self._data_browser, "is_single_group_selected")
            and self._data_browser.is_single_group_selected()
        )

        if single_group_selected and len(selected_group_ids) == 1:
            group_id = selected_group_ids[0]
            group_member_runs = (
                self._data_browser.get_group_member_run_numbers(group_id)
                if hasattr(self._data_browser, "get_group_member_run_numbers")
                else []
            )
            run_map = {int(ds.run_number): ds for ds in targets}
            ordered_group_targets = [run_map[rn] for rn in group_member_runs if rn in run_map]
            if not ordered_group_targets:
                ordered_group_targets = targets

            if self._current_dataset is not None:
                for dataset in ordered_group_targets:
                    if self._run_numbers_match(self._current_dataset, dataset):
                        return dataset
            return ordered_group_targets[0]

        current_selected = (
            self._data_browser.get_current_dataset()
            if hasattr(self._data_browser, "get_current_dataset")
            else None
        )
        if current_selected is not None:
            for dataset in targets:
                if self._run_numbers_match(current_selected, dataset):
                    return dataset

        if self._current_dataset is not None:
            for dataset in targets:
                if self._run_numbers_match(self._current_dataset, dataset):
                    return dataset

        return targets[-1]

    def _build_vector_axis_datasets(
        self,
        datasets: list[MuonDataset],
    ) -> dict[str, list[MuonDataset]]:
        """Return per-axis cloned datasets for vector ``ALL`` subplot rendering."""
        axis_map: dict[str, list[MuonDataset]] = {"P_x": [], "P_y": [], "P_z": []}
        for axis in ("P_x", "P_y", "P_z"):
            for dataset in datasets:
                payload = self._extract_grouping_overrides(dataset)
                if not isinstance(payload, dict):
                    continue
                run = getattr(dataset, "run", None)
                if run is None:
                    continue
                groups = payload.get("groups", {})
                names = payload.get("group_names")
                pairs = self._vector_axis_pairs_for_grouping(groups, names)
                if axis not in pairs:
                    continue

                payload["vector_axis"] = axis
                clone_dataset = copy.deepcopy(dataset)
                applied, _ = self._apply_grouping_settings_to_dataset(clone_dataset, payload)
                if applied:
                    axis_map[axis].append(clone_dataset)
        return axis_map

    def _synchronize_targets_to_axis(
        self,
        targets: list[MuonDataset],
        axis: str | None,
    ) -> int:
        """Ensure *targets* use a consistent vector-axis component.

        Returns the number of datasets updated.
        """
        if axis not in {"P_x", "P_y", "P_z"}:
            return 0

        updated = 0
        for dataset in targets:
            if dataset is None:
                continue
            run = getattr(dataset, "run", None)
            grouping = getattr(run, "grouping", None)
            if not isinstance(grouping, dict):
                continue

            current_axis = self._normalize_vector_axis(grouping.get("vector_axis"))
            if current_axis == axis:
                continue

            payload = self._extract_grouping_overrides(dataset)
            if not isinstance(payload, dict):
                continue
            pairs = self._vector_axis_pairs_for_grouping(
                payload.get("groups", {}),
                payload.get("group_names"),
            )
            if axis not in pairs:
                continue

            fwd_gid, bwd_gid = pairs[axis]
            payload["forward_group"] = fwd_gid
            payload["backward_group"] = bwd_gid
            payload["vector_axis"] = axis

            applied, _ = self._apply_grouping_settings_to_dataset(dataset, payload)
            if applied:
                updated += 1

        return updated

    def _render_current_selection_plot(self) -> None:
        """Render current single/multi selection using active polarization settings."""
        targets = self._selected_or_current_datasets()
        if not targets:
            return

        if not self._overlay_enabled() and len(targets) > 1:
            chosen = self._select_non_overlay_target(targets)
            if chosen is None:
                return
            targets = [chosen]

        if len(targets) == 1:
            self._current_dataset = targets[0]

        active_axis = None
        if hasattr(self._plot_panel, "get_current_polarization_axis"):
            active_axis = self._normalize_vector_axis(self._plot_panel.get_current_polarization_axis())

        if active_axis == "ALL" and hasattr(self._plot_panel, "plot_vector_subplots"):
            axis_datasets = self._build_vector_axis_datasets(targets)
            if all(axis_datasets.get(axis) for axis in ("P_x", "P_y", "P_z")):
                self._plot_panel.plot_vector_subplots(axis_datasets)
                return

        if len(targets) > 1:
            self._plot_panel.plot_datasets(targets)
            return
        self._plot_panel.plot_dataset(targets[0])

    def _update_fit_block_state(self) -> None:
        """Disable ambiguous fitting workflows when vector ALL mode is active."""
        if not hasattr(self._fit_panel, "set_fit_blocked"):
            return

        active_axis = None
        if hasattr(self._plot_panel, "get_current_polarization_axis"):
            active_axis = self._normalize_vector_axis(self._plot_panel.get_current_polarization_axis())

        blocked = active_axis == "ALL"
        reason = (
            "Vector All mode is ambiguous for fitting. "
            "Select x, y, or z before running a fit."
        )
        self._fit_panel.set_fit_blocked(blocked, reason if blocked else "")

    def _on_plot_polarization_axis_changed(self, axis_text: str) -> None:
        """Recompute displayed datasets using the selected vector polarization axis."""
        axis = self._normalize_vector_axis(axis_text)
        if axis is None:
            return

        if axis == "ALL":
            self._render_current_selection_plot()
            self._refresh_vector_axis_selector()
            self._update_fit_block_state()
            self._log_panel.log("Set vector polarization axis to ALL.")
            return

        selected = list(self._data_browser.get_selected_datasets())
        targets = selected if selected else ([self._current_dataset] if self._current_dataset else [])

        updated = 0
        for dataset in targets:
            if dataset is None:
                continue
            run = getattr(dataset, "run", None)
            grouping = getattr(run, "grouping", None)
            if not isinstance(grouping, dict):
                continue
            payload = self._extract_grouping_overrides(dataset)
            if not isinstance(payload, dict):
                continue

            pairs = self._vector_axis_pairs_for_grouping(
                payload.get("groups", {}),
                payload.get("group_names"),
            )
            if axis not in pairs:
                continue
            fwd_gid, bwd_gid = pairs[axis]
            payload["forward_group"] = fwd_gid
            payload["backward_group"] = bwd_gid
            payload["vector_axis"] = axis

            applied, _dt_applied = self._apply_grouping_settings_to_dataset(dataset, payload)
            if applied:
                updated += 1

        if updated <= 0:
            return

        self._data_browser._rebuild_table()
        self._render_current_selection_plot()
        self._refresh_vector_axis_selector()
        self._update_fit_block_state()
        self._log_panel.log(f"Set vector polarization axis to {axis} for {updated} dataset(s).")

    # ── slots ──────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        """Prompt the user to select one or more data files and load them."""
        from asymmetry.core.io.base import LoaderRegistry

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open μSR data files",
            self._last_open_dir,
            LoaderRegistry.file_dialog_filter(),
        )
        if paths:
            selected_dir = os.path.dirname(paths[0])
            if selected_dir:
                self._last_open_dir = selected_dir
                self._settings.setValue("io/last_open_dir", selected_dir)
            self._load_files(paths)

    def _on_export_current_plot(self) -> None:
        """Export the current main plot view to GLE/PDF/EPS."""
        self._plot_panel.export_current_plot()

    def _on_export_logbook(self) -> None:
        """Export the data-browser logbook table to TSV or RTF."""
        if not self._data_browser.get_all_datasets():
            self.statusBar().showMessage("No datasets available to export")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Logbook",
            default_export_path(self._default_logbook_export_name()),
            "Tab-separated values (*.tsv);;Rich Text Format (*.rtf);;All files (*)",
        )
        if not path:
            return

        path_obj = Path(path)
        selected_filter_lower = (selected_filter or "").lower()
        filter_wants_rtf = "rich text format" in selected_filter_lower
        filter_wants_tsv = "tab-separated values" in selected_filter_lower

        if not path_obj.suffix:
            if filter_wants_rtf:
                path_obj = path_obj.with_suffix(".rtf")
            elif filter_wants_tsv:
                path_obj = path_obj.with_suffix(".tsv")
            else:
                path_obj = path_obj.with_suffix(".tsv")
        path = str(path_obj)

        is_rtf = path_obj.suffix.lower() == ".rtf"

        try:
            if is_rtf:
                exported_count = self._data_browser.export_logbook_rtf(path)
                fmt = "RTF"
            else:
                exported_count = self._data_browser.export_logbook_tsv(path)
                fmt = "TSV"
            remember_export_path(path)
            self._log_panel.log(f"Exported logbook ({fmt}) to {path} ({exported_count} datasets).")
            self.statusBar().showMessage(f"Logbook exported: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export logbook:\n{e}")
            self._log_panel.log(f"ERROR exporting logbook: {e}")

    def _default_logbook_export_name(self) -> str:
        """Return default logbook export filename, preferring project name."""
        if self._current_project_path:
            stem = Path(self._current_project_path).stem.strip()
            safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_")
            if safe_stem:
                return f"{safe_stem}_logbook.tsv"
        return "logbook.tsv"

    def _load_files(self, paths: list[str]) -> None:
        """Load multiple data files."""
        successful = 0
        failed = 0
        last_dataset = None
        apply_comment_field_to_all = False
        overwrite_existing_to_all = False
        auto_grouping_payload = self._get_project_grouping_template()
        auto_grouping_attempts = 0
        auto_grouping_applied = 0

        for path in paths:
            should_overwrite_existing = False
            if self._is_source_file_loaded(path):
                if not overwrite_existing_to_all:
                    overwrite_choice = self._prompt_overwrite_existing_dataset(path)
                    if overwrite_choice == QMessageBox.StandardButton.No:
                        self._log_panel.log(f"Skipped already-loaded file: {path}")
                        continue
                    if overwrite_choice == QMessageBox.StandardButton.YesToAll:
                        overwrite_existing_to_all = True
                should_overwrite_existing = True

            try:
                loaded = self._load_file(path)
                if loaded is None:
                    continue

                datasets = loaded if isinstance(loaded, list) else [loaded]
                if not datasets:
                    continue

                if should_overwrite_existing:
                    removed = self._remove_datasets_for_source_file(path)
                    self._log_panel.log(
                        f"Updated file {path} (replaced {removed} existing dataset(s))."
                    )

                for dataset in datasets:
                    # Offer to apply field extracted from comment when available.
                    apply_choice = self._maybe_apply_comment_field(
                        dataset,
                        path,
                        apply_comment_field_to_all,
                    )
                    if apply_choice == "cancel":
                        self._log_panel.log("File loading cancelled by user")
                        break
                    if apply_choice == "yes_to_all":
                        apply_comment_field_to_all = True

                    if auto_grouping_payload is not None:
                        auto_grouping_attempts += 1
                        grouping_payload = dict(auto_grouping_payload)
                        active_axis = None
                        if hasattr(self._plot_panel, "get_current_polarization_axis"):
                            active_axis = self._normalize_vector_axis(
                                self._plot_panel.get_current_polarization_axis()
                            )
                        if active_axis in {"P_x", "P_y", "P_z"}:
                            grouping_payload["vector_axis"] = active_axis

                        applied, _ = self._apply_grouping_settings_to_dataset(dataset, grouping_payload)
                        if applied:
                            auto_grouping_applied += 1

                    self._data_browser.add_dataset(dataset)
                    if dataset:
                        last_dataset = dataset
                        successful += 1
                else:
                    self._log_panel.log(f"Loaded {path}")
                    continue

                break
            except Exception as e:
                self._log_panel.log(f"ERROR loading {path}: {e}")
                failed += 1

        # Plot the last successfully loaded dataset
        if last_dataset:
            self._plot_panel.plot_dataset(last_dataset)

        # Update selected datasets for global fitting
        self._update_selected_datasets()

        # Update status message
        if successful > 0:
            msg = f"Loaded {successful} file(s)"
            if failed > 0:
                msg += f", {failed} failed"
            self._log_panel.log(msg)
            self.statusBar().showMessage(msg)
        elif failed > 0:
            self.statusBar().showMessage(f"Failed to load {failed} file(s)")

        if auto_grouping_payload is not None and auto_grouping_attempts > 0:
            skipped = auto_grouping_attempts - auto_grouping_applied
            self._log_panel.log(
                "Auto-applied existing project grouping to "
                f"{auto_grouping_applied} dataset(s); skipped {skipped}."
            )

    def _dataset_source_file(self, dataset) -> str:
        """Return the dataset source file path, if available."""
        source_file = ""
        run = getattr(dataset, "run", None)
        if run is not None:
            source_file = str(getattr(run, "source_file", "") or "")
        if not source_file:
            metadata = getattr(dataset, "metadata", {})
            if isinstance(metadata, dict):
                source_file = str(metadata.get("source_file", "") or "")
        return source_file

    def _is_source_file_loaded(self, path: str) -> bool:
        """Return True when a source file is already represented in the browser."""
        target = _normalise_source_path(path)
        for dataset in self._data_browser.get_all_datasets():
            source_file = self._dataset_source_file(dataset)
            if source_file and _normalise_source_path(source_file) == target:
                return True
        return False

    def _remove_datasets_for_source_file(self, path: str) -> int:
        """Remove all datasets that originate from *path* and return count."""
        target = _normalise_source_path(path)
        run_numbers_to_remove: list[int] = []
        for dataset in self._data_browser.get_all_datasets():
            source_file = self._dataset_source_file(dataset)
            if source_file and _normalise_source_path(source_file) == target:
                run_numbers_to_remove.append(int(dataset.run_number))

        for run_number in run_numbers_to_remove:
            self._data_browser._remove_run_number(run_number)

        if run_numbers_to_remove:
            self._data_browser._rebuild_table()

        return len(run_numbers_to_remove)

    def _prompt_overwrite_existing_dataset(self, path: str) -> QMessageBox.StandardButton:
        """Ask whether a duplicate file should overwrite currently loaded datasets."""
        run_numbers = []
        target = _normalise_source_path(path)
        for dataset in self._data_browser.get_all_datasets():
            source_file = self._dataset_source_file(dataset)
            if source_file and _normalise_source_path(source_file) == target:
                run_numbers.append(int(dataset.run_number))

        run_numbers_text = "unknown"
        if run_numbers:
            run_numbers_text = ", ".join(str(rn) for rn in sorted(set(run_numbers)))

        answer = QMessageBox.question(
            self,
            "Data File Already Loaded",
            "This data file is already in the Data Browser.\n\n"
            f"Run number(s): {run_numbers_text}\n\n"
            "Do you want to update this dataset and overwrite the existing one?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.YesToAll,
            QMessageBox.StandardButton.No,
        )
        if answer in {
            QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.YesToAll,
        }:
            return answer
        return QMessageBox.StandardButton.No

    def _get_project_grouping_template(self) -> dict | None:
        """Return grouping payload from the highest-run dataset in the browser.

        When multiple datasets have different grouping definitions, this chooses
        the payload from the largest run number currently loaded in the data
        browser so newly added files inherit the latest in-browser grouping.
        """
        browser_datasets: list = []
        if hasattr(self._data_browser, "get_all_datasets"):
            browser_datasets = list(self._data_browser.get_all_datasets())

        ranked: list[tuple[float, object]] = []
        for dataset in browser_datasets:
            try:
                rank = float(int(dataset.run_number))
            except (TypeError, ValueError):
                rank = float("-inf")
            ranked.append((rank, dataset))

        if not ranked and self._current_dataset is not None:
            ranked.append((float("-inf"), self._current_dataset))

        ranked.sort(key=lambda item: item[0], reverse=True)
        for _, dataset in ranked:
            payload = self._extract_grouping_overrides(dataset)
            if isinstance(payload, dict) and isinstance(payload.get("groups"), dict):
                groups = payload.get("groups", {})
                if groups:
                    return payload
        return None

    def _load_file(self, path: str):
        from asymmetry.core.io import load

        dataset = load(path)
        return dataset

    def _on_get_info_requested(self, run_number: int) -> None:
        """Open run-information dialog for a selected dataset row."""
        dataset = self._data_browser.get_dataset(run_number)
        if dataset is None:
            return
        dialog = RunInfoDialog(
            dataset,
            self,
            included_fields=set(self._data_browser.get_extra_columns()),
        )
        dialog.set_browser_field_inclusion_requested.connect(self._on_run_info_field_inclusion_changed)
        dialog.exec()

    def _on_run_info_field_inclusion_changed(self, field_key: str, include: bool) -> None:
        """Apply include/exclude requests from the Run Info dialog."""
        if include:
            self._data_browser.add_extra_column(field_key)
        else:
            self._data_browser.remove_extra_column(field_key)

    def _on_grouping_requested(self, run_number: int) -> None:
        """Open shared grouping dialog focused on a selected run."""
        selected_run_numbers = [
            int(ds.run_number)
            for ds in self._data_browser.get_selected_datasets()
        ]
        if int(run_number) not in selected_run_numbers:
            selected_run_numbers = [int(run_number)]
        self._open_shared_grouping_dialog(
            selected_run_number=run_number,
            selected_run_numbers=selected_run_numbers,
        )

    def _on_grouping_current(self) -> None:
        """Open shared grouping dialog for all datasets in the active project."""
        selected_run = None if self._current_dataset is None else int(self._current_dataset.run_number)
        selected_run_numbers = [
            int(ds.run_number)
            for ds in self._data_browser.get_selected_datasets()
        ]
        self._open_shared_grouping_dialog(
            selected_run_number=selected_run,
            selected_run_numbers=selected_run_numbers if selected_run_numbers else None,
        )

    def _open_shared_grouping_dialog(
        self,
        *,
        selected_run_number: int | None = None,
        selected_run_numbers: list[int] | None = None,
    ) -> None:
        """Show shared grouping dialog and apply settings to selected datasets."""
        all_datasets = (
            self._data_browser.get_all_datasets()
            if hasattr(self._data_browser, "get_all_datasets")
            else []
        )
        (
            dialog_datasets,
            reference_dataset,
            dialog_selected_run_number,
            dialog_selected_run_numbers,
            combined_target_run_number,
        ) = self._resolve_grouping_dialog_context(
            all_datasets=all_datasets,
            selected_run_number=selected_run_number,
            selected_run_numbers=selected_run_numbers,
        )

        use_wim_dialog = False
        if reference_dataset is not None and reference_dataset.run is not None:
            source_file = str(getattr(reference_dataset.run, "source_file", "") or "")
            use_wim_dialog = source_file.lower().endswith(".wim")

        if use_wim_dialog:
            wim_dialog_datasets = [
                ds
                for ds in dialog_datasets
                if ds.run is not None
                and str(getattr(ds.run, "source_file", "") or "").lower().endswith(".wim")
            ]
            filtered_selected = None
            if dialog_selected_run_numbers is not None:
                selected_set = {int(v) for v in dialog_selected_run_numbers}
                filtered_selected = [
                    int(ds.run_number)
                    for ds in wim_dialog_datasets
                    if int(ds.run_number) in selected_set
                ]
            dialog = WimGroupingDialog(
                wim_dialog_datasets,
                selected_run_number=dialog_selected_run_number,
                selected_run_numbers=filtered_selected,
                parent=self,
            )
        else:
            dialog = GroupingDialog(
                dialog_datasets,
                selected_run_number=dialog_selected_run_number,
                selected_run_numbers=dialog_selected_run_numbers,
                parent=self,
            )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        grouping_result = dialog.get_grouping_result()
        if not grouping_result:
            return

        run_numbers = {int(v) for v in grouping_result.get("run_numbers", [])}
        alpha = float(grouping_result.get("alpha", 1.0))
        use_deadtime = bool(grouping_result.get("deadtime_correction", False))
        use_background = bool(grouping_result.get("background_correction", False))

        updated = 0
        skipped = 0
        deadtime_applied = 0
        deadtime_missing = 0
        background_applied = 0
        background_missing = 0
        first_updated_dataset = None

        for dataset in dialog_datasets:
            if run_numbers and int(dataset.run_number) not in run_numbers:
                continue
            applied, dt_applied = self._apply_grouping_settings_to_dataset(dataset, grouping_result)
            if not applied:
                skipped += 1
                continue
            if use_deadtime and not dt_applied:
                deadtime_missing += 1
            if dt_applied:
                deadtime_applied += 1
            if use_background:
                grouping = dataset.run.grouping if dataset.run is not None else {}
                if isinstance(grouping, dict) and grouping.get("background_method") in {
                    "estimated",
                    "fixed",
                }:
                    background_applied += 1
                else:
                    background_missing += 1

            if dataset is self._current_dataset:
                self._fit_panel.set_dataset(self._get_fit_dataset(dataset))
            if first_updated_dataset is None:
                first_updated_dataset = dataset
            updated += 1

        rebuilt_combined_dataset = None
        if updated > 0 and combined_target_run_number is not None and hasattr(
            self._data_browser,
            "rebuild_combined_dataset",
        ):
            rebuilt_combined_dataset = self._data_browser.rebuild_combined_dataset(
                combined_target_run_number
            )
            if rebuilt_combined_dataset is not None:
                first_updated_dataset = rebuilt_combined_dataset
                if (
                    self._current_dataset is not None
                    and int(self._current_dataset.run_number) == int(combined_target_run_number)
                ):
                    self._current_dataset = rebuilt_combined_dataset
                    self._fit_panel.set_dataset(self._get_fit_dataset(rebuilt_combined_dataset))

        if updated > 0:
            self._data_browser._rebuild_table()
            self._render_current_selection_plot()
            self._refresh_vector_axis_selector()

        deadtime_msg = "off"
        if use_deadtime:
            deadtime_msg = f"on (applied={deadtime_applied}, missing={deadtime_missing})"
        background_msg = "off"
        if use_background:
            background_msg = f"on (applied={background_applied}, missing={background_missing})"

        self._log_panel.log(
            f"Applied grouping to {updated} dataset(s); skipped {skipped}. "
            f"F={grouping_result['forward_group']}, "
            f"B={grouping_result['backward_group']}, alpha={alpha:.6g}, "
            f"deadtime={deadtime_msg}, background={background_msg}"
        )

    def _resolve_grouping_dialog_context(
        self,
        *,
        all_datasets: list,
        selected_run_number: int | None,
        selected_run_numbers: list[int] | None,
    ) -> tuple[list, object | None, int | None, list[int] | None, int | None]:
        """Return grouping dialog datasets plus any combined-row target context."""
        reference_dataset = None
        if selected_run_number is not None:
            reference_dataset = next(
                (ds for ds in all_datasets if int(ds.run_number) == int(selected_run_number)),
                None,
            )
        if reference_dataset is None:
            reference_dataset = self._current_dataset

        dialog_datasets = list(all_datasets)
        dialog_selected_run_number = selected_run_number
        dialog_selected_run_numbers = list(selected_run_numbers) if selected_run_numbers is not None else None
        combined_target_run_number = None

        if (
            reference_dataset is not None
            and hasattr(self._data_browser, "is_combined_dataset")
            and self._data_browser.is_combined_dataset(int(reference_dataset.run_number))
            and hasattr(self._data_browser, "get_combined_source_datasets")
        ):
            combined_target_run_number = int(reference_dataset.run_number)
            source_datasets = self._data_browser.get_combined_source_datasets(
                combined_target_run_number
            )
            if source_datasets:
                dialog_datasets = source_datasets
                reference_dataset = source_datasets[0]
                dialog_selected_run_number = int(reference_dataset.run_number)
                dialog_selected_run_numbers = [int(ds.run_number) for ds in source_datasets]

        return (
            dialog_datasets,
            reference_dataset,
            dialog_selected_run_number,
            dialog_selected_run_numbers,
            combined_target_run_number,
        )

    def _extract_grouping_overrides(self, dataset) -> dict | None:
        """Return grouping settings that should persist in project files."""
        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if not isinstance(grouping, dict):
            return None

        groups_raw = grouping.get("groups")
        if not isinstance(groups_raw, dict):
            return None

        groups: dict[int, list[object]] = {}
        for key, values in groups_raw.items():
            try:
                gid = int(key)
            except (TypeError, ValueError):
                continue
            entries = self._normalize_group_entries(values)
            if entries:
                groups[gid] = entries

        if not groups:
            return None

        t0_default = 0
        if run is not None and getattr(run, "histograms", None):
            try:
                t0_default = int(run.histograms[0].t0_bin)
            except (TypeError, ValueError, IndexError):
                t0_default = 0

        try:
            t0_bin = int(grouping.get("t0_bin", t0_default))
        except (TypeError, ValueError):
            t0_bin = t0_default

        raw_t_good = grouping.get("t_good_offset")
        if raw_t_good is None:
            try:
                raw_t_good = int(grouping.get("first_good_bin", t0_bin)) - t0_bin
            except (TypeError, ValueError):
                raw_t_good = 0
        try:
            t_good_offset = max(0, int(raw_t_good))
        except (TypeError, ValueError):
            t_good_offset = 0

        first_good_bin = max(0, t0_bin + t_good_offset)
        try:
            last_good_bin = int(grouping.get("last_good_bin", 0))
        except (TypeError, ValueError):
            last_good_bin = 0
        try:
            bin_index_base = 1 if int(grouping.get("bin_index_base", 0)) == 1 else 0
        except (TypeError, ValueError):
            bin_index_base = 0

        payload = {
            "groups": groups,
            "forward_group": int(grouping.get("forward_group", 1)),
            "backward_group": int(grouping.get("backward_group", 2)),
            "alpha": float(grouping.get("alpha", 1.0)),
            "t0_bin": t0_bin,
            "t_good_offset": t_good_offset,
            "first_good_bin": first_good_bin,
            "last_good_bin": last_good_bin,
            "bin_index_base": bin_index_base,
            "bunching_factor": int(grouping.get("bunching_factor", 1)),
            "source_bunching_factor": int(
                grouping.get("source_bunching_factor", grouping.get("bunching_factor", 1))
            ),
            "deadtime_correction": bool(grouping.get("deadtime_correction", False)),
            "background_correction": bool(grouping.get("background_correction", False)),
        }

        for axis in ("P_x", "P_y", "P_z"):
            key = self._vector_alpha_key(axis)
            legacy_key = self._legacy_vector_alpha_key(axis)
            if key is None:
                continue
            try:
                if key in grouping:
                    payload[key] = float(grouping.get(key))
                elif legacy_key in grouping:
                    payload[key] = float(grouping.get(legacy_key))
            except (TypeError, ValueError):
                continue

        vector_axis = self._normalize_vector_axis(grouping.get("vector_axis"))
        if vector_axis is not None:
            payload["vector_axis"] = vector_axis

        group_names_raw = grouping.get("group_names")
        if isinstance(group_names_raw, dict) and group_names_raw:
            payload["group_names"] = {
                int(k): str(v) for k, v in group_names_raw.items()
            }

        grouping_preset = grouping.get("grouping_preset")
        if grouping_preset:
            payload["grouping_preset"] = str(grouping_preset)

        instrument_name = grouping.get("instrument")
        if instrument_name:
            payload["instrument"] = str(instrument_name)

        if "dead_time_us" in grouping and isinstance(grouping.get("dead_time_us"), list):
            payload["dead_time_us"] = list(grouping.get("dead_time_us", []))
        if "good_frames" in grouping:
            payload["good_frames"] = grouping.get("good_frames")
        for key in (
            "background_ranges",
            "background_range",
            "background_forward_range",
            "background_backward_range",
            "background_fixed_values",
            "background_values",
            "background_method",
            "background_fix",
            "bkg_fix",
        ):
            if key in grouping:
                payload[key] = grouping.get(key)
        for key in (
            "detector_t0_bins",
            "detector_first_good_bins",
            "detector_last_good_bins",
            "histogram_labels",
        ):
            if isinstance(grouping.get(key), list):
                payload[key] = list(grouping.get(key, []))
        return payload

    def _normalize_group_entries(self, values) -> list[object]:
        """Return grouping entries preserving detector/t0 pairs when present."""
        if not isinstance(values, list):
            return []

        entries: list[object] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                parsed: list[int] = []
                for item in list(value)[:2]:
                    try:
                        parsed.append(int(item))
                    except (TypeError, ValueError):
                        parsed = []
                        break
                if not parsed:
                    continue
                entries.append(parsed[0] if len(parsed) == 1 else tuple(parsed))
                continue

            try:
                entries.append(int(value))
            except (TypeError, ValueError):
                continue

        return entries

    def _group_detector_indices(self, values) -> list[int]:
        """Return zero-based detector indices for one grouping entry list."""
        indices: list[int] = []
        for value in self._normalize_group_entries(values):
            detector = value[0] if isinstance(value, tuple) else value
            indices.append(max(0, int(detector) - 1))
        return indices

    def _grouping_source_arrays(
        self,
        dataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return immutable source arrays for regrouping datasets without histograms.

        ``.wim`` datasets only carry processed asymmetry arrays, so repeated
        bunching edits must always start from the original loaded data rather
        than the most recently rebinned view.
        """
        cached = getattr(dataset, "_grouping_source_arrays_cache", None)
        if isinstance(cached, tuple) and len(cached) == 3:
            return cached

        cached = (
            np.asarray(dataset.time, dtype=float).copy(),
            np.asarray(dataset.asymmetry, dtype=float).copy(),
            np.asarray(dataset.error, dtype=float).copy(),
        )
        setattr(dataset, "_grouping_source_arrays_cache", cached)
        return cached

    def _apply_grouping_settings_to_dataset(self, dataset, grouping_result: dict) -> tuple[bool, bool]:
        """Apply grouping settings to one dataset and recompute asymmetry.

        Returns
        -------
        tuple[bool, bool]
            ``(applied, deadtime_applied)``.
        """
        run = dataset.run
        if run is None:
            return False, False
        existing_grouping = run.grouping if isinstance(run.grouping, dict) else {}

        groups_raw = grouping_result.get("groups", {})
        if not isinstance(groups_raw, dict):
            groups_raw = {}

        groups: dict[int, list[object]] = {}
        for key, values in groups_raw.items():
            try:
                gid = int(key)
            except (TypeError, ValueError):
                continue
            entries = self._normalize_group_entries(values)
            if entries:
                groups[gid] = entries

        try:
            forward_gid = int(grouping_result.get("forward_group", 1))
            backward_gid = int(grouping_result.get("backward_group", 2))
        except (TypeError, ValueError):
            return False, False

        group_names_for_axis = grouping_result.get("group_names")
        if not isinstance(group_names_for_axis, dict) and isinstance(existing_grouping, dict):
            group_names_for_axis = existing_grouping.get("group_names")
        axis_pairs = self._vector_axis_pairs_for_grouping(groups, group_names_for_axis)
        vector_axis = self._normalize_vector_axis(
            grouping_result.get("vector_axis", existing_grouping.get("vector_axis"))
        )
        if axis_pairs:
            if vector_axis not in axis_pairs:
                vector_axis = "P_z"
            forward_gid, backward_gid = axis_pairs[vector_axis]

        vector_alphas = self._resolve_vector_alpha_values(grouping_result, existing_grouping)

        forward_idx = self._group_detector_indices(groups.get(forward_gid, []))
        backward_idx = self._group_detector_indices(groups.get(backward_gid, []))

        if run.histograms:
            max_bin = len(run.histograms[0].counts) - 1
            t0_default = int(run.histograms[0].t0_bin)
        else:
            source_time, source_asymmetry, source_error = self._grouping_source_arrays(dataset)
            max_bin = max(0, len(source_time) - 1)
            t0_default = int(existing_grouping.get("t0_bin", 0))

        try:
            t0_bin = int(grouping_result.get("t0_bin", existing_grouping.get("t0_bin", t0_default)))
        except (TypeError, ValueError):
            t0_bin = t0_default
        t0_bin = max(0, min(max_bin, t0_bin))

        raw_t_good_offset = grouping_result.get("t_good_offset")
        if raw_t_good_offset is None:
            try:
                raw_first_good = int(
                    grouping_result.get(
                        "first_good_bin",
                        existing_grouping.get("first_good_bin", t0_bin),
                    )
                )
                raw_t_good_offset = raw_first_good - t0_bin
            except (TypeError, ValueError):
                raw_t_good_offset = 0
        try:
            t_good_offset = int(raw_t_good_offset)
        except (TypeError, ValueError):
            t_good_offset = 0
        t_good_offset = max(0, min(max_bin - t0_bin, t_good_offset))
        first_good = t0_bin + t_good_offset

        try:
            bin_index_base = 1 if int(grouping_result.get("bin_index_base", existing_grouping.get("bin_index_base", 0))) == 1 else 0
        except (TypeError, ValueError):
            bin_index_base = 0

        try:
            last_good_raw = int(grouping_result.get("last_good_bin", max_bin))
        except (TypeError, ValueError):
            last_good_raw = max_bin
        last_good = max(first_good, min(max_bin, last_good_raw))
        if axis_pairs and vector_axis in vector_alphas:
            alpha = float(vector_alphas[vector_axis])
        else:
            try:
                alpha = float(grouping_result.get("alpha", existing_grouping.get("alpha", 1.0)))
            except (TypeError, ValueError):
                alpha = 1.0
        bunch_factor = int(grouping_result.get("bunching_factor", 1))
        period_mode = str(grouping_result.get("period_mode", PeriodMode.RED))
        enforce_source_bunching = bool(grouping_result.get("enforce_source_bunching", False))
        if "enforce_source_bunching" not in grouping_result:
            source_file = str(getattr(run, "source_file", "") or "")
            enforce_source_bunching = source_file.lower().endswith(".wim")

        source_bunch_factor = 1
        additional_bunch_factor = max(1, bunch_factor)
        if enforce_source_bunching:
            source_bunch_factor = int(
                grouping_result.get(
                    "source_bunching_factor",
                    run.grouping.get("source_bunching_factor", run.grouping.get("bunching_factor", 1)),
                )
            )
            source_bunch_factor = max(1, source_bunch_factor)
            if bunch_factor < source_bunch_factor:
                return False, False
            if bunch_factor % source_bunch_factor != 0:
                return False, False
            additional_bunch_factor = max(1, bunch_factor // source_bunch_factor)
        use_deadtime = bool(grouping_result.get("deadtime_correction", False))
        use_background = bool(
            grouping_result.get("background_correction", False)
            and self._dataset_supports_background_correction(dataset)
        )

        if not run.histograms:
            source_last_bin = len(source_time) - 1
            source_file = str(getattr(run, "source_file", "") or "")
            # WiMDA-exported asymmetry tables have already had their early-bin
            # handling applied upstream, so the stored first-good-bin value is
            # informational only and must not trim the imported points again.
            lo = 0 if source_file.lower().endswith(".wim") else max(0, first_good)
            hi = min(source_last_bin, last_good)
            if lo <= hi:
                time_out = source_time[lo : hi + 1].copy()
                asym_out = source_asymmetry[lo : hi + 1].copy()
                err_out = source_error[lo : hi + 1].copy()
                if additional_bunch_factor > 1:
                    time_out, asym_out, err_out = rebin(
                        time_out,
                        asym_out,
                        err_out,
                        additional_bunch_factor,
                    )
                dataset.time = time_out
                dataset.asymmetry = asym_out
                dataset.error = err_out

            if not isinstance(run.grouping, dict):
                run.grouping = {}
            if groups:
                run.grouping["groups"] = groups
                run.grouping["forward_group"] = forward_gid
                run.grouping["backward_group"] = backward_gid
            run.grouping["alpha"] = float(alpha if alpha > 0 else 1.0)
            if axis_pairs:
                run.grouping["alpha_x"] = float(vector_alphas.get("P_x", run.grouping["alpha"]))
                run.grouping["alpha_y"] = float(vector_alphas.get("P_y", run.grouping["alpha"]))
                run.grouping["alpha_z"] = float(vector_alphas.get("P_z", run.grouping["alpha"]))
            run.grouping["t0_bin"] = t0_bin
            run.grouping["t_good_offset"] = t_good_offset
            run.grouping["first_good_bin"] = first_good
            run.grouping["last_good_bin"] = last_good
            run.grouping["bin_index_base"] = bin_index_base
            run.grouping["bunching_factor"] = bunch_factor
            run.grouping["deadtime_correction"] = False
            run.grouping["background_correction"] = use_background
            if not use_background:
                run.grouping.pop("background_method", None)
                run.grouping.pop("background_values", None)
            run.grouping["period_mode"] = period_mode
            if enforce_source_bunching:
                run.grouping["source_bunching_factor"] = source_bunch_factor
            group_names = grouping_result.get("group_names")
            if isinstance(group_names, dict) and group_names:
                run.grouping["group_names"] = {int(k): str(v) for k, v in group_names.items()}
            if vector_axis and axis_pairs:
                run.grouping["vector_axis"] = vector_axis
            preset_name = grouping_result.get("grouping_preset")
            if preset_name:
                run.grouping["grouping_preset"] = str(preset_name)
            else:
                run.grouping.pop("grouping_preset", None)
            instrument_name = grouping_result.get("instrument")
            if instrument_name:
                run.grouping["instrument"] = str(instrument_name)
            else:
                run.grouping.pop("instrument", None)
            return True, False

        if not groups:
            return False, False
        if not forward_idx or not backward_idx:
            return False, False
        if max(forward_idx, default=-1) >= len(run.histograms):
            return False, False
        if max(backward_idx, default=-1) >= len(run.histograms):
            return False, False

        if not isinstance(run.grouping, dict):
            run.grouping = {}

        # Keep histogram time-zero consistent with grouping metadata. PSI files
        # can have per-detector t0 values, so preserve relative detector
        # offsets when the user edits the common t0 control.
        if isinstance(grouping_result.get("detector_t0_bins"), list):
            run.grouping["detector_t0_bins"] = list(
                grouping_result.get("detector_t0_bins", [])
            )
        if isinstance(grouping_result.get("detector_first_good_bins"), list):
            run.grouping["detector_first_good_bins"] = list(
                grouping_result.get("detector_first_good_bins", [])
            )
        if isinstance(grouping_result.get("detector_last_good_bins"), list):
            run.grouping["detector_last_good_bins"] = list(
                grouping_result.get("detector_last_good_bins", [])
            )
        if isinstance(grouping_result.get("histogram_labels"), list):
            run.grouping["histogram_labels"] = list(
                grouping_result.get("histogram_labels", [])
            )

        detector_t0_bins = run.grouping.get("detector_t0_bins")
        if (
            isinstance(detector_t0_bins, list)
            and len(detector_t0_bins) == len(run.histograms)
        ):
            try:
                previous_common_t0 = int(existing_grouping.get("t0_bin", t0_default))
            except (TypeError, ValueError):
                previous_common_t0 = t0_default
            delta = int(t0_bin) - previous_common_t0
            run.histograms = [
                Histogram(
                    counts=hist.counts,
                    bin_width=hist.bin_width,
                    t0_bin=max(0, int(detector_t0_bins[i]) + delta),
                    good_bin_start=hist.good_bin_start,
                    good_bin_end=hist.good_bin_end,
                )
                for i, hist in enumerate(run.histograms)
            ]
        elif run.histograms and any(int(hist.t0_bin) != t0_bin for hist in run.histograms):
            run.histograms = [
                Histogram(
                    counts=hist.counts,
                    bin_width=hist.bin_width,
                    t0_bin=t0_bin,
                    good_bin_start=hist.good_bin_start,
                    good_bin_end=hist.good_bin_end,
                )
                for hist in run.histograms
            ]

        if isinstance(grouping_result.get("dead_time_us"), list):
            run.grouping["dead_time_us"] = list(grouping_result.get("dead_time_us", []))
        if "good_frames" in grouping_result:
            run.grouping["good_frames"] = grouping_result.get("good_frames")
        for key in (
            "background_ranges",
            "background_range",
            "background_forward_range",
            "background_backward_range",
            "background_fixed_values",
            "background_fix",
            "bkg_fix",
        ):
            if key in grouping_result:
                run.grouping[key] = grouping_result.get(key)

        use_deadtime = bool(use_deadtime and has_file_deadtime(run.grouping, len(run.histograms)))
        if not use_deadtime:
            run.grouping.pop("deadtime_method", None)

        working_histograms, dt_applied = self._prepare_grouping_histograms(
            run.histograms,
            run.grouping,
            use_deadtime,
        )

        common_t0 = common_t0_for_groups(working_histograms, forward_idx, backward_idx)
        forward = apply_grouping_aligned(
            working_histograms,
            forward_idx,
            common_t0_bin=common_t0,
        )
        backward = apply_grouping_aligned(
            working_histograms,
            backward_idx,
            common_t0_bin=common_t0,
        )
        n_grouped = min(len(forward), len(backward))
        forward = forward[:n_grouped]
        backward = backward[:n_grouped]

        if use_background:
            bin_width = float(working_histograms[0].bin_width) if working_histograms else 1.0
            facility = str(
                run.metadata.get(
                    "facility",
                    dataset.metadata.get("facility", dataset.metadata.get("instrument", "")),
                )
            )
            bkg_result = apply_grouped_background_correction(
                forward,
                backward,
                grouping=run.grouping,
                t0_bin=common_t0,
                bin_width_us=bin_width,
                facility=facility,
            )
            forward = bkg_result.forward
            backward = bkg_result.backward
            if bkg_result.applied:
                run.grouping["background_method"] = bkg_result.method
                if bkg_result.values is not None:
                    run.grouping["background_values"] = [
                        float(bkg_result.values[0]),
                        float(bkg_result.values[1]),
                    ]
                if bkg_result.ranges is not None:
                    run.grouping["background_ranges"] = [
                        [int(v) for v in bkg_result.ranges[0]],
                        [int(v) for v in bkg_result.ranges[1]],
                    ]
            else:
                run.grouping["background_method"] = bkg_result.method
                run.grouping.pop("background_values", None)
        else:
            run.grouping.pop("background_method", None)
            run.grouping.pop("background_values", None)

        run_alpha = alpha if alpha > 0 else 1.0

        asymmetry, error = compute_asymmetry(forward, backward, alpha=run_alpha)
        asymmetry = asymmetry * 100.0
        error = error * 100.0
        if run.histograms:
            bin_width = float(run.histograms[0].bin_width)
        else:
            bin_width = 1.0
        time_axis = (
            np.arange(len(asymmetry), dtype=np.float64) - float(common_t0)
        ) * bin_width

        lo = max(0, first_good)
        hi = min(len(asymmetry) - 1, last_good)
        if lo <= hi:
            time_out = time_axis[lo : hi + 1].copy()
            asym_out = asymmetry[lo : hi + 1].copy()
            err_out = error[lo : hi + 1].copy()
            # Datasets can already be bunched in-source (e.g. .wim). Apply
            # only the extra bunching requested relative to source binning.
            if additional_bunch_factor > 1:
                time_out, asym_out, err_out = rebin(
                    time_out,
                    asym_out,
                    err_out,
                    additional_bunch_factor,
                )
            dataset.time = time_out
            dataset.asymmetry = asym_out
            dataset.error = err_out

        run.grouping.update(
            {
                "groups": groups,
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "alpha": float(run_alpha),
                "t0_bin": t0_bin,
                "t_good_offset": t_good_offset,
                "first_good_bin": first_good,
                "last_good_bin": last_good,
                "bin_index_base": bin_index_base,
                "bunching_factor": bunch_factor,
                "deadtime_correction": use_deadtime,
                "background_correction": use_background,
                "period_mode": period_mode,
            }
        )
        if axis_pairs:
            run.grouping["alpha_x"] = float(vector_alphas.get("P_x", run_alpha))
            run.grouping["alpha_y"] = float(vector_alphas.get("P_y", run_alpha))
            run.grouping["alpha_z"] = float(vector_alphas.get("P_z", run_alpha))
        if (
            isinstance(detector_t0_bins, list)
            and len(detector_t0_bins) == len(run.histograms)
        ):
            run.grouping["detector_t0_bins"] = [int(hist.t0_bin) for hist in run.histograms]
        if enforce_source_bunching:
            run.grouping["source_bunching_factor"] = source_bunch_factor
        # Persist group names if provided
        group_names = grouping_result.get("group_names")
        if isinstance(group_names, dict) and group_names:
            run.grouping["group_names"] = {int(k): str(v) for k, v in group_names.items()}
        if vector_axis and axis_pairs:
            run.grouping["vector_axis"] = vector_axis
        preset_name = grouping_result.get("grouping_preset")
        if preset_name:
            run.grouping["grouping_preset"] = str(preset_name)
        else:
            run.grouping.pop("grouping_preset", None)
        instrument_name = grouping_result.get("instrument")
        if instrument_name:
            run.grouping["instrument"] = str(instrument_name)
        else:
            run.grouping.pop("instrument", None)
        return True, dt_applied

    def _prepare_grouping_histograms(self, histograms, grouping: dict, use_deadtime: bool):
        """Return histograms prepared for grouping, with optional deadtime correction.

        Parameters
        ----------
        histograms
            Original run histograms.
        grouping
            Run grouping dictionary potentially containing ``dead_time_us``.
        use_deadtime
            Whether deadtime correction should be applied.

        Returns
        -------
        tuple[list[Histogram], bool]
            Prepared histogram list and a flag indicating whether deadtime was
            actually applied.
        """
        if not use_deadtime:
            return list(histograms), False
        return prepare_histograms_with_deadtime(histograms, grouping, use_deadtime)

    def _dataset_supports_background_correction(self, dataset) -> bool:
        """Return whether this dataset should expose PSI-style background correction."""
        run = getattr(dataset, "run", None)
        metadata = dict(getattr(dataset, "metadata", {}) or {})
        if run is not None:
            metadata.update(getattr(run, "metadata", {}) or {})
        source_file = str(getattr(run, "source_file", "") if run is not None else "")
        if not source_file:
            source_file = str(metadata.get("source_file", ""))
        return supports_background_correction(metadata=metadata, source_file=source_file)

    def _apply_deadtime_correction(
        self,
        counts,
        tau_us: float,
        bin_width_us: float,
        *,
        num_good_frames: float = 1.0,
    ):
        """Apply non-paralyzable deadtime correction to histogram counts.

        The corrected counts are computed as:

        ``N_corr = N / (1 - N * tau / (dt * n_frames))``

        where ``tau`` is detector deadtime, ``dt`` is bin width, and
        ``n_frames`` is the number of good frames (Mantid-compatible).
        Denominators are clamped away from zero for numerical stability.
        """
        return apply_deadtime_correction(
            counts,
            tau_us,
            bin_width_us,
            num_good_frames=num_good_frames,
        )

    def _maybe_apply_comment_field(
        self,
        dataset,
        path: str,
        apply_to_all: bool,
    ) -> str:
        """Optionally apply comment-derived field value.

        Returns one of: "none", "yes_to_all", "cancel".
        """
        meta = dataset.metadata
        candidate = meta.get("field_comment_candidate")
        header = meta.get("field_header")

        # Only prompt when a plausible candidate exists and header field is
        # missing or zero.
        if candidate is None:
            return "none"

        header_value = 0.0 if header is None else float(header)
        if header_value != 0.0:
            return "none"

        if apply_to_all:
            meta["field"] = float(candidate)
            if dataset.run is not None:
                dataset.run.metadata["field"] = float(candidate)
            self._log_panel.log(
                f"Applied comment field {candidate:.1f} G to run {dataset.run_number}"
            )
            return "yes_to_all"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Apply Field From Comment?")
        msg.setText(
            "Field in header is 0 G, but comment contains a field candidate."
        )
        msg.setInformativeText(
            f"File: {path}\n"
            f"Run: {dataset.run_number}\n"
            f"Comment field candidate: {candidate:.1f} G\n\n"
            "Apply this value as B (G)?"
        )
        yes_btn = msg.addButton("Yes", QMessageBox.ButtonRole.YesRole)
        no_btn = msg.addButton("No", QMessageBox.ButtonRole.NoRole)
        yes_all_btn = msg.addButton("Yes to All", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(yes_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == yes_all_btn:
            meta["field"] = float(candidate)
            if dataset.run is not None:
                dataset.run.metadata["field"] = float(candidate)
            self._log_panel.log(
                f"Applied comment field {candidate:.1f} G to run {dataset.run_number}"
            )
            return "yes_to_all"
        if clicked == yes_btn:
            meta["field"] = float(candidate)
            if dataset.run is not None:
                dataset.run.metadata["field"] = float(candidate)
            self._log_panel.log(
                f"Applied comment field {candidate:.1f} G to run {dataset.run_number}"
            )
            return "none"
        if clicked == cancel_btn:
            return "cancel"

        # "No" keeps original header field value.
        if clicked == no_btn:
            self._log_panel.log(
                f"Kept header field {header_value:.1f} G for run {dataset.run_number}"
            )
        return "none"

    def _on_fit(self) -> None:
        """Show and raise the Fit dock panel."""
        self._dock_fit.show()
        self._dock_fit.raise_()
        self._log_panel.log("Opened Fit panel")

    def _on_fourier(self) -> None:
        """Show and raise the Fourier dock panel."""
        self._dock_fourier.show()
        self._dock_fourier.raise_()
        self._log_panel.log("Opened Fourier panel")

    def _on_fit_parameters(self) -> None:
        """Show and raise the Fitted Parameters dock panel."""
        self._dock_fit_parameters.show()
        self._dock_fit_parameters.raise_()
        self._log_panel.log("Opened Fit Parameters panel")

    def _on_global_parameter_fit(self) -> None:
        """Show the Global Parameter Fit window if cross-group results exist."""
        if self._global_parameter_fit_window is None or not self._global_parameter_fit_window.has_result():
            return
        self._global_parameter_fit_window.show()
        self._global_parameter_fit_window.raise_()
        self._global_parameter_fit_window.activateWindow()

    def _update_global_parameter_fit_menu_style(self, has_result: bool) -> None:
        if not hasattr(self, "_global_parameter_fit_action"):
            return
        if has_result:
            self._global_parameter_fit_action.setText("Global Parameter Fit [available]")
        else:
            self._global_parameter_fit_action.setText("Global Parameter Fit")
        self._global_parameter_fit_action.setEnabled(bool(has_result))
        if hasattr(self, "_global_parameter_fit_toolbar_action"):
            self._global_parameter_fit_toolbar_action.setEnabled(bool(has_result))

    def _on_about(self) -> None:
        """Show the About dialog with version information."""
        from asymmetry import __version__

        QMessageBox.about(
            self,
            "About Asymmetry",
            f"Asymmetry v{__version__}\n\nA Python library for μSR data analysis.",
        )

    def _reset_layout(self) -> None:
        """Reset dock panels to their default layout positions (not yet implemented)."""

    def _on_dataset_selected(self, run_number: int) -> None:
        """Handle dataset selection from data browser."""
        self._active_group_context = None
        if hasattr(self._plot_panel, "set_active_label_group"):
            self._plot_panel.set_active_label_group(None)
        dataset = self._data_browser.get_dataset(run_number)
        if dataset:
            self._current_dataset = dataset
            active_axis = None
            if hasattr(self._plot_panel, "get_current_polarization_axis"):
                active_axis = self._normalize_vector_axis(
                    self._plot_panel.get_current_polarization_axis()
                )
            if active_axis in {"P_x", "P_y", "P_z"}:
                self._synchronize_targets_to_axis([dataset], active_axis)
            self._render_current_selection_plot()
            self._refresh_vector_axis_selector()
            self._update_fit_block_state()
            self._fit_panel.set_dataset(self._get_fit_dataset(dataset))
            self._log_panel.log(f"Selected run {run_number}")
            self.statusBar().showMessage(f"Viewing run {run_number}")

    def _on_group_selected(self, group_id: str) -> None:
        """Track selected data-group context for grouped global-fit workflows."""
        group_name = self._data_browser.get_group_name(group_id)
        if group_name is None:
            self._active_group_context = None
            if hasattr(self._plot_panel, "set_active_label_group"):
                self._plot_panel.set_active_label_group(None)
            return
        self._active_group_context = (group_id, group_name)
        if hasattr(self._plot_panel, "set_active_label_group"):
            self._plot_panel.set_active_label_group(group_id)
        self.statusBar().showMessage(f"Selected group: {group_name}")

    def _on_overlay_toggled(self, enabled: bool) -> None:
        """Refresh plotting and fit context when overlay mode changes."""
        self._update_selected_datasets()
        mode = "enabled" if enabled else "disabled"
        self._log_panel.log(f"Plot overlay {mode}.")

    def _on_fit_range_changed(self, _x_min: float, _x_max: float) -> None:
        """Refresh fit inputs when the selected fit x-range changes."""
        if self._current_dataset is not None:
            self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))
        self._update_selected_datasets()

    def _on_fit_completed(self, fit_result, fitted_curve, component_curves) -> None:
        """Handle completed fit from fit panel."""
        t_fit, y_fit = fitted_curve
        fit_function = None
        if hasattr(self._fit_panel, "single_fit_formula_string"):
            fit_function = self._fit_panel.single_fit_formula_string()
        self._plot_panel.plot_fit(
            t_fit,
            y_fit,
            label="Fit",
            component_curves=component_curves,
            fit_result=fit_result,
            fit_function=fit_function,
        )
        self._log_panel.log(
            f"Fit completed: χ²ᵣ = {fit_result.reduced_chi_squared:.4f}"
        )

    def _on_preview_requested(self, fit_result, fitted_curve, component_curves) -> None:
        """Handle preview request from fit panel."""
        t_fit, y_fit = fitted_curve
        fit_function = None
        if hasattr(self._fit_panel, "single_fit_formula_string"):
            fit_function = self._fit_panel.single_fit_formula_string()
        self._plot_panel.plot_fit(
            t_fit,
            y_fit,
            label="Preview",
            component_curves=component_curves,
            fit_result=None,
            fit_function=fit_function,
        )

    def _on_share_single_function_with_group(self, source_run_number: int) -> None:
        """Copy single-fit function settings from one run to its data-group peers."""
        if not hasattr(self._data_browser, "get_group_id_for_run"):
            self.statusBar().showMessage("Data-group sharing unavailable in this browser mode")
            return

        group_id = self._data_browser.get_group_id_for_run(source_run_number)
        if not group_id:
            self.statusBar().showMessage("Selected run is not in a data group")
            return

        member_runs = []
        if hasattr(self._data_browser, "get_group_member_run_numbers"):
            member_runs = self._data_browser.get_group_member_run_numbers(group_id)
        if not member_runs:
            self.statusBar().showMessage("No data-group members found to share with")
            return

        target_runs = [rn for rn in member_runs if int(rn) != int(source_run_number)]
        if not target_runs:
            self.statusBar().showMessage("Data group has no other members to share with")
            return

        updated = 0
        if hasattr(self._fit_panel, "share_single_function_state"):
            updated = int(self._fit_panel.share_single_function_state(source_run_number, target_runs))

        group_name = (
            self._data_browser.get_group_name(group_id)
            if hasattr(self._data_browser, "get_group_name")
            else group_id
        )
        self._log_panel.log(
            f"Shared fit function from run {source_run_number} to {updated} run(s) in group {group_name}"
        )
        self.statusBar().showMessage(
            f"Shared fit function to {updated} run(s) in group {group_name}"
        )

    def _on_global_fit_completed(self, results_dict, global_params) -> None:
        """Handle completed global fit.

        Parameters
        ----------
        results_dict : dict
            Dictionary mapping run_number -> (FitResult, fitted_curve_tuple,
            component_curves).
        global_params : ParameterSet
            The fitted global parameters.
        """
        # Normalize run-number keys first (Qt signal transport can coerce key types).
        normalized_results = {}
        for run_key, payload in results_dict.items():
            try:
                run_number = int(run_key)
            except (TypeError, ValueError):
                continue
            normalized_results[run_number] = payload

        # Normalize payload shape for backward compatibility:
        #   legacy: (result, fitted_curve)
        #   current: (result, fitted_curve, component_curves)
        normalized_payloads = {}
        for run_number, payload in normalized_results.items():
            if not isinstance(payload, tuple) or len(payload) < 2:
                continue
            if len(payload) >= 3:
                result, fitted_curve, component_curves = payload[0], payload[1], payload[2]
            else:
                result, fitted_curve = payload[0], payload[1]
                component_curves = []
            normalized_payloads[run_number] = (result, fitted_curve, component_curves)

        # Store fit curves for all datasets
        global_fit_function = None
        if hasattr(self._fit_panel, "global_fit_formula_string"):
            global_fit_function = self._fit_panel.global_fit_formula_string()
        fit_curves = {}
        for run_number, (result, fitted_curve, component_curves) in normalized_payloads.items():
            t_fit, y_fit = fitted_curve
            axis_key = None
            dataset = self._data_browser.get_dataset(run_number)
            if dataset is not None:
                run = getattr(dataset, "run", None)
                grouping = getattr(run, "grouping", None)
                if isinstance(grouping, dict):
                    axis_key = self._normalize_vector_axis(grouping.get("vector_axis"))
                    if axis_key == "ALL":
                        axis_key = None
            fit_curves[run_number] = (
                t_fit,
                y_fit,
                "Global Fit",
                component_curves,
                result,
                global_fit_function,
                axis_key,
            )

        self._fit_panel.register_global_fit_results(normalized_payloads)

        # Set all fit curves in plot panel
        self._plot_panel.set_global_fits(fit_curves)

        # Push fitted parameters into the trends panel.
        trends_results = {
            run_number: (result, fitted_curve)
            for run_number, (result, fitted_curve, _component_curves) in normalized_payloads.items()
        }
        datasets_by_run = {}
        for run_number in normalized_payloads:
            dataset = self._data_browser.get_dataset(run_number)
            if dataset is not None:
                datasets_by_run[run_number] = dataset
        group_id = None
        group_name = None
        selected_group_ids = (
            self._data_browser.get_selected_group_ids()
            if hasattr(self._data_browser, "get_selected_group_ids")
            else []
        )
        if len(selected_group_ids) == 1:
            group_id = selected_group_ids[0]
            group_name = (
                self._data_browser.get_group_name(group_id)
                if hasattr(self._data_browser, "get_group_name")
                else None
            )
        elif self._active_group_context is not None:
            group_id, group_name = self._active_group_context

        self._fit_parameters_panel.set_fit_results(
            trends_results,
            datasets_by_run,
            global_params,
            group_id=group_id,
            group_name=group_name,
        )
        self._dock_fit_parameters.show()
        self._dock_fit_parameters.raise_()

        # Log summary
        successful_results = [
            payload for payload in normalized_payloads.values()
            if payload and payload[0].success
        ]
        n_datasets = len(successful_results)
        if n_datasets == 0:
            self._log_panel.log(
                "Global fit completed but no successful dataset results were available"
            )
            self.statusBar().showMessage(
                "Global fit completed with no successful results"
            )
            return

        avg_chi2r = sum(payload[0].reduced_chi_squared for payload in successful_results) / n_datasets
        self._log_panel.log(
            f"Global fit completed: {n_datasets} datasets, "
            f"average χ²ᵣ = {avg_chi2r:.3f}"
        )
        self.statusBar().showMessage(f"Global fit completed for {n_datasets} datasets")

    def _on_cross_group_fit_completed(self, parameter_name, groups, output) -> None:
        """Display cross-group fit output in the Global Parameter Fit window."""
        if self._global_parameter_fit_window is None:
            self._global_parameter_fit_window = GlobalParameterFitWindow(self)
        fit_result = getattr(output, "fit_result", None)
        model = getattr(output, "model", None)
        x_key = getattr(output, "x_key", "run")
        fit_x_min = getattr(output, "fit_x_min", float("nan"))
        fit_x_max = getattr(output, "fit_x_max", float("nan"))
        if fit_result is None or model is None:
            return
        self._global_parameter_fit_window.set_results(
            parameter_name=parameter_name,
            x_key=x_key,
            groups=groups,
            model=model,
            result=fit_result,
            fit_x_min=fit_x_min,
            fit_x_max=fit_x_max,
        )
        self._global_parameter_fit_window.show()
        self._global_parameter_fit_window.raise_()
        self._global_parameter_fit_window.activateWindow()
        self._update_global_parameter_fit_menu_style(True)

    def _on_fit_parameters_group_fits_deleted(self, group_id: str, run_numbers: object) -> None:
        """Clear run-level fit state when a Fit Parameters group is deleted."""
        if not isinstance(run_numbers, (list, tuple, set)):
            return

        normalized_runs: list[int] = []
        seen: set[int] = set()
        for run_number in run_numbers:
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            if run_key in seen:
                continue
            seen.add(run_key)
            normalized_runs.append(run_key)

        if not normalized_runs:
            return

        self._fit_panel.clear_fits_for_runs(normalized_runs)
        self._plot_panel.clear_fits_for_runs(normalized_runs)
        self._log_panel.log(
            f"Deleted fit(s) for group {group_id}: cleared {len(normalized_runs)} dataset fit entry/entries"
        )
        self.statusBar().showMessage(f"Deleted fit(s) for group {group_id}")

    def _update_selected_datasets(self, *_args) -> None:
        """Update the fit panel with currently selected datasets."""
        selected = self._data_browser.get_selected_datasets()
        active_axis = None
        if hasattr(self._plot_panel, "get_current_polarization_axis"):
            active_axis = self._normalize_vector_axis(self._plot_panel.get_current_polarization_axis())

        if selected and active_axis in {"P_x", "P_y", "P_z"}:
            updated = self._synchronize_targets_to_axis(selected, active_axis)
            if updated > 0:
                self._data_browser._rebuild_table()
                selected = self._data_browser.get_selected_datasets()

        selected_group_ids = (
            self._data_browser.get_selected_group_ids()
            if hasattr(self._data_browser, "get_selected_group_ids")
            else []
        )
        active_plot_group_id = selected_group_ids[0] if len(selected_group_ids) == 1 else None
        if hasattr(self._plot_panel, "set_active_label_group"):
            self._plot_panel.set_active_label_group(active_plot_group_id)
        if len(selected_group_ids) == 1:
            gid = selected_group_ids[0]
            gname = (
                self._data_browser.get_group_name(gid)
                if hasattr(self._data_browser, "get_group_name")
                else None
            )
            if gname is not None:
                self._active_group_context = (gid, gname)
        elif selected_group_ids:
            self._active_group_context = None
        if self._current_dataset is not None:
            current_run = self._current_dataset.run_number
            if self._data_browser.get_dataset(current_run) is None:
                self._current_dataset = None
                self._plot_panel.clear()
                self._fit_panel.set_dataset(None)

        # Multi-selection render mode depends on the plot-panel Overlay toggle.
        if len(selected) > 1:
            self._render_current_selection_plot()
            self._refresh_vector_axis_selector()
            self._update_fit_block_state()
            if self._overlay_enabled():
                run_labels = ", ".join(str(ds.run_label) for ds in selected)
                self.statusBar().showMessage(f"Viewing runs {run_labels}")
            elif self._current_dataset is not None:
                self.statusBar().showMessage(f"Viewing run {self._current_dataset.run_label}")
        elif self._current_dataset is not None:
            self._refresh_vector_axis_selector()
            self._update_fit_block_state()
        else:
            self._update_fit_block_state()

        analysis_datasets = [
            dataset
            for dataset in (
                self._get_fit_dataset(ds) for ds in selected
            )
            if dataset is not None
        ]

        # Refresh the single-fit tab with the currently active dataset so that
        # bunch-factor or fit-range changes are reflected immediately.
        if self._current_dataset is not None:
            self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))

        self._fit_panel.set_datasets(analysis_datasets)

    def _get_fit_dataset(self, dataset):
        """Return analysis dataset restricted to the active fit range."""
        analysis_dataset = self._plot_panel.get_analysis_dataset(dataset)
        return self._plot_panel.get_fit_dataset(analysis_dataset)

    # ── project save / open ────────────────────────────────────────────

    def _on_new_project(self) -> None:
        """Clear all state to start a fresh project."""
        reply = QMessageBox.question(
            self,
            "New Project",
            "Clear the current session and start a new project?\n"
            "Unsaved changes will be lost.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return
        self._clear_all_state()
        self._current_project_path = None
        self._update_window_title()
        self._log_panel.log("Started new project")
        self.statusBar().showMessage("New project")

    def _on_open_project(self) -> None:
        """Open a project file chosen via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._last_open_dir,
            _PROJECT_FILE_FILTER,
        )
        if path:
            self._open_project_file(path)

    def _on_save_project(self) -> None:
        """Save the current project to its existing path, or prompt if new."""
        if self._current_project_path:
            self._write_project(self._current_project_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self) -> None:
        """Save the current project to a user-selected path."""
        default = (
            self._current_project_path
            or os.path.join(self._last_open_dir, "project.asymp")
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            default,
            _PROJECT_FILE_FILTER,
        )
        if path:
            if not path.endswith(".asymp"):
                path += ".asymp"
            self._write_project(path)

    def _write_project(self, path: str) -> None:
        """Collect state and write to *path*, updating recent projects."""
        try:
            state = self.collect_project_state()
            save_project(state, path)
            self._current_project_path = path
            self._add_recent_project(path)
            self._update_window_title()
            self._log_panel.log(f"Project saved: {path}")
            self.statusBar().showMessage(f"Saved: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(
                self, "Save Failed", f"Could not save project:\n{e}"
            )
            self._log_panel.log(f"ERROR saving project: {e}")

    def _open_project_file(self, path: str) -> None:
        """Load and restore a project from *path*."""
        try:
            state = load_project(path)
        except UnsupportedSchemaVersion as e:
            QMessageBox.critical(self, "Unsupported Project File", str(e))
            self._log_panel.log(f"ERROR opening project: {e}")
            return
        except Exception as e:
            QMessageBox.critical(
                self, "Could Not Open Project", f"Failed to read project file:\n{e}"
            )
            self._log_panel.log(f"ERROR opening project: {e}")
            return

        self._clear_all_state()
        self.restore_project_state(state, path)
        self._current_project_path = path
        self._add_recent_project(path)
        self._update_window_title()

    def collect_project_state(self) -> dict:
        """Return a full serialisable snapshot of the current application state.

        Returns
        -------
        dict
            Project state suitable for passing to
            :func:`~asymmetry.core.project.save_project`.
        """
        from asymmetry import __version__

        # Build dataset list (source files + field overrides).
        # Important: combined datasets remove their source runs from
        # ``_datasets``, so we must also capture source runs from
        # ``_combined_source_datasets`` to guarantee they can be recreated.
        datasets = []
        seen_run_numbers: set[int] = set()

        def _append_dataset_entry(dataset) -> None:
            run_number = int(dataset.run_number)
            if run_number in seen_run_numbers:
                return

            source_file = ""
            if dataset.run:
                source_file = dataset.run.source_file
            if not source_file:
                source_file = str(dataset.metadata.get("source_file", ""))

            datasets.append({
                "run_number": run_number,
                "source_file": source_file,
                "metadata_overrides": {
                    "field": float(dataset.metadata.get("field", 0.0)),
                },
                "grouping_overrides": self._extract_grouping_overrides(dataset),
            })
            seen_run_numbers.add(run_number)

        for run_number, dataset in self._data_browser._datasets.items():
            if run_number in self._data_browser._combined_datasets:
                continue  # Combined datasets are captured separately.
            _append_dataset_entry(dataset)

        combined_sources = getattr(self._data_browser, "_combined_source_datasets", {})
        for source_datasets in combined_sources.values():
            for dataset in source_datasets:
                _append_dataset_entry(dataset)

        # Combined dataset definitions.
        combined_datasets = [
            {
                "combined_run_number": int(crn),
                "source_run_numbers": [int(r) for r in src_runs],
            }
            for crn, src_runs in self._data_browser._combined_datasets.items()
        ]

        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "created_with_app_version": __version__,
            "datasets": datasets,
            "combined_datasets": combined_datasets,
            "browser_state": self._data_browser.get_state(),
            "plot_state": self._plot_panel.get_state(),
            "single_fit_state": self._fit_panel.get_single_state(),
            "global_fit_state": self._fit_panel.get_global_state(),
            "fit_ui_state": self._fit_panel.get_ui_state(),
            "fit_parameters_state": self._fit_parameters_panel.get_state(),
            "global_parameter_fit_window_state": (
                self._global_parameter_fit_window.get_state()
                if self._global_parameter_fit_window is not None
                else None
            ),
            "fourier_state": self._fourier_panel.get_state(),
        }

    def restore_project_state(self, state: dict, project_path: str) -> None:
        """Restore the full application state from a project file state dict.

        Parameters
        ----------
        state : dict
            Validated project state as returned by
            :func:`~asymmetry.core.project.load_project`.
        project_path : str
            Absolute path to the project file (used to resolve relative paths).
        """
        project_dir = os.path.dirname(os.path.abspath(project_path))
        loaded_run_numbers: set[int] = set()

        # ── resolve source file paths, prompting once for a fallback dir ──
        def _resolve_source_file(source_file: str) -> str | None:
            """Return resolved path to *source_file*, or None if not found."""
            if not source_file:
                return None
            if os.path.exists(source_file):
                return source_file
            rel = os.path.join(project_dir, source_file)
            if os.path.exists(rel):
                return rel
            return None

        datasets_info = state.get("datasets", [])

        # First pass: attempt to resolve all paths with no user interaction.
        resolved_paths: dict[int, str | None] = {}
        missing_info: list[dict] = []  # entries whose files cannot be found
        for ds_info in datasets_info:
            sf = ds_info.get("source_file", "")
            rn = ds_info.get("run_number")
            resolved = _resolve_source_file(sf) if sf else None
            resolved_paths[rn] = resolved
            if resolved is None:
                missing_info.append(ds_info)

        # If any files are missing, offer the user one chance to redirect to a new directory.
        fallback_dir: str | None = None
        if missing_info:
            names = ", ".join(
                os.path.basename(d.get("source_file", "?")) for d in missing_info[:5]
            )
            suffix = f" and {len(missing_info) - 5} more" if len(missing_info) > 5 else ""
            answer = QMessageBox.question(
                self,
                "Data Files Not Found",
                f"The following data file(s) could not be found:\n\n"
                f"  {names}{suffix}\n\n"
                f"Would you like to locate the directory where these files are now stored?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                fallback_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Locate Data Directory",
                    project_dir,
                )

        # Second pass: apply fallback directory for still-missing files.
        if fallback_dir:
            for ds_info in missing_info:
                sf = ds_info.get("source_file", "")
                rn = ds_info.get("run_number")
                candidate = os.path.join(fallback_dir, os.path.basename(sf))
                if os.path.exists(candidate):
                    resolved_paths[rn] = candidate

        # ── load source files ──────────────────────────────────────────
        loaded_file_cache: dict[str, object] = {}
        for ds_info in datasets_info:
            rn = ds_info.get("run_number")
            source_file = ds_info.get("source_file", "")
            if not source_file:
                self._log_panel.log(
                    f"WARNING: Run {rn} has no source file; skipping."
                )
                continue

            resolved = resolved_paths.get(rn)
            if not resolved:
                self._log_panel.log(
                    f"WARNING: Source file not found: {source_file}; skipping."
                )
                continue

            try:
                if resolved in loaded_file_cache:
                    loaded_obj = loaded_file_cache[resolved]
                else:
                    loaded_obj = self._load_file(resolved)
                    loaded_file_cache[resolved] = loaded_obj

                if loaded_obj is None:
                    continue

                candidates = loaded_obj if isinstance(loaded_obj, list) else [loaded_obj]
                dataset = None
                for cand in candidates:
                    if cand is None:
                        continue
                    try:
                        if int(cand.run_number) == int(rn):
                            dataset = cand
                            break
                    except (TypeError, ValueError):
                        continue
                if dataset is None and len(candidates) == 1:
                    dataset = candidates[0]
                if dataset is None:
                    self._log_panel.log(
                        f"WARNING: Run {rn} not found in loaded file {source_file}; skipping."
                    )
                    continue
                if int(dataset.run_number) in loaded_run_numbers:
                    continue

                # Apply saved metadata overrides without prompting.
                for key, val in ds_info.get("metadata_overrides", {}).items():
                    dataset.metadata[key] = val
                    if dataset.run:
                        dataset.run.metadata[key] = val

                grouping_overrides = ds_info.get("grouping_overrides")
                if isinstance(grouping_overrides, dict):
                    self._apply_grouping_settings_to_dataset(dataset, grouping_overrides)

                self._data_browser.add_dataset(dataset)
                loaded_run_numbers.add(dataset.run_number)
            except Exception as e:
                self._log_panel.log(f"ERROR loading {source_file}: {e}")

        # ── recreate combined datasets ─────────────────────────────────
        # Map saved combined IDs to restored IDs so selection can be adjusted.
        combined_id_map: dict[int, int] = {}
        for combined_info in state.get("combined_datasets", []):
            old_id = combined_info.get("combined_run_number")
            src_runs = combined_info.get("source_run_numbers", [])
            if all(rn in loaded_run_numbers for rn in src_runs):
                new_id = self._data_browser.add_combined_dataset(src_runs)
                if new_id is not None and old_id is not None:
                    combined_id_map[int(old_id)] = new_id
            else:
                missing = [rn for rn in src_runs if rn not in loaded_run_numbers]
                self._log_panel.log(
                    f"WARNING: Could not recreate combined dataset "
                    f"{src_runs}; missing runs: {missing}"
                )

        # ── fix up browser state: remap old combined IDs ───────────────
        browser_state = dict(state.get("browser_state", {}))
        if combined_id_map and "selected_run_numbers" in browser_state:
            browser_state["selected_run_numbers"] = [
                combined_id_map.get(rn, rn)
                for rn in browser_state["selected_run_numbers"]
            ]
        self._data_browser.restore_state(browser_state)

        # ── restore plot state ─────────────────────────────────────────
        plot_state = state.get("plot_state", {})
        current_run = plot_state.get("current_run_number")
        if current_run is not None:
            current_run = combined_id_map.get(int(current_run), int(current_run))
        current_dataset = (
            self._data_browser.get_dataset(current_run)
            if current_run is not None
            else None
        )
        if current_dataset is not None:
            self._current_dataset = current_dataset
        self._plot_panel.restore_state(plot_state, current_dataset)

        # Propagate current dataset to fit panel.
        if current_dataset is not None:
            self._fit_panel.set_dataset(self._get_fit_dataset(current_dataset))
        self._update_selected_datasets()

        restored_axis = self._normalize_vector_axis(plot_state.get("polarization_axis"))
        if restored_axis == "ALL":
            # restore_state redraws via plot_dataset(dataset), which is a single
            # axis path. Force a selection-aware rerender so ALL mode shows
            # stacked vector subplots immediately after project load.
            self._render_current_selection_plot()
            self._refresh_vector_axis_selector()

        # ── restore fit panel states (after dataset propagation) ──────
        single_fit_state = state.get("single_fit_state")
        if single_fit_state:
            self._fit_panel.restore_single_state(single_fit_state)

        global_fit_state = state.get("global_fit_state")
        if global_fit_state:
            self._fit_panel.restore_global_state(global_fit_state)

        fit_ui_state = state.get("fit_ui_state")
        if fit_ui_state:
            self._fit_panel.restore_ui_state(fit_ui_state)

        fit_parameters_state = state.get("fit_parameters_state")
        if fit_parameters_state:
            self._fit_parameters_panel.restore_state(fit_parameters_state)

        restored_cross_group = getattr(self._fit_parameters_panel, "last_cross_group_fit", None)
        global_parameter_fit_window_state = state.get("global_parameter_fit_window_state")
        if isinstance(restored_cross_group, dict):
            fit_result = restored_cross_group.get("fit_result")
            model = restored_cross_group.get("model")
            groups = restored_cross_group.get("groups")
            parameter_name = restored_cross_group.get("parameter_name")
            x_key = restored_cross_group.get("x_key", "run")
            fit_x_min = restored_cross_group.get("fit_x_min", float("nan"))
            fit_x_max = restored_cross_group.get("fit_x_max", float("nan"))
            if (
                fit_result is not None
                and model is not None
                and isinstance(groups, list)
                and isinstance(parameter_name, str)
            ):
                if self._global_parameter_fit_window is None:
                    self._global_parameter_fit_window = GlobalParameterFitWindow(self)
                self._global_parameter_fit_window.set_results(
                    parameter_name=parameter_name,
                    x_key=str(x_key),
                    groups=groups,
                    model=model,
                    result=fit_result,
                    fit_x_min=float(fit_x_min),
                    fit_x_max=float(fit_x_max),
                )
                if isinstance(global_parameter_fit_window_state, dict):
                    self._global_parameter_fit_window.restore_state(global_parameter_fit_window_state)
                self._global_parameter_fit_window.show()
                self._global_parameter_fit_window.raise_()
                self._global_parameter_fit_window.activateWindow()
                self._update_global_parameter_fit_menu_style(True)
            else:
                self._update_global_parameter_fit_menu_style(False)
        else:
            self._update_global_parameter_fit_menu_style(False)

        # ── restore Fourier state ──────────────────────────────────────
        fourier_state = state.get("fourier_state")
        if fourier_state:
            self._fourier_panel.restore_state(fourier_state)

        # Open fit-related docks automatically when project contains saved
        # results/state for those panes.
        if _has_saved_fit_results(single_fit_state, global_fit_state):
            self._dock_fit.show()
            self._dock_fit.raise_()

        if _has_saved_fit_parameters_results(fit_parameters_state):
            self._dock_fit_parameters.show()
            self._dock_fit_parameters.raise_()

        n_loaded = len(loaded_run_numbers)
        self._log_panel.log(
            f"Project opened: {n_loaded} run(s) loaded from {project_path}"
        )
        self.statusBar().showMessage(
            f"Opened project — {n_loaded} run(s) loaded"
        )

    def _clear_all_state(self) -> None:
        """Reset every panel to its empty initial state."""
        self._current_dataset = None
        self._data_browser.clear()
        self._plot_panel.clear()
        self._fit_panel.set_dataset(None)
        self._fit_panel.set_datasets([])
        self._fit_parameters_panel.clear()
        if self._global_parameter_fit_window is not None:
            self._global_parameter_fit_window.close()
            self._global_parameter_fit_window = None
        self._update_global_parameter_fit_menu_style(False)

    def _add_recent_project(self, path: str) -> None:
        """Add *path* to the front of the recent-projects list in QSettings."""
        raw = self._settings.value("project/recent_files")
        recent: list[str] = list(raw) if isinstance(raw, (list, tuple)) else []
        # Remove any existing entry for this path (avoid duplicates).
        recent = [p for p in recent if p != path]
        recent.insert(0, path)
        recent = recent[:_MAX_RECENT_PROJECTS]
        self._settings.setValue("project/recent_files", recent)
        self._update_recent_projects_menu()

    def _update_recent_projects_menu(self) -> None:
        """Rebuild the Recent Projects submenu from QSettings."""
        self._recent_menu.clear()
        raw = self._settings.value("project/recent_files")
        recent: list[str] = list(raw) if isinstance(raw, (list, tuple)) else []
        if not recent:
            action = self._recent_menu.addAction("(No recent projects)")
            action.setEnabled(False)
            return
        for path in recent:
            action = self._recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda checked=False, p=path: self._open_project_file(p))
        self._recent_menu.addSeparator()
        self._recent_menu.addAction(
            "Clear Recent Projects", self._clear_recent_projects
        )

    def _clear_recent_projects(self) -> None:
        """Remove all entries from the recent-projects list."""
        self._settings.remove("project/recent_files")
        self._update_recent_projects_menu()

    def _update_window_title(self) -> None:
        """Update window title to reflect the current project file name."""
        if self._current_project_path:
            name = Path(self._current_project_path).stem
            self.setWindowTitle(f"Asymmetry — {name}")
        else:
            self.setWindowTitle("Asymmetry \u2014 \u03bcSR Data Analysis")


def _has_saved_fit_results(single_fit_state: dict | None, global_fit_state: dict | None) -> bool:
    """Return True when project state contains persisted fit result text."""
    default_single = "No fit performed yet"
    single_html = ""
    if isinstance(single_fit_state, dict):
        single_html = str(single_fit_state.get("result_html", "")).strip()
    if single_html and single_html != default_single:
        return True

    global_html = ""
    if isinstance(global_fit_state, dict):
        global_html = str(global_fit_state.get("result_html", "")).strip()
    if global_html and "No fit performed yet" not in global_html:
        return True

    return False


def _has_saved_fit_parameters_results(fit_parameters_state: dict | None) -> bool:
    """Return True when project state contains persisted fitted-parameter rows."""
    if not isinstance(fit_parameters_state, dict):
        return False
    rows = fit_parameters_state.get("rows", [])
    return isinstance(rows, list) and len(rows) > 0
