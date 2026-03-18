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

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QIcon
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
from asymmetry.core.transform import apply_grouping, compute_asymmetry
from asymmetry.core.transform.rebin import rebin
from asymmetry.core.utils.constants import PeriodMode

_MAX_RECENT_PROJECTS = 10
_PROJECT_FILE_FILTER = "Asymmetry projects (*.asymp);;All files (*)"

from asymmetry.gui.panels.data_browser import DataBrowserPanel
from asymmetry.gui.panels.fit_panel import FitPanel
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from asymmetry.gui.windows.grouping_dialog import GroupingDialog
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
        from importlib.resources import as_file, files

        resources = files("asymmetry").joinpath("resources")
        logo = resources.joinpath("logo_256x256.png")
        with as_file(logo) as icon_path:
            if icon_path.exists():
                return QIcon(str(icon_path))
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError):
        pass

    # Fallback: try direct path (for development)
    try:
        from pathlib import Path

        resources_dir = Path(__file__).parent.parent / "resources"
        icon_path = resources_dir / "logo_256x256.png"
        if icon_path.exists():
            return QIcon(str(icon_path))
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
                        applied, _ = self._apply_grouping_settings_to_dataset(dataset, auto_grouping_payload)
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
        dialog = GroupingDialog(
            all_datasets,
            selected_run_number=selected_run_number,
            selected_run_numbers=selected_run_numbers,
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

        updated = 0
        skipped = 0
        deadtime_applied = 0
        deadtime_missing = 0
        first_updated_dataset = None

        for dataset in all_datasets:
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

            if dataset is self._current_dataset:
                self._fit_panel.set_dataset(self._get_fit_dataset(dataset))
            if first_updated_dataset is None:
                first_updated_dataset = dataset
            updated += 1

        if updated > 0:
            self._data_browser._rebuild_table()
            selected_after = self._data_browser.get_selected_datasets()
            if len(selected_after) > 1:
                self._plot_panel.plot_datasets(selected_after)
            elif self._current_dataset is not None:
                self._plot_panel.plot_dataset(self._current_dataset)
            elif first_updated_dataset is not None:
                self._plot_panel.plot_dataset(first_updated_dataset)

        deadtime_msg = "off"
        if use_deadtime:
            deadtime_msg = f"on (applied={deadtime_applied}, missing={deadtime_missing})"

        self._log_panel.log(
            f"Applied grouping to {updated} dataset(s); skipped {skipped}. "
            f"F={grouping_result['forward_group']}, "
            f"B={grouping_result['backward_group']}, alpha={alpha:.6g}, "
            f"deadtime={deadtime_msg}"
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

        groups: dict[int, list[int]] = {}
        for key, values in groups_raw.items():
            try:
                gid = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(values, list):
                continue
            detectors: list[int] = []
            for value in values:
                try:
                    detectors.append(int(value))
                except (TypeError, ValueError):
                    continue
            if detectors:
                groups[gid] = detectors

        if not groups:
            return None

        payload = {
            "groups": groups,
            "forward_group": int(grouping.get("forward_group", 1)),
            "backward_group": int(grouping.get("backward_group", 2)),
            "alpha": float(grouping.get("alpha", 1.0)),
            "first_good_bin": int(grouping.get("first_good_bin", 0)),
            "last_good_bin": int(grouping.get("last_good_bin", 0)),
            "bunching_factor": int(grouping.get("bunching_factor", 1)),
            "deadtime_correction": bool(grouping.get("deadtime_correction", False)),
            "period_mode": str(grouping.get("period_mode", PeriodMode.RED)),
        }

        if "dead_time_us" in grouping and isinstance(grouping.get("dead_time_us"), list):
            payload["dead_time_us"] = list(grouping.get("dead_time_us", []))
        if "good_frames" in grouping:
            payload["good_frames"] = grouping.get("good_frames")
        return payload

    def _apply_grouping_settings_to_dataset(self, dataset, grouping_result: dict) -> tuple[bool, bool]:
        """Apply grouping settings to one dataset and recompute asymmetry.

        Returns
        -------
        tuple[bool, bool]
            ``(applied, deadtime_applied)``.
        """
        run = dataset.run
        if run is None or not run.histograms:
            return False, False

        groups_raw = grouping_result.get("groups", {})
        if not isinstance(groups_raw, dict):
            return False, False

        groups: dict[int, list[int]] = {}
        for key, values in groups_raw.items():
            try:
                gid = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(values, list):
                continue
            detectors: list[int] = []
            for value in values:
                try:
                    detectors.append(int(value))
                except (TypeError, ValueError):
                    continue
            if detectors:
                groups[gid] = detectors
        if not groups:
            return False, False

        try:
            forward_gid = int(grouping_result.get("forward_group", 1))
            backward_gid = int(grouping_result.get("backward_group", 2))
        except (TypeError, ValueError):
            return False, False

        forward_idx = [max(0, int(v) - 1) for v in groups.get(forward_gid, [])]
        backward_idx = [max(0, int(v) - 1) for v in groups.get(backward_gid, [])]
        if not forward_idx or not backward_idx:
            return False, False
        if max(forward_idx, default=-1) >= len(run.histograms):
            return False, False
        if max(backward_idx, default=-1) >= len(run.histograms):
            return False, False

        first_good = int(grouping_result.get("first_good_bin", 0))
        last_good = int(grouping_result.get("last_good_bin", len(run.histograms[0].counts) - 1))
        alpha = float(grouping_result.get("alpha", 1.0))
        bunch_factor = int(grouping_result.get("bunching_factor", 1))
        use_deadtime = bool(grouping_result.get("deadtime_correction", False))
        period_mode = str(grouping_result.get("period_mode", run.grouping.get("period_mode", PeriodMode.RED)))

        if not isinstance(run.grouping, dict):
            run.grouping = {}
        if isinstance(grouping_result.get("dead_time_us"), list):
            run.grouping["dead_time_us"] = list(grouping_result.get("dead_time_us", []))
        if "good_frames" in grouping_result:
            run.grouping["good_frames"] = grouping_result.get("good_frames")

        source_histograms = list(run.histograms)
        grouping_for_mode = run.grouping
        if isinstance(run.grouping, dict):
            source_histograms, grouping_for_mode = self._select_histograms_for_period_mode(
                run.histograms,
                run.grouping,
                period_mode,
            )
            run.grouping["period_mode"] = period_mode
            run.grouping["good_frames"] = grouping_for_mode.get("good_frames", run.grouping.get("good_frames", 1.0))
            run.grouping["dead_time_us"] = list(grouping_for_mode.get("dead_time_us", run.grouping.get("dead_time_us", [])))

        working_histograms, dt_applied = self._prepare_grouping_histograms(
            source_histograms,
            grouping_for_mode,
            use_deadtime,
        )

        run_alpha = alpha if alpha > 0 else 1.0

        if (
            isinstance(run.grouping, dict)
            and period_mode in {str(PeriodMode.GREEN_MINUS_RED), str(PeriodMode.GREEN_PLUS_RED)}
            and isinstance(run.grouping.get("period_histograms"), list)
            and len(run.grouping.get("period_histograms", [])) == 2
        ):
            asymmetry, error, dt_applied = self._compute_period_mode_asymmetry(
                run.grouping,
                period_mode,
                forward_idx,
                backward_idx,
                run_alpha,
                use_deadtime,
            )
        else:
            forward = apply_grouping(working_histograms, forward_idx)
            backward = apply_grouping(working_histograms, backward_idx)
            asymmetry, error = compute_asymmetry(forward, backward, alpha=run_alpha)
        asymmetry = asymmetry * 100.0
        error = error * 100.0
        time_axis = run.histograms[0].time_axis

        lo = max(0, first_good)
        hi = min(len(asymmetry) - 1, last_good)
        if lo <= hi:
            time_out = time_axis[lo : hi + 1].copy()
            asym_out = asymmetry[lo : hi + 1].copy()
            err_out = error[lo : hi + 1].copy()
            if bunch_factor > 1:
                time_out, asym_out, err_out = rebin(time_out, asym_out, err_out, bunch_factor)
            dataset.time = time_out
            dataset.asymmetry = asym_out
            dataset.error = err_out

        run.grouping.update(
            {
                "groups": groups,
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "alpha": float(run_alpha),
                "first_good_bin": first_good,
                "last_good_bin": last_good,
                "bunching_factor": bunch_factor,
                "deadtime_correction": use_deadtime,
                "period_mode": period_mode,
            }
        )
        return True, dt_applied

    def _compute_period_mode_asymmetry(
        self,
        grouping: dict,
        period_mode: str,
        forward_idx: list[int],
        backward_idx: list[int],
        alpha: float,
        use_deadtime: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Compute two-period combined asymmetry directly in asymmetry space."""
        period_hist = grouping.get("period_histograms") if isinstance(grouping, dict) else None
        if not isinstance(period_hist, list) or len(period_hist) != 2:
            raise ValueError("Expected two period histogram sets for period-mode asymmetry")

        period_good_frames = grouping.get("period_good_frames") if isinstance(grouping, dict) else None
        period_dead_time = grouping.get("period_dead_time_us") if isinstance(grouping, dict) else None

        def _period_meta(index: int) -> dict:
            good_frames = 1.0
            if isinstance(period_good_frames, list) and len(period_good_frames) > index:
                try:
                    good_frames = float(period_good_frames[index])
                except (TypeError, ValueError):
                    good_frames = 1.0
            dead_time_us: list[float] = []
            if isinstance(period_dead_time, list) and len(period_dead_time) > index:
                raw = period_dead_time[index]
                if isinstance(raw, list):
                    dead_time_us = [float(v) for v in raw]
            return {"good_frames": good_frames, "dead_time_us": dead_time_us}

        red_hist = period_hist[0] if isinstance(period_hist[0], list) else []
        green_hist = period_hist[1] if isinstance(period_hist[1], list) else []
        red_working, red_dt = self._prepare_grouping_histograms(red_hist, _period_meta(0), use_deadtime)
        green_working, green_dt = self._prepare_grouping_histograms(green_hist, _period_meta(1), use_deadtime)

        red_forward = apply_grouping(red_working, forward_idx)
        red_backward = apply_grouping(red_working, backward_idx)
        green_forward = apply_grouping(green_working, forward_idx)
        green_backward = apply_grouping(green_working, backward_idx)

        red_asym, red_err = compute_asymmetry(red_forward, red_backward, alpha=alpha)
        green_asym, green_err = compute_asymmetry(green_forward, green_backward, alpha=alpha)

        if period_mode == str(PeriodMode.GREEN_PLUS_RED):
            combined_asym = green_asym + red_asym
        else:
            combined_asym = green_asym - red_asym
        combined_err = np.sqrt(np.square(green_err) + np.square(red_err))
        return combined_asym, combined_err, (red_dt or green_dt)

    def _select_histograms_for_period_mode(
        self,
        histograms,
        grouping: dict,
        period_mode: str,
    ):
        """Return histograms and deadtime metadata for the selected period mode."""
        period_hist = grouping.get("period_histograms") if isinstance(grouping, dict) else None
        period_good_frames = grouping.get("period_good_frames") if isinstance(grouping, dict) else None
        period_dead_time = grouping.get("period_dead_time_us") if isinstance(grouping, dict) else None
        if not isinstance(period_hist, list) or len(period_hist) != 2:
            return list(histograms), grouping

        red_hist = period_hist[0] if isinstance(period_hist[0], list) else list(histograms)
        green_hist = period_hist[1] if isinstance(period_hist[1], list) else list(histograms)

        red_good = 1.0
        green_good = 1.0
        if isinstance(period_good_frames, list) and len(period_good_frames) == 2:
            try:
                red_good = float(period_good_frames[0])
                green_good = float(period_good_frames[1])
            except (TypeError, ValueError):
                red_good = 1.0
                green_good = 1.0

        red_dt: list[float] = []
        green_dt: list[float] = []
        if isinstance(period_dead_time, list) and len(period_dead_time) == 2:
            red_dt = [float(v) for v in period_dead_time[0]] if isinstance(period_dead_time[0], list) else []
            green_dt = [float(v) for v in period_dead_time[1]] if isinstance(period_dead_time[1], list) else []

        selected_mode = period_mode
        if selected_mode == str(PeriodMode.GREEN):
            return list(green_hist), {
                **grouping,
                "good_frames": green_good,
                "dead_time_us": green_dt,
            }

        if selected_mode == str(PeriodMode.GREEN_MINUS_RED):
            return list(red_hist), {
                **grouping,
                "good_frames": max(green_good + red_good, 1.0),
                "dead_time_us": [0.0] * len(red_hist),
            }

        if selected_mode == str(PeriodMode.GREEN_PLUS_RED):
            return list(red_hist), {
                **grouping,
                "good_frames": max(green_good + red_good, 1.0),
                "dead_time_us": [0.0] * len(red_hist),
            }

        return list(red_hist), {
            **grouping,
            "good_frames": red_good,
            "dead_time_us": red_dt,
        }

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

        dead_time_us = grouping.get("dead_time_us") if isinstance(grouping, dict) else None
        if not isinstance(dead_time_us, list):
            return list(histograms), False
        if len(dead_time_us) < len(histograms):
            return list(histograms), False

        good_frames = 1.0
        try:
            good_frames = float(grouping.get("good_frames", 1.0))
        except (TypeError, ValueError):
            good_frames = 1.0
        if good_frames <= 0.0:
            good_frames = 1.0

        corrected: list[Histogram] = []
        applied_any = False
        for i, hist in enumerate(histograms):
            try:
                tau_us = float(dead_time_us[i])
            except (TypeError, ValueError):
                tau_us = 0.0

            counts = hist.counts
            if tau_us > 0.0:
                counts = self._apply_deadtime_correction(
                    counts,
                    tau_us,
                    hist.bin_width,
                    num_good_frames=good_frames,
                )
                applied_any = True

            corrected.append(
                Histogram(
                    counts=counts,
                    bin_width=hist.bin_width,
                    t0_bin=hist.t0_bin,
                    good_bin_start=hist.good_bin_start,
                    good_bin_end=hist.good_bin_end,
                )
            )

        return corrected, applied_any

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
        n = np.asarray(counts, dtype=np.float64)
        if tau_us <= 0.0 or bin_width_us <= 0.0 or num_good_frames <= 0.0:
            return n.copy()

        denom = 1.0 - (n * tau_us / (float(bin_width_us) * float(num_good_frames)))
        denom = np.clip(denom, 1.0e-6, None)
        return n / denom

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
        QMessageBox.about(
            self,
            "About Asymmetry",
            "Asymmetry v0.1.0\n\nA Python library for μSR data analysis.",
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
            self._plot_panel.plot_dataset(dataset)
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
            fit_curves[run_number] = (
                t_fit,
                y_fit,
                "Global Fit",
                component_curves,
                result,
                global_fit_function,
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

        # When multiple datasets are selected, overlay them all on the plot.
        if len(selected) > 1:
            self._plot_panel.plot_datasets(selected)
            run_labels = ", ".join(str(ds.run_label) for ds in selected)
            self.statusBar().showMessage(f"Viewing runs {run_labels}")

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
