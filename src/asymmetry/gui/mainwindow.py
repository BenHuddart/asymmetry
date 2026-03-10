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

_MAX_RECENT_PROJECTS = 10
_PROJECT_FILE_FILTER = "Asymmetry projects (*.asymp);;All files (*)"

from asymmetry.gui.panels.data_browser import DataBrowserPanel
from asymmetry.gui.panels.fit_panel import FitPanel
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.panels.plot_panel import PlotPanel


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
    * Plot panel bunch factor, axis limits, and fit-curve overlay
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

        self._setup_menus()
        self._setup_toolbar()
        self._setup_panels()

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
        tb.addAction("Fit", self._on_fit)
        tb.addAction("FFT", self._on_fourier)
        tb.addAction("Params", self._on_fit_parameters)

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
        self._data_browser.selection_changed.connect(self._update_selected_datasets)
        self._plot_panel.bunch_factor_changed.connect(self._on_bunch_factor_changed)
        self._fit_panel.fit_completed.connect(self._on_fit_completed)
        self._fit_panel.global_fit_completed.connect(self._on_global_fit_completed)

        # Update selected datasets for global fitting whenever selection changes
        self._update_selected_datasets()

    # ── slots ──────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        """Prompt the user to select one or more .wim data files and load them."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open μSR data files",
            self._last_open_dir,
            "WiMDA files (*.wim);;All files (*)",
        )
        if paths:
            selected_dir = os.path.dirname(paths[0])
            if selected_dir:
                self._last_open_dir = selected_dir
                self._settings.setValue("io/last_open_dir", selected_dir)
            self._load_files(paths)

    def _load_files(self, paths: list[str]) -> None:
        """Load multiple data files."""
        successful = 0
        failed = 0
        last_dataset = None
        apply_comment_field_to_all = False

        for path in paths:
            try:
                dataset = self._load_file(path)
                if dataset is None:
                    continue

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

                self._data_browser.add_dataset(dataset)
                self._log_panel.log(f"Loaded {path}")
                if dataset:
                    last_dataset = dataset
                    successful += 1
            except Exception as e:
                self._log_panel.log(f"ERROR loading {path}: {e}")
                failed += 1

        # Plot the last successfully loaded dataset
        if last_dataset:
            # Clear any previous fit curve when loading new data
            self._plot_panel._fit_curve = None
            self._plot_panel._fit_curves = {}
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

    def _load_file(self, path: str) -> None:
        from asymmetry.core.io import load

        dataset = load(path)
        return dataset

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
        dataset = self._data_browser.get_dataset(run_number)
        if dataset:
            self._current_dataset = dataset
            # Clear any previous fit curve when switching datasets
            self._plot_panel._fit_curve = None
            self._plot_panel.plot_dataset(dataset)
            self._fit_panel.set_dataset(self._plot_panel.get_analysis_dataset(dataset))
            self._log_panel.log(f"Selected run {run_number}")
            self.statusBar().showMessage(f"Viewing run {run_number}")

    def _on_bunch_factor_changed(self, _factor: int) -> None:
        """Refresh fit inputs so fitting follows the current bunch factor."""
        if self._current_dataset is not None:
            current_run = self._current_dataset.run_number
            if self._data_browser.get_dataset(current_run) is not None:
                self._fit_panel.set_dataset(
                    self._plot_panel.get_analysis_dataset(self._current_dataset)
                )
        self._update_selected_datasets()

    def _on_fit_completed(self, fit_result, fitted_curve) -> None:
        """Handle completed fit from fit panel."""
        t_fit, y_fit = fitted_curve
        self._plot_panel.plot_fit(t_fit, y_fit, label="Fit")
        self._log_panel.log(
            f"Fit completed: χ²ᵣ = {fit_result.reduced_chi_squared:.4f}"
        )

    def _on_global_fit_completed(self, results_dict, global_params) -> None:
        """Handle completed global fit.

        Parameters
        ----------
        results_dict : dict
            Dictionary mapping run_number -> (FitResult, fitted_curve_tuple).
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

        # Store fit curves for all datasets
        fit_curves = {}
        for run_number, (result, fitted_curve) in normalized_results.items():
            t_fit, y_fit = fitted_curve
            fit_curves[run_number] = (t_fit, y_fit, "Global Fit")

        # Set all fit curves in plot panel
        self._plot_panel.set_global_fits(fit_curves)

        # Push fitted parameters into the trends panel.
        datasets_by_run = {}
        for run_number in normalized_results:
            dataset = self._data_browser.get_dataset(run_number)
            if dataset is not None:
                datasets_by_run[run_number] = dataset
        self._fit_parameters_panel.set_fit_results(
            normalized_results, datasets_by_run, global_params,
        )
        self._dock_fit_parameters.show()
        self._dock_fit_parameters.raise_()

        # Log summary
        successful_results = [
            payload for payload in normalized_results.values()
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

        avg_chi2r = sum(r.reduced_chi_squared for r, _ in successful_results) / n_datasets
        self._log_panel.log(
            f"Global fit completed: {n_datasets} datasets, "
            f"average χ²ᵣ = {avg_chi2r:.3f}"
        )
        self.statusBar().showMessage(f"Global fit completed for {n_datasets} datasets")

    def _update_selected_datasets(self) -> None:
        """Update the fit panel with currently selected datasets."""
        selected = self._data_browser.get_selected_datasets()
        if self._current_dataset is not None:
            current_run = self._current_dataset.run_number
            if self._data_browser.get_dataset(current_run) is None:
                self._current_dataset = None
                self._plot_panel.clear()
                self._fit_panel.set_dataset(None)
        analysis_datasets = [
            dataset
            for dataset in (
                self._plot_panel.get_analysis_dataset(ds) for ds in selected
            )
            if dataset is not None
        ]
        self._fit_panel.set_datasets(analysis_datasets)

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
                dataset = self._load_file(resolved)
                if dataset is None:
                    continue
                # Apply saved metadata overrides without prompting.
                for key, val in ds_info.get("metadata_overrides", {}).items():
                    dataset.metadata[key] = val
                    if dataset.run:
                        dataset.run.metadata[key] = val
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
            self._fit_panel.set_dataset(
                self._plot_panel.get_analysis_dataset(current_dataset)
            )
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
