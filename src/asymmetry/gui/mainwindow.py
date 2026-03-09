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

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

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
    """Top-level application window."""

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

        self._setup_menus()
        self._setup_toolbar()
        self._setup_panels()

        self.statusBar().showMessage("Ready")

    # ── menus ──────────────────────────────────────────────────────────

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        file_menu.addAction("&Open…", self._on_open)
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
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("Open", self._on_open)
        tb.addAction("Fit", self._on_fit)
        tb.addAction("FFT", self._on_fourier)
        tb.addAction("Params", self._on_fit_parameters)

    # ── panels / docks ─────────────────────────────────────────────────

    def _setup_panels(self) -> None:
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
        self._fit_panel.fit_completed.connect(self._on_fit_completed)
        self._fit_panel.global_fit_completed.connect(self._on_global_fit_completed)

        # Update selected datasets for global fitting whenever selection changes
        self._update_selected_datasets()

    # ── slots ──────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        from PySide6.QtWidgets import QFileDialog

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
        self._dock_fit.show()
        self._dock_fit.raise_()
        self._log_panel.log("Opened Fit panel")

    def _on_fourier(self) -> None:
        self._dock_fourier.show()
        self._dock_fourier.raise_()
        self._log_panel.log("Opened Fourier panel")

    def _on_fit_parameters(self) -> None:
        self._dock_fit_parameters.show()
        self._dock_fit_parameters.raise_()
        self._log_panel.log("Opened Fit Parameters panel")

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Asymmetry",
            "Asymmetry v0.1.0\n\nA Python library for μSR data analysis.",
        )

    def _reset_layout(self) -> None:
        pass  # TODO: restore default dock positions

    def _on_dataset_selected(self, run_number: int) -> None:
        """Handle dataset selection from data browser."""
        dataset = self._data_browser.get_dataset(run_number)
        if dataset:
            self._current_dataset = dataset
            # Clear any previous fit curve when switching datasets
            self._plot_panel._fit_curve = None
            self._plot_panel.plot_dataset(dataset)
            self._fit_panel.set_dataset(dataset)
            self._log_panel.log(f"Selected run {run_number}")
            self.statusBar().showMessage(f"Viewing run {run_number}")

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
        self._fit_panel.set_datasets(selected)
