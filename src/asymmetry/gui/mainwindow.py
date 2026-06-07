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
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QActionGroup, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import Histogram, MuonDataset
from asymmetry.core.fitting import build_grouped_time_domain_datasets, fit_result_summary
from asymmetry.core.fourier import (
    GroupSpectrumConfig,
    build_group_signal_dataset,
    canonical_fourier_display_mode,
    compute_average_group_spectrum,
    estimate_fft_phase,
    fft_complex_asymmetry,
    fourier_mode_uses_phase_correction,
)
from asymmetry.core.io.periods import (
    combine_period_asymmetry,
    select_period_histograms,
)
from asymmetry.core.maxent import (
    MaxEntCancelledError,
    MaxEntConfig,
    MaxEntState,
    estimate_maxent_workload,
    maxent,
)
from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    load_project,
    save_project,
)
from asymmetry.core.representation import (
    FitSeries,
    FitSlot,
    RepresentationType,
    canonical_model_matches,
)
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.transform import (
    apply_deadtime_correction,
    apply_grouped_background_correction,
    apply_grouping_aligned,
    build_field_scan,
    common_t0_for_groups,
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
    has_file_deadtime,
    has_resolved_deadtime,
    prepare_histograms_with_deadtime,
    supports_background_correction,
)
from asymmetry.core.transform.rebin import rebin
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    PeriodMode,
)
from asymmetry.gui.export_paths import default_export_path, remember_export_path
from asymmetry.gui.gle_settings import GleSetupDialog
from asymmetry.gui.panels.alc_panel import ALCFitPanel, ALCScanView
from asymmetry.gui.panels.data_browser import DataBrowserPanel
from asymmetry.gui.panels.fit_panel import FitPanel
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.panels.maxent_panel import MaxEntPanel
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.panels.plot_workspace_panel import PlotWorkspacePanel
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import header_font
from asymmetry.gui.styles.widgets import build_segmented_button_qss
from asymmetry.gui.ui_manager import (
    UI_SCALE_OPTIONS,
    UI_SCALE_SETTINGS_KEY,
    UIManager,
)
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from asymmetry.gui.windows.grouping_dialog import GroupingDialog
from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow
from asymmetry.gui.windows.run_info_dialog import RunInfoDialog

_MAX_RECENT_PROJECTS = 10
_PROJECT_FILE_FILTER = "Asymmetry projects (*.asymp);;All files (*)"
_COMPACT_MODE_SETTINGS_KEY = "ui/compact_mode"
_UI_SCALE_SETTINGS_KEY = UI_SCALE_SETTINGS_KEY
_UI_SCALE_OPTIONS = UI_SCALE_OPTIONS
_VIEW_MODE_COUNT = 3
_PERF_LOGGING_SETTINGS_KEY = "debug/perf_logging"
_PERF_LOGGING_ENV_VAR = "ASYMMETRY_PERF_LOGGING"
_PLOT_DECIMATION_SETTINGS_KEY = "plot/enable_decimation"
_MAXENT_WARN_PEAK_MATRIX_BYTES = 1 * 1024**3
_MAXENT_WARN_TOTAL_MATRIX_BYTES = 8 * 1024**3
_MAXENT_WARN_TOTAL_OBSERVATIONS = 500_000


def _normalise_source_path(path: str) -> str:
    """Return a canonical string for source-file path comparisons."""
    return os.path.normcase(os.path.abspath(os.path.realpath(path)))


def _format_bytes(num_bytes: int) -> str:
    """Return a compact human-readable byte count."""
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


class MaxEntWorker(QObject):
    """Background worker for MaxEnt calculation."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        run,
        config: MaxEntConfig,
        *,
        cycles: int,
        state: MaxEntState | None,
    ) -> None:
        super().__init__()
        self._run = run
        self._config = config
        self._cycles = int(cycles)
        self._state = state
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._cancel_requested = True

    def run(self) -> None:
        """Run the MaxEnt calculation and emit one terminal signal."""
        try:
            result = maxent(
                self._run,
                self._config,
                cycles=self._cycles,
                state=self._state,
                progress_callback=self.progress.emit,
                cancel_callback=lambda: bool(self._cancel_requested),
            )
        except MaxEntCancelledError:
            self.cancelled.emit()
        except Exception as exc:  # pragma: no cover - exercised through GUI smoke tests
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)


def _load_window_icon() -> QIcon | None:
    """Load window icon from package resources.

    Returns None if icon cannot be loaded.
    """
    from asymmetry.gui.app import _icon_from_pixmap, _load_resource_pixmap, _macos_icon_pixmap

    # Try importlib.resources (preferred for installed packages)
    try:
        from importlib.resources import files

        logo = files("asymmetry.resources").joinpath("logo_256x256.png")
        if logo.is_file():
            pixmap = QPixmap()
            if pixmap.loadFromData(logo.read_bytes(), "PNG"):
                icon = _icon_from_pixmap(_macos_icon_pixmap(pixmap))
                if icon is not None and not icon.isNull():
                    return icon
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError, OSError):
        pass

    icon = _icon_from_pixmap(_macos_icon_pixmap(_load_resource_pixmap("logo_256x256.png")))
    if icon is not None:
        return icon

    # Fallback: try direct path (for development)
    try:
        resources_dir = Path(__file__).parent.parent / "resources"
        icon_path = resources_dir / "logo_256x256.png"
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            icon = _icon_from_pixmap(_macos_icon_pixmap(pixmap))
            if icon is not None and not icon.isNull():
                return icon
    except (OSError, ValueError):
        pass

    return None


def _coerce_bool(value: object, default: bool = False) -> bool:
    """Return a conservative bool conversion for values loaded from QSettings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


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
    data file must remain accessible at its original path (or at the same
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

    Previously not saved (now persisted where available):

    * Raw asymmetry / time / error arrays (reloaded from source files)
    * Fit *result* statistics (χ², uncertainties) — only the fitted
      parameter values that were written back to the parameter table
    * Fourier transform output (cached spectra are now saved when computed)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Asymmetry — μSR Data Analysis")

        self.compact_mode = False

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
        # Recipe-only representation/batch state for the redesign.  Frequency
        # spectra are recomputed from FrequencyFFT recipes on load; the
        # Frequency spectra are cached by representation.  _frequency_spectra_by_run
        # remains the FFT cache alias for existing tests and transitional project
        # fallbacks.
        self._project_model = ProjectModel()
        self._next_batch_index = 1
        self._next_scan_index = 1
        self._alc_mode = False
        self._frequency_spectra_by_run: dict[int, list[MuonDataset]] = {}
        self._frequency_spectra_by_rep: dict[RepresentationType, dict[int, list[MuonDataset]]] = {
            RepresentationType.FREQ_FFT: self._frequency_spectra_by_run,
            RepresentationType.FREQ_MAXENT: {},
        }
        self._maxent_state_by_run: dict[int, MaxEntState] = {}
        self._maxent_panel_state_by_run: dict[int, dict] = {}
        self._maxent_thread: QThread | None = None
        self._maxent_worker: MaxEntWorker | None = None
        self._maxent_active_run_number: int | None = None
        self._maxent_active_run = None
        self._maxent_active_config: MaxEntConfig | None = None
        self._maxent_active_cycles: int = 0
        self._maxent_started_at: float | None = None
        # Frequency representation the fit-panel datasets were last collected
        # from, and its snapshot at global-fit launch.  The async completion
        # handler resolves run datasets against the LAUNCH snapshot: the
        # collection pin is refreshed by view/selection changes and would
        # otherwise drift mid-fit.
        self._last_frequency_fit_rep_type: RepresentationType | None = None
        self._active_global_fit_rep_type: RepresentationType | None = None
        self._fourier_group_phase_state_by_run: dict[int, dict[str, object]] = {}
        self._global_parameter_fit_window: GlobalParameterFitWindow | None = None
        self._multi_group_fit_window: MultiGroupFitWindow | None = None
        self._fit_stack: QStackedWidget | None = None
        self._ui_scale_action_group = QActionGroup(self)
        self._ui_scale_action_group.setExclusive(True)
        self._ui_scale_actions: dict[float, object] = {}
        self._view_modes = [self._default_view_mode_state() for _ in range(_VIEW_MODE_COUNT)]
        self._active_view_mode_index = 0
        self._syncing_view_mode_ui = False
        self._applying_view_mode = False
        self._syncing_bunch_context = False
        self._applying_inspector_domain = False

        self._setup_menus()
        self._create_toolbars()
        self._create_docks()
        self._ui_manager = UIManager(self)
        self._connect_actions()
        self._ui_manager.restore_settings()
        self._restore_plot_ranges_from_settings()

        # Check for SciPy availability and warn if using fallback
        from asymmetry.core.fitting.diffusion import is_scipy_available

        if not is_scipy_available():
            self._log_panel.log(
                "⚠️  WARNING: SciPy is unavailable or broken. "
                "Diffusion model will use slower NumPy fallback for numerical integration. "
                "Please repair SciPy in your Python environment."
            )

        self._update_status_selection()

    def _perf_logging_is_enabled(self) -> bool:
        """Return whether lightweight GUI performance logging is enabled."""
        env_value = os.environ.get(_PERF_LOGGING_ENV_VAR)
        if env_value is not None:
            return _coerce_bool(env_value, default=False)
        return _coerce_bool(self._settings.value(_PERF_LOGGING_SETTINGS_KEY), default=False)

    def _plot_decimation_is_enabled(self) -> bool:
        """Return whether plot display decimation is enabled."""
        return _coerce_bool(self._settings.value(_PLOT_DECIMATION_SETTINGS_KEY), default=True)

    def _perf_dataset_metrics(
        self, datasets: MuonDataset | list[MuonDataset] | None
    ) -> dict[str, int]:
        """Return compact metrics for one or more datasets."""
        if datasets is None:
            items: list[MuonDataset] = []
        elif isinstance(datasets, list):
            items = [dataset for dataset in datasets if dataset is not None]
        else:
            items = [datasets]

        points = 0
        histograms = 0
        histogram_bins = 0
        for dataset in items:
            points += int(getattr(dataset, "n_points", len(getattr(dataset, "time", []))))
            run = getattr(dataset, "run", None)
            if run is None:
                continue
            run_histograms = list(getattr(run, "histograms", []) or [])
            histograms += len(run_histograms)
            histogram_bins += sum(len(hist.counts) for hist in run_histograms)

        return {
            "datasets": len(items),
            "points": points,
            "histograms": histograms,
            "histogram_bins": histogram_bins,
        }

    def _log_perf_event(self, event: str, started_at: float, **fields: object) -> None:
        """Append a PERF log entry when debug timing is enabled."""
        if not self._perf_logging_is_enabled():
            return

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        detail_parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
        detail_text = f" ({', '.join(detail_parts)})" if detail_parts else ""
        self._log_panel.log(f"PERF {event}: {elapsed_ms:.1f} ms{detail_text}")

    # ── menus ──────────────────────────────────────────────────────────

    def _setup_menus(self) -> None:
        """Build the application menu bar."""
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
        scale_menu = view_menu.addMenu("UI Scale")
        for scale in _UI_SCALE_OPTIONS:
            action = scale_menu.addAction(f"{int(round(scale * 100))}%")
            action.setCheckable(True)
            self._ui_scale_action_group.addAction(action)
            self._ui_scale_actions[scale] = action
        view_menu.addSeparator()
        view_menu.addAction("Show Data", self._on_show_data)
        view_menu.addAction("Show Fit", self._on_fit)
        view_menu.addAction("Show Fourier", self._on_fourier)
        view_menu.addAction("Show Fit Parameters", self._on_fit_parameters)
        view_menu.addAction("Show Log", self._on_show_log)

        # Options
        options_menu = mb.addMenu("&Options")
        self._use_temperature_from_log_action = options_menu.addAction("Use temperature from log")
        self._use_temperature_from_log_action.setCheckable(True)
        self._use_temperature_from_log_action.toggled.connect(
            self._on_use_temperature_from_log_toggled
        )
        self._perf_logging_action = options_menu.addAction("Enable performance logging")
        self._perf_logging_action.setCheckable(True)
        self._perf_logging_action.setChecked(self._perf_logging_is_enabled())
        self._perf_logging_action.toggled.connect(self._on_perf_logging_toggled)
        self._plot_decimation_action = options_menu.addAction("Enable plot decimation")
        self._plot_decimation_action.setCheckable(True)
        self._plot_decimation_action.setChecked(self._plot_decimation_is_enabled())
        self._plot_decimation_action.toggled.connect(self._on_plot_decimation_toggled)

        # Setup
        setup_menu = mb.addMenu("&Setup")
        setup_menu.addAction("GLE Setup…", self._on_gle_setup)

        # Help
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("&About…", self._on_about)

    # ── toolbar ────────────────────────────────────────────────────────

    def _create_toolbars(self) -> None:
        """Create the primary toolbar."""
        self._main_toolbar = QToolBar("Main")
        self._main_toolbar.setMovable(False)
        self.addToolBar(self._main_toolbar)

        self._main_toolbar.addAction("Open", self._on_open)
        self._main_toolbar.addAction("Export logbook", self._on_export_logbook)
        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction("Grouping", self._on_grouping_current)
        self._main_toolbar.addAction("Fit", self._on_fit)
        self._main_toolbar.addAction("FFT", self._on_fourier)
        self._main_toolbar.addAction("Params", self._on_fit_parameters)
        self._global_parameter_fit_toolbar_action = self._main_toolbar.addAction(
            "Global Fit", self._on_global_parameter_fit
        )
        self._global_parameter_fit_toolbar_action.setEnabled(False)
        # ALC mode: integral-asymmetry field scan. Enabled only for the F-B
        # asymmetry representation (see _on_plot_workspace_view_changed).
        self._alc_mode_action = self._main_toolbar.addAction("ALC mode")
        self._alc_mode_action.setCheckable(True)
        self._alc_mode_action.setEnabled(False)
        self._alc_mode_action.setToolTip(
            "Integral-asymmetry field scan (ALC / repolarisation / QLCR). "
            "Available for the Forward-Backward asymmetry representation."
        )
        self._alc_mode_action.toggled.connect(self._on_alc_mode_toggled)
        self._main_toolbar.addSeparator()

        # Domain → representation segmented control, grouped 2 + 2 under
        # "Time" and "Frequency" headers.  One exclusive group spans all four
        # buttons so only a single representation is ever active.
        self._domain_button_group = QButtonGroup(self)
        self._domain_button_group.setExclusive(True)
        self._domain_buttons: list[QPushButton] = []
        _domain_qss = build_segmented_button_qss()

        def _domain_cluster(header: str, specs: list[tuple[str, str]]) -> QWidget:
            container = QWidget()
            column = QVBoxLayout(container)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(1)
            heading = QLabel(header)
            heading.setFont(header_font())
            heading.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            column.addWidget(heading)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            for label, token in specs:
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setStyleSheet(_domain_qss)
                btn.clicked.connect(
                    lambda _checked=False, v=token: self._on_domain_button_clicked(v)
                )
                self._domain_button_group.addButton(btn)
                self._domain_buttons.append(btn)
                row.addWidget(btn)
            column.addLayout(row)
            return container

        self._main_toolbar.addWidget(
            _domain_cluster(
                "Time domain",
                [("F-B asymmetry", "fb_asymmetry"), ("Individual groups", "groups")],
            )
        )
        self._main_toolbar.addSeparator()
        self._main_toolbar.addWidget(
            _domain_cluster(
                "Frequency domain",
                [("FFT", "frequency"), ("MaxEnt", "maxent")],
            )
        )
        self._domain_buttons[0].setChecked(True)
        self._domain_buttons[3].setEnabled(False)
        self._domain_buttons[3].setToolTip("Maximum-entropy spectra from grouped counts")

        # Stretch spacer — pushes View / Bunch to the right edge
        _stretch = QWidget()
        _stretch.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._main_toolbar.addWidget(_stretch)

        self._main_toolbar.addWidget(QLabel("View:"))
        self._view_mode_button_group = QButtonGroup(self)
        self._view_mode_button_group.setExclusive(True)
        self._view_mode_buttons: list[QPushButton] = []
        _view_qss = build_segmented_button_qss(min_width=28, padding_h=6)
        for index in range(_VIEW_MODE_COUNT):
            button = QPushButton(str(index + 1))
            button.setCheckable(True)
            button.setStyleSheet(_view_qss)
            button.clicked.connect(
                lambda _checked=False, idx=index: self._on_view_mode_button_clicked(idx)
            )
            self._view_mode_button_group.addButton(button, index)
            self._view_mode_buttons.append(button)
            self._main_toolbar.addWidget(button)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addWidget(QLabel("Bunch:"))
        self._view_bunch_spin = QSpinBox()
        self._view_bunch_spin.setRange(1, 1000)
        self._view_bunch_spin.setMaximumWidth(70)
        self._view_bunch_spin.valueChanged.connect(self._on_main_bunch_factor_changed)
        self._main_toolbar.addWidget(self._view_bunch_spin)

        self._set_view_mode_button_states(self._active_view_mode_index)
        self._set_view_bunch_spin_value(
            self._view_modes[self._active_view_mode_index]["bunch_factor"]
        )

    def _setup_toolbar(self) -> None:
        """Backward-compatible wrapper for older tests/tools."""
        self._create_toolbars()

    @staticmethod
    def _default_view_mode_state() -> dict[str, float | int]:
        """Return the default persisted state for one main-window view mode."""
        return {
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
        }

    def _set_view_mode_button_states(self, active_index: int) -> None:
        """Highlight the currently active saved view mode."""
        self._syncing_view_mode_ui = True
        try:
            for index, button in enumerate(getattr(self, "_view_mode_buttons", [])):
                button.setChecked(index == active_index)
        finally:
            self._syncing_view_mode_ui = False

    def _set_view_bunch_spin_value(self, value: int) -> None:
        """Update the visible bunch-factor spinbox without re-entering handlers."""
        if not hasattr(self, "_view_bunch_spin"):
            return
        previous = self._view_bunch_spin.blockSignals(True)
        self._view_bunch_spin.setValue(max(1, int(value)))
        self._view_bunch_spin.blockSignals(previous)

    def _snapshot_active_view_mode(self) -> None:
        """Store the current bunch factor and axis limits into the active mode."""
        if not self._view_modes:
            return
        mode = self._view_modes[self._active_view_mode_index]
        if hasattr(self, "_view_bunch_spin"):
            mode["bunch_factor"] = max(1, int(self._view_bunch_spin.value()))
        if hasattr(self, "_plot_panel") and hasattr(self._plot_panel, "get_view_limits"):
            x_min, x_max, y_min, y_max = self._plot_panel.get_view_limits()
            mode["x_min"] = float(x_min)
            mode["x_max"] = float(x_max)
            mode["y_min"] = float(y_min)
            mode["y_max"] = float(y_max)

    def _collect_view_modes_state(self) -> dict[str, object]:
        """Return a serializable payload for the saved main-window view modes."""
        self._snapshot_active_view_mode()
        return {
            "active_index": int(self._active_view_mode_index),
            "modes": [dict(mode) for mode in self._view_modes],
        }

    def _restore_view_modes_state(self, state: object) -> None:
        """Restore persisted main-window view modes from project state."""
        restored_modes = [self._default_view_mode_state() for _ in range(_VIEW_MODE_COUNT)]
        active_index = 0

        if isinstance(state, dict):
            raw_modes = state.get("modes")
            if isinstance(raw_modes, list):
                for index in range(min(len(raw_modes), _VIEW_MODE_COUNT)):
                    raw_mode = raw_modes[index]
                    if not isinstance(raw_mode, dict):
                        continue
                    mode = self._default_view_mode_state()
                    for key, default in mode.items():
                        raw_value = raw_mode.get(key, default)
                        try:
                            mode[key] = (
                                int(raw_value) if key == "bunch_factor" else float(raw_value)
                            )
                        except (TypeError, ValueError):
                            mode[key] = default
                    mode["bunch_factor"] = max(1, int(mode["bunch_factor"]))
                    restored_modes[index] = mode
            try:
                active_index = int(state.get("active_index", 0))
            except (TypeError, ValueError):
                active_index = 0

        self._view_modes = restored_modes
        self._active_view_mode_index = max(0, min(_VIEW_MODE_COUNT - 1, active_index))
        self._set_view_mode_button_states(self._active_view_mode_index)
        self._set_view_bunch_spin_value(
            self._view_modes[self._active_view_mode_index]["bunch_factor"]
        )

    def _resolve_view_bunch_targets(self) -> tuple[list[MuonDataset], set[int]]:
        """Return datasets that should receive a bunch-factor update."""
        resolved: list[MuonDataset] = []
        seen_run_numbers: set[int] = set()
        combined_targets: set[int] = set()

        for dataset in self._selected_or_current_datasets():
            try:
                run_number = int(dataset.run_number)
            except (TypeError, ValueError):
                continue

            if (
                hasattr(self._data_browser, "is_combined_dataset")
                and self._data_browser.is_combined_dataset(run_number)
                and hasattr(self._data_browser, "get_combined_source_datasets")
            ):
                combined_targets.add(run_number)
                for source_dataset in self._data_browser.get_combined_source_datasets(run_number):
                    try:
                        source_run = int(source_dataset.run_number)
                    except (TypeError, ValueError):
                        continue
                    if source_run in seen_run_numbers:
                        continue
                    seen_run_numbers.add(source_run)
                    resolved.append(source_dataset)
                continue

            if run_number in seen_run_numbers:
                continue
            seen_run_numbers.add(run_number)
            resolved.append(dataset)

        return resolved, combined_targets

    def _apply_bunch_factor_to_context(self, bunch_factor: int) -> None:
        """Apply the visible bunch factor using grouping where possible."""
        bunch_factor = max(1, int(bunch_factor))
        if self._syncing_bunch_context:
            return

        self._syncing_bunch_context = True
        try:
            targets, combined_targets = self._resolve_view_bunch_targets()
            grouping_targets: list[tuple[MuonDataset, dict]] = []
            for dataset in targets:
                payload = self._extract_grouping_overrides(dataset)
                if isinstance(payload, dict):
                    payload["bunching_factor"] = bunch_factor
                    grouping_targets.append((dataset, payload))
                    continue

                run = getattr(dataset, "run", None)
                grouping = getattr(run, "grouping", None)
                if isinstance(grouping, dict):
                    grouping["bunching_factor"] = bunch_factor

            if grouping_targets:
                updated = 0
                for dataset, payload in grouping_targets:
                    applied, _dt_applied = self._apply_grouping_settings_to_dataset(
                        dataset, payload
                    )
                    if applied:
                        updated += 1

                if hasattr(self._plot_panel, "set_bunch_factor"):
                    self._plot_panel.set_bunch_factor(1, emit_signal=False)

                if updated > 0:
                    for combined_run_number in combined_targets:
                        if not hasattr(self._data_browser, "rebuild_combined_dataset"):
                            continue
                        rebuilt_combined_dataset = self._data_browser.rebuild_combined_dataset(
                            combined_run_number
                        )
                        if (
                            rebuilt_combined_dataset is not None
                            and self._current_dataset is not None
                            and int(self._current_dataset.run_number) == int(combined_run_number)
                        ):
                            self._current_dataset = rebuilt_combined_dataset

                    if hasattr(self._data_browser, "_rebuild_table"):
                        self._data_browser._rebuild_table()
                    if self._current_dataset is not None:
                        self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))
                    self._render_current_selection_plot()
                    self._refresh_vector_axis_selector()
                    self._update_fit_block_state()
                    self._update_selected_datasets()
                return

            if hasattr(self._plot_panel, "set_bunch_factor"):
                self._plot_panel.set_bunch_factor(bunch_factor, emit_signal=False)
            if self._current_dataset is not None:
                self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))
            if targets:
                self._render_current_selection_plot()
            self._update_selected_datasets()
        finally:
            self._syncing_bunch_context = False

    def _apply_view_mode(self, mode_index: int) -> None:
        """Activate one saved view mode and redraw the current view."""
        mode = self._view_modes[mode_index]
        self._applying_view_mode = True
        try:
            self._set_view_bunch_spin_value(int(mode["bunch_factor"]))
            self._apply_bunch_factor_to_context(int(mode["bunch_factor"]))
            if hasattr(self._plot_panel, "set_view_limits"):
                self._plot_panel.set_view_limits(
                    float(mode["x_min"]),
                    float(mode["x_max"]),
                    float(mode["y_min"]),
                    float(mode["y_max"]),
                )
        finally:
            self._applying_view_mode = False

    def _on_view_mode_button_clicked(self, mode_index: int) -> None:
        """Switch between saved main-window view modes."""
        if self._syncing_view_mode_ui or mode_index == self._active_view_mode_index:
            return
        self._snapshot_active_view_mode()
        self._active_view_mode_index = mode_index
        self._set_view_mode_button_states(mode_index)
        self._apply_view_mode(mode_index)

    def _on_main_bunch_factor_changed(self, value: int) -> None:
        """Apply bunch-factor edits from the main toolbar and persist them into the active mode."""
        bunch_factor = max(1, int(value))
        self._view_modes[self._active_view_mode_index]["bunch_factor"] = bunch_factor
        if self._applying_view_mode:
            return
        self._apply_bunch_factor_to_context(bunch_factor)

    def _set_frequency_axis_relative_check(self, enabled: bool) -> None:
        """Synchronize the frequency-axis checkbox without re-entry."""
        check = getattr(self, "_frequency_axis_relative_check", None)
        if check is None:
            return
        previous = check.blockSignals(True)
        check.setChecked(bool(enabled))
        check.blockSignals(previous)

    def _on_plot_view_limits_changed(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """Persist live x/y limit edits into the active saved view mode."""
        if self._applying_view_mode:
            return
        mode = self._view_modes[self._active_view_mode_index]
        mode["x_min"] = float(x_min)
        mode["x_max"] = float(x_max)
        mode["y_min"] = float(y_min)
        mode["y_max"] = float(y_max)

    # ── panels / docks ─────────────────────────────────────────────────

    def _create_docks(self) -> None:
        """Create and dock all child panels, then connect inter-panel signals."""
        # Enable dock nesting for proper splitter behavior
        self.setDockNestingEnabled(True)

        # Central plot
        try:
            self._plot_panel = PlotPanel(domain="time")
        except TypeError:
            self._plot_panel = PlotPanel()
        try:
            self._frequency_plot_panel = PlotPanel(domain="frequency")
        except TypeError:
            self._frequency_plot_panel = PlotPanel()
        if hasattr(self._plot_panel, "set_decimation_enabled"):
            self._plot_panel.set_decimation_enabled(
                self._plot_decimation_is_enabled(),
                redraw=False,
            )
        if hasattr(self._frequency_plot_panel, "set_decimation_enabled"):
            self._frequency_plot_panel.set_decimation_enabled(
                self._plot_decimation_is_enabled(),
                redraw=False,
            )
        self._plot_workspace = PlotWorkspacePanel(
            time_panel=self._plot_panel,
            frequency_panel=self._frequency_plot_panel,
        )
        self._plot_workspace.active_domain_changed.connect(self._on_plot_workspace_domain_changed)
        if hasattr(self._plot_workspace, "active_view_changed"):
            self._plot_workspace.active_view_changed.connect(self._on_plot_workspace_view_changed)
        self._frequency_axis_relative_check = getattr(
            self._frequency_plot_panel, "_frequency_axis_relative_check", None
        )
        self.setCentralWidget(self._plot_workspace)

        # Left dock — data browser / logbook
        self._data_browser = DataBrowserPanel()
        self._dock_data_browser = QDockWidget("DATA BROWSER", self)
        self._dock_data_browser.setWidget(self._data_browser)
        self._dock_data_browser.setMinimumWidth(220)
        self._dock_data_browser.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_data_browser)

        # Right dock — fit controls
        self._fit_panel = FitPanel()
        self._multi_group_fit_window = MultiGroupFitWindow(self)
        self._multi_group_fit_window.grouped_fit_completed.connect(
            lambda grouped_datasets, results_dict: self._on_grouped_fit_completed(
                grouped_datasets,
                results_dict,
                fit_function=self._multi_group_fit_window.grouped_fit_formula_string()
                if self._multi_group_fit_window is not None
                else None,
            )
        )
        self._multi_group_fit_window.grouped_preview_requested.connect(
            lambda grouped_datasets, preview_curves: self._on_grouped_preview_requested(
                grouped_datasets,
                preview_curves,
                fit_function=self._multi_group_fit_window.grouped_fit_formula_string()
                if self._multi_group_fit_window is not None
                else None,
            )
        )
        # Bespoke ALC-mode build panel, swapped into the Fit dock when ALC mode
        # is on (see _sync_fit_dock_mode).
        self._alc_fit_panel = ALCFitPanel()
        self._fit_stack = QStackedWidget(self)
        self._fit_stack.addWidget(self._fit_panel)
        self._fit_stack.addWidget(self._multi_group_fit_window)
        self._fit_stack.addWidget(self._alc_fit_panel)
        self._dock_fit = QDockWidget("Fit", self)
        self._dock_fit.setWidget(self._fit_stack)
        self._dock_fit.setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fit)

        # Right dock — spectrum controls (FFT / MaxEnt, tabbed with fit)
        self._fourier_panel = FourierPanel()
        self._maxent_panel = MaxEntPanel()
        self._spectrum_stack = QStackedWidget(self)
        self._spectrum_stack.addWidget(self._fourier_panel)
        self._spectrum_stack.addWidget(self._maxent_panel)
        self._dock_fourier = QDockWidget("Spectrum", self)
        self._dock_fourier.setWidget(self._spectrum_stack)
        self._dock_fourier.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fourier)
        self.tabifyDockWidget(self._dock_fit, self._dock_fourier)

        # Right dock — fitted parameter trends (tabbed with fit/fourier). In ALC
        # mode the dock swaps to the bespoke scan view via _parameters_stack.
        self._fit_parameters_panel = FitParametersPanel()
        self._alc_scan_view = ALCScanView()
        self._parameters_stack = QStackedWidget(self)
        self._parameters_stack.addWidget(self._fit_parameters_panel)
        self._parameters_stack.addWidget(self._alc_scan_view)
        self._dock_fit_parameters = QDockWidget("Parameters", self)
        self._dock_fit_parameters.setWidget(self._parameters_stack)
        self._dock_fit_parameters.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fit_parameters)
        self.tabifyDockWidget(self._dock_fit, self._dock_fit_parameters)

        # Analysis docks are opened on demand from toolbar/menu actions.
        self._dock_fit.hide()
        self._dock_fourier.hide()
        self._dock_fit_parameters.hide()
        # Gate the FitSeries browser highlight on Parameters dock visibility.
        self._dock_fit_parameters.visibilityChanged.connect(
            self._on_parameters_dock_visibility_changed
        )

        # Bottom dock — log panel
        self._log_panel = LogPanel()
        self._dock_log = QDockWidget("LOG", self)
        self._dock_log.setWidget(self._log_panel)
        self._dock_log.setMinimumHeight(96)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_log)

        # ── Structured status bar ──────────────────────────────────────────
        _sb = self.statusBar()
        _sb.setContentsMargins(4, 0, 4, 0)

        self._status_sel_label = QLabel("")
        self._status_sel_label.setFont(mono_font(10.5))
        _sb.addPermanentWidget(self._status_sel_label, 1)

        self._status_coords_label = QLabel("")
        self._status_coords_label.setFont(mono_font(10.5))
        self._status_coords_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        _sb.addPermanentWidget(self._status_coords_label)

        self._last_fit_chi2: float | None = None

        # Set compact-friendly defaults while keeping the central plot dominant.
        self.resizeDocks([self._dock_data_browser], [360], Qt.Orientation.Horizontal)
        self.resizeDocks([self._dock_log], [140], Qt.Orientation.Vertical)

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
        if hasattr(self._frequency_plot_panel, "fit_range_changed"):
            self._frequency_plot_panel.fit_range_changed.connect(self._on_fit_range_changed)
        if hasattr(self._plot_panel, "cursor_coords_changed"):
            self._plot_panel.cursor_coords_changed.connect(self._on_cursor_coords_changed)
        if hasattr(self._fit_panel, "fit_range_edit_committed"):
            self._fit_panel.fit_range_edit_committed.connect(self._on_fit_range_edit_committed)
        if self._multi_group_fit_window is not None and hasattr(
            self._multi_group_fit_window, "fit_range_edit_committed"
        ):
            self._multi_group_fit_window.fit_range_edit_committed.connect(
                self._on_fit_range_edit_committed
            )
        if hasattr(self._plot_panel, "bunch_factor_changed"):
            self._plot_panel.bunch_factor_changed.connect(self._update_selected_datasets)
        if hasattr(self._plot_panel, "view_limits_changed"):
            self._plot_panel.view_limits_changed.connect(self._on_plot_view_limits_changed)
        if hasattr(self._plot_panel, "overlay_toggled"):
            self._plot_panel.overlay_toggled.connect(self._on_overlay_toggled)
        if hasattr(self._plot_panel, "time_view_changed"):
            self._plot_panel.time_view_changed.connect(self._on_plot_time_view_changed)
        if hasattr(self._plot_panel, "polarization_axis_changed"):
            self._plot_panel.polarization_axis_changed.connect(
                self._on_plot_polarization_axis_changed
            )
        self._fit_panel.fit_completed.connect(self._on_fit_completed)
        if hasattr(self._fit_panel, "global_fit_started"):
            self._fit_panel.global_fit_started.connect(self._on_global_fit_started)
        self._fit_panel.global_fit_completed.connect(self._on_global_fit_completed)
        self._alc_fit_panel.build_requested.connect(self._on_scan_requested)
        self._alc_fit_panel.fit_range_edit_committed.connect(self._on_fit_range_edit_committed)
        if hasattr(self._fit_panel, "grouped_fit_completed"):
            self._fit_panel.grouped_fit_completed.connect(self._on_grouped_fit_completed)
        if hasattr(self._fit_panel, "preview_requested"):
            self._fit_panel.preview_requested.connect(self._on_preview_requested)
        if hasattr(self._fit_panel, "share_function_with_group_requested"):
            self._fit_panel.share_function_with_group_requested.connect(
                self._on_share_single_function_with_group
            )
        if hasattr(self._fit_panel, "add_single_fit_to_series_requested"):
            self._fit_panel.add_single_fit_to_series_requested.connect(
                self._on_add_single_fit_to_series_requested
            )
        if hasattr(self._fit_parameters_panel, "cross_group_fit_completed"):
            self._fit_parameters_panel.cross_group_fit_completed.connect(
                self._on_cross_group_fit_completed
            )
        if hasattr(self._fit_parameters_panel, "delete_group_fits_requested"):
            self._fit_parameters_panel.delete_group_fits_requested.connect(
                self._on_fit_parameters_group_fits_deleted
            )
        if hasattr(self._fit_parameters_panel, "series_selection_changed"):
            self._fit_parameters_panel.series_selection_changed.connect(
                self._on_trend_series_selected
            )
        if hasattr(self._fit_parameters_panel, "series_rename_requested"):
            self._fit_parameters_panel.series_rename_requested.connect(
                self._on_series_rename_requested
            )
        if hasattr(self._fit_parameters_panel, "series_select_members_requested"):
            self._fit_parameters_panel.series_select_members_requested.connect(
                self._on_series_select_members_requested
            )
        if hasattr(self._fit_parameters_panel, "series_delete_requested"):
            self._fit_parameters_panel.series_delete_requested.connect(
                self._on_series_delete_requested
            )

        if hasattr(self._fourier_panel, "_fft_btn"):
            self._fourier_panel._fft_btn.clicked.connect(self._on_compute_fourier)
        if hasattr(self._fourier_panel, "_apply_to_selection_btn"):
            self._fourier_panel._apply_to_selection_btn.clicked.connect(
                self._on_apply_fourier_to_selection
            )
        if hasattr(self._fourier_panel, "_auto_phase_btn"):
            self._fourier_panel._auto_phase_btn.clicked.connect(self._on_fill_fourier_phases)
        if hasattr(self._maxent_panel, "_cycle_one_btn"):
            self._maxent_panel._cycle_one_btn.clicked.connect(lambda: self._on_compute_maxent(1))
        if hasattr(self._maxent_panel, "_cycle_five_btn"):
            self._maxent_panel._cycle_five_btn.clicked.connect(lambda: self._on_compute_maxent(5))
        if hasattr(self._maxent_panel, "_cycle_twentyfive_btn"):
            self._maxent_panel._cycle_twentyfive_btn.clicked.connect(
                lambda: self._on_compute_maxent(25)
            )
        if hasattr(self._maxent_panel, "_converge_btn"):
            self._maxent_panel._converge_btn.clicked.connect(lambda: self._on_compute_maxent(50))
        if hasattr(self._maxent_panel, "_restart_btn"):
            self._maxent_panel._restart_btn.clicked.connect(self._on_restart_maxent)
        if hasattr(self._maxent_panel, "_cancel_btn"):
            self._maxent_panel._cancel_btn.clicked.connect(self._on_cancel_maxent)
        if hasattr(self._maxent_panel, "_apply_to_selection_btn"):
            self._maxent_panel._apply_to_selection_btn.clicked.connect(
                self._on_apply_maxent_to_selection
            )

        # Update selected datasets for global fitting whenever selection changes
        self._update_selected_datasets()

    def _on_plot_workspace_domain_changed(self, _domain: str) -> None:
        """Refresh fit blocking when the active plot workspace tab changes."""
        if self._plot_workspace.active_domain() == "frequency":
            self._sync_frequency_plot_for_current_dataset()
            if hasattr(self._fit_panel, "set_domain"):
                self._fit_panel.set_domain("frequency")
            self._fit_panel.set_dataset(self._active_frequency_fit_dataset())
            self._set_frequency_fit_datasets_for_selection()
        else:
            if hasattr(self._fit_panel, "set_domain"):
                self._fit_panel.set_domain("time")
            if self._current_dataset is not None:
                self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))
                if self._multi_group_fit_window is not None:
                    self._multi_group_fit_window.set_dataset(
                        self._get_fit_dataset(self._current_dataset)
                    )
            self._update_selected_datasets()
        self._refresh_time_view_selector()
        self._refresh_vector_axis_selector()
        self._update_fit_block_state()

    def _setup_panels(self) -> None:
        """Backward-compatible wrapper for older tests/tools."""
        self._create_docks()

    def _connect_actions(self) -> None:
        """Connect actions created in menu/toolbar setup methods."""
        self._ui_manager.bind_actions()

    def _show_panel(self, panel_key: str) -> None:
        """Show a panel in the standard dock layout."""
        self._ui_manager.show_panel(panel_key)

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

    def _vector_axis_state_for_dataset(
        self, dataset
    ) -> tuple[dict[str, tuple[int, int]], str | None]:
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

        if hasattr(self, "_plot_workspace") and self._plot_workspace.active_domain() != "time":
            self._plot_panel.set_polarization_axes([])
            return
        if (
            hasattr(self._plot_panel, "current_time_view_mode")
            and self._plot_panel.current_time_view_mode() != "fb_asymmetry"
        ):
            self._plot_panel.set_polarization_axes([])
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
            current = (
                first_axis if first_axis in available else (available[0] if available else None)
            )
        self._plot_panel.set_polarization_axes(available, current)

    def _refresh_time_view_selector(self) -> None:
        """Keep the top-level plot tabs and internal time-view state in sync."""
        if not hasattr(self._plot_panel, "set_time_view_modes"):
            return

        modes = ["fb_asymmetry"]
        selected = list(self._data_browser.get_selected_datasets())
        if len(selected) > 1 and self._overlay_enabled():
            current_mode = None
            if hasattr(self._plot_panel, "current_time_view_mode"):
                current_mode = self._plot_panel.current_time_view_mode()
            self._plot_panel.set_time_view_modes(modes, current_mode=current_mode)
            if hasattr(self._plot_workspace, "set_available_views"):
                self._plot_workspace.set_available_views(modes)
                if self._plot_workspace.active_domain() == "time":
                    self._plot_workspace.set_active_view("fb_asymmetry")
            self._domain_buttons[1].setEnabled(False)
            self._domain_buttons[3].setEnabled(False)
            return

        target = self._current_dataset or (selected[0] if len(selected) == 1 else None)
        if target is not None and self._grouped_time_domain_display_datasets(target):
            modes.append("groups")
        if self._dataset_supports_maxent(target):
            modes.append("maxent")

        self._domain_buttons[1].setEnabled("groups" in modes)
        self._domain_buttons[3].setEnabled("maxent" in modes)

        current_mode = None
        if hasattr(self._plot_panel, "current_time_view_mode"):
            current_mode = self._plot_panel.current_time_view_mode()
        self._plot_panel.set_time_view_modes(modes, current_mode=current_mode)
        if hasattr(self._plot_workspace, "set_available_views"):
            self._plot_workspace.set_available_views(modes)
            if self._plot_workspace.active_domain() != "time":
                return
            active_view = current_mode if current_mode in modes else modes[0]
            if hasattr(self._plot_workspace, "active_view"):
                workspace_view = self._plot_workspace.active_view()
                if workspace_view in modes:
                    active_view = workspace_view
            self._plot_workspace.set_active_view(active_view)

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
        started_at = time.perf_counter()
        targets: list[MuonDataset] = []
        rendered_targets: list[MuonDataset] = []
        render_mode = "empty"
        try:
            targets = self._selected_or_current_datasets()
            rendered_targets = list(targets)
            if not targets:
                return

            if (
                hasattr(self._plot_panel, "current_time_view_mode")
                and self._plot_panel.current_time_view_mode() == "groups"
                and self._plot_workspace.active_domain() == "time"
                and len(targets) == 1
            ):
                grouped_targets = self._grouped_time_domain_display_datasets(targets[0])
                if grouped_targets and hasattr(
                    self._plot_panel, "plot_grouped_time_domain_subplots"
                ):
                    render_mode = "grouped_time"
                    rendered_targets = list(grouped_targets)
                    self._plot_panel.plot_grouped_time_domain_subplots(grouped_targets)
                    return

            if not self._overlay_enabled() and len(targets) > 1:
                chosen = self._select_non_overlay_target(targets)
                if chosen is None:
                    render_mode = "overlay_empty"
                    rendered_targets = []
                    return
                targets = [chosen]
                rendered_targets = list(targets)

            if len(targets) == 1:
                self._current_dataset = targets[0]

            active_axis = None
            if hasattr(self._plot_panel, "get_current_polarization_axis"):
                active_axis = self._normalize_vector_axis(
                    self._plot_panel.get_current_polarization_axis()
                )

            if active_axis == "ALL" and hasattr(self._plot_panel, "plot_vector_subplots"):
                axis_datasets = self._build_vector_axis_datasets(targets)
                if all(axis_datasets.get(axis) for axis in ("P_x", "P_y", "P_z")):
                    render_mode = "vector_all"
                    rendered_targets = [
                        dataset
                        for axis in ("P_x", "P_y", "P_z")
                        for dataset in axis_datasets.get(axis, [])
                    ]
                    self._plot_panel.plot_vector_subplots(axis_datasets)
                    return

            if len(targets) > 1:
                render_mode = "overlay"
                self._plot_panel.plot_datasets(targets)
                return

            render_mode = "single"
            self._plot_panel.plot_dataset(targets[0])
        finally:
            self._log_perf_event(
                "selection_plot",
                started_at,
                domain=self._plot_workspace.active_domain(),
                mode=render_mode,
                **self._perf_dataset_metrics(rendered_targets),
            )

    def _current_fit_block_state(self) -> tuple[bool, str]:
        """Return whether the current plot context should block fitting."""
        if hasattr(self, "_plot_workspace") and self._plot_workspace.active_domain() == "frequency":
            if self._active_frequency_fit_dataset() is None:
                return (
                    True,
                    f"Compute a {self._frequency_status_name()} spectrum for the active run before fitting in the frequency domain.",
                )
            return False, ""

        active_axis = None
        if hasattr(self._plot_panel, "get_current_polarization_axis"):
            active_axis = self._normalize_vector_axis(
                self._plot_panel.get_current_polarization_axis()
            )

        blocked = active_axis == "ALL"
        reason = "Vector All mode is ambiguous for fitting. Select x, y, or z before running a fit."
        return blocked, reason if blocked else ""

    def _update_fit_block_state(self) -> None:
        """Disable ambiguous fitting workflows when the current view is not fit-safe."""
        self._sync_fit_dock_mode()
        if (
            hasattr(self, "_fit_panel")
            and hasattr(self, "_plot_workspace")
            and hasattr(self._fit_panel, "set_domain")
        ):
            self._fit_panel.set_domain(self._plot_workspace.active_domain())
        blocked, reason = self._current_fit_block_state()
        if hasattr(self._fit_panel, "set_fit_blocked"):
            self._fit_panel.set_fit_blocked(blocked, reason)
        if self._multi_group_fit_window is not None:
            self._multi_group_fit_window.set_fit_blocked(blocked, reason)

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
        targets = (
            selected if selected else ([self._current_dataset] if self._current_dataset else [])
        )

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
        self._plot_workspace.export_current_plot()

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
            safe_stem = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem
            ).strip("_")
            if safe_stem:
                return f"{safe_stem}_logbook.tsv"
        return "logbook.tsv"

    def _load_files(self, paths: list[str]) -> None:
        """Load multiple data files."""
        started_at = time.perf_counter()
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

                        applied, _ = self._apply_grouping_settings_to_dataset(
                            dataset, grouping_payload
                        )
                        if applied:
                            auto_grouping_applied += 1

                    self._data_browser.add_dataset(dataset)
                    if dataset:
                        last_dataset = dataset
                        successful += 1
                else:
                    self._log_panel.log(f"Loaded {path}", tag="load")
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

        self._log_perf_event(
            "load_files_batch",
            started_at,
            files=len(paths),
            loaded=successful,
            failed=failed,
            auto_grouped=auto_grouping_applied,
            last_run=None if last_dataset is None else int(last_dataset.run_number),
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
        started_at = time.perf_counter()
        from asymmetry.core.io import load

        dataset = load(path)
        metrics_target: MuonDataset | list[MuonDataset] | None
        if dataset is None:
            metrics_target = None
        elif isinstance(dataset, list):
            metrics_target = dataset
        else:
            metrics_target = dataset
        self._log_perf_event(
            "load_file",
            started_at,
            file=Path(path).name,
            **self._perf_dataset_metrics(metrics_target),
        )
        return dataset

    def _on_get_info_requested(self, run_number: int) -> None:
        """Open run-information dialog for a selected dataset row."""
        dataset = self._data_browser.get_dataset(run_number)
        if dataset is None:
            return
        dialog = RunInfoDialog(
            dataset,
            self,
            included_fields=self._run_info_included_fields_for_dataset(run_number),
        )
        dialog.set_browser_field_inclusion_requested.connect(
            lambda field_key, include, rn=run_number: self._on_run_info_field_inclusion_changed(
                field_key,
                include,
                rn,
            )
        )
        dialog.exec()

    def _run_info_included_fields_for_dataset(self, run_number: int) -> set[str]:
        """Return Run Info checkbox state for a specific dataset."""
        included = set(self._data_browser.get_extra_columns())
        if self._data_browser.dataset_uses_temperature_from_log(run_number):
            included.add("temperature")
        else:
            included.discard("temperature")
        return included

    def _on_run_info_field_inclusion_changed(
        self,
        field_key: str,
        include: bool,
        run_number: int | None = None,
    ) -> None:
        """Apply include/exclude requests from the Run Info dialog."""
        if field_key == "temperature" and run_number is not None:
            self._data_browser.set_dataset_temperature_from_log(run_number, include)
            return
        if include:
            self._data_browser.add_extra_column(field_key)
        else:
            self._data_browser.remove_extra_column(field_key)
        if field_key == "temperature":
            self._sync_temperature_log_option_action()

    def _on_use_temperature_from_log_toggled(self, checked: bool) -> None:
        """Toggle Data Browser temperature display between header and log mean."""
        if not hasattr(self, "_data_browser"):
            return
        self._data_browser.set_use_temperature_from_log(checked)

    def _on_perf_logging_toggled(self, checked: bool) -> None:
        """Persist and report GUI performance logging state."""
        self._settings.setValue(_PERF_LOGGING_SETTINGS_KEY, bool(checked))
        state = "enabled" if checked else "disabled"
        self._log_panel.log(f"Performance logging {state}.")
        self.statusBar().showMessage(f"Performance logging {state}")

    def _on_plot_decimation_toggled(self, checked: bool) -> None:
        """Persist and apply the display-decimation policy for plot panels."""
        enabled = bool(checked)
        self._settings.setValue(_PLOT_DECIMATION_SETTINGS_KEY, enabled)
        if hasattr(self, "_plot_panel") and hasattr(self._plot_panel, "set_decimation_enabled"):
            self._plot_panel.set_decimation_enabled(enabled)
        if hasattr(self, "_frequency_plot_panel") and hasattr(
            self._frequency_plot_panel, "set_decimation_enabled"
        ):
            self._frequency_plot_panel.set_decimation_enabled(enabled)
        state = "enabled" if enabled else "disabled"
        self._log_panel.log(f"Plot decimation {state}.")
        self.statusBar().showMessage(f"Plot decimation {state}")

    def _sync_temperature_log_option_action(self) -> None:
        """Keep the Options menu temperature action aligned with browser state."""
        action = getattr(self, "_use_temperature_from_log_action", None)
        if action is None or not hasattr(self, "_data_browser"):
            return
        action.blockSignals(True)
        action.setChecked(self._data_browser.use_temperature_from_log())
        action.blockSignals(False)

    def _on_grouping_requested(self, run_number: int) -> None:
        """Open shared grouping dialog focused on a selected run."""
        selected_run_numbers = [
            int(ds.run_number) for ds in self._data_browser.get_selected_datasets()
        ]
        if int(run_number) not in selected_run_numbers:
            selected_run_numbers = [int(run_number)]
        self._open_shared_grouping_dialog(
            selected_run_number=run_number,
            selected_run_numbers=selected_run_numbers,
        )

    def _on_grouping_current(self) -> None:
        """Open shared grouping dialog for all datasets in the active project."""
        selected_run = (
            None if self._current_dataset is None else int(self._current_dataset.run_number)
        )
        selected_run_numbers = [
            int(ds.run_number) for ds in self._data_browser.get_selected_datasets()
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
            _reference_dataset,
            dialog_selected_run_number,
            dialog_selected_run_numbers,
            combined_target_run_number,
        ) = self._resolve_grouping_dialog_context(
            all_datasets=all_datasets,
            selected_run_number=selected_run_number,
            selected_run_numbers=selected_run_numbers,
        )

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
        if (
            updated > 0
            and combined_target_run_number is not None
            and hasattr(
                self._data_browser,
                "rebuild_combined_dataset",
            )
        ):
            rebuilt_combined_dataset = self._data_browser.rebuild_combined_dataset(
                combined_target_run_number
            )
            if rebuilt_combined_dataset is not None:
                first_updated_dataset = rebuilt_combined_dataset
                if self._current_dataset is not None and int(
                    self._current_dataset.run_number
                ) == int(combined_target_run_number):
                    self._current_dataset = rebuilt_combined_dataset
                    self._fit_panel.set_dataset(self._get_fit_dataset(rebuilt_combined_dataset))

        if updated > 0:
            bunch_factor = max(1, int(grouping_result.get("bunching_factor", 1)))
            self._view_modes[self._active_view_mode_index]["bunch_factor"] = bunch_factor
            self._set_view_bunch_spin_value(bunch_factor)
            if hasattr(self._plot_panel, "set_bunch_factor"):
                self._plot_panel.set_bunch_factor(1, emit_signal=False)
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
        dialog_selected_run_numbers = (
            list(selected_run_numbers) if selected_run_numbers is not None else None
        )
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
            "deadtime_correction": bool(grouping.get("deadtime_correction", False)),
            "background_correction": bool(grouping.get("background_correction", False)),
        }
        if "period_mode" in grouping:
            payload["period_mode"] = str(grouping.get("period_mode", PeriodMode.RED))
        if isinstance(grouping.get("period_good_frames"), list):
            payload["period_good_frames"] = list(grouping.get("period_good_frames", []))
        if isinstance(grouping.get("period_dead_time_us"), list):
            payload["period_dead_time_us"] = copy.deepcopy(grouping.get("period_dead_time_us", []))

        deadtime_mode = str(grouping.get("deadtime_mode", grouping.get("deadtime_method", "off")))
        if deadtime_mode == "load":
            deadtime_mode = "manual"
        payload["deadtime_mode"] = deadtime_mode
        if grouping.get("deadtime_method"):
            payload["deadtime_method"] = str(grouping.get("deadtime_method"))
        for key in (
            "deadtime_manual_us",
            "deadtime_estimated_us",
            "deadtime_reference_run",
        ):
            if key in grouping:
                payload[key] = grouping.get(key)

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
            payload["group_names"] = {int(k): str(v) for k, v in group_names_raw.items()}
        included_groups_raw = grouping.get("included_groups")
        if isinstance(included_groups_raw, dict) and included_groups_raw:
            payload["included_groups"] = {int(k): bool(v) for k, v in included_groups_raw.items()}

        grouping_preset = grouping.get("grouping_preset")
        if grouping_preset:
            payload["grouping_preset"] = str(grouping_preset)

        instrument_name = grouping.get("instrument")
        if instrument_name:
            payload["instrument"] = str(instrument_name)

        if deadtime_mode != "file":
            if isinstance(grouping.get("dead_time_us"), list):
                payload["dead_time_us"] = list(grouping.get("dead_time_us", []))
            elif isinstance(grouping.get("deadtime_loaded_us"), list):
                payload["dead_time_us"] = list(grouping.get("deadtime_loaded_us", []))
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
        """Return immutable source arrays for regrouping datasets without histograms."""
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

    def _apply_grouping_settings_to_dataset(
        self, dataset, grouping_result: dict
    ) -> tuple[bool, bool]:
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
            bin_index_base = (
                1
                if int(
                    grouping_result.get(
                        "bin_index_base", existing_grouping.get("bin_index_base", 0)
                    )
                )
                == 1
                else 0
            )
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
        period_mode = str(
            grouping_result.get(
                "period_mode",
                existing_grouping.get("period_mode", PeriodMode.RED),
            )
        )
        bunch_factor = max(1, bunch_factor)
        use_deadtime = bool(grouping_result.get("deadtime_correction", False))
        deadtime_mode = (
            str(
                grouping_result.get(
                    "deadtime_mode",
                    existing_grouping.get(
                        "deadtime_mode", existing_grouping.get("deadtime_method", "off")
                    ),
                )
            )
            .strip()
            .lower()
            or "off"
        )
        if deadtime_mode == "load":
            deadtime_mode = "manual"
        use_background = bool(
            grouping_result.get("background_correction", False)
            and self._dataset_supports_background_correction(dataset)
        )

        if not run.histograms:
            source_last_bin = len(source_time) - 1
            lo = max(0, first_good)
            hi = min(source_last_bin, last_good)
            if lo <= hi:
                time_out = source_time[lo : hi + 1].copy()
                asym_out = source_asymmetry[lo : hi + 1].copy()
                err_out = source_error[lo : hi + 1].copy()
                if bunch_factor > 1:
                    time_out, asym_out, err_out = rebin(
                        time_out,
                        asym_out,
                        err_out,
                        bunch_factor,
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
            run.grouping["deadtime_correction"] = bool(
                grouping_result.get("deadtime_correction", False)
            )
            run.grouping["deadtime_mode"] = deadtime_mode
            if grouping_result.get("deadtime_method"):
                run.grouping["deadtime_method"] = str(grouping_result.get("deadtime_method"))
            else:
                run.grouping.pop("deadtime_method", None)
            for key in (
                "deadtime_manual_us",
                "deadtime_estimated_us",
                "deadtime_reference_run",
            ):
                if key in grouping_result:
                    run.grouping[key] = grouping_result.get(key)
                else:
                    run.grouping.pop(key, None)
            run.grouping.pop("deadtime_source_path", None)
            run.grouping.pop("deadtime_loaded_us", None)
            if deadtime_mode != "file":
                if isinstance(grouping_result.get("dead_time_us"), list):
                    run.grouping["dead_time_us"] = list(grouping_result.get("dead_time_us", []))
                elif isinstance(grouping_result.get("deadtime_loaded_us"), list):
                    run.grouping["dead_time_us"] = list(
                        grouping_result.get("deadtime_loaded_us", [])
                    )
            run.grouping["background_correction"] = use_background
            if not use_background:
                run.grouping.pop("background_method", None)
                run.grouping.pop("background_values", None)
            run.grouping["period_mode"] = period_mode
            group_names = grouping_result.get("group_names")
            if isinstance(group_names, dict) and group_names:
                run.grouping["group_names"] = {int(k): str(v) for k, v in group_names.items()}
            included_groups = grouping_result.get("included_groups")
            if isinstance(included_groups, dict):
                run.grouping["included_groups"] = {
                    int(k): bool(v) for k, v in included_groups.items()
                }
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
            run.grouping["detector_t0_bins"] = list(grouping_result.get("detector_t0_bins", []))
        if isinstance(grouping_result.get("detector_first_good_bins"), list):
            run.grouping["detector_first_good_bins"] = list(
                grouping_result.get("detector_first_good_bins", [])
            )
        if isinstance(grouping_result.get("detector_last_good_bins"), list):
            run.grouping["detector_last_good_bins"] = list(
                grouping_result.get("detector_last_good_bins", [])
            )
        if isinstance(grouping_result.get("histogram_labels"), list):
            run.grouping["histogram_labels"] = list(grouping_result.get("histogram_labels", []))

        detector_t0_bins = run.grouping.get("detector_t0_bins")
        if isinstance(detector_t0_bins, list) and len(detector_t0_bins) == len(run.histograms):
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

        if has_file_deadtime(existing_grouping, len(run.histograms)):
            run.grouping["deadtime_file_us"] = list(existing_grouping.get("dead_time_us", []))
        elif isinstance(existing_grouping.get("deadtime_file_us"), list):
            run.grouping["deadtime_file_us"] = list(existing_grouping.get("deadtime_file_us", []))

        run.grouping["deadtime_mode"] = deadtime_mode
        if grouping_result.get("deadtime_method"):
            run.grouping["deadtime_method"] = str(grouping_result.get("deadtime_method"))
        else:
            run.grouping.pop("deadtime_method", None)
        for key in (
            "deadtime_manual_us",
            "deadtime_estimated_us",
            "deadtime_reference_run",
        ):
            if key in grouping_result:
                run.grouping[key] = grouping_result.get(key)
            else:
                run.grouping.pop(key, None)
        run.grouping.pop("deadtime_source_path", None)
        run.grouping.pop("deadtime_loaded_us", None)

        if deadtime_mode == "file":
            if isinstance(run.grouping.get("deadtime_file_us"), list):
                run.grouping["dead_time_us"] = list(run.grouping.get("deadtime_file_us", []))
        elif isinstance(grouping_result.get("dead_time_us"), list):
            run.grouping["dead_time_us"] = list(grouping_result.get("dead_time_us", []))
        elif isinstance(grouping_result.get("deadtime_loaded_us"), list):
            run.grouping["dead_time_us"] = list(grouping_result.get("deadtime_loaded_us", []))
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

        reduction_grouping = dict(run.grouping)
        reduction_grouping.update(
            {
                "groups": groups,
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "alpha": float(alpha if alpha > 0 else 1.0),
                "t0_bin": t0_bin,
                "t_good_offset": t_good_offset,
                "first_good_bin": first_good,
                "last_good_bin": last_good,
                "bin_index_base": bin_index_base,
                "bunching_factor": bunch_factor,
                "deadtime_correction": use_deadtime,
                "deadtime_mode": deadtime_mode,
                "background_correction": use_background,
                "period_mode": period_mode,
            }
        )
        if axis_pairs:
            reduction_grouping["alpha_x"] = float(vector_alphas.get("P_x", alpha))
            reduction_grouping["alpha_y"] = float(vector_alphas.get("P_y", alpha))
            reduction_grouping["alpha_z"] = float(vector_alphas.get("P_z", alpha))

        if not use_deadtime:
            run.grouping.pop("deadtime_method", None)

        run_alpha = alpha if alpha > 0 else 1.0
        has_two_period_histograms = bool(
            isinstance(reduction_grouping.get("period_histograms"), list)
            and len(reduction_grouping.get("period_histograms", [])) == 2
        )

        background_state: dict[str, object] | None = None
        dt_applied = False
        time_axis: np.ndarray
        asymmetry: np.ndarray
        error: np.ndarray

        if has_two_period_histograms and period_mode in {
            str(PeriodMode.GREEN_MINUS_RED),
            str(PeriodMode.GREEN_PLUS_RED),
        }:
            red_histograms, red_grouping = self._period_histograms_for_mode(
                run.histograms,
                reduction_grouping,
                period_index=0,
            )
            green_histograms, green_grouping = self._period_histograms_for_mode(
                run.histograms,
                reduction_grouping,
                period_index=1,
            )
            red_time, red_asym, red_err, red_dt, red_background = (
                self._reduce_grouped_histograms_to_asymmetry(
                    histograms=red_histograms,
                    grouping=red_grouping,
                    dataset=dataset,
                    run=run,
                    forward_idx=forward_idx,
                    backward_idx=backward_idx,
                    alpha=run_alpha,
                    use_deadtime=use_deadtime,
                    deadtime_mode=deadtime_mode,
                    use_background=use_background,
                )
            )
            green_time, green_asym, green_err, green_dt, green_background = (
                self._reduce_grouped_histograms_to_asymmetry(
                    histograms=green_histograms,
                    grouping=green_grouping,
                    dataset=dataset,
                    run=run,
                    forward_idx=forward_idx,
                    backward_idx=backward_idx,
                    alpha=run_alpha,
                    use_deadtime=use_deadtime,
                    deadtime_mode=deadtime_mode,
                    use_background=use_background,
                )
            )
            time_axis, asymmetry, error = combine_period_asymmetry(
                red_time,
                red_asym,
                red_err,
                green_time,
                green_asym,
                green_err,
                period_mode,
            )
            if len(time_axis) == 0:
                return False, bool(red_dt or green_dt)
            dt_applied = bool(red_dt or green_dt)
            if use_background:
                if red_background == green_background:
                    background_state = red_background
                else:
                    method = None
                    for state in (green_background, red_background):
                        if isinstance(state, dict) and isinstance(state.get("method"), str):
                            method = str(state.get("method"))
                            break
                    if method:
                        background_state = {"method": method}
        else:
            period_index = 1 if period_mode == str(PeriodMode.GREEN) else 0
            selected_histograms, selected_grouping = self._period_histograms_for_mode(
                run.histograms,
                reduction_grouping,
                period_index=period_index,
            )
            time_axis, asymmetry, error, dt_applied, background_state = (
                self._reduce_grouped_histograms_to_asymmetry(
                    histograms=selected_histograms,
                    grouping=selected_grouping,
                    dataset=dataset,
                    run=run,
                    forward_idx=forward_idx,
                    backward_idx=backward_idx,
                    alpha=run_alpha,
                    use_deadtime=use_deadtime,
                    deadtime_mode=deadtime_mode,
                    use_background=use_background,
                )
            )
            if has_two_period_histograms and period_mode in {
                str(PeriodMode.RED),
                str(PeriodMode.GREEN),
            }:
                if "good_frames" in selected_grouping:
                    run.grouping["good_frames"] = selected_grouping.get("good_frames")
                if isinstance(selected_grouping.get("dead_time_us"), list):
                    run.grouping["dead_time_us"] = list(selected_grouping.get("dead_time_us", []))

        if use_background:
            if isinstance(background_state, dict) and isinstance(
                background_state.get("method"), str
            ):
                run.grouping["background_method"] = str(background_state.get("method"))
            else:
                run.grouping.pop("background_method", None)
            values = background_state.get("values") if isinstance(background_state, dict) else None
            if isinstance(values, list) and len(values) == 2:
                run.grouping["background_values"] = [float(values[0]), float(values[1])]
            else:
                run.grouping.pop("background_values", None)
            ranges = background_state.get("ranges") if isinstance(background_state, dict) else None
            if isinstance(ranges, list) and len(ranges) == 2:
                run.grouping["background_ranges"] = [
                    [int(v) for v in ranges[0]],
                    [int(v) for v in ranges[1]],
                ]
        else:
            run.grouping.pop("background_method", None)
            run.grouping.pop("background_values", None)

        lo = max(0, first_good)
        hi = min(len(asymmetry) - 1, last_good)
        if lo <= hi:
            time_out = time_axis[lo : hi + 1].copy()
            asym_out = asymmetry[lo : hi + 1].copy()
            err_out = error[lo : hi + 1].copy()
            if bunch_factor > 1:
                time_out, asym_out, err_out = rebin(
                    time_out,
                    asym_out,
                    err_out,
                    bunch_factor,
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
                "deadtime_mode": deadtime_mode,
                "background_correction": use_background,
                "period_mode": period_mode,
            }
        )
        if axis_pairs:
            run.grouping["alpha_x"] = float(vector_alphas.get("P_x", run_alpha))
            run.grouping["alpha_y"] = float(vector_alphas.get("P_y", run_alpha))
            run.grouping["alpha_z"] = float(vector_alphas.get("P_z", run_alpha))
        if isinstance(detector_t0_bins, list) and len(detector_t0_bins) == len(run.histograms):
            run.grouping["detector_t0_bins"] = [int(hist.t0_bin) for hist in run.histograms]
        # Persist group names if provided
        group_names = grouping_result.get("group_names")
        if isinstance(group_names, dict) and group_names:
            run.grouping["group_names"] = {int(k): str(v) for k, v in group_names.items()}
        included_groups = grouping_result.get("included_groups")
        if isinstance(included_groups, dict):
            run.grouping["included_groups"] = {int(k): bool(v) for k, v in included_groups.items()}
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

    def _period_histograms_for_mode(
        self,
        histograms: list[Histogram],
        grouping: dict,
        *,
        period_index: int,
    ) -> tuple[list[Histogram], dict]:
        """Return period-specific histograms plus effective grouping metadata.

        Thin wrapper over :func:`asymmetry.core.io.periods.select_period_histograms`
        so the GUI and the scriptable core API share one implementation of the
        red/green period-selection rule.
        """
        return select_period_histograms(histograms, grouping, period_index)

    def _reduce_grouped_histograms_to_asymmetry(
        self,
        *,
        histograms: list[Histogram],
        grouping: dict,
        dataset,
        run,
        forward_idx: list[int],
        backward_idx: list[int],
        alpha: float,
        use_deadtime: bool,
        deadtime_mode: str,
        use_background: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, object] | None]:
        """Return reduced grouped asymmetry arrays for one histogram source."""
        effective_use_deadtime = bool(use_deadtime)
        if effective_use_deadtime:
            if deadtime_mode == "file":
                effective_use_deadtime = has_file_deadtime(grouping, len(histograms))
            else:
                effective_use_deadtime = has_resolved_deadtime(grouping, len(histograms))

        working_histograms, dt_applied = self._prepare_grouping_histograms(
            histograms,
            grouping,
            effective_use_deadtime,
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

        background_state: dict[str, object] | None = None
        bkg_result = None
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
                grouping=grouping,
                t0_bin=common_t0,
                bin_width_us=bin_width,
                facility=facility,
            )
            forward = bkg_result.forward
            backward = bkg_result.backward
            background_state = {"method": bkg_result.method}
            if bkg_result.applied:
                if bkg_result.values is not None:
                    background_state["values"] = [
                        float(bkg_result.values[0]),
                        float(bkg_result.values[1]),
                    ]
                if bkg_result.ranges is not None:
                    background_state["ranges"] = [
                        [int(v) for v in bkg_result.ranges[0]],
                        [int(v) for v in bkg_result.ranges[1]],
                    ]

        if (
            bkg_result is not None
            and bkg_result.applied
            and bkg_result.forward_error is not None
            and bkg_result.backward_error is not None
        ):
            asymmetry, error = compute_asymmetry_with_count_errors(
                forward,
                backward,
                bkg_result.forward_error,
                bkg_result.backward_error,
                alpha=alpha,
            )
        else:
            asymmetry, error = compute_asymmetry(forward, backward, alpha=alpha)

        bin_width = float(working_histograms[0].bin_width) if working_histograms else 1.0
        time_axis = (np.arange(len(asymmetry), dtype=np.float64) - float(common_t0)) * bin_width
        return (
            time_axis,
            np.asarray(asymmetry * 100.0, dtype=np.float64),
            np.asarray(error * 100.0, dtype=np.float64),
            dt_applied,
            background_state,
        )

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
        msg.setText("Field in header is 0 G, but comment contains a field candidate.")
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
        """Show the fit dock with the fit surface matching the current time view."""
        self._sync_fit_dock_mode()
        self._show_panel("fit")
        if self._should_launch_multi_group_fit_window():
            self._log_panel.log("Opened Multi-Group Fit panel")
            return
        self._log_panel.log("Opened Fit panel")

    def _should_launch_multi_group_fit_window(self) -> bool:
        """Return True when Fit should open the multi-group fit window."""
        if self._plot_workspace.active_domain() != "time":
            return False
        if not hasattr(self._plot_panel, "current_time_view_mode"):
            return False
        if self._plot_panel.current_time_view_mode() != "groups":
            return False
        return bool(self._grouped_time_domain_display_datasets())

    def _sync_fit_dock_mode(self) -> None:
        """Swap the fit dock between regular, grouped and ALC content."""
        if self._fit_stack is None:
            return

        # ALC mode takes precedence: the Fit dock shows the bespoke scan-build
        # panel and the Parameters dock shows the scan view.
        if self._alc_mode:
            self._fit_stack.setCurrentWidget(self._alc_fit_panel)
            self._parameters_stack.setCurrentWidget(self._alc_scan_view)
            self._dock_fit.setWindowTitle("ALC scan")
            return
        self._parameters_stack.setCurrentWidget(self._fit_parameters_panel)

        show_grouped = self._should_launch_multi_group_fit_window()
        current_widget = self._multi_group_fit_window if show_grouped else self._fit_panel
        self._fit_stack.setCurrentWidget(current_widget)

        if show_grouped and self._multi_group_fit_window is not None:
            dataset = (
                self._get_fit_dataset(self._current_dataset) if self._current_dataset else None
            )
            self._multi_group_fit_window.set_dataset(dataset)
            self._dock_fit.setWindowTitle(self._multi_group_fit_window.dock_title())
        else:
            self._dock_fit.setWindowTitle("Fit")  # inspector tab label — title case per spec

    def _on_alc_mode_toggled(self, checked: bool) -> None:
        """Enter/leave ALC mode: swap the Fit and Parameters docks accordingly."""
        if checked and self._active_representation_type() != RepresentationType.TIME_FB_ASYMMETRY:
            # Guard: ALC mode is only valid for the F-B asymmetry representation.
            self._alc_mode_action.setChecked(False)
            return
        self._alc_mode = bool(checked)
        self._sync_fit_dock_mode()
        if self._alc_mode:
            # Echo the current integration window and surface the docks.
            t_min, t_max = (None, None)
            if hasattr(self._plot_panel, "get_fit_range"):
                t_min, t_max = self._plot_panel.get_fit_range()
            self._alc_fit_panel.set_fit_range_display(t_min, t_max)
            for dock in (self._dock_fit, self._dock_fit_parameters):
                dock.show()
                dock.raise_()

    # Maps each toolbar domain token to (ordered visible dock keys, default raised key).
    # Fourier is hidden in the groups domain; mgfit is surfaced by swapping _fit_stack.
    _INSPECTOR_DOMAIN_CONFIG: dict[str, tuple[list[str], str]] = {
        "fb_asymmetry": (["fit", "fourier", "fit_parameters"], "fit"),
        "groups": (["fit", "fit_parameters"], "fit"),
        "frequency": (["fourier", "fit", "fit_parameters"], "fourier"),
        "maxent": (["fourier", "fit", "fit_parameters"], "fourier"),
    }

    def _apply_inspector_for_domain(self, view: str) -> None:
        """Show/hide/raise the right inspector docks for *view* and sync the fit stack."""
        config = self._INSPECTOR_DOMAIN_CONFIG.get(view)
        if config is None:
            return

        visible_keys, default_key = config
        dock_map = {
            "fit": self._dock_fit,
            "fourier": self._dock_fourier,
            "fit_parameters": self._dock_fit_parameters,
        }
        visible_set = set(visible_keys)

        self._applying_inspector_domain = True
        try:
            for key, dock in dock_map.items():
                if dock.isFloating():
                    continue
                if key in visible_set:
                    dock.show()
                else:
                    dock.hide()

            default_dock = dock_map[default_key]
            if not default_dock.isFloating():
                default_dock.raise_()
        finally:
            self._applying_inspector_domain = False

        # Sync _fit_stack page: groups domain surfaces mgfit when grouped data is present,
        # all other domains revert to single-fit so the dock title reads "Fit".
        self._sync_fit_dock_mode()

        # Qt sometimes skips the dock tab-bar relayout after programmatic
        # show/hide of tabified docks, leaving the bottom tabs missing until
        # the window is manually resized. Defer a relayout nudge so the tab
        # bar is rebuilt once the visibility changes settle.
        QTimer.singleShot(0, self, self._refresh_inspector_tab_bar)

    def _refresh_inspector_tab_bar(self) -> None:
        """Force the right dock area to relayout so its tab bar reappears.

        Works around a Qt quirk where the tab bar of a tabified dock group is
        not re-shown after programmatic show/hide cycles. ``resizeDocks``
        walks the same relayout path as a manual window resize, which is the
        user-visible recovery for the missing tabs.
        """
        docks = [
            dock
            for dock in (self._dock_fit, self._dock_fourier, self._dock_fit_parameters)
            if dock.isVisible() and not dock.isFloating()
        ]
        if len(docks) < 2:
            return
        self.resizeDocks(docks, [dock.width() for dock in docks], Qt.Orientation.Horizontal)
        # Belt-and-braces: if the relayout still left the group's tab bar
        # hidden, re-show it directly. Dock tab bars are direct QTabBar
        # children of the QMainWindow; ours is the one listing a dock title.
        dock_titles = {dock.windowTitle() for dock in docks}
        direct_only = Qt.FindChildOption.FindDirectChildrenOnly
        for tab_bar in self.findChildren(QTabBar, options=direct_only):
            tab_texts = {tab_bar.tabText(i) for i in range(tab_bar.count())}
            if tab_texts & dock_titles and not tab_bar.isVisible():
                tab_bar.show()

    def _on_fourier(self) -> None:
        """Show and raise the Fourier dock panel."""
        self._sync_spectrum_panel_for_view("frequency")
        self._show_panel("fourier")
        if self._current_dataset is not None:
            self._sync_fourier_panel_for_dataset(self._current_dataset)
        self._log_panel.log("Opened Fourier panel")

    def _sync_spectrum_panel_for_view(self, view: str | None = None) -> None:
        """Switch the spectrum controls between FFT and MaxEnt."""
        token = view
        if token is None and hasattr(self, "_plot_workspace"):
            token = self._plot_workspace.active_view()
        is_maxent = str(token).strip().lower() == "maxent"
        if hasattr(self, "_spectrum_stack"):
            self._spectrum_stack.setCurrentWidget(
                self._maxent_panel if is_maxent else self._fourier_panel
            )
        if hasattr(self, "_dock_fourier"):
            self._dock_fourier.setWindowTitle("MaxEnt" if is_maxent else "Fourier")

    def _active_frequency_rep_type(self) -> RepresentationType:
        """Return the active frequency representation type."""
        if hasattr(self, "_plot_workspace") and self._plot_workspace.active_view() == "maxent":
            return RepresentationType.FREQ_MAXENT
        return RepresentationType.FREQ_FFT

    def _frequency_cache(
        self,
        rep_type: RepresentationType | None = None,
    ) -> dict[int, list[MuonDataset]]:
        """Return the in-memory spectrum cache for a frequency representation."""
        resolved = rep_type or self._active_frequency_rep_type()
        if resolved == RepresentationType.FREQ_FFT:
            self._frequency_spectra_by_rep[RepresentationType.FREQ_FFT] = (
                self._frequency_spectra_by_run
            )
        return self._frequency_spectra_by_rep.setdefault(resolved, {})

    def _frequency_status_name(self, rep_type: RepresentationType | None = None) -> str:
        resolved = rep_type or self._active_frequency_rep_type()
        return "MaxEnt" if resolved == RepresentationType.FREQ_MAXENT else "FFT"

    def _set_fourier_status(self, message: str, *, success: bool = False) -> None:
        """Update the Fourier panel status text and main-window status bar."""
        if self._active_frequency_rep_type() == RepresentationType.FREQ_MAXENT:
            self._maxent_panel.set_status(message, success=success)
        else:
            self._fourier_panel.set_fft_status(message, success=success)
        self.statusBar().showMessage(str(message))

    def _fourier_group_names_for_dataset(self, dataset: MuonDataset | None) -> dict[int, str]:
        """Return detector-group names for the provided dataset, if available."""
        if dataset is None or dataset.run is None or not isinstance(dataset.run.grouping, dict):
            return {}
        groups = dataset.run.grouping.get("groups")
        if not isinstance(groups, dict):
            return {}
        group_names = (
            dataset.run.grouping.get("group_names")
            if isinstance(dataset.run.grouping.get("group_names"), dict)
            else {}
        )

        resolved: dict[int, str] = {}
        for raw_group_id in sorted(groups):
            try:
                group_id = int(raw_group_id)
            except (TypeError, ValueError):
                continue
            resolved[group_id] = str(group_names.get(group_id, f"Group {group_id}"))
        return resolved

    def _dataset_supports_maxent(self, dataset: MuonDataset | None) -> bool:
        """Return whether *dataset* has the raw grouped counts MaxEnt needs."""
        return bool(
            dataset is not None
            and dataset.run is not None
            and dataset.run.histograms
            and self._fourier_group_names_for_dataset(dataset)
        )

    def _sync_fourier_panel_for_dataset(self, dataset: MuonDataset | None) -> None:
        """Refresh the Fourier group-phase table for the active run."""
        group_names = self._fourier_group_names_for_dataset(dataset)
        run_number = None if dataset is None else int(dataset.run_number)
        state = (
            None if run_number is None else self._fourier_group_phase_state_by_run.get(run_number)
        )
        self._fourier_panel.restore_group_phase_state(state, group_names)
        if hasattr(self, "_maxent_panel"):
            self._sync_maxent_panel_for_dataset(dataset)

    def _store_maxent_panel_state_for_dataset(self, dataset: MuonDataset | None) -> None:
        """Persist the current MaxEnt panel state for one dataset/run."""
        if dataset is None:
            return
        try:
            run_number = int(dataset.run_number)
        except (TypeError, ValueError):
            return
        self._maxent_panel_state_by_run[run_number] = self._maxent_panel.get_state()

    @staticmethod
    def _derive_group_enabled_table(state: dict, group_names: dict[int, str]) -> dict[int, bool]:
        """Derive per-group inclusion from a stored ``selected_group_ids`` list.

        Groups the stored selection never knew about — absent from both the
        selection and the phase table, e.g. after re-grouping the run or when
        a recipe was copied from a run with different groups — default to
        enabled, preserving the panel's unknown-groups-default-enabled
        convention.  Groups that were known but unselected stay disabled.
        """
        selected = state.get("selected_group_ids")
        if not isinstance(selected, list):
            return {int(group_id): True for group_id in group_names}
        selected_ids: set[int] = set()
        for group_id in selected:
            try:
                selected_ids.add(int(group_id))
            except (TypeError, ValueError):
                continue
        known = set(selected_ids)
        phases = state.get("group_phase_degrees")
        if isinstance(phases, dict):
            for key in phases:
                try:
                    known.add(int(key))
                except (TypeError, ValueError):
                    continue
        return {
            int(group_id): (int(group_id) in selected_ids if int(group_id) in known else True)
            for group_id in group_names
        }

    def _maxent_state_from_config(self, config: dict, group_names: dict[int, str]) -> dict | None:
        """Return a panel-state dict from a MaxEnt recipe block.

        The block is normalised through the typed ``MaxEntConfig`` boundary so
        malformed entries in a loaded project degrade gracefully instead of
        raising out of dataset selection.
        """
        if not isinstance(config, dict):
            return None
        state = MaxEntConfig.from_dict(config).to_dict()
        state["group_enabled_table"] = self._derive_group_enabled_table(state, group_names)
        return state

    def _normalise_maxent_panel_state(self, state: dict, group_names: dict[int, str]) -> dict:
        """Return a MaxEnt panel state compatible with the current run groups."""
        normalised = dict(state)
        if "group_enabled_table" not in normalised and isinstance(
            normalised.get("selected_group_ids"), list
        ):
            normalised["group_enabled_table"] = self._derive_group_enabled_table(
                normalised, group_names
            )
        return normalised

    def _maxent_state_from_representation(
        self, run_number: int, group_names: dict[int, str]
    ) -> dict | None:
        """Return a panel-state dict from a persisted MaxEnt recipe, if available."""
        representation = self._project_model.representation(
            run_number, RepresentationType.FREQ_MAXENT
        )
        if representation is None:
            return None
        config = representation.maxent_config()
        if not config:
            return None
        return self._maxent_state_from_config(config, group_names)

    def _inherited_maxent_panel_state(self) -> dict:
        """Return current non-group MaxEnt settings for a new run."""
        state = self._maxent_panel.get_state()
        state.pop("selected_group_ids", None)
        state.pop("group_enabled_table", None)
        state.pop("group_phase_degrees", None)
        return state

    def _sync_maxent_panel_for_dataset(self, dataset: MuonDataset | None) -> None:
        """Refresh the MaxEnt group table for the active run."""
        group_names = self._fourier_group_names_for_dataset(dataset)
        run_number = None if dataset is None else int(dataset.run_number)
        if run_number is None:
            self._maxent_panel.set_group_definitions({})
            return
        state = self._maxent_panel_state_by_run.get(run_number)
        if state is None:
            state = self._maxent_state_from_representation(run_number, group_names)
        if state is None:
            state = self._inherited_maxent_panel_state()
        state = self._normalise_maxent_panel_state(state, group_names)
        self._maxent_panel.set_group_definitions(group_names)
        self._maxent_panel.restore_state(state)

    def _store_fourier_group_phase_state_for_dataset(self, dataset: MuonDataset | None) -> None:
        """Persist the current Fourier group-phase UI state for one dataset/run."""
        if dataset is None:
            return
        try:
            run_number = int(dataset.run_number)
        except (TypeError, ValueError):
            return
        self._fourier_group_phase_state_by_run[run_number] = self._fourier_panel.group_phase_state()

    def _estimate_dataset_fourier_phase(self, dataset: MuonDataset, state: dict) -> float:
        """Estimate a single phase correction for one time-domain dataset."""
        freqs, spectrum = fft_complex_asymmetry(
            dataset,
            window=str(state.get("window", "none")),
            padding_factor=int(state.get("padding", 1)),
            phase_degrees=0.0,
            t0_offset_us=float(state.get("t0_offset_us", 0.0)),
            subtract_average_signal=bool(state.get("subtract_average_signal", True)),
            filter_start_us=float(state.get("filter_start_us", 0.0)),
            filter_time_constant_us=float(state.get("filter_time_constant_us", 1.5)),
        )
        min_frequency, max_frequency = self._resolve_fourier_phase_window_mhz(dataset, freqs)
        return estimate_fft_phase(
            freqs,
            spectrum,
            method=str(state.get("auto_phase_method", "Peak")).strip().lower(),
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

    def _fourier_center_frequency_mhz(self, dataset: MuonDataset | None) -> float | None:
        """Return the expected FFT center frequency from the run field, if available."""
        if dataset is None:
            return None
        field_value = dataset.metadata.get("field")
        if field_value is None and dataset.run is not None:
            field_value = dataset.run.metadata.get("field")
        try:
            field_gauss = float(field_value)
        except (TypeError, ValueError):
            return None
        return field_gauss * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA

    def _resolve_fourier_phase_window_mhz(
        self,
        dataset: MuonDataset,
        freqs: np.ndarray,
    ) -> tuple[float, float | None]:
        """Return the MHz window that should drive automatic phase estimation."""
        frequencies = np.asarray(freqs, dtype=float)
        positive = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]
        if positive.size == 0:
            return 0.0, None

        expected_center = self._fourier_center_frequency_mhz(dataset)
        preferred_half_width_mhz = 10.0
        candidate_window: tuple[float, float] | None = None
        if hasattr(self._frequency_plot_panel, "get_frequency_view_window_mhz"):
            raw_window = self._frequency_plot_panel.get_frequency_view_window_mhz(
                reference_dataset=dataset
            )
            if raw_window is not None:
                lo, hi = sorted((float(raw_window[0]), float(raw_window[1])))
                if hi > 0.0:
                    if expected_center is None:
                        candidate_window = (max(0.0, lo), hi)
                    elif (
                        hasattr(
                            self._frequency_plot_panel,
                            "is_frequency_axis_relative_to_reference",
                        )
                        and self._frequency_plot_panel.is_frequency_axis_relative_to_reference()
                    ) or (lo <= expected_center <= hi):
                        narrowed_lo = max(lo, expected_center - preferred_half_width_mhz)
                        narrowed_hi = min(hi, expected_center + preferred_half_width_mhz)
                        if narrowed_hi > narrowed_lo:
                            candidate_window = (max(0.0, narrowed_lo), narrowed_hi)
                        else:
                            candidate_window = (max(0.0, lo), hi)

        if candidate_window is None and expected_center is not None:
            candidate_window = (
                max(0.0, expected_center - preferred_half_width_mhz),
                expected_center + preferred_half_width_mhz,
            )

        if candidate_window is None:
            return 0.0, None

        lo, hi = candidate_window
        has_overlap = np.any((positive >= lo) & (positive <= hi))
        if has_overlap:
            return lo, hi

        if expected_center is not None:
            fallback_lo = max(0.0, expected_center - preferred_half_width_mhz)
            fallback_hi = expected_center + preferred_half_width_mhz
            if np.any((positive >= fallback_lo) & (positive <= fallback_hi)):
                return fallback_lo, fallback_hi

        return 0.0, None

    def _estimate_group_fourier_phases(self, dataset: MuonDataset, state: dict) -> dict[int, float]:
        """Estimate one phase correction per detector group."""
        if dataset.run is None:
            return {}
        phases: dict[int, float] = {}
        prepared_histograms, reference_t0_bin = self._precompute_group_fourier_inputs(dataset)
        for group_id in self._fourier_group_names_for_dataset(dataset):
            group_dataset = build_group_signal_dataset(
                dataset.run,
                group_id,
                center_signal=False,
                reference_t0_bin=reference_t0_bin,
                prepared_histograms=prepared_histograms,
            )
            phases[group_id] = self._estimate_dataset_fourier_phase(group_dataset, state)
        return phases

    def _resolve_group_phase_degrees(
        self,
        selected_group_ids: list[int],
        state: dict,
        *,
        apply_phase_correction: bool,
        auto_phase: bool,
        use_phase_table: bool,
        manual_phase: float,
        group_phase_table: dict[int, float],
        prepared_histograms: list[Histogram] | None,
        reference_t0_bin: int | None,
    ) -> dict[int, float]:
        """Resolve concrete per-group phase corrections for the active run.

        Mirrors the previous inline auto/table/manual selection so the shared
        spectrum core (and recipe recompute) receives fully-resolved phases.
        """
        if (
            not apply_phase_correction
            or self._current_dataset is None
            or self._current_dataset.run is None
        ):
            return {}
        resolved: dict[int, float] = {}
        for group_id in selected_group_ids:
            if auto_phase and not use_phase_table:
                group_dataset = build_group_signal_dataset(
                    self._current_dataset.run,
                    group_id,
                    center_signal=False,
                    reference_t0_bin=reference_t0_bin,
                    prepared_histograms=prepared_histograms,
                )
                resolved[group_id] = self._estimate_dataset_fourier_phase(group_dataset, state)
            elif use_phase_table:
                resolved[group_id] = group_phase_table.get(group_id, manual_phase)
            else:
                resolved[group_id] = manual_phase
        return resolved

    def _fourier_display_ylabel(self, display: str) -> str:
        """Return a display-specific y-axis label for FFT plots."""
        return {
            "cos": "FFT Cos (a.u.)",
            "imaginary": "FFT Imaginary (a.u.)",
            "magnitude": "FFT Magnitude (a.u.)",
            "phase_corrected": "FFT Phase-Corrected (a.u.)",
            "phase_opt_real": "FFT phaseOptReal (a.u.)",
            "phase_spectrum": "FFT Phase Spectrum (deg)",
            "power": "FFT Power (a.u.)",
            "power_sqrt": "FFT (Power)^1/2 (a.u.)",
            "real": "FFT Real (a.u.)",
            "sin": "FFT Sin (a.u.)",
        }.get(canonical_fourier_display_mode(display), "FFT (a.u.)")

    def _build_fourier_value_dataset(
        self,
        source_dataset: MuonDataset,
        freqs: np.ndarray,
        values: np.ndarray,
        *,
        display: str,
        run_label: str,
        error: np.ndarray | None = None,
    ) -> MuonDataset:
        """Convert one real-valued Fourier display channel into a plottable dataset."""
        metadata = dict(source_dataset.metadata)
        metadata.update(
            {
                "run_number": source_dataset.run_number,
                "run_label": str(run_label),
                "plot_domain": "frequency",
                "x_label": "Frequency (MHz)",
                "y_label": self._fourier_display_ylabel(display),
                "fourier_display": str(display),
            }
        )
        return MuonDataset(
            time=np.asarray(freqs, dtype=float),
            asymmetry=np.asarray(values, dtype=float),
            error=(
                np.asarray(error, dtype=float)
                if error is not None
                else np.zeros_like(values, dtype=float)
            ),
            metadata=metadata,
            run=None,
        )

    def _store_frequency_spectra_for_run(
        self,
        run_number: int,
        spectra: list[MuonDataset],
        *,
        rep_type: RepresentationType | None = None,
    ) -> None:
        """Cache computed frequency spectra for one run-number context."""
        cache = self._frequency_cache(rep_type)
        cache[int(run_number)] = list(spectra)

    def _record_frequency_fft_recipe(
        self,
        run_number: int,
        config: GroupSpectrumConfig,
        spectrum: MuonDataset,
    ) -> None:
        """Persist the generation recipe for a run's FFT representation.

        The recipe lets the spectrum be recomputed on project load instead of
        storing the spectrum arrays.  The freshly computed spectrum is cached on
        the representation so it need not be recomputed immediately.
        """
        representation = self._project_model.ensure_dataset(int(run_number)).ensure(
            RepresentationType.FREQ_FFT
        )
        representation.recipe = {"fourier_config": config.to_dict()}
        representation.cache_datasets([spectrum])

    def _restore_frequency_representations(self, state: dict) -> None:
        """Rebuild FFT spectra from recipes (authoritative over stored arrays).

        Reads the v6 ``representations``/``batches`` into the project model,
        recomputes each FrequencyFFT spectrum from its recipe using the loaded
        runs, and refreshes the in-memory plot cache.  Runs that cannot be
        recomputed keep whatever the legacy array fallback restored.
        """
        self._project_model = ProjectModel.from_project_state(state)
        runs_by_number: dict[int, object] = {}
        if hasattr(self._data_browser, "get_all_datasets"):
            for dataset in self._data_browser.get_all_datasets():
                if dataset.run is not None:
                    runs_by_number[int(dataset.run_number)] = dataset.run
        self._project_model.recompute_all(runs_by_number)
        for run_number, container in self._project_model.datasets.items():
            for rep_type in (RepresentationType.FREQ_FFT, RepresentationType.FREQ_MAXENT):
                representation = container.get(rep_type)
                if representation is not None and representation.primary is not None:
                    self._frequency_cache(rep_type)[int(run_number)] = [representation.primary]

    def _serialize_frequency_spectra_state(self) -> dict[str, list[dict[str, object]]]:
        """Return a serializable snapshot of cached Fourier spectra."""
        serialized: dict[str, list[dict[str, object]]] = {}
        for run_number, spectra in self._frequency_spectra_by_run.items():
            run_payload: list[dict[str, object]] = []
            for spectrum in spectra:
                run_payload.append(
                    {
                        "time": np.asarray(spectrum.time, dtype=float).tolist(),
                        "asymmetry": np.asarray(spectrum.asymmetry, dtype=float).tolist(),
                        "error": np.asarray(spectrum.error, dtype=float).tolist(),
                        "metadata": dict(spectrum.metadata),
                    }
                )
            serialized[str(int(run_number))] = run_payload
        return serialized

    def _restore_frequency_spectra_state(self, state: object) -> None:
        """Restore cached Fourier spectra from serialized project state."""
        self._frequency_spectra_by_run = {}
        self._frequency_spectra_by_rep = {
            RepresentationType.FREQ_FFT: self._frequency_spectra_by_run,
            RepresentationType.FREQ_MAXENT: {},
        }
        if not isinstance(state, dict):
            return

        for run_key, entries in state.items():
            try:
                run_number = int(run_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(entries, list):
                continue

            restored: list[MuonDataset] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    time = np.asarray(entry.get("time", []), dtype=float)
                    asymmetry = np.asarray(entry.get("asymmetry", []), dtype=float)
                    error = np.asarray(entry.get("error", []), dtype=float)
                except (TypeError, ValueError):
                    continue
                if time.size == 0 or asymmetry.size == 0:
                    continue
                if error.size != asymmetry.size:
                    error = np.zeros_like(asymmetry, dtype=float)
                metadata = entry.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata = dict(metadata)
                metadata.setdefault("run_number", run_number)
                restored.append(
                    MuonDataset(
                        time=time,
                        asymmetry=asymmetry,
                        error=error,
                        metadata=metadata,
                        run=None,
                    )
                )

            if restored:
                self._frequency_cache(RepresentationType.FREQ_FFT)[run_number] = restored

    def _sync_frequency_plot_for_run(
        self,
        run_number: int | None,
        *,
        preserve_x_limits: bool = False,
    ) -> None:
        """Render the cached frequency spectra for *run_number*, or clear the tab."""
        preserved_x_limits: tuple[float, float] | None = None
        preserved_y_limits: tuple[float, float] | None = None
        if hasattr(self._frequency_plot_panel, "get_view_limits"):
            current_dataset = getattr(self._frequency_plot_panel, "_current_dataset", None)
            current_run_number: int | None = None
            if current_dataset is not None:
                try:
                    current_run_number = int(current_dataset.run_number)
                except (TypeError, ValueError):
                    current_run_number = None
            same_run = run_number is not None and current_run_number == int(run_number)
            if preserve_x_limits or same_run:
                x_min, x_max, y_min, y_max = self._frequency_plot_panel.get_view_limits()
                preserved_x_limits = (float(x_min), float(x_max))
                preserved_y_limits = (float(y_min), float(y_max))

        if run_number is None:
            self._frequency_plot_panel.clear()
            if preserved_x_limits is not None and preserved_y_limits is not None:
                self._frequency_plot_panel.set_view_limits(
                    preserved_x_limits[0],
                    preserved_x_limits[1],
                    preserved_y_limits[0],
                    preserved_y_limits[1],
                )
            return

        rep_type = self._active_frequency_rep_type()
        spectra = list(self._frequency_cache(rep_type).get(int(run_number), []))
        if not spectra:
            self._frequency_plot_panel.clear()
            if preserved_x_limits is not None and preserved_y_limits is not None:
                self._frequency_plot_panel.set_view_limits(
                    preserved_x_limits[0],
                    preserved_x_limits[1],
                    preserved_y_limits[0],
                    preserved_y_limits[1],
                )
            self._set_fourier_status(
                f"No {self._frequency_status_name(rep_type)} computed for run {run_number}."
            )
            return

        if len(spectra) == 1:
            self._frequency_plot_panel.plot_dataset(spectra[0])
        else:
            self._frequency_plot_panel.plot_datasets(spectra)

        if preserved_x_limits is not None:
            _current_x_min, _current_x_max, y_min, y_max = (
                self._frequency_plot_panel.get_view_limits()
            )
            self._frequency_plot_panel.set_view_limits(
                preserved_x_limits[0],
                preserved_x_limits[1],
                y_min,
                y_max,
            )
        if (
            hasattr(self, "_plot_workspace")
            and self._plot_workspace.active_domain() == "frequency"
            and hasattr(self, "_fit_panel")
        ):
            if hasattr(self._fit_panel, "set_domain"):
                self._fit_panel.set_domain("frequency")
            self._fit_panel.set_dataset(self._active_frequency_fit_dataset())
            self._set_frequency_fit_datasets_for_selection()

    def _sync_frequency_plot_for_current_dataset(self) -> None:
        """Render the cached frequency spectra for the current dataset selection."""
        run_number = (
            None if self._current_dataset is None else int(self._current_dataset.run_number)
        )
        self._sync_frequency_plot_for_run(run_number)

    def _selected_fourier_group_ids(self, dataset: MuonDataset) -> list[int]:
        """Return the detector groups currently enabled for grouped Fourier transforms."""
        group_names = self._fourier_group_names_for_dataset(dataset)
        enabled = self._fourier_panel.group_enabled_table()
        return [group_id for group_id in group_names if enabled.get(int(group_id), True)]

    def _precompute_group_fourier_inputs(
        self,
        dataset: MuonDataset,
    ) -> tuple[list[Histogram] | None, int | None]:
        """Prepare grouped Fourier intermediates once per run.

        Reusing deadtime-prepared histograms and a shared reference t0 avoids
        repeating setup work for each detector-group FFT.
        """
        if dataset.run is None:
            return None, None
        grouping = dataset.run.grouping if isinstance(dataset.run.grouping, dict) else {}
        groups = grouping.get("groups") if isinstance(grouping, dict) else None
        histograms = list(dataset.run.histograms)
        if not isinstance(groups, dict) or not histograms:
            return None, None

        apply_deadtime = bool(grouping.get("deadtime_correction", False))
        prepared_histograms, _ = prepare_histograms_with_deadtime(
            histograms,
            grouping,
            apply_deadtime,
        )

        all_group_indices: list[list[int]] = []
        for values in groups.values():
            if not isinstance(values, list):
                continue
            normalized: list[int] = []
            for value in values:
                detector = value[0] if isinstance(value, (list, tuple)) and value else value
                try:
                    normalized.append(max(0, int(detector) - 1))
                except (TypeError, ValueError):
                    continue
            if normalized:
                all_group_indices.append(normalized)

        reference_t0_bin = 0
        if all_group_indices:
            reference_t0_bin = common_t0_for_groups(prepared_histograms, *all_group_indices)
        return prepared_histograms, int(reference_t0_bin)

    def _current_fourier_time_window_us(self) -> tuple[float | None, float | None]:
        """Return the active time-domain window to use for grouped FFTs."""
        if not hasattr(self._plot_panel, "get_fit_range"):
            return None, None
        fit_range = self._plot_panel.get_fit_range()
        if fit_range is None:
            return None, None
        try:
            t_min = float(fit_range[0])
            t_max = float(fit_range[1])
        except (TypeError, ValueError, IndexError):
            return None, None
        if not np.isfinite(t_min) or not np.isfinite(t_max):
            return None, None
        if t_max < t_min:
            t_min, t_max = t_max, t_min
        return t_min, t_max

    def _on_fill_fourier_phases(self) -> None:
        """Estimate one phase correction per included detector group."""
        state = self._fourier_panel.get_state()
        if self._current_dataset is None:
            self._set_fourier_status("Select a run before estimating group phases.")
            return
        self._sync_fourier_panel_for_dataset(self._current_dataset)
        phases = self._estimate_group_fourier_phases(self._current_dataset, state)
        if not phases:
            self._set_fourier_status("No detector groups are available for phase estimation.")
            return
        self._fourier_panel.set_group_phases(phases, auto_filled=True)
        self._fourier_panel._use_phase_table_check.setChecked(True)
        self._set_fourier_status(
            f"Estimated phases for {len(phases)} detector groups.", success=True
        )

    def _on_compute_fourier(self) -> None:
        """Compute one averaged grouped FFT spectrum for the active run."""
        started_at = time.perf_counter()
        state = self._fourier_panel.get_state()
        display = str(state.get("display", "Real"))
        padding = int(state.get("padding", 1))
        selected_group_ids: list[int] = []
        spectra: list[MuonDataset] = []
        apply_phase_correction = fourier_mode_uses_phase_correction(display)
        window = str(state.get("window", "none"))
        filter_start_us = float(state.get("filter_start_us", 0.0))
        filter_time_constant_us = float(state.get("filter_time_constant_us", 1.5))
        t0_offset_us = float(state.get("t0_offset_us", 0.0))
        manual_phase = float(state.get("phase_degrees", 0.0))
        auto_phase = bool(state.get("auto_phase", False))
        use_phase_table = bool(state.get("use_phase_table", False))
        estimate_average_error = bool(state.get("estimate_average_error", False))
        subtract_average_signal = bool(state.get("subtract_average_signal", True))
        group_phase_table = {
            int(group_id): float(phase)
            for group_id, phase in self._fourier_panel.group_phase_table().items()
        }
        self._fourier_panel.clear_average_summary()

        spectra_by_run: dict[int, list[MuonDataset]] = {}

        try:
            if self._current_dataset is None or self._current_dataset.run is None:
                self._set_fourier_status(
                    "Select a grouped run before computing the Fourier transform."
                )
                return

            self._sync_fourier_panel_for_dataset(self._current_dataset)
            if auto_phase and apply_phase_correction:
                estimated = self._estimate_group_fourier_phases(self._current_dataset, state)
                if use_phase_table and estimated:
                    self._fourier_panel.set_group_phases(estimated, auto_filled=True)
                    group_phase_table.update(estimated)
            group_names = self._fourier_group_names_for_dataset(self._current_dataset)
            if not group_names:
                self._set_fourier_status("The active run does not define detector groups.")
                return

            selected_group_ids = self._selected_fourier_group_ids(self._current_dataset)
            if not selected_group_ids:
                self._set_fourier_status(
                    "Select at least one detector group before computing the Fourier transform."
                )
                return

            fourier_t_min_us, fourier_t_max_us = self._current_fourier_time_window_us()
            prepared_histograms, reference_t0_bin = self._precompute_group_fourier_inputs(
                self._current_dataset
            )

            # Resolve the auto/table/manual phase choice into concrete per-group
            # values, then delegate the spectrum maths to the shared core so a
            # generated spectrum and a recipe-recomputed one are identical.
            group_phase_degrees = self._resolve_group_phase_degrees(
                selected_group_ids,
                state,
                apply_phase_correction=apply_phase_correction,
                auto_phase=auto_phase,
                use_phase_table=use_phase_table,
                manual_phase=manual_phase,
                group_phase_table=group_phase_table,
                prepared_histograms=prepared_histograms,
                reference_t0_bin=reference_t0_bin,
            )

            fourier_config = GroupSpectrumConfig(
                display=display,
                window=window,
                padding=padding,
                filter_start_us=filter_start_us,
                filter_time_constant_us=filter_time_constant_us,
                t0_offset_us=t0_offset_us,
                subtract_average_signal=subtract_average_signal,
                estimate_average_error=estimate_average_error,
                t_min_us=fourier_t_min_us,
                t_max_us=fourier_t_max_us,
                selected_group_ids=list(selected_group_ids),
                group_phase_degrees=group_phase_degrees,
            )
            average_dataset = compute_average_group_spectrum(
                self._current_dataset.run,
                fourier_config,
                prepared_histograms=prepared_histograms,
                reference_t0_bin=reference_t0_bin,
            )

            if average_dataset is not None:
                averaged_display = average_dataset.asymmetry
                averaged_error = average_dataset.error
                if averaged_error.size > 0 and np.any(averaged_error > 0.0):
                    sn = np.divide(
                        np.abs(averaged_display),
                        averaged_error,
                        out=np.zeros_like(averaged_display),
                        where=averaged_error > 0.0,
                    )
                    peak_signal_to_noise = float(np.nanmax(sn)) if sn.size else 0.0
                    self._fourier_panel.set_average_summary(
                        mean_error=float(np.nanmean(averaged_error))
                        if averaged_error.size
                        else 0.0,
                        peak_signal_to_noise=peak_signal_to_noise,
                        group_count=len(selected_group_ids),
                    )
                spectra.append(average_dataset)
                spectra_by_run[int(self._current_dataset.run_number)] = list(spectra)
                self._record_frequency_fft_recipe(
                    int(self._current_dataset.run_number),
                    fourier_config,
                    average_dataset,
                )

            if not spectra:
                self._set_fourier_status(
                    "No FFT spectra could be generated from the current selection."
                )
                return

            for run_number, run_spectra in spectra_by_run.items():
                self._store_frequency_spectra_for_run(
                    run_number,
                    run_spectra,
                    rep_type=RepresentationType.FREQ_FFT,
                )

            active_run_number = None
            if self._current_dataset is not None:
                current_run_number = int(self._current_dataset.run_number)
                if current_run_number in spectra_by_run:
                    active_run_number = current_run_number
            if active_run_number is None and spectra_by_run:
                active_run_number = next(iter(spectra_by_run))

            self._sync_frequency_plot_for_run(active_run_number, preserve_x_limits=True)
            self._plot_workspace.set_active_view("frequency")
            self._show_panel("fourier")
            suffix = "s" if len(spectra) != 1 else ""
            self._set_fourier_status(
                f"Computed {len(spectra)} Fourier spectrum{suffix}.", success=True
            )
            self._log_panel.log(
                f"Computed averaged grouped Fourier spectrum using {display.lower()} display."
            )
        finally:
            self._log_perf_event(
                "compute_fourier",
                started_at,
                run=None
                if self._current_dataset is None
                else int(self._current_dataset.run_number),
                groups=len(selected_group_ids),
                padding=padding,
                display=display,
                spectra=len(spectra),
            )

    def _on_apply_fourier_to_selection(self) -> None:
        """Copy the active run's FFT recipe to the other selected runs.

        Implements the "apply to series / all" affordance: the current run's
        generated Fourier configuration is copied onto each other selected run's
        FrequencyFFT representation, and their spectra are (re)generated.  This
        keeps a series consistently configured for comparison without retuning
        each run by hand.
        """
        if self._current_dataset is None or self._current_dataset.run is None:
            self._set_fourier_status("Select a run before applying Fourier settings.")
            return
        source_run = int(self._current_dataset.run_number)
        source_rep = self._project_model.representation(source_run, RepresentationType.FREQ_FFT)
        if source_rep is None or not source_rep.recipe.get("fourier_config"):
            self._set_fourier_status("Compute an FFT first, then apply it to the selection.")
            return

        config_dict = dict(source_rep.recipe["fourier_config"])
        applied = 0
        for dataset in self._data_browser.get_selected_datasets():
            if dataset.run is None:
                continue
            run_number = int(dataset.run_number)
            if run_number == source_run:
                continue
            representation = self._project_model.ensure_dataset(run_number).ensure(
                RepresentationType.FREQ_FFT
            )
            representation.recipe = {"fourier_config": dict(config_dict)}
            representation.invalidate()
            try:
                spectra = representation.ensure_computed(dataset.run)
            except (ValueError, RuntimeError):
                continue
            if spectra:
                self._frequency_spectra_by_run[run_number] = [spectra[0]]
                applied += 1

        if applied == 0:
            self._set_fourier_status("Select additional runs to apply the Fourier settings to.")
            return
        self._set_fourier_status(f"Applied Fourier settings to {applied} run(s).", success=True)
        self._log_panel.log(f"Applied Fourier settings to {applied} run(s).")

    def _record_frequency_maxent_recipe(
        self,
        run_number: int,
        config: MaxEntConfig,
        spectrum: MuonDataset,
        diagnostics: dict,
        *,
        total_cycles: int | None = None,
    ) -> None:
        """Persist the generation recipe and compact diagnostics for MaxEnt."""
        representation = self._project_model.ensure_dataset(int(run_number)).ensure(
            RepresentationType.FREQ_MAXENT
        )
        config_dict = config.to_dict()
        if total_cycles is not None:
            # The button's config carries only the last increment ("+1" -> 1);
            # the recipe must describe the cumulative resumed computation so a
            # from-scratch recompute reproduces the displayed spectrum.
            config_dict["outer_cycles"] = max(1, int(total_cycles))
        representation.recipe = {"maxent_config": config_dict}
        representation.result_metadata = {
            "cycles": int(diagnostics.get("cycles", [0])[-1]) if diagnostics.get("cycles") else 0,
            "diagnostics": dict(diagnostics),
        }
        representation.cache_datasets([spectrum])
        # No panel-state store write here: the recipe just recorded is served
        # on demand by _sync_maxent_panel_for_dataset's fallback chain, and an
        # unconditional write would clobber edits the user made (and stored
        # via a dataset switch) while the worker was running.

    def _on_restart_maxent(self) -> None:
        """Drop resumable MaxEnt state for the active run."""
        if self._maxent_thread is not None:
            self._set_fourier_status("Cancel the running MaxEnt calculation before restarting.")
            return
        if self._current_dataset is None:
            self._set_fourier_status("Select a run before restarting MaxEnt.")
            return
        run_number = int(self._current_dataset.run_number)
        self._maxent_state_by_run.pop(run_number, None)
        representation = self._project_model.representation(
            run_number, RepresentationType.FREQ_MAXENT
        )
        if representation is not None:
            representation.invalidate()
            representation.result_metadata = {}
        self._maxent_panel.set_diagnostics(None)
        self._set_fourier_status(f"Restarted MaxEnt state for run {run_number}.", success=True)

    def _maxent_workload_is_unsafe(self, estimate) -> bool:
        """Return whether a MaxEnt workload should ask for confirmation."""
        return bool(
            estimate.peak_dense_matrix_bytes >= _MAXENT_WARN_PEAK_MATRIX_BYTES
            or estimate.total_dense_matrix_bytes >= _MAXENT_WARN_TOTAL_MATRIX_BYTES
            or estimate.total_observations >= _MAXENT_WARN_TOTAL_OBSERVATIONS
        )

    def _confirm_maxent_workload(self, estimate, config: MaxEntConfig) -> bool:
        """Ask the user whether to continue with a risky MaxEnt workload."""
        if not self._maxent_workload_is_unsafe(estimate):
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Large MaxEnt calculation")
        msg.setText("The selected MaxEnt settings may be slow or memory intensive.")
        msg.setInformativeText(
            "\n".join(
                [
                    f"Run: {estimate.run_number}",
                    f"Selected groups: {estimate.selected_group_count}",
                    f"Time points per group: up to {estimate.max_time_points:,}",
                    f"Spectrum points: {estimate.n_spectrum_points:,}",
                    f"Dense-equivalent peak matrix: {_format_bytes(estimate.peak_dense_matrix_bytes)}",
                    f"Dense-equivalent matrices per pass: {_format_bytes(estimate.total_dense_matrix_bytes)}",
                    f"MaxEnt binning factor: {int(config.time_binning_factor)}",
                    "",
                    "Asymmetry evaluates the projection in chunks where possible, but this "
                    "setting still represents a large numerical workload.",
                    "Reducing the time range, increasing MaxEnt binning, or using fewer spectrum "
                    "points will usually make the calculation safer.",
                ]
            )
        )
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        proceed_btn = msg.addButton("Proceed anyway", QMessageBox.ButtonRole.AcceptRole)
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        return msg.clickedButton() == proceed_btn

    def _launch_maxent_worker(
        self,
        *,
        run,
        config: MaxEntConfig,
        cycles: int,
        state: MaxEntState | None,
        run_number: int,
    ) -> None:
        """Start a background MaxEnt worker."""
        self._maxent_thread = QThread(self)
        self._maxent_worker = MaxEntWorker(run, config, cycles=int(cycles), state=state)
        self._maxent_active_run_number = int(run_number)
        self._maxent_active_run = run
        self._maxent_active_config = config
        self._maxent_active_cycles = int(cycles)
        self._maxent_started_at = time.perf_counter()
        self._maxent_worker.moveToThread(self._maxent_thread)
        self._maxent_thread.started.connect(self._maxent_worker.run)
        self._maxent_worker.progress.connect(self._on_maxent_progress)
        self._maxent_worker.finished.connect(self._on_maxent_worker_finished)
        self._maxent_worker.error.connect(self._on_maxent_worker_error)
        self._maxent_worker.cancelled.connect(self._on_maxent_worker_cancelled)
        self._maxent_worker.finished.connect(self._maxent_thread.quit)
        self._maxent_worker.error.connect(self._maxent_thread.quit)
        self._maxent_worker.cancelled.connect(self._maxent_thread.quit)
        self._maxent_worker.finished.connect(self._maxent_worker.deleteLater)
        self._maxent_worker.error.connect(self._maxent_worker.deleteLater)
        self._maxent_worker.cancelled.connect(self._maxent_worker.deleteLater)
        self._maxent_thread.finished.connect(self._cleanup_maxent_thread)
        self._maxent_thread.finished.connect(self._maxent_thread.deleteLater)
        self._maxent_panel.set_busy(True)
        self._set_fourier_status(f"Computing MaxEnt spectrum for run {run_number}...")
        self._maxent_thread.start()

    def _on_cancel_maxent(self) -> None:
        """Request cancellation of the active MaxEnt worker."""
        if self._maxent_worker is None:
            self._set_fourier_status("No MaxEnt calculation is running.")
            return
        self._maxent_worker.cancel()
        self._set_fourier_status("Cancelling MaxEnt calculation...")

    def _on_maxent_progress(self, current: int, total: int, message: str) -> None:
        """Update MaxEnt progress from the worker."""
        self._maxent_panel.set_progress(current, total, message)
        self.statusBar().showMessage(str(message))

    def _on_maxent_worker_finished(self, result) -> None:
        """Store and display a completed MaxEnt result."""
        run_number = self._maxent_active_run_number
        if run_number is None:
            run_number = int(result.metadata.get("run_number", 0))
        self._maxent_state_by_run[int(run_number)] = result.state
        spectrum = result.as_dataset(self._maxent_active_run)
        diagnostics = result.diagnostics.to_dict()
        config = self._maxent_active_config or MaxEntConfig()
        self._record_frequency_maxent_recipe(
            int(run_number),
            config,
            spectrum,
            diagnostics,
            total_cycles=int(result.state.cycle),
        )
        self._store_frequency_spectra_for_run(
            int(run_number),
            [spectrum],
            rep_type=RepresentationType.FREQ_MAXENT,
        )
        self._maxent_panel.set_diagnostics(diagnostics)
        if hasattr(self._plot_workspace, "set_available_views"):
            enabled = set(self._plot_workspace.enabled_views())
            enabled.add("maxent")
            self._plot_workspace.set_available_views(sorted(enabled))
        # Jump to the result only while the computed run is still the browser
        # selection; if the user navigated elsewhere mid-compute, leave both
        # their view and plot alone (the result stays cached for later).
        current_run = (
            None if self._current_dataset is None else int(self._current_dataset.run_number)
        )
        if current_run == int(run_number):
            already_maxent = self._plot_workspace.active_view() == "maxent"
            self._plot_workspace.set_active_view("maxent")
            if already_maxent:
                self._sync_frequency_plot_for_run(int(run_number), preserve_x_limits=True)
            self._show_panel("fourier")
        message = (
            f"Computed MaxEnt spectrum for run {run_number} through cycle {result.state.cycle}."
        )
        self._set_fourier_status(message, success=True)
        self._log_panel.log(message)
        self._log_maxent_perf()

    def _on_maxent_worker_error(self, message: str) -> None:
        """Handle a failed MaxEnt worker."""
        run_number = self._maxent_active_run_number
        if run_number is not None and "incompatible" in str(message).lower():
            self._maxent_state_by_run.pop(int(run_number), None)
            message = f"{message} Restart MaxEnt, then run cycles again."
        self._set_fourier_status(f"MaxEnt failed: {message}")
        self._log_panel.log(f"MaxEnt failed: {message}")
        self._log_maxent_perf()

    def _on_maxent_worker_cancelled(self) -> None:
        """Handle worker cancellation."""
        self._set_fourier_status("MaxEnt calculation cancelled.")
        self._log_panel.log("MaxEnt calculation cancelled.")
        self._log_maxent_perf()

    def _cleanup_maxent_thread(self) -> None:
        """Clear MaxEnt worker state after the thread exits."""
        self._maxent_panel.set_busy(False)
        self._maxent_thread = None
        self._maxent_worker = None
        self._maxent_active_run_number = None
        self._maxent_active_run = None
        self._maxent_active_config = None
        self._maxent_active_cycles = 0
        self._maxent_started_at = None

    def _log_maxent_perf(self) -> None:
        """Record MaxEnt timing if performance logging is enabled."""
        started_at = self._maxent_started_at
        if started_at is None:
            return
        self._log_perf_event(
            "compute_maxent",
            started_at,
            run=self._maxent_active_run_number,
            cycles=self._maxent_active_cycles,
        )

    def _on_compute_maxent(self, cycles: int) -> None:
        """Compute or resume a MaxEnt spectrum for the active run."""
        if self._maxent_thread is not None:
            self._set_fourier_status("A MaxEnt calculation is already running.")
            return
        if self._current_dataset is None or self._current_dataset.run is None:
            self._set_fourier_status("Select a grouped run before computing MaxEnt.")
            return
        if not self._dataset_supports_maxent(self._current_dataset):
            self._set_fourier_status("The active run does not define grouped raw counts.")
            return

        # Store the in-table edits before syncing, mirroring the dataset-switch
        # flow: the sync rebuilds the group table from the per-run store, so an
        # unsaved phase/include edit would otherwise be wiped right before the
        # config is read.
        self._store_maxent_panel_state_for_dataset(self._current_dataset)
        self._sync_maxent_panel_for_dataset(self._current_dataset)
        run_number = int(self._current_dataset.run_number)
        config = self._maxent_panel.maxent_config(cycles=int(cycles))
        if (
            config.t_min_us is not None
            and config.t_max_us is not None
            and config.t_max_us <= config.t_min_us
        ):
            self._set_fourier_status("MaxEnt end time must be greater than the start time.")
            return
        state = self._maxent_state_by_run.get(run_number)
        estimate = estimate_maxent_workload(self._current_dataset.run, config)
        if not self._confirm_maxent_workload(estimate, config):
            self._set_fourier_status("MaxEnt calculation cancelled before launch.")
            return
        self._launch_maxent_worker(
            run=self._current_dataset.run,
            config=config,
            cycles=int(cycles),
            state=state,
            run_number=run_number,
        )

    def _on_apply_maxent_to_selection(self) -> None:
        """Copy the active run's MaxEnt recipe to other selected runs."""
        if self._current_dataset is None or self._current_dataset.run is None:
            self._set_fourier_status("Select a run before applying MaxEnt settings.")
            return
        source_run = int(self._current_dataset.run_number)
        source_rep = self._project_model.representation(source_run, RepresentationType.FREQ_MAXENT)
        config_dict = source_rep.maxent_config() if source_rep is not None else {}
        if not config_dict:
            self._set_fourier_status("Compute a MaxEnt spectrum first, then apply it.")
            return
        applied = 0
        for dataset in self._data_browser.get_selected_datasets():
            if dataset.run is None or not self._dataset_supports_maxent(dataset):
                continue
            run_number = int(dataset.run_number)
            if run_number == source_run:
                continue
            representation = self._project_model.ensure_dataset(run_number).ensure(
                RepresentationType.FREQ_MAXENT
            )
            representation.recipe = {"maxent_config": dict(config_dict)}
            representation.invalidate()
            # This discards any previous result outright, so the persisted
            # diagnostics must go with it (invalidate deliberately keeps them).
            representation.result_metadata = {}
            self._frequency_cache(RepresentationType.FREQ_MAXENT).pop(run_number, None)
            # Drop any stored panel draft so the new recipe (served by the
            # sync fallback chain, with group inclusion re-derived for the
            # target's own grouping) becomes the panel state on next visit.
            self._maxent_panel_state_by_run.pop(run_number, None)
            applied += 1
        if applied == 0:
            self._set_fourier_status("Select additional grouped runs to apply MaxEnt settings to.")
            return
        message = (
            f"Applied MaxEnt settings to {applied} run(s); compute each run to update spectra."
        )
        self._set_fourier_status(message, success=True)
        self._log_panel.log(message)

    def _on_fit_parameters(self) -> None:
        """Show and raise the Fitted Parameters dock panel."""
        self._show_panel("fit_parameters")
        self._log_panel.log("Opened Fit Parameters panel")

    def _on_show_data(self) -> None:
        """Show and raise the data browser panel for the current layout mode."""
        self._show_panel("data")

    def _on_show_log(self) -> None:
        """Show and raise the log panel for the current layout mode."""
        self._show_panel("log")

    def set_compact_mode(self, enabled: bool) -> None:
        """Legacy compatibility shim after compact-mode removal."""
        del enabled
        self.compact_mode = False

    def _on_global_parameter_fit(self) -> None:
        """Show the Global Parameter Fit window if cross-group results exist."""
        if (
            self._global_parameter_fit_window is None
            or not self._global_parameter_fit_window.has_result()
        ):
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

    def _on_gle_setup(self) -> None:
        """Open the GLE executable configuration dialog."""
        dialog = GleSetupDialog(self)
        dialog.exec()

    def _on_about(self) -> None:
        """Show the About dialog with version information."""
        from asymmetry import __version__

        QMessageBox.about(
            self,
            "About Asymmetry",
            f"Asymmetry v{__version__}\n\nA Python library for μSR data analysis.",
        )

    def _reset_layout(self) -> None:
        """Reset dock panels to the default compact-friendly layout."""
        self._ui_manager.reset_layout()

    def _on_dataset_selected(self, run_number: int) -> None:
        """Handle dataset selection from data browser."""
        started_at = time.perf_counter()
        dataset = None
        self._store_fourier_group_phase_state_for_dataset(self._current_dataset)
        self._store_maxent_panel_state_for_dataset(self._current_dataset)
        self._active_group_context = None
        if hasattr(self._plot_panel, "set_active_label_group"):
            self._plot_panel.set_active_label_group(None)
        try:
            dataset = self._data_browser.get_dataset(run_number)
            if dataset:
                self._current_dataset = dataset
                self._sync_fourier_panel_for_dataset(dataset)
                if hasattr(self._frequency_plot_panel, "update_frequency_reference"):
                    self._frequency_plot_panel.update_frequency_reference(dataset)
                active_axis = None
                if hasattr(self._plot_panel, "get_current_polarization_axis"):
                    active_axis = self._normalize_vector_axis(
                        self._plot_panel.get_current_polarization_axis()
                    )
                if active_axis in {"P_x", "P_y", "P_z"}:
                    self._synchronize_targets_to_axis([dataset], active_axis)
                self._render_current_selection_plot()
                self._sync_frequency_plot_for_current_dataset()
                self._refresh_vector_axis_selector()
                self._update_fit_block_state()
                if self._plot_workspace.active_domain() == "frequency":
                    if hasattr(self._fit_panel, "set_domain"):
                        self._fit_panel.set_domain("frequency")
                    self._fit_panel.set_dataset(self._active_frequency_fit_dataset())
                    self._set_frequency_fit_datasets_for_selection()
                    _fit_range = self._frequency_plot_panel.get_fit_range()
                else:
                    if hasattr(self._fit_panel, "set_domain"):
                        self._fit_panel.set_domain("time")
                    self._fit_panel.set_dataset(self._get_fit_dataset(dataset))
                    _fit_range = self._plot_panel.get_fit_range()
                if hasattr(self._fit_panel, "set_fit_range_display"):
                    self._fit_panel.set_fit_range_display(*_fit_range)
                if self._multi_group_fit_window is not None:
                    self._multi_group_fit_window.set_dataset(self._get_fit_dataset(dataset))
                    if hasattr(self._multi_group_fit_window, "set_fit_range_display"):
                        self._multi_group_fit_window.set_fit_range_display(*_fit_range)
                self._log_panel.log(f"Selected run {run_number}")
                self.statusBar().showMessage(f"Viewing run {run_number}")
        finally:
            self._log_perf_event(
                "dataset_selected",
                started_at,
                run=run_number,
                domain=self._plot_workspace.active_domain(),
                cached_frequency=int(run_number in self._frequency_cache()),
                **self._perf_dataset_metrics(dataset),
            )

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

    def _on_plot_time_view_changed(self, _mode: str) -> None:
        """Re-render the main time-domain plot after an explicit view switch."""
        if (
            hasattr(self._plot_workspace, "active_domain")
            and self._plot_workspace.active_domain() == "time"
        ):
            if hasattr(self._plot_workspace, "set_active_view"):
                self._plot_workspace.set_active_view(_mode)
        self._sync_fit_dock_mode()
        self._render_current_selection_plot()
        self._refresh_vector_axis_selector()
        self._update_fit_block_state()
        if self._current_dataset is not None:
            if _mode == "groups":
                self.statusBar().showMessage(
                    f"Viewing individual groups for run {self._current_dataset.run_label}"
                )
            else:
                self.statusBar().showMessage(f"Viewing run {self._current_dataset.run_label}")

    def _on_domain_button_clicked(self, view: str) -> None:
        """Switch the workspace to *view* and keep the Domain buttons in sync."""
        self._plot_workspace.set_active_view(view)
        self._sync_domain_buttons(self._plot_workspace.active_view())

    def _sync_domain_buttons(self, view: str) -> None:
        """Update toolbar Domain button checked state to match *view*."""
        _tokens = ("fb_asymmetry", "groups", "frequency", "maxent")
        for idx, btn in enumerate(getattr(self, "_domain_buttons", [])):
            btn.setChecked(_tokens[idx] == view)

    def _on_plot_workspace_view_changed(self, view: str) -> None:
        """Map top-level workspace tab changes onto the shared time plot panel state."""
        # Keep the spectrum dock's stacked panel in lockstep with the view for
        # every transition: leaving "maxent" for a time view must restore the
        # FFT controls (and the "Fourier" title), or FFT status messages land
        # on the hidden page while stale MaxEnt controls stay visible.
        self._sync_spectrum_panel_for_view(view)
        if view not in {"frequency", "maxent"}:
            if hasattr(self._plot_panel, "set_current_time_view_mode"):
                self._plot_panel.set_current_time_view_mode(view, emit_signal=False)
            self._on_plot_time_view_changed(view)
        else:
            self._sync_frequency_plot_for_current_dataset()
        self._sync_domain_buttons(view)
        self._apply_inspector_for_domain(view)
        self._update_status_selection()
        # ALC mode is only valid for the F-B asymmetry view; enable the toggle
        # there and auto-exit ALC mode when the user leaves it.
        is_fb = view == "fb_asymmetry"
        self._alc_mode_action.setEnabled(is_fb)
        if not is_fb and self._alc_mode_action.isChecked():
            self._alc_mode_action.setChecked(False)  # fires _on_alc_mode_toggled(False)
        # Refresh the trend panel for the newly-active representation.
        self._refresh_trend_panel()

    def _on_fit_range_changed(self, x_min: float, x_max: float) -> None:
        """Refresh fit inputs when the selected fit x-range changes."""
        if hasattr(self._fit_panel, "set_fit_range_display"):
            self._fit_panel.set_fit_range_display(x_min, x_max)
        # Keep the ALC fit panel's integration-window spinboxes in sync.
        self._alc_fit_panel.set_fit_range_display(x_min, x_max)
        if self._multi_group_fit_window is not None and hasattr(
            self._multi_group_fit_window, "set_fit_range_display"
        ):
            self._multi_group_fit_window.set_fit_range_display(x_min, x_max)
        if self._current_dataset is not None:
            if self._plot_workspace.active_domain() == "frequency":
                if hasattr(self._fit_panel, "set_domain"):
                    self._fit_panel.set_domain("frequency")
                self._fit_panel.set_dataset(self._active_frequency_fit_dataset())
                self._set_frequency_fit_datasets_for_selection()
            else:
                if hasattr(self._fit_panel, "set_domain"):
                    self._fit_panel.set_domain("time")
                self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))
            if self._multi_group_fit_window is not None:
                self._multi_group_fit_window.set_dataset(
                    self._get_fit_dataset(self._current_dataset)
                )
        self._update_selected_datasets()

    def _on_fit_range_edit_committed(self, x_min: float, x_max: float) -> None:
        """Push a spinbox-committed fit range from the Fit panel to the plot."""
        panel = (
            self._frequency_plot_panel
            if self._plot_workspace.active_domain() == "frequency"
            else self._plot_panel
        )
        panel.set_fit_range(x_min, x_max)

    def _on_fit_completed(self, fit_result, fitted_curve, component_curves) -> None:
        """Handle completed fit from fit panel."""
        t_fit, y_fit = fitted_curve
        fit_function = None
        if hasattr(self._fit_panel, "single_fit_formula_string"):
            fit_function = self._fit_panel.single_fit_formula_string()
        panel = (
            self._frequency_plot_panel
            if self._plot_workspace.active_domain() == "frequency"
            else self._plot_panel
        )
        panel.plot_fit(
            t_fit,
            y_fit,
            label="Fit",
            component_curves=component_curves,
            fit_result=fit_result,
            fit_function=fit_function,
        )
        self._last_fit_chi2 = float(fit_result.reduced_chi_squared)
        self._record_single_fit_slot(fit_result)
        self._log_panel.log(f"Fit completed: χ²ᵣ = {fit_result.reduced_chi_squared:.4f}", tag="fit")

    def _active_representation_type(self) -> RepresentationType | None:
        """Map the active workspace view to its representation type."""
        if not hasattr(self, "_plot_workspace"):
            return None
        return {
            "fb_asymmetry": RepresentationType.TIME_FB_ASYMMETRY,
            "groups": RepresentationType.TIME_GROUPS,
            "frequency": RepresentationType.FREQ_FFT,
            "maxent": RepresentationType.FREQ_MAXENT,
        }.get(self._plot_workspace.active_view())

    @staticmethod
    def _fit_result_summary(fit_result) -> dict:
        """Return a JSON-serialisable summary of a fit result.

        Delegates to the shared core helper so the run-batch and grouped-series
        recording paths produce identically shaped ``results_by_run`` entries.
        """
        return fit_result_summary(fit_result)

    # ── Representation-aware trend panel (Phase 4) ────────────────────────────

    def _build_series_rows(self, series: FitSeries) -> list[dict]:
        """Build the row-dict list for one ``FitSeries`` to pass to the trend panel.

        Each entry is a plain dict with keys ``run_number``, ``run_label``,
        ``field``, ``temperature``, ``values``, ``errors``.  Only successful
        members are included.

        For frequency-domain series, dataset metadata is read from the cached
        frequency spectra (``_frequency_spectra_by_run``) rather than the data
        browser, mirroring the old ``set_fit_results`` path's ``is_frequency_fit``
        branch.  The time-domain data browser is used as a fallback when the
        spectrum is not cached in memory.
        """
        rows: list[dict] = []
        is_frequency = series.rep_type in (
            RepresentationType.FREQ_FFT,
            RepresentationType.FREQ_MAXENT,
        )
        for member_key in series.member_run_numbers:
            summary = series.results_by_run.get(member_key)
            if not summary or not summary.get("success"):
                continue

            if series.member_kind == "groups":
                source_run = series.source_run_for(member_key)
                dataset = (
                    self._data_browser.get_dataset(source_run)
                    if hasattr(self._data_browser, "get_dataset")
                    else None
                )
                group_idx = abs(member_key) % 1000
                run_label = f"R{source_run}/G{group_idx}"
            else:
                # Frequency-domain spectra may carry richer / more-accurate
                # metadata than the time-domain entry in the data browser.
                dataset = None
                if is_frequency:
                    spectra = self._frequency_cache(series.rep_type).get(member_key, [])
                    dataset = spectra[0] if spectra else None
                if dataset is None and hasattr(self._data_browser, "get_dataset"):
                    dataset = self._data_browser.get_dataset(member_key)
                run_label = None

            meta = getattr(dataset, "metadata", {}) or {}
            if run_label is None:
                run_label = str(meta.get("run_label") or member_key)

            rows.append(
                {
                    "run_number": member_key,
                    "run_label": run_label,
                    "field": float(meta.get("field", 0.0)),
                    "temperature": float(meta.get("temperature", 0.0)),
                    "values": dict(summary.get("parameters", {})),
                    "errors": dict(summary.get("uncertainties", {})),
                }
            )
        return rows

    def _refresh_trend_panel(self) -> None:
        """Reload the trend panel from the project model for the active representation.

        This is the *pull*-based entry point called after every fit that records
        a ``FitSeries`` and whenever the active representation changes.  It
        replaces the old per-fit UUID push.
        """
        if not hasattr(self, "_fit_parameters_panel"):
            return
        rep_type = self._active_representation_type()
        if rep_type is None:
            return

        # Gather all series for the active representation, in creation order
        # (batch-N sorts before batch-(N+1) because IDs are "batch-<index>").
        series_for_rep = sorted(
            (s for s in self._project_model.batches.values() if s.rep_type == rep_type),
            key=lambda s: s.batch_id,
        )

        entries: list[tuple[str, str, list[dict]]] = []
        highlight_map: dict[str, list[int]] = {}
        for idx, series in enumerate(series_for_rep, start=1):
            row_dicts = self._build_series_rows(series)
            if not row_dicts:
                continue
            batch_id = series.batch_id
            name = series.display_name(f"Series {idx}")
            entries.append((batch_id, name, row_dicts))
            # Runs to highlight: source runs for group series, member keys for run series.
            if series.member_kind == "groups":
                highlight_map[batch_id] = sorted(set(series.member_source_run.values()))
            else:
                highlight_map[batch_id] = list(series.member_run_numbers)

        if hasattr(self._fit_parameters_panel, "load_representation_series"):
            self._fit_parameters_panel.load_representation_series(
                entries,
                highlight_runs_by_id=highlight_map,
            )

        if entries and hasattr(self, "_dock_fit_parameters"):
            self._dock_fit_parameters.show()
            self._dock_fit_parameters.raise_()
        elif not entries and hasattr(self._data_browser, "set_highlighted_runs"):
            # No series remain — ensure the browser highlight is cleared.
            self._data_browser.set_highlighted_runs(set())

    def _on_trend_series_selected(self, batch_id: str) -> None:
        """Highlight the member runs of the active fit series in the data browser."""
        series = self._project_model.batch(batch_id)
        if series is None:
            if hasattr(self._data_browser, "set_highlighted_runs"):
                self._data_browser.set_highlighted_runs(set())
            return
        if series.member_kind == "groups":
            runs = set(series.member_source_run.values())
        else:
            runs = set(series.member_run_numbers)
        if hasattr(self._data_browser, "set_highlighted_runs"):
            self._data_browser.set_highlighted_runs(runs)

    def _on_parameters_dock_visibility_changed(self, visible: bool) -> None:
        """Gate the FitSeries browser highlight on Parameters dock visibility.

        When the dock is hidden (e.g. user switches to the Fit tab), the red
        tint is cleared so it doesn't persist as a confusing indicator.  When
        the dock is re-shown, the highlight is restored directly (not through
        the signal chain) so the guard works correctly in both real and headless
        environments.
        """
        if not hasattr(self, "_data_browser"):
            return
        if not hasattr(self._data_browser, "set_highlighted_runs"):
            return
        if visible:
            active_id = getattr(self._fit_parameters_panel, "_active_group_id", None)
            if active_id:
                series = self._project_model.batch(active_id)
                if series is not None:
                    if series.member_kind == "groups":
                        runs = set(series.member_source_run.values())
                    else:
                        runs = set(series.member_run_numbers)
                    self._data_browser.set_highlighted_runs(runs)
        else:
            self._data_browser.set_highlighted_runs(set())

    def _on_series_rename_requested(self, batch_id: str, new_label: str) -> None:
        """Persist a user rename of a FitSeries and refresh the trend panel."""
        label = new_label.strip() or None
        if self._project_model.rename_batch(batch_id, label):
            display = new_label.strip() if new_label.strip() else batch_id
            self._log_panel.log(f'Renamed series {batch_id} to "{display}".', tag="fit")
            self.statusBar().showMessage(f'Series renamed to "{display}".')
            self._refresh_trend_panel()

    def _on_series_select_members_requested(self, batch_id: str) -> None:
        """Perform a true browser selection of a FitSeries' member datasets."""
        series = self._project_model.batch(batch_id)
        if series is None:
            return
        if series.member_kind == "groups":
            runs = set(series.member_source_run.values())
        else:
            runs = set(series.member_run_numbers)
        if hasattr(self._data_browser, "select_runs"):
            self._data_browser.select_runs(runs)

    def _on_series_delete_requested(self, batch_id: str) -> None:
        """Remove a FitSeries from the project and clear its dataset fits."""
        series = self._project_model.batch(batch_id)
        if series is None:
            return
        # A computed series (integral scan) owns no per-run FitSlots, so dropping
        # it must NOT clear the runs' fit overlays — those belong to a real fit
        # that may share the same run numbers.
        if series.is_computed:
            self._project_model.remove_batch(batch_id)
            self._refresh_trend_panel()
            return
        if series.member_kind == "groups":
            runs = list(series.member_source_run.values())
        else:
            runs = list(series.member_run_numbers)
        self._project_model.remove_batch(batch_id)
        # Clear fit panel and plot panel state for the affected runs.
        self._on_fit_parameters_group_fits_deleted(batch_id, runs)
        self._refresh_trend_panel()

    def _record_single_fit_slot(self, fit_result) -> None:
        """Write the active representation's single FitSlot into the project model."""
        rep_type = self._active_representation_type()
        if rep_type is None or self._current_dataset is None:
            return
        if not hasattr(self._fit_panel, "get_single_state"):
            return
        state = self._fit_panel.get_single_state()
        if not isinstance(state, dict):
            return
        run_number = int(self._current_dataset.run_number)
        representation = self._project_model.ensure_dataset(run_number).ensure(rep_type)
        representation.fit = FitSlot(
            model=state.get("composite_model"),
            parameters=[dict(p) for p in state.get("parameters", []) if isinstance(p, dict)],
            result={
                **self._fit_result_summary(fit_result),
                "result_html": state.get("result_html"),
            },
            provenance="single",
        )
        # Editing a batch member's model via a single fit may diverge it.
        self._project_model.refresh_divergence()

    def _record_global_fit_batch(self, normalized_payloads: dict, global_params) -> None:
        """Persist a completed batch/global fit as a FitSeries + member FitSlots.

        A batch fit (all parameters local/fixed) and a global fit (>=1 parameter
        classified ``global``) are the same operation; the parameter classifier
        decides which.  Each member's representation gets a FitSlot pointing back
        to the batch, and the batch carries the run-by-run results for trending.
        """
        rep_type = self._active_representation_type()
        if rep_type is None or not normalized_payloads:
            return
        if not hasattr(self._fit_panel, "get_global_state"):
            return
        state = self._fit_panel.get_global_state()
        if not isinstance(state, dict):
            return

        param_roles: dict[str, str] = {}
        for entry in state.get("parameters", []):
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            role = str(entry.get("type", "local")).strip().lower()
            if role not in ("global", "local", "fixed"):
                role = "local"
            param_roles[str(entry["name"])] = role
        provenance = "global" if any(r == "global" for r in param_roles.values()) else "batch"

        member_runs = sorted(int(r) for r in normalized_payloads)
        canonical_model = state.get("composite_model")
        results_by_run = {
            int(run): self._fit_result_summary(payload[0])
            for run, payload in normalized_payloads.items()
        }
        batch = FitSeries(
            f"batch-{self._next_batch_index}",
            rep_type,
            member_run_numbers=member_runs,
            order_key="field",
            canonical_model=canonical_model,
            param_roles=param_roles,
            results_by_run=results_by_run,
        )
        self._next_batch_index += 1

        runs_by_number: dict[int, object] = {}
        if hasattr(self._data_browser, "get_dataset"):
            for run in member_runs:
                dataset = self._data_browser.get_dataset(run)
                if dataset is not None and dataset.run is not None:
                    runs_by_number[run] = dataset.run
        batch.sort_members(runs_by_number)
        self._project_model.add_batch(batch)

        template_parameters = [dict(p) for p in state.get("parameters", []) if isinstance(p, dict)]
        for run, payload in normalized_payloads.items():
            representation = self._project_model.ensure_dataset(int(run)).ensure(rep_type)
            representation.fit = FitSlot(
                model=canonical_model,
                parameters=template_parameters,
                result=self._fit_result_summary(payload[0]),
                provenance=provenance,
                batch_id=batch.batch_id,
            )
        # Fresh batch members all share the canonical model (no divergence yet).
        self._project_model.refresh_divergence()

    #: Display quantity name for the integral-asymmetry scan (percent units).
    _SCAN_QUANTITY = "Integral asymmetry (%)"

    def _on_scan_requested(self) -> None:
        """Build an integral-asymmetry field scan (ALC mode) from the selected runs.

        Integrates each run's asymmetry over the current fit-range window (one
        value per run, in percent), records it as a model-less ("computed")
        ``FitSeries`` for persistence/series management, and renders it in the
        bespoke ALC scan view.
        """
        rep_type = self._active_representation_type()
        if rep_type != RepresentationType.TIME_FB_ASYMMETRY:
            QMessageBox.information(
                self,
                "Integral scan",
                "Integral-scan (ALC) mode applies to the Forward-Backward asymmetry only.",
            )
            return

        datasets = (
            self._fit_panel.batch_datasets() if hasattr(self._fit_panel, "batch_datasets") else []
        )
        runs = [d.run for d in datasets if getattr(d, "run", None) is not None]
        if len(runs) < 2:
            QMessageBox.information(
                self, "Integral scan", "Select at least two runs to build a field scan."
            )
            return

        t_min = t_max = None
        if hasattr(self._plot_panel, "get_fit_range"):
            t_min, t_max = self._plot_panel.get_fit_range()

        try:
            # order_key="field": an ALC/repolarisation scan is plotted vs field,
            # so field is both the ordering and exclusion key (a run with no
            # field log is not a valid scan point and is dropped + logged).
            scan = build_field_scan(
                runs, t_min=t_min, t_max=t_max, method="integral", order_key="field"
            )
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Integral scan", f"Could not build the scan: {exc}")
            return
        if scan.n_points < 2:
            QMessageBox.warning(
                self, "Integral scan", "The scan has too few usable points to plot."
            )
            return

        if scan.excluded:
            dropped = ", ".join(f"{run} ({reason})" for run, reason in scan.excluded)
            self._log_panel.log(f"Integral scan: excluded {len(scan.excluded)} run(s): {dropped}")

        # The core integral is fractional; display/store in percent to match the
        # time-domain asymmetry plots.
        value_pct = scan.value * 100.0
        error_pct = scan.error * 100.0
        quantity = self._SCAN_QUANTITY
        results_by_run = {
            int(run_number): {
                "success": True,
                "parameters": {quantity: float(value)},
                "uncertainties": {quantity: float(error)},
                "chi_squared": 0.0,
                "reduced_chi_squared": 0.0,
            }
            for run_number, value, error in zip(scan.run_numbers, value_pct, error_pct)
        }
        series = FitSeries(
            f"scan-{self._next_scan_index}",
            rep_type,
            label=f"Integral scan {self._next_scan_index}",
            member_run_numbers=list(scan.run_numbers),
            order_key="field",
            canonical_model=None,
            param_roles={},
            results_by_run=results_by_run,
        )
        self._next_scan_index += 1
        runs_by_number = {
            int(d.run.run_number): d.run for d in datasets if getattr(d, "run", None) is not None
        }
        series.sort_members(runs_by_number)
        self._project_model.add_batch(series)

        # Render in the bespoke ALC view (x = field, y = integral asymmetry %).
        self._alc_scan_view.show_scan(
            scan.x,
            value_pct,
            error_pct,
            list(scan.run_numbers),
            x_label=scan.x_label,
            y_label=quantity,
        )
        self._log_panel.log(f"Built integral scan '{series.label}' ({scan.n_points} points).")

    def _active_grouped_state(self) -> dict:
        """Return the grouped-fit classification from the active grouped surface.

        Prefers the multi-group fit window (the real grouped-fit surface when the
        Individual-groups representation is active); falls back to the fit panel.
        """
        if (
            self._active_representation_type() == RepresentationType.TIME_GROUPS
            and self._multi_group_fit_window is not None
            and hasattr(self._multi_group_fit_window, "get_grouped_state")
        ):
            state = self._multi_group_fit_window.get_grouped_state()
            if isinstance(state, dict) and state:
                return state
        if hasattr(self._fit_panel, "get_grouped_state"):
            return self._fit_panel.get_grouped_state()
        return {}

    def _record_grouped_fit_series(self, grouped_datasets, results_dict) -> None:
        """Persist a completed grouped fit as a ``FitSeries(member_kind="groups")``.

        Each ``(run, group)`` member is keyed by its synthetic group key so the
        series' ``results_by_run`` drives parameter trending exactly like a run
        series.  The two-tier classification is recorded as physics ``param_roles``
        plus the always-per-group ``nuisance_params`` block.  Each source run's
        grouped representation gets one pointer ``FitSlot`` into the series.
        """
        if not isinstance(grouped_datasets, list) or not isinstance(results_dict, dict):
            return
        if not results_dict:
            return
        state = self._active_grouped_state()
        if not isinstance(state, dict) or not state:
            return

        rep_type = RepresentationType.TIME_GROUPS
        source_by_key: dict[int, int] = {}
        for dataset in grouped_datasets:
            metadata = getattr(dataset, "metadata", {}) or {}
            try:
                key = int(metadata.get("run_number"))
            except (TypeError, ValueError):
                continue
            source_by_key[key] = int(metadata.get("source_run_number", abs(key) // 1000))

        member_keys: list[int] = []
        member_source_run: dict[int, int] = {}
        results_by_run: dict[int, dict] = {}
        for raw_key, payload in results_dict.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            fit_result = payload[0] if isinstance(payload, tuple) and payload else payload
            member_keys.append(key)
            member_source_run[key] = source_by_key.get(key, abs(key) // 1000)
            results_by_run[key] = self._fit_result_summary(fit_result)
        if not member_keys:
            return

        physics_roles = {
            str(name): str(role)
            for name, role in (state.get("param_roles") or {}).items()
            if role in ("global", "local", "fixed")
        }
        nuisance_params = [str(name) for name in (state.get("nuisance_params") or [])]
        canonical_model = state.get("composite_model")
        provenance = "global" if any(r == "global" for r in physics_roles.values()) else "batch"

        series = FitSeries(
            f"batch-{self._next_batch_index}",
            rep_type,
            member_kind="groups",
            member_run_numbers=member_keys,
            member_source_run=member_source_run,
            order_key="run",
            canonical_model=canonical_model,
            param_roles=physics_roles,
            nuisance_params=nuisance_params,
            results_by_run=results_by_run,
        )
        self._next_batch_index += 1

        runs_by_number: dict[int, object] = {}
        if hasattr(self._data_browser, "get_dataset"):
            for run in set(member_source_run.values()):
                dataset = self._data_browser.get_dataset(run)
                if dataset is not None and dataset.run is not None:
                    runs_by_number[run] = dataset.run
        series.sort_members(runs_by_number)
        self._project_model.add_batch(series)

        for run in sorted(set(member_source_run.values())):
            representation = self._project_model.ensure_dataset(int(run)).ensure(rep_type)
            representation.fit = FitSlot(
                model=canonical_model,
                result={"series_id": series.batch_id},
                provenance=provenance,
                batch_id=series.batch_id,
            )
        # Fresh group-series members all share the canonical model; clear stale
        # divergence state from any earlier series on the same representations.
        self._project_model.refresh_divergence()

    def _add_single_fit_to_series(self, run_number: int, series_id: str) -> bool:
        """Add a compatible single fit (one run) as a member of an existing series.

        Compatibility = the run's stored single-fit model matches the series'
        canonical model (reuses ``canonical_model_matches``). Run-membered series
        only; group series grow by re-running the batch. Returns ``True`` when the
        member was added.
        """
        series = self._project_model.batch(str(series_id))
        if series is None or series.member_kind != "runs":
            return False
        representation = self._project_model.representation(int(run_number), series.rep_type)
        if representation is None or representation.fit.is_empty():
            return False
        if not canonical_model_matches(representation.fit.model, series.canonical_model):
            return False

        run_number = int(run_number)
        series.add_member(run_number)
        representation.fit.batch_id = series.batch_id
        representation.fit.provenance = "global" if series.is_global() else "batch"
        if isinstance(representation.fit.result, dict):
            series.results_by_run[run_number] = dict(representation.fit.result)

        runs_by_number: dict[int, object] = {}
        if hasattr(self._data_browser, "get_dataset"):
            for member in series.member_run_numbers:
                dataset = self._data_browser.get_dataset(member)
                if dataset is not None and dataset.run is not None:
                    runs_by_number[member] = dataset.run
        series.sort_members(runs_by_number)
        self._project_model.refresh_divergence()
        return True

    def _series_fallback_name(self, series) -> str:
        """Return the positional "Series N" fallback label consistent with the trend panel."""
        rep_type = series.rep_type
        series_for_rep = sorted(
            (s for s in self._project_model.batches.values() if s.rep_type == rep_type),
            key=lambda s: s.batch_id,
        )
        for idx, s in enumerate(series_for_rep, start=1):
            if s.batch_id == series.batch_id:
                return f"Series {idx}"
        return series.batch_id

    def _on_add_single_fit_to_series_requested(self) -> None:
        """Handle the Single tab's 'Add to Series…' action.

        Finds run-membered series whose canonical model matches the active run's
        single fit and adds the run to one (prompting if several match). The
        trend panel is refreshed after a successful addition.
        """
        if self._current_dataset is None:
            self.statusBar().showMessage("Select a run with a single fit to add to a series.")
            return
        run = int(self._current_dataset.run_number)
        rep_type = self._active_representation_type()
        if rep_type is None:
            return
        representation = self._project_model.representation(run, rep_type)
        if representation is None or representation.fit.is_empty():
            self.statusBar().showMessage(f"Run {run} has no single fit to add — fit it first.")
            return

        compatible = [
            series
            for series in self._project_model.batches.values()
            if series.member_kind == "runs"
            and series.rep_type == rep_type
            and run not in series.member_run_numbers
            and canonical_model_matches(representation.fit.model, series.canonical_model)
        ]
        if not compatible:
            self.statusBar().showMessage(f"No compatible batch series for run {run}'s fit.")
            return

        if len(compatible) == 1:
            series = compatible[0]
        else:
            labels = [
                f"{s.display_name(self._series_fallback_name(s))} ({len(s.member_run_numbers)} runs)"
                for s in compatible
            ]
            choice, ok = QInputDialog.getItem(
                self, "Add to Series", "Compatible series:", labels, 0, False
            )
            if not ok:
                return
            series = compatible[labels.index(choice)]

        series_label = series.display_name(self._series_fallback_name(series))
        if self._add_single_fit_to_series(run, series.batch_id):
            self._log_panel.log(f"Added run {run} to {series_label}.", tag="fit")
            self.statusBar().showMessage(f"Added run {run} to {series_label}.")
            self._refresh_trend_panel()

    def _on_preview_requested(self, fit_result, fitted_curve, component_curves) -> None:
        """Handle preview request from fit panel."""
        t_fit, y_fit = fitted_curve
        fit_function = None
        if hasattr(self._fit_panel, "single_fit_formula_string"):
            fit_function = self._fit_panel.single_fit_formula_string()
        panel = (
            self._frequency_plot_panel
            if self._plot_workspace.active_domain() == "frequency"
            else self._plot_panel
        )
        panel.plot_fit(
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

        # Resolve target datasets so that file-specific parameter defaults
        # (e.g. B_L from the run's applied field) can be seeded per member.
        all_browser_datasets: dict[int, MuonDataset] = {}
        if hasattr(self._data_browser, "_datasets"):
            for rn, ds in self._data_browser._datasets.items():
                try:
                    all_browser_datasets[int(rn)] = ds
                except (TypeError, ValueError):
                    pass

        updated = 0
        if hasattr(self._fit_panel, "share_single_function_state"):
            updated = int(
                self._fit_panel.share_single_function_state(
                    source_run_number,
                    target_runs,
                    datasets_by_run=all_browser_datasets or None,
                )
            )

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

    def _on_global_fit_started(self) -> None:
        """Snapshot launch-time fit context before any UI refresh changes it."""
        self._active_global_fit_rep_type = self._last_frequency_fit_rep_type

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
        is_frequency_fit = (
            hasattr(self._fit_panel, "domain") and self._fit_panel.domain() == "frequency"
        )
        global_fit_function = None
        if hasattr(self._fit_panel, "global_fit_formula_string"):
            global_fit_function = self._fit_panel.global_fit_formula_string()
        fit_curves = {}
        for run_number, (result, fitted_curve, component_curves) in normalized_payloads.items():
            t_fit, y_fit = fitted_curve
            axis_key = None
            dataset = (
                # Resolve against the representation snapshotted at fit
                # launch, not the active view at result-arrival time (and not
                # the collection pin, which view/selection refreshes rewrite
                # mid-fit).
                self._frequency_cache(self._active_global_fit_rep_type).get(run_number, [None])[0]
                if is_frequency_fit
                else self._data_browser.get_dataset(run_number)
            )
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
                "Batch Fit",
                component_curves,
                result,
                global_fit_function,
                axis_key,
            )

        self._fit_panel.register_global_fit_results(normalized_payloads)
        self._record_global_fit_batch(normalized_payloads, global_params)

        # Set all fit curves in plot panel
        panel = self._frequency_plot_panel if is_frequency_fit else self._plot_panel
        panel.set_global_fits(fit_curves)

        # Reload the trend panel from the project model (pull-based, Phase 4).
        # _record_global_fit_batch has already stored the new FitSeries, so
        # _refresh_trend_panel picks it up keyed by series batch_id rather than
        # the old ad-hoc UUID group_id.
        self._refresh_trend_panel()

        # Log summary
        successful_results = [
            payload for payload in normalized_payloads.values() if payload and payload[0].success
        ]
        n_datasets = len(successful_results)
        if n_datasets == 0:
            self._log_panel.log(
                "Batch fit completed but no successful dataset results were available"
            )
            self.statusBar().showMessage("Batch fit completed with no successful results")
            return

        avg_chi2r = (
            sum(payload[0].reduced_chi_squared for payload in successful_results) / n_datasets
        )
        self._last_fit_chi2 = float(avg_chi2r)
        self._log_panel.log(
            f"Batch fit completed: {n_datasets} datasets, average χ²ᵣ = {avg_chi2r:.3f}",
            tag="fit",
        )
        self.statusBar().showMessage(f"Batch fit completed for {n_datasets} datasets")

    def _on_grouped_fit_completed(
        self,
        grouped_datasets,
        results_dict,
        fit_function: str | None = None,
    ) -> None:
        """Handle completed grouped time-domain fit."""
        if not isinstance(grouped_datasets, list) or not isinstance(results_dict, dict):
            return

        self._record_grouped_fit_series(grouped_datasets, results_dict)
        # Pull-based refresh: show the newly recorded series in the trend panel.
        self._refresh_trend_panel()

        fit_curves = {}
        if fit_function is None and hasattr(self._fit_panel, "global_fit_formula_string"):
            fit_function = self._fit_panel.global_fit_formula_string()
        for run_number, payload in results_dict.items():
            if not isinstance(payload, tuple) or len(payload) < 2:
                continue
            fit_result = payload[0]
            fitted_curve = payload[1]
            component_curves = payload[2] if len(payload) >= 3 else []
            t_fit, y_fit = fitted_curve
            fit_curves[int(run_number)] = (
                t_fit,
                y_fit,
                "Grouped Fit",
                component_curves,
                fit_result,
                fit_function,
                None,
            )

        self._plot_panel.set_global_fits(fit_curves)
        if hasattr(self._plot_panel, "plot_grouped_time_domain_subplots"):
            self._plot_panel.plot_grouped_time_domain_subplots(grouped_datasets)
        self._log_panel.log(f"Grouped time-domain fit completed: {len(grouped_datasets)} groups")
        self.statusBar().showMessage(
            f"Grouped time-domain fit completed: {len(grouped_datasets)} groups"
        )

    def _on_grouped_preview_requested(
        self,
        grouped_datasets,
        preview_curves,
        fit_function: str | None = None,
    ) -> None:
        """Handle grouped time-domain preview requests from the grouped fit dock."""
        if not isinstance(grouped_datasets, list) or not isinstance(preview_curves, dict):
            return

        fit_payloads = {}
        for run_number, payload in preview_curves.items():
            if not isinstance(payload, tuple) or len(payload) < 2:
                continue
            fitted_curve = payload[1]
            component_curves = payload[2] if len(payload) >= 3 else []
            t_fit, y_fit = fitted_curve
            fit_payloads[int(run_number)] = (
                t_fit,
                y_fit,
                "Grouped Preview",
                component_curves,
                None,
                fit_function,
                None,
            )

        self._plot_panel.set_global_fits(fit_payloads)
        if hasattr(self._plot_panel, "plot_grouped_time_domain_subplots"):
            self._plot_panel.plot_grouped_time_domain_subplots(grouped_datasets)

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
            active_axis = self._normalize_vector_axis(
                self._plot_panel.get_current_polarization_axis()
            )

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
                self._frequency_plot_panel.clear()
                self._fit_panel.set_dataset(None)

            self._refresh_time_view_selector()

        # Multi-selection render mode depends on the plot-panel Overlay toggle.
        if len(selected) > 1:
            self._render_current_selection_plot()
            self._refresh_vector_axis_selector()
            self._update_fit_block_state()
            if self._overlay_enabled():
                self.statusBar().showMessage(self._selection_status_message(selected))
            elif self._current_dataset is not None:
                self.statusBar().showMessage(f"Viewing run {self._current_dataset.run_label}")
        elif self._current_dataset is not None:
            self._refresh_vector_axis_selector()
            self._update_fit_block_state()
        else:
            self._update_fit_block_state()
        self._update_status_selection()

        is_frequency_domain = (
            hasattr(self, "_plot_workspace") and self._plot_workspace.active_domain() == "frequency"
        )
        if hasattr(self, "_fit_panel") and hasattr(self._fit_panel, "set_domain"):
            self._fit_panel.set_domain("frequency" if is_frequency_domain else "time")

        if is_frequency_domain:
            analysis_datasets = self._frequency_fit_datasets_for_selected_runs()
        else:
            analysis_datasets = [
                dataset
                for dataset in (self._get_fit_dataset(ds) for ds in selected)
                if dataset is not None
            ]

        # Refresh the single-fit tab with the currently active dataset so that
        # bunch-factor or fit-range changes are reflected immediately.
        if is_frequency_domain:
            self._fit_panel.set_dataset(self._active_frequency_fit_dataset())
        elif self._current_dataset is not None:
            self._fit_panel.set_dataset(self._get_fit_dataset(self._current_dataset))

        self._fit_panel.set_datasets(analysis_datasets)
        # The grouped surface fits a *series* across the selected runs.
        if (
            self._multi_group_fit_window is not None
            and not is_frequency_domain
            and hasattr(self._multi_group_fit_window, "set_member_datasets")
        ):
            self._multi_group_fit_window.set_member_datasets(analysis_datasets)
        if is_frequency_domain:
            self._apply_frequency_missing_spectra_status(len(analysis_datasets))

    def _selection_status_message(self, selected: list) -> str:
        """Return a compact status message for multi-run selections."""
        run_labels = [str(ds.run_label) for ds in selected]
        if len(run_labels) <= 12:
            return f"Viewing runs {', '.join(run_labels)}"
        preview = ", ".join(run_labels[:12])
        return f"Viewing {len(run_labels)} runs: {preview}, ..."

    def _update_status_selection(self) -> None:
        """Refresh the center status bar label with current selection + domain."""
        if not hasattr(self, "_status_sel_label"):
            return
        all_ds = (
            self._data_browser.get_all_datasets()
            if hasattr(self._data_browser, "get_all_datasets")
            else []
        )
        selected = list(self._data_browser.get_selected_datasets())
        n_sel = len(selected)
        n_total = len(all_ds)
        _domain_labels = {
            "fb_asymmetry": "F-B asymmetry",
            "groups": "individual groups",
            "frequency": "frequency",
        }
        domain = _domain_labels.get(
            self._plot_workspace.active_view() if hasattr(self, "_plot_workspace") else "",
            "",
        )
        parts = []
        if n_sel == 1 and selected:
            parts.append(f"Run {selected[0].run_label}")
        elif n_sel > 1:
            parts.append(f"{n_sel} of {n_total} runs selected")
        elif n_total > 0:
            parts.append(f"{n_total} run{'s' if n_total != 1 else ''} loaded")
        if domain:
            parts.append(f"{domain} view")
        self._status_sel_label.setText(" · ".join(parts))

    def _on_cursor_coords_changed(self, x: object, y: object) -> None:
        """Update the status bar right label with the current cursor position."""
        if not hasattr(self, "_status_coords_label"):
            return
        if x is None or y is None:
            self._status_coords_label.setText("")
            return
        domain = self._plot_workspace.active_view() if hasattr(self, "_plot_workspace") else ""
        if domain == "frequency":
            text = f"ν = {float(x):.3f} MHz  |F| = {float(y):.4g}"
        else:
            text = f"x = {float(x):.3f} μs  y = {float(y):.2f} %"
            if self._last_fit_chi2 is not None:
                text += f"  χ²/ν = {self._last_fit_chi2:.3f}"
        self._status_coords_label.setText(text)

    def _get_fit_dataset(self, dataset):
        """Return analysis dataset restricted to the active fit range."""
        analysis_dataset = self._plot_panel.get_analysis_dataset(dataset)
        return self._plot_panel.get_fit_dataset(analysis_dataset)

    def _active_frequency_fit_dataset(self) -> MuonDataset | None:
        """Return the currently displayed Fourier spectrum clipped to its fit range."""
        if not hasattr(self, "_frequency_plot_panel"):
            return None
        dataset = getattr(self._frequency_plot_panel, "_current_dataset", None)
        if dataset is None:
            return None
        analysis_dataset = self._frequency_plot_panel.get_analysis_dataset(dataset)
        fit_dataset = self._frequency_plot_panel.get_fit_dataset(analysis_dataset)
        return self._frequency_dataset_with_fit_errors(fit_dataset)

    def _frequency_dataset_with_fit_errors(self, dataset: MuonDataset | None) -> MuonDataset | None:
        """Return a frequency dataset with finite positive errors for least squares."""
        if dataset is None:
            return None
        error = np.asarray(dataset.error, dtype=float)
        if error.shape == np.asarray(dataset.asymmetry).shape and np.any(
            np.isfinite(error) & (error > 0.0)
        ):
            safe_error = np.where(np.isfinite(error) & (error > 0.0), error, np.nan)
            fallback = float(np.nanmedian(safe_error))
            if not np.isfinite(fallback) or fallback <= 0.0:
                fallback = 1.0
            safe_error = np.where(np.isfinite(safe_error), safe_error, fallback)
        else:
            y = np.asarray(dataset.asymmetry, dtype=float)
            scale = float(np.nanstd(y)) if y.size else 1.0
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            safe_error = np.full_like(y, scale * 0.05, dtype=float)
        return MuonDataset(
            time=np.asarray(dataset.time, dtype=float).copy(),
            asymmetry=np.asarray(dataset.asymmetry, dtype=float).copy(),
            error=np.asarray(safe_error, dtype=float),
            metadata=dict(dataset.metadata),
            run=dataset.run,
        )

    def _frequency_fit_datasets_for_selected_runs(self) -> list[MuonDataset]:
        """Return cached spectra for selected browser runs in the active frequency view."""
        selected = self._data_browser.get_selected_datasets()
        datasets: list[MuonDataset] = []
        missing_run_numbers: list[int] = []
        # Pin the representation the fit datasets are collected from: the
        # async fit-completion handler must resolve run datasets against this
        # same cache, not whichever view happens to be active when the result
        # arrives (the user may switch FFT <-> MaxEnt mid-fit).
        rep_type = self._active_frequency_rep_type()
        self._last_frequency_fit_rep_type = rep_type
        for source in selected:
            try:
                run_number = int(source.run_number)
            except (TypeError, ValueError):
                continue
            spectra = list(self._frequency_cache(rep_type).get(run_number, []))
            if not spectra:
                missing_run_numbers.append(run_number)
                continue
            dataset = spectra[0]
            analysis_dataset = self._frequency_plot_panel.get_analysis_dataset(dataset)
            fit_dataset = self._frequency_plot_panel.get_fit_dataset(analysis_dataset)
            if fit_dataset is not None:
                safe_dataset = self._frequency_dataset_with_fit_errors(fit_dataset)
                if safe_dataset is not None:
                    datasets.append(safe_dataset)
        self._last_frequency_fit_missing_run_numbers = missing_run_numbers
        return datasets

    def _set_frequency_fit_datasets_for_selection(self) -> list[MuonDataset]:
        """Set frequency global-fit datasets and report selected uncached runs."""
        datasets = self._frequency_fit_datasets_for_selected_runs()
        self._fit_panel.set_datasets(datasets)
        self._apply_frequency_missing_spectra_status(len(datasets))
        return datasets

    def _apply_frequency_missing_spectra_status(self, cached_count: int) -> None:
        """Show an actionable V1 status for global frequency fits with uncached runs."""
        missing_run_numbers = list(getattr(self, "_last_frequency_fit_missing_run_numbers", []))
        if not missing_run_numbers:
            return
        if hasattr(self._fit_panel, "set_frequency_missing_spectra_status"):
            self._fit_panel.set_frequency_missing_spectra_status(missing_run_numbers, cached_count)
        preview = ", ".join(str(run_number) for run_number in missing_run_numbers[:5])
        if len(missing_run_numbers) > 5:
            preview += f", +{len(missing_run_numbers) - 5} more"
        self.statusBar().showMessage(
            f"Compute {self._frequency_status_name()} spectra for selected run(s) {preview} before global frequency fitting."
        )

    def _grouped_time_domain_display_datasets(
        self,
        dataset: MuonDataset | None = None,
    ) -> list[MuonDataset]:
        """Return grouped time-domain display datasets for the active dataset."""
        source = self._current_dataset if dataset is None else dataset
        if source is None:
            return []
        source_dataset = self._plot_panel.get_analysis_dataset(source)
        if source_dataset is None:
            return []
        try:
            return build_grouped_time_domain_datasets(source_dataset)
        except ValueError:
            return []

    # ── project save / open ────────────────────────────────────────────

    def _on_new_project(self) -> None:
        """Clear all state to start a fresh project."""
        reply = QMessageBox.question(
            self,
            "New Project",
            "Clear the current session and start a new project?\nUnsaved changes will be lost.",
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
        default = self._current_project_path or os.path.join(self._last_open_dir, "project.asymp")
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
            QMessageBox.critical(self, "Save Failed", f"Could not save project:\n{e}")
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

            datasets.append(
                {
                    "run_number": run_number,
                    "source_file": source_file,
                    "metadata_overrides": {
                        "field": float(dataset.metadata.get("field", 0.0)),
                    },
                    "grouping_overrides": self._extract_grouping_overrides(dataset),
                }
            )
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

        plot_state = self._plot_panel.get_state()
        plot_state["fit_curve"] = None
        plot_state["fit_curves"] = {}
        plot_state["fit_curves_by_key"] = {}
        plot_state["fit_components"] = None
        plot_state["fit_components_by_run"] = {}
        plot_state["fit_components_by_key"] = {}
        plot_state["workspace_state"] = self._plot_workspace.get_state()
        plot_state["frequency_plot_state"] = self._frequency_plot_panel.get_state()
        self._store_fourier_group_phase_state_for_dataset(self._current_dataset)
        self._store_maxent_panel_state_for_dataset(self._current_dataset)

        def _prune_single_fit_state(state: dict | None) -> dict | None:
            if not isinstance(state, dict):
                return state
            pruned = dict(state)
            pruned.pop("wizard_state", None)
            raw_states = pruned.get("states_by_run")
            if isinstance(raw_states, dict):
                pruned["states_by_run"] = {
                    str(run_number): {
                        key: value
                        for key, value in dict(run_state).items()
                        if key != "wizard_state"
                    }
                    for run_number, run_state in raw_states.items()
                    if isinstance(run_state, dict)
                }
            return pruned

        def _prune_global_fit_state(state: dict | None) -> dict | None:
            if not isinstance(state, dict):
                return state
            pruned = dict(state)
            pruned.pop("wizard_state", None)
            pruned.pop("wizard_state_by_run_set", None)
            return pruned

        time_fit_state = (
            self._fit_panel.get_domain_state("time")
            if hasattr(self._fit_panel, "get_domain_state")
            else {
                "single_fit_state": self._fit_panel.get_single_state(),
                "global_fit_state": self._fit_panel.get_global_state(),
                "fit_ui_state": self._fit_panel.get_ui_state(),
            }
        )
        frequency_fit_state = (
            self._fit_panel.get_domain_state("frequency")
            if hasattr(self._fit_panel, "get_domain_state")
            else {
                "domain": "frequency",
                "single_fit_state": {},
                "global_fit_state": {},
                "fit_ui_state": {},
            }
        )

        project_state = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "created_with_app_version": __version__,
            "datasets": datasets,
            "combined_datasets": combined_datasets,
            "browser_state": self._data_browser.get_state(),
            "plot_state": plot_state,
            "view_modes_state": self._collect_view_modes_state(),
            "single_fit_state": _prune_single_fit_state(time_fit_state.get("single_fit_state")),
            "global_fit_state": _prune_global_fit_state(time_fit_state.get("global_fit_state")),
            "multi_group_fit_state": _prune_global_fit_state(
                self._multi_group_fit_window.get_state()
                if self._multi_group_fit_window is not None
                and hasattr(self._multi_group_fit_window, "get_state")
                else None
            ),
            "fit_ui_state": time_fit_state.get("fit_ui_state", {}),
            "frequency_fit_state": {
                "domain": "frequency",
                "single_fit_state": _prune_single_fit_state(
                    frequency_fit_state.get("single_fit_state")
                ),
                "global_fit_state": _prune_global_fit_state(
                    frequency_fit_state.get("global_fit_state")
                ),
                "fit_ui_state": frequency_fit_state.get("fit_ui_state", {}),
            },
            "fit_parameters_state": self._fit_parameters_panel.get_state(),
            "global_parameter_fit_window_state": (
                self._global_parameter_fit_window.get_state()
                if self._global_parameter_fit_window is not None
                else None
            ),
            "fourier_state": {
                **self._fourier_panel.get_state(),
                "group_phase_state_by_run": {
                    str(run_number): dict(run_state)
                    for run_number, run_state in self._fourier_group_phase_state_by_run.items()
                },
            },
            "maxent_state": self._maxent_panel.get_state(),
            "maxent_state_by_run": {
                str(run_number): dict(run_state)
                for run_number, run_state in self._maxent_panel_state_by_run.items()
            },
            "fourier_spectra_state": self._serialize_frequency_spectra_state(),
        }
        # Recipe-only representation/batch state (v6).  Frequency spectra are
        # recomputed from these recipes on load; the array snapshot above is a
        # transitional fallback removed in cleanup.
        self._project_model.write_to_project_state(project_state)
        return project_state

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
            names = ", ".join(os.path.basename(d.get("source_file", "?")) for d in missing_info[:5])
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
                self._log_panel.log(f"WARNING: Run {rn} has no source file; skipping.")
                continue

            resolved = resolved_paths.get(rn)
            if not resolved:
                self._log_panel.log(f"WARNING: Source file not found: {source_file}; skipping.")
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
                combined_id_map.get(rn, rn) for rn in browser_state["selected_run_numbers"]
            ]
        self._data_browser.restore_state(browser_state)
        self._sync_temperature_log_option_action()

        # ── restore plot state ─────────────────────────────────────────
        plot_state = state.get("plot_state", {})
        current_run = plot_state.get("current_run_number")
        if current_run is not None:
            current_run = combined_id_map.get(int(current_run), int(current_run))
        current_dataset = (
            self._data_browser.get_dataset(current_run) if current_run is not None else None
        )
        if current_dataset is not None:
            self._current_dataset = current_dataset
        self._plot_panel.restore_state(plot_state, current_dataset)
        self._frequency_plot_panel.restore_state(plot_state.get("frequency_plot_state", {}), None)
        self._restore_view_modes_state(state.get("view_modes_state"))
        if state.get("view_modes_state") is not None:
            self._apply_view_mode(self._active_view_mode_index)
        else:
            self._snapshot_active_view_mode()
        self._plot_workspace.restore_state(plot_state.get("workspace_state"))
        self._restore_frequency_spectra_state(state.get("fourier_spectra_state"))
        self._restore_frequency_representations(state)
        if self._plot_workspace.active_domain() == "frequency":
            self._sync_frequency_plot_for_current_dataset()
        if (
            current_dataset is not None
            and hasattr(self._plot_panel, "get_fit_range")
            and hasattr(self._plot_panel, "_set_fit_range")
        ):
            fit_range = self._plot_panel.get_fit_range()
            if fit_range is not None:
                self._plot_panel._set_fit_range(
                    float(fit_range[0]),
                    float(fit_range[1]),
                    emit_signal=False,
                    redraw=True,
                )
                if hasattr(self._plot_panel, "get_view_limits") and hasattr(
                    self._plot_panel, "set_view_limits"
                ):
                    x_min, x_max, y_min, y_max = self._plot_panel.get_view_limits()
                    fit_min = float(min(fit_range[0], fit_range[1]))
                    fit_max = float(max(fit_range[0], fit_range[1]))
                    widened_x_min = min(float(x_min), fit_min)
                    widened_x_max = max(float(x_max), fit_max)
                    if widened_x_min != x_min or widened_x_max != x_max:
                        self._plot_panel.set_view_limits(
                            widened_x_min,
                            widened_x_max,
                            float(y_min),
                            float(y_max),
                        )
        if (
            self._plot_workspace.active_domain() == "frequency"
            and hasattr(self._frequency_plot_panel, "has_plot_content")
            and not self._frequency_plot_panel.has_plot_content()
        ):
            self._plot_workspace.set_active_domain("time")

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

        multi_group_fit_state = state.get("multi_group_fit_state")
        if (
            multi_group_fit_state
            and self._multi_group_fit_window is not None
            and hasattr(self._multi_group_fit_window, "restore_state")
        ):
            self._multi_group_fit_window.restore_state(multi_group_fit_state)

        fit_ui_state = state.get("fit_ui_state")
        if fit_ui_state:
            self._fit_panel.restore_ui_state(fit_ui_state)

        frequency_fit_state = state.get("frequency_fit_state")
        if isinstance(frequency_fit_state, dict) and hasattr(
            self._fit_panel, "restore_domain_state"
        ):
            self._fit_panel.restore_domain_state("frequency", frequency_fit_state)
            if self._plot_workspace.active_domain() == "frequency":
                if hasattr(self._fit_panel, "set_domain"):
                    self._fit_panel.set_domain("frequency")
                self._fit_panel.set_dataset(self._active_frequency_fit_dataset())
                self._set_frequency_fit_datasets_for_selection()

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
                    self._global_parameter_fit_window.restore_state(
                        global_parameter_fit_window_state
                    )
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
            raw_group_phase_states = fourier_state.get("group_phase_state_by_run", {})
            self._fourier_group_phase_state_by_run = {}
            if isinstance(raw_group_phase_states, dict):
                for run_number, run_state in raw_group_phase_states.items():
                    try:
                        parsed_run = int(run_number)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(run_state, dict):
                        self._fourier_group_phase_state_by_run[parsed_run] = dict(run_state)
            self._fourier_panel.restore_state(fourier_state)
            if current_dataset is not None:
                if int(current_dataset.run_number) not in self._fourier_group_phase_state_by_run:
                    self._fourier_group_phase_state_by_run[int(current_dataset.run_number)] = {
                        "group_enabled_table": dict(fourier_state.get("group_enabled_table", {})),
                        "group_phase_table": dict(fourier_state.get("group_phase_table", {})),
                        "group_auto_filled_ids": list(
                            fourier_state.get("group_auto_filled_ids", [])
                        ),
                    }
                self._sync_fourier_panel_for_dataset(current_dataset)

        maxent_state = state.get("maxent_state")
        if isinstance(maxent_state, dict):
            self._maxent_panel.restore_state(maxent_state)
        raw_maxent_states = state.get("maxent_state_by_run")
        self._maxent_panel_state_by_run = {}
        if isinstance(raw_maxent_states, dict):
            for run_number, run_state in raw_maxent_states.items():
                try:
                    parsed_run = int(run_number)
                except (TypeError, ValueError):
                    continue
                if isinstance(run_state, dict):
                    self._maxent_panel_state_by_run[parsed_run] = dict(run_state)
        if current_dataset is not None:
            self._sync_maxent_panel_for_dataset(current_dataset)

        # Open fit-related docks automatically when project contains saved
        # results/state for those panes.
        if _has_saved_fit_results(single_fit_state, global_fit_state):
            self._show_panel("fit")

        if _has_saved_fit_parameters_results(fit_parameters_state):
            self._show_panel("fit_parameters")

        n_loaded = len(loaded_run_numbers)
        self._log_panel.log(f"Project opened: {n_loaded} run(s) loaded from {project_path}")
        self.statusBar().showMessage(f"Opened project — {n_loaded} run(s) loaded")

    def _clear_all_state(self) -> None:
        """Reset every panel to its empty initial state."""
        self._current_dataset = None
        self._frequency_spectra_by_run = {}
        self._frequency_spectra_by_rep = {
            RepresentationType.FREQ_FFT: self._frequency_spectra_by_run,
            RepresentationType.FREQ_MAXENT: {},
        }
        self._maxent_state_by_run = {}
        self._maxent_panel_state_by_run = {}
        self._fourier_group_phase_state_by_run = {}
        self._data_browser.clear()
        self._plot_workspace.clear()
        if hasattr(self._frequency_plot_panel, "_frequency_x_unit_combo"):
            self._frequency_plot_panel._frequency_x_unit_combo.setCurrentIndex(0)
        if hasattr(self._frequency_plot_panel, "set_frequency_axis_relative_to_reference"):
            self._frequency_plot_panel.set_frequency_axis_relative_to_reference(False)
        if hasattr(self._fit_panel, "clear"):
            self._fit_panel.clear()
        else:
            self._fit_panel.set_dataset(None)
            self._fit_panel.set_datasets([])
        self._fit_parameters_panel.clear()
        if self._global_parameter_fit_window is not None:
            self._global_parameter_fit_window.close()
            self._global_parameter_fit_window = None
        self._update_global_parameter_fit_menu_style(False)
        self._sync_temperature_log_option_action()

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
        self._recent_menu.addAction("Clear Recent Projects", self._clear_recent_projects)

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

    def closeEvent(self, event) -> None:
        """Stop background work and save plot axis ranges before closing."""
        # The MaxEnt thread is parented to this window: letting the window be
        # destroyed while it runs aborts the process (QThread::~QThread calls
        # qFatal). Request cooperative cancellation and wait for the worker.
        if getattr(self, "_maxent_worker", None) is not None:
            self._maxent_worker.cancel()
        thread = getattr(self, "_maxent_thread", None)
        if thread is not None:
            thread.quit()
            if not thread.wait(10_000):
                # The engine checks cancellation per kernel chunk, so this is
                # unlikely — but if the worker is still inside numpy, unparent
                # the thread so a timed-out wait degrades to a leaked thread
                # instead of a qFatal abort when the window is destroyed.
                thread.setParent(None)
                thread.finished.connect(thread.deleteLater)
        if hasattr(self, "_plot_panel") and hasattr(self._plot_panel, "get_view_limits"):
            x_min, x_max, y_min, y_max = self._plot_panel.get_view_limits()
            self._settings.setValue("plot/time_x_min", float(x_min))
            self._settings.setValue("plot/time_x_max", float(x_max))
            self._settings.setValue("plot/time_y_min", float(y_min))
            self._settings.setValue("plot/time_y_max", float(y_max))
        if hasattr(self, "_frequency_plot_panel") and hasattr(
            self._frequency_plot_panel, "get_view_limits"
        ):
            x_min, x_max, y_min, y_max = self._frequency_plot_panel.get_view_limits()
            self._settings.setValue("plot/freq_x_min", float(x_min))
            self._settings.setValue("plot/freq_x_max", float(x_max))
            self._settings.setValue("plot/freq_y_min", float(y_min))
            self._settings.setValue("plot/freq_y_max", float(y_max))
        self._settings.sync()
        super().closeEvent(event)

    def _restore_plot_ranges_from_settings(self) -> None:
        """Restore saved x/y axis ranges from QSettings if available."""
        if self._settings.contains("plot/time_x_min") and hasattr(
            self._plot_panel, "set_view_limits"
        ):
            x_min = self._settings.value("plot/time_x_min", 0.0, float)
            x_max = self._settings.value("plot/time_x_max", 10.0, float)
            y_min = self._settings.value("plot/time_y_min", -30.0, float)
            y_max = self._settings.value("plot/time_y_max", 30.0, float)
            self._plot_panel.set_view_limits(x_min, x_max, y_min, y_max)
            if hasattr(self._plot_panel, "_limits_initialized"):
                self._plot_panel._limits_initialized = True
        if self._settings.contains("plot/freq_x_min") and hasattr(
            self._frequency_plot_panel, "set_view_limits"
        ):
            x_min = self._settings.value("plot/freq_x_min", 0.0, float)
            x_max = self._settings.value("plot/freq_x_max", 20.0, float)
            y_min = self._settings.value("plot/freq_y_min", 0.0, float)
            y_max = self._settings.value("plot/freq_y_max", 10.0, float)
            self._frequency_plot_panel.set_view_limits(x_min, x_max, y_min, y_max)
            if hasattr(self._frequency_plot_panel, "_limits_initialized"):
                self._frequency_plot_panel._limits_initialized = True


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
