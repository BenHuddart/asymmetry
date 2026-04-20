"""Non-modal guided fit wizard for ordered global-fit dataset series."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalFitWizardCandidatePortfolio,
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    build_global_fit_wizard_screening_recommendation,
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio,
    build_global_fit_wizard_candidate_portfolio,
    build_global_fit_wizard_recommendation,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
    is_rate_like_parameter,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.gui.panels.log_panel import LogPanel


_DEFAULT_PHASE_ONE_SINGLE_FIT_HELPER = (
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio
)
_DEFAULT_SCREENING_BUILDER = build_global_fit_wizard_screening_recommendation
_DEFAULT_GLOBAL_FIT_BUILDER = build_global_fit_wizard_recommendation


class AnalysisLogWindow(QMainWindow):
    """Read-only log window for long-running wizard analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Fit Wizard Log")
        self.resize(720, 420)
        self._log_panel = LogPanel(self)
        self.setCentralWidget(self._log_panel)

    def clear(self) -> None:
        self._log_panel.clear()

    def log(self, message: str) -> None:
        self._log_panel.log(message)

    def to_plain_text(self) -> str:
        return self._log_panel.to_plain_text()


class GlobalFitWizardWorker(QObject):
    """Run global-fit wizard analysis off the UI thread."""

    finished = Signal(int, str, object)
    error = Signal(int, str, str)
    progress = Signal(int, str)
    single_fit_precomputed = Signal(int, object)

    def __init__(
        self,
        mode: str,
        request_id: int,
        datasets: list[MuonDataset],
        current_model: CompositeModel | None,
        current_parameter_types: dict[str, str],
        current_values: dict[str, float],
        parameter_bounds: dict[str, tuple[float, float]],
        existing_single_fit_recommendations_by_run: dict[int, object] | None,
        metric: SelectionMetric,
        selected_template_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._mode = str(mode)
        self._request_id = request_id
        self._datasets = datasets
        self._current_model = current_model
        self._current_parameter_types = current_parameter_types
        self._current_values = current_values
        self._parameter_bounds = parameter_bounds
        self._existing_single_fit_recommendations_by_run = dict(
            existing_single_fit_recommendations_by_run or {}
        )
        self._metric = metric
        self._selected_template_keys = tuple(
            key for key in selected_template_keys if isinstance(key, str)
        )

    def run(self) -> None:
        try:
            single_fit_recommendations_before_analysis = dict(
                self._existing_single_fit_recommendations_by_run
            )
            screening_builder_is_custom = (
                build_global_fit_wizard_screening_recommendation
                is not _DEFAULT_SCREENING_BUILDER
            )
            optimization_builder_is_custom = (
                build_global_fit_wizard_recommendation is not _DEFAULT_GLOBAL_FIT_BUILDER
            )
            skip_implicit_phase_one = (
                ((self._mode == "screening" and screening_builder_is_custom)
                or (self._mode == "optimize" and optimization_builder_is_custom))
                and build_or_complete_single_fit_wizard_recommendations_for_global_portfolio
                is _DEFAULT_PHASE_ONE_SINGLE_FIT_HELPER
                and not self._existing_single_fit_recommendations_by_run
            )
            if skip_implicit_phase_one:
                single_fit_recommendations_by_run = dict(
                    self._existing_single_fit_recommendations_by_run
                )
            else:
                _portfolio, single_fit_recommendations_by_run, _generated_runs = (
                    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
                        self._datasets,
                        current_model=self._current_model,
                        existing_recommendations_by_run=self._existing_single_fit_recommendations_by_run,
                        progress_callback=lambda message: self.progress.emit(
                            self._request_id,
                            message,
                        ),
                    )
                )
            progress_callback = lambda message: self.progress.emit(
                self._request_id,
                message,
            )
            if self._mode == "screening":
                recommendation = build_global_fit_wizard_screening_recommendation(
                    self._datasets,
                    current_model=self._current_model,
                    current_parameter_types=self._current_parameter_types,
                    current_values=self._current_values,
                    parameter_bounds=self._parameter_bounds,
                    single_fit_recommendations_by_run=single_fit_recommendations_by_run,
                    metric=self._metric,
                    progress_callback=progress_callback,
                )
            else:
                recommendation = build_global_fit_wizard_recommendation(
                    self._datasets,
                    current_model=self._current_model,
                    current_parameter_types=self._current_parameter_types,
                    current_values=self._current_values,
                    parameter_bounds=self._parameter_bounds,
                    single_fit_recommendations_by_run=single_fit_recommendations_by_run,
                    metric=self._metric,
                    progress_callback=progress_callback,
                    selected_template_keys=self._selected_template_keys,
                )
            updated_single_fit_recommendations = {
                int(run_number): recommendation
                for run_number, recommendation in single_fit_recommendations_by_run.items()
                if single_fit_recommendations_before_analysis.get(int(run_number))
                is not recommendation
            }
            if updated_single_fit_recommendations:
                self.single_fit_precomputed.emit(
                    self._request_id,
                    updated_single_fit_recommendations,
                )
        except Exception as exc:
            self.error.emit(self._request_id, self._mode, str(exc))
            return
        self.finished.emit(self._request_id, self._mode, recommendation)


class GlobalFitWizardParameterSetupDialog(QDialog):
    """Collect parameter-role expectations and bounds for the shortlisted families."""

    def __init__(
        self,
        portfolio: GlobalFitWizardCandidatePortfolio,
        *,
        current_parameter_types: dict[str, str],
        current_parameter_bounds: dict[str, tuple[float, float]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Fit Wizard Parameter Setup")
        self.resize(860, 520)

        self._portfolio = portfolio
        self._configuration: dict[str, object] | None = None

        root = QVBoxLayout(self)
        intro = QLabel(
            "The wizard will explore the candidate families below. Review the combined "
            "parameter list and set your expected Global/Local behaviour and bounds before "
            "the expensive search starts."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        family_titles = ", ".join(template.title for template in portfolio.templates)
        family_label = QLabel(f"Candidate families: {family_titles}")
        family_label.setWordWrap(True)
        root.addWidget(family_label)

        defaults_label = QLabel(
            "Defaults: amplitudes start as Global with positive bounds, rate-like parameters "
            "start as Local with positive bounds, and background terms stay Global unless you "
            "change them."
        )
        defaults_label.setWordWrap(True)
        root.addWidget(defaults_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Parameter", "Expected Role", "Bounds", "Used By"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        root.addWidget(self._table)

        self._parameter_names, self._usage_by_name = _portfolio_parameter_usage(portfolio.templates)
        self._table.setRowCount(len(self._parameter_names))
        for row, name in enumerate(self._parameter_names):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            role_combo = QComboBox()
            role_combo.addItems(["Global", "Local", "Fixed"])
            role_combo.setCurrentText(
                _default_parameter_role(name, current_parameter_types=current_parameter_types)
            )
            self._table.setCellWidget(row, 1, role_combo)

            bounds_item = QTableWidgetItem(
                _format_bounds_text(
                    _default_parameter_bounds(name, current_parameter_bounds=current_parameter_bounds)
                )
            )
            self._table.setItem(row, 2, bounds_item)

            usage_titles = self._usage_by_name[name]
            usage_item = QTableWidgetItem(
                ", ".join(usage_titles[:3]) + (", ..." if len(usage_titles) > 3 else "")
            )
            usage_item.setToolTip("\n".join(usage_titles))
            usage_item.setFlags(usage_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 3, usage_item)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root.addWidget(button_box)

    def configuration(self) -> dict[str, object] | None:
        return copy.deepcopy(self._configuration)

    def accept(self) -> None:  # type: ignore[override]
        types: dict[str, str] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for row, name in enumerate(self._parameter_names):
            role_combo = self._table.cellWidget(row, 1)
            role = role_combo.currentText() if isinstance(role_combo, QComboBox) else "Global"
            bounds_item = self._table.item(row, 2)
            try:
                min_val, max_val = _parse_bounds_text(bounds_item.text() if bounds_item else "-inf, inf")
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Bounds", f"{name}: {exc}")
                return
            types[name] = role
            bounds[name] = (min_val, max_val)
        self._configuration = {"types": types, "bounds": bounds}
        super().accept()


class GlobalFitWizardWindow(QMainWindow):
    """Present a guided workflow for global-fit model recommendation."""

    apply_assessment_requested = Signal(object, object)
    analysis_cached = Signal(object, str, object)
    parameter_setup_applied = Signal(object)
    single_fit_recommendations_generated = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Fit Wizard")
        self.resize(1260, 900)

        self._datasets: list[MuonDataset] = []
        self._current_model: CompositeModel | None = None
        self._current_parameter_types: dict[str, str] = {}
        self._current_values: dict[str, float] = {}
        self._parameter_bounds: dict[str, tuple[float, float]] = {}
        self._recommendation: GlobalFitWizardRecommendation | None = None
        self._selected_key: str | None = None
        self._screening_selected_keys: set[str] = set()
        self._running_template_keys: set[str] = set()
        self._analysis_request_id = 0
        self._analysis_mode = "screening"
        self._analysis_in_progress = False
        self._analysis_thread: QThread | None = None
        self._analysis_worker: GlobalFitWizardWorker | None = None
        self._log_window: AnalysisLogWindow | None = None
        self._cached_log_text = ""
        self._cached_signature: dict[str, object] | None = None
        self._single_fit_recommendations_by_run: dict[int, object] = {}

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self._heading_label = QLabel("Global Fit Wizard")
        heading_font = QFont(self._heading_label.font())
        heading_font.setPointSize(max(heading_font.pointSize() + 4, 14))
        heading_font.setBold(True)
        self._heading_label.setFont(heading_font)
        layout.addWidget(self._heading_label)

        self._status_label = QLabel(
            "Open the global fit wizard on a field or temperature series "
            "to compare common model families and recommended "
            "Global/Local parameter roles."
        )
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        controls_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Build Screening Table")
        self._refresh_btn.clicked.connect(self._start_analysis)
        controls_row.addWidget(self._refresh_btn)
        self._optimize_btn = QPushButton("Optimize Selected")
        self._optimize_btn.clicked.connect(self._start_selected_optimisation)
        controls_row.addWidget(self._optimize_btn)
        controls_row.addWidget(QLabel("Ranking Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        controls_row.addWidget(self._metric_combo)
        metric_info_btn = QPushButton("Metric Info")
        metric_info_btn.clicked.connect(self._show_metric_info)
        controls_row.addWidget(metric_info_btn)
        warning_info_btn = QPushButton("Warning Info")
        warning_info_btn.clicked.connect(self._show_warning_info)
        controls_row.addWidget(warning_info_btn)
        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        controls_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(220)
        controls_row.addWidget(self._progress_bar)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._overview_tab = QWidget()
        self._portfolio_tab = QWidget()
        self._compare_tab = QWidget()
        self._optimized_tab = QWidget()
        self._roles_tab = QWidget()
        self._apply_tab = QWidget()
        self._tabs.addTab(self._overview_tab, "1. Series Overview")
        self._tabs.addTab(self._portfolio_tab, "2. Candidate Portfolio")
        self._tabs.addTab(self._compare_tab, "3. Single-Fit Screening")
        self._tabs.addTab(self._optimized_tab, "4. Global Optimized Fits")
        self._tabs.addTab(self._roles_tab, "5. Parameter Sharing")
        self._tabs.addTab(self._apply_tab, "6. Apply")

        self._build_overview_tab()
        self._build_portfolio_tab()
        self._build_compare_tab()
        self._build_optimized_tab()
        self._build_roles_tab()
        self._build_apply_tab()

        self._refresh_btn.setEnabled(False)
        self._metric_combo.setEnabled(False)
        self._optimize_btn.setEnabled(False)

    def set_analysis_context(
        self,
        datasets: list[MuonDataset],
        *,
        current_model: CompositeModel | None = None,
        current_parameter_types: dict[str, str] | None = None,
        current_values: dict[str, float] | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        existing_single_fit_recommendations_by_run: dict[int, object] | None = None,
    ) -> None:
        """Prepare the window for a new ordered dataset series."""
        self._datasets = list(datasets)
        self._current_model = current_model
        self._current_parameter_types = dict(current_parameter_types or {})
        self._current_values = dict(current_values or {})
        self._parameter_bounds = dict(parameter_bounds or {})
        self._single_fit_recommendations_by_run = dict(existing_single_fit_recommendations_by_run or {})
        self._recommendation = None
        self._selected_key = None
        self._screening_selected_keys = set()
        self._running_template_keys = set()
        self._cached_log_text = ""
        self._cached_signature = None
        self._analysis_request_id += 1
        run_labels = ", ".join(dataset.run_label for dataset in self._datasets[:4])
        if len(self._datasets) > 4:
            run_labels += ", …"
        self._heading_label.setText(f"Global Fit Wizard — {len(self._datasets)} datasets")
        self._status_label.setText(
            f"Ready to analyze the selected series ({run_labels}). "
            "Review the candidate portfolio, then build the screening table before choosing which candidates to optimize globally."
        )
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(SelectionMetric.AICC.value)
        self._metric_combo.blockSignals(False)
        self._set_empty_state()
        self._set_busy(False)

    def _build_overview_tab(self) -> None:
        layout = QVBoxLayout(self._overview_tab)
        self._overview_banner = QLabel("")
        self._overview_banner.setWordWrap(True)
        layout.addWidget(self._overview_banner)
        self._overview_table = QTableWidget(0, 6)
        self._overview_table.setHorizontalHeaderLabels(
            ["Run", "Field (G)", "Temperature (K)", "Osc.", "KT-like", "Multi-rate"]
        )
        self._overview_table.horizontalHeader().setStretchLastSection(False)
        self._overview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._overview_table)

    def _build_portfolio_tab(self) -> None:
        layout = QVBoxLayout(self._portfolio_tab)
        self._portfolio_banner = QLabel("")
        self._portfolio_banner.setWordWrap(True)
        layout.addWidget(self._portfolio_banner)
        self._portfolio_table = QTableWidget(0, 4)
        self._portfolio_table.setHorizontalHeaderLabels(
            ["Candidate", "Category", "Parameters", "Rationale"]
        )
        self._portfolio_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._portfolio_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._portfolio_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self._portfolio_table)

    def _build_compare_tab(self) -> None:
        layout = QVBoxLayout(self._compare_tab)
        self._compare_banner = QLabel("")
        self._compare_banner.setWordWrap(True)
        layout.addWidget(self._compare_banner)
        self._compare_table = QTableWidget(0, 9)
        self._compare_table.setHorizontalHeaderLabels(
            [
                "Candidate",
                "Screening Score",
                "AIC",
                "AICc",
                "BIC",
                "Status",
                "Global Fit",
                "Params",
                "Local",
            ]
        )
        self._compare_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._compare_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._compare_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._compare_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        self._compare_table.itemSelectionChanged.connect(self._on_compare_selection_changed)
        layout.addWidget(self._compare_table)

        self._compare_warning_text = QTextEdit()
        self._compare_warning_text.setReadOnly(True)
        self._compare_warning_text.setMinimumHeight(150)
        layout.addWidget(self._compare_warning_text)

    def _build_optimized_tab(self) -> None:
        layout = QVBoxLayout(self._optimized_tab)
        self._optimized_banner = QLabel("")
        self._optimized_banner.setWordWrap(True)
        layout.addWidget(self._optimized_banner)
        self._optimized_table = QTableWidget(0, 8)
        self._optimized_table.setHorizontalHeaderLabels(
            ["Candidate", "Score", "AIC", "AICc", "BIC", "Gate", "Global", "Local"]
        )
        self._optimized_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._optimized_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._optimized_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._optimized_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self._optimized_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self._optimized_table.itemSelectionChanged.connect(self._on_optimized_selection_changed)
        layout.addWidget(self._optimized_table)

    def _build_roles_tab(self) -> None:
        layout = QVBoxLayout(self._roles_tab)
        self._roles_banner = QLabel("")
        self._roles_banner.setWordWrap(True)
        layout.addWidget(self._roles_banner)
        self._roles_table = QTableWidget(0, 7)
        self._roles_table.setHorizontalHeaderLabels(
            ["Parameter", "Role", "Global Score", "Local Score", "Δ", "TV", "Roughness"]
        )
        self._roles_table.horizontalHeader().setStretchLastSection(False)
        self._roles_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._roles_table)
        self._roles_rationale_text = QTextEdit()
        self._roles_rationale_text.setReadOnly(True)
        self._roles_rationale_text.setMinimumHeight(120)
        layout.addWidget(self._roles_rationale_text)

    def _build_apply_tab(self) -> None:
        layout = QVBoxLayout(self._apply_tab)
        self._apply_banner = QLabel("")
        self._apply_banner.setWordWrap(True)
        layout.addWidget(self._apply_banner)
        self._apply_selection_label = QLabel("")
        self._apply_selection_label.setWordWrap(True)
        layout.addWidget(self._apply_selection_label)
        self._apply_text = QTextEdit()
        self._apply_text.setReadOnly(True)
        layout.addWidget(self._apply_text)
        button_row = QHBoxLayout()
        self._apply_recommended_btn = QPushButton("Apply Recommended Fit")
        self._apply_recommended_btn.clicked.connect(self._apply_recommended_fit)
        button_row.addWidget(self._apply_recommended_btn)
        self._apply_selected_btn = QPushButton("Apply Selected Fit")
        self._apply_selected_btn.clicked.connect(self._apply_selected_fit)
        button_row.addWidget(self._apply_selected_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

    def _set_busy(self, busy: bool) -> None:
        self._analysis_in_progress = busy
        self._progress_label.setVisible(busy)
        self._progress_bar.setVisible(busy)
        self._progress_label.setText(
            "Working..." if busy else ""
        )
        self._refresh_btn.setEnabled(bool(self._datasets) and not busy)
        self._metric_combo.setEnabled(self._recommendation is not None and not busy)
        has_screening_selection = bool(self._screening_selected_keys)
        self._optimize_btn.setEnabled(
            self._recommendation is not None and has_screening_selection and not busy
        )

    def _set_empty_state(self) -> None:
        self._overview_banner.setText("")
        self._portfolio_banner.setText("")
        self._compare_banner.setText("")
        self._optimized_banner.setText("")
        self._roles_banner.setText("")
        self._apply_banner.setText("")
        self._apply_selection_label.setText("")
        self._compare_warning_text.setPlainText("")
        self._roles_rationale_text.setPlainText("")
        self._apply_text.setPlainText("")
        for table in (
            self._overview_table,
            self._portfolio_table,
            self._compare_table,
            self._optimized_table,
            self._roles_table,
        ):
            table.setRowCount(0)
        self._screening_selected_keys = set()
        self._running_template_keys = set()
        self._apply_recommended_btn.setEnabled(False)
        self._apply_selected_btn.setEnabled(False)
        self._optimize_btn.setEnabled(False)

    def _start_analysis(self) -> None:
        if len(self._datasets) < 2:
            self._status_label.setText("Global fit wizard requires at least two datasets.")
            return
        if self._analysis_in_progress:
            return
        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait()
            self._cleanup_analysis_thread()

        try:
            portfolio = build_global_fit_wizard_candidate_portfolio(
                self._datasets,
                current_model=self._current_model,
            )
        except Exception as exc:
            self._status_label.setText(f"Global fit wizard setup failed: {exc}")
            return

        setup_config: dict[str, object] | None = None
        if portfolio.mixed_axes_warning is None and portfolio.templates:
            setup_config = self._prompt_parameter_setup(portfolio)
            if setup_config is None:
                self._status_label.setText(
                    "Analysis cancelled before the parameter setup was applied."
                )
                return
            self._apply_parameter_setup(setup_config)

        signature = self._analysis_signature()
        if self._cached_signature == signature and self._recommendation is not None:
            self._status_label.setText(self._recommendation.summary)
            self._populate_from_recommendation()
            return

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._cached_signature = copy.deepcopy(signature)
        self._set_busy(True)
        self._set_empty_state()
        self._show_log_window()
        self._status_label.setText(
            "Building the single-fit screening table in the background. "
            "The main window stays responsive while the shared candidate portfolio is screened."
        )
        self._append_progress_log(
            request_id,
            f"Starting screening for {len(self._datasets)} datasets.",
        )

        self._launch_worker(
            GlobalFitWizardWorker(
                "screening",
                request_id,
                self._datasets,
                self._current_model,
                self._current_parameter_types,
                self._current_values,
                self._parameter_bounds,
                self._single_fit_recommendations_by_run,
                SelectionMetric.from_value(self._metric_combo.currentText()),
            )
        )

    def _start_selected_optimisation(self) -> None:
        if self._recommendation is None or not self._screening_selected_keys:
            return
        if self._analysis_in_progress:
            return
        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._analysis_mode = "optimize"
        self._running_template_keys = set(self._screening_selected_keys)
        self._set_busy(True)
        selected_titles = [
            assessment.template.title
            for assessment in self._recommendation.assessments
            if assessment.template.key in self._running_template_keys
        ]
        self._show_log_window()
        self._status_label.setText(
            "Running coupled global optimisation for the selected candidates. Progress is streamed to the log window."
        )
        self._append_progress_log(
            request_id,
            "Starting coupled global optimisation for: " + ", ".join(selected_titles) + ".",
        )
        self._populate_compare_table()
        self._launch_worker(
            GlobalFitWizardWorker(
                "optimize",
                request_id,
                self._datasets,
                self._current_model,
                self._current_parameter_types,
                self._current_values,
                self._parameter_bounds,
                self._single_fit_recommendations_by_run,
                SelectionMetric.from_value(self._metric_combo.currentText()),
                tuple(sorted(self._screening_selected_keys)),
            )
        )

    def _launch_worker(self, worker: GlobalFitWizardWorker) -> None:
        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait()
            self._cleanup_analysis_thread()
        self._analysis_mode = worker._mode
        if (
            (worker._mode == "screening" and build_global_fit_wizard_screening_recommendation is not _DEFAULT_SCREENING_BUILDER)
            or (worker._mode == "optimize" and build_global_fit_wizard_recommendation is not _DEFAULT_GLOBAL_FIT_BUILDER)
        ):
            self._analysis_worker = worker
            self._analysis_worker.finished.connect(self._on_analysis_finished)
            self._analysis_worker.error.connect(self._on_analysis_error)
            self._analysis_worker.progress.connect(self._append_progress_log)
            self._analysis_worker.single_fit_precomputed.connect(self._on_single_fit_precomputed)
            self._analysis_worker.run()
            self._analysis_worker = None
            return
        self._analysis_thread = QThread(self)
        self._analysis_worker = worker
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.progress.connect(self._append_progress_log)
        self._analysis_worker.single_fit_precomputed.connect(self._on_single_fit_precomputed)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_thread.quit)
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.error.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._cleanup_analysis_thread)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._analysis_thread.start()

    def _on_analysis_finished(self, request_id: int, mode: str, recommendation: object) -> None:
        thread = self._analysis_thread
        if thread is not None:
            thread.quit()
            thread.wait()
            self._cleanup_analysis_thread()
        if request_id != self._analysis_request_id:
            self._set_busy(False)
            return
        if not isinstance(recommendation, GlobalFitWizardRecommendation):
            self._set_busy(False)
            self._status_label.setText("Global fit wizard analysis returned an unexpected result.")
            return

        if mode == "optimize" and self._recommendation is not None:
            self._recommendation = merge_global_fit_wizard_recommendations(
                self._recommendation,
                recommendation,
            )
        else:
            self._recommendation = recommendation
        self._running_template_keys = set()
        self._selected_key = self._recommended_or_first_optimized_key(self._recommendation)
        self._status_label.setText(self._recommendation.summary)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(self._recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._append_progress_log(request_id, self._recommendation.summary)
        self.analysis_cached.emit(
            self._recommendation,
            self.current_log_text(),
            copy.deepcopy(self._cached_signature) if self._cached_signature is not None else None,
        )
        self._set_busy(False)
        self._populate_from_recommendation()

    def _on_analysis_error(self, request_id: int, mode: str, message: str) -> None:
        thread = self._analysis_thread
        if thread is not None:
            thread.quit()
            thread.wait()
            self._cleanup_analysis_thread()
        if request_id != self._analysis_request_id:
            self._set_busy(False)
            return
        self._running_template_keys = set()
        if mode == "screening":
            self._recommendation = None
        self._set_busy(False)
        self._status_label.setText(f"Global fit wizard analysis failed: {message}")
        self._append_progress_log(request_id, f"Analysis failed: {message}")
        if self._recommendation is None:
            self._set_empty_state()
        else:
            self._populate_from_recommendation()

    def _on_single_fit_precomputed(self, request_id: int, payload: object) -> None:
        if request_id != self._analysis_request_id or not isinstance(payload, dict):
            return
        typed_payload = {
            int(run_number): recommendation
            for run_number, recommendation in payload.items()
        }
        self._single_fit_recommendations_by_run.update(typed_payload)
        self.single_fit_recommendations_generated.emit(typed_payload)

    def _cleanup_analysis_thread(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None

    def _show_log_window(self) -> None:
        if self._log_window is None:
            self._log_window = AnalysisLogWindow(self)
        self._log_window.clear()
        self._log_window.show()
        self._log_window.raise_()
        self._log_window.activateWindow()

    def _append_progress_log(self, request_id: int, message: str) -> None:
        if request_id != self._analysis_request_id:
            return
        if self._log_window is None:
            self._cached_log_text = "\n".join(filter(None, [self._cached_log_text, message]))
            return
        self._log_window.log(message)
        self._cached_log_text = self._log_window.to_plain_text()

    def set_cached_recommendation(
        self,
        recommendation: GlobalFitWizardRecommendation,
        *,
        signature: dict[str, object] | None = None,
        log_text: str = "",
        status_text: str | None = None,
    ) -> None:
        """Populate the window from an already-computed recommendation."""
        self._recommendation = recommendation
        self._cached_signature = copy.deepcopy(signature) if isinstance(signature, dict) else None
        self._selected_key = self._recommended_or_first_optimized_key(recommendation)
        self._cached_log_text = str(log_text or "")
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._status_label.setText(status_text or recommendation.summary)
        self._set_busy(False)
        self._populate_from_recommendation()

    def _analysis_signature(self) -> dict[str, object]:
        return {
            "run_numbers": [int(dataset.run_number) for dataset in self._datasets],
            "model": self._current_model.to_dict() if self._current_model is not None else None,
            "types": {str(key): str(value) for key, value in self._current_parameter_types.items()},
            "values": {str(key): float(value) for key, value in self._current_values.items()},
            "bounds": {
                str(key): [float(bounds[0]), float(bounds[1])]
                for key, bounds in self._parameter_bounds.items()
            },
        }

    def _prompt_parameter_setup(
        self,
        portfolio: GlobalFitWizardCandidatePortfolio,
    ) -> dict[str, object] | None:
        dialog = GlobalFitWizardParameterSetupDialog(
            portfolio,
            current_parameter_types=self._current_parameter_types,
            current_parameter_bounds=self._parameter_bounds,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.configuration()

    def _apply_parameter_setup(self, config: dict[str, object]) -> None:
        types = config.get("types")
        bounds = config.get("bounds")
        if not isinstance(types, dict) or not isinstance(bounds, dict):
            return

        typed_types = {
            str(name): str(role)
            for name, role in types.items()
            if isinstance(name, str)
        }
        typed_bounds: dict[str, tuple[float, float]] = {}
        for name, raw_bounds in bounds.items():
            if not isinstance(name, str) or not isinstance(raw_bounds, tuple | list) or len(raw_bounds) != 2:
                continue
            try:
                typed_bounds[name] = (float(raw_bounds[0]), float(raw_bounds[1]))
            except (TypeError, ValueError):
                continue

        self._current_parameter_types.update(typed_types)
        self._parameter_bounds.update(typed_bounds)
        for name, (min_val, max_val) in typed_bounds.items():
            if name not in self._current_values:
                continue
            self._current_values[name] = float(np.clip(self._current_values[name], min_val, max_val))

        self.parameter_setup_applied.emit(
            {
                "types": copy.deepcopy(typed_types),
                "bounds": copy.deepcopy(typed_bounds),
            }
        )

    def current_recommendation(self) -> GlobalFitWizardRecommendation | None:
        return self._recommendation

    def current_log_text(self) -> str:
        if self._log_window is not None:
            self._cached_log_text = self._log_window.to_plain_text()
        return self._cached_log_text

    def _populate_from_recommendation(self) -> None:
        if self._recommendation is None:
            self._set_empty_state()
            return

        recommendation = self._recommendation
        if recommendation.mixed_axes_warning:
            self._overview_banner.setText(recommendation.mixed_axes_warning)
            self._portfolio_banner.setText(recommendation.mixed_axes_warning)
            self._compare_banner.setText(recommendation.mixed_axes_warning)
            self._optimized_banner.setText(recommendation.mixed_axes_warning)
            self._apply_banner.setText(recommendation.mixed_axes_warning)
        else:
            self._overview_banner.setText(
                f"Series ordered by {recommendation.series_axis_label}. "
                "The wizard compares one common fit function across "
                f"{len(recommendation.dataset_order)} datasets."
            )
            self._portfolio_banner.setText(
                "Candidate families are listed here for reference so you can review the model expression, "
                "category, and rationale before moving to screening."
            )
            self._compare_banner.setText(
                "Single-fit screening ranks candidates using independent per-dataset fits only. "
                "These rows have not yet been optimized for coupled global fitting. Select one or more rows "
                "to launch coupled optimisation and follow progress in the log window."
            )
            self._optimized_banner.setText(
                recommendation.summary
                + " Successful runs appear one row per converged global/local assignment."
            )
            self._apply_banner.setText(recommendation.summary)
        self._roles_banner.setText(
            "Role recommendations use penalized score differences plus continuity diagnostics. "
            "Fixed parameters are left untouched."
        )

        self._populate_overview_table()
        self._populate_portfolio_table()
        self._populate_compare_table()
        self._populate_optimized_table()
        self._sync_selected_assessment()
        self._update_roles_table()
        self._update_apply_page()

    def _populate_overview_table(self) -> None:
        if self._recommendation is None:
            return
        self._overview_table.setRowCount(len(self._datasets))
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        for row, run_number in enumerate(self._recommendation.dataset_order):
            dataset = by_run.get(int(run_number))
            fingerprint = self._recommendation.fingerprints_by_run[int(run_number)]
            run_label = dataset.run_label if dataset else str(run_number)
            field_text = (
                f"{float((dataset.metadata if dataset else {}).get('field', 0.0)):.6g}"
                if dataset
                else "0"
            )
            temperature_text = (
                f"{float((dataset.metadata if dataset else {}).get('temperature', 0.0)):.6g}"
                if dataset
                else "0"
            )
            self._overview_table.setItem(row, 0, QTableWidgetItem(run_label))
            self._overview_table.setItem(row, 1, QTableWidgetItem(field_text))
            self._overview_table.setItem(row, 2, QTableWidgetItem(temperature_text))
            self._overview_table.setItem(
                row,
                3,
                QTableWidgetItem("Yes" if fingerprint.oscillatory_hint else "No"),
            )
            self._overview_table.setItem(
                row,
                4,
                QTableWidgetItem("Yes" if fingerprint.kt_like_hint else "No"),
            )
            self._overview_table.setItem(
                row,
                5,
                QTableWidgetItem("Yes" if fingerprint.multi_rate_hint else "No"),
            )

    def _populate_portfolio_table(self) -> None:
        if self._recommendation is None:
            return
        recommended_assessment = self._recommendation.recommended_assessment
        self._portfolio_table.setRowCount(len(self._recommendation.templates))
        for row, template in enumerate(self._recommendation.templates):
            title_item = QTableWidgetItem(template.title)
            if recommended_assessment is not None and template.key == recommended_assessment.template.key:
                title_item.setFont(_bold_font(title_item.font()))
            self._portfolio_table.setItem(row, 0, title_item)
            self._portfolio_table.setItem(row, 1, QTableWidgetItem(template.category))
            self._portfolio_table.setItem(
                row,
                2,
                QTableWidgetItem(str(len(template.model.param_names))),
            )
            self._portfolio_table.setItem(row, 3, QTableWidgetItem(template.rationale))
        self._ensure_candidate_column_width(self._portfolio_table)

    def _populate_compare_table(self) -> None:
        if self._recommendation is None:
            return
        recommended_assessment = self._recommendation.recommended_assessment
        assessments = self._recommendation.sorted_prescreen_assessments()
        self._compare_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            title_item = QTableWidgetItem(assessment.template.title)
            title_item.setData(Qt.ItemDataRole.UserRole, assessment.template.key)
            if (
                recommended_assessment is not None
                and assessment.template.key == recommended_assessment.template.key
            ):
                title_item.setFont(_bold_font(title_item.font()))
            self._compare_table.setItem(row, 0, title_item)
            self._compare_table.setItem(
                row,
                1,
                _numeric_item(assessment.metric_value(self._recommendation.metric)),
            )
            self._compare_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._compare_table.setItem(
                row,
                3,
                (
                    _numeric_item(assessment.aicc)
                    if assessment.aicc is not None
                    else QTableWidgetItem("AIC")
                ),
            )
            self._compare_table.setItem(row, 4, _numeric_item(assessment.bic))
            self._compare_table.setItem(
                row,
                5,
                QTableWidgetItem(self._recommendation.optimization_status_for_key(assessment.template.key)),
            )
            self._compare_table.setItem(
                row,
                6,
                QTableWidgetItem(
                    "Running"
                    if assessment.template.key in self._running_template_keys
                    else ("Yes" if not assessment.prescreen_only else "No")
                ),
            )
            self._compare_table.setItem(
                row,
                7,
                QTableWidgetItem(str(assessment.parameter_count)),
            )
            self._compare_table.setItem(
                row,
                8,
                QTableWidgetItem(str(len(assessment.local_param_names))),
            )
        self._ensure_candidate_column_width(self._compare_table)
        self._restore_screening_selection()

    def _populate_optimized_table(self) -> None:
        if self._recommendation is None:
            return
        assessments = self._recommendation.sorted_optimized_assessments()
        self._optimized_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            title_item = QTableWidgetItem(assessment.template.title)
            title_item.setData(Qt.ItemDataRole.UserRole, assessment.selection_key)
            if assessment.selection_key == self._recommendation.recommended_key:
                title_item.setFont(_bold_font(title_item.font()))
            self._optimized_table.setItem(row, 0, title_item)
            self._optimized_table.setItem(row, 1, _numeric_item(assessment.metric_value(self._recommendation.metric)))
            self._optimized_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._optimized_table.setItem(
                row,
                3,
                _numeric_item(assessment.aicc) if assessment.aicc is not None else QTableWidgetItem("AIC"),
            )
            self._optimized_table.setItem(row, 4, _numeric_item(assessment.bic))
            self._optimized_table.setItem(
                row,
                5,
                QTableWidgetItem("Pass" if assessment.residual_gate_passed else "Warn"),
            )
            self._optimized_table.setItem(
                row,
                6,
                QTableWidgetItem(", ".join(assessment.global_param_names) or "None"),
            )
            self._optimized_table.setItem(
                row,
                7,
                QTableWidgetItem(", ".join(assessment.local_param_names) or "None"),
            )
        self._ensure_candidate_column_width(self._optimized_table)

    def _sync_selected_assessment(self) -> None:
        if self._recommendation is None:
            return
        target_key = self._selected_key or self._recommended_or_first_optimized_key(self._recommendation)
        self._selected_key = target_key
        self._select_row_for_key(self._optimized_table, target_key)
        self._update_compare_warning_text()

    def _selected_assessment(self) -> GlobalCandidateAssessment | None:
        if self._recommendation is None:
            return None
        assessment = self._recommendation.assessment_for_key(self._selected_key)
        if assessment is not None and not assessment.prescreen_only:
            return assessment
        return self._recommendation.recommended_assessment

    def _on_compare_selection_changed(self) -> None:
        selected_keys: set[str] = set()
        for index in self._compare_table.selectionModel().selectedRows():
            item = self._compare_table.item(index.row(), 0)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(key, str):
                selected_keys.add(key)
        self._screening_selected_keys = selected_keys
        self._set_busy(self._analysis_in_progress)
        self._update_compare_warning_text()

    def _on_optimized_selection_changed(self) -> None:
        selected_items = self._optimized_table.selectedItems()
        if not selected_items:
            return
        key = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(key, str):
            self._selected_key = key
        self._update_compare_warning_text()
        self._update_roles_table()
        self._update_apply_page()

    def _update_compare_warning_text(self) -> None:
        if self._recommendation is None:
            self._compare_warning_text.setPlainText("")
            return
        lines: list[str] = []
        selected_assessments = [
            assessment
            for assessment in self._recommendation.sorted_prescreen_assessments()
            if assessment.template.key in self._screening_selected_keys
        ]
        if not selected_assessments:
            lines.append(
                "Select one or more screening rows to queue them for coupled global optimisation."
            )
            if self._recommendation.recommended_assessment is not None:
                lines.append("")
                lines.append(
                    "The optimized-results tab will list each converged global/local assignment after global fitting completes."
                )
        else:
            lines.append(
                f"Selected for optimisation: {', '.join(assessment.template.title for assessment in selected_assessments)}"
            )
            primary = selected_assessments[0]
            lines.append("")
            lines.append(
                "These screening scores come from independent per-dataset fits and do not yet include "
                "coupled global parameter sharing."
            )
            lines.append(f"Status: {self._recommendation.optimization_status_for_key(primary.template.key)}")
            lines.append(f"AIC = {primary.aic:.3f}")
            lines.append(
                f"AICc = {primary.aicc:.3f}"
                if primary.aicc is not None
                else "AICc fell back to AIC for this candidate."
            )
            lines.append(f"BIC = {primary.bic:.3f}")
        self._compare_warning_text.setPlainText("\n".join(lines))

    def _update_roles_table(self) -> None:
        assessment = self._selected_assessment()
        if assessment is None:
            self._roles_table.setRowCount(0)
            self._roles_rationale_text.setPlainText(
                "Run a coupled global optimisation for at least one candidate to inspect parameter-sharing diagnostics."
            )
            return
        recommendations = list(assessment.parameter_recommendations)
        if not recommendations:
            self._roles_table.setRowCount(0)
            self._roles_rationale_text.setPlainText(
                "This table is derived directly from the exhaustive wavefront search results for the selected assignment."
            )
            return
        self._roles_table.setRowCount(len(recommendations))
        rationale_lines: list[str] = []
        for row, recommendation in enumerate(recommendations):
            self._roles_table.setItem(
                row,
                0,
                QTableWidgetItem(get_param_info(recommendation.name).unicode_label()),
            )
            self._roles_table.setItem(row, 1, QTableWidgetItem(recommendation.recommended_role))
            self._roles_table.setItem(row, 2, _numeric_item(recommendation.global_score))
            self._roles_table.setItem(row, 3, _numeric_item(recommendation.local_score))
            self._roles_table.setItem(row, 4, _numeric_item(recommendation.score_delta))
            self._roles_table.setItem(
                row,
                5,
                _numeric_item(recommendation.total_variation),
            )
            self._roles_table.setItem(row, 6, _numeric_item(recommendation.roughness))
            rationale_lines.append(f"{recommendation.name}: {recommendation.rationale}")
        self._roles_rationale_text.setPlainText("\n".join(rationale_lines))

    def _update_apply_page(self) -> None:
        assessment = self._selected_assessment()
        if self._recommendation is None:
            self._apply_selection_label.setText("")
            self._apply_text.setPlainText("")
            self._apply_recommended_btn.setEnabled(False)
            self._apply_selected_btn.setEnabled(False)
            return

        if assessment is None:
            self._apply_selection_label.setText(
                "No globally optimized candidate is selected yet."
            )
            self._apply_text.setPlainText(
                "Choose one or more rows from the screening table and run coupled global optimisation first."
            )
            self._apply_recommended_btn.setEnabled(False)
            self._apply_selected_btn.setEnabled(False)
            return

        recommended = self._recommendation.recommended_assessment
        if recommended is None:
            self._apply_selection_label.setText(
                f"Selected candidate: {assessment.template.title}. "
                "No automatic recommendation is available."
            )
        else:
            self._apply_selection_label.setText(
                f"Recommended: {recommended.template.title}. Selected: {assessment.template.title}."
            )

        lines = [
            f"Global parameters: {', '.join(assessment.global_param_names) or 'None'}",
            f"Local parameters: {', '.join(assessment.local_param_names) or 'None'}",
        ]
        if assessment.series_warnings:
            lines.append("")
            lines.append("Series warnings:")
            lines.extend(f"• {warning}" for warning in assessment.series_warnings)
        self._apply_text.setPlainText("\n".join(lines))

        self._apply_recommended_btn.setEnabled(recommended is not None)
        self._apply_selected_btn.setEnabled(assessment.is_successful)

    def _on_metric_changed(self, text: str) -> None:
        if self._recommendation is None:
            return
        selected_key = self._selected_key
        self._recommendation = rerank_global_fit_wizard_recommendation(
            self._recommendation,
            SelectionMetric.from_value(text),
        )
        self._selected_key = selected_key or self._recommendation.recommended_key
        self._status_label.setText(self._recommendation.summary)
        self._populate_from_recommendation()
        if isinstance(self._cached_signature, dict):
            self.analysis_cached.emit(
                self._recommendation,
                self.current_log_text(),
                copy.deepcopy(self._cached_signature),
            )

    def _apply_recommended_fit(self) -> None:
        if self._recommendation is None or self._recommendation.recommended_assessment is None:
            return
        self.apply_assessment_requested.emit(
            self._recommendation.recommended_assessment,
            self._recommendation,
        )
        self.statusBar().showMessage(
            "Applied recommended global fit: "
            f"{self._recommendation.recommended_assessment.template.title}"
        )

    def _apply_selected_fit(self) -> None:
        assessment = self._selected_assessment()
        if self._recommendation is None or assessment is None:
            return
        self.apply_assessment_requested.emit(assessment, self._recommendation)
        self.statusBar().showMessage(f"Applied selected global fit: {assessment.template.title}")

    def _show_metric_info(self) -> None:
        QMessageBox.information(
            self,
            "Global Fit Wizard Metrics",
            (
                "The screening table and optimized-results table both rerank the same candidates by the "
                "chosen information criterion.\n\n"
                "Screening rows are based on independent per-dataset fits only. Optimized rows rerun the "
                "candidate under coupled global parameter sharing before being compared.\n\n"
                "AICc is the default because it adds a small-sample correction when the total fitted point "
                "count is not large compared with the number of free parameters."
            ),
        )

    def _show_warning_info(self) -> None:
        QMessageBox.information(
            self,
            "Global Fit Wizard Warnings",
            (
                "The screening table intentionally does not claim that a candidate is good for global fitting. "
                "It only reports how promising the function looks when each dataset is fit independently.\n\n"
                "Warnings in the optimized-results tab combine per-run residual checks with ordered-series "
                "continuity diagnostics after the coupled global optimisation has run."
            ),
        )

    def _ensure_candidate_column_width(
        self,
        table: QTableWidget,
        *,
        minimum_width: int = 420,
    ) -> None:
        if table.columnCount() <= 0:
            return
        table.resizeColumnToContents(0)
        table.setColumnWidth(0, max(table.columnWidth(0), minimum_width))

    def _recommended_or_first_optimized_key(
        self,
        recommendation: GlobalFitWizardRecommendation | None,
    ) -> str | None:
        if recommendation is None:
            return None
        if recommendation.recommended_key is not None:
            return recommendation.recommended_key
        optimized = recommendation.sorted_optimized_assessments()
        if optimized:
            return optimized[0].selection_key
        return None

    def _select_row_for_key(self, table: QTableWidget, key: str | None) -> None:
        if key is None:
            return
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                table.selectRow(row)
                return

    def _restore_screening_selection(self) -> None:
        self._compare_table.blockSignals(True)
        self._compare_table.clearSelection()
        for row in range(self._compare_table.rowCount()):
            item = self._compare_table.item(row, 0)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(key, str) and key in self._screening_selected_keys:
                self._compare_table.selectRow(row)
        self._compare_table.blockSignals(False)

    def closeEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802
        if self._analysis_in_progress:
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)


def _numeric_item(value: float) -> QTableWidgetItem:
    item = QTableWidgetItem(f"{float(value):.3f}")
    item.setData(Qt.ItemDataRole.UserRole, float(value))
    return item


def _bold_font(font: QFont) -> QFont:
    updated = QFont(font)
    updated.setBold(True)
    return updated


def _portfolio_parameter_usage(
    templates: tuple[CandidateTemplate, ...],
) -> tuple[list[str], dict[str, list[str]]]:
    ordered_names: list[str] = []
    usage_by_name: dict[str, list[str]] = {}
    seen: set[str] = set()
    for template in templates:
        for name in template.model.param_names:
            usage_by_name.setdefault(name, [])
            if template.title not in usage_by_name[name]:
                usage_by_name[name].append(template.title)
            if name in seen:
                continue
            seen.add(name)
            ordered_names.append(name)
    return ordered_names, usage_by_name


def _default_parameter_role(
    name: str,
    *,
    current_parameter_types: dict[str, str],
) -> str:
    current = str(current_parameter_types.get(name, "")).strip()
    if current == "Fixed":
        return "Fixed"
    if is_background_parameter(name):
        return "Global"
    if is_amplitude_parameter(name):
        return "Global"
    if _is_positive_rate_parameter(name) or _is_phase_parameter(name):
        return "Local"
    if current in {"Global", "Local"}:
        return current
    return "Global"


def _default_parameter_bounds(
    name: str,
    *,
    current_parameter_bounds: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    if is_background_parameter(name):
        return -float("inf"), float("inf")
    if is_amplitude_parameter(name) or _is_positive_rate_parameter(name):
        return 0.0, float("inf")
    if name in current_parameter_bounds:
        return current_parameter_bounds[name]
    default_min = get_param_info(name).default_min
    return (float(default_min), float("inf")) if default_min is not None else (-float("inf"), float("inf"))


def _is_positive_rate_parameter(name: str) -> bool:
    lower_name = name.lower()
    if "phase" in lower_name:
        return False
    return is_rate_like_parameter(name)


def _is_phase_parameter(name: str) -> bool:
    return "phase" in name.lower()


def _format_bounds_text(bounds: tuple[float, float]) -> str:
    return f"{_format_bound_value(bounds[0])}, {_format_bound_value(bounds[1])}"


def _format_bound_value(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == -float("inf"):
        return "-inf"
    return f"{float(value):.6g}"


def _parse_bounds_text(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("bounds must be written as 'min, max'")
    min_val = _parse_bound_value(parts[0])
    max_val = _parse_bound_value(parts[1])
    if np.isfinite(min_val) and np.isfinite(max_val) and min_val > max_val:
        raise ValueError(f"invalid bounds: {min_val} > {max_val}")
    return min_val, max_val


def _parse_bound_value(text: str) -> float:
    lowered = text.strip().lower()
    if lowered == "-inf":
        return -float("inf")
    if lowered == "inf":
        return float("inf")
    try:
        value = float(lowered)
    except ValueError as exc:
        raise ValueError(f"could not parse bound '{text}'") from exc
    if not np.isfinite(value):
        raise ValueError(f"bound '{text}' must be finite or +/-inf")
    return value
