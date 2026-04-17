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
    build_global_fit_wizard_candidate_portfolio,
    build_global_fit_wizard_recommendation,
    rerank_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
    is_rate_like_parameter,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.gui.panels.log_panel import LogPanel


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

    finished = Signal(int, object)
    error = Signal(int, str)
    progress = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        datasets: list[MuonDataset],
        current_model: CompositeModel | None,
        current_parameter_types: dict[str, str],
        current_values: dict[str, float],
        parameter_bounds: dict[str, tuple[float, float]],
        metric: SelectionMetric,
        search_strategy: str,
    ) -> None:
        super().__init__()
        self._request_id = request_id
        self._datasets = datasets
        self._current_model = current_model
        self._current_parameter_types = current_parameter_types
        self._current_values = current_values
        self._parameter_bounds = parameter_bounds
        self._metric = metric
        self._search_strategy = str(search_strategy)

    def run(self) -> None:
        try:
            recommendation = build_global_fit_wizard_recommendation(
                self._datasets,
                current_model=self._current_model,
                current_parameter_types=self._current_parameter_types,
                current_values=self._current_values,
                parameter_bounds=self._parameter_bounds,
                metric=self._metric,
                search_strategy=self._search_strategy,
                progress_callback=lambda message: self.progress.emit(
                    self._request_id,
                    message,
                ),
            )
        except Exception as exc:
            self.error.emit(self._request_id, str(exc))
            return
        self.finished.emit(self._request_id, recommendation)


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
        self._analysis_request_id = 0
        self._analysis_in_progress = False
        self._analysis_thread: QThread | None = None
        self._analysis_worker: GlobalFitWizardWorker | None = None
        self._log_window: AnalysisLogWindow | None = None
        self._cached_log_text = ""
        self._cached_signature: dict[str, object] | None = None

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
        self._refresh_btn = QPushButton("Start Analysis")
        self._refresh_btn.clicked.connect(self._start_analysis)
        controls_row.addWidget(self._refresh_btn)
        controls_row.addWidget(QLabel("Ranking Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        controls_row.addWidget(self._metric_combo)
        controls_row.addWidget(QLabel("Search Strategy:"))
        self._strategy_combo = QComboBox()
        self._strategy_combo.addItems(["legacy", "staged_v1", "staged_v2"])
        self._strategy_combo.setCurrentText("legacy")
        controls_row.addWidget(self._strategy_combo)
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
        self._roles_tab = QWidget()
        self._apply_tab = QWidget()
        self._tabs.addTab(self._overview_tab, "1. Series Overview")
        self._tabs.addTab(self._portfolio_tab, "2. Candidate Portfolio")
        self._tabs.addTab(self._compare_tab, "3. Compare Fits")
        self._tabs.addTab(self._roles_tab, "4. Parameter Sharing")
        self._tabs.addTab(self._apply_tab, "5. Apply")

        self._build_overview_tab()
        self._build_portfolio_tab()
        self._build_compare_tab()
        self._build_roles_tab()
        self._build_apply_tab()

        self._refresh_btn.setEnabled(False)
        self._metric_combo.setEnabled(False)
        self._strategy_combo.setEnabled(False)

    def set_analysis_context(
        self,
        datasets: list[MuonDataset],
        *,
        current_model: CompositeModel | None = None,
        current_parameter_types: dict[str, str] | None = None,
        current_values: dict[str, float] | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Prepare the window for a new ordered dataset series."""
        self._datasets = list(datasets)
        self._current_model = current_model
        self._current_parameter_types = dict(current_parameter_types or {})
        self._current_values = dict(current_values or {})
        self._parameter_bounds = dict(parameter_bounds or {})
        self._recommendation = None
        self._selected_key = None
        self._cached_log_text = ""
        self._cached_signature = None
        self._analysis_request_id += 1
        run_labels = ", ".join(dataset.run_label for dataset in self._datasets[:4])
        if len(self._datasets) > 4:
            run_labels += ", …"
        self._heading_label.setText(f"Global Fit Wizard — {len(self._datasets)} datasets")
        self._status_label.setText(
            f"Ready to analyze the selected series ({run_labels}). "
            "Click Start Analysis to compare common global-fit candidates."
        )
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(SelectionMetric.AICC.value)
        self._metric_combo.blockSignals(False)
        self._set_empty_state()
        self._set_busy(False)

    def set_search_strategy(self, search_strategy: str) -> None:
        """Update the selected search strategy."""
        strategy = str(search_strategy).strip() or "legacy"
        index = self._strategy_combo.findText(strategy)
        if index >= 0:
            self._strategy_combo.setCurrentIndex(index)

    def current_search_strategy(self) -> str:
        """Return the selected search strategy."""
        return self._strategy_combo.currentText() or "legacy"

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
        self._portfolio_table.horizontalHeader().setStretchLastSection(True)
        self._portfolio_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._portfolio_table)

    def _build_compare_tab(self) -> None:
        layout = QVBoxLayout(self._compare_tab)
        self._compare_banner = QLabel("")
        self._compare_banner.setWordWrap(True)
        layout.addWidget(self._compare_banner)
        self._compare_table = QTableWidget(0, 8)
        self._compare_table.setHorizontalHeaderLabels(
            ["Candidate", "Score", "AIC", "AICc", "BIC", "Gate", "Params", "Local"]
        )
        self._compare_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._compare_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._compare_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._compare_table.itemSelectionChanged.connect(self._on_compare_selection_changed)
        layout.addWidget(self._compare_table)

        self._compare_warning_text = QTextEdit()
        self._compare_warning_text.setReadOnly(True)
        self._compare_warning_text.setMinimumHeight(150)
        layout.addWidget(self._compare_warning_text)

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
        self._progress_label.setText("Analysis in progress..." if busy else "")
        self._refresh_btn.setEnabled(bool(self._datasets) and not busy)
        self._metric_combo.setEnabled(self._recommendation is not None and not busy)
        self._strategy_combo.setEnabled(bool(self._datasets) and not busy)

    def _set_empty_state(self) -> None:
        self._overview_banner.setText("")
        self._portfolio_banner.setText("")
        self._compare_banner.setText("")
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
            self._roles_table,
        ):
            table.setRowCount(0)
        self._apply_recommended_btn.setEnabled(False)
        self._apply_selected_btn.setEnabled(False)

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
            "Running global fit wizard analysis in the background. "
            "The main window stays responsive while recommendations "
            "are prepared."
        )
        self._append_progress_log(
            request_id,
            f"Starting analysis for {len(self._datasets)} datasets using "
            f"{self.current_search_strategy()}.",
        )

        self._analysis_thread = QThread(self)
        self._analysis_worker = GlobalFitWizardWorker(
            request_id,
            self._datasets,
            self._current_model,
            self._current_parameter_types,
            self._current_values,
            self._parameter_bounds,
            SelectionMetric.AICC,
            self.current_search_strategy(),
        )
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.progress.connect(self._append_progress_log)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_thread.quit)
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.error.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._cleanup_analysis_thread)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._analysis_thread.start()

    def _on_analysis_finished(self, request_id: int, recommendation: object) -> None:
        thread = self._analysis_thread
        if thread is not None:
            thread.quit()
            thread.wait()
        if request_id != self._analysis_request_id:
            self._set_busy(False)
            return
        if not isinstance(recommendation, GlobalFitWizardRecommendation):
            self._set_busy(False)
            self._status_label.setText("Global fit wizard analysis returned an unexpected result.")
            return

        self._recommendation = recommendation
        self._selected_key = recommendation.recommended_key
        if self._selected_key is None and recommendation.assessments:
            self._selected_key = recommendation.assessments[0].template.key
        self._status_label.setText(recommendation.summary)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._append_progress_log(request_id, recommendation.summary)
        self.analysis_cached.emit(
            recommendation,
            self.current_log_text(),
            copy.deepcopy(self._cached_signature) if self._cached_signature is not None else None,
        )
        self._set_busy(False)
        self._populate_from_recommendation()

    def _on_analysis_error(self, request_id: int, message: str) -> None:
        thread = self._analysis_thread
        if thread is not None:
            thread.quit()
            thread.wait()
        if request_id != self._analysis_request_id:
            self._set_busy(False)
            return
        self._recommendation = None
        self._set_busy(False)
        self._status_label.setText(f"Global fit wizard analysis failed: {message}")
        self._append_progress_log(request_id, f"Analysis failed: {message}")
        self._set_empty_state()

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
    ) -> None:
        """Populate the window from an already-computed recommendation."""
        self._recommendation = recommendation
        self._cached_signature = copy.deepcopy(signature) if isinstance(signature, dict) else None
        self._selected_key = recommendation.recommended_key
        if self._selected_key is None and recommendation.assessments:
            self._selected_key = recommendation.assessments[0].template.key
        self._cached_log_text = str(log_text or "")
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._status_label.setText(recommendation.summary)
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
            "search_strategy": self.current_search_strategy(),
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
            self._compare_banner.setText(recommendation.mixed_axes_warning)
            self._apply_banner.setText(recommendation.mixed_axes_warning)
        else:
            self._overview_banner.setText(
                f"Series ordered by {recommendation.series_axis_label}. "
                "The wizard compares one common fit function across "
                f"{len(recommendation.dataset_order)} datasets."
            )
            self._compare_banner.setText(recommendation.summary)
            self._apply_banner.setText(recommendation.summary)
        self._portfolio_banner.setText(recommendation.summary)
        self._roles_banner.setText(
            "Role recommendations use penalized score differences plus continuity diagnostics. "
            "Fixed parameters are left untouched."
        )

        self._populate_overview_table()
        self._populate_portfolio_table()
        self._populate_compare_table()
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
        self._portfolio_table.setRowCount(len(self._recommendation.templates))
        for row, template in enumerate(self._recommendation.templates):
            title_item = QTableWidgetItem(template.title)
            if template.key == self._recommendation.recommended_key:
                title_item.setFont(_bold_font(title_item.font()))
            self._portfolio_table.setItem(row, 0, title_item)
            self._portfolio_table.setItem(row, 1, QTableWidgetItem(template.category))
            self._portfolio_table.setItem(
                row,
                2,
                QTableWidgetItem(str(len(template.model.param_names))),
            )
            self._portfolio_table.setItem(row, 3, QTableWidgetItem(template.rationale))

    def _populate_compare_table(self) -> None:
        if self._recommendation is None:
            return
        assessments = self._recommendation.sorted_assessments()
        self._compare_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            title_item = QTableWidgetItem(assessment.template.title)
            title_item.setData(Qt.ItemDataRole.UserRole, assessment.template.key)
            if assessment.template.key == self._recommendation.recommended_key:
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
                QTableWidgetItem("Pass" if assessment.residual_gate_passed else "Warn"),
            )
            self._compare_table.setItem(
                row,
                6,
                QTableWidgetItem(str(assessment.parameter_count)),
            )
            self._compare_table.setItem(
                row,
                7,
                QTableWidgetItem(str(len(assessment.local_param_names))),
            )

    def _sync_selected_assessment(self) -> None:
        if self._recommendation is None:
            return
        target_key = self._selected_key or self._recommendation.recommended_key
        if target_key is None and self._recommendation.assessments:
            target_key = self._recommendation.assessments[0].template.key
        self._selected_key = target_key
        for row in range(self._compare_table.rowCount()):
            item = self._compare_table.item(row, 0)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole) == target_key:
                self._compare_table.selectRow(row)
                break
        self._update_compare_warning_text()

    def _selected_assessment(self) -> GlobalCandidateAssessment | None:
        if self._recommendation is None:
            return None
        return (
            self._recommendation.assessment_for_key(self._selected_key)
            or self._recommendation.recommended_assessment
        )

    def _on_compare_selection_changed(self) -> None:
        selected_items = self._compare_table.selectedItems()
        if not selected_items:
            return
        key = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(key, str):
            self._selected_key = key
        self._update_compare_warning_text()
        self._update_roles_table()
        self._update_apply_page()

    def _update_compare_warning_text(self) -> None:
        assessment = self._selected_assessment()
        if assessment is None:
            self._compare_warning_text.setPlainText("")
            return
        lines: list[str] = []
        if assessment.residual_gate_passed:
            lines.append("Residual and continuity checks passed.")
        else:
            lines.append("Warnings:")
            for warning in assessment.series_warnings:
                lines.append(f"• {warning}")
            for diagnostic in assessment.run_diagnostics:
                if diagnostic.gate_reasons:
                    lines.append(
                        f"• Run {diagnostic.run_label}: {', '.join(diagnostic.gate_reasons)}"
                    )
        lines.append(f"AIC = {assessment.aic:.3f}")
        lines.append(
            f"AICc = {assessment.aicc:.3f}"
            if assessment.aicc is not None
            else "AICc fell back to AIC for this candidate."
        )
        lines.append(f"BIC = {assessment.bic:.3f}")
        self._compare_warning_text.setPlainText("\n".join(lines))

    def _update_roles_table(self) -> None:
        assessment = self._selected_assessment()
        if assessment is None:
            self._roles_table.setRowCount(0)
            self._roles_rationale_text.setPlainText("")
            return
        recommendations = list(assessment.parameter_recommendations)
        if not recommendations:
            self._roles_table.setRowCount(0)
            self._roles_rationale_text.setPlainText(
                "Detailed role retests were only generated for the final candidates to keep "
                "the wizard runtime bounded. Apply still uses this candidate's actual "
                "Global/Local assignment."
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
        if self._recommendation is None or assessment is None:
            self._apply_selection_label.setText("")
            self._apply_text.setPlainText("")
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
                "AIC rewards fit quality while penalising parameter count.\n\n"
                "AICc is the default because it adds a small-sample correction "
                "when the total number of fitted "
                "points is not large compared with the number of free parameters.\n\n"
                "BIC penalises complexity more strongly and tends to prefer "
                "simpler shared-parameter descriptions."
            ),
        )

    def _show_warning_info(self) -> None:
        QMessageBox.information(
            self,
            "Global Fit Wizard Warnings",
            (
                "Warnings combine per-run residual checks with ordered-series "
                "continuity diagnostics.\n\n"
                "The wizard looks for clusters of residual failures, abrupt "
                "fingerprint changes, and sharply varying "
                "Local parameter traces across the selected field or temperature series."
            ),
        )

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
