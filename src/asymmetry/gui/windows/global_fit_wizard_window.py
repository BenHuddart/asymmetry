"""Non-modal guided fit wizard for ordered global-fit dataset series."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    ConfidenceTier,
    RecommendationVerdict,
    SelectionMetric,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardCandidatePortfolio,
    GlobalFitWizardRecommendation,
    build_global_fit_wizard_candidate_portfolio,
    build_global_fit_wizard_recommendation,
    build_global_fit_wizard_screening_recommendation,
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
    is_rate_like_parameter,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fitting.wizard_scope import (
    DEFAULT_EFFORT_TIER,
    EFFORT_TIER_DESCRIPTIONS,
    EFFORT_TIER_LABELS,
    EffortTier,
    WizardScope,
    effort_tier_from_payload,
    estimate_screening_cost,
    resolve_scope_for_datasets,
)
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.screen_sizing import resize_to_available
from asymmetry.gui.widgets.wizard_scope_selector import WizardScopeSelector
from asymmetry.gui.windows.wizard_base import WizardWindowBase

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


@dataclass
class _GlobalAnalysisResult:
    """Plain result object returned by the global-fit-wizard analysis task.

    Folds the old ``GlobalFitWizardWorker`` signals into one value: the analysis
    ``mode`` (screening/optimize) and any ``updated_single_fit_recommendations``
    (formerly the one-shot ``single_fit_precomputed`` signal) ride alongside the
    ``recommendation`` so the base's single ``finished`` path can apply them.
    """

    mode: str
    recommendation: object
    updated_single_fit_recommendations: dict[int, object]


def _run_global_fit_wizard_analysis(
    worker,
    *,
    mode: str,
    datasets: list[MuonDataset],
    current_model: CompositeModel | None,
    current_parameter_types: dict[str, str],
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    existing_single_fit_recommendations_by_run: dict[int, object] | None,
    metric: SelectionMetric,
    selected_template_keys: tuple[str, ...] = (),
    scope: dict | None = None,
    effort_tier: EffortTier = DEFAULT_EFFORT_TIER,
) -> _GlobalAnalysisResult:
    """Run the global-fit wizard analysis off the GUI thread.

    Moved from the former ``GlobalFitWizardWorker.run`` body; exceptions now
    propagate to ``TaskWorker.run`` (→ the base error slot) instead of being
    caught and re-emitted, and progress goes through ``worker.progress.emit``.
    Cooperative cancel is honoured between builder phases: the base passes a
    ``FitCancelledError`` in ``_cancel_exceptions()`` so ``TaskWorker`` reports
    it as a cancellation rather than a failure. ``scope`` is the serialised
    ``WizardScope`` payload from the Scope tab (``None`` → whole time domain);
    it is converted here (worker thread) and forwarded to every builder.
    ``effort_tier`` is the user-facing effort slider (PR 5); it only affects the
    coupled-optimisation builder (``mode == "optimize"``) — the independent
    per-run screening pass has no tier concept.
    """
    resolved_scope = WizardScope.from_payload(scope) if scope is not None else None

    def _raise_if_cancelled() -> None:
        if worker.is_cancelled():
            raise FitCancelledError("Analysis cancelled.")

    existing = dict(existing_single_fit_recommendations_by_run or {})
    selected_template_keys = tuple(key for key in selected_template_keys if isinstance(key, str))

    single_fit_recommendations_before_analysis = dict(existing)
    screening_builder_is_custom = (
        build_global_fit_wizard_screening_recommendation is not _DEFAULT_SCREENING_BUILDER
    )
    optimization_builder_is_custom = (
        build_global_fit_wizard_recommendation is not _DEFAULT_GLOBAL_FIT_BUILDER
    )
    skip_implicit_phase_one = (
        (
            (mode == "screening" and screening_builder_is_custom)
            or (mode == "optimize" and optimization_builder_is_custom)
        )
        and build_or_complete_single_fit_wizard_recommendations_for_global_portfolio
        is _DEFAULT_PHASE_ONE_SINGLE_FIT_HELPER
        and not existing
    )
    _raise_if_cancelled()
    if skip_implicit_phase_one:
        single_fit_recommendations_by_run = dict(existing)
    else:
        _portfolio, single_fit_recommendations_by_run, _generated_runs = (
            build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
                datasets,
                current_model=current_model,
                existing_recommendations_by_run=existing,
                progress_callback=lambda message: worker.progress.emit(0, 0, message),
                scope=resolved_scope,
            )
        )

    def progress_callback(message):
        return worker.progress.emit(0, 0, message)

    _raise_if_cancelled()
    if mode == "screening":
        recommendation = build_global_fit_wizard_screening_recommendation(
            datasets,
            current_model=current_model,
            current_parameter_types=current_parameter_types,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            single_fit_recommendations_by_run=single_fit_recommendations_by_run,
            metric=metric,
            progress_callback=progress_callback,
            scope=resolved_scope,
        )
    else:
        recommendation = build_global_fit_wizard_recommendation(
            datasets,
            current_model=current_model,
            current_parameter_types=current_parameter_types,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            single_fit_recommendations_by_run=single_fit_recommendations_by_run,
            metric=metric,
            progress_callback=progress_callback,
            selected_template_keys=selected_template_keys,
            scope=resolved_scope,
            effort_tier=effort_tier,
        )
    updated_single_fit_recommendations = {
        int(run_number): rec
        for run_number, rec in single_fit_recommendations_by_run.items()
        if single_fit_recommendations_before_analysis.get(int(run_number)) is not rec
    }
    return _GlobalAnalysisResult(
        mode=mode,
        recommendation=recommendation,
        updated_single_fit_recommendations=updated_single_fit_recommendations,
    )


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
                    _default_parameter_bounds(
                        name, current_parameter_bounds=current_parameter_bounds
                    )
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
                min_val, max_val = _parse_bounds_text(
                    bounds_item.text() if bounds_item else "-inf, inf"
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Bounds", f"{name}: {exc}")
                return
            types[name] = role
            bounds[name] = (min_val, max_val)
        self._configuration = {"types": types, "bounds": bounds}
        super().accept()


class GlobalFitWizardWindow(WizardWindowBase):
    """Present a guided workflow for global-fit model recommendation."""

    apply_assessment_requested = Signal(object, object)
    analysis_cached = Signal(object, str, object)
    parameter_setup_applied = Signal(object)
    single_fit_recommendations_generated = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        # WizardWindowBase.__init__ builds the shared frame and calls
        # _build_tabs() before this body resumes.
        super().__init__(parent)
        self.setWindowTitle("Global Fit Wizard")
        # Cap the default to the available screen so the title bar never opens
        # clipped above the menu bar on a 13-inch laptop; the tab bodies scroll
        # so the spacious preferred size applies only when the display fits it
        # (P1-5).
        resize_to_available(self, 1180, 740)

        heading_font = QFont(self._heading_label.font())
        heading_font.setPointSize(max(heading_font.pointSize() + 4, 14))
        heading_font.setBold(True)
        self._heading_label.setFont(heading_font)
        self._heading_label.setText("Global Fit Wizard")
        self._status_label.setText(
            "Open the global fit wizard on a field or temperature series "
            "to compare common model families and recommended "
            "Global/Local parameter roles."
        )

        # Stale banner sits under the status label (heading/status/controls/tabs
        # is the base's central layout order, so index 2 lands it just above the
        # controls row). Shown after a Scope edit invalidates the shown results.
        self._stale_banner = QLabel(
            "Scope changed since the last analysis — the results below are stale. "
            "Re-run the screening."
        )
        self._stale_banner.setWordWrap(True)
        self._stale_banner.setStyleSheet(f"color: {tokens.ERROR}; font-weight: 600;")
        self._stale_banner.setVisible(False)
        self._central_layout.insertWidget(2, self._stale_banner)

        self._refresh_btn.setEnabled(False)
        self._metric_combo.setEnabled(False)
        self._optimize_btn.setEnabled(False)

    def _build_tabs(self) -> None:
        # The base calls this during __init__, before the subclass body resumes,
        # so the result-state members are initialised here.
        self._datasets: list[MuonDataset] = []
        self._current_model: CompositeModel | None = None
        self._current_parameter_types: dict[str, str] = {}
        self._current_values: dict[str, float] = {}
        self._parameter_bounds: dict[str, tuple[float, float]] = {}
        self._recommendation: GlobalFitWizardRecommendation | None = None
        self._selected_key: str | None = None
        self._screening_selected_keys: set[str] = set()
        self._running_template_keys: set[str] = set()
        self._analysis_mode = "screening"
        self._log_window: AnalysisLogWindow | None = None
        self._single_fit_recommendations_by_run: dict[int, object] = {}
        # A Scope edit invalidates the shown results; screening must be re-run.
        self._analysis_stale = False

        # Insert the window's controls before the base-owned progress widgets
        # (index 0..6), then a trailing stretch keeps the row left-aligned.
        self._refresh_btn = QPushButton("Build Screening Table")
        self._refresh_btn.clicked.connect(self._start_analysis)
        self._controls_row.insertWidget(0, self._refresh_btn)
        self._optimize_btn = QPushButton("Optimize Selected")
        self._optimize_btn.clicked.connect(self._start_selected_optimisation)
        self._controls_row.insertWidget(1, self._optimize_btn)
        self._controls_row.insertWidget(2, QLabel("Ranking Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        self._controls_row.insertWidget(3, self._metric_combo)
        metric_info_btn = QPushButton("Metric Info")
        metric_info_btn.clicked.connect(self._show_metric_info)
        self._controls_row.insertWidget(4, metric_info_btn)
        warning_info_btn = QPushButton("Warning Info")
        warning_info_btn.clicked.connect(self._show_warning_info)
        self._controls_row.insertWidget(5, warning_info_btn)
        self._controls_row.insertWidget(6, QLabel("Effort:"))
        self._effort_combo = QComboBox()
        for tier in EffortTier:
            self._effort_combo.addItem(EFFORT_TIER_LABELS[tier], userData=tier.value)
            self._effort_combo.setItemData(
                self._effort_combo.count() - 1,
                EFFORT_TIER_DESCRIPTIONS[tier],
                Qt.ItemDataRole.ToolTipRole,
            )
        self._effort_combo.setCurrentIndex(self._effort_combo.findData(DEFAULT_EFFORT_TIER.value))
        self._effort_combo.setToolTip(EFFORT_TIER_DESCRIPTIONS[DEFAULT_EFFORT_TIER])
        self._effort_combo.currentIndexChanged.connect(self._on_effort_tier_changed)
        self._controls_row.insertWidget(7, self._effort_combo)
        # Cancel button lives in the controls row; visible only while busy (see
        # _update_action_enablement). It routes through the base's single cancel
        # entry point, which cancels the live TaskWorker cooperatively.
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_current_analysis)
        self._controls_row.insertWidget(8, self._cancel_btn)
        self._controls_row.addStretch()

        self._scope_tab = QWidget()
        self._overview_tab = QWidget()
        self._portfolio_tab = QWidget()
        self._compare_tab = QWidget()
        self._optimized_tab = QWidget()
        self._roles_tab = QWidget()
        self._apply_tab = QWidget()
        self._tabs.addTab(self._scope_tab, "1. Scope")
        self._tabs.addTab(self._overview_tab, "2. Series Overview")
        self._tabs.addTab(self._portfolio_tab, "3. Candidate Portfolio")
        self._tabs.addTab(self._compare_tab, "4. Single-Fit Screening")
        self._tabs.addTab(self._optimized_tab, "5. Global Optimized Fits")
        self._tabs.addTab(self._roles_tab, "6. Parameter Sharing")
        self._tabs.addTab(self._apply_tab, "7. Apply")

        self._build_scope_tab()
        self._build_overview_tab()
        self._build_portfolio_tab()
        self._build_compare_tab()
        self._build_optimized_tab()
        self._build_roles_tab()
        self._build_apply_tab()

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
        self._single_fit_recommendations_by_run = dict(
            existing_single_fit_recommendations_by_run or {}
        )
        self._recommendation = None
        self._selected_key = None
        self._screening_selected_keys = set()
        self._running_template_keys = set()
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._cached_log_text = ""
        self._cached_signature = None
        self._analysis_request_id += 1
        # Install the scope resolver and reset the selector to Auto (signal-silent),
        # then refresh so is_valid() sees a populated tree before the final
        # _set_busy(False) below evaluates the button states.
        self._scope_selector.set_resolver(self._resolve_scope)
        self._scope_selector.set_scope(None)
        self._scope_selector.refresh_from_context()
        self._tabs.setCurrentIndex(0)
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
        # Run / Field / Temperature are known now, so show the series immediately
        # rather than an empty table until screening. The classification columns
        # stay "—" until a recommendation is built (see _populate_overview_table).
        self._populate_overview_table()
        if self._datasets:
            self._overview_banner.setText(
                f"{len(self._datasets)} runs selected. "
                "Run screening to classify each run (Osc. / KT-like / Multi-rate)."
            )
        self._set_busy(False)

    def _build_scope_tab(self) -> None:
        layout = QVBoxLayout(self._scope_tab)
        intro = QLabel(
            "Choose which candidate families the wizard screens across the series. Start "
            "from a preset (or Auto, inferred from run metadata) and include/exclude "
            "individual components as needed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._scope_selector = WizardScopeSelector()
        self._scope_selector.scope_changed.connect(self._on_scope_changed)
        self._scope_selector.validity_changed.connect(self._on_scope_validity_changed)
        layout.addWidget(self._scope_selector, 1)

    def _resolve_scope(self, preset_id: str, overrides: dict) -> dict:
        """Adapt the core scope resolver to the WizardScopeSelector dict contract.

        Groups in-registry-order TIME-domain components by their display
        ``category``; frequency-domain components are skipped entirely. The scope
        is resolved across the whole selected series (union of in-scope sets).
        """
        if not self._datasets:
            return {
                "effective_preset": preset_id,
                "note": "Load a series first",
                "families": [],
                "estimate": [0, 0],
            }
        scope = WizardScope.from_payload(
            {
                "version": 1,
                "preset": preset_id,
                "include": overrides.get("include", []),
                "exclude": overrides.get("exclude", []),
            }
        )
        resolution = resolve_scope_for_datasets(list(self._datasets), scope)
        included = resolution.included_set
        reasons = {exc.name: exc.reason for exc in resolution.excluded_components}

        families: list[dict] = []
        by_category: dict[str, dict] = {}
        for name, definition in COMPONENTS.items():
            if definition.domain != "time":
                continue
            category = definition.category
            family = by_category.get(category)
            if family is None:
                family = {"key": category, "title": category, "components": []}
                by_category[category] = family
                families.append(family)
            family["components"].append(
                {
                    "name": name,
                    "included": name in included,
                    "reason": reasons.get(name, ""),
                    "cost": definition.cost.value,
                }
            )

        return {
            "effective_preset": resolution.effective_preset.value,
            "note": resolution.inference_note,
            "families": families,
            "estimate": list(estimate_screening_cost(resolution)),
        }

    def _on_scope_changed(self, _scope: object) -> None:
        # A stale screening table's selection no longer corresponds to the new
        # scope, so clear it before disabling "Optimize Selected" via _set_busy.
        self._screening_selected_keys = set()
        self._mark_analysis_stale("Scope changed")

    def _on_scope_validity_changed(self, is_valid: bool) -> None:
        if not is_valid and not self._analysis_in_progress:
            self._status_label.setText(
                "Select at least one candidate family on the Scope tab to enable screening."
            )
        self._set_busy(self._analysis_in_progress)

    def _mark_analysis_stale(self, reason: str) -> None:
        """Flag the displayed results as stale after a scope edit.

        Follows the ignore-stale convention: an in-flight analysis is orphaned by
        bumping the request id (its terminal signal is discarded by the base's
        staleness guard on arrival). We also cancel the live worker cooperatively
        so it stops wasting cycles, then clear busy.
        """
        if self._analysis_in_progress:
            self._cancel_current_analysis()
            self._analysis_request_id += 1
            self._set_busy(False)
            self._status_label.setText(
                f"{reason} while analysis was running; that result will be discarded. "
                "Re-run the screening."
            )
        if self._recommendation is not None:
            self._analysis_stale = True
            self._stale_banner.setVisible(True)
        self._set_busy(self._analysis_in_progress)

    def _cancel_exceptions(self) -> tuple[type[BaseException], ...]:
        return (FitCancelledError,)

    def _build_overview_tab(self) -> None:
        layout = QVBoxLayout(self._overview_tab)
        self._overview_banner = QLabel("")
        self._overview_banner.setWordWrap(True)
        layout.addWidget(self._overview_banner)
        # Unmissable, series-level flag when any run's single-fit shows no
        # significant structure. Hidden until screening surfaces such a run.
        self._verdict_banner = QLabel("")
        self._verdict_banner.setWordWrap(True)
        self._verdict_banner.setVisible(False)
        layout.addWidget(self._verdict_banner)
        self._overview_table = QTableWidget(0, 8)
        self._overview_table.setHorizontalHeaderLabels(
            [
                "Run",
                "Field (G)",
                "Temperature (K)",
                "Osc.",
                "KT-like",
                "Multi-rate",
                "Confidence",
                "Recommendation",
            ]
        )
        self._overview_table.horizontalHeader().setStretchLastSection(True)
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
        self._portfolio_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
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
        self._compare_table.horizontalHeader().setSectionResizeMode(
            8, QHeaderView.ResizeMode.Stretch
        )
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
        self._optimized_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch
        )
        self._optimized_table.horizontalHeader().setSectionResizeMode(
            7, QHeaderView.ResizeMode.Stretch
        )
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

    def _update_action_enablement(self, busy: bool) -> None:
        self._progress_label.setText("Working..." if busy else "")
        self._cancel_btn.setVisible(busy)
        self._refresh_btn.setEnabled(
            bool(self._datasets) and not busy and self._scope_selector.is_valid()
        )
        self._metric_combo.setEnabled(self._recommendation is not None and not busy)
        has_screening_selection = bool(self._screening_selected_keys)
        self._optimize_btn.setEnabled(
            self._recommendation is not None
            and has_screening_selection
            and not busy
            and not self._analysis_stale
        )

    def _set_empty_state(self) -> None:
        self._overview_banner.setText("")
        self._verdict_banner.setText("")
        self._verdict_banner.setVisible(False)
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

        if not self._scope_selector.is_valid():
            self._status_label.setText(
                "Select at least one candidate family on the Scope tab to enable screening."
            )
            return

        scope_payload = copy.deepcopy(self._scope_selector.current_scope())
        try:
            portfolio = build_global_fit_wizard_candidate_portfolio(
                self._datasets,
                current_model=self._current_model,
                scope=WizardScope.from_payload(scope_payload),
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

        # Same-signature short-circuit: serve the cached recommendation without
        # recomputing. Scope is in the signature, so a change-then-revert to the
        # cached scope makes any stale flag obsolete — clear it. The base
        # bumps/caches on _run_analysis(), so this stays a subclass decision (the
        # single-fit window always recomputes).
        if (
            self._cached_signature == self._analysis_signature()
            and self._recommendation is not None
        ):
            self._analysis_stale = False
            self._stale_banner.setVisible(False)
            self._status_label.setText(self._recommendation.summary)
            self._populate_from_recommendation()
            return

        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._analysis_mode = "screening"
        self._show_log_window()
        self._status_label.setText(
            "Building the single-fit screening table in the background. "
            "The main window stays responsive while the shared candidate portfolio is screened."
        )
        self._append_log(f"Starting screening for {len(self._datasets)} datasets.")
        # Base: bump request id, cache signature, set busy, _reset_result_state(),
        # then run _create_worker_task() off-thread.
        self._run_analysis()

    def _start_selected_optimisation(self) -> None:
        if self._recommendation is None or not self._screening_selected_keys:
            return
        if self._analysis_in_progress:
            return
        self._analysis_mode = "optimize"
        self._running_template_keys = set(self._screening_selected_keys)
        selected_titles = [
            assessment.template.title
            for assessment in self._recommendation.assessments
            if assessment.template.key in self._running_template_keys
        ]
        self._show_log_window()
        self._status_label.setText(
            "Running coupled global optimisation for the selected candidates. Progress is streamed to the log window."
        )
        self._append_log(
            "Starting coupled global optimisation for: " + ", ".join(selected_titles) + "."
        )
        self._populate_compare_table()
        self._run_analysis()

    def _create_worker_task(self, request_id: int):
        # Capture inputs at submit time; the closure runs on the worker thread
        # and must touch no widgets — it returns a plain _GlobalAnalysisResult.
        mode = self._analysis_mode
        datasets = list(self._datasets)
        current_model = self._current_model
        current_parameter_types = dict(self._current_parameter_types)
        current_values = dict(self._current_values)
        parameter_bounds = dict(self._parameter_bounds)
        existing = dict(self._single_fit_recommendations_by_run)
        metric = SelectionMetric.from_value(self._metric_combo.currentText())
        selected_keys = tuple(sorted(self._screening_selected_keys)) if mode == "optimize" else ()
        scope_payload = copy.deepcopy(self._scope_selector.current_scope())
        effort_tier = self.current_effort_tier()

        def task(worker):
            return _run_global_fit_wizard_analysis(
                worker,
                mode=mode,
                datasets=datasets,
                current_model=current_model,
                current_parameter_types=current_parameter_types,
                current_values=current_values,
                parameter_bounds=parameter_bounds,
                existing_single_fit_recommendations_by_run=existing,
                metric=metric,
                selected_template_keys=selected_keys,
                scope=scope_payload,
                effort_tier=effort_tier,
            )

        return task

    def _populate_results(self, result: object) -> None:
        # Apply the single-fit update the worker computed (formerly the one-shot
        # single_fit_precomputed signal, now folded into the result object).
        if result.updated_single_fit_recommendations:
            typed_payload = {
                int(run_number): rec
                for run_number, rec in result.updated_single_fit_recommendations.items()
            }
            self._single_fit_recommendations_by_run.update(typed_payload)
            self.single_fit_recommendations_generated.emit(typed_payload)

        recommendation = result.recommendation
        if result.mode == "optimize" and self._recommendation is not None:
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
        self._append_log(self._recommendation.summary)
        self.analysis_cached.emit(
            self._recommendation,
            self.current_log_text(),
            copy.deepcopy(self._cached_signature) if self._cached_signature is not None else None,
        )
        # The base cleared busy before calling us, while _recommendation was
        # still the old value; re-assert enablement now it is set.
        self._update_action_enablement(False)
        self._populate_from_recommendation()

    def _reset_result_state(self) -> None:
        # Screening starts from a clean slate; optimize merges into the existing
        # screening recommendation, so it keeps the current result and the
        # running-template highlight rather than clearing them.
        if self._analysis_mode == "optimize":
            return
        self._set_empty_state()

    def _on_analysis_failed(self, message: str) -> None:
        # The base has already cleared busy and run the request-id staleness
        # guard; _analysis_mode was stashed when the run started.
        self._running_template_keys = set()
        if self._analysis_mode == "screening":
            self._recommendation = None
        self._status_label.setText(f"Global fit wizard analysis failed: {message}")
        self._append_log(f"Analysis failed: {message}")
        if self._recommendation is None:
            self._set_empty_state()
        else:
            self._populate_from_recommendation()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        # Base already guarded the request id; stream to the log window.
        self._append_log(message)

    def _show_log_window(self) -> None:
        if self._log_window is None:
            self._log_window = AnalysisLogWindow(self)
        self._log_window.clear()
        self._log_window.show()
        self._log_window.raise_()
        self._log_window.activateWindow()

    def _append_log(self, message: str) -> None:
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
        # Restore scope from the signature. Legacy signatures without a scope key
        # restore as Auto. Cached state is never stale. set_scope is a no-op on
        # the tree when no resolver is installed (no prior set_analysis_context).
        signature_dict = signature if isinstance(signature, dict) else {}
        cached_scope = signature_dict.get("scope")
        self._scope_selector.set_scope(cached_scope if isinstance(cached_scope, dict) else None)
        # Legacy signatures without an "effort_tier" key restore as Balanced.
        self._set_effort_tier(effort_tier_from_payload(signature_dict.get("effort_tier")))
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._status_label.setText(status_text or recommendation.summary)
        self._set_busy(False)
        self._populate_from_recommendation()

    def current_effort_tier(self) -> EffortTier:
        """The effort tier currently selected on the slider (default Balanced)."""
        return effort_tier_from_payload(self._effort_combo.currentData())

    def _set_effort_tier(self, tier: EffortTier) -> None:
        index = self._effort_combo.findData(tier.value)
        if index < 0:
            return
        self._effort_combo.blockSignals(True)
        self._effort_combo.setCurrentIndex(index)
        self._effort_combo.blockSignals(False)
        self._effort_combo.setToolTip(EFFORT_TIER_DESCRIPTIONS[tier])

    def _on_effort_tier_changed(self, _index: int) -> None:
        tier = self.current_effort_tier()
        self._effort_combo.setToolTip(EFFORT_TIER_DESCRIPTIONS[tier])
        # A different effort tier changes which fits the next analysis run would
        # perform, so any already-shown results are stale — mirrors _on_scope_changed.
        self._mark_analysis_stale("Effort level changed")

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
            "scope": self._scope_selector.current_scope(),
            "effort_tier": self.current_effort_tier().value,
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
            str(name): str(role) for name, role in types.items() if isinstance(name, str)
        }
        typed_bounds: dict[str, tuple[float, float]] = {}
        for name, raw_bounds in bounds.items():
            if (
                not isinstance(name, str)
                or not isinstance(raw_bounds, tuple | list)
                or len(raw_bounds) != 2
            ):
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
            self._current_values[name] = float(
                np.clip(self._current_values[name], min_val, max_val)
            )

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
        """List one row per selected run in the Series Overview.

        Run / Field / Temperature are known as soon as the series is set, so the
        overview is populated immediately by :meth:`set_analysis_context` — it no
        longer sits empty until screening. The Osc. / KT-like / Multi-rate columns
        come from per-run fingerprints, which only exist once screening has built
        a recommendation; until then they show ``"—"``. Before screening the rows
        follow the input order; once a recommendation exists they follow its
        series-axis ordering (``dataset_order``).
        """
        recommendation = self._recommendation
        if recommendation is not None:
            run_order = [int(run_number) for run_number in recommendation.dataset_order]
            fingerprints = recommendation.fingerprints_by_run
        else:
            run_order = [int(dataset.run_number) for dataset in self._datasets]
            fingerprints = None
        self._overview_table.setRowCount(len(run_order))
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        for row, run_number in enumerate(run_order):
            dataset = by_run.get(int(run_number))
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
            if fingerprints is not None:
                fingerprint = fingerprints[int(run_number)]
                osc_text = "Yes" if fingerprint.oscillatory_hint else "No"
                kt_text = "Yes" if fingerprint.kt_like_hint else "No"
                multi_text = "Yes" if fingerprint.multi_rate_hint else "No"
            else:
                osc_text = kt_text = multi_text = "—"
            self._overview_table.setItem(row, 0, QTableWidgetItem(run_label))
            self._overview_table.setItem(row, 1, QTableWidgetItem(field_text))
            self._overview_table.setItem(row, 2, QTableWidgetItem(temperature_text))
            self._overview_table.setItem(row, 3, QTableWidgetItem(osc_text))
            self._overview_table.setItem(row, 4, QTableWidgetItem(kt_text))
            self._overview_table.setItem(row, 5, QTableWidgetItem(multi_text))
            confidence_item, recommendation_item = self._overview_confidence_items(int(run_number))
            self._overview_table.setItem(row, 6, confidence_item)
            self._overview_table.setItem(row, 7, recommendation_item)
        self._update_verdict_banner(run_order)

    def _overview_confidence_items(
        self, run_number: int
    ) -> tuple[QTableWidgetItem, QTableWidgetItem]:
        """Build the Confidence / Recommendation cells for one run.

        Both come from that run's single-fit recommendation, which carries the
        confidence tier, verdict, and caveat. A run whose best single fit shows
        no significant structure (a null baseline) is marked in red so a
        pure-noise run is never silently presented as a good fit; a
        medium-confidence run carries its caveat as an amber tooltip. When no
        single-fit recommendation exists yet (before screening) both cells show
        ``"—"``.
        """
        rec = self._single_fit_recommendations_by_run.get(int(run_number))
        confidence = getattr(rec, "confidence", None)
        verdict = getattr(rec, "verdict", None)
        caveat = str(getattr(rec, "caveat", "") or "")
        if rec is None or confidence is None:
            return QTableWidgetItem("—"), QTableWidgetItem("—")

        confidence_item = QTableWidgetItem(_confidence_label(confidence))
        if verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
            recommendation_item = QTableWidgetItem("No significant structure")
            for item in (confidence_item, recommendation_item):
                item.setForeground(QColor(tokens.ERROR))
                item.setFont(_bold_font(item.font()))
            if caveat:
                recommendation_item.setToolTip(caveat)
        else:
            recommended = getattr(rec, "recommended_assessment", None)
            title = getattr(getattr(recommended, "template", None), "title", "")
            recommendation_item = QTableWidgetItem(str(title) or "—")
            if confidence is ConfidenceTier.MEDIUM:
                confidence_item.setForeground(QColor(tokens.WARN))
                if caveat:
                    confidence_item.setToolTip(caveat)
                    recommendation_item.setToolTip(caveat)
            elif confidence is ConfidenceTier.HIGH:
                confidence_item.setForeground(QColor(tokens.OK))
        return confidence_item, recommendation_item

    def _update_verdict_banner(self, run_order: list[int]) -> None:
        """Raise an unmissable series-level banner for null-structure runs.

        Any run whose single-fit verdict is NO_SIGNIFICANT_STRUCTURE means the
        data on that run carry no structure worth a richer model. A single red
        table cell is easy to miss, so this surfaces the count at series level.
        """
        flagged: list[str] = []
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        for run_number in run_order:
            rec = self._single_fit_recommendations_by_run.get(int(run_number))
            if getattr(rec, "verdict", None) is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
                dataset = by_run.get(int(run_number))
                flagged.append(dataset.run_label if dataset else str(run_number))
        if not flagged:
            self._verdict_banner.setVisible(False)
            self._verdict_banner.setText("")
            return
        labels = ", ".join(flagged)
        self._verdict_banner.setText(
            f"No significant structure on {len(flagged)} run(s): {labels}. "
            "The data there do not support a richer model than a flat/exponential baseline."
        )
        self._verdict_banner.setStyleSheet(f"color: {tokens.ERROR}; font-weight: 600;")
        self._verdict_banner.setVisible(True)

    def _populate_portfolio_table(self) -> None:
        if self._recommendation is None:
            return
        recommended_assessment = self._recommendation.recommended_assessment
        self._portfolio_table.setRowCount(len(self._recommendation.templates))
        for row, template in enumerate(self._recommendation.templates):
            title_item = QTableWidgetItem(template.title)
            if (
                recommended_assessment is not None
                and template.key == recommended_assessment.template.key
            ):
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
                QTableWidgetItem(
                    self._recommendation.optimization_status_for_key(assessment.template.key)
                ),
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
            self._optimized_table.setItem(
                row, 1, _numeric_item(assessment.metric_value(self._recommendation.metric))
            )
            self._optimized_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._optimized_table.setItem(
                row,
                3,
                _numeric_item(assessment.aicc)
                if assessment.aicc is not None
                else QTableWidgetItem("AIC"),
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
        target_key = self._selected_key or self._recommended_or_first_optimized_key(
            self._recommendation
        )
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
            lines.append(
                f"Status: {self._recommendation.optimization_status_for_key(primary.template.key)}"
            )
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
            self._apply_selection_label.setText("No globally optimized candidate is selected yet.")
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


def _numeric_item(value: float) -> QTableWidgetItem:
    item = QTableWidgetItem(f"{float(value):.3f}")
    item.setData(Qt.ItemDataRole.UserRole, float(value))
    return item


def _bold_font(font: QFont) -> QFont:
    updated = QFont(font)
    updated.setBold(True)
    return updated


_CONFIDENCE_LABELS = {
    ConfidenceTier.HIGH: "High",
    ConfidenceTier.MEDIUM: "Medium",
    ConfidenceTier.NONE: "—",
}


def _confidence_label(confidence: ConfidenceTier) -> str:
    return _CONFIDENCE_LABELS.get(confidence, "—")


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
    return (
        (float(default_min), float("inf"))
        if default_min is not None
        else (-float("inf"), float("inf"))
    )


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
