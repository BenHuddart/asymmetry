"""Non-modal guided fit wizard for ordered global-fit dataset series.

The window is a three-state, answer-first shell built on
:class:`~asymmetry.gui.windows.wizard_base.WizardWindowBase` via the
``_build_central()`` hook (so ``self._tabs`` stays ``None`` and no tab
scaffolding is created). The three states live in a ``QStackedWidget``:

* **Setup** — the series overview (one row per run, populated as soon as the
  context arrives), the scope selector, a collapsed *Guide the search
  (optional)* section housing the embedded parameter-expectations editor
  (formerly a blocking modal dialog), the search settings, and a prominent
  *Run screening* button.
* **Running** — a streaming decision trail whose steps light up as the core
  reports progress, above a collapsed *Live log* section that captures every
  progress message inline (formerly a separate log window).
* **Result** — the series answer card (verdict + overlaid data/fit traces +
  local-parameter trend + apply + alternatives) above the screening shortlist
  and a finished decision trail whose steps expand to the demoted detail
  tables (portfolio, optimized fits, parameter roles, apply).

The external surface (``set_analysis_context``, ``set_cached_recommendation``,
the ``apply_assessment_requested``/``analysis_cached``/
``single_fit_recommendations_generated``/``parameter_setup_applied`` signals,
and the seven-key ``_analysis_signature``) is unchanged so ``global_tab.py``
needs no edits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
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
from asymmetry.core.fitting.wizard_narrative import TrailStep
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
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    make_section_header,
    make_warning_banner,
)
from asymmetry.gui.widgets.collapsible_section import CollapsibleSection
from asymmetry.gui.widgets.decision_trail import DecisionTrail
from asymmetry.gui.widgets.screen_sizing import resize_to_available
from asymmetry.gui.widgets.wizard_scope_selector import WizardScopeSelector
from asymmetry.gui.widgets.wizard_series_card import (
    SeriesRunTrace,
    SeriesTrend,
    WizardSeriesCard,
)
from asymmetry.gui.windows.wizard_base import WizardWindowBase

_DEFAULT_PHASE_ONE_SINGLE_FIT_HELPER = (
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio
)
_DEFAULT_SCREENING_BUILDER = build_global_fit_wizard_screening_recommendation
_DEFAULT_GLOBAL_FIT_BUILDER = build_global_fit_wizard_recommendation

#: Stacked-widget page indices.
_PAGE_SETUP = 0
_PAGE_RUNNING = 1
_PAGE_RESULT = 2

#: Progress-message prefix → running-trail step key, per analysis mode. Matched
#: case-insensitively by prefix so the core's coarse phase messages light the
#: right step regardless of the run label/count they interpolate; an unmatched
#: message only updates the trail status line (and the live log).
_SCREENING_PROGRESS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("preparing consolidated", "context"),
    ("preparing per-dataset single-fit", "screening"),
    ("preparing missing single-fit", "screening"),
    ("running phase-1 single-fit", "screening"),
    ("single-fit table", "screening"),
    ("using completed per-run single-fit", "ranking"),
)

_OPTIMIZE_PROGRESS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("preparing consolidated", "prepare"),
    ("preparing per-dataset single-fit", "prepare"),
    ("preparing missing single-fit", "prepare"),
    ("running phase-1 single-fit", "prepare"),
    ("single-fit table", "prepare"),
    ("using completed per-run single-fit", "prepare"),
    ("running coupled global optimisation", "optimize"),
    ("coupled global optimisation will evaluate", "optimize"),
    ("coupled optimisation", "optimize"),
    ("using serial wavefront", "optimize"),
    ("completed exhaustive coupled optimisation", "roles"),
    ("completed heuristic coupled optimisation", "roles"),
    ("completed coupled optimisation", "roles"),
)


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
                cancel_callback=worker.is_cancelled,
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
            cancel_callback=worker.is_cancelled,
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
            cancel_callback=worker.is_cancelled,
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


class GlobalFitWizardWindow(WizardWindowBase):
    """Present a guided workflow for global-fit model recommendation."""

    apply_assessment_requested = Signal(object, object)
    analysis_cached = Signal(object, str, object)
    parameter_setup_applied = Signal(object)
    single_fit_recommendations_generated = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        # WizardWindowBase.__init__ builds the shared frame and calls
        # _build_central() before this body resumes.
        super().__init__(parent)
        self.setWindowTitle("Global Fit Wizard")
        # Cap the default to the available screen so the title bar never opens
        # clipped above the menu bar on a 13-inch laptop; the page bodies scroll
        # so the spacious preferred size applies only when the display fits it
        # (P1-5).
        resize_to_available(self, 1180, 740)

        self._heading_label.setText("Global Fit Wizard")
        self._status_label.setText(
            "Open the global fit wizard on a field or temperature series "
            "to compare common model families and recommended "
            "Global/Local parameter roles."
        )

        self._refresh_btn.setEnabled(False)
        self._metric_combo.setEnabled(False)
        self._optimize_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Content region (overrides the base tab scaffolding)
    # ------------------------------------------------------------------

    def _build_central(self) -> QWidget:
        """Build the three-state stacked content region.

        Runs during ``WizardWindowBase.__init__`` (before this subclass body
        resumes), so it initialises every result-state member the old
        ``_build_tabs`` used to, then constructs the Setup/Running/Result pages
        and the deep panels injected into the result trail.
        """
        # --- Result / analysis state ---
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
        self._single_fit_recommendations_by_run: dict[int, object] = {}
        # A Scope edit invalidates the shown results; screening must be re-run.
        self._analysis_stale = False
        # Row order of the embedded expectations table; empty when the table
        # could not be populated (portfolio failure / mixed axes / no context).
        self._expectation_parameter_names: list[str] = []

        # Base controls row: the progress label/bar already occupy indices 0-1;
        # only Cancel joins them (visible while busy — see
        # _update_action_enablement). Everything else lives on the pages.
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_current_analysis)
        self._controls_row.addWidget(self._cancel_btn)
        self._controls_row.addStretch()

        # Stale banner sits above the stack. Shown after a Scope edit
        # invalidates the shown results.
        self._stale_banner = make_warning_banner(
            "Scope changed since the last analysis — the results below are stale. "
            "Re-run the screening."
        )
        self._stale_banner.setVisible(False)
        self._central_layout.addWidget(self._stale_banner)

        # Deep panels: built once here, injected once into the result trail's
        # step expansions (set_step_detail_widget persists across set_steps
        # rebuilds, so the trail can be re-derived without re-injection).
        self._portfolio_panel = self._build_portfolio_panel()
        self._optimized_panel = self._build_optimized_panel()
        self._roles_panel = self._build_roles_panel()
        self._apply_panel = self._build_apply_panel()

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_setup_page())
        self._stack.addWidget(self._build_running_page())
        self._stack.addWidget(self._build_result_page())
        return self._stack

    # ------------------------------------------------------------------
    # Page construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_scroll_page(content: QWidget) -> QScrollArea:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        page.setWidget(content)
        return page

    def _build_setup_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)

        # --- Series overview: populated as soon as the context arrives. ---
        layout.addWidget(make_section_header("Series"))
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

        # --- Scope. ---
        layout.addWidget(make_section_header("Scope"))
        scope_intro = QLabel(
            "Choose which candidate families the wizard screens across the series. Start "
            "from a preset (or Auto, inferred from run metadata) and include/exclude "
            "individual components as needed."
        )
        scope_intro.setWordWrap(True)
        layout.addWidget(scope_intro)
        self._scope_selector = WizardScopeSelector()
        # A floor keeps the family tree usable inside the scrolling page.
        self._scope_selector.setMinimumHeight(260)
        self._scope_selector.scope_changed.connect(self._on_scope_changed)
        self._scope_selector.validity_changed.connect(self._on_scope_validity_changed)
        layout.addWidget(self._scope_selector)

        # --- Optional parameter expectations (embedded ex-dialog). ---
        layout.addWidget(self._build_expectations_section())

        # --- Search settings. ---
        layout.addWidget(make_section_header("Search settings"))
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Ranking Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        settings_row.addWidget(self._metric_combo)
        metric_info_btn = QPushButton("Metric Info")
        metric_info_btn.clicked.connect(self._show_metric_info)
        settings_row.addWidget(metric_info_btn)
        # Single honest optimisation mode. Every EffortTier now resolves to the
        # exact bounded-wavefront engine (see EffortTier / _EFFORT_TIER_SEARCH_ENGINE):
        # PR 2's exact bounds made it near-minimal and 12-way parallel, so the
        # former heuristic Low/Balanced tiers were empirically slower with no
        # fit-count headroom. A four-position slider where every position did the
        # same work would be misleading, so the visible control is a single
        # disabled item. The 1-item combo (rather than a bare QLabel) keeps
        # current_effort_tier()/_set_effort_tier() and the payload round-trip
        # working unchanged, so a future scope-based quick-look tier can be added
        # without reworking persistence.
        settings_row.addWidget(QLabel("Search:"))
        self._effort_combo = QComboBox()
        self._effort_combo.addItem(
            EFFORT_TIER_LABELS[EffortTier.EXHAUSTIVE], userData=EffortTier.EXHAUSTIVE.value
        )
        self._effort_combo.setCurrentIndex(0)
        self._effort_combo.setEnabled(False)
        self._effort_combo.setToolTip(EFFORT_TIER_DESCRIPTIONS[EffortTier.EXHAUSTIVE])
        settings_row.addWidget(self._effort_combo)
        warning_info_btn = QPushButton("Warning Info")
        warning_info_btn.clicked.connect(self._show_warning_info)
        settings_row.addWidget(warning_info_btn)
        settings_row.addStretch()
        layout.addLayout(settings_row)

        # --- Primary CTA. ---
        cta_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Run screening")
        self._refresh_btn.setStyleSheet(build_primary_button_qss())
        self._refresh_btn.clicked.connect(self._start_analysis)
        cta_row.addWidget(self._refresh_btn)
        cta_row.addStretch()
        layout.addLayout(cta_row)

        layout.addStretch()
        return self._make_scroll_page(content)

    def _build_expectations_section(self) -> CollapsibleSection:
        """Build the embedded parameter-expectations editor (ex modal dialog).

        Screening no longer blocks on a dialog: the table is populated from the
        candidate portfolio when the context arrives, and *Run screening* reads
        it in place (invalid bounds surface inline and stop the run).
        """
        self._expectations_section = CollapsibleSection(
            "Guide the search (optional)", expanded=False
        )
        hint = QLabel(
            "The wizard explores the candidate families below. Review the combined "
            "parameter list and set your expected Global/Local behaviour and bounds "
            "before the expensive search starts. Defaults: amplitudes start as Global "
            "with positive bounds, rate-like parameters start as Local with positive "
            "bounds, and background terms stay Global unless you change them."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._expectations_section.addWidget(hint)

        # Shown instead of the table when the portfolio cannot be built for the
        # current context (build failure / mixed series axes).
        self._expectations_warning_label = QLabel("")
        self._expectations_warning_label.setWordWrap(True)
        self._expectations_warning_label.setVisible(False)
        self._expectations_section.addWidget(self._expectations_warning_label)

        self._expectations_table = QTableWidget(0, 4)
        self._expectations_table.setHorizontalHeaderLabels(
            ["Parameter", "Expected Role", "Bounds", "Used By"]
        )
        self._expectations_table.horizontalHeader().setStretchLastSection(True)
        self._expectations_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._expectations_section.addWidget(self._expectations_table)

        self._expectations_error_label = QLabel("")
        self._expectations_error_label.setWordWrap(True)
        self._expectations_error_label.setStyleSheet(f"color: {tokens.ERROR};")
        self._expectations_error_label.setVisible(False)
        self._expectations_section.addWidget(self._expectations_error_label)
        return self._expectations_section

    def _build_running_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._running_header_label = QLabel("")
        header_font = QFont(self._running_header_label.font())
        header_font.setBold(True)
        self._running_header_label.setFont(header_font)
        layout.addWidget(self._running_header_label)
        self._running_trail = DecisionTrail()
        layout.addWidget(self._running_trail)
        self._log_section = CollapsibleSection("Live log", expanded=False)
        self._log_panel = LogPanel()
        self._log_panel.setMinimumHeight(180)
        self._log_section.addWidget(self._log_panel)
        layout.addWidget(self._log_section)
        layout.addStretch()
        return page

    def _build_result_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)

        # Series answer card at the top.
        self._series_card = WizardSeriesCard()
        self._series_card.apply_requested.connect(self._apply_recommended_fit)
        self._series_card.selection_changed.connect(self._on_card_selection_changed)
        layout.addWidget(self._series_card)

        # Screening shortlist: pick candidates for coupled optimisation.
        layout.addWidget(make_section_header("Screening shortlist"))
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
        # Bounded height: a one-line hint must not open a 200px blank box, and
        # the details never need more than a few lines (the page scrolls).
        self._compare_warning_text.setMinimumHeight(70)
        self._compare_warning_text.setMaximumHeight(160)
        layout.addWidget(self._compare_warning_text)
        optimize_row = QHBoxLayout()
        self._optimize_btn = QPushButton("Optimize selected")
        self._optimize_btn.setStyleSheet(build_primary_button_qss())
        self._optimize_btn.clicked.connect(self._start_selected_optimisation)
        optimize_row.addWidget(self._optimize_btn)
        optimize_row.addStretch()
        layout.addLayout(optimize_row)

        # Finished decision trail hosting the demoted detail tables. The panels
        # are injected once; set_steps rebuilds re-apply them by key.
        self._result_trail = DecisionTrail()
        layout.addWidget(self._result_trail)
        self._result_trail.set_step_detail_widget("portfolio", self._portfolio_panel)
        self._result_trail.set_step_detail_widget("optimized", self._optimized_panel)
        self._result_trail.set_step_detail_widget("roles", self._roles_panel)
        self._result_trail.set_step_detail_widget("apply", self._apply_panel)

        layout.addStretch()
        self._result_page = self._make_scroll_page(content)
        return self._result_page

    # ------------------------------------------------------------------
    # Deep panels (built once; injected into the result trail)
    # ------------------------------------------------------------------

    def _build_portfolio_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
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
        return panel

    def _build_optimized_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
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
        return panel

    def _build_roles_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
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
        return panel

    def _build_apply_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
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
        return panel

    # ------------------------------------------------------------------
    # External surface (unchanged contract)
    # ------------------------------------------------------------------

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
        """Prepare the window for a new ordered dataset series (→ Setup)."""
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
        self._reset_log()
        self._cached_signature = None
        self._analysis_request_id += 1
        # Install the scope resolver and reset the selector to Auto (signal-silent),
        # then refresh so is_valid() sees a populated tree before the final
        # _set_busy(False) below evaluates the button states.
        self._scope_selector.set_resolver(self._resolve_scope)
        self._scope_selector.set_scope(None)
        self._scope_selector.refresh_from_context()
        self._populate_expectations_from_context()
        self._stack.setCurrentIndex(_PAGE_SETUP)
        run_label_chips = [dataset.run_label for dataset in self._datasets[:4]]
        if len(self._datasets) > 4:
            run_label_chips.append("…")
        run_labels = ", ".join(run_label_chips)
        self._heading_label.setText("Global Fit Wizard")
        self._status_label.setToolTip("")
        self.set_context_chips([f"{len(self._datasets)} datasets", *run_label_chips])
        self._status_label.setText(
            f"Ready to analyze the selected series ({run_labels}). "
            "Review the candidate portfolio, then run the screening before choosing which candidates to optimize globally."
        )
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(SelectionMetric.AICC.value)
        self._metric_combo.blockSignals(False)
        self._set_empty_state()
        # Run / Field / Temperature are known now, so show the series immediately
        # rather than an empty table until screening. The classification columns
        # stay "—" until a recommendation is built (see _populate_overview_table).
        self._populate_series_preview()
        self._set_busy(False)

    def _populate_series_preview(self) -> None:
        """List the loaded runs in the SERIES section before any screening.

        Shared by ``set_analysis_context`` and the screening-failure path: the
        series stays loaded either way, so the Setup page must keep listing the
        runs rather than showing a blank table.
        """
        self._populate_overview_table()
        if self._datasets:
            self._overview_banner.setText(
                f"{len(self._datasets)} runs selected. "
                "Run screening to classify each run (Osc. / KT-like / Multi-rate)."
            )

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
        # scope, so clear it before disabling "Optimize selected" via _set_busy.
        self._screening_selected_keys = set()
        self._mark_analysis_stale("Scope changed")

    def _on_scope_validity_changed(self, is_valid: bool) -> None:
        if not is_valid and not self._analysis_in_progress:
            self._status_label.setText(
                "Select at least one candidate family in the Scope section to enable screening."
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

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _show_running(self) -> None:
        """Enter the Running state: stream trail placeholders for the mode."""
        if self._analysis_mode == "optimize":
            self._running_header_label.setText("Optimizing selected candidates…")
            self._running_trail.stream_placeholders(_optimize_placeholder_steps())
            self._running_trail.set_status("Preparing selected candidates…")
        else:
            self._running_header_label.setText("Screening the series…")
            self._running_trail.stream_placeholders(_screening_placeholder_steps())
            self._running_trail.set_status("Reading series conditions…")
        self._stack.setCurrentIndex(_PAGE_RUNNING)

    def _update_action_enablement(self, busy: bool) -> None:
        self._progress_label.setText("Working..." if busy else "")
        self._cancel_btn.setVisible(busy)
        self._refresh_btn.setEnabled(
            bool(self._datasets) and not busy and self._scope_selector.is_valid()
        )
        self._metric_combo.setEnabled(self._recommendation is not None and not busy)
        selected_count = len(self._screening_selected_keys)
        self._optimize_btn.setText(
            f"Optimize selected ({selected_count})" if selected_count else "Optimize selected"
        )
        self._optimize_btn.setEnabled(
            self._recommendation is not None
            and selected_count > 0
            and not busy
            and not self._analysis_stale
        )
        # A run that ends without a populate transition (cancel, stale-orphan,
        # failure) must not leave the window parked on the Running page: land on
        # Result when a recommendation is still shown, else back on Setup.
        if (
            not busy
            and not self._analysis_in_progress
            and self._stack.currentIndex() == _PAGE_RUNNING
        ):
            self._stack.setCurrentIndex(
                _PAGE_RESULT if self._recommendation is not None else _PAGE_SETUP
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
        self._series_card.clear()
        self._series_card.set_apply_enabled(False)
        self._result_trail.set_steps(())
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
                "Select at least one candidate family in the Scope section to enable screening."
            )
            return

        # Read the embedded expectations table (the ex-dialog). Invalid bounds
        # stop the run and surface inline instead of via a modal warning.
        try:
            setup_config = self._read_expectations_configuration()
        except ValueError as exc:
            self._expectations_error_label.setText(f"Invalid bounds — {exc}")
            self._expectations_error_label.setVisible(True)
            self._expectations_section.setExpanded(True)
            self._status_label.setText(f"Fix the parameter expectations before screening: {exc}")
            return
        self._expectations_error_label.setText("")
        self._expectations_error_label.setVisible(False)
        if setup_config is not None:
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
        self._reset_log()
        self._status_label.setText(
            "Building the single-fit screening table in the background. "
            "The main window stays responsive while the shared candidate portfolio is screened."
        )
        self._append_log(f"Starting screening for {len(self._datasets)} datasets.")
        self._show_running()
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
        self._reset_log()
        self._status_label.setText(
            "Running coupled global optimisation for the selected candidates. "
            "Progress is streamed to the live log."
        )
        self._append_log(
            "Starting coupled global optimisation for: " + ", ".join(selected_titles) + "."
        )
        self._populate_compare_table()
        self._show_running()
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
        # guard; _analysis_mode was stashed when the run started. A failed
        # screening leaves nothing to show (→ Setup); a failed optimize keeps the
        # existing screening recommendation on the Result page.
        self._running_template_keys = set()
        if self._analysis_mode == "screening":
            self._recommendation = None
        # Keep the header's status line to the failure's first line — a
        # multi-line exception message (e.g. a multiprocessing bootstrap error)
        # would otherwise balloon the header band. The full text stays in the
        # log and in the status line's tooltip.
        failure_text = str(message).strip() or "unknown error"
        self._status_label.setText(
            f"Global fit wizard analysis failed: {failure_text.splitlines()[0]}"
        )
        self._status_label.setToolTip(failure_text)
        self._append_log(f"Analysis failed: {message}")
        if self._recommendation is None:
            self._set_empty_state()
            self._populate_series_preview()
            self._stack.setCurrentIndex(_PAGE_SETUP)
        else:
            self._populate_from_recommendation()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        # Base already guarded the request id; stream to the live log and the
        # running trail (prefix table per mode; unmatched → status line only).
        text = (message or "").strip()
        if text:
            self._append_log(text)
        lowered = text.lower()
        prefixes = (
            _OPTIMIZE_PROGRESS_PREFIXES
            if self._analysis_mode == "optimize"
            else _SCREENING_PROGRESS_PREFIXES
        )
        matched = next(
            (key for prefix, key in prefixes if lowered.startswith(prefix)),
            None,
        )
        if matched is not None:
            self._running_trail.activate_step(matched)
        if text:
            self._running_trail.set_status(text)

    # ------------------------------------------------------------------
    # Inline log (formerly a separate log window)
    # ------------------------------------------------------------------

    def _reset_log(self) -> None:
        """Clear the inline log at run start (as the old log window used to)."""
        self._log_panel.clear()
        self._cached_log_text = ""

    def _append_log(self, message: str) -> None:
        self._log_panel.log(message)
        self._cached_log_text = "\n".join(filter(None, [self._cached_log_text, message]))

    # ------------------------------------------------------------------
    # Cached restore / signature (unchanged contract)
    # ------------------------------------------------------------------

    def set_cached_recommendation(
        self,
        recommendation: GlobalFitWizardRecommendation,
        *,
        signature: dict[str, object] | None = None,
        log_text: str = "",
        status_text: str | None = None,
    ) -> None:
        """Populate the window from an already-computed recommendation (→ Result)."""
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
        # The effort tier is retained in the payload for forward-compatibility,
        # but every tier now runs the exact engine and the visible control is a
        # single "Optimize" mode. Restoring a legacy Low/Balanced payload is a
        # no-op on the one-item control (it stays on the exact mode), which is the
        # correct behaviour since all tiers resolve to the same exact search.
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
        """The effort tier the wizard will run.

        The visible control is a single disabled "Optimize" item, so this always
        returns the exact tier. The method (and the payload it feeds) is retained
        so a future scope-based quick-look tier can be surfaced without reworking
        persistence.
        """
        return effort_tier_from_payload(self._effort_combo.currentData())

    def _set_effort_tier(self, tier: EffortTier) -> None:
        # The one-item control only carries the exact (Optimize) tier; a legacy
        # Low/Balanced payload finds no matching item and is left on the exact
        # mode, which is correct now that every tier runs the exact engine.
        index = self._effort_combo.findData(tier.value)
        if index < 0:
            return
        self._effort_combo.blockSignals(True)
        self._effort_combo.setCurrentIndex(index)
        self._effort_combo.blockSignals(False)

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

    # ------------------------------------------------------------------
    # Parameter expectations (embedded ex-dialog)
    # ------------------------------------------------------------------

    def _set_expectations_warning(self, text: str) -> None:
        """Show ``text`` instead of the expectations table (empty text restores it)."""
        self._expectations_warning_label.setText(text)
        self._expectations_warning_label.setVisible(bool(text))
        self._expectations_table.setVisible(not text)

    def _populate_expectations_from_context(self) -> None:
        """Rebuild the expectations table from the current context's portfolio.

        A portfolio failure (or a mixed-axes series) leaves the table empty and
        shows the reason inline; screening then proceeds without a parameter
        setup, exactly as the old dialog-skipping branch did.
        """
        self._expectations_error_label.setText("")
        self._expectations_error_label.setVisible(False)
        self._expectation_parameter_names = []
        self._expectations_table.setRowCount(0)
        if not self._datasets:
            self._set_expectations_warning("")
            return
        try:
            portfolio = build_global_fit_wizard_candidate_portfolio(
                self._datasets,
                current_model=self._current_model,
                scope=WizardScope.from_payload(copy.deepcopy(self._scope_selector.current_scope())),
            )
        except Exception as exc:
            self._set_expectations_warning(f"Global fit wizard setup failed: {exc}")
            return
        if portfolio.mixed_axes_warning:
            self._set_expectations_warning(portfolio.mixed_axes_warning)
            return
        if not portfolio.templates:
            self._set_expectations_warning("No candidate families are in scope for this series.")
            return
        self._set_expectations_warning("")

        names, usage_by_name = _portfolio_parameter_usage(portfolio.templates)
        self._expectation_parameter_names = names
        self._expectations_table.setRowCount(len(names))
        for row, name in enumerate(names):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._expectations_table.setItem(row, 0, name_item)

            role_combo = QComboBox()
            role_combo.addItems(["Global", "Local", "Fixed"])
            role_combo.setCurrentText(
                _default_parameter_role(name, current_parameter_types=self._current_parameter_types)
            )
            self._expectations_table.setCellWidget(row, 1, role_combo)

            bounds_item = QTableWidgetItem(
                _format_bounds_text(
                    _default_parameter_bounds(name, current_parameter_bounds=self._parameter_bounds)
                )
            )
            self._expectations_table.setItem(row, 2, bounds_item)

            usage_titles = usage_by_name[name]
            usage_item = QTableWidgetItem(
                ", ".join(usage_titles[:3]) + (", ..." if len(usage_titles) > 3 else "")
            )
            usage_item.setToolTip("\n".join(usage_titles))
            usage_item.setFlags(usage_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._expectations_table.setItem(row, 3, usage_item)

    def _read_expectations_configuration(self) -> dict[str, object] | None:
        """Read the expectations table into a parameter-setup config.

        Returns ``None`` when the table is unpopulated (portfolio failure /
        mixed axes), matching the old dialog-skipping branch. Raises
        ``ValueError`` naming the offending parameter on unparseable bounds.
        """
        if not self._expectation_parameter_names:
            return None
        types: dict[str, str] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for row, name in enumerate(self._expectation_parameter_names):
            role_combo = self._expectations_table.cellWidget(row, 1)
            role = role_combo.currentText() if isinstance(role_combo, QComboBox) else "Global"
            bounds_item = self._expectations_table.item(row, 2)
            try:
                min_val, max_val = _parse_bounds_text(
                    bounds_item.text() if bounds_item else "-inf, inf"
                )
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            types[name] = role
            bounds[name] = (min_val, max_val)
        return {"types": types, "bounds": bounds}

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

    # ------------------------------------------------------------------
    # Result-state population
    # ------------------------------------------------------------------

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
                "to launch coupled optimisation and follow progress in the live log."
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
        self._populate_series_card()
        self._populate_result_trail()
        self._stack.setCurrentIndex(_PAGE_RESULT)
        # Land at the top of the result page: a prior scroll position (or focus
        # handoff from the shortlist's Optimize button) would otherwise open the
        # page mid-card, cutting the verdict off above the viewport.
        self._result_page.verticalScrollBar().setValue(0)

    def _populate_result_trail(self) -> None:
        """Re-derive the finished trail's headlines from the recommendation.

        The step keys are stable, so the deep panels injected once in
        ``_build_result_page`` survive every rebuild; only the mechanical
        count-based headlines change.
        """
        recommendation = self._recommendation
        if recommendation is None:
            self._result_trail.set_steps(())
            return
        optimized_count = len(recommendation.sorted_optimized_assessments())
        self._result_trail.set_steps(
            (
                TrailStep(
                    "portfolio",
                    f"Candidate portfolio — {len(recommendation.templates)} families considered",
                    "portfolio",
                    (),
                ),
                TrailStep(
                    "optimized",
                    f"Global optimized fits — {optimized_count} converged",
                    "optimized",
                    (),
                ),
                TrailStep("roles", "Parameter sharing diagnostics", "roles", ()),
                TrailStep("apply", "Apply to the fit panel", "apply", ()),
            )
        )

    # --- Series answer card adapter -----------------------------------

    @staticmethod
    def _series_axis_value(dataset: MuonDataset | None, axis_key: str | None) -> float | None:
        """The run's position along the series axis, or None when unavailable."""
        if dataset is None or not axis_key:
            return None
        try:
            value = float(dataset.metadata.get(axis_key))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _series_run_traces(
        self,
        recommendation: GlobalFitWizardRecommendation,
        assessment: GlobalCandidateAssessment | None,
    ) -> list[SeriesRunTrace]:
        """One trace per run in series order; fit overlays from ``assessment``.

        None-safe by design: a screening-only selection (no optimized
        assessment) or a run missing from ``fitted_curves_by_run`` still draws
        its data points, just without a fit line.
        """
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        curves = assessment.fitted_curves_by_run if assessment is not None else {}
        traces: list[SeriesRunTrace] = []
        for run_number in recommendation.dataset_order:
            run_number = int(run_number)
            dataset = by_run.get(run_number)
            if dataset is None:
                continue
            entry = curves.get(run_number)
            fitted_time = fitted_curve = None
            if isinstance(entry, tuple | list) and len(entry) == 2:
                fitted_time = np.asarray(entry[0], dtype=float)
                fitted_curve = np.asarray(entry[1], dtype=float)
            traces.append(
                SeriesRunTrace(
                    run_label=dataset.run_label,
                    axis_value=self._series_axis_value(dataset, recommendation.series_axis_key),
                    time=np.asarray(dataset.time, dtype=float),
                    asymmetry=np.asarray(dataset.asymmetry, dtype=float),
                    error=np.asarray(dataset.error, dtype=float),
                    fitted_time=fitted_time,
                    fitted_curve=fitted_curve,
                )
            )
        return traces

    def _series_trend(
        self,
        recommendation: GlobalFitWizardRecommendation,
        assessment: GlobalCandidateAssessment | None,
    ) -> SeriesTrend | None:
        """The first local parameter's fitted values across the series.

        Only an optimized assessment carries per-run fitted values worth
        trending; any missing piece (run fit, parameter, axis value) yields
        ``None`` rather than a partially honest trend.
        """
        if assessment is None or assessment.prescreen_only or not assessment.local_param_names:
            return None
        name = assessment.local_param_names[0]
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        axis_values: list[float] = []
        values: list[float] = []
        errors: list[float | None] = []
        for run_number in recommendation.dataset_order:
            run_number = int(run_number)
            dataset = by_run.get(run_number)
            fit_result = assessment.fit_results_by_run.get(run_number)
            axis_value = self._series_axis_value(dataset, recommendation.series_axis_key)
            if fit_result is None or axis_value is None:
                return None
            try:
                value = float(fit_result.parameters[name].value)
            except (KeyError, TypeError, ValueError):
                return None
            if not np.isfinite(value):
                return None
            axis_values.append(axis_value)
            values.append(value)
            uncertainty = fit_result.uncertainties.get(name)
            errors.append(float(uncertainty) if uncertainty is not None else None)
        if len(values) < 2:
            return None
        return SeriesTrend(
            parameter_label=get_param_info(name).unicode_label(),
            axis_label=recommendation.series_axis_label,
            axis_values=tuple(axis_values),
            values=tuple(values),
            errors=(
                tuple(errors)  # type: ignore[arg-type]
                if all(error is not None for error in errors)
                else None
            ),
        )

    def _populate_series_card(self) -> None:
        recommendation = self._recommendation
        if recommendation is None:
            self._series_card.clear()
            return
        assessment = self._selected_assessment()
        self._series_card.set_series(
            self._series_run_traces(recommendation, assessment),
            recommendation.series_axis_label,
        )
        self._series_card.set_trend(self._series_trend(recommendation, assessment))
        recommended = recommendation.recommended_assessment
        # The recommendation carries no confidence tier, so no chip is shown
        # (tier=None) — the summary line carries the confidence prose instead.
        # Before any optimisation there is no recommended assessment; lead with
        # the top screening candidate as a plain fact rather than repeating the
        # summary as both headline and prose.
        if recommended is not None:
            self._series_card.set_verdict(recommended.template.title, recommendation.summary, None)
        else:
            prescreen = recommendation.sorted_prescreen_assessments()
            if prescreen:
                self._series_card.set_verdict(
                    f"Leading candidate: {prescreen[0].template.title}",
                    recommendation.summary,
                    None,
                )
            else:
                self._series_card.set_verdict(recommendation.summary, "", None)
        candidates = []
        for candidate in recommendation.sorted_optimized_assessments():
            if (
                recommendation.recommended_key is not None
                and candidate.selection_key == recommendation.recommended_key
            ):
                continue
            candidates.append(candidate)
            if len(candidates) == 3:
                break
        # Several optimized assignments of the SAME template differ only in
        # their Global/Local split, so a bare title cannot tell them apart —
        # append the local-parameter signature whenever the title collides with
        # the recommendation or another alternative.
        titles = [candidate.template.title for candidate in candidates]
        if recommended is not None:
            titles.append(recommended.template.title)
        alternatives: list[tuple[str, str, str]] = []
        for candidate in candidates:
            label = candidate.template.title
            if titles.count(label) > 1:
                label = f"{label} · local: {', '.join(candidate.local_param_names) or 'none'}"
            tooltip = (
                f"Global: {', '.join(candidate.global_param_names) or 'None'}\n"
                f"Local: {', '.join(candidate.local_param_names) or 'None'}"
            )
            alternatives.append((candidate.selection_key, label, tooltip))
        self._series_card.set_alternatives(alternatives)
        self._series_card.set_selected_key(self._selected_key)

    def _on_card_selection_changed(self, key: str) -> None:
        """Route a card alternative pick through the optimized-selection path."""
        if not isinstance(key, str):
            return
        self._selected_key = key
        self._select_row_for_key(self._optimized_table, key)
        self._update_compare_warning_text()
        self._update_roles_table()
        self._update_apply_page()

    # --- Detail tables --------------------------------------------------

    def _populate_overview_table(self) -> None:
        """List one row per selected run in the Series overview.

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
            # Keep the answer card's alternatives strip in step (no-op when
            # already selected, so table↔card sync converges).
            self._series_card.set_selected_key(key)
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
                    "The optimized-fits step below will list each converged global/local assignment after global fitting completes."
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
            self._series_card.set_apply_enabled(False)
            return

        if assessment is None:
            self._apply_selection_label.setText("No globally optimized candidate is selected yet.")
            self._apply_text.setPlainText(
                "Choose one or more rows from the screening table and run coupled global optimisation first."
            )
            self._apply_recommended_btn.setEnabled(False)
            self._apply_selected_btn.setEnabled(False)
            self._series_card.set_apply_enabled(False)
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
        # The card's Apply mirrors "apply recommended" exactly.
        self._series_card.set_apply_enabled(recommended is not None)
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
                "Warnings in the optimized-results step combine per-run residual checks with ordered-series "
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


def _screening_placeholder_steps() -> tuple[TrailStep, ...]:
    """Pending trail headlines for a screening run (before results are known)."""
    return (
        TrailStep("context", "Reading series conditions…", "context", ()),
        TrailStep("portfolio", "Choosing candidate families…", "portfolio", ()),
        TrailStep("screening", "Screening each run independently…", "screening", ()),
        TrailStep("ranking", "Ranking candidates across the series…", "ranking", ()),
    )


def _optimize_placeholder_steps() -> tuple[TrailStep, ...]:
    """Pending trail headlines for a coupled-optimisation run."""
    return (
        TrailStep("prepare", "Preparing selected candidates…", "prepare", ()),
        TrailStep("optimize", "Running coupled global optimisation…", "optimize", ()),
        TrailStep("roles", "Scoring Global/Local parameter roles…", "roles", ()),
        TrailStep("ranking", "Ranking optimized fits…", "ranking", ()),
    )


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
