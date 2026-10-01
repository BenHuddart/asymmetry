"""Non-modal guided fit wizard for ordered global-fit dataset series.

A :class:`WizardStepper` runs across the top of a ``QStackedWidget`` of five
step views, Scope → Screen → Compare → Phases → Apply (design:
``docs/plans/global-wizard-stepper.md``). The window is built on
:class:`~asymmetry.gui.windows.wizard_base.WizardWindowBase` through the
``_build_central()`` hook, so ``self._tabs`` stays ``None``.

* **Scope** — the series overview, the model family picker, *Guide the search
  (optional)*, the search settings and *Run screening*.
* **Screen** — the family leaderboard, ticked for the shortlist, beside a
  preview of the selected family's per-run fits.
* **Compare** — the optimised role splits, A against B.
* **Phases** — the transitions penalty path and a fit overlay coloured by phase.
* **Apply** — what will be handed to the global fit tab, and the apply action.

A running analysis shows its :class:`RunProgress` inside the step it feeds
(D2). The external surface (``set_analysis_context``,
``set_cached_recommendation``, ``current_recommendation``, the signals and the
seven-key ``_analysis_signature``) is unchanged for ``global_tab.py``.
"""

from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass, replace
from functools import partial

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
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import FieldGeometry, ParameterKind
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
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
    format_transition_boundaries,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
    transitions_summary,
)
from asymmetry.core.fitting.global_search.partition import PartitionPath
from asymmetry.core.fitting.global_search.sharing_ladder import ADEQUACY_SIGMA
from asymmetry.core.fitting.global_search.trend_objective import (
    RungSummary,
    SelectionObjective,
    cost_text,
    left_out_note,
    summarise_rungs,
)
from asymmetry.core.fitting.model_comparison import shortlist, summarise_candidates
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fitting.wizard_narrative import TrailStep
from asymmetry.core.fitting.wizard_scope import (
    DEFAULT_EFFORT_TIER,
    EFFORT_TIER_DESCRIPTIONS,
    EFFORT_TIER_LABELS,
    EffortTier,
    FitTimeEstimates,
    WizardScope,
    describe_scope,
    effort_tier_from_payload,
    set_user_field_direction,
)
from asymmetry.gui.styles import metrics, tokens
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    clear_layout,
    make_section_header,
    make_warning_banner,
)
from asymmetry.gui.utils.fit_times import record_fit_times, shared_fit_time_store
from asymmetry.gui.utils.formatting import format_param_label
from asymmetry.gui.utils.phase_colors import (
    EXCLUDED_PHASE_HATCH_COLOR,
    format_axis_range,
    phase_color,
)
from asymmetry.gui.widgets.elided_label import ElidedLabel
from asymmetry.gui.widgets.key_value_grid import KeyValueGrid
from asymmetry.gui.widgets.model_compare_panel import ModelComparePanel, RungRow
from asymmetry.gui.widgets.model_family_picker import ModelFamilyPicker
from asymmetry.gui.widgets.panel_section import PanelSection
from asymmetry.gui.widgets.parameter_trace_strip import ParameterTraceStrip, TracedRung, rung_traces
from asymmetry.gui.widgets.run_progress import RunProgress
from asymmetry.gui.widgets.screen_sizing import resize_to_available
from asymmetry.gui.widgets.screening_leaderboard import OptimisationStatus, ScreeningLeaderboard
from asymmetry.gui.widgets.series_fit_canvas import SeriesFitCanvas
from asymmetry.gui.widgets.transitions_card import (
    PhaseSummary,
    TransitionRow,
    TransitionsCard,
)
from asymmetry.gui.widgets.wizard_stepper import StepState, WizardStepper
from asymmetry.gui.windows.wizard_base import WizardWindowBase

_DEFAULT_PHASE_ONE_SINGLE_FIT_HELPER = (
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio
)
_DEFAULT_SCREENING_BUILDER = build_global_fit_wizard_screening_recommendation
_DEFAULT_GLOBAL_FIT_BUILDER = build_global_fit_wizard_recommendation

#: The model family picker's height floor on the scrolling Scope view, in table rows.
_PICKER_MIN_ROWS = 18

#: The steps, in journey order: key and title.
_STEPS = (
    ("scope", "Scope"),
    ("screen", "Screen"),
    ("compare", "Compare"),
    ("phases", "Phases"),
    ("apply", "Apply"),
)

#: The objective switch (plan D1 of the trend objective): label and one-line hint, default first.
_OBJECTIVES: dict[SelectionObjective, tuple[str, str]] = {
    SelectionObjective.TREND: (
        "Best for trending",
        "Recommends the sharing pattern whose per-run parameters vary most cleanly along "
        "the series, among the models that fit adequately.",
    ),
    SelectionObjective.STATISTICAL: (
        "Best statistical fit",
        "Recommends the model and sharing pattern with the best information criterion.",
    ),
}

#: Analysis modes whose result *adds to* the standing screening recommendation
#: rather than replacing it (so the shortlist and the path survive the run).
_MERGING_MODES = ("optimize", "optimize_phases")

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

_OPTIMIZE_PHASES_PROGRESS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("preparing consolidated", "prepare"),
    ("preparing per-dataset single-fit", "prepare"),
    ("preparing missing single-fit", "prepare"),
    ("running phase-1 single-fit", "prepare"),
    ("single-fit table", "prepare"),
    ("using completed per-run single-fit", "prepare"),
    ("series alphabet", "prepare"),
    ("optimising", "phases"),
    ("phase ", "phases"),
    ("separable role search", "phases"),
)

#: The core announces the per-phase run as "Optimising N distinct phase(s) …"
#: before it reports any individual phase, so the window can count "phase i of
#: N" off the two messages without knowing how the core chose its N (it fits the
#: selected solution's phases *and* the neighbours it verifies).
_PHASE_TOTAL_ANNOUNCEMENT = re.compile(r"^optimising (\d+) distinct phase")
_PHASE_STEP_PREFIX = "phase "


@dataclass(frozen=True)
class _RunMode:
    """How one analysis mode shows its run, in the step whose view hosts it (D2)."""

    step: str
    header: str
    #: The hosting step's stepper summary while the run goes.
    step_summary: str
    placeholders: tuple[TrailStep, ...]
    opening_status: str
    #: Progress-message prefix → trail step key.
    prefixes: tuple[tuple[str, str], ...]


_RUN_MODES: dict[str, _RunMode] = {
    "screening": _RunMode(
        step="screen",
        header="Screening the series…",
        step_summary="Screening…",
        placeholders=(
            TrailStep("context", "Reading series conditions…", "context", ()),
            TrailStep("portfolio", "Choosing candidate families…", "portfolio", ()),
            TrailStep("screening", "Screening each run independently…", "screening", ()),
            TrailStep("ranking", "Ranking candidates across the series…", "ranking", ()),
        ),
        opening_status="Reading series conditions…",
        prefixes=_SCREENING_PROGRESS_PREFIXES,
    ),
    "optimize": _RunMode(
        step="compare",
        header="Optimizing selected candidates…",
        step_summary="Optimising the shortlist…",
        placeholders=(
            TrailStep("prepare", "Preparing selected candidates…", "prepare", ()),
            TrailStep("optimize", "Running coupled global optimisation…", "optimize", ()),
            TrailStep("roles", "Scoring Global/Local parameter roles…", "roles", ()),
            TrailStep("ranking", "Ranking optimized fits…", "ranking", ()),
        ),
        opening_status="Preparing selected candidates…",
        prefixes=_OPTIMIZE_PROGRESS_PREFIXES,
    ),
    # The "phases" headline is rewritten to "Optimising phase i of N…" as the
    # core reports each phase, so a long run says how far through it is.
    "optimize_phases": _RunMode(
        step="phases",
        header="Optimizing each phase…",
        step_summary="Optimising phases…",
        placeholders=(
            TrailStep("prepare", "Preparing the series screening table…", "prepare", ()),
            TrailStep("phases", "Optimising each phase…", "phases", ()),
        ),
        opening_status="Preparing the series screening table…",
        prefixes=_OPTIMIZE_PHASES_PROGRESS_PREFIXES,
    ),
}

#: ``optimization_status_for_key`` words for a listed family, as leaderboard badges.
_OPTIMISATION_STATUSES = {
    "Not optimized": OptimisationStatus.NOT_OPTIMISED,
    "Optimized": OptimisationStatus.OPTIMISED,
    "Optimization failed": OptimisationStatus.FAILED,
}


@dataclass(frozen=True)
class _ModelTarget:
    """Apply hands optimised candidate ``key`` (Compare's A) to the global fit tab."""

    key: str


@dataclass(frozen=True)
class _PhasesTarget:
    """Apply hands the phases of path row ``partition_k`` to the project."""

    partition_k: int


_ApplyTarget = _ModelTarget | _PhasesTarget


@dataclass
class _GlobalAnalysisResult:
    """Plain result object returned by the global-fit-wizard analysis task.

    Folds the old ``GlobalFitWizardWorker`` signals into one value: the analysis
    ``mode`` (screening/optimize) and any ``updated_single_fit_recommendations``
    (formerly the one-shot ``single_fit_precomputed`` signal) ride alongside the
    ``recommendation`` so the base's single ``finished`` path can apply them.
    ``fitted_assessments`` are the screening fits this run made, for the
    per-machine fit-time store.
    """

    mode: str
    recommendation: object
    updated_single_fit_recommendations: dict[int, object]
    fitted_assessments: tuple[CandidateAssessment, ...]


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
    objective: SelectionObjective,
    selected_template_keys: tuple[str, ...] = (),
    scope: WizardScope = WizardScope(),
    fit_times: FitTimeEstimates,
    effort_tier: EffortTier = DEFAULT_EFFORT_TIER,
    partition_path: PartitionPath | None = None,
    partition_k: int | None = None,
) -> _GlobalAnalysisResult:
    """Run the global-fit wizard analysis off the GUI thread.

    Moved from the former ``GlobalFitWizardWorker.run`` body; exceptions now
    propagate to ``TaskWorker.run`` (→ the base error slot) instead of being
    caught and re-emitted, and progress goes through ``worker.progress.emit``.
    Cooperative cancel is honoured between builder phases: the base passes a
    ``FitCancelledError`` in ``_cancel_exceptions()`` so ``TaskWorker`` reports
    it as a cancellation rather than a failure. ``scope`` is the picker's
    frozen :class:`WizardScope`, forwarded to every builder with the
    ``fit_times`` judgement the picker described it by.
    ``effort_tier`` is the user-facing effort slider (PR 5); it only affects the
    coupled-optimisation builder (``mode == "optimize"``) — the independent
    per-run screening pass has no tier concept. ``objective`` is what that
    builder searches and ranks for; screening is the same under either.

    ``partition_path``/``partition_k`` carry the *Transitions* pick of the
    ``"optimize_phases"`` mode: the same coupled-optimisation builder then runs
    the separable role search **per phase** of that path solution (and of the
    neighbours it verifies) instead of once across the whole series. They travel
    together — the core refuses one without the other — and both stay ``None``
    for every other mode.
    """

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
            or (mode in _MERGING_MODES and optimization_builder_is_custom)
        )
        and build_or_complete_single_fit_wizard_recommendations_for_global_portfolio
        is _DEFAULT_PHASE_ONE_SINGLE_FIT_HELPER
        and not existing
    )
    _raise_if_cancelled()
    portfolio = None
    fitted_assessments: tuple[CandidateAssessment, ...] = ()
    if skip_implicit_phase_one:
        single_fit_recommendations_by_run = dict(existing)
    else:
        screening_table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
            datasets,
            current_model=current_model,
            existing_recommendations_by_run=existing,
            progress_callback=lambda message: worker.progress.emit(0, 0, message),
            scope=scope,
            fit_times=fit_times,
            cancel_callback=worker.is_cancelled,
        )
        # ``existing`` now holds the runs' own single-run Fit Wizard analyses
        # (what the fit tabs cache and what a later series analysis reuses); the
        # completed table is the rectangular per-run score table the builders
        # consume, and it travels with the portfolio it was completed against.
        portfolio = screening_table.portfolio
        single_fit_recommendations_by_run = screening_table.recommendations_by_run
        fitted_assessments = screening_table.fitted_assessments

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
            scope=scope,
            fit_times=fit_times,
            portfolio=portfolio,
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
            scope=scope,
            fit_times=fit_times,
            effort_tier=effort_tier,
            portfolio=portfolio,
            cancel_callback=worker.is_cancelled,
            partition_path=partition_path,
            partition_k=partition_k,
            objective=objective,
        )
    updated_single_fit_recommendations = {
        int(run_number): rec
        for run_number, rec in existing.items()
        if single_fit_recommendations_before_analysis.get(int(run_number)) is not rec
    }
    return _GlobalAnalysisResult(
        mode=mode,
        recommendation=recommendation,
        updated_single_fit_recommendations=updated_single_fit_recommendations,
        fitted_assessments=fitted_assessments,
    )


class GlobalFitWizardWindow(WizardWindowBase):
    """Present a guided workflow for global-fit model recommendation."""

    apply_assessment_requested = Signal(object, object)
    #: ``(recommendation, partition_k)`` — apply the optimised partition as
    #: phase data groups, one global-fit series per phase.
    apply_phases_requested = Signal(object, int)
    analysis_cached = Signal(object, str, object)
    parameter_setup_applied = Signal(object)
    single_fit_recommendations_generated = Signal(object)
    #: ``(run numbers, FieldGeometry | None)`` — the user answered the field
    #: direction for the runs whose files record none.
    field_direction_answered = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        # WizardWindowBase.__init__ builds the shared frame and calls
        # _build_central() before this body resumes.
        super().__init__(parent)
        self.setWindowTitle("Global Fit Wizard")
        # Cap the default to the available screen so the title bar never opens
        # clipped above the menu bar on a 13-inch laptop; the step views scroll
        # so the spacious preferred size applies only when the display fits it
        # (P1-5).
        resize_to_available(self, 1180, 740)

        self._heading_label.setText("Global Fit Wizard")
        self._status_label.setText(
            "Open the global fit wizard on a field or temperature series "
            "to compare common model families and recommended "
            "Global/Local parameter roles."
        )
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Content region (overrides the base tab scaffolding)
    # ------------------------------------------------------------------

    def _build_central(self) -> QWidget:
        """Build the stepper and the five step views.

        Runs during ``WizardWindowBase.__init__``, before this subclass body
        resumes, so it initialises the analysis state too.
        """
        self._datasets: list[MuonDataset] = []
        self._current_model: CompositeModel | None = None
        self._current_parameter_types: dict[str, str] = {}
        self._current_values: dict[str, float] = {}
        self._parameter_bounds: dict[str, tuple[float, float]] = {}
        self._recommendation: GlobalFitWizardRecommendation | None = None
        # Families whose coupled optimisation is running, from the ticks at its start.
        self._running_template_keys: set[str] = set()
        self._analysis_mode = "screening"
        # The step a run was started from, where a cancelled or failed run lands.
        self._run_origin = "scope"
        self._single_fit_recommendations_by_run: dict[int, object] = {}
        # Transitions state. ``_partition_k`` indexes the recommendation's
        # penalty path; ``None`` is the honest answer for a series that has no
        # path at all (fewer than two minimum-length phases fit in it).
        self._partition_k: int | None = None
        # A phase picked in the per-phase strip; ``None`` draws the row's phases without fits.
        self._selected_phase_segment: int | None = None
        self._phase_progress_seen = 0
        self._phase_progress_total = 0
        # What Continue (Compare) or Apply phases (Phases) chose, and what was applied.
        self._apply_target: _ApplyTarget | None = None
        self._applied_target: _ApplyTarget | None = None
        # A Scope edit invalidates the shown results; screening must be re-run.
        self._analysis_stale = False
        # Compare's rungs by key under the trend objective; empty under the statistical one.
        self._rung_summaries: dict[str, RungSummary] = {}
        # The selected path row's optimised phases as rungs, in phase order; likewise.
        self._phase_rung_summaries: tuple[RungSummary, ...] = ()
        # Row order of the embedded expectations table; empty when the table
        # could not be populated (portfolio failure / mixed axes / no context).
        self._expectation_parameter_names: list[str] = []

        # The base controls row holds the busy label and bar; Cancel lives in
        # the running step's progress block.
        self._controls_row.addStretch()

        # Stale banner sits above the stepper. Shown after a Scope edit
        # invalidates the shown results.
        self._stale_banner = make_warning_banner(
            "Scope changed since the last analysis, so these results are stale. "
            "Run screening again to refresh them."
        )
        self._stale_banner.setVisible(False)
        self._central_layout.addWidget(self._stale_banner)
        # Shown while the optimised fits answer the objective the user has since left.
        self._objective_banner = make_warning_banner()
        self._objective_banner.setVisible(False)
        self._central_layout.addWidget(self._objective_banner)

        self._stepper = WizardStepper(_STEPS)
        self._stepper.step_requested.connect(self._show_step)
        # Above the base's busy row, so the steps never shift while a run starts.
        self._central_layout.insertWidget(0, self._stepper)

        # One progress block, moved into the view of whichever step is running.
        self._run_progress = RunProgress()
        self._run_progress.cancel_requested.connect(self._cancel_current_analysis)
        self._progress_hosts: dict[str, QWidget] = {}

        self._views: dict[str, QWidget] = {
            "scope": self._build_scope_view(),
            "screen": self._build_screen_view(),
            "compare": self._build_compare_view(),
            "phases": self._build_phases_view(),
            "apply": self._build_apply_view(),
        }
        self._stack = QStackedWidget()
        for view in self._views.values():
            self._stack.addWidget(view)
        self._progress_hosts["screen"].layout().addWidget(self._run_progress)
        self._run_progress.hide()
        return self._stack

    # ------------------------------------------------------------------
    # View construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_scroll_page(content: QWidget) -> QScrollArea:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        page.setWidget(content)
        return page

    def _progress_host(self, step: str) -> QWidget:
        """An empty slot at the top of ``step``'s view where its run's progress shows."""
        host = QWidget()
        # Never taller than the progress block, so a hidden step body leaves the slack below.
        host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        self._progress_hosts[step] = host
        return host

    @staticmethod
    def _footer_row(button: QPushButton) -> QHBoxLayout:
        """A step's footer: its primary action on the right."""
        button.setStyleSheet(build_primary_button_qss())
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button)
        return row

    def _build_scope_view(self) -> QWidget:
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
        self._picker = ModelFamilyPicker(
            lambda scope: describe_scope(self._datasets, scope, self._fit_times())
        )
        # A floor keeps the family cards usable inside the scrolling view.
        self._picker.setMinimumHeight(metrics.row_height() * _PICKER_MIN_ROWS)
        self._picker.scope_changed.connect(
            lambda _scope: self._mark_analysis_stale("Scope changed")
        )
        self._picker.validity_changed.connect(self._on_scope_validity_changed)
        self._picker.direction_answered.connect(self._answer_field_direction)
        layout.addWidget(self._picker)

        # --- Optional parameter expectations (embedded ex-dialog). ---
        layout.addWidget(self._build_expectations_section())

        # --- Search settings. ---
        layout.addWidget(make_section_header("Search settings"))
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Recommend:"))
        self._objective_combo = QComboBox()
        for index, (objective, (label, hint)) in enumerate(_OBJECTIVES.items()):
            self._objective_combo.addItem(label, userData=objective)
            self._objective_combo.setItemData(index, hint, Qt.ItemDataRole.ToolTipRole)
        self._objective_combo.currentIndexChanged.connect(self._on_objective_changed)
        settings_row.addWidget(self._objective_combo)
        settings_row.addWidget(QLabel("Ranking Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        settings_row.addWidget(self._metric_combo)
        metric_info_btn = QPushButton("Metric Info")
        metric_info_btn.clicked.connect(self._show_metric_info)
        settings_row.addWidget(metric_info_btn)
        # Single honest optimisation mode. Every EffortTier resolves to the
        # exact bounded-wavefront engine, so the visible control is a single
        # disabled item; the 1-item combo keeps current_effort_tier() and the
        # payload round-trip working unchanged.
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
        self._objective_hint = _muted_label(_OBJECTIVES[SelectionObjective.TREND][1])
        layout.addWidget(self._objective_hint)
        layout.addStretch()

        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self._make_scroll_page(content), 1)
        self._refresh_btn = QPushButton("Run screening")
        self._refresh_btn.clicked.connect(self._start_analysis)
        view_layout.addLayout(self._footer_row(self._refresh_btn))
        return view

    def _build_expectations_section(self) -> PanelSection:
        """Build the embedded parameter-expectations editor (ex modal dialog).

        Screening no longer blocks on a dialog: the table is populated from the
        candidate portfolio when the context arrives, and *Run screening* reads
        it in place (invalid bounds surface inline and stop the run).
        """
        self._expectations_section = PanelSection(
            "Guide the search (optional)", collapsible=True, expanded=False
        )
        self._expectations_section.addWidget(
            _muted_label(
                "The wizard explores the candidate families below. Review the combined "
                "parameter list and set your expected Global/Local behaviour and bounds "
                "before the expensive search starts. Defaults: amplitudes start as Global "
                "with positive bounds, rate-like parameters start as Local with positive "
                "bounds, and background terms stay Global unless you change them."
            )
        )

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

    def _build_screen_view(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(
            _muted_label(
                "Each family was fitted to every run on its own. Tick the families to "
                "optimise with coupled global fits; click one to preview its fits."
            )
        )

        body = QHBoxLayout()
        board = QVBoxLayout()
        self._leaderboard = ScreeningLeaderboard()
        self._leaderboard.ticked_changed.connect(
            lambda _keys: self._update_action_enablement(self._analysis_in_progress)
        )
        self._leaderboard.selected_changed.connect(self._preview_screening_family)
        board.addWidget(self._leaderboard)
        self._screen_empty = _muted_label("No screening table was kept with this result.")
        board.addWidget(self._screen_empty)
        board.addStretch()
        # Two thirds of the width, so family titles do not elide at 1180 px.
        body.addLayout(board, 2)
        self._screen_canvas = SeriesFitCanvas()
        body.addWidget(self._screen_canvas, 1)
        layout.addLayout(body, 1)

        details = PanelSection("Details", collapsible=True, expanded=False)
        details.addWidget(
            _muted_label(
                "The screening scores behind the leaderboard, and the candidate families "
                "the portfolio considered. Screening scores come from independent "
                "per-run fits, without coupled parameter sharing."
            )
        )
        self._screening_table = QTableWidget(0, 9)
        self._screening_table.setHorizontalHeaderLabels(
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
        self._screening_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._screening_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._screening_table.horizontalHeader().setSectionResizeMode(
            8, QHeaderView.ResizeMode.Stretch
        )
        details.addWidget(self._screening_table)
        details.addWidget(make_section_header("Candidate portfolio"))
        self._portfolio_table = QTableWidget(0, 4)
        self._portfolio_table.setHorizontalHeaderLabels(
            ["Candidate", "Category", "Parameters", "Rationale"]
        )
        self._portfolio_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self._portfolio_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._portfolio_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        details.addWidget(self._portfolio_table)
        layout.addWidget(details)

        self._screen_body = QWidget()
        body_layout = QVBoxLayout(self._screen_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self._make_scroll_page(content), 1)
        self._optimise_btn = QPushButton()
        self._optimise_btn.clicked.connect(self._start_selected_optimisation)
        body_layout.addLayout(self._footer_row(self._optimise_btn))

        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self._progress_host("screen"))
        view_layout.addWidget(self._screen_body, 1)
        view_layout.addStretch()
        return view

    def _build_compare_view(self) -> QWidget:
        # Not a scroll page: the panel fits the view, so Continue stays in sight.
        self._compare_body = QWidget()
        layout = QVBoxLayout(self._compare_body)
        layout.setContentsMargins(0, 0, 0, 0)
        self._compare_panel = ModelComparePanel()
        self._compare_panel.a_changed.connect(self._show_compare_a)
        self._compare_panel.continue_requested.connect(
            lambda key: self._choose_apply_target(_ModelTarget(key))
        )
        layout.addWidget(self._compare_panel, 1)

        details = PanelSection("Details", collapsible=True, expanded=False)
        details_content = QWidget()
        details_layout = QVBoxLayout(details_content)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self._details_intro = _muted_label("")
        details_layout.addWidget(self._details_intro)
        self._optimised_table = QTableWidget(0, 8)
        self._optimised_table.setHorizontalHeaderLabels(
            ["Candidate", "Score", "AIC", "AICc", "BIC", "Gate", "Global", "Local"]
        )
        self._optimised_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._optimised_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for column in (6, 7):
            self._optimised_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch
            )
        details_layout.addWidget(self._optimised_table)

        # What follows A depends on the objective: the role search's diagnostics, or a rung's cost.
        self._roles_details = QWidget()
        roles_layout = QVBoxLayout(self._roles_details)
        roles_layout.setContentsMargins(0, 0, 0, 0)
        roles_layout.addWidget(make_section_header("Parameter roles for A"))
        self._roles_table = QTableWidget(0, 5)
        self._roles_table.setHorizontalHeaderLabels(
            ["Parameter", "Role", "Global Score", "Local Score", "Δ"]
        )
        self._roles_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        roles_layout.addWidget(self._roles_table)
        self._roles_rationale = QLabel("")
        self._roles_rationale.setWordWrap(True)
        roles_layout.addWidget(self._roles_rationale)

        self._rung_details = QWidget()
        rung_layout = QVBoxLayout(self._rung_details)
        rung_layout.setContentsMargins(0, 0, 0, 0)
        rung_layout.addWidget(make_section_header("Cost of A"))
        self._rung_grid = KeyValueGrid()
        rung_layout.addWidget(self._rung_grid)
        self._run_costs_table = QTableWidget(0, 4)
        self._run_costs_table.setHorizontalHeaderLabels(["Run", "χ²ᵣ", "Cost (σ)", "Note"])
        self._run_costs_table.horizontalHeader().setStretchLastSection(True)
        self._run_costs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._run_costs_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        rung_layout.addWidget(self._run_costs_table)

        self._details_stack = QStackedWidget()
        self._details_stack.addWidget(self._roles_details)
        self._details_stack.addWidget(self._rung_details)
        details_layout.addWidget(self._details_stack)
        details.addWidget(self._make_scroll_page(details_content))
        layout.addWidget(details)

        self._compare_empty = QWidget()
        empty_layout = QHBoxLayout(self._compare_empty)
        empty_layout.addWidget(_muted_label("Optimise the shortlist on the Screen step first."))
        to_screen = QPushButton("Go to Screen")
        to_screen.clicked.connect(partial(self._show_step, "screen"))
        empty_layout.addWidget(to_screen)
        empty_layout.addStretch()

        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self._progress_host("compare"))
        view_layout.addWidget(self._compare_body, 1)
        view_layout.addWidget(self._compare_empty)
        view_layout.addStretch()
        return view

    def _build_phases_view(self) -> QWidget:
        content = QWidget()
        layout = QHBoxLayout(content)
        self._transitions_card = TransitionsCard()
        self._transitions_card.selection_changed.connect(self._on_transition_row_changed)
        self._transitions_card.optimize_requested.connect(self._start_phase_optimisation)
        self._transitions_card.apply_requested.connect(
            lambda k: self._choose_apply_target(_PhasesTarget(k))
        )
        self._transitions_card.phase_selected.connect(self._on_phase_selected)
        card_column = QVBoxLayout()
        card_column.addWidget(self._transitions_card)
        # Under the trend objective each optimised phase's answer is a ladder rung.
        self._phase_rungs = QVBoxLayout()
        self._phase_rungs.setSpacing(6)
        card_column.addLayout(self._phase_rungs)
        self._phase_rung_rows: dict[int, RungRow] = {}
        card_column.addStretch(1)
        layout.addLayout(card_column, 2)
        plots = QVBoxLayout()
        self._phases_canvas = SeriesFitCanvas()
        plots.addWidget(self._phases_canvas, 3)
        self._phases_strip = ParameterTraceStrip()
        plots.addWidget(self._phases_strip, 2)
        layout.addLayout(plots, 3)

        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self._progress_host("phases"))
        view_layout.addWidget(self._make_scroll_page(content), 1)
        return view

    def _build_apply_view(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        self._apply_title = QLabel("")
        self._apply_title.setWordWrap(True)
        title_font = QFont(self._apply_title.font())
        title_font.setBold(True)
        self._apply_title.setFont(title_font)
        layout.addWidget(self._apply_title)
        self._apply_note = _muted_label("")
        layout.addWidget(self._apply_note)

        # Model mode: Global/Local/Fixed. Phases mode: one row per phase.
        self._apply_roles_section = PanelSection("Parameter roles")
        self._apply_roles = KeyValueGrid()
        self._apply_roles_section.addWidget(self._apply_roles)
        layout.addWidget(self._apply_roles_section)

        # Runs a rung exempts: listed, and left out of the coupled series (plan D17).
        self._apply_left_out = QLabel("")
        self._apply_left_out.setWordWrap(True)
        layout.addWidget(self._apply_left_out)

        self._apply_values_section = PanelSection("Starting values")
        self._apply_values_table = QTableWidget(0, 2)
        self._apply_values_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._apply_values_table.horizontalHeader().setStretchLastSection(True)
        self._apply_values_table.verticalHeader().setVisible(False)
        self._apply_values_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._apply_values_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._apply_values_section.addWidget(self._apply_values_table)
        layout.addWidget(self._apply_values_section)

        self._apply_warnings = QLabel("")
        self._apply_warnings.setWordWrap(True)
        self._apply_warnings.setStyleSheet(f"color: {tokens.WARN};")
        layout.addWidget(self._apply_warnings)

        self._apply_why = PanelSection("Why these roles?", collapsible=True, expanded=False)
        self._apply_rationale = QLabel("")
        self._apply_rationale.setWordWrap(True)
        self._apply_why.addWidget(self._apply_rationale)
        layout.addWidget(self._apply_why)
        layout.addStretch()

        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self._make_scroll_page(content), 1)
        self._apply_btn = QPushButton()
        self._apply_btn.clicked.connect(self._apply_chosen)
        view_layout.addLayout(self._footer_row(self._apply_btn))
        return view

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
        """Prepare the window for a new ordered dataset series (→ Scope)."""
        self._datasets = list(datasets)
        self._current_model = current_model
        self._current_parameter_types = dict(current_parameter_types or {})
        self._current_values = dict(current_values or {})
        self._parameter_bounds = dict(parameter_bounds or {})
        self._single_fit_recommendations_by_run = dict(
            existing_single_fit_recommendations_by_run or {}
        )
        self._recommendation = None
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._run_progress.restore_log("")
        self._run_progress.hide()
        self._cached_signature = None
        self._analysis_request_id += 1
        # Signal-silent, so the final _set_busy(False) evaluates the new runs' validity.
        self._picker.set_scope(WizardScope())
        self._populate_expectations_from_context()
        self._show_step("scope")
        run_label_chips = [dataset.run_label for dataset in self._datasets[:4]]
        if len(self._datasets) > 4:
            run_label_chips.append("…")
        run_labels = ", ".join(run_label_chips)
        self._heading_label.setText("Global Fit Wizard")
        self._status_label.setToolTip("")
        self.set_context_chips([f"{len(self._datasets)} datasets", *run_label_chips])
        self._status_label.setText(
            f"Ready to analyze the selected series ({run_labels}). "
            "Choose the families to screen, then run the screening."
        )
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(SelectionMetric.AICC.value)
        self._metric_combo.blockSignals(False)
        self._show_objective(SelectionObjective.TREND)
        self._set_empty_state()
        # Run / Field / Temperature are known now, so show the series immediately
        # rather than an empty table until screening. The classification columns
        # stay "—" until a recommendation is built (see _populate_overview_table).
        self._populate_series_preview()
        self._set_busy(False)

    def _populate_series_preview(self) -> None:
        """List the loaded runs in the Series section before any screening.

        Shared by ``set_analysis_context`` and a screening run's reset: the
        series stays loaded either way, so Scope keeps listing the runs rather
        than showing a blank table.
        """
        self._populate_overview_table()
        if self._datasets:
            self._overview_banner.setText(
                f"{len(self._datasets)} runs selected. "
                "Run screening to classify each run (Osc. / KT-like / Multi-rate)."
            )

    def _answer_field_direction(self, geometry: FieldGeometry | None) -> None:
        """Save the answer on the runs whose files record no direction."""
        set_user_field_direction(self._datasets, geometry)
        self._picker.refresh()
        self._mark_analysis_stale("Field direction changed")
        self.field_direction_answered.emit(
            frozenset(dataset.run_number for dataset in self._datasets), geometry
        )

    def _on_scope_validity_changed(self, is_valid: bool) -> None:
        if not is_valid and not self._analysis_in_progress:
            self._status_label.setText(
                "Select at least one candidate family in the Scope section to enable screening."
            )
        self._set_busy(self._analysis_in_progress)

    def _mark_analysis_stale(self, reason: str) -> None:
        """Flag the displayed results as stale after a scope or field-direction edit.

        Follows the ignore-stale convention: an in-flight analysis is orphaned by
        bumping the request id (its terminal signal is discarded by the base's
        staleness guard on arrival). We also cancel the live worker cooperatively
        so it stops wasting cycles, then clear busy. The results stay viewable,
        and Screen, Compare and Phases read stale until screening runs again.
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
    # Steps and runs
    # ------------------------------------------------------------------

    def _show_step(self, key: str) -> None:
        self._stack.setCurrentWidget(self._views[key])
        self._stepper.set_current(key)

    def _host_progress(self, step: str) -> None:
        """Move the progress block into ``step``'s view."""
        self._run_progress.parentWidget().layout().removeWidget(self._run_progress)
        self._progress_hosts[step].layout().addWidget(self._run_progress)

    def _begin_run(self, mode: str, opening_log: str) -> None:
        """Run ``mode``'s analysis with its progress in the step it feeds (D2)."""
        run = _RUN_MODES[mode]
        self._analysis_mode = mode
        self._run_origin = self._stepper.current_key()
        self._run_progress.start(run.header, run.placeholders)
        self._run_progress.trail.set_status(run.opening_status)
        self._run_progress.append_log(opening_log)
        self._host_progress(run.step)
        self._run_progress.show()
        self._show_step(run.step)
        # Base: bump request id, cache signature, set busy, _reset_result_state(),
        # then run _create_worker_task() off-thread.
        self._run_analysis()

    def _refresh_stepper(self) -> None:
        """Derive every step's state and summary (plan D10) from the window's facts."""
        recommendation = self._recommendation
        done = StepState.STALE if self._analysis_stale else StepState.DONE
        ready = StepState.STALE if self._analysis_stale else StepState.READY
        objective_stale = self._objective_stale()
        self._objective_banner.setVisible(objective_stale)
        if objective_stale:
            self._objective_banner.setText(
                f"These fits were optimised as “{_OBJECTIVES[recommendation.objective][0]}”. "
                "Optimise the shortlist on the Screen step again, and the phases on the Phases "
                f"step, to rank them as “{_OBJECTIVES[self.current_objective()][0]}”."
            )
        steps: dict[str, tuple[StepState, str]] = {
            "scope": (
                StepState.READY if recommendation is None else StepState.DONE,
                self._picker.summary(),
            )
        }
        if recommendation is None:
            steps |= dict.fromkeys(
                ("screen", "compare", "phases"), (StepState.PENDING, "Not screened yet")
            )
        else:
            prescreen = recommendation.sorted_prescreen_assessments()
            steps["screen"] = (
                done,
                f"{prescreen[0].template.title} leads" if prescreen else "Screening done",
            )
            optimised = len(recommendation.optimized_assessments())
            steps["compare"] = (
                (
                    StepState.STALE if objective_stale else done,
                    _compare_step_summary(recommendation),
                )
                if optimised
                else (ready, "Next: optimise the shortlist")
            )
            path = recommendation.partition_path
            # Skipped when the path's own pick has no break, unless the user has
            # already optimised a row that has one.
            if path is None or (
                path.solutions[path.selected_k].breaks == 0
                and not any(
                    path.solutions[k].breaks >= 1 for k, _ in recommendation.phase_assessments
                )
            ):
                steps["phases"] = (StepState.SKIPPED, "No transition found")
            else:
                solution = path.solutions[self._partition_k]
                steps["phases"] = (
                    (StepState.STALE if objective_stale else done)
                    if recommendation.phase_assessments
                    else ready,
                    (
                        f"{solution.breaks} transition{'' if solution.breaks == 1 else 's'} · "
                        + format_transition_boundaries(solution, recommendation.series_axis_label)
                    )
                    if solution.breaks
                    else "No transition in this row",
                )
        target = self._apply_target
        if target is None:
            steps["apply"] = (StepState.PENDING, "Pick a model first")
        elif target == self._applied_target:
            steps["apply"] = (StepState.DONE, f"Applied: {self._target_title(target)}")
        else:
            steps["apply"] = (StepState.READY, f"Ready: {self._target_title(target)}")
        if self._analysis_in_progress:
            run = _RUN_MODES[self._analysis_mode]
            steps[run.step] = (StepState.RUNNING, run.step_summary)
        for key, (state, summary) in steps.items():
            self._stepper.set_step(key, state, summary)

    def _update_action_enablement(self, busy: bool) -> None:
        self._progress_label.setText("Working..." if busy else "")
        self._refresh_btn.setEnabled(bool(self._datasets) and not busy and self._picker.is_valid())
        has_result = self._recommendation is not None
        self._metric_combo.setEnabled(has_result and not busy)
        self._objective_combo.setEnabled(not busy)
        ticked = len(self._leaderboard.ticked())
        self._optimise_btn.setText(f"Optimise {ticked} famil{'y' if ticked == 1 else 'ies'} →")
        self._optimise_btn.setEnabled(
            has_result and ticked > 0 and not busy and not self._analysis_stale
        )
        self._transitions_card.set_actions_enabled(not busy and not self._analysis_stale)
        if not busy:
            self._run_progress.finish()
        self._refresh_stepper()

    def _set_empty_state(self) -> None:
        self._overview_banner.setText("")
        self._verdict_banner.setText("")
        self._verdict_banner.setVisible(False)
        for table in (
            self._overview_table,
            self._portfolio_table,
            self._screening_table,
            self._optimised_table,
            self._roles_table,
            self._run_costs_table,
        ):
            table.setRowCount(0)
        self._rung_summaries = {}
        self._running_template_keys = set()
        self._screen_body.setVisible(False)
        self._compare_body.setVisible(False)
        self._compare_empty.setVisible(True)
        self._transitions_card.clear()
        clear_layout(self._phase_rungs)
        self._phase_rung_rows = {}
        self._phase_rung_summaries = ()
        self._phases_strip.setVisible(False)
        self._phases_canvas.set_series([], [], [])
        self._phases_canvas.set_curves(None, None)
        self._partition_k = None
        self._selected_phase_segment = None
        self._apply_target = None
        self._applied_target = None
        self._populate_apply()

    def _start_analysis(self) -> None:
        if len(self._datasets) < 2:
            self._status_label.setText("Global fit wizard requires at least two datasets.")
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

        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        # Same-signature short-circuit: serve the cached recommendation without
        # recomputing. Scope is in the signature, so a change-then-revert to the
        # cached scope makes any stale flag obsolete, as cleared above.
        if (
            self._cached_signature == self._analysis_signature()
            and self._recommendation is not None
        ):
            self._status_label.setText(self._recommendation.summary)
            self._repopulate()
            self._show_step("screen")
            return

        self._status_label.setText(
            "Building the single-fit screening table in the background. "
            "The main window stays responsive while the shared candidate portfolio is screened."
        )
        self._begin_run("screening", f"Starting screening for {len(self._datasets)} datasets.")

    def _start_selected_optimisation(self) -> None:
        self._running_template_keys = set(self._leaderboard.ticked())
        selected_titles = [
            assessment.template.title
            for assessment in self._recommendation.sorted_prescreen_assessments()
            if assessment.template.key in self._running_template_keys
        ]
        self._status_label.setText(
            "Running coupled global optimisation for the selected candidates. "
            "Progress is streamed to the live log."
        )
        # The leaderboard shows the families as running.
        self._repopulate()
        self._begin_run(
            "optimize",
            "Starting coupled global optimisation for: " + ", ".join(selected_titles) + ".",
        )

    def _start_phase_optimisation(self, partition_k: int) -> None:
        """Run the coupled role search per phase of path solution *partition_k*."""
        self._partition_k = partition_k
        self._phase_progress_seen = 0
        self._phase_progress_total = 0
        solution = self._recommendation.partition_path.solutions[partition_k]
        self._status_label.setText(
            "Running the coupled global optimisation once per phase. "
            "Progress is streamed to the live log."
        )
        self._begin_run(
            "optimize_phases",
            f"Starting per-phase coupled optimisation for the {solution.breaks}-break partition.",
        )

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
        objective = self.current_objective()
        selected_keys = tuple(sorted(self._running_template_keys)) if mode == "optimize" else ()
        scope = self._picker.scope()
        fit_times = self._fit_times()
        effort_tier = self.current_effort_tier()
        # The path and the row index travel together — the core refuses one
        # without the other — and only the per-phase mode has either.
        if mode == "optimize_phases":
            partition_path = self._recommendation.partition_path
            partition_k = self._partition_k
        else:
            partition_path = None
            partition_k = None

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
                objective=objective,
                selected_template_keys=selected_keys,
                scope=scope,
                fit_times=fit_times,
                effort_tier=effort_tier,
                partition_path=partition_path,
                partition_k=partition_k,
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

        merging = result.mode in _MERGING_MODES
        if merging:
            self._recommendation = merge_global_fit_wizard_recommendations(
                self._recommendation,
                result.recommendation,
            )
        else:
            # A screening holds no optimised fit, so it carries the chosen objective.
            self._recommendation = replace(
                result.recommendation, objective=self.current_objective()
            )
            self._apply_target = None
            self._applied_target = None
        recommendation = self._recommendation
        self._running_template_keys = set()
        self._selected_phase_segment = None
        self._partition_k = self._default_partition_k(recommendation)
        self._status_label.setText(recommendation.summary)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._run_progress.append_log(recommendation.summary)
        self.analysis_cached.emit(
            recommendation,
            self.current_log_text(),
            copy.deepcopy(self._cached_signature) if self._cached_signature is not None else None,
        )
        # A new screening pre-ticks the shortlist (D7); an optimise keeps the ticks.
        self._populate_from_recommendation(
            self._leaderboard.ticked() if merging else _shortlist_keys(recommendation),
            _recommended_optimised_key(recommendation),
        )
        self._show_step(_RUN_MODES[result.mode].step)
        record_fit_times(result.fitted_assessments)
        self._picker.refresh()

    def _fit_times(self) -> FitTimeEstimates:
        """The series' slow judgement: the picker's tags are what the analysis leaves out."""
        return shared_fit_time_store().estimates(self._datasets)

    def _reset_result_state(self) -> None:
        # Screening starts from a clean slate — no recommendation, so a cancel or
        # failure lands on Scope rather than on emptied results; both optimise
        # modes merge into the existing screening recommendation, so they keep it.
        if self._analysis_mode in _MERGING_MODES:
            return
        self._recommendation = None
        self._set_empty_state()
        self._populate_series_preview()
        self._update_action_enablement(self._analysis_in_progress)

    def _on_analysis_failed(self, message: str) -> None:
        # Keep the header's status line to the failure's first line — a
        # multi-line exception message (e.g. a multiprocessing bootstrap error)
        # would otherwise balloon the header band. The full text stays in the
        # log and in the status line's tooltip.
        failure_text = str(message).strip() or "unknown error"
        self._status_label.setText(
            f"Global fit wizard analysis failed: {failure_text.splitlines()[0]}"
        )
        self._status_label.setToolTip(failure_text)
        self._run_progress.append_log(f"Analysis failed: {message}")
        self._land_after_interrupted_run()

    def _land_after_interrupted_run(self) -> None:
        """A run that ends without a result lands on the step it was started from.

        Screening starts from Scope and drops the recommendation first, so a
        failed or cancelled screening lands on Scope with nothing to repopulate.
        """
        self._running_template_keys = set()
        if self._recommendation is not None:
            self._repopulate()
        self._show_step(self._run_origin)

    _on_analysis_cancelled = _land_after_interrupted_run

    def _on_progress(self, current: int, total: int, message: str) -> None:
        # Base already guarded the request id; stream to the live log and the
        # running trail (prefix table per mode; unmatched → status line only).
        text = (message or "").strip()
        if text:
            self._run_progress.append_log(text)
        lowered = text.lower()
        matched = next(
            (
                key
                for prefix, key in _RUN_MODES[self._analysis_mode].prefixes
                if lowered.startswith(prefix)
            ),
            None,
        )
        if matched is not None:
            self._run_progress.trail.activate_step(matched)
        if self._analysis_mode == "optimize_phases":
            self._update_phase_progress(lowered)
        if text:
            self._run_progress.trail.set_status(text)

    def _update_phase_progress(self, lowered: str) -> None:
        """Count the core's per-phase messages into the "phases" step headline."""
        announcement = _PHASE_TOTAL_ANNOUNCEMENT.match(lowered)
        if announcement is not None:
            self._phase_progress_total = int(announcement.group(1))
            self._phase_progress_seen = 0
            return
        if not lowered.startswith(_PHASE_STEP_PREFIX):
            return
        self._phase_progress_seen += 1
        self._run_progress.trail.set_step_headline(
            "phases",
            f"Optimising phase {self._phase_progress_seen} of {self._phase_progress_total}…",
        )

    def current_log_text(self) -> str:
        return self._run_progress.log_text()

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
        """Show an already-computed recommendation for the context's runs.

        Call :meth:`set_analysis_context` with the recommendation's runs first.
        Lands on Compare when the recommendation holds optimised rows, else on
        Screen.
        """
        context_runs = {int(dataset.run_number) for dataset in self._datasets}
        missing = sorted(set(map(int, recommendation.dataset_order)) - context_runs)
        if missing:
            raise ValueError(
                "set_cached_recommendation needs a context holding every run of the "
                f"recommendation; runs {missing} are missing"
            )
        self._recommendation = recommendation
        self._cached_signature = copy.deepcopy(signature) if isinstance(signature, dict) else None
        self._selected_phase_segment = None
        self._partition_k = self._default_partition_k(recommendation)
        self._apply_target = None
        self._applied_target = None
        # Restore the scope the cached result was screened under; a signature
        # without one keeps the default. Cached state is never stale.
        signature_dict = signature if isinstance(signature, dict) else {}
        if "scope" in signature_dict:
            self._picker.set_scope(WizardScope.from_payload(signature_dict["scope"]))
        # Every effort tier runs the exact engine and the visible control is a
        # single "Optimize" mode, so a legacy Low/Balanced payload stays on it.
        self._set_effort_tier(effort_tier_from_payload(signature_dict.get("effort_tier")))
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._show_objective(recommendation.objective)
        self._status_label.setText(status_text or recommendation.summary)
        landing = "compare" if recommendation.optimized_assessments() else "screen"
        self._run_progress.restore_log(log_text)
        self._host_progress(landing)
        self._run_progress.setVisible(bool(log_text))
        self._set_busy(False)
        self._populate_from_recommendation(
            _shortlist_keys(recommendation), _recommended_optimised_key(recommendation)
        )
        self._show_step(landing)

    def current_objective(self) -> SelectionObjective:
        """What the next optimisation will search and rank for (the Scope step's switch)."""
        return self._objective_combo.currentData()

    def _show_objective(self, objective: SelectionObjective) -> None:
        """Put the switch on ``objective`` without treating it as the user's change."""
        self._objective_combo.blockSignals(True)
        self._objective_combo.setCurrentIndex(self._objective_combo.findData(objective))
        self._objective_combo.blockSignals(False)
        self._objective_hint.setText(_OBJECTIVES[objective][1])

    def _on_objective_changed(self) -> None:
        """Take the user's objective; fits optimised for the other one read stale.

        Screening is the same under either objective, so it stays. A
        recommendation that holds no optimised fit takes the choice at once,
        which is how the choice is stored with the result (as the metric is);
        one that holds some keeps saying what they were optimised for until
        the shortlist, or the phases, are optimised again.
        """
        objective = self.current_objective()
        self._objective_hint.setText(_OBJECTIVES[objective][1])
        recommendation = self._recommendation
        if recommendation is not None and not (
            recommendation.optimized_assessments() or recommendation.phase_assessments
        ):
            self._recommendation = replace(recommendation, objective=objective)
            if isinstance(self._cached_signature, dict):
                self.analysis_cached.emit(
                    self._recommendation,
                    self.current_log_text(),
                    copy.deepcopy(self._cached_signature),
                )
        self._set_busy(self._analysis_in_progress)

    def _objective_stale(self) -> bool:
        """Whether the optimised fits answer another objective than the one now chosen."""
        recommendation = self._recommendation
        return (
            recommendation is not None and recommendation.objective is not self.current_objective()
        )

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
            "scope": self._picker.scope().to_payload(),
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
                scope=self._picker.scope(),
                fit_times=self._fit_times(),
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
        # A name shared by several templates takes the kind the first one declares.
        kinds: dict[str, ParameterKind] = {}
        for template in reversed(portfolio.templates):
            kinds.update(template.model.parameter_kinds())
        self._expectation_parameter_names = names
        self._expectations_table.setRowCount(len(names))
        for row, name in enumerate(names):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._expectations_table.setItem(row, 0, name_item)

            role_combo = QComboBox()
            role_combo.addItems(["Global", "Local", "Fixed"])
            role_combo.setCurrentText(
                _default_parameter_role(
                    name, kinds[name], current_parameter_types=self._current_parameter_types
                )
            )
            self._expectations_table.setCellWidget(row, 1, role_combo)

            bounds_item = QTableWidgetItem(
                _format_bounds_text(
                    _default_parameter_bounds(
                        name, kinds[name], current_parameter_bounds=self._parameter_bounds
                    )
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

    def _populate_from_recommendation(self, ticked: tuple[str, ...], a_key: str | None) -> None:
        """Render the recommendation in every step, with these shortlist ticks and A."""
        recommendation = self._recommendation
        self._overview_banner.setText(
            recommendation.mixed_axes_warning
            or (
                f"Series ordered by {recommendation.series_axis_label}. "
                "The wizard compares one common fit function across "
                f"{len(recommendation.dataset_order)} datasets."
            )
        )
        self._populate_overview_table()
        self._populate_portfolio_table()
        datasets, labels, axis_values = self._series_in_order(recommendation)
        # The Screen and Compare headers name the ranking metric (D11).
        metric_label = f"Δ{recommendation.metric.value} from best"

        prescreen = recommendation.sorted_prescreen_assessments()
        self._leaderboard.set_candidates(
            summarise_candidates(prescreen, self._datasets, recommendation.metric),
            {int(dataset.run_number): label for dataset, label in zip(datasets, labels)},
            metric_label,
            ticked=ticked,
            statuses={
                assessment.selection_key: (
                    OptimisationStatus.RUNNING
                    if assessment.template.key in self._running_template_keys
                    else _OPTIMISATION_STATUSES[
                        recommendation.optimization_status_for_key(assessment.template.key)
                    ]
                )
                for assessment in prescreen
            },
        )
        self._leaderboard.setVisible(bool(prescreen))
        self._screen_empty.setVisible(not prescreen)
        self._screen_canvas.set_series(datasets, labels, axis_values)
        self._preview_screening_family(self._leaderboard.selected_key())
        self._populate_screening_table(prescreen)
        self._screen_body.setVisible(True)

        self._compare_panel.set_series(
            datasets, labels, axis_values, recommendation.series_axis_label
        )
        # The objective picks what Compare lists: ladder rungs, or role-search nodes.
        if recommendation.objective is SelectionObjective.TREND:
            optimised = recommendation.sorted_rungs()
            rungs = summarise_rungs(optimised, self._datasets, recommendation.metric)
            self._rung_summaries = {rung.key: rung for rung in rungs}
            self._compare_panel.set_ladders(
                rungs, recommended_key=recommendation.recommended_key, a_key=a_key
            )
            self._details_intro.setText(
                "The rungs' information criteria, and what A costs against the same model "
                "with every parameter local: for the series, and run by run."
            )
            self._details_stack.setCurrentWidget(self._rung_details)
        else:
            optimised = recommendation.sorted_optimized_assessments()
            self._rung_summaries = {}
            self._compare_panel.set_candidates(
                summarise_candidates(optimised, self._datasets, recommendation.metric),
                metric_label,
                a_key=a_key,
            )
            self._details_intro.setText(
                "The optimised fits' scores, and the parameter-sharing diagnostics for A. "
                "Role recommendations use penalized score differences; fixed parameters "
                "are left untouched."
            )
            self._details_stack.setCurrentWidget(self._roles_details)
        self._compare_body.setVisible(bool(optimised))
        self._compare_empty.setVisible(not optimised)
        self._populate_optimised_table(optimised)
        if optimised:
            self._show_compare_a(self._compare_panel.a_key())

        self._populate_transitions_card()
        self._draw_phases()
        self._populate_apply()
        self._update_action_enablement(self._analysis_in_progress)

    def _repopulate(self) -> None:
        """Re-render the same recommendation, keeping the user's ticks and A."""
        self._populate_from_recommendation(self._leaderboard.ticked(), self._compare_panel.a_key())

    def _run_labels(self) -> dict[int, str]:
        return {int(dataset.run_number): dataset.run_label for dataset in self._datasets}

    def _series_in_order(
        self, recommendation: GlobalFitWizardRecommendation
    ) -> tuple[list[MuonDataset], list[str], list[float | None]]:
        """The runs in series order, their labels, and their series-axis values."""
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        datasets = [by_run[int(run)] for run in recommendation.dataset_order]
        return (
            datasets,
            [dataset.run_label for dataset in datasets],
            [
                self._series_axis_value(dataset, recommendation.series_axis_key)
                for dataset in datasets
            ],
        )

    def _preview_screening_family(self, key: str | None) -> None:
        """Draw family ``key``'s per-run screening fits; ``None`` for an empty board."""
        recommendation = self._recommendation
        self._screen_canvas.set_curves(
            None
            if key is None
            else summarise_candidates(
                [recommendation.assessment_for_key(key)], self._datasets, recommendation.metric
            )[0],
            None,
        )

    def _show_compare_a(self, key: str) -> None:
        """Follow candidate A in the Details section and Continue."""
        recommendation = self._recommendation
        assessment = recommendation.assessment_for_key(key)
        self._compare_panel.set_continue_enabled(assessment.is_successful)
        if recommendation.objective is SelectionObjective.TREND:
            self._show_rung_cost(self._rung_summaries[key])
            return
        recommendations = assessment.parameter_recommendations
        self._roles_table.setRowCount(len(recommendations))
        for row, parameter in enumerate(recommendations):
            self._roles_table.setItem(
                row, 0, QTableWidgetItem(get_param_info(parameter.name).unicode_label())
            )
            self._roles_table.setItem(row, 1, QTableWidgetItem(parameter.recommended_role))
            self._roles_table.setItem(row, 2, _numeric_item(parameter.global_score))
            self._roles_table.setItem(row, 3, _numeric_item(parameter.local_score))
            self._roles_table.setItem(row, 4, _numeric_item(parameter.score_delta))
        self._roles_rationale.setText(_role_rationale(assessment))

    def _show_rung_cost(self, summary: RungSummary) -> None:
        """Details for a rung: its χ²ᵣ against all-local, and each run's cost in σ."""
        rung = summary.rung
        self._rung_grid.set_rows(
            [
                ("χ²ᵣ, every parameter local", f"{rung.all_local_chi2r:.4f}"),
                ("χ²ᵣ, this rung", f"{summary.chi2r:.4f}"),
                ("Cost", f"{rung.series_cost:+.2f}σ (tolerance {ADEQUACY_SIGMA:g}σ)"),
                (
                    "Trend quality",
                    html.escape(
                        " · ".join(
                            f"{_symbol(name)} {quality.quality:.2f}"
                            for name, quality in rung.trend.parameters.items()
                        )
                        or "nothing left local"
                    ),
                ),
            ]
        )
        notes = {
            **dict.fromkeys(rung.offending_runs, f"over the {ADEQUACY_SIGMA:g}σ tolerance"),
            **dict.fromkeys(rung.exempt_runs, "exempt: keeps its own amplitude"),
        }
        self._run_costs_table.setRowCount(len(summary.runs))
        for row, run in enumerate(summary.runs):
            note = notes.get(run.run_number, "")
            cells = (
                QTableWidgetItem(run.run_label),
                _numeric_item(run.reduced_chi_squared),
                QTableWidgetItem(f"{rung.run_costs[run.run_number]:+.2f}"),
                QTableWidgetItem(note),
            )
            for column, cell in enumerate(cells):
                if note:
                    cell.setForeground(QColor(tokens.WARN))
                self._run_costs_table.setItem(row, column, cell)
        _fit_height_to_rows(self._run_costs_table)

    def _choose_apply_target(self, target: _ApplyTarget) -> None:
        """Continue from Compare or Phases to review ``target`` on Apply (D5)."""
        self._apply_target = target
        self._populate_apply()
        self._refresh_stepper()
        self._show_step("apply")

    def _target_title(self, target: _ApplyTarget) -> str:
        match target:
            case _ModelTarget(key=key):
                return self._recommendation.assessment_for_key(key).template.title
            case _PhasesTarget(partition_k=partition_k):
                solution = self._recommendation.partition_path.solutions[partition_k]
                return f"{sum(not segment.excluded for segment in solution.segments)} phases"

    def _populate_apply(self) -> None:
        """Review what the chosen target will hand over (D5)."""
        target = self._apply_target
        recommendation = self._recommendation
        model = isinstance(target, _ModelTarget)
        for widget in (self._apply_values_section, self._apply_warnings, self._apply_why):
            widget.setVisible(model)
        self._apply_left_out.setVisible(False)
        self._apply_roles_section.setVisible(target is not None)
        self._apply_btn.setVisible(target is not None)
        match target:
            case None:
                self._apply_title.setText("Pick a model on the Compare step first.")
                self._apply_note.setText(
                    "Continue with A on Compare, or Apply phases on Phases, to review "
                    "what will be handed to the global fit tab."
                )
                return
            case _ModelTarget(key=key):
                assessment = recommendation.assessment_for_key(key)
                self._apply_title.setText(assessment.template.title)
                self._apply_roles_section.set_title("Parameter roles")
                roles = assessment.applied_roles
                self._apply_roles.set_rows(
                    [
                        (
                            role,
                            html.escape(
                                ", ".join(_symbol(n) for n, r in roles.items() if r == role)
                                or "none"
                            ),
                        )
                        for role in ("Global", "Local", "Fixed")
                    ]
                )
                if assessment.exempt_runs:
                    self._apply_left_out.setText(left_out_note(assessment))
                    self._apply_left_out.setVisible(True)
                # The global fit tab seeds every parameter from its first run's fit.
                first_run = next(
                    dataset
                    for dataset in self._datasets
                    if int(dataset.run_number) not in assessment.exempt_runs
                )
                self._apply_values_section.set_hint(
                    f"From run {first_run.run_label}'s fit; local parameters start "
                    "there on every run."
                )
                parameters = list(
                    assessment.fit_results_by_run[int(first_run.run_number)].parameters
                )
                self._apply_values_table.setRowCount(len(parameters))
                for row, parameter in enumerate(parameters):
                    self._apply_values_table.setItem(
                        row, 0, QTableWidgetItem(format_param_label(parameter.name))
                    )
                    self._apply_values_table.setItem(
                        row, 1, QTableWidgetItem(f"{parameter.value:.6g}")
                    )
                _fit_height_to_rows(self._apply_values_table)
                self._apply_warnings.setText(
                    "\n".join(f"• {warning}" for warning in assessment.series_warnings)
                )
                self._apply_warnings.setVisible(bool(assessment.series_warnings))
                self._apply_rationale.setText(
                    _rung_rationale(self._rung_summaries[key], self._run_labels())
                    if recommendation.objective is SelectionObjective.TREND
                    else _role_rationale(assessment)
                )
                self._apply_btn.setText("Apply to the global fit tab")
                self._apply_btn.setEnabled(assessment.is_successful)
            case _PhasesTarget(partition_k=partition_k):
                solution = recommendation.partition_path.solutions[partition_k]
                self._apply_title.setText(
                    transitions_summary(solution, recommendation.series_axis_label)
                )
                self._apply_roles_section.set_title("Phases")
                self._apply_roles.set_rows(
                    [
                        (
                            f"Phase {phase.ordinal} · {phase.range_text}",
                            html.escape(
                                f"{phase.template_title} · {phase.roles_text} · "
                                f"{phase.confidence_text}"
                            ),
                        )
                        for phase in self._phase_summaries(recommendation, partition_k)
                    ]
                )
                left_out = [
                    left_out_note(assessment)
                    for (k, _segment), assessment in sorted(
                        recommendation.phase_assessments.items()
                    )
                    if k == partition_k and assessment.exempt_runs
                ]
                self._apply_left_out.setText("\n".join(left_out))
                self._apply_left_out.setVisible(bool(left_out))
                self._apply_btn.setText("Apply phases")
                self._apply_btn.setEnabled(True)
        self._apply_note.setText(
            "Applied to the global fit tab."
            if target == self._applied_target
            else "Review what will be handed to the global fit tab, then apply it."
        )

    def _apply_chosen(self) -> None:
        """Hand the chosen target to the global fit tab (the unchanged signals)."""
        recommendation = self._recommendation
        target = self._apply_target
        match target:
            case _ModelTarget(key=key):
                self.apply_assessment_requested.emit(
                    recommendation.assessment_for_key(key), recommendation
                )
            case _PhasesTarget(partition_k=partition_k):
                self.apply_phases_requested.emit(recommendation, partition_k)
            case _:
                raise RuntimeError("Apply is only offered once a target has been chosen")
        self.statusBar().showMessage(f"Applied: {self._target_title(target)}")
        self._applied_target = target
        self._populate_apply()
        self._refresh_stepper()

    # --- Phases ---------------------------------------------------------

    @staticmethod
    def _default_partition_k(
        recommendation: GlobalFitWizardRecommendation,
    ) -> int | None:
        """Which path row to select for *recommendation*, or ``None`` for no path.

        An optimised recommendation names the row its per-phase verification
        settled on (the elbow can move onto a neighbour the exact fits measured);
        a screening-only one names the closed-form elbow.
        """
        path = recommendation.partition_path
        if path is None:
            return None
        if recommendation.recommended_partition_k is not None:
            return recommendation.recommended_partition_k
        return path.selected_k

    @staticmethod
    def _excluded_note(solution) -> str:
        """``"excluded: runs 28, 29"`` for a solution carrying end stubs."""
        runs = [
            int(run)
            for segment in solution.segments
            if segment.excluded
            for run in segment.run_numbers
        ]
        if not runs:
            return ""
        listed = ", ".join(str(run) for run in runs)
        return f"excluded: run {listed}" if len(runs) == 1 else f"excluded: runs {listed}"

    def _transition_rows(
        self,
        recommendation: GlobalFitWizardRecommendation,
    ) -> list[TransitionRow]:
        """One card row per penalty-path solution."""
        path = recommendation.partition_path
        verified = {k for k, _segment_index in recommendation.phase_assessments}
        axis_label = recommendation.series_axis_label
        return [
            TransitionRow(
                breaks=solution.breaks,
                boundaries_text=format_transition_boundaries(solution, axis_label),
                # The top of the path has nothing to improve on, so its gain is
                # a placeholder rather than a measured zero.
                gain_text="" if index == 0 else f"{solution.gain:.1f}",
                is_elbow=index == path.selected_k,
                is_verified=index in verified,
                excluded_note=self._excluded_note(solution),
                summary=transitions_summary(solution, axis_label),
            )
            for index, solution in enumerate(path.solutions)
        ]

    def _phase_summaries(
        self,
        recommendation: GlobalFitWizardRecommendation,
        partition_k: int,
    ) -> list[PhaseSummary]:
        """One strip entry per optimised phase of solution *partition_k*.

        Empty until that solution has been optimised: tier 3 writes an assessment
        for *every* non-excluded segment of a solution it scored, so a solution
        either has all of them or none.
        """
        solution = recommendation.partition_path.solutions[partition_k]
        by_run = {int(dataset.run_number): dataset for dataset in self._datasets}
        summaries: list[PhaseSummary] = []
        ordinal = 0
        for segment_index, segment in enumerate(solution.segments):
            if segment.excluded:
                continue
            assessment = recommendation.phase_assessments[(partition_k, segment_index)]
            ordinal += 1
            values = [
                self._series_axis_value(by_run[int(run)], recommendation.series_axis_key)
                for run in segment.run_numbers
            ]
            spans = [value for value in values if value is not None]
            summaries.append(
                PhaseSummary(
                    segment_index=segment_index,
                    ordinal=ordinal,
                    color=phase_color(ordinal),
                    range_text=(
                        format_axis_range(min(spans), max(spans), recommendation.series_axis_key)
                        if spans
                        # No run records an axis value, so name the runs rather
                        # than invent an axis span.
                        else f"runs {segment.run_numbers[0]}–{segment.run_numbers[-1]}"
                    ),
                    template_title=assessment.template.title,
                    roles_text=(
                        f"Global: {', '.join(assessment.global_param_names) or 'none'} · "
                        f"Local: {', '.join(assessment.local_param_names) or 'none'}"
                    ),
                    confidence_text=_phase_verdict_text(assessment, recommendation.objective),
                )
            )
        return summaries

    def _populate_transitions_card(self) -> None:
        """Render the path, the selected row's phases, and the actions."""
        recommendation = self._recommendation
        clear_layout(self._phase_rungs)
        self._phase_rung_rows = {}
        self._phase_rung_summaries = ()
        if recommendation.partition_path is None:
            self._transitions_card.clear()
            return
        self._transitions_card.set_rows(self._transition_rows(recommendation), self._partition_k)
        verified = any(k == self._partition_k for k, _ in recommendation.phase_assessments)
        phases = self._phase_summaries(recommendation, self._partition_k) if verified else []
        ladders = recommendation.objective is SelectionObjective.TREND
        # A ladder rung has more to say than a chip holds: it gets a row of its own below.
        self._transitions_card.set_phases(() if ladders else phases)
        if ladders:
            # Each phase is its own pool: one answer, and the band taken on its runs.
            self._phase_rung_summaries = tuple(
                summarise_rungs(
                    [recommendation.phase_assessments[(self._partition_k, phase.segment_index)]],
                    self._datasets,
                    recommendation.metric,
                )[0]
                for phase in phases
            )
            run_labels = self._run_labels()
            for phase, rung in zip(phases, self._phase_rung_summaries, strict=True):
                # The phase's colour on a left stripe, as in the Data Browser; the
                # title elides rather than wrapping the column taller.
                heading = QWidget()
                heading_row = QHBoxLayout(heading)
                heading_row.setContentsMargins(0, 4, 0, 0)
                stripe = QFrame()
                stripe.setFixedWidth(4)
                stripe.setStyleSheet(f"background: {phase.color};")
                heading_row.addWidget(stripe)
                title = ElidedLabel(
                    f"Phase {phase.ordinal} · {phase.range_text} · {phase.template_title}"
                )
                title.setFont(_bold_font(title.font()))
                heading_row.addWidget(title, 1)
                self._phase_rungs.addWidget(heading)
                row = RungRow(rung, run_labels, self, recommended=False, slots=False)
                row.picked.connect(
                    lambda _key, index=phase.segment_index: self._on_phase_selected(index)
                )
                self._phase_rung_rows[phase.segment_index] = row
                self._phase_rungs.addWidget(row)
        self._transitions_card.set_actions_enabled(
            not self._analysis_in_progress and not self._analysis_stale
        )

    def _phase_colour_by_run(self) -> dict[int, str]:
        """Phase colour per run for the selected path row; empty when none applies.

        A break-free row is the whole series in one phase, which the axis
        gradient already says better than a single flat colour would.
        """
        recommendation = self._recommendation
        if self._partition_k is None:
            return {}
        solution = recommendation.partition_path.solutions[self._partition_k]
        if solution.breaks < 1:
            return {}
        colours: dict[int, str] = {}
        ordinal = 0
        for segment in solution.segments:
            if segment.excluded:
                # An excluded stub belongs to no phase, so it wears the "no
                # phase" grey the Data Browser hatches its rows with.
                colours.update(
                    {int(run): EXCLUDED_PHASE_HATCH_COLOR for run in segment.run_numbers}
                )
                continue
            ordinal += 1
            colours.update({int(run): phase_color(ordinal) for run in segment.run_numbers})
        return colours

    def _draw_phases(self) -> None:
        """Overlay the series coloured by phase, with the picked phase's fit as A (D6)."""
        recommendation = self._recommendation
        datasets, labels, axis_values = self._series_in_order(recommendation)
        colours = self._phase_colour_by_run()
        self._phases_canvas.set_series(
            datasets,
            labels,
            axis_values,
            [colours[int(dataset.run_number)] for dataset in datasets] if colours else None,
        )
        phase = (
            None
            if self._selected_phase_segment is None
            else recommendation.phase_assessments[(self._partition_k, self._selected_phase_segment)]
        )
        self._phases_canvas.set_curves(
            None
            if phase is None
            else summarise_candidates([phase], self._datasets, recommendation.metric)[0],
            None,
        )
        for segment_index, row in self._phase_rung_rows.items():
            row.set_slot("A" if segment_index == self._selected_phase_segment else "")
        # One trace per parameter a phase leaves local, each phase in its own colour.
        self._phases_strip.setVisible(bool(self._phase_rung_summaries))
        if self._phase_rung_summaries:
            self._phases_strip.set_traces(
                rung_traces(
                    [
                        TracedRung(rung, f"Phase {ordinal}", phase_color(ordinal))
                        for ordinal, rung in enumerate(self._phase_rung_summaries, start=1)
                    ]
                ),
                recommendation.series_axis_label,
                [
                    estimate
                    for estimate, _half_gap in recommendation.partition_path.solutions[
                        self._partition_k
                    ].boundaries
                ],
            )

    def _on_transition_row_changed(self, index: int) -> None:
        """A path row was picked: recolour the overlay and re-offer the actions."""
        self._partition_k = index
        self._selected_phase_segment = None
        self._populate_transitions_card()
        self._draw_phases()
        self._refresh_stepper()

    def _on_phase_selected(self, segment_index: int) -> None:
        """Show one phase's assessment in the overlay."""
        self._selected_phase_segment = segment_index
        self._draw_phases()

    @staticmethod
    def _series_axis_value(dataset: MuonDataset, axis_key: str) -> float | None:
        """The run's position along the series axis, or None when unavailable.

        A run-ordered series is positioned by run number, exactly as the core
        orders it (``_axis_value``): the axis is "Run", and the run number is not
        a metadata field to look up.
        """
        if not axis_key:
            return None
        if axis_key == "run":
            return float(dataset.run_number)
        try:
            value = float(dataset.metadata.get(axis_key))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

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

    def _populate_screening_table(self, assessments: list[GlobalCandidateAssessment]) -> None:
        recommendation = self._recommendation
        self._screening_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            self._screening_table.setItem(row, 0, QTableWidgetItem(assessment.template.title))
            self._screening_table.setItem(
                row, 1, _numeric_item(assessment.metric_value(recommendation.metric))
            )
            self._screening_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._screening_table.setItem(
                row,
                3,
                (
                    _numeric_item(assessment.aicc)
                    if assessment.aicc is not None
                    else QTableWidgetItem("AIC")
                ),
            )
            self._screening_table.setItem(row, 4, _numeric_item(assessment.bic))
            self._screening_table.setItem(
                row,
                5,
                QTableWidgetItem(
                    recommendation.optimization_status_for_key(assessment.template.key)
                ),
            )
            self._screening_table.setItem(
                row,
                6,
                QTableWidgetItem(
                    "Running"
                    if assessment.template.key in self._running_template_keys
                    else ("Yes" if not assessment.prescreen_only else "No")
                ),
            )
            self._screening_table.setItem(row, 7, QTableWidgetItem(str(assessment.parameter_count)))
            self._screening_table.setItem(
                row, 8, QTableWidgetItem(str(len(assessment.local_param_names)))
            )
        self._ensure_candidate_column_width(self._screening_table)

    def _populate_optimised_table(self, assessments: list[GlobalCandidateAssessment]) -> None:
        recommendation = self._recommendation
        self._optimised_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            title_item = QTableWidgetItem(assessment.template.title)
            if assessment.selection_key == recommendation.recommended_key:
                title_item.setFont(_bold_font(title_item.font()))
            self._optimised_table.setItem(row, 0, title_item)
            self._optimised_table.setItem(
                row, 1, _numeric_item(assessment.metric_value(recommendation.metric))
            )
            self._optimised_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._optimised_table.setItem(
                row,
                3,
                _numeric_item(assessment.aicc)
                if assessment.aicc is not None
                else QTableWidgetItem("AIC"),
            )
            self._optimised_table.setItem(row, 4, _numeric_item(assessment.bic))
            self._optimised_table.setItem(
                row,
                5,
                QTableWidgetItem("Pass" if assessment.residual_gate_passed else "Warn"),
            )
            self._optimised_table.setItem(
                row, 6, QTableWidgetItem(", ".join(assessment.global_param_names) or "None")
            )
            self._optimised_table.setItem(
                row, 7, QTableWidgetItem(", ".join(assessment.local_param_names) or "None")
            )
        self._ensure_candidate_column_width(self._optimised_table)

    def _on_metric_changed(self, text: str) -> None:
        """Re-rank in place (D11), on whichever step is showing."""
        self._recommendation = rerank_global_fit_wizard_recommendation(
            self._recommendation,
            SelectionMetric.from_value(text),
        )
        self._status_label.setText(self._recommendation.summary)
        self._repopulate()
        if isinstance(self._cached_signature, dict):
            self.analysis_cached.emit(
                self._recommendation,
                self.current_log_text(),
                copy.deepcopy(self._cached_signature),
            )

    def _show_metric_info(self) -> None:
        QMessageBox.information(
            self,
            "Global Fit Wizard Metrics",
            (
                "The Screen step ranks the candidates by the chosen information criterion, "
                "as the difference from the best row. So does the Compare step under "
                f"“{_OBJECTIVES[SelectionObjective.STATISTICAL][0]}”; under "
                f"“{_OBJECTIVES[SelectionObjective.TREND][0]}” it lists each model's "
                "sharing ladder by trend quality, and the criterion only breaks ties.\n\n"
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
                "The Screen step intentionally does not claim that a candidate is good for global fitting. "
                "It only reports how promising the function looks when each dataset is fit independently.\n\n"
                "Warnings on the Compare step combine per-run residual checks with a check for "
                "an abrupt change in the spectra along the series, after the coupled global "
                "optimisation has run."
            ),
        )

    def _ensure_candidate_column_width(
        self,
        table: QTableWidget,
        *,
        minimum_width: int = 420,
    ) -> None:
        table.resizeColumnToContents(0)
        table.setColumnWidth(0, max(table.columnWidth(0), minimum_width))


def _shortlist_keys(recommendation: GlobalFitWizardRecommendation) -> tuple[str, ...]:
    """The families pre-ticked for optimisation when a screening arrives (D7)."""
    return shortlist(
        {
            assessment.selection_key: assessment.metric_value(recommendation.metric)
            for assessment in recommendation.sorted_prescreen_assessments()
        }
    )


def _compare_step_summary(recommendation: GlobalFitWizardRecommendation) -> str:
    """The Compare step's stepper line: what was optimised, counted as its objective lists it."""
    if recommendation.objective is SelectionObjective.TREND:
        ladders = len({rung.template.key for rung in recommendation.sorted_rungs()})
        return f"{ladders} sharing ladder{'' if ladders == 1 else 's'} climbed"
    optimised = len(recommendation.optimized_assessments())
    return f"{optimised} role split{'' if optimised == 1 else 's'} optimised"


def _recommended_optimised_key(recommendation: GlobalFitWizardRecommendation) -> str | None:
    """Compare's default A: the recommended optimised row, else ``None`` (the first row)."""
    recommended = recommendation.recommended_assessment
    if recommended is None or recommended.prescreen_only:
        return None
    return recommended.selection_key


def _rung_rationale(summary: RungSummary, run_labels: dict[int, str]) -> str:
    """Why a ladder rung shares what it shares: its cost, its trends, and the ladder's findings."""
    trends = " · ".join(
        f"{_symbol(name)} {quality.quality:.2f}"
        for name, quality in summary.rung.trend.parameters.items()
    )
    return "\n".join(
        [
            cost_text(summary.rung, run_labels),
            f"Trend quality of what stays local: {trends}."
            if trends
            else "Nothing is left local, so there is no trend to judge.",
            *(finding.detail for finding in summary.findings),
        ]
    )


def _role_rationale(assessment: GlobalCandidateAssessment) -> str:
    """One line per parameter: why the role search gave it its role."""
    if not assessment.parameter_recommendations:
        return "No per-parameter rationale was recorded for this candidate."
    return "\n".join(
        f"{parameter.name}: {parameter.rationale}"
        for parameter in assessment.parameter_recommendations
    )


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
    return label


def _symbol(name: str) -> str:
    return format_param_label(name, include_unit=False)


def _phase_verdict_text(
    assessment: GlobalCandidateAssessment, objective: SelectionObjective
) -> str:
    """How far to trust one phase's answer, in one short phrase.

    A ladder rung says what it costs and how what it leaves local trends. A
    role-search node whose per-run residual gates all pass is the wizard's
    "high" tier; anything else is "medium" and the reader is told to look at
    the warnings.
    """
    if objective is SelectionObjective.TREND:
        rung = assessment.rung
        is_trend, worst, _mean = rung.trend.ordering_key
        trend = f"trend {worst:.2f}" if is_trend else "nothing left local"
        return f"Cost {rung.series_cost:+.1f}σ · {trend}"
    if assessment.residual_gate_passed:
        return "High confidence"
    return "Medium confidence — check the warnings"


def _fit_height_to_rows(table: QTableWidget) -> None:
    """Make ``table`` tall enough for every row, so its view scrolls rather than the table."""
    table.setFixedHeight(
        table.horizontalHeader().sizeHint().height()
        + table.verticalHeader().length()
        + 2 * table.frameWidth()
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


#: Kinds a series is expected to hold steady, and kinds it is expected to move.
_GLOBAL_BY_DEFAULT = frozenset({ParameterKind.BACKGROUND, ParameterKind.AMPLITUDE})
_LOCAL_BY_DEFAULT = frozenset(
    {
        ParameterKind.RATE,
        ParameterKind.FREQUENCY,
        ParameterKind.STATIC_WIDTH,
        ParameterKind.SHAPE,
        ParameterKind.PHASE,
    }
)
#: Kinds that start bounded below at zero, whatever bounds the Fit tab holds.
_NON_NEGATIVE = (_LOCAL_BY_DEFAULT - {ParameterKind.PHASE}) | {ParameterKind.AMPLITUDE}


def _default_parameter_role(
    name: str,
    kind: ParameterKind,
    *,
    current_parameter_types: dict[str, str],
) -> str:
    current = str(current_parameter_types.get(name, "")).strip()
    if current == "Fixed":
        return "Fixed"
    if kind in _GLOBAL_BY_DEFAULT:
        return "Global"
    if kind in _LOCAL_BY_DEFAULT:
        return "Local"
    if current in {"Global", "Local"}:
        return current
    return "Global"


def _default_parameter_bounds(
    name: str,
    kind: ParameterKind,
    *,
    current_parameter_bounds: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    if kind is ParameterKind.BACKGROUND:
        return -float("inf"), float("inf")
    if kind in _NON_NEGATIVE:
        return 0.0, float("inf")
    if name in current_parameter_bounds:
        return current_parameter_bounds[name]
    default_min = get_param_info(name).default_min
    return (
        (float(default_min), float("inf"))
        if default_min is not None
        else (-float("inf"), float("inf"))
    )


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
