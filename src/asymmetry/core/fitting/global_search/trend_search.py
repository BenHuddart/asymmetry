"""The trend objective's search: a sharing ladder for every template in the band.

Where the statistical objective runs the separable role search, this climbs
``climb_sharing_ladder`` once per competing template, on the wizard's spawn
pool, and turns every rung into a candidate the rest of the wizard reads. It
imports the wizard's machinery, so the wizard imports it where it calls it.
Design: ``docs/plans/global-wizard-trend-objective.md`` (D7–D9, D13, Phase 4).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.fraction_form import signal_fraction_form
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    _append_metric,
    _assemble_assignment_assessment,
    _axis_value,
    _drain_separable_tasks,
    _fresh_task_instrumentation,
    _global_candidate_assessment_key,
    _progress_log,
    _record_counter,
    _run_separable_anchor_task,
    _separable_prescreen_results_for_template,
    _separable_search_datasets,
    _SeparableAnchorTask,
    _set_metric,
    _with_dense_curves,
)
from asymmetry.core.fitting.global_search.sharing_ladder import LadderRung, climb_sharing_ladder
from asymmetry.core.fitting.global_search.trend_objective import (
    CandidateRung,
    templates_within_band,
)
from asymmetry.core.fitting.parameters import ParameterSet

__all__ = ["run_trend_search"]


@dataclass(frozen=True)
class _LadderTask:
    """Climb one template's sharing ladder from its all-local node."""

    template_key: str
    template: CandidateTemplate
    #: The series as fit records, at the search resolution.
    datasets: list[MuonDataset]
    #: The all-local node: the ladder's first rung and what every cost is against.
    anchor: GlobalCandidateAssessment
    base_by_run: dict[int, ParameterSet]
    fixed_param_names: tuple[str, ...]
    axis_key: str
    metric: SelectionMetric
    #: ``time.time()`` after which the ladder stops adding further parameters.
    #: Wall-clock, so that every worker process reads the same moment.
    budget_ends: float


@dataclass(frozen=True)
class _LadderResult:
    template_key: str
    #: Every rung in the order it was climbed, with no curves and no residuals.
    assessments: tuple[GlobalCandidateAssessment, ...]
    #: Whether the budget ended the climb before every further parameter was tried.
    restricted: bool
    instrumentation: dict[str, object]


def _names_in_form(
    template_model: CompositeModel, rung_model: CompositeModel, names: tuple[str, ...]
) -> tuple[str, ...]:
    """Parameters of the template as given, as the model a rung was fitted in names them."""
    if rung_model.param_names == template_model.param_names:
        return names
    form = signal_fraction_form(template_model)
    grouped_name = {given: grouped for grouped, given in form.carried.items()}
    return tuple(grouped_name[name] for name in names)


def _rung_assessment(
    task: _LadderTask, rung: LadderRung, verdict: CandidateRung
) -> GlobalCandidateAssessment:
    """Score one coupled rung through the wizard's own assembly."""
    fixed = _names_in_form(task.template.model, rung.model, task.fixed_param_names)
    sharing_run = next(run for run in rung.results_by_run if run not in rung.exempt_runs)
    return _assemble_assignment_assessment(
        task.datasets,
        replace(task.template, model=rung.model),
        base_by_run={run: result.parameters for run, result in rung.results_by_run.items()},
        results_by_run=dict(rung.results_by_run),
        fitted_global=ParameterSet(
            [replace(rung.results_by_run[sharing_run].parameters[name]) for name in rung.shared]
        ),
        global_param_names=rung.shared,
        local_param_names=tuple(
            name for name in rung.model.param_names if name not in fixed and name not in rung.shared
        ),
        fixed_param_names=fixed,
        axis_key=task.axis_key,
        metric=task.metric,
        fit_success=rung.converged,
        dense_curves=False,
        rung=verdict,
    )


def _run_ladder_task(task: _LadderTask) -> _LadderResult:
    """Climb the ladder and return its rungs as candidates.

    The budget (plan D8) is a stopwatch, not a forecast: a further parameter is
    tried only while a rung as long as the last one would still end before
    ``budget_ends``. The background and amplitude rungs are always climbed.
    """
    instrumentation = _fresh_task_instrumentation()
    previous = time.time()
    restricted = False

    def climb_further() -> bool:
        nonlocal previous, restricted
        now = time.time()
        restricted = now + (now - previous) > task.budget_ends
        previous = now
        return not restricted

    ladder = climb_sharing_ladder(
        task.datasets,
        task.template.model,
        all_local_results=task.anchor.fit_results_by_run,
        base_by_run=task.base_by_run,
        axis_values=[_axis_value(dataset, task.axis_key) for dataset in task.datasets],
        cancel_callback=lambda: False,
        climb_further=climb_further,
    )
    _record_counter(instrumentation, "trend_rungs_fitted", len(ladder.rungs) - 1)
    verdicts = [
        CandidateRung.from_ladder_rung(
            rung,
            preselected=rung is ladder.preselected,
            amplitude_unshareable_runs=ladder.amplitude_unshareable_runs,
            all_local_chi2r=_series_reduced_chi_squared(task.anchor),
        )
        for rung in ladder.rungs
    ]
    climbed = [
        replace(task.anchor, rung=verdicts[0]),
        *(
            _rung_assessment(task, rung, verdict)
            for rung, verdict in zip(ladder.rungs[1:], verdicts[1:], strict=True)
        ),
    ]
    return _LadderResult(
        template_key=task.template_key,
        assessments=tuple(
            replace(
                assessment,
                assessment_key=_global_candidate_assessment_key(
                    task.template_key,
                    global_param_names=assessment.global_param_names,
                    local_param_names=assessment.local_param_names,
                ),
            )
            for assessment in climbed
        ),
        restricted=restricted,
        instrumentation=instrumentation,
    )


def _prescreen_fits_for_ladder(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment | None,
    *,
    resolution_matches: bool,
) -> dict[int, FitResult] | None:
    """The pre-screen's per-run fits when a ladder can start from them, else ``None``.

    Beyond what the separable search asks of them, they must carry their
    degrees of freedom, which a cost is measured in. Fits restored from a
    project file carry neither those nor a covariance, and are fitted again.
    """
    results = _separable_prescreen_results_for_template(
        datasets, assessment, resolution_matches=resolution_matches
    )
    if results is None or not all(result.dof for result in results.values()):
        return None
    return results


def _series_reduced_chi_squared(assessment: GlobalCandidateAssessment) -> float:
    results = assessment.fit_results_by_run.values()
    return sum(result.chi_squared for result in results) / sum(result.dof for result in results)


def run_trend_search(
    datasets: list[MuonDataset],
    *,
    templates: Sequence[CandidateTemplate],
    always_competing: Collection[str],
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    prescreen_assessments: dict[str, GlobalCandidateAssessment],
    axis_key: str,
    metric: SelectionMetric,
    progress_callback: Callable[[str], None] | None,
    search_strategy: str,
    instrumentation: dict[str, object] | None,
    cancel_callback: Callable[[], bool] | None,
    search_rebin_factor: int,
    prescreen_rebin_factor: int,
    time_budget_seconds: float | None,
    backstop_seconds: float | None,
    materialise_curves: bool,
) -> tuple[GlobalCandidateAssessment, ...]:
    """Every rung of every competing template's sharing ladder, as wizard candidates.

    Each of ``templates`` gets its all-local node the way the separable search
    builds it: straight from the pre-screen's per-run fits when those sit at the
    search resolution, fitted here otherwise. The templates whose all-local
    series χ²ᵣ lies within the band of the best one's then compete
    (:func:`templates_within_band`), with those named in ``always_competing``
    — the ones the data identified, or the ones the user ticked — and each
    competitor's ladder is one pool task. A template whose all-local node did
    not converge on every run has no ladder and returns nothing.

    ``time_budget_seconds`` is the budget of plan D8: once it has run out, a
    ladder climbs its background and amplitude rungs and no further, and the
    template is recorded under ``trend_restricted_templates`` (one entry per
    search, so per phase when the caller searches phase by phase). A template is
    lost only to ``backstop_seconds``, which stops the pool as it does for the
    separable search.

    Everything is fitted, and reported, at ``search_rebin_factor``: a rung's
    cost is measured against all-local fits of the same records, and the fit a
    user applies seeds the Batch tab's own fit of the native ones.

    ``materialise_curves`` builds the dense curves of everything returned, on
    the search-resolution records; a caller that keeps only some rungs passes
    false and materialises those itself (the dense-curve contract on
    :class:`GlobalCandidateAssessment`).
    """
    _set_metric(instrumentation, "separable_search_rebin_factor", int(search_rebin_factor))
    search_datasets = _separable_search_datasets(
        datasets, search_rebin_factor=search_rebin_factor, instrumentation=instrumentation
    )
    fit_records = [dataset.fit_record() for dataset in search_datasets]
    deadline = None if backstop_seconds is None else time.monotonic() + backstop_seconds
    budget_ends = math.inf if time_budget_seconds is None else time.time() + time_budget_seconds
    resolution_matches = int(prescreen_rebin_factor) == int(search_rebin_factor)

    _progress_log(
        progress_callback,
        f"Sharing ladders: assembling all-local fits for {len(templates)} candidate(s).",
    )
    anchors = _drain_separable_tasks(
        [
            _SeparableAnchorTask(
                template_key=template.key,
                template=template,
                datasets=fit_records,
                base_by_run=template_contexts[template.key][0],
                fixed_param_names=template_contexts[template.key][1],
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                prescreen_results_by_run=_prescreen_fits_for_ladder(
                    search_datasets,
                    prescreen_assessments.get(template.key),
                    resolution_matches=resolution_matches,
                ),
            )
            for template in templates
        ],
        _run_separable_anchor_task,
        activity="Sharing ladders (all-local fits)",
        progress_callback=progress_callback,
        instrumentation=instrumentation,
        cancel_callback=cancel_callback,
        deadline=deadline,
    )
    climbable = {
        result.template_key: result.state
        for result in anchors
        if result.state.anchor_assessment is not None
        and result.state.anchor_assessment.is_successful
        and all(fit.dof for fit in result.state.anchor_assessment.fit_results_by_run.values())
    }
    competing = templates_within_band(
        {
            key: _series_reduced_chi_squared(state.anchor_assessment)
            for key, state in climbable.items()
        }
    ) | (set(always_competing) & set(climbable))
    for template in templates:
        if template.key in competing:
            _append_metric(instrumentation, "trend_competing_templates", template.key)
    _progress_log(
        progress_callback,
        f"Sharing ladders: {len(climbable)} of {len(templates)} candidate(s) fitted every run; "
        f"climbing a ladder for the {len(competing)} that fit about as well as the best one "
        "or that the data or the selection named.",
    )

    ladders = _drain_separable_tasks(
        [
            _LadderTask(
                template_key=template.key,
                template=template,
                datasets=fit_records,
                anchor=climbable[template.key].anchor_assessment,
                base_by_run=template_contexts[template.key][0],
                fixed_param_names=template_contexts[template.key][1],
                axis_key=axis_key,
                metric=metric,
                budget_ends=budget_ends,
            )
            for template in templates
            if template.key in competing
        ],
        _run_ladder_task,
        activity="Sharing ladders",
        progress_callback=progress_callback,
        instrumentation=instrumentation,
        cancel_callback=cancel_callback,
        deadline=deadline,
    )
    for ladder in ladders:
        if ladder.restricted:
            _append_metric(instrumentation, "trend_restricted_templates", ladder.template_key)
        chosen = next(rung for rung in ladder.assessments if rung.rung.preselected)
        _progress_log(
            progress_callback,
            f"{chosen.template.title}: {len(ladder.assessments) - 1} sharing pattern(s) fitted; "
            f"pre-selected Global[{', '.join(chosen.global_param_names) or 'none'}]"
            + (
                " (further parameters left untried: over the time budget)." * ladder.restricted
                or "."
            ),
        )
    by_key = {ladder.template_key: ladder.assessments for ladder in ladders}
    climbed = tuple(
        assessment
        for template in templates
        if template.key in by_key
        for assessment in by_key[template.key]
    )
    if not materialise_curves:
        return climbed
    return tuple(
        _with_dense_curves(
            assessment,
            search_datasets,
            base_by_run={
                run: result.parameters for run, result in assessment.fit_results_by_run.items()
            },
        )
        for assessment in climbed
    )
