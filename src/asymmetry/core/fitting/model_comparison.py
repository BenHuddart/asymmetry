"""Qt-free judgements behind the fit wizards' model-comparison panel.

Evidence weights, the shortlist rule, normalised residuals, parameter flags and
the neutral :class:`CandidateSummary` a comparison panel renders. The summary is
written for N runs; the single-run wizard adapts to it with N = 1. Design:
``docs/plans/global-wizard-stepper.md`` (D4, D7, D8) and
``docs/plans/fit-wizard-compare.md`` (D2, D3).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.parameters import Parameter

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.fit_wizard import (
        CandidateAssessment,
        FitWizardRecommendation,
        SelectionMetric,
    )
    from asymmetry.core.fitting.global_fit_wizard import GlobalCandidateAssessment

Curve = tuple[NDArray[np.float64], NDArray[np.float64]]
_Assessment = TypeVar("_Assessment", "GlobalCandidateAssessment", "CandidateAssessment")

#: Burnham & Anderson: a model more than 10 information units behind the best
#: has "essentially no support", so it is not pre-ticked for optimisation.
SHORTLIST_MAX_DELTA = 10.0
#: Each shortlisted family costs a coupled role search; three bounds that cost.
SHORTLIST_MAX_SIZE = 3
#: A value within this fraction of max(|value|, |finite bounds|, 1) of a finite
#: bound is at that bound. Shared with the residual gate's bound-hit reasons.
AT_BOUND_RELATIVE_TOLERANCE = 1e-6
#: Two estimates of one shared global differ when |a − b| / √(σa² + σb²) ≥ this.
DIFFERENCE_SIGMA_THRESHOLD = 2.0
#: A per-run χ²ᵣ at most this is a good fit, and at most ``FAIR_CHI2_MAX`` a fair one.
GOOD_CHI2_MAX = 1.5
FAIR_CHI2_MAX = 5.0


def score_deltas(scores: Sequence[float]) -> list[float]:
    """Each score minus the best finite score; a non-finite score's delta is ``inf``."""
    values = np.asarray(scores, dtype=float)
    finite = np.isfinite(values)
    deltas = np.full(values.shape, math.inf)
    if finite.any():
        deltas[finite] = values[finite] - values[finite].min()
    return deltas.tolist()


def information_weights(scores: Sequence[float]) -> list[float]:
    """Akaike-style evidence weights, w ∝ exp(−Δ/2), from information-criterion values.

    A non-finite score gets weight ``0.0`` and is left out of the normalisation;
    when every score is non-finite there is nothing to rank and every weight is
    ``0.0``.
    """
    raw = np.exp(-np.asarray(score_deltas(scores), dtype=float) / 2.0)
    total = float(raw.sum())
    if total == 0.0:
        return [0.0] * len(raw)
    return (raw / total).tolist()


def shortlist(scores: Mapping[str, float]) -> tuple[str, ...]:
    """The keys within ``SHORTLIST_MAX_DELTA`` of the best score, best first, at most ``SHORTLIST_MAX_SIZE``."""
    keys = list(scores)
    deltas = score_deltas([scores[key] for key in keys])
    ranked = sorted(
        (delta, index) for index, delta in enumerate(deltas) if delta <= SHORTLIST_MAX_DELTA
    )
    return tuple(keys[index] for _delta, index in ranked[:SHORTLIST_MAX_SIZE])


def normalised_residuals(dataset: MuonDataset, curve: Curve) -> Curve:
    """``(y − f(t)) / σ`` on the dataset's own time grid within the curve's span, f linearly interpolated."""
    curve_time, curve_value = curve
    time = np.asarray(dataset.time, dtype=float)
    inside = (time >= curve_time[0]) & (time <= curve_time[-1])
    t = time[inside]
    model = np.interp(t, curve_time, curve_value)
    residual = (np.asarray(dataset.asymmetry, dtype=float)[inside] - model) / np.asarray(
        dataset.error, dtype=float
    )[inside]
    return t, residual


class FitGrade(Enum):
    """How well one run's fit reads from its χ²ᵣ."""

    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


def grade_reduced_chi_squared(value: float) -> FitGrade:
    """GOOD ≤ ``GOOD_CHI2_MAX`` < FAIR ≤ ``FAIR_CHI2_MAX`` < POOR; a non-finite χ²ᵣ is POOR."""
    if value <= GOOD_CHI2_MAX:
        return FitGrade.GOOD
    if value <= FAIR_CHI2_MAX:
        return FitGrade.FAIR
    return FitGrade.POOR


def bound_side(parameter: Parameter) -> Literal["lower", "upper"] | None:
    """Which finite bound the value sits at, within ``AT_BOUND_RELATIVE_TOLERANCE``."""
    # Infinite bounds are left out of the scale: an infinite |max| would make
    # the tolerance infinite and put every value "at" the lower bound.
    scale = max(
        abs(parameter.value),
        abs(parameter.min) if np.isfinite(parameter.min) else 0.0,
        abs(parameter.max) if np.isfinite(parameter.max) else 0.0,
        1.0,
    )
    tolerance = AT_BOUND_RELATIVE_TOLERANCE * scale
    if np.isfinite(parameter.min) and abs(parameter.value - parameter.min) <= tolerance:
        return "lower"
    if np.isfinite(parameter.max) and abs(parameter.value - parameter.max) <= tolerance:
        return "upper"
    return None


class ParameterFlag(Enum):
    """A warning about one fitted parameter; the value is its display text."""

    NOT_FINITE = "not finite"
    AT_LOWER_BOUND = "at lower bound"
    AT_UPPER_BOUND = "at upper bound"
    POORLY_DETERMINED = "poorly determined"


_BOUND_FLAGS = {"lower": ParameterFlag.AT_LOWER_BOUND, "upper": ParameterFlag.AT_UPPER_BOUND}


def parameter_flags(parameter: Parameter, error: float) -> tuple[ParameterFlag, ...]:
    """The flags a fitted value earns; a constrained parameter never moved, so earns none."""
    if parameter.is_constrained:
        return ()
    flags: list[ParameterFlag] = []
    side = bound_side(parameter)
    if not math.isfinite(parameter.value):
        flags.append(ParameterFlag.NOT_FINITE)
    elif side is not None:
        flags.append(_BOUND_FLAGS[side])
    if math.isfinite(error) and error > abs(parameter.value):
        flags.append(ParameterFlag.POORLY_DETERMINED)
    return tuple(flags)


class ParameterRole(Enum):
    """A parameter's part in a candidate; FITTED is a free parameter of a single-run fit."""

    GLOBAL = "Global"
    LOCAL = "Local"
    FITTED = "Fitted"
    FIXED = "Fixed"


@dataclass(frozen=True)
class Estimate:
    """A fitted value and its 1σ error (NaN when the fit reports none)."""

    value: float
    error: float


@dataclass(frozen=True)
class ParameterRow:
    """One parameter of a candidate.

    ``values`` is a single :class:`Estimate` for a global shared by a coupled
    fit, and otherwise one estimate per run in the candidate's run order.
    ``run_flags`` holds the flags each run's fit earned, in that run order,
    whether or not the value is shared.
    """

    name: str
    role: ParameterRole
    values: Estimate | tuple[Estimate, ...]
    run_flags: tuple[tuple[ParameterFlag, ...], ...]

    @property
    def flags(self) -> tuple[ParameterFlag, ...]:
        """The union of the flags earned in any run, in first-seen order."""
        return tuple(dict.fromkeys(flag for flags in self.run_flags for flag in flags))


@dataclass(frozen=True)
class RunCurves:
    """One run's dense fit curve and the normalised residuals taken against it."""

    fit: Curve
    residuals: Curve


@dataclass(frozen=True)
class RunFit:
    """One run's fit inside a candidate: its curves and χ²ᵣ.

    ``curves`` is ``None`` while the fit's dense curve has not been built: a
    single-run build draws only the rows it answers with (the dense-curve
    contract on :class:`~asymmetry.core.fitting.fit_wizard.CandidateAssessment`),
    and the owner builds any other row on demand. ``axis_value`` is NaN for a
    lone run, which has no series axis.
    """

    run_number: int
    run_label: str
    axis_value: float
    curves: RunCurves | None
    reduced_chi_squared: float


@dataclass(frozen=True)
class CandidateSummary:
    """Everything a comparison panel shows about one candidate, for N runs.

    ``delta`` and ``weight`` are relative to the pool the summary was built in
    (see :func:`summarise_candidates`). ``gate_summary`` is empty when every run
    passed its residual gate.
    """

    key: str
    title: str
    metric_value: float
    delta: float
    weight: float
    gate_passed: bool
    gate_summary: str
    series_warnings: tuple[str, ...]
    runs: tuple[RunFit, ...]
    parameters: tuple[ParameterRow, ...]
    prescreen: bool

    def names(self, role: ParameterRole) -> tuple[str, ...]:
        """The parameters playing ``role``, in row order."""
        return tuple(row.name for row in self.parameters if row.role is role)

    @property
    def curves_built(self) -> bool:
        """Whether every run carries its dense curve (see :class:`RunFit`)."""
        return all(run.curves is not None for run in self.runs)


def _scored(
    assessments: Sequence[_Assessment], metric: SelectionMetric
) -> Iterator[tuple[_Assessment, float, float, float]]:
    """Each assessment with its score, Δ and evidence weight across exactly this pool."""
    scores = [assessment.metric_value(metric) for assessment in assessments]
    return zip(assessments, scores, score_deltas(scores), information_weights(scores), strict=True)


def summarise_candidates(
    assessments: Sequence[GlobalCandidateAssessment],
    datasets: Sequence[MuonDataset],
    metric: SelectionMetric,
) -> tuple[CandidateSummary, ...]:
    """Summaries in input order, with Δ and weight taken across exactly this pool.

    The caller picks the pool: optimised rows for Compare, pre-screen rows for
    Screen. Runs follow each assessment's ``run_diagnostics`` (series order);
    a run with no fit result (an incomplete pre-screen) is left out of ``runs``.
    """
    datasets_by_run = {int(dataset.run_number): dataset for dataset in datasets}
    return tuple(
        _candidate_summary(assessment, datasets_by_run, score, delta, weight)
        for assessment, score, delta, weight in _scored(assessments, metric)
    )


def summarise_single_candidates(
    recommendation: FitWizardRecommendation,
    dataset: MuonDataset,
    metric: SelectionMetric,
) -> tuple[CandidateSummary, ...]:
    """One single-run summary per candidate row, ranked on ``metric``, with Δ and weight across all.

    ``dataset`` is the record the recommendation was built from. A row the
    build left without dense curves gets a :class:`RunFit` whose ``curves`` is
    ``None``. Null baselines and disqualified rows are listed too, marked in
    the title as the wizard's compare table marks them.
    """
    return tuple(
        _single_candidate_summary(assessment, dataset, score, delta, weight)
        for assessment, score, delta, weight in _scored(
            recommendation.sorted_assessments(metric), metric
        )
    )


def _single_candidate_summary(
    assessment: CandidateAssessment,
    dataset: MuonDataset,
    score: float,
    delta: float,
    weight: float,
) -> CandidateSummary:
    fit = assessment.fit_result
    curve = (assessment.fitted_time, assessment.fitted_curve)
    run = RunFit(
        run_number=int(dataset.run_number),
        run_label=dataset.run_label,
        axis_value=math.nan,
        curves=(
            RunCurves(curve, normalised_residuals(dataset, curve))
            if assessment.fitted_time.size
            else None
        ),
        reduced_chi_squared=fit.reduced_chi_squared,
    )
    free = {parameter.name for parameter in fit.parameters.free_parameters}
    parameters = tuple(
        _parameter_row(name, role, [fit], shared=False)
        for role, names in (
            (ParameterRole.FITTED, [name for name in fit.parameters.names if name in free]),
            (ParameterRole.FIXED, [name for name in fit.parameters.names if name not in free]),
        )
        for name in names
    )
    title = assessment.template.title
    if assessment.is_null_baseline:
        title = f"{title} (baseline)"
    elif assessment.is_disqualified:
        title = f"{title} (disqualified)"
    return CandidateSummary(
        key=assessment.template.key,
        title=title,
        metric_value=score,
        delta=delta,
        weight=weight,
        gate_passed=assessment.residual_gate_passed,
        gate_summary=_gate_summary(
            [(run.run_number, run.run_label, assessment.residual_gate_reasons)], parameters, (run,)
        ),
        series_warnings=(),
        runs=(run,),
        parameters=parameters,
        prescreen=False,
    )


def _candidate_summary(
    assessment: GlobalCandidateAssessment,
    datasets_by_run: Mapping[int, MuonDataset],
    score: float,
    delta: float,
    weight: float,
) -> CandidateSummary:
    runs = tuple(
        RunFit(
            run_number=diagnostic.run_number,
            run_label=diagnostic.run_label,
            axis_value=diagnostic.axis_value,
            curves=RunCurves(
                assessment.fitted_curves_by_run[diagnostic.run_number],
                normalised_residuals(
                    datasets_by_run[diagnostic.run_number],
                    assessment.fitted_curves_by_run[diagnostic.run_number],
                ),
            ),
            reduced_chi_squared=assessment.fit_results_by_run[
                diagnostic.run_number
            ].reduced_chi_squared,
        )
        for diagnostic in assessment.run_diagnostics
        if diagnostic.run_number in assessment.fit_results_by_run
    )
    fits = [assessment.fit_results_by_run[run.run_number] for run in runs]
    roles = (
        *((name, ParameterRole.GLOBAL) for name in assessment.global_param_names),
        *((name, ParameterRole.LOCAL) for name in assessment.local_param_names),
        *((name, ParameterRole.FIXED) for name in assessment.fixed_param_names),
    )
    parameters = tuple(
        _parameter_row(
            name,
            role,
            fits,
            shared=role is ParameterRole.GLOBAL and not assessment.prescreen_only,
        )
        for name, role in roles
    )
    return CandidateSummary(
        key=assessment.selection_key,
        title=assessment.template.title,
        metric_value=score,
        delta=delta,
        weight=weight,
        gate_passed=assessment.residual_gate_passed,
        gate_summary=_gate_summary(
            [(d.run_number, d.run_label, d.gate_reasons) for d in assessment.run_diagnostics],
            parameters,
            runs,
        ),
        series_warnings=assessment.series_warnings,
        runs=runs,
        # Only a coupled fit shares a global; a pre-screen fits each run alone.
        parameters=parameters,
        prescreen=assessment.prescreen_only,
    )


def _parameter_row(
    name: str, role: ParameterRole, fits: Sequence[FitResult], *, shared: bool
) -> ParameterRow:
    estimates: list[Estimate] = []
    run_flags: list[tuple[ParameterFlag, ...]] = []
    for fit in fits:
        # A failed fit carries no parameters; its value is unknown, not a runaway.
        if name not in fit.parameters:
            estimates.append(Estimate(math.nan, math.nan))
            run_flags.append(())
            continue
        error = float(fit.uncertainties.get(name, math.nan))
        parameter = fit.parameters[name]
        estimates.append(Estimate(float(parameter.value), error))
        run_flags.append(parameter_flags(parameter, error))
    return ParameterRow(
        name=name,
        role=role,
        values=estimates[0] if shared else tuple(estimates),
        run_flags=tuple(run_flags),
    )


def _gate_summary(
    reasons_by_run: Sequence[tuple[int, str, Sequence[str]]],
    parameters: Sequence[ParameterRow],
    runs: Sequence[RunFit],
) -> str:
    """The gate reasons, minus the bound hits the parameter flags already carry.

    ``reasons_by_run`` holds (run number, run label, reasons); a reason names its
    runs only when there is more than one run to tell apart. The gate's "X at
    lower/upper bound" reason and the parameter flag share one definition
    (:func:`bound_side`), so a flagged bound hit is said once, by the flag.
    """
    flagged = {
        (run.run_number, f"{row.name} {flag.value}")
        for row in parameters
        for run, flags in zip(runs, row.run_flags, strict=True)
        for flag in flags
        if flag in _BOUND_FLAGS.values()
    }
    runs_by_reason: dict[str, list[str]] = {}
    for run_number, run_label, reasons in reasons_by_run:
        for reason in reasons:
            if (run_number, reason) not in flagged:
                runs_by_reason.setdefault(reason, []).append(run_label)
    if len(reasons_by_run) == 1:
        return "; ".join(runs_by_reason)
    return "; ".join(
        f"{reason} ({'run' if len(labels) == 1 else 'runs'} {', '.join(labels)})"
        for reason, labels in runs_by_reason.items()
    )


@dataclass(frozen=True)
class ParameterPair:
    """One parameter name across candidates A and B; a side lacking it is ``None``."""

    name: str
    a: ParameterRow | None
    b: ParameterRow | None

    @property
    def sigma_difference(self) -> float | None:
        """k = |a − b| / √(σa² + σb²) for a global both candidates share, else ``None``.

        k is NaN when the errors are zero or unknown: the difference is unmeasured.
        """
        if self.a is None or self.b is None:
            return None
        a, b = self.a.values, self.b.values
        if not (isinstance(a, Estimate) and isinstance(b, Estimate)):
            return None
        combined = math.hypot(a.error, b.error)
        return abs(a.value - b.value) / combined if combined > 0.0 else math.nan

    @property
    def difference_text(self) -> str:
        """``"differs by 3.1σ"`` when k ≥ ``DIFFERENCE_SIGMA_THRESHOLD``, else empty."""
        k = self.sigma_difference
        if k is None or not k >= DIFFERENCE_SIGMA_THRESHOLD:
            return ""
        return f"differs by {k:.1f}σ"


def compare_parameters(a: CandidateSummary, b: CandidateSummary) -> tuple[ParameterPair, ...]:
    """A's and B's parameters paired by name: globals, then locals, then fixed.

    A name's section is its role in A, or in B when A lacks it; within a section
    A's order comes first, then names only B has.
    """
    rows_a = {row.name: row for row in a.parameters}
    rows_b = {row.name: row for row in b.parameters}
    section = {name: row.role for name, row in {**rows_b, **rows_a}.items()}
    names = [*rows_a, *(name for name in rows_b if name not in rows_a)]
    role_order = tuple(ParameterRole)
    return tuple(
        ParameterPair(name, rows_a.get(name), rows_b.get(name))
        for name in sorted(names, key=lambda name: role_order.index(section[name]))
    )
