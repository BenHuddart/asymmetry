"""Qt-free judgements behind the fit wizards' model-comparison panel.

Evidence weights, the shortlist rule, normalised residuals, parameter flags and
the neutral :class:`CandidateSummary` a comparison panel renders. The summary is
written for N runs; the single-run wizard adapts to it with N = 1. Design:
``docs/plans/global-wizard-stepper.md`` (D4, D7, D8).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.parameters import Parameter

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.fit_wizard import SelectionMetric
    from asymmetry.core.fitting.global_fit_wizard import (
        GlobalCandidateAssessment,
        RunResidualDiagnostic,
    )

Curve = tuple[NDArray[np.float64], NDArray[np.float64]]

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

    RUNS_AWAY = "runs away"
    POORLY_DETERMINED = "poorly determined"


def parameter_flags(parameter: Parameter, error: float) -> tuple[ParameterFlag, ...]:
    """The flags a fitted value earns; a constrained parameter never moved, so earns none."""
    if parameter.is_constrained:
        return ()
    flags: list[ParameterFlag] = []
    if not math.isfinite(parameter.value) or bound_side(parameter) is not None:
        flags.append(ParameterFlag.RUNS_AWAY)
    if math.isfinite(error) and error > abs(parameter.value):
        flags.append(ParameterFlag.POORLY_DETERMINED)
    return tuple(flags)


class ParameterRole(Enum):
    GLOBAL = "Global"
    LOCAL = "Local"
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
    ``flags`` is the union of the flags earned in any run.
    """

    name: str
    role: ParameterRole
    values: Estimate | tuple[Estimate, ...]
    flags: tuple[ParameterFlag, ...]


@dataclass(frozen=True)
class RunFit:
    """One run's fit inside a candidate: its curve, normalised residuals and χ²ᵣ."""

    run_number: int
    run_label: str
    axis_value: float
    curve: Curve
    residuals: Curve
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
    global_names: tuple[str, ...]
    local_names: tuple[str, ...]
    fixed_names: tuple[str, ...]
    metric_value: float
    delta: float
    weight: float
    gate_passed: bool
    gate_summary: str
    series_warnings: tuple[str, ...]
    runs: tuple[RunFit, ...]
    parameters: tuple[ParameterRow, ...]
    prescreen: bool


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
    scores = [assessment.metric_value(metric) for assessment in assessments]
    deltas = score_deltas(scores)
    weights = information_weights(scores)
    datasets_by_run = {int(dataset.run_number): dataset for dataset in datasets}
    return tuple(
        _candidate_summary(assessment, datasets_by_run, score, delta, weight)
        for assessment, score, delta, weight in zip(
            assessments, scores, deltas, weights, strict=True
        )
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
            curve=assessment.fitted_curves_by_run[diagnostic.run_number],
            residuals=normalised_residuals(
                datasets_by_run[diagnostic.run_number],
                assessment.fitted_curves_by_run[diagnostic.run_number],
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
    return CandidateSummary(
        key=assessment.selection_key,
        title=assessment.template.title,
        global_names=assessment.global_param_names,
        local_names=assessment.local_param_names,
        fixed_names=assessment.fixed_param_names,
        metric_value=score,
        delta=delta,
        weight=weight,
        gate_passed=assessment.residual_gate_passed,
        gate_summary=_gate_summary(assessment.run_diagnostics),
        series_warnings=assessment.series_warnings,
        runs=runs,
        # Only a coupled fit shares a global; a pre-screen fits each run alone.
        parameters=tuple(
            _parameter_row(
                name,
                role,
                fits,
                shared=role is ParameterRole.GLOBAL and not assessment.prescreen_only,
            )
            for name, role in roles
        ),
        prescreen=assessment.prescreen_only,
    )


def _parameter_row(
    name: str, role: ParameterRole, fits: Sequence[FitResult], *, shared: bool
) -> ParameterRow:
    estimates: list[Estimate] = []
    flags: dict[ParameterFlag, None] = {}
    for fit in fits:
        # A failed fit carries no parameters; its value is unknown, not a runaway.
        if name not in fit.parameters:
            estimates.append(Estimate(math.nan, math.nan))
            continue
        error = float(fit.uncertainties.get(name, math.nan))
        parameter = fit.parameters[name]
        estimates.append(Estimate(float(parameter.value), error))
        flags.update(dict.fromkeys(parameter_flags(parameter, error)))
    return ParameterRow(
        name=name,
        role=role,
        values=estimates[0] if shared else tuple(estimates),
        flags=tuple(flags),
    )


def _gate_summary(diagnostics: Sequence[RunResidualDiagnostic]) -> str:
    runs_by_reason: dict[str, list[str]] = {}
    for diagnostic in diagnostics:
        for reason in diagnostic.gate_reasons:
            runs_by_reason.setdefault(reason, []).append(diagnostic.run_label)
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
