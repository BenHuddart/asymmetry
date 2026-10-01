"""The trend objective: which sharing pattern of which template the wizard recommends.

Every candidate here is a rung of a template's sharing ladder
(:mod:`~asymmetry.core.fitting.global_search.sharing_ladder`). The templates
that compete are those that fit about as well as the best one; among their
pre-selected rungs the one whose local parameters trend best wins. Design:
``docs/plans/global-wizard-trend-objective.md`` (D1, D7–D9, D13–D14, D18).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING

from asymmetry.core.fitting.global_search.sharing_ladder import (
    ADEQUACY_SIGMA,
    LadderRung,
    RungVerdict,
)
from asymmetry.core.fitting.model_comparison import CandidateSummary, summarise_candidates
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fitting.trend_quality import CandidateTrend, PassDiagnostic, TraceQuality

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.fit_wizard import SelectionMetric
    from asymmetry.core.fitting.global_fit_wizard import GlobalCandidateAssessment

__all__ = [
    "RESIDUAL_CLUSTER_WARNING",
    "TEMPLATE_BAND",
    "TREND_TIE",
    "CandidateRung",
    "RungFinding",
    "RungSummary",
    "SelectionObjective",
    "band_limit",
    "cost_text",
    "rung_findings",
    "summarise_rungs",
    "templates_within_band",
    "trend_contenders",
    "trend_recommendation",
    "trend_sort_key",
]

#: A template competes on trend quality while its all-local series χ²ᵣ is within
#: this share of the best template's (plan D9, D18).
TEMPLATE_BAND = 0.03
#: Worst trend qualities this close are a tie: the better-fitting template leads.
TREND_TIE = 0.05
#: How the series warning for clustered residual failures opens. The trend
#: summary names the failing runs itself and leaves that warning out.
RESIDUAL_CLUSTER_WARNING = "Residual warnings cluster across runs"


class SelectionObjective(Enum):
    """What the wizard's recommendation optimises (plan D1)."""

    #: The fit whose local parameters trend best among those that fit adequately.
    TREND = "trend"
    #: The best information criterion over the separable role search.
    STATISTICAL = "statistical"


@dataclass(frozen=True)
class CandidateRung(RungVerdict):
    """What being a rung of a sharing ladder adds to a wizard candidate."""

    #: Whether this is the rung its template's ladder pre-selects (plan D9).
    preselected: bool
    #: The end block the template's ladder could not share the amplitudes
    #: through (plan D12). A finding of the whole climb, so every rung of one
    #: ladder carries the same runs; empty when the climb found none.
    amplitude_unshareable_runs: tuple[int, ...]
    #: The template's all-local series χ²ᵣ: what the band compares (plan D9).
    #: The same on every rung of one ladder.
    all_local_chi2r: float

    @classmethod
    def from_ladder_rung(
        cls,
        rung: LadderRung,
        *,
        preselected: bool,
        amplitude_unshareable_runs: tuple[int, ...],
        all_local_chi2r: float,
    ) -> CandidateRung:
        return cls(
            **{item.name: getattr(rung, item.name) for item in fields(RungVerdict)},
            preselected=preselected,
            amplitude_unshareable_runs=amplitude_unshareable_runs,
            all_local_chi2r=all_local_chi2r,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "exempt_runs": list(self.exempt_runs),
            "series_cost": self.series_cost,
            "run_costs": [[run, cost] for run, cost in self.run_costs.items()],
            "trend": {
                name: [quality.determined, quality.signal, quality.zigzag]
                for name, quality in self.trend.parameters.items()
            },
            "hard_to_justify": list(self.hard_to_justify),
            "pass_disagreements": {
                name: {
                    "zigzag_axis_order": diagnostic.zigzag_axis_order,
                    "zigzag_run_order": diagnostic.zigzag_run_order,
                    "passes": [list(runs) for runs in diagnostic.passes],
                }
                for name, diagnostic in self.pass_disagreements.items()
            },
            "preselected": self.preselected,
            "amplitude_unshareable_runs": list(self.amplitude_unshareable_runs),
            "all_local_chi2r": self.all_local_chi2r,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CandidateRung:
        return cls(
            exempt_runs=tuple(int(run) for run in payload["exempt_runs"]),
            series_cost=float(payload["series_cost"]),
            run_costs={int(run): float(cost) for run, cost in payload["run_costs"]},
            trend=CandidateTrend(
                {
                    name: TraceQuality(determined=determined, signal=signal, zigzag=zigzag)
                    for name, (determined, signal, zigzag) in payload["trend"].items()
                }
            ),
            hard_to_justify=tuple(payload["hard_to_justify"]),
            pass_disagreements={
                name: PassDiagnostic(
                    zigzag_axis_order=float(entry["zigzag_axis_order"]),
                    zigzag_run_order=float(entry["zigzag_run_order"]),
                    passes=tuple(tuple(int(run) for run in runs) for runs in entry["passes"]),
                )
                for name, entry in payload["pass_disagreements"].items()
            },
            preselected=bool(payload["preselected"]),
            amplitude_unshareable_runs=tuple(
                int(run) for run in payload["amplitude_unshareable_runs"]
            ),
            all_local_chi2r=float(payload["all_local_chi2r"]),
        )


def templates_within_band(chi2r_by_key: Mapping[str, float]) -> set[str]:
    """Templates whose all-local series χ²ᵣ is within the band of the best one's (plan D9)."""
    if not chi2r_by_key:
        return set()
    limit = (1.0 + TEMPLATE_BAND) * min(chi2r_by_key.values())
    return {key for key, chi2r in chi2r_by_key.items() if chi2r <= limit}


def band_limit(assessments: Sequence[GlobalCandidateAssessment]) -> float:
    """The largest all-local χ²ᵣ a template may have and still contend (plan D9).

    Taken over the ladders that kept a fitted pre-selected rung; infinite when
    none did.
    """
    return (1.0 + TEMPLATE_BAND) * min(
        (
            assessment.rung.all_local_chi2r
            for assessment in assessments
            if assessment.rung.preselected and assessment.is_successful
        ),
        default=math.inf,
    )


def _ranked(
    assessment: GlobalCandidateAssessment, metric: SelectionMetric
) -> tuple[tuple[bool, float, float], bool, float]:
    """Larger is better: trend quality, then easy to justify, then the criterion."""
    rung = assessment.rung
    return (rung.trend.ordering_key, not rung.hard_to_justify, -assessment.metric_value(metric))


def _symbols(names: Sequence[str]) -> str:
    return ", ".join(get_param_info(name).unicode_label(include_unit=False) for name in names)


def _runs_phrase(runs: Sequence[int], labels: Mapping[int, str]) -> str:
    """``"run 7"`` or ``"runs 3, 8–10"``: runs that follow in run number are given as a range."""
    ordered = sorted(runs)
    stretches: list[list[int]] = []
    for run in ordered:
        if stretches and run == stretches[-1][-1] + 1:
            stretches[-1].append(run)
        else:
            stretches.append([run])
    listed = ", ".join(
        labels[stretch[0]] if len(stretch) == 1 else f"{labels[stretch[0]]}–{labels[stretch[-1]]}"
        for stretch in stretches
    )
    return f"{'run' if len(ordered) == 1 else 'runs'} {listed}"


@dataclass(frozen=True)
class RungFinding:
    """Something to know about a rung before choosing it: a short name and one sentence."""

    label: str
    detail: str


def rung_findings(
    assessment: GlobalCandidateAssessment, *, in_band: bool
) -> tuple[RungFinding, ...]:
    """What the ladder found out about a rung, in plain words (plan D3, D9, D12, D15).

    ``in_band`` says whether the rung's template fits about as well as the best
    one (:func:`band_limit`). The end-block finding never asserts missing
    asymmetry: an amplitude that really steps at one end of the series reads
    the same to the ladder.
    """
    rung = assessment.rung
    labels = {
        diagnostic.run_number: diagnostic.run_label for diagnostic in assessment.run_diagnostics
    }
    findings: list[RungFinding] = []
    if rung.exempt_runs:
        exempt = _runs_phrase(rung.exempt_runs, labels)
        findings.append(
            RungFinding(
                f"{exempt.capitalize()} exempt",
                f"{exempt.capitalize()} "
                f"{'keeps its' if len(rung.exempt_runs) == 1 else 'keep their'} own amplitude: "
                "the asymmetry there stands apart from the series.",
            )
        )
    if rung.amplitude_unshareable_runs:
        block = _runs_phrase(rung.amplitude_unshareable_runs, labels)
        findings.append(
            RungFinding(
                f"Amplitude not shared through {block}",
                f"Amplitude could not be shared through {block}: possible missing asymmetry "
                "there, or an amplitude that really changes.",
            )
        )
    if rung.hard_to_justify:
        shared = _symbols(rung.hard_to_justify)
        findings.append(
            RungFinding(
                f"Hard to justify: shares {shared}",
                f"Shares {shared} while the same component's amplitude varies.",
            )
        )
    if rung.pass_disagreements:
        disputed = _symbols(tuple(rung.pass_disagreements))
        findings.append(
            RungFinding(
                f"Passes disagree on {disputed}",
                f"Runs taken in separate passes disagree on {disputed}.",
            )
        )
    if not in_band:
        band = f"{100.0 * TEMPLATE_BAND:g} %"
        findings.append(
            RungFinding(
                f"Outside the {band} band",
                f"Fits worse than the best model by more than {band}; listed for comparison.",
            )
        )
    return tuple(findings)


def cost_text(rung: RungVerdict, labels: Mapping[int, str]) -> str:
    """One sentence on what a rung costs against its template's all-local fits (plan D9)."""
    sentence = (
        f"Sharing raises the series χ²ᵣ by {rung.series_cost:+.1f}σ over the fit with every "
        f"parameter local; the tolerance is {ADEQUACY_SIGMA:g}σ."
    )
    if rung.offending_runs:
        offending = _runs_phrase(rung.offending_runs, labels)
        sentence += (
            f" {offending.capitalize()} "
            f"{'is' if len(rung.offending_runs) == 1 else 'are'} over it on "
            f"{'its' if len(rung.offending_runs) == 1 else 'their'} own."
        )
    return sentence


@dataclass(frozen=True)
class RungSummary(CandidateSummary):
    """A rung of a sharing ladder, as the Compare step lists it (the trend objective).

    Beside what every candidate shows, a rung has a cost against its template's
    all-local fits, the trend quality of what it leaves local, and the ladder's
    findings about it.
    """

    rung: CandidateRung
    #: Every run's coupled fit converged.
    converged: bool
    #: The rung's series χ²ᵣ, over the series' own degrees of freedom.
    chi2r: float
    #: Whether the rung's template fits about as well as the best one.
    in_band: bool
    findings: tuple[RungFinding, ...]

    @property
    def adequate(self) -> bool:
        """Fitted, and within the cost tolerance for the series and every non-exempt run."""
        return self.converged and self.rung.within_tolerance


def summarise_rungs(
    assessments: Sequence[GlobalCandidateAssessment],
    datasets: Sequence[MuonDataset],
    metric: SelectionMetric,
) -> tuple[RungSummary, ...]:
    """One summary per ladder rung, in input order; the band is taken across exactly this pool.

    Every assessment must be a rung (``assessment.rung``), which is what a
    recommendation of the trend objective holds.
    """
    limit = band_limit(assessments)
    return tuple(
        _rung_summary(assessment, summary, in_band=assessment.rung.all_local_chi2r <= limit)
        for assessment, summary in zip(
            assessments, summarise_candidates(assessments, datasets, metric), strict=True
        )
    )


def _rung_summary(
    assessment: GlobalCandidateAssessment, summary: CandidateSummary, *, in_band: bool
) -> RungSummary:
    fits = assessment.fit_results_by_run.values()
    free_names = len(assessment.global_param_names) + len(assessment.local_param_names)
    # ν of a coupled fit: the runs' own, plus one per column the sharing saves.
    dof = sum(fit.dof for fit in fits) + free_names * len(fits) - assessment.parameter_count
    return RungSummary(
        **{item.name: getattr(summary, item.name) for item in fields(CandidateSummary)},
        rung=assessment.rung,
        converged=assessment.is_successful,
        chi2r=sum(fit.chi_squared for fit in fits) / dof,
        in_band=in_band,
        findings=rung_findings(assessment, in_band=in_band),
    )


def trend_contenders(
    assessments: Sequence[GlobalCandidateAssessment], metric: SelectionMetric
) -> list[GlobalCandidateAssessment]:
    """The pre-selected rung of each template inside the band, the best trend first.

    The band is taken over the templates that have a ladder among
    ``assessments``: a template climbed because the data identified it, or
    because the user ticked it, is listed with its rungs but contends only when
    it fits about as well as the best of them. Rungs whose worst trend quality
    is within :data:`TREND_TIE` of the leader's trend equally well: among them
    the rung that is not hard to justify leads, then the template that fits
    best all-local. A rung with no local parameter orders
    below every rung that has one, so it leads only when nothing else is
    adequate.
    """
    rungs = [assessment for assessment in assessments if assessment.rung is not None]
    limit = band_limit(rungs)
    ranked = sorted(
        (
            assessment
            for assessment in rungs
            if assessment.rung.preselected
            and assessment.is_successful
            and assessment.rung.all_local_chi2r <= limit
        ),
        key=lambda assessment: _ranked(assessment, metric),
        reverse=True,
    )
    if not ranked:
        return []
    is_trend, worst, _mean = ranked[0].rung.trend.ordering_key
    level = [
        assessment
        for assessment in ranked
        if assessment.rung.trend.is_trend == is_trend
        and worst - assessment.rung.trend.ordering_key[1] <= TREND_TIE
    ]
    level.sort(
        key=lambda assessment: (
            bool(assessment.rung.hard_to_justify),
            assessment.rung.all_local_chi2r,
        )
    )
    return level + [assessment for assessment in ranked if assessment not in level]


def trend_sort_key(
    assessments: Sequence[GlobalCandidateAssessment], metric: SelectionMetric
) -> Callable[[GlobalCandidateAssessment], tuple]:
    """A key that orders ladder rungs for display, smaller first.

    Template by template: the contending templates in the order of
    :func:`trend_contenders`, so the recommended rung leads, then the templates
    outside the band by how well they fit. Within a template the pre-selected
    rung, then the other adequate rungs by trend quality, then the rungs that
    cost too much by their cost.
    """
    contending = {
        assessment.template.key: (0, float(index))
        for index, assessment in enumerate(trend_contenders(assessments, metric))
    }

    def key(assessment: GlobalCandidateAssessment) -> tuple:
        rung = assessment.rung
        adequate = assessment.is_successful and rung.within_tolerance
        (is_trend, worst, mean), justified, score = _ranked(assessment, metric)
        return (
            contending.get(assessment.template.key, (1, rung.all_local_chi2r)),
            not rung.preselected,
            not adequate,
            (not is_trend, -worst, -mean, not justified, -score)
            if adequate
            else (rung.series_cost,),
        )

    return key


def trend_recommendation(
    assessments: Sequence[GlobalCandidateAssessment], metric: SelectionMetric
) -> tuple[str | None, tuple[str, ...], str]:
    """``(recommended_key, comparable_keys, summary)`` under the trend objective.

    The first of :func:`trend_contenders` is recommended. The second, the best
    pre-selected rung of another template, is the comparable alternative: its
    template is inside the band, so it fits about as well.

    A run that fails its residual gate does not disqualify a rung. Adequacy is
    measured against the same template's all-local fits and the band admits the
    template, so a model that fits imperfectly is still recommended, with the
    runs named in the summary.
    """
    contenders = trend_contenders(assessments, metric)
    if not contenders:
        return (
            None,
            (),
            "No candidate could be fitted on every run, so there is no sharing pattern "
            "to recommend. Inspect the optimized-results table before applying a model.",
        )
    best = contenders[0]
    rung = best.rung
    labels = {diagnostic.run_number: diagnostic.run_label for diagnostic in best.run_diagnostics}
    shared = _symbols(best.global_param_names)
    local = _symbols(tuple(rung.trend.parameters))
    if not shared:
        pattern = f"no parameter shared and {local} varying from run to run"
    elif not local:
        pattern = f"{shared} shared across the series and nothing left to vary from run to run"
    else:
        pattern = f"{shared} shared across the series and {local} varying from run to run"
    sentences = [
        f"Recommended for its trends: {best.template.title}, with {pattern}.",
        *(finding.detail for finding in rung_findings(best, in_band=True)),
    ]
    failing = [
        diagnostic.run_number for diagnostic in best.run_diagnostics if not diagnostic.gate_passed
    ]
    if failing:
        sentences.append(
            f"The model leaves structured residuals on {_runs_phrase(failing, labels)}; "
            "review them before applying."
        )
    sentences.extend(
        warning
        for warning in best.series_warnings
        if not warning.startswith(RESIDUAL_CLUSTER_WARNING)
    )
    if len(contenders) > 1:
        sentences.append(
            f"{contenders[1].template.title} fits about as well and is listed to compare."
        )
    return (
        best.selection_key,
        tuple(assessment.selection_key for assessment in contenders[:2])
        if len(contenders) > 1
        else (),
        " ".join(sentences),
    )
