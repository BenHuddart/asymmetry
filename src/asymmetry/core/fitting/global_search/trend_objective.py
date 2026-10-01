"""The trend objective: which sharing pattern of which template the wizard recommends.

Every candidate here is a rung of a template's sharing ladder
(:mod:`~asymmetry.core.fitting.global_search.sharing_ladder`). The templates
that compete are those that fit about as well as the best one; among their
pre-selected rungs the one whose local parameters trend best wins. Design:
``docs/plans/global-wizard-trend-objective.md`` (D1, D7–D9, D13–D14, D18).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING

from asymmetry.core.fitting.global_search.sharing_ladder import LadderRung, RungVerdict
from asymmetry.core.fitting.trend_quality import CandidateTrend, PassDiagnostic, TraceQuality

if TYPE_CHECKING:
    from asymmetry.core.fitting.fit_wizard import SelectionMetric
    from asymmetry.core.fitting.global_fit_wizard import GlobalCandidateAssessment

__all__ = [
    "TEMPLATE_BAND",
    "CandidateRung",
    "SelectionObjective",
    "templates_within_band",
    "trend_contenders",
    "trend_recommendation",
    "trend_sort_key",
]

#: A template competes on trend quality while its all-local series χ²ᵣ is within
#: this share of the best template's (plan D9, D18).
TEMPLATE_BAND = 0.03


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


def _ranked(
    assessment: GlobalCandidateAssessment, metric: SelectionMetric
) -> tuple[tuple[bool, float, float], bool, float]:
    """Larger is better: trend quality, then easy to justify, then the criterion."""
    rung = assessment.rung
    return (rung.trend.ordering_key, not rung.hard_to_justify, -assessment.metric_value(metric))


def _run_list(runs: Sequence[int], labels: Mapping[int, str]) -> str:
    return ", ".join(labels[run] for run in runs)


def trend_contenders(
    assessments: Sequence[GlobalCandidateAssessment], metric: SelectionMetric
) -> list[GlobalCandidateAssessment]:
    """The pre-selected rung of each template inside the band, the best trend first.

    The band is taken over the templates that have a ladder among
    ``assessments``: a template climbed because the data identified it, or
    because the user ticked it, is listed with its rungs but contends only when
    it fits about as well as the best of them. Ties go to the rung that is not
    hard to justify, then to ``metric``. A rung with no local parameter orders
    below every rung that has one, so it leads only when nothing else is
    adequate.
    """
    preselected = [
        assessment
        for assessment in assessments
        if assessment.rung is not None and assessment.rung.preselected and assessment.is_successful
    ]
    band = templates_within_band(
        {assessment.template.key: assessment.rung.all_local_chi2r for assessment in preselected}
    )
    return sorted(
        (assessment for assessment in preselected if assessment.template.key in band),
        key=lambda assessment: _ranked(assessment, metric),
        reverse=True,
    )


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
    shared = ", ".join(best.global_param_names)
    local = ", ".join(rung.trend.parameters)
    if not shared:
        pattern = f"no parameter shared and {local} varying from run to run"
    elif not local:
        pattern = f"{shared} shared across the series and nothing left to vary from run to run"
    else:
        pattern = f"{shared} shared across the series and {local} varying from run to run"
    sentences = [f"Recommended for its trends: {best.template.title}, with {pattern}."]
    if rung.exempt_runs:
        sentences.append(
            f"Runs {_run_list(rung.exempt_runs, labels)} keep their own "
            f"{', '.join(best.exemptions)}: their asymmetry stands apart from the series."
        )
    if rung.amplitude_unshareable_runs:
        sentences.append(
            "The amplitude could not be shared through runs "
            f"{_run_list(rung.amplitude_unshareable_runs, labels)} at one end of the series: "
            "possible missing asymmetry there, or an amplitude that really changes."
        )
    if rung.hard_to_justify:
        sentences.append(
            f"Sharing {', '.join(rung.hard_to_justify)} while the amplitude of the same "
            "component varies is hard to justify."
        )
    if rung.pass_disagreements:
        sentences.append(
            f"Runs taken in separate passes disagree on {', '.join(rung.pass_disagreements)}."
        )
    failing = [
        diagnostic.run_label for diagnostic in best.run_diagnostics if not diagnostic.gate_passed
    ]
    if failing:
        sentences.append(
            f"The model leaves structured residuals on runs {', '.join(failing)}; "
            "review them before applying."
        )
    if best.series_warnings:
        sentences.append(f"The series as a whole was flagged: {' '.join(best.series_warnings)}")
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
