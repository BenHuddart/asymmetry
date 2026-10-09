"""Screen one run against the fit wizard's candidate portfolio.

:func:`screen_run` is the scripted form of the desktop Fit Wizard: it hands one
reduced spectrum to
:func:`~asymmetry.core.fitting.build_fit_wizard_recommendation` and returns the
recommendation, the decision narrative, the ranked candidate table and — the
part the rest of the workflow consumes — a :class:`~asymmetry.core.workflow.recipe.FitRecipe`
built from the recommended candidate's own fitted parameters.

Geometry
--------

The wizard scopes its candidate families by the run's applied-field geometry,
which :func:`~asymmetry.core.fitting.wizard_scope.resolve_scope_for_dataset`
reads from the recorded ``field_direction``/``field_state`` token. Real files
are unreliable there: ISIS stamps ``TF`` on the zero-field runs of a scan that
opened with a weak transverse-field calibration, and some files record nothing
at all. So the geometry is resolved here first and the wizard is handed a
dataset copy whose ``field_direction`` says so, in this order:

1. the caller's explicit override (``"user"``);
2. the geometry the folder's survey resolved for this run (``"survey"``) — which
   may itself have been *measured* from Larmor precession, the one source that
   can speak for a file recording no field state at all;
3. the survey's metadata rule on this dataset alone
   (:func:`~asymmetry.core.workflow.survey.run_geometry`: a recorded field of
   exactly zero means zero field), as ``"field"`` or ``"file"``;
4. nothing (``"none"``).

The survey and the wizard can then never disagree about what a run is, and the
result records which source decided.

Scope names
-----------

A scope name (:data:`SCOPE_PRESETS`) chooses only the physics classes the wizard
looks for; it never chooses a geometry. ``--scope lf-dynamics`` on a zero-field
run screens zero-field dynamics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import PhysicsClass, geometry_from_field_direction
from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.fit_wizard import (
    MIN_CYCLES_IN_EFFECTIVE_WINDOW,
    build_fit_wizard_recommendation,
    effective_window_duration,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.wizard_narrative import render_log_text
from asymmetry.core.fitting.wizard_scope import WizardScope
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.survey import run_geometry

#: A candidate within this many AICc of the recommendation is one the run hardly
#: tells apart from it: the runner-up gets a recipe of its own, for the scan to decide.
#: Only between relaxation models with at most one parameter more — a precession
#: model's alternatives (another envelope) are not a different reading of the scan.
RUNNER_UP_AICC = 10.0

#: Wizard categories that model relaxation alone, and the one that precesses.
_RELAXATION = frozenset({"General", "Multi-rate", "KT-like"})
PRECESSION_CATEGORY = "Oscillatory"


def _runner_up(recommendation: Any, lined: bool) -> Any | None:
    """The relaxation assessment worth a recipe beside the recommended one, if any.

    A close one beside a relaxation recommendation; beside a precession model
    the spectrum shows no line for (*lined* false) — which is fitting a
    relaxation's shape — the best relaxation model, however far behind.
    """
    recommended = recommendation.recommended_assessment
    expression = recommended.template.model.component_expression_string()
    candidates = [
        assessment
        for assessment in sorted(recommendation.assessments, key=lambda item: item.selected_score)
        if assessment.is_successful
        and not assessment.is_disqualified
        and assessment.template.category in _RELAXATION
        and assessment.aicc is not None
        and assessment.template.model.component_expression_string() != expression
    ]
    if recommended.aicc is None:
        return None
    if recommended.template.category in _RELAXATION:
        candidates = [
            assessment
            for assessment in candidates
            if 0.0 <= assessment.aicc - recommended.aicc <= RUNNER_UP_AICC
            and assessment.parameter_count <= recommended.parameter_count + 1
        ]
    elif recommended.template.category != PRECESSION_CATEGORY or lined:
        return None
    return candidates[0] if candidates else None


#: Where a run's geometry came from, in the order the resolution tries them.
GEOMETRY_SOURCES = ("user", "survey", "field", "file", "none")

#: Scope names a caller may give, each a shortcut for the physics classes looked
#: for (empty: every class). Geometry always comes from the run.
SCOPE_PRESETS: dict[str, frozenset[PhysicsClass]] = {
    "auto": frozenset(),
    "zf-static-magnetism": frozenset({PhysicsClass.MAGNETISM}),
    "tf-knight-precession": frozenset({PhysicsClass.MAGNETISM}),
    "tf-superconductor": frozenset({PhysicsClass.SUPERCONDUCTIVITY, PhysicsClass.MAGNETISM}),
    "lf-dynamics": frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM}),
    "fluoride-fmuf": frozenset({PhysicsClass.MOLECULAR}),
    "muonium-radical": frozenset({PhysicsClass.MUONIUM}),
    "all": frozenset(),
}


def _named_geometry(value: str) -> str:
    """``"ZF"``/``"TF"``/``"LF"`` for a geometry token, or :class:`ValueError`."""
    geometry = geometry_from_field_direction(value)
    if geometry is None:
        raise ValueError(f"Unknown geometry {value!r}; expected ZF, TF or LF.")
    return geometry.value


def resolve_geometry(
    dataset: MuonDataset,
    override: str | None,
    survey_geometry: str | None = None,
) -> tuple[str | None, str]:
    """Return this run's ``(geometry, source)`` — see the module docstring.

    *survey_geometry* is the geometry the folder's survey resolved for this run;
    it loses to an explicit *override* and beats this dataset's own metadata.
    Raises :class:`ValueError` for a token that is not a geometry.
    """
    if override is not None:
        return _named_geometry(override), "user"
    if survey_geometry is not None:
        return _named_geometry(survey_geometry), "survey"
    resolved = run_geometry(dataset.metadata)
    if resolved is None:
        return None, "none"
    field = dataset.metadata.get("field")
    if field is not None and float(field) == 0.0:
        return resolved, "field"
    return resolved, "file"


@dataclass(frozen=True)
class ScreenCandidate:
    """One row of the ranked candidate table."""

    key: str
    title: str
    category: str
    aicc: float | None
    chi2_red: float
    parameter_count: int
    is_recommended: bool
    is_comparable: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "key": self.key,
            "title": self.title,
            "category": self.category,
            "aicc": self.aicc,
            "chi2_red": self.chi2_red,
            "parameter_count": self.parameter_count,
            "is_recommended": self.is_recommended,
            "is_comparable": self.is_comparable,
        }


@dataclass(frozen=True)
class ScreenResult:
    """Everything :func:`screen_run` found for one run."""

    run_number: int
    geometry: str | None
    geometry_source: str
    scope_preset: str
    #: Components added to and dropped from the scope.
    scope_include: list[str]
    scope_exclude: list[str]
    scope_note: str
    confidence: str
    verdict: str
    caveat: str
    summary: str
    recommended_key: str | None
    candidates: list[ScreenCandidate]
    narrative: str
    #: :func:`serialize_fit_wizard_recommendation` output, ``compact=True``.
    recommendation: dict[str, Any]
    #: The recipe built from the recommended candidate, or ``None`` when the
    #: wizard made no recommendation (the payload then says so).
    recipe: FitRecipe | None
    #: The recipe of the best relaxation model when it is a close runner-up to a
    #: relaxation recommendation (:data:`RUNNER_UP_AICC`), or when the
    #: recommendation precesses and the spectrum shows no line; else ``None``.
    runner_up: FitRecipe | None
    #: The spectral lines the search detected that complete enough cycles to be precession.
    lines: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "geometry": self.geometry,
            "geometry_source": self.geometry_source,
            "scope_preset": self.scope_preset,
            "scope_include": list(self.scope_include),
            "scope_exclude": list(self.scope_exclude),
            "scope_note": self.scope_note,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "caveat": self.caveat,
            "summary": self.summary,
            "recommended_key": self.recommended_key,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "narrative": self.narrative,
            "recommendation": self.recommendation,
            "recipe": None if self.recipe is None else self.recipe.to_dict(),
            "runner_up": None if self.runner_up is None else self.runner_up.to_dict(),
            "lines": list(self.lines),
        }


def screen_run(
    dataset: MuonDataset,
    *,
    geometry: str | None = None,
    survey_geometry: str | None = None,
    scope_preset: str = "auto",
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    run_number: int,
) -> ScreenResult:
    """Screen *dataset* against the wizard's candidate models.

    *geometry* overrides the file's recorded field direction (``"ZF"``,
    ``"TF"`` or ``"LF"``); *survey_geometry* is what the folder's survey
    resolved for this run, used when the caller gives no override;
    *scope_preset* is one of :data:`SCOPE_PRESETS`, naming the physics looked
    for. *include* and *exclude* name time-domain components to add to or drop
    from the scope (exclude wins). Raises :class:`ValueError` for a value outside any of these
    vocabularies — this is the boundary where they are checked.
    """
    if scope_preset not in SCOPE_PRESETS:
        raise ValueError(
            f"Unknown scope {scope_preset!r}; expected one of {', '.join(SCOPE_PRESETS)}."
        )
    include = frozenset(include)
    exclude = frozenset(exclude)
    unknown = sorted((include | exclude) - set(COMPONENTS))
    if unknown:
        raise ValueError(
            f"Unknown component(s) {', '.join(unknown)}; the wizard's components are "
            f"{', '.join(sorted(COMPONENTS))}."
        )
    resolved_geometry, geometry_source = resolve_geometry(dataset, geometry, survey_geometry)

    # Hand the wizard the geometry this workflow resolved, not the file's raw
    # token, so its scope matches the survey's reading of the same run.
    metadata = dict(dataset.metadata)
    if resolved_geometry is not None:
        metadata["field_direction"] = resolved_geometry
    scoped = replace(dataset, metadata=metadata)

    recommendation = build_fit_wizard_recommendation(
        scoped,
        scope=WizardScope(
            physics=SCOPE_PRESETS[scope_preset],
            include_components=include,
            exclude_components=exclude,
        ),
    )

    comparable = set(recommendation.comparable_keys)
    candidates = [
        ScreenCandidate(
            key=assessment.template.key,
            title=assessment.template.title,
            category=assessment.template.category,
            aicc=None if assessment.aicc is None else float(assessment.aicc),
            chi2_red=float(assessment.fit_result.reduced_chi_squared),
            parameter_count=int(assessment.parameter_count),
            is_recommended=assessment.template.key == recommendation.recommended_key,
            is_comparable=assessment.template.key in comparable,
        )
        for assessment in sorted(recommendation.assessments, key=lambda item: item.selected_score)
    ]

    recommended = recommendation.recommended_assessment
    recipe = (
        None
        if recommended is None
        else FitRecipe.from_assessment(recommended, run_number=run_number, seed_field=dataset.field)
    )
    serialized = serialize_fit_wizard_recommendation(recommendation, compact=True)
    # A "line" completing too few cycles in the informative window is relaxation
    # leaking into the lowest bins, not precession (the survey's rule too).
    duration = effective_window_duration(scoped)
    lines = [
        peak
        for peak in serialized["peak_analysis"]["peaks"]
        if peak["frequency_mhz"] * duration >= MIN_CYCLES_IN_EFFECTIVE_WINDOW
    ]
    alternative = None if recommended is None else _runner_up(recommendation, bool(lines))
    runner_up = (
        None
        if alternative is None
        else FitRecipe.from_assessment(alternative, run_number=run_number, seed_field=dataset.field)
    )

    return ScreenResult(
        run_number=int(run_number),
        geometry=resolved_geometry,
        geometry_source=geometry_source,
        scope_preset=scope_preset,
        scope_include=sorted(include),
        scope_exclude=sorted(exclude),
        scope_note=recommendation.scope_note,
        confidence=recommendation.confidence.value,
        verdict=recommendation.verdict.value,
        caveat=recommendation.caveat,
        summary=recommendation.summary,
        recommended_key=recommendation.recommended_key,
        candidates=candidates,
        narrative=render_log_text(recommendation),
        recommendation=serialized,
        lines=lines,
        recipe=recipe,
        runner_up=runner_up,
    )


__all__ = [
    "GEOMETRY_SOURCES",
    "PRECESSION_CATEGORY",
    "RUNNER_UP_AICC",
    "SCOPE_PRESETS",
    "ScreenCandidate",
    "ScreenResult",
    "resolve_geometry",
    "screen_run",
]
