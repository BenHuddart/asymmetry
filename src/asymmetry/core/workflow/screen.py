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
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import geometry_from_field_direction
from asymmetry.core.fitting.fit_wizard import (
    build_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.wizard_narrative import render_log_text
from asymmetry.core.fitting.wizard_scope import WizardScope, WizardScopePreset
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.survey import run_geometry

#: Where a run's geometry came from, in the order the resolution tries them.
GEOMETRY_SOURCES = ("user", "survey", "field", "file", "none")

#: Scope presets a caller may name (the fit wizard's own vocabulary).
SCOPE_PRESETS = tuple(preset.value for preset in WizardScopePreset)


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "geometry": self.geometry,
            "geometry_source": self.geometry_source,
            "scope_preset": self.scope_preset,
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
        }


def screen_run(
    dataset: MuonDataset,
    *,
    geometry: str | None = None,
    survey_geometry: str | None = None,
    scope_preset: str = WizardScopePreset.AUTO.value,
    run_number: int,
) -> ScreenResult:
    """Screen *dataset* against the wizard's candidate models.

    *geometry* overrides the file's recorded field direction (``"ZF"``,
    ``"TF"`` or ``"LF"``); *survey_geometry* is what the folder's survey
    resolved for this run, used when the caller gives no override;
    *scope_preset* is one of :data:`SCOPE_PRESETS`. Raises :class:`ValueError`
    for a value outside either vocabulary — this is the boundary where that
    vocabulary is checked.
    """
    preset = WizardScopePreset(scope_preset)
    resolved_geometry, geometry_source = resolve_geometry(dataset, geometry, survey_geometry)

    # Hand the wizard the geometry this workflow resolved, not the file's raw
    # token, so its scope matches the survey's reading of the same run.
    metadata = dict(dataset.metadata)
    if resolved_geometry is not None:
        metadata["field_direction"] = resolved_geometry
    scoped = replace(dataset, metadata=metadata)

    recommendation = build_fit_wizard_recommendation(scoped, scope=WizardScope(preset=preset))

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
        else FitRecipe.from_assessment(recommended, run_number=run_number)
    )

    return ScreenResult(
        run_number=int(run_number),
        geometry=resolved_geometry,
        geometry_source=geometry_source,
        scope_preset=preset.value,
        scope_note=recommendation.scope_note,
        confidence=recommendation.confidence.value,
        verdict=recommendation.verdict.value,
        caveat=recommendation.caveat,
        summary=recommendation.summary,
        recommended_key=recommendation.recommended_key,
        candidates=candidates,
        narrative=render_log_text(recommendation),
        recommendation=serialize_fit_wizard_recommendation(recommendation, compact=True),
        recipe=recipe,
    )


__all__ = [
    "GEOMETRY_SOURCES",
    "SCOPE_PRESETS",
    "ScreenCandidate",
    "ScreenResult",
    "resolve_geometry",
    "screen_run",
]
