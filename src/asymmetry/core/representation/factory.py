"""Declarative registry mapping representation types to their classes."""

from __future__ import annotations

from asymmetry.core.representation.base import FitSlot, Representation, RepresentationType
from asymmetry.core.representation.frequency import FrequencyFFT, FrequencyMaxEnt
from asymmetry.core.representation.time import (
    TimeFBAsymmetry,
    TimeGroups,
    TimeMaxEntReconstruction,
)

#: The single declarative source of truth for representation construction.
REPRESENTATION_REGISTRY: dict[RepresentationType, type[Representation]] = {
    RepresentationType.TIME_FB_ASYMMETRY: TimeFBAsymmetry,
    RepresentationType.TIME_GROUPS: TimeGroups,
    RepresentationType.TIME_MAXENT_RECON: TimeMaxEntReconstruction,
    RepresentationType.FREQ_FFT: FrequencyFFT,
    RepresentationType.FREQ_MAXENT: FrequencyMaxEnt,
}


def _coerce_type(rep_type: RepresentationType | str) -> RepresentationType:
    if isinstance(rep_type, RepresentationType):
        return rep_type
    return RepresentationType(str(rep_type))


def make_representation(
    rep_type: RepresentationType | str,
    recipe: dict | None = None,
    *,
    fit: FitSlot | None = None,
    trend_state: dict | None = None,
    result_metadata: dict | None = None,
) -> Representation:
    """Construct a representation of *rep_type* with the given recipe/state."""
    resolved = _coerce_type(rep_type)
    cls = REPRESENTATION_REGISTRY[resolved]
    return cls(
        recipe=recipe,
        fit=fit,
        trend_state=trend_state,
        result_metadata=result_metadata,
    )


def representation_from_dict(data: dict) -> Representation:
    """Reconstruct a representation from its :meth:`Representation.to_dict` form."""
    if not isinstance(data, dict):
        raise ValueError("Representation data must be a dict.")
    rep_type = _coerce_type(data["rep_type"])
    return make_representation(
        rep_type,
        recipe=data.get("recipe"),
        fit=FitSlot.from_dict(data.get("fit")),
        trend_state=data.get("trend_state"),
        result_metadata=data.get("result_metadata"),
    )


__all__ = [
    "REPRESENTATION_REGISTRY",
    "make_representation",
    "representation_from_dict",
]
