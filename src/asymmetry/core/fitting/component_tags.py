"""Scoping tags for fit components: field geometry, physics class, and cost.

These enums annotate every built-in :class:`~asymmetry.core.fitting.composite.ComponentDefinition`
with the experimental context in which it is physically meaningful (applied-field
geometry), the physics regime it belongs to (for wizard scoping), and a coarse
evaluation-cost hint (for tiered wizard screening). The tags let the fit wizard
narrow the candidate set to components that make sense for a given run before it
spends time trialling fits.

Each enum is a ``str`` mixin so its members serialise as plain strings in
``.asymp`` projects and JSON without a custom encoder. This module imports only
the standard library, keeping it usable from the Qt-free core.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class FieldGeometry(str, Enum):
    """Applied-field geometry a component is physically meaningful for."""

    ZF = "ZF"
    TF = "TF"
    LF = "LF"


#: All three geometries — the default for a component with no geometry restriction.
ALL_GEOMETRIES: frozenset[FieldGeometry] = frozenset(FieldGeometry)


class PhysicsClass(str, Enum):
    """Physics regime a component belongs to, used to scope the fit wizard."""

    MAGNETISM = "magnetism"
    SUPERCONDUCTIVITY = "superconductivity"
    MOLECULAR = "molecular"  # F-mu-F / nuclear-dipolar entangled states
    MUONIUM = "muonium"  # muonium / radical states
    DYNAMICS = "dynamics"  # fluctuations / diffusion / motional narrowing
    GENERIC_RELAXATION = "generic-relaxation"
    BACKGROUND = "background"
    SPECTRAL = "spectral"  # frequency-domain lineshapes
    CUSTOM = "custom"  # sentinel default for user functions; no built-in may carry it


class ComputationalCost(str, Enum):
    """Relative evaluation-cost hint for tiered wizard screening."""

    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"


def _one_or_many(values: object) -> Iterable[object]:
    """Normalise a bare string/enum member into a one-element iterable.

    A bare ``str`` is iterable character-by-character, which would make
    ``coerce_geometries("ZF")`` fail on the char ``'Z'``. Treat a single string
    or enum member as a one-element collection; leave other iterables alone.
    """
    if isinstance(values, (str, Enum)):
        return (values,)
    if isinstance(values, Iterable):
        return values
    raise ValueError(f"expected a string, enum member, or iterable of them, got {values!r}")


def coerce_geometries(values: object) -> frozenset[FieldGeometry]:
    """Coerce strings/enum members into a ``frozenset`` of :class:`FieldGeometry`.

    Accepts a bare token or an iterable of them. Raises :class:`ValueError`
    naming the offending token on the first unrecognised value.
    """
    result: set[FieldGeometry] = set()
    for value in _one_or_many(values):
        try:
            result.add(FieldGeometry(value))
        except ValueError as exc:
            raise ValueError(f"unknown field geometry {value!r}") from exc
    return frozenset(result)


def coerce_physics_classes(values: object) -> frozenset[PhysicsClass]:
    """Coerce strings/enum members into a ``frozenset`` of :class:`PhysicsClass`.

    Accepts a bare token or an iterable of them. Raises :class:`ValueError`
    naming the offending token on the first unrecognised value.
    """
    result: set[PhysicsClass] = set()
    for value in _one_or_many(values):
        try:
            result.add(PhysicsClass(value))
        except ValueError as exc:
            raise ValueError(f"unknown physics class {value!r}") from exc
    return frozenset(result)


def coerce_cost(value: object) -> ComputationalCost:
    """Coerce a string/enum member into a :class:`ComputationalCost`.

    Raises :class:`ValueError` naming the offending token on a bad value.
    """
    try:
        return ComputationalCost(value)
    except ValueError as exc:
        raise ValueError(f"unknown computational cost {value!r}") from exc


#: Case-insensitive lookup from the loader field-direction vocabulary (see
#: :func:`asymmetry.core.io.base.field_direction_from_text`) and the raw
#: ``field_state`` tokens onto a :class:`FieldGeometry`.
_FIELD_DIRECTION_TO_GEOMETRY: dict[str, FieldGeometry] = {
    "transverse": FieldGeometry.TF,
    "tf": FieldGeometry.TF,
    "longitudinal": FieldGeometry.LF,
    "lf": FieldGeometry.LF,
    "zero field": FieldGeometry.ZF,
    "zf": FieldGeometry.ZF,
}


def geometry_from_field_direction(text: str) -> FieldGeometry | None:
    """Map a loader field-direction label onto a :class:`FieldGeometry`.

    Accepts the loader word forms ``"Transverse"``/``"Longitudinal"``/
    ``"Zero field"`` (see :func:`asymmetry.core.io.base.field_direction_from_text`)
    and the raw ``"TF"``/``"LF"``/``"ZF"`` tokens carried in
    ``metadata["field_state"]``. Matching is case-insensitive on the word forms
    and tokens alike. Returns ``None`` for an empty string or any unrecognised
    value (geometry unknown → no restriction).
    """
    return _FIELD_DIRECTION_TO_GEOMETRY.get(str(text).strip().lower())
