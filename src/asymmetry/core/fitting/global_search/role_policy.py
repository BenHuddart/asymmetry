"""How readily the role search lets a parameter of each kind vary run by run."""

from __future__ import annotations

from asymmetry.core.fitting.component_tags import ParameterKind

#: Penalty class for making a parameter Local: the higher, the stronger the
#: evidence needed. Rates and frequencies are what a series is expected to
#: move; the record's scale and baseline are expected to stay put.
_LOCALISATION_PRIORITY: dict[ParameterKind, int] = {
    ParameterKind.RATE: 0,
    ParameterKind.FREQUENCY: 0,
    ParameterKind.PHASE: 1,
    ParameterKind.SHAPE: 1,
    ParameterKind.STATIC_WIDTH: 1,
    ParameterKind.FIELD: 1,
    ParameterKind.GEOMETRY: 1,
    ParameterKind.FRACTION: 1,
    ParameterKind.AMPLITUDE: 3,
    ParameterKind.BACKGROUND: 4,
}

#: Factor on the staged deviation threshold, by penalty class.
_THRESHOLD_SCALE: dict[int, float] = {0: 1.0, 1: 1.25, 3: 2.0, 4: 3.0}


def localisation_priorities(kinds: dict[str, ParameterKind]) -> dict[str, int]:
    """Return each named parameter's penalty class (see ``CompositeModel.parameter_kinds``)."""
    return {name: _LOCALISATION_PRIORITY[kind] for name, kind in kinds.items()}


def localisation_threshold_scale(kind: ParameterKind) -> float:
    """Return how much stronger the staged evidence must be to localize this kind."""
    return _THRESHOLD_SCALE[_LOCALISATION_PRIORITY[kind]]


def allows_rate_first_localization(kind: ParameterKind) -> bool:
    """Return whether the staged search may localize this kind in its first pass."""
    return _LOCALISATION_PRIORITY[kind] == 0
