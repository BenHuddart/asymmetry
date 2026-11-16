"""Parameter expectation rows and authored bounds for global wizard setup."""

import numpy as np

from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.fit_wizard import CandidateTemplate
from asymmetry.core.fitting.parameters import get_param_info


def _portfolio_parameter_usage(
    templates: tuple[CandidateTemplate, ...],
) -> tuple[list[str], dict[str, list[str]]]:
    ordered_names: list[str] = []
    usage_by_name: dict[str, list[str]] = {}
    seen: set[str] = set()
    for template in templates:
        for name in template.model.param_names:
            usage_by_name.setdefault(name, [])
            if template.title not in usage_by_name[name]:
                usage_by_name[name].append(template.title)
            if name in seen:
                continue
            seen.add(name)
            ordered_names.append(name)
    return ordered_names, usage_by_name


#: Kinds a series is expected to hold steady, and kinds it is expected to move.
_GLOBAL_BY_DEFAULT = frozenset({ParameterKind.BACKGROUND, ParameterKind.AMPLITUDE})
_LOCAL_BY_DEFAULT = frozenset(
    {
        ParameterKind.RATE,
        ParameterKind.FREQUENCY,
        ParameterKind.STATIC_WIDTH,
        ParameterKind.SHAPE,
        ParameterKind.PHASE,
    }
)
#: Kinds that start bounded below at zero, whatever bounds the Fit tab holds.
_NON_NEGATIVE = (_LOCAL_BY_DEFAULT - {ParameterKind.PHASE}) | {ParameterKind.AMPLITUDE}


def _default_parameter_role(
    name: str,
    kind: ParameterKind,
    *,
    current_parameter_types: dict[str, str],
) -> str:
    current = str(current_parameter_types.get(name, "")).strip()
    if current == "Fixed":
        return "Fixed"
    if kind in _GLOBAL_BY_DEFAULT:
        return "Global"
    if kind in _LOCAL_BY_DEFAULT:
        return "Local"
    if current in {"Global", "Local"}:
        return current
    return "Global"


def _default_parameter_bounds(
    name: str,
    kind: ParameterKind,
    *,
    current_parameter_bounds: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    if kind is ParameterKind.BACKGROUND:
        return -float("inf"), float("inf")
    if kind in _NON_NEGATIVE:
        return 0.0, float("inf")
    if name in current_parameter_bounds:
        return current_parameter_bounds[name]
    default_min = get_param_info(name).default_min
    return (
        (float(default_min), float("inf"))
        if default_min is not None
        else (-float("inf"), float("inf"))
    )


def _format_bounds_text(bounds: tuple[float, float]) -> str:
    return f"{_format_bound_value(bounds[0])}, {_format_bound_value(bounds[1])}"


def _format_bound_value(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == -float("inf"):
        return "-inf"
    return f"{float(value):.6g}"


def _parse_bounds_text(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("bounds must be written as 'min, max'")
    min_val = _parse_bound_value(parts[0])
    max_val = _parse_bound_value(parts[1])
    if np.isfinite(min_val) and np.isfinite(max_val) and min_val > max_val:
        raise ValueError(f"invalid bounds: {min_val} > {max_val}")
    return min_val, max_val


def _parse_bound_value(text: str) -> float:
    lowered = text.strip().lower()
    if lowered == "-inf":
        return -float("inf")
    if lowered == "inf":
        return float("inf")
    try:
        value = float(lowered)
    except ValueError as exc:
        raise ValueError(f"could not parse bound '{text}'") from exc
    if not np.isfinite(value):
        raise ValueError(f"bound '{text}' must be finite or +/-inf")
    return value
