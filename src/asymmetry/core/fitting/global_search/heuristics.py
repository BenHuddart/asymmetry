"""Physics-aware heuristics for staged global-search role decisions."""

from __future__ import annotations


def is_background_parameter(name: str) -> bool:
    """Return whether a parameter looks like a background term."""
    lower_name = name.lower()
    return lower_name in {"a_bg", "bg", "background"} or "background" in lower_name


def is_amplitude_parameter(name: str) -> bool:
    """Return whether a parameter looks like an amplitude term."""
    lower_name = name.lower()
    return lower_name.startswith("a_") or lower_name in {"a", "amplitude"}


def is_rate_like_parameter(name: str) -> bool:
    """Return whether a parameter looks like a relaxation/rate/shape term."""
    lower_name = name.lower()
    return any(
        token in lower_name
        for token in (
            "lambda",
            "sigma",
            "delta",
            "beta",
            "nu",
            "freq",
            "phase",
            "tau",
            "width",
            "rate",
        )
    )


def allows_rate_first_localization(name: str) -> bool:
    """Return whether staged_v1 should localize this parameter in the first pass."""
    if is_background_parameter(name):
        return False
    if is_amplitude_parameter(name):
        return False
    return is_rate_like_parameter(name)


def parameter_localisation_priority(name: str) -> int:
    """Return a heuristic penalty class for localizing one parameter."""
    if is_background_parameter(name):
        return 4
    if is_amplitude_parameter(name):
        return 3
    if is_rate_like_parameter(name):
        return 0
    return 1


def localisation_threshold_scale(name: str) -> float:
    """Return how much stronger the staged evidence must be to localize a parameter."""
    priority = parameter_localisation_priority(name)
    if priority >= 4:
        return 3.0
    if priority >= 3:
        return 2.0
    if priority >= 1:
        return 1.25
    return 1.0
