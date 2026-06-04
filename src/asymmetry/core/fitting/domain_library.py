"""Domain-filtered views over the fit-function registries.

The redesign organises analysis around two domains -- ``"time"`` and
``"frequency"`` -- and each data representation may only be fit with functions
appropriate to its domain.  Rather than duplicating the model/component
registries, this module is the single source of truth for *which* registered
functions belong to *which* domain, and for the default model a fresh
representation should start from.

The frequency component set is seeded from
:data:`asymmetry.core.fitting.spectral.FREQUENCY_COMPONENT_NAMES` so there is
exactly one canonical list of frequency-domain components in the codebase.
"""

from __future__ import annotations

from asymmetry.core.fitting.composite import COMPONENTS, ComponentDefinition, CompositeModel
from asymmetry.core.fitting.models import MODELS, ModelDefinition
from asymmetry.core.fitting.spectral import FREQUENCY_COMPONENT_NAMES, default_frequency_model

#: The analysis domains recognised by the application.
DOMAINS: tuple[str, ...] = ("time", "frequency")


# Guard: every canonical frequency component name must be tagged
# ``domain="frequency"``.  This keeps the domain tags (the filter source of
# truth) and ``FREQUENCY_COMPONENT_NAMES`` from drifting apart.
_missing_frequency_tags = {
    name
    for name in FREQUENCY_COMPONENT_NAMES
    if COMPONENTS.get(name) is None or COMPONENTS[name].domain != "frequency"
}
if _missing_frequency_tags:
    raise RuntimeError(
        "FREQUENCY_COMPONENT_NAMES lists components not tagged domain='frequency': "
        f"{sorted(_missing_frequency_tags)}"
    )


def _normalise_domain(domain: str) -> str:
    """Return *domain* lower-cased, raising on an unknown value."""
    token = str(domain).strip().lower()
    if token not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}.")
    return token


def components_for_domain(domain: str) -> dict[str, ComponentDefinition]:
    """Return the registered composite components for *domain*.

    Parameters
    ----------
    domain : str
        ``"time"`` or ``"frequency"``.

    Returns
    -------
    dict[str, ComponentDefinition]
        Mapping of component name to definition, preserving registry order.
    """
    token = _normalise_domain(domain)
    return {
        name: definition
        for name, definition in COMPONENTS.items()
        if definition.domain == token
    }


def models_for_domain(domain: str) -> dict[str, ModelDefinition]:
    """Return the registered built-in models for *domain*."""
    token = _normalise_domain(domain)
    return {name: definition for name, definition in MODELS.items() if definition.domain == token}


def default_model_for_domain(domain: str) -> CompositeModel:
    """Return the default starting model for a representation in *domain*.

    Time defaults to ``Exponential + Constant``; frequency defaults to the V1
    peak model (``GaussianPeak + ConstantBackground``).
    """
    token = _normalise_domain(domain)
    if token == "frequency":
        return default_frequency_model()
    return CompositeModel(["Exponential", "Constant"], operators=["+"])


__all__ = [
    "DOMAINS",
    "components_for_domain",
    "default_model_for_domain",
    "models_for_domain",
]
