"""Public registration facade for user-defined fit functions.

This module is the single supported way for user code (plugin files in
``~/.asymmetry/user_functions/``, packaged plugins, or interactive scripts) to
add fit functions to Asymmetry's registries:

* :func:`register_component` — time- or frequency-domain composite components
  (the functions offered by the fit-function builder pickers);
* :func:`register_parameter_component` — parameter-vs-x trend components.

The facade enforces the registry naming rules (N4, ``docs/ARCHITECTURE.md``
§4.3): a registered name must be a bare expression-grammar atom, must carry an
explicit analysis domain, and must be unique across **all** fit-function
registries (``COMPONENTS``, ``MODELS``, ``PARAMETER_MODEL_COMPONENTS``), so a
name registered here identifies its registry and domain unambiguously.

Every function is validated at registration — signature arity, vectorised
evaluation, and finite output on a probe grid at the default parameter values
— so a broken plugin fails with a clear :class:`UserFunctionError` at load
time, never mid-fit. Validation happens before any registry mutation, so a
failed registration leaves all registries untouched.

Trust model: user functions are ordinary Python imported into the running
process, with full interpreter privileges — the same trust model as WiMDA's
plugin DLLs. Only install plugin files you trust.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from asymmetry.core.fitting.component_docs import register_component_documentation
from asymmetry.core.fitting.composite import COMPONENTS, ComponentDefinition
from asymmetry.core.fitting.domain_library import DOMAINS
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    SCOPES,
    ParameterModelComponentDefinition,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fitting.registration import (
    REGISTRY_NAME_RE,
    RESERVED_NAMES,
    insert_definition,
)

__all__ = [
    "UserFunctionError",
    "register_component",
    "register_parameter_component",
]


class UserFunctionError(ValueError):
    """A user-function registration failed validation.

    Raised at registration (load) time with a message naming the offending
    function and rule, so plugin authors get actionable errors instead of
    mid-fit failures.
    """


#: Probe grids for load-time validation. Time includes t = 0 (catches 1/t
#: singularities) and late times (catches overflow); the parameter-model grid
#: starts just above 0 so legitimate x → 0 divergences (e.g. Curie-Weiss-like
#: trends) are not rejected.
_PROBE_GRIDS: dict[str, np.ndarray] = {
    "time": np.linspace(0.0, 32.0, 257),
    "frequency": np.linspace(0.0, 50.0, 257),
    "parameter": np.linspace(1e-3, 300.0, 257),
}

_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")

#: Collector hooked by :mod:`asymmetry.core.plugins` during discovery so the
#: load report can attribute registrations to their source file/entry point.
#: Entries are ``(kind, name)`` with kind ``"component"`` or
#: ``"parameter_component"``.
_active_collector: list[tuple[str, str]] | None = None


def _record_registration(kind: str, name: str) -> None:
    if _active_collector is not None:
        _active_collector.append((kind, name))


def _all_registered_names() -> dict[str, str]:
    """Return every registered fit-function name mapped to a registry label."""
    names: dict[str, str] = {}
    for label, registry in (
        ("a fit component", COMPONENTS),
        ("a built-in model", MODELS),
        ("a parameter-trend component", PARAMETER_MODEL_COMPONENTS),
    ):
        for name in registry:
            names.setdefault(name, label)
    return names


def _check_name(name: object) -> str:
    """Apply the N4 name rules: grammar atom, not reserved, globally unique."""
    if not isinstance(name, str) or not REGISTRY_NAME_RE.fullmatch(name):
        raise UserFunctionError(
            f"Invalid component name {name!r}: names must match "
            "[A-Za-z_][A-Za-z0-9_]* so they can appear in fit expressions."
        )
    if name in RESERVED_NAMES:
        raise UserFunctionError(f"Invalid component name {name!r}: reserved grammar token.")
    existing = _all_registered_names()
    if name in existing:
        raise UserFunctionError(
            f"Component name {name!r} is already registered as {existing[name]}. "
            "Names must be unique across all fit-function registries; pick a "
            "distinct name for the user function."
        )
    return name


def _check_params(
    name: str,
    param_names: Sequence[str],
    param_defaults: Mapping[str, float] | None,
) -> tuple[list[str], dict[str, float]]:
    """Validate parameter names/defaults, filling missing defaults with 1.0."""
    if isinstance(param_names, str) or not isinstance(param_names, Sequence):
        raise UserFunctionError(
            f"{name}: param_names must be a sequence of parameter-name strings."
        )
    params = [str(p) for p in param_names]
    for pname in params:
        if not REGISTRY_NAME_RE.fullmatch(pname):
            raise UserFunctionError(
                f"{name}: invalid parameter name {pname!r} (must be a valid identifier)."
            )
    if len(set(params)) != len(params):
        raise UserFunctionError(f"{name}: duplicate parameter names in {params}.")

    defaults_in = dict(param_defaults or {})
    unknown = sorted(set(defaults_in) - set(params))
    if unknown:
        raise UserFunctionError(
            f"{name}: param_defaults has entries for unknown parameter(s) {unknown}."
        )
    defaults: dict[str, float] = {}
    for pname in params:
        try:
            defaults[pname] = float(defaults_in.get(pname, 1.0))
        except (TypeError, ValueError) as exc:
            raise UserFunctionError(
                f"{name}: default for parameter '{pname}' is not a number "
                f"({defaults_in.get(pname)!r})."
            ) from exc
    return params, defaults


def _check_metadata(name: str, description: object, formula_template: object) -> None:
    if not isinstance(description, str) or not description.strip():
        raise UserFunctionError(f"{name}: a non-empty description is required.")
    if not isinstance(formula_template, str) or not formula_template.strip():
        raise UserFunctionError(f"{name}: a non-empty formula_template is required.")


def _check_formula_template(name: str, formula_template: str, params: Sequence[str]) -> None:
    """Every ``{placeholder}`` in the template must name a declared parameter."""
    stray = sorted(
        {
            token
            for token in _TEMPLATE_PLACEHOLDER_RE.findall(formula_template)
            if token not in params
        }
    )
    if stray:
        raise UserFunctionError(
            f"{name}: formula_template placeholders {stray} are not declared "
            f"parameters {list(params)}."
        )


def _probe_function(
    name: str,
    function: Callable[..., object],
    defaults: Mapping[str, float],
    grid: np.ndarray,
) -> None:
    """Call *function* on the probe grid at defaults; require finite ndarray output."""
    if not callable(function):
        raise UserFunctionError(f"{name}: function is not callable.")
    try:
        with np.errstate(all="ignore"):
            out = function(grid, **dict(defaults))
    except Exception as exc:
        raise UserFunctionError(
            f"{name}: probe evaluation failed — the function must be vectorised, "
            f"accepting an x array plus parameters {sorted(defaults)} as keyword "
            f"arguments ({type(exc).__name__}: {exc})."
        ) from exc
    arr = np.asarray(out)
    if not np.issubdtype(arr.dtype, np.number):
        raise UserFunctionError(
            f"{name}: probe evaluation returned non-numeric output of dtype {arr.dtype}."
        )
    if arr.shape != grid.shape:
        raise UserFunctionError(
            f"{name}: probe evaluation returned shape {arr.shape} for input shape "
            f"{grid.shape} — the function must be vectorised over x."
        )
    if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
        raise UserFunctionError(
            f"{name}: probe evaluation produced non-finite values (NaN/Inf) at the "
            "default parameter values; fix the function or its defaults."
        )


def register_component(
    name: str,
    function: Callable[..., np.ndarray],
    param_names: Sequence[str],
    *,
    domain: str,
    description: str,
    formula_template: str,
    param_defaults: Mapping[str, float] | None = None,
    latex_equation: str = "",
    applicability: str = "",
    references: Iterable[str] = (),
    category: str = "User",
    fixed_params: Sequence[str] = (),
) -> ComponentDefinition:
    """Register a user fit component for the time- or frequency-domain pickers.

    ``function`` must be vectorised: called as ``function(x, **params)`` with
    ``x`` an ndarray (time in µs or frequency in MHz, by ``domain``) and one
    keyword per entry of ``param_names``, returning an ndarray of the same
    shape. ``domain`` is required (``"time"`` or ``"frequency"``) — see the
    registry-naming note in ``docs/ARCHITECTURE.md`` §4.3.

    Returns the registered :class:`ComponentDefinition` (flagged ``user=True``).
    Raises :class:`UserFunctionError` on any validation failure, in which case
    no registry is modified.
    """
    _check_name(name)
    if not isinstance(domain, str) or domain.strip().lower() not in DOMAINS:
        raise UserFunctionError(
            f"{name}: a valid domain is required (one of {DOMAINS}); got {domain!r}. "
            "The domain places the component in the matching picker and plots."
        )
    domain_token = domain.strip().lower()
    _check_metadata(name, description, formula_template)
    params, defaults = _check_params(name, param_names, param_defaults)
    _check_formula_template(name, formula_template, params)
    if not isinstance(category, str) or not category.strip():
        raise UserFunctionError(f"{name}: category must be a non-empty string.")
    fixed = tuple(str(p) for p in fixed_params)
    unknown_fixed = sorted(set(fixed) - set(params))
    if unknown_fixed:
        raise UserFunctionError(
            f"{name}: fixed_params {unknown_fixed} are not declared parameters."
        )
    _probe_function(name, function, defaults, _PROBE_GRIDS[domain_token])

    definition = ComponentDefinition(
        name=name,
        description=str(description),
        function=function,
        param_names=params,
        param_defaults=defaults,
        param_info={p: get_param_info(p) for p in params},
        formula_template=str(formula_template),
        latex_equation=str(latex_equation),
        category=str(category),
        domain=domain_token,
        fixed_params=fixed,
        user=True,
    )
    insert_definition(COMPONENTS, definition, registry_label="COMPONENTS")
    register_component_documentation(
        name,
        kind="fit",
        applicability=str(applicability),
        references=tuple(str(ref) for ref in references),
    )
    _record_registration("component", name)
    return definition


def register_parameter_component(
    name: str,
    function: Callable[..., np.ndarray],
    param_names: Sequence[str],
    *,
    description: str,
    formula_template: str,
    param_defaults: Mapping[str, float] | None = None,
    latex_equation: str = "",
    applicability: str = "",
    references: Iterable[str] = (),
    scopes: Sequence[str] = ("common",),
    fwhm_factor: float | None = None,
) -> ParameterModelComponentDefinition:
    """Register a user parameter-vs-x trend component.

    ``function`` must be vectorised: called as ``function(x, **params)`` with
    ``x`` the trend variable (temperature, field, …) and one keyword per entry
    of ``param_names``. ``scopes`` restricts where the component is offered
    (subset of ``{"common", "field", "temperature"}``).

    Returns the registered :class:`ParameterModelComponentDefinition`
    (flagged ``user=True``). Raises :class:`UserFunctionError` on any
    validation failure, in which case no registry is modified.
    """
    _check_name(name)
    _check_metadata(name, description, formula_template)
    params, defaults = _check_params(name, param_names, param_defaults)
    _check_formula_template(name, formula_template, params)
    scope_tokens = tuple(str(s).strip().lower() for s in scopes)
    invalid_scopes = sorted(set(scope_tokens) - set(SCOPES))
    if not scope_tokens or invalid_scopes:
        raise UserFunctionError(
            f"{name}: scopes must be a non-empty subset of {sorted(SCOPES)}; got {tuple(scopes)!r}."
        )
    if fwhm_factor is not None:
        try:
            fwhm_factor = float(fwhm_factor)
        except (TypeError, ValueError) as exc:
            raise UserFunctionError(f"{name}: fwhm_factor must be a number or None.") from exc
    _probe_function(name, function, defaults, _PROBE_GRIDS["parameter"])

    definition = ParameterModelComponentDefinition(
        name=name,
        description=str(description),
        function=function,
        param_names=params,
        param_defaults=defaults,
        param_info={p: get_param_info(p) for p in params},
        formula_template=str(formula_template),
        latex_equation=str(latex_equation),
        scopes=scope_tokens,
        fwhm_factor=fwhm_factor,
        user=True,
    )
    insert_definition(
        PARAMETER_MODEL_COMPONENTS,
        definition,
        registry_label="PARAMETER_MODEL_COMPONENTS",
    )
    register_component_documentation(
        name,
        kind="parameter_model",
        applicability=str(applicability),
        references=tuple(str(ref) for ref in references),
    )
    _record_registration("parameter_component", name)
    return definition
