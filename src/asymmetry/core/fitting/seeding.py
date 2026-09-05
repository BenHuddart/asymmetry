"""One seeding function for fit-parameter start values, bounds and Fix state.

:func:`seed_parameters` answers "what should this model's parameters start at
for this data?" in one place, so every fit surface asks the same question and
gets the same answer. It composes five layers, each later one overriding the
values the earlier ones set:

1. **component static defaults** — ``param_defaults``, each parameter's
   ``default_min`` (``-inf`` when it has none), and the model's
   ``fixed_by_default_params``;
2. **record scale** (time domain, with a dataset) — amplitude-role parameters
   from the record's early-mean-minus-tail estimate and background parameters
   from its tail (see :func:`record_scale_estimate`);
3. **applied field** — ``field`` / ``B_L`` from a non-zero applied field;
4. **frequency-domain peaks** — the displayed spectrum's dominant peak for one
   dataset, or the per-parameter mean across a batch's datasets;
5. **individual-groups overrides** — background and ``phase`` held at zero,
   because that fit carries the absolute phase in its per-group nuisances.

``Seed.run_bound`` marks a value that *describes the run it was seeded from*
rather than the physics the user is fitting: the applied field and the frequency
peak. When a form is carried from one run to another, a run-bound value is stale
by construction and is re-seeded from the new run, while a value the user typed
or fitted is kept.

Every key returned is a name in ``model.param_names``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
)
from asymmetry.core.fitting.models import LINEAR_PARAM_ROLE_NAMES
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, suggest_trend_seeds
from asymmetry.core.fitting.parameters import get_param_info, split_parameter_name
from asymmetry.core.fitting.spectral import seed_peak_parameters_from_dataset

#: Base names seeded from the run's applied field.
_FIELD_SEED_BASE_NAMES: frozenset[str] = frozenset({"field", "B_L"})

#: Base names that carry the *record's* asymmetry amplitude. Deliberately the
#: intersection of the linear (amplitude/background) roles with the amplitude
#: predicate rather than the predicate alone: ``is_amplitude_parameter`` matches
#: any ``a_*`` name, which would sweep in physical quantities that merely start
#: that way (``A_hf``, a hyperfine coupling; ``a_L``, a Lorentzian weight).
_AMPLITUDE_ROLE_BASE_NAMES: frozenset[str] = frozenset(
    name
    for name in LINEAR_PARAM_ROLE_NAMES
    if is_amplitude_parameter(name) and not is_background_parameter(name)
)

#: Parameters that start held whatever model they appear in. ``shape_factor_a``
#: is a switch, not a fitted quantity: ``0`` selects the Carrington-Manzano
#: interpolation and any positive value the generalized reduced-gap law, so a
#: fit must not wander across that boundary unless the user releases it.
_ALWAYS_FIXED_PARAM_NAMES: frozenset[str] = frozenset({"shape_factor_a"})


@dataclass(frozen=True)
class Seed:
    """One parameter's starting value, bounds, Fix state and provenance."""

    value: float
    fixed: bool = False
    min: float = -math.inf
    max: float = math.inf
    #: True when the value describes the run it came from (applied field,
    #: frequency peak) rather than a physics guess that travels with the form.
    run_bound: bool = False


@dataclass(frozen=True)
class SeedContext:
    """What the seeding layers know about the data being fitted.

    ``dataset`` is the single record a surface fits; ``datasets`` the members of
    a batch. ``field_gauss`` is the applied field the surface attributes to the
    fit — one run's field, or the mean across a batch's members — with ``None``
    (or zero) meaning "no applied field to seed from".
    """

    dataset: MuonDataset | None = None
    datasets: tuple[MuonDataset, ...] = ()
    field_gauss: float | None = None
    domain: str = "time"
    individual_groups: bool = False


def record_scale_window_counts(n_points: int) -> tuple[int, int]:
    """Return the (early, late) sample counts framing a record's scale estimate.

    Both windows are a fixed fraction of the record with a five-sample floor, so
    a short record still has something to average.
    """
    n = max(int(n_points), 1)
    return min(n, max(5, n // 20)), min(n, max(5, n // 10))


def record_scale_estimate(
    time: NDArray[np.float64], asymmetry: NDArray[np.float64]
) -> tuple[float, float]:
    """Return one record's ``(amplitude, tail)`` scale estimate.

    The tail is the mean of the late window and the amplitude the early
    window's mean above it — the crude but robust "how big is this signal and
    what does it settle to" reading that both the fit wizard's fingerprint and
    the record-scale seeding layer are built on.
    """
    early_count, late_count = record_scale_window_counts(len(time))
    y = np.asarray(asymmetry, dtype=float)
    tail = float(np.mean(y[-late_count:]))
    amplitude = float(np.mean(y[:early_count]) - tail)
    return amplitude, tail


def seed_parameters(model: CompositeModel, context: SeedContext) -> dict[str, Seed]:
    """Return a starting :class:`Seed` for every parameter of *model*.

    The layers are applied in the order documented at the top of this module;
    each one overrides the values of those before it.
    """
    seeds = _static_default_seeds(model, context)
    _apply_values(seeds, _record_scale_values(model, context), run_bound=False)
    _apply_values(seeds, _applied_field_values(model, context), run_bound=True)
    _apply_values(seeds, _frequency_peak_values(model, context), run_bound=True)
    _hold_at_zero(seeds, _individual_group_held_names(model, context))
    return seeds


def seed_trend_parameters(
    model: ParameterCompositeModel, x: NDArray[np.float64], y: NDArray[np.float64]
) -> dict[str, Seed]:
    """Return a starting :class:`Seed` for every parameter of a trend model.

    Static component defaults with :func:`~asymmetry.core.fitting.parameter_models.suggest_trend_seeds`
    over the top, which is what makes a critical-temperature component converge
    without a hand reseed. A trend is fitted against a series rather than a run,
    so nothing here is run-bound.
    """
    seeds = _base_seeds(model.param_names, model.param_defaults, frozenset())
    _apply_values(seeds, suggest_trend_seeds(model, x, y), run_bound=False)
    return seeds


# ── layers ──────────────────────────────────────────────────────────────────


def _static_default_seeds(model: CompositeModel, context: SeedContext) -> dict[str, Seed]:
    """Layer 1: each component's declared defaults, lower bound and Fix state."""
    return _base_seeds(model.param_names, model.param_defaults, model.fixed_by_default_params())


def _record_scale_values(model: CompositeModel, context: SeedContext) -> dict[str, float]:
    """Layer 2: amplitude and background parameters from the record's own scale."""
    if context.domain != "time" or context.dataset is None:
        return {}
    amplitude, tail = record_scale_estimate(context.dataset.time, context.dataset.asymmetry)
    values: dict[str, float] = {}
    for param_name in model.param_names:
        base_name, _index = split_parameter_name(param_name)
        if is_background_parameter(base_name):
            values[param_name] = tail
        elif base_name in _AMPLITUDE_ROLE_BASE_NAMES:
            values[param_name] = amplitude
    return values


def _applied_field_values(model: CompositeModel, context: SeedContext) -> dict[str, float]:
    """Layer 3: ``field`` / ``B_L`` from the applied field, when there is one."""
    field_gauss = context.field_gauss
    if field_gauss is None or field_gauss == 0.0:
        return {}
    return {
        param_name: float(field_gauss)
        for param_name in model.param_names
        if split_parameter_name(param_name)[0] in _FIELD_SEED_BASE_NAMES
    }


def _frequency_peak_values(model: CompositeModel, context: SeedContext) -> dict[str, float]:
    """Layer 4: peak height/centre/width/background read off the spectrum.

    A batch seeds from the per-parameter mean across its members, so one
    member's noisy spectrum cannot drag the whole series' start value.
    """
    if context.domain != "frequency":
        return {}
    if context.datasets:
        values_by_name: dict[str, list[float]] = {}
        for dataset in context.datasets:
            for name, value in seed_peak_parameters_from_dataset(dataset, model).items():
                values_by_name.setdefault(name, []).append(float(value))
        return {name: float(np.mean(values)) for name, values in values_by_name.items()}
    if context.dataset is None:
        return {}
    return {
        name: float(value)
        for name, value in seed_peak_parameters_from_dataset(context.dataset, model).items()
    }


def _individual_group_held_names(model: CompositeModel, context: SeedContext) -> list[str]:
    """Layer 5: the parameters an individual-groups fit holds at zero.

    That fit gives every detector group its own background and its own absolute
    phase nuisance, so the shared model's background and ``phase`` would be
    degenerate with them.
    """
    if not context.individual_groups:
        return []
    held: list[str] = []
    for param_name in model.param_names:
        base_name, _index = split_parameter_name(param_name)
        if is_background_parameter(base_name) or base_name == "phase":
            held.append(param_name)
    return held


# ── composition helpers ─────────────────────────────────────────────────────


def _base_seeds(
    param_names: list[str],
    param_defaults: dict[str, float],
    fixed_by_default: set[str],
) -> dict[str, Seed]:
    """Return the static seed of every named parameter."""
    fixed_names = set(fixed_by_default) | _ALWAYS_FIXED_PARAM_NAMES
    return {
        param_name: Seed(
            value=float(param_defaults[param_name]),
            fixed=param_name in fixed_names,
            min=_default_min(param_name),
            max=math.inf,
        )
        for param_name in param_names
    }


def _default_min(param_name: str) -> float:
    """Return a parameter's default lower bound (``-inf`` when it has none)."""
    default_min = get_param_info(param_name).default_min
    return -math.inf if default_min is None else float(default_min)


def _apply_values(seeds: dict[str, Seed], values: dict[str, float], *, run_bound: bool) -> None:
    """Overwrite the value (and run-boundness) of each seed a layer names."""
    for param_name, value in values.items():
        seeds[param_name] = replace(seeds[param_name], value=float(value), run_bound=run_bound)


def _hold_at_zero(seeds: dict[str, Seed], param_names: Iterable[str]) -> None:
    """Set each named seed to zero and Fix it."""
    for param_name in param_names:
        seeds[param_name] = replace(seeds[param_name], value=0.0, fixed=True, run_bound=False)


__all__ = [
    "Seed",
    "SeedContext",
    "record_scale_estimate",
    "record_scale_window_counts",
    "seed_parameters",
    "seed_trend_parameters",
]
