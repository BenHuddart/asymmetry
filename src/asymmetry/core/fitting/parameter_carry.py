"""Carry parameter state across a model change, by component instance.

A parameter's value, constraints and roles belong to the component instance
that owns it, so when the model changes they follow that instance wherever it
moves and whatever its parameters are now called. A component instance that
did not exist before carries nothing and is left for the caller to seed —
nothing is keyed by name across a model change.

The map from new component index to old component index is the ``origins``
sequence supplied by whoever edited the model (the function builder keeps one
alongside its rows). Where no such map exists — a model retyped as text —
:func:`align_component_names` recovers one by matching component names in
order.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

from asymmetry.core.fitting.parameters import AffineTie, Parameter, ParameterSet

T = TypeVar("T")

#: Identifiers inside a free-form ``Parameter.expr`` constraint. Every token
#: that looks like a name is treated as a possible parameter reference; names
#: the old model never had (functions, constants) translate to themselves.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Entry keys that describe a *fit* of the old model rather than a starting
#: point, and so are cleared when an entry is carried as a seed.
_UNCERTAINTY_KEYS: tuple[str, ...] = ("uncertainty", "uncertainty_asymmetric")


class ParameterIdentity(ABC):
    """What a parameter *is*, independent of how the model spells it.

    Two models' parameters denote the same quantity when their identities
    compare equal after :meth:`remap` has expressed the new model's component
    indices in the old model's terms.
    """

    @abstractmethod
    def remap(self, origins: Sequence[int | None]) -> ParameterIdentity | None:
        """Return this identity with component indices mapped back through ``origins``.

        ``origins[i]`` is the index, in the old model, of the component that is
        now at index ``i`` — or ``None`` when that component is new. The result
        is ``None`` when any component this identity references is new, which is
        exactly the case where there is nothing to carry from.
        """


@dataclass(frozen=True)
class ComponentParameter(ParameterIdentity):
    """A component's own parameter, addressed by its local (unmapped) name."""

    component: int
    local_name: str

    def remap(self, origins: Sequence[int | None]) -> ComponentParameter | None:
        origin = origins[self.component]
        if origin is None:
            return None
        return replace(self, component=origin)


@dataclass(frozen=True)
class GroupAmplitude(ParameterIdentity):
    """A fraction group's shared amplitude, keyed by the group's component set.

    The amplitude belongs to the group as a whole, so it only carries when
    every component of the group survives into the same group.
    """

    components: frozenset[int]

    def remap(self, origins: Sequence[int | None]) -> GroupAmplitude | None:
        mapped: set[int] = set()
        for component in self.components:
            origin = origins[component]
            if origin is None:
                return None
            mapped.add(origin)
        return GroupAmplitude(frozenset(mapped))


@dataclass(frozen=True)
class FractionWeight(ParameterIdentity):
    """A free fraction weight, keyed by the component that starts its term."""

    term_start: int

    def remap(self, origins: Sequence[int | None]) -> FractionWeight | None:
        origin = origins[self.term_start]
        if origin is None:
            return None
        return replace(self, term_start=origin)


class SupportsParameterIdentities(Protocol):
    """Any model that can name what each of its parameters is."""

    def parameter_identities(self) -> dict[str, ParameterIdentity]: ...


def align_component_names(old: Sequence[str], new: Sequence[str]) -> tuple[int | None, ...]:
    """Match ``new`` component names onto ``old`` ones, in order.

    Returns one entry per new component: the index of the old component it
    continues, or ``None`` when it is new. Equal names are matched as a longest
    common subsequence, so an inserted or removed component shifts the ones
    after it instead of re-pairing them. Where the choice is a tie, the earlier
    new component is left unmatched — a repeated name is continued by its first
    occurrence.

    This is the fallback used when the edit itself did not record which
    component became which (a model retyped as an expression).
    """
    old_names = list(old)
    new_names = list(new)
    n_old = len(old_names)
    n_new = len(new_names)

    # lengths[i][j] is the LCS length of old_names[i:] and new_names[j:].
    lengths = [[0] * (n_new + 1) for _ in range(n_old + 1)]
    for i in range(n_old - 1, -1, -1):
        for j in range(n_new - 1, -1, -1):
            if old_names[i] == new_names[j]:
                lengths[i][j] = 1 + lengths[i + 1][j + 1]
            else:
                lengths[i][j] = max(lengths[i + 1][j], lengths[i][j + 1])

    origins: list[int | None] = []
    i = 0
    j = 0
    while j < n_new:
        if i >= n_old:
            origins.append(None)
            j += 1
        elif old_names[i] == new_names[j]:
            origins.append(i)
            i += 1
            j += 1
        elif lengths[i][j + 1] >= lengths[i + 1][j]:
            origins.append(None)
            j += 1
        else:
            i += 1
    return tuple(origins)


def carried_names(
    old_identities: Mapping[str, ParameterIdentity],
    new_identities: Mapping[str, ParameterIdentity],
    origins: Sequence[int | None],
) -> dict[str, str]:
    """Map each new parameter name to the old name it continues.

    New parameters (no surviving predecessor) are absent. ``origins`` may
    repeat an old index — a duplicated component — in which case every
    successor maps to the same old name and all of them carry.
    """
    by_identity = {identity: name for name, identity in old_identities.items()}
    mapping: dict[str, str] = {}
    for new_name, identity in new_identities.items():
        old_identity = identity.remap(origins)
        if old_identity is None:
            continue
        old_name = by_identity.get(old_identity)
        if old_name is None:
            continue
        mapping[new_name] = old_name
    return mapping


def carry_parameters(
    old_identities: Mapping[str, ParameterIdentity],
    new_identities: Mapping[str, ParameterIdentity],
    origins: Sequence[int | None],
    state: Mapping[str, T],
) -> dict[str, T]:
    """Re-key ``state`` from the old model's parameter names onto the new model's.

    Generic over the payload: whatever was stored against a parameter that
    survives the model change reappears under the surviving parameter's new
    name. Parameters with no predecessor — and predecessors with no entry in
    ``state`` — are simply absent from the result, so the caller seeds them.
    """
    return {
        new_name: state[old_name]
        for new_name, old_name in carried_names(old_identities, new_identities, origins).items()
        if old_name in state
    }


def _successor_names(carried: Mapping[str, str]) -> dict[str, str]:
    """Invert a new→old name map, keeping the first successor of each old name."""
    successors: dict[str, str] = {}
    for new_name, old_name in carried.items():
        successors.setdefault(old_name, new_name)
    return successors


def _translate_reference(
    name: str,
    successors: Mapping[str, str],
    old_identities: Mapping[str, ParameterIdentity],
) -> str | None:
    """Translate one referenced parameter name; ``None`` when it vanished.

    A name the old model owned resolves to its successor, or to ``None`` when
    it has none. A name the old model never owned — a free auxiliary parameter
    driving a tie, or a function name inside an expression — is untouched by
    the model change and stays as it is.
    """
    if name in successors:
        return successors[name]
    if name in old_identities:
        return None
    return name


def _translate_tie(
    tie: AffineTie,
    successors: Mapping[str, str],
    old_identities: Mapping[str, ParameterIdentity],
) -> AffineTie | None:
    """Re-target a tie's referenced names; ``None`` when it must be dropped."""
    main = _translate_reference(tie.main, successors, old_identities)
    if main is None:
        return None
    if tie.offset is None:
        return replace(tie, main=main)
    offset = _translate_reference(tie.offset, successors, old_identities)
    if offset is None:
        return None
    return replace(tie, main=main, offset=offset)


def _translate_expr(
    expr: str,
    successors: Mapping[str, str],
    old_identities: Mapping[str, ParameterIdentity],
) -> str | None:
    """Re-target the names inside an expression constraint; ``None`` to drop it."""
    resolved: dict[str, str] = {}
    for name in set(_IDENTIFIER_RE.findall(expr)):
        translated = _translate_reference(name, successors, old_identities)
        if translated is None:
            return None
        resolved[name] = translated
    return _IDENTIFIER_RE.sub(lambda match: resolved[match.group(0)], expr)


def carry_parameter_set(
    old_model: SupportsParameterIdentities,
    new_model: SupportsParameterIdentities,
    origins: Sequence[int | None],
    parameters: ParameterSet,
) -> ParameterSet:
    """Carry a :class:`ParameterSet` onto ``new_model``'s parameter names.

    Value, bounds, ``fixed`` and ``link_group`` follow the component instance.
    Ties and expression constraints have their referenced names re-targeted
    through the same map and are dropped when a referenced parameter did not
    survive. Parameters with no predecessor are absent from the result.
    """
    old_identities = old_model.parameter_identities()
    new_identities = new_model.parameter_identities()
    carried = carried_names(old_identities, new_identities, origins)
    successors = _successor_names(carried)

    result = ParameterSet()
    for new_name, old_name in carried.items():
        if old_name not in parameters:
            continue
        old = parameters[old_name]
        result.add(
            Parameter(
                name=new_name,
                value=old.value,
                min=old.min,
                max=old.max,
                fixed=old.fixed,
                expr=(
                    None
                    if old.expr is None
                    else _translate_expr(old.expr, successors, old_identities)
                ),
                link_group=old.link_group,
                tie=(
                    None if old.tie is None else _translate_tie(old.tie, successors, old_identities)
                ),
            )
        )
    return result


def carry_parameter_entries(
    old_model: SupportsParameterIdentities,
    new_model: SupportsParameterIdentities,
    origins: Sequence[int | None],
    entries: Sequence[dict],
) -> list[dict]:
    """Carry the GUI's list-of-dict parameter state onto ``new_model``'s names.

    Each carried entry is a *seed*: the uncertainty keys are cleared, because
    they described a fit of the old model. The ``tie`` dict is re-targeted or
    dropped exactly as :func:`carry_parameter_set` does; every other key is
    passed through untouched. Entries whose parameter has no successor are
    absent from the result, so the caller's own seeds stand.
    """
    old_identities = old_model.parameter_identities()
    new_identities = new_model.parameter_identities()
    carried = carried_names(old_identities, new_identities, origins)
    successors = _successor_names(carried)
    by_name = {str(entry["name"]): entry for entry in entries}

    result: list[dict] = []
    for new_name, old_name in carried.items():
        if old_name not in by_name:
            continue
        entry = dict(by_name[old_name])
        entry["name"] = new_name
        for key in _UNCERTAINTY_KEYS:
            entry[key] = None
        tie = entry["tie"]
        if tie is not None:
            translated = _translate_tie(AffineTie.from_dict(tie), successors, old_identities)
            entry["tie"] = None if translated is None else translated.to_dict()
        result.append(entry)
    return result
