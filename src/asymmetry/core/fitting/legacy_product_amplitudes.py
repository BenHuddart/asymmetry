"""Alpha-only migration of saved amplitudes onto the one-scale-per-product policy.

**This module is temporary and is deleted at v1.0**, together with its call
sites and ``tests/core/test_legacy_product_amplitudes.py`` — the deletion list
lives in ``RELEASING.md`` § "Delete at v1.0".

Before the policy landed, :class:`~asymmetry.core.fitting.composite.CompositeModel`
named amplitudes by *spelling*: a flat ``*``/``/`` chain shared one ``A``, while
any parenthesis switched the whole model to a regime that kept every component's
own ``A`` (bar a leaf multiplied by an additive group).  Parameter *names* are
persisted next to the model dict — in projects, in the fit-wizard cache and in
global-wizard payloads — so saved data carries amplitude entries the new model
no longer exposes:

* amplitudes of suppressed product factors (``A_2``/``A_4`` in a wizard
  multiplet ``(Osc*Exp)+(Osc*Exp)+Const``, usually pinned to ``1.0``);
* ``A_1`` in a flat ``Constant * Exponential``, where the old evaluator
  multiplied by both ``A_bg`` and ``A_1``;
* a lone amplitude spelled ``A`` in a parenthesised model, now ``A_1``, and an
  ``A_bg_2`` that is now plain ``A_bg`` because the colliding ``A_bg`` is
  suppressed.

The migration is self-contained: it carries a frozen copy of the *pre-policy*
naming algorithm (:func:`legacy_parameter_mapping`, ported from
``composite.py`` at 812db6d) and folds saved amplitudes onto the surviving
names.  Nothing here influences the live model, and the live model knows
nothing about it.

**Fold rules.**  For each product of the new expression tree, the surviving
amplitude's value becomes the product of every leaf factor's legacy amplitude
value that is present in the saved data (the survivor's own included); relative
uncertainties add in quadrature (an entry without an uncertainty contributes
none); the survivor is fixed only if every folded entry was fixed; the survivor
keeps its own bounds.  Renamed survivors are renamed, folded-away entries are
removed, and every other entry passes through untouched, in order.

A product whose factors include a sum has *no* surviving leaf scale (the sum's
terms carry the scale instead) — only reachable by the degenerate spelling
``(A + B) * C * D``, where the old rule suppressed just the factor adjacent to
the group.  There, the product of the present legacy leaf amplitudes on the
product's *other* factors (``k``) is distributed onto the sum's terms rather
than dropped: each term's own surviving scale — a leaf's own amplitude, a
nested product's survivor, or (recursively) a nested fraction group's
amplitude — is multiplied by ``k``, and ``k``'s relative uncertainty adds in
quadrature to that scale's own.  Fixedness of the scaled entries is unchanged:
they were the user's own parameters, and ``k`` is simply absorbed into their
value.  A fraction group's terms are represented by its single group
amplitude, which alone absorbs ``k``.  When two sum factors of the same
product both carry scales (the degenerate ``(A + B) * (C + D)``), only the
first sum receives the distribution — the shape is already degenerate, and one
target is enough to stay deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from asymmetry.core.fitting.composite import (
    _UNIT_AMPLITUDE_SENTINEL,
    CompositeModel,
    ExprLeaf,
    ExprProduct,
    ExprSum,
    iter_nodes,
    leaf_indices,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

#: Parameters that act as a component's scale factor (frozen copy of the
#: pre-policy ``CompositeModel._is_scaling_parameter``).
_SCALING_PARAMETERS = frozenset({"A", "A_bg"})


# ---------------------------------------------------------------------------
# The frozen pre-policy naming algorithm
# ---------------------------------------------------------------------------


def _has_scaling_parameter(param_names: Sequence[str]) -> bool:
    return any(pname in _SCALING_PARAMETERS for pname in param_names)


def _rhs_group_contains_additive_operator(
    rhs_index: int,
    operators: Sequence[str],
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    component_count: int,
) -> bool:
    """Return True if the group opening at ``rhs_index`` has a top-level + or -."""
    balance = 1
    for k in range(rhs_index, component_count):
        if k == rhs_index:
            balance += max(open_parentheses[k] - 1, 0)
        else:
            balance += open_parentheses[k]

        if k > rhs_index and balance > 0 and operators[k - 1] in {"+", "-"}:
            return True

        balance -= close_parentheses[k]
        if balance <= 0:
            break
    return False


def _lhs_group_contains_additive_operator(
    lhs_index: int,
    operators: Sequence[str],
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
) -> bool:
    """Return True if the group closing at ``lhs_index`` has a top-level + or -."""
    balance = 1
    for k in range(lhs_index, -1, -1):
        if k == lhs_index:
            balance += max(close_parentheses[k] - 1, 0)
        else:
            balance += close_parentheses[k]

        if k < lhs_index and balance > 0 and operators[k] in {"+", "-"}:
            return True

        balance -= open_parentheses[k]
        if balance <= 0:
            break
    return False


def _legacy_suppressed_amplitudes(
    param_names_by_component: Sequence[Sequence[str]],
    operators: Sequence[str],
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
) -> list[bool]:
    """Frozen copy of the pre-policy ``_identify_suppressed_amplitudes``.

    In a parenthesised expression, a leaf multiplied or divided by an additive
    group lost its amplitude; a flat expression suppressed nothing (it shared
    one ``A`` per chain instead, see :func:`_legacy_param_mapping`).
    """
    suppress = [False] * len(param_names_by_component)
    if not (any(open_parentheses) or any(close_parentheses)):
        return suppress

    component_count = len(param_names_by_component)
    for op_index, op in enumerate(operators):
        if op not in {"*", "/"}:
            continue

        lhs_index = op_index
        rhs_index = op_index + 1
        if rhs_index >= component_count:
            continue

        rhs_is_additive_group = open_parentheses[
            rhs_index
        ] > 0 and _rhs_group_contains_additive_operator(
            rhs_index, operators, open_parentheses, close_parentheses, component_count
        )
        lhs_is_additive_group = close_parentheses[
            lhs_index
        ] > 0 and _lhs_group_contains_additive_operator(
            lhs_index, operators, open_parentheses, close_parentheses
        )

        if (
            rhs_is_additive_group
            and not lhs_is_additive_group
            and _has_scaling_parameter(param_names_by_component[lhs_index])
        ):
            suppress[lhs_index] = True
        if (
            lhs_is_additive_group
            and not rhs_is_additive_group
            and _has_scaling_parameter(param_names_by_component[rhs_index])
        ):
            suppress[rhs_index] = True

    return suppress


def _legacy_param_mapping(
    param_names_by_component: Sequence[Sequence[str]],
    operators: Sequence[str],
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    fraction_term_number_by_component: Mapping[int, int],
    fraction_group_by_component: Mapping[int, tuple[int, int]],
) -> list[dict[str, str]]:
    """Frozen copy of the pre-policy ``CompositeModel._build_param_mapping``.

    A flat expression shared one ``A`` per multiplicative chain (named after the
    chain's first component); any parenthesis kept every component's own ``A``,
    indexed only when the name collided.  Collision counts included suppressed
    scales, which is why an old ``Constant`` next to a suppressed ``Constant``
    was named ``A_bg_2``.
    """
    name_counts: dict[str, int] = {}
    for param_names in param_names_by_component:
        for pname in param_names:
            name_counts[pname] = name_counts.get(pname, 0) + 1

    component_count = len(param_names_by_component)
    share_chain_amplitude = not (any(open_parentheses) or any(close_parentheses))
    suppressed = _legacy_suppressed_amplitudes(
        param_names_by_component, operators, open_parentheses, close_parentheses
    )

    amplitude_group_starts: list[int] = []
    current_start = 1
    for idx in range(1, component_count + 1):
        if idx > 1 and operators[idx - 2] in {"+", "-"}:
            # Start a new amplitude group after additive operators.
            current_start = idx
        amplitude_group_starts.append(current_start)

    mappings: list[dict[str, str]] = []
    used_names: set[str] = set()
    for idx, param_names in enumerate(param_names_by_component, start=1):
        mapping: dict[str, str] = {}
        for pname in param_names:
            is_scaling = pname in _SCALING_PARAMETERS
            if is_scaling and (
                fraction_group_by_component.get(idx - 1) is not None or suppressed[idx - 1]
            ):
                mapping[pname] = _UNIT_AMPLITUDE_SENTINEL
                continue
            if pname == "A" and share_chain_amplitude:
                mapping[pname] = f"A_{amplitude_group_starts[idx - 1]}"
            elif name_counts[pname] > 1:
                term_number = fraction_term_number_by_component.get(idx - 1)
                if term_number is not None:
                    candidate = f"{pname}_{term_number}"
                    mapping[pname] = candidate if candidate not in used_names else f"{pname}_{idx}"
                else:
                    mapping[pname] = f"{pname}_{idx}"
            else:
                mapping[pname] = pname
            used_names.add(mapping[pname])
        mappings.append(mapping)
    return mappings


def legacy_parameter_mapping(model: CompositeModel) -> list[dict[str, str]]:
    """Return the parameter names ``model`` carried before the amplitude policy.

    One dict per component, local parameter name → the name the pre-policy
    ``CompositeModel`` exposed (or the unit-amplitude sentinel for a scale it
    suppressed).  Fraction naming is unchanged by the policy, so the model's own
    fraction bookkeeping is read straight off it.
    """
    return _legacy_param_mapping(
        [tuple(component.param_names) for component in model.components],
        model.operators,
        model.open_parentheses,
        model.close_parentheses,
        model._fraction_term_number_by_component,
        model._fraction_group_by_component,
    )


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fold:
    """One surviving amplitude and the legacy names that fold into it.

    ``legacy_names`` is in factor order with the survivor's own legacy name
    first (when it had one), so the first *present* entry supplies the survivor's
    bounds and its position in the parameter list.
    """

    new_name: str
    legacy_names: tuple[str, ...]


@dataclass(frozen=True)
class _Distribution:
    """A product's leaf amplitudes with no surviving scale, spread over a sum.

    ``k_names`` are the legacy names of the product's leaf factors that lost
    their scale to a sum factor and have no current counterpart; their product
    (``k``) multiplies each of ``target_names`` — the current name of every
    scale the sum's terms carry.
    """

    k_names: tuple[str, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class _FoldPlan:
    """Every rename/fold/distribute/drop the saved amplitudes of one model need."""

    folds: tuple[_Fold, ...]
    distributions: tuple[_Distribution, ...]
    dropped: frozenset[str]

    def fold_by_legacy_name(self) -> dict[str, _Fold]:
        return {name: fold for fold in self.folds for name in fold.legacy_names}


def _scale_names(model: CompositeModel) -> tuple[dict[int, str], dict[int, str]]:
    """Return ``(legacy, current)`` scale names keyed by component index.

    A component whose scale is suppressed under a scheme is absent from that
    scheme's dict.
    """
    legacy_mapping = legacy_parameter_mapping(model)
    current_mapping = model.parameter_mapping()
    legacy: dict[int, str] = {}
    current: dict[int, str] = {}
    for index, component in enumerate(model.components):
        for pname in component.param_names:
            if pname not in _SCALING_PARAMETERS:
                continue
            if legacy_mapping[index][pname] != _UNIT_AMPLITUDE_SENTINEL:
                legacy[index] = legacy_mapping[index][pname]
            if current_mapping[index][pname] != _UNIT_AMPLITUDE_SENTINEL:
                current[index] = current_mapping[index][pname]
    return legacy, current


def _sum_target_names(
    node: ExprSum, current: Mapping[int, str], model: CompositeModel
) -> tuple[str, ...]:
    """Return the current scale name(s) that carry ``node``'s terms.

    A fraction-group sum is scaled by its single group amplitude. A plain
    sum's terms each keep their own current scale — a leaf's own amplitude, a
    nested product's survivor (``current`` already reflects that survivor, so
    filtering the term's leaves against it is enough), or, recursively, a
    nested fraction group's amplitude.
    """
    if node.fraction_group is not None:
        return (model._fraction_group_amplitude_name(node.fraction_group),)
    names: list[str] = []
    for term in node.terms:
        if isinstance(term, ExprSum):
            names.extend(_sum_target_names(term, current, model))
        else:
            names.extend(current[index] for index in leaf_indices(term) if index in current)
    return tuple(names)


def _fold_plan(model: CompositeModel) -> _FoldPlan:
    """Pair legacy amplitude names with the names the model exposes now."""
    legacy, current = _scale_names(model)

    # Every surviving scale starts paired with its own component; a product then
    # hands its survivor the scales of the factors the policy suppressed.
    contributors: dict[int, list[int]] = {index: [index] for index in current}
    distributions: list[_Distribution] = []
    distributed_sources: set[int] = set()
    for node in iter_nodes(model.expression_tree()):
        if not isinstance(node, ExprProduct):
            continue
        leaves = [factor.index for factor in node.factors if isinstance(factor, ExprLeaf)]
        survivors = [index for index in leaves if index in current]
        suppressed = [index for index in leaves if index not in current and index in legacy]
        if survivors:
            contributors[survivors[0]] = [survivors[0], *suppressed]
            continue
        if not suppressed:
            continue
        sums = [factor for factor in node.factors if isinstance(factor, ExprSum)]
        if not sums:
            continue
        targets = _sum_target_names(sums[0], current, model)
        if not targets:
            continue
        distributions.append(_Distribution(tuple(legacy[index] for index in suppressed), targets))
        distributed_sources.update(suppressed)

    folds: list[_Fold] = []
    for index, sources in contributors.items():
        legacy_names = tuple(legacy[source] for source in sources if source in legacy)
        if legacy_names and legacy_names != (current[index],):
            folds.append(_Fold(current[index], legacy_names))

    contributing = {source for sources in contributors.values() for source in sources}
    contributing |= distributed_sources
    dropped = frozenset(
        name for index, name in legacy.items() if index not in contributing and name not in current
    )
    return _FoldPlan(tuple(folds), tuple(distributions), dropped)


def _carries_legacy_names(model: CompositeModel, names: Iterable[str]) -> bool:
    """Return True when some saved name is not a parameter of ``model``.

    The cheap precondition shared by every entry point: data already saved under
    the current policy names only parameters the model exposes, so it skips the
    fold entirely (and never pays for the legacy mapping or the tree walk).
    """
    exposed = set(model.param_names)
    return any(name not in exposed for name in names)


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Best-effort ``float(value)`` for a hand-edited or truncated saved value."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _folded_value(values: Sequence[float]) -> float:
    return math.prod(values)


def _folded_uncertainty(value: float, pairs: Sequence[tuple[float, float | None]]) -> float | None:
    """Combine per-factor ``(value, uncertainty)`` into the survivor's uncertainty.

    Relative uncertainties add in quadrature; a factor without an uncertainty
    contributes none, and a factor whose value is zero has no relative
    uncertainty to contribute (the folded value is zero regardless).  Returns
    ``None`` when no factor carried an uncertainty.
    """
    relative_squares = [
        (uncertainty / factor) ** 2
        for factor, uncertainty in pairs
        if uncertainty is not None and factor != 0.0
    ]
    if not any(uncertainty is not None for _, uncertainty in pairs):
        return None
    return abs(value) * math.sqrt(sum(relative_squares))


# ---------------------------------------------------------------------------
# Entry points, one per saved shape
# ---------------------------------------------------------------------------


def fold_legacy_product_amplitude_values(
    model: CompositeModel, values: Mapping[str, float]
) -> dict[str, float]:
    """Fold a legacy value dict onto ``model``'s current amplitude names.

    Each surviving amplitude takes the product of the legacy amplitude values
    present for its product's factors; folded-away and dropped keys are removed.
    A product with no surviving leaf scale instead distributes the product of
    its present legacy leaf values onto the sum's own surviving scales (see the
    module docstring). Every other key passes through untouched.  A no-op (a
    shallow copy) when the keys already match the model's parameters.
    """
    if not _carries_legacy_names(model, values):
        return dict(values)

    plan = _fold_plan(model)
    folded = {name: value for name, value in values.items() if name not in plan.dropped}
    for fold in plan.folds:
        present = [name for name in fold.legacy_names if name in values]
        if not present:
            continue
        for name in present:
            folded.pop(name, None)
        folded[fold.new_name] = _folded_value([_coerce_float(values[name]) for name in present])
    for distribution in plan.distributions:
        present = [name for name in distribution.k_names if name in values]
        if not present:
            continue
        for name in present:
            folded.pop(name, None)
        k = _folded_value([_coerce_float(values[name]) for name in present])
        for target in distribution.target_names:
            if target in folded:
                folded[target] = folded[target] * k
    return folded


def _entry_is_fixed(entry: Mapping[str, object]) -> bool:
    """Read fixedness from either saved entry shape (a flag, or a role column)."""
    return bool(entry.get("fixed", False)) or entry.get("type") == "Fixed"


def _folded_entry(fold: _Fold, present: Sequence[dict]) -> dict:
    """Build the survivor's entry from the entries folding into it, in order."""
    folded = dict(present[0])
    folded["name"] = fold.new_name
    values = [_coerce_float(entry.get("value", 0.0)) for entry in present]
    value = _folded_value(values)
    folded["value"] = value
    if len(present) == 1:
        return folded

    # The survivor absorbed other factors: its fixedness, uncertainty and any
    # asymmetric band are properties of the fold, not of the entry it came from.
    # (An entry shape that spells fixedness as a role column keeps the
    # survivor's own role — the role is the user's global/local choice too.)
    if "fixed" in folded:
        folded["fixed"] = all(_entry_is_fixed(entry) for entry in present)
    saved = [
        None if entry.get("uncertainty") is None else _coerce_float(entry["uncertainty"])
        for entry in present
    ]
    uncertainty = _folded_uncertainty(value, list(zip(values, saved, strict=True)))
    if uncertainty is not None or "uncertainty" in folded:
        folded["uncertainty"] = uncertainty
    if "uncertainty_asymmetric" in folded:
        folded["uncertainty_asymmetric"] = None
    return folded


def _distributed_entry(
    entry: dict, distribution: _Distribution, entry_by_name: Mapping[str, dict]
) -> dict:
    """Scale ``entry`` — a distribution target — by the product of its ``k`` factors.

    The target's own fixedness is left untouched: it was the user's own
    parameter, and ``k`` is absorbed into its value. Relative uncertainties
    (the target's own and each present ``k`` factor's) add in quadrature.
    """
    k_entries = [entry_by_name[name] for name in distribution.k_names if name in entry_by_name]
    if not k_entries:
        return entry

    own_value = _coerce_float(entry.get("value", 0.0))
    k_values = [_coerce_float(k_entry.get("value", 0.0)) for k_entry in k_entries]
    value = own_value * _folded_value(k_values)

    distributed = dict(entry)
    distributed["value"] = value
    own_uncertainty = (
        None if entry.get("uncertainty") is None else _coerce_float(entry["uncertainty"])
    )
    k_uncertainties = [
        None if k_entry.get("uncertainty") is None else _coerce_float(k_entry["uncertainty"])
        for k_entry in k_entries
    ]
    pairs = [(own_value, own_uncertainty), *zip(k_values, k_uncertainties, strict=True)]
    uncertainty = _folded_uncertainty(value, pairs)
    if uncertainty is not None or "uncertainty" in distributed:
        distributed["uncertainty"] = uncertainty
    if "uncertainty_asymmetric" in distributed:
        distributed["uncertainty_asymmetric"] = None
    return distributed


def fold_legacy_product_amplitude_entries(
    model: CompositeModel, parameters: list[dict]
) -> list[dict]:
    """Fold a list of saved parameter-state dicts onto the current names.

    Each entry carries at least ``name`` and ``value`` plus metadata
    (``fixed``/``min``/``max``/``uncertainty``/``type``/``bounds``).  The
    survivor's entry keeps its own bounds and takes the folded value, fixedness
    and uncertainty; folded-away entries are removed; a distribution target
    keeps its own bounds and fixedness but has its value (and uncertainty)
    scaled by its product's leftover legacy amplitudes; every other entry
    passes through untouched, preserving order.  A no-op (a shallow copy) when
    the names already match the model's parameters.
    """
    entry_by_name: dict[str, dict] = {}
    for entry in parameters:
        if isinstance(entry, dict) and "name" in entry:
            entry_by_name.setdefault(str(entry["name"]), entry)
    if not _carries_legacy_names(model, entry_by_name):
        return [dict(entry) for entry in parameters]

    plan = _fold_plan(model)
    fold_by_name = plan.fold_by_legacy_name()
    k_names = {name for distribution in plan.distributions for name in distribution.k_names}
    distribution_by_target = {
        target: distribution
        for distribution in plan.distributions
        for target in distribution.target_names
    }

    folded: list[dict] = []
    emitted: set[str] = set()
    for entry in parameters:
        if not isinstance(entry, dict):
            folded.append(entry)
            continue
        name = str(entry.get("name", ""))
        if name in plan.dropped or name in k_names:
            continue
        fold = fold_by_name.get(name)
        if fold is None:
            base = dict(entry)
        else:
            if fold.new_name in emitted:
                # A later factor of a product already folded into its survivor.
                continue
            emitted.add(fold.new_name)
            present = [
                entry_by_name[legacy] for legacy in fold.legacy_names if legacy in entry_by_name
            ]
            base = _folded_entry(fold, present)
        distribution = distribution_by_target.get(base["name"])
        folded.append(
            _distributed_entry(base, distribution, entry_by_name) if distribution else base
        )
    return folded


def fold_legacy_product_amplitude_set(
    model: CompositeModel,
    parameter_set: ParameterSet,
    uncertainties: Mapping[str, float],
) -> tuple[ParameterSet, dict[str, float]]:
    """Fold a :class:`ParameterSet` and its uncertainty dict onto the current names.

    The :class:`ParameterSet` analogue of
    :func:`fold_legacy_product_amplitude_entries`, for the fit results cached by
    the wizards (whose uncertainties live beside the parameters rather than on
    them).  The survivor keeps its own bounds, ``expr``, link group and tie, and
    is fixed only if every parameter folded into it was.  A distribution target
    keeps its own bounds, fixedness, ``expr``, link group and tie, with its
    value (and uncertainty) scaled by its product's leftover legacy amplitudes.
    Returns the inputs unchanged when the names already match the model's
    parameters.
    """
    if not _carries_legacy_names(model, parameter_set.names):
        return parameter_set, dict(uncertainties)

    plan = _fold_plan(model)
    fold_by_name = plan.fold_by_legacy_name()
    k_names = {name for distribution in plan.distributions for name in distribution.k_names}
    distribution_by_target = {
        target: distribution
        for distribution in plan.distributions
        for target in distribution.target_names
    }
    parameter_by_name = {parameter.name: parameter for parameter in parameter_set}

    folded = ParameterSet()
    folded_uncertainties = {
        name: value
        for name, value in uncertainties.items()
        if name not in plan.dropped and name not in fold_by_name and name not in k_names
    }
    emitted: set[str] = set()
    for parameter in parameter_set:
        if parameter.name in plan.dropped or parameter.name in k_names:
            continue
        fold = fold_by_name.get(parameter.name)
        if fold is None:
            folded.add(parameter)
            continue
        if fold.new_name in emitted:
            continue
        emitted.add(fold.new_name)
        present = [
            parameter_by_name[legacy] for legacy in fold.legacy_names if legacy in parameter_by_name
        ]
        values = [float(member.value) for member in present]
        value = _folded_value(values)
        survivor = present[0]
        folded.add(
            Parameter(
                name=fold.new_name,
                value=value,
                min=survivor.min,
                max=survivor.max,
                fixed=all(member.fixed for member in present),
                expr=survivor.expr,
                link_group=survivor.link_group,
                tie=survivor.tie,
            )
        )
        uncertainty = _folded_uncertainty(
            value,
            [
                (member_value, uncertainties.get(member.name))
                for member_value, member in zip(values, present, strict=True)
            ],
        )
        if uncertainty is not None:
            folded_uncertainties[fold.new_name] = uncertainty

    for target, distribution in distribution_by_target.items():
        k_members = [
            parameter_by_name[name] for name in distribution.k_names if name in parameter_by_name
        ]
        if target not in folded or not k_members:
            continue
        member = folded[target]
        k_pairs = [
            (float(k_member.value), uncertainties.get(k_member.name)) for k_member in k_members
        ]
        value = member.value * _folded_value([k_value for k_value, _ in k_pairs])
        folded.add(
            Parameter(
                name=target,
                value=value,
                min=member.min,
                max=member.max,
                fixed=member.fixed,
                expr=member.expr,
                link_group=member.link_group,
                tie=member.tie,
            )
        )
        uncertainty = _folded_uncertainty(
            value, [(member.value, folded_uncertainties.get(target)), *k_pairs]
        )
        if uncertainty is not None:
            folded_uncertainties[target] = uncertainty
    return folded, folded_uncertainties


def fold_legacy_product_amplitude_names(
    model: CompositeModel, names: Sequence[str]
) -> tuple[str, ...]:
    """Rename/drop legacy amplitudes in a saved parameter-role name tuple.

    The global wizard caches ``global``/``local``/``fixed`` parameter-role
    tuples beside its recommendation; a folded-away amplitude, and a
    distribution's leftover legacy amplitude, has no role under the current
    policy, and a renamed survivor keeps the role it had.  Order is preserved
    and duplicates collapse.
    """
    if not _carries_legacy_names(model, names):
        return tuple(names)

    plan = _fold_plan(model)
    rename = {legacy: fold.new_name for fold in plan.folds for legacy in fold.legacy_names[:1]}
    folded_away = {legacy for fold in plan.folds for legacy in fold.legacy_names[1:]}
    folded_away |= {name for distribution in plan.distributions for name in distribution.k_names}
    migrated: list[str] = []
    for name in names:
        if name in plan.dropped or name in folded_away:
            continue
        candidate = rename.get(name, name)
        if candidate not in migrated:
            migrated.append(candidate)
    return tuple(migrated)


def fold_legacy_product_amplitude_state(state: Mapping[str, object]) -> dict[str, object]:
    """Fold the amplitudes of a saved single-/global-fit form payload.

    ``state`` carries a ``composite_model`` dict and a ``parameters`` list of
    parameter-state dicts.  When the model reconstructs, its parameters are
    folded via :func:`fold_legacy_product_amplitude_entries`; otherwise (missing
    or malformed model payload) the state is returned with a shallow-copied
    ``parameters`` list and is otherwise unchanged.
    """
    folded = dict(state)
    parameters = state.get("parameters")
    model_data = state.get("composite_model")
    if not isinstance(parameters, list) or not isinstance(model_data, dict):
        return folded
    try:
        model = CompositeModel.from_dict(model_data, allow_missing=True)
    except (ValueError, KeyError, TypeError):
        # A hand-edited or truncated payload: nothing to fold against.
        return folded
    folded["parameters"] = fold_legacy_product_amplitude_entries(model, parameters)
    return folded
