"""Permanent converters for released positional fraction parameter payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping

from asymmetry.core.fitting.composite import CompositeModel


def _legacy_fraction_numbering(model: CompositeModel) -> list[list[int]]:
    """Return the old ``fraction_<k>`` numbers per group, in group order.

    The retired scheme numbered one ``fraction_<k>`` per additive term,
    consecutively across all groups in group order (see the pre-migration
    ``_build_fraction_param_number_map``). This reconstructs that numbering so
    legacy value dicts can be located and migrated.
    """
    numbering: list[list[int]] = []
    next_number = 1
    for group in model.fraction_groups:
        term_starts = model._fraction_group_term_starts(group)
        group_numbers = list(range(next_number, next_number + len(term_starts)))
        numbering.append(group_numbers)
        next_number += len(term_starts)
    return numbering


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Reject corrupt stored fractions rather than inventing a physical weight."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid stored fraction value {value!r}") from None
    if not math.isfinite(converted):
        raise ValueError(f"Invalid stored fraction value {value!r}")
    return converted


def has_legacy_fraction_values(model: CompositeModel, values: Mapping[str, float]) -> bool:
    """Return True when ``values`` carries old ``fraction_<k>`` keys for ``model``.

    Only the numbers that the model's fraction groups would have used under the
    retired positional scheme count — a stray ``fraction_9`` unrelated to any
    group does not trigger migration.
    """
    for group_numbers in _legacy_fraction_numbering(model):
        if any(f"fraction_{k}" in values for k in group_numbers):
            return True
    return False


def migrate_legacy_fraction_values(
    model: CompositeModel, values: Mapping[str, float]
) -> dict[str, float]:
    """Convert a legacy fraction value dict to the n-1 free-parameter scheme.

    For each group whose legacy keys ``fraction_a..fraction_b`` are present (one
    per term), compute the old normalized weights (clamp ≥ 0, divide by the sum;
    equal weights when the sum ≤ 1e-30) and assign the first n-1 to the group's
    new free parameters, dropping the last (now the derived remainder). All
    non-fraction keys pass through untouched; consumed legacy keys are removed.
    The model's structural :meth:`CompositeModel.to_dict` is unaffected — only
    value dicts need migrating.
    """
    migrated = dict(values)
    for group, group_numbers in zip(
        model.fraction_groups, _legacy_fraction_numbering(model), strict=True
    ):
        legacy_keys = [f"fraction_{k}" for k in group_numbers]
        if not any(key in migrated for key in legacy_keys):
            continue
        raw = [max(_coerce_float(migrated.get(key, 0.0)), 0.0) for key in legacy_keys]
        total = sum(raw)
        if total <= 1e-30:
            weights = [1.0 / float(len(raw))] * len(raw)
        else:
            weights = [value / total for value in raw]
        # Assign the first n-1 normalized weights to the new free parameters;
        # the last term is the derived remainder and gets no key.
        term_starts = model._fraction_group_term_starts(group)
        for idx, weight in zip(term_starts[:-1], weights[:-1], strict=True):
            migrated[model._fraction_param_name(idx)] = weight
        for key in legacy_keys:
            migrated.pop(key, None)
    return migrated


def _legacy_fraction_rename_map(model: CompositeModel) -> dict[str, str | None]:
    """Map each legacy ``fraction_<k>`` name to its new name (or ``None`` to drop).

    For every group the positional legacy keys ``fraction_a..fraction_b`` (one
    per additive term) map to the group's term starts ``t_1..t_n``. Under the
    n-1 scheme ``t_1..t_{n-1}`` become free parameters (their new names) and
    ``t_n`` — the derived remainder — has no parameter, so its legacy key maps
    to ``None`` (drop). Only names that a value/entry dict must be rewritten
    against appear here.
    """
    rename: dict[str, str | None] = {}
    for group, group_numbers in zip(
        model.fraction_groups, _legacy_fraction_numbering(model), strict=True
    ):
        term_starts = model._fraction_group_term_starts(group)
        for number, idx in zip(group_numbers[:-1], term_starts[:-1], strict=True):
            rename[f"fraction_{number}"] = model._fraction_param_name(idx)
        rename[f"fraction_{group_numbers[-1]}"] = None
    return rename


def migrate_legacy_fraction_parameter_entries(
    model: CompositeModel, parameters: list[dict]
) -> list[dict]:
    """Migrate a list of parameter-state dicts from the legacy fraction scheme.

    Each entry is a GUI/serialised parameter-state dict carrying at least a
    ``"name"`` and ``"value"`` (plus metadata such as ``fixed``/``min``/``max``/
    ``uncertainty``/``type``/``bounds``). Legacy ``fraction_<k>`` entries are
    rewritten in place: the first n-1 terms of each group keep their metadata but
    are renamed to the new free-parameter name and take the normalised migrated
    weight as their value; the dropped last-term entry is removed. Every other
    entry passes through untouched, preserving order. A no-op (returns a shallow
    copy) when the parameter list carries no legacy fraction keys for the model.
    """
    if not has_legacy_fraction_values(
        model, {str(entry.get("name", "")): 0.0 for entry in parameters}
    ):
        return [dict(entry) for entry in parameters]
    value_map = {
        str(entry["name"]): entry.get("value", 0.0)
        for entry in parameters
        if isinstance(entry, dict) and "name" in entry
    }
    if not has_legacy_fraction_values(model, value_map):
        return [dict(entry) for entry in parameters]

    migrated_values = migrate_legacy_fraction_values(model, value_map)
    rename = _legacy_fraction_rename_map(model)

    migrated: list[dict] = []
    for entry in parameters:
        if not isinstance(entry, dict):
            migrated.append(entry)
            continue
        name = str(entry.get("name", ""))
        if name in rename:
            new_name = rename[name]
            if new_name is None:
                # Dropped derived-remainder term: no parameter under the new scheme.
                continue
            new_entry = dict(entry)
            new_entry["name"] = new_name
            new_entry["value"] = migrated_values.get(new_name, entry.get("value"))
            migrated.append(new_entry)
        else:
            migrated.append(dict(entry))
    return migrated
