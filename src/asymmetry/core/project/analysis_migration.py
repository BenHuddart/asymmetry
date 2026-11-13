"""Convert released active analysis payloads before domain or editor restoration.

Frozen reduction recipes, calibration histories and execution attribution are
records of calculations, not editable parameter tables, and are never walked.
"""

from __future__ import annotations

import re
from copy import deepcopy

from asymmetry.core.fitting.analysis_history import constraint_signature
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.project.legacy_fractions import (
    _legacy_fraction_rename_map,
    migrate_legacy_fraction_parameter_entries,
)
from asymmetry.core.project.legacy_product_amplitudes import (
    _carries_legacy_names,
    _entry_is_fixed,
    _fold_plan,
    fold_legacy_product_amplitude_entries,
    fold_legacy_product_amplitude_names,
)

WIZARD_PAYLOAD_VERSION = 1


def analysis_changed_names(model: CompositeModel, names: list[str]) -> set[str]:
    """Identify changes evidenced by old names, never by unrelated auxiliaries."""
    present = set(names)
    exposed = set(model.param_names)
    changed = set(_legacy_fraction_rename_map(model)).intersection(present)
    if _carries_legacy_names(model, names):
        plan = _fold_plan(model)
        for fold in plan.folds:
            if (set(fold.legacy_names) - exposed).intersection(present):
                changed.update(fold.legacy_names)
                changed.add(fold.new_name)
        for distribution in plan.distributions:
            if set(distribution.k_names).intersection(present):
                changed.update(distribution.k_names)
                changed.update(distribution.target_names)
    return changed


def incompatible_bounds(row: dict) -> bool:
    """Nontrivial raw-scale bounds need explicit authoring after a fold."""
    constraint = constraint_signature([row])[0]
    minimum, maximum = constraint["min"], constraint["max"]
    if row["name"].startswith("fraction_"):
        return minimum not in (-float("inf"), 0.0) or maximum not in (float("inf"), 1.0)
    if row.get("fixed") and row.get("value") == 1.0:
        return False
    return minimum not in (-float("inf"), 0.0) or maximum != float("inf")


def incompatible_fixed_scales(model: CompositeModel, rows: list[dict]) -> bool:
    """Fixed raw factors can restrict a folded scale even with open saved bounds."""
    names = {row["name"]: row for row in rows}
    if not _carries_legacy_names(model, names):
        return False
    plan = _fold_plan(model)
    for fold in plan.folds:
        present = [names[name] for name in fold.legacy_names if name in names]
        if not present or all(_entry_is_fixed(row) for row in present):
            continue
        if _entry_is_fixed(present[0]) or any(
            _entry_is_fixed(row) and row.get("value") != 1.0 for row in present[1:]
        ):
            return True
    return any(
        any(
            name in names and (not _entry_is_fixed(names[name]) or names[name].get("value") != 1.0)
            for name in distribution.k_names
        )
        for distribution in plan.distributions
    )


def preserve_analysis(record: dict, model: CompositeModel, rows: list[dict]) -> bool:
    """Archive unresolved naming or constraints before changing active rows."""
    if any(component.missing for component in model.components):
        record["analysis_history"] = {"reason": "missing_model", "original": deepcopy(record)}
        return False
    changed = analysis_changed_names(model, [row["name"] for row in rows])

    affected = any(
        (
            row["name"] in changed
            and (
                row.get("tie")
                or row.get("expr")
                or row.get("link_group") is not None
                or (row["name"].startswith("fraction_") and row.get("fixed"))
                or incompatible_bounds(row)
            )
        )
        or (
            isinstance(row.get("tie"), dict)
            and any(row["tie"].get(key) in changed for key in ("main", "offset"))
        )
        or (row.get("expr") and changed.intersection(re.findall(r"\b[A-Za-z_]\w*\b", row["expr"])))
        for row in rows
    )
    if affected or incompatible_fixed_scales(model, rows):
        record["analysis_history"] = {"reason": "constraints", "original": deepcopy(record)}
    return True


def active_parameters(record: dict, model: CompositeModel, rows: list[dict]) -> list[dict]:
    """Snapshot values are usable; incompatible original constraints stay archival."""
    converted = analysis_parameters(model, rows)
    history = record.get("analysis_history")
    if history is not None and history["reason"] == "constraints":
        converted = [{**row, "tie": None, "expr": None, "link_group": None} for row in converted]
        changed = analysis_changed_names(model, [row["name"] for row in rows])
        for row in converted:
            if row["name"] in changed and not row["name"].startswith("f_"):
                row.update(min=-float("inf"), max=float("inf"))
                if "bounds" in row:
                    row["bounds"] = "-inf, inf"
            if row["name"].startswith("f_"):
                row.update(min=0.0, max=1.0)
                if "bounds" in row:
                    row["bounds"] = "0, 1"
        history["constraint_table"] = record.get("analysis_history", {}).get(
            "constraint_table", "parameters"
        )
        history["canonical_constraints"] = constraint_signature(converted)
    return converted


def pending_record(record: dict) -> bool:
    """A missing dependency is retried only at its versioned persistence boundary."""
    history = record.get("analysis_history")
    if history is None or history["reason"] != "missing_model":
        return False
    original = deepcopy(history["original"])
    record.clear()
    record.update(original)
    return True


def analysis_parameters(model: CompositeModel, entries: list[dict]) -> list[dict]:
    """Preserve released normalization and product folding conventions."""
    return fold_legacy_product_amplitude_entries(
        model, migrate_legacy_fraction_parameter_entries(model, entries)
    )


def analysis_names(model: CompositeModel, names) -> list[str]:
    """Map a stored name table with the same model as its parameter table."""
    rename = _legacy_fraction_rename_map(model)
    return list(
        fold_legacy_product_amplitude_names(
            model, [rename.get(name, name) for name in names if rename.get(name, name) is not None]
        )
    )


def analysis_result(result: dict, model: CompositeModel) -> None:
    """Rewrite only active result parameter fields, retaining recorded statistics."""
    parameters = result.get("parameters")
    if not isinstance(parameters, (dict, list)):
        return
    mapping = isinstance(parameters, dict)
    uncertainties = result.get("uncertainties", {}) or {}
    entries = (
        [{"name": name, "value": value} for name, value in parameters.items()]
        if mapping
        else parameters
    )
    asymmetric = result.get("uncertainties_asymmetric", {}) or {}
    entries = [
        {
            **row,
            **({"uncertainty": uncertainties[row["name"]]} if row["name"] in uncertainties else {}),
            **(
                {"uncertainty_asymmetric": asymmetric[row["name"]]}
                if row["name"] in asymmetric
                else {}
            ),
        }
        for row in entries
    ]
    converted = analysis_parameters(model, entries)
    result["parameters"] = (
        {row["name"]: row["value"] for row in converted}
        if mapping
        else [
            {k: v for k, v in row.items() if k not in {"uncertainty", "uncertainty_asymmetric"}}
            for row in converted
        ]
    )
    if "uncertainties" in result:
        result["uncertainties"] = {
            row["name"]: row["uncertainty"]
            for row in converted
            if row.get("uncertainty") is not None
        }
    if isinstance(result.get("uncertainties_asymmetric"), dict):
        mapped = {
            row["name"]: row["uncertainty_asymmetric"]
            for row in converted
            if row.get("uncertainty_asymmetric") is not None
        }
        if len(mapped) < len(asymmetric):
            result["parameter_history"] = {
                "parameters": deepcopy(parameters),
                "uncertainties_asymmetric": deepcopy(asymmetric),
            }
        result["uncertainties_asymmetric"] = mapped
    for key in ("params_at_bound", "parameter_names", "free_param_names"):
        if isinstance(result.get(key), list):
            result[key] = analysis_names(model, result[key])


def wizard_assessment(assessment: dict, *, legacy: bool = True) -> None:
    """An assessment's template owns all its local/global parameter names."""
    pending = pending_record(assessment)
    if not legacy and not pending:
        return
    descriptor = assessment.get("template", {}).get("model")
    if not isinstance(descriptor, dict):
        return
    model = CompositeModel.from_dict(descriptor, allow_missing=True)
    results = [
        assessment.get("fit_result"),
        *(assessment.get("fit_results_by_run", {}) or {}).values(),
    ]
    rows = [
        row
        for result in results
        if isinstance(result, dict)
        for row in result.get("parameters", [])
    ]
    rows.extend(assessment.get("global_parameters", []))
    if not preserve_analysis(assessment, model, rows):
        return
    for result in [
        assessment.get("fit_result"),
        *(assessment.get("fit_results_by_run", {}) or {}).values(),
    ]:
        if isinstance(result, dict):
            analysis_result(result, model)
            if (assessment.get("analysis_history") or {}).get("reason") == "constraints":
                result["parameters"] = active_parameters(assessment, model, result["parameters"])
    if isinstance(assessment.get("global_parameters"), list):
        assessment["global_parameters"] = active_parameters(
            assessment, model, assessment["global_parameters"]
        )
    for key in ("global_param_names", "local_param_names", "fixed_param_names"):
        if isinstance(assessment.get(key), list):
            assessment[key] = analysis_names(model, assessment[key])
    recommendations = []
    for entry in assessment.get("parameter_recommendations", []):
        if isinstance(entry, dict) and "name" in entry:
            names = analysis_names(model, [entry["name"]])
            if names:
                entry["name"] = names[0]
                recommendations.append(entry)

    if "parameter_recommendations" in assessment:
        assessment["parameter_recommendations"] = recommendations


def migrate_wizard_payload(payload: dict) -> dict:
    """Version the standalone/workdir recommendation codec independently of projects."""
    version = payload.get("analysis_payload_version", 0)
    if version not in (0, WIZARD_PAYLOAD_VERSION):
        raise ValueError(f"Unsupported wizard analysis payload version {version!r}")
    converted = deepcopy(payload)
    for assessment in converted.get("assessments", []):
        wizard_assessment(assessment, legacy=version == 0)
    for entry in converted.get("phase_assessments", []):
        wizard_assessment(entry["assessment"], legacy=version == 0)
    converted["analysis_payload_version"] = WIZARD_PAYLOAD_VERSION
    return converted


def fit_editor(state: dict, model: CompositeModel | None = None, *, legacy: bool = True) -> None:
    """Convert a form and its explicitly owned per-run/wizard cache children."""
    pending = pending_record(state)
    descriptor = state.get("composite_model")
    if isinstance(descriptor, dict):
        model = CompositeModel.from_dict(descriptor, allow_missing=True)
    if model is not None and (legacy or pending):
        table = "group_model_parameters" if state.get("group_model_parameters") else "parameters"
        rows = state.get(table, [])
        physics_model = model.with_default_fraction_groups()
        context_model = physics_model if table == "group_model_parameters" else model
        if preserve_analysis(state, context_model, rows):
            if state.get("analysis_history") is not None:
                state["analysis_history"]["constraint_table"] = table
            for key in ("parameters", "group_parameters", "group_model_parameters"):
                if isinstance(state.get(key), list):
                    state[key] = group_parameter_rows(
                        physics_model if key != "parameters" else model, state[key]
                    )
            if table in state:
                state[table] = active_parameters(state, context_model, rows)
                if state.get("analysis_history") is not None and any(
                    key in state for key in ("group_parameters", "group_model_parameters")
                ):
                    active_rows = [
                        row for row in state[table] if row["name"] in context_model.param_names
                    ]
                    state["analysis_history"]["canonical_constraints"] = constraint_signature(
                        active_rows
                    )
    for key in ("single_fit_state", "global_fit_state", "ui_state", "single", "batch"):
        if isinstance(state.get(key), dict):
            fit_editor(state[key], model, legacy=legacy)
    for key in ("states_by_run", "states_by_run_set"):
        for child in (state.get(key, {}) or {}).values():
            if isinstance(child, dict):
                fit_editor(child, model, legacy=legacy)
    caches = [state.get("wizard_state")]
    stored = state.get("wizard_state_by_run_set", [])
    caches.extend(stored.values() if isinstance(stored, dict) else stored)
    for cache in caches:
        if isinstance(cache, dict) and isinstance(cache.get("recommendation"), dict):
            cache["recommendation"] = migrate_wizard_payload(cache["recommendation"])


def fit_slot(slot: dict, *, legacy: bool = True) -> None:
    pending = pending_record(slot)
    descriptor = slot.get("model")
    model = (
        CompositeModel.from_dict(descriptor, allow_missing=True)
        if isinstance(descriptor, dict)
        else None
    )
    if model is not None and (legacy or pending):
        if preserve_analysis(slot, model, slot.get("parameters", [])):
            slot["parameters"] = active_parameters(slot, model, slot.get("parameters", []))
            if isinstance(slot.get("result"), dict):
                analysis_result(slot["result"], model)
    if isinstance(slot.get("ui_state"), dict):
        fit_editor(slot["ui_state"], model, legacy=legacy)
        if (
            slot.get("analysis_history") is not None
            and slot["ui_state"].get("analysis_history") is None
        ):
            slot["ui_state"]["analysis_history"] = deepcopy(slot["analysis_history"])


def group_parameter_rows(model: CompositeModel, rows: list[dict]) -> list[dict]:
    """Each group's values use the same model and normalization as the outer table."""
    converted = analysis_parameters(model, rows)
    groups = {key for row in rows for key in row.get("group_values", {})}
    for group in groups:
        entries = [
            {**row, "value": row.get("group_values", {}).get(group, row.get("value", 0.0))}
            for row in rows
        ]
        values = {row["name"]: row["value"] for row in analysis_parameters(model, entries)}
        for row in converted:
            row.setdefault("group_values", {})[group] = values[row["name"]]
    return converted


def _project_dataset_analysis(dataset: dict, legacy: bool) -> None:
    representations = dataset.get("representations")
    if not isinstance(representations, dict):
        return
    for rep in representations.values():
        if not isinstance(rep, dict):
            continue
        for fit_set in (rep.get("single_fits", {}) or {}).values():
            for slot in fit_set.get("fits", []):
                fit_slot(slot, legacy=legacy)
        for key in ("fit",):
            if isinstance(rep.get(key), dict):
                fit_slot(rep[key], legacy=legacy)


def migrate_analysis_state(data: dict, *, legacy: bool = True) -> dict:
    """Project v38→v39: visit active fit records through their schema owners."""
    migrated = deepcopy(data)
    for key in ("datasets", "combined_datasets"):
        for index, dataset in enumerate(migrated.get(key, []) or []):
            if not isinstance(dataset, dict):
                continue
            try:
                _project_dataset_analysis(dataset, legacy)
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError(f"project.{key}[{index}].representations: {exc}") from exc
    for state in (migrated.get("fit_states", {}) or {}).values():
        if isinstance(state, dict):
            fit_editor(state, legacy=legacy)
    if isinstance(migrated.get("multi_group_fit_state"), dict):
        fit_editor(migrated["multi_group_fit_state"], legacy=legacy)
    models = {}
    changed_by_series = {}
    for series in migrated.get("batches", []) or []:
        if not isinstance(series, dict):
            continue
        pending = pending_record(series)
        descriptor = series.get("canonical_model")
        if not isinstance(descriptor, dict):
            continue
        model = CompositeModel.from_dict(descriptor, allow_missing=True)
        models[series["batch_id"]] = model
        source_names = list(series.get("param_roles", {})) + [
            row["name"] for row in series.get("recipe", {}).get("parameters", [])
        ]
        changed_by_series[series["batch_id"]] = (
            analysis_changed_names(model, source_names)
            if not model.missing_component_names
            else set()
        )

        if not legacy and not pending:
            continue
        if not preserve_analysis(series, model, series.get("recipe", {}).get("parameters", [])):
            continue
        if series.get("analysis_history") is None and any(
            name in changed_by_series[series["batch_id"]] and str(role).lower() != "local"
            for name, role in series.get("param_roles", {}).items()
        ):
            series["analysis_history"] = {"reason": "constraints", "original": deepcopy(series)}
        for key in ("param_roles", "shared_params"):
            if isinstance(series.get(key), dict):
                series[key] = {
                    names[0]: value
                    for name, value in series[key].items()
                    if (names := analysis_names(model, [name]))
                }
        if isinstance(series.get("nuisance_params"), list):
            series["nuisance_params"] = analysis_names(model, series["nuisance_params"])
        if isinstance(series.get("recipe"), dict):
            from asymmetry.core.representation.series import normalise_recipe

            recipe = series["recipe"]
            recipe["parameters"] = active_parameters(series, model, recipe.get("parameters", []))
            for row in recipe["parameters"]:
                row.setdefault(
                    "bounds", f"{row.get('min', -float('inf'))}, {row.get('max', float('inf'))}"
                )
                row.setdefault(
                    "type", str(series.get("param_roles", {}).get(row["name"], "local")).title()
                )
            series["recipe"] = normalise_recipe(recipe)
            if series.get("analysis_history") is not None:
                history = series["analysis_history"]
                history["canonical_constraints"] = constraint_signature(
                    [
                        row
                        for row in series["recipe"]["parameters"]
                        if row["name"] in model.param_names
                    ]
                )
        for result in (series.get("results_by_run", {}) or {}).values():
            if isinstance(result, dict):
                analysis_result(result, model)
    for joint in migrated.get("joint_fits", []):
        prior_changes = (joint.get("analysis_history") or {}).get("changed_members", {})
        pending = pending_record(joint)
        if not legacy and not pending:
            continue
        members = joint.get("member_batch_ids", [])
        if any(models[batch].missing_component_names for batch in members if batch in models):
            joint["analysis_history"] = {
                "reason": "missing_model",
                "original": deepcopy(joint),
                "changed_members": {
                    batch: sorted(
                        set(prior_changes.get(batch, [])) | changed_by_series.get(batch, set())
                    )
                    for batch in members
                },
            }
            continue
        original = deepcopy(joint)
        shared = []
        affected = False
        for row in joint.get("shared", []):
            if any(
                name in (changed_by_series.get(batch, set()) | set(prior_changes.get(batch, [])))
                for batch, name in row.get("members", {}).items()
            ):
                affected = True
                continue
            shared.append(row)
        if "shared" in joint:
            joint["shared"] = shared
        if affected:
            from asymmetry.core.fitting.analysis_history import shared_constraint_signature

            joint["analysis_history"] = {
                "reason": "constraints",
                "original": original,
                "canonical_shared": shared_constraint_signature(shared),
            }
    from asymmetry.core.project.correction_migration import project_corrections

    if legacy:
        project_corrections(migrated)
    return migrated
