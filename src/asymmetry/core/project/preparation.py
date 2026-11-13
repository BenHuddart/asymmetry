"""Prepare owned project declarations before a caller replaces active state.

Availability is resolved later: source files and model plugins need not exist,
while historical analysis membership and constraint archives remain inspectable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding, SourceDeclaration
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.project.calibration import PreparedGrouping, calibration_values
from asymmetry.core.project.json_values import decode_non_finite
from asymmetry.core.project.profiles import GroupingProfile
from asymmetry.core.representation.container import DatasetRepresentations
from asymmetry.core.representation.global_fit_study import GlobalFitStudy
from asymmetry.core.representation.group import DataGroup
from asymmetry.core.representation.joint_fit import JointFit
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import FitSeries
from asymmetry.core.transform.rotating_frame import RotatingFrame


class ProjectValidationError(ValueError):
    """Malformed owned project content, with its serialized field location."""


@dataclass(frozen=True)
class PreparedProject:
    """Detached canonical JSON and its existing domain owners, ready to apply.

    Applying transfers ownership to the session; the objects are not a reusable
    snapshot of widgets or a certification of external source contents.
    """

    state: dict
    profiles: list[GroupingProfile]
    model: ProjectModel
    studies: dict[str, GlobalFitStudy]
    groupings: dict[int, PreparedGrouping]
    sources: tuple[SourceDeclaration, ...] = ()


def _shape(value, expected, path):
    if not isinstance(value, expected):
        raise ProjectValidationError(f"{path}: expected {expected.__name__}")
    return value


def _fields(data, keys, expected, path, *, nullable=True):
    for key in keys:
        if key in data and (not nullable or data[key] is not None):
            _shape(data[key], expected, f"{path}.{key}")


def _identity(value, expected, path):
    _shape(value, expected, path)
    if isinstance(value, bool) or (expected is str and not value):
        raise ProjectValidationError(f"{path}: expected a nonempty {expected.__name__} identity")
    return value


def _unique(value, seen, path):
    if value in seen:
        raise ProjectValidationError(f"{path}: duplicate identity {value!r}")
    seen.add(value)


def _construct(constructor, value, path):
    try:
        result = constructor(value)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ProjectValidationError(f"{path}: {exc}") from exc
    if result is None:
        raise ProjectValidationError(f"{path}: invalid {constructor.__qualname__} declaration")
    return result


def _rows(data, key, path):
    rows = data.get(key, [])
    _shape(rows, list, f"{path}.{key}")
    for index, row in enumerate(rows):
        _shape(row, dict, f"{path}.{key}[{index}]")
    return rows


def _analysis_shape(record, path):
    """Check active containers, leaving names, draft rows and archives to their owners."""
    _fields(
        record,
        (
            "model",
            "canonical_model",
            "composite_model",
            "result",
            "ui_state",
            "analysis_history",
            "param_roles",
            "recipe",
            "extra",
        ),
        dict,
        path,
    )
    for index, row in enumerate(_rows(record, "parameters", path)):
        _shape(row.get("name"), str, f"{path}.parameters[{index}].name")
        _fields(row, ("tie",), dict, f"{path}.parameters[{index}]")
    # The archive may contain original names, constraints and unset uncertainties.
    # It is never traversed as an active parameter table.


def _fit_editor_shape(state, path, *, floating_parameters=False):
    _analysis_shape(state, path)
    if floating_parameters:
        for index, row in enumerate(state.get("parameters", [])):
            value = row.get("value")
            if isinstance(value, str) and not value.strip():
                continue
            _numeric_fields(row, ("value",), f"{path}.parameters[{index}]")
    for key in ("group_parameters", "group_model_parameters"):
        _rows(state, key, path)
    for key in ("single_fit_state", "global_fit_state", "ui_state", "single", "batch"):
        if state.get(key) is not None:
            _shape(state[key], dict, f"{path}.{key}")
            _fit_editor_shape(
                state[key],
                f"{path}.{key}",
                floating_parameters=key in ("global_fit_state", "batch"),
            )
    for key in ("states_by_run", "states_by_run_set"):
        if state.get(key) is not None:
            _shape(state[key], dict, f"{path}.{key}")
            for identity, child in state[key].items():
                child_path = f"{path}.{key}[{identity!r}]"
                _shape(child, dict, child_path)
                _fit_editor_shape(child, child_path, floating_parameters=floating_parameters)


def _numeric_fields(state, keys, path):
    for key in keys:
        value = state.get(key)
        if value is not None:
            try:
                float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProjectValidationError(f"{path}.{key}: expected a number") from exc


def _curve_shape(curve, path):
    _shape(curve, dict, path)
    for key in ("t", "y"):
        values = _shape(curve.get(key), list, f"{path}.{key}")
        for index, value in enumerate(values):
            _numeric_fields({"value": value}, ("value",), f"{path}.{key}[{index}]")
    if len(curve["t"]) != len(curve["y"]):
        raise ProjectValidationError(f"{path}: curve coordinates must have equal lengths")


def _plot_shape(plot, path):
    _fields(
        plot,
        (
            "rotating_frame",
            "rrf",
            "projection_memory",
            "axis_limits",
            "fit_curves",
            "fit_curves_by_key",
            "fit_components_by_run",
            "fit_components_by_key",
            "fit_metadata",
            "fit_metadata_by_key",
            "annotations_by_group",
        ),
        dict,
        path,
        nullable=False,
    )
    for identity, selected in (plot.get("projection_memory") or {}).items():
        _shape(selected, list, f"{path}.projection_memory[{identity!r}]")
    _numeric_fields(plot, ("fit_x_min", "fit_x_max"), path)
    if plot.get("fit_curve"):
        _curve_shape(plot["fit_curve"], f"{path}.fit_curve")
    for key in ("fit_curves", "fit_curves_by_key"):
        for identity, curve in (plot.get(key) or {}).items():
            _curve_shape(curve, f"{path}.{key}[{identity!r}]")
            if key == "fit_curves":
                try:
                    int(identity)
                except (ValueError, TypeError) as exc:
                    raise ProjectValidationError(
                        f"{path}.{key}: expected integer run keys"
                    ) from exc
    for key in ("fit_components_by_run", "fit_components_by_key"):
        for identity, entries in (plot.get(key) or {}).items():
            _shape(entries, list, f"{path}.{key}[{identity!r}]")
            if key == "fit_components_by_run":
                _construct(int, identity, f"{path}.{key}[{identity!r}]")
            for index, entry in enumerate(entries):
                _shape(entry, dict, f"{path}.{key}[{identity!r}][{index}]")
    if "axis_limits" in plot:
        limits = plot["axis_limits"]
        axes = _shape(limits.get("axes"), dict, f"{path}.axis_limits.axes")
        for axis, state in axes.items():
            axis_path = f"{path}.axis_limits.axes[{axis!r}]"
            _shape(state, dict, axis_path)
            missing = {"auto", "held", "quantity"} - state.keys()
            if missing:
                raise ProjectValidationError(f"{axis_path}: missing fields {sorted(missing)}")
            if state["held"] is not None:
                values = _shape(state["held"], list, f"{axis_path}.held")
                if len(values) != 2:
                    raise ProjectValidationError(f"{axis_path}.held: expected two bounds")
                for index, value in enumerate(values):
                    bound = _construct(float, value, f"{axis_path}.held[{index}]")
                    if not math.isfinite(bound):
                        raise ProjectValidationError(
                            f"{axis_path}.held[{index}]: expected a finite axis bound"
                        )


def _construct_optional_identity(value, path):
    """Decode optional source facts at the existing project input boundary."""
    return (
        None
        if value is None
        else _construct(ContentIdentity.from_dict, value, f"{path}.content_identity")
    )


def _grouping_shape(record, path, sources):
    _fields(
        record,
        (
            "groups",
            "group_names",
            "extra",
            "background_run",
            "alpha_policy",
            "beta_policy",
            "deadtime_policy",
            "background_policy",
            "t0_policy",
            "good_window_policy",
            "fingerprint",
        ),
        dict,
        path,
    )
    for key in ("alpha_policy", "beta_policy", "deadtime_policy"):
        if key in record:
            policy_path = f"{path}.{key}"
            policy = _shape(record[key], dict, policy_path)
            for scalar in ("value", "manual_us", "estimated_us"):
                if scalar in policy:
                    _construct(float, policy[scalar], f"{policy_path}.{scalar}")
            _numeric_fields(policy, ("error",), policy_path)
            if "values" in policy:
                for index, value in enumerate(
                    _shape(policy["values"], list, f"{policy_path}.values")
                ):
                    _construct(float, value, f"{policy_path}.values[{index}]")
    fingerprint = record.get("fingerprint")
    if fingerprint is not None:
        _fields(fingerprint, ("instrument",), str, f"{path}.fingerprint")
        if "histogram_count" in fingerprint:
            _identity(fingerprint["histogram_count"], int, f"{path}.fingerprint.histogram_count")
    groups = record.get("groups")
    if isinstance(groups, dict):
        for key, members in groups.items():
            _shape(members, list, f"{path}.groups[{key!r}]")
    reference = record.get("background_run")
    if reference is not None:
        ref_path = f"{path}.background_run"
        _fields(reference, ("source_file",), str, ref_path)
        identity = _construct_optional_identity(reference.get("content_identity"), ref_path)
        sources.append(
            SourceDeclaration(
                "reference",
                reference.get("source_file", ""),
                identity,
                reference.get("run_number", reference.get("legacy_run_number")),
                reference.get("period_number"),
            )
        )
        _fields(reference, ("timing", "scale_request", "historical_scale"), dict, ref_path)
        for key in ("run_number", "period_number", "legacy_run_number"):
            if reference.get(key) is not None:
                _identity(reference[key], int, f"{ref_path}.{key}")
    background = record.get("background_policy")
    if background is not None and background.get("details") is not None:
        _shape(background["details"], dict, f"{path}.background_policy.details")
        _grouping_shape(background["details"], f"{path}.background_policy.details", sources)
    _fields(record, ("projection_alpha_policies",), dict, path)
    for calibration in _rows(record, "calibration_records", path):
        identity = _construct_optional_identity(calibration.get("content_identity"), path)
        sources.append(
            SourceDeclaration(
                "calibration",
                calibration.get("source_file", ""),
                identity,
                calibration.get("source_run_number"),
                calibration.get("period_number"),
            )
        )
    for key in ("alpha_policy", "beta_policy"):
        quick = record.get(key, {}).get("quick_calibration")
        if quick is not None:
            identity = _construct_optional_identity(quick.get("content_identity"), path)
            sources.append(
                SourceDeclaration(
                    "calibration",
                    quick.get("source_file", ""),
                    identity,
                    quick.get("run_number"),
                    quick.get("period_number"),
                )
            )
    for index, projection in enumerate(_rows(record, "projections", path)):
        _fields(projection, ("label",), str, f"{path}.projections[{index}]")


def _representation_shape(representations, path):
    _shape(representations, dict, path)
    for key, rep in representations.items():
        rep_path = f"{path}[{key!r}]"
        _shape(rep, dict, rep_path)
        if rep.get("rep_type", key) != key:
            raise ProjectValidationError(f"{rep_path}.rep_type: must match representation key")
        _fields(rep, ("recipe", "trend_state", "result_metadata", "single_fits"), dict, rep_path)
        for projection, fit_set in (rep.get("single_fits") or {}).items():
            fit_path = f"{rep_path}.single_fits[{projection!r}]"
            _shape(fit_set, dict, fit_path)
            ids = set()
            for index, slot in enumerate(_rows(fit_set, "fits", fit_path)):
                slot_path = f"{fit_path}.fits[{index}]"
                _analysis_shape(slot, slot_path)
                if slot.get("ui_state") is not None:
                    _fit_editor_shape(slot["ui_state"], f"{slot_path}.ui_state")
                _unique(_identity(slot.get("fit_id"), str, f"{slot_path}.fit_id"), ids, slot_path)
            selected = fit_set.get("open_id")
            if selected is not None and selected not in ids:
                raise ProjectValidationError(f"{fit_path}.open_id: must name a saved fit")


def _fit_domain_shape(block, domain, path):
    _shape(block, dict, path)
    tag = block.get("domain")
    if tag is not None and str(tag).strip().lower() != domain:
        raise ProjectValidationError(f"{path}.domain: must match its owning domain")
    _fields(
        block, ("single_fit_state", "global_fit_state", "fit_ui_state"), dict, path, nullable=False
    )
    for key in ("single_fit_state", "global_fit_state"):
        if key in block:
            _fit_editor_shape(
                block[key], f"{path}.{key}", floating_parameters=key == "global_fit_state"
            )


def _ui_shape(state):
    """Known restore containers only; arbitrary user metadata is opaque."""
    _fields(
        state,
        (
            "fit_states",
            "fit_parameters_state",
            "fourier_state",
            "fourier_spectra_state",
            "multi_group_fit_state",
            "maxent_state",
            "maxent_state_by_run",
            "view_modes_state",
            "joint_fit_window_state",
            "global_parameter_fit_window_state",
            "knight_shift_analysis_state",
        ),
        dict,
        "project",
    )
    for key in ("plot_state", "browser_state"):
        if key in state:
            _shape(state[key], dict, f"project.{key}")
    plot = state.get("plot_state") or {}
    _fields(plot, ("frequency_plot_state",), dict, "project.plot_state", nullable=False)
    _fields(plot, ("workspace_state",), dict, "project.plot_state")
    if plot.get("current_run_number") is not None:
        _identity(plot["current_run_number"], int, "project.plot_state.current_run_number")
    _plot_shape(plot, "project.plot_state")
    if plot.get("frequency_plot_state") is not None:
        _plot_shape(plot["frequency_plot_state"], "project.plot_state.frequency_plot_state")
    browser = state.get("browser_state") or {}
    _fields(
        browser,
        ("filters", "temperature_from_log_overrides", "field_from_log_overrides"),
        dict,
        "project.browser_state",
        nullable=False,
    )
    _fields(
        browser,
        ("selected_run_numbers", "selected_group_ids", "extra_columns", "data_groups"),
        list,
        "project.browser_state",
        nullable=False,
    )
    if "sort_column" in browser:
        _construct(int, browser["sort_column"], "project.browser_state.sort_column")
    for key, values in (browser.get("filters") or {}).items():
        path = f"project.browser_state.filters[{key!r}]"
        _construct(int, key, path)
        _shape(values, list, path)
        _construct(set, values, path)
    for key in ("selected_run_numbers", "selected_group_ids"):
        if key in browser:
            _construct(set, browser[key], f"project.browser_state.{key}")
    parameters = state.get("fit_parameters_state") or {}
    _fields(parameters, ("inferred_x_key",), str, "project.fit_parameters_state")
    fourier = state.get("fourier_state") or {}
    _fields(fourier, ("auto_phase_method",), str, "project.fourier_state", nullable=False)
    if "phase_fit_tolerance_degrees" in fourier:
        _construct(
            float,
            fourier["phase_fit_tolerance_degrees"],
            "project.fourier_state.phase_fit_tolerance_degrees",
        )
    if fourier.get("exclusion_ranges") is not None:
        for index, row in enumerate(
            _shape(fourier["exclusion_ranges"], list, "project.fourier_state.exclusion_ranges")
        ):
            path = f"project.fourier_state.exclusion_ranges[{index}]"
            _shape(row, list, path)
            if len(row) != 2:
                raise ProjectValidationError(f"{path}: expected centre and width")
            for value in row:
                _construct(float, value, path)
    if "padding" in fourier:
        _identity(fourier["padding"], int, "project.fourier_state.padding")
    _numeric_fields(
        fourier,
        ("phase_fit_min_mhz", "phase_fit_max_mhz"),
        "project.fourier_state",
    )
    fit_states = state.get("fit_states") or {}
    for domain in ("time", "frequency"):
        if domain in fit_states:
            _fit_domain_shape(fit_states[domain], domain, f"project.fit_states[{domain!r}]")
    if "time" not in fit_states:
        for key in ("single_fit_state", "global_fit_state", "fit_ui_state"):
            if state.get(key) is not None:
                _shape(state[key], dict, f"project.{key}")
                if key != "fit_ui_state":
                    _fit_editor_shape(
                        state[key], f"project.{key}", floating_parameters=key == "global_fit_state"
                    )
    if "frequency" not in fit_states and state.get("frequency_fit_state") is not None:
        _fit_domain_shape(state["frequency_fit_state"], "frequency", "project.frequency_fit_state")
    if state.get("multi_group_fit_state") is not None:
        _fit_editor_shape(state["multi_group_fit_state"], "project.multi_group_fit_state")
    views = state.get("view_modes_state")
    if views is not None:
        _rows(views, "modes", "project.view_modes_state")


def prepare_project(
    state: dict, *, checkpoint: Callable[[], None] = lambda: None
) -> PreparedProject:
    """Migrate/retry once, then prepare detached canonical project owners.

    ``checkpoint`` may raise cancellation between declarations. No source file
    is opened and no execution attribution is changed by preparation.
    """
    from asymmetry.core.project.schema import UnsupportedSchemaVersion, migrate_to_current

    checkpoint()
    _shape(state, dict, "project")
    # Reject the envelope before migrations can consume malformed containers.
    if "datasets" not in state:
        raise ProjectValidationError("project.datasets: missing required keys: datasets")
    _shape(state["datasets"], list, "project.datasets")
    if "schema_version" not in state:
        raise ProjectValidationError(
            "project.schema_version: missing required keys: schema_version"
        )
    version = state.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProjectValidationError("project.schema_version: expected int")
    try:
        canonical = migrate_to_current(state)
    except UnsupportedSchemaVersion:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ProjectValidationError(f"project: malformed migration input: {exc}") from exc
    return _prepare_canonical(canonical, checkpoint)


def _prepare_canonical(state, checkpoint):
    _ui_shape(state)
    model = ProjectModel()
    groupings = {}
    sources = []
    runs = set()
    for key in ("datasets", "combined_datasets"):
        if key in state:
            _shape(state[key], list, f"project.{key}")
    for key, id_key in (("datasets", "run_number"), ("combined_datasets", "combined_run_number")):
        for index, entry in enumerate(_rows(state, key, "project")):
            checkpoint()
            path = f"project.{key}[{index}]"
            number = _identity(entry.get(id_key), int, f"{path}.{id_key}")
            _unique(number, runs, f"{path}.{id_key}")
            if "source_file" in entry:
                _shape(entry["source_file"], str, f"{path}.source_file")
            binding = _construct(
                SourceBinding.from_dict, entry.get("source_binding"), f"{path}.source_binding"
            )
            if key == "datasets":
                sources.append(
                    SourceDeclaration(
                        "sample", entry.get("source_file", ""), binding.expected, number
                    )
                )
            for member in _rows(entry, "constituent_sources", path):
                member_binding = _construct(
                    SourceBinding.from_dict,
                    member.get("source_binding"),
                    f"{path}.constituent_sources",
                )
                sources.append(
                    SourceDeclaration(
                        "constituent",
                        member.get("source_file", ""),
                        member_binding.expected,
                        member.get("source_run_number", member.get("run_number")),
                        member.get("data_period_number", member.get("period_number")),
                        member.get("coefficient", 1.0),
                        member.get("input_ordinal"),
                        member.get("dae_period_number"),
                    )
                )
            _fields(entry, ("profile",), str, path)
            if "metadata_overrides" in entry:
                _shape(entry["metadata_overrides"], dict, f"{path}.metadata_overrides")
            _fields(
                entry,
                ("grouping_overrides", "representations", "rotating_frame"),
                dict,
                path,
            )
            if entry.get("grouping_overrides") is not None:
                _grouping_shape(entry["grouping_overrides"], f"{path}.grouping_overrides", sources)
                groupings[number] = PreparedGrouping(
                    _construct(
                        calibration_values,
                        entry["grouping_overrides"],
                        f"{path}.grouping_overrides",
                    )
                )
            if key == "combined_datasets":
                source_numbers = _shape(
                    entry.get("source_run_numbers"), list, f"{path}.source_run_numbers"
                )
                for source in source_numbers:
                    _identity(source, int, f"{path}.source_run_numbers")
            if entry.get("representations") is not None:
                _representation_shape(entry["representations"], f"{path}.representations")
                model.datasets[number] = _construct(
                    DatasetRepresentations.from_dict,
                    {"run_number": number, "representations": entry["representations"]},
                    f"{path}.representations",
                )
                for representation in model.datasets[number]:
                    for record in representation.result_metadata.get("effective_recipes", []):
                        if record.get("effective_recipe") is not None:
                            recipe = _construct(
                                EffectiveReductionRecipe.from_dict,
                                record["effective_recipe"],
                                f"{path}.result_metadata.effective_recipes",
                            )
                            sources.extend(recipe.sources())
            if entry.get("rotating_frame") is not None:
                model.rotating_frames[number] = _construct(
                    RotatingFrame.from_dict, entry["rotating_frame"], f"{path}.rotating_frame"
                )
    profiles = []
    names = set()
    for index, raw in enumerate(_rows(state, "grouping_profiles", "project")):
        checkpoint()
        path = f"project.grouping_profiles[{index}]"
        _grouping_shape(raw, path, sources)
        name = _identity(raw.get("name"), str, f"{path}.name")
        profile = _construct(GroupingProfile.from_dict, raw, path)
        fingerprint = profile.fingerprint
        _unique(
            (name, fingerprint.instrument.strip().lower(), fingerprint.histogram_count),
            names,
            f"{path}.name",
        )
        profiles.append(profile)
    studies = {}
    for key, id_key, constructor, registry in (
        ("batches", "batch_id", FitSeries.from_dict, model.batches),
        ("data_groups", "group_id", DataGroup.from_dict, model.data_groups),
        ("joint_fits", "joint_id", JointFit.from_dict, model.joint_fits),
        ("global_fit_studies", "study_id", GlobalFitStudy.from_dict, studies),
    ):
        seen = set()
        for index, raw in enumerate(_rows(state, key, "project")):
            checkpoint()
            path = f"project.{key}[{index}]"
            identity = _identity(raw.get(id_key), str, f"{path}.{id_key}")
            _unique(identity, seen, f"{path}.{id_key}")
            _analysis_shape(raw, path)
            for member_key in (
                "member_run_numbers",
                "last_fitted_members",
                "excluded_run_numbers",
                "trend_excluded_runs",
            ):
                if raw.get(member_key) is not None:
                    for member in _shape(raw[member_key], list, f"{path}.{member_key}"):
                        _identity(member, int, f"{path}.{member_key}")
            _fields(raw, ("results_by_run", "member_source_run", "member_frames"), dict, path)
            for member, result in (raw.get("results_by_run") or {}).items():
                _shape(result, dict, f"{path}.results_by_run[{member!r}]")
            if key == "batches" and raw.get("recipe") is not None:
                _analysis_shape(raw["recipe"], f"{path}.recipe")
                _fields(raw["recipe"], ("fit_range", "coadd"), dict, f"{path}.recipe")
            if key == "joint_fits":
                members = _shape(raw.get("member_batch_ids"), list, f"{path}.member_batch_ids")
                for member in members:
                    _identity(member, str, f"{path}.member_batch_ids")
                for index, row in enumerate(_rows(raw, "shared", path)):
                    row_path = f"{path}.shared[{index}]"
                    _shape(row.get("members"), dict, f"{row_path}.members")
                    _construct(float, row.get("value", 0.0), f"{row_path}.value")
                    _numeric_fields(row, ("min", "max"), row_path)
                    if not row.get("name") or len(row["members"]) < 2:
                        raise ProjectValidationError(
                            f"{row_path}: shared row requires two named members"
                        )
            _fields(raw, ("config",), dict, path)
            if key == "global_fit_studies":
                _rows(raw, "groups", path)
            record = _construct(constructor, raw, path)
            if key == "global_fit_studies" and len(record.groups) != len(raw.get("groups", [])):
                raise ProjectValidationError(f"{path}.groups: malformed group cannot be discarded")
            registry[identity] = record
    active = state.get("active_series") or {}
    _shape(active, dict, "project.active_series")
    for rep_type, batch_id in active.items():
        _identity(batch_id, str, f"project.active_series[{rep_type!r}]")
        series = model.batches.get(batch_id)
        if series is None or series.rep_type.value != rep_type:
            raise ProjectValidationError(
                f"project.active_series[{rep_type!r}]: must name a series of this representation"
            )
    model.active_series = dict(active)
    checkpoint()
    return PreparedProject(state, profiles, model, studies, groupings, tuple(sources))


def load_prepared_project(path: str | Path, *, checkpoint: Callable[[], None] = lambda: None):
    """Read project JSON and prepare its owned declarations without acquisition I/O."""
    checkpoint()
    raw = decode_non_finite(json.loads(Path(path).read_text(encoding="utf-8")))
    return prepare_project(raw, checkpoint=checkpoint)
