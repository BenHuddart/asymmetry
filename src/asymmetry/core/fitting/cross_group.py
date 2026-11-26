"""Prepare and execute cross-group parameter fits from recorded configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
    global_fit_parameter_model,
    parse_fit_windows,
    validate_fit_windows,
    windows_mask,
)


class CrossGroupFitInputError(ValueError):
    """A rejected user selection, classified for presentation by a front end."""

    def __init__(self, reason: Literal["windows", "range", "groups"], message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class CrossGroupFitInputs:
    """Detached engine inputs with the configuration snapshot they resolve.

    Arrays and configuration attribution are detached at the input boundary so
    subsequent editor changes cannot alter an already prepared execution.
    """

    model: ParameterCompositeModel
    groups: list[ParameterGroupData]
    global_params: list[str]
    local_params: list[str]
    fixed_params: dict[str, float]
    initial_params: dict[str, float]
    parameter_bounds: dict[str, tuple[float, float]]
    error_mode: ErrorMode
    error_value: float | None
    xerr: dict[str, np.ndarray] | None
    fit_x_min: float
    fit_x_max: float
    config: dict[str, object]


@dataclass
class CrossGroupFitRun:
    """Completed cross-group execution and its selected data/configuration."""

    model: ParameterCompositeModel
    result: CrossGroupFitResult
    fitted_groups: list[ParameterGroupData]
    fit_x_min: float
    fit_x_max: float
    config: dict[str, object]


def prepare_cross_group_fit(
    groups: list[ParameterGroupData], config: dict[str, object]
) -> CrossGroupFitInputs:
    """Resolve a user or recorded configuration once into detached fit inputs.

    Windows override the shared range; groups retaining fewer than two points
    are omitted. Removed parameter rows are dropped, duplicate rows use their
    first entry, and missing model parameters start Global at the model default
    with open bounds. Extra configuration attribution is preserved.
    """
    model_data = config.get("model")
    if not isinstance(model_data, dict):
        raise ValueError("Config is missing a model.")
    model = ParameterCompositeModel.from_dict(model_data)

    windows = parse_fit_windows(config.get("windows"))
    fit_x_min_raw = config.get("fit_x_min")
    fit_x_max_raw = config.get("fit_x_max")
    x_min = float(fit_x_min_raw) if isinstance(fit_x_min_raw, (int, float)) else -float("inf")
    x_max = float(fit_x_max_raw) if isinstance(fit_x_max_raw, (int, float)) else float("inf")

    if windows:
        try:
            validate_fit_windows(windows)
        except ValueError as exc:
            raise CrossGroupFitInputError("windows", str(exc)) from exc
    elif np.isfinite(x_min) and np.isfinite(x_max) and x_max <= x_min:
        raise CrossGroupFitInputError("range", "x max must be greater than x min.")

    fitted_groups: list[ParameterGroupData] = []
    for group in groups:
        x = np.asarray(group.x, dtype=float)
        if windows:
            mask = np.isfinite(x) & windows_mask(x, windows)
        else:
            mask = np.isfinite(x)
            if np.isfinite(x_min):
                mask &= x >= x_min
            if np.isfinite(x_max):
                mask &= x <= x_max
        if np.count_nonzero(mask) < 2:
            continue
        group_xe = group.xerr
        fitted_groups.append(
            ParameterGroupData(
                group_id=group.group_id,
                group_name=group.group_name,
                width_context=group.width_context.select(mask) if group.width_context else None,
                x=np.asarray(group.x, dtype=float)[mask],
                y=np.asarray(group.y, dtype=float)[mask],
                yerr=None if group.yerr is None else np.asarray(group.yerr, dtype=float)[mask],
                yerr_estimated=None if group.yerr_estimated is None else group.yerr_estimated[mask],
                group_variable_value=float(group.group_variable_value),
                xerr=(None if group_xe is None else np.asarray(group_xe, dtype=float)[mask]),
            )
        )

    if len(fitted_groups) < 2:
        raise CrossGroupFitInputError(
            "groups", "Not enough groups have at least two points in the selected fitting range."
        )

    global_params: list[str] = []
    local_params: list[str] = []
    fixed_params: dict[str, float] = {}
    initial_params: dict[str, float] = {}
    parameter_bounds: dict[str, tuple[float, float]] = {}

    model_param_names = list(model.param_names)
    model_param_name_set = set(model_param_names)
    rows = config.get("parameter_rows")
    healed_rows: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for entry in rows if isinstance(rows, list) else []:
        if not isinstance(entry, dict):
            continue
        pname = str(entry.get("name", "")).strip()
        if not pname:
            continue
        if pname not in model_param_name_set:
            continue
        if pname in seen_names:
            continue
        seen_names.add(pname)
        value = (
            float(entry.get("initial", 0.0))
            if isinstance(entry.get("initial"), (int, float))
            else 0.0
        )
        pmin = float(entry.get("min", -float("inf")))
        pmax = float(entry.get("max", float("inf")))
        initial_params[pname] = value
        parameter_bounds[pname] = (pmin, pmax)
        role = str(entry.get("type", "Global"))
        if role == "Local":
            local_params.append(pname)
        elif role == "Fixed":
            fixed_params[pname] = value
        else:
            global_params.append(pname)
        healed_rows.append(
            {"name": pname, "initial": value, "min": pmin, "max": pmax, "type": role}
        )

    for pname in model_param_names:
        if pname in seen_names:
            continue
        default_value = float(model.param_defaults.get(pname, 0.0))
        initial_params[pname] = default_value
        parameter_bounds[pname] = (-float("inf"), float("inf"))
        global_params.append(pname)
        healed_rows.append(
            {
                "name": pname,
                "initial": default_value,
                "min": -float("inf"),
                "max": float("inf"),
                "type": "Global",
            }
        )

    healed_config = deepcopy(config)
    healed_config["model"] = model.to_dict()
    healed_config["parameter_rows"] = healed_rows

    error_mode = ErrorMode(str(config.get("error_mode", ErrorMode.COLUMN.value)))
    error_value_raw = config.get("error_value")
    error_value = float(error_value_raw) if isinstance(error_value_raw, (int, float)) else None

    xerr_map: dict[str, np.ndarray] | None = None
    if bool(config.get("use_x_errors", False)):
        xerr_map = {
            group.group_id: np.asarray(group.xerr, dtype=float)
            for group in fitted_groups
            if group.xerr is not None
        }
        if not xerr_map:
            xerr_map = None

    return CrossGroupFitInputs(
        model=ParameterCompositeModel(
            component_names=list(model.component_names), operators=list(model.operators)
        ),
        groups=fitted_groups,
        global_params=global_params,
        local_params=local_params,
        fixed_params=fixed_params,
        initial_params=initial_params,
        parameter_bounds=parameter_bounds,
        error_mode=error_mode,
        error_value=error_value,
        xerr=xerr_map,
        fit_x_min=x_min if np.isfinite(x_min) else float("nan"),
        fit_x_max=x_max if np.isfinite(x_max) else float("nan"),
        config=healed_config,
    )


def run_cross_group_fit(inputs: CrossGroupFitInputs) -> CrossGroupFitRun:
    """Execute prepared inputs, retaining unsuccessful solver results as records."""
    result = global_fit_parameter_model(
        groups=inputs.groups,
        model=inputs.model,
        global_params=inputs.global_params,
        local_params=inputs.local_params,
        fixed_params=inputs.fixed_params,
        initial_params=inputs.initial_params,
        parameter_bounds=inputs.parameter_bounds,
        error_mode=inputs.error_mode,
        error_value=inputs.error_value,
        xerr=inputs.xerr,
    )
    return CrossGroupFitRun(
        model=inputs.model,
        result=result,
        fitted_groups=inputs.groups,
        fit_x_min=inputs.fit_x_min,
        fit_x_max=inputs.fit_x_max,
        config=inputs.config,
    )


def run_cross_group_fit_from_config(
    groups: list[ParameterGroupData], config: dict[str, object]
) -> CrossGroupFitRun:
    """Prepare and run a recorded configuration without a graphical front end."""
    return run_cross_group_fit(prepare_cross_group_fit(groups, config))
