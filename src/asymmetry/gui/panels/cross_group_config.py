"""Pure config→fit bridge for cross-group global parameter fits.

The cross-group fit dialog (``cross_group_fit_dialog.py``) collects a *config*
dict — the model, a shared fit range and/or windows, the per-parameter
Global/Local/Fixed roles with bounds and seeds, the error mode, and the
effective-variance x-error toggle — and turns it into the argument bundle for
:func:`asymmetry.core.fitting.parameter_models.global_fit_parameter_model`. The
*refit* path (:meth:`MainWindow._on_global_fit_refit_requested`) needs to run
that same mapping against freshly re-assembled trend groups, off the GUI thread.

Factoring the mapping here — deliberately widget-free — lets both callers share
one implementation and keeps the worker safe to call from a background thread
(it touches only plain data and core fitting code). ``config`` uses the exact
shape produced by :meth:`CrossGroupFitDialog._collect_config`:

- ``model``: a :class:`ParameterCompositeModel` ``to_dict()`` payload.
- ``fit_x_min`` / ``fit_x_max``: shared range bounds (``None`` = open side).
- ``windows``: optional ``list[[lo, hi], …]`` union (overrides the range).
- ``parameter_rows``: ``[{name, initial, min, max, type}, …]`` where ``type`` is
  ``"Global" | "Local" | "Fixed"``.
- ``error_mode`` / ``error_value``: :class:`ErrorMode` value + scalar.
- ``use_x_errors``: whether to pass per-group σ_x for effective-variance.

``run_cross_group_fit_from_config`` self-heals a config whose
``parameter_rows`` have drifted from ``model``'s current ``param_names`` (a
model edited after the config was saved, e.g. via a resaved study whose
dialog cache went stale — see ``CrossGroupFitDialog._on_model_edited``): rows
naming a parameter the model no longer has are dropped, and any model
parameter missing a row is added as Global with the model's default initial
value and open bounds. The healed config — not the one passed in — comes
back on :attr:`CrossGroupFitRun.config` so a caller that persists it (the
refit path) fixes the stored config in place instead of re-saving the same
stale rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

__all__ = [
    "CrossGroupFitRun",
    "run_cross_group_fit_from_config",
]


@dataclass
class CrossGroupFitRun:
    """Result bundle from :func:`run_cross_group_fit_from_config`."""

    model: ParameterCompositeModel
    result: CrossGroupFitResult
    #: The groups actually fitted, after windows/range slicing (a group with
    #: fewer than two surviving points is dropped).
    fitted_groups: list[ParameterGroupData]
    fit_x_min: float
    fit_x_max: float
    #: The config actually used to run the fit, healed against ``model``:
    #: ``parameter_rows`` entries naming a parameter the model no longer has
    #: are dropped, and any model parameter missing a row is appended as
    #: Global with the model's default initial value/bounds. Callers that
    #: persist the config that produced a result (e.g. the refit path) should
    #: save THIS, not the config they passed in, so a corrupted/stale config
    #: self-heals in place instead of re-failing on the next refit.
    config: dict[str, object] = field(default_factory=dict)


def run_cross_group_fit_from_config(
    groups: list[ParameterGroupData],
    config: dict,
) -> CrossGroupFitRun:
    """Run a cross-group fit described by *config* over *groups*.

    Pure and widget-free, so it is safe to invoke from a worker thread. Mirrors
    :meth:`CrossGroupFitDialog._run_fit`: slice each group to the window union
    (or ``[x_min, x_max]``), map the parameter roles to global/local/fixed +
    seeds + bounds, and call :func:`global_fit_parameter_model`.

    Raises :class:`ValueError` for a malformed config (bad windows, inverted
    range) or when fewer than two groups keep two points in the fitting range —
    the caller surfaces the message. A fit that runs but does not converge is
    *not* an error here: it comes back as an unsuccessful
    :class:`CrossGroupFitResult`, matching the dialog's "use anyway" path.

    Parameter roles are classified against ``model.param_names`` (see module
    docstring): rows for parameters the model no longer has are dropped, and
    model parameters missing a row default to Global. The resulting healed
    config is returned on :attr:`CrossGroupFitRun.config`.
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
        validate_fit_windows(windows)
    elif np.isfinite(x_min) and np.isfinite(x_max) and x_max <= x_min:
        raise ValueError("x max must be greater than x min.")

    # Slice each group to the window union (or the [x_min, x_max] range) so the
    # masking lives in one place, mirroring the dialog's Run path.
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
        group_xe = getattr(group, "xerr", None)
        fitted_groups.append(
            ParameterGroupData(
                group_id=group.group_id,
                group_name=group.group_name,
                x=np.asarray(group.x, dtype=float)[mask],
                y=np.asarray(group.y, dtype=float)[mask],
                yerr=np.asarray(group.yerr, dtype=float)[mask],
                group_variable_value=float(group.group_variable_value),
                xerr=(None if group_xe is None else np.asarray(group_xe, dtype=float)[mask]),
            )
        )

    if len(fitted_groups) < 2:
        raise ValueError(
            "Not enough groups have at least two points in the selected fitting range."
        )

    global_params: list[str] = []
    local_params: list[str] = []
    fixed_params: dict[str, float] = {}
    initial_params: dict[str, float] = {}
    parameter_bounds: dict[str, tuple[float, float]] = {}

    # Self-heal: classify roles from the CURRENT model's param_names, not
    # blindly from the stored rows. A model edit (component add/remove) after
    # the config was saved leaves ``parameter_rows`` out of sync with
    # ``model.param_names`` — e.g. a removed component's rows linger, or a
    # newly added component's params are missing entirely. Feeding stale rows
    # straight to global_fit_parameter_model trips its "Unknown parameter
    # classification" guard (parameter_models.py) instead of fitting.
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
            # Dropped: this parameter no longer exists on the model (e.g. its
            # component was removed).
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

    # Any model parameter missing a row (e.g. a newly added component, or one
    # that arrived after this config was saved) defaults to Global with the
    # model's own default initial value and open bounds.
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

    healed_config = dict(config)
    healed_config["model"] = model.to_dict()
    healed_config["parameter_rows"] = healed_rows

    error_mode = ErrorMode(str(config.get("error_mode", ErrorMode.COLUMN.value)))
    error_value_raw = config.get("error_value")
    error_value = float(error_value_raw) if isinstance(error_value_raw, (int, float)) else None

    # Effective-variance σ_x only when the user opted in and every group carries
    # it; the core ignores it under NONE/SCATTER regardless.
    xerr_map: dict[str, np.ndarray] | None = None
    if bool(config.get("use_x_errors", False)):
        xerr_map = {
            group.group_id: np.asarray(group.xerr, dtype=float)
            for group in fitted_groups
            if getattr(group, "xerr", None) is not None
        }
        if not xerr_map:
            xerr_map = None

    model_snapshot = ParameterCompositeModel(
        component_names=list(model.component_names),
        operators=list(model.operators),
    )
    result = global_fit_parameter_model(
        groups=fitted_groups,
        model=model_snapshot,
        global_params=global_params,
        local_params=local_params,
        fixed_params=fixed_params,
        initial_params=initial_params,
        parameter_bounds=parameter_bounds,
        error_mode=error_mode,
        error_value=error_value,
        xerr=xerr_map,
    )

    return CrossGroupFitRun(
        model=model_snapshot,
        result=result,
        fitted_groups=fitted_groups,
        fit_x_min=(x_min if np.isfinite(x_min) else float("nan")),
        fit_x_max=(x_max if np.isfinite(x_max) else float("nan")),
        config=healed_config,
    )
