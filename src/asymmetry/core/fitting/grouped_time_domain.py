"""Grouped time-domain fitting helpers.

This module adapts WiMDA-style multi-group count fitting onto Asymmetry's
existing simultaneous-fit engine. The first slice intentionally keeps the
engine unchanged and expresses each included group as one temporary fitting
domain.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
)
from asymmetry.core.fitting.parameters import ParameterSet, split_parameter_name
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.grouping import (
    apply_grouping_aligned,
    common_t0_for_groups,
    effective_group_indices,
)
from asymmetry.core.transform.rebin import rebin_counts
from asymmetry.core.utils.constants import MUON_LIFETIME_US

GROUP_NUISANCE_PARAMS: tuple[str, ...] = (
    "N0",
    "background",
    "amplitude",
    "relative_phase",
)


def validate_grouped_model_contract(
    model_param_names: list[str] | tuple[str, ...],
    *,
    model_values: dict[str, float],
    fixed_params: set[str] | list[str] | tuple[str, ...],
) -> None:
    """Reject grouped-model parameter ownership that conflicts with group nuisances.

    Grouped time-domain fits reserve the overall scale and constant background for
    the per-group nuisance block. The shared model is therefore interpreted as a
    normalized polarization-like function: model amplitude parameters must be
    fixed at ``1`` and model background parameters must be fixed at ``0``.
    Fraction parameters remain free and are not treated as amplitude conflicts.
    """
    fixed = {str(name) for name in fixed_params}

    background_conflicts: list[str] = []
    amplitude_conflicts: list[str] = []
    for name in model_param_names:
        if is_background_parameter(name):
            value = float(model_values.get(name, 0.0))
            if name not in fixed or not np.isclose(value, 0.0):
                background_conflicts.append(str(name))
            continue
        if is_amplitude_parameter(name):
            value = float(model_values.get(name, 1.0))
            if name not in fixed or not np.isclose(value, 1.0):
                amplitude_conflicts.append(str(name))

    messages: list[str] = []
    if background_conflicts:
        joined = ", ".join(background_conflicts)
        messages.append(
            "Grouped time-domain fit-function backgrounds must stay in the per-group "
            f"parameter block; set these fit-function parameters to Fixed = 0: {joined}."
        )
    if amplitude_conflicts:
        joined = ", ".join(amplitude_conflicts)
        messages.append(
            "Grouped time-domain fit-function amplitudes must be normalized so the "
            f"per-group amplitude owns the overall scale; set these fit-function parameters to Fixed = 1: {joined}."
        )
    if messages:
        raise ValueError(" ".join(messages))


def normalize_to_grouped_contract(
    model_param_names: list[str] | tuple[str, ...],
    base_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Force a model's parameters to the normalised-polarisation contract.

    The grouped time-domain fit (and the multi-group *simulation* that mirrors
    it) reserve the overall scale for the per-group amplitude and the constant
    background for the per-group ``N0``, so the shared model must be a
    unit-amplitude, zero-baseline polarisation: amplitude parameters fixed at
    ``1`` and background parameters fixed at ``0``. Returns ``base_values`` (a
    copy, defaulting to ``{}``) with those parameters overridden — the single
    definition of the contract, used by both the fit seed cache and the
    multi-group simulate dialog.
    """
    values = dict(base_values or {})
    for name in model_param_names:
        if is_amplitude_parameter(name):
            values[name] = 1.0
        elif is_background_parameter(name):
            values[name] = 0.0
    return values


@dataclass
class GroupedTimeDomainGroup:
    """One lifetime-corrected grouped-count domain for simultaneous fitting."""

    group_id: Hashable
    group_name: str
    time: NDArray[np.float64]
    counts: NDArray[np.float64]
    error: NDArray[np.float64]
    source_run_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupedTimeDomainFitResult:
    """Result bundle for grouped time-domain fitting."""

    success: bool
    group_results: dict[Hashable, FitResult]
    shared_parameters: ParameterSet
    message: str = ""


def build_grouped_time_domain_datasets(
    dataset: MuonDataset,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    lifetime_corrected: bool = True,
) -> list[MuonDataset]:
    """Return grouped-count datasets for one grouped run.

    With ``lifetime_corrected=True`` (the default) the counts are scaled by
    ``exp(t / tau_mu)``; with ``False`` the raw detector counts are returned, on
    which Poisson statistics are exact.
    """
    groups = build_grouped_time_domain_groups(
        dataset, t_min=t_min, t_max=t_max, lifetime_corrected=lifetime_corrected
    )
    y_label = "Lifetime-corrected counts" if lifetime_corrected else "Counts"
    grouped_datasets: list[MuonDataset] = []
    for index, group in enumerate(groups, start=1):
        synthetic_run_number = _group_dataset_run_number(group.source_run_number, index)
        metadata = {
            **dict(group.metadata),
            "run_number": synthetic_run_number,
            "run_label": str(group.group_name),
            "group_id": group.group_id,
            "group_name": group.group_name,
            "source_run_number": group.source_run_number,
            "grouped_time_domain": True,
            "x_label": "Time (μs)",
            "y_label": y_label,
        }
        grouped_datasets.append(
            MuonDataset(
                time=np.asarray(group.time, dtype=float).copy(),
                asymmetry=np.asarray(group.counts, dtype=float).copy(),
                error=np.asarray(group.error, dtype=float).copy(),
                metadata=metadata,
                run=None,
            )
        )
    return grouped_datasets


def grouped_time_domain_available(dataset: MuonDataset | None) -> bool:
    """Cheap availability probe for the grouped time-domain build.

    Answers "would :func:`build_grouped_time_domain_groups` find enough
    groups?" without any array work, by running the same setup validation the
    build itself runs (source run with histograms, grouping with at least two
    included non-empty detector groups) — the GUI gates views and toolbar
    buttons on this for every selection change, where building the full
    per-group traces just to test truthiness is far too expensive.

    A pathological run that passes this probe can still fail the full build
    (e.g. every bin masked out of the fit window); callers already treat a
    failed build as no-data at render time.
    """
    if dataset is None:
        return False
    try:
        _run, grouping, groups_raw, included_groups = _count_group_setup(dataset)
    except ValueError:
        return False
    return len(_resolve_group_indices(grouping, groups_raw, included_groups)) >= 2


@dataclass
class _CountGroupContext:
    """Shared per-run setup for building one or more count-domain group traces.

    Computed once over a chosen set of detector groups so that single-group,
    forward/backward, and full multi-group builders all share the same t0
    alignment, good-bin window, deadtime preparation, and bunching.
    """

    run: Any
    prepared_histograms: list[Any]
    common_t0: int
    first_good: int
    last_good: int
    bunch_factor: int
    bin_width: float
    axis_start: float
    group_names: dict[Any, Any]


def _count_group_setup(dataset: MuonDataset) -> tuple[Any, dict, dict, dict]:
    """Validate a dataset and return ``(run, grouping, groups_raw, included_groups)``."""
    run = dataset.run
    if run is None:
        raise ValueError("Grouped time-domain fitting requires a dataset with a source run")
    if not run.histograms:
        raise ValueError("Grouped time-domain fitting requires detector histograms")

    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups_raw = grouping.get("groups")
    if not isinstance(groups_raw, dict) or not groups_raw:
        raise ValueError("Grouped time-domain fitting requires grouping definitions")
    included_raw = grouping.get("included_groups")
    included_groups = included_raw if isinstance(included_raw, dict) else {}
    return run, grouping, groups_raw, included_groups


def _resolve_group_indices(
    grouping: dict, groups_raw: dict, included_groups: dict
) -> list[tuple[int, list[int]]]:
    """Return ``(group_id, detector_indices)`` for every included, non-empty group."""
    groups: list[tuple[int, list[int]]] = []
    for raw_group_id in sorted(groups_raw, key=str):
        try:
            group_id = int(raw_group_id)
        except (TypeError, ValueError):
            continue
        if not bool(included_groups.get(group_id, True)):
            continue
        indices = effective_group_indices(grouping, group_id)
        if indices:
            groups.append((group_id, indices))
    return groups


def _count_group_context(
    dataset: MuonDataset,
    grouping: dict,
    run: Any,
    group_specs: list[tuple[int, list[int]]],
) -> _CountGroupContext:
    """Prepare the shared count-build context over the given detector groups."""
    apply_deadtime = bool(grouping.get("deadtime_correction", False))
    prepared_histograms, _ = prepare_histograms_with_deadtime(
        list(run.histograms),
        grouping,
        apply_deadtime,
    )
    common_t0 = common_t0_for_groups(prepared_histograms, *(indices for _, indices in group_specs))

    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    max_last_good = max(0, min(len(hist.counts) for hist in prepared_histograms) - 1)
    try:
        last_good = int(grouping.get("last_good_bin", max_last_good))
    except (TypeError, ValueError):
        last_good = max_last_good
    last_good = max(first_good, min(max_last_good, last_good))
    try:
        bunch_factor = max(1, int(grouping.get("bunching_factor", 1)))
    except (TypeError, ValueError):
        bunch_factor = 1

    bin_width = float(prepared_histograms[0].bin_width)
    axis_start = first_good - common_t0
    group_names = (
        grouping.get("group_names") if isinstance(grouping.get("group_names"), dict) else {}
    )
    return _CountGroupContext(
        run=run,
        prepared_histograms=prepared_histograms,
        common_t0=common_t0,
        first_good=first_good,
        last_good=last_good,
        bunch_factor=bunch_factor,
        bin_width=bin_width,
        axis_start=axis_start,
        group_names=group_names,
    )


def _build_one_count_group(
    ctx: _CountGroupContext,
    group_id: int,
    indices: list[int],
    *,
    t_min: float | None,
    t_max: float | None,
    lifetime_corrected: bool,
    exclude: tuple[float, float] | None = None,
) -> GroupedTimeDomainGroup | None:
    """Build one count-domain group trace from a prepared context.

    ``exclude`` drops an interior ``[t0, t1]`` window of bins from the trace
    (WiMDA's second time range), endpoints inclusive — for rejecting a laser/RF
    artefact or a spike without splitting the fit. Returns ``None`` when the
    group yields no usable bins inside the fit window.
    """
    run = ctx.run
    counts = apply_grouping_aligned(
        ctx.prepared_histograms,
        indices,
        common_t0_bin=ctx.common_t0,
    )
    if counts.size == 0:
        return None
    trimmed_counts = np.asarray(counts[ctx.first_good : ctx.last_good + 1], dtype=np.float64)
    time = (np.arange(trimmed_counts.size, dtype=float) + float(ctx.axis_start)) * ctx.bin_width
    if ctx.bunch_factor > 1:
        time, trimmed_counts = rebin_counts(time, trimmed_counts, ctx.bunch_factor)

    if lifetime_corrected:
        scale = np.exp(time / float(MUON_LIFETIME_US))
        out_counts = trimmed_counts * scale
        errors = np.sqrt(np.clip(trimmed_counts, 1.0, None)) * scale
    else:
        out_counts = trimmed_counts
        errors = np.sqrt(np.clip(trimmed_counts, 1.0, None))

    mask = np.ones_like(time, dtype=bool)
    if t_min is not None:
        mask &= time >= float(t_min)
    if t_max is not None:
        mask &= time <= float(t_max)
    if exclude is not None:
        ex0, ex1 = float(exclude[0]), float(exclude[1])
        if ex1 > ex0:
            mask &= ~((time >= ex0) & (time <= ex1))
    if not np.any(mask):
        return None

    group_name = ctx.group_names.get(group_id, f"Group {group_id}")
    metadata = dict(run.metadata)
    metadata.update(
        {
            "group_id": int(group_id),
            "group_name": str(group_name),
            "grouped_time_domain": True,
            "grouped_time_domain_bunch_factor": ctx.bunch_factor,
            "grouped_time_domain_lifetime_corrected": bool(lifetime_corrected),
        }
    )
    return GroupedTimeDomainGroup(
        group_id=int(group_id),
        group_name=str(group_name),
        time=np.asarray(time[mask], dtype=float).copy(),
        counts=np.asarray(out_counts[mask], dtype=float).copy(),
        error=np.asarray(errors[mask], dtype=float).copy(),
        source_run_number=int(run.run_number),
        metadata=metadata,
    )


def build_grouped_time_domain_groups(
    dataset: MuonDataset,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    lifetime_corrected: bool = True,
    exclude: tuple[float, float] | None = None,
) -> list[GroupedTimeDomainGroup]:
    """Build grouped count domains from one dataset (all included groups)."""
    run, grouping, groups_raw, included_groups = _count_group_setup(dataset)
    specs = _resolve_group_indices(grouping, groups_raw, included_groups)
    if len(specs) < 2:
        raise ValueError(
            "Grouped time-domain fitting requires at least two included detector groups"
        )
    ctx = _count_group_context(dataset, grouping, run, specs)

    built_groups: list[GroupedTimeDomainGroup] = []
    for group_id, indices in specs:
        group = _build_one_count_group(
            ctx,
            group_id,
            indices,
            t_min=t_min,
            t_max=t_max,
            lifetime_corrected=lifetime_corrected,
            exclude=exclude,
        )
        if group is not None:
            built_groups.append(group)

    if len(built_groups) < 2:
        raise ValueError("Grouped time-domain fitting produced fewer than two usable groups")
    return built_groups


def build_count_groups(
    dataset: MuonDataset,
    group_ids: list[int],
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    lifetime_corrected: bool = True,
    exclude: tuple[float, float] | None = None,
) -> list[GroupedTimeDomainGroup]:
    """Build several named detector groups sharing ONE t0 alignment.

    All requested groups go through a single context, so the common t0 (and
    good-bin window, bunching, deadtime preparation) is computed over the union
    of their detectors and every trace lands on one time axis. The
    forward/backward count fit needs this: building the two banks separately
    would give them independent ``common_t0`` values and misalign the shared
    physics whenever the F and B detectors carry different ``t0_bin``.

    Unlike :func:`build_grouped_time_domain_groups` this does not require the
    groups to be in the ``included_groups`` map. Returns one group per id, in the
    requested order.
    """
    run, grouping, _groups_raw, _included = _count_group_setup(dataset)
    specs: list[tuple[int, list[int]]] = []
    for raw_gid in group_ids:
        gid = int(raw_gid)
        indices = effective_group_indices(grouping, gid)
        if not indices:
            raise ValueError(f"Count-domain fitting found no detectors for group {raw_gid!r}")
        specs.append((gid, indices))
    ctx = _count_group_context(dataset, grouping, run, specs)

    built: list[GroupedTimeDomainGroup] = []
    for gid, indices in specs:
        group = _build_one_count_group(
            ctx,
            gid,
            indices,
            t_min=t_min,
            t_max=t_max,
            lifetime_corrected=lifetime_corrected,
            exclude=exclude,
        )
        if group is None:
            raise ValueError(f"Count-domain group {gid!r} produced no usable bins")
        built.append(group)
    return built


def build_count_group(
    dataset: MuonDataset,
    group_id: int,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    lifetime_corrected: bool = True,
    exclude: tuple[float, float] | None = None,
) -> GroupedTimeDomainGroup:
    """Build one named detector group's count-domain trace.

    Unlike :func:`build_grouped_time_domain_groups` this does not require two
    included groups: it serves single-histogram count fits, which select a
    specific detector group regardless of the ``included_groups`` map.
    ``exclude`` drops an interior time window of bins.
    """
    return build_count_groups(
        dataset,
        [group_id],
        t_min=t_min,
        t_max=t_max,
        lifetime_corrected=lifetime_corrected,
        exclude=exclude,
    )[0]


def fit_grouped_time_domain(
    groups: list[GroupedTimeDomainGroup],
    polarization_model_fn,
    global_params: list[str],
    local_params: list[str],
    initial_params: dict[Hashable, ParameterSet],
    *,
    fit_engine: FitEngine | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "migrad",
    max_calls: int = 10000,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> GroupedTimeDomainFitResult:
    """Fit one shared polarization model across several grouped count traces.

    The first-slice contract is intentionally strict:

    - each group is a separate fitting domain
    - model-function parameters may be global or fixed only
    - local parameters may only come from the group nuisance block
    - the observed signal is assumed to be lifetime-corrected grouped counts

    Parameters
    ----------
    groups
        Included groups from one active dataset.
    polarization_model_fn
        Callable returning a normalized polarization-like signal.
    global_params
        Shared free parameters across all groups. These may include both
        physical model parameters and nuisance parameters selected as global.
    local_params
        Per-group free parameters. These are restricted to
        ``GROUP_NUISANCE_PARAMS``.
    initial_params
        Parameter sets keyed by ``group_id``. Every set must include the group
        nuisance parameters and any model parameters referenced by
        ``global_params``.
    """
    if len(groups) < 2:
        raise ValueError("Need at least two groups for grouped time-domain fitting")

    group_ids = [group.group_id for group in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("Grouped time-domain fitting requires unique group ids")

    local_only = set(local_params) - set(GROUP_NUISANCE_PARAMS)
    if local_only:
        raise ValueError(
            "Local grouped time-domain parameters must come from the group block: "
            f"{sorted(local_only)}"
        )

    overlapping = set(global_params) & set(local_params)
    if overlapping:
        raise ValueError(f"Global and local grouped parameters overlap: {sorted(overlapping)}")

    missing_sets = [group_id for group_id in group_ids if group_id not in initial_params]
    if missing_sets:
        raise ValueError(f"Missing grouped time-domain initial parameters: {missing_sets}")

    required_names = set(GROUP_NUISANCE_PARAMS) | set(global_params) | set(local_params)
    for group_id in group_ids:
        missing_names = sorted(required_names - set(initial_params[group_id].names))
        if missing_names:
            raise ValueError(
                f"Grouped time-domain parameters for {group_id!r} are missing: {missing_names}"
            )

    model_fn = build_grouped_count_model(polarization_model_fn)

    engine = fit_engine or FitEngine()
    temporary_datasets: list[MuonDataset] = []
    temporary_initial_params: dict[int, ParameterSet] = {}
    internal_to_group: dict[int, Hashable] = {}

    for idx, group in enumerate(groups, start=1):
        internal_id = -idx
        internal_to_group[internal_id] = group.group_id
        time = np.asarray(group.time, dtype=float)
        counts = np.asarray(group.counts, dtype=float)
        error = np.asarray(group.error, dtype=float)
        if time.shape != counts.shape or time.shape != error.shape:
            raise ValueError(
                f"Grouped time-domain arrays for {group.group_id!r} must share one shape"
            )
        metadata = dict(group.metadata)
        metadata.update(
            {
                "run_number": internal_id,
                "group_id": group.group_id,
                "group_name": group.group_name,
                "source_run_number": group.source_run_number,
            }
        )
        temporary_datasets.append(
            MuonDataset(
                time=time.copy(),
                asymmetry=counts.copy(),
                error=error.copy(),
                metadata=metadata,
                run=None,
            )
        )
        temporary_initial_params[internal_id] = initial_params[group.group_id]

    internal_results, shared_parameters = engine.global_fit(
        temporary_datasets,
        model_fn,
        global_params=global_params,
        local_params=local_params,
        initial_params=temporary_initial_params,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=cancel_callback,
    )

    group_results = {
        internal_to_group[internal_id]: result
        for internal_id, result in internal_results.items()
        if internal_id in internal_to_group
    }
    success = bool(group_results) and all(result.success for result in group_results.values())
    if success:
        message = "Grouped time-domain fit successful"
    elif group_results:
        failed = [str(group_id) for group_id, result in group_results.items() if not result.success]
        message = f"Grouped time-domain fit failed for: {', '.join(failed)}"
    else:
        message = "Grouped time-domain fit produced no results"
    return GroupedTimeDomainFitResult(
        success=success,
        group_results=group_results,
        shared_parameters=shared_parameters,
        message=message,
    )


def _supports_phase_parameter(model_fn) -> bool:
    signature = inspect.signature(model_fn)
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return any(split_parameter_name(str(name))[0] == "phase" for name in signature.parameters)


def _group_dataset_run_number(source_run_number: int | None, group_index: int) -> int:
    source = abs(int(source_run_number or 0))
    return -((source * 1000) + max(1, int(group_index)))


def build_grouped_count_model(polarization_model_fn):
    phase_supported = _supports_phase_parameter(polarization_model_fn)

    def grouped_count_model(t, **kwargs):
        time = np.asarray(t, dtype=float)
        n0 = float(kwargs.pop("N0"))
        background = float(kwargs.pop("background"))
        amplitude = float(kwargs.pop("amplitude"))
        relative_phase = float(kwargs.pop("relative_phase"))

        model_kwargs = dict(kwargs)
        if phase_supported:
            phase_keys = [
                key for key in list(model_kwargs) if split_parameter_name(str(key))[0] == "phase"
            ]
            if "phase" in model_kwargs and "phase" not in phase_keys:
                phase_keys.append("phase")
            if phase_keys:
                for phase_key in phase_keys:
                    model_kwargs[phase_key] = (
                        float(model_kwargs.get(phase_key, 0.0)) + relative_phase
                    )
            else:
                model_kwargs["phase"] = relative_phase
        elif not np.isclose(relative_phase, 0.0):
            raise ValueError(
                "Grouped time-domain fitting requires a phase-capable model when "
                "relative_phase is non-zero"
            )

        polarization = np.asarray(polarization_model_fn(time, **model_kwargs), dtype=float)
        return n0 * (1.0 + amplitude * polarization) + background * np.exp(
            time / float(MUON_LIFETIME_US)
        )

    return grouped_count_model


def build_fb_count_model(polarization_model_fn):
    """Forward/backward count model with the detector balance ``alpha`` free.

    Mirrors WiMDA ``fgFB``: one shared ``N0`` is split as ``N0·√alpha`` for the
    forward histogram and ``N0/√alpha`` for the backward one, the shared physics
    polarization enters with a forward (``+``) / backward (``-``) sign, and each
    side carries its own background. ``sign`` is supplied per dataset as a fixed
    parameter (``+1`` forward, ``-1`` backward); ``alpha`` and ``N0`` are shared
    (global) and the physics-model parameters are global too, so ``alpha`` and
    its correlation with the amplitude fall straight out of the joint fit.

    The returned signal is on the **lifetime-corrected** count scale (the
    background carries ``exp(t / tau_mu)``); raw-count callers multiply the whole
    model by ``exp(-t / tau_mu)``.
    """

    def fb_count_model(t, **kwargs):
        time = np.asarray(t, dtype=float)
        alpha = float(kwargs.pop("alpha"))
        n0 = float(kwargs.pop("N0"))
        background = float(kwargs.pop("background"))
        sign = float(kwargs.pop("sign"))

        ralp = np.sqrt(abs(alpha))
        if sign >= 0.0:
            scale = ralp
        else:
            scale = 1.0 / ralp if ralp > 0.0 else 0.0

        polarization = np.asarray(polarization_model_fn(time, **kwargs), dtype=float)
        return n0 * scale * (1.0 + sign * polarization) + background * np.exp(
            time / float(MUON_LIFETIME_US)
        )

    return fb_count_model


#: The three member relationships a grouped-series fit can use.
GROUPED_SERIES_RELATIONSHIPS: tuple[str, ...] = ("individual", "batch", "global")


@dataclass
class GroupedSeriesFitResult:
    """Result bundle for a multi-member grouped time-domain *series* fit.

    ``member_results`` is keyed by the synthetic group-member key
    (:func:`_group_dataset_run_number`) so it maps directly onto a group
    :class:`~asymmetry.core.representation.series.FitSeries`'s
    ``results_by_run``.  ``member_source_run`` / ``member_group_id`` resolve each
    key back to its physical run and detector group.  ``shared_parameters`` holds
    the cross-run global parameters for a ``"global"`` relationship and is empty
    for ``"individual"`` / ``"batch"``.
    """

    success: bool
    relationship: str
    member_results: dict[int, FitResult]
    member_source_run: dict[int, int]
    member_group_id: dict[int, Hashable]
    shared_parameters: ParameterSet
    message: str = ""


def fit_grouped_series(
    relationship: str,
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    global_params: list[str],
    local_params: list[str],
    initial_params: dict[int, dict[Hashable, ParameterSet]],
    *,
    fit_engine: FitEngine | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "migrad",
    max_calls: int = 10000,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> GroupedSeriesFitResult:
    """Fit a series of grouped runs with one of three member relationships.

    The *physics* (fit-function) parameters in ``global_params`` are always
    shared across the detector groups within a run.  The ``relationship``
    governs how they relate *across* runs (members):

    * ``"individual"`` / ``"batch"`` – each run is fit independently (its physics
      values are not shared with other runs).  Both run the same engine path;
      they differ only in how the caller records the resulting series.
    * ``"global"`` – the physics parameters are shared across *all* runs and
      groups via a single simultaneous fit.

    The per-group nuisance block (``local_params`` ⊆ :data:`GROUP_NUISANCE_PARAMS`)
    is always estimated separately for each ``(run, group)``.

    Parameters
    ----------
    members
        Mapping of run number → that run's included
        :class:`GroupedTimeDomainGroup` list (group order is the trend order).
    initial_params
        Nested mapping ``run -> {group_id -> ParameterSet}``.  Every set must
        include the nuisance block plus the model parameters referenced by
        ``global_params`` / ``local_params``.

    Note
    ----
    A *mixed* global fit (some physics shared across runs, others joined only
    within a run) is not yet supported: the simultaneous engine shares a global
    parameter across every dataset in the call, so per-run scoping would need an
    engine-level change.  Use ``"global"`` to share all free physics across runs,
    or ``"batch"`` to keep them per-run.
    """
    if relationship not in GROUPED_SERIES_RELATIONSHIPS:
        raise ValueError(
            f"Unknown grouped-series relationship {relationship!r}; "
            f"expected one of {GROUPED_SERIES_RELATIONSHIPS}"
        )
    if not members:
        raise ValueError("Grouped-series fitting requires at least one member run")

    local_only = set(local_params) - set(GROUP_NUISANCE_PARAMS)
    if local_only:
        raise ValueError(
            "Local grouped time-domain parameters must come from the group block: "
            f"{sorted(local_only)}"
        )
    overlapping = set(global_params) & set(local_params)
    if overlapping:
        raise ValueError(f"Global and local grouped parameters overlap: {sorted(overlapping)}")

    engine = fit_engine or FitEngine()
    if relationship in ("individual", "batch"):
        return _fit_grouped_series_independent(
            relationship,
            members,
            polarization_model_fn,
            global_params,
            local_params,
            initial_params,
            engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            minos=minos,
            cancel_callback=cancel_callback,
        )
    return _fit_grouped_series_global(
        members,
        polarization_model_fn,
        global_params,
        local_params,
        initial_params,
        engine=engine,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=cancel_callback,
    )


def _fit_grouped_series_independent(
    relationship: str,
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    global_params: list[str],
    local_params: list[str],
    initial_params: dict[int, dict[Hashable, ParameterSet]],
    *,
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> GroupedSeriesFitResult:
    """Run one independent grouped joint fit per member run (no cross-run sharing)."""
    member_results: dict[int, FitResult] = {}
    member_source_run: dict[int, int] = {}
    member_group_id: dict[int, Hashable] = {}
    messages: list[str] = []
    for raw_run, groups in members.items():
        # Cooperative cancel between member fits (the minimum abort granularity): a
        # cancelled series records nothing and the loop stops cleanly here.
        if cancel_callback is not None and bool(cancel_callback()):
            raise FitCancelledError("Fit cancelled.")
        run = int(raw_run)
        run_initial = initial_params.get(run, {})
        result = fit_grouped_time_domain(
            groups,
            polarization_model_fn,
            global_params=global_params,
            local_params=local_params,
            initial_params=run_initial,
            fit_engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            minos=minos,
            cancel_callback=cancel_callback,
        )
        messages.append(f"run {run}: {result.message}")
        for index, group in enumerate(groups, start=1):
            key = _group_dataset_run_number(run, index)
            group_result = result.group_results.get(group.group_id)
            if group_result is None:
                continue
            member_results[key] = group_result
            member_source_run[key] = run
            member_group_id[key] = group.group_id
    success = bool(member_results) and all(r.success for r in member_results.values())
    return GroupedSeriesFitResult(
        success=success,
        relationship=relationship,
        member_results=member_results,
        member_source_run=member_source_run,
        member_group_id=member_group_id,
        shared_parameters=ParameterSet(),
        message="; ".join(messages),
    )


def _fit_grouped_series_global(
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    global_params: list[str],
    local_params: list[str],
    initial_params: dict[int, dict[Hashable, ParameterSet]],
    *,
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> GroupedSeriesFitResult:
    """Fit every ``(run, group)`` simultaneously, sharing physics across all runs."""
    temporary_datasets: list[MuonDataset] = []
    temporary_initial: dict[int, ParameterSet] = {}
    member_source_run: dict[int, int] = {}
    member_group_id: dict[int, Hashable] = {}
    required_names = set(GROUP_NUISANCE_PARAMS) | set(global_params) | set(local_params)

    for raw_run, groups in members.items():
        run = int(raw_run)
        run_initial = initial_params.get(run, {})
        for index, group in enumerate(groups, start=1):
            key = _group_dataset_run_number(run, index)
            if group.group_id not in run_initial:
                raise ValueError(
                    f"Missing grouped-series initial parameters for run {run} "
                    f"group {group.group_id!r}"
                )
            param_set = run_initial[group.group_id]
            missing_names = sorted(required_names - set(param_set.names))
            if missing_names:
                raise ValueError(
                    f"Grouped-series parameters for run {run} group {group.group_id!r} "
                    f"are missing: {missing_names}"
                )
            time = np.asarray(group.time, dtype=float)
            counts = np.asarray(group.counts, dtype=float)
            error = np.asarray(group.error, dtype=float)
            if time.shape != counts.shape or time.shape != error.shape:
                raise ValueError(
                    f"Grouped-series arrays for run {run} group {group.group_id!r} "
                    "must share one shape"
                )
            metadata = dict(group.metadata)
            metadata.update(
                {
                    "run_number": key,
                    "group_id": group.group_id,
                    "group_name": group.group_name,
                    "source_run_number": run,
                }
            )
            temporary_datasets.append(
                MuonDataset(
                    time=time.copy(),
                    asymmetry=counts.copy(),
                    error=error.copy(),
                    metadata=metadata,
                    run=None,
                )
            )
            temporary_initial[key] = param_set
            member_source_run[key] = run
            member_group_id[key] = group.group_id

    if len(temporary_datasets) < 2:
        raise ValueError("Global grouped-series fitting requires at least two (run, group) members")

    model_fn = build_grouped_count_model(polarization_model_fn)
    internal_results, shared_parameters = engine.global_fit(
        temporary_datasets,
        model_fn,
        global_params=global_params,
        local_params=local_params,
        initial_params=temporary_initial,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=cancel_callback,
    )

    member_results = {
        key: internal_results[key] for key in member_source_run if key in internal_results
    }
    success = bool(member_results) and all(r.success for r in member_results.values())
    if success:
        message = "Grouped-series global fit successful"
    elif member_results:
        failed = [str(member_group_id[key]) for key, r in member_results.items() if not r.success]
        message = f"Grouped-series global fit failed for groups: {', '.join(failed)}"
    else:
        message = "Grouped-series global fit produced no results"
    return GroupedSeriesFitResult(
        success=success,
        relationship="global",
        member_results=member_results,
        member_source_run=member_source_run,
        member_group_id=member_group_id,
        shared_parameters=shared_parameters,
        message=message,
    )
