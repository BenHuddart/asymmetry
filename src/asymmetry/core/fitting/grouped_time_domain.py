"""Grouped time-domain fitting helpers.

This module adapts WiMDA-style multi-group count fitting onto Asymmetry's
existing simultaneous-fit engine. The first slice intentionally keeps the
engine unchanged and expresses each included group as one temporary fitting
domain.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import (
    COST_FACTORIES,
    POISSON_COST,
    CostFactory,
    FitCancelledError,
    FitEngine,
    FitResult,
    _reject_affine_ties,
)
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, split_parameter_name
from asymmetry.core.fitting.process_pool import open_spawn_pool
from asymmetry.core.fitting.series_seeding import recommend_series_seeding
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


def _raw_count_model(model_fn):
    """Wrap a lifetime-corrected count model to predict raw counts.

    ``build_grouped_count_model`` returns the lifetime-corrected expectation
    (background carries ``e^(t/τ_μ)``); multiplying the whole thing by
    ``e^(−t/τ_μ)`` gives the raw-count expectation the Cash statistic compares
    against. Same construction the count-domain driver uses.
    """

    def raw(t, **kwargs):
        time = np.asarray(t, dtype=float)
        corrected = np.asarray(model_fn(time, **kwargs), dtype=float)
        return corrected * np.exp(-time / float(MUON_LIFETIME_US))

    return raw


def _raw_group_counts(
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    metadata: dict,
) -> NDArray[np.float64]:
    """Invert the lifetime correction on a trace to recover raw Poisson counts.

    ``_build_one_count_group`` records whether it applied the ``e^(t/τ_μ)``
    correction; when it did, ``raw = corrected · e^(−t/τ_μ)`` restores the
    Poisson-distributed (rebinned) counts. An already-raw trace passes through.
    """
    if not bool(metadata.get("grouped_time_domain_lifetime_corrected", True)):
        return np.asarray(counts, dtype=float)
    return np.asarray(counts, dtype=float) * np.exp(
        -np.asarray(time, dtype=float) / float(MUON_LIFETIME_US)
    )


def _resolve_grouped_cost(cost: str):
    """Validate a grouped-fit cost name and return ``(use_poisson, cost_factory)``.

    Shared by the single-run and global-series grouped fitters so the cost
    contract (valid names, the Poisson→raw-count routing) lives in one place.
    """
    if cost not in COST_FACTORIES:
        raise ValueError(
            f"Unknown grouped fit cost {cost!r}; expected one of {sorted(COST_FACTORIES)}"
        )
    use_poisson = cost == "poisson"
    return use_poisson, (POISSON_COST if use_poisson else None)


def _raw_count_dataset_fields(time, counts, metadata):
    """Return (raw_counts, poisson_error, metadata) for a Poisson grouped fit.

    Inverts the lifetime correction to recover the raw Poisson counts and floors
    a Poisson √N error (unused by Cash, but ``global_fit`` rejects zero errors);
    the returned metadata records the trace is no longer lifetime-corrected so it
    cannot be double-corrected downstream.
    """
    raw = _raw_group_counts(time, counts, metadata)
    error = np.sqrt(np.clip(raw, 1.0, None))
    fixed_metadata = dict(metadata)
    fixed_metadata["grouped_time_domain_lifetime_corrected"] = False
    return raw, error, fixed_metadata


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
    cost: str = "poisson",
) -> GroupedTimeDomainFitResult:
    """Fit one shared polarization model across several grouped count traces.

    The first-slice contract is intentionally strict:

    - each group is a separate fitting domain
    - model-function parameters may be global or fixed only
    - local parameters may only come from the group nuisance block
    - the observed signal is assumed to be lifetime-corrected grouped counts

    ``cost`` selects the fit objective, matching the count-domain modes'
    convention:

    - ``"poisson"`` (default) — the Cash statistic on the **raw** Poisson
      counts (the lifetime correction is inverted and the model is multiplied
      by ``e^(−t/τ_μ)`` to predict raw counts). This is the statistically
      faithful objective: at low counts the √N-Gaussian weight biases the fit,
      and Cash removes that bias. Reported ``chi_squared`` is the Cash value
      (asymptotically χ²-distributed).
    - ``"gaussian"`` — √N-weighted least squares on the lifetime-corrected
      counts, WiMDA's weighting, kept byte-for-byte as the historical baseline.
      (√N weighting is invariant to the lifetime correction, so this is exactly
      the pre-cost-factory grouped fit.)

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
    _reject_affine_ties(initial_params.values(), "Grouped time-domain fitting")

    required_names = set(GROUP_NUISANCE_PARAMS) | set(global_params) | set(local_params)
    for group_id in group_ids:
        missing_names = sorted(required_names - set(initial_params[group_id].names))
        if missing_names:
            raise ValueError(
                f"Grouped time-domain parameters for {group_id!r} are missing: {missing_names}"
            )

    # Poisson (Cash) fits the raw counts against a raw-count expectation, so the
    # objective sees true Poisson statistics; Gaussian keeps the historical
    # lifetime-corrected √N least squares (cost_factory=None → byte-identical).
    use_poisson, cost_factory = _resolve_grouped_cost(cost)
    base_model_fn = build_grouped_count_model(polarization_model_fn)
    model_fn = _raw_count_model(base_model_fn) if use_poisson else base_model_fn

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
        if use_poisson:
            counts, error, metadata = _raw_count_dataset_fields(time, counts, metadata)
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
        cost_factory=cost_factory,
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
    #: The seeding mode actually used ("as_provided"/"chain"); for Auto this is the
    #: resolved choice. ``seeding_reason`` is a human-readable explanation the GUI
    #: surfaces so Auto is never silent.
    seeding_used: str = "as_provided"
    seeding_reason: str = ""


#: Concrete grouped-series seeding mechanisms. ``"as_provided"`` uses each member's
#: caller-built seed (the per-run or averaged seeds the GUI prepares); ``"chain"``
#: carries member N's fitted values into member N+1 (the WiMDA ``itPrevious``
#: analogue). ``"auto"`` is resolved to one of these by
#: :func:`recommend_grouped_series_seeding`.
GROUPED_SERIES_SEEDING: tuple[str, ...] = ("as_provided", "chain", "auto")

#: Auto chooses chaining only for an ordered scan of at least this many members.
_CHAIN_MIN_MEMBERS = 3


@dataclass(frozen=True)
class SeedingRecommendation:
    """The seeding mode Auto picked for a grouped series, with a human reason."""

    mode: str  # "as_provided" or "chain"
    reason: str


def recommend_grouped_series_seeding(
    member_runs: Sequence[int],
    order_key: dict[int, float] | None,
    *,
    min_members: int = _CHAIN_MIN_MEMBERS,
) -> SeedingRecommendation:
    """Pick a seeding mode for an independent grouped series (the "Auto" policy).

    Chaining from the previous run wins on **ordered scans** — a temperature or
    field sweep where each member's best seed is its neighbour, especially across a
    transition. It is pointless or harmful on unordered/repeat collections (a
    diverged member would poison the chain). So Auto chains only when a usable
    numeric ``order_key`` (run → temperature/field) spans a real range over at least
    ``min_members`` members; otherwise it leaves the caller's seeds in place. The
    returned ``reason`` is meant to be surfaced to the user — Auto is never silent.

    Delegates to the shared :func:`recommend_series_seeding` policy so the grouped and
    F-B asymmetry batch paths agree on when to chain; the result is wrapped in the
    grouped :class:`SeedingRecommendation` for backward compatibility.
    """
    shared = recommend_series_seeding(member_runs, order_key, min_members=min_members)
    return SeedingRecommendation(shared.mode, shared.reason)


def _chained_initial_from_member(
    prev_result: GroupedTimeDomainFitResult,
    provided: dict[Hashable, ParameterSet],
) -> dict[Hashable, ParameterSet]:
    """Build the next member's seed from ``prev_result``'s fitted values.

    Each group's seed keeps the provided structure (names, bounds, fixed, links) but
    takes the previous fit's fitted values, re-pinned through the normalised
    polarisation contract (amplitude→1, background→0, per W5). A group whose previous
    fit is missing or unsuccessful falls back to its provided seed.
    """
    chained: dict[Hashable, ParameterSet] = {}
    for group_id, seed in provided.items():
        group_result = prev_result.group_results.get(group_id)
        if group_result is None or not getattr(group_result, "success", False):
            chained[group_id] = seed
            continue
        fitted_names = set(group_result.parameters.names)
        carried = {
            p.name: (
                float(group_result.parameters[p.name].value) if p.name in fitted_names else p.value
            )
            for p in seed
        }
        normalised = normalize_to_grouped_contract([p.name for p in seed], carried)
        rebuilt = ParameterSet()
        for p in seed:
            rebuilt.add(
                Parameter(
                    name=p.name,
                    value=normalised[p.name],
                    min=p.min,
                    max=p.max,
                    fixed=p.fixed,
                    link_group=p.link_group,
                )
            )
        chained[group_id] = rebuilt
    return chained


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
    seeding: str = "as_provided",
    order_key: dict[int, float] | None = None,
    cost: str = "poisson",
    cross_run_local_params: list[str] | None = None,
    max_workers: int | None = None,
    block_separable: bool = False,
    profile_shared_errors: bool = False,
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

    A *mixed* fit (some physics shared across runs, others fitted per run) is
    expressed by routing through ``"global"`` and listing the per-run physics in
    ``cross_run_local_params``: those become engine local parameters grouped by
    source run, while the rest stay shared across all runs.

    ``max_workers`` opts into process-level parallelism of the independent
    (``"as_provided"``) batch path, where each run's joint fit is fully
    independent. ``None`` (the default) keeps the sequential, in-process path; a
    positive count dispatches up to ``min(max_workers, n_runs)`` runs across a
    spawn-based pool (the GUI passes ``os.cpu_count()``). For a ``"global"`` series
    it instead parallelises the inner per-run fits of the block-separable solver
    (see ``block_separable``); for chained seeding it has no effect (each run
    depends on the previous, so the chain is inherently sequential).

    ``block_separable`` permits the ``"global"`` path to solve large mixed fits by
    *alternating block minimisation* instead of one monolithic Minuit problem: the
    runs are independent except for the handful of cross-run-shared physics params,
    so the solver alternates between fitting each run independently (the shared
    params held — the same per-run work the batch path parallelises) and a small fit
    of just the shared params. This scales linearly in run count where the monolithic
    fit blows up superlinearly. It engages only above a free-parameter threshold
    (small global fits keep the monolithic path and its exact joint covariance); the
    shared-parameter uncertainties it reports are conditional on the fitted locals.
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

    if seeding not in GROUPED_SERIES_SEEDING:
        raise ValueError(
            f"Unknown grouped-series seeding {seeding!r}; expected one of {GROUPED_SERIES_SEEDING}"
        )
    _reject_affine_ties(
        (ps for run_sets in initial_params.values() for ps in run_sets.values()),
        "Grouped time-domain fitting",
    )

    engine = fit_engine or FitEngine()
    if relationship in ("individual", "batch"):
        # Resolve Auto to a concrete mechanism using the order-key policy; explicit
        # modes pass through with a descriptive reason.
        if seeding == "auto":
            recommendation = recommend_grouped_series_seeding(list(members), order_key)
        elif seeding == "chain":
            recommendation = SeedingRecommendation("chain", "chain from previous run (requested)")
        else:
            recommendation = SeedingRecommendation("as_provided", "independent seeds (requested)")
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
            seeding=recommendation.mode,
            seeding_reason=recommendation.reason,
            order_key=order_key,
            cost=cost,
            max_workers=max_workers,
        )
    # A "global" series is one simultaneous fit, so sequential chaining does not
    # apply; the seeding choice is recorded as-is for transparency.
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
        cost=cost,
        cross_run_local_params=cross_run_local_params,
        max_workers=max_workers,
        block_separable=block_separable,
        profile_shared_errors=profile_shared_errors,
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
    seeding: str = "as_provided",
    seeding_reason: str = "",
    order_key: dict[int, float] | None = None,
    cost: str = "poisson",
    max_workers: int | None = None,
) -> GroupedSeriesFitResult:
    """Run one independent grouped joint fit per member run (no cross-run sharing).

    With ``seeding="chain"`` each member is seeded from the previous member's fitted
    values (re-normalised to the grouped contract), iterating in ``order_key`` order
    so the chain follows the physical scan; a failed member resets the next member to
    its provided seed.

    With ``seeding="as_provided"`` the per-run fits share no state, so a positive
    ``max_workers`` dispatches them across a process pool (``None``/``1`` → the
    in-process sequential path). Results are independent of worker count. Chained
    seeding always runs sequentially.
    """
    # Chaining follows the physical scan order; fall back to the caller's member order
    # when no order key is supplied.
    run_order = list(members)
    if seeding == "chain" and order_key:
        run_order = sorted(run_order, key=lambda r: (order_key.get(int(r), math.inf), int(r)))

    member_results: dict[int, FitResult] = {}
    member_source_run: dict[int, int] = {}
    member_group_id: dict[int, Hashable] = {}
    messages: dict[int, str] = {}

    def record(run: int, result: GroupedTimeDomainFitResult) -> None:
        messages[run] = f"run {run}: {result.message}"
        for index, group in enumerate(members[run], start=1):
            key = _group_dataset_run_number(run, index)
            group_result = result.group_results.get(group.group_id)
            if group_result is None:
                continue
            member_results[key] = group_result
            member_source_run[key] = run
            member_group_id[key] = group.group_id

    parallel_results: dict[int, GroupedTimeDomainFitResult] | None = None
    workers = _resolve_grouped_series_workers(max_workers, len(run_order))
    if (
        seeding != "chain"
        and workers > 1
        and _grouped_series_payload_picklable(polarization_model_fn)
    ):
        # Independent seeds → the members share no state, so each run's joint fit is
        # a self-contained, deterministic problem. Dispatch them across processes
        # (iminuit calls back into Python for every cost evaluation, so threads would
        # serialise on the GIL; processes give real parallelism). Results are
        # bit-identical to the sequential path regardless of worker count; a pool that
        # cannot start (or breaks) returns ``None`` and we fall through to sequential.
        parallel_results = _fit_members_parallel(
            run_order,
            members,
            polarization_model_fn,
            global_params,
            local_params,
            initial_params,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            minos=minos,
            cost=cost,
            cancel_callback=cancel_callback,
            workers=workers,
        )

    if parallel_results is not None:
        for run in (int(r) for r in run_order):
            record(run, parallel_results[run])
    else:
        previous: GroupedTimeDomainFitResult | None = None
        for raw_run in run_order:
            groups = members[raw_run]
            # Cooperative cancel between member fits (the minimum abort granularity):
            # a cancelled series records nothing and the loop stops cleanly here.
            if cancel_callback is not None and bool(cancel_callback()):
                raise FitCancelledError("Fit cancelled.")
            run = int(raw_run)
            provided = initial_params.get(run, {})
            if seeding == "chain" and previous is not None:
                run_initial = _chained_initial_from_member(previous, provided)
            else:
                run_initial = provided
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
                cost=cost,
            )
            # Only chain from a successful member; a failed fit resets the next member
            # to its provided seed rather than propagating a diverged seed down the scan.
            previous = result if result.success else None
            record(run, result)

    # Report messages in the (physical-scan) run order, independent of completion order.
    ordered_messages = [messages[int(r)] for r in run_order if int(r) in messages]
    success = bool(member_results) and all(r.success for r in member_results.values())
    return GroupedSeriesFitResult(
        success=success,
        relationship=relationship,
        member_results=member_results,
        member_source_run=member_source_run,
        member_group_id=member_group_id,
        shared_parameters=ParameterSet(),
        message="; ".join(ordered_messages),
        seeding_used=seeding,
        seeding_reason=seeding_reason,
    )


def _resolve_grouped_series_workers(max_workers: int | None, n_runs: int) -> int:
    """Resolve the worker count for an independent batch (clamped to ``[1, n_runs]``).

    Parallelism is opt-in: ``None`` (the default) keeps the sequential, in-process
    path so programmatic callers and spies are unaffected. A positive count is
    honoured but never exceeds the run count (extra workers would just sit idle).
    The GUI passes ``os.cpu_count()`` to auto-size the pool to the host.
    """
    if max_workers is None or n_runs <= 1:
        return 1
    return max(1, min(int(max_workers), n_runs))


def _grouped_series_payload_picklable(model_fn) -> bool:
    """Whether the per-run payload can cross a process boundary.

    The group arrays and :class:`ParameterSet` seeds are always picklable; the one
    real risk is the polarization model function (e.g. a user-defined Python model
    captured as a closure). If it cannot be pickled the caller falls back to the
    in-process sequential path rather than crashing on dispatch.
    """
    import pickle

    try:
        pickle.dumps(model_fn)
    except Exception:
        return False
    return True


def _grouped_member_worker(payload):
    """Process-pool entry point: fit one run's groups and return ``(run, result)``.

    Module-level (so it survives the ``spawn`` start method) and engine-free — each
    worker builds its own :class:`FitEngine`. Cancellation is handled in the parent
    between completions, so no cancel callback crosses the boundary.
    """
    (
        run,
        groups,
        polarization_model_fn,
        global_params,
        local_params,
        run_initial,
        t_min,
        t_max,
        method,
        max_calls,
        minos,
        cost,
    ) = payload
    result = fit_grouped_time_domain(
        groups,
        polarization_model_fn,
        global_params=global_params,
        local_params=local_params,
        initial_params=run_initial,
        fit_engine=None,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=None,
        cost=cost,
    )
    return run, result


def _fit_members_parallel(
    run_order: list[int],
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    global_params: list[str],
    local_params: list[str],
    initial_params: dict[int, dict[Hashable, ParameterSet]],
    *,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    minos: bool,
    cost: str,
    cancel_callback: Callable[[], bool] | None,
    workers: int,
) -> dict[int, GroupedTimeDomainFitResult] | None:
    """Fit every independent member across a process pool; return ``{run: result}``.

    Members are dispatched to a spawn-based :class:`ProcessPoolExecutor` and collected
    as they complete (the caller folds them in by run number, so completion order does
    not matter). Results are bit-identical to the sequential path regardless of worker
    count. Returns ``None`` — signalling the caller to run sequentially instead — when
    a spawn-safe pool cannot start or the pool breaks mid-run (a constrained or frozen
    environment); raises :class:`FitCancelledError` on a cooperative cancel.

    Cancellation is coarse: a requested cancel stops collecting further results, but an
    in-flight member fit runs to completion (the same per-member abort granularity as
    the sequential path).
    """
    payloads = [
        (
            int(raw_run),
            members[raw_run],
            polarization_model_fn,
            global_params,
            local_params,
            initial_params.get(int(raw_run), {}),
            t_min,
            t_max,
            method,
            max_calls,
            minos,
            cost,
        )
        for raw_run in run_order
    ]
    if cancel_callback is not None and bool(cancel_callback()):
        raise FitCancelledError("Fit cancelled.")
    # No spawn-safe workers here (e.g. a restricted sandbox) → the caller falls back to
    # the sequential path, which produces identical results.
    executor = open_spawn_pool(workers)
    if executor is None:
        return None
    results: dict[int, GroupedTimeDomainFitResult] = {}
    try:
        futures = {
            executor.submit(_grouped_member_worker, payload): payload[0] for payload in payloads
        }
        for future in as_completed(futures):
            if cancel_callback is not None and bool(cancel_callback()):
                raise FitCancelledError("Fit cancelled.")
            run, result = future.result()
            results[run] = result
    except BrokenExecutor:
        # A worker died for an environmental reason (not a fit failure — failed fits
        # return success=False without raising). Abandon parallelism and let the
        # caller re-run the batch sequentially rather than report partial results.
        return None
    finally:
        # Drop pending work immediately; in-flight processes finish on their own.
        executor.shutdown(wait=False, cancel_futures=True)
    return results


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
    cost: str = "poisson",
    cross_run_local_params: list[str] | None = None,
    max_workers: int | None = None,
    block_separable: bool = False,
    profile_shared_errors: bool = False,
) -> GroupedSeriesFitResult:
    """Fit every ``(run, group)`` simultaneously.

    Physics in ``global_params`` is shared across all runs and groups, except the
    subset in ``cross_run_local_params``, which is shared across each run's groups
    but fitted independently *per run* (the mixed Global/Local case). The
    per-group nuisance block is always per ``(run, group)``.

    With ``block_separable`` and a large free-parameter count the joint problem is
    solved by alternating block minimisation (see :func:`_fit_grouped_series_global_blockwise`)
    rather than one monolithic Minuit fit; otherwise the monolithic path runs.
    """
    cross_run_local = set(cross_run_local_params or [])
    use_poisson, cost_factory = _resolve_grouped_cost(cost)
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
            if use_poisson:
                counts, error, metadata = _raw_count_dataset_fields(time, counts, metadata)
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

    # Split physics: cross-run-global stays shared across every dataset; the
    # per-run subset becomes an engine local parameter grouped by its source run,
    # so a run's groups share one value while runs stay independent.
    engine_global = [name for name in global_params if name not in cross_run_local]
    engine_local = list(local_params) + [name for name in global_params if name in cross_run_local]
    local_param_groups = (
        {name: dict(member_source_run) for name in cross_run_local} if cross_run_local else None
    )

    base_model_fn = build_grouped_count_model(polarization_model_fn)
    model_fn = _raw_count_model(base_model_fn) if use_poisson else base_model_fn

    # The joint objective is a sum over (run, group) coupled across runs ONLY through
    # the cross-run-shared physics (engine_global); for fixed shared values the runs
    # are independent. A monolithic Minuit fit ignores that structure and scales
    # superlinearly, so for a large free-parameter count solve it by alternating block
    # minimisation instead (shared params held → independent per-run fits, then a small
    # shared-only fit). Small fits keep the monolithic path and its exact joint errors.
    representative = temporary_initial[next(iter(temporary_initial))]
    free_global = [name for name in engine_global if not representative[name].fixed]
    if (
        block_separable
        and free_global
        and _grouped_global_is_large(
            representative,
            truly_global=engine_global,
            per_run_physics=list(cross_run_local),
            nuisances=list(local_params),
            n_runs=len(members),
            n_members=len(temporary_datasets),
        )
    ):
        return _fit_grouped_series_global_blockwise(
            members,
            polarization_model_fn,
            truly_global=engine_global,
            per_run_physics=list(cross_run_local),
            nuisances=list(local_params),
            initial_params=initial_params,
            temporary_datasets=temporary_datasets,
            temporary_initial=temporary_initial,
            member_source_run=member_source_run,
            member_group_id=member_group_id,
            model_fn=model_fn,
            cost_factory=cost_factory,
            engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            cost=cost,
            cancel_callback=cancel_callback,
            max_workers=max_workers,
            profile_shared_errors=profile_shared_errors,
        )

    internal_results, shared_parameters = engine.global_fit(
        temporary_datasets,
        model_fn,
        global_params=engine_global,
        local_params=engine_local,
        initial_params=temporary_initial,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=cancel_callback,
        cost_factory=cost_factory,
        local_param_groups=local_param_groups,
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


#: Below this many free parameters the monolithic global fit is fast and gives exact
#: joint errors, so block minimisation is not worth its conditional-error trade-off.
_BLOCK_SEPARABLE_MIN_FREE_PARAMS = 64


def _grouped_global_is_large(
    representative: ParameterSet,
    *,
    truly_global: list[str],
    per_run_physics: list[str],
    nuisances: list[str],
    n_runs: int,
    n_members: int,
    threshold: int = _BLOCK_SEPARABLE_MIN_FREE_PARAMS,
) -> bool:
    """Whether the monolithic joint fit would carry enough free params to be worth
    block minimisation.

    Counts free shared params once, free per-run physics once per run, and free
    nuisances once per ``(run, group)`` member — the same accounting the engine uses
    to build the Minuit vector.
    """
    free_global = sum(1 for name in truly_global if not representative[name].fixed)
    free_phys = sum(1 for name in per_run_physics if not representative[name].fixed)
    free_nuis = sum(1 for name in nuisances if not representative[name].fixed)
    free_total = free_global + free_phys * n_runs + free_nuis * n_members
    return free_total >= threshold


def _copy_param_set_with(
    source: ParameterSet,
    *,
    fix: set[str] | None = None,
    free: set[str] | None = None,
    values: dict[str, float] | None = None,
) -> ParameterSet:
    """Return a copy of ``source`` with selected parameters re-fixed/-freed/-revalued.

    Preserves every parameter's bounds and link metadata; only the ``fixed`` flag (for
    names in ``fix``/``free``) and ``value`` (for names in ``values``) are overridden.
    """
    fix = fix or set()
    free = free or set()
    values = values or {}
    out = ParameterSet()
    for p in source:
        fixed = p.fixed
        if p.name in fix:
            fixed = True
        if p.name in free:
            fixed = False
        out.add(
            Parameter(
                name=p.name,
                value=float(values.get(p.name, p.value)),
                min=p.min,
                max=p.max,
                fixed=fixed,
                link_group=p.link_group,
            )
        )
    return out


def _blockwise_inner_payload(
    run,
    members,
    polarization_model_fn,
    inner_global,
    nuisances,
    run_seeds,
    t_min,
    t_max,
    method,
    max_calls,
    cost,
):
    """Build the ``_grouped_member_worker`` argument tuple for one run's inner fit."""
    return (
        int(run),
        members[run],
        polarization_model_fn,
        inner_global,
        nuisances,
        run_seeds[int(run)],
        t_min,
        t_max,
        method,
        max_calls,
        False,
        cost,
    )


def _blockwise_inner_fit_runs(
    run_order: list[int],
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    inner_global: list[str],
    nuisances: list[str],
    run_seeds: dict[int, dict[Hashable, ParameterSet]],
    *,
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    cost: str,
    cancel_callback: Callable[[], bool] | None,
    executor: ProcessPoolExecutor | None,
) -> dict[int, GroupedTimeDomainFitResult]:
    """Fit every run independently for one block-minimisation round → ``{run: result}``.

    With the shared params held (they are fixed inside ``run_seeds``) each run is a
    self-contained grouped fit. When ``executor`` is supplied the per-run fits are
    submitted to that persistent pool (created once for the whole solve, not per
    round, so the spawn cost is paid only at start-up); otherwise they run in a
    sequential loop. A broken pool raises :class:`BrokenExecutor` for the caller to
    handle.
    """
    if executor is not None:
        futures = {
            executor.submit(
                _grouped_member_worker,
                _blockwise_inner_payload(
                    run,
                    members,
                    polarization_model_fn,
                    inner_global,
                    nuisances,
                    run_seeds,
                    t_min,
                    t_max,
                    method,
                    max_calls,
                    cost,
                ),
            ): int(run)
            for run in run_order
        }
        results: dict[int, GroupedTimeDomainFitResult] = {}
        for future in as_completed(futures):
            if cancel_callback is not None and bool(cancel_callback()):
                raise FitCancelledError("Fit cancelled.")
            run, result = future.result()
            results[run] = result
        return results

    results = {}
    for raw_run in run_order:
        if cancel_callback is not None and bool(cancel_callback()):
            raise FitCancelledError("Fit cancelled.")
        run = int(raw_run)
        results[run] = fit_grouped_time_domain(
            members[raw_run],
            polarization_model_fn,
            global_params=inner_global,
            local_params=nuisances,
            initial_params=run_seeds[run],
            fit_engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            minos=False,
            cancel_callback=cancel_callback,
            cost=cost,
        )
    return results


def _inner_round_with_fallback(
    executor: ProcessPoolExecutor | None,
    run_order: list[int],
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    inner_global: list[str],
    nuisances: list[str],
    run_seeds: dict[int, dict[Hashable, ParameterSet]],
    *,
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    cost: str,
    cancel_callback: Callable[[], bool] | None,
) -> tuple[dict[int, GroupedTimeDomainFitResult], ProcessPoolExecutor | None]:
    """Run one inner block round; on a broken pool, retry sequentially.

    Returns ``(results, executor)`` where the returned executor is ``None`` once the
    pool has broken for an environmental reason, so callers stop dispatching to a dead
    pool on later rounds. The single home for the pool-failure policy shared by the
    alternating loop and the error-profiling probes.
    """
    kwargs = dict(
        engine=engine,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        cost=cost,
        cancel_callback=cancel_callback,
    )
    if executor is not None:
        try:
            results = _blockwise_inner_fit_runs(
                run_order,
                members,
                polarization_model_fn,
                inner_global,
                nuisances,
                run_seeds,
                executor=executor,
                **kwargs,
            )
            return results, executor
        except BrokenExecutor:
            executor = None
    results = _blockwise_inner_fit_runs(
        run_order,
        members,
        polarization_model_fn,
        inner_global,
        nuisances,
        run_seeds,
        executor=None,
        **kwargs,
    )
    return results, None


def _blockwise_run_rounds(
    *,
    run_order: list[int],
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    truly_global: list[str],
    per_run_physics: list[str],
    nuisances: list[str],
    inner_global: list[str],
    run_seeds: dict[int, dict[Hashable, ParameterSet]],
    shared_values: dict[str, float],
    temporary_datasets: list[MuonDataset],
    member_source_run: dict[int, int],
    member_group_id: dict[int, Hashable],
    model_fn,
    cost_factory: CostFactory | None,
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    cost: str,
    cancel_callback: Callable[[], bool] | None,
    executor: ProcessPoolExecutor | None,
    max_rounds: int,
    tol: float,
) -> tuple[dict[int, GroupedTimeDomainFitResult], dict[str, float], dict[str, float], int]:
    """Run the alternating rounds; return ``(inner_results, shared, shared_unc, rounds)``.

    Each round: (a) fit every run independently with the shared params held, then
    (b) fit only the shared params with all locals held. A broken pool drops to the
    sequential path for that round and the rest of the solve.
    """
    inner_results: dict[int, GroupedTimeDomainFitResult] = {}
    shared_uncertainties: dict[str, float] = {}
    rounds_run = 0
    for _round in range(max_rounds):
        if cancel_callback is not None and bool(cancel_callback()):
            raise FitCancelledError("Fit cancelled.")
        rounds_run += 1

        # (a) Pin the shared params, then fit each run independently.
        for run in run_order:
            run = int(run)
            run_seeds[run] = {
                gid: _copy_param_set_with(ps, fix=set(truly_global), values=shared_values)
                for gid, ps in run_seeds[run].items()
            }
        inner_results, executor = _inner_round_with_fallback(
            executor,
            run_order,
            members,
            polarization_model_fn,
            inner_global,
            nuisances,
            run_seeds,
            engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            cost=cost,
            cancel_callback=cancel_callback,
        )

        # Warm-start the next round: carry each fitted local value back into the seeds.
        updatable = set(per_run_physics) | set(nuisances)
        for run in run_order:
            run = int(run)
            result = inner_results.get(run)
            if result is None:
                continue
            for gid, seed in run_seeds[run].items():
                fitted = result.group_results.get(gid)
                if fitted is None:
                    continue
                carried = {
                    name: float(fitted.parameters[name].value)
                    for name in updatable
                    if name in fitted.parameters.names
                }
                run_seeds[run][gid] = _copy_param_set_with(seed, values=carried)

        # (b) Hold all locals at their fitted values; fit only the shared params.
        outer_initial = {
            key: _copy_param_set_with(
                run_seeds[member_source_run[key]][member_group_id[key]],
                fix=set(per_run_physics) | set(nuisances),
                free=set(truly_global),
                values=shared_values,
            )
            for key in member_source_run
        }
        outer_results, outer_shared = engine.global_fit(
            temporary_datasets,
            model_fn,
            global_params=list(truly_global),
            local_params=list(nuisances) + list(per_run_physics),
            initial_params=outer_initial,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            minos=False,
            cancel_callback=cancel_callback,
            cost_factory=cost_factory,
            minuit_strategy=0,
        )
        new_shared = {name: float(outer_shared[name].value) for name in truly_global}
        shared_uncertainties = {
            p.name: float(u)
            for p in outer_shared
            for u in (_param_uncertainty(outer_results, p.name),)
            if u is not None
        }

        # The shared params are the ONLY cross-run coupling, so once a full alternation
        # leaves them unchanged the inner fits reproduce the same locals and the
        # iteration is at its fixed point — that is the convergence test.
        shared_converged = all(
            _relative_change(shared_values[name], new_shared[name]) <= tol for name in truly_global
        )
        shared_values = new_shared
        if rounds_run >= 2 and shared_converged:
            break

    return inner_results, shared_values, shared_uncertainties, rounds_run


def _fit_grouped_series_global_blockwise(
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    *,
    truly_global: list[str],
    per_run_physics: list[str],
    nuisances: list[str],
    initial_params: dict[int, dict[Hashable, ParameterSet]],
    temporary_datasets: list[MuonDataset],
    temporary_initial: dict[int, ParameterSet],
    member_source_run: dict[int, int],
    member_group_id: dict[int, Hashable],
    model_fn,
    cost_factory: CostFactory | None,
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    cost: str,
    cancel_callback: Callable[[], bool] | None,
    max_workers: int | None,
    profile_shared_errors: bool = False,
    max_rounds: int = 12,
    tol: float = 1.0e-4,
) -> GroupedSeriesFitResult:
    """Solve the mixed global/local grouped fit by alternating block minimisation.

    The joint objective couples runs only through ``truly_global`` (the cross-run
    shared physics). Each round (a) holds those shared params and fits every run
    independently — its per-run physics ``per_run_physics`` shared across the run's
    groups, ``nuisances`` per ``(run, group)`` — then (b) holds all locals and fits
    only the shared params. Iterating to a fixed point reaches the joint optimum while
    each sub-problem stays small and well-conditioned, so cost scales linearly in run
    count. Shared-parameter uncertainties come from step (b) and are conditional on the
    fitted locals; per-(run, group) uncertainties come from step (a).
    """
    run_order = list(members)
    workers = _resolve_grouped_series_workers(max_workers, len(run_order))

    # Mutable per-run seeds carry the structural template (bounds, fixed phases) and the
    # latest fitted values across rounds; the shared params are pinned fixed for step (a).
    run_seeds: dict[int, dict[Hashable, ParameterSet]] = {
        int(run): {gid: _copy_param_set_with(ps) for gid, ps in groups.items()}
        for run, groups in initial_params.items()
    }
    representative = temporary_initial[next(iter(temporary_initial))]
    shared_values: dict[str, float] = {
        name: float(representative[name].value) for name in truly_global
    }
    # Captured from the ORIGINAL seeds: the round loop pins the shared params fixed in
    # run_seeds, so their free/fixed status must be read before that mutation.
    free_shared = [name for name in truly_global if not representative[name].fixed]
    inner_global = list(per_run_physics) + list(truly_global)

    # One persistent pool for the whole solve — the inner per-run fits repeat every
    # round, so creating the (spawn) pool per round would pay the import cost N times
    # and erase the parallel gain. Created once here, reused across rounds, torn down
    # in finally (covering the cancel path).
    executor: ProcessPoolExecutor | None = None
    if workers > 1 and _grouped_series_payload_picklable(polarization_model_fn):
        executor = open_spawn_pool(workers)

    try:
        inner_results, shared_values, shared_uncertainties, rounds_run = _blockwise_run_rounds(
            run_order=run_order,
            members=members,
            polarization_model_fn=polarization_model_fn,
            truly_global=truly_global,
            per_run_physics=per_run_physics,
            nuisances=nuisances,
            inner_global=inner_global,
            run_seeds=run_seeds,
            shared_values=shared_values,
            temporary_datasets=temporary_datasets,
            member_source_run=member_source_run,
            member_group_id=member_group_id,
            model_fn=model_fn,
            cost_factory=cost_factory,
            engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            cost=cost,
            cancel_callback=cancel_callback,
            executor=executor,
            max_rounds=max_rounds,
            tol=tol,
        )
        # Rigorous (marginal) shared-parameter errors: profile the few shared params,
        # re-optimising all locals at each probe (the numerical Schur complement), while
        # the pool is still alive. Falls back to the conditional errors on any failure.
        profiled_errors = (
            _profile_shared_covariance(
                truly_global=truly_global,
                free_shared=free_shared,
                shared_values=shared_values,
                conditional=shared_uncertainties,
                run_seeds=run_seeds,
                run_order=run_order,
                members=members,
                polarization_model_fn=polarization_model_fn,
                inner_global=inner_global,
                nuisances=nuisances,
                engine=engine,
                t_min=t_min,
                t_max=t_max,
                method=method,
                max_calls=max_calls,
                cost=cost,
                cancel_callback=cancel_callback,
                executor=executor,
            )
            if profile_shared_errors
            else None
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    errors_profiled = bool(profiled_errors)
    if profiled_errors:
        shared_uncertainties = {**shared_uncertainties, **profiled_errors}

    # Final per-(run, group) results come from the last inner round; inject the converged
    # shared values + their (conditional or profiled) uncertainties so each member reports
    # the full set.
    member_results: dict[int, FitResult] = {}
    for key, run in member_source_run.items():
        result = inner_results.get(int(run))
        if result is None:
            continue
        group_result = result.group_results.get(member_group_id[key])
        if group_result is None:
            continue
        member_results[key] = _inject_shared_parameters(
            group_result, shared_values, shared_uncertainties
        )

    shared_parameters = ParameterSet(
        [
            Parameter(
                name=name,
                value=shared_values[name],
                min=temporary_initial[next(iter(temporary_initial))][name].min,
                max=temporary_initial[next(iter(temporary_initial))][name].max,
            )
            for name in truly_global
        ]
    )

    success = bool(member_results) and all(r.success for r in member_results.values())
    if success:
        error_note = (
            "shared-parameter errors profiled over the locals"
            if errors_profiled
            else "shared-parameter errors are conditional on the fitted locals"
        )
        message = (
            f"Grouped-series global fit successful (block-separable solver, "
            f"{rounds_run} round{'s' if rounds_run != 1 else ''}; {error_note})"
        )
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


def _relative_change(old: float, new: float) -> float:
    """Scale-free change between two scalars (absolute when ``old`` is ~0)."""
    denom = max(abs(old), abs(new), 1.0e-12)
    return abs(new - old) / denom


def _param_uncertainty(results: dict[int, FitResult], name: str) -> float | None:
    """First finite uncertainty reported for ``name`` across per-dataset results."""
    for result in results.values():
        unc = getattr(result, "uncertainties", None) or {}
        value = unc.get(name)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return None


def _inject_shared_parameters(
    group_result: FitResult,
    shared_values: dict[str, float],
    shared_uncertainties: dict[str, float],
) -> FitResult:
    """Return ``group_result`` with the converged shared params set on its parameter set.

    The per-run inner fit held the shared params fixed at the round's value; this stamps
    the final converged value (and conditional uncertainty) so each member reports the
    complete physics set, matching the monolithic path's per-member result shape.
    """
    params = group_result.parameters
    rebuilt = ParameterSet()
    for p in params:
        if p.name in shared_values:
            rebuilt.add(Parameter(name=p.name, value=shared_values[p.name], min=p.min, max=p.max))
        else:
            rebuilt.add(p)
    group_result.parameters = rebuilt
    merged_unc = dict(getattr(group_result, "uncertainties", None) or {})
    merged_unc.update(shared_uncertainties)
    group_result.uncertainties = merged_unc
    return group_result


def _profile_shared_covariance(
    *,
    truly_global: list[str],
    free_shared: list[str],
    shared_values: dict[str, float],
    conditional: dict[str, float],
    run_seeds: dict[int, dict[Hashable, ParameterSet]],
    run_order: list[int],
    members: dict[int, list[GroupedTimeDomainGroup]],
    polarization_model_fn,
    inner_global: list[str],
    nuisances: list[str],
    engine: FitEngine,
    t_min: float | None,
    t_max: float | None,
    method: str,
    max_calls: int,
    cost: str,
    cancel_callback: Callable[[], bool] | None,
    executor: ProcessPoolExecutor | None,
) -> dict[str, float] | None:
    """Marginal (profiled) uncertainties for the free shared params, or ``None``.

    A *conditional* error (the shared-only outer step) ignores that the per-run locals
    would readjust as a shared param moves. This profiles the objective in the shared
    subspace: each probe re-optimises every local (one block inner round, reusing the
    pool), so the resulting reduced Hessian — finite-differenced over the shared params,
    inverted as ``2·H⁻¹`` — is the numerical Schur complement, i.e. the same marginal
    covariance a full joint HESSE would give for those params, at a fraction of the cost.

    Returns a name→σ map for the free shared params whose profiled variance is finite and
    positive; ``None`` if the curvature is unusable (caller keeps the conditional errors).
    """
    # Only free shared params have an error to profile (a fixed shared param is held).
    names = list(free_shared)
    n = len(names)
    if n == 0:
        return None

    fix_shared = set(truly_global)
    # The pool is reused across probes; a probe that breaks it reverts to sequential.
    pool = [executor]

    def chi2_at(overrides: dict[str, float]) -> float:
        values = dict(shared_values)
        values.update(overrides)
        seeds = {
            int(run): {
                gid: _copy_param_set_with(ps, fix=fix_shared, values=values)
                for gid, ps in run_seeds[int(run)].items()
            }
            for run in run_order
        }
        results, pool[0] = _inner_round_with_fallback(
            pool[0],
            run_order,
            members,
            polarization_model_fn,
            inner_global,
            nuisances,
            seeds,
            engine=engine,
            t_min=t_min,
            t_max=t_max,
            method=method,
            max_calls=max_calls,
            cost=cost,
            cancel_callback=cancel_callback,
        )
        total = 0.0
        for run in run_order:
            grouped = results.get(int(run))
            # A failed probe fit reports chi_squared=0.0 by default; summing it would
            # understate this probe's objective and corrupt the finite-difference
            # Hessian. Treat any non-converged probe as unusable → the caller keeps the
            # conditional errors rather than profiling from a poisoned curvature.
            if grouped is None or not grouped.success:
                return float("nan")
            for group_result in grouped.group_results.values():
                total += float(getattr(group_result, "chi_squared", 0.0) or 0.0)
        return total

    deltas: dict[str, float] = {}
    for name in names:
        sigma = conditional.get(name)
        if sigma is not None and np.isfinite(sigma) and sigma > 0.0:
            step = float(sigma)
        else:
            step = 1.0e-3 * max(abs(shared_values[name]), 1.0)
        deltas[name] = max(step, 1.0e-9)

    chi0 = chi2_at({})
    plus = {name: chi2_at({name: shared_values[name] + deltas[name]}) for name in names}
    minus = {name: chi2_at({name: shared_values[name] - deltas[name]}) for name in names}
    if not all(np.isfinite([chi0, *plus.values(), *minus.values()])):
        return None

    hessian = np.zeros((n, n))
    for i, name in enumerate(names):
        hessian[i, i] = (plus[name] - 2.0 * chi0 + minus[name]) / deltas[name] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = names[i], names[j]
            corner = chi2_at(
                {ni: shared_values[ni] + deltas[ni], nj: shared_values[nj] + deltas[nj]}
            )
            if not np.isfinite(corner):
                return None
            mixed = (corner - plus[ni] - plus[nj] + chi0) / (deltas[ni] * deltas[nj])
            hessian[i, j] = hessian[j, i] = mixed

    try:
        # χ² (and Cash) are 2·NLL, so the parameter covariance is 2·(∂²χ²/∂θ²)⁻¹.
        covariance = 2.0 * np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None

    profiled: dict[str, float] = {}
    for i, name in enumerate(names):
        variance = covariance[i, i]
        if np.isfinite(variance) and variance > 0.0:
            profiled[name] = float(np.sqrt(variance))
    return profiled or None
