"""Request-scoped detector count inputs for grouped fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_corrections import (
    count_background_domain,
    require_count_background,
)
from asymmetry.core.transform.corrections import (
    CorrectionOutcome,
    require_complete_dataset,
)
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow
from asymmetry.core.transform.grouping import effective_group_indices, resolve_alignment
from asymmetry.core.transform.rebin import binning_spec, rebin_counts
from asymmetry.core.transform.reduce import correction_flags_from_grouping
from asymmetry.core.transform.scoped_counts import ScopedCounts, observation_domain
from asymmetry.core.transform.t0 import common_t0_time_us, effective_detector_t0_bins
from asymmetry.core.utils.constants import MUON_LIFETIME_US

if TYPE_CHECKING:
    from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainGroup


def count_time_selection(first, last, factor, axis_start, width, residual, t_min, t_max, exclude):
    """Use merged-bin centres for selection, retaining every contributing raw bin."""
    size = last - first + 1
    time = (np.arange(size, dtype=float) + axis_start) * width + residual
    if factor > 1:
        time = time[: size // factor * factor].reshape(-1, factor).mean(axis=1)
    mask = np.ones(time.size, dtype=bool)
    if t_min is not None:
        mask &= time >= float(t_min)
    if t_max is not None:
        mask &= time <= float(t_max)
    if exclude is not None and float(exclude[1]) > float(exclude[0]):
        mask &= ~((time >= float(exclude[0])) & (time <= float(exclude[1])))
    return time, mask


@dataclass
class _CountGroupContext:
    """Shared detector alignment, window, correction, and bunching settings."""

    run: Any
    prepared: ScopedCounts
    alignment: AlignmentCrop
    first_good: int
    last_good: int
    bunch_factor: int
    bin_width: float
    axis_start: float
    #: Exact-t0 residual from the common bin centre, in µs.
    t0_residual_us: float
    group_names: dict[Any, Any]
    background_preflight: dict[str, CorrectionOutcome]


def _count_group_setup(dataset: MuonDataset) -> tuple[Any, dict, dict, dict]:
    """Require histograms and a grouping at the count-fit input boundary."""
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
    grouping: dict, groups_raw: dict, included_groups: dict, n_histograms: int
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
        indices = effective_group_indices(grouping, group_id, n_histograms=n_histograms)
        groups.append((group_id, indices))
    return groups


def _count_group_context(
    dataset: MuonDataset,
    grouping: dict,
    run: Any,
    group_specs: list[tuple[int, list[int]]],
    t_min,
    t_max,
    exclude,
) -> _CountGroupContext | None:
    require_complete_dataset(dataset)
    prepared_histograms = list(run.histograms)
    detector_t0_bins = effective_detector_t0_bins(prepared_histograms, grouping)
    alignment = resolve_alignment(
        prepared_histograms,
        *(indices for _, indices in group_specs),
        detector_t0_bins=detector_t0_bins,
    )
    window = alignment.window(grouping)
    first_good, last_good = window.first, window.last
    bunch_factor = binning_spec(grouping).factor
    selected = sorted({i for _, indices in group_specs for i in indices})
    bin_width = float(prepared_histograms[selected[0]].bin_width)
    axis_start = first_good - alignment.common_t0_bin
    t0_residual_us = (alignment.common_t0_bin + 0.5) * bin_width - common_t0_time_us(
        prepared_histograms,
        grouping,
        alignment.common_t0_bin,
        selected_indices=selected,
    )
    group_names = (
        grouping.get("group_names") if isinstance(grouping.get("group_names"), dict) else {}
    )
    _, mask = count_time_selection(
        first_good,
        last_good,
        bunch_factor,
        axis_start,
        bin_width,
        t0_residual_us,
        t_min,
        t_max,
        exclude,
    )
    if not np.any(mask):
        return None
    windows = [
        BinWindow(first_good + i * bunch_factor, first_good + (i + 1) * bunch_factor - 1)
        for i in np.flatnonzero(mask)
    ]
    parts = [(alignment, selected, windows)]
    if correction_flags_from_grouping(grouping).use_background:
        parts.extend(
            count_background_domain(prepared_histograms, grouping, group_specs, run.metadata)
        )
    domain = observation_domain(parts)
    prepared = ScopedCounts.prepare(
        prepared_histograms,
        grouping,
        domain,
        use_deadtime=bool(grouping.get("deadtime_correction", False)),
    )
    background = {}
    if correction_flags_from_grouping(grouping).use_background:
        background = require_count_background(prepared, grouping, group_specs, run.metadata)
    return _CountGroupContext(
        run=run,
        prepared=prepared,
        alignment=alignment,
        first_good=first_good,
        last_good=last_good,
        bunch_factor=bunch_factor,
        bin_width=bin_width,
        axis_start=axis_start,
        t0_residual_us=t0_residual_us,
        group_names=group_names,
        background_preflight=background,
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
) -> GroupedTimeDomainGroup:
    """Align counts and measured variance before bunching and masking."""
    from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainGroup

    run = ctx.run
    aligned = [ctx.prepared.aligned([index], ctx.alignment) for index in indices]
    terminal_bin = ctx.last_good + 1
    detector_counts = np.asarray([trace.counts[ctx.first_good : terminal_bin] for trace in aligned])
    counts = detector_counts.sum(axis=0)
    variance = counts
    if any(trace.variance is not None for trace in aligned):
        variance = np.sum(
            [trace.independent_variance[ctx.first_good : terminal_bin] for trace in aligned], axis=0
        )
    trimmed_counts = counts
    bunch_factor = ctx.bunch_factor
    time, mask = count_time_selection(
        ctx.first_good,
        terminal_bin - 1,
        bunch_factor,
        ctx.axis_start,
        ctx.bin_width,
        ctx.t0_residual_us,
        t_min,
        t_max,
        exclude,
    )
    if bunch_factor > 1:
        _, variance = rebin_counts(np.arange(variance.size), variance, bunch_factor)
        detector_counts = np.asarray(
            [
                rebin_counts(np.arange(trace.size), trace, bunch_factor)[1]
                for trace in detector_counts
            ]
        )
        trimmed_counts = detector_counts.sum(axis=0)

    scale = np.exp(time / MUON_LIFETIME_US) if lifetime_corrected else 1.0
    out_counts = trimmed_counts * scale
    errors = np.sqrt(np.clip(variance, 1.0, None)) * scale

    group_name = ctx.group_names.get(group_id, f"Group {group_id}")
    metadata = dict(run.metadata)
    metadata.update(
        {
            "group_id": int(group_id),
            "group_name": str(group_name),
            "grouped_time_domain": True,
            "count_background_preflight": {
                label: outcome.to_dict() for label, outcome in ctx.background_preflight.items()
            },
            "grouped_alignment": ctx.alignment.to_dict(),
            "grouped_time_domain_bunch_factor": bunch_factor,
            "grouped_time_domain_lifetime_corrected": bool(lifetime_corrected),
            "count_fit_raw_windows": tuple(
                (ctx.first_good + i * bunch_factor, ctx.first_good + (i + 1) * bunch_factor - 1)
                for i in np.flatnonzero(mask)
            ),
        }
    )
    return GroupedTimeDomainGroup(
        group_id=int(group_id),
        group_name=str(group_name),
        time=np.asarray(time[mask], dtype=float).copy(),
        counts=np.asarray(out_counts[mask], dtype=float).copy(),
        error=np.asarray(errors[mask], dtype=float).copy(),
        detector_counts=detector_counts[:, mask].copy(),
        deadtime=ctx.prepared.outcome,
        detector_indices=tuple(indices),
        source_run_number=int(run.run_number),
        metadata=metadata,
    )
