"""Request-bound sample/reference count preparation for spectral consumers."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.group_background import (
    _apply_group_background_correction,
    group_background_payload,
)
from asymmetry.core.fourier.group_time import group_time_grid, time_range_mask
from asymmetry.core.transform.background import (
    BackgroundCorrectionResult,
    grouped_background_ranges,
    resolve_background_mode,
    resolve_facility,
    tail_background_window,
)
from asymmetry.core.transform.constituent_counts import (
    PreparedConstituents,
    active_acquisitions,
    resolve_group_observation,
)
from asymmetry.core.transform.corrections import (
    CorrectionOutcomes,
    reference_observation_metadata,
    result_tail_skip,
)
from asymmetry.core.transform.deadtime import DeadtimeDomain, deadtime_operation_inputs
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow, crop_to_good_support
from asymmetry.core.transform.grouping import (
    common_t0_for_groups,
    effective_group_indices,
    group_reference_t0,
    resolve_alignment,
)
from asymmetry.core.transform.preparation import count_metadata_identity
from asymmetry.core.transform.reduce import correction_flags_from_grouping
from asymmetry.core.transform.reference_scale import ReferenceExposure
from asymmetry.core.transform.reference_timing import ReferenceTiming
from asymmetry.core.transform.scoped_counts import (
    ScopedCounts,
    observation_domain,
    scoped_source_digest,
)
from asymmetry.core.transform.t0 import effective_detector_t0_bins, require_selected_timing


def histogram_grid_identity(histograms: list[Histogram]) -> tuple:
    """Snapshot raw-bin timing and observed support without reading count values."""
    return tuple((h.t0_bin, h.t0_time_us, h.bin_width, h.n_bins) for h in histograms)


def preparation_identity(
    run: Run, reference: ScopedCounts | None = None, timing: ReferenceTiming | None = None
) -> tuple:
    """Bind products to their actual arrays, period and authored reduction settings."""
    from asymmetry.core.io import reference_file_identity

    if (reference is None) != (timing is None):
        raise ValueError("Prepared reference counts and timing request must be supplied together")
    from asymmetry.core.transform.constituent_counts import constituent_metadata

    retained = "period_inputs" in run.grouping
    grouping = {
        key: value
        for key, value in run.grouping.items()
        if not retained
        or key
        not in {
            "period_inputs",
            "period_histograms",
            "period_reduced",
            "period_mapping",
            "constituent_coefficients",
        }
    }
    histograms = (
        [h for _, source, _ in active_acquisitions(run.grouping) for h in source.histograms]
        if retained
        else run.histograms
    )
    if retained:
        grouping["constituents"] = constituent_metadata(run.grouping, active_only=True)
    return (
        reference_file_identity(run.grouping.get("background_run"))
        if correction_flags_from_grouping(run.grouping).use_background
        and resolve_background_mode(run.grouping) == "reference_run"
        else None,
        tuple((id(h.counts), id(h.count_variance)) for h in histograms),
        histogram_grid_identity(histograms),
        json.dumps(grouping, default=str),
        json.dumps(count_metadata_identity(run.metadata), default=str),
        run.source_file,
        scoped_source_digest(
            reference.histograms,
            DeadtimeDomain(
                tuple(
                    (i, (BinWindow(0, reference.histograms[i].n_bins - 1),))
                    for i in timing.search_indices(reference.domain.detector_indices)
                )
            ),
        )
        if reference is not None
        else None,
        histogram_grid_identity(
            [reference.histograms[i] for i in reference.domain.detector_indices]
        )
        if reference is not None
        else None,
    )


@dataclass(frozen=True)
class GroupLayout:
    """One group's existing signal grid and active background coordinate."""

    alignment: AlignmentCrop
    window: BinWindow
    grouping: dict
    background_t0: int
    reference_alignment: AlignmentCrop | None


@dataclass(frozen=True)
class GroupPreparation:
    """A declared group cohort, with atomic sample/reference deadtime outcomes."""

    identity: tuple
    use_deadtime: bool
    reference_t0_bin: int
    layouts: dict[int, GroupLayout]
    sample: ScopedCounts | PreparedConstituents
    reference: ScopedCounts | None
    reference_timing: ReferenceTiming | None
    reference_scale: float | None
    time_range: tuple | None
    backgrounds: dict[int, BackgroundCorrectionResult]

    def correction_metadata(self) -> dict:
        """Expose actual outcomes for every contributing group, including skips."""
        return {
            "correction_outcomes": {
                str(gid): CorrectionOutcomes(self.sample.outcome, result.outcome).to_dict()
                for gid, result in self.backgrounds.items()
            }
        }

    def require_request(self, run, group_id, use_deadtime, reference_t0_bin, time_range):
        """Reject stale products or a group outside the prepared cohort."""
        self.sample.require_source()
        if self.reference is not None:
            self.reference.require_source()
        if (
            self.time_range != time_range
            or self.identity != preparation_identity(run, self.reference, self.reference_timing)
            or group_id not in self.layouts
            or self.use_deadtime != use_deadtime
            or reference_t0_bin is not None
            and self.reference_t0_bin != reference_t0_bin
        ):
            raise ValueError(
                "Prepared grouped counts do not match the requested source, scope or correction settings"
            )


def prepare_group_counts(
    run,
    group_ids,
    *,
    use_deadtime=None,
    reference_t0_bin=None,
    cache=None,
    time_range=None,
    cancel_check=None,
):
    """Resolve all selected and active estimator observations before any inverse."""
    from asymmetry.core.io import resolve_background_reference

    if time_range == (None, None):
        time_range = None
    identity_run = run
    observation = None
    group_ids = list(group_ids)
    if "period_inputs" in run.grouping:
        specs = [
            (gid, effective_group_indices(run.grouping, gid, n_histograms=len(run.histograms)))
            for gid in group_ids
        ]
        observation, run, _ = resolve_group_observation(run, specs, cancel_check=cancel_check)
    grouping = run.grouping
    histograms = list(run.histograms)
    apply_deadtime = (
        bool(grouping.get("deadtime_correction", False))
        if use_deadtime is None
        else bool(use_deadtime)
    )
    common = (
        group_reference_t0(histograms, grouping)
        if reference_t0_bin is None
        else int(reference_t0_bin)
    )
    common = max(0, common)
    detector_t0 = effective_detector_t0_bins(histograms, grouping)
    background = correction_flags_from_grouping(grouping).use_background
    background_t0 = 0
    if background:
        background_t0 = (
            common_t0_for_groups(
                histograms,
                sorted(
                    {
                        i
                        for gid in group_ids
                        for i in effective_group_indices(
                            grouping, gid, n_histograms=len(histograms)
                        )
                    }
                ),
                detector_t0_bins=detector_t0,
            )
            if observation is not None
            else common_t0_for_groups(
                histograms,
                effective_group_indices(
                    grouping, grouping.get("forward_group", 1), n_histograms=len(histograms)
                ),
                effective_group_indices(
                    grouping, grouping.get("backward_group", 2), n_histograms=len(histograms)
                ),
                detector_t0_bins=detector_t0,
            )
        )
    reference = None
    if background and resolve_background_mode(grouping) == "reference_run":
        reference = resolve_background_reference(
            grouping.get("background_run"),
            sample_exposure=ReferenceExposure.from_grouping(grouping),
            cache=cache,
        )
    group_ids = list(group_ids)
    reference_timing = None
    if reference is not None:
        from asymmetry.core.transform.grouping import ChannelMembership

        selected = sorted(
            {
                i
                for gid in group_ids
                for i in effective_group_indices(
                    identity_run.grouping, gid, n_histograms=len(identity_run.histograms)
                )
            }
        )
        ChannelMembership(tuple(i + 1 for i in selected)).indices(len(reference.histograms))
        reference_timing = reference.timing.resolve(
            reference.histograms, [selected], cancel_check=cancel_check
        )
    layouts, sample_parts, reference_parts = {}, [], []
    for group_id in group_ids:
        indices = effective_group_indices(grouping, group_id, n_histograms=len(histograms))
        require_selected_timing(grouping, indices, len(histograms))
        alignment = resolve_alignment(
            histograms, indices, common_t0_bin=common, detector_t0_bins=detector_t0
        )
        t0 = alignment.common_t0_bin
        declared = grouping.get("t0_bin")
        declared = t0 if declared is None else int(declared)
        if observation is not None and "good_window_policy" in grouping:
            from asymmetry.core.project.profiles import GoodWindowPolicy, resolve_run_good_window

            first, last = resolve_run_good_window(
                GoodWindowPolicy.from_dict(grouping["good_window_policy"]),
                run,
                grouping,
                alignment=alignment,
            )
            window = alignment.window(dict(first_good_bin=first, last_good_bin=last))
        else:
            window = alignment.window(grouping, offset_bins=t0 + alignment.first_bin - declared)
        if reference is not None:
            alignment, window = crop_to_good_support(alignment, window)
            t0 = alignment.common_t0_bin
        local_grouping = dict(
            grouping, t0_bin=t0, first_good_bin=window.first, last_good_bin=window.last
        )
        time, edges = group_time_grid(histograms, local_grouping, indices, alignment, window)
        mask = time_range_mask(time, time_range)
        windows = [BinWindow(int(edges[i]), int(edges[i + 1]) - 1) for i in np.flatnonzero(mask)]
        if background:
            payload = group_background_payload(local_grouping, group_id)
            mode = resolve_background_mode(payload)
            if mode == "range":
                windows.append(
                    grouped_background_ranges(
                        payload,
                        t0_bin=t0,
                        acquisition_start_bin=background_t0 - t0,
                        length=alignment.length,
                        bin_width_us=histograms[indices[0]].bin_width,
                        facility=resolve_facility(metadata=run.metadata, grouping=payload),
                    )[0]
                )
            elif mode == "tail_fit":
                first, last = tail_background_window(
                    alignment.length, t0, window.last, first_good_bin=window.first
                )
                if first <= last:
                    windows.append(BinWindow(first, last))
        sample_parts.append((alignment, indices, windows))
        reference_alignment = None
        if reference is not None:
            sample = observation.reference_sample if observation is not None else None
            if windows:
                length = max(window.last for window in windows) + 1
                alignment = replace(alignment, length=length)
                window = BinWindow(window.first, min(window.last, length - 1))
                local_grouping["last_good_bin"] = window.last
            physical = effective_group_indices(
                identity_run.grouping, group_id, n_histograms=len(identity_run.histograms)
            )
            reference_alignment = reference.alignment(
                sample.histograms if sample is not None else histograms,
                sample.grouping if sample is not None else grouping,
                alignment,
                physical,
                timing_grouping=reference_timing,
            )
            reference_parts.append((reference_alignment, physical, windows))
        layouts[int(group_id)] = GroupLayout(
            alignment, window, local_grouping, background_t0, reference_alignment
        )
    if reference is not None:
        indices = sorted({index for _, selected, _ in sample_parts for index in selected})
        if observation is None:
            reference.require_independent(histograms, reference_observation_metadata(run), indices)
        else:
            for member in observation.members:
                reference.require_independent(
                    member.histograms,
                    member.metadata,
                    sorted(
                        {
                            i
                            for gid in group_ids
                            for i in effective_group_indices(
                                identity_run.grouping,
                                gid,
                                n_histograms=len(identity_run.histograms),
                            )
                        }
                    ),
                )
    domain = observation_domain(sample_parts)
    if observation is None:
        inputs, domains = [(histograms, grouping)], [domain]
    else:
        inputs, domains = observation.participants(domain)
    n_sources = len(inputs)
    if reference is not None:
        inputs.append((reference.histograms, reference.deadtime_grouping(grouping)))
        domains.append(observation_domain(reference_parts))
    inputs = deadtime_operation_inputs(inputs, apply_deadtime, domains=domains)
    sample = (
        ScopedCounts.prepare(*inputs[0], domains[0], use_deadtime=apply_deadtime)
        if observation is None
        else PreparedConstituents.prepare(
            observation,
            inputs[:n_sources],
            domains[:n_sources],
            use_deadtime=apply_deadtime,
            cancel_check=cancel_check,
        )
    )
    CorrectionOutcomes(deadtime=sample.outcome).require_complete()
    products = [sample]
    if reference is not None:
        products.append(ScopedCounts.prepare(*inputs[-1], domains[-1], use_deadtime=apply_deadtime))
    backgrounds, original = {}, {}
    for group_id, layout in layouts.items():
        indices = effective_group_indices(grouping, group_id, n_histograms=len(histograms))
        original[group_id] = products[0].aligned(indices, layout.alignment)
        backgrounds[group_id] = _apply_group_background_correction(
            original[group_id],
            grouping=layout.grouping,
            group_id=group_id,
            t0_bin=layout.alignment.common_t0_bin,
            background_t0_bin=layout.background_t0,
            bin_width=histograms[indices[0]].bin_width,
            metadata=run.metadata,
            source_file=run.source_file,
            reference_counts=products[1].aligned(
                effective_group_indices(
                    identity_run.grouping, group_id, n_histograms=len(identity_run.histograms)
                ),
                layout.reference_alignment,
                retain_variance=True,
            )
            if reference is not None
            else None,
            reference_scale=reference.scale if reference is not None else None,
        )
    if reference is not None:
        backgrounds = {
            gid: replace(
                result,
                outcome=replace(
                    result.outcome,
                    effective=dict(
                        result.outcome.effective,
                        reference=reference.provenance(),
                        reference_alignment=layouts[gid].reference_alignment.to_dict(),
                    ),
                ),
            )
            for gid, result in backgrounds.items()
        }
    skip = result_tail_skip({str(gid): result.outcome for gid, result in backgrounds.items()})
    if skip is not None:
        warnings.warn(skip.warning, UserWarning, stacklevel=2)
        backgrounds = {
            gid: replace(result, forward=original[gid], backward=original[gid], outcome=skip)
            for gid, result in backgrounds.items()
        }
    return GroupPreparation(
        preparation_identity(
            identity_run,
            products[1] if reference is not None else None,
            reference.timing if reference is not None else None,
        ),
        apply_deadtime,
        common,
        layouts,
        products[0],
        products[1] if reference is not None else None,
        reference.timing if reference is not None else None,
        reference.scale if reference is not None else None,
        time_range,
        backgrounds,
    )
