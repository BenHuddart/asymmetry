"""Resolve acquired members and project scoped detector corrections onto one grid."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.corrections import CorrectionStatus, reference_observation_metadata
from asymmetry.core.transform.deadtime import DeadtimeDomain
from asymmetry.core.transform.fb_counts import GroupCounts
from asymmetry.core.transform.fb_deadtime import deadtime_corrected_pair
from asymmetry.core.transform.grouping import grouping_balances, resolve_forward_backward_layout
from asymmetry.core.transform.t0 import (
    effective_detector_t0_bins,
    effective_t0_sources,
    resolve_selected_timing,
    selected_t0_reference,
)

CONSTITUENT_PREPARATION_POLICY = "acquired_correct_align_v1"


@dataclass(frozen=True)
class AcquiredCounts:
    """One acquisition's raw buffers, live request, and offset in the selected layout."""

    histograms: list[Histogram]
    grouping: dict
    metadata: dict | None
    period_number: int | None = None


@dataclass(frozen=True)
class ConstituentCounts:
    """Borrowed acquisition buffers with one common observed-bin layout."""

    members: tuple[AcquiredCounts, ...]
    histograms: list[Histogram]
    grouping: dict
    pair: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def layout(self):
        """Resolve common support using the ordinary detector alignment owner."""
        layout = resolve_forward_backward_layout(
            self.histograms, self.grouping, resolved_indices=self.pair
        )
        return replace(layout, alignment=self.acquired_alignment(layout.alignment))

    def acquired_alignment(self, alignment):
        """Attach original acquired period/channel ordinals to flattened detector offsets."""
        if self.members[0].period_number is None:
            return alignment
        identities = []
        offset = 0
        for member in self.members:
            identities.extend(
                (offset + i, member.period_number, i)
                for i in range(len(member.histograms))
                if offset + i in dict(alignment.detector_offsets)
            )
            offset += len(member.histograms)
        return replace(alignment, acquired_detectors=tuple(identities))

    @property
    def reference_sample(self):
        """Acquisition containing the deterministic elapsed-time reference channel."""
        index = selected_t0_reference(
            self.histograms, self.grouping, selected_indices=sorted(sum(self.pair, ()))
        )
        offset = 0
        for member in self.members:
            offset += len(member.histograms)
            if index < offset:
                return member
        raise ValueError("Selected timing reference must belong to an acquired constituent")

    def participants(self, domain):
        """Split flat detector ordinals into their original correction inputs/domains."""
        inputs, domains = [], []
        offset = 0
        for member in self.members:
            size = len(member.histograms)
            inputs.append((member.histograms, member.grouping))
            domains.append(
                DeadtimeDomain(
                    tuple(
                        (i - offset, windows)
                        for i, windows in domain.observations
                        if offset <= i < offset + size
                    )
                )
            )
            offset += size
        return inputs, domains

    def corrected_pair(self, inputs, domains, crop, *, use_deadtime, mode, cancel_check=None):
        """Correct each acquired detector once, then add independent aligned counts."""
        forward, backward, outcomes = [], [], []
        offset = 0
        for member, (histograms, grouping), domain in zip(
            self.members, inputs, domains, strict=True
        ):
            if cancel_check is not None:
                cancel_check()
            size = len(histograms)

            def local(values):
                return tuple(
                    (i - offset, value) for i, value in values if offset <= i < offset + size
                )

            alignment = replace(
                crop,
                detector_offsets=local(crop.detector_offsets),
                timing_residuals_us=local(crop.timing_residuals_us),
                timing_sources=local(crop.timing_sources),
            )
            pair = tuple(
                tuple(i - offset for i in indices if offset <= i < offset + size)
                for indices in self.pair
            )
            f, b, outcome = deadtime_corrected_pair(
                histograms, grouping, *pair, alignment, domain, use_deadtime=use_deadtime, mode=mode
            )
            forward.append(f)
            backward.append(b)
            outcomes.append(outcome)
            offset += size
        outcome = next(
            (item for item in outcomes if not item.usable),
            next(
                (item for item in outcomes if item.status == CorrectionStatus.SKIPPED), outcomes[0]
            ),
        )
        if len(outcomes) > 1 and all(item.status == CorrectionStatus.APPLIED for item in outcomes):
            outcome = replace(
                outcome,
                effective={
                    "constituents": [
                        {"data_period_number": member.period_number, **item.to_dict()}
                        for member, item in zip(self.members, outcomes, strict=True)
                    ]
                },
            )

        def pooled(groups):
            if len(groups) == 1:
                return groups[0]
            return GroupCounts(
                np.sum([g.counts for g in groups], axis=0),
                np.sum([g.independent_variance for g in groups], axis=0),
            )

        return pooled(forward), pooled(backward), outcome


def resolve_constituent_counts(histograms, grouping, pair, metadata=None, *, cancel_check=None):
    """Resolve selected originals against current intent; Ignore never enters timing work."""
    originals = grouping.get("period_inputs")
    if originals is None:
        return ConstituentCounts(
            (AcquiredCounts(histograms, grouping, metadata),), histograms, grouping, pair
        )

    target = grouping.get("period_mode", "red")
    if target not in {"red", "green"}:
        raise ValueError("Constituent preparation requires one selected mapped set")
    members = []
    for number, original in enumerate(originals, 1):
        if (
            grouping["period_mapping"].get(str(number), grouping["period_mapping"].get(number))
            != target
        ):
            continue
        if cancel_check is not None:
            cancel_check()
        effective = grouping_for_acquisition(original, grouping)
        policy = effective.get("t0_policy")
        if policy is not None:
            # A held-set search cache cannot identify an original acquisition.
            resolve_selected_timing(
                original.histograms,
                effective,
                original.metadata,
                mode=policy["mode"],
                offset_bins=policy.get("offset_bins", 0),
                resolved_indices=pair,
                cancel_check=cancel_check,
            )
        members.append(
            AcquiredCounts(
                original.histograms,
                effective,
                reference_observation_metadata(original),
                original.metadata.get("data_period_number", number),
            )
        )
    if not members:
        raise ValueError(f"Mapped {target} set has no acquired constituents")
    if len(members) == 1:
        member = members[0]
        return ConstituentCounts(tuple(members), member.histograms, member.grouping, pair)
    return pool_constituents(members, grouping, pair)


def pool_constituents(members, grouping, pair):
    """Put acquired detector buffers on one layout without changing authored membership."""
    flat = [h for member in members for h in member.histograms]
    expanded = tuple(
        tuple(
            i + sum(len(m.histograms) for m in members[:j])
            for j, member in enumerate(members)
            for i in indices
        )
        for indices in pair
    )
    effective = dict(grouping)
    effective["alpha"], effective["beta"] = grouping_balances(grouping)
    for key in ("projections", "vector_axis", "effective_t0_period"):
        effective.pop(key, None)
    effective.update(
        groups={1: [i + 1 for i in expanded[0]], 2: [i + 1 for i in expanded[1]]},
        forward_group=1,
        backward_group=2,
        excluded_detectors=[],
        effective_detector_t0_bins=[
            v for m in members for v in effective_detector_t0_bins(m.histograms, m.grouping)
        ],
        effective_t0_sources=[
            v for m in members for v in effective_t0_sources(m.grouping, len(m.histograms))
        ],
        detector_t0_bins=[h.t0_bin for h in flat],
        detector_first_good_bins=[h.good_bin_start for h in flat],
        detector_last_good_bins=[
            h.n_bins - 1 if h.good_bin_end < 0 else h.good_bin_end for h in flat
        ],
    )
    if "first_good_bin" not in effective and "last_good_bin" not in effective:
        effective.setdefault("good_window_policy", {"mode": "from_file"})
    return ConstituentCounts(tuple(members), flat, effective, expanded)


def grouping_for_acquisition(run, live: dict) -> dict:
    """Combine current authored intent with one original acquisition's file facts."""
    from asymmetry.core.io.periods import histogram_grouping_facts
    from asymmetry.core.project.profiles import _PER_RUN_FACT_KEYS, _copy_per_run_facts
    from asymmetry.core.transform.deadtime import file_deadtime_table
    from asymmetry.core.transform.reduce import correction_flags_from_grouping

    timing = histogram_grouping_facts(run.histograms, live)
    effective = {key: value for key, value in timing.items() if key not in _PER_RUN_FACT_KEYS}
    if "t0_policy" in live:
        for key in (
            "effective_detector_t0_bins",
            "effective_t0_sources",
            "effective_t0_period",
            "t0_search_strategy",
            "t0_search_spread_bins",
        ):
            effective.pop(key, None)
    _copy_per_run_facts(effective, run.grouping, run)
    if correction_flags_from_grouping(live).deadtime_mode not in {"manual", "estimate"}:
        effective["dead_time_us"] = list(file_deadtime_table(run.grouping))
    return effective


def constituent_metadata(grouping: dict) -> dict:
    """Identify acquired facts and preparation semantics without scanning count buffers."""
    from asymmetry.core.transform.deadtime import file_deadtime_table

    return {
        "policy": CONSTITUENT_PREPARATION_POLICY,
        "mapping": grouping["period_mapping"],
        "selected_set": grouping.get("period_mode", "red"),
        "inputs": [
            {
                "source_file": run.source_file,
                "run_number": run.metadata.get("source_run_number", run.run_number),
                "period_number": number,
                "data_period_number": run.metadata.get("data_period_number"),
                "dae_period_number": run.metadata.get("dae_period_number"),
                "grid": [
                    (
                        h.t0_bin,
                        h.t0_time_us,
                        h.bin_width,
                        h.n_bins,
                        h.good_bin_start,
                        h.good_bin_end,
                    )
                    for h in run.histograms
                ],
                "good_frames": run.grouping.get("good_frames"),
                "deadtime_file_us": file_deadtime_table(run.grouping),
                "t0_source": run.grouping.get("t0_source"),
                "detector_t0_sources": run.grouping.get("detector_t0_sources"),
            }
            for number, run in enumerate(grouping["period_inputs"], 1)
        ],
    }
