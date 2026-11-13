"""Resolve acquired members and project scoped detector corrections onto one grid."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.corrections import (
    CorrectionOutcome,
    CorrectionStatus,
    reference_observation_metadata,
)
from asymmetry.core.transform.counting_noise import combine_noise
from asymmetry.core.transform.deadtime import DeadtimeDomain
from asymmetry.core.transform.fb_counts import GroupCounts, crop_to_good_support
from asymmetry.core.transform.grouping import grouping_balances, resolve_forward_backward_layout
from asymmetry.core.transform.t0 import (
    effective_detector_t0_bins,
    effective_t0_sources,
    resolve_selected_timing,
    selected_t0_reference,
)

if TYPE_CHECKING:
    from asymmetry.core.transform.scoped_counts import ScopedCounts

CONSTITUENT_PREPARATION_POLICY = "acquired_correct_align_v1"
ACQUIRED_RUN_PREPARATION_POLICY = "own_acquisition_correct_align_weight_v1"


@dataclass(frozen=True)
class AcquiredCounts:
    """One acquisition's raw buffers, live request, and offset in the selected layout."""

    histograms: list[Histogram]
    grouping: dict
    metadata: dict | None
    period_number: int | None = None
    coefficient: float = 1.0
    arithmetic: bool = False
    input_number: int | None = None


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
        from asymmetry.core.transform.background import resolve_background_mode

        if (
            self.grouping.get("background_correction")
            and resolve_background_mode(self.grouping) == "reference_run"
        ):
            crop, window = crop_to_good_support(layout.alignment, layout.window)
            layout = replace(
                layout,
                alignment=crop,
                window=window,
                t0_time_us=layout.t0_time_us
                - (crop.first_bin - layout.alignment.first_bin) * layout.bin_width_us,
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
        prepared = PreparedConstituents.prepare(
            self, inputs, domains, use_deadtime=use_deadtime, mode=mode, cancel_check=cancel_check
        )
        return (
            prepared.aligned(list(self.pair[0]), crop),
            prepared.aligned(list(self.pair[1]), crop),
            prepared.outcome,
        )


@dataclass(frozen=True)
class PreparedConstituents:
    """Scoped original preparations projected with signed coefficients only afterwards."""

    observation: ConstituentCounts
    sources: tuple[ScopedCounts, ...]
    outcome: CorrectionOutcome

    @property
    def histograms(self):
        return self.observation.histograms

    def require_source(self):
        for source in self.sources:
            source.require_source()

    @classmethod
    def prepare(cls, observation, inputs, domains, *, use_deadtime, mode=None, cancel_check=None):
        from asymmetry.core.transform.scoped_counts import ScopedCounts, require_independent_noise

        for member, domain in zip(observation.members, domains, strict=True):
            require_independent_noise(
                member.histograms,
                domain,
                raw_sources=member.arithmetic or len(observation.members) > 1,
            )
        sources = []
        for (histograms, grouping), domain in zip(inputs, domains, strict=True):
            if cancel_check is not None:
                cancel_check()
            sources.append(
                ScopedCounts.prepare(
                    histograms,
                    grouping,
                    domain,
                    use_deadtime=use_deadtime,
                    mode=mode,
                    allow_incomplete=True,
                )
            )
        outcomes = [source.outcome for source in sources]
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
                        {
                            "source": acquired_source_identity(member),
                            "data_period_number": member.period_number,
                            "coefficient": member.coefficient,
                            **item.to_dict(),
                        }
                        for member, item in zip(observation.members, outcomes, strict=True)
                    ]
                },
            )
        return cls(observation, tuple(sources), outcome)

    def aligned(self, indices, alignment, *, retain_variance=False):
        groups, offset = [], 0
        for member, source in zip(self.observation.members, self.sources, strict=True):
            size = len(member.histograms)
            selected = [i - offset for i in indices if offset <= i < offset + size]
            if selected:

                def local(values):
                    return tuple(
                        (i - offset, value) for i, value in values if offset <= i < offset + size
                    )

                crop = replace(
                    alignment,
                    detector_offsets=local(alignment.detector_offsets),
                    timing_residuals_us=local(alignment.timing_residuals_us),
                    timing_sources=local(alignment.timing_sources),
                )
                group = source.aligned(
                    selected,
                    crop,
                    retain_variance=retain_variance or len(self.sources) > 1 or member.arithmetic,
                )
                c = member.coefficient
                groups.append(
                    group
                    if c == 1
                    else GroupCounts(
                        c * group.counts,
                        c * c * group.independent_variance,
                        noise=group.noise.scale(np.full(group.counts.size, c))
                        if group.noise is not None
                        else None,
                    )
                )
            offset += size
        return GroupCounts(
            np.sum([g.counts for g in groups], axis=0),
            np.sum([g.independent_variance for g in groups], axis=0)
            if any(g.variance is not None for g in groups)
            else None,
            noise=combine_noise(*(g.noise for g in groups))
            if all(g.noise is not None for g in groups)
            else None,
        )


def resolve_group_observation(run, group_specs, *, cancel_check=None):
    """Expand arbitrary requested groups onto the same retained acquisition owner."""
    selected = sorted({i for _, indices in group_specs for i in indices})
    observation = resolve_constituent_counts(
        run.histograms,
        run.grouping,
        (tuple(selected), ()),
        reference_observation_metadata(run),
        cancel_check=cancel_check,
    )
    offsets = np.cumsum([0, *(len(m.histograms) for m in observation.members[:-1])])
    specs = [
        (gid, [int(offset) + i for offset in offsets for i in indices])
        for gid, indices in group_specs
    ]
    grouping = dict(observation.grouping)
    grouping["groups"] = {gid: [i + 1 for i in indices] for gid, indices in specs}
    grouping["forward_group"] = run.grouping.get("forward_group", 1)
    grouping["backward_group"] = run.grouping.get("backward_group", 2)
    for key in ("period_inputs", "constituent_coefficients", "period_histograms", "period_reduced"):
        grouping.pop(key, None)
    return observation, replace(run, histograms=observation.histograms, grouping=grouping), specs


def resolve_constituent_counts(histograms, grouping, pair, metadata=None, *, cancel_check=None):
    """Resolve selected originals against current intent; Ignore never enters timing work."""
    originals = grouping.get("period_inputs")
    if originals is None:
        return ConstituentCounts(
            (AcquiredCounts(histograms, grouping, metadata),), histograms, grouping, pair
        )

    target = grouping.get("period_mode", "red")
    members = []
    for number, original, coefficient in active_acquisitions(grouping):
        if cancel_check is not None:
            cancel_check()
        if "period_inputs" in original.grouping or not original.histograms:
            raise ValueError(
                "Select original acquired periods; nested inspection counts cannot supply acquired inputs"
            )
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
                selected_indices=sorted(sum(pair, ())),
                cancel_check=cancel_check,
            )
        members.append(
            AcquiredCounts(
                original.histograms,
                effective,
                reference_observation_metadata(original),
                original.metadata.get(
                    "data_period_number",
                    original.metadata.get(
                        "period_number", 1 if "constituent_coefficients" in grouping else number
                    ),
                ),
                coefficient,
                True,
                number,
            )
        )
    if not members:
        raise ValueError(f"Mapped {target} set has no acquired constituents")
    return pool_constituents(members, grouping, pair)


def active_acquisitions(grouping):
    """Select nonzero acquired inputs without resolving clocks or reading count buffers."""
    target = grouping.get("period_mode", "red")
    if target not in {"red", "green", "green_plus_red"}:
        raise ValueError("Grouped counts require Red or Green; use F/B asymmetry for G−R.")
    targets = {"red", "green"} if target == "green_plus_red" else {target}
    originals = grouping["period_inputs"]
    weights = grouping.get("constituent_coefficients", (1.0,) * len(originals))
    return [
        (number, original, coefficient)
        for number, (original, coefficient) in enumerate(zip(originals, weights, strict=True), 1)
        if coefficient != 0
        and grouping["period_mapping"].get(str(number), grouping["period_mapping"].get(number))
        in targets
    ]


def single_acquisition_model_source(run):
    """Resolve a unit singleton's own facts; multiple or weighted sources need another model."""
    if "constituent_coefficients" not in run.grouping:
        return run
    if run.grouping.get("period_mode") == "green_minus_red":
        return None
    selected = active_acquisitions(run.grouping)
    if len(selected) != 1 or selected[0][2] != 1:
        return None
    original = selected[0][1]
    return replace(original, grouping=grouping_for_acquisition(original, run.grouping))


def acquired_source_identity(member):
    """Keep acquisition identity in correction/fit records without copying file log arrays."""
    metadata = member.metadata or {}
    return {
        "source_file": metadata.get("source_file", ""),
        "run_number": metadata.get("source_run_number", metadata.get("run_number")),
    }


def acquired_source_descriptors(observation):
    """Map flat detector ordinals to source/channel coordinates without storing arrays."""
    descriptors, offset = [], 0
    for member in observation.members:
        metadata = acquired_source_identity(member)
        size = len(member.histograms)
        descriptors.append(
            (
                offset,
                size,
                metadata.get("source_file", ""),
                metadata["run_number"],
                member.input_number,
                member.coefficient,
                (member.metadata or {}).get("data_period_number"),
                (member.metadata or {}).get("dae_period_number"),
            )
        )
        offset += size
    return tuple(descriptors)


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
    if ("period_inputs" in grouping and "constituent_coefficients" not in grouping) or (
        "first_good_bin" not in effective and "last_good_bin" not in effective
    ):
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


def acquisition_grid(histograms):
    """Identify each acquired clock and good support without reading count values."""
    return tuple(
        (h.t0_bin, h.t0_time_us, h.bin_width, h.n_bins, h.good_bin_start, h.good_bin_end)
        for h in histograms
    )


def held_period_grids(grouping):
    """Describe held acquisition support for ordinary inputs without retained originals."""
    return tuple(acquisition_grid(period) for period in grouping.get("period_histograms", ()))


def constituent_metadata(grouping: dict, *, active_only: bool = False) -> dict:
    """Identify acquired facts and preparation semantics without scanning count buffers."""
    from asymmetry.core.data.correspondence import declared_period_labels, detector_roles
    from asymmetry.core.transform.deadtime import file_deadtime_table

    active = (
        {number for number, _, _ in active_acquisitions(grouping)}
        if active_only
        else set(range(1, len(grouping["period_inputs"]) + 1))
    )
    return {
        "policy": CONSTITUENT_PREPARATION_POLICY,
        "mapping": {
            key: value for key, value in grouping["period_mapping"].items() if int(key) in active
        },
        **(
            {
                "coefficients": tuple(
                    c for i, c in enumerate(grouping["constituent_coefficients"], 1) if i in active
                ),
                "run_preparation_policy": ACQUIRED_RUN_PREPARATION_POLICY,
            }
            if "constituent_coefficients" in grouping
            else {}
        ),
        "selected_set": grouping.get("period_mode", "red"),
        "inputs": [
            {
                "detector_roles": detector_roles(run),
                "declared_period_labels": declared_period_labels(run),
                "source_file": run.source_file,
                "content_identity": run.source_binding.observed.to_dict()
                if run.source_binding.observed is not None
                else None,
                "run_number": run.metadata.get("source_run_number", run.run_number),
                "period_number": number,
                "data_period_number": run.metadata.get("data_period_number"),
                "dae_period_number": run.metadata.get("dae_period_number"),
                "grid": acquisition_grid(run.histograms),
                "good_frames": run.grouping.get("good_frames"),
                "deadtime_file_us": file_deadtime_table(run.grouping),
                "t0_source": run.grouping.get("t0_source"),
                "detector_t0_sources": run.grouping.get("detector_t0_sources"),
            }
            for number, run in enumerate(grouping["period_inputs"], 1)
            if number in active
        ],
    }
