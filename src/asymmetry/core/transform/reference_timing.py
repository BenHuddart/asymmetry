"""Source-bound timing requests and native reference observation alignment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.t0 import (
    acquired_t0_sources,
    detector_t0_overrides,
    effective_detector_t0_bins,
    effective_t0_sources,
    require_selected_timing,
    resolve_selected_timing,
    selected_t0_reference,
    source_is_pulsed,
)

if TYPE_CHECKING:
    from asymmetry.core.transform.fb_counts import AlignmentCrop


def reference_timing_request(grouping):
    """Copy authored source intent without validating an advisory save/refresh boundary."""
    policy = grouping.get("t0_policy")
    if policy is not None:
        return {"t0_policy": deepcopy(policy)}
    if "effective_detector_t0_bins" in grouping:
        return {"effective_detector_t0_bins": deepcopy(grouping["effective_detector_t0_bins"])}
    return {"t0_policy": {"mode": "from_file"}}


@dataclass(frozen=True)
class ResolvedReferenceTiming:
    """Worker-prepared anchors reused only under their owner's source/request identity."""

    bins: tuple[int, ...]
    sources: tuple[str, ...]
    strategy: str

    @classmethod
    def from_grouping(cls, histograms, grouping):
        return cls(
            tuple(effective_detector_t0_bins(histograms, grouping)),
            tuple(effective_t0_sources(grouping, len(histograms))),
            grouping.get("t0_search_strategy", ""),
        )

    def grouping(self):
        """Materialise the existing resolved policy shape for alignment."""
        return {
            "effective_detector_t0_bins": list(self.bins),
            "effective_t0_sources": list(self.sources),
            "t0_search_strategy": self.strategy,
        }


@dataclass(frozen=True)
class ReferenceTiming:
    """Own timing intent, or an explicitly supplied source-local absolute table.

    Policy outcomes are recomputed from the selected acquisition; an absolute
    table without a policy remains an authored table, never a sample offset.
    """

    mode: str = "from_file"
    offset_bins: int = 0
    absolute_bins: tuple[int, ...] | None = None
    file_sources: tuple[str, ...] = ()
    pulsed: bool | None = None

    def __post_init__(self):
        if self.mode not in {"from_file", "manual", "auto_detect"}:
            raise ValueError("Reference timing requires a known timing policy")
        if isinstance(self.offset_bins, bool) or not isinstance(self.offset_bins, Integral):
            raise ValueError("Reference Manual timing requires an integer source-relative offset")
        if self.mode != "manual" and self.offset_bins != 0:
            raise ValueError("Only Reference Manual timing carries an offset")
        if self.pulsed is not None and not isinstance(self.pulsed, bool):
            raise ValueError("Reference timing source must be pulsed, continuous, or unknown")
        if self.absolute_bins is not None:
            if self.mode != "from_file":
                raise ValueError("Reference timing requires a policy or an absolute table")
            bins = detector_t0_overrides(
                {"effective_detector_t0_bins": self.absolute_bins}, len(self.absolute_bins)
            )
            object.__setattr__(self, "absolute_bins", tuple(bins))
        object.__setattr__(self, "file_sources", tuple(self.file_sources))

    @classmethod
    def from_grouping(cls, histograms, grouping, metadata):
        """Capture only timing facts and intent, without searching count arrays."""
        request = reference_timing_request(grouping)
        policy = request.get("t0_policy")
        mode, offset, absolute = "from_file", 0, None
        if policy is None:
            bins = detector_t0_overrides(request, len(histograms))
            absolute = tuple(bins) if bins is not None else None
        else:
            if not isinstance(policy, dict) or "mode" not in policy:
                raise ValueError("Reference timing requires an explicit policy mode")
            mode = policy["mode"]
            if mode == "manual":
                offset = policy.get("offset_bins")
        return cls(
            mode,
            offset,
            absolute,
            tuple(acquired_t0_sources(grouping, len(histograms))),
            source_is_pulsed(metadata),
        )

    def request(self) -> dict:
        """Export authored intent for source-bound project replay."""
        if self.absolute_bins is not None:
            return {"effective_detector_t0_bins": list(self.absolute_bins)}
        return {
            "t0_policy": {
                "mode": self.mode,
                **({"offset_bins": self.offset_bins} if self.mode == "manual" else {}),
            }
        }

    def search_indices(self, indices):
        """Raw channels whose complete count records determine effective anchors."""
        if self.absolute_bins is not None:
            return ()
        return tuple(
            i
            for i in indices
            if self.mode == "auto_detect"
            or (self.file_sources and self.file_sources[i] == "missing")
        )

    def resolve(self, histograms, groups, *, cancel_check=None):
        """Use the existing timing policy evaluator for the consumed channels."""
        grouping = {}
        if self.file_sources:
            grouping["detector_t0_sources"] = list(self.file_sources)
        if self.absolute_bins is not None:
            grouping["effective_detector_t0_bins"] = list(self.absolute_bins)
        else:
            indices = sorted({i for group in groups for i in group})
            resolve_selected_timing(
                histograms,
                grouping,
                mode=self.mode,
                offset_bins=self.offset_bins,
                pulsed=self.pulsed,
                selected_indices=indices,
                cancel_check=cancel_check,
            )
        require_selected_timing(grouping, sorted({i for g in groups for i in g}), len(histograms))
        return grouping


def reference_alignment(
    histograms: list[Histogram],
    timing: ReferenceTiming,
    sample: list[Histogram],
    sample_grouping: dict,
    crop: AlignmentCrop,
    *groups: list[int],
    timing_grouping: dict | None = None,
    cancel_check=None,
) -> AlignmentCrop:
    """Map elapsed integer bins, retaining each reference residual against sample t0.

    One ULP admits scalar float64 representation rounding. Fractional t0
    differences are recorded, not applied as interpolation or extra support.
    """
    from asymmetry.core.transform.grouping import (
        ChannelMembership,
        require_compatible_bin_width,
        resolve_alignment,
    )

    indices = sorted({i for group in groups for i in group})
    ChannelMembership(tuple(i + 1 for i in indices)).indices(len(histograms))
    width = sample[indices[0]].bin_width
    for i in indices:
        current = histograms[i].bin_width
        require_compatible_bin_width(width, current)
    grouping = (
        timing.resolve(histograms, groups, cancel_check=cancel_check)
        if timing_grouping is None
        else timing_grouping
    )
    require_selected_timing(grouping, indices, len(histograms))
    alignment = resolve_alignment(
        histograms,
        *groups,
        detector_t0_bins=effective_detector_t0_bins(histograms, grouping),
        common_t0_bin=crop.common_t0_bin + crop.first_bin,
        first_bin=crop.first_bin,
        sample_length=crop.length,
    )
    anchor = selected_t0_reference(sample, sample_grouping, selected_indices=indices)
    ref_anchor = selected_t0_reference(histograms, grouping, selected_indices=indices)

    def fraction(hist):
        return hist.t0_time_us_effective - (hist.t0_bin + 0.5) * hist.bin_width

    difference = fraction(histograms[ref_anchor]) - fraction(sample[anchor])
    sources = effective_t0_sources(grouping, len(histograms))
    return replace(
        alignment,
        timing_residuals_us=tuple(
            (i, value + difference) for i, value in alignment.timing_residuals_us
        ),
        timing_sources=tuple((i, sources[i]) for i in indices),
        timing_strategy=grouping.get("t0_search_strategy", ""),
    )
