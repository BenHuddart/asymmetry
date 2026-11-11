"""Raw loading retains editable requests whose independent quick-look is unavailable."""

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.data.ratio_validity import RatioValidity
from asymmetry.core.transform.asymmetry import ratio_with_count_errors
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow, linear_combination_variance
from asymmetry.core.transform.grouping import (
    independent_pair_indices,
    resolve_forward_backward_layout,
)
from asymmetry.core.transform.t0 import (
    common_t0_time_us,
    effective_detector_t0_bins,
    good_window_for_groups,
    resolve_selected_timing,
    selected_t0_reference,
)
from asymmetry.core.transform.units import PERCENT_PER_FRACTION


@dataclass(frozen=True)
class QuickAsymmetry:
    """A loader-derived pair or an explicit unavailable reason alongside raw data."""

    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    time: NDArray[np.float64]
    common_t0: int
    alignment: AlignmentCrop | None
    window: BinWindow | None
    unavailable_reason: str | None = None
    t0_time_us: float | None = None
    timing: dict = field(default_factory=dict)
    ratio_validity: RatioValidity | None = None

    @property
    def file_grouping(self) -> dict:
        """Canonical uncropped bounds, including an explicitly unavailable window."""
        start = self.alignment.first_bin if self.alignment is not None else 0
        first, last = (
            (start + 1, start)
            if self.window is None
            else (self.window.first + start, self.window.last + start)
        )
        return {
            **self.timing,
            **({"t0_time_us": self.t0_time_us} if self.t0_time_us is not None else {}),
            "t0_bin": self.common_t0,
            "first_good_bin": first,
            "last_good_bin": last,
            "t_good_offset": first - self.common_t0,
        }

    @property
    def metadata(self) -> dict:
        """Publish alignment only for a valid pair, otherwise its repair diagnostic."""
        if self.unavailable_reason is not None:
            return {"correction_error": self.unavailable_reason}
        return {"grouped_alignment": self.alignment.to_dict()}


def quick_asymmetry(
    histograms: list[Histogram], grouping: dict, *, metadata: dict | None = None, cancel_check=None
) -> QuickAsymmetry:
    """Validate the default pair before exposing independent-count uncertainties."""
    alignment = None
    common_t0 = max((hist.t0_bin for hist in histograms), default=0)
    t0_time_us = None
    timing = dict(grouping)
    try:
        from asymmetry.core.transform.constituent_counts import (
            PreparedConstituents,
            resolve_constituent_counts,
        )

        pair = independent_pair_indices(grouping, n_histograms=len(histograms))
        observation = resolve_constituent_counts(
            histograms, grouping, pair, metadata, cancel_check=cancel_check
        )
        histograms, timing = observation.histograms, dict(observation.grouping)
        timing["good_window_policy"] = {"mode": "from_file"}
        if "effective_detector_t0_bins" not in timing:
            policy = timing.get("t0_policy", {"mode": "from_file"})
            resolve_selected_timing(
                histograms,
                timing,
                metadata,
                mode=policy["mode"],
                offset_bins=policy.get("offset_bins", 0),
            )
        indices = sorted(sum(observation.pair, ()))
        common_t0 = max(effective_detector_t0_bins(histograms, timing)[i] for i in indices)
        reference = selected_t0_reference(histograms, timing, selected_indices=indices)
        if histograms[reference].t0_time_us is not None:
            t0_time_us = common_t0_time_us(histograms, timing, common_t0, selected_indices=indices)
        # File-good quick looks derive bounds from acquired histograms, never
        # a scalar window left by another selected pair or an unavailable one.
        file_grouping = {
            key: value
            for key, value in timing.items()
            if key not in {"first_good_bin", "last_good_bin", "good_window_policy"}
        }
        layout = resolve_forward_backward_layout(histograms, file_grouping)
        alignment = observation.acquired_alignment(layout.alignment)
        common_t0 = layout.common_t0_bin + alignment.first_bin
        first, last = good_window_for_groups(
            [i for i, _ in alignment.detector_offsets],
            effective_detector_t0_bins(histograms, timing),
            [hist.good_bin_start for hist in histograms],
            [
                hist.good_bin_end if hist.good_bin_end >= 0 else hist.n_bins - 1
                for hist in histograms
            ],
            common_t0_bin=layout.common_t0_bin + alignment.first_bin,
            n_bins=alignment.first_bin + alignment.length,
        )
        window = alignment.window({"first_good_bin": first, "last_good_bin": last})
        from asymmetry.core.transform.fb_deadtime import fb_deadtime_domain

        domain = fb_deadtime_domain(
            replace(layout, window=window),
            timing,
            *observation.pair,
            use_background=False,
            facility=None,
            metadata=metadata,
        )
        inputs, domains = observation.participants(domain)
        prepared = PreparedConstituents.prepare(
            observation, inputs, domains, use_deadtime=False, mode="off", cancel_check=cancel_check
        )
        forward, backward = (
            prepared.aligned(list(indices), alignment, retain_variance=True)
            for indices in observation.pair
        )
        edges = np.arange(alignment.length + 1)
        vf = linear_combination_variance(forward, edges)
        vb = linear_combination_variance(backward, edges)
    except ValueError as exc:
        return QuickAsymmetry(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            common_t0,
            alignment,
            None,
            str(exc),
            t0_time_us=t0_time_us,
            timing={
                k: v
                for k, v in timing.items()
                if "period_inputs" not in grouping
                and k.startswith(("effective_", "t0_search", "t0_policy"))
            },
        )
    asymmetry, error, validity = ratio_with_count_errors(
        forward.counts,
        backward.counts,
        np.sqrt(vf),
        np.sqrt(vb),
        alpha=layout.alpha,
    )
    return QuickAsymmetry(
        asymmetry * PERCENT_PER_FRACTION,
        error * PERCENT_PER_FRACTION,
        (np.arange(alignment.length) + 0.5) * layout.bin_width_us - layout.t0_time_us,
        common_t0,
        alignment,
        window,
        ratio_validity=validity,
        t0_time_us=t0_time_us,
        timing={
            k: v
            for k, v in timing.items()
            if "period_inputs" not in grouping
            and k.startswith(("effective_", "t0_search", "t0_policy"))
        },
    )
