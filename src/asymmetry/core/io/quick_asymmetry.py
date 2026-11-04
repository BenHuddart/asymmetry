"""Raw loading retains editable requests whose independent quick-look is unavailable."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.t0 import (
    effective_detector_t0_bins,
    good_window_for_groups,
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

    @property
    def file_grouping(self) -> dict:
        """Canonical uncropped bounds, including an explicitly unavailable window."""
        start = self.alignment.first_bin if self.alignment is not None else 0
        first, last = (
            (1, 0) if self.window is None else (self.window.first + start, self.window.last + start)
        )
        return {
            **({"t0_time_us": self.t0_time_us} if self.t0_time_us is not None else {}),
            "t0_bin": self.common_t0,
            "first_good_bin": first,
            "last_good_bin": last,
            "t_good_offset": max(0, first - self.common_t0),
        }

    @property
    def metadata(self) -> dict:
        """Publish alignment only for a valid pair, otherwise its repair diagnostic."""
        if self.unavailable_reason is not None:
            return {"correction_error": self.unavailable_reason}
        return {"grouped_alignment": self.alignment.to_dict()}


def quick_asymmetry(histograms: list[Histogram], grouping: dict) -> QuickAsymmetry:
    """Validate the default pair before exposing independent-count uncertainties."""
    try:
        grouped = group_forward_backward(histograms, grouping)
        alignment = grouped.counts.alignment
        first, last = good_window_for_groups(
            [i for i, _ in alignment.detector_offsets],
            effective_detector_t0_bins(histograms, grouping),
            [hist.good_bin_start for hist in histograms],
            [
                hist.good_bin_end if hist.good_bin_end >= 0 else hist.n_bins - 1
                for hist in histograms
            ],
            common_t0_bin=grouped.common_t0 + alignment.first_bin,
            n_bins=alignment.first_bin + alignment.length,
        )
        window = alignment.window({"first_good_bin": first, "last_good_bin": last})
    except ValueError as exc:
        return QuickAsymmetry(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            max((hist.t0_bin for hist in histograms), default=0),
            None,
            None,
            str(exc),
        )
    asymmetry, error = compute_asymmetry(grouped.forward, grouped.backward, alpha=grouped.alpha)
    return QuickAsymmetry(
        asymmetry * PERCENT_PER_FRACTION,
        error * PERCENT_PER_FRACTION,
        (np.arange(grouped.forward.size) + 0.5) * grouped.counts.bin_width_us
        - grouped.counts.t0_time_us,
        grouped.common_t0 + grouped.counts.alignment.first_bin,
        alignment,
        window,
        t0_time_us=(
            grouped.counts.t0_time_us + alignment.first_bin * grouped.counts.bin_width_us
            if histograms[
                selected_t0_reference(
                    histograms,
                    grouping,
                    selected_indices=[i for i, _ in alignment.detector_offsets],
                )
            ].t0_time_us
            is not None
            else None
        ),
    )
