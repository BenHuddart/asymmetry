"""Check background availability for count domains without inventing an F/B pair."""

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.background import apply_grouped_background_correction
from asymmetry.core.transform.corrections import CorrectionOutcomes
from asymmetry.core.transform.grouping import align_forward_backward_counts
from asymmetry.core.transform.t0 import effective_detector_t0_bins


def require_count_background(
    histograms: list[Histogram],
    grouping: dict,
    group_specs: list[tuple[int, list[int]]],
    metadata: dict,
) -> None:
    """Check the requested background on count groups, which may be the same group.

    This operation checks correction availability only; it never forms a ratio
    or assumes that the two selections are independent observations.
    """
    forward, backward, crop = align_forward_backward_counts(
        histograms,
        group_specs[0][1],
        group_specs[-1][1],
        effective_detector_t0_bins(histograms, grouping),
    )
    background = apply_grouped_background_correction(
        forward,
        backward,
        grouping=grouping,
        t0_bin=crop.common_t0_bin,
        acquisition_start_bin=crop.first_bin,
        bin_width_us=histograms[group_specs[0][1][0]].bin_width,
        last_good_bin=crop.window(grouping).last,
        metadata=metadata,
    )
    CorrectionOutcomes(background=background.outcome).require_complete()
