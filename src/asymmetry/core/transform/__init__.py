"""Data transformations: asymmetry calculation, grouping, rebinning."""

from asymmetry.core.transform.asymmetry import compute_asymmetry, estimate_alpha
from asymmetry.core.transform.background import (
    BackgroundCorrectionResult,
    apply_grouped_background_correction,
    supports_background_correction,
)
from asymmetry.core.transform.deadtime import (
    apply_deadtime_correction,
    has_file_deadtime,
    prepare_histograms_with_deadtime,
)
from asymmetry.core.transform.grouping import (
    apply_grouping,
    apply_grouping_aligned,
    common_t0_for_groups,
)
from asymmetry.core.transform.rebin import rebin

__all__ = [
    "compute_asymmetry",
    "estimate_alpha",
    "BackgroundCorrectionResult",
    "apply_grouped_background_correction",
    "supports_background_correction",
    "apply_deadtime_correction",
    "has_file_deadtime",
    "prepare_histograms_with_deadtime",
    "apply_grouping",
    "apply_grouping_aligned",
    "common_t0_for_groups",
    "rebin",
]
