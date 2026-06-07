"""Data transformations: asymmetry calculation, grouping, rebinning."""

from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
    estimate_alpha,
)
from asymmetry.core.transform.background import (
    BackgroundCorrectionResult,
    apply_grouped_background_correction,
    supports_background_correction,
)
from asymmetry.core.transform.deadtime import (
    apply_deadtime_correction,
    calibrate_deadtime_from_histograms,
    estimate_deadtime_from_histograms,
    has_file_deadtime,
    has_resolved_deadtime,
    parse_deadtime_calibration_text,
    prepare_histograms_with_deadtime,
)
from asymmetry.core.transform.grouping import (
    GroupedForwardBackward,
    apply_grouping,
    apply_grouping_aligned,
    common_t0_for_groups,
    effective_grouping,
    group_forward_backward,
    resolve_group_indices,
)
from asymmetry.core.transform.integral import (
    FieldScan,
    FieldScanPoint,
    build_field_scan,
    differentiate_scan,
    integrate_asymmetry,
    integrate_curve,
    integrate_run,
)
from asymmetry.core.transform.rebin import rebin

__all__ = [
    "compute_asymmetry",
    "compute_asymmetry_with_count_errors",
    "estimate_alpha",
    "BackgroundCorrectionResult",
    "apply_grouped_background_correction",
    "supports_background_correction",
    "apply_deadtime_correction",
    "calibrate_deadtime_from_histograms",
    "estimate_deadtime_from_histograms",
    "has_file_deadtime",
    "has_resolved_deadtime",
    "parse_deadtime_calibration_text",
    "prepare_histograms_with_deadtime",
    "apply_grouping",
    "apply_grouping_aligned",
    "common_t0_for_groups",
    "resolve_group_indices",
    "effective_grouping",
    "group_forward_backward",
    "GroupedForwardBackward",
    "integrate_asymmetry",
    "integrate_curve",
    "integrate_run",
    "build_field_scan",
    "differentiate_scan",
    "FieldScan",
    "FieldScanPoint",
    "rebin",
]
