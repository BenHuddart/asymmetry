"""Capture count-fit residual calibration only when a complete total table exists."""

import warnings

from asymmetry.core.transform.corrections import (
    CorrectionStatus,
    correction_metadata,
    correction_warnings,
)
from asymmetry.core.transform.deadtime import DeadtimeCalibration


def record_count_deadtime_calibration(result, dataset, groups):
    """Keep a successful fit usable when coefficient availability prevents promotion."""
    result.correction_metadata = correction_metadata(groups[0].metadata)
    for message in dict.fromkeys(
        w for group in groups for w in correction_warnings(group.metadata)
    ):
        if message not in result.warnings:
            result.warnings.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)
    if not result.success or "DT0" not in result.parameters:
        return
    base = groups[0].deadtime
    missing = (
        [i + 1 for i, tau in enumerate(base.effective["tau_us"]) if tau is None]
        if base.status is CorrectionStatus.APPLIED
        else []
    )
    reason = None
    if base.status is CorrectionStatus.SKIPPED:
        reason = (
            "File deadtime correction was skipped because required coefficients are unavailable."
        )
    elif missing:
        reason = "coefficients are unknown for detector(s) " + ", ".join(map(str, missing)) + "."
    if reason is not None:
        message = f"Deadtime promotion unavailable: {reason} Fit results remain usable."
        result.warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=2)
        return
    result.deadtime_calibration = DeadtimeCalibration.from_increment(
        base,
        result.parameters["DT0"].value,
        n_histograms=len(dataset.run.histograms),
        detector_indices=sorted({i for group in groups for i in group.detector_indices}),
        error_us=result.uncertainties.get("DT0"),
        source_run=dataset.run_number,
    )
