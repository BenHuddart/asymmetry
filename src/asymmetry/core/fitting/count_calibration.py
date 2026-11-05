"""Capture count-fit residual calibration only when a complete total table exists."""

import warnings

from asymmetry.core.transform.corrections import CorrectionStatus
from asymmetry.core.transform.deadtime import DeadtimeCalibration


def record_count_deadtime_calibration(result, dataset, groups):
    """Keep a successful fit usable when unknown unused coefficients prevent promotion."""
    if not result.success or "DT0" not in result.parameters:
        return
    base = groups[0].deadtime
    missing = (
        [i + 1 for i, tau in enumerate(base.effective["tau_us"]) if tau is None]
        if base.status is CorrectionStatus.APPLIED
        else []
    )
    if missing:
        message = (
            "Deadtime promotion unavailable: coefficients are unknown for detector(s) "
            + ", ".join(map(str, missing))
            + ". Fit results remain usable."
        )
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
