"""Fresh acquired-count requests are independent of cached curve diagnostics."""

import numpy as np
import pytest

from asymmetry.core.fitting.grouped_time_domain import build_count_groups
from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, fit_tf_calibration
from asymmetry.core.transform.corrections import (
    CorrectionUnavailableError,
    require_complete_dataset,
)
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run, calibrate_source
from asymmetry.core.workflow.reduction import ReductionSettings
from tests.core.test_tf_calibration_fit import _raw_tf


@pytest.mark.parametrize("entry", ["direct", "resolved", "workflow", "grouped"])
def test_current_count_request_rebuilds_from_histograms_despite_old_curve_failure(entry):
    source = _raw_tf(beta=1)
    before = [h.counts.copy() for h in source.run.histograms]
    source.metadata["correction_error"] = "Old curve used an unavailable detector group"
    source.run.metadata["correction_error"] = source.metadata["correction_error"]
    if entry == "direct":
        assert fit_tf_calibration(source, TFCalibrationRequest(1, 2)).ok
    elif entry == "grouped":
        groups = build_count_groups(source, [1, 2], lifetime_corrected=False)
        assert len(groups) == 2
        assert all(group.counts.size for group in groups)
    else:
        options = CalibrationOptions("tf_calibration_fit", envelope="exponential")
        result = (
            calibrate_source(source, options)
            if entry == "resolved"
            else calibrate_run(source, ReductionSettings(), options)
        )
        assert result.alpha is not None, result.message
    for expected, histogram in zip(before, source.run.histograms, strict=True):
        np.testing.assert_array_equal(expected, histogram.counts)
    with pytest.raises(CorrectionUnavailableError, match="Old curve"):
        require_complete_dataset(source)


def test_fresh_count_request_still_rejects_current_missing_reference():
    source = _raw_tf(beta=1)
    source.run.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 99},
    )
    with pytest.raises(ValueError, match="reference|Reference"):
        calibrate_source(source, CalibrationOptions("tf_calibration_fit", envelope="exponential"))


def test_fresh_count_request_still_requires_current_acquired_observations():
    source = _raw_tf(beta=1)
    source.run.metadata["raw_period_histograms_available"] = False
    with pytest.raises(ValueError, match="acquired period"):
        fit_tf_calibration(source, TFCalibrationRequest(1, 2))


def test_fresh_grouped_count_arrays_do_not_inherit_an_old_curve_diagnosis():
    from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets

    source = _raw_tf(beta=1)
    reason = "Old curve used an unavailable detector group"
    source.metadata["correction_error"] = reason
    source.run.metadata["correction_error"] = reason
    groups = build_grouped_time_domain_datasets(source, lifetime_corrected=False)
    assert len(groups) == 2
    for group in groups:
        require_complete_dataset(group)
        assert "correction_error" not in group.metadata
    assert source.metadata["correction_error"] == reason
    assert source.run.metadata["correction_error"] == reason
