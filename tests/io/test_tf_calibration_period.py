"""TF calibration delegates acquired-period selection to the existing owner."""

import numpy as np
import pytest

from asymmetry.core.fitting.tf_calibration import fit_tf_calibration
from tests.core.test_tf_calibration_fit import _raw_tf, _request


def test_selected_period_reuses_acquired_observation_owner():
    from asymmetry.core.io.periods import source_run_of

    red, green = _raw_tf(beta=1), _raw_tf(beta=1, seed=43)
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.metadata.update(run_number=7101, source_run_number=7101, period_count=2)
    red.run.grouping.update(
        period_histograms=[red.run.histograms, green.run.histograms],
        period_reduced=[empty, empty],
        period_good_frames=[10.0, 20.0],
    )
    result = fit_tf_calibration(red, _request(period=2))
    direct = fit_tf_calibration(green, _request())
    assert result.ok and direct.ok
    assert result.alpha == direct.alpha
    assert result.fit.covariance == direct.fit.covariance
    assert result.source_run_number == source_run_of(red.run)
    assert result.period_number == 2


@pytest.mark.parametrize("already_selected", [False, True])
def test_cached_period_reduction_cannot_substitute_for_missing_acquired_histograms(
    already_selected,
):
    from asymmetry.core.fitting.likelihood import FitInputError
    from asymmetry.core.io.periods import select_period

    red = _raw_tf(beta=1)
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.metadata.update(run_number=7101, source_run_number=7101, period_count=2)
    red.run.grouping.update(period_reduced=[empty, empty], period_good_frames=[10.0, 20.0])
    dataset = select_period(red, 2) if already_selected else red
    with pytest.raises(FitInputError, match="acquired period histograms"):
        fit_tf_calibration(dataset, _request(period=None if already_selected else 2))


def test_workflow_source_period_is_separate_from_target_difference():
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import GREEN_MINUS_RED, ReductionSettings

    red, green = _raw_tf(beta=1), _raw_tf(beta=1, seed=43)
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.metadata.update(source_run_number=7101, period_count=2)
    red.run.grouping.update(
        period_histograms=[red.run.histograms, green.run.histograms],
        period_reduced=[empty, empty],
        period_good_frames=[10.0, 20.0],
    )
    settings = ReductionSettings(period=GREEN_MINUS_RED)
    options = CalibrationOptions(
        "tf_calibration_fit", envelope="exponential", period=2, frequency_mhz=1.3
    )
    result = calibrate_run(red, settings, options).record.result
    assert result.period_number == 2
    assert result.source_run_number == 7101
    direct = fit_tf_calibration(green, _request())
    assert result.alpha == direct.alpha
    assert result.fit.covariance == direct.fit.covariance
    assert result.fit.domains[0].raw_windows == direct.fit.domains[0].raw_windows
    with pytest.raises(ValueError, match="acquired source period"):
        calibrate_run(red, settings, CalibrationOptions("tf_calibration_fit"))
    with pytest.raises(ValueError, match="acquired source period"):
        calibrate_run(red, ReductionSettings(), CalibrationOptions("tf_calibration_fit"))


def test_workflow_does_not_calibrate_cached_period():
    from asymmetry.core.fitting.likelihood import FitInputError
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    data = _raw_tf(beta=1)
    empty = (np.array([]), np.array([]), np.array([]))
    data.run.metadata.update(period_count=2)
    data.run.grouping.update(period_reduced=[empty, empty])
    with pytest.raises(FitInputError, match="acquired period histograms"):
        calibrate_run(data, ReductionSettings(), CalibrationOptions("tf_calibration_fit", period=2))


@pytest.mark.parametrize(
    "mapping",
    [
        {1: "ignore", 2: "red", 3: "red", 4: "ignore"},
        {1: "ignore", 2: "green", 3: "red", 4: "ignore"},
    ],
)
def test_workflow_never_labels_mapped_counts_as_an_acquired_period(mapping):
    from asymmetry.core.io.periods import combine_mapped_periods
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    periods = [_raw_tf(beta=1, seed=i) for i in range(4)]
    for i, data in enumerate(periods, 1):
        data.run.metadata.update(source_run_number=7101, period_number=i, period_count=4)
    mapped = combine_mapped_periods(periods, mapping)
    options = CalibrationOptions(
        "tf_calibration_fit", envelope="exponential", period=1, frequency_mhz=1.3
    )
    with pytest.raises(ValueError, match="mapped.*acquired"):
        calibrate_run(mapped, ReductionSettings(), options)
    from dataclasses import replace

    from asymmetry.core.io.periods import period_run

    red = period_run(mapped.run, 0)
    selected = replace(mapped, run=red, metadata=red.metadata)
    ratio = calibrate_run(selected, ReductionSettings(), CalibrationOptions("ratio"))
    assert ratio.alpha is not None


@pytest.mark.parametrize("period", [None, 3])
def test_workflow_preserves_an_already_selected_acquired_period(period):
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    data = _raw_tf(beta=1)
    data.run.metadata.update(source_run_number=7101, period_number=3, period_count=4)
    options = CalibrationOptions(
        "tf_calibration_fit", envelope="exponential", frequency_mhz=1.3, period=period
    )
    outcome = calibrate_run(data, ReductionSettings(), options)
    assert outcome.record.result.period_number == 3


def test_workflow_rejects_mismatched_selected_source_period():
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    data = _raw_tf(beta=1)
    data.run.metadata.update(source_run_number=7101, period_number=3, period_count=4)
    with pytest.raises(ValueError, match="requested acquired period"):
        calibrate_run(data, ReductionSettings(), CalibrationOptions("tf_calibration_fit", period=2))
