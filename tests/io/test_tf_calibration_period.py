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
    assert ratio.period_number is None
    assert dict(ratio.result.period_mapping) == mapping
    from asymmetry.core.workflow.calibration_results import QuickCalibrationResult
    from asymmetry.core.workflow.jsonio import json_safe

    assert QuickCalibrationResult.from_dict(json_safe(ratio.result.to_dict())) == ratio.result


@pytest.mark.parametrize("method", ["tf_calibration_fit", "ratio", "diamagnetic", "general"])
@pytest.mark.parametrize("period", [None, 3])
def test_workflow_preserves_an_already_selected_acquired_period(period, method):
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    data = _raw_tf(beta=1)
    data.run.metadata.update(source_run_number=7101, period_number=3, period_count=4)
    options = CalibrationOptions(
        method,
        envelope="exponential" if method == "tf_calibration_fit" else None,
        frequency_mhz=1.3 if method == "tf_calibration_fit" else None,
        period=period,
    )
    outcome = calibrate_run(data, ReductionSettings(), options)
    assert outcome.period_number == 3


@pytest.mark.parametrize("method", ["tf_calibration_fit", "ratio", "diamagnetic", "general"])
def test_workflow_rejects_mismatched_selected_source_period(method):
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    data = _raw_tf(beta=1)
    data.run.metadata.update(source_run_number=7101, period_number=3, period_count=4)
    with pytest.raises(ValueError, match="requested acquired period"):
        calibrate_run(data, ReductionSettings(), CalibrationOptions(method, period=2))


def _calibration_periods(*, cached_only=False):
    """Different bank balances make accidental parent-period reuse observable."""
    red, green = _raw_tf(beta=1), _raw_tf(beta=1, seed=43)
    from asymmetry.core.utils.constants import MUON_LIFETIME_US

    time = red.run.histograms[0].time_axis
    decay = np.exp(-time / MUON_LIFETIME_US)
    polarization = 0.2 * np.exp(-0.3 * time)
    for data in (red, green):
        data.run.histograms[0].counts = np.rint(20000 * decay * (1 + polarization))
        data.run.histograms[1].counts = np.rint(16000 * decay * (1 - polarization))
    green.run.histograms[1].counts *= 2
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.metadata.update(run_number=7101, source_run_number=7101, period_count=2)
    red.run.grouping.update(period_reduced=[empty, empty], period_good_frames=[10.0, 20.0])
    if not cached_only:
        red.run.grouping["period_histograms"] = [red.run.histograms, green.run.histograms]
    return red, green


@pytest.mark.parametrize("method", [None, "ratio", "diamagnetic", "general", "tf_calibration_fit"])
@pytest.mark.parametrize(
    "source_form",
    ["combined_default", "combined_green", "selected_default", "selected_green", "list_green"],
)
def test_all_workflow_methods_reject_cached_only_source_counts(method, source_form):
    from asymmetry.core.io.periods import select_period
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    combined, _ = _calibration_periods(cached_only=True)
    selected = select_period(combined, 2)
    source = (
        [combined, selected]
        if source_form == "list_green"
        else selected
        if source_form.startswith("selected")
        else combined
    )
    options = CalibrationOptions(method, period=2 if source_form.endswith("green") else None)
    with pytest.raises(ValueError, match="acquired.*period"):
        calibrate_run(source, ReductionSettings(), options)


@pytest.mark.parametrize("method", [None, "ratio", "diamagnetic", "general"])
@pytest.mark.parametrize("cached_curves", [False, True])
def test_default_combined_quick_source_records_effective_red_period(method, cached_curves):
    import json

    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.calibration_results import QuickCalibrationResult
    from asymmetry.core.workflow.jsonio import dumps
    from asymmetry.core.workflow.reduction import ReductionSettings

    combined, _ = _calibration_periods()
    if not cached_curves:
        combined.run.grouping.pop("period_reduced")
    outcome = calibrate_run(combined, ReductionSettings(), CalibrationOptions(method))
    assert outcome.period_number == 1
    from asymmetry.core.transform.asymmetry import estimate_alpha_detailed
    from asymmetry.core.transform.fb_counts import GroupCounts
    from asymmetry.core.workflow.calibration import calibration_source

    red = calibration_source(combined, ReductionSettings(), CalibrationOptions(method)).run
    assert red.metadata["period_number"] == 1
    assert red.grouping["good_frames"] == 10.0
    expected = estimate_alpha_detailed(
        GroupCounts(combined.run.histograms[0].counts),
        GroupCounts(combined.run.histograms[1].counts),
        method=method or "ratio",
        time_us=combined.run.histograms[0].time_axis,
        first_good_bin=4,
        last_good_bin=498,
    )
    assert outcome.alpha == expected.alpha
    if method is not None:
        assert outcome.result.estimate == expected
    restored = QuickCalibrationResult.from_dict(json.loads(dumps(outcome.result.to_dict())))
    assert restored == outcome.result
