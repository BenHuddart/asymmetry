"""Explicit calibration packing preserves count-domain support and covariance."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting import tf_calibration
from asymmetry.core.fitting.tf_calibration import (
    TFCalibrationRequest,
    TFEnvelope,
    fit_tf_calibration,
)
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run, calibrate_source
from asymmetry.core.workflow.reduction import ReductionSettings, resolve_reduction_grouping
from tests.core.test_tf_calibration_fit import _raw_tf


@pytest.mark.parametrize("factor", [None, 1, 4, 7])
@pytest.mark.parametrize("entry", ["workflow", "source"])
def test_public_packing_matches_existing_count_path_and_independent_sums(
    monkeypatch, factor, entry
):
    data = _raw_tf()
    data.run.grouping.update(binning_mode="fixed", bunching_factor=3)
    original_grouping = dict(data.run.grouping)
    settings = ReductionSettings(rebin=5)
    grouping = (
        resolve_reduction_grouping(data.run, settings) if entry == "workflow" else original_grouping
    )
    if factor is not None:
        grouping = dict(grouping, binning_mode="fixed", bunching_factor=factor)
    prepared = replace(data, run=replace(data.run, grouping=grouping))
    direct = fit_tf_calibration(
        prepared, TFCalibrationRequest(1, 2, fit_beta=True, frequency_mhz=1.3)
    )
    build = tf_calibration.build_count_groups
    captured = []

    def capture(*args, **kwargs):
        groups = build(*args, **kwargs)
        captured.extend(groups)
        return groups

    monkeypatch.setattr(tf_calibration, "build_count_groups", capture)
    options = CalibrationOptions(
        "tf_calibration_fit", True, TFEnvelope.EXPONENTIAL, frequency_mhz=1.3, count_packing=factor
    )
    outcome = (
        calibrate_run(data, settings, options)
        if entry == "workflow"
        else calibrate_source(data, options)
    )
    result = outcome.record.result
    assert result == direct
    assert result.ok, result.message
    assert result.alpha_error > 0 and result.beta_error > 0
    assert result.alpha_beta_correlation is not None
    effective = 3 if factor is None else factor
    # The inclusive acquisition window is 4..498; incomplete final groups are dropped.
    windows = [(first, first + effective - 1) for first in range(4, 500 - effective, effective)]
    for group, domain, histogram in zip(
        captured, result.fit.domains, data.run.histograms, strict=True
    ):
        assert [(w.first, w.last) for w in domain.raw_windows] == windows
        np.testing.assert_array_equal(
            group.counts, [histogram.counts[a : b + 1].sum() for a, b in windows]
        )
        native = np.arange(histogram.counts.size) * histogram.bin_width
        np.testing.assert_allclose(group.time, [native[a : b + 1].mean() for a, b in windows])
    assert data.run.grouping == original_grouping


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "4", float("inf")])
def test_invalid_packing_rejected_at_request_boundary(bad):
    with pytest.raises(ValueError, match="packing.*integer"):
        CalibrationOptions("tf_calibration_fit", count_packing=bad)


@pytest.mark.parametrize("method", [None, "ratio", "general", "diamagnetic"])
def test_quick_methods_reject_explicit_packing(method):
    with pytest.raises(ValueError, match="packing.*tf_calibration_fit"):
        CalibrationOptions(method, count_packing=1)


@pytest.mark.parametrize("held", [False, True])
def test_packing_selected_period_and_centre_time_cuts(held):
    red, green = _raw_tf(seed=31), _raw_tf(seed=72)
    red.run.grouping.update(binning_mode="variable", bunching_factor=9)
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.grouping.update(
        period_histograms=[red.run.histograms, green.run.histograms],
        period_reduced=[empty, empty],
    )
    if not held:
        red.run.metadata["period_number"] = 1
        green.run.metadata["period_number"] = 2
    loaded = red if held else [red, green]
    options = CalibrationOptions(
        "tf_calibration_fit",
        True,
        TFEnvelope.EXPONENTIAL,
        period=2,
        frequency_mhz=1.3,
        count_packing=4,
        exclude=(2, 3),
    )
    outcome = calibrate_run(loaded, ReductionSettings(rebin=7, t_min=0.5, t_max=5), options)
    green.run.grouping.update(binning_mode="fixed", bunching_factor=4)
    direct = fit_tf_calibration(
        green,
        TFCalibrationRequest(
            1, 2, fit_beta=True, frequency_mhz=1.3, t_min=0.5, t_max=5, exclude=(2, 3)
        ),
    )
    result = outcome.record.result
    assert result.ok and direct.ok
    assert result.alpha == direct.alpha
    assert result.fit.covariance == direct.fit.covariance
    for actual, expected in zip(result.fit.domains, direct.fit.domains, strict=True):
        assert actual.raw_windows == expected.raw_windows
        assert actual.time_us == expected.time_us
        assert all(w.last - w.first + 1 == 4 for w in actual.raw_windows)
        assert all(0.5 <= t <= 5 and not 2 <= t <= 3 for t in actual.time_us)
    assert result.period_number == 2


def test_packing_does_not_bypass_background_preflight():
    from tests.core.test_tf_deadtime_calibration import _deadtime_tf

    data = _deadtime_tf()
    # This native bin lies outside the fitted signal but inside the requested background.
    data.run.histograms[0].counts[520] = 1000000
    options = CalibrationOptions(
        "tf_calibration_fit",
        envelope=TFEnvelope.EXPONENTIAL,
        frequency_mhz=1.3,
        count_packing=4,
    )
    settings = ReductionSettings(
        deadtime="from_file", background="range", background_range=(510, 550)
    )
    with pytest.raises(ValueError, match="saturat|occupancy"):
        calibrate_run(data, settings, options)
