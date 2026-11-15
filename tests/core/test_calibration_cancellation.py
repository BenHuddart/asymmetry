"""Cancellation stops actual calibration work before the next work unit."""

from threading import Event

import pytest

from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
from asymmetry.core.workflow.reduction import ReductionSettings
from tests.core.test_tf_calibration_fit import _raw_tf


@pytest.mark.parametrize(
    "method,owner,seam",
    [
        ("diamagnetic", "asymmetry.core.transform.asymmetry", "_diamagnetic_objective"),
        ("tf_calibration_fit", "asymmetry.core.fitting.count_inputs", "count_time_selection"),
    ],
)
def test_cancel_inside_calibration_work_stops_before_next_unit(monkeypatch, method, owner, seam):
    import importlib

    module = importlib.import_module(owner)
    original = vars(module)[seam]
    cancelled = Event()
    calls = []

    def cancel_after_work(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(1)
        cancelled.set()
        return value

    monkeypatch.setattr(module, seam, cancel_after_work)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            _raw_tf(beta=1),
            ReductionSettings(),
            CalibrationOptions(method),
            cancel_callback=cancelled.is_set,
        )
    assert len(calls) == 1, "Calibration continued into another unit after cancellation"


@pytest.mark.parametrize("stage", ["hesse", "minos"])
def test_covariance_cancellation_propagates_without_packing_a_fit(monkeypatch, stage):
    from iminuit import Minuit

    from asymmetry.core.fitting import count_domain
    from asymmetry.core.fitting.tf_calibration import TFEnvelope

    cancelled = Event()
    original_stage = vars(Minuit)[stage]
    original_drive = count_domain.drive_minuit
    returned = []
    entered = []

    def cancel_in_covariance(self, *args, **kwargs):
        entered.append(1)
        cancelled.set()
        return original_stage(self, *args, **kwargs)

    def observe_drive(*args, **kwargs):
        result = original_drive(*args, **kwargs)
        returned.append(1)
        return result

    monkeypatch.setattr(Minuit, stage, cancel_in_covariance)
    monkeypatch.setattr(count_domain, "drive_minuit", observe_drive)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            _raw_tf(beta=1),
            ReductionSettings(),
            CalibrationOptions("tf_calibration_fit", envelope=TFEnvelope.EXPONENTIAL, minos=True),
            cancel_callback=cancelled.is_set,
        )
    assert entered == [1]
    assert returned == [], "MINOS cancellation became a normal solver return"


@pytest.mark.parametrize("method", ["ratio", "diamagnetic", "general", "tf_calibration_fit"])
def test_pre_cancel_prevents_preparation(monkeypatch, method):
    from asymmetry.core.workflow import calibration

    def unexpected(*args, **kwargs):
        pytest.fail("A pre-cancelled calibration started source preparation")

    monkeypatch.setattr(calibration, "calibration_source", unexpected)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            _raw_tf(beta=1),
            ReductionSettings(),
            CalibrationOptions(method),
            cancel_callback=lambda: True,
        )


def test_general_diagnostic_stops_before_another_subwindow(monkeypatch):
    import inspect

    from asymmetry.core.transform import asymmetry

    original = asymmetry.np.sum
    cancelled = Event()
    windows = []

    def sum_in_diagnostic(*args, **kwargs):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == "_general_flatness" and "lo" in frame.f_locals:
            window = (frame.f_locals["lo"], frame.f_locals["hi"])
            if not windows or windows[-1] != window:
                windows.append(window)
                cancelled.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(asymmetry.np, "sum", sum_in_diagnostic)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            _raw_tf(beta=1),
            ReductionSettings(),
            CalibrationOptions("general"),
            cancel_callback=cancelled.is_set,
        )
    assert len(windows) == 1, "General continued through diagnostic sub-windows"


@pytest.mark.parametrize("method", ["ratio", "diamagnetic", "general", "tf_calibration_fit"])
def test_detector_correction_cancel_stops_before_next_detector(monkeypatch, method):
    from asymmetry.core.transform import scoped_counts
    from tests.core.test_tf_deadtime_calibration import _deadtime_tf

    data = _deadtime_tf(fit_beta=False, mode="from_file")
    original = scoped_counts.apply_deadtime_correction
    cancelled = Event()
    detectors = []

    def correct_one(*args, **kwargs):
        result = original(*args, **kwargs)
        detectors.append(kwargs["detector"])
        cancelled.set()
        return result

    monkeypatch.setattr(scoped_counts, "apply_deadtime_correction", correct_one)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            data,
            ReductionSettings(deadtime="from_file"),
            CalibrationOptions(method),
            cancel_callback=cancelled.is_set,
        )
    assert len(detectors) == 1


@pytest.mark.parametrize("method", ["ratio", "diamagnetic", "general", "tf_calibration_fit"])
def test_never_cancelled_preserves_complete_scientific_result(method):
    data = _raw_tf(beta=1)
    options = CalibrationOptions(method)
    plain = calibrate_run(data, ReductionSettings(), options)
    polled = calibrate_run(data, ReductionSettings(), options, cancel_callback=lambda: False)
    assert (
        polled.result == plain.result
        if plain.record is None
        else (polled.record.result == plain.record.result)
    )


def test_cancel_between_envelopes_returns_no_partial_selection(monkeypatch):
    from asymmetry.core.fitting import tf_calibration

    original = tf_calibration._fit_prepared_tf
    cancelled = Event()
    completed = []

    def completed_candidate(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(result)
        cancelled.set()
        return result

    monkeypatch.setattr(tf_calibration, "_fit_prepared_tf", completed_candidate)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            _raw_tf(beta=1),
            ReductionSettings(),
            CalibrationOptions("tf_calibration_fit"),
            cancel_callback=cancelled.is_set,
        )
    assert len(completed) == 1 and completed[0].ok


def test_cancel_during_reference_correction_stops_before_next_reference_detector(monkeypatch):
    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.transform import scoped_counts
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest
    from asymmetry.core.workflow.calibration import calibrate_source
    from tests.core.test_reference_timing import timing_runs

    sample, reference = timing_runs()
    sample.grouping.update(
        deadtime_correction=True,
        deadtime_mode="from_file",
        deadtime_file_us=[0.02, 0.02],
        good_frames=100000,
    )
    resolved = BackgroundReference(
        reference.histograms,
        ReferenceScaleRequest("manual", 1).resolve(),
        run_number=reference.run_number,
        deadtime_normaliser=100000,
        file_dead_time_us=(0.02, 0.02),
    )
    empty = np.empty(0)
    data = MuonDataset(empty, empty, empty, run=sample)
    original = scoped_counts.apply_deadtime_correction
    cancelled = Event()
    reference_detectors = []

    def correct(*args, **kwargs):
        result = original(*args, **kwargs)
        if np.max(args[0]) < 100:
            reference_detectors.append(kwargs["detector"])
            cancelled.set()
        return result

    monkeypatch.setattr(scoped_counts, "apply_deadtime_correction", correct)
    with pytest.raises(FitCancelledError):
        calibrate_source(
            data,
            CalibrationOptions("ratio"),
            reference_resolver=lambda _: resolved,
            cancel_callback=cancelled.is_set,
        )
    assert reference_detectors == [0]
