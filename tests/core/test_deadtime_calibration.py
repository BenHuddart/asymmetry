"""Signed calibration against the published ISIS line and a recorded binary check."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.deadtime import calibrate_deadtime_from_histograms
from asymmetry.core.utils.constants import MUON_LIFETIME_US


@pytest.mark.parametrize("coefficient", [-0.02, 0.0, 0.02])
def test_cal_matches_mantid_6_16_1_signed_reference(coefficient):
    # Mantid 6.16.1 CalMuonDeadTime binary: issue #389 records this input and
    # recovery within 2e-12 µs. No third-party implementation source is used.
    width, frames = 0.016, 1e6
    time = 0.504 + np.arange(282) * width
    true = 1e4 * np.exp(-time / MUON_LIFETIME_US)
    observed = true / (1 + true * coefficient / (width * frames))
    result = calibrate_deadtime_from_histograms(
        [Histogram(observed, bin_width=width)], num_good_frames=frames, max_bins=282
    )
    assert result is not None
    assert result[0] == pytest.approx(coefficient, abs=1e-9)


def test_cal_matches_published_regression():
    # RUMDA (RAL-94-080), §4: M exp(t/τμ) = intercept + slope M;
    # d = -slope Δt F / intercept, with d and Δt both in µs.
    width, frames, coefficient = 0.02, 1000, 0.04
    time = (np.arange(180) + 0.5) * width
    true = 180 * np.exp(-time / MUON_LIFETIME_US)
    measured = true / (1 + true * coefficient / (width * frames))
    slope, intercept = np.polyfit(measured, measured * np.exp(time / MUON_LIFETIME_US), 1)
    independent = -slope * width * frames / intercept
    calibrated = calibrate_deadtime_from_histograms(
        [Histogram(measured, bin_width=width)], num_good_frames=frames
    )
    assert calibrated[0] == pytest.approx(independent, rel=1e-6)


@pytest.mark.parametrize("base", [0.0, 0.01])
def test_fit_promote_reduce_recovers_true_counts(base):
    from asymmetry.core.data.dataset import MuonDataset, Run
    from asymmetry.core.fitting.count_domain import fit_fb_alpha
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.transform import corrected_grouped_counts
    from asymmetry.core.transform.deadtime import promote_deadtime_to_grouping

    width, frames, true_coefficient = 0.01, 1e6, 0.02
    time = np.arange(1000) * width
    true = np.array(
        [1e5 * rate * np.exp(-time / MUON_LIFETIME_US) for rate in (1.0, 0.6, 0.8, 0.5)]
    )
    measured = true / (1 + true * true_coefficient / (width * frames))
    run = Run(
        run_number=72,
        histograms=[Histogram(values, bin_width=width) for values in measured],
        grouping={
            "groups": {1: [1, 2], 2: [3, 4]},
            "first_good_bin": 0,
            "last_good_bin": 999,
            "good_frames": frames,
            "deadtime_correction": bool(base),
            "deadtime_mode": "manual",
            "deadtime_method": "manual",
            "dead_time_us": [base] * 4,
        },
    )
    dataset = MuonDataset(
        time, np.zeros_like(time), np.ones_like(time), metadata={"run_number": 72}, run=run
    )
    params = ParameterSet(
        [
            Parameter("N0", 1.6e5, min=0),
            Parameter("alpha", 1.6 / 1.3),
            Parameter("background", 0, fixed=True),
            Parameter("background_b", 0, fixed=True),
            Parameter("DT0", 0.015),
        ]
    )
    result = fit_fb_alpha(dataset, 1, 2, lambda t: np.zeros_like(t), params, cost="gaussian")
    assert result.success
    fitted = result.group_results[1]
    assert fitted.parameters["DT0"].value == pytest.approx(true_coefficient - base, abs=1e-7)
    calibration = fitted.deadtime_calibration
    assert calibration == result.group_results[2].deadtime_calibration
    assert calibration.tau_us == pytest.approx([true_coefficient] * 4, abs=1e-7)
    assert calibration.source_run == 72 and calibration.method == "count_fit"
    # The captured base is immutable even if the live table changes before promotion.
    run.grouping["dead_time_us"] = [0.3] * 4
    promote_deadtime_to_grouping(run.grouping, calibration)
    reduced = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=run.grouping,
        forward_idx=[0, 1],
        backward_idx=[2, 3],
        use_deadtime=True,
        deadtime_mode="manual",
        use_background=False,
    )
    np.testing.assert_allclose(reduced.forward.counts, np.sum(true[:2], axis=0), rtol=1e-6)
    np.testing.assert_allclose(reduced.backward.counts, np.sum(true[2:], axis=0), rtol=1e-6)


@pytest.mark.parametrize("frames", [None, 0.0, -1.0, np.nan])
def test_calibration_rejects_missing_or_invalid_exposure(frames):
    from asymmetry.core.transform.deadtime import DeadtimeCorrectionError

    with pytest.raises(DeadtimeCorrectionError, match="good frames"):
        calibrate_deadtime_from_histograms(
            [Histogram(np.ones(10), bin_width=0.01)], num_good_frames=frames
        )


def test_calibration_record_errors_and_cooperative_cancel():
    from concurrent.futures import CancelledError
    from dataclasses import FrozenInstanceError

    from asymmetry.core.transform.deadtime import fit_deadtime_calibration

    width, frames = 0.02, 1e6
    time = np.arange(200) * width
    true = 1e4 * np.exp(-time / MUON_LIFETIME_US)
    histograms = [Histogram(true / (1 + true * 0.02 / (width * frames)), bin_width=width)]
    result = fit_deadtime_calibration(histograms, num_good_frames=frames, source_run=7)
    assert result.errors_us[0] > 0
    with pytest.raises(FrozenInstanceError):
        result.method = "estimate"
    with pytest.raises(CancelledError):
        fit_deadtime_calibration(histograms, num_good_frames=frames, cancel_callback=lambda: True)


def test_applied_base_outcome_survives_worker_serialization():
    import pickle

    from asymmetry.core.transform.corrections import CorrectionOutcome, CorrectionStatus

    original = CorrectionOutcome(
        "deadtime",
        "manual",
        CorrectionStatus.APPLIED,
        effective={"tau_us": [0.01, -0.02], "good_frames": 1e6, "source": "calibrate"},
        uncertainty="coefficient covariance unavailable",
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    with pytest.raises(TypeError):
        restored.effective["tau_us"] = [0.0, 0.0]
