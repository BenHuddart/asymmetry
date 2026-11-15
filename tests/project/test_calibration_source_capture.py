"""Quick history retains the resolved detector request across durable transport."""

import json
from dataclasses import replace

import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.project.calibration import apply_calibration_outcome
from asymmetry.core.project.profiles import (
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_source
from tests.core.test_tf_calibration_fit import _raw_tf


@pytest.mark.parametrize("method", ["ratio", "diamagnetic", "general"])
def test_quick_capture_retains_members_and_exclusions_after_regrouping(method):
    data = _raw_tf(beta=1, backgrounds=(0, 0))
    f, b = data.run.histograms
    data.run.histograms = [Histogram(h.counts.copy(), h.bin_width) for h in (f, b, f, b, f, b)]
    data.run.grouping.update(groups={1: [1, 3, 5], 2: [2, 4, 6]}, excluded_detectors=[5, 6])
    result = calibrate_source(data, CalibrationOptions(method))
    assert result.alpha is not None, result.message
    captured = result.result
    assert captured.forward_indices == (0, 2)
    assert captured.backward_indices == (1, 3)
    assert captured.excluded_indices == (4, 5)
    before = json.dumps(captured.to_dict(), sort_keys=True)
    profile = GroupingProfile("source", ProfileFingerprint("test", 6))
    apply_calibration_outcome(profile, result)
    profile.groups = {1: [2, 4], 2: [1, 3]}
    data.run.grouping["groups"][1][:] = [2, 4]
    data.run.grouping["excluded_detectors"].clear()
    data.run.histograms[0].counts[:] = 0
    restored = GroupingProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert json.dumps(restored.alpha_policy.result.to_dict(), sort_keys=True) == before
    assert restored.alpha_policy.value == result.alpha
    assert resolve_effective_grouping(restored, data.run)["alpha"] == result.alpha
    with pytest.raises(TypeError):
        restored.alpha_policy.result.forward_indices[0] = 99


def test_quick_legacy_membership_stays_unknown_on_json_restore():
    from asymmetry.core.workflow.calibration_results import QuickCalibrationResult

    captured = calibrate_source(_raw_tf(beta=1), CalibrationOptions("ratio")).result
    payload = json.loads(json.dumps(captured.to_dict()))
    for name in ("forward_indices", "backward_indices", "excluded_indices"):
        payload.pop(name, None)
    restored = QuickCalibrationResult.from_dict(payload)
    assert restored.forward_indices is None
    assert restored.backward_indices is None
    assert restored.excluded_indices is None
    assert restored.estimate == captured.estimate


def test_quick_nested_alignment_and_mapping_are_immutable_after_json():
    from asymmetry.core.workflow.calibration_results import QuickCalibrationResult

    captured = calibrate_source(_raw_tf(beta=1), CalibrationOptions("ratio")).result
    captured = replace(captured, period_mapping=((1, "red"), (2, "red")))
    payload = json.loads(json.dumps(captured.to_dict()))
    restored = QuickCalibrationResult.from_dict(payload)
    before = restored.to_dict()
    payload["period_mapping"][0][1] = "green"
    payload["alignment"]["detector_offsets"][0][1] = 99
    payload["alignment"]["timing_sources"][0][1] = "changed"
    assert restored.to_dict() == before
    for nested in (
        restored.period_mapping,
        restored.alignment.detector_offsets,
        restored.alignment.timing_sources,
        restored.alignment.timing_residuals_us,
        restored.alignment.acquired_detectors,
    ):
        assert isinstance(nested, tuple)
        assert all(isinstance(item, tuple) for item in nested)


def test_tf_exclusions_and_nested_domains_survive_actual_json():
    from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, fit_tf_calibration
    from asymmetry.core.project.calibration import (
        CalibrationRecord,
        calibration_payload,
        restore_calibration,
    )
    from asymmetry.core.project.json_values import decode_non_finite, encode_non_finite

    data = _raw_tf(beta=1)
    data.run.histograms.append(data.run.histograms[0])
    data.run.grouping["excluded_detectors"] = [3]
    fitted = fit_tf_calibration(data, TFCalibrationRequest(1, 2))
    assert fitted.excluded_indices == (2,)
    record = CalibrationRecord("capture", fitted)
    payload = json.loads(json.dumps(encode_non_finite(calibration_payload(record))))
    restored = restore_calibration(decode_non_finite(payload))
    assert restored == record
    payload["excluded_indices"][0] = 99
    payload["domains"][0]["alignment"]["detector_offsets"][0][1] = "changed"
    assert restored == record
    assert restored.result.excluded_indices == (2,)
    for domain in restored.result.fit.domains:
        assert isinstance(domain.detector_indices, tuple)
        for name in (
            "detector_offsets",
            "timing_residuals_us",
            "timing_sources",
            "acquired_detectors",
        ):
            values = getattr(domain.alignment, name)
            assert isinstance(values, tuple)
            assert all(isinstance(item, tuple) for item in values)
    legacy = calibration_payload(record)
    legacy.pop("excluded_indices")
    assert restore_calibration(legacy).result.excluded_indices is None


@pytest.mark.parametrize("method", ["ratio", "diamagnetic", "general"])
def test_equal_group_sums_still_retain_distinct_member_partitions(method):
    data = _raw_tf(beta=1, backgrounds=(0, 0))
    histogram = data.run.histograms[0]
    data.run.histograms = [
        Histogram(histogram.counts.copy(), histogram.bin_width) for _ in range(4)
    ]
    data.run.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
    first = calibrate_source(data, CalibrationOptions(method)).result
    data.run.grouping["groups"] = {1: [1, 3], 2: [2, 4]}
    second = calibrate_source(data, CalibrationOptions(method)).result
    assert first.estimate == second.estimate
    assert first.alignment == second.alignment
    assert first.to_dict() != second.to_dict()
