"""Existing form capture retains frozen calibration until a manual edit."""

import pytest

from asymmetry.core.project.calibration import CalibrationRecord, apply_calibration
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.gui.windows.grouping.profile_bridge import profile_from_form_payload
from tests.project.test_tf_calibration_records import candidate

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("manual_beta", [False, True])
def test_form_capture_preserves_history_and_authored_beta(manual_beta):
    profile = GroupingProfile("TF", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("frozen", candidate()))
    payload = {"groups": {1: [1], 2: [2]}, "alpha": 1.2, "beta": 1.1}
    if not manual_beta:
        payload.update(beta_method="tf_calibration_fit", beta_reference_run=123, beta_error=0.3)
    captured = profile_from_form_payload(
        payload,
        name=profile.name,
        fingerprint=profile.fingerprint,
        alpha_policy=profile.alpha_policy,
        calibration_source=profile,
    )
    restored = GroupingProfile.from_dict(captured.to_dict())
    assert restored.alpha_policy.record is restored.calibration_records[0]
    if manual_beta:
        assert restored.beta_policy.method == ""
        assert restored.beta_policy.value == 1.1
    else:
        assert restored.beta_policy.record is restored.alpha_policy.record
