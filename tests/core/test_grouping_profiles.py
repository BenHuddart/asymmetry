"""Tests for project-level grouping profiles (:mod:`asymmetry.core.project.profiles`).

Covers profile round-trip serialization, fingerprint matching, byte-identical
resolution for each alpha/deadtime/background policy mode, and the loaded-run
inheritance helper. These are pure-core tests (no Qt / no GUI).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GroupingProfile,
    ProfileFingerprint,
    active_profile_for_run,
    effective_grouping_for_loaded_run,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)

# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #


def _run(
    *,
    instrument: str = "EMU",
    n_hist: int = 4,
    grouping: dict | None = None,
    metadata: dict | None = None,
) -> Run:
    """Build a Run with *n_hist* simple, distinguishable histograms."""
    histograms = [
        Histogram(
            counts=np.arange(10 * i, 10 * i + 20, dtype=float) + 1.0,
            bin_width=0.016,
            t0_bin=5,
            good_bin_start=6,
            good_bin_end=19,
        )
        for i in range(n_hist)
    ]
    base_grouping = {"instrument": instrument}
    if grouping:
        base_grouping.update(grouping)
    return Run(
        run_number=1,
        histograms=histograms,
        grouping=base_grouping,
        metadata=metadata or {"instrument": instrument},
    )


def _per_run_facts() -> dict:
    """A run's file-derived grouping facts (never stored on a profile)."""
    return {
        "t0_bin": 5,
        "t_good_offset": 1,
        "first_good_bin": 6,
        "last_good_bin": 19,
        "bin_index_base": 0,
        "detector_t0_bins": [5, 5, 5, 5],
        "detector_first_good_bins": [6, 6, 6, 6],
        "detector_last_good_bins": [19, 19, 19, 19],
        "histogram_labels": ["F1", "F2", "B1", "B2"],
        "good_frames": 1000.0,
    }


def _base_profile(**overrides) -> GroupingProfile:
    kwargs = {
        "name": "Default (EMU)",
        "fingerprint": ProfileFingerprint("EMU", 4),
        "groups": {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
    }
    kwargs.update(overrides)
    return GroupingProfile(**kwargs)


# --------------------------------------------------------------------------- #
# Serialization round-trip
# --------------------------------------------------------------------------- #


def test_profile_round_trips_through_dict():
    profile = _base_profile(
        included_groups={1: True, 2: True},
        excluded_detectors=[3],
        projections=[{"label": "P_z", "forward_group": 1, "backward_group": 2, "alpha": 1.1}],
        alpha_policy=AlphaPolicy(
            mode="calibrated", value=1.23, error=0.02, method="ratio", source_run=42
        ),
        deadtime_policy=DeadtimePolicy(
            mode="manual", values=[0.01, 0.02, 0.03, 0.04], method="calibrate", source_run=7
        ),
        background_policy=BackgroundPolicy(
            mode="fixed", details={"background_fixed_values": [3.0, 4.0]}
        ),
        binning_mode="variable",
        bin0_us=0.05,
        bin10_us=0.5,
        bunching_factor=2,
        period_mode="green_minus_red",
        extra={"grouping_preset": "Longitudinal", "alpha_x": 1.0},
    )
    restored = GroupingProfile.from_dict(profile.to_dict())
    assert restored.to_dict() == profile.to_dict()


def test_policy_dicts_round_trip_each_mode():
    for policy in (
        AlphaPolicy(mode="fixed", value=1.0),
        AlphaPolicy(mode="calibrated", value=1.2, error=0.01, method="ratio", source_run=3),
        AlphaPolicy(mode="per_run_estimate"),
    ):
        assert AlphaPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    for policy in (
        DeadtimePolicy(mode="off"),
        DeadtimePolicy(mode="from_file"),
        DeadtimePolicy(
            mode="manual", values=[0.1, 0.2], manual_us=0.1, method="calibrate", source_run=9
        ),
        DeadtimePolicy(mode="estimate", estimated_us=0.05, source_run=9),
    ):
        assert DeadtimePolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    for policy in (
        BackgroundPolicy(mode="none"),
        BackgroundPolicy(mode="reference_run", details={"background_run": {"run_number": 5}}),
    ):
        assert BackgroundPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()


def test_from_dict_rejects_unknown_modes_gracefully():
    assert AlphaPolicy.from_dict({"mode": "bogus"}).mode == "fixed"
    assert DeadtimePolicy.from_dict({"mode": "bogus"}).mode == "off"
    assert BackgroundPolicy.from_dict({"mode": "bogus"}).mode == "none"


# --------------------------------------------------------------------------- #
# Fingerprint matching
# --------------------------------------------------------------------------- #


def test_fingerprint_from_run_reads_grouping_instrument_and_histogram_count():
    run = _run(instrument="MuSR", n_hist=64)
    fp = profile_fingerprint_for_run(run)
    assert fp == ProfileFingerprint("MuSR", 64)


def test_fingerprint_falls_back_to_metadata_instrument():
    run = _run(n_hist=4)
    run.grouping.pop("instrument")
    run.metadata["instrument"] = "HIFI"
    assert profile_fingerprint_for_run(run).instrument == "HIFI"


def test_fingerprint_matching_is_case_insensitive_on_instrument():
    assert ProfileFingerprint("EMU", 4).matches(ProfileFingerprint("emu", 4))
    assert not ProfileFingerprint("EMU", 4).matches(ProfileFingerprint("EMU", 8))
    assert not ProfileFingerprint("EMU", 4).matches(ProfileFingerprint("MuSR", 4))


def test_active_profile_selection_by_fingerprint():
    emu = _base_profile(name="EMU-A", fingerprint=ProfileFingerprint("EMU", 4), active=True)
    musr = _base_profile(name="MuSR-A", fingerprint=ProfileFingerprint("MuSR", 4), active=True)
    inactive = _base_profile(name="EMU-B", fingerprint=ProfileFingerprint("EMU", 4), active=False)
    run = _run(instrument="EMU", n_hist=4)
    assert active_profile_for_run([inactive, musr, emu], run) is emu
    assert active_profile_for_run([inactive, musr], run) is None


# --------------------------------------------------------------------------- #
# Resolution: byte-identical payloads per policy mode
# --------------------------------------------------------------------------- #


def test_resolve_fixed_alpha_and_no_corrections():
    run = _run(grouping=_per_run_facts())
    profile = _base_profile(alpha_policy=AlphaPolicy(mode="fixed", value=1.5))
    resolved = resolve_effective_grouping(profile, run)
    expected = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": [],
        "bunching_factor": 1,
        "instrument": "EMU",
        "t0_bin": 5,
        "t_good_offset": 1,
        "first_good_bin": 6,
        "last_good_bin": 19,
        "bin_index_base": 0,
        "detector_t0_bins": [5, 5, 5, 5],
        "detector_first_good_bins": [6, 6, 6, 6],
        "detector_last_good_bins": [19, 19, 19, 19],
        "histogram_labels": ["F1", "F2", "B1", "B2"],
        "good_frames": 1000.0,
        "alpha": 1.5,
        "deadtime_correction": False,
        "deadtime_mode": "off",
        "background_correction": False,
        "background_mode": "none",
    }
    assert resolved == expected


def test_resolve_calibrated_alpha_carries_provenance():
    run = _run(grouping=_per_run_facts())
    profile = _base_profile(
        alpha_policy=AlphaPolicy(
            mode="calibrated", value=1.2, error=0.03, method="ratio", source_run=42
        )
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["alpha"] == 1.2
    assert resolved["alpha_method"] == "ratio"
    assert resolved["alpha_error"] == 0.03
    assert resolved["alpha_reference_run"] == 42


def test_resolve_per_run_estimate_computes_integral_ratio():
    from asymmetry.core.transform.asymmetry import estimate_alpha
    from asymmetry.core.transform.grouping import (
        apply_grouping_aligned,
        common_t0_for_groups,
    )

    run = _run(grouping=_per_run_facts())
    profile = _base_profile(alpha_policy=AlphaPolicy(mode="per_run_estimate"))
    resolved = resolve_effective_grouping(profile, run)

    forward_idx, backward_idx = [0, 1], [2, 3]
    common_t0 = common_t0_for_groups(run.histograms, forward_idx, backward_idx)
    f = apply_grouping_aligned(run.histograms, forward_idx, common_t0_bin=common_t0)
    b = apply_grouping_aligned(run.histograms, backward_idx, common_t0_bin=common_t0)
    expected_alpha = estimate_alpha(f, b, first_good_bin=6, last_good_bin=19)

    assert resolved["alpha"] == pytest.approx(expected_alpha)
    assert resolved["alpha_method"] == "per_run_estimate"
    assert expected_alpha != 1.0  # the fixtures give an unbalanced ratio


def test_resolve_deadtime_from_file_uses_run_values():
    facts = _per_run_facts()
    facts["dead_time_us"] = [0.011, 0.012, 0.013, 0.014]
    run = _run(grouping=facts)
    profile = _base_profile(deadtime_policy=DeadtimePolicy(mode="from_file"))
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["deadtime_correction"] is True
    assert resolved["deadtime_mode"] == "file"
    assert resolved["deadtime_method"] == "file"
    assert resolved["dead_time_us"] == [0.011, 0.012, 0.013, 0.014]


def test_resolve_deadtime_manual_broadcasts_stored_values():
    run = _run(grouping=_per_run_facts())
    profile = _base_profile(
        deadtime_policy=DeadtimePolicy(
            mode="manual", values=[0.01, 0.02, 0.03, 0.04], method="calibrate", source_run=7
        )
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["deadtime_mode"] == "manual"
    assert resolved["deadtime_method"] == "calibrate"
    assert resolved["dead_time_us"] == [0.01, 0.02, 0.03, 0.04]
    assert resolved["deadtime_reference_run"] == 7


def test_resolve_deadtime_estimate_broadcasts_single_value_per_detector():
    run = _run(n_hist=4, grouping=_per_run_facts())
    profile = _base_profile(
        deadtime_policy=DeadtimePolicy(mode="estimate", estimated_us=0.05, source_run=9)
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["deadtime_mode"] == "estimate"
    assert resolved["deadtime_estimated_us"] == 0.05
    assert resolved["dead_time_us"] == [0.05, 0.05, 0.05, 0.05]
    assert resolved["deadtime_reference_run"] == 9


def test_resolve_background_fixed_carries_values():
    run = _run(grouping=_per_run_facts())
    profile = _base_profile(
        background_policy=BackgroundPolicy(
            mode="fixed",
            details={"background_fixed_values": [3.0, 4.0], "background_method": "count_fit"},
        )
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["background_correction"] is True
    assert resolved["background_mode"] == "fixed"
    assert resolved["background_fixed_values"] == [3.0, 4.0]
    assert resolved["background_method"] == "count_fit"


def test_resolve_background_reference_run_carries_payload():
    run = _run(grouping=_per_run_facts())
    ref_payload = {"run_number": 5, "source_file": "/x/ref.nxs", "good_frames_reference": 900.0}
    profile = _base_profile(
        background_policy=BackgroundPolicy(
            mode="reference_run", details={"background_run": ref_payload}
        )
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["background_mode"] == "reference_run"
    assert resolved["background_run"] == ref_payload


def test_resolve_out_of_range_detector_ids_do_not_crash_and_are_dropped_at_reduction():
    from asymmetry.core.transform.grouping import group_forward_backward

    run = _run(n_hist=4, grouping=_per_run_facts())
    # Group 2 references detector 9, which this 4-detector run does not have.
    profile = _base_profile(groups={1: [1, 2], 2: [3, 4, 9]})
    resolved = resolve_effective_grouping(profile, run)
    # Resolution keeps the id verbatim; the reduction chokepoint drops it.
    assert resolved["groups"][2] == [3, 4, 9]
    grouped = group_forward_backward(run.histograms, resolved)
    assert len(grouped.backward) > 0  # detector 9 dropped, 3 and 4 still summed


def test_resolve_binning_and_period_mode():
    run = _run(grouping=_per_run_facts())
    profile = _base_profile(
        binning_mode="variable", bin0_us=0.05, bin10_us=0.5, period_mode="green"
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["binning_mode"] == "variable"
    assert resolved["bin0_us"] == 0.05
    assert resolved["bin10_us"] == 0.5
    assert resolved["period_mode"] == "green"


def test_resolve_fixed_binning_omits_width_knobs():
    run = _run(grouping=_per_run_facts())
    profile = _base_profile(binning_mode="fixed", bunching_factor=4)
    resolved = resolve_effective_grouping(profile, run)
    assert "binning_mode" not in resolved
    assert "bin0_us" not in resolved
    assert resolved["bunching_factor"] == 4


# --------------------------------------------------------------------------- #
# profile_from_payload
# --------------------------------------------------------------------------- #


def test_profile_from_payload_lifts_shareable_fields_only():
    payload = {
        **_per_run_facts(),
        "groups": {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "F", 2: "B"},
        "included_groups": {1: True, 2: True},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": [3],
        "alpha": 1.4,
        "alpha_method": "ratio",
        "alpha_reference_run": 12,
        "alpha_error": 0.02,
        "grouping_preset": "Longitudinal",
        "deadtime_correction": True,
        "deadtime_mode": "manual",
        "dead_time_us": [0.01, 0.02, 0.03, 0.04],
        "background_correction": True,
        "background_mode": "fixed",
        "background_fixed_values": [1.0, 2.0],
        "instrument": "EMU",
    }
    profile = profile_from_payload(payload, "P1", ProfileFingerprint("EMU", 4))
    assert profile.groups == {1: [1, 2], 2: [3, 4]}
    assert profile.excluded_detectors == [3]
    assert profile.alpha_policy.mode == "calibrated"
    assert profile.alpha_policy.value == 1.4
    assert profile.alpha_policy.source_run == 12
    assert profile.deadtime_policy.mode == "manual"
    assert profile.deadtime_policy.values == [0.01, 0.02, 0.03, 0.04]
    assert profile.background_policy.mode == "fixed"
    assert profile.background_policy.details["background_fixed_values"] == [1.0, 2.0]
    assert profile.extra["grouping_preset"] == "Longitudinal"
    # Per-run facts must NOT have leaked into the profile serialization.
    serialized = profile.to_dict()
    for per_run_key in ("t0_bin", "first_good_bin", "detector_t0_bins", "good_frames"):
        assert per_run_key not in serialized


def test_profile_from_payload_bare_alpha_is_fixed():
    payload = {"groups": {1: [1], 2: [2]}, "alpha": 1.1, "instrument": "EMU"}
    profile = profile_from_payload(payload, "P", ProfileFingerprint("EMU", 2))
    assert profile.alpha_policy.mode == "fixed"
    assert profile.alpha_policy.value == 1.1


def test_profile_from_payload_round_trips_through_resolve():
    """A payload lifted into a profile and resolved reproduces the shareable keys."""
    payload = {
        **_per_run_facts(),
        "groups": {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.25,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "background_correction": False,
        "instrument": "EMU",
    }
    run = _run(grouping=_per_run_facts())
    profile = profile_from_payload(payload, "P", ProfileFingerprint("EMU", 4))
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["alpha"] == 1.25
    assert resolved["groups"] == {1: [1, 2], 2: [3, 4]}
    assert resolved["forward_group"] == 1
    assert resolved["backward_group"] == 2


# --------------------------------------------------------------------------- #
# Loaded-run inheritance helper
# --------------------------------------------------------------------------- #


def test_loaded_run_inherits_matching_active_profile():
    run = _run(instrument="EMU", n_hist=4, grouping=_per_run_facts())
    profile = _base_profile(alpha_policy=AlphaPolicy(mode="fixed", value=1.7))
    grouping = effective_grouping_for_loaded_run([profile], run)
    assert grouping["alpha"] == 1.7
    assert grouping["groups"] == {1: [1, 2], 2: [3, 4]}


def test_loaded_run_without_matching_profile_keeps_loader_default():
    facts = _per_run_facts()
    facts["alpha"] = 1.0
    run = _run(instrument="EMU", n_hist=4, grouping=facts)
    musr_profile = _base_profile(fingerprint=ProfileFingerprint("MuSR", 4))
    grouping = effective_grouping_for_loaded_run([musr_profile], run)
    # No matching profile → the run's own loader grouping (copied) is returned.
    assert grouping == run.grouping
    assert grouping is not run.grouping  # a copy, safe to mutate


def test_loaded_run_ignores_inactive_matching_profile():
    run = _run(instrument="EMU", n_hist=4, grouping=_per_run_facts())
    inactive = _base_profile(active=False, alpha_policy=AlphaPolicy(mode="fixed", value=9.9))
    grouping = effective_grouping_for_loaded_run([inactive], run)
    assert grouping == run.grouping
