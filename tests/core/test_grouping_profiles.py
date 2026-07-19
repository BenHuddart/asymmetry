"""Tests for project-level grouping profiles (:mod:`asymmetry.core.project.profiles`).

Covers profile round-trip serialization, fingerprint matching, byte-identical
resolution for each alpha/deadtime/background policy mode, and the loaded-run
inheritance helper. These are pure-core tests (no Qt / no GUI).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GroupingProfile,
    ProfileFingerprint,
    T0Policy,
    assigned_profile_for_run,
    default_profile_for_run,
    detect_instrument_for_run,
    effective_grouping_for_loaded_run,
    named_profile_for_run,
    profile_fingerprint_for_run,
    profile_from_payload,
    reconcile_instrument_for_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import (
    EFFECTIVE_DETECTOR_T0_KEY,
    group_forward_backward,
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


def test_default_profile_selection_by_fingerprint():
    emu = _base_profile(name="EMU-A", fingerprint=ProfileFingerprint("EMU", 4), active=True)
    musr = _base_profile(name="MuSR-A", fingerprint=ProfileFingerprint("MuSR", 4), active=True)
    non_default = _base_profile(
        name="EMU-B", fingerprint=ProfileFingerprint("EMU", 4), active=False
    )
    run = _run(instrument="EMU", n_hist=4)
    assert default_profile_for_run([non_default, musr, emu], run) is emu
    assert default_profile_for_run([non_default, musr], run) is None


def test_named_profile_requires_matching_name_and_fingerprint():
    emu = _base_profile(name="Sample A", fingerprint=ProfileFingerprint("EMU", 4), active=True)
    emu_b = _base_profile(name="Sample B", fingerprint=ProfileFingerprint("EMU", 4), active=False)
    musr = _base_profile(name="Sample B", fingerprint=ProfileFingerprint("MuSR", 4), active=True)
    run = _run(instrument="EMU", n_hist=4)
    # Name + fingerprint both match — the default flag is irrelevant.
    assert named_profile_for_run([emu, musr, emu_b], "Sample B", run) is emu_b
    # A same-named profile of another fingerprint never resolves onto the run.
    assert named_profile_for_run([musr], "Sample B", run) is None
    assert named_profile_for_run([emu, emu_b], "Sample C", run) is None


def test_assigned_profile_prefers_assignment_and_falls_back_to_default():
    default = _base_profile(name="Sample A", fingerprint=ProfileFingerprint("EMU", 4), active=True)
    other = _base_profile(name="Sample B", fingerprint=ProfileFingerprint("EMU", 4), active=False)
    run = _run(instrument="EMU", n_hist=4)
    profiles = [default, other]
    # A recorded assignment wins over the default, even when not flagged default.
    assert assigned_profile_for_run(profiles, run, "Sample B") is other
    # No assignment (or a stale name) falls back to the fingerprint's default.
    assert assigned_profile_for_run(profiles, run, None) is default
    assert assigned_profile_for_run(profiles, run, "Deleted sample") is default
    # No default either → None.
    assert assigned_profile_for_run([other], run, "Deleted sample") is None


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


# --------------------------------------------------------------------------- #
# T0Policy
# --------------------------------------------------------------------------- #


def _run_with_detector_t0(detector_t0: list[int]) -> Run:
    """A run whose histograms carry distinct per-detector t0 bins."""
    histograms = [
        Histogram(
            counts=np.arange(10 * i, 10 * i + 20, dtype=float) + 1.0,
            bin_width=0.016,
            t0_bin=t0,
            good_bin_start=t0 + 1,
            good_bin_end=19,
        )
        for i, t0 in enumerate(detector_t0)
    ]
    facts = {
        "instrument": "EMU",
        "t0_bin": max(detector_t0),
        "first_good_bin": max(detector_t0) + 1,
        "last_good_bin": 19,
        "detector_t0_bins": list(detector_t0),
    }
    return Run(run_number=1, histograms=histograms, grouping=facts, metadata={"instrument": "EMU"})


def test_t0_policy_round_trips_each_mode():
    for policy in (
        T0Policy(mode="from_file"),
        T0Policy(mode="manual", value=7),
        T0Policy(mode="auto_detect", strategy="prompt_peak", spread_bins=2, source_run=3),
    ):
        assert T0Policy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()


def test_t0_policy_from_dict_rejects_unknown_mode():
    assert T0Policy.from_dict({"mode": "bogus"}).mode == "from_file"


def test_t0_policy_default_is_omitted_from_profile_dict():
    """A from_file (default) t0 policy leaves no ``t0_policy`` key — no schema bump."""
    profile = _base_profile()
    assert profile.t0_policy.mode == "from_file"
    assert "t0_policy" not in profile.to_dict()


def test_t0_policy_manual_serializes_in_profile_dict():
    profile = _base_profile(t0_policy=T0Policy(mode="manual", value=9))
    data = profile.to_dict()
    assert data["t0_policy"] == {"mode": "manual", "value": 9}
    assert GroupingProfile.from_dict(data).t0_policy.to_dict() == {"mode": "manual", "value": 9}


def test_t0_from_file_is_bit_identical_to_default_resolution():
    """from_file resolution reproduces today's file-derived t0 payload exactly."""
    run = _run(grouping=_per_run_facts())
    default_profile = _base_profile()  # T0Policy() defaults to from_file
    resolved = resolve_effective_grouping(default_profile, run)
    assert resolved["t0_bin"] == 5
    assert resolved["first_good_bin"] == 6
    assert resolved["last_good_bin"] == 19
    assert resolved["detector_t0_bins"] == [5, 5, 5, 5]
    assert EFFECTIVE_DETECTOR_T0_KEY not in resolved


def test_t0_manual_shifts_common_t0_and_publishes_effective_bins_non_destructively():
    """Manual t0 offsets each detector's file t0 in the payload only."""
    run = _run_with_detector_t0([5, 5, 6, 6])  # file common t0 (max over groups) = 6
    before = [int(h.t0_bin) for h in run.histograms]
    profile = _base_profile(t0_policy=T0Policy(mode="manual", value=9))
    resolved = resolve_effective_grouping(profile, run)

    # delta = 9 - 6 = 3; each detector's file t0 is shifted by +3.
    assert resolved["t0_bin"] == 9
    assert resolved[EFFECTIVE_DETECTOR_T0_KEY] == [8, 8, 9, 9]
    # first_good_bin shifts with t0 so the good-window offset is preserved.
    assert resolved["first_good_bin"] == 7 + 3
    # The run's histograms are UNCHANGED — nothing was rewritten.
    assert [int(h.t0_bin) for h in run.histograms] == before == [5, 5, 6, 6]


def test_t0_manual_effective_bins_drive_reduction_alignment():
    """The effective per-detector t0 override changes what reduction aligns to."""
    run = _run_with_detector_t0([5, 5, 5, 5])
    file_profile = _base_profile()
    manual_profile = _base_profile(t0_policy=T0Policy(mode="manual", value=8))

    file_grouped = group_forward_backward(
        run.histograms, resolve_effective_grouping(file_profile, run)
    )
    manual_grouped = group_forward_backward(
        run.histograms, resolve_effective_grouping(manual_profile, run)
    )
    # All detectors shifted by the same delta → common_t0 moves by the delta.
    assert file_grouped.common_t0 == 5
    assert manual_grouped.common_t0 == 8


def test_t0_manual_matching_file_value_is_a_no_op():
    run = _run_with_detector_t0([5, 5, 6, 6])  # file common t0 = 6
    profile = _base_profile(t0_policy=T0Policy(mode="manual", value=6))
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["t0_bin"] == 6
    assert EFFECTIVE_DETECTOR_T0_KEY not in resolved  # delta == 0, no override


def test_t0_auto_detect_runs_search_per_run():
    """auto_detect fills t0 from a synthetic prompt-peak run and records provenance."""
    # Continuous (PSI) source → prompt-peak strategy → argmax bin.
    peak_bin = 7
    counts = np.ones(20, dtype=float)
    counts[peak_bin] = 100.0
    histograms = [
        Histogram(
            counts=counts.copy(), bin_width=0.016, t0_bin=3, good_bin_start=4, good_bin_end=19
        )
        for _ in range(4)
    ]
    run = Run(
        run_number=1,
        histograms=histograms,
        grouping={"instrument": "EMU", "t0_bin": 3, "detector_t0_bins": [3, 3, 3, 3]},
        metadata={"instrument": "EMU", "facility": "PSI"},
    )
    profile = _base_profile(t0_policy=T0Policy(mode="auto_detect"))
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["t0_bin"] == peak_bin
    assert resolved["t0_search_strategy"] == "prompt_peak"
    assert resolved["t0_search_spread_bins"] == 0
    # Effective override shifts every detector from file t0 3 -> 7.
    assert resolved[EFFECTIVE_DETECTOR_T0_KEY] == [7, 7, 7, 7]
    # Histograms untouched.
    assert all(int(h.t0_bin) == 3 for h in run.histograms)


# --------------------------------------------------------------------------- #
# profile_from_payload / migration inference for t0
# --------------------------------------------------------------------------- #


def test_profile_from_payload_infers_from_file_when_t0_matches_file():
    payload = {
        "groups": {1: [1], 2: [2]},
        "instrument": "EMU",
        "t0_bin": 6,
        "detector_t0_bins": [5, 5, 6, 6],  # file common t0 = 6 == stored t0
    }
    profile = profile_from_payload(payload, "P", ProfileFingerprint("EMU", 4))
    assert profile.t0_policy.mode == "from_file"


def test_profile_from_payload_infers_manual_when_t0_differs_from_file():
    payload = {
        "groups": {1: [1], 2: [2]},
        "instrument": "EMU",
        "t0_bin": 9,
        "detector_t0_bins": [5, 5, 6, 6],  # file common t0 = 6 != stored 9
    }
    profile = profile_from_payload(payload, "P", ProfileFingerprint("EMU", 4))
    assert profile.t0_policy.mode == "manual"
    assert profile.t0_policy.value == 9


def test_profile_from_payload_infers_manual_from_effective_override():
    payload = {
        "groups": {1: [1], 2: [2]},
        "instrument": "EMU",
        "t0_bin": 8,
        "effective_detector_t0_bins": [8, 8, 8, 8],
    }
    profile = profile_from_payload(payload, "P", ProfileFingerprint("EMU", 4))
    assert profile.t0_policy.mode == "manual"


def test_profile_from_payload_no_detector_table_is_from_file():
    """A common-t0 NeXus payload (no per-detector table) has no manual signal."""
    payload = {"groups": {1: [1], 2: [2]}, "instrument": "EMU", "t0_bin": 40}
    profile = profile_from_payload(payload, "P", ProfileFingerprint("EMU", 2))
    assert profile.t0_policy.mode == "from_file"


# --------------------------------------------------------------------------- #
# Instrument self-healing on re-detection
# --------------------------------------------------------------------------- #

#: The real FLAME eight-counter label set (Forw/Back/Righ/Left + the four split
#: corner plates) from the investigation. ``detect_instrument`` returns FLAME for
#: this label set even when the instrument token is a generic "PSI".
_FLAME_LABELS = ["Forw", "Back", "Righ", "Left", "R_F", "R_B", "L_F", "L_B"]


def _flame_run(*, grouping_instrument: str) -> Run:
    """An 8-histogram run whose fresh detection is FLAME but whose grouping
    carries *grouping_instrument* (e.g. a stale "GPS")."""
    return _run(
        instrument=grouping_instrument,
        n_hist=8,
        grouping={"instrument": grouping_instrument},
        metadata={
            "facility": "PSI",
            "instrument": "PSI",
            "histogram_labels": list(_FLAME_LABELS),
        },
    )


def test_detect_instrument_for_run_uses_run_metadata():
    run = _flame_run(grouping_instrument="GPS")
    assert detect_instrument_for_run(run) == "FLAME"


def test_reconcile_heals_stale_instrument_and_discards_structure():
    """A FLAME run persisted as GPS heals to FLAME and drops GPS structure."""
    run = _flame_run(grouping_instrument="GPS")
    stale_payload = {
        "instrument": "GPS",
        "groups": {1: [1], 2: [2]},
        "group_names": {1: "Up", 2: "Down"},
        "forward_group": 1,
        "backward_group": 2,
        "grouping_preset": "Transverse (Vector)",
        "projections": [{"label": "Up-Down", "forward_group": 1, "backward_group": 2}],
        "vector_axis": "P_y",
        # Instrument-independent settings that must survive.
        "t0_bin": 5,
        "first_good_bin": 6,
        "alpha": 1.23,
        "deadtime_mode": "off",
    }
    reconciled, note = reconcile_instrument_for_payload(run, stale_payload)

    assert note is not None
    assert "GPS" in note and "FLAME" in note
    # The run's own identity is corrected in place.
    assert run.grouping["instrument"] == "FLAME"
    assert run.metadata["instrument"] == "FLAME"
    # The fingerprint now reports FLAME (8), not GPS.
    assert profile_fingerprint_for_run(run) == ProfileFingerprint("FLAME", 8)
    # Stale GPS structure is gone; instrument-independent settings kept.
    assert reconciled["instrument"] == "FLAME"
    for key in (
        "groups",
        "group_names",
        "forward_group",
        "backward_group",
        "grouping_preset",
        "projections",
        "vector_axis",
    ):
        assert key not in reconciled
    assert reconciled["t0_bin"] == 5
    assert reconciled["first_good_bin"] == 6
    assert reconciled["alpha"] == 1.23
    assert reconciled["deadtime_mode"] == "off"


def test_reconcile_does_not_inherit_wrong_fingerprint_profile():
    """After healing, the run must not inherit a (GPS, 8) profile."""
    run = _flame_run(grouping_instrument="GPS")
    gps_profile = GroupingProfile(
        name="Default (GPS)",
        fingerprint=ProfileFingerprint("GPS", 8),
        groups={1: [1], 2: [2]},
        group_names={1: "Up", 2: "Down"},
    )
    reconcile_instrument_for_payload(run, {"instrument": "GPS", "groups": {1: [1]}})
    assert default_profile_for_run([gps_profile], run) is None


def test_reconcile_none_detection_keeps_stored_value():
    """When detection is inconclusive, the stored instrument is kept."""
    # A 2-histogram run with no instrument hints -> detect_instrument returns None.
    run = _run(instrument="GPS", n_hist=2, metadata={"instrument": ""})
    assert detect_instrument_for_run(run) is None
    payload = {"instrument": "GPS", "groups": {1: [1], 2: [2]}}
    reconciled, note = reconcile_instrument_for_payload(run, payload)
    assert note is None
    assert reconciled is payload  # unchanged object
    assert run.grouping["instrument"] == "GPS"


def test_reconcile_matching_value_leaves_payload_byte_identical():
    """A payload whose stored instrument matches detection is untouched."""
    run = _flame_run(grouping_instrument="FLAME")
    payload = {
        "instrument": "FLAME",
        "groups": {1: [1], 2: [2]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 2,
        "backward_group": 1,
        "alpha": 1.0,
    }
    snapshot = copy.deepcopy(payload)
    reconciled, note = reconcile_instrument_for_payload(run, payload)
    assert note is None
    assert reconciled is payload  # same object, not a copy
    assert payload == snapshot  # byte-identical


def test_reconcile_gps_variants_are_same_family_no_heal():
    """GPS vs GPS-RD are variants of one instrument; never a spurious heal."""
    # An 11-histogram GPS ROOT run detects as "GPS-RD"; a payload stored as the
    # display-name "GPS" must not be "healed" to GPS-RD.
    run = _run(
        instrument="GPS",
        n_hist=11,
        metadata={
            "facility": "PSI",
            "psi_format": "psi-root",
            "instrument": "LMU_BULKMUSR_GPS",
            "histogram_labels": [
                "Forw",
                "Back",
                "Up_B",
                "Up_F",
                "Down_B",
                "Down_F",
                "Right_B",
                "Right_F",
                "Left_B",
                "Left_F",
                "Mob-RL",
            ],
        },
    )
    assert detect_instrument_for_run(run) == "GPS-RD"
    payload = {"instrument": "GPS", "groups": {1: [1]}}
    reconciled, note = reconcile_instrument_for_payload(run, payload)
    assert note is None
    assert reconciled is payload


def test_profile_color_round_trips_and_defaults_none():
    profile = _base_profile(color="#5f4a87")
    data = profile.to_dict()
    assert data["color"] == "#5f4a87"
    assert GroupingProfile.from_dict(data).color == "#5f4a87"
    colorless = _base_profile()
    assert "color" not in colorless.to_dict()
    assert GroupingProfile.from_dict(colorless.to_dict()).color is None
