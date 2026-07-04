"""Tests for the v11->v12 grouping-profile schema migration.

Pure-core: exercises :func:`asymmetry.core.project.schema.migrate_to_current`
directly (no GUI). Covers the all-identical collapse, divergence-keeps-overrides,
multi-instrument bucketing, the missing-metadata conservative path, and that a
v11 project migrates without altering per-dataset resolution behaviour.
"""

from __future__ import annotations

import copy

from asymmetry.core.project.profiles import (
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    migrate_to_current,
    validate,
)


def _v11_state(datasets: list[dict]) -> dict:
    return {
        "schema_version": 11,
        "created_with_app_version": "0.1.0",
        "datasets": datasets,
        "combined_datasets": [],
    }


def _overrides(
    *,
    instrument: str = "EMU",
    n_hist: int = 4,
    alpha: float = 1.0,
    groups: dict | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "groups": groups or {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": alpha,
        "t0_bin": 5,
        "t_good_offset": 1,
        "first_good_bin": 6,
        "last_good_bin": 250,
        "bin_index_base": 0,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "background_correction": False,
        # A per-detector list makes the histogram count explicit for fingerprinting.
        "detector_t0_bins": [5] * n_hist,
        "instrument": instrument,
    }
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------- #
# All-identical collapse
# --------------------------------------------------------------------------- #


def test_identical_runs_collapse_to_single_active_profile():
    state = _v11_state(
        [
            {"run_number": 1, "source_file": "a.nxs", "grouping_overrides": _overrides(alpha=1.2)},
            {"run_number": 2, "source_file": "b.nxs", "grouping_overrides": _overrides(alpha=1.2)},
            {"run_number": 3, "source_file": "c.nxs", "grouping_overrides": _overrides(alpha=1.2)},
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    validate(migrated)

    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    profiles = migrated["grouping_profiles"]
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["name"] == "Default (EMU)"
    assert profile["active"] is True
    assert profile["fingerprint"] == {"instrument": "EMU", "histogram_count": 4}
    assert profile["alpha_policy"] == {"mode": "fixed", "value": 1.2}

    # Each dataset now references the profile and drops its per-run copy.
    for ds in migrated["datasets"]:
        assert ds.get("profile") == "Default (EMU)"
        assert "grouping_overrides" not in ds


# --------------------------------------------------------------------------- #
# Divergence
# --------------------------------------------------------------------------- #


def test_divergent_runs_use_majority_profile_and_keep_outliers():
    state = _v11_state(
        [
            {"run_number": 1, "source_file": "a.nxs", "grouping_overrides": _overrides(alpha=1.0)},
            {"run_number": 2, "source_file": "b.nxs", "grouping_overrides": _overrides(alpha=1.0)},
            # Divergent: different alpha → keeps its own overrides.
            {"run_number": 3, "source_file": "c.nxs", "grouping_overrides": _overrides(alpha=2.5)},
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))

    assert len(migrated["grouping_profiles"]) == 1
    assert migrated["grouping_profiles"][0]["alpha_policy"]["value"] == 1.0

    by_run = {ds["run_number"]: ds for ds in migrated["datasets"]}
    assert by_run[1]["profile"] == "Default (EMU)"
    assert "grouping_overrides" not in by_run[1]
    assert by_run[2]["profile"] == "Default (EMU)"
    # Outlier keeps its full payload and is NOT bound to the profile.
    assert "profile" not in by_run[3]
    assert by_run[3]["grouping_overrides"]["alpha"] == 2.5


def test_majority_tie_breaks_to_first_run():
    state = _v11_state(
        [
            {"run_number": 1, "source_file": "a.nxs", "grouping_overrides": _overrides(alpha=1.0)},
            {"run_number": 2, "source_file": "b.nxs", "grouping_overrides": _overrides(alpha=2.0)},
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    # 1-1 tie → the first run's payload wins the profile.
    assert migrated["grouping_profiles"][0]["alpha_policy"]["value"] == 1.0
    by_run = {ds["run_number"]: ds for ds in migrated["datasets"]}
    assert by_run[1].get("profile") == "Default (EMU)"
    assert "profile" not in by_run[2]
    assert by_run[2]["grouping_overrides"]["alpha"] == 2.0


# --------------------------------------------------------------------------- #
# Multi-instrument
# --------------------------------------------------------------------------- #


def test_multi_instrument_project_yields_one_profile_per_fingerprint():
    state = _v11_state(
        [
            {
                "run_number": 1,
                "source_file": "e.nxs",
                "grouping_overrides": _overrides(instrument="EMU", n_hist=4),
            },
            {
                "run_number": 2,
                "source_file": "m.nxs",
                "grouping_overrides": _overrides(instrument="MuSR", n_hist=64),
            },
            {
                "run_number": 3,
                "source_file": "e2.nxs",
                "grouping_overrides": _overrides(instrument="EMU", n_hist=4),
            },
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    profiles = {p["name"]: p for p in migrated["grouping_profiles"]}
    assert set(profiles) == {"Default (EMU)", "Default (MuSR)"}
    assert profiles["Default (EMU)"]["fingerprint"]["histogram_count"] == 4
    assert profiles["Default (MuSR)"]["fingerprint"]["histogram_count"] == 64
    for ds in migrated["datasets"]:
        assert "profile" in ds and "grouping_overrides" not in ds


def test_same_instrument_different_histogram_count_are_distinct_fingerprints():
    state = _v11_state(
        [
            {
                "run_number": 1,
                "source_file": "a.nxs",
                "grouping_overrides": _overrides(instrument="GPS", n_hist=4),
            },
            {
                "run_number": 2,
                "source_file": "b.nxs",
                "grouping_overrides": _overrides(
                    instrument="GPS", n_hist=8, groups={1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}
                ),
            },
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    assert len(migrated["grouping_profiles"]) == 2
    counts = sorted(p["fingerprint"]["histogram_count"] for p in migrated["grouping_profiles"])
    assert counts == [4, 8]


# --------------------------------------------------------------------------- #
# Missing-metadata conservative path
# --------------------------------------------------------------------------- #


def test_missing_instrument_keeps_overrides_and_creates_no_profile():
    overrides = _overrides()
    overrides.pop("instrument")
    state = _v11_state([{"run_number": 1, "source_file": "a.nxs", "grouping_overrides": overrides}])
    migrated = migrate_to_current(copy.deepcopy(state))
    assert migrated["grouping_profiles"] == []
    ds = migrated["datasets"][0]
    assert "profile" not in ds
    assert ds["grouping_overrides"]["alpha"] == 1.0


def test_missing_histogram_count_keeps_overrides():
    # No per-detector list and no groups → histogram count is unknowable.
    overrides = {
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "instrument": "EMU",
    }
    state = _v11_state([{"run_number": 1, "source_file": "a.nxs", "grouping_overrides": overrides}])
    migrated = migrate_to_current(copy.deepcopy(state))
    assert migrated["grouping_profiles"] == []
    assert "grouping_overrides" in migrated["datasets"][0]


def test_histogram_count_inferred_from_groups_when_no_detector_list():
    overrides = _overrides(groups={1: [1, 2], 2: [3, 4]})
    overrides.pop("detector_t0_bins")  # force the group-based fallback
    state = _v11_state([{"run_number": 1, "source_file": "a.nxs", "grouping_overrides": overrides}])
    migrated = migrate_to_current(copy.deepcopy(state))
    assert len(migrated["grouping_profiles"]) == 1
    assert migrated["grouping_profiles"][0]["fingerprint"]["histogram_count"] == 4


def test_dataset_without_grouping_overrides_is_untouched():
    state = _v11_state(
        [{"run_number": 1, "source_file": "a.nxs", "metadata_overrides": {"field": 100.0}}]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    assert migrated["grouping_profiles"] == []
    ds = migrated["datasets"][0]
    assert "profile" not in ds
    assert ds["metadata_overrides"] == {"field": 100.0}


# --------------------------------------------------------------------------- #
# Behavioural equivalence: migrated profile resolves back to the same shareable
# grouping the v11 override carried.
# --------------------------------------------------------------------------- #


def test_v11_project_migrates_to_v12_preserving_other_state():
    state = _v11_state([])
    state["fit_states"] = {"time": {"domain": "time"}, "frequency": {"domain": "frequency"}}
    state["browser_state"] = {"extra_columns": [{"id": "x", "label": "X", "kind": "custom"}]}
    migrated = migrate_to_current(copy.deepcopy(state))
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrated["grouping_profiles"] == []
    # Unrelated top-level state is preserved verbatim.
    assert migrated["fit_states"] == state["fit_states"]
    assert migrated["browser_state"] == state["browser_state"]


def test_migrated_profile_resolves_to_original_shareable_grouping():
    import numpy as np

    from asymmetry.core.data.dataset import Histogram, Run

    original = _overrides(alpha=1.35)
    state = _v11_state(
        [
            {
                "run_number": 1,
                "source_file": "a.nxs",
                "grouping_overrides": copy.deepcopy(original),
            },
            {
                "run_number": 2,
                "source_file": "b.nxs",
                "grouping_overrides": copy.deepcopy(original),
            },
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    profile = GroupingProfile.from_dict(migrated["grouping_profiles"][0])

    run = Run(
        run_number=1,
        histograms=[Histogram(counts=np.ones(20), bin_width=0.016, t0_bin=5) for _ in range(4)],
        grouping={
            "instrument": "EMU",
            "t0_bin": 5,
            "first_good_bin": 6,
            "last_good_bin": 19,
        },
    )
    resolved = resolve_effective_grouping(profile, run)
    assert resolved["alpha"] == 1.35
    assert resolved["groups"] == original["groups"]
    assert resolved["forward_group"] == 1
    assert resolved["backward_group"] == 2
    # Fingerprint carries through so a reload re-inherits the profile.
    assert profile.fingerprint == ProfileFingerprint("EMU", 4)


# --------------------------------------------------------------------------- #
# t0 policy inference during migration
# --------------------------------------------------------------------------- #


def test_migration_infers_from_file_t0_when_stored_matches_file():
    """A payload whose t0 equals its file common t0 migrates to from_file (default)."""
    state = _v11_state(
        [
            {"run_number": 1, "source_file": "a.nxs", "grouping_overrides": _overrides()},
            {"run_number": 2, "source_file": "b.nxs", "grouping_overrides": _overrides()},
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    profile = migrated["grouping_profiles"][0]
    # from_file is the default → no t0_policy key stored (no schema bump).
    assert "t0_policy" not in profile
    assert GroupingProfile.from_dict(profile).t0_policy.mode == "from_file"


def test_migration_infers_manual_t0_when_stored_differs_from_file():
    """A payload whose common t0 was shifted from the file t0 migrates to manual."""
    shifted = _overrides(extra={"t0_bin": 9})  # detector_t0_bins stay [5,5,5,5]
    state = _v11_state(
        [
            {"run_number": 1, "source_file": "a.nxs", "grouping_overrides": shifted},
            {"run_number": 2, "source_file": "b.nxs", "grouping_overrides": copy.deepcopy(shifted)},
        ]
    )
    migrated = migrate_to_current(copy.deepcopy(state))
    profile = migrated["grouping_profiles"][0]
    assert profile["t0_policy"] == {"mode": "manual", "value": 9}
    assert GroupingProfile.from_dict(profile).t0_policy.value == 9
