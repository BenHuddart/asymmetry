"""Released instrument choices migrate without rewriting scientific payloads or raw hints."""

import copy

import numpy as np
import pytest

from asymmetry.core.instruments.resolution import (
    RecordedInstrumentChoice,
    UnresolvedInstrumentIdentity,
)
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    load_project,
    migrate_to_current,
    save_project,
)


def saved_overrides(instrument, count):
    return {
        "instrument": instrument,
        "groups": {"1": [1], "2": [2]},
        "group_names": {"1": "Authored positive", "2": "Authored negative"},
        "forward_group": 2,
        "backward_group": 1,
        "alpha": 1.23,
        "beta": 0.91,
        "detector_t0_bins": [2] * count,
        "t0_policy": {"mode": "manual", "offset_bins": 3},
        "good_window_policy": {"mode": "manual", "first_bin": 2, "last_bin": 4},
        "background_correction": False,
        "deadtime_correction": False,
    }


def state40(entries, profiles=()):
    return {
        "schema_version": 40,
        "datasets": entries,
        "combined_datasets": [],
        "grouping_profiles": list(profiles),
    }


def profile40(name, instrument, count, *, active=True):
    return {
        "name": name,
        "fingerprint": {"instrument": instrument, "histogram_count": count},
        "active": active,
        "groups": {"1": [1], "2": [2]},
        "group_names": {"1": "Authored positive", "2": "Authored negative"},
        "forward_group": 2,
        "backward_group": 1,
        "alpha_policy": {"mode": "fixed", "value": 1.23},
    }


@pytest.mark.parametrize(
    "label,count,key,acquisition",
    (
        ("EMU", 96, "isis.emu", "nexus96"),
        ("emu", 96, "isis.emu", "nexus96"),
        ("HIFI", 64, "isis.hifi", "nexus64"),
        ("HiFi", 64, "isis.hifi", "nexus64"),
        ("MuSR", 64, "isis.musr", "nexus64"),
        ("FLAME", 8, "psi.flame", "eight_channel"),
        ("HAL", 17, "psi.hal9500", "full17"),
        ("GPS", 6, "psi.gps", "bin_combined"),
        ("gps-rd", 11, "psi.gps", "root_split"),
        ("GPS-RD15", 15, "psi.gps", "root_combined_split"),
        ("LEM", 7, "psi.lem", {"unavailable": "undeclared"}),
    ),
)
def test_stored_layout_choices_have_historical_namespace_and_revision(
    label, count, key, acquisition
):
    overrides = saved_overrides(label, count)
    original = state40(
        [{"run_number": 1, "source_file": "not-opened.nxs", "grouping_overrides": overrides}]
    )
    before = copy.deepcopy(original)
    migrated = migrate_to_current(original)
    entry = migrated["datasets"][0]
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert entry["instrument_choice"]["selection"] == {
        "profile_key": key,
        "acquisition": acquisition,
        "profile_revision": "1",
        "origin": "legacy",
    }
    assert entry["grouping_overrides"] == {k: v for k, v in overrides.items() if k != "instrument"}
    assert "source_assessment" not in entry
    assert original == before
    assert migrate_to_current(migrated) == migrated


def test_unknown_stored_identity_remains_opaque_and_retains_every_other_field():
    overrides = saved_overrides("  Custom spectrometer  ", 7)
    overrides.update(
        exclusions=[3],
        included_groups=[2, 1],
        projections=[{"label": "Author", "forward_group": 2, "backward_group": 1, "alpha": 0.75}],
        calibration_record={
            "unavailable_request": {
                "timing": {"source": {"pulsed": None, "reason": "absent", "origin": "metadata"}}
            }
        },
        background_run={
            "run_number": 42,
            "timing": {"t0_policy": {"mode": "manual", "offset_bins": None}},
        },
        extra={"instrument": "HIFI", "scalar": 7},
    )
    entry = {
        "run_number": 1,
        "source_file": "not-opened.root",
        "profile": "Kept base",
        "grouping_overrides": overrides,
        "analysis_history": {"instrument": "GPS-RD", "recipe": {"instrument": "HIFI"}},
        "metadata_overrides": {"instrument": "HIFI", "facility": "PSI"},
    }
    original = state40([entry])
    migrated = migrate_to_current(original)
    updated = migrated["datasets"][0]
    choice = RecordedInstrumentChoice.from_dict(updated["instrument_choice"])
    assert isinstance(choice.selection, UnresolvedInstrumentIdentity)
    assert choice.selection.label == "  Custom spectrometer  "
    expected = copy.deepcopy(entry)
    expected["grouping_overrides"].pop("instrument")
    assert {k: v for k, v in updated.items() if k != "instrument_choice"} == expected
    assert original["datasets"][0] == entry


def test_stored_hifi_layout_and_psi_raw_hifi_are_different_evidence():
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.instruments.resolution import resolve_instrument

    entry = {
        "run_number": 1,
        "source_file": "not-opened.root",
        "grouping_overrides": saved_overrides("HIFI", 64),
        "metadata_overrides": {"instrument": "HIFI", "facility": "PSI"},
    }
    migrated = migrate_to_current(state40([entry]))["datasets"][0]
    assert migrated["instrument_choice"]["selection"]["profile_key"] == "isis.hifi"
    assert migrated["metadata_overrides"] == entry["metadata_overrides"]
    run = Run(
        1,
        [Histogram(np.ones(4), 0.1) for _ in range(64)],
        metadata=entry["metadata_overrides"],
        instrument_choice=RecordedInstrumentChoice.from_dict(migrated["instrument_choice"]),
    )
    context = resolve_instrument(run)
    assert context.requires_choice
    assert context.observed.selection.profile_key == "psi.hal9500"
    assert run.instrument_choice.selection.profile_key == "isis.hifi"


def test_following_and_released_names_and_profile_payloads_survive_case_variants():
    profiles = [profile40("Sample", "eMu", 96), profile40("Other", "EMU", 96, active=False)]
    overrides = saved_overrides("EMU", 96)
    original = state40(
        [
            {"run_number": 1, "source_file": "a.nxs", "profile": "Sample"},
            {
                "run_number": 2,
                "source_file": "b.nxs",
                "profile": "Other",
                "grouping_overrides": overrides,
            },
        ],
        profiles,
    )
    migrated = migrate_to_current(original)
    assert migrated["datasets"][0]["profile"] == "Sample"
    assert "grouping_overrides" not in migrated["datasets"][0]
    assert migrated["datasets"][1]["profile"] == "Other"
    for index, profile in enumerate(migrated["grouping_profiles"]):
        assert profile["fingerprint"] == {
            "instrument": "isis.emu",
            "histogram_count": 96,
            "acquisition": "nexus96",
        }
        assert {k: v for k, v in profile.items() if k != "fingerprint"} == {
            k: v for k, v in profiles[index].items() if k != "fingerprint"
        }
    assert [
        row["instrument_choice"]["selection"]["profile_key"] for row in migrated["datasets"]
    ] == ["isis.emu", "isis.emu"]


def test_saved_project_roundtrip_prepares_current_fingerprints_without_file_access(tmp_path):
    project = tmp_path / "legacy-instrument.asymp"
    original = state40(
        [{"run_number": 1, "source_file": "not-opened.nxs", "profile": "Sample"}],
        [profile40("Sample", "EMU", 96)],
    )
    save_project(original, project)
    migrated = load_project(project)
    assert migrated["datasets"][0]["instrument_choice"]["selection"]["profile_key"] == "isis.emu"
    save_project(migrated, project)
    assert load_project(project) == migrated
    assert "source_assessment" not in migrated["datasets"][0]


@pytest.mark.parametrize(
    "label,count,acquisition",
    (
        ("GPS", 11, "root_split"),
        ("GPS", 15, "root_combined_split"),
        ("GPS-RD", 15, {"unavailable": "incompatible"}),
        ("GPS-RD15", 11, {"unavailable": "incompatible"}),
        ("EMU", 4, {"unavailable": "incompatible"}),
    ),
)
def test_saved_count_selects_gps_family_without_switching_explicit_map(label, count, acquisition):
    migrated = migrate_to_current(
        state40(
            [
                {
                    "run_number": 1,
                    "source_file": "gps15.root",
                    "grouping_overrides": saved_overrides(label, count),
                }
            ]
        )
    )
    assert migrated["datasets"][0]["instrument_choice"]["selection"]["acquisition"] == acquisition


@pytest.mark.parametrize("label,acquisition", (("EMU", "nexus96"), ("GPS-RD", "root_split")))
def test_missing_count_keeps_recorded_map_without_inventing_acquired_count(label, acquisition):
    overrides = saved_overrides(label, 7)
    overrides.pop("detector_t0_bins")
    migrated = migrate_to_current(
        state40(
            [{"run_number": 1, "source_file": "gps15.root", "grouping_overrides": overrides}],
            [profile40("Named", label, 0)],
        )
    )
    assert migrated["datasets"][0]["instrument_choice"]["selection"]["acquisition"] == acquisition
    assert migrated["grouping_profiles"][0]["fingerprint"]["histogram_count"] == 0
    assert migrated["grouping_profiles"][0]["fingerprint"]["acquisition"] == acquisition


def test_ambiguous_same_named_followers_remain_pending_without_a_default_choice():
    profiles = [profile40("Shared", "EMU", 96), profile40("Shared", "HIFI", 64)]
    original = state40(
        [{"run_number": 1, "source_file": "emu123.nxs", "profile": "Shared"}], profiles
    )
    migrated = migrate_to_current(original)
    entry = migrated["datasets"][0]
    choice = RecordedInstrumentChoice.from_dict(entry["instrument_choice"])
    assert isinstance(choice.selection, UnresolvedInstrumentIdentity)
    assert choice.selection.label == "profile:Shared"
    assert ("profile", "Shared") in choice.evidence
    assert entry["profile"] == "Shared"
    assert len(migrated["grouping_profiles"]) == 2
    assert [profile["active"] for profile in migrated["grouping_profiles"]] == [True, True]
    assert original["grouping_profiles"] == profiles


def test_released_override_without_active_identity_does_not_adopt_reattach_profile():
    overrides = {"alpha": 1.9, "groups": {"1": [1], "2": [2]}, "detector_t0_bins": [2] * 96}
    migrated = migrate_to_current(
        state40(
            [
                {
                    "run_number": 1,
                    "source_file": "a.nxs",
                    "profile": "Reattach",
                    "grouping_overrides": overrides,
                }
            ],
            [profile40("Reattach", "EMU", 96)],
        )
    )
    entry = migrated["datasets"][0]
    assert entry["profile"] == "Reattach"
    assert entry["grouping_overrides"] == overrides
    assert entry["instrument_choice"]["selection"] == {"unresolved": "profile:Reattach"}


def test_raw_metadata_and_channel_count_do_not_manufacture_a_recorded_choice():
    overrides = saved_overrides("EMU", 96)
    overrides.pop("instrument")
    entry = {
        "run_number": 1,
        "source_file": "emu123.nxs",
        "grouping_overrides": overrides,
        "metadata_overrides": {"facility": "PSI", "instrument": "HIFI"},
    }
    migrated = migrate_to_current(state40([entry]))
    assert migrated["datasets"][0] == entry
    assert "source_assessment" not in migrated["datasets"][0]


def test_combined_entries_and_preexisting_captured_source_are_preserved():
    source = {"pulsed": None, "reason": "absent", "origin": "metadata"}
    overrides = saved_overrides("HIFI", 64)
    entry = {
        "run_number": 1000,
        "source_run_numbers": [1, 2],
        "grouping_overrides": overrides,
        "source_assessment": source,
        "analysis_history": {"instrument": "HAL", "recipe": {"instrument": "HIFI"}},
    }
    original = state40([])
    original["combined_datasets"] = [entry]
    migrated = migrate_to_current(original)
    updated = migrated["combined_datasets"][0]
    assert updated["instrument_choice"]["selection"]["profile_key"] == "isis.hifi"
    expected = copy.deepcopy(entry)
    expected["grouping_overrides"].pop("instrument")
    assert {key: value for key, value in updated.items() if key != "instrument_choice"} == expected
    assert updated["source_assessment"] == source


def test_v11_profile_conversion_uses_released_fingerprint_shape_before_canonicalization():
    original = {
        "schema_version": 11,
        "datasets": [
            {
                "run_number": 1,
                "source_file": "a.nxs",
                "grouping_overrides": saved_overrides("EMU", 96),
            },
            {
                "run_number": 2,
                "source_file": "b.nxs",
                "grouping_overrides": saved_overrides("EMU", 96),
            },
        ],
        "combined_datasets": [],
    }
    migrated = migrate_to_current(original)
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    profile = migrated["grouping_profiles"][0]
    assert profile["name"] == "Default (EMU)"
    assert profile["fingerprint"] == {
        "instrument": "isis.emu",
        "histogram_count": 96,
        "acquisition": "nexus96",
    }
    for entry in migrated["datasets"]:
        assert entry["profile"] == "Default (EMU)"
        assert "grouping_overrides" not in entry
        assert entry["instrument_choice"]["selection"]["profile_key"] == "isis.emu"
