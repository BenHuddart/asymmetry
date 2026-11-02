"""Saved objectives are inferred only where the producing fit path is known."""

from copy import deepcopy

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, migrate_to_current
from asymmetry.core.representation import FitSeries, FitSlot, RepresentationType


def _summary():
    return {
        "success": True,
        "chi_squared": 12.0,
        "reduced_chi_squared": 1.2,
        "parameters": {},
        "quality": {"verdict": "good"},
    }


def test_v27_migration_records_known_gaussian_and_unknown_group_objectives():
    slot = FitSlot(result=_summary()).to_dict()
    project = {
        "schema_version": 27,
        "datasets": [{"nested": {"slot": slot}}],
        "batches": [
            {"member_kind": kind, "results_by_run": {"1": _summary()}}
            for kind in ["runs", "groups"]
        ],
    }
    original = deepcopy(project)
    migrated = migrate_to_current(project)
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    restored_slot = FitSlot.from_dict(migrated["datasets"][0]["nested"]["slot"])
    assert restored_slot.result["objective"] == "gaussian"
    runs, groups = migrated["batches"]
    assert runs["results_by_run"]["1"]["objective"] == "gaussian"
    assert runs["results_by_run"]["1"]["quality"]["verdict"] == "good"
    assert groups["results_by_run"]["1"]["objective"] is None
    assert groups["results_by_run"]["1"]["quality"] is None
    assert groups["results_by_run"]["1"]["quality_reason"] == "Fit objective was not recorded."
    assert migrate_to_current(migrated) == migrated
    assert project == original


def test_objective_summary_round_trips_through_saved_fit_and_series():
    summary = _summary() | {
        "objective": "poisson",
        "minimum_expected_count": 3.0,
        "quality": None,
        "quality_reason": "Sparse Poisson counts.",
    }
    slot = FitSlot(result=summary)
    assert FitSlot.from_dict(slot.to_dict()).result == summary
    series = FitSeries(
        batch_id="test",
        rep_type=RepresentationType.TIME_GROUPS,
        member_kind="groups",
        results_by_run={1: summary},
    )
    assert FitSeries.from_dict(series.to_dict()).results_by_run[1] == summary
