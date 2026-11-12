"""Saved scalar results retain their values and honest estimator identity."""

from copy import deepcopy

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, migrate_to_current
from asymmetry.core.representation import FitSeries, RepresentationType


def test_legacy_project_scan_is_not_relabelled_as_count_integral():
    series = FitSeries(
        "scan-1",
        RepresentationType.TIME_FB_ASYMMETRY,
        results_by_run={1: {"parameters": {"Integral asymmetry": 80.0}}},
        extra={"kind": "alc_scan", "rf_mode": True},
    ).to_dict()
    series["extra"].pop("estimator")
    series["extra"].pop("time_window_us")
    old = {"schema_version": 36, "datasets": [], "project_model": {"batches": [series]}}
    original = deepcopy(old)
    migrated = migrate_to_current(old)
    saved = migrated["project_model"]["batches"][0]
    assert saved["extra"]["estimator"] == "legacy_unspecified"
    assert saved["results_by_run"]["1"]["parameters"] == series["results_by_run"]["1"]["parameters"]
    assert saved["results_by_run"]["1"]["observation"] is None
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrate_to_current(migrated) == migrated
    assert old == original


def test_explicit_saved_scan_identity_roundtrips():
    series = FitSeries(
        "scan-1",
        RepresentationType.TIME_FB_ASYMMETRY,
        extra={"kind": "alc_scan", "estimator": "integral", "time_window_us": [0, 1]},
    )
    assert FitSeries.from_dict(series.to_dict()).extra == series.extra


def test_schema37_preserves_stored_values_and_unknown_observation_status():
    data = {
        "schema_version": 37,
        "batches": [
            {
                "extra": {
                    "kind": "alc_scan",
                    "estimator": "legacy_unspecified",
                    "time_window_us": None,
                },
                "results_by_run": {"1": {"parameters": {"A": 0.2}, "uncertainties": {"A": 1.0}}},
            }
        ],
    }
    migrated = migrate_to_current(data)
    series = migrated["batches"][0]
    assert series["extra"]["estimator"] == "legacy_unspecified"
    assert series["extra"]["correlated_pairs"] == []
    assert series["results_by_run"]["1"]["observation"] is None
    assert series["results_by_run"]["1"]["uncertainties"] == {"A": 1.0}
    assert "observation" not in data["batches"][0]["results_by_run"]["1"]
    assert migrate_to_current(migrated) == migrated


def test_schema37_retains_legacy_null_extra():
    data = {"schema_version": 37, "project_model": {"batches": [{"extra": None}]}}
    assert migrate_to_current(data)["project_model"]["batches"][0]["extra"] is None
