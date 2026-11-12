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
    assert saved["results_by_run"] == series["results_by_run"]
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
