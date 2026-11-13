"""Released flags migrate once without inventing numerical scan results."""

from copy import deepcopy

from asymmetry.core.project.schema import migrate_to_current


def test_settings_only_scan_migration_preserves_retained_values():
    extra = {
        "kind": "alc_scan",
        "baseline_fitted": True,
        "peaks_fitted": True,
        "rf_fitted": False,
        "peaks": [["Gaussian", 200.0, 20.0, -5.0]],
        "regions": [[50.0, 120.0]],
        "computation_revision": None,
    }
    original = {"schema_version": 39, "batches": [{"extra": deepcopy(extra)}]}
    current = migrate_to_current(original)
    migrated = current["batches"][0]["extra"]
    assert migrated["fit_history"] == {
        "legacy_settings": extra,
        "records": [],
        "active": {},
        "missing": ["baseline", "peaks"],
    }
    assert migrated["peaks"] == extra["peaks"]
    assert migrated["regions"] == extra["regions"]
    assert migrated["computation_revision"] is None
    assert "baseline_fitted" not in migrated
    assert original["batches"][0]["extra"] == extra
    assert migrate_to_current(current) == current


def test_invalid_scan_fit_reference_is_rejected_before_application():
    import pytest

    from asymmetry.core.project import ProjectValidationError, prepare_project

    state = {
        "schema_version": 40,
        "datasets": [],
        "batches": [
            {
                "batch_id": "scan",
                "representation_type": "time_fb_asymmetry",
                "extra": {
                    "kind": "alc_scan",
                    "fit_history": {
                        "records": [],
                        "active": {"baseline": 0},
                        "missing": [],
                    },
                },
            }
        ],
    }
    with pytest.raises(ProjectValidationError, match="extra.fit_history.*active execution"):
        prepare_project(state)
