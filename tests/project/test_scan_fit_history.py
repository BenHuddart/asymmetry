"""Released flags migrate once without inventing numerical scan results."""

from copy import deepcopy

from asymmetry.core.project.schema import migrate_to_current


def test_legacy_column_floor_recovers_known_scan_and_single_series_executions():
    result = {
        "success": True,
        "parameters": [{"name": "c", "value": 5 / 3}],
        "uncertainties": {"c": 1 / 6**0.5},
        "chi_squared": 16 / 3,
        "error_mode": "column",
    }
    original = {
        "schema_version": 42,
        "fit_parameters_state": {
            "model_fits": {
                "single": {"ranges": [{"result": result | {"n_points": 3}}]},
                "ambiguous": {"ranges": [{"result": dict(result)}]},
            },
            "cross_group_fit_configs": {"groups": {"error_mode": "column"}},
        },
        "global_fit_windows": {"a": {"local_model_fits": {"c": {"ranges": [{"result": result}]}}}},
        "batches": [
            {"extra": {"fit_history": {"records": [{"kind": "baseline", "result": dict(result)}]}}}
        ],
    }

    migrated = migrate_to_current(original)

    state = migrated["fit_parameters_state"]
    expected = result | {"error_mode": "column_stabilized"}
    assert state["model_fits"]["single"]["ranges"][0]["result"] == expected | {"n_points": 3}
    assert state["model_fits"]["ambiguous"]["ranges"][0]["result"] == result
    assert state["cross_group_fit_configs"]["groups"]["error_mode"] == "column"
    local_fits = migrated["global_fit_windows"]["a"]["local_model_fits"]
    assert local_fits["c"]["ranges"][0]["result"] == expected
    assert migrated["batches"][0]["extra"]["fit_history"]["records"][0]["result"] == expected
    assert original["fit_parameters_state"]["model_fits"]["single"]["ranges"][0]["result"] == (
        result | {"n_points": 3}
    )
    assert migrated == migrate_to_current(migrated)


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
