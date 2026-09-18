"""Tests for the v21->v22 joint-fit schema migration and its persistence path.

Pure-core: exercises :func:`asymmetry.core.project.schema.migrate_to_current`
directly (no GUI), plus the full write/save/load/read round trip through
:class:`ProjectModel` and :func:`save_project`/:func:`load_project`. Mirrors
the neighbouring v14->v15 (``test_datagroup_series_migration.py``) and v19->v20
(``test_series_recipe_migration.py``) migration tests: a pre-v22 project has
no ``joint_fits`` block at all and migrates clean with an empty registry and
every series unstamped, exactly like the v12->v13 ``global_fit_studies`` step
this one is modelled on.
"""

from __future__ import annotations

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    load_project,
    migrate_to_current,
    save_project,
    validate,
)
from asymmetry.core.representation.joint_fit import JointFit
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import FitSeries

_MODEL = CompositeModel(["Exponential"]).to_dict()


def _v21_series(batch_id: str, **kwargs) -> dict:
    series = {
        "batch_id": batch_id,
        "rep_type": "time_fb_asymmetry",
        "member_kind": "runs",
        "member_run_numbers": [10, 11],
        "member_source_run": {},
        "order_key": "run",
        "canonical_model": dict(_MODEL),
        "param_roles": {"A": "global"},
        "nuisance_params": [],
        "results_by_run": {},
        "extra": {},
        "source_group_id": None,
        "group_id": None,
        "excluded_run_numbers": [],
        "last_fitted_members": [10, 11],
        "recipe": {
            "parameters": [],
            "fit_range": {"min": None, "max": None},
            "seeding": "auto",
            "coadd": {"mode": "off", "window": 2},
        },
        "trend_excluded_runs": [],
        # No "joint_fit_id"/"shared_params" keys at all: a genuine v21 save
        # predates the feature entirely.
    }
    series.update(kwargs)
    return series


def _v21_state(*, batches=None, **extra) -> dict:
    state: dict = {
        "schema_version": 21,
        "created_with_app_version": "0.1.0",
        "datasets": [],
    }
    if batches is not None:
        state["batches"] = batches
    state.update(extra)
    return state


def test_v21_project_migrates_with_empty_joint_fits_registry():
    """v22 is additive: no rewrite of ``batches`` on disk is needed at all.

    The migration only bumps the version and defaults the new top-level
    ``joint_fits`` list — a series dict with no ``joint_fit_id``/
    ``shared_params`` keys is exactly what a genuine v21 save looks like, and
    :meth:`FitSeries.from_dict` is what resolves "absent" to "unstamped" (see
    :func:`test_v21_project_round_trips_through_project_model`).
    """
    state = _v21_state(batches=[_v21_series("b1")])
    result = migrate_to_current(state)
    validate(result)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION == 22
    assert result["joint_fits"] == []
    series = result["batches"][0]
    assert "joint_fit_id" not in series
    assert "shared_params" not in series


def test_v21_project_with_no_batches_migrates_clean():
    state = _v21_state()
    result = migrate_to_current(state)
    validate(result)
    assert result["schema_version"] == 22
    assert result["joint_fits"] == []


def test_v21_project_round_trips_through_project_model():
    state = _v21_state(batches=[_v21_series("b1"), _v21_series("b2")])
    migrated = migrate_to_current(state)
    model = ProjectModel.from_project_state(migrated)
    assert model.joint_fits == {}
    assert model.batch("b1").joint_fit_id is None
    assert model.batch("b2").shared_params == {}


def test_migrate_is_a_no_op_when_joint_fits_already_present():
    """An idempotence check: re-migrating an already-v22 state changes nothing."""
    state = _v21_state(batches=[_v21_series("b1")])
    state["schema_version"] = 22
    state["joint_fits"] = [{"joint_id": "j1"}]  # malformed, but migration must not touch it
    result = migrate_to_current(state)
    assert result["schema_version"] == 22
    assert result["joint_fits"] == [{"joint_id": "j1"}]


# ── full round trip: ProjectModel -> project dict -> file -> ProjectModel ──


def _series_with_stamp(batch_id: str, joint_id: str) -> FitSeries:
    return FitSeries(
        batch_id,
        "time_fb_asymmetry",
        member_run_numbers=[10, 11],
        canonical_model=dict(_MODEL),
        param_roles={"A": "global"},
        joint_fit_id=joint_id,
        shared_params={"A": "A_shared"},
    )


def _joint_fit(joint_id: str, members: tuple[str, ...]) -> JointFit:
    return JointFit(
        joint_id,
        label="Ordered + Para",
        rep_type="time_fb_asymmetry",
        member_batch_ids=list(members),
        shared=[
            {
                "name": "A_shared",
                "members": {m: "A" for m in members},
                "value": 0.2,
                "min": 0.0,
                "max": 1.0,
            }
        ],
        result={
            "shared_values": {"A_shared": 0.21},
            "shared_uncertainties": {"A_shared": 0.01},
            "shared_covariance": [[1e-4]],
            "chi_squared": 50.0,
            "dof": 40,
            "reduced_chi_squared": 1.25,
            "series_reduced_chi_squared": {members[0]: 1.1, members[1]: 1.4},
            "fitted_at": "2026-09-18T00:00:00",
        },
    )


def _project_shell() -> dict:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "created_with_app_version": "0.1.0",
        "datasets": [],
    }


def test_project_model_round_trips_through_to_dict():
    model = ProjectModel()
    model.add_batch(_series_with_stamp("b1", "j1"))
    model.add_batch(_series_with_stamp("b2", "j1"))
    model.add_joint_fit(_joint_fit("j1", ("b1", "b2")))

    restored = ProjectModel.from_dict(model.to_dict())

    assert set(restored.joint_fits) == {"j1"}
    assert restored.joint_fit("j1").member_batch_ids == ["b1", "b2"]
    assert restored.joint_fit("j1").result == model.joint_fit("j1").result
    assert restored.batch("b1").joint_fit_id == "j1"
    assert restored.batch("b2").shared_params == {"A": "A_shared"}


def test_project_model_round_trips_through_save_and_load_project(tmp_path):
    model = ProjectModel()
    model.add_batch(_series_with_stamp("b1", "j1"))
    model.add_batch(_series_with_stamp("b2", "j1"))
    model.add_joint_fit(_joint_fit("j1", ("b1", "b2")))

    project = _project_shell()
    model.write_to_project_state(project)

    path = tmp_path / "project.asymp"
    save_project(project, path)
    loaded = load_project(path)
    validate(loaded)

    restored = ProjectModel.from_project_state(loaded)
    assert set(restored.joint_fits) == {"j1"}
    joint = restored.joint_fit("j1")
    assert joint.label == "Ordered + Para"
    assert joint.member_batch_ids == ["b1", "b2"]
    assert joint.shared[0]["name"] == "A_shared"
    assert joint.shared[0]["min"] == 0.0
    assert joint.result["chi_squared"] == 50.0
    assert restored.batch("b1").joint_fit_id == "j1"
    assert restored.batch("b2").joint_fit_id == "j1"
