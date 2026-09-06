"""Tests for the v14->v15 DataGroup/FitSeries unification schema migration.

Pure-core: exercises :func:`asymmetry.core.project.schema.migrate_to_current`
directly (no GUI). Covers the D9 cases — a run series whose ``source_group_id``
resolves to a live group adopts it as ``group_id``; a group-less run series
freezes to ``group_id=None``; a detector-group (``member_kind="groups"``)
series is only given the additive defaults; a project with no ``data_groups``
block at all migrates clean — plus the v15 write/read round-trip.
"""

from __future__ import annotations

from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    migrate_to_current,
    validate,
)
from asymmetry.core.representation.group import PhaseSpec
from asymmetry.core.representation.project_model import ProjectModel


def _v14_state(*, data_groups=None, batches=None) -> dict:
    state: dict = {
        "schema_version": 14,
        "created_with_app_version": "0.1.0",
        "datasets": [],
    }
    if data_groups is not None:
        state["data_groups"] = data_groups
    if batches is not None:
        state["batches"] = batches
    return state


def _run_series(batch_id: str, *, source_group_id=None, members=(1, 2), member_kind="runs") -> dict:
    return {
        "batch_id": batch_id,
        "rep_type": "time_fb_asymmetry",
        "member_kind": member_kind,
        "member_run_numbers": list(members),
        "member_source_run": {},
        "order_key": "run",
        "canonical_model": None,
        "param_roles": {},
        "nuisance_params": [],
        "results_by_run": {},
        "diverged_runs": [],
        "extra": {},
        "source_group_id": source_group_id,
    }


def test_v15_bumps_version_and_defaults_group_kind():
    state = _v14_state(
        data_groups=[{"group_id": "grp-1", "name": "B = 60 G", "member_run_numbers": [1, 2]}]
    )
    result = migrate_to_current(state)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION == 19
    assert result["data_groups"][0]["kind"] == "user"


def test_case_a_series_with_resolvable_source_group_id_adopts_group_id():
    state = _v14_state(
        data_groups=[{"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2]}],
        batches=[_run_series("b1", source_group_id="grp-1", members=(1, 2))],
    )
    result = migrate_to_current(state)
    series = result["batches"][0]
    assert series["group_id"] == "grp-1"
    assert series["excluded_run_numbers"] == []
    assert series["last_fitted_members"] == [1, 2]


def test_case_b_group_less_series_freezes_to_none():
    # source_group_id present but names no live group -> frozen (group_id None).
    state = _v14_state(
        data_groups=[{"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2]}],
        batches=[
            _run_series("b1", source_group_id="ghost", members=(5, 6)),
            _run_series("b2", source_group_id=None, members=(7, 8)),
        ],
    )
    result = migrate_to_current(state)
    dangling, no_provenance = result["batches"]
    assert dangling["group_id"] is None
    assert no_provenance["group_id"] is None
    assert no_provenance["last_fitted_members"] == [7, 8]


def test_case_c_detector_group_series_untouched_semantics():
    state = _v14_state(
        data_groups=[{"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2]}],
        # A "groups" series whose source_group_id happens to name a live group is
        # still never group-resolved — detector-group series keep frozen semantics.
        batches=[
            _run_series("g1", source_group_id="grp-1", members=(-1001, -1002), member_kind="groups")
        ],
    )
    result = migrate_to_current(state)
    series = result["batches"][0]
    assert series["group_id"] is None
    assert series["excluded_run_numbers"] == []
    assert series["last_fitted_members"] == [-1001, -1002]


def test_case_d_no_data_groups_block_migrates_clean():
    # Pre-Phase-7 save: no data_groups key at all, one group-less run series.
    state = _v14_state(batches=[_run_series("b1", source_group_id=None, members=(3, 4))])
    assert "data_groups" not in state
    result = migrate_to_current(state)
    validate(result)
    assert result["schema_version"] == 19
    series = result["batches"][0]
    assert series["group_id"] is None
    assert series["last_fitted_members"] == [3, 4]


def test_migration_tolerates_junk_shapes():
    state = _v14_state(data_groups=["not-a-dict", 5], batches=["junk", None])
    result = migrate_to_current(state)
    # Junk entries pass through untouched; no raise.
    assert result["data_groups"] == ["not-a-dict", 5]
    assert result["batches"] == ["junk", None]


def test_migration_is_idempotent_on_already_migrated_fields():
    state = _v14_state(
        data_groups=[
            {"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2], "kind": "auto"}
        ],
        batches=[
            {
                **_run_series("b1", source_group_id="grp-1", members=(1, 2)),
                "group_id": None,
                "excluded_run_numbers": [1],
                "last_fitted_members": [2],
            }
        ],
    )
    result = migrate_to_current(state)
    # Pre-existing kind / group_id / exclusions / last_fitted survive the migration.
    assert result["data_groups"][0]["kind"] == "auto"
    series = result["batches"][0]
    assert series["group_id"] is None
    assert series["excluded_run_numbers"] == [1]
    assert series["last_fitted_members"] == [2]


def test_v15_round_trips_through_write_read():
    state = _v14_state(
        data_groups=[{"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2]}],
        batches=[_run_series("b1", source_group_id="grp-1", members=(1, 2))],
    )
    migrated = migrate_to_current(state)

    model = ProjectModel.from_project_state(migrated)
    group = model.data_group("grp-1")
    assert group is not None and group.kind == "user"
    series = model.batch("b1")
    assert series.group_id == "grp-1"
    assert series.last_fitted_members == [1, 2]
    assert series.excluded_run_numbers == []

    # Write back out and re-read: the new fields survive the round-trip.
    project: dict = {"datasets": []}
    model.write_to_project_state(project)
    assert project["data_groups"][0]["kind"] == "user"
    assert project["batches"][0]["group_id"] == "grp-1"

    rebuilt = ProjectModel.from_project_state(project)
    assert rebuilt.batch("b1").group_id == "grp-1"
    assert rebuilt.batch("b1").last_fitted_members == [1, 2]
    assert rebuilt.data_group("grp-1").kind == "user"


# ── v18 -> v19: nested phase data groups (Global Fit Wizard transitions) ──────


def _v18_state(*, data_groups=None) -> dict:
    state: dict = {
        "schema_version": 18,
        "created_with_app_version": "0.1.0",
        "datasets": [],
    }
    if data_groups is not None:
        state["data_groups"] = data_groups
    return state


def test_v19_adds_phase_defaults_to_existing_groups():
    state = _v18_state(
        data_groups=[
            {"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2], "kind": "user"}
        ]
    )
    result = migrate_to_current(state)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION == 19
    group = result["data_groups"][0]
    assert group["parent_group_id"] is None
    assert group["phase_ordinal"] is None
    assert group["phase_range"] is None
    assert group["phase_boundaries"] == {"lower": None, "upper": None}
    assert group["phase_color"] is None
    assert group["phase_provenance"] == {}


def test_v19_migration_tolerates_no_data_groups_block():
    state = _v18_state()
    result = migrate_to_current(state)
    validate(result)
    assert result["schema_version"] == 19
    assert "data_groups" not in result


def test_v19_migration_tolerates_junk_group_entries():
    state = _v18_state(data_groups=["not-a-dict", 5])
    result = migrate_to_current(state)
    assert result["data_groups"] == ["not-a-dict", 5]


def test_v18_project_with_groups_migrates_and_round_trips():
    """A v18 project with an ordinary group migrates to v19 and round-trips."""
    state = _v18_state(
        data_groups=[{"group_id": "grp-1", "name": "scan", "member_run_numbers": [1, 2, 3]}]
    )
    migrated = migrate_to_current(state)
    assert migrated["schema_version"] == 19

    model = ProjectModel.from_project_state(migrated)
    parent = model.data_group("grp-1")
    assert parent is not None
    assert not parent.is_phase

    ids = model.create_phase_groups(
        "grp-1",
        [
            PhaseSpec(
                ordinal=1,
                name="Phase I",
                member_run_numbers=(1, 2),
                phase_range=(1.0, 2.0),
                phase_boundaries={"lower": None, "upper": (2.5, 0.5)},
                phase_color="#2F4DA0",
                phase_provenance={"axis_key": "temperature"},
            )
        ],
    )

    project: dict = {"datasets": []}
    model.write_to_project_state(project)

    rebuilt = ProjectModel.from_project_state(project)
    phases = rebuilt.phase_groups_for("grp-1")
    assert [p.group_id for p in phases] == ids
    phase = phases[0]
    assert phase.name == "Phase I"
    assert phase.phase_range == (1.0, 2.0)
    assert phase.phase_boundaries == {"lower": None, "upper": (2.5, 0.5)}
    assert phase.phase_color == "#2F4DA0"
    assert phase.phase_provenance == {"axis_key": "temperature"}


def test_v19_project_with_phase_group_loads_and_resolves():
    """A file already at v19 (no migration involved) loads its phase group."""
    state = {
        "schema_version": 19,
        "created_with_app_version": "0.1.0",
        "datasets": [],
        "data_groups": [
            {
                "group_id": "grp-1",
                "name": "scan",
                "member_run_numbers": [1, 2, 3],
                "order_key": "run",
                "kind": "user",
                "parent_group_id": None,
                "phase_ordinal": None,
                "phase_range": None,
                "phase_boundaries": {"lower": None, "upper": None},
                "phase_color": None,
                "phase_provenance": {},
            },
            {
                "group_id": "phase-1",
                "name": "Phase I",
                "member_run_numbers": [1, 2],
                "order_key": "run",
                "kind": "user",
                "parent_group_id": "grp-1",
                "phase_ordinal": 1,
                "phase_range": [1.0, 2.0],
                "phase_boundaries": {"lower": None, "upper": [2.5, 0.5]},
                "phase_color": "#2F4DA0",
                "phase_provenance": {"axis_key": "temperature"},
            },
        ],
    }
    validate(state)
    model = ProjectModel.from_project_state(state)
    phases = model.phase_groups_for("grp-1")
    assert len(phases) == 1
    assert phases[0].group_id == "phase-1"
    assert phases[0].phase_range == (1.0, 2.0)
    assert phases[0].phase_boundaries == {"lower": None, "upper": (2.5, 0.5)}
    assert model.excluded_runs_for("grp-1") == [3]
