"""Tests for the v19->v20 series-recipe / active-series schema migration (D11).

Pure-core: exercises :func:`asymmetry.core.project.schema.migrate_to_current`
directly (no GUI). A v19 project's per-run ``FitSlot`` pointers are folded into
the series that owned them — the Batch tab's parameter template and fit window
become ``recipe``, ``include_in_trend=False`` becomes
``trend_excluded_runs`` — member slots are dropped, single fits (with their
``ui_state``) survive untouched, and the newest series per representation
becomes active.
"""

from __future__ import annotations

from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    migrate_to_current,
    validate,
)
from asymmetry.core.representation.project_model import ProjectModel

_FB = "time_fb_asymmetry"
_GROUPS = "time_groups"

_MODEL = {"component_names": ["Exponential", "Constant"], "operators": ["+"]}

_TEMPLATE = [
    {"name": "A", "value": 0.2, "type": "Local", "bounds": "0, 1", "seeded": True},
    {"name": "Lambda", "value": 1.5, "type": "Global", "bounds": "-inf, inf", "seeded": True},
]


def _slot(**kwargs) -> dict:
    slot = {
        "model": None,
        "parameters": [],
        "result": None,
        "provenance": "none",
        "batch_id": None,
        "diverged": False,
        "include_in_trend": True,
    }
    slot.update(kwargs)
    return slot


def _dataset(run_number: int, *, rep_type=_FB, fit=None, projection_fits=None) -> dict:
    representation = {
        "rep_type": rep_type,
        "recipe": {},
        "fit": fit if fit is not None else _slot(),
        "trend_state": {},
        "result_metadata": {},
    }
    if projection_fits is not None:
        representation["projection_fits"] = projection_fits
    return {"run_number": run_number, "representations": {rep_type: representation}}


def _series(batch_id: str, *, rep_type=_FB, members=(10, 11), member_kind="runs", **kwargs) -> dict:
    series = {
        "batch_id": batch_id,
        "rep_type": rep_type,
        "member_kind": member_kind,
        "member_run_numbers": list(members),
        "member_source_run": {},
        "order_key": "run",
        "canonical_model": dict(_MODEL),
        "param_roles": {"A": "local", "Lambda": "global"},
        "nuisance_params": [],
        "results_by_run": {str(m): {"fit_range": "0.100–8.000 µs"} for m in members},
        "diverged_runs": [],
        "extra": {},
        "source_group_id": None,
        "group_id": None,
        "excluded_run_numbers": [],
        "last_fitted_members": list(members),
    }
    series.update(kwargs)
    return series


def _v19_state(*, datasets=None, batches=None, **extra) -> dict:
    state: dict = {
        "schema_version": 19,
        "created_with_app_version": "0.1.0",
        "datasets": list(datasets or []),
    }
    if batches is not None:
        state["batches"] = batches
    state.update(extra)
    return state


# ── (a) a batch series with member pointer slots ─────────────────────────────


def _batch_project() -> dict:
    """A v19 batch over runs 10 and 11, each holding a member pointer slot."""
    member_slot = _slot(
        model=dict(_MODEL),
        parameters=[dict(row) for row in _TEMPLATE],
        result={"success": True, "fit_range": "0.100–8.000 µs"},
        provenance="global",
        batch_id="b1",
    )
    return _v19_state(
        datasets=[_dataset(10, fit=member_slot), _dataset(11, fit=dict(member_slot))],
        batches=[_series("b1")],
    )


def test_batch_series_seeds_recipe_from_the_member_template():
    result = migrate_to_current(_batch_project())
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION == 20
    recipe = result["batches"][0]["recipe"]
    assert recipe["parameters"] == [
        {"name": "A", "value": 0.2, "type": "Local", "bounds": "0, 1", "seeded": False},
        {
            "name": "Lambda",
            "value": 1.5,
            "type": "Global",
            "bounds": "-inf, inf",
            "seeded": False,
        },
    ]
    assert recipe["fit_range"] == {"min": 0.1, "max": 8.0}
    assert recipe["seeding"] == "auto"
    assert recipe["coadd"] == {"mode": "off", "window": 2}


def test_batch_member_slots_are_dropped():
    result = migrate_to_current(_batch_project())
    for entry in result["datasets"]:
        slot = entry["representations"][_FB]["fit"]
        assert slot == {"model": None, "parameters": [], "result": None, "provenance": "none"}


def test_series_loses_diverged_runs_and_gains_an_active_pointer():
    state = _batch_project()
    state["batches"][0]["diverged_runs"] = [11]
    result = migrate_to_current(state)
    assert "diverged_runs" not in result["batches"][0]
    assert result["active_series"] == {_FB: "b1"}


def test_newest_series_per_representation_becomes_active():
    state = _batch_project()
    state["batches"].append(_series("b2"))
    state["batches"].append(_series("f1", rep_type="freq_fft"))
    result = migrate_to_current(state)
    # Last in list order wins per representation; other representations are
    # tracked independently.
    assert result["active_series"] == {_FB: "b2", "freq_fft": "f1"}


def test_unparsable_or_absent_fit_range_becomes_an_unbounded_window():
    state = _batch_project()
    state["batches"][0]["results_by_run"] = {"10": {"fit_range": "as fitted"}}
    assert migrate_to_current(state)["batches"][0]["recipe"]["fit_range"] == {
        "min": None,
        "max": None,
    }
    state["batches"][0]["results_by_run"] = {}
    assert migrate_to_current(state)["batches"][0]["recipe"]["fit_range"] == {
        "min": None,
        "max": None,
    }


def test_legacy_ascii_hyphen_fit_range_still_parses():
    state = _batch_project()
    state["batches"][0]["results_by_run"] = {"10": {"fit_range": "0.1-8 µs"}}
    assert migrate_to_current(state)["batches"][0]["recipe"]["fit_range"] == {
        "min": 0.1,
        "max": 8.0,
    }


# ── (b) an include_in_trend=False member ─────────────────────────────────────


def test_excluded_member_becomes_a_series_trend_exclusion():
    state = _batch_project()
    state["datasets"][1]["representations"][_FB]["fit"]["include_in_trend"] = False
    result = migrate_to_current(state)
    assert result["batches"][0]["trend_excluded_runs"] == [11]

    model = ProjectModel.from_project_state(result)
    assert model.batch("b1").trend_member_run_numbers() == [10]


def test_included_members_leave_no_exclusions():
    assert migrate_to_current(_batch_project())["batches"][0]["trend_excluded_runs"] == []


# ── (c) a single-fit slot with ui_state ──────────────────────────────────────


def test_single_fit_slot_keeps_ui_state_and_loses_only_the_moved_fields():
    single = _slot(
        model=dict(_MODEL),
        parameters=[dict(row) for row in _TEMPLATE],
        result={"success": True},
        provenance="single",
        include_in_trend=False,
        ui_state={"composite_model": dict(_MODEL), "result_html": "<p>ok</p>"},
    )
    state = _v19_state(datasets=[_dataset(10, fit=single)], batches=[])
    slot = migrate_to_current(state)["datasets"][0]["representations"][_FB]["fit"]
    assert slot["provenance"] == "single"
    assert slot["model"] == _MODEL
    assert slot["parameters"] == _TEMPLATE
    assert slot["result"] == {"success": True}
    assert slot["ui_state"] == {"composite_model": _MODEL, "result_html": "<p>ok</p>"}
    assert "batch_id" not in slot
    assert "diverged" not in slot
    assert "include_in_trend" not in slot


def test_projection_fits_are_migrated_like_the_default_slot():
    state = _v19_state(
        datasets=[
            _dataset(
                10,
                rep_type=_GROUPS,
                fit=_slot(provenance="none"),
                projection_fits={
                    "P_x": _slot(
                        model=dict(_MODEL),
                        provenance="single",
                        ui_state={"result_html": "keep"},
                    ),
                    "P_y": _slot(model=dict(_MODEL), provenance="batch", batch_id="b1"),
                },
            )
        ],
        batches=[],
    )
    projections = migrate_to_current(state)["datasets"][0]["representations"][_GROUPS][
        "projection_fits"
    ]
    assert projections["P_x"]["ui_state"] == {"result_html": "keep"}
    assert "batch_id" not in projections["P_x"]
    assert projections["P_y"] == {
        "model": None,
        "parameters": [],
        "result": None,
        "provenance": "none",
    }


# ── (d) a computed (model-less) scan series ──────────────────────────────────


def test_computed_scan_series_gets_an_empty_recipe():
    scan = _series(
        "scan-1",
        canonical_model=None,
        param_roles={},
        results_by_run={"10": {"success": True, "parameters": {"Integral asymmetry": 0.1}}},
    )
    result = migrate_to_current(_v19_state(datasets=[_dataset(10)], batches=[scan]))
    recipe = result["batches"][0]["recipe"]
    assert recipe["parameters"] == []
    assert recipe["fit_range"] == {"min": None, "max": None}

    model = ProjectModel.from_project_state(result)
    assert model.batch("scan-1").is_computed
    assert model.batch("scan-1").recipe == recipe


def test_computed_scan_never_becomes_the_active_series():
    """A scan has no fit to draw, so the newest *model-bearing* series is active."""
    scan = _series(
        "scan-1",
        canonical_model=None,
        param_roles={},
        results_by_run={"10": {"success": True, "parameters": {"Integral asymmetry": 0.1}}},
    )
    result = migrate_to_current(_v19_state(datasets=[_dataset(10)], batches=[_series("b1"), scan]))
    assert result["active_series"] == {_FB: "b1"}

    only_scan = migrate_to_current(_v19_state(datasets=[_dataset(10)], batches=[scan]))
    assert only_scan["active_series"] == {}


# ── (e) a member_kind="groups" series ────────────────────────────────────────


def test_group_series_reads_its_template_from_the_source_run_slot():
    member_slot = _slot(
        model=dict(_MODEL),
        parameters=[dict(row) for row in _TEMPLATE],
        provenance="global",
        batch_id="g1",
        include_in_trend=False,
    )
    grouped = _series(
        "g1",
        rep_type=_GROUPS,
        member_kind="groups",
        members=(-10001, -10002),
        member_source_run={"-10001": 10, "-10002": 10},
        last_fitted_members=[-10001, -10002],
    )
    state = _v19_state(
        datasets=[_dataset(10, rep_type=_GROUPS, fit=member_slot)],
        batches=[grouped],
    )
    result = migrate_to_current(state)
    series = result["batches"][0]
    assert [row["name"] for row in series["recipe"]["parameters"]] == ["A", "Lambda"]
    # The gate lived on the shared source-run slot, so every synthetic member of
    # that run is excluded together.
    assert series["trend_excluded_runs"] == [-10002, -10001]
    assert result["active_series"] == {_GROUPS: "g1"}
    # The shared pointer slot is dropped like any other member slot.
    assert result["datasets"][0]["representations"][_GROUPS]["fit"]["provenance"] == "none"


def test_group_series_without_a_source_map_decodes_the_synthetic_key():
    member_slot = _slot(parameters=[dict(_TEMPLATE[0])], provenance="batch", batch_id="g1")
    grouped = _series(
        "g1",
        rep_type=_GROUPS,
        member_kind="groups",
        members=(-2961001, -2961002),
        last_fitted_members=[-2961001, -2961002],
    )
    state = _v19_state(
        datasets=[_dataset(2961, rep_type=_GROUPS, fit=member_slot)],
        batches=[grouped],
    )
    result = migrate_to_current(state)
    assert [row["name"] for row in result["batches"][0]["recipe"]["parameters"]] == ["A"]


# ── (f) a project with no batches ────────────────────────────────────────────


def test_project_with_no_batches_migrates_clean():
    state = _v19_state(datasets=[_dataset(10)])
    assert "batches" not in state
    result = migrate_to_current(state)
    validate(result)
    assert result["schema_version"] == 20
    assert "active_series" not in result
    model = ProjectModel.from_project_state(result)
    assert model.batches == {}
    assert model.active_series == {}


def test_empty_batches_list_yields_an_empty_active_map():
    result = migrate_to_current(_v19_state(datasets=[], batches=[]))
    assert result["active_series"] == {}


# ── (g) orphan pointers and junk ─────────────────────────────────────────────


def test_member_slot_naming_a_dead_series_is_dropped_without_raising():
    orphan = _slot(
        model=dict(_MODEL),
        parameters=[dict(row) for row in _TEMPLATE],
        provenance="batch",
        batch_id="gone",
    )
    state = _v19_state(datasets=[_dataset(10, fit=orphan)], batches=[_series("b1", members=(11,))])
    result = migrate_to_current(state)
    # The orphan pointer slot is emptied; the live series seeds nothing from it.
    assert result["datasets"][0]["representations"][_FB]["fit"]["provenance"] == "none"
    assert result["batches"][0]["recipe"]["parameters"] == []
    ProjectModel.from_project_state(result)  # loads without raising


def test_malformed_entries_are_skipped_not_raised_on():
    state = _v19_state(
        datasets=["junk", None, {"run_number": "nope"}, {"run_number": 10, "representations": 5}],
        batches=["junk", None, {"batch_id": "b1", "rep_type": _FB}],
    )
    result = migrate_to_current(state)
    # Only the readable series survives; the junk carries nothing a reader
    # could use and would abort the open.
    assert len(result["batches"]) == 1
    assert result["batches"][0]["batch_id"] == "b1"
    assert result["batches"][0]["recipe"]["fit_range"] == {"min": None, "max": None}
    assert result["batches"][0]["trend_excluded_runs"] == []


def test_unloadable_batches_are_dropped_and_the_good_one_still_loads():
    """A series ``FitSeries.from_dict`` could not read must not reach the open.

    ``from_dict`` indexes ``batch_id`` and ``rep_type``; carrying such an entry
    through the migration would turn one junk record into a project that cannot
    be opened at all.
    """
    state = _v19_state(
        batches=[
            _series("b1"),
            {"label": "junk"},
            "not-even-a-dict",
            {"batch_id": 17, "rep_type": _FB},
            {"batch_id": "b2", "rep_type": "not_a_representation"},
        ],
    )
    result = migrate_to_current(state)
    validate(result)

    assert [entry["batch_id"] for entry in result["batches"]] == ["b1"]

    model = ProjectModel.from_project_state(result)
    assert list(model.batches) == ["b1"]


def test_migrated_project_round_trips_through_the_project_model():
    state = _batch_project()
    state["datasets"][1]["representations"][_FB]["fit"]["include_in_trend"] = False
    migrated = migrate_to_current(state)

    model = ProjectModel.from_project_state(migrated)
    series = model.batch("b1")
    assert series.recipe["fit_range"] == {"min": 0.1, "max": 8.0}
    assert series.trend_excluded_runs == [11]
    assert model.active_series_id(_FB) == "b1"
    identity = series.recipe_identity()

    project: dict = {"datasets": [{"run_number": 10}, {"run_number": 11}]}
    model.write_to_project_state(project)
    rebuilt = ProjectModel.from_project_state(project)
    assert rebuilt.batch("b1").recipe_identity() == identity
    assert rebuilt.batch("b1").trend_excluded_runs == [11]
    assert rebuilt.active_series_id(_FB) == "b1"
