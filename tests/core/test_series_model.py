"""Unit tests for the FitSeries object (Phase 1)."""

from __future__ import annotations

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.base import FitSlot
from asymmetry.core.representation.group import DATA_GROUP_KINDS, DataGroup
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import (
    FitSeries,
    canonical_model_matches,
    default_recipe,
)

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _batch(**kwargs) -> FitSeries:
    defaults = dict(
        batch_id="b1",
        rep_type=RepresentationType.TIME_FB_ASYMMETRY,
        member_run_numbers=[10, 11, 12],
        order_key="run",
        canonical_model=CompositeModel(["Exponential", "Constant"]).to_dict(),
        param_roles={"A": "global", "Lambda": "local", "A_bg": "fixed"},
    )
    defaults.update(kwargs)
    return FitSeries(**defaults)


# ── classifier-derived scope ────────────────────────────────────────────────


def test_is_global_when_any_param_is_global():
    assert _batch().is_global()


def test_pure_batch_when_no_global_param():
    batch = _batch(param_roles={"A": "local", "Lambda": "local"})
    assert not batch.is_global()


def test_param_role_partitions():
    batch = _batch()
    assert batch.global_params() == ["A"]
    assert batch.local_params() == ["Lambda"]
    assert batch.fixed_params() == ["A_bg"]


def test_invalid_param_role_dropped():
    batch = _batch(param_roles={"A": "global", "B": "bogus"})
    assert "B" not in batch.param_roles


# ── ordering ────────────────────────────────────────────────────────────────


def test_sort_members_by_field_and_temperature():
    runs = {
        10: Run(run_number=10, metadata={"field": 300.0, "temperature": 2.0}),
        11: Run(run_number=11, metadata={"field": 100.0, "temperature": 8.0}),
        12: Run(run_number=12, metadata={"field": 200.0, "temperature": 5.0}),
    }
    batch = _batch(order_key="field")
    batch.sort_members(runs)
    assert batch.member_run_numbers == [11, 12, 10]

    batch.order_key = "temperature"
    batch.sort_members(runs)
    assert batch.member_run_numbers == [10, 12, 11]


def test_sort_members_run_fallback_when_no_runs():
    batch = _batch(member_run_numbers=[12, 10, 11], order_key="field")
    batch.sort_members({})  # missing runs -> fall back to run number
    assert batch.member_run_numbers == [10, 11, 12]


def test_unknown_order_key_defaults_to_run():
    assert FitSeries("b", RepresentationType.FREQ_FFT, order_key="weird").order_key == "run"


# ── membership ──────────────────────────────────────────────────────────────


def test_add_and_remove_member_cleans_derived_state():
    batch = _batch(results_by_run={11: {"chi": 1.0}}, trend_excluded_runs=[11])
    batch.add_member(11)  # idempotent
    assert batch.member_run_numbers.count(11) == 1
    batch.add_member(20)
    assert 20 in batch.member_run_numbers
    batch.remove_member(11)
    assert 11 not in batch.member_run_numbers
    assert 11 not in batch.results_by_run
    assert batch.trend_excluded_runs == []


# ── trend exclusions (D4) ───────────────────────────────────────────────────


def test_trend_excluded_runs_gate_trend_membership():
    batch = _batch(trend_excluded_runs=[11])
    assert batch.trend_member_run_numbers() == [10, 12]
    batch.trend_excluded_runs = []
    assert batch.trend_member_run_numbers() == [10, 11, 12]


def test_trend_excluded_runs_sorted_deduped_and_round_trip():
    batch = _batch(trend_excluded_runs=[12, 10, 12])
    assert batch.trend_excluded_runs == [10, 12]
    restored = FitSeries.from_dict(batch.to_dict())
    assert restored.trend_excluded_runs == [10, 12]
    assert restored.trend_member_run_numbers() == [11]
    # A pre-v20 dict with no key at all defaults to "nothing excluded".
    payload = batch.to_dict()
    del payload["trend_excluded_runs"]
    assert FitSeries.from_dict(payload).trend_excluded_runs == []


def test_trend_exclusion_is_per_series_not_per_run():
    """Two series over the same runs keep independent trend gates (D4)."""
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1"))
    pm.add_batch(_batch(batch_id="b2"))
    pm.set_trend_excluded("b1", 11, True)
    assert pm.batch("b1").trend_member_run_numbers() == [10, 12]
    assert pm.batch("b2").trend_member_run_numbers() == [10, 11, 12]
    pm.set_trend_excluded("b1", 11, False)
    assert pm.batch("b1").trend_member_run_numbers() == [10, 11, 12]
    # Unknown series: nothing to toggle, nothing raised.
    pm.set_trend_excluded("missing", 11, True)


# ── recipe (D2) ─────────────────────────────────────────────────────────────


def test_recipe_defaults_and_round_trip():
    fresh = _batch()
    assert fresh.recipe == {
        "parameters": [],
        "fit_range": {"min": None, "max": None},
        "seeding": "auto",
        "coadd": {"mode": "off", "window": 2},
    }
    series = _batch(
        recipe={
            "parameters": [
                {"name": "A", "value": 0.2, "type": "Local", "bounds": "0, 1", "seeded": True}
            ],
            "fit_range": {"min": 0.1, "max": 8},
            "seeding": "carry",
            "coadd": {"mode": "window", "window": 4},
        }
    )
    restored = FitSeries.from_dict(series.to_dict())
    assert restored.recipe == series.recipe
    assert restored.recipe["fit_range"] == {"min": 0.1, "max": 8.0}
    assert restored.recipe["coadd"] == {"mode": "window", "window": 4}
    assert restored.recipe["seeding"] == "carry"


def test_recipe_absent_from_dict_defaults_tolerantly():
    """A pre-v20 series dict carries no recipe at all."""
    payload = _batch().to_dict()
    del payload["recipe"]
    assert FitSeries.from_dict(payload).recipe == default_recipe()


def test_partial_recipe_is_completed_on_construction():
    series = _batch(recipe={"seeding": "manual"})
    assert series.recipe["seeding"] == "manual"
    assert series.recipe["fit_range"] == {"min": None, "max": None}
    assert series.recipe["coadd"] == {"mode": "off", "window": 2}
    assert series.recipe["parameters"] == []


# ── recipe identity (D3) ────────────────────────────────────────────────────


def _identity_series(**kwargs) -> FitSeries:
    defaults = dict(
        last_fitted_members=[10, 11, 12],
        recipe={
            "parameters": [
                {"name": "A", "value": 0.2, "type": "Local", "bounds": "0, 1", "seeded": False}
            ],
            "fit_range": {"min": 0.0, "max": 8.0},
        },
    )
    defaults.update(kwargs)
    return _batch(**defaults)


def test_recipe_identity_differs_when_fit_range_differs():
    narrow = _identity_series(
        recipe={"parameters": [], "fit_range": {"min": 0.0, "max": 6.0}},
    )
    wide = _identity_series(
        recipe={"parameters": [], "fit_range": {"min": 0.0, "max": 8.0}},
    )
    assert narrow.recipe_identity() != wide.recipe_identity()


def test_recipe_identity_ignores_label_and_batch_id():
    """Renaming never changes identity (D3)."""
    first = _identity_series(batch_id="b1", label="Field sweep")
    second = _identity_series(batch_id="b2", label=None)
    assert first.recipe_identity() == second.recipe_identity()


def test_recipe_identity_differs_when_a_member_is_excluded():
    full = _identity_series()
    subset = _identity_series(last_fitted_members=[10, 11], excluded_run_numbers=[12])
    assert full.recipe_identity() != subset.recipe_identity()


def test_recipe_identity_differs_when_bounds_differ():
    loose = _identity_series()
    tight = _identity_series(
        recipe={
            "parameters": [
                {"name": "A", "value": 0.2, "type": "Local", "bounds": "0, 0.5", "seeded": False}
            ],
            "fit_range": {"min": 0.0, "max": 8.0},
        }
    )
    assert loose.recipe_identity() != tight.recipe_identity()


def test_recipe_identity_ignores_member_order_and_results():
    """The member *set* is identity; its ordering and results are not."""
    first = _identity_series(last_fitted_members=[10, 11, 12])
    second = _identity_series(
        last_fitted_members=[12, 10, 11],
        results_by_run={10: {"success": True}},
    )
    assert first.recipe_identity() == second.recipe_identity()


def test_recipe_identity_differs_on_model_roles_and_representation():
    base = _identity_series()
    assert (
        _identity_series(canonical_model=CompositeModel(["Gaussian"]).to_dict()).recipe_identity()
        != base.recipe_identity()
    )
    assert _identity_series(param_roles={"A": "local"}).recipe_identity() != base.recipe_identity()
    assert (
        _identity_series(rep_type=RepresentationType.FREQ_FFT).recipe_identity()
        != base.recipe_identity()
    )


def test_recipe_identity_is_stable_across_a_round_trip():
    series = _identity_series()
    assert FitSeries.from_dict(series.to_dict()).recipe_identity() == series.recipe_identity()


# ── canonical model comparison ──────────────────────────────────────────────


def test_canonical_model_matches_normalised():
    a = CompositeModel(["Exponential", "Constant"]).to_dict()
    b = CompositeModel(["Exponential", "Constant"]).to_dict()
    assert canonical_model_matches(a, b)


def test_canonical_model_mismatch_on_different_components():
    a = CompositeModel(["Exponential", "Constant"]).to_dict()
    b = CompositeModel(["Gaussian", "Constant"]).to_dict()
    assert not canonical_model_matches(a, b)


def test_canonical_model_matches_handles_none():
    assert canonical_model_matches(None, None)
    assert not canonical_model_matches(None, {"component_names": ["Exponential"]})


# ── persistence ─────────────────────────────────────────────────────────────


def test_batch_round_trip():
    batch = _batch(results_by_run={10: {"chi": 1.0}, 11: {"chi": 2.0}})
    restored = FitSeries.from_dict(batch.to_dict())
    assert restored.batch_id == batch.batch_id
    assert restored.rep_type == batch.rep_type
    assert restored.member_run_numbers == batch.member_run_numbers
    assert restored.param_roles == batch.param_roles
    assert restored.canonical_model == batch.canonical_model
    assert restored.results_by_run == batch.results_by_run
    assert restored.is_global() == batch.is_global()


# ── member kind & group series ───────────────────────────────────────────────


def test_member_kind_defaults_to_runs_and_validates():
    assert _batch().member_kind == "runs"
    assert _batch(member_kind="groups").member_kind == "groups"
    assert _batch(member_kind="bogus").member_kind == "runs"  # invalid falls back


def test_source_run_for_runs_is_identity():
    batch = _batch()
    assert batch.source_run_for(11) == 11


def test_source_run_for_groups_uses_map_then_decodes_key():
    # Synthetic group keys: -((source*1000)+group_index).
    batch = _batch(
        member_kind="groups",
        member_run_numbers=[-10001, -10002, -11001],
        member_source_run={-10001: 10, -10002: 10},
    )
    assert batch.source_run_for(-10001) == 10  # from map
    assert batch.source_run_for(-11001) == 11  # decoded from key (|key| // 1000)


def test_group_members_sort_by_source_run_then_key():
    runs = {
        10: Run(run_number=10, metadata={"field": 300.0}),
        11: Run(run_number=11, metadata={"field": 100.0}),
    }
    batch = _batch(
        member_kind="groups",
        member_run_numbers=[-10002, -11001, -10001, -11002],
        member_source_run={-10001: 10, -10002: 10, -11001: 11, -11002: 11},
        order_key="field",
    )
    batch.sort_members(runs)
    # Run 11 (field 100) before run 10 (field 300); within a run, groups in
    # ascending group-index order (|key| = run*1000+index).
    assert batch.member_run_numbers == [-11001, -11002, -10001, -10002]


def test_add_member_records_source_run_for_groups():
    batch = _batch(member_kind="groups", member_run_numbers=[])
    batch.add_member(-12003, source_run=12)
    assert batch.member_run_numbers == [-12003]
    assert batch.source_run_for(-12003) == 12
    batch.remove_member(-12003)
    assert -12003 not in batch.member_source_run


def test_round_trip_preserves_group_fields():
    batch = _batch(
        member_kind="groups",
        member_run_numbers=[-10001, -10002],
        member_source_run={-10001: 10, -10002: 10},
        nuisance_params=["N0", "background", "amplitude", "relative_phase"],
    )
    restored = FitSeries.from_dict(batch.to_dict())
    assert restored.member_kind == "groups"
    assert restored.member_run_numbers == [-10001, -10002]
    assert restored.member_source_run == {-10001: 10, -10002: 10}
    assert restored.nuisance_params == ["N0", "background", "amplitude", "relative_phase"]


# ── DataGroup.kind (D4) ───────────────────────────────────────────────────────


def test_data_group_kind_defaults_to_user():
    assert DataGroup("g1", "scan").kind == "user"
    assert set(DATA_GROUP_KINDS) == {"user", "auto"}


def test_data_group_kind_stored_and_invalid_coerced_to_user():
    assert DataGroup("g1", "scan", kind="auto").kind == "auto"
    assert DataGroup("g1", "scan", kind="bogus").kind == "user"  # invalid falls back


def test_data_group_kind_round_trips_and_tolerant_read():
    group = DataGroup("g1", "scan", member_run_numbers=[1, 2], order_key="field", kind="auto")
    restored = DataGroup.from_dict(group.to_dict())
    assert restored.kind == "auto"
    assert restored.member_run_numbers == [1, 2]
    assert restored.order_key == "field"
    # Pre-v15 dict with no ``kind`` key defaults to "user".
    assert DataGroup.from_dict({"group_id": "g2", "name": "old"}).kind == "user"


# ── DataGroup phase fields (D1, Global Fit Wizard transitions) ───────────────


def test_data_group_is_not_a_phase_by_default():
    group = DataGroup("g1", "scan", member_run_numbers=[1, 2, 3])
    assert group.parent_group_id is None
    assert not group.is_phase
    assert group.phase_ordinal is None
    assert group.phase_range is None
    assert group.phase_boundaries == {"lower": None, "upper": None}
    assert group.phase_color is None
    assert group.phase_provenance == {}


def test_data_group_phase_fields_set_and_is_phase_true():
    phase = DataGroup(
        "p1",
        "Phase I",
        member_run_numbers=[10, 11, 12],
        parent_group_id="series-1",
        phase_ordinal=1,
        phase_range=(1.8, 16.0),
        phase_boundaries={"lower": None, "upper": (16.5, 0.5)},
        phase_color="#2F4DA0",
        phase_provenance={"axis_key": "temperature", "gains": [12.4]},
    )
    assert phase.is_phase
    assert phase.parent_group_id == "series-1"
    assert phase.phase_ordinal == 1
    assert phase.phase_range == (1.8, 16.0)
    assert phase.phase_boundaries == {"lower": None, "upper": (16.5, 0.5)}
    assert phase.phase_color == "#2F4DA0"
    assert phase.phase_provenance == {"axis_key": "temperature", "gains": [12.4]}


def test_data_group_phase_fields_round_trip():
    phase = DataGroup(
        "p1",
        "Phase I",
        member_run_numbers=[10, 11, 12],
        parent_group_id="series-1",
        phase_ordinal=1,
        phase_range=(1.8, 16.0),
        phase_boundaries={"lower": None, "upper": (16.5, 0.5)},
        phase_color="#2F4DA0",
        phase_provenance={"axis_key": "temperature"},
    )
    restored = DataGroup.from_dict(phase.to_dict())
    assert restored.is_phase
    assert restored.parent_group_id == "series-1"
    assert restored.phase_ordinal == 1
    assert restored.phase_range == (1.8, 16.0)
    assert restored.phase_boundaries == {"lower": None, "upper": (16.5, 0.5)}
    assert restored.phase_color == "#2F4DA0"
    assert restored.phase_provenance == {"axis_key": "temperature"}


def test_data_group_phase_fields_tolerant_defaults_from_dict():
    # Pre-v19 dict carrying none of the phase keys migrates to "not a phase".
    restored = DataGroup.from_dict({"group_id": "g1", "name": "scan", "member_run_numbers": [1, 2]})
    assert not restored.is_phase
    assert restored.phase_range is None
    assert restored.phase_boundaries == {"lower": None, "upper": None}
    assert restored.phase_color is None
    assert restored.phase_provenance == {}


# ── group_id / exclusions / staleness (D1) ────────────────────────────────────


def test_new_series_fields_round_trip_and_default():
    fresh = _batch()
    assert fresh.group_id is None
    assert fresh.excluded_run_numbers == []
    assert fresh.last_fitted_members == []

    series = _batch(
        group_id="grp-1",
        excluded_run_numbers=[12, 10, 12],  # deduped + sorted
        last_fitted_members=[10, 11],
    )
    assert series.excluded_run_numbers == [10, 12]
    restored = FitSeries.from_dict(series.to_dict())
    assert restored.group_id == "grp-1"
    assert restored.excluded_run_numbers == [10, 12]
    assert restored.last_fitted_members == [10, 11]


def test_effective_members_applies_exclusions_in_group_order():
    group = DataGroup("grp-1", "scan", member_run_numbers=[30, 10, 20])
    series = _batch(group_id="grp-1", member_run_numbers=[10, 20, 30], excluded_run_numbers=[20])
    # Group order preserved (30, 10), excluded run dropped.
    assert series.effective_members(group) == [30, 10]


def test_effective_members_frozen_series_returns_member_snapshot():
    group = DataGroup("grp-1", "scan", member_run_numbers=[30, 10, 20])
    frozen = _batch(group_id=None, member_run_numbers=[10, 11])  # group_id None => frozen
    assert frozen.effective_members(group) == [10, 11]
    # Even given a group, a None group_id ignores it entirely.
    assert frozen.effective_members(None) == [10, 11]


def test_effective_members_groups_kind_ignores_group():
    group = DataGroup("grp-1", "scan", member_run_numbers=[30, 10, 20])
    grouped = _batch(member_kind="groups", group_id="grp-1", member_run_numbers=[-10001, -10002])
    assert grouped.effective_members(group) == [-10001, -10002]


def test_is_stale_true_when_effective_differs_from_last_fitted():
    group = DataGroup("grp-1", "scan", member_run_numbers=[10, 20, 30])
    series = _batch(group_id="grp-1", excluded_run_numbers=[], last_fitted_members=[10, 20])
    # Group has a run (30) never fitted -> stale.
    assert series.is_stale(group)


def test_is_stale_false_when_membership_matches_order_insensitively():
    group = DataGroup("grp-1", "scan", member_run_numbers=[30, 10, 20])
    series = _batch(group_id="grp-1", last_fitted_members=[10, 20, 30])
    # Same set, different order -> not stale (order resolved at fit time).
    assert not series.is_stale(group)


def test_is_stale_respects_exclusions():
    group = DataGroup("grp-1", "scan", member_run_numbers=[10, 20, 30])
    series = _batch(group_id="grp-1", excluded_run_numbers=[30], last_fitted_members=[10, 20])
    assert not series.is_stale(group)


def test_frozen_and_groups_series_are_never_stale():
    group = DataGroup("grp-1", "scan", member_run_numbers=[10, 20, 30])
    frozen = _batch(group_id=None, member_run_numbers=[10], last_fitted_members=[10])
    assert not frozen.is_stale(group)
    grouped = _batch(
        member_kind="groups",
        group_id="grp-1",
        member_run_numbers=[-10001],
        last_fitted_members=[-10001],
    )
    assert not grouped.is_stale(group)


# ── label / display_name ─────────────────────────────────────────────────────


def test_label_defaults_to_none():
    assert _batch().label is None


def test_label_stored_and_trimmed():
    assert _batch(label="  My Series  ").label == "My Series"


def test_empty_label_normalised_to_none():
    assert _batch(label="   ").label is None
    assert _batch(label="").label is None


def test_display_name_returns_label_when_set():
    assert _batch(label="Field sweep").display_name("Series 1") == "Field sweep"


def test_display_name_uses_fallback_when_no_label():
    assert _batch().display_name("Series 3") == "Series 3"


def test_label_round_trips_via_dict():
    batch = _batch(label="My label")
    restored = FitSeries.from_dict(batch.to_dict())
    assert restored.label == "My label"


def test_none_label_round_trips():
    batch = _batch()
    d = batch.to_dict()
    assert d["label"] is None
    restored = FitSeries.from_dict(d)
    assert restored.label is None


def test_from_dict_missing_label_is_none():
    d = _batch().to_dict()
    del d["label"]
    restored = FitSeries.from_dict(d)
    assert restored.label is None


# ── ProjectModel.remove_batch / rename_batch ─────────────────────────────────


def test_remove_batch_returns_series_and_unpops():
    pm = ProjectModel()
    s = _batch(batch_id="b1")
    pm.add_batch(s)
    removed = pm.remove_batch("b1")
    assert removed is s
    assert pm.batch("b1") is None


def test_remove_batch_unknown_id_returns_none():
    pm = ProjectModel()
    assert pm.remove_batch("no-such-id") is None


def test_remove_batch_sibling_survives():
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1"))
    pm.add_batch(_batch(batch_id="b2"))
    pm.remove_batch("b1")
    assert pm.batch("b2") is not None


def test_remove_batch_leaves_member_single_fits_alone():
    """Deleting a series touches only the series (D6)."""
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1", member_run_numbers=[10]))
    rep = pm.ensure_dataset(10).ensure(RepresentationType.TIME_FB_ASYMMETRY)
    rep.fit = FitSlot(model={"component_names": ["Exponential"]}, provenance="single")
    pm.remove_batch("b1")
    assert pm.batch("b1") is None
    assert rep.fit.model == {"component_names": ["Exponential"]}
    assert rep.fit.provenance == "single"


# ── active series (D5) ───────────────────────────────────────────────────────


def test_active_series_set_read_and_cleared():
    pm = ProjectModel()
    assert pm.active_series_id(_FB) is None
    pm.set_active_series(_FB, "b1")
    assert pm.active_series_id(_FB) == "b1"
    # Keyed by representation value, and accepts the value string too.
    assert pm.active_series == {"time_fb_asymmetry": "b1"}
    assert pm.active_series_id("time_fb_asymmetry") == "b1"
    # One active series per representation: another representation is separate.
    pm.set_active_series(RepresentationType.FREQ_FFT, "f1")
    assert pm.active_series_id(_FB) == "b1"
    pm.set_active_series(_FB, None)
    assert pm.active_series_id(_FB) is None
    assert pm.active_series_id(RepresentationType.FREQ_FFT) == "f1"


def test_remove_batch_clears_active_series_pointer():
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1"))
    pm.add_batch(_batch(batch_id="b2"))
    pm.set_active_series(_FB, "b1")
    pm.set_active_series(RepresentationType.FREQ_FFT, "b2")
    pm.remove_batch("b1")
    assert pm.active_series_id(_FB) is None
    assert pm.active_series_id(RepresentationType.FREQ_FFT) == "b2"


def test_rename_batch_sets_label():
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1"))
    assert pm.rename_batch("b1", "New label")
    assert pm.batch("b1").label == "New label"


def test_rename_batch_clears_label_with_none():
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1", label="Old"))
    assert pm.rename_batch("b1", None)
    assert pm.batch("b1").label is None


def test_rename_batch_clears_label_with_empty_string():
    pm = ProjectModel()
    pm.add_batch(_batch(batch_id="b1", label="Old"))
    assert pm.rename_batch("b1", "")
    assert pm.batch("b1").label is None


def test_rename_batch_unknown_id_returns_false():
    pm = ProjectModel()
    assert not pm.rename_batch("no-such-id", "X")
