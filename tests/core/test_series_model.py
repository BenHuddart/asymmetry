"""Unit tests for the FitSeries object (Phase 1)."""

from __future__ import annotations

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.base import FitSlot
from asymmetry.core.representation.group import DATA_GROUP_KINDS, DataGroup
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import FitSeries, canonical_model_matches


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
    batch = _batch(results_by_run={11: {"chi": 1.0}}, diverged_runs={11})
    batch.add_member(11)  # idempotent
    assert batch.member_run_numbers.count(11) == 1
    batch.add_member(20)
    assert 20 in batch.member_run_numbers
    batch.remove_member(11)
    assert 11 not in batch.member_run_numbers
    assert 11 not in batch.results_by_run
    assert not batch.is_diverged(11)


# ── divergence ──────────────────────────────────────────────────────────────


def test_divergence_flags_exclude_from_trend():
    batch = _batch()
    batch.mark_diverged(11)
    assert batch.is_diverged(11)
    assert batch.trend_member_run_numbers() == [10, 12]
    batch.clear_diverged(11)
    assert batch.trend_member_run_numbers() == [10, 11, 12]


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
    batch = _batch(results_by_run={10: {"chi": 1.0}, 11: {"chi": 2.0}}, diverged_runs={11})
    restored = FitSeries.from_dict(batch.to_dict())
    assert restored.batch_id == batch.batch_id
    assert restored.rep_type == batch.rep_type
    assert restored.member_run_numbers == batch.member_run_numbers
    assert restored.param_roles == batch.param_roles
    assert restored.canonical_model == batch.canonical_model
    assert restored.results_by_run == batch.results_by_run
    assert restored.diverged_runs == batch.diverged_runs
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


def test_remove_batch_clears_fit_slot_batch_id():
    pm = ProjectModel()
    s = _batch(batch_id="b1", member_run_numbers=[10])
    pm.add_batch(s)
    rep = pm.ensure_dataset(10).ensure(RepresentationType.TIME_FB_ASYMMETRY)
    rep.fit = FitSlot(model={}, provenance="batch", batch_id="b1")
    pm.remove_batch("b1")
    assert rep.fit.batch_id is None
    assert rep.fit.provenance == "single"


def test_remove_batch_clears_group_series_source_run_slots():
    pm = ProjectModel()
    s = FitSeries(
        "b1",
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=[-10001],
        member_source_run={-10001: 10},
        canonical_model={},
    )
    pm.add_batch(s)
    rep = pm.ensure_dataset(10).ensure(RepresentationType.TIME_GROUPS)
    rep.fit = FitSlot(model={}, provenance="batch", batch_id="b1")
    pm.remove_batch("b1")
    assert rep.fit.batch_id is None
    assert rep.fit.provenance == "single"


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
