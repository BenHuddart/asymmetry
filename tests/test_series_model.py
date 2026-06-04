"""Unit tests for the FitSeries object (Phase 1)."""

from __future__ import annotations

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import RepresentationType
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
