"""Unit tests for ProjectModel (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import (
    DataGroup,
    FitSeries,
    FitSlot,
    PhaseSpec,
    RepresentationType,
    make_representation,
)
from asymmetry.core.representation.project_model import ProjectModel

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _run(run_number: int = 7) -> Run:
    return Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.array([100.0, 80.0, 60.0, 40.0, 20.0]), bin_width=0.1, t0_bin=0),
            Histogram(counts=np.array([50.0, 45.0, 40.0, 35.0, 30.0]), bin_width=0.1, t0_bin=0),
        ],
        metadata={"field": 120.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 4,
        },
    )


def test_ensure_dataset_and_representation_access():
    model = ProjectModel()
    container = model.ensure_dataset(7)
    container.ensure(RepresentationType.TIME_FB_ASYMMETRY)
    assert model.representation(7, RepresentationType.TIME_FB_ASYMMETRY) is not None
    assert model.representation(7, RepresentationType.FREQ_FFT) is None
    assert model.representation(99, RepresentationType.TIME_FB_ASYMMETRY) is None


def test_add_and_get_batch():
    model = ProjectModel()
    batch = FitSeries("b1", RepresentationType.TIME_FB_ASYMMETRY, member_run_numbers=[1, 2])
    model.add_batch(batch)
    assert model.batch("b1") is batch
    assert model.batch("missing") is None


def _model_dict() -> dict:
    return CompositeModel(["Exponential", "Constant"], operators=["+"]).to_dict()


def _batch(batch_id: str, *, model=None, runs=(1, 2), roles=None, fit_range="0.1-8 µs"):
    return FitSeries(
        batch_id,
        _FB,
        member_run_numbers=list(runs),
        canonical_model=model if model is not None else _model_dict(),
        param_roles=roles or {"A": "local"},
        results_by_run={r: {"fit_range": fit_range} for r in runs},
    )


def test_active_series_round_trips_through_to_dict():
    model = ProjectModel()
    model.add_batch(_batch("b1"))
    model.set_active_series(_FB, "b1")

    restored = ProjectModel.from_dict(model.to_dict())
    assert restored.active_series_id(_FB) == "b1"
    # Absent block (a pre-v20 standalone payload) loads with no active series.
    payload = model.to_dict()
    del payload["active_series"]
    assert ProjectModel.from_dict(payload).active_series == {}
    assert ProjectModel.from_dict(None).active_series == {}


def test_active_series_round_trips_through_project_state():
    project = {"datasets": [{"run_number": 1, "source_file": "/tmp/a.nxs"}]}
    model = ProjectModel()
    model.add_batch(_batch("b1"))
    model.set_active_series(_FB, "b1")
    model.write_to_project_state(project)
    assert project["active_series"] == {"time_fb_asymmetry": "b1"}

    rebuilt = ProjectModel.from_project_state(project)
    assert rebuilt.active_series_id(_FB) == "b1"
    # A project saved before v20 has no active_series key at all.
    del project["active_series"]
    assert ProjectModel.from_project_state(project).active_series == {}


def test_trend_excluded_runs_round_trip_through_project_state():
    project = {"datasets": []}
    model = ProjectModel()
    model.add_batch(_batch("b1", runs=(1, 2, 3)))
    model.set_trend_excluded("b1", 2, True)

    model.write_to_project_state(project)
    rebuilt = ProjectModel.from_project_state(project)
    assert rebuilt.batch("b1").trend_excluded_runs == [2]
    assert rebuilt.batch("b1").trend_member_run_numbers() == [1, 3]


def test_standalone_round_trip():
    model = ProjectModel()
    container = model.ensure_dataset(7)
    rep = container.ensure(RepresentationType.FREQ_FFT)
    rep.recipe = {"fourier_config": {"display": "Cos"}}
    rep.fit = FitSlot(provenance="single", result={"result_html": "ok"})
    model.add_batch(FitSeries("b1", RepresentationType.FREQ_FFT, member_run_numbers=[7]))

    restored = ProjectModel.from_dict(model.to_dict())
    restored_rep = restored.representation(7, RepresentationType.FREQ_FFT)
    assert restored_rep is not None
    assert restored_rep.recipe == {"fourier_config": {"display": "Cos"}}
    assert restored_rep.fit.provenance == "single"
    assert restored.batch("b1").member_run_numbers == [7]


def test_project_state_integration_round_trip():
    project = {
        "datasets": [
            {"run_number": 7, "source_file": "/tmp/a.nxs"},
            {"run_number": 8, "source_file": "/tmp/b.nxs"},
        ],
    }
    model = ProjectModel()
    model.ensure_dataset(7).ensure(RepresentationType.TIME_FB_ASYMMETRY)
    model.add_batch(
        FitSeries("b1", RepresentationType.TIME_FB_ASYMMETRY, member_run_numbers=[7, 8])
    )
    model.write_to_project_state(project)

    # Dataset 7 has a representation; dataset 8 gets an empty map.
    assert "time_fb_asymmetry" in project["datasets"][0]["representations"]
    assert project["datasets"][1]["representations"] == {}
    assert project["batches"][0]["batch_id"] == "b1"

    rebuilt = ProjectModel.from_project_state(project)
    assert rebuilt.representation(7, RepresentationType.TIME_FB_ASYMMETRY) is not None
    assert rebuilt.batch("b1").member_run_numbers == [7, 8]


def test_recompute_all_populates_primary_and_survives_bad_recipe():
    model = ProjectModel()
    container = model.ensure_dataset(7)
    fb = make_representation(RepresentationType.TIME_FB_ASYMMETRY)
    container.by_type[RepresentationType.TIME_FB_ASYMMETRY] = fb
    # MaxEnt opts out of load-time recomputation: it is an expensive iterative
    # reconstruction that must not run synchronously during project load.
    maxent = make_representation(
        RepresentationType.FREQ_MAXENT,
        recipe={"maxent_config": {"selected_group_ids": [999]}},
    )
    container.by_type[RepresentationType.FREQ_MAXENT] = maxent

    assert fb.primary is None
    assert maxent.recompute_on_load is False
    model.recompute_all({7: _run(7)})
    assert fb.primary is not None
    assert maxent.primary is None  # deferred: recomputed on demand, not at load


def test_recompute_all_preserves_persisted_result_metadata():
    """Neither a skipped nor a failed recompute may destroy loaded metadata."""
    model = ProjectModel()
    container = model.ensure_dataset(7)
    maxent = make_representation(
        RepresentationType.FREQ_MAXENT,
        recipe={"maxent_config": {"n_spectrum_points": 64}},
        result_metadata={"cycles": 25, "diagnostics": {"chi2": [1.0]}},
    )
    container.by_type[RepresentationType.FREQ_MAXENT] = maxent

    model.recompute_all({7: _run(7)})
    assert maxent.result_metadata["cycles"] == 25

    maxent.invalidate()  # recipe-change invalidation keeps persisted metadata
    assert maxent.result_metadata["cycles"] == 25


def test_computed_series_is_flagged():
    """A model-less "computed" series (e.g. an integral scan) owns no fit model."""
    pm = ProjectModel()
    real = _batch("b1", runs=(10, 11))
    pm.add_batch(real)
    scan = FitSeries(
        "scan-1",
        _FB,
        member_run_numbers=[10, 11],
        canonical_model=None,
        param_roles={},
        results_by_run={
            10: {"success": True, "parameters": {"Integral asymmetry": 0.1}},
            11: {"success": True, "parameters": {"Integral asymmetry": 0.2}},
        },
    )
    pm.add_batch(scan)
    assert scan.is_computed is True
    assert real.is_computed is False


def test_recompute_all_skips_missing_runs():
    model = ProjectModel()
    container = model.ensure_dataset(7)
    fb = make_representation(RepresentationType.TIME_FB_ASYMMETRY)
    container.by_type[RepresentationType.TIME_FB_ASYMMETRY] = fb
    model.recompute_all({})  # run 7 not supplied
    assert fb.primary is None


def test_maxent_recipe_and_reconstruction_survive_project_round_trip():
    """A MaxEnt recipe carrying every new field, plus the TIME_MAXENT_RECON
    representation, must survive a full project round-trip unchanged — the
    schema is additive (``MaxEntConfig.from_dict`` defaults absent keys), so no
    migration is needed and ``recompute_on_load`` stays off for both."""
    from asymmetry.core.maxent import MaxEntConfig

    config = MaxEntConfig(
        n_spectrum_points=512,
        mode="zf_lf",
        selected_group_ids=[1, 2],
        pulse_mode="double",
        pulse_half_width_us=0.08,
        pulse_separation_us=0.324,
        exclude_t_min_us=1.5,
        exclude_t_max_us=2.5,
        specbg_enabled=True,
        specbg_gaussian_width_mhz=0.2,
        specbg_lorentzian_width_mhz=0.15,
        specbg_lorentzian_fraction=0.3,
        show_reconstruction=True,
    )
    recipe = {"maxent_config": config.to_dict()}

    model = ProjectModel()
    container = model.ensure_dataset(7)
    spectrum_rep = container.ensure(RepresentationType.FREQ_MAXENT)
    spectrum_rep.recipe = dict(recipe)
    spectrum_rep.result_metadata = {"cycles": 25}
    recon_rep = container.ensure(RepresentationType.TIME_MAXENT_RECON)
    recon_rep.recipe = dict(recipe)

    # Both expensive representations opt out of load-time recomputation.
    assert spectrum_rep.recompute_on_load is False
    assert recon_rep.recompute_on_load is False

    # Standalone and project-state round-trips must both preserve the pair.
    for restored in (
        ProjectModel.from_dict(model.to_dict()),
        ProjectModel.from_project_state(
            _project_state_with(ProjectModel.from_dict(model.to_dict()))
        ),
    ):
        restored_spectrum = restored.representation(7, RepresentationType.FREQ_MAXENT)
        restored_recon = restored.representation(7, RepresentationType.TIME_MAXENT_RECON)
        assert restored_spectrum is not None
        assert restored_recon is not None
        # The recipe block survives verbatim, and rebuilding the config recovers
        # every new field (no schema migration, defaults untouched).
        assert restored_spectrum.recipe == recipe
        assert restored_recon.recipe == recipe
        rebuilt = MaxEntConfig.from_dict(restored_spectrum.recipe["maxent_config"])
        assert rebuilt.mode == "zf_lf"
        assert rebuilt.pulse_mode == "double"
        assert rebuilt.exclude_t_min_us == 1.5
        assert rebuilt.specbg_enabled is True
        assert rebuilt.show_reconstruction is True
        assert restored_spectrum.recompute_on_load is False
        assert restored_recon.recompute_on_load is False
        assert restored_spectrum.result_metadata["cycles"] == 25


def _project_state_with(model: ProjectModel) -> dict:
    """Write *model* into a fresh project-state dict (its datasets registered)."""
    project = {"datasets": [{"run_number": 7, "source_file": "/tmp/a.nxs"}]}
    model.write_to_project_state(project)
    return project


def test_fitseries_extra_round_trips():
    # The freeform `extra` dict (carries the ALC scan's analysis) survives
    # to_dict/from_dict; ordinary series default to an empty dict.
    series = FitSeries(
        "scan-1",
        _FB,
        extra={"kind": "alc_scan", "regions": [[0.0, 100.0]], "baseline_fitted": True},
    )
    restored = FitSeries.from_dict(series.to_dict())
    assert restored.extra == {
        "kind": "alc_scan",
        "regions": [[0.0, 100.0]],
        "baseline_fitted": True,
    }
    assert FitSeries("b", _FB).extra == {}
    assert FitSeries.from_dict(FitSeries("b", _FB).to_dict()).extra == {}


# ── Phase 7: DataGroup <-> FitSeries linking (D1, README §6 Option B) ──────


def test_fitseries_source_group_id_round_trips():
    series = FitSeries("b1", _FB, member_run_numbers=[1, 2], source_group_id="grp-1")
    restored = FitSeries.from_dict(series.to_dict())
    assert restored.source_group_id == "grp-1"
    # Default / ad-hoc series carry no provenance.
    assert FitSeries("b2", _FB).source_group_id is None
    assert FitSeries.from_dict(FitSeries("b2", _FB).to_dict()).source_group_id is None


def test_data_group_round_trips_through_project_model():
    model = ProjectModel()
    model.add_data_group(DataGroup("grp-1", "T = 150 K", member_run_numbers=[7, 8]))
    assert model.data_group("grp-1").name == "T = 150 K"
    assert model.data_group("missing") is None

    restored = ProjectModel.from_dict(model.to_dict())
    restored_group = restored.data_group("grp-1")
    assert restored_group is not None
    assert restored_group.name == "T = 150 K"
    assert restored_group.member_run_numbers == [7, 8]
    assert restored_group.order_key == "run"


def test_data_groups_round_trip_through_project_state():
    project = {"datasets": []}
    model = ProjectModel()
    model.add_data_group(DataGroup("grp-1", "B = 60 G", member_run_numbers=[1, 2, 3]))
    model.write_to_project_state(project)

    assert project["data_groups"][0]["group_id"] == "grp-1"

    rebuilt = ProjectModel.from_project_state(project)
    assert rebuilt.data_group("grp-1").member_run_numbers == [1, 2, 3]


def test_data_groups_absent_block_loads_empty_without_crashing():
    """A project saved before Phase 7 has no top-level data_groups key at all."""
    legacy_project = {
        "datasets": [{"run_number": 7, "source_file": "/tmp/a.nxs"}],
        "batches": [],
    }
    assert "data_groups" not in legacy_project
    model = ProjectModel.from_project_state(legacy_project)
    assert model.data_groups == {}

    # And from_dict is equally tolerant of the standalone form.
    assert ProjectModel.from_dict({"representations_by_run": {}}).data_groups == {}
    assert ProjectModel.from_dict(None).data_groups == {}


def test_series_for_group_computes_back_references():
    """Back-references are computed from batches, not stored on the group (D1)."""
    model = ProjectModel()
    model.add_data_group(DataGroup("grp-1", "T = 150 K", member_run_numbers=[1, 2]))
    linked = _batch("b1", runs=(1, 2))
    linked.source_group_id = "grp-1"
    unrelated = _batch("b2", runs=(3, 4))
    model.add_batch(linked)
    model.add_batch(unrelated)

    assert model.series_for_group("grp-1") == [linked]
    assert model.series_for_group("missing") == []

    # Editing the group's membership does not retroactively touch the series
    # already built from it — the back-reference is recomputed, not cached.
    model.data_group("grp-1").member_run_numbers = [1]
    assert model.series_for_group("grp-1") == [linked]
    assert linked.member_run_numbers == [1, 2]


# ── group mutation API (D1/D4/D7) ─────────────────────────────────────────────


def test_create_data_group_mints_uuid_and_honours_kind():
    model = ProjectModel()
    group = model.create_data_group("scan", [1, 2], kind="auto")
    assert group.kind == "auto"
    assert group.group_id  # a minted uuid4
    assert model.data_group(group.group_id) is group
    explicit = model.create_data_group("named", [3], group_id="grp-x", order_key="field")
    assert explicit.group_id == "grp-x"
    assert explicit.order_key == "field"
    assert explicit.kind == "user"


def test_rename_data_group_promotes_auto_to_user():
    model = ProjectModel()
    group = model.create_data_group("Runs 1–2", [1, 2], kind="auto")
    assert model.rename_data_group(group.group_id, "Field scan")
    assert group.name == "Field scan"
    assert group.kind == "user"  # promoted on rename (D4)
    # Renaming a user group leaves its kind alone.
    assert model.rename_data_group(group.group_id, "Field scan v2")
    assert group.kind == "user"
    assert not model.rename_data_group("missing", "x")


def test_set_data_group_members_replaces_membership():
    model = ProjectModel()
    group = model.create_data_group("scan", [1, 2])
    assert model.set_data_group_members(group.group_id, [3, 4, 5])
    assert group.member_run_numbers == [3, 4, 5]
    assert not model.set_data_group_members("missing", [1])


def test_find_auto_group_matches_member_set_only_for_auto_kind():
    model = ProjectModel()
    auto = model.create_data_group("auto", [2, 1], kind="auto")
    model.create_data_group("user", [5, 6], kind="user")
    # Set identity, order-insensitive.
    assert model.find_auto_group([1, 2]) is auto
    assert model.find_auto_group([2, 1]) is auto
    # A user group with matching members is never reused.
    assert model.find_auto_group([5, 6]) is None
    assert model.find_auto_group([9]) is None


def test_remove_data_group_freeze_branch_snapshots_and_orphans():
    model = ProjectModel()
    group = model.create_data_group("scan", [1, 2, 3])
    linked = _batch("b1", runs=(1, 2, 3))
    linked.group_id = group.group_id
    # last_fitted is the actual snapshot: run 3 was excluded at fit time.
    linked.last_fitted_members = [1, 2]
    model.add_batch(linked)

    removed = model.remove_data_group(group.group_id, orphan_series=True)
    assert removed == []  # nothing deleted
    assert model.data_group(group.group_id) is None
    # Series survives, frozen: group_id cleared, members snapshotted to what was fit.
    assert model.batch("b1") is linked
    assert linked.group_id is None
    assert linked.member_run_numbers == [1, 2]


def test_remove_data_group_delete_branch_removes_series_and_returns_ids():
    model = ProjectModel()
    group = model.create_data_group("scan", [1, 2])
    linked = _batch("b1", runs=(1, 2))
    linked.group_id = group.group_id
    # Run 1 has its own single fit; deleting the group's series must not touch it.
    rep1 = model.ensure_dataset(1).ensure(_FB)
    rep1.fit = FitSlot(model=_model_dict(), provenance="single")
    model.add_batch(linked)
    model.set_active_series(_FB, "b1")

    removed = model.remove_data_group(group.group_id, orphan_series=False)
    assert removed == ["b1"]
    assert model.batch("b1") is None
    # The run keeps its single fit; the active pointer at the deleted series goes.
    assert rep1.fit.provenance == "single"
    assert rep1.fit.model == _model_dict()
    assert model.active_series_id(_FB) is None


def test_remove_data_group_resolves_legacy_source_group_id_fallback():
    model = ProjectModel()
    group = model.create_data_group("scan", [1, 2])
    # A migrated/legacy series linked only by source_group_id (group_id None).
    legacy = _batch("b1", runs=(1, 2))
    legacy.source_group_id = group.group_id
    model.add_batch(legacy)

    removed = model.remove_data_group(group.group_id, orphan_series=False)
    assert removed == ["b1"]
    assert model.batch("b1") is None


def test_remove_data_group_unknown_id_returns_empty():
    assert ProjectModel().remove_data_group("missing", orphan_series=True) == []


# ── phase groups (Global Fit Wizard transitions, D1) ──────────────────────────


def _run_with_temperature(run_number: int, temperature: float) -> Run:
    return Run(run_number=run_number, metadata={"temperature": temperature})


def _phase(ordinal, name, members, *, lower=None, upper=None, color=None):
    return PhaseSpec(
        ordinal=ordinal,
        name=name,
        member_run_numbers=tuple(members),
        phase_range=None,
        phase_boundaries={"lower": lower, "upper": upper},
        phase_color=color,
        phase_provenance={},
    )


def test_create_phase_groups_creates_nested_groups_in_ordinal_order():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4, 5, 6], order_key="temperature")
    ids = model.create_phase_groups(
        parent.group_id,
        [
            _phase(2, "Phase II", [4, 5, 6], lower=(3.5, 0.5)),
            _phase(1, "Phase I", [1, 2, 3], upper=(3.5, 0.5)),
        ],
    )
    phases = model.phase_groups_for(parent.group_id)
    assert [p.name for p in phases] == ["Phase I", "Phase II"]
    assert [p.group_id for p in phases] == ids
    assert phases[0].parent_group_id == parent.group_id
    assert phases[0].member_run_numbers == [1, 2, 3]
    assert phases[0].is_phase
    assert phases[0].order_key == "temperature"  # inherited from the parent
    assert phases[0].phase_boundaries == {"lower": None, "upper": (3.5, 0.5)}
    assert phases[1].phase_boundaries == {"lower": (3.5, 0.5), "upper": None}
    assert model.excluded_runs_for(parent.group_id) == []


def test_create_phase_groups_rejects_members_outside_parent():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3])
    with pytest.raises(ValueError):
        model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [1, 2, 99])])
    assert model.phase_groups_for(parent.group_id) == []


def test_create_phase_groups_rejects_overlapping_phases():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4])
    with pytest.raises(ValueError):
        model.create_phase_groups(
            parent.group_id,
            [_phase(1, "Phase I", [1, 2, 3]), _phase(2, "Phase II", [3, 4])],
        )
    assert model.phase_groups_for(parent.group_id) == []


def test_create_phase_groups_unknown_parent_raises():
    with pytest.raises(ValueError):
        ProjectModel().create_phase_groups("missing", [])


def test_create_phase_groups_excludes_unclaimed_members():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4, 5])
    model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [1, 2, 3])])
    assert model.excluded_runs_for(parent.group_id) == [4, 5]


def test_create_phase_groups_replaces_existing_partition_and_deletes_old_series():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4, 5, 6])
    model.create_phase_groups(
        parent.group_id,
        [_phase(1, "Phase I", [1, 2, 3]), _phase(2, "Phase II", [4, 5, 6])],
    )
    old_phase_1 = model.phase_groups_for(parent.group_id)[0]
    series = _batch("b1", runs=(1, 2, 3))
    series.group_id = old_phase_1.group_id
    model.add_batch(series)

    model.create_phase_groups(
        parent.group_id,
        [_phase(1, "New Phase I", [1, 2]), _phase(2, "New Phase II", [3, 4, 5, 6])],
    )
    # A re-partition deletes the previous phases' series (not freeze) — the
    # fits belonged to a partition that no longer exists.
    assert model.data_group(old_phase_1.group_id) is None
    assert model.batch("b1") is None
    new_phases = model.phase_groups_for(parent.group_id)
    assert [p.name for p in new_phases] == ["New Phase I", "New Phase II"]


def test_phase_groups_for_unknown_parent_returns_empty():
    assert ProjectModel().phase_groups_for("missing") == []


def test_find_auto_group_never_returns_a_phase_group():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3])
    (phase_id,) = model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [1, 2, 3])])
    # Even though the phase's member set matches exactly, it is never
    # returned — a phase is always kind="user", never "auto".
    assert model.find_auto_group([1, 2, 3]) is None
    assert model.data_group(phase_id).kind == "user"


def test_rename_data_group_on_a_phase_keeps_phase_fields():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3])
    (phase_id,) = model.create_phase_groups(
        parent.group_id, [_phase(1, "Phase I", [1, 2, 3], lower=(0.5, 0.1))]
    )
    assert model.rename_data_group(phase_id, "Low-T phase")
    phase = model.data_group(phase_id)
    assert phase.name == "Low-T phase"
    assert phase.is_phase
    assert phase.parent_group_id == parent.group_id
    assert phase.phase_ordinal == 1
    assert phase.phase_boundaries == {"lower": (0.5, 0.1), "upper": None}


def test_remove_data_group_cascades_to_phase_groups_delete_branch():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4])
    ids = model.create_phase_groups(
        parent.group_id, [_phase(1, "Phase I", [1, 2]), _phase(2, "Phase II", [3, 4])]
    )
    for i, gid in enumerate(ids):
        series = _batch(f"b{i}", runs=tuple(model.data_group(gid).member_run_numbers))
        series.group_id = gid
        model.add_batch(series)

    removed = model.remove_data_group(parent.group_id, orphan_series=False)
    assert set(removed) == {"b0", "b1"}
    assert model.data_group(parent.group_id) is None
    for gid in ids:
        assert model.data_group(gid) is None
    assert model.batch("b0") is None
    assert model.batch("b1") is None


def test_remove_data_group_cascades_to_phase_groups_freeze_branch():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3])
    (phase_id,) = model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [1, 2, 3])])
    series = _batch("b1", runs=(1, 2, 3))
    series.group_id = phase_id
    series.last_fitted_members = [1, 2, 3]
    model.add_batch(series)

    removed = model.remove_data_group(parent.group_id, orphan_series=True)
    assert removed == []
    assert model.data_group(parent.group_id) is None
    assert model.data_group(phase_id) is None
    assert model.batch("b1") is series
    assert series.group_id is None
    assert series.member_run_numbers == [1, 2, 3]


def test_move_run_to_phase_updates_membership_and_range():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4], order_key="temperature")
    ids = model.create_phase_groups(
        parent.group_id, [_phase(1, "Phase I", [1, 2]), _phase(2, "Phase II", [3, 4])]
    )
    runs_by_number = {r: _run_with_temperature(r, float(r)) for r in (1, 2, 3, 4)}

    model.move_run_to_phase(
        2, ids[1], series_group_id=parent.group_id, runs_by_number=runs_by_number
    )

    phase1 = model.data_group(ids[0])
    phase2 = model.data_group(ids[1])
    assert phase1.member_run_numbers == [1]
    assert phase2.member_run_numbers == [2, 3, 4]  # re-ordered along the axis
    assert phase1.phase_range == (1.0, 1.0)
    assert phase2.phase_range == (2.0, 4.0)
    # Parent membership is untouched — a phase's members are a live subset.
    assert parent.member_run_numbers == [1, 2, 3, 4]


def test_move_run_to_phase_to_excluded():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3], order_key="run")
    ids = model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [1, 2, 3])])

    model.move_run_to_phase(2, None, series_group_id=parent.group_id, runs_by_number={})

    phase = model.data_group(ids[0])
    assert phase.member_run_numbers == [1, 3]
    assert model.excluded_runs_for(parent.group_id) == [2]


def test_move_run_to_phase_marks_both_affected_series_stale():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3, 4], order_key="run")
    ids = model.create_phase_groups(
        parent.group_id, [_phase(1, "Phase I", [1, 2]), _phase(2, "Phase II", [3, 4])]
    )
    series1 = _batch("b1", runs=(1, 2))
    series1.group_id = ids[0]
    series1.last_fitted_members = [1, 2]
    series2 = _batch("b2", runs=(3, 4))
    series2.group_id = ids[1]
    series2.last_fitted_members = [3, 4]
    model.add_batch(series1)
    model.add_batch(series2)
    phase1 = model.data_group(ids[0])
    phase2 = model.data_group(ids[1])
    assert not series1.is_stale(phase1)
    assert not series2.is_stale(phase2)

    model.move_run_to_phase(2, ids[1], series_group_id=parent.group_id, runs_by_number={})

    assert series1.is_stale(phase1)
    assert series2.is_stale(phase2)


def test_move_run_to_phase_unknown_phase_raises():
    model = ProjectModel()
    parent = model.create_data_group("scan", [1, 2, 3])
    model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [1, 2, 3])])
    with pytest.raises(ValueError):
        model.move_run_to_phase(1, "missing", series_group_id=parent.group_id, runs_by_number={})


def test_move_run_to_phase_run_not_in_targets_parent_raises():
    model = ProjectModel()
    model.create_data_group("scan a", [1, 2, 3])
    parent_b = model.create_data_group("scan b", [10, 11, 12])
    (phase_b,) = model.create_phase_groups(parent_b.group_id, [_phase(1, "Phase I", [10, 11, 12])])
    with pytest.raises(ValueError, match="not a member of series group"):
        model.move_run_to_phase(1, phase_b, series_group_id=parent_b.group_id, runs_by_number={})


def test_move_run_to_phase_touches_only_the_named_series():
    # A run in two partitioned series: excluding it from one leaves the
    # other's phase untouched.
    model = ProjectModel()
    first = model.create_data_group("scan A", [1, 2, 3], order_key="run")
    second = model.create_data_group("scan B", [2, 3, 4], order_key="run")
    first_ids = model.create_phase_groups(first.group_id, [_phase(1, "Phase I", [1, 2, 3])])
    second_ids = model.create_phase_groups(second.group_id, [_phase(1, "Phase I", [2, 3, 4])])

    model.move_run_to_phase(2, None, series_group_id=first.group_id, runs_by_number={})

    assert model.data_group(first_ids[0]).member_run_numbers == [1, 3]
    assert model.data_group(second_ids[0]).member_run_numbers == [2, 3, 4]


def test_move_run_to_phase_rejects_a_phase_of_another_series():
    model = ProjectModel()
    first = model.create_data_group("scan A", [1, 2], order_key="run")
    second = model.create_data_group("scan B", [1, 2], order_key="run")
    model.create_phase_groups(first.group_id, [_phase(1, "Phase I", [1, 2])])
    second_ids = model.create_phase_groups(second.group_id, [_phase(1, "Phase I", [1, 2])])
    with pytest.raises(ValueError, match="not a phase of series group"):
        model.move_run_to_phase(1, second_ids[0], series_group_id=first.group_id, runs_by_number={})


def test_phase_members_follow_the_parent_order_not_run_number():
    # The parent lists its members in sweep order; a phase inherits that order.
    model = ProjectModel()
    parent = model.create_data_group("scan", [30, 10, 20], order_key="temperature")
    ids = model.create_phase_groups(parent.group_id, [_phase(1, "Phase I", [20, 10, 30])])
    assert model.data_group(ids[0]).member_run_numbers == [30, 10, 20]


def test_phase_group_names_use_conventional_numerals():
    from asymmetry.core.representation.group import phase_group_name

    assert [phase_group_name(n) for n in (1, 4, 9, 14, 40, 49, 90, 400, 1994)] == [
        "Phase I",
        "Phase IV",
        "Phase IX",
        "Phase XIV",
        "Phase XL",
        "Phase XLIX",
        "Phase XC",
        "Phase CD",
        "Phase MCMXCIV",
    ]
