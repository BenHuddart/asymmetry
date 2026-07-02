"""Unit tests for ProjectModel (Phase 2)."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import (
    DataGroup,
    FitSeries,
    FitSlot,
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


def test_remove_superseded_batches_dedupes_identical_rerun():
    """Re-running the same batch supersedes the earlier identical series.

    The replacement inherits the superseded twin's ``batch_id`` so the chip and
    any back-references stay stable across the re-run.
    """
    model = ProjectModel()
    first = _batch("b1")
    model.add_batch(first)
    second = _batch("b2")  # same model / runs / roles / range as b1
    removed = model.remove_superseded_batches(second)
    assert removed == ["b1"]
    assert second.batch_id == "b1"  # inherited the stable id
    model.add_batch(second)
    assert model.batch("b1") is second
    assert model.batch("b2") is None


def test_remove_superseded_ignores_fit_range_and_roles():
    """Fit window and Global/Local split are attributes, not identity (D4).

    Re-running the same members+model with a different fit window or a different
    parameter classification supersedes the earlier series in place rather than
    spawning a duplicate trend pill.
    """
    model = ProjectModel()
    model.add_batch(_batch("b1", fit_range="0.1-8 µs", roles={"A": "global"}))
    # Same members + model, but a narrower window and an all-local split.
    other = _batch("b2", fit_range="0.2-6 µs", roles={"A": "local"})
    assert model.remove_superseded_batches(other) == ["b1"]


def test_remove_superseded_keeps_distinct_model():
    """A different model is not a duplicate even over the same runs."""
    model = ProjectModel()
    model.add_batch(_batch("b1"))
    other = _batch(
        "b2",
        model=CompositeModel(["Gaussian", "Constant"], operators=["+"]).to_dict(),
    )
    assert model.remove_superseded_batches(other) == []


def test_remove_superseded_ignores_computed_series():
    """Computed (model-less) scans over the same runs are not de-duplicated."""
    model = ProjectModel()
    scan = FitSeries("s1", _FB, member_run_numbers=[1, 2], canonical_model=None)
    model.add_batch(scan)
    other = FitSeries("s2", _FB, member_run_numbers=[1, 2], canonical_model=None)
    assert model.remove_superseded_batches(other) == []


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


def _batched_model(pm: ProjectModel, model: dict) -> FitSeries:
    for run_number in (10, 11):
        rep = pm.ensure_dataset(run_number).ensure(_FB)
        rep.fit = FitSlot(model=model, provenance="batch", batch_id="b1")
    batch = FitSeries(
        "b1", _FB, member_run_numbers=[10, 11], canonical_model=model, param_roles={"A": "local"}
    )
    pm.add_batch(batch)
    return batch


def test_refresh_divergence_flags_excludes_and_re_includes():
    model_a = CompositeModel(["Exponential", "Constant"]).to_dict()
    model_b = CompositeModel(["Gaussian", "Constant"]).to_dict()
    pm = ProjectModel()
    batch = _batched_model(pm, model_a)

    pm.refresh_divergence()
    assert pm.trend_runs_for_batch(batch) == [10, 11]
    assert not batch.is_diverged(11)

    # Edit member 11's model -> diverged, excluded from trend by default.
    pm.representation(11, _FB).fit.model = model_b
    pm.refresh_divergence()
    assert batch.is_diverged(11)
    assert pm.representation(11, _FB).fit.diverged
    assert pm.trend_runs_for_batch(batch) == [10]

    # Manual re-inclusion is honoured and preserved across refresh.
    pm.set_member_trend_inclusion("b1", 11, True)
    assert pm.trend_runs_for_batch(batch) == [10, 11]
    pm.refresh_divergence()
    assert pm.trend_runs_for_batch(batch) == [10, 11]
    assert batch.is_diverged(11)  # still flagged, just re-included

    # Revert the model -> re-converges, flag cleared.
    pm.representation(11, _FB).fit.model = model_a
    pm.refresh_divergence()
    assert not batch.is_diverged(11)
    assert pm.trend_runs_for_batch(batch) == [10, 11]


def test_computed_series_is_flagged_and_skips_divergence():
    # A model-less "computed" series (e.g. an integral scan) must not flip the
    # divergence/trend state of a real fit that shares the same run numbers.
    model_a = CompositeModel(["Exponential", "Constant"]).to_dict()
    pm = ProjectModel()
    real = _batched_model(pm, model_a)  # real fit on runs 10, 11
    pm.refresh_divergence()
    assert pm.representation(11, _FB).fit.include_in_trend is True

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
    assert scan.is_computed is True
    assert real.is_computed is False
    pm.add_batch(scan)

    pm.refresh_divergence()
    # The real fit's per-run state is untouched by the computed series.
    assert pm.representation(11, _FB).fit.diverged is False
    assert pm.representation(11, _FB).fit.include_in_trend is True
    assert not real.is_diverged(11)


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


def test_fitseries_source_group_id_excluded_from_identity_signature():
    """Provenance is an attribute, not identity (same lesson as param_roles/fit_range).

    Re-running the same members+model under a different group association still
    supersedes the earlier series in place rather than spawning a duplicate.
    """
    model = ProjectModel()
    first = _batch("b1")
    first.source_group_id = "grp-1"
    model.add_batch(first)
    second = _batch("b2")
    second.source_group_id = "grp-2"  # different provenance, same identity
    assert model.remove_superseded_batches(second) == ["b1"]


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


def test_remove_data_group_leaves_its_series_untouched():
    model = ProjectModel()
    model.add_data_group(DataGroup("grp-1", "T = 150 K", member_run_numbers=[1, 2]))
    linked = _batch("b1", runs=(1, 2))
    linked.source_group_id = "grp-1"
    model.add_batch(linked)

    removed = model.remove_data_group("grp-1")
    assert removed is not None and removed.group_id == "grp-1"
    assert model.data_group("grp-1") is None
    assert model.batch("b1") is linked
    assert linked.source_group_id == "grp-1"
