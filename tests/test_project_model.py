"""Unit tests for ProjectModel (Phase 2)."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import (
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
