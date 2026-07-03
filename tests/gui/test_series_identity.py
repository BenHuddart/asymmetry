"""Phase 1: series identity, replace-in-place, unified naming (D4/D8, F13/F22).

Covers the narrowed identity signature (roles/fit-window are attributes, not
identity), label + ``batch_id`` carry-over when a re-run supersedes its twin,
load-time dedupe of duplicate-era projects, the unified naming helper, and a GUI
round-trip proving a re-run with a *changed classification* updates the single
chip in place instead of spawning a duplicate.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import (
    FitSeries,
    RepresentationType,
    default_series_label,
    member_range,
)
from asymmetry.core.representation.project_model import ProjectModel

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _model(names=("Exponential", "Constant")) -> dict:
    return CompositeModel(list(names), operators=["+"] * (len(names) - 1)).to_dict()


def _series(
    batch_id: str,
    *,
    label=None,
    model=None,
    runs=(1, 2),
    roles=None,
    fit_range="0.1-8 µs",
    member_kind="runs",
    member_source_run=None,
) -> FitSeries:
    return FitSeries(
        batch_id,
        _FB,
        label=label,
        member_kind=member_kind,
        member_run_numbers=list(runs),
        member_source_run=member_source_run,
        canonical_model=model if model is not None else _model(),
        param_roles=roles or {"A": "local"},
        results_by_run={r: {"fit_range": fit_range} for r in runs},
    )


# ── identity signature narrowing ────────────────────────────────────────────


def test_signature_ignores_roles_and_fit_range():
    """Same members+model but different Global/Local split or window → supersede."""
    model = ProjectModel()
    model.add_batch(_series("b1", roles={"A": "global"}, fit_range="0.1-8 µs"))
    rerun = _series("b2", roles={"A": "local"}, fit_range="0.2-6 µs")
    assert model.superseded_batch_ids(rerun) == ["b1"]


def test_signature_distinguishes_model():
    model = ProjectModel()
    model.add_batch(_series("b1", model=_model(("Exponential", "Constant"))))
    other = _series("b2", model=_model(("Gaussian", "Constant")))
    assert model.superseded_batch_ids(other) == []


def test_signature_distinguishes_members():
    model = ProjectModel()
    model.add_batch(_series("b1", runs=(1, 2)))
    other = _series("b2", runs=(1, 3))
    assert model.superseded_batch_ids(other) == []


def test_signature_uses_decoded_source_runs_for_partial_group_map():
    """Group series with an empty member_source_run map still key off real runs.

    Regression: computing signature members from raw member_source_run.values()
    yields () for a legacy group series with no map, collapsing DISTINCT group
    fits in dedupe_batches. Routing through source_runs() (synthetic-key decode)
    keeps them distinct.
    """
    model = ProjectModel()
    # Two grouped series over different runs, both WITHOUT a member_source_run map
    # (synthetic keys run*1000+group; source_run_for decodes abs(key)//1000).
    a = FitSeries(
        "g1",
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=[-2961001, -2961002],
        canonical_model=_model(),
    )
    b = FitSeries(
        "g2",
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=[-2967001, -2967002],
        canonical_model=_model(),
    )
    model.add_batch(a)
    model.add_batch(b)
    # Distinct source runs (2961 vs 2967) → not duplicates → not collapsed.
    assert model.dedupe_batches() == []
    assert set(model.batches) == {"g1", "g2"}


def test_source_runs_accessor_decodes_group_keys():
    series = FitSeries(
        "g",
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=[-2961002, -2961001],
        canonical_model=_model(),
    )
    assert series.source_runs() == [2961]


def test_signature_distinguishes_group_subsets_over_same_source_runs():
    """Two grouped fits over the same source run(s) but different detector-group
    subsets are DISTINCT series and must not supersede/merge each other.

    Regression: keying identity on the deduplicated physical runs (source_runs())
    collapsed both to the same signature; keying on the synthetic (run, group)
    member keys keeps them apart.
    """
    model = ProjectModel()
    two_groups = FitSeries(
        "g-2",
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=[-2961001, -2961002],  # run 2961, groups 1 & 2
        canonical_model=_model(),
    )
    three_groups = FitSeries(
        "g-3",
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=[-2961001, -2961002, -2961003],  # run 2961, groups 1-3
        canonical_model=_model(),
    )
    model.add_batch(two_groups)
    model.add_batch(three_groups)
    # Same source run + same model, but different group subsets → NOT duplicates.
    assert model.superseded_batch_ids(three_groups) == []
    assert model.dedupe_batches() == []
    assert set(model.batches) == {"g-2", "g-3"}


def test_signature_distinguishes_representation():
    model = ProjectModel()
    model.add_batch(_series("b1"))
    other = FitSeries(
        "b2",
        RepresentationType.FREQ_FFT,
        member_run_numbers=[1, 2],
        canonical_model=_model(),
    )
    assert model.superseded_batch_ids(other) == []


# ── label + batch_id carry-over on replacement ──────────────────────────────


def test_replacement_inherits_user_label_and_batch_id():
    """A user rename and the stable id survive an in-place re-run."""
    model = ProjectModel()
    model.add_batch(_series("b1", label="Field sweep", roles={"A": "global"}))
    rerun = _series("b2", label=None, roles={"A": "local"})
    removed = model.remove_superseded_batches(rerun)
    assert removed == ["b1"]
    assert rerun.batch_id == "b1"  # stable id inherited
    assert rerun.label == "Field sweep"  # user rename carried forward
    model.add_batch(rerun)
    assert list(model.batches) == ["b1"]


def test_replacement_without_user_label_stays_unlabelled():
    """An unlabelled twin leaves the replacement unlabelled (renders the default)."""
    model = ProjectModel()
    model.add_batch(_series("b1", label=None))
    rerun = _series("b2", label=None)
    model.remove_superseded_batches(rerun)
    assert rerun.batch_id == "b1"
    assert rerun.label is None


def test_replacement_keeps_new_label_over_old():
    """A rename on the replacement itself is not clobbered by carry-over."""
    model = ProjectModel()
    model.add_batch(_series("b1", label="Old name"))
    rerun = _series("b2", label="New name")
    model.remove_superseded_batches(rerun)
    assert rerun.label == "New name"


# ── load-time dedupe ─────────────────────────────────────────────────────────


def test_dedupe_collapses_duplicates_keeping_most_recent():
    """Legacy duplicate-era batches collapse to the last (freshest) one."""
    model = ProjectModel()
    model.add_batch(_series("b1", label="Field sweep"))
    model.add_batch(_series("b2"))  # identical signature, no label, more recent
    records = model.dedupe_batches()
    assert list(model.batches) == ["b2"]  # kept the most recent
    assert model.batch("b2").label == "Field sweep"  # user label carried forward
    assert len(records) == 1
    assert records[0]["kept"] == "b2"
    assert records[0]["dropped"] == ["b1"]


def test_dedupe_no_op_when_unique():
    model = ProjectModel()
    model.add_batch(_series("b1", runs=(1, 2)))
    model.add_batch(_series("b2", runs=(3, 4)))
    assert model.dedupe_batches() == []
    assert set(model.batches) == {"b1", "b2"}


def test_dedupe_ignores_computed_series():
    """Two model-less scans over the same runs are distinct results, not dupes."""
    model = ProjectModel()
    model.add_batch(FitSeries("s1", _FB, member_run_numbers=[1, 2], canonical_model=None))
    model.add_batch(FitSeries("s2", _FB, member_run_numbers=[1, 2], canonical_model=None))
    assert model.dedupe_batches() == []
    assert set(model.batches) == {"s1", "s2"}


# ── unified naming helper ────────────────────────────────────────────────────


def test_member_range_run_series():
    assert member_range(_series("b", runs=(2923, 2960, 2941))) == "2923–2960"
    assert member_range(_series("b", runs=(2960,))) == "2960"


def test_member_range_group_series_prefix():
    series = _series(
        "b",
        member_kind="groups",
        runs=(-2961001, -2967001),
        member_source_run={-2961001: 2961, -2967001: 2967},
    )
    assert member_range(series) == "groups 2961–2967"


def test_default_series_label_model_and_members():
    label = default_series_label(_series("b", runs=(2923, 2960)))
    assert label == "Exponential + Constant · 2923–2960"


def test_default_series_label_appends_group_suffix():
    label = default_series_label(_series("b", runs=(2961, 2967)), group_name="B = 60 G")
    assert label == "Exponential + Constant · 2961–2967 · B = 60 G"


def test_default_series_label_computed_series():
    """A model-less series with members still yields a sane default (no model)."""
    scan = FitSeries("s", _FB, member_run_numbers=[1, 2], canonical_model=None)
    assert default_series_label(scan) == "1–2"


# ── GUI: re-run with changed classification updates the single chip ──────────

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark_gui = pytest.mark.gui


@pytest.fixture
def mw():
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QApplication.instance() or QApplication([])
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _gui_dataset(run_number: int, field: float = 100.0):
    from asymmetry.core.data.dataset import Histogram, MuonDataset, Run

    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": field},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    return MuonDataset(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01, 0.01]),
        {"run_number": run_number},
        run,
    )


def _gui_result():
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    return FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        parameters=ParameterSet([Parameter("A", 0.2)]),
        uncertainties={"A": 0.01},
    )


@pytest.mark.gui
def test_rerun_with_changed_classification_updates_single_chip(mw, monkeypatch):
    from asymmetry.core.fitting.parameters import ParameterSet

    for rn in (10, 11):
        mw._data_browser.add_dataset(_gui_dataset(rn))
    mw._plot_workspace.set_active_view("fb_asymmetry")

    curve = (np.array([0.0, 0.3]), np.array([0.1, 0.05]))
    results = {rn: (_gui_result(), curve, []) for rn in (10, 11)}

    def _state(role):
        return lambda: {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "parameters": [{"name": "A", "type": role}],
            "result_html": "",
        }

    # First run: all-Local (a plain batch).
    monkeypatch.setattr(mw._fit_panel, "get_global_state", _state("Local"))
    mw._on_global_fit_completed(results, ParameterSet())
    batches = [s for s in mw._project_model.batches.values() if not s.is_computed]
    assert len(batches) == 1
    original_id = batches[0].batch_id
    assert not batches[0].is_global()

    # Re-run the SAME members+model with a Global classification. The narrowed
    # identity means this supersedes rather than duplicates: one chip, same id,
    # now global.
    monkeypatch.setattr(mw._fit_panel, "get_global_state", _state("Global"))
    mw._on_global_fit_completed(results, ParameterSet())
    batches = [s for s in mw._project_model.batches.values() if not s.is_computed]
    assert len(batches) == 1
    assert batches[0].batch_id == original_id  # chip stayed in place
    assert batches[0].is_global()  # reflects the new classification


@pytest.mark.gui
def test_reseed_batch_index_avoids_collision_with_loaded_batches(mw):
    """A fresh batch id after load must not collide with a restored batch-N."""
    from asymmetry.core.representation import FitSeries as _FitSeries
    from asymmetry.core.representation import RepresentationType as _Rep

    mw._next_batch_index = 1  # fresh window
    mw._project_model.add_batch(
        _FitSeries("batch-3", _Rep.TIME_FB_ASYMMETRY, member_run_numbers=[1, 2])
    )
    mw._reseed_batch_index()
    assert mw._next_batch_id() == "batch-4"  # past the loaded batch-3
