"""Phase 4: completed fits write per-representation FitSlots / Batches."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _dataset(run_number: int, field: float = 100.0) -> MuonDataset:
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


def _result(rchi: float = 0.5) -> FitResult:
    return FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=rchi,
        parameters=ParameterSet([Parameter("A", 0.2), Parameter("Lambda", 0.5)]),
        uncertainties={"A": 0.01, "Lambda": 0.02},
    )


_CURVE = (np.array([0.0, 0.3]), np.array([0.1, 0.05]))


def test_single_fit_writes_representation_slot(mw, monkeypatch):
    ds = _dataset(300)
    mw._data_browser.add_dataset(ds)
    mw._on_dataset_selected(300)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "parameters": [{"name": "A", "value": 0.2}],
            "result_html": "ok",
        },
    )

    mw._on_fit_completed(_result(rchi=0.7), _CURVE, [])

    rep = mw._project_model.representation(300, RepresentationType.TIME_FB_ASYMMETRY)
    assert rep is not None
    assert rep.fit.provenance == "single"
    assert rep.fit.model["component_names"] == ["Exponential"]
    assert rep.fit.result["reduced_chi_squared"] == pytest.approx(0.7)
    assert rep.fit.result["parameters"]["A"] == pytest.approx(0.2)
    # Provenance is stamped onto the persisted record (Item 2: enrich wiring).
    assert rep.fit.result["model_name"] == "Exponential"
    assert rep.fit.result["provenance"] == "single"
    assert "timestamp" in rep.fit.result
    assert rep.fit.result["npar"] == 2  # both A and Lambda carry a HESSE sigma


def test_single_fit_overlay_and_slot_agree_on_run(mw, monkeypatch):
    """The plotted overlay and the persisted slot key under the same run.

    Regression: ``plot_fit`` used to key the overlay under the plot panel's own
    ``_current_dataset`` while ``_record_single_fit_slot`` keyed the slot under
    the main window's selected dataset. In a multi-run overlay stacked view
    those differ, so the displayed curve and the saved fit disagreed on the run.
    Both now source the run from ``_single_fit_run_number()`` (the selected
    dataset). Here the panel's current dataset is forced to a *different* run to
    prove the overlay still follows the selected one.
    """
    ds = _dataset(300)
    mw._data_browser.add_dataset(ds)
    mw._on_dataset_selected(300)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "parameters": [{"name": "A", "value": 0.2}],
            "result_html": "ok",
        },
    )

    # Simulate the multi-run stacked view where the panel's current dataset is a
    # different overlaid run (999) than the selected/fitted run (300).
    mw._plot_panel._current_dataset = _dataset(999)

    captured: dict[str, object] = {}
    real_plot_fit = mw._plot_panel.plot_fit

    def _capture(*args, **kwargs):
        captured["run_number"] = kwargs.get("run_number")
        return real_plot_fit(*args, **kwargs)

    monkeypatch.setattr(mw._plot_panel, "plot_fit", _capture)

    mw._on_fit_completed(_result(rchi=0.7), _CURVE, [])

    # The overlay was keyed under the selected run, not the panel's stale 999.
    assert captured["run_number"] == 300
    # And the persisted slot lives under that same run.
    assert mw._project_model.representation(300, RepresentationType.TIME_FB_ASYMMETRY) is not None
    assert mw._project_model.representation(999, RepresentationType.TIME_FB_ASYMMETRY) is None


def test_single_fit_slot_targets_active_domain(mw, monkeypatch):
    ds = _dataset(301)
    mw._data_browser.add_dataset(ds)
    mw._on_dataset_selected(301)
    mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
    mw._plot_workspace.set_active_view("groups")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {
            "composite_model": {"component_names": ["Gaussian"], "operators": []},
            "parameters": [],
            "result_html": "",
        },
    )

    mw._on_fit_completed(_result(), _CURVE, [])

    assert mw._project_model.representation(301, RepresentationType.TIME_GROUPS) is not None
    assert mw._project_model.representation(301, RepresentationType.TIME_FB_ASYMMETRY) is None


def test_global_fit_creates_batch_and_member_slots(mw, monkeypatch):
    for run_number, field in [(10, 100.0), (11, 50.0), (12, 150.0)]:
        mw._data_browser.add_dataset(_dataset(run_number, field))
    mw._on_dataset_selected(10)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential", "Constant"], "operators": ["+"]},
            "parameters": [{"name": "A", "type": "Local"}, {"name": "Lambda", "type": "Local"}],
            "result_html": "",
        },
    )
    payloads = {rn: (_result(rchi=0.4 + 0.1 * i), _CURVE, []) for i, rn in enumerate([10, 11, 12])}

    mw._on_global_fit_completed(payloads, ParameterSet())

    assert len(mw._project_model.batches) == 1
    batch = next(iter(mw._project_model.batches.values()))
    assert set(batch.member_run_numbers) == {10, 11, 12}
    # Ordered by field (50, 100, 150) -> runs 11, 10, 12.
    assert batch.member_run_numbers == [11, 10, 12]
    assert not batch.is_global()  # all-local -> pure batch
    assert set(batch.results_by_run) == {10, 11, 12}

    rep = mw._project_model.representation(11, RepresentationType.TIME_FB_ASYMMETRY)
    assert rep.fit.provenance == "batch"
    assert rep.fit.batch_id == batch.batch_id
    # Every batch member record carries the composite model name + provenance.
    for summary in batch.results_by_run.values():
        assert summary["model_name"] == "Exponential + Constant"
        assert summary["provenance"] == "batch"
        assert "timestamp" in summary
    # The member FitSlot result mirrors the series entry (no recompute drift).
    assert rep.fit.result["model_name"] == "Exponential + Constant"
    assert rep.fit.result["provenance"] == "batch"


def test_global_classified_parameter_yields_global_provenance(mw, monkeypatch):
    for run_number in (10, 11):
        mw._data_browser.add_dataset(_dataset(run_number))
    mw._on_dataset_selected(10)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "parameters": [{"name": "A", "type": "Global"}, {"name": "Lambda", "type": "Local"}],
            "result_html": "",
        },
    )
    payloads = {rn: (_result(), _CURVE, []) for rn in (10, 11)}

    mw._on_global_fit_completed(payloads, ParameterSet())

    batch = next(iter(mw._project_model.batches.values()))
    assert batch.is_global()
    assert batch.global_params() == ["A"]
    rep = mw._project_model.representation(10, RepresentationType.TIME_FB_ASYMMETRY)
    assert rep.fit.provenance == "global"
    assert rep.fit.batch_id == batch.batch_id


def _group_member(source_run: int, group: int) -> MuonDataset:
    """Synthetic grouped-fit member dataset for one (run, group)."""
    return MuonDataset(
        np.array([0.0, 0.1]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        {
            "run_number": -((source_run * 1000) + group),
            "source_run_number": source_run,
            "group_id": group,
            "group_name": f"g{group}",
        },
        None,
    )


def test_grouped_batch_creates_group_series_and_pointer_slot(mw, monkeypatch):
    # A multi-run batch (≥2 source runs) is the unit that records a FitSeries.
    for rn in (42, 43):
        mw._data_browser.add_dataset(_dataset(rn))
    mw._on_dataset_selected(42)
    mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
    mw._plot_workspace.set_active_view("groups")
    monkeypatch.setattr(
        mw._multi_group_fit_window,
        "get_grouped_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "param_roles": {"Lambda": "global"},
            "nuisance_params": ["N0", "background", "amplitude", "relative_phase"],
        },
    )
    # One group each across two source runs (42, 43).
    grouped_datasets = [_group_member(42, 1), _group_member(43, 1)]
    results = {-42001: (_result(rchi=0.3), _CURVE, []), -43001: (_result(rchi=0.6), _CURVE, [])}

    mw._record_grouped_fit_series(grouped_datasets, results)

    assert len(mw._project_model.batches) == 1
    series = next(iter(mw._project_model.batches.values()))
    assert series.member_kind == "groups"
    assert set(series.member_run_numbers) == {-42001, -43001}
    assert series.member_source_run == {-42001: 42, -43001: 43}
    assert set(series.results_by_run) == {-42001, -43001}
    assert series.results_by_run[-42001]["reduced_chi_squared"] == pytest.approx(0.3)
    assert series.is_global()  # Lambda global -> global provenance
    assert series.nuisance_params == ["N0", "background", "amplitude", "relative_phase"]

    rep = mw._project_model.representation(42, RepresentationType.TIME_GROUPS)
    assert rep is not None
    assert rep.fit.provenance == "global"
    assert rep.fit.batch_id == series.batch_id


def test_single_grouped_fit_writes_slot_not_series(mw, monkeypatch):
    # A single-dataset grouped fit (one source run) behaves like an ordinary
    # single fit: NO FitSeries is created (a one-point series is un-trendable);
    # the per-group results are stored on the dataset's grouped FitSlot instead.
    mw._data_browser.add_dataset(_dataset(42))
    mw._on_dataset_selected(42)
    mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
    mw._plot_workspace.set_active_view("groups")
    monkeypatch.setattr(
        mw._multi_group_fit_window,
        "get_grouped_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "param_roles": {"Lambda": "global"},
            "nuisance_params": ["N0", "background", "amplitude", "relative_phase"],
        },
    )
    grouped_datasets = [_group_member(42, 1), _group_member(42, 2)]
    results = {-42001: (_result(rchi=0.3), _CURVE, []), -42002: (_result(rchi=0.6), _CURVE, [])}

    batch_id = mw._record_grouped_fit_series(grouped_datasets, results)

    # No series recorded.
    assert batch_id is None
    assert mw._project_model.batches == {}

    # The grouped representation's FitSlot carries the single fit's per-group
    # results (provenance "single", no batch id).
    rep = mw._project_model.representation(42, RepresentationType.TIME_GROUPS)
    assert rep is not None
    assert rep.fit.provenance == "single"
    assert rep.fit.batch_id is None
    assert not rep.fit.is_empty()
    assert set(rep.fit.result["groups"]) == {"-42001", "-42002"}
    assert rep.fit.result["groups"]["-42001"]["reduced_chi_squared"] == pytest.approx(0.3)


def test_add_compatible_single_fit_to_series(mw, monkeypatch):
    for run_number, field in [(10, 100.0), (11, 50.0), (12, 150.0), (13, 200.0)]:
        mw._data_browser.add_dataset(_dataset(run_number, field))
    mw._on_dataset_selected(10)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    model = {"component_names": ["Exponential", "Constant"], "operators": ["+"]}
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": model,
            "parameters": [{"name": "A", "type": "Local"}],
            "result_html": "",
        },
    )
    mw._on_global_fit_completed({rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet())
    series = next(iter(mw._project_model.batches.values()))
    assert set(series.member_run_numbers) == {10, 11}

    # Single-fit run 12 with a matching model, then add it to the series.
    mw._on_dataset_selected(12)
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {"composite_model": model, "parameters": [], "result_html": ""},
    )
    mw._on_fit_completed(_result(), _CURVE, [])

    assert mw._add_single_fit_to_series(12, series.batch_id) is True
    assert 12 in series.member_run_numbers
    rep = mw._project_model.representation(12, RepresentationType.TIME_FB_ASYMMETRY)
    assert rep.fit.batch_id == series.batch_id
    assert 12 in series.results_by_run

    # An incompatible model (different components) is rejected.
    mw._on_dataset_selected(13)
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {
            "composite_model": {"component_names": ["Gaussian"], "operators": []},
            "parameters": [],
            "result_html": "",
        },
    )
    mw._on_fit_completed(_result(), _CURVE, [])

    assert mw._add_single_fit_to_series(13, series.batch_id) is False
    assert 13 not in series.member_run_numbers


def test_add_to_series_action_finds_and_adds_compatible_series(mw, monkeypatch):
    for run_number, field in [(10, 100.0), (11, 50.0), (12, 150.0)]:
        mw._data_browser.add_dataset(_dataset(run_number, field))
    mw._on_dataset_selected(10)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    model = {"component_names": ["Exponential", "Constant"], "operators": ["+"]}
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": model,
            "parameters": [{"name": "A", "type": "Local"}],
            "result_html": "",
        },
    )
    mw._on_global_fit_completed({rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet())
    series = next(iter(mw._project_model.batches.values()))

    # Single-fit run 12 with a matching model, then trigger the action.
    mw._on_dataset_selected(12)
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {"composite_model": model, "parameters": [], "result_html": ""},
    )
    mw._on_fit_completed(_result(), _CURVE, [])

    # Exactly one compatible series → added without a chooser prompt.
    mw._on_add_single_fit_to_series_requested()

    assert 12 in series.member_run_numbers
    rep = mw._project_model.representation(12, RepresentationType.TIME_FB_ASYMMETRY)
    assert rep.fit.batch_id == series.batch_id


def test_editing_member_model_diverges_and_excludes_from_trend(mw, monkeypatch):
    for run_number, field in [(10, 100.0), (11, 50.0)]:
        mw._data_browser.add_dataset(_dataset(run_number, field))
    mw._on_dataset_selected(10)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": {"component_names": ["Exponential", "Constant"], "operators": ["+"]},
            "parameters": [{"name": "A", "type": "Local"}],
            "result_html": "",
        },
    )
    mw._on_global_fit_completed({rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet())
    batch = next(iter(mw._project_model.batches.values()))
    assert set(mw._project_model.trend_runs_for_batch(batch)) == {10, 11}

    # Single-fit member 11 with a different model -> diverges, excluded from trend.
    mw._on_dataset_selected(11)
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {
            "composite_model": {"component_names": ["Gaussian"], "operators": []},
            "parameters": [],
            "result_html": "",
        },
    )
    mw._on_fit_completed(_result(), _CURVE, [])

    assert batch.is_diverged(11)
    assert mw._project_model.trend_runs_for_batch(batch) == [10]
