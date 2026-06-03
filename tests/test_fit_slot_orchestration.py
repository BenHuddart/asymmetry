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
        "get_single_state",
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


def test_single_fit_slot_targets_active_domain(mw, monkeypatch):
    ds = _dataset(301)
    mw._data_browser.add_dataset(ds)
    mw._on_dataset_selected(301)
    mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
    mw._plot_workspace.set_active_view("groups")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_state",
        lambda: {"composite_model": {"component_names": ["Gaussian"], "operators": []},
                 "parameters": [], "result_html": ""},
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
    payloads = {
        rn: (_result(rchi=0.4 + 0.1 * i), _CURVE, [])
        for i, rn in enumerate([10, 11, 12])
    }

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
