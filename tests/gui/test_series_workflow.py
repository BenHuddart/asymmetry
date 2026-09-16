"""Series workflow (docs/plans/series-workflow.md, D3-D6): recording, overlays, delete.

The GUI half of "the Batch tab edits one series": an identical re-run replaces
its series in place, anything else records a new one beside it, the plot draws
the *active* series on every run it covers, and deleting a series takes only
its own overlay with it.
"""

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
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

_FB = RepresentationType.TIME_FB_ASYMMETRY
_MODEL = {"component_names": ["Exponential", "Constant"], "operators": ["+"]}
_CURVE = (np.array([0.0, 0.3]), np.array([0.1, 0.05]))


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _new_window() -> MainWindow:
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


@pytest.fixture
def mw(app):
    return _new_window()


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
        {"run_number": run_number, "field": field},
        run,
    )


def _result(value: float = 0.2) -> FitResult:
    return FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        parameters=ParameterSet([Parameter("A", value), Parameter("Lambda", 0.5)]),
        uncertainties={"A": 0.01, "Lambda": 0.02},
    )


def _load_runs(mw: MainWindow, runs: list[int]) -> None:
    for index, run_number in enumerate(runs):
        mw._data_browser.add_dataset(_dataset(run_number, field=100.0 + 10.0 * index))
    mw._on_dataset_selected(runs[0])
    mw._plot_workspace.set_active_view("fb_asymmetry")


def _stub_batch_form(mw: MainWindow, monkeypatch) -> None:
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": _MODEL,
            "parameters": [{"name": "A", "type": "Local"}, {"name": "Lambda", "type": "Local"}],
            "result_html": "",
        },
    )


def _set_batch_range(mw: MainWindow, low: float, high: float) -> None:
    """Set the Batch tab's fit-range fields — the series' recipe window (D2)."""
    tab = mw._fit_panel._global_tab
    tab._fit_range_min_spin.setValue(low)
    tab._fit_range_max_spin.setValue(high)


def _run_batch(mw: MainWindow, runs: list[int], value: float = 0.2) -> str:
    mw._on_global_fit_completed({run: (_result(value), _CURVE, []) for run in runs}, ParameterSet())
    return mw._project_model.active_series_id(_FB)


def _single_fit(mw: MainWindow, run_number: int, monkeypatch) -> None:
    mw._on_dataset_selected(run_number)
    monkeypatch.setattr(
        mw._fit_panel,
        "get_single_form_state",
        lambda: {"composite_model": _MODEL, "parameters": [], "result_html": ""},
    )
    mw._on_fit_completed(_result(), _CURVE, [])


# ── D3: identical replaces, anything else is a new series ────────────────────


def test_identical_rerun_replaces_results_keeping_id_and_label(mw, monkeypatch):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)

    first_id = _run_batch(mw, [10, 11], value=0.2)
    mw._on_series_rename_requested(first_id, "My campaign")

    second_id = _run_batch(mw, [10, 11], value=0.9)

    assert second_id == first_id
    assert len(mw._project_model.batches) == 1
    series = mw._project_model.batch(first_id)
    assert series.label == "My campaign"
    # The re-run's results replaced the first run's, in place.
    assert series.results_by_run[10]["parameters"]["A"] == pytest.approx(0.9)


def test_changed_fit_range_records_a_second_series_and_leaves_the_first(mw, monkeypatch):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    first_id = _run_batch(mw, [10, 11], value=0.2)

    _set_batch_range(mw, 0.0, 4.0)
    second_id = _run_batch(mw, [10, 11], value=0.9)

    assert second_id != first_id
    assert len(mw._project_model.batches) == 2
    first = mw._project_model.batch(first_id)
    assert first.results_by_run[10]["parameters"]["A"] == pytest.approx(0.2)
    assert first.recipe["fit_range"] == {"min": 0.0, "max": 8.0}
    second = mw._project_model.batch(second_id)
    assert second.recipe["fit_range"] == {"min": 0.0, "max": 4.0}
    # The just-run series is the active one (D5).
    assert mw._project_model.active_series_id(_FB) == second_id


def test_colliding_default_labels_are_disambiguated(mw, monkeypatch):
    """Two series that would render the same default label read distinctly (D10)."""
    _load_runs(mw, [10, 11, 12])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    first_id = _run_batch(mw, [10, 11])
    # A different member set over the same model and window: a new series whose
    # default label is the first one's.
    second_id = _run_batch(mw, [10, 11, 12])

    assert second_id != first_id
    first, second = mw._project_model.batch(first_id), mw._project_model.batch(second_id)
    assert first.label is None
    assert second.label == f"{mw._series_fallback_name(first)} (2)"


# ── D5: overlays follow the active series ────────────────────────────────────


def test_run_in_two_series_draws_the_active_one_and_follows_the_chip(mw, monkeypatch):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    first_id = _run_batch(mw, [10, 11])
    _set_batch_range(mw, 0.0, 4.0)
    second_id = _run_batch(mw, [10, 11])

    panel = mw._plot_panel
    # Both series' curves are stored; run 10 draws the active (latest) one.
    assert panel.has_fits_for_series(first_id)
    assert panel.has_fits_for_series(second_id)
    assert panel.active_fit_id() == second_id
    assert panel.shown_fit_ids(10) == [second_id]

    # Pressing the first series' chip makes it active — and the overlay follows.
    mw._on_trend_series_selected(first_id)
    assert mw._project_model.active_series_id(_FB) == first_id
    assert panel.active_fit_id() == first_id
    assert panel.shown_fit_ids(10) == [first_id]


def test_deleting_a_series_leaves_the_other_series_and_the_single_overlay(mw, monkeypatch):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    first_id = _run_batch(mw, [10, 11])
    _set_batch_range(mw, 0.0, 4.0)
    second_id = _run_batch(mw, [10, 11])
    _single_fit(mw, 10, monkeypatch)

    panel = mw._plot_panel
    assert panel.shown_fit_ids(10) == ["single"]  # a just-run single fit shows

    mw._on_series_delete_requested(second_id)

    assert mw._project_model.batch(second_id) is None
    assert mw._project_model.batch(first_id) is not None
    assert not panel.has_fits_for_series(second_id)
    assert panel.has_fits_for_series(first_id)
    # Run 10's own single fit is untouched, in the project and on the plot.
    assert mw._project_model.representation(10, _FB).fit.provenance == "single"
    assert panel._fit_curve_for_dataset(_dataset(10), fit_id="single") is not None


def test_active_series_seeds_the_single_tab_form_for_a_member_with_no_slot(mw, monkeypatch):
    """D5: a run the active series fitted restores its form from that series."""
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    _run_batch(mw, [10, 11])

    payload = mw._single_fit_restore_payload(_dataset(11))

    series = mw._project_model.batch(mw._project_model.active_series_id(_FB))
    assert payload is not None
    assert payload["composite_model"]["component_names"] == ["Exponential", "Constant"]
    # One row per model parameter, seeded from the series' recorded result.
    from asymmetry.core.fitting.composite import CompositeModel

    expected = set(CompositeModel.from_dict(series.canonical_model).param_names)
    assert {p["name"] for p in payload["parameters"]} == expected


# ── D4: trend gating lives on the series ─────────────────────────────────────


def test_trend_row_inclusion_round_trips_through_trend_excluded_runs(mw, monkeypatch):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    batch_id = _run_batch(mw, [10, 11])
    series = mw._project_model.batch(batch_id)

    def _row(run: int) -> dict:
        return next(r for r in mw._build_series_rows(series) if r["run_number"] == run)

    assert _row(10)["include_in_trend"] is True

    mw._on_member_trend_inclusion_changed(batch_id, 10, False)
    assert series.trend_excluded_runs == [10]
    assert _row(10)["include_in_trend"] is False
    assert _row(11)["include_in_trend"] is True

    mw._on_member_trend_inclusion_changed(batch_id, 10, True)
    assert series.trend_excluded_runs == []
    assert _row(10)["include_in_trend"] is True


def test_trend_exclusion_is_per_series_not_per_run(mw, monkeypatch):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    first_id = _run_batch(mw, [10, 11])
    _set_batch_range(mw, 0.0, 4.0)
    second_id = _run_batch(mw, [10, 11])

    mw._on_member_trend_inclusion_changed(second_id, 10, False)

    assert mw._project_model.batch(second_id).trend_excluded_runs == [10]
    assert mw._project_model.batch(first_id).trend_excluded_runs == []
    assert mw._project_model.batch(first_id).trend_member_run_numbers() == [10, 11]


# ── D5: the active series survives a save/load ───────────────────────────────


def test_project_reload_restores_the_active_series_and_selects_it(mw, monkeypatch, tmp_path):
    _load_runs(mw, [10, 11])
    _stub_batch_form(mw, monkeypatch)
    _set_batch_range(mw, 0.0, 8.0)
    first_id = _run_batch(mw, [10, 11])
    _set_batch_range(mw, 0.0, 4.0)
    _run_batch(mw, [10, 11])
    # Leave the *first* series active, so the restore cannot pass by accident
    # (it is neither the newest nor the last recorded).
    mw._on_trend_series_selected(first_id)

    path = tmp_path / "series.asymp"
    state = mw.collect_project_state()
    assert state["active_series"] == {_FB.value: first_id}
    save_project(state, path)

    # These runs were built in-memory, so the reopen's "data files not found"
    # prompt would block; decline it and let the rest of the restore run.
    from PySide6.QtWidgets import QMessageBox

    import asymmetry.gui.mainwindow as mw_module

    monkeypatch.setattr(
        mw_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    restored = _new_window()
    restored.restore_project_state(load_project(path), str(path))

    assert restored._project_model.active_series_id(_FB) == first_id
    assert restored._fit_parameters_panel._active_group_id == first_id
