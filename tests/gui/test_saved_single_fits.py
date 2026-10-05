"""Saved single fits: the Single tab's Saved fits row, the recorder and the plot overlays.

``docs/plans/single-fit-compare.md`` D1 (automatic recording), D5 (names) and D6
(each saved fit draws under its own id). Fits run through the real Single tab
against a stub engine, so the recorder sees the form exactly as a user leaves it.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.core.fitting.composite import CompositeModel  # noqa: E402
from asymmetry.core.fitting.engine import FitResult  # noqa: E402
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.core.representation import FitSlot, RepresentationType  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from asymmetry.gui.panels.plot_panel import SAVED_FIT_ID_PREFIX, SINGLE_FIT_ID  # noqa: E402
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY  # noqa: E402

_RUN = 500
_EXP = CompositeModel(["Exponential", "Constant"], operators=["+"])
_GAUSS = CompositeModel(["Gaussian", "Constant"], operators=["+"])


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    run = Run(
        run_number=_RUN,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    window._data_browser.add_dataset(
        MuonDataset(
            np.array([0.0, 0.1, 0.2, 0.3]),
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.01, 0.01, 0.01, 0.01]),
            {"run_number": _RUN},
            run,
        )
    )
    window._on_dataset_selected(_RUN)
    window._plot_workspace.set_active_view("fb_asymmetry")
    return window


def _tab(mw):
    return mw._fit_panel._single_tab


def _fit(mw, model: CompositeModel | None = None, chi_squared: float = 60.0) -> None:
    """Fit the bound run with *model* (or the form as it stands) against a stub engine."""
    tab = _tab(mw)
    if model is not None:
        tab._set_composite_model(model)
    names = list(tab._composite_model.param_names)

    def _engine_fit(ds, model_fn, parameters, *, minos=False, cancel_callback=None):
        return FitResult(
            success=True,
            chi_squared=chi_squared,
            reduced_chi_squared=chi_squared / 60.0,
            dof=60,
            parameters=ParameterSet(
                [Parameter(name=name, value=float(i + 1)) for i, name in enumerate(names)]
            ),
            uncertainties=dict.fromkeys(names, 0.01),
        )

    tab._fit_engine = SimpleNamespace(fit=_engine_fit)
    tab._run_fit()
    assert tab.wait_for_fit()


def _fit_set(mw):
    rep = mw._project_model.representation(_RUN, RepresentationType.TIME_FB_ASYMMETRY)
    return rep, rep.fit_set(None)


def _count(mw) -> str:
    return _tab(mw)._saved_fits_section._suffix_label.text()


def _hint(mw) -> str:
    """The line under the selector; empty while it is hidden (the form is the open fit)."""
    hint = _tab(mw)._saved_fit_hint
    return hint.text() if not hint.isHidden() else ""


def test_a_second_model_saves_beside_the_first_and_both_draw(mw) -> None:
    _fit(mw, _EXP)
    _fit(mw, _GAUSS)

    rep, fit_set = _fit_set(mw)
    exp, gauss = fit_set.fits
    assert fit_set.open_id == gauss.fit_id
    assert rep.fit_name(exp).startswith("Exponential + Constant · ")
    names = [entry.name for entry in mw._saved_fit_catalogue().entries]
    assert names == [rep.fit_name(exp), rep.fit_name(gauss)]
    # The open fit draws as the run's single fit; the other under its own id.
    stored = mw._plot_panel.stored_fit_ids(_RUN)
    assert {SINGLE_FIT_ID, SAVED_FIT_ID_PREFIX + exp.fit_id} <= stored
    assert mw._plot_panel.fit_label(SAVED_FIT_ID_PREFIX + exp.fit_id) == (
        f"Single fit · {rep.fit_name(exp)}"
    )
    assert (_count(mw), _hint(mw)) == ("2 on this run", "")
    assert _tab(mw)._compare_fits_btn.isEnabled()


def test_refitting_an_unchanged_setup_replaces_the_open_fit(mw) -> None:
    _fit(mw, _EXP, chi_squared=80.0)
    _fit(mw, chi_squared=60.0)

    rep, fit_set = _fit_set(mw)
    assert len(fit_set.fits) == 1
    assert rep.fit.result["chi_squared"] == pytest.approx(60.0)
    assert (_count(mw), _hint(mw)) == ("1 on this run", "")
    assert not _tab(mw)._compare_fits_btn.isEnabled()


def test_new_fit_saves_an_unchanged_setup_beside_the_open_fit(mw) -> None:
    _fit(mw, _EXP)
    _tab(mw)._new_fit_btn.click()
    assert _hint(mw) == "New fit: the next Fit is saved beside this run's other fits."
    _fit(mw)

    rep, fit_set = _fit_set(mw)
    first, second = fit_set.fits
    assert rep.fit_name(second) == f"{rep.fit_name(first)} (2)"
    # New fit is spent by the fit it applied to.
    assert not _tab(mw).records_new_fit()


def test_the_row_says_what_the_next_fit_will_do(mw) -> None:
    _fit(mw, _EXP)
    _tab(mw)._set_composite_model(_GAUSS)
    assert _hint(mw) == "Edited: the next Fit is saved as a new fit."

    _fit(mw)
    rep, fit_set = _fit_set(mw)
    _tab(mw)._set_composite_model(_EXP)
    assert _hint(mw) == f"Edited: the next Fit replaces “{rep.fit_name(fit_set.fits[0])}”."


def test_opening_a_saved_fit_restores_its_form_and_swaps_the_curves(mw) -> None:
    _fit(mw, _EXP)
    _fit(mw, _GAUSS)
    _rep, fit_set = _fit_set(mw)
    exp, gauss = fit_set.fits

    mw._on_saved_fit_open_requested(exp.fit_id)

    assert fit_set.open_id == exp.fit_id
    assert _tab(mw)._composite_model.component_names == ["Exponential", "Constant"]
    stored = mw._plot_panel.stored_fit_ids(_RUN)
    assert SAVED_FIT_ID_PREFIX + gauss.fit_id in stored
    assert SAVED_FIT_ID_PREFIX + exp.fit_id not in stored
    assert mw._plot_panel.shown_fit_ids(_RUN) == [SINGLE_FIT_ID]
    assert _hint(mw) == ""


def test_rename_then_delete_down_to_no_fit(mw) -> None:
    _fit(mw, _EXP)
    _fit(mw, _GAUSS)
    rep, fit_set = _fit_set(mw)
    exp, gauss = fit_set.fits

    mw._on_saved_fit_rename_requested(exp.fit_id, "Plain relaxation")
    assert [entry.name for entry in mw._saved_fit_catalogue().entries][0] == "Plain relaxation"

    mw._on_saved_fit_delete_requested(gauss.fit_id)
    assert fit_set.open_id == exp.fit_id
    assert _tab(mw)._composite_model.component_names == ["Exponential", "Constant"]

    mw._on_saved_fit_delete_requested(exp.fit_id)
    assert rep.fit.is_empty()
    assert SINGLE_FIT_ID not in mw._plot_panel.stored_fit_ids(_RUN)
    assert _count(mw) == ""
    assert not _tab(mw)._delete_fit_btn.isEnabled()


# ── the Compare window (D2, D7–D9) ──────────────────────────────────────────


def _wait_for_curves(window, fit_id: str) -> None:
    """Let the window's curve worker for *fit_id* finish and its result land."""
    app = QApplication.instance()
    for _ in range(500):
        app.processEvents()
        if fit_id in window._curves:
            return
        QThread.msleep(10)
    raise AssertionError(f"no curves were built for {fit_id}")


def test_compare_ranks_the_runs_fits_and_opens_a(mw) -> None:
    _fit(mw, _EXP, chi_squared=80.0)
    _fit(mw, _GAUSS, chi_squared=60.0)
    _rep, fit_set = _fit_set(mw)
    exp, gauss = fit_set.fits

    _tab(mw)._compare_fits_btn.click()
    window = mw._saved_fit_compare_window
    assert window is not None and window.isVisible()
    panel = window.panel
    # Same window and points: ranked on AICc, the lower χ² first; the open fit is A.
    assert list(panel._summaries) == [gauss.fit_id, exp.fit_id]
    assert panel._summaries[exp.fit_id].delta == pytest.approx(20.0)
    assert panel.a_key() == gauss.fit_id
    _wait_for_curves(window, gauss.fit_id)
    assert window._curves[gauss.fit_id][1].fit[0].size > 0

    panel.set_a(exp.fit_id)
    panel.continue_requested.emit(panel.a_key())
    assert fit_set.open_id == exp.fit_id
    assert _tab(mw)._composite_model.component_names == ["Exponential", "Constant"]
    window.close()


def test_compare_follows_a_new_fit_on_the_run(mw) -> None:
    _fit(mw, _EXP)
    _fit(mw, _GAUSS)
    _tab(mw)._compare_fits_btn.click()
    window = mw._saved_fit_compare_window

    _fit(mw, CompositeModel(["Exponential", "Gaussian", "Constant"], operators=["+", "+"]))

    assert len(window.panel._summaries) == 3
    window.close()
    QApplication.instance().processEvents()


def test_opening_renaming_or_deleting_a_saved_fit_is_unsaved_work(mw) -> None:
    _fit(mw, _EXP)
    _fit(mw, _GAUSS)
    _rep, fit_set = _fit_set(mw)
    exp, gauss = fit_set.fits
    for emit in (
        lambda: _tab(mw).saved_fit_open_requested.emit(exp.fit_id),
        lambda: _tab(mw).saved_fit_rename_requested.emit(exp.fit_id, "Plain"),
        lambda: _tab(mw).saved_fit_delete_requested.emit(gauss.fit_id),
    ):
        mw._clear_dirty()
        emit()
        assert mw._dirty


def test_a_legacy_fit_beside_a_new_one_opens_compares_and_draws_nothing_it_cannot(mw) -> None:
    """A migrated v5-era fit (model and table, no structured result) never breaks the row."""
    rep = mw._project_model.ensure_dataset(_RUN).ensure(RepresentationType.TIME_FB_ASYMMETRY)
    names = list(_GAUSS.param_names)
    rep.set_fit_for(
        None,
        FitSlot(
            model=_GAUSS.to_dict(),
            parameters=[{"name": name, "value": 1.0} for name in names],
            result={"result_html": "<b>legacy</b>"},
            provenance="single",
        ),
    )
    legacy = rep.fit
    _fit(mw, _EXP)
    assert len(rep.fit_set(None).fits) == 2

    _tab(mw)._compare_fits_btn.click()
    window = mw._saved_fit_compare_window
    assert window.panel._summaries[legacy.fit_id].delta == math.inf

    mw._on_saved_fit_open_requested(legacy.fit_id)
    assert _tab(mw)._composite_model.component_names == ["Gaussian", "Constant"]
    window.close()


def test_losing_the_binding_closes_the_compare_window(mw) -> None:
    _fit(mw, _EXP)
    _fit(mw, _GAUSS)
    _tab(mw)._compare_fits_btn.click()
    assert mw._saved_fit_compare_window is not None
    # The run is gone from the Single tab (no record bound): nothing to compare.
    mw._current_dataset = None
    mw._sync_saved_fit_overlays()
    QApplication.instance().processEvents()
    assert mw._saved_fit_compare_window is None or not mw._saved_fit_compare_window.isVisible()
