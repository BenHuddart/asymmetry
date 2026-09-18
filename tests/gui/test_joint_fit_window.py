"""Joint fit (docs/plans/joint-fit.md, phase 3): the window, the menu, recording.

Covers what the GUI half of a joint fit has to get right: two series may only
be members together when they share no run (D3), the shared table is proposed
from the models and keeps the user's own rows (D7), a run writes each member's
results in place and stamps them (D8), a solo re-run detaches a member and the
joint fit says so (D9), and the record round-trips through a project file.
"""

from __future__ import annotations

import dataclasses
import os
import time

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.series import FitSeries
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY
from asymmetry.gui.windows import joint_fit_window as joint_window_module
from asymmetry.gui.windows.joint_fit_window import JointFitWindow, JointSeriesEntry
from tests._qt_helpers import wait_for

_FB = RepresentationType.TIME_FB_ASYMMETRY
_MODEL_A = {"component_names": ["Exponential", "Constant"], "operators": ["+"]}
_MODEL_B = {"component_names": ["Gaussian", "Constant"], "operators": ["+"]}

#: The shared truth both synthetic series are generated with.
_TRUE_BACKGROUND = 0.05

_ROWS_A = [
    {"name": "A_1", "value": 0.2, "type": "Global", "bounds": "-inf, inf"},
    {"name": "Lambda", "value": 0.5, "type": "Local", "bounds": "0, 10"},
    {"name": "A_bg", "value": 0.04, "type": "Global", "bounds": "-inf, inf"},
]
_ROLES_A = {"A_1": "global", "Lambda": "local", "A_bg": "global"}
_ROWS_B = [
    {"name": "A_1", "value": 0.15, "type": "Global", "bounds": "-inf, inf"},
    {"name": "sigma", "value": 0.3, "type": "Local", "bounds": "0, 10"},
    {"name": "A_bg", "value": 0.06, "type": "Global", "bounds": "-inf, inf"},
]
_ROLES_B = {"A_1": "global", "sigma": "local", "A_bg": "global"}


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


# ── fixtures: synthetic data and recorded series ────────────────────────────


def _dataset(run_number: int, asymmetry: np.ndarray) -> MuonDataset:
    time = np.linspace(0.05, 4.0, asymmetry.size)
    counts = np.linspace(100.0, 10.0, asymmetry.size)
    run = Run(
        run_number=run_number,
        histograms=[Histogram(counts, 0.016, 0), Histogram(counts * 0.9, 0.016, 0)],
        metadata={"field": 100.0 + run_number, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": asymmetry.size - 1,
        },
    )
    return MuonDataset(
        time,
        asymmetry,
        np.full(asymmetry.size, 0.004),
        {"run_number": run_number, "field": 100.0 + run_number, "temperature": 5.0},
        run,
    )


def _exponential_dataset(run_number: int, relax: float) -> MuonDataset:
    time = np.linspace(0.05, 4.0, 80)
    return _dataset(run_number, 0.2 * np.exp(-relax * time) + _TRUE_BACKGROUND)


def _gaussian_dataset(run_number: int, sigma: float) -> MuonDataset:
    time = np.linspace(0.05, 4.0, 80)
    return _dataset(run_number, 0.15 * np.exp(-((sigma * time) ** 2) / 2) + _TRUE_BACKGROUND)


def _series(
    batch_id: str,
    runs: list[int],
    model: dict,
    rows: list[dict],
    roles: dict[str, str],
    *,
    label: str | None = None,
) -> FitSeries:
    """A recorded, run-membered series with a full recipe — what a joint fit composes."""
    return FitSeries(
        batch_id,
        _FB,
        label=label,
        member_run_numbers=list(runs),
        canonical_model=dict(model),
        param_roles=dict(roles),
        results_by_run={
            run: {"success": True, "parameters": {"A_bg": 0.0}, "reduced_chi_squared": 9.0}
            for run in runs
        },
        last_fitted_members=list(runs),
        recipe={
            "parameters": [dict(row) for row in rows],
            "fit_range": {"min": 0.05, "max": 4.0},
        },
    )


def _load_project(mw: MainWindow) -> None:
    """Load four runs and make the F-B asymmetry view active."""
    for dataset in (
        _exponential_dataset(1001, 0.8),
        _exponential_dataset(1002, 1.1),
        _gaussian_dataset(1003, 0.7),
        _gaussian_dataset(1004, 0.9),
    ):
        mw._data_browser.add_dataset(dataset)
    mw._on_dataset_selected(1001)
    mw._plot_workspace.set_active_view("fb_asymmetry")


def _record_two_series(mw: MainWindow) -> tuple[FitSeries, FitSeries]:
    series_a = _series("batch-a", [1001, 1002], _MODEL_A, _ROWS_A, _ROLES_A, label="Ordered")
    series_b = _series("batch-b", [1003, 1004], _MODEL_B, _ROWS_B, _ROLES_B, label="Para")
    mw._project_model.add_batch(series_a)
    mw._project_model.add_batch(series_b)
    return series_a, series_b


def _entry(batch_id: str, label: str, members: tuple[int, ...], **kwargs) -> JointSeriesEntry:
    """A window-level series entry, for the tests that need no project at all."""
    return JointSeriesEntry(
        batch_id=batch_id,
        label=label,
        rep_type=_FB,
        model_text=kwargs.get("model_text", "Exp + Const"),
        members=members,
        status="2/2",
        blocked_reason=kwargs.get("blocked_reason", ""),
        model=kwargs.get("model"),
        roles=kwargs.get("roles", {}),
        recipe=kwargs.get("recipe", {"parameters": [], "fit_range": {"min": None, "max": None}}),
    )


def _tick(window: JointFitWindow, batch_id: str, checked: bool = True) -> None:
    """Tick (or untick) a series row the way a click on its checkbox does."""
    table = window._series_table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item.data(Qt.ItemDataRole.UserRole) == batch_id:
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"no row for {batch_id}")


def _series_row(window: JointFitWindow, batch_id: str) -> int:
    table = window._series_table
    for row in range(table.rowCount()):
        if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == batch_id:
            return row
    raise AssertionError(f"no row for {batch_id}")


def _shared_row_names(window: JointFitWindow) -> dict[str, bool]:
    """``{shared name: ticked}`` as the shared table currently reads."""
    table = window._shared_table
    return {
        table.item(row, 0).text(): table.item(row, 0).checkState() == Qt.CheckState.Checked
        for row in range(table.rowCount())
    }


# ── D3: members never overlap ───────────────────────────────────────────────


def test_ticking_a_series_disables_an_overlapping_one(app) -> None:
    window = JointFitWindow()
    entries = [
        _entry("batch-a", "Ordered", (1001, 1002)),
        _entry("batch-b", "Para", (1003, 1004)),
        _entry("batch-c", "Overlapping", (1002, 1005)),
    ]
    window.set_providers(lambda: entries, lambda _bid: [])
    window.start_new(_FB)

    assert window._series_table.rowCount() == 3

    _tick(window, "batch-a")
    row_c = _series_row(window, "batch-c")
    item_c = window._series_table.item(row_c, 0)
    assert not (item_c.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    assert item_c.toolTip() == "Shares runs with Ordered"
    # A series that does not overlap stays available.
    row_b = _series_row(window, "batch-b")
    assert window._series_table.item(row_b, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable

    _tick(window, "batch-a", checked=False)
    row_c = _series_row(window, "batch-c")
    item_c = window._series_table.item(row_c, 0)
    assert item_c.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert item_c.toolTip() == ""


def test_group_and_model_less_series_are_listed_with_their_reason(app) -> None:
    window = JointFitWindow()
    entries = [
        _entry("batch-g", "Groups", (1001,), blocked_reason="Detector-group series"),
        _entry("batch-s", "Scan", (1001,), blocked_reason="No fit model"),
    ]
    window.set_providers(lambda: entries, lambda _bid: [])
    window.start_new(_FB)

    tooltips = {
        window._series_table.item(row, 0).text(): window._series_table.item(row, 0).toolTip()
        for row in range(window._series_table.rowCount())
    }
    assert tooltips == {"Groups": "Detector-group series", "Scan": "No fit model"}


# ── D7: autodetection proposes the default shared table ─────────────────────


def test_suggest_ticks_exact_rows_and_keeps_user_rows(app) -> None:
    window = JointFitWindow()
    entries = [
        _entry(
            "batch-a",
            "Ordered",
            (1001, 1002),
            model=CompositeModel.from_dict(_MODEL_A),
            roles=_ROLES_A,
            recipe={"parameters": _ROWS_A, "fit_range": {"min": 0.05, "max": 4.0}},
        ),
        _entry(
            "batch-b",
            "Para",
            (1003, 1004),
            model=CompositeModel.from_dict(_MODEL_B),
            roles=_ROLES_B,
            recipe={"parameters": _ROWS_B, "fit_range": {"min": 0.05, "max": 4.0}},
        ),
    ]
    window.set_providers(lambda: entries, lambda _bid: [])
    window.start_new(_FB)
    _tick(window, "batch-a")
    _tick(window, "batch-b")

    window._on_suggest_clicked()
    proposed = _shared_row_names(window)
    # A_bg is the same parameter of the same component in both models: exact,
    # so it is ticked. A_1 sits on different components: offered, unticked.
    assert proposed["A_bg"] is True
    assert proposed["A_1"] is False
    # D6: the shared row is seeded from the first contributing series' recipe.
    a_bg = next(row for row in window._shared_rows if row.name == "A_bg")
    assert a_bg.value == pytest.approx(0.04)

    # A row the user added by hand survives a second proposal.
    window._shared_rows.append(joint_window_module._SharedRow(name="phase", tier="user"))
    window._populate_shared_table()
    window._on_suggest_clicked()
    assert "phase" in _shared_row_names(window)


def test_a_shared_row_with_one_contributor_cannot_be_ticked(app) -> None:
    window = JointFitWindow()
    entries = [
        _entry("batch-a", "Ordered", (1001,), roles=_ROLES_A),
        _entry("batch-b", "Para", (1003,), roles=_ROLES_B),
    ]
    window.set_providers(lambda: entries, lambda _bid: [])
    window.start_new(_FB)
    _tick(window, "batch-a")
    _tick(window, "batch-b")

    window._shared_rows.append(
        joint_window_module._SharedRow(name="A_bg", members={"batch-a": "A_bg"}, ticked=True)
    )
    window._populate_shared_table()
    item = window._shared_table.item(0, 0)
    assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    assert "at least two series" in item.toolTip()
    assert window._shared_rows[0].ticked is False


# ── D8: a run records into the member series and the JointFit record ────────


def _prepare_joint_fit(mw: MainWindow) -> JointFitWindow:
    """Open the window on both series with the suggested shared table, unrun."""
    mw._on_new_joint_fit()
    window = mw._joint_fit_window
    _tick(window, "batch-a")
    _tick(window, "batch-b")
    window._on_suggest_clicked()
    return window


def _run_joint_fit(mw: MainWindow, app) -> JointFitWindow:
    window = _prepare_joint_fit(mw)
    window._on_run_clicked()
    wait_for(lambda: window._worker is None, app, timeout_s=30.0)
    return window


def test_joint_run_records_results_stamps_members_and_lists_the_record(mw, app) -> None:
    _load_project(mw)
    series_a, series_b = _record_two_series(mw)

    window = _run_joint_fit(mw, app)

    assert window.has_result()
    joint_fits = mw._project_model.joint_fits
    assert len(joint_fits) == 1
    joint = next(iter(joint_fits.values()))
    assert joint.member_batch_ids == ["batch-a", "batch-b"]
    assert [row["name"] for row in joint.shared] == ["A_bg"]
    assert joint.result["shared_values"]["A_bg"] == pytest.approx(_TRUE_BACKGROUND, abs=0.01)
    assert joint.result["fitted_at"]

    # Results land under the members' existing ids, in the Batch-tab shape.
    for series in (series_a, series_b):
        assert sorted(series.results_by_run) == sorted(series.member_run_numbers)
        summary = series.results_by_run[series.member_run_numbers[0]]
        assert summary["success"] is True
        assert summary["provenance"] == "joint"
        assert summary["parameters"]["A_bg"] == pytest.approx(_TRUE_BACKGROUND, abs=0.01)
        # D5: each member says which shared column its own name feeds.
        assert series.joint_fit_id == joint.joint_id
        assert series.shared_params == {"A_bg": "A_bg"}

    labels = [action.text() for action in mw._joint_fits_menu.actions()]
    assert labels == ["Joint: Ordered + Para"]
    assert not joint.is_stale(mw._project_model)


def test_a_failed_joint_fit_records_nothing(mw, app, monkeypatch) -> None:
    _load_project(mw)
    series_a, _series_b = _record_two_series(mw)
    previous = dict(series_a.results_by_run)

    class _Failed:
        success = False
        message = "the minimiser gave up"

    monkeypatch.setattr(joint_window_module, "fit_joint", lambda *a, **k: _Failed())
    window = _run_joint_fit(mw, app)

    assert mw._project_model.joint_fits == {}
    assert series_a.results_by_run == previous
    assert series_a.joint_fit_id is None
    assert "gave up" in window._results_card.content_html()


class _ExplodingModel:
    """A model whose evaluation is a test failure — proves the draw path is dry."""

    def function(self, *_args, **_kwargs):
        raise AssertionError("the overlay path evaluated a model on the GUI thread")


def test_the_completion_payload_carries_worker_evaluated_curves(mw, app) -> None:
    """The fitted curves arrive with the result, so drawing evaluates nothing."""
    _load_project(mw)
    _record_two_series(mw)

    captured: list[tuple] = []
    window = _prepare_joint_fit(mw)
    window.joint_fit_completed.connect(
        lambda launch, result, curves: captured.append((launch, result, curves))
    )
    window._on_run_clicked()
    wait_for(lambda: window._worker is None, app, timeout_s=30.0)

    assert len(captured) == 1
    launch, result, curves = captured[0]
    assert set(curves) == {"batch-a", "batch-b"}
    assert sorted(curves["batch-a"]) == [1001, 1002]
    times, values = curves["batch-a"][1001]
    assert len(times) == len(values) > 0
    assert np.all(np.isfinite(values))

    # Re-drawing with every model replaced by one that raises on evaluation:
    # the overlay path must be satisfied by the curves it was handed.
    dry_launch = dataclasses.replace(
        launch, models={batch_id: _ExplodingModel() for batch_id in launch.member_batch_ids}
    )
    mw._draw_joint_fit_overlays(dry_launch, result, curves)


# ── v1 is time-domain only ──────────────────────────────────────────────────


def test_a_frequency_representation_offers_no_joint_fit(mw, app) -> None:
    _load_project(mw)
    _record_two_series(mw)

    mw._plot_workspace.set_active_view("frequency")
    mw._on_new_joint_fit()
    window = mw._joint_fit_window
    assert window._series_table.rowCount() == 0
    assert window.series_notice() == "Joint fits are available for time-domain series only."

    # Back on a time representation the same series are offered again.
    mw._plot_workspace.set_active_view("fb_asymmetry")
    mw._on_new_joint_fit()
    assert window._series_table.rowCount() == 2
    assert window.series_notice() == ""


# ── D9: a solo re-run detaches a member ─────────────────────────────────────


def test_solo_rerun_of_a_member_detaches_it_and_marks_the_joint_fit_stale(mw, app) -> None:
    _load_project(mw)
    series_a, _series_b = _record_two_series(mw)
    window = _run_joint_fit(mw, app)
    joint = next(iter(mw._project_model.joint_fits.values()))

    # The Batch tab re-running series A alone comes through _record_fit_series
    # with an identical recipe, so it replaces A's results in place.
    rerun = _series("batch-rerun", [1001, 1002], _MODEL_A, _ROWS_A, _ROLES_A, label="Ordered")
    recorded = mw._record_fit_series(rerun, source_runs=[1001, 1002])

    assert recorded == "batch-a"
    assert series_a.joint_fit_id is None
    assert series_a.shared_params == {}
    assert joint.is_stale(mw._project_model)
    assert "re-run on its own" in joint.stale_reason(mw._project_model)
    assert window._stale_banner.isVisible()
    assert "re-run on its own" in window._stale_label.text()


# ── D11: the record round-trips through a project file ──────────────────────


def test_project_save_and_load_reopens_the_joint_fit(mw, app, tmp_path, monkeypatch) -> None:
    _load_project(mw)
    _record_two_series(mw)
    _run_joint_fit(mw, app)
    joint_id = next(iter(mw._project_model.joint_fits))

    path = tmp_path / "joint.asymp"
    save_project(mw.collect_project_state(), str(path))

    # These runs were built in-memory, so the reopen's "data files not found"
    # prompt would block; decline it and let the rest of the restore run.
    import asymmetry.gui.mainwindow as mw_module

    monkeypatch.setattr(
        mw_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: mw_module.QMessageBox.StandardButton.No),
    )
    reopened = MainWindow()
    reopened.restore_project_state(load_project(str(path)), str(path))

    restored = reopened._project_model.joint_fits
    assert list(restored) == [joint_id]
    assert [row["name"] for row in restored[joint_id].shared] == ["A_bg"]
    for batch_id in ("batch-a", "batch-b"):
        assert reopened._project_model.batch(batch_id).joint_fit_id == joint_id

    # The submenu lists it, and the window reopens on the saved record with its
    # members ticked and its shared table loaded.
    assert [action.text() for action in reopened._joint_fits_menu.actions()] == [
        "Joint: Ordered + Para"
    ]
    window = reopened._joint_fit_window
    assert window is not None
    assert window.open_joint_id() == joint_id
    assert [entry.batch_id for entry in window.checked_series()] == ["batch-a", "batch-b"]
    assert list(_shared_row_names(window)) == ["A_bg"]


# ── Stop ────────────────────────────────────────────────────────────────────


def test_stop_cancels_the_run_and_leaves_the_members_untouched(mw, app, monkeypatch) -> None:
    _load_project(mw)
    series_a, series_b = _record_two_series(mw)
    before = {
        series_a.batch_id: dict(series_a.results_by_run),
        series_b.batch_id: dict(series_b.results_by_run),
    }

    def _blocking_fit(_problems, _shared, *, cancel_callback=None, **_kwargs):
        # Sleep rather than spin: a busy loop in a worker thread holds the GIL
        # and starves the GUI thread this test then has to pump.
        while not cancel_callback():
            time.sleep(0.005)
        raise FitCancelledError("cancelled")

    monkeypatch.setattr(joint_window_module, "fit_joint", _blocking_fit)

    mw._on_new_joint_fit()
    window = mw._joint_fit_window
    _tick(window, "batch-a")
    _tick(window, "batch-b")
    window._on_suggest_clicked()
    window._on_run_clicked()
    window._on_stop_clicked()
    wait_for(lambda: window._worker is None, app, timeout_s=30.0)

    assert "stopped" in window._results_card.content_html().lower()
    assert mw._project_model.joint_fits == {}
    assert series_a.results_by_run == before["batch-a"]
    assert series_b.results_by_run == before["batch-b"]
    assert series_a.joint_fit_id is None


# ── D10: deleting a member cascades into the record and the window ──────────


def test_delete_button_drops_the_record_but_keeps_the_members_and_their_results(
    mw, app, monkeypatch
) -> None:
    _load_project(mw)
    series_a, series_b = _record_two_series(mw)

    window = _prepare_joint_fit(mw)
    # Nothing recorded yet: there is no joint fit to delete.
    assert not window._delete_btn.isEnabled()
    window._on_run_clicked()
    wait_for(lambda: window._worker is None, app, timeout_s=30.0)
    assert window._delete_btn.isEnabled()

    before = {
        series_a.batch_id: dict(series_a.results_by_run),
        series_b.batch_id: dict(series_b.results_by_run),
    }
    monkeypatch.setattr(
        joint_window_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: joint_window_module.QMessageBox.StandardButton.Yes),
    )
    window._on_delete_clicked()

    assert mw._project_model.joint_fits == {}
    for series, batch_id in ((series_a, "batch-a"), (series_b, "batch-b")):
        assert series.joint_fit_id is None
        assert series.shared_params == {}
        # The constraint goes; the fit that honoured it stays (D10).
        assert series.results_by_run == before[batch_id]
    assert [action.text() for action in mw._joint_fits_menu.actions()] == ["(no joint fits yet)"]
    assert window.open_joint_id() is None
    assert not window._delete_btn.isEnabled()


def test_declining_the_delete_confirmation_keeps_the_joint_fit(mw, app, monkeypatch) -> None:
    _load_project(mw)
    _record_two_series(mw)
    window = _run_joint_fit(mw, app)
    joint_id = window.open_joint_id()

    monkeypatch.setattr(
        joint_window_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: joint_window_module.QMessageBox.StandardButton.No),
    )
    window._on_delete_clicked()

    assert list(mw._project_model.joint_fits) == [joint_id]
    assert mw._project_model.batch("batch-a").joint_fit_id == joint_id


def test_deleting_a_member_series_drops_the_joint_fit_and_clears_the_window(mw, app) -> None:
    _load_project(mw)
    _record_two_series(mw)
    window = _run_joint_fit(mw, app)
    assert window.open_joint_id() is not None

    mw._on_series_delete_requested("batch-a")

    # Two members minus one leaves fewer than two: ProjectModel removes the
    # record, and the window detaches from the id it would have updated.
    assert mw._project_model.joint_fits == {}
    assert mw._project_model.batch("batch-b").joint_fit_id is None
    assert window.open_joint_id() is None
    assert [action.text() for action in mw._joint_fits_menu.actions()] == ["(no joint fits yet)"]
