"""The Batch tab edits one series (docs/plans/series-workflow.md, Phase 3).

D1/D7/D8 from the Batch tab's side: the tab always has an *open* series (a
recorded one, or a draft that has never run), a browser click never rewrites it,
opening a series restores its whole recipe, and a run either replaces the series
it was editing or moves the tab onto the new one it recorded.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

_FB = RepresentationType.TIME_FB_ASYMMETRY
_CURVE = (np.array([0.0, 0.3]), np.array([0.1, 0.05]))
_RUNS = [10, 11, 12]


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


def _group_over(mw: MainWindow, runs: list[int], name: str) -> str:
    """Load *runs*, group them and take the Batch tab into the group (D7)."""
    for index, run_number in enumerate(runs):
        mw._data_browser.add_dataset(_dataset(run_number, field=100.0 + 10.0 * index))
    mw._on_dataset_selected(runs[0])
    mw._plot_workspace.set_active_view("fb_asymmetry")
    group_id = mw._data_browser.create_data_group(runs, name=name)
    mw._on_fit_group_requested(group_id)
    return group_id


def _tab(mw: MainWindow):
    return mw._fit_panel._global_tab


def _set_range(mw: MainWindow, low: float, high: float) -> None:
    tab = _tab(mw)
    tab._fit_range_min_spin.setValue(low)
    tab._fit_range_max_spin.setValue(high)


def _run(mw: MainWindow, runs: list[int], value: float = 0.2) -> str:
    mw._on_global_fit_completed({run: (_result(value), _CURVE, []) for run in runs}, ParameterSet())
    return mw._fit_panel.open_series_id()


def _untick(mw: MainWindow, run_number: int) -> None:
    members = _tab(mw)._members_list
    for row in range(members.count()):
        item = members.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == run_number:
            item.setCheckState(Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"run {run_number} is not in the members list")


def _member_runs(mw: MainWindow) -> list[int]:
    return sorted(int(ds.run_number) for ds in mw._fit_panel.batch_datasets())


def _select_in_browser(mw: MainWindow, monkeypatch, runs: list[int]) -> None:
    """Make the browser report *runs* as the selection and publish the change."""
    datasets = [mw._data_browser.get_dataset(run) for run in runs]
    monkeypatch.setattr(mw._data_browser, "get_selected_datasets", lambda: list(datasets))
    mw._update_selected_datasets()


def _status_tag(mw: MainWindow) -> str:
    return _tab(mw)._series_group._suffix_label.text()


def _hint_shown(mw: MainWindow) -> bool:
    """Whether the selection hint is up (``isVisible`` needs a shown window)."""
    return not _tab(mw)._selection_hint.isHidden()


# ── D7: a browser click never rewrites the open series ───────────────────────


def test_selecting_runs_leaves_an_open_series_untouched_and_hints(mw, monkeypatch):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    batch_id = _run(mw, _RUNS)
    tab = _tab(mw)
    before = (
        tab._composite_model.to_dict(),
        tab.current_recipe(),
        _member_runs(mw),
        tab.bound_group_id(),
    )

    _select_in_browser(mw, monkeypatch, [11])

    assert mw._fit_panel.open_series_id() == batch_id
    assert tab._composite_model.to_dict() == before[0]
    assert tab.current_recipe() == before[1]
    assert _member_runs(mw) == before[2]
    assert tab.bound_group_id() == before[3]
    assert _hint_shown(mw)
    assert "You selected runs 11" in tab._selection_hint_banner.text()
    assert "keeps its members until you say otherwise" in tab._selection_hint_banner.text()


def test_a_selection_matching_the_members_takes_the_hint_down(mw, monkeypatch):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    _run(mw, _RUNS)

    _select_in_browser(mw, monkeypatch, [11])
    assert _hint_shown(mw)

    _select_in_browser(mw, monkeypatch, _RUNS)
    assert not _hint_shown(mw)


def test_new_series_from_selection_drops_to_a_draft_over_the_selection(mw, monkeypatch):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    _run(mw, _RUNS)

    _select_in_browser(mw, monkeypatch, [10, 11])
    # Through the hint's own button, so the click wiring is covered too.
    _tab(mw)._hint_new_series_btn.click()

    tab = _tab(mw)
    assert mw._fit_panel.open_series_id() is None
    assert _member_runs(mw) == [10, 11]
    assert tab.bound_group_id() is None
    assert _status_tag(mw) == "Draft"
    assert not _hint_shown(mw)


# ── D1: opening a series restores its recipe ─────────────────────────────────


def test_opening_a_then_b_then_a_restores_a_exactly(mw):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    _untick(mw, 12)
    first_id = _run(mw, [10, 11])
    first_recipe = _tab(mw).current_recipe()

    # A second series over the same group: a different window and every member.
    members = _tab(mw)._members_list
    members.item(members.count() - 1).setCheckState(Qt.CheckState.Checked)
    _set_range(mw, 0.5, 4.0)
    second_id = _run(mw, _RUNS)
    assert second_id != first_id

    mw._open_series_in_batch_tab(first_id)

    assert mw._fit_panel.open_series_id() == first_id
    assert _tab(mw).current_recipe() == first_recipe
    assert _member_runs(mw) == [10, 11]
    assert _tab(mw).bound_group_id() == mw._project_model.batch(first_id).group_id
    assert mw._project_model.active_series_id(_FB) == first_id


def test_the_edited_tag_flips_on_a_row_edit_and_clears_on_reopen(mw):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    batch_id = _run(mw, _RUNS)
    assert _status_tag(mw).startswith("Fitted ")

    _tab(mw)._param_table.item(0, 1).setText("0.123")
    assert _status_tag(mw) == "Edited · results show last run"

    mw._open_series_in_batch_tab(batch_id)
    assert _status_tag(mw).startswith("Fitted ")


# ── D3: a run either replaces the open series or moves the tab onto a new one ─


def test_an_identical_rerun_keeps_the_open_series(mw):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    first_id = _run(mw, _RUNS, value=0.2)
    mw._on_series_rename_requested(first_id, "My campaign")

    second_id = _run(mw, _RUNS, value=0.9)

    assert second_id == first_id
    assert mw._fit_panel.open_series_id() == first_id
    assert mw._project_model.batch(first_id).label == "My campaign"
    assert len(mw._project_model.batches) == 1
    assert _tab(mw)._results_card._notice.text() == "Re-ran My campaign — results replaced."


def test_a_changed_window_records_a_second_series_and_opens_it(mw):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    first_id = _run(mw, _RUNS, value=0.2)

    _set_range(mw, 0.0, 4.0)
    second_id = _run(mw, _RUNS, value=0.9)

    assert second_id != first_id
    assert mw._fit_panel.open_series_id() == second_id
    first = mw._project_model.batch(first_id)
    assert first.recipe["fit_range"] == {"min": 0.0, "max": 8.0}
    assert first.results_by_run[10]["parameters"]["A"] == pytest.approx(0.2)
    notice = _tab(mw)._results_card._notice.text()
    assert notice.startswith("Saved as a new series: ")


# ── D7: "Fit this group…" seeds from the group's newest series ────────────────


def test_fit_this_group_seeds_from_the_groups_newest_series(mw):
    group_id = _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    _untick(mw, 12)
    batch_id = _run(mw, [10, 11])
    recorded = mw._project_model.batch(batch_id)

    # Come back through the browser's own "Fit this group…".
    mw._on_fit_group_requested(group_id)

    tab = _tab(mw)
    assert mw._fit_panel.open_series_id() is None, "a fresh draft, not the recorded series"
    assert _status_tag(mw) == "Draft"
    assert tab.current_recipe() == recorded.recipe
    assert tab.bound_group_id() == group_id
    # The group's whole membership, with the series' exclusions unticked.
    assert _member_runs(mw) == [10, 11]
    assert tab._members_list.count() == 3


# ── The selector menu ────────────────────────────────────────────────────────


def test_the_selector_menu_lists_the_representations_series_by_group(mw, monkeypatch):
    group_id = _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    batch_id = _run(mw, _RUNS)
    tab = _tab(mw)

    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        type(tab),
        "_exec_menu",
        lambda _self, menu, _pos: seen.extend(
            (action.text(), action.isEnabled()) for action in menu.actions() if action.text()
        ),
    )
    tab._show_series_menu()

    assert seen[0] == ("New series from browser selection", True)
    # The owning group heads its own section, and the series reads
    # "<model> · <range> · <status>".
    assert ("Scan A", False) in seen
    series = mw._project_model.batch(batch_id)
    entry = mw._series_menu_text(series, mw._project_model.data_group(group_id))
    assert entry.startswith("Exponential + Constant · 0–8 µs · 3/3")
    assert (entry, True) in seen


# ── Duplicate, rename, delete ────────────────────────────────────────────────


def test_duplicate_makes_a_draft_named_after_its_source(mw):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    batch_id = _run(mw, _RUNS)
    tab = _tab(mw)
    source_recipe = tab.current_recipe()
    source_name = tab._selector_name()

    tab.duplicate_open_series()

    assert mw._fit_panel.open_series_id() is None
    assert _status_tag(mw) == "Draft"
    assert tab.current_recipe() == source_recipe
    assert _member_runs(mw) == _RUNS
    assert tab.bound_group_id() == mw._project_model.batch(batch_id).group_id
    assert tab._selector_name() == f"{source_name} (copy)"


def test_deleting_the_open_series_leaves_a_draft_over_its_runs(mw, monkeypatch):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    batch_id = _run(mw, _RUNS)
    monkeypatch.setattr(
        QMessageBox, "exec", lambda _self: QMessageBox.StandardButton.Ok, raising=False
    )

    _tab(mw)._delete_open_series()

    assert mw._project_model.batch(batch_id) is None
    assert mw._fit_panel.open_series_id() is None
    assert _member_runs(mw) == _RUNS
    assert _status_tag(mw) == "Draft"


# ── Persistence: the open series survives save/reload ────────────────────────


def test_project_reload_reopens_the_series_the_tab_was_editing(mw, monkeypatch, tmp_path):
    _group_over(mw, _RUNS, "Scan A")
    _set_range(mw, 0.0, 8.0)
    first_id = _run(mw, _RUNS)
    _set_range(mw, 0.0, 4.0)
    _run(mw, _RUNS)
    # Go back to the first series, so the reopen cannot pass by landing on the
    # newest one by accident.
    mw._open_series_in_batch_tab(first_id)

    path = tmp_path / "series.asymp"
    state = mw.collect_project_state()
    assert state["fit_states"]["time"]["global_fit_state"]["open_series_id"] == first_id
    save_project(state, path)

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    restored = _new_window()
    restored.restore_project_state(load_project(path), str(path))

    assert restored._fit_panel.open_series_id() == first_id
    assert restored._project_model.active_series_id(_FB) == first_id
