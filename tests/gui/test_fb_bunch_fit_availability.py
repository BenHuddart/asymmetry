"""F–B fit inputs and saved fits track the toolbar's recorded bin merges."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.input_attribution import fit_input_attribution
from asymmetry.core.fitting.likelihood import dataset_observations
from asymmetry.core.representation import FitSeries, RepresentationType

pytestmark = [pytest.mark.gui]
pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel, Qt  # noqa: E402

from tests.gui.bunch_helpers import BIN_WIDTH_US, histogram_run_dataset  # noqa: E402
from tests.gui.bunch_helpers import mainwindow as mainwindow  # noqa: E402


@pytest.mark.parametrize("recall", [False, True], ids=["direct", "saved-view"])
def test_fit_availability_tracks_bunching_and_run_selection(mainwindow, recall):
    long, short, other = [
        histogram_run_dataset(np.full(n, 200.0), np.full(n, 150.0), run_number=run)
        for run, n in [(4242, 256), (4243, 16), (4244, 256)]
    ]
    browser = mainwindow._data_browser
    for dataset in (long, short, other):
        browser.add_dataset(dataset)

    browser.select_runs({long.run_number})
    bunch = mainwindow._view_bunch_spin
    view_one, view_two, _ = mainwindow._view_mode_buttons
    single = mainwindow._fit_panel._single_tab
    batch = mainwindow._fit_panel._global_tab
    reason = "No complete bins at Bunch 32. Reduce Bunch."
    plot = mainwindow._plot_panel
    fit_window = (0.10, 0.80)
    plot.set_fit_range(*fit_window)

    view_two.click()
    bunch.setValue(1)
    view_one.click()
    bunch.setValue(32)
    if recall:
        view_two.click()
        assert bunch.value() == 1
        view_one.click()
    assert bunch.value() == 32
    assert single._fit_btn.isEnabled()

    browser.select_runs({short.run_number})
    assert mainwindow._fit_panel.single_dataset().n_points == 0
    assert not single._fit_btn.isEnabled()
    assert single._fit_btn.toolTip() == reason
    assert mainwindow.statusBar().currentMessage() == reason
    assert plot.get_fit_range() == fit_window

    browser.select_runs({long.run_number, short.run_number, other.run_number})
    assert not single._fit_btn.isEnabled()
    assert batch._fit_btn.isEnabled()
    assert {ds.run_number for ds in batch.batch_datasets()} == {long.run_number, other.run_number}
    assert batch._unavailable_members == {short.run_number: reason}

    browser.select_runs({long.run_number})
    assert single._fit_btn.isEnabled()
    assert plot.get_fit_range() == fit_window
    np.testing.assert_allclose(mainwindow._fit_panel.single_dataset().time, [0.248, 0.76])
    assert reason not in mainwindow.statusBar().currentMessage()

    browser.select_runs({short.run_number})
    assert not single._fit_btn.isEnabled()
    if recall:
        view_two.click()
        assert single._fit_btn.isEnabled()
        assert mainwindow._fit_panel.single_dataset().n_points == 9
        view_one.click()
        assert not single._fit_btn.isEnabled()
        assert mainwindow.statusBar().currentMessage() == reason
    bunch.setValue(1)
    assert mainwindow._fit_panel.single_dataset().n_points == 9
    assert single._fit_btn.isEnabled()
    assert reason not in mainwindow.statusBar().currentMessage()

    bunch.setValue(32)
    assert not single._fit_btn.isEnabled()
    assert mainwindow.statusBar().currentMessage() == reason
    assert plot.get_fit_range() == fit_window
    bunch.setValue(1)
    assert single._fit_btn.isEnabled()
    assert plot.get_fit_range() == fit_window
    assert mainwindow._fit_panel.single_dataset().n_points == 9


@pytest.mark.parametrize("recall", [False, True], ids=["direct", "saved-view"])
def test_fit_window_survives_an_empty_bunched_overlay(mainwindow, recall):
    browser = mainwindow._data_browser
    runs = [4242, 4243]
    for run in runs:
        browser.add_dataset(
            histogram_run_dataset(np.full(16, 200.0), np.full(16, 150.0), run_number=run)
        )
    browser.select_runs({runs[0]})
    plot = mainwindow._plot_panel
    plot.set_overlay_enabled(True, emit_signal=True)
    browser.select_runs(set(runs))
    bunch = mainwindow._view_bunch_spin
    view_one, view_two, _ = mainwindow._view_mode_buttons
    if recall:
        view_two.click()
        bunch.setValue(32)
        view_one.click()

    fit_window = (0.10, 0.80)
    plot.set_fit_range(*fit_window)
    panel = mainwindow._fit_panel
    single = panel._single_tab._fit_btn
    batch = panel._global_tab._fit_btn
    assert single.isEnabled() and batch.isEnabled()
    assert panel.single_dataset().n_points == 9

    if recall:
        view_two.click()
    else:
        bunch.setValue(32)
    assert bunch.value() == 32
    assert not single.isEnabled() and not batch.isEnabled()
    assert mainwindow.statusBar().currentMessage() == "No complete bins at Bunch 32. Reduce Bunch."
    assert plot.get_fit_range() == fit_window

    if recall:
        view_one.click()
    else:
        bunch.setValue(1)
    assert bunch.value() == 1
    assert plot.get_fit_range() == fit_window
    assert single.isEnabled() and batch.isEnabled()
    assert panel.single_dataset().n_points == 9
    assert {ds.run_number for ds in panel.batch_datasets()} == set(runs)


def test_selecting_a_group_keeps_the_empty_fit_reason(mainwindow):
    browser = mainwindow._data_browser
    runs = [4242, 4243]
    for run in runs:
        browser.add_dataset(
            histogram_run_dataset(np.full(16, 200.0), np.full(16, 150.0), run_number=run)
        )
    group = browser.create_data_group(runs, name="Short runs")
    table = browser._table
    header = next(
        table.model().index(row, 0)
        for row in range(table.rowCount())
        if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == f"group:{group}"
    )
    browser.select_runs({runs[0]})
    mainwindow._view_bunch_spin.setValue(32)
    fit = mainwindow._fit_panel._single_tab._fit_btn
    reason = "No complete bins at Bunch 32. Reduce Bunch."

    select = QItemSelectionModel.SelectionFlag
    table.selectionModel().select(header, select.ClearAndSelect | select.Rows)
    assert not fit.isEnabled()
    assert fit.toolTip() == reason
    assert mainwindow.statusBar().currentMessage() == reason

    mainwindow._view_bunch_spin.setValue(1)
    browser.select_runs({runs[0]})
    table.selectionModel().select(header, select.ClearAndSelect | select.Rows)
    assert fit.isEnabled()
    assert mainwindow.statusBar().currentMessage() == "Selected group: Short runs"


def test_fit_availability_tracks_the_window_independently_of_bunching(mainwindow):
    dataset = histogram_run_dataset(np.full(64, 200.0), np.full(64, 150.0))
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({dataset.run_number})
    mainwindow._view_bunch_spin.setValue(4)
    plot = mainwindow._plot_panel
    fit = mainwindow._fit_panel._single_tab._fit_btn

    plot.set_fit_range(0.0, 0.01)
    assert mainwindow._fit_panel.single_dataset().n_points == 0
    assert not fit.isEnabled()
    assert fit.toolTip() == "No bins in the fit window."
    assert mainwindow.statusBar().currentMessage() == fit.toolTip()

    plot.set_fit_range(0.0, 1.0)
    assert fit.isEnabled()
    assert fit.toolTip() == ""
    assert "No bins" not in mainwindow.statusBar().currentMessage()


def test_real_fit_keeps_attribution_when_bunch_and_zoom_change(mainwindow):
    index = np.arange(256, dtype=float)
    forward = 190 + 3 * (index % 13) + index
    backward = 160 + 2 * (index % 11) + index / 2
    dataset = histogram_run_dataset(forward, backward)
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({dataset.run_number})
    panel = mainwindow._fit_panel
    tab = panel._single_tab
    plot = mainwindow._plot_panel
    fit_window = (0.10, 3.55)
    plot.set_fit_range(*fit_window)
    view_one, view_two, _ = mainwindow._view_mode_buttons

    def check_fit_input(factor):
        t = index.reshape(-1, factor).mean(axis=1) * BIN_WIDTH_US
        f = forward.reshape(-1, factor).sum(axis=1)
        b = backward.reshape(-1, factor).sum(axis=1)
        mask = (t >= fit_window[0]) & (t <= fit_window[1])
        observations = dataset_observations(panel.single_dataset())
        np.testing.assert_allclose(observations.time, t[mask], atol=1e-15)
        np.testing.assert_allclose(observations.values, (100 * (f - b) / (f + b))[mask])
        sigma = 200 * np.sqrt(b**2 * f + f**2 * b) / (f + b) ** 2
        np.testing.assert_allclose(observations.sigma, sigma[mask])
        assert plot.get_fit_range() == fit_window
        return fit_input_attribution(observations, panel.single_dataset())

    view_two.click()
    mainwindow._view_bunch_spin.setValue(1)
    view_one.click()
    mainwindow._view_bunch_spin.setValue(4)
    producing_input = check_fit_input(4)
    plot.set_view_limits(0.2, 0.8, -30.0, 30.0)
    assert check_fit_input(4) == producing_input

    tab._set_composite_model(CompositeModel(["Constant"]))
    tab._fit_btn.click()
    assert tab.wait_for_fit()
    rep = mainwindow._project_model.representation(
        dataset.run_number, RepresentationType.TIME_FB_ASYMMETRY
    )
    (original,) = rep.fit_set(None).fits
    assert original.result["input_attribution"] == producing_input
    saved_attribution = deepcopy(original.result["input_attribution"])
    assert not mainwindow._saved_fit_catalogue().entries[0].stale_reason

    view_two.click()
    assert mainwindow._view_bunch_spin.value() == 1
    native_input = check_fit_input(1)
    assert native_input["n"] > producing_input["n"]
    assert "Historical fit" in mainwindow._saved_fit_catalogue().entries[0].stale_reason
    assert original.result["input_attribution"] == saved_attribution

    tab._fit_btn.click()
    assert tab.wait_for_fit()
    _, native = rep.fit_set(None).fits
    assert native.result["input_attribution"] == native_input

    view_one.click()
    assert mainwindow._view_bunch_spin.value() == 4
    assert check_fit_input(4) == producing_input
    assert not mainwindow._saved_fit_catalogue().entries[0].stale_reason
    assert original.result["input_attribution"] == saved_attribution
    assert dataset.run.grouping["bunching_factor"] == 1


@pytest.mark.parametrize("reopened", [False, True], ids=["group-draft", "reopened"])
def test_batch_pool_tracks_empty_inputs_without_losing_the_recipe(mainwindow, reopened):
    runs = [4242, 4243, 4244, 4245]
    browser = mainwindow._data_browser
    for run, n in zip(runs, [256, 16, 256, 16], strict=True):
        browser.add_dataset(
            histogram_run_dataset(np.full(n, 200.0), np.full(n, 150.0), run_number=run)
        )
    browser.select_runs({runs[0]})
    plot = mainwindow._plot_panel
    plot.set_fit_range(0.0, 0.8)
    group = browser.create_data_group(runs, name="Mixed records")
    mainwindow._on_fit_group_requested(group)
    tab = mainwindow._fit_panel._global_tab

    excluded = next(
        entry
        for row in range(tab._members_list.count())
        if (entry := tab._members_list.item(row)).data(Qt.ItemDataRole.UserRole) == runs[-1]
    )
    excluded.setCheckState(Qt.CheckState.Unchecked)
    tab.set_fit_range_display(0.0, 0.8)
    recipe = tab.current_recipe()
    bunch = mainwindow._view_bunch_spin
    bunch.setValue(32)
    assert tab.bound_group_id() == group
    assert {ds.run_number for ds in tab.batch_datasets()} == {runs[0], runs[2]}
    assert tab._fit_btn.isEnabled()

    if reopened:
        series = FitSeries(
            "recorded",
            RepresentationType.TIME_FB_ASYMMETRY,
            member_run_numbers=runs[:-1],
            excluded_run_numbers=[runs[-1]],
            group_id=group,
            recipe=recipe,
        )
        mainwindow._project_model.add_batch(series)
        mainwindow._open_series_in_batch_tab(series.batch_id)
        assert mainwindow._fit_panel.open_series_id() == series.batch_id

    reason = "No complete bins at Bunch 32. Reduce Bunch."
    assert tab.current_recipe() == recipe
    assert tab._unavailable_members == {runs[1]: reason, runs[3]: reason}
    assert {ds.run_number for ds in tab.batch_datasets()} == {runs[0], runs[2]}
    assert tab._fit_btn.isEnabled()

    included = next(
        entry
        for row in range(tab._members_list.count())
        if (entry := tab._members_list.item(row)).data(Qt.ItemDataRole.UserRole) == runs[0]
    )
    included.setCheckState(Qt.CheckState.Unchecked)
    assert not tab._fit_btn.isEnabled()
    included.setCheckState(Qt.CheckState.Checked)

    tab.set_fit_range_display(0.0, 0.01)
    tab._on_fit_range_spinbox_committed()
    assert tab.batch_datasets() == []
    assert not tab._fit_btn.isEnabled()
    assert tab._fit_btn.toolTip() == "No bins in the fit window."
    assert plot.get_fit_range() == (0.0, 0.8)

    plot.fit_range_guide_changed.emit(0.0, 0.8)
    assert tab._fit_btn.isEnabled()
    assert {ds.run_number for ds in tab.batch_datasets()} == {runs[0], runs[2]}

    bunch.setValue(1)
    assert tab.bound_group_id() == group
    assert tab.current_recipe() == recipe
    assert not tab._unavailable_members
    assert {ds.run_number for ds in tab.batch_datasets()} == set(runs[:-1])


def test_group_draft_uses_its_own_window_before_launch(mainwindow):
    browser = mainwindow._data_browser
    runs = [4242, 4243]
    for run in runs:
        browser.add_dataset(
            histogram_run_dataset(np.full(16, 200.0), np.full(16, 150.0), run_number=run)
        )
    browser.select_runs({runs[0]})
    group = browser.create_data_group(runs, name="Short records")
    tab = mainwindow._fit_panel._global_tab
    tab.set_fit_range_display(0.3, 0.8)
    mainwindow._on_fit_group_requested(group)

    assert tab.bound_group_id() == group
    assert tab._unavailable_members == dict.fromkeys(runs, "No bins in the fit window.")
    assert tab.batch_datasets() == []
    assert not tab._fit_btn.isEnabled()

    tab.set_fit_range_display(0.0, 0.20)
    tab._on_fit_range_spinbox_committed()
    assert tab._fit_btn.isEnabled()
    tab._set_composite_model(CompositeModel(["Constant"]))
    tab._fit_btn.click()
    assert tab.wait_for_fit()
    series = mainwindow._project_model.batch(mainwindow._fit_panel.open_series_id())
    assert set(series.results_by_run) == set(runs)
    for summary in series.results_by_run.values():
        assert summary["success"]
        assert summary["input_attribution"]["n"] == 13


@pytest.mark.parametrize(
    "window,indices,summary",
    [
        ({"min": 0.10, "max": None}, np.arange(7, 16), "≥ 0.100 µs"),
        ({"min": None, "max": 0.10}, np.arange(7), "≤ 0.100 µs"),
    ],
    ids=["lower-bound", "upper-bound"],
)
def test_open_series_keeps_a_legacy_bound_with_an_empty_member_pool(
    mainwindow, window, indices, summary
):
    browser = mainwindow._data_browser
    runs = [4242, 4243]
    for run in runs:
        browser.add_dataset(
            histogram_run_dataset(np.full(16, 200.0), np.full(16, 150.0), run_number=run)
        )
    browser.select_runs({runs[0]})
    series = FitSeries(
        "legacy",
        RepresentationType.TIME_FB_ASYMMETRY,
        member_run_numbers=runs,
        recipe={"fit_range": window},
    )
    mainwindow._project_model.add_batch(series)
    bunch = mainwindow._view_bunch_spin
    bunch.setValue(32)
    tab = mainwindow._fit_panel._global_tab
    tab.set_fit_range_display(0.0, 0.80)
    mainwindow._open_series_in_batch_tab(series.batch_id)
    assert not tab._fit_btn.isEnabled()
    assert tab.current_recipe()["fit_range"] == window
    assert tab.current_fit_range_text() == summary

    bunch.setValue(1)
    assert tab._fit_btn.isEnabled()
    assert {ds.run_number for ds in tab.batch_datasets()} == set(runs)
    for dataset in tab.batch_datasets():
        np.testing.assert_allclose(dataset.time, indices * BIN_WIDTH_US)
    assert tab.current_recipe()["fit_range"] == window
    tab._set_composite_model(CompositeModel(["Constant"]))
    tab._fit_btn.click()
    assert tab.wait_for_fit()
    fitted = mainwindow._project_model.batch(mainwindow._fit_panel.open_series_id())
    assert fitted.recipe["fit_range"] == window
    assert {item["fit_range"] for item in fitted.results_by_run.values()} == {summary}
