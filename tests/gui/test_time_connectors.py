"""Connector presentation, publication ownership, and real viewport lifecycle."""

import weakref
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import PlotPanel

pytestmark = [pytest.mark.gui]


def dataset(run=1, n=10001):
    x = np.linspace(0, 10, n)
    return MuonDataset(
        time=x,
        asymmetry=20 * np.cos(2 * np.pi * 18.2 * x + run),
        error=np.full(n, 0.3),
        metadata={"run_number": run},
    )


def render(panel, mode):
    data = [dataset(1), dataset(2)]
    if mode == "single":
        panel.plot_dataset(data[0])
    elif mode == "stacked":
        panel._current_polarization_axis = "ALL"
        panel.plot_vector_subplots({"P_x": [data[0]], "P_y": [data[1]]})
    elif mode == "grouped":
        panel._current_time_view_mode = "groups"
        panel.plot_grouped_time_domain_subplots(data)
    else:
        if mode == "waterfall":
            panel.set_waterfall_enabled(True)
        panel.plot_datasets(data)


@pytest.mark.parametrize("mode", ["single", "overlay", "stacked", "grouped", "waterfall"])
def test_viewport_keeps_owned_arrays_artists_and_skips_analysis(qapp, monkeypatch, mode):
    panel = PlotPanel()
    panel.set_connect_points_enabled(True)
    render(panel, mode)
    view = panel._time_view
    artists = (
        [trace.marker for trace in view.traces]
        + [trace.bars for trace in view.traces]
        + [trace.connector for trace in view.traces]
        + panel._fit_span_artists
        + panel._fit_min_handles
        + panel._fit_max_handles
    )
    readout = (panel._last_plot_time, panel._last_plot_asymmetry, panel._last_plot_error)
    preparation = Mock(side_effect=AssertionError("Viewport changes must use the published view"))
    monkeypatch.setattr(panel, "get_analysis_dataset", preparation)
    panel.set_view_limits(2, 3, -30, 60)
    assert panel._time_view is view
    assert all(
        a is b
        for a, b in zip(
            readout,
            (panel._last_plot_time, panel._last_plot_asymmetry, panel._last_plot_error),
            strict=True,
        )
    )
    current = (
        [trace.marker for trace in view.traces]
        + [trace.bars for trace in view.traces]
        + [trace.connector for trace in view.traces]
        + panel._fit_span_artists
        + panel._fit_min_handles
        + panel._fit_max_handles
    )
    assert artists == current
    panel._auto_y_btn.click()
    preparation.assert_not_called()
    for trace in view.traces:
        np.testing.assert_allclose(
            trace.connector.get_xdata()[np.isfinite(trace.connector.get_xdata())].min(),
            trace.arrays[0][np.searchsorted(trace.arrays[0], 2) - 1],
        )


def test_toggle_preserves_limits_fits_and_errorbar_legend_without_analysis(qapp, monkeypatch):
    panel = PlotPanel()
    data = dataset()
    panel._fit_curves[1] = (data.time, data.asymmetry, "Model A")
    panel.plot_dataset(data)
    view = panel._time_view
    before = panel.get_view_limits()
    preparation = Mock(side_effect=AssertionError("A presentation toggle must not reduce data"))
    monkeypatch.setattr(panel, "get_analysis_dataset", preparation)
    panel.set_connect_points_enabled(True)
    panel._canvas.draw()
    trace = view.traces[0]
    assert trace.connector.get_visible()
    assert trace.connector.get_linewidth() < view.fits[0][0].get_linewidth()
    assert (
        trace.connector.get_zorder()
        < view.fits[0][0].get_zorder()
        < trace.bars.get_zorder()
        < trace.marker.get_zorder()
    )
    assert view.fits[0][0].get_linestyle() == "-"
    assert [t.get_text() for t in panel._ax.get_legend().get_texts()] == ["1", "Fit: Model A"]
    glyphs = panel._ax.get_legend().findobj()
    assert any(
        isinstance(g, type(trace.marker))
        or (isinstance(g, type(view.fits[0][0])) and g.get_linestyle() == "--")
        for g in glyphs
    )
    assert panel.get_view_limits() == before
    panel.set_connect_points_enabled(False)
    assert not trace.connector.get_visible()
    assert panel._time_view is view
    preparation.assert_not_called()


def test_geometry_uses_source_not_strided_markers_and_refreshes_for_resize(qapp):
    panel = PlotPanel()
    panel._max_render_points_per_trace = 30
    panel.set_connect_points_enabled(True)
    data = dataset(n=100001)
    panel.plot_dataset(data)
    trace = panel._time_view.traces[0]
    assert trace.marker.get_xdata().size <= 31
    assert trace.connector.get_xdata().size > 1000
    first = trace.connector.viewport
    panel._figure.set_dpi(panel._figure.dpi * 2)
    panel._canvas.draw()
    assert trace.connector.viewport[1] > first[1]
    panel.set_view_limits(2, 2.001, -30, 30)
    expected = trace.connector.geometry.vertices((2, 2.001), None)
    np.testing.assert_array_equal(np.column_stack(trace.connector.get_data()), expected)


def test_publication_replaces_owned_snapshot_and_mutable_aliases_cannot_change_it(qapp):
    panel = PlotPanel()
    data = dataset()
    original = data.asymmetry.copy()
    alias = data.asymmetry.view()
    data.asymmetry.flags.writeable = False
    panel.plot_dataset(data)
    old_view = weakref.ref(panel._time_view)
    alias[:] += 5
    panel.set_view_limits(2, 4, -30, 30)
    np.testing.assert_array_equal(panel._last_plot_asymmetry, original)
    panel.plot_dataset(data)
    assert old_view() is None
    np.testing.assert_array_equal(panel._last_plot_asymmetry, original + 5)
    panel.clear()
    assert not panel._time_view.traces


def test_changed_data_error_mask_labels_and_fits_replace_view(qapp):
    panel = PlotPanel()
    data = dataset()
    panel.plot_dataset(data)
    first = panel._time_view
    data.error[20] = np.nan
    data.asymmetry[10] += 3
    data.metadata["comment"] = "Edited"
    panel._fit_curves[1] = (data.time, np.zeros_like(data.time), "Changed model")
    panel.plot_dataset(data)
    assert panel._time_view is not first
    panel.set_connect_points_enabled(True)
    panel._canvas.draw()
    trace = panel._time_view.traces[0]
    assert 20 not in trace.connector.geometry.indices
    assert trace.arrays[1][10] == data.asymmetry[10]
    assert panel._time_view.fits[0][2] == "Changed model"


def test_unchanged_paint_reuses_geometry_and_vector_export_uses_full_source(qapp, monkeypatch):
    panel = PlotPanel()
    panel.set_connect_points_enabled(True)
    panel.plot_dataset(dataset(n=100001))
    trace = panel._time_view.traces[0]
    selector = Mock(wraps=trace.connector.geometry.vertices)
    monkeypatch.setattr(trace.connector.geometry, "vertices", selector)
    panel._canvas.draw()
    selector.assert_not_called()
    panel._figure.savefig(BytesIO(), format="svg")
    assert trace.connector.viewport[1] is None
    assert trace.connector.get_xdata().size == 100001
    panel._canvas.draw()
    assert trace.connector.viewport[1] is not None


def test_project_menu_persists_setting_and_disables_unsupported_views(qapp):
    window = MainWindow()
    action = window._plot_display_menu.actions()[0]
    assert not action.isChecked() and action.isEnabled()
    action.trigger()
    assert window._plot_panel.connect_points_enabled()
    assert window._plot_panel.get_state()["connect_points"] is True
    window._plot_workspace.set_active_view("frequency")
    assert not action.isEnabled() and action.isChecked()
    window._plot_workspace.set_active_view("fb_asymmetry")
    assert action.isEnabled()
    window._plot_panel.restore_state({"connect_points": True})
    assert action.isChecked()
    window._plot_panel.restore_state({})
    assert not action.isChecked()


def test_raw_counts_and_frequency_keep_their_existing_rendering(qapp):
    for domain, mode in [("time", "raw_counts"), ("frequency", "fb_asymmetry")]:
        panel = PlotPanel(domain=domain)
        panel._current_time_view_mode = mode
        panel.set_connect_points_enabled(True)
        panel.plot_dataset(dataset())
        assert panel._time_view is None


@pytest.mark.parametrize("mode", ["single", "overlay", "stacked", "grouped", "waterfall"])
def test_viewport_image_matches_a_fresh_publication(qapp, mode):
    panel = PlotPanel()
    panel.set_connect_points_enabled(True)
    render(panel, mode)
    panel.set_view_limits(2, 3, -30, 60)
    panel._canvas.draw()
    reused = np.asarray(panel._canvas.buffer_rgba()).copy()
    panel._redraw_current_view()
    panel._canvas.draw()
    np.testing.assert_array_equal(reused, np.asarray(panel._canvas.buffer_rgba()))


def test_saved_project_roundtrip_older_state_and_new_project_defaults(qapp, tmp_path):
    from asymmetry.core.project import load_project, save_project

    window = MainWindow()
    window._plot_panel.set_connect_points_enabled(True)
    path = tmp_path / "connectors.asymp"
    save_project(window.collect_project_state(), path)
    state = load_project(path)
    other = MainWindow()
    other.restore_project_state(state, str(path))
    assert other._plot_panel.connect_points_enabled()
    assert other._plot_display_menu.actions()[0].isChecked()
    other._clear_all_state()
    assert not other._plot_panel.connect_points_enabled()
    assert window._plot_panel.connect_points_enabled()
    state["plot_state"].pop("connect_points")
    other.restore_project_state(state, str(path))
    assert not other._plot_panel.connect_points_enabled()


def test_gle_uses_full_gapped_source_and_numeric_export_is_unchanged(qapp, tmp_path):
    panel = PlotPanel()
    data = dataset()
    data.error[20] = np.nan
    panel.plot_dataset(data)
    before = panel.get_current_plot_export_data()[0]["data"]
    panel.set_connect_points_enabled(True)
    payloads = panel.get_current_plot_export_data()
    for name in before:
        np.testing.assert_array_equal(before[name], payloads[0]["data"][name])
    snapshot = panel._gle_snapshot(payloads)
    ax = Mock()
    snapshot._plot_export_payloads_on_axis(
        ax,
        payloads,
        axis_key=None,
        written_files=[],
        dat_writes=[],
        gle_path=tmp_path / "plot.gle",
        colors=["black"],
        show_legend=True,
    )
    guides = [call for call in ax.plot.call_args_list if call.args[2] == "--"]
    assert len(guides) == 1
    x = guides[0].args[0]
    assert np.isnan(x).sum() == 1
    assert np.isfinite(x).sum() == data.time.size - 1
    assert guides[0].kwargs["linewidth"] == 0.65


def test_disabling_decimation_restores_all_visible_connector_points(qapp):
    panel = PlotPanel()
    panel.plot_dataset(dataset(n=100001))
    panel.set_connect_points_enabled(True)
    panel._canvas.draw()
    view = panel._time_view
    panel.set_decimation_enabled(False)
    assert panel._time_view is view
    assert view.traces[0].connector.viewport[1] is None
    assert view.traces[0].connector.get_xdata().size == 100001


def test_confidence_boundaries_keep_separate_guides_and_one_legend_entry(qapp, monkeypatch):
    panel = PlotPanel()
    data = dataset()
    monkeypatch.setattr(panel, "_low_count_mask_for_dataset", lambda *_a, **_k: data.time < 5)
    panel.set_connect_points_enabled(True)
    panel.plot_dataset(data)
    low, main = panel._time_view.traces
    assert low.marker.get_color() == "0.6"
    assert np.nanmax(low.connector.get_xdata()) < 5
    assert np.nanmin(main.connector.get_xdata()) >= 5
    assert [t.get_text() for t in panel._ax.get_legend().get_texts()] == ["1"]
    assert all(a.flags.writeable for a in (data.time, data.asymmetry, data.error))


def test_project_boundary_rejects_non_boolean_connector_setting(qapp):
    from asymmetry.core.project.preparation import ProjectValidationError, prepare_project

    window = MainWindow()
    state = window.collect_project_state()
    state["plot_state"]["connect_points"] = "enabled"
    with pytest.raises(ProjectValidationError, match="connect_points"):
        prepare_project(state)


def test_overlay_fit_legends_identify_the_run_when_model_names_match(qapp):
    panel = PlotPanel()
    data = [dataset(1), dataset(2)]
    for item in data:
        panel._fit_curves[item.run_number] = (item.time, item.asymmetry, "Shared model")
    panel.plot_datasets(data)
    panel.set_connect_points_enabled(True)
    labels = [text.get_text() for text in panel._ax.get_legend().get_texts()]
    assert len(labels) == 4
    assert set(labels) == {"1", "2", "Fit: 1 — Shared model", "Fit: 2 — Shared model"}


def test_clearing_fit_visibility_override_publishes_the_default_fit_view(qapp):
    panel = PlotPanel()
    data = dataset()
    panel._fit_curves[1] = (data.time, data.asymmetry, "Default model")
    panel.set_shown_fits(1, [])
    panel.plot_dataset(data)
    view = panel._time_view
    assert not view.fits
    panel.clear_shown_fits(1)
    assert panel._time_view is not view
    assert len(panel._time_view.fits) == 1
    panel.set_view_limits(2, 4, -30, 30)
    assert len(panel._time_view.fits) == 1


@pytest.mark.parametrize("combined", [False, True])
def test_reconstruction_snapshots_exclude_connectors(qapp, combined):
    panel = PlotPanel()
    panel.set_connect_points_enabled(True)
    panel.plot_maxent_reconstruction([dataset(n=101)], combined=combined)
    assert not panel._gle_snapshot(panel._collect_export_payloads()).connect_points


def test_frequency_dataset_on_time_panel_excludes_connector_exports(qapp):
    panel = PlotPanel()
    panel.set_connect_points_enabled(True)
    data = dataset(n=101)
    data.metadata["plot_domain"] = "frequency"
    panel.plot_dataset(data)
    assert panel._time_view is None
    assert not panel._gle_snapshot(panel._collect_export_payloads()).connect_points
