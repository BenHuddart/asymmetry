"""Phase 4: per-series trend surface and representation-aware panel.

Tests cover:
- FitParametersPanel.load_representation_series (pull-based refresh).
- MainWindow._refresh_trend_panel after batch/grouped fits.
- Representation change swaps the trend-panel content.
- Data-browser series highlighting.
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
from asymmetry.core.representation import FitSeries, FitSlot, RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        {"run_number": run_number, "field": field},
        run,
    )


def _result(rchi: float = 0.5, **param_kw) -> FitResult:
    params = {"A": 0.2, "Lambda": 0.5, **param_kw}
    return FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=rchi,
        parameters=ParameterSet([Parameter(name, val) for name, val in params.items()]),
        uncertainties={name: 0.01 for name in params},
    )


_CURVE = (np.array([0.0, 0.3]), np.array([0.1, 0.05]))


# ---------------------------------------------------------------------------
# FitParametersPanel.load_representation_series
# ---------------------------------------------------------------------------


class TestLoadRepresentationSeries:
    """The pull-based reload entry point produces the correct panel state."""

    def test_panel_groups_keyed_by_batch_id(self, app):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        panel = FitParametersPanel()
        rows_a = [
            {
                "run_number": 10,
                "run_label": "10",
                "field": 100.0,
                "temperature": 20.0,
                "values": {"A": 0.2},
                "errors": {"A": 0.01},
            },
        ]
        rows_b = [
            {
                "run_number": 20,
                "run_label": "20",
                "field": 200.0,
                "temperature": 20.0,
                "values": {"A": 0.3},
                "errors": {"A": 0.01},
            },
        ]
        panel.load_representation_series(
            [("batch-0", "Series 1", rows_a), ("batch-1", "Series 2", rows_b)],
        )

        assert "batch-0" in panel._group_fit_results
        assert "batch-1" in panel._group_fit_results
        assert panel._group_fit_results["batch-0"].group_name == "Series 1"
        assert panel._group_fit_results["batch-1"].group_name == "Series 2"

    def test_most_recent_series_becomes_active(self, app):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        def _row(rn):
            return [
                {
                    "run_number": rn,
                    "run_label": str(rn),
                    "field": 0.0,
                    "temperature": 0.0,
                    "values": {"A": 0.1},
                    "errors": {"A": 0.01},
                }
            ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("batch-0", "Series 1", _row(10)), ("batch-1", "Series 2", _row(20))],
        )

        assert panel._active_group_id == "batch-1"

    def test_existing_active_series_preserved_on_reload(self, app):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        def _row(rn):
            return [
                {
                    "run_number": rn,
                    "run_label": str(rn),
                    "field": 0.0,
                    "temperature": 0.0,
                    "values": {"A": 0.1},
                    "errors": {"A": 0.01},
                }
            ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("batch-0", "Series 1", _row(10)), ("batch-1", "Series 2", _row(20))],
        )
        # Manually activate Series 1 to simulate user clicking it.
        panel._active_group_id = "batch-0"

        # Reload with same entries.
        panel.load_representation_series(
            [("batch-0", "Series 1", _row(10)), ("batch-1", "Series 2", _row(20))],
        )

        # Active series should stay at batch-0.
        assert panel._active_group_id == "batch-0"

    def test_clear_resets_series_buttons(self, app):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        panel = FitParametersPanel()
        row = [
            {
                "run_number": 1,
                "run_label": "1",
                "field": 0.0,
                "temperature": 0.0,
                "values": {"A": 0.1},
                "errors": {"A": 0.01},
            }
        ]
        panel.load_representation_series([("batch-0", "Series 1", row)])
        assert "batch-0" in panel._group_fit_results

        panel.clear()
        assert panel._group_fit_results == {}
        assert panel._active_group_id is None


# ---------------------------------------------------------------------------
# MainWindow._refresh_trend_panel
# ---------------------------------------------------------------------------


class TestRefreshTrendPanel:
    """_refresh_trend_panel drives the panel from the project model."""

    def test_global_fit_populates_trend_panel(self, mw, monkeypatch):
        """After a batch fit _refresh_trend_panel shows the new series."""
        for rn, field in [(10, 100.0), (11, 50.0)]:
            mw._data_browser.add_dataset(_dataset(rn, field))
        mw._on_dataset_selected(10)
        mw._plot_workspace.set_active_view("fb_asymmetry")
        monkeypatch.setattr(
            mw._fit_panel,
            "get_global_state",
            lambda: {
                "composite_model": {"component_names": ["Exponential"], "operators": []},
                "parameters": [{"name": "A", "type": "Local"}],
                "result_html": "",
            },
        )
        payloads = {rn: (_result(), _CURVE, []) for rn in (10, 11)}
        mw._on_global_fit_started()  # the fit panel's launch signal, as in production
        mw._on_global_fit_completed(payloads, ParameterSet())

        panel = mw._fit_parameters_panel
        # The panel should have one series entry keyed by the batch id.
        assert len(panel._group_fit_results) == 1
        gdata = next(iter(panel._group_fit_results.values()))
        run_numbers = {r.run_number for r in gdata.rows}
        assert run_numbers == {10, 11}

    def test_representation_switch_swaps_trend_content(self, mw, monkeypatch):
        """Switching from fb_asymmetry → groups → fb_asymmetry swaps series."""
        # Step 1: batch fit in FB Asymmetry.
        for rn in (10, 11):
            mw._data_browser.add_dataset(_dataset(rn))
        mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
        mw._plot_workspace.set_active_view("fb_asymmetry")
        monkeypatch.setattr(
            mw._fit_panel,
            "get_global_state",
            lambda: {
                "composite_model": {"component_names": ["Exponential"], "operators": []},
                "parameters": [{"name": "A", "type": "Local"}],
                "result_html": "",
            },
        )
        mw._on_global_fit_started()  # the fit panel's launch signal, as in production
        mw._on_global_fit_completed(
            {rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet()
        )
        # Confirm the FB Asymmetry series was recorded.
        assert len(mw._project_model.batches) == 1

        # The ad-hoc batch auto-created its group (D3), which refreshes the browser
        # and re-derives the available views from the (stub) datasets; re-pin them
        # so the rep-switch precondition holds (real grouped-capable data keeps
        # "groups" available on its own).
        mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])

        # Step 2: grouped fit in Groups representation.
        mw._plot_workspace.set_active_view("groups")
        monkeypatch.setattr(
            mw._multi_group_fit_window,
            "get_grouped_state",
            lambda: {
                "composite_model": {"component_names": ["Exponential"], "operators": []},
                "param_roles": {"Lambda": "local"},
                "nuisance_params": [],
            },
        )
        # A multi-run batch (10, 11) records a groups series.
        grouped_datasets = [
            MuonDataset(
                np.array([0.0, 0.1]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                {"run_number": -10001, "source_run_number": 10},
                None,
            ),
            MuonDataset(
                np.array([0.0, 0.1]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                {"run_number": -11001, "source_run_number": 11},
                None,
            ),
        ]
        results = {-10001: (_result(), _CURVE, []), -11001: (_result(), _CURVE, [])}
        mw._on_grouped_fit_completed(grouped_datasets, results)

        # Now the panel should show the groups series.
        panel = mw._fit_parameters_panel
        group_ids = set(panel._group_fit_results.keys())
        # All should be groups-rep series.
        for gid in group_ids:
            batch = mw._project_model.batch(gid)
            assert batch is not None
            assert batch.rep_type == RepresentationType.TIME_GROUPS

        # Step 3: switch back to FB Asymmetry → see its series.
        mw._plot_workspace.set_active_view("fb_asymmetry")
        for gid in panel._group_fit_results:
            batch = mw._project_model.batch(gid)
            assert batch is not None
            assert batch.rep_type == RepresentationType.TIME_FB_ASYMMETRY

    def test_grouped_batch_result_appears_in_trend_panel(self, mw, monkeypatch):
        """Grouped fits were previously invisible to the trend panel; Phase 4 fixes that.

        A grouped *batch* (≥2 source runs) records a series and, with global
        physics, collapses to one trend point per source run.
        """
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
                "nuisance_params": ["N0"],
            },
        )
        grouped_datasets = [self._group_member(42, 1), self._group_member(43, 1)]
        results = {
            -42001: (_result(rchi=0.35), _CURVE, []),
            -43001: (_result(rchi=0.4), _CURVE, []),
        }
        mw._on_grouped_fit_completed(grouped_datasets, results)

        panel = mw._fit_parameters_panel
        assert len(panel._group_fit_results) == 1
        gdata = next(iter(panel._group_fit_results.values()))
        # Physics is global (Lambda), so the series collapses to one trend point
        # per source run, keyed by the run (42, 43), not the synthetic group key.
        assert {r.run_number for r in gdata.rows} == {42, 43}

    def test_grouped_batch_renders_only_current_run_subplots(self, mw, monkeypatch):
        """A batch grouped fit must not render every (run, group) member.

        Rendering all members produced one dense subplot per member (e.g. 84 for
        21 runs × 4 groups), freezing the GUI for seconds when the values came
        back. Only the current run's detector groups should be drawn.
        """
        for rn in (42, 43):
            mw._data_browser.add_dataset(_dataset(rn))
        mw._on_dataset_selected(42)
        mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
        mw._plot_workspace.set_active_view("groups")
        monkeypatch.setattr(
            mw._multi_group_fit_window, "get_grouped_state", self._grouped_state_stub
        )

        rendered: list[list] = []
        monkeypatch.setattr(
            mw._plot_panel,
            "plot_grouped_time_domain_subplots",
            lambda datasets: rendered.append(list(datasets)),
        )

        # 2-run batch, 2 groups each (4 members total).
        grouped_datasets = [
            self._group_member(42, 1),
            self._group_member(42, 2),
            self._group_member(43, 1),
            self._group_member(43, 2),
        ]
        results = {
            -42001: (_result(), _CURVE, []),
            -42002: (_result(), _CURVE, []),
            -43001: (_result(), _CURVE, []),
            -43002: (_result(), _CURVE, []),
        }
        mw._on_grouped_fit_completed(grouped_datasets, results)

        assert rendered, "grouped subplots were never rendered"
        drawn = rendered[-1]
        # Only one run's groups (2), not all four members.
        assert len(drawn) == 2
        source_runs = {mw._grouped_member_source_run(ds) for ds in drawn}
        assert source_runs == {42}

    def _grouped_state_stub(self):
        return {
            "composite_model": {"component_names": ["Exponential"], "operators": []},
            "param_roles": {"Lambda": "global"},
            "nuisance_params": ["N0"],
        }

    def _group_member(self, source_run: int, group: int) -> MuonDataset:
        return MuonDataset(
            np.array([0.0, 0.1]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            {"run_number": -((source_run * 1000) + group), "source_run_number": source_run},
            None,
        )

    def test_single_grouped_fit_does_not_surface_param_panel(self, mw, monkeypatch):
        # A single grouped fit (one source run) stays on the plot/fit view like an
        # ordinary single fit — it must not raise the fit-parameters dock.
        mw._data_browser.add_dataset(_dataset(42))
        mw._on_dataset_selected(42)
        mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
        mw._plot_workspace.set_active_view("groups")
        monkeypatch.setattr(
            mw._multi_group_fit_window, "get_grouped_state", self._grouped_state_stub
        )
        surfaced: list[str] = []
        monkeypatch.setattr(mw, "_show_panel", lambda key: surfaced.append(key))

        # Two groups, one source run (42).
        datasets = [self._group_member(42, 1), self._group_member(42, 2)]
        results = {-42001: (_result(), _CURVE, []), -42002: (_result(), _CURVE, [])}
        mw._on_grouped_fit_completed(datasets, results)

        assert "fit_parameters" not in surfaced
        # A single grouped fit records no series — it stores its result on the
        # dataset's grouped FitSlot like an ordinary single fit.
        assert mw._project_model.batches == {}
        assert mw._fit_parameters_panel._group_fit_results == {}
        rep = mw._project_model.representation(42, RepresentationType.TIME_GROUPS)
        assert rep is not None and rep.fit.provenance == "single"

    def test_batch_grouped_fit_surfaces_param_panel(self, mw, monkeypatch):
        # A multi-run batch grouped fit still surfaces the trend panel.
        for rn in (42, 43):
            mw._data_browser.add_dataset(_dataset(rn))
        mw._on_dataset_selected(42)
        mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
        mw._plot_workspace.set_active_view("groups")
        monkeypatch.setattr(
            mw._multi_group_fit_window, "get_grouped_state", self._grouped_state_stub
        )
        surfaced: list[str] = []
        monkeypatch.setattr(mw, "_show_panel", lambda key: surfaced.append(key))

        # One group each across two source runs (42, 43).
        datasets = [self._group_member(42, 1), self._group_member(43, 1)]
        results = {-42001: (_result(), _CURVE, []), -43001: (_result(), _CURVE, [])}
        mw._on_grouped_fit_completed(datasets, results)

        assert "fit_parameters" in surfaced

    def test_add_to_series_refreshes_trend_panel(self, mw, monkeypatch):
        """After Add-to-Series the trend panel shows the extended membership."""
        for rn in (10, 11):
            mw._data_browser.add_dataset(_dataset(rn))
        mw._plot_workspace.set_active_view("fb_asymmetry")
        model = {"component_names": ["Exponential"], "operators": []}
        monkeypatch.setattr(
            mw._fit_panel,
            "get_global_state",
            lambda: {
                "composite_model": model,
                "parameters": [{"name": "A", "type": "Local"}],
                "result_html": "",
            },
        )
        mw._on_global_fit_started()  # the fit panel's launch signal, as in production
        mw._on_global_fit_completed(
            {rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet()
        )
        series = next(iter(mw._project_model.batches.values()))

        # Single-fit run 12, then add it to the series.
        mw._data_browser.add_dataset(_dataset(12))
        mw._on_dataset_selected(12)
        monkeypatch.setattr(
            mw._fit_panel,
            "get_single_form_state",
            lambda: {"composite_model": model, "parameters": [], "result_html": ""},
        )
        mw._on_fit_completed(_result(), _CURVE, [])
        mw._add_single_fit_to_series(12, series.batch_id)
        mw._refresh_trend_panel()

        panel = mw._fit_parameters_panel
        gdata = panel._group_fit_results.get(series.batch_id)
        run_numbers = {r.run_number for r in (gdata.rows if gdata else [])}
        assert 12 in run_numbers


# ---------------------------------------------------------------------------
# Data-browser series highlighting
# ---------------------------------------------------------------------------


class TestDataBrowserHighlighting:
    """set_highlighted_runs tints series-member rows in the browser."""

    def test_set_highlighted_runs_stores_set(self, app):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        browser = DataBrowserPanel()
        browser.set_highlighted_runs({10, 11})
        assert browser._highlighted_runs == {10, 11}

    def test_clear_highlights_with_empty_set(self, app):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        browser = DataBrowserPanel()
        browser.set_highlighted_runs({5})
        browser.set_highlighted_runs(set())
        assert browser._highlighted_runs == set()

    def test_trend_series_selected_highlights_member_runs(self, mw, monkeypatch):
        """series_selection_changed → browser highlights the series' member runs."""
        for rn in (10, 11):
            mw._data_browser.add_dataset(_dataset(rn))
        mw._plot_workspace.set_active_view("fb_asymmetry")
        monkeypatch.setattr(
            mw._fit_panel,
            "get_global_state",
            lambda: {
                "composite_model": {"component_names": ["Exponential"], "operators": []},
                "parameters": [{"name": "A", "type": "Local"}],
                "result_html": "",
            },
        )
        mw._on_global_fit_started()  # the fit panel's launch signal, as in production
        mw._on_global_fit_completed(
            {rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet()
        )
        series = next(iter(mw._project_model.batches.values()))

        # Simulate the user clicking the series button.
        mw._on_trend_series_selected(series.batch_id)
        assert mw._data_browser._highlighted_runs == {10, 11}

    def test_unknown_series_clears_highlights(self, mw):
        """Selecting an unknown batch_id clears browser highlights."""
        mw._data_browser.set_highlighted_runs({99})
        mw._on_trend_series_selected("no-such-batch")
        assert mw._data_browser._highlighted_runs == set()

    def test_browser_clear_resets_highlighted_runs(self, app):
        """DataBrowserPanel.clear() removes stale series highlights (fix for review finding 4)."""
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        browser = DataBrowserPanel()
        browser.set_highlighted_runs({42, 43})
        browser.clear()
        # Stale highlights must not survive into the next project.
        assert browser._highlighted_runs == set()

    def test_initial_highlight_fires_without_user_click(self, mw, monkeypatch):
        """Browser highlights appear immediately after a fit, not only after a button click (fix for finding 5)."""
        for rn in (10, 11):
            mw._data_browser.add_dataset(_dataset(rn))
        mw._plot_workspace.set_active_view("fb_asymmetry")
        monkeypatch.setattr(
            mw._fit_panel,
            "get_global_state",
            lambda: {
                "composite_model": {"component_names": ["Exponential"], "operators": []},
                "parameters": [{"name": "A", "type": "Local"}],
                "result_html": "",
            },
        )
        # Reset any pre-existing highlights.
        mw._data_browser.set_highlighted_runs(set())

        mw._on_global_fit_started()  # the fit panel's launch signal, as in production
        mw._on_global_fit_completed(
            {rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet()
        )
        # The trend panel refresh must have emitted series_selection_changed
        # automatically, causing the browser to highlight members without a click.
        assert mw._data_browser._highlighted_runs == {10, 11}


# ---------------------------------------------------------------------------
# Regression: review findings
# ---------------------------------------------------------------------------


class TestReviewFindings:
    """Regression tests for bugs identified in the code review."""

    def test_build_series_rows_frequency_uses_spectra_cache(self, mw, monkeypatch):
        """Fix #3: _build_series_rows uses _frequency_spectra_by_run for FFT series."""
        # Populate a fake frequency spectrum with known field/temperature.
        freq_dataset = MuonDataset(
            np.array([0.0, 1.0]),
            np.array([0.1, 0.1]),
            np.array([0.01, 0.01]),
            {"run_number": 10, "field": 250.0, "temperature": 5.0, "run_label": "10"},
            None,
        )
        mw._frequency_spectra_by_run[10] = [freq_dataset]

        # Also add a time-domain dataset with different metadata so we can
        # detect which source was used.
        td_dataset = _dataset(10, field=100.0)  # field=100, not 250
        mw._data_browser.add_dataset(td_dataset)

        series = FitSeries(
            "batch-freq",
            RepresentationType.FREQ_FFT,
            member_run_numbers=[10],
            canonical_model={"component_names": ["Exponential"], "operators": []},
            results_by_run={
                10: {"success": True, "parameters": {"A": 0.2}, "uncertainties": {"A": 0.01}}
            },
        )

        rows = mw._build_series_rows(series)
        assert len(rows) == 1
        # Must have picked the frequency spectrum (field=250) not the time-domain
        # dataset (field=100).
        assert rows[0]["field"] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Visibility-gated highlight, rename, select, delete
# ---------------------------------------------------------------------------


def _setup_one_series(mw, monkeypatch, model="Exponential"):
    """Add two datasets, run a global fit, and return the resulting FitSeries."""
    for rn in (10, 11):
        mw._data_browser.add_dataset(_dataset(rn))
    mw._plot_workspace.set_active_view("fb_asymmetry")
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": {"component_names": [model], "operators": []},
            "parameters": [{"name": "A", "type": "Local"}],
            "result_html": "",
        },
    )
    mw._on_global_fit_started()  # the fit panel's launch signal, as in production
    mw._on_global_fit_completed({rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet())
    return next(iter(mw._project_model.batches.values()))


class TestVisibilityGatedHighlight:
    """Parameters dock visibility gates the FitSeries browser highlight."""

    def test_hiding_dock_clears_highlight(self, mw, monkeypatch):
        _setup_one_series(mw, monkeypatch)
        # Simulate the dock becoming visible — restores the highlight directly.
        mw._on_parameters_dock_visibility_changed(True)
        assert mw._data_browser._highlighted_runs != set()
        # Simulate hiding the dock.
        mw._on_parameters_dock_visibility_changed(False)
        assert mw._data_browser._highlighted_runs == set()

    def test_showing_dock_restores_highlight(self, mw, monkeypatch):
        _setup_one_series(mw, monkeypatch)
        mw._on_parameters_dock_visibility_changed(False)
        assert mw._data_browser._highlighted_runs == set()
        mw._on_parameters_dock_visibility_changed(True)
        assert mw._data_browser._highlighted_runs == {10, 11}

    def test_visibilitychanged_false_clears_regardless_of_series_selection(self, mw, monkeypatch):
        """The visibilityChanged(False) handler always clears, even if series_selection_changed fires."""
        _setup_one_series(mw, monkeypatch)
        mw._on_parameters_dock_visibility_changed(True)
        assert mw._data_browser._highlighted_runs != set()
        # Fire visibilityChanged(False) — must clear.
        mw._on_parameters_dock_visibility_changed(False)
        assert mw._data_browser._highlighted_runs == set()
        # Even a direct _on_trend_series_selected call after a hide doesn't change
        # the intent — the gate is driven by the visibilityChanged cycle.


def _twin_series(mw, first, model: str) -> FitSeries:
    """A second, group-less (Standalone) series over *first*'s runs and results,
    fitted with another model."""
    twin = FitSeries(
        "batch-twin",
        first.rep_type,
        member_kind="runs",
        member_run_numbers=list(first.member_run_numbers),
        canonical_model={"component_names": [model], "operators": []},
        results_by_run=dict(first.results_by_run),
    )
    mw._project_model.add_batch(twin)
    mw._refresh_trend_panel()
    return twin


class TestShortSeriesPillNames:
    """The host computes the short pill label and disambiguates collisions.

    A batch/global fit always records into a data group — an explicit binding,
    or an auto-minted one named after the run range ("Runs 10–11", D3) when
    the user never bound one — so ``_setup_one_series``'s series sits under a
    real (non-"Standalone") section header (item 1). Its own chip therefore
    reads ``<model> · <range>`` only; ``_twin_series`` deliberately adds its
    series with no ``group_id`` at all, landing it in the "Standalone"
    section, whose chip still carries the member run range (item 2).
    """

    def test_single_series_pill_is_just_the_model_under_its_group_header(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        panel = mw._fit_parameters_panel
        assert panel._group_fit_results[series.batch_id].short_name == "Exponential"
        assert panel._group_button_map[series.batch_id].text() == "Exponential"

    def test_standalone_series_pill_carries_the_model_and_the_member_range(self, mw, monkeypatch):
        first = _setup_one_series(mw, monkeypatch)
        second = _twin_series(mw, first, "Gaussian")

        short = {
            bid: group.short_name
            for bid, group in mw._fit_parameters_panel._group_fit_results.items()
        }
        # first: grouped, so just the model (no configured fit range here).
        assert short[first.batch_id] == "Exponential"
        # second: group-less, so the model plus its own member range.
        assert short[second.batch_id] == "Gaussian · 10–11"

    def test_renamed_series_keeps_its_label_as_the_pill(self, mw, monkeypatch):
        first = _setup_one_series(mw, monkeypatch)
        second = _twin_series(mw, first, "Gaussian")
        mw._on_series_rename_requested(first.batch_id, "Cooldown")

        panel = mw._fit_parameters_panel
        # The user's name is never shortened, and never disambiguated; the
        # group-less second series' short name does not depend on the first's
        # label (it was never a collision-driven suffix to begin with).
        assert panel._group_fit_results[first.batch_id].short_name == "Cooldown"
        assert panel._group_button_map[first.batch_id].text() == "Cooldown"
        assert panel._group_fit_results[second.batch_id].short_name == "Gaussian · 10–11"


class TestSeriesRenameAndLabel:
    """_on_series_rename_requested updates label and refreshes panel."""

    def test_rename_sets_label_and_refreshes(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        mw._on_series_rename_requested(series.batch_id, "Field sweep")
        assert mw._project_model.batch(series.batch_id).label == "Field sweep"
        # Panel button should now show the new label.
        panel = mw._fit_parameters_panel
        button = panel._group_button_map.get(series.batch_id)
        assert button is not None
        assert button.text() == "Field sweep"

    def test_rename_empty_string_reverts_to_fallback(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        mw._on_series_rename_requested(series.batch_id, "Field sweep")
        mw._on_series_rename_requested(series.batch_id, "")
        assert mw._project_model.batch(series.batch_id).label is None
        # Clearing the label reverts to the unified default (D10: "<model> ·
        # <fit range>", the window omitted when the recipe leaves it open), not
        # a bare positional "Series N": the pill drops to the model alone (its
        # auto-minted group's own section header already names the run range,
        # item 2) and the full default name moves to the tooltip.
        button = mw._fit_parameters_panel._group_button_map.get(series.batch_id)
        assert button is not None and button.text() == "Exponential"
        assert button.toolTip().startswith("Exponential\n")

    def test_add_to_series_chooser_shows_user_label(self, mw, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        series = _setup_one_series(mw, monkeypatch)
        mw._on_series_rename_requested(series.batch_id, "My named series")
        # Load a third dataset with a single fit, compatible model.
        from asymmetry.core.representation import RepresentationType

        mw._data_browser.add_dataset(_dataset(99))
        rep = mw._project_model.ensure_dataset(99).ensure(RepresentationType.TIME_FB_ASYMMETRY)
        rep.fit = FitSlot(
            model={"component_names": ["Exponential"], "operators": []},
            provenance="single",
        )
        # Create a second compatible series so the chooser dialog is triggered.
        from asymmetry.core.representation.series import FitSeries

        s2 = FitSeries(
            "batch-99",
            RepresentationType.TIME_FB_ASYMMETRY,
            member_kind="runs",
            member_run_numbers=[10],
            canonical_model={"component_names": ["Exponential"], "operators": []},
        )
        mw._project_model.add_batch(s2)

        captured_items: list = []
        monkeypatch.setattr(
            QInputDialog,
            "getItem",
            lambda _self, _title, _label, items, *_a, **_kw: (
                captured_items.extend(items) or (items[0], False)
            ),
        )
        mw._current_dataset = mw._data_browser.get_dataset(99)
        monkeypatch.setattr(
            mw, "_active_representation_type", lambda: RepresentationType.TIME_FB_ASYMMETRY
        )
        mw._on_add_single_fit_to_series_requested()

        assert any("My named series" in item for item in captured_items)


class TestSeriesSelectMembers:
    """_on_series_select_members_requested performs a true browser selection."""

    def test_select_members_selects_run_series(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        mw._on_series_select_members_requested(series.batch_id)
        selected = set(mw._data_browser._get_selected_run_numbers())
        assert selected == {10, 11}

    def test_select_members_does_not_change_highlight(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        mw._on_parameters_dock_visibility_changed(True)
        highlight_before = set(mw._data_browser._highlighted_runs)
        mw._on_series_select_members_requested(series.batch_id)
        assert mw._data_browser._highlighted_runs == highlight_before


class TestSeriesDelete:
    """_on_series_delete_requested removes series and clears highlight."""

    def test_delete_removes_batch_from_project(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        bid = series.batch_id
        mw._on_series_delete_requested(bid)
        assert mw._project_model.batch(bid) is None

    def test_delete_clears_browser_highlight(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        mw._on_parameters_dock_visibility_changed(True)
        mw._on_series_delete_requested(series.batch_id)
        assert mw._data_browser._highlighted_runs == set()

    def test_delete_removes_panel_button(self, mw, monkeypatch):
        series = _setup_one_series(mw, monkeypatch)
        bid = series.batch_id
        mw._on_series_delete_requested(bid)
        assert bid not in mw._fit_parameters_panel._group_button_map

    def test_delete_unknown_batch_is_noop(self, mw):
        # Should not raise.
        mw._on_series_delete_requested("no-such-batch")
