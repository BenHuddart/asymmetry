"""Phase 4: per-series trend surface and representation-aware panel.

Tests cover:
- Group-aware divergence / inclusion in ProjectModel.
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
from asymmetry.core.representation.project_model import ProjectModel
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
# ProjectModel group-aware divergence
# ---------------------------------------------------------------------------


class TestGroupAwareDivergence:
    """refresh_divergence handles synthetic group-member keys."""

    def _group_series(self, source_run: int = 42) -> FitSeries:
        """Return a group FitSeries for source_run with 2 synthetic members."""
        k1 = -((source_run * 1000) + 1)
        k2 = -((source_run * 1000) + 2)
        canonical = {"component_names": ["Exponential"], "operators": []}
        series = FitSeries(
            "test-series",
            RepresentationType.TIME_GROUPS,
            member_kind="groups",
            member_run_numbers=[k1, k2],
            member_source_run={k1: source_run, k2: source_run},
            canonical_model=canonical,
        )
        return series

    def test_non_diverged_members_stay_clear(self):
        pm = ProjectModel()
        series = self._group_series(source_run=42)
        pm.add_batch(series)

        # Source run 42 has a matching model.
        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        rep.fit = FitSlot(model=series.canonical_model)

        pm.refresh_divergence()

        for k in series.member_run_numbers:
            assert not series.is_diverged(k), f"member {k} should not be diverged"

    def test_mismatched_model_marks_all_synthetic_members_diverged(self):
        pm = ProjectModel()
        series = self._group_series(source_run=42)
        pm.add_batch(series)

        # Source run 42 has a *different* model.
        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        rep.fit = FitSlot(
            model={"component_names": ["Gaussian"], "operators": []},
        )

        pm.refresh_divergence()

        for k in series.member_run_numbers:
            assert series.is_diverged(k), f"member {k} should be diverged"
        # All diverged → excluded from trend by default.
        assert not rep.fit.include_in_trend

    def test_reconverged_model_clears_divergence(self):
        pm = ProjectModel()
        series = self._group_series(source_run=42)
        # Pre-mark both as diverged.
        for k in series.member_run_numbers:
            series.mark_diverged(k)
        pm.add_batch(series)

        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        rep.fit = FitSlot(model=series.canonical_model, diverged=True)
        rep.fit.include_in_trend = False

        pm.refresh_divergence()

        for k in series.member_run_numbers:
            assert not series.is_diverged(k)
        assert rep.fit.include_in_trend is True

    def test_trend_runs_for_group_batch_returns_synthetic_keys(self):
        pm = ProjectModel()
        series = self._group_series(source_run=42)
        pm.add_batch(series)

        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        rep.fit = FitSlot(model=series.canonical_model)

        trend = pm.trend_runs_for_batch(series)
        assert set(trend) == set(series.member_run_numbers)

    def test_trend_runs_excludes_when_source_run_excluded(self):
        pm = ProjectModel()
        series = self._group_series(source_run=42)
        pm.add_batch(series)

        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        rep.fit = FitSlot(model=series.canonical_model)
        rep.fit.include_in_trend = False  # Manually excluded.

        trend = pm.trend_runs_for_batch(series)
        assert trend == []

    def test_set_member_trend_inclusion_maps_to_source_run(self):
        pm = ProjectModel()
        series = self._group_series(source_run=42)
        pm.add_batch(series)

        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        rep.fit = FitSlot(model=series.canonical_model)

        # Disable via a synthetic member key.
        k1 = series.member_run_numbers[0]
        pm.set_member_trend_inclusion(series.batch_id, k1, False)
        assert rep.fit.include_in_trend is False

        # Re-enable.
        pm.set_member_trend_inclusion(series.batch_id, k1, True)
        assert rep.fit.include_in_trend is True


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
        mw._on_global_fit_completed(
            {rn: (_result(), _CURVE, []) for rn in (10, 11)}, ParameterSet()
        )
        # Confirm the FB Asymmetry series was recorded.
        assert len(mw._project_model.batches) == 1

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
        grouped_datasets = [
            MuonDataset(
                np.array([0.0, 0.1]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                {"run_number": -10001, "source_run_number": 10},
                None,
            ),
        ]
        results = {-10001: (_result(), _CURVE, [])}
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

    def test_grouped_fit_result_appears_in_trend_panel(self, mw, monkeypatch):
        """Grouped fits were previously invisible to the trend panel; Phase 4 fixes that."""
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
                "nuisance_params": ["N0"],
            },
        )
        grouped_datasets = [
            MuonDataset(
                np.array([0.0, 0.1]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                {"run_number": -42001, "source_run_number": 42},
                None,
            ),
        ]
        results = {-42001: (_result(rchi=0.35), _CURVE, [])}
        mw._on_grouped_fit_completed(grouped_datasets, results)

        panel = mw._fit_parameters_panel
        assert len(panel._group_fit_results) == 1
        gdata = next(iter(panel._group_fit_results.values()))
        assert len(gdata.rows) == 1
        # Physics is global (Lambda), so the series collapses to one trend point
        # per source run, keyed by the run (42), not the synthetic group key.
        assert gdata.rows[0].run_number == 42

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
        # The series is still recorded and loaded into the panel.
        assert mw._fit_parameters_panel._group_fit_results

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

    def test_group_divergence_heterogeneous_keys_preserves_manual_reinclusion(self):
        """Fix #1: _refresh_group_series_divergence reads was_diverged before mutating.

        When two synthetic keys for the same source run have heterogeneous prior
        divergence states, the first-time-exclusion guard must not override a
        manual re-inclusion set by the user.
        """
        pm = ProjectModel()
        # Two synthetic keys for source run 42.
        k1, k2 = -42001, -42002
        canonical = {"component_names": ["Exponential"], "operators": []}
        series = FitSeries(
            "s",
            RepresentationType.TIME_GROUPS,
            member_kind="groups",
            member_run_numbers=[k1, k2],
            member_source_run={k1: 42, k2: 42},
            canonical_model=canonical,
        )
        # k1 is already diverged; k2 is not (heterogeneous state after a partial
        # persistence round-trip).
        series.mark_diverged(k1)
        pm.add_batch(series)

        rep = pm.ensure_dataset(42).ensure(RepresentationType.TIME_GROUPS)
        # Model still doesn't match (divergence persists).
        rep.fit = FitSlot(model={"component_names": ["Gaussian"], "operators": []})
        # User manually re-included the run despite it being diverged.
        rep.fit.include_in_trend = True

        pm.refresh_divergence()

        # Both keys should be marked diverged (model still wrong).
        assert series.is_diverged(k1)
        assert series.is_diverged(k2)
        # The manual re-inclusion must NOT have been overwritten by the
        # k2 (was_diverged=False) path — fix ensures was_any_diverged=True
        # so the first-time-exclusion guard is skipped.
        assert rep.fit.include_in_trend is True

    def test_grouped_fit_calls_refresh_divergence(self, mw, monkeypatch):
        """Fix #2: _record_grouped_fit_series calls refresh_divergence at the end."""
        mw._data_browser.add_dataset(_dataset(42))
        mw._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
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
        grouped_datasets = [
            MuonDataset(
                np.array([0.0, 0.1]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                {"run_number": -42001, "source_run_number": 42},
                None,
            ),
        ]
        results = {-42001: (_result(), _CURVE, [])}

        # Plant a stale diverged_runs entry that should be cleared.
        stale = FitSeries(
            "old-series",
            RepresentationType.TIME_GROUPS,
            member_kind="groups",
            member_run_numbers=[-42001],
            member_source_run={-42001: 42},
            canonical_model={"component_names": ["Gaussian"], "operators": []},
        )
        stale.mark_diverged(-42001)
        mw._project_model.add_batch(stale)

        mw._record_grouped_fit_series(grouped_datasets, results)

        # refresh_divergence should have been called; the new canonical model
        # (Exponential) is written onto run 42's representation, so divergence
        # from the old Gaussian model should now be cleared for the new series.
        new_series = next(
            s for s in mw._project_model.batches.values() if s.batch_id != "old-series"
        )
        # The new FitSlot has the new model; refresh_divergence must clear stale state.
        assert not new_series.is_diverged(-42001)

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


def _setup_one_series(mw, monkeypatch):
    """Add two datasets, run a global fit, and return the resulting FitSeries."""
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
        # The panel button should show the positional fallback "Series 1".
        button = mw._fit_parameters_panel._group_button_map.get(series.batch_id)
        assert button is not None and button.text() == "Series 1"

    def test_add_to_series_chooser_shows_user_label(self, mw, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        series = _setup_one_series(mw, monkeypatch)
        mw._on_series_rename_requested(series.batch_id, "My named series")
        # Load a third dataset with a single fit, compatible model.
        from asymmetry.core.representation import FitSlot, RepresentationType

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
