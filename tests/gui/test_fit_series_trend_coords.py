"""Regression tests for batch FitSeries per-run T/B propagation (trend X-axis).

A batch over runs spanning a range of temperatures/fields must carry each
member run's real temperature/field into the recorded ``FitSeries`` so the
parameter-trend X = T(K)/B(G) plots every point at its real coordinate — even
after the dataset leaves the data browser or the project is saved and reopened.
Previously the trend row defaulted a missing coordinate to ``0.0``, collapsing
every point to T = 0 (the headline bug from overnight GUI testing).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.series import FitSeries
from asymmetry.gui.mainwindow import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def win(qapp: QApplication) -> MainWindow:
    w = MainWindow()
    return w


class _StubFitResult:
    """Minimal duck-typed fit result for ``fit_result_summary``."""

    def __init__(self, params: dict[str, float]) -> None:
        self.parameters = ParameterSet([Parameter(name=n, value=v) for n, v in params.items()])
        self.uncertainties = {n: 0.01 for n in params}
        self.minos_errors = None
        self.success = True
        self.chi_squared = 1.0
        self.reduced_chi_squared = 1.0


def _add_dataset(win: MainWindow, run_number: int, temperature: float, field: float) -> None:
    ds = MuonDataset(
        time=np.linspace(0.0, 16.0, 8),
        asymmetry=np.zeros(8),
        error=np.ones(8),
        metadata={"run_number": run_number, "temperature": temperature, "field": field},
    )
    win._data_browser.add_dataset(ds)


def _run_batch(win: MainWindow, coords: dict[int, tuple[float, float]]) -> str:
    """Drive ``_record_global_fit_batch`` over *coords* (run -> (T, B))."""
    for run, (temp, field) in coords.items():
        _add_dataset(win, run, temp, field)

    # Stub the fit-panel global state the record path reads.
    win._fit_panel.get_global_state = lambda: {  # type: ignore[method-assign]
        "parameters": [{"name": "sigma", "type": "local"}],
        "composite_model": {"component_names": ["Gaussian", "Constant"], "operators": ["+"]},
    }
    win._fit_panel.batch_fit_range_text = lambda: "0-16"  # type: ignore[method-assign]
    win._active_representation_type = (  # type: ignore[method-assign]
        lambda: RepresentationType.TIME_FB_ASYMMETRY
    )

    payloads = {
        run: (_StubFitResult({"sigma": 1.0}), (np.zeros(2), np.zeros(2)), []) for run in coords
    }
    batch_id = win._record_global_fit_batch(payloads, None)
    assert batch_id is not None
    return batch_id


def test_batch_series_summary_carries_per_run_temperature_field(win: MainWindow) -> None:
    coords = {1277: (10.0, 400.0), 1280: (70.0, 400.0), 1276: (125.0, 400.0)}
    batch_id = _run_batch(win, coords)

    series = win._project_model.batch(batch_id)
    assert series is not None
    for run, (temp, field) in coords.items():
        summary = series.results_by_run[run]
        assert summary["temperature"] == pytest.approx(temp)
        assert summary["field"] == pytest.approx(field)


def test_trend_rows_keep_temperature_after_browser_cleared(win: MainWindow) -> None:
    """The stamped coordinate survives the dataset leaving the browser (reload/stale)."""
    coords = {1277: (10.0, 400.0), 1280: (70.0, 400.0)}
    batch_id = _run_batch(win, coords)
    series = win._project_model.batch(batch_id)

    # Mimic a project reopened with these runs no longer loaded in the browser.
    win._data_browser._datasets.clear()

    rows = {row["run_number"]: row for row in win._build_series_rows(series)}
    for run, (temp, field) in coords.items():
        assert rows[run]["temperature"] == pytest.approx(temp)
        assert rows[run]["field"] == pytest.approx(field)
        # The bug planted these at 0.0; assert that never happens.
        assert rows[run]["temperature"] != 0.0


def test_batch_series_has_informative_default_label(win: MainWindow) -> None:
    coords = {1276: (125.0, 400.0), 1289: (10.0, 400.0)}
    batch_id = _run_batch(win, coords)
    series = win._project_model.batch(batch_id)
    # ``label`` is reserved for user renames; the informative default (model +
    # run range) is rendered on demand as the display fallback.
    assert series.label is None
    default = win._series_fallback_name(series)
    assert "1276" in default and "1289" in default


def test_missing_metadata_point_is_off_axis_not_zero(win: MainWindow) -> None:
    """A run with no recorded T/B and no backing dataset is NaN (off-axis), not 0."""
    series = FitSeries(
        "batch-stale",
        RepresentationType.TIME_FB_ASYMMETRY,
        member_run_numbers=[9999],
        order_key="temperature",
        results_by_run={
            9999: {
                "success": True,
                "parameters": {"sigma": 1.0},
                "uncertainties": {"sigma": 0.05},
            }
        },
    )
    win._data_browser._datasets.clear()
    (row,) = win._build_series_rows(series)
    assert math.isnan(row["temperature"])
    assert math.isnan(row["field"])


def _group_series(
    batch_id: str,
    runs: list[int],
    n_groups: int,
    roles: dict[str, str],
    freq_by_run: dict[int, float],
) -> FitSeries:
    """Build a TIME_GROUPS FitSeries: each (run, group) member shares the run's
    global physics value, replicated across that run's groups."""
    member_run_numbers: list[int] = []
    member_source_run: dict[int, int] = {}
    results_by_run: dict[int, dict] = {}
    for run in runs:
        for g in range(1, n_groups + 1):
            key = -(run * 1000 + g)
            member_run_numbers.append(key)
            member_source_run[key] = run
            results_by_run[key] = {
                "success": True,
                "parameters": {"freq": freq_by_run[run], "amp": 0.2 + 0.01 * g},
                "uncertainties": {"freq": 0.01, "amp": 0.001},
                "temperature": 10.0,
                "field": 400.0,
            }
    return FitSeries(
        batch_id,
        RepresentationType.TIME_GROUPS,
        member_kind="groups",
        member_run_numbers=member_run_numbers,
        member_source_run=member_source_run,
        param_roles=roles,
        nuisance_params=[],
        results_by_run=results_by_run,
    )


def test_group_series_collapses_to_one_row_per_run_when_physics_global(win: MainWindow) -> None:
    # Per-run angle values ride on the source-run dataset metadata.
    for run, angle in ((1276, "0"), (1280, "30")):
        win._data_browser.add_dataset(
            MuonDataset(
                time=np.linspace(0.0, 16.0, 8),
                asymmetry=np.zeros(8),
                error=np.ones(8),
                metadata={
                    "run_number": run,
                    "temperature": 10.0,
                    "field": 400.0,
                    "custom_fields": {"angle": angle},
                },
            )
        )
    series = _group_series("batch-g", [1276, 1280], 2, {"freq": "global"}, {1276: 5.0, 1280: 6.0})

    rows = win._build_series_rows(series)
    # Two runs × two groups, but a single global physics value per run → 2 rows.
    assert len(rows) == 2
    by_run = {row["run_number"]: row for row in rows}
    assert set(by_run) == {1276, 1280}
    assert by_run[1276]["values"]["freq"] == pytest.approx(5.0)
    assert by_run[1280]["values"]["freq"] == pytest.approx(6.0)
    # The source-run angle reaches the collapsed row (drives the Angle trend axis).
    assert by_run[1276]["custom_values"]["angle"] == "0"
    assert by_run[1280]["custom_values"]["angle"] == "30"


def test_collapsed_group_row_drops_per_group_nuisance_values(win: MainWindow) -> None:
    # A collapsed row represents the whole run via one group member, so a per-group
    # nuisance ("amp") must NOT survive — otherwise it is offered as a trend Y that
    # silently plots only the representative group's value per run.
    series = _group_series("batch-g3", [1276, 1280], 2, {"freq": "global"}, {1276: 5.0, 1280: 6.0})
    series.nuisance_params = ["amp"]
    rows = win._build_series_rows(series)
    assert len(rows) == 2
    for row in rows:
        assert "freq" in row["values"]
        assert "amp" not in row["values"]
        assert "amp" not in row["errors"]


def test_group_series_collapses_to_one_row_per_run_even_with_local_physics(
    win: MainWindow,
) -> None:
    # A "local" role means per-RUN (independent across runs); the model-function
    # parameter is still shared across a run's detector groups (only the nuisance
    # block is per-group). So the series collapses to one trend point per source run
    # just like the all-global case — the parameters tab shows model params per run.
    series = _group_series(
        "batch-g2", [1276, 1280], 2, {"freq": "global", "amp": "local"}, {1276: 5.0, 1280: 6.0}
    )
    rows = win._build_series_rows(series)
    assert len(rows) == 2
    by_run = {row["run_number"]: row for row in rows}
    assert set(by_run) == {1276, 1280}
    assert by_run[1276]["values"]["freq"] == pytest.approx(5.0)
    assert by_run[1280]["values"]["freq"] == pytest.approx(6.0)


def test_fit_series_shared_parameters_reads_global_roles_from_results() -> None:
    """FitSeries.shared_parameters surfaces the global-role params' value + error."""
    series = _group_series(
        "batch-sp", [1276, 1280], 2, {"freq": "global", "amp": "local"}, {1276: 5.0, 1280: 6.0}
    )
    shared = series.shared_parameters()
    assert set(shared) == {"freq"}  # only the global-role param
    assert shared["freq"]["value"] == pytest.approx(5.0)  # first member's shared value
    assert shared["freq"]["error"] == pytest.approx(0.01)
    # A series with no global role has no shared parameters.
    assert (
        _group_series("batch-b", [1276], 2, {"freq": "local"}, {1276: 5.0}).shared_parameters()
        == {}
    )


# ── Phase 7: DataGroup <-> FitSeries linking (D1, README §6 Option B) ──────


def test_batch_launched_from_group_stamps_source_group_id(win: MainWindow) -> None:
    """A batch whose members are exactly one data group's members gets provenance."""
    coords = {1277: (10.0, 400.0), 1280: (70.0, 400.0)}
    for run, (temp, field) in coords.items():
        _add_dataset(win, run, temp, field)
    gid = win._data_browser.create_data_group(list(coords), name="T scan")

    batch_id = _run_batch(win, coords)
    series = win._project_model.batch(batch_id)
    assert series.source_group_id == gid


def test_adhoc_batch_spanning_two_groups_leaves_source_group_id_none(win: MainWindow) -> None:
    """A batch whose members straddle two different groups has no single provenance."""
    coords = {1277: (10.0, 400.0), 1280: (70.0, 400.0), 1281: (90.0, 400.0), 1282: (95.0, 400.0)}
    for run, (temp, field) in coords.items():
        _add_dataset(win, run, temp, field)
    win._data_browser.create_data_group([1277, 1281], name="grp-a")
    win._data_browser.create_data_group([1280, 1282], name="grp-b")

    # Batch over one member from each group — no single group covers this set.
    batch_id = _run_batch(win, {1277: coords[1277], 1280: coords[1280]})
    series = win._project_model.batch(batch_id)
    assert series.source_group_id is None


def test_fit_this_group_prefills_batch_regardless_of_visibility(win: MainWindow) -> None:
    """F9 regression: a group hidden by a column filter/sort still fits its real members.

    ``_on_fit_group_requested`` must build the batch dataset list from the
    group's stored member run numbers, not the browser's live/visible table
    selection — the trap that let a filter silently drop invisible group runs
    from a batch.
    """
    coords = {2961: (10.0, 60.0), 2962: (10.0, 60.0), 2963: (10.0, 60.0)}
    for run, (temp, field) in coords.items():
        _add_dataset(win, run, temp, field)
    gid = win._data_browser.create_data_group(list(coords), name="B = 60 G")

    # Simulate F9: a column filter hides every row (including this group's),
    # and nothing is selected in the live table.
    win._data_browser._column_filters = {0: {"nonexistent-value"}}
    win._data_browser._apply_row_visibility()
    assert win._data_browser.get_selected_datasets() == []

    win._on_fit_group_requested(gid)

    fed_runs = sorted(int(ds.run_number) for ds in win._fit_panel._all_datasets)
    assert fed_runs == sorted(coords)


def test_show_series_from_group_filters_trend_panel(win: MainWindow) -> None:
    coords_a = {1277: (10.0, 400.0), 1280: (70.0, 400.0)}
    coords_b = {1281: (10.0, 500.0), 1282: (70.0, 500.0)}
    for run, (temp, field) in {**coords_a, **coords_b}.items():
        _add_dataset(win, run, temp, field)
    gid_a = win._data_browser.create_data_group(list(coords_a), name="B = 400 G")
    win._data_browser.create_data_group(list(coords_b), name="B = 500 G")

    batch_a = _run_batch(win, coords_a)
    batch_b = _run_batch(win, coords_b)
    assert win._project_model.batch(batch_a).source_group_id == gid_a

    assert win._project_model.series_for_group(gid_a) == [win._project_model.batch(batch_a)]
    win._on_show_group_series_requested(gid_a)
    shown_ids = set(win._fit_parameters_panel._group_fit_results)
    assert batch_a in shown_ids
    assert batch_b not in shown_ids


def test_saved_project_carries_data_groups_and_reload_relinks_provenance(win: MainWindow) -> None:
    """Full save→reload: the browser group and the series it produced both survive."""
    coords = {1277: (10.0, 400.0), 1280: (70.0, 400.0)}
    for run, (temp, field) in coords.items():
        _add_dataset(win, run, temp, field)
    gid = win._data_browser.create_data_group(list(coords), name="T scan")
    batch_id = _run_batch(win, coords)
    assert win._project_model.batch(batch_id).source_group_id == gid

    state = win.collect_project_state()
    saved_group_ids = {g["group_id"] for g in state["data_groups"]}
    assert gid in saved_group_ids

    win._data_browser.restore_state(state["browser_state"])
    win._restore_frequency_representations(state)
    win._sync_data_groups_to_project_model()

    assert win._project_model.data_group(gid) is not None
    reloaded_series = win._project_model.batch(batch_id)
    assert reloaded_series is not None
    assert reloaded_series.source_group_id == gid
    assert win._project_model.series_for_group(gid) == [reloaded_series]


def test_core_only_data_group_survives_reload_without_browser_state_twin(
    win: MainWindow,
) -> None:
    """A group written only via the core API (no browser_state.data_groups twin) is not
    silently dropped on load.

    Regression: the sync used to run unconditionally right after
    ProjectModel.from_project_state parsed the top-level ``data_groups`` block,
    overwriting it with whatever the browser's legacy browser_state.data_groups
    block restored — which is empty/absent for a project authored purely against
    the core API (e.g. a script using ProjectModel.add_data_group directly).
    """
    coords = {1277: (10.0, 400.0), 1280: (70.0, 400.0)}
    for run, (temp, field) in coords.items():
        _add_dataset(win, run, temp, field)

    state = {
        "browser_state": {},  # no legacy data_groups entry
        "data_groups": [
            {
                "group_id": "core-only-grp",
                "name": "Core-only group",
                "member_run_numbers": [1277, 1280],
                "order_key": "run",
            }
        ],
    }
    win._data_browser.restore_state(state["browser_state"])
    win._restore_frequency_representations(state)
    win._seed_browser_groups_from_project_model()
    win._sync_data_groups_to_project_model()

    assert win._data_browser.get_group_name("core-only-grp") == "Core-only group"
    assert sorted(win._data_browser.get_group_member_run_numbers("core-only-grp")) == [
        1277,
        1280,
    ]
    restored_group = win._project_model.data_group("core-only-grp")
    assert restored_group is not None
    assert sorted(restored_group.member_run_numbers) == [1277, 1280]
