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
    # Default label carries the model and the run range, not a bare "Series N".
    assert "1276" in series.label and "1289" in series.label


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
