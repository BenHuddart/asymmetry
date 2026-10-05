"""Runs from vector and single-pair groupings in one project.

A run shows projections only when its own grouping declares them; overlaid runs
share the union of projections, each subplot holding only the runs that measure
it (a longitudinal run sits on ``P_z`` alone).
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_hex
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.mainwindow import MainWindow

pytestmark = [pytest.mark.gui]

_N_BINS = 64
_VECTOR_PROJECTIONS = [
    {"label": "P_x", "forward_group": 5, "backward_group": 6},
    {"label": "P_y", "forward_group": 3, "backward_group": 4},
    {"label": "P_z", "forward_group": 1, "backward_group": 2},
]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    QSettings().setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _dataset(run_number: int, *, vector: bool) -> MuonDataset:
    t = np.arange(_N_BINS) * 0.016
    n_det = 6 if vector else 2
    histograms = [
        Histogram(
            counts=1000.0 * np.exp(-t / 2.2) * (1.0 + 0.1 * (d + 1) * np.cos(3.0 * t)),
            bin_width=0.016,
        )
        for d in range(n_det)
    ]
    grouping: dict = {
        "groups": {gid: [gid] for gid in range(1, n_det + 1)},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": _N_BINS - 1,
        "bunching_factor": 1,
        "deadtime_correction": False,
    }
    if vector:
        grouping["projections"] = [dict(p) for p in _VECTOR_PROJECTIONS]
        grouping["vector_axis"] = "P_z"
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number},
        grouping=grouping,
    )
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _load(mainwindow: MainWindow, *datasets: MuonDataset) -> None:
    with mainwindow._data_browser.batch_updates():
        for dataset in datasets:
            mainwindow._data_browser.add_dataset(dataset)


def _select_chips(mainwindow: MainWindow, labels: list[str]) -> None:
    bar = mainwindow._plot_panel._projection_bar
    bar.set_selected(labels)
    bar.selection_changed.emit(bar.selected_labels())


def test_single_pair_payload_clears_a_runs_vector_projections(mainwindow: MainWindow) -> None:
    dataset = _dataset(301, vector=True)
    _load(mainwindow, dataset)
    payload = mainwindow._extract_grouping_overrides(dataset)
    payload["groups"] = {1: [1], 2: [2]}
    payload.pop("projections")

    applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

    assert applied
    assert "projections" not in dataset.run.grouping
    assert "vector_axis" not in dataset.run.grouping


def test_single_pair_run_shows_one_plain_pane(mainwindow: MainWindow) -> None:
    vector, plain = _dataset(302, vector=True), _dataset(303, vector=False)
    _load(mainwindow, vector, plain)
    mainwindow._data_browser.select_runs([302])
    _select_chips(mainwindow, ["P_x", "P_y", "P_z"])

    mainwindow._data_browser.select_runs([303])

    panel = mainwindow._plot_panel
    assert panel._projection_bar.isHidden()
    assert len(panel._figure.axes) == 1
    assert mainwindow._current_single_fit_projection() is None


def test_mixed_overlay_places_each_run_on_its_own_projections(mainwindow: MainWindow) -> None:
    vector, plain = _dataset(304, vector=True), _dataset(305, vector=False)
    _load(mainwindow, vector, plain)
    panel = mainwindow._plot_panel
    panel.set_overlay_enabled(True, emit_signal=True)
    mainwindow._data_browser.select_runs([304, 305])

    assert list(panel._projection_bar._chips) == ["P_x", "P_y", "P_z"]

    _select_chips(mainwindow, ["P_x", "P_y", "P_z"])

    members = {
        label: [ds.run_number for ds in datasets]
        for label, datasets in panel._vector_subplot_datasets.items()
    }
    assert members == {"P_x": [304], "P_y": [304], "P_z": [304, 305]}
    axes = panel._subplot_axes_by_polarization
    vector_colors = {
        label: [to_hex(line.get_color()) for line in axes[label].get_legend().get_lines()]
        for label in ("P_x", "P_z")
    }
    # The vector run keeps its colour on every subplot it shares with the plain run.
    assert vector_colors["P_x"][0] == vector_colors["P_z"][0]
    assert len(vector_colors["P_z"]) == 2

    _select_chips(mainwindow, ["P_x"])
    assert [ds.run_number for ds in panel._current_datasets] == [304]

    _select_chips(mainwindow, ["P_z"])
    assert sorted(ds.run_number for ds in panel._current_datasets) == [304, 305]
