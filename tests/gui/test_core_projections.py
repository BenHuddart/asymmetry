"""The core projection reducer gives the GUI's stacked-subplot projections, period by period."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.projections import reduce_run_projections
from asymmetry.core.transform.rotating_frame import RotatingFrame
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.mainwindow import MainWindow
from tests.core.vector_synthetic import synthetic_vector_run


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    QSettings().setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _turning(t):
    angle = -2.0 * math.pi * 1.5 * t
    return np.stack([np.cos(angle), np.sin(angle), np.exp(-t)])


def _still(t):
    return np.stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])


@pytest.mark.parametrize(("mode", "index"), [(PeriodMode.RED, 0), (PeriodMode.GREEN, 1)])
def test_core_projections_match_the_gui_vector_subplots(mainwindow, mode, index):
    run = synthetic_vector_run([_turning, _still], alphas=(1.1, 0.9, 1.2))
    run.grouping["period_mode"] = str(mode)
    dataset = MuonDataset(
        time=np.zeros(1),
        asymmetry=np.zeros(1),
        error=np.ones(1),
        metadata=dict(run.metadata),
        run=run,
    )
    gui = mainwindow._build_vector_axis_datasets([dataset])
    core = reduce_run_projections(run, index)
    for label, reduced in core.items():
        shown = gui[label][0]
        np.testing.assert_allclose(shown.time, reduced.time)
        np.testing.assert_allclose(shown.asymmetry, reduced.asymmetry)
        np.testing.assert_allclose(shown.error, reduced.error)


def test_a_new_project_forgets_the_previous_projects_rotating_frames(mainwindow):
    mainwindow._project_model.rotating_frames[5] = RotatingFrame.typed_frequency(1.5, 1)
    mainwindow._clear_all_state()
    assert mainwindow._project_model.rotating_frames == {}


def test_applied_exact_timing_uses_selected_reference_after_spectator_and_manual_edits(mainwindow):
    from asymmetry.core.data.dataset import Histogram
    from tests.gui.test_mainwindow_additional import _make_dataset

    dataset = _make_dataset(7398, with_grouping=False)
    dataset.run.histograms = [
        Histogram(np.ones(4) * level, 0.1, 1, t0_time_us=exact)
        for level, exact in ((10.0, 0.12), (20.0, 0.14))
    ]
    payload = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 1,
        "first_good_bin": 0,
        "last_good_bin": 3,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "background_correction": False,
    }
    assert mainwindow._apply_grouping_settings_to_dataset(dataset, payload)[0]
    assert dataset.run.grouping["t0_time_us"] == pytest.approx(0.12)
    axis = dataset.time.copy()
    dataset.run.histograms.insert(0, Histogram(np.ones(4), 9.0, 1, t0_time_us=100.0))
    payload["groups"] = {1: [2], 2: [3]}
    assert mainwindow._apply_grouping_settings_to_dataset(dataset, payload)[0]
    assert dataset.run.grouping["t0_time_us"] == pytest.approx(0.12)
    np.testing.assert_array_equal(dataset.time, axis)
    payload.update(t0_bin=3)
    assert mainwindow._apply_grouping_settings_to_dataset(dataset, payload)[0]
    assert dataset.run.grouping["t0_time_us"] == pytest.approx(0.32)
    np.testing.assert_allclose(dataset.time, [axis[-1] - 0.2], atol=1e-15)
    for histogram in dataset.run.histograms[1:]:
        histogram.t0_time_us = None
    payload["t0_bin"] = 1
    assert mainwindow._apply_grouping_settings_to_dataset(dataset, payload)[0]
    assert "t0_time_us" not in dataset.run.grouping
