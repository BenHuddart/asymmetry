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
