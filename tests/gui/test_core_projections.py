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
    np.testing.assert_allclose(dataset.time, axis - 0.2, atol=1e-15)
    for histogram in dataset.run.histograms[1:]:
        histogram.t0_time_us = None
    payload["t0_bin"] = 1
    assert mainwindow._apply_grouping_settings_to_dataset(dataset, payload)[0]
    assert "t0_time_us" not in dataset.run.grouping


def test_shared_detector_rotation_reaches_display_and_single_fit_input(mainwindow):
    from asymmetry.core.transform.rotating_frame import Provenance, rotate_transverse
    from tests.core.test_projection_noise import small_run
    from tests.gui.test_rotating_frame_projection import _rotate, _show

    run = small_run()
    dataset = MuonDataset(np.zeros(1), np.zeros(1), np.ones(1), run=run)
    frame = RotatingFrame.typed_frequency(0.61, 1).with_values(
        Provenance.TYPED, sense=1, rf_phase_deg=23.0, gain=1.7
    )
    mainwindow._project_model.rotating_frames[run.run_number] = frame
    panel = _show(mainwindow, dataset)
    _rotate(panel)
    panel.set_fit_target_projection("P′_y")
    expected = rotate_transverse(*reduce_run_projections(run, 0).values(), frame, weights=(1.0,))[1]
    shown = panel._vector_subplot_datasets["P′_y"][0]
    np.testing.assert_allclose(shown.error, expected.error)
    np.testing.assert_allclose(mainwindow._fit_panel.single_dataset().error, expected.error)
    assert shown.counting_noise is not None
    assert shown.fit_record().counting_noise is None


def test_cohort_snapshot_and_rotation_use_cancellable_scientific_worker(
    mainwindow, monkeypatch, qapp
):
    from PySide6.QtCore import QThread

    import asymmetry.core.transform.projections as core
    import asymmetry.gui.utils.reduction_cache as cache
    import asymmetry.gui.utils.vector_projections as adapter
    from asymmetry.core.transform.rotating_frame import FrameSnapshot
    from tests.core.test_projection_noise import small_run

    run = small_run()
    dataset = MuonDataset(np.zeros(1), np.zeros(1), np.ones(1), run=run)
    original = core.reduce_selected_projections
    checks = []

    def checked(*args, **kwargs):
        assert QThread.currentThread() != qapp.thread()
        assert kwargs["cancel_check"] is not None
        kwargs["cancel_check"]()
        checks.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "_INLINE_INPUT_BYTES", 0)
    monkeypatch.setattr(core, "reduce_selected_projections", checked)
    monkeypatch.setattr(adapter, "reduce_selected_projections", checked)
    mainwindow._build_vector_axis_datasets([dataset])
    snapshot = FrameSnapshot(RotatingFrame.typed_frequency(0.61, 1), (1.0,), "owned")
    rotated = mainwindow._rotated_curves(dataset, snapshot)
    assert len(checks) == 2
    assert rotated["P′_x"].counting_noise is not None


def test_authored_dataset_conditions_survive_core_vector_preparation(mainwindow):
    from asymmetry.core.transform.rotating_frame import FrameSnapshot
    from tests.core.test_projection_noise import small_run

    run = small_run()
    run.metadata.update(field=2.0, temperature=8.0, title="acquired")
    dataset = MuonDataset(
        np.zeros(1),
        np.zeros(1),
        np.ones(1),
        run=run,
        metadata={
            "field": 50.0,
            "temperature": 20.0,
            "title": "authored",
            "projection": "old-label",
            "grouped_alignment": {"old": True},
        },
    )
    cohorts = mainwindow._build_vector_axis_datasets([dataset])
    rotated = mainwindow._rotated_curves(
        dataset, FrameSnapshot(RotatingFrame.typed_frequency(0.61, 1), (1.0,), "owned")
    )
    for label, curve in [(label, curves[0]) for label, curves in cohorts.items()] + list(
        rotated.items()
    ):
        assert curve.field == 50.0
        assert curve.temperature == 20.0
        assert curve.metadata["title"] == "authored"
        assert curve.metadata["projection"] == label
        assert "old" not in curve.metadata["grouped_alignment"]
