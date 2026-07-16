"""The MaxEnt workload warning must never block scripted/offscreen sessions."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore

from asymmetry.core.maxent import MaxEntConfig, MaxEntWorkloadEstimate
from asymmetry.gui.mainwindow import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _unsafe_estimate() -> MaxEntWorkloadEstimate:
    """An estimate far past every warning threshold (a raw full-res .mdu run)."""
    return MaxEntWorkloadEstimate(
        run_number=686,
        selected_group_count=8,
        time_points_per_group=(388_894,) * 8,
        n_spectrum_points=1 << 19,
        peak_dense_matrix_bytes=388_894 * (1 << 19) * 8,
        total_dense_matrix_bytes=8 * 388_894 * (1 << 19) * 8,
    )


def test_workload_confirm_proceeds_headless_without_modal(qapp: QApplication) -> None:
    """Offscreen there is no user to dismiss the modal, so ``exec()`` would
    hang the session (it blocked the corpus screenshot scenarios); the warning
    must route to the log and proceed instead."""
    window = MainWindow()
    estimate = _unsafe_estimate()
    assert window._maxent_workload_is_unsafe(estimate) is True

    # Must return promptly (no modal event loop) and allow the calculation.
    assert window._confirm_maxent_workload(estimate, MaxEntConfig()) is True


def test_workload_confirm_safe_estimate_never_asks(qapp: QApplication) -> None:
    window = MainWindow()
    estimate = MaxEntWorkloadEstimate(
        run_number=1,
        selected_group_count=2,
        time_points_per_group=(2000, 2000),
        n_spectrum_points=1024,
        peak_dense_matrix_bytes=2000 * 1024 * 8,
        total_dense_matrix_bytes=2 * 2000 * 1024 * 8,
    )
    assert window._maxent_workload_is_unsafe(estimate) is False
    assert window._confirm_maxent_workload(estimate, MaxEntConfig()) is True
