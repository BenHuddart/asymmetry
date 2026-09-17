"""GUI-side series identity: batch-id minting after a project load.

The core half of series identity — ``FitSeries.recipe_identity`` (D3) and the
default-label scheme (D10) — lives in ``tests/core/test_series_model.py`` and
``tests/core/test_series_naming.py``. What is left here needs a ``MainWindow``.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def mw():
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QApplication.instance() or QApplication([])
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


@pytest.mark.gui
def test_reseed_batch_index_avoids_collision_with_loaded_batches(mw):
    """A fresh batch id after load must not collide with a restored batch-N."""
    from asymmetry.core.representation import FitSeries as _FitSeries
    from asymmetry.core.representation import RepresentationType as _Rep

    mw._next_batch_index = 1  # fresh window
    mw._project_model.add_batch(
        _FitSeries("batch-3", _Rep.TIME_FB_ASYMMETRY, member_run_numbers=[1, 2])
    )
    mw._reseed_batch_index()
    assert mw._next_batch_id() == "batch-4"  # past the loaded batch-3
