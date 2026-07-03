"""Bug #7: the batch-seeding control on the Batch tab and the Analysis ▸ Batch
seeding menu must stay in sync both ways.

The seeding modes (Auto / Independent seeds / Chain from previous run) governed
the Batch tab but lived only in the menu bar, disjoint from the tab — "Chain"
(the natural mode for an ordered T/B scan) was easy to miss. The control is now
mirrored on the tab; selecting it in one place updates the other.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore  # noqa: E402
from PySide6.QtWidgets import QApplication  # type: ignore  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    win = MainWindow()
    yield win
    win.close()
    win.deleteLater()


def _combo(win: MainWindow):
    return win._fit_panel._global_tab._seeding_combo


def test_menu_selection_updates_batch_tab_combo(mainwindow: MainWindow) -> None:
    # Default: both sides on "auto".
    assert mainwindow._batch_seeding_actions["auto"].isChecked()
    assert _combo(mainwindow).currentData() == "auto"

    # Choosing a mode from the menu drives the on-tab combo and the panel mode.
    mainwindow._batch_seeding_actions["chain"].trigger()
    assert _combo(mainwindow).currentData() == "chain"
    assert mainwindow._fit_panel._global_tab._batch_seeding_mode == "chain"


def test_batch_tab_combo_updates_menu(mainwindow: MainWindow) -> None:
    combo = _combo(mainwindow)
    combo.setCurrentIndex(combo.findData("as_provided"))

    # The menu's action group mirrors the on-tab selection.
    assert mainwindow._batch_seeding_actions["as_provided"].isChecked()
    assert not mainwindow._batch_seeding_actions["auto"].isChecked()
    assert mainwindow._fit_panel._global_tab._batch_seeding_mode == "as_provided"
