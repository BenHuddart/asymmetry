"""Options → Advanced → "Rotating reference frame" toggle (item 2).

App-level chrome (QSettings, default off) gating the entire RRF surface;
auto-enables when a project carrying active RRF parameters is opened.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.gui.mainwindow import _RRF_ADVANCED_SETTINGS_KEY

pytestmark = [pytest.mark.gui]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    QSettings().setValue("ui/scale", 1.0)
    QSettings().setValue(_RRF_ADVANCED_SETTINGS_KEY, False)
    win = mw_module.MainWindow()
    yield win
    win.close()
    win.deleteLater()


def _rrf_controls(win):
    return win._plot_panel._rrf_controls


def test_default_off_controls_absent(window):
    assert not window._rrf_advanced_action.isChecked()
    assert not _rrf_controls(window).feature_enabled()


def test_toggle_on_persists_and_enables_feature(window):
    window._rrf_advanced_action.setChecked(True)  # fires the handler
    assert _rrf_controls(window).feature_enabled()
    from asymmetry.gui.mainwindow import _coerce_bool

    assert _coerce_bool(QSettings().value(_RRF_ADVANCED_SETTINGS_KEY), default=False) is True
    window._rrf_advanced_action.setChecked(False)
    assert not _rrf_controls(window).feature_enabled()


def test_setting_persists_across_restart(window, qapp):
    QSettings().setValue(_RRF_ADVANCED_SETTINGS_KEY, True)
    fresh = mw_module.MainWindow()
    try:
        assert fresh._rrf_advanced_action.isChecked()
        assert fresh._plot_panel._rrf_controls.feature_enabled()
    finally:
        fresh.close()
        fresh.deleteLater()


def test_open_project_with_active_rrf_auto_enables(window):
    # Toggle off; a project carrying an active RRF frame opens.
    assert not window._rrf_advanced_action.isChecked()
    controls = _rrf_controls(window)
    controls._freq_spin.setValue(30.0)
    controls._enable_check.setChecked(True)  # active params, feature still off
    assert window._plot_panel.rrf_has_active_parameters()
    window._auto_enable_rrf_for_active_project()
    assert window._rrf_advanced_action.isChecked()
    assert controls.feature_enabled()
