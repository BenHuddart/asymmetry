"""Options → Advanced → "Rotating reference frame" toggle (item 2).

App-level chrome (QSettings, default off) gating the entire RRF surface;
auto-enables when a project carrying active RRF parameters is opened.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings

import asymmetry.gui.mainwindow as mw_module
from asymmetry.gui.mainwindow import _RRF_ADVANCED_SETTINGS_KEY

pytestmark = [pytest.mark.gui]


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


def test_magnitude_display_keeps_lab_observations_in_actual_fit_panel(window):
    import numpy as np

    from tests.gui.test_rrf_controls import _dataset

    dataset = _dataset()
    window._data_browser.add_dataset(dataset)
    window._data_browser.select_runs([dataset.run_number])
    tab = window._fit_panel._single_tab
    original = tab._current_dataset
    assert original is not None
    before = (original.time.copy(), original.asymmetry.copy(), original.error.copy())
    window._rrf_advanced_action.setChecked(True)
    controls = window._plot_panel._rrf_controls
    controls._freq_spin.setValue(30.0)
    controls._component_combo.setCurrentIndex(controls._component_combo.findData("magnitude"))
    controls._phase_spin.setValue(37.0)
    controls._enable_check.setChecked(True)
    plotted = window._plot_panel._last_plot_asymmetry
    assert np.nanmin(plotted) > 0
    current = tab._current_dataset
    for actual, expected in zip(
        (current.time, current.asymmetry, current.error), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert np.min(current.asymmetry) < 0
    window._dirty = False
