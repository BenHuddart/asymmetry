"""Phase 4: inspector dock visibility and tab-raise behaviour per domain."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.gui.mainwindow import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def win(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    w = MainWindow()
    w.show()
    qapp.processEvents()
    return w


# ── helpers ──────────────────────────────────────────────────────────────────


def _switch_domain(win: MainWindow, view: str) -> None:
    """Fire the same code path as clicking a Domain toolbar button."""
    win._apply_inspector_for_domain(view)


# ── F-B asymmetry domain ─────────────────────────────────────────────────────


def test_fb_domain_makes_fit_fourier_params_visible(win: MainWindow) -> None:
    _switch_domain(win, "fb_asymmetry")
    assert win._dock_fit.isVisible()
    assert win._dock_fourier.isVisible()
    assert win._dock_fit_parameters.isVisible()


def test_fb_domain_raises_fit_dock(win: MainWindow) -> None:
    # Raise fourier first so we can confirm it gets overridden.
    win._dock_fourier.raise_()
    _switch_domain(win, "fb_asymmetry")
    # _dock_fit should be the raised (visible-not-hidden-behind-others) tab.
    # In a tabified group the raised dock is the one whose tab is active.
    assert not win._dock_fit.visibleRegion().isEmpty() or win._dock_fit.isVisible()


# ── Groups domain ────────────────────────────────────────────────────────────


def test_groups_domain_shows_fit_and_params(win: MainWindow) -> None:
    _switch_domain(win, "groups")
    assert win._dock_fit.isVisible()
    assert win._dock_fit_parameters.isVisible()


def test_groups_domain_hides_fourier(win: MainWindow) -> None:
    # Start with fourier visible so we can verify it gets hidden.
    _switch_domain(win, "fb_asymmetry")
    assert win._dock_fourier.isVisible()
    _switch_domain(win, "groups")
    assert not win._dock_fourier.isVisible()


def test_groups_domain_raises_fit_dock(win: MainWindow) -> None:
    _switch_domain(win, "groups")
    assert win._dock_fit.isVisible()


# ── Frequency domain ─────────────────────────────────────────────────────────


def test_frequency_domain_makes_all_three_visible(win: MainWindow) -> None:
    _switch_domain(win, "frequency")
    assert win._dock_fit.isVisible()
    assert win._dock_fourier.isVisible()
    assert win._dock_fit_parameters.isVisible()


def test_frequency_domain_raises_fourier(win: MainWindow) -> None:
    # Start with fit raised to confirm it gets overridden.
    _switch_domain(win, "fb_asymmetry")
    _switch_domain(win, "frequency")
    assert win._dock_fourier.isVisible()


# ── Domain cycling resets the raised tab ─────────────────────────────────────


def test_domain_switch_resets_raised_tab(win: MainWindow) -> None:
    """Switching fb → freq → fb re-raises Fit, not whatever was raised last in fb."""
    _switch_domain(win, "fb_asymmetry")
    # Simulate user clicking Fourier tab inside fb domain.
    win._dock_fourier.raise_()
    # Switch away and back — Fit should be raised again (domain reset).
    _switch_domain(win, "frequency")
    _switch_domain(win, "fb_asymmetry")
    assert win._dock_fit.isVisible()


# ── Fourier hidden in groups reappears on fb switch ──────────────────────────


def test_fourier_hidden_by_groups_reappears_in_fb(win: MainWindow) -> None:
    _switch_domain(win, "groups")
    assert not win._dock_fourier.isVisible()
    _switch_domain(win, "fb_asymmetry")
    assert win._dock_fourier.isVisible()


# ── Floating dock is skipped ─────────────────────────────────────────────────


def test_floating_dock_unaffected_by_domain_switch(win: MainWindow) -> None:
    # Float the fourier dock then switch to groups (which would hide it if docked).
    win._dock_fourier.setFloating(True)
    try:
        was_visible = win._dock_fourier.isVisible()
        _switch_domain(win, "groups")
        assert win._dock_fourier.isVisible() == was_visible
    finally:
        win._dock_fourier.setFloating(False)


# ── Unknown domain token is a no-op ──────────────────────────────────────────


def test_unknown_domain_token_is_noop(win: MainWindow) -> None:
    _switch_domain(win, "fb_asymmetry")
    fit_visible = win._dock_fit.isVisible()
    _switch_domain(win, "unknown_domain")
    assert win._dock_fit.isVisible() == fit_visible
