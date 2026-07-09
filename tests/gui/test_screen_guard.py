"""Tests for the macOS-26 QScreen use-after-free guard (``gui.screen_guard``).

The guard cannot be exercised against a real dangling ``QScreen`` in a test, so
the re-anchoring decision is driven through a fake application object; a second
test checks that installing the guard and running the handler against the real
session ``QApplication`` is inert and error-free.
"""

from __future__ import annotations

import pytest

from asymmetry.gui import screen_guard

pytestmark = [pytest.mark.gui]


class _FakeWindow:
    def __init__(self, screen, *, raises: bool = False) -> None:
        self._screen = screen
        self._raises = raises
        self.set_to = None

    def screen(self):
        if self._raises:
            raise RuntimeError("window torn down mid-reconfiguration")
        return self._screen

    def setScreen(self, screen) -> None:  # noqa: N802 - mirrors the Qt API
        self.set_to = screen


class _FakeApp:
    def __init__(self, primary, screens, windows) -> None:
        self._primary = primary
        self._screens = screens
        self._windows = windows

    def primaryScreen(self):  # noqa: N802 - mirrors the Qt API
        return self._primary

    def screens(self):
        return self._screens

    def topLevelWindows(self):  # noqa: N802 - mirrors the Qt API
        return self._windows


def _patch_app(monkeypatch, app) -> None:
    monkeypatch.setattr(screen_guard.QGuiApplication, "instance", staticmethod(lambda: app))


def test_reanchors_only_windows_on_a_dead_screen(monkeypatch) -> None:
    primary = object()  # a live screen
    dead = object()  # a screen no longer in screens()
    on_live = _FakeWindow(primary)
    on_dead = _FakeWindow(dead)
    _patch_app(monkeypatch, _FakeApp(primary, [primary], [on_live, on_dead]))

    screen_guard.reanchor_stale_windows()

    # The window whose screen vanished is re-pointed at the primary; the window
    # already on a live screen is left untouched (multi-monitor placement kept).
    assert on_dead.set_to is primary
    assert on_live.set_to is None


def test_reanchors_window_with_null_screen(monkeypatch) -> None:
    primary = object()
    orphan = _FakeWindow(None)
    _patch_app(monkeypatch, _FakeApp(primary, [primary], [orphan]))

    screen_guard.reanchor_stale_windows()

    assert orphan.set_to is primary


def test_survives_a_window_torn_down_mid_reconfiguration(monkeypatch) -> None:
    primary = object()
    dying = _FakeWindow(None, raises=True)
    healthy = _FakeWindow(None)
    _patch_app(monkeypatch, _FakeApp(primary, [primary], [dying, healthy]))

    # Must not propagate the RuntimeError, and must still reach later windows.
    screen_guard.reanchor_stale_windows()

    assert healthy.set_to is primary


def test_noop_without_application_or_primary(monkeypatch) -> None:
    _patch_app(monkeypatch, None)
    screen_guard.reanchor_stale_windows()  # no application → returns quietly

    _patch_app(monkeypatch, _FakeApp(None, [], []))
    screen_guard.reanchor_stale_windows()  # no primary screen → returns quietly


def test_install_and_run_against_real_qapp_is_inert(qapp) -> None:
    if qapp is None:
        pytest.skip("PySide6 unavailable")
    # Installing the guard connects the display-change signals without error, and
    # running the handler against the real (single-screen, offscreen) app is a
    # no-op: every window is already on a live screen.
    screen_guard.install_screen_change_guard(qapp)
    screen_guard.reanchor_stale_windows()
