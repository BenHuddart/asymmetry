"""Tests for the in-process gleplot editor launcher (``gui/utils/gle_editor.py``).

Covers feature detection, opening/closing real editor windows against the
project venv's gleplot (>= 1.6, embedding API present), graceful fallback
when the API is missing, and the two integration points on ``MainWindow``
(the Analysis menu action and the ``closeEvent`` shutdown hook).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.utils.gle_editor import (
    close_all_gle_editors,
    gle_editor_available,
    launch_gle_editor,
    open_gle_editor_count,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _close_stray_editors(qapp: QApplication) -> None:
    """Belt-and-braces: never let a failed assertion leak an editor window."""
    yield
    close_all_gle_editors()


def _write_minimal_gle(tmp_path: Path) -> Path:
    """Export a minimal ``.gle`` figure via the real gleplot API."""
    glp = pytest.importorskip("gleplot")
    fig = glp.figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 2, 3])
    gle_path = tmp_path / "editor_launch_test.gle"
    fig.savefig(str(gle_path))
    return gle_path


def test_gle_editor_available_in_project_venv(qapp: QApplication) -> None:
    """The project venv pins a local editable gleplot >= 1.6 with the API."""
    assert gle_editor_available() is True


@pytest.mark.skipif(not gle_editor_available(), reason="gleplot embedding API not installed")
def test_launch_gle_editor_opens_and_closes_window(qapp: QApplication, tmp_path: Path) -> None:
    gle_path = _write_minimal_gle(tmp_path)
    assert open_gle_editor_count() == 0

    opened = launch_gle_editor(gle_path)

    assert opened is True
    assert open_gle_editor_count() == 1

    close_all_gle_editors()

    assert open_gle_editor_count() == 0


@pytest.mark.skipif(not gle_editor_available(), reason="gleplot embedding API not installed")
def test_launch_gle_editor_blank_opens_window(qapp: QApplication) -> None:
    assert open_gle_editor_count() == 0

    opened = launch_gle_editor(None)

    assert opened is True
    assert open_gle_editor_count() == 1

    close_all_gle_editors()
    assert open_gle_editor_count() == 0


def test_gle_editor_available_false_when_api_missing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``gleplot.gui`` module that fails to import a real ``open_editor``.

    Setting the ``sys.modules`` entry to ``None`` is a negative-import cache:
    ``from gleplot.gui import open_editor`` raises ``ImportError`` exactly like
    an older gleplot build lacking the submodule.
    """
    monkeypatch.setitem(sys.modules, "gleplot.gui", None)

    assert gle_editor_available() is False


def test_launch_gle_editor_returns_false_when_api_missing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "gleplot.gui", None)

    assert launch_gle_editor(None) is False
    assert open_gle_editor_count() == 0


# ── MainWindow integration: menu action + closeEvent shutdown ──────────────


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def _find_action(window: MainWindow, text: str):
    """Find a top-level menu's action by its exact text."""
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.text() == text:
                return action
    return None


def test_gle_editor_menu_action_exists(mainwindow: MainWindow) -> None:
    action = _find_action(mainwindow, "GLE Figure &Editor…")
    assert action is not None


def test_gle_editor_menu_action_launches_editor(
    mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(mw_module, "launch_gle_editor", lambda path: calls.append(path) or True)

    mainwindow._on_open_gle_editor()

    assert calls == [None]


def test_gle_editor_menu_action_shows_info_when_unavailable(
    mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mw_module, "launch_gle_editor", lambda path: False)
    infos: list[tuple[object, ...]] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **_kwargs: infos.append(args))

    mainwindow._on_open_gle_editor()

    assert len(infos) == 1
    assert "gleplot" in infos[0][2]


def test_close_event_calls_close_all_gle_editors(
    mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(mw_module, "close_all_gle_editors", lambda: calls.append(True))

    mainwindow.close()

    assert calls == [True]
