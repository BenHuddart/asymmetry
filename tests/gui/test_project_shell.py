"""ProjectShell: several open projects hosted as tabs in one window.

Each tab is a whole ``MainWindow`` constructed as a plain child widget (see
``asymmetry.gui.shell``). These tests pin the behaviour the shell itself is
responsible for: per-tab isolation with a shared reduction-cache budget, tab
routing for new/open/close, the quit-time multi-project guard, tab-label and
menu-bar mirroring of the active project, the process-wide bulk-load /
recent-projects / UI-scale broadcasts reaching every tab, and floating-dock
visibility across a tab switch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.core.project import save_project  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from asymmetry.gui.shell import ProjectShell  # noqa: E402


@pytest.fixture
def shell(qapp: QApplication):
    s = ProjectShell()
    s.show()
    qapp.processEvents()
    yield s


def _ds(run_number: int = 11) -> MuonDataset:
    n = 8
    meta = {"run_number": run_number, "field": 100.0, "temperature": 10.0}
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.full(n, 110.0), bin_width=0.01),
            Histogram(counts=np.full(n, 90.0), bin_width=0.01),
        ],
        metadata=dict(meta),
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "bunching_factor": 1,
        },
    )
    t = np.arange(n) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros(n),
        error=np.full(n, 0.01),
        metadata=dict(meta),
        run=run,
    )


# ── per-tab isolation ────────────────────────────────────────────────────


def test_hosted_page_is_a_child_widget_with_no_window_of_its_own(
    shell: ProjectShell, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page must never own a window: on macOS the native menu bar would give a
    top-level page its own native window, which survives into the shell and
    offsets every mapToGlobal in the tab, so clicks land on the wrong widget.
    Offscreen has no native menu bar, so the guard that bites on every platform
    is the construction order: the page is already a plain widget by the time
    its menu bar is built.
    """
    is_window_at_menu_build: list[bool] = []
    setup_menus = MainWindow._setup_menus

    def spy(self: MainWindow) -> None:
        is_window_at_menu_build.append(self.isWindow())
        setup_menus(self)

    monkeypatch.setattr(MainWindow, "_setup_menus", spy)
    page = shell.add_project()
    qapp.processEvents()

    assert is_window_at_menu_build == [False]
    assert not page.isWindow()
    assert page.windowHandle() is None
    assert page.window() is shell
    origin = page.mapTo(shell, QPoint(0, 0))
    assert page.mapToGlobal(QPoint(0, 0)) == shell.mapToGlobal(origin)


def test_tabs_hold_independent_state(shell: ProjectShell, qapp: QApplication) -> None:
    page1 = shell.add_project()
    page2 = shell.add_project()
    page1._data_browser.add_dataset(_ds())
    qapp.processEvents()

    assert page1._data_browser.all_datasets()
    assert not page2._data_browser.all_datasets()
    assert page1._dirty is True
    assert page2._dirty is False
    assert page1._data_browser is not page2._data_browser
    assert page1._project_model is not page2._project_model
    assert shell.pages()[0]._reduction_cache is shell.pages()[1]._reduction_cache


# ── open_project tab routing ─────────────────────────────────────────────


def test_open_project_into_untouched_tab_replaces_in_place(shell: ProjectShell, tmp_path) -> None:
    page = shell.add_project()
    path = str(tmp_path / "seed.asymp")
    save_project(page.collect_project_state(), path)

    shell.open_project(path)

    assert shell._tabs.count() == 1
    assert shell.current_page() is page
    assert page._current_project_path == path


def test_open_project_into_touched_tab_opens_new_tab(shell: ProjectShell, tmp_path) -> None:
    page = shell.add_project()
    path = str(tmp_path / "seed.asymp")
    save_project(page.collect_project_state(), path)
    page._data_browser.add_dataset(_ds())

    shell.open_project(path)

    assert shell._tabs.count() == 2
    new_page = shell.current_page()
    assert new_page is not page
    assert new_page._current_project_path == path
    assert page._data_browser.all_datasets()


def test_new_project_from_hosted_page_opens_new_tab(
    shell: ProjectShell, qapp: QApplication
) -> None:
    page = shell.add_project()
    page._data_browser.add_dataset(_ds())
    qapp.processEvents()

    page._on_new_project()

    assert shell._tabs.count() == 2
    assert shell.current_page() is not page
    assert page._data_browser.all_datasets()


# ── close-tab guard ───────────────────────────────────────────────────────


@pytest.mark.real_save_guard
def test_close_tab_cancel_keeps_tab(shell: ProjectShell, monkeypatch: pytest.MonkeyPatch) -> None:
    page1 = shell.add_project()
    shell.add_project()
    page1._mark_dirty()
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel)
    )
    try:
        shell.close_page(page1)
        assert shell._tabs.count() == 2
        assert page1 in shell.pages()
    finally:
        # Force-clear so autouse teardown's shell.close() does not re-prompt
        # via the un-stubbed (real_save_guard) _maybe_save and hang offscreen.
        for page in shell.pages():
            page._dirty = False


@pytest.mark.real_save_guard
def test_close_tab_discard_removes_tab(
    shell: ProjectShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    page1 = shell.add_project()
    shell.add_project()
    page1._mark_dirty()
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard)
    )
    try:
        shell.close_page(page1)
        assert shell._tabs.count() == 1
        assert page1 not in shell.pages()
    finally:
        for page in shell.pages():
            page._dirty = False


@pytest.mark.real_save_guard
def test_closing_last_tab_leaves_fresh_untitled_tab(
    shell: ProjectShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = shell.add_project()
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard)
    )
    try:
        shell.close_page(page)
        assert shell._tabs.count() == 1
        fresh = shell.current_page()
        assert not fresh._data_browser.all_datasets()
        assert fresh._dirty is False
    finally:
        for page in shell.pages():
            page._dirty = False


def test_close_page_deletes_its_menu_bar(shell: ProjectShell, qapp: QApplication) -> None:
    page1 = shell.add_project()
    shell.add_project()
    bar = shell._menu_bars.widget(0)

    shell.close_page(page1)
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

    assert shell._menu_bars.count() == 1
    assert not shiboken6.isValid(bar)


# ── quit guard ────────────────────────────────────────────────────────────


@pytest.mark.real_save_guard
def test_quit_second_cancel_aborts_and_shuts_down_nothing(
    shell: ProjectShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [shell.add_project() for _ in range(3)]
    pages[0]._mark_dirty()
    pages[2]._mark_dirty()
    responses = iter([QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Cancel])
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: next(responses)))
    shutdown_calls: list[bool] = []
    pages[0]._shutdown_workers = lambda: shutdown_calls.append(True)
    try:
        assert shell.close() is False
        assert shell._tabs.count() == 3
        assert shell.pages() == pages
        assert shutdown_calls == []
    finally:
        for page in shell.pages():
            page._dirty = False


@pytest.mark.real_save_guard
def test_quit_all_discard_closes_shell(
    shell: ProjectShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [shell.add_project() for _ in range(3)]
    pages[0]._mark_dirty()
    pages[2]._mark_dirty()
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard)
    )
    try:
        assert shell.close() is True
    finally:
        for page in shell.pages():
            page._dirty = False


# ── tab label / window-modified mirroring ─────────────────────────────────


def test_tab_label_and_window_modified_mirror_page_state(
    shell: ProjectShell, qapp: QApplication
) -> None:
    page = shell.add_project()
    assert shell._tabs.tabText(0) == "Untitled"

    page._data_browser.add_dataset(_ds())
    qapp.processEvents()
    assert shell._tabs.tabText(0) == "Untitled*"
    assert shell.isWindowModified() is True
    assert shell.windowTitle() == page.windowTitle()

    page._clear_dirty()
    assert shell._tabs.tabText(0) == "Untitled"


def test_tab_label_shows_file_stem_after_open(shell: ProjectShell, tmp_path) -> None:
    page = shell.add_project()
    path = str(tmp_path / "myproj.asymp")
    save_project(page.collect_project_state(), path)

    shell.open_project(path)

    assert shell._tabs.tabText(0) == "myproj"


# ── menu-bar stack follows the active tab ─────────────────────────────────


def test_menu_bar_stack_follows_active_tab(shell: ProjectShell) -> None:
    shell.add_project()
    page2 = shell.add_project()

    shell._tabs.setCurrentIndex(1)

    assert shell._menu_bars.currentIndex() == 1
    assert shell._pages.currentWidget() is page2
    assert shell._menu_bars.currentWidget() is not shell._menu_bars.widget(0)
    assert shell._menu_bars.currentWidget().actions()[0].text() == "&File"


# ── bulk-load gate is process-wide, not per-page ──────────────────────────


def test_bulk_load_gate_blocks_other_tab(
    shell: ProjectShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    page1 = shell.add_project()
    page2 = shell.add_project()
    monkeypatch.setattr(mw_module, "_bulk_load_owner", page1)

    assert page2._load_paths_with_progress(["x.dat"]) == {}
    assert page2.statusBar().currentMessage() == mw_module._BULK_LOAD_BUSY_MESSAGE
    assert page2._apply_items_with_progress("t", [("a", lambda: None)]) is False


def test_apply_items_marks_page_busy_for_the_close_guards(shell: ProjectShell) -> None:
    page = shell.add_project()
    seen: list[bool] = []

    page._apply_items_with_progress("t", [("a", lambda: seen.append(page._bulk_load_active))])

    assert seen == [True]
    assert page._bulk_load_active is False


# ── window() resolves the shell's UI manager for an embedded page ────────


def test_embedded_widget_window_resolves_shell_ui_manager(
    shell: ProjectShell, qapp: QApplication
) -> None:
    page1 = shell.add_project()
    shell.add_project()
    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()

    widget = page1._fit_parameters_panel
    assert widget.window() is shell
    assert widget.window()._ui_manager is page1._ui_manager
    assert widget._ui_scale_sync_connected is True


# ── recent-projects broadcast reaches every tab ───────────────────────────


def test_recent_projects_broadcast_reaches_other_tabs(shell: ProjectShell, tmp_path) -> None:
    settings = QSettings()
    previous = settings.value("project/recent_files")
    page1 = shell.add_project()
    page2 = shell.add_project()
    try:
        page1._add_recent_project(str(tmp_path / "x.asymp"))
        assert page2._recent_menu.actions()[0].text() == "x.asymp"
    finally:
        if previous is None:
            settings.remove("project/recent_files")
        else:
            settings.setValue("project/recent_files", previous)


# ── UI-scale broadcast reaches every tab ──────────────────────────────────


def test_ui_scale_broadcast_reaches_other_tabs(shell: ProjectShell) -> None:
    settings = QSettings()
    previous = settings.value(mw_module._UI_SCALE_SETTINGS_KEY)
    page1 = shell.add_project()
    page2 = shell.add_project()
    try:
        page1._ui_manager.set_ui_scale(1.1)
        assert page2._ui_manager.ui_scale == 1.1
        assert page2._ui_scale_actions[1.1].isChecked()
    finally:
        if previous is None:
            settings.remove(mw_module._UI_SCALE_SETTINGS_KEY)
        else:
            settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, previous)


# ── floating docks follow the active tab ──────────────────────────────────


def test_floating_dock_hidden_on_tab_switch_and_restored_on_return(
    shell: ProjectShell, qapp: QApplication
) -> None:
    page1 = shell.add_project()
    shell.add_project()
    shell._tabs.setCurrentIndex(0)

    dock = page1._dock_log
    dock.setFloating(True)
    dock.show()
    qapp.processEvents()

    shell._tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert dock.isVisible() is False

    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert dock.isVisible() is True


# ── Ctrl+W / Ctrl+Q routing ────────────────────────────────────────────────


def test_close_project_routes_to_shell_close_page(shell: ProjectShell) -> None:
    page = shell.add_project()
    calls: list[MainWindow] = []
    shell.close_page = calls.append

    page._on_close_project()

    assert calls == [page]


def test_exit_routes_to_shell_close(shell: ProjectShell) -> None:
    shell.add_project()
    calls: list[bool] = []
    shell.close = lambda: calls.append(True)

    shell.current_page()._on_exit()

    assert calls == [True]


# ── standalone MainWindow (no shell) fallback ─────────────────────────────


@pytest.mark.parametrize("method_name", ["_on_close_project", "_on_exit"])
def test_standalone_mainwindow_closes_without_shell(qapp: QApplication, method_name: str) -> None:
    window = MainWindow()
    window.show()
    qapp.processEvents()

    getattr(window, method_name)()

    assert window.isVisible() is False
