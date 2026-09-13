"""The application window: several open projects as tabs over one shell.

Each tab is a whole :class:`~asymmetry.gui.mainwindow.MainWindow`, constructed
as a plain child widget, so a project keeps its own docks, panels, and session
state and the two layers stay separable. The shell owns only what must be
shared: the tab strip, the stack of per-page menu bars, the reduction-cache
budget, and the quit sequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.mainwindow import MainWindow, _load_window_icon
from asymmetry.gui.screen_guard import place_window_on_screen
from asymmetry.gui.styles import tokens
from asymmetry.gui.utils.gle_editor import close_all_gle_editors
from asymmetry.gui.utils.reduction_cache import ReductionCache

if TYPE_CHECKING:
    from asymmetry.gui.ui_manager import UIManager

# The row draws the single separator under the menu bar and tab strip, so the
# menu bar's own bottom border is suppressed rather than doubled up.
_TAB_ROW_QSS = f"""
#projectTabRow {{
    background: {tokens.SURFACE_ALT};
    border-bottom: 1px solid {tokens.BORDER};
}}
#projectTabRow QMenuBar {{
    border: none;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: transparent;
    color: {tokens.TEXT_MUTED};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 2px 8px 2px 10px;
    margin-right: 1px;
}}
QTabBar::tab:selected {{
    color: {tokens.TEXT};
    border-bottom: 2px solid {tokens.ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {tokens.TEXT};
}}
"""


class ProjectShell(QWidget):
    """Hosts one :class:`MainWindow` per open project, switched by a tab bar."""

    def __init__(self) -> None:
        super().__init__()
        # The [*] placeholder mirrors the active page's title verbatim, so the
        # unsaved marker follows the tab the user is looking at.
        self.setWindowTitle("Asymmetry — μSR Data Analysis[*]")
        icon = _load_window_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        place_window_on_screen(self)

        # Cache entries are keyed on id(run) and released by finalizers, so
        # nothing crosses projects; sharing one cache shares its memory budget
        # rather than handing every tab a budget of its own.
        self._reduction_cache = ReductionCache()
        # A floating dock is a child *window*: hiding its page does not hide
        # it, so the outgoing page's floating docks are hidden by hand and
        # restored when that page comes back.
        self._hidden_floating_docks: dict[MainWindow, list[QDockWidget]] = {}

        self._menu_bars = QStackedWidget()
        self._menu_bars.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

        self._tabs = QTabBar()
        self._tabs.setDocumentMode(True)
        self._tabs.setDrawBase(False)
        self._tabs.setExpanding(False)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self._tabs.setTabsClosable(True)

        row = QWidget()
        row.setObjectName("projectTabRow")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(_TAB_ROW_QSS)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        row_layout.addWidget(self._menu_bars)
        row_layout.addSpacing(16)
        row_layout.addWidget(self._tabs, 1, Qt.AlignmentFlag.AlignBottom)

        self._pages = QStackedWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(row)
        layout.addWidget(self._pages, 1)

        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.tabCloseRequested.connect(
            lambda index: self.close_page(self._pages.widget(index))
        )

    @property
    def _ui_manager(self) -> UIManager:
        # Panels resolve the UI manager through self.window(), which is this
        # shell once a page is embedded; answer for the visible project.
        return self.current_page()._ui_manager

    def pages(self) -> list[MainWindow]:
        """Return the open projects in tab order."""
        return [self._pages.widget(index) for index in range(self._pages.count())]

    def current_page(self) -> MainWindow:
        """Return the project shown in the active tab."""
        return self._pages.currentWidget()

    def add_project(self) -> MainWindow:
        """Open an empty project in a new tab and make it current."""
        page = MainWindow(self._pages)
        page._shell = self
        page._reduction_cache = self._reduction_cache
        self._pages.addWidget(page)
        # Taking the bar reparents it out of the page's own layout, which drops
        # the page's menu-bar pointer — so page.menuBar() must never be called
        # again, or Qt mints a fresh, empty bar in its place.
        self._menu_bars.addWidget(page.menuBar())
        index = self._tabs.addTab("")
        page.windowTitleChanged.connect(lambda _title, page=page: self._refresh_tab(page))
        page.dirty_changed.connect(lambda _dirty, page=page: self._refresh_tab(page))
        self._tabs.setCurrentIndex(index)
        self._refresh_tab(page)
        return page

    def open_project(self, path: str) -> None:
        """Open the project at *path*, in the current tab if it is still untouched."""
        page = self.current_page()
        if (
            page._dirty
            or page._current_project_path is not None
            or page._data_browser.all_datasets()
        ):
            page = self.add_project()
        page._open_project_file(path)

    def close_page(self, page: MainWindow) -> None:
        """Close one project's tab, after its own unsaved-work guard agrees."""
        # A child widget's close() still delivers a QCloseEvent, so this runs
        # the page's Save/Discard/Cancel prompt and worker shutdown.
        if not page.close():
            return
        index = self._pages.indexOf(page)
        self._hidden_floating_docks.pop(page, None)
        # Both stacks shrink before the tab does, so the currentChanged the
        # removal emits already sees matching indices.
        self._pages.removeWidget(page)
        # The bar was reparented into the stack, so deleting the page alone
        # would leave it (and its action tree) alive under the shell.
        bar = self._menu_bars.widget(index)
        self._menu_bars.removeWidget(bar)
        bar.deleteLater()
        self._tabs.removeTab(index)
        page.deleteLater()
        if self._pages.count() == 0:
            self.add_project()

    def _refresh_tab(self, page: MainWindow) -> None:
        index = self._pages.indexOf(page)
        path = page._current_project_path
        name = Path(path).stem if path else "Untitled"
        self._tabs.setTabText(index, f"{name}*" if page._dirty else name)
        self._tabs.setTabToolTip(index, path or "")
        if page is self.current_page():
            self.setWindowTitle(page.windowTitle())
            self.setWindowModified(page.isWindowModified())

    def _on_tab_changed(self, index: int) -> None:
        if index < 0:
            # The last tab was just removed; close_page adds a fresh one.
            return
        outgoing = self._pages.currentWidget()
        page = self._pages.widget(index)
        if outgoing is not page:
            floating = [
                dock
                for dock in outgoing.findChildren(QDockWidget)
                if dock.isFloating() and dock.isVisible()
            ]
            self._hidden_floating_docks[outgoing] = floating
            for dock in floating:
                dock.hide()
        self._menu_bars.setCurrentIndex(index)
        self._pages.setCurrentIndex(index)
        for dock in self._hidden_floating_docks.pop(page, []):
            dock.show()
        self.setWindowTitle(page.windowTitle())
        self.setWindowModified(page.isWindowModified())

    def closeEvent(self, event) -> None:
        """Quit: every project gets its guard before anything is torn down."""
        pages = self.pages()
        # A bulk load's nested event loop is on the stack; kick off the same
        # cooperative-cancel-then-abandon escape its Cancel button uses so the
        # next quit attempt succeeds.
        for page in pages:
            if page._bulk_load_active:
                if page._bulk_load_cancel is not None:
                    page._bulk_load_cancel()
                page.statusBar().showMessage("Stopping file load — try closing again in a moment.")
                event.ignore()
                return
        # Prompt for all of them before closing any, so a Cancel late in the
        # walk leaves every project exactly as it was.
        for page in pages:
            if not page._maybe_save("quitting"):
                event.ignore()
                return
        for page in pages:
            if page._project_save_active:
                page.statusBar().showMessage("Saving project — try closing again in a moment.")
                event.ignore()
                return
        for page in pages:
            page._shutdown_workers()
        # Parentless gleplot editor windows are shared across the tabs, so they
        # are closed once here rather than by each page.
        close_all_gle_editors()
        super().closeEvent(event)
