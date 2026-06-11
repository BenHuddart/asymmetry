"""Design-handoff visual grammar: dock headers, joined segments, tabs, status."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QTabWidget

import asymmetry.gui.mainwindow as mw_module
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    build_segmented_cell_qss,
    build_segmented_container_qss,
)
from asymmetry.gui.widgets.dock_header import DockHeader


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


# ── dock headers replace (never duplicate) the Qt title bar ──────────────────


def test_log_and_browser_docks_use_custom_title_bars(win: MainWindow) -> None:
    """setTitleBarWidget REPLACES the native title — one header, never two."""
    assert isinstance(win._dock_log.titleBarWidget(), DockHeader)
    assert isinstance(win._dock_data_browser.titleBarWidget(), DockHeader)
    # windowTitle stays set for menus/tabs/floating chrome.
    assert win._dock_log.windowTitle()
    assert win._dock_data_browser.windowTitle()


def test_log_header_counts_entries(win: MainWindow) -> None:
    before = win._log_panel.entry_count()
    win._log_panel.log("hello", tag="load")
    assert win._log_panel.entry_count() == before + 1
    assert f"{before + 1} entries" in win._log_dock_header._meta_label.text()


def test_browser_header_shows_selection_count(win: MainWindow) -> None:
    win._update_status_selection()
    # No data loaded → meta stays empty rather than reading "0 of 0".
    assert win._browser_dock_header._meta_label.text() == ""


def test_dock_header_close_feeds_closed_tab_memory(win: MainWindow, qapp: QApplication) -> None:
    """The custom close button raises a real Close event (dock.close())."""
    win._dock_log.close()
    qapp.processEvents()
    assert not win._dock_log.isVisible()
    win._on_show_log()
    assert win._dock_log.isVisible()


# ── inspector tabs: top position ─────────────────────────────────────────────


def test_right_dock_area_tabs_are_north(win: MainWindow) -> None:
    assert win.tabPosition(Qt.DockWidgetArea.RightDockWidgetArea) == QTabWidget.TabPosition.North


# ── joined segmented control QSS ─────────────────────────────────────────────


def test_segmented_cells_only_round_outer_corners() -> None:
    first = build_segmented_cell_qss(first=True, last=False)
    middle = build_segmented_cell_qss(first=False, last=False)
    last = build_segmented_cell_qss(first=False, last=True)
    assert "border-top-left-radius: 3px" in first
    assert "border-top-right-radius: 0px" in first
    assert "border-top-left-radius: 0px" in middle
    assert "border-top-right-radius: 0px" in middle
    assert "border-top-right-radius: 3px" in last
    # Internal dividers on all but the last cell.
    assert "border-right: 1px solid" in first
    assert "border-right: 1px solid" in middle
    assert "border-right" not in last.split("QPushButton:checked")[0].replace(
        "border-top-right-radius", ""
    ).replace("border-bottom-right-radius", "")


def test_segmented_container_owns_the_outer_border() -> None:
    qss = build_segmented_container_qss()
    assert f"border: 1px solid {tokens.BORDER}" in qss
    assert "border-radius: 4px" in qss


def test_domain_buttons_live_in_joined_cells(win: MainWindow) -> None:
    for btn in win._domain_buttons:
        ss = btn.styleSheet()
        assert "font-weight: 600" in ss
        assert "border: none" in ss  # cells are borderless; the frame owns it


# ── primary action treatment ─────────────────────────────────────────────────


def test_primary_button_qss_is_accent_tinted() -> None:
    qss = build_primary_button_qss()
    assert tokens.ACCENT_SOFT in qss
    assert tokens.ACCENT in qss
    assert "QPushButton:disabled" in qss


def test_panel_primary_actions_carry_the_treatment(win: MainWindow) -> None:
    assert tokens.ACCENT in win._fourier_panel._fft_btn.styleSheet()
    assert tokens.ACCENT in win._alc_fit_panel._build_btn.styleSheet()


# ── status bar additions ─────────────────────────────────────────────────────


def test_status_bar_has_state_dot_and_chi2_slot(win: MainWindow) -> None:
    assert win._status_state_label.text() == "● Idle"
    assert win._status_chi2_label.text() == ""
    win._set_status_chi2(1.0432)
    assert win._status_chi2_label.text() == "χ²/ν = 1.04"
    win._set_status_chi2(None)
    assert win._status_chi2_label.text() == ""


def test_deck_docks_use_slim_title_bars(win: MainWindow, qapp: QApplication) -> None:
    """Tabified deck docks show buttons-only headers while docked — the tab
    strip above already names the pane, so no repeated 'Fit' title."""
    for dock in (win._dock_fit, win._dock_fourier, win._dock_fit_parameters):
        header = dock.titleBarWidget()
        assert isinstance(header, DockHeader)
        assert not header._title_label.isVisible()
    # Floating restores the title (tracking windowTitle), since the tab bar
    # is no longer there to identify the pane.
    win._dock_fit.setFloating(True)
    qapp.processEvents()
    try:
        assert win._dock_fit.titleBarWidget()._title_label.isVisible()
        assert win._dock_fit.titleBarWidget()._title_label.text() == "FIT"
    finally:
        win._dock_fit.setFloating(False)


def test_toolbar_cluster_captions_are_uppercase(win: MainWindow) -> None:
    """Deliberate handoff deviation: stacked uppercase domain captions."""
    from PySide6.QtWidgets import QLabel

    texts = {label.text() for label in win._main_toolbar.findChildren(QLabel)}
    assert "TIME DOMAIN" in texts
    assert "FREQUENCY DOMAIN" in texts


def test_browser_headers_match_handoff(win: MainWindow) -> None:
    """Plain upright units, no resting sort arrow, slimmer numeric columns."""
    browser = win._data_browser
    assert browser._COLUMNS == ["Run", "Title", "T (K)", "B (G)"]
    header = browser._table.horizontalHeader()
    assert not header.isSortIndicatorShown()
    assert header.sectionSize(0) <= 100
    assert header.sectionSize(2) <= 60
    assert header.sectionSize(3) <= 60
