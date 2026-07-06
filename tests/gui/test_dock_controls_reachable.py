"""Regression for branch ``fix/dock-responsive``: the right inspector dock.

Round-2 GUI finding (``_findings/windows-gui/`` + ``GUI_LORE.md``): the
right-hand inspector dock clipped at the maximised screen edge and would not
widen. The dock opened squeezed to ~236 px — *below* the inspector panels'
minimum width (the Parameters panel needs ~332) — so controls fell off the
right edge: the per-Y-parameter **Model Fit** buttons (``FitParametersPanel``)
and the MaxEnt Min/Max-frequency + Time Start/End fields (``MaxEntPanel``). The
plot|dock splitter was a no-op, so there was no way to recover the width.

Root cause was the dock container, not the panels (they already contain their
own ``QScrollArea``\\s). Two coupled defects: dock widths set in ``__init__`` did
not stick (the layout was not settled yet) so the deck opened at its squeezed
minimum, and the deck could not be widened afterwards. The fix (see
``MainWindow.showEvent`` / ``_apply_default_dock_widths`` and
``INSPECTOR_DOCK_DEFAULT_WIDTH``) re-applies a controls-fitting default width
once the window is shown, and resizes the three tabified right docks as a group
so the splitter widens them. The default width is now adaptive (font-metric
minimum, a fraction of the window width in between, capped so the plot stays
dominant); tests read it from ``MainWindow._inspector_default_width`` rather than
a fixed pixel constant.

These tests drive the *real* ``MainWindow`` dock layout (a standalone panel
already scrolls and so cannot reproduce the integration bug — that is why the
previous proxy test was skipped).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QScrollArea

from asymmetry.gui.mainwindow import MainWindow

#: The squeezed width the dock opened at while the bug was live. The fix must
#: clear this comfortably so the panels' controls are not clipped.
_CLIPPED_DOCK_WIDTH = 236


def _settle(app: QApplication, times: int = 6) -> None:
    """Pump the event loop so the deferred first-show dock resize fires."""
    for _ in range(times):
        app.processEvents()


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def shown_window(app: QApplication):
    """A MainWindow shown at a maximised-like width with its layout settled."""
    window = MainWindow()
    window.resize(1600, 850)  # room for the full default deck width
    window.show()
    _settle(app)
    try:
        yield window, app
    finally:
        window.close()
        window.deleteLater()


def _deck_region_width(window: MainWindow) -> int:
    """Width of the tabified inspector region.

    The panes share one region, but only the *raised* tab reports the true
    region width — the others report a stale minimum. So the region width is the
    max over the visible, docked panes, independent of which tab is raised.
    """
    widths = [
        dock.width()
        for dock in (window._dock_fit, window._dock_fourier, window._dock_fit_parameters)
        if dock.isVisible() and not dock.isFloating()
    ]
    return max(widths, default=0)


def test_inspector_deck_opens_at_controls_fitting_width(shown_window) -> None:
    """The deck must open wide enough to show its controls, not the squeezed min."""
    window, _app = shown_window
    # Comfortably past the clipped width, and up to the adaptive default.
    assert _deck_region_width(window) > _CLIPPED_DOCK_WIDTH
    assert _deck_region_width(window) >= window._inspector_default_width() - 1


def test_inspector_deck_is_widenable(shown_window) -> None:
    """The plot|dock splitter must widen the deck (it was a no-op when maximised).

    Exercised through ``resizeDocks`` on the tabified right-dock group, which
    walks the same relayout path as a manual separator drag. The window is given
    extra room first so the widen is not capped by the central plot's minimum.
    """
    window, app = shown_window
    window.resize(2000, 850)
    _settle(app, 3)
    group = [window._dock_fit, window._dock_fourier, window._dock_fit_parameters]
    before = _deck_region_width(window)
    window.resizeDocks(group, [620, 620, 620], Qt.Orientation.Horizontal)
    _settle(app, 3)
    after = _deck_region_width(window)
    assert after > before
    assert after >= 560  # reached most of the requested 620


def test_reset_layout_restores_controls_fitting_width(shown_window) -> None:
    """View > Reset Layout must restore the same controls-fitting deck width.

    Reset previously resized only ``_dock_fit`` to a stale 340 — a no-op against
    its tabified siblings — leaving the deck inconsistent with the launch
    default. It now routes through the same default-width helper.
    """
    window, app = shown_window
    group = [window._dock_fit, window._dock_fourier, window._dock_fit_parameters]
    window.resizeDocks(group, [_CLIPPED_DOCK_WIDTH] * 3, Qt.Orientation.Horizontal)
    _settle(app, 3)
    window._ui_manager.reset_layout()
    _settle(app)
    assert _deck_region_width(window) >= window._inspector_default_width() - 1


@pytest.mark.parametrize(
    "panel_attr",
    ["_fit_parameters_panel", "_maxent_panel"],
)
def test_clipped_panels_keep_a_scroll_fallback_in_the_dock(shown_window, panel_attr) -> None:
    """Each panel that clipped lives under a QScrollArea, so a narrow dock scrolls.

    Together with the controls-fitting default width and the widenable splitter,
    this is the design intent: at a usable width the controls show outright, and
    below it the panel scrolls rather than clipping silently.
    """
    window, _app = shown_window
    panel = getattr(window, panel_attr)
    # Each panel embeds its own QScrollArea, so when the dock is dragged below
    # the panel's natural width its controls scroll into reach rather than
    # clipping silently. This is the narrow-width fallback beneath the
    # controls-fitting default width and the widenable splitter.
    assert panel.findChild(QScrollArea) is not None
