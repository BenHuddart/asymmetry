"""Acceptance target for branch ``fix/dock-responsive`` (marked xfail until fixed).

Round-2 GUI finding (multiple, see ``_findings/windows-gui/`` + ``GUI_LORE.md``):
the right-hand dock clips at the maximised screen edge (1456 px) and will not
widen via the plot|dock splitter. Consequences observed live:
  * the per-Y-parameter **Model Fit** buttons in ``FitParametersPanel`` are
    off-screen (had to drag a horizontal scrollbar to reach them);
  * the **MaxEnt** Min/Max-frequency and Time Start/End fields in ``MaxEntPanel``
    are off-screen (forced reliance on Auto-window-from-field).

Requirement (design-agnostic): at a *narrow* dock width these panels must keep
their controls reachable — i.e. provide horizontal scrolling (a QScrollArea in
the hierarchy) or fit a sensible default dock width. The fix may choose either;
this test accepts both. It is marked ``xfail`` because no fix exists yet — when
the branch lands, drop the marker so the ratchet keeps it passing.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea

from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.maxent_panel import MaxEntPanel

_NARROW_DOCK_WIDTH = 340  # a realistic right-dock width on a 1456 px screen


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _controls_reachable(widget) -> bool:
    """True if the panel scrolls horizontally or fits the narrow dock width."""
    has_scroll = widget.findChild(QScrollArea) is not None
    fits = widget.minimumSizeHint().width() <= _NARROW_DOCK_WIDTH
    return has_scroll or fits


@pytest.mark.skip(
    reason=(
        "fix/dock-responsive: MANUAL/VISUAL acceptance. The standalone panels "
        "already contain QScrollAreas, so a headless proxy XPASSes — yet the live "
        "controls (Model-Fit buttons, MaxEnt freq/time fields) clip in the "
        "maximised right dock because the dock width is fixed and the plot|dock "
        "splitter will not widen. Root cause is the dock container/splitter, not "
        "the panels. Acceptance: reproduce per GUI_LORE (maximised window, drag "
        "the Y-parameters horizontal scrollbar to find Model-Fit), choose a fix "
        "(horizontal scroll at the dock container, sensible min-width, or a "
        "widenable splitter), then replace this with a regression test that "
        "drives the real dock layout. The _controls_reachable() helper below is "
        "kept as the design intent (scroll OR fit a narrow dock)."
    )
)
@pytest.mark.parametrize("factory", [FitParametersPanel, MaxEntPanel])
def test_dock_panel_controls_reachable_at_narrow_width(app, factory):
    panel = factory()
    try:
        assert _controls_reachable(panel)
    finally:
        panel.close()
        panel.deleteLater()
