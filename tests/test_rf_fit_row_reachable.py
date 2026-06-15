"""RED target for branch ``fix/rf-fit-row-clipping``.

Round-3 GUI finding (Benzene RF, ``_findings/windows-gui/Round3_progress.md``): the
#105 "RF resonance (A_µ, A_p)" fit row (v_RF · A_µ₀ · A_p₀ · **Fit RF resonance**
button) is laid out horizontally and is wider than the inspector deck. It lives in
``ALCScanView`` whose analysis area is a ``QScrollArea(setWidgetResizable=True)`` —
i.e. vertical scroll only — so the over-wide row is **clipped** and the Fit button
overflows past the right screen edge at maximised width, with no way to reach it.

Contract: the RF-resonance fit controls must be reachable at the deck's default
width (~360 px). Fixed by wrapping the seed spinboxes onto one labeled row each
(``_build_rf_group``), so the group's content no longer demands more than the deck
width. This test asserts that contract holds.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPushButton

from asymmetry.gui.panels.alc_panel import ALCScanView

_DECK_DEFAULT_WIDTH = 360  # INSPECTOR_DOCK_DEFAULT_WIDTH (#104)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _find_rf_fit_button(view) -> QPushButton:
    for btn in view.findChildren(QPushButton):
        if "RF resonance" in btn.text():
            return btn
    raise AssertionError("Fit RF resonance button not found in ALCScanView")


def test_rf_fit_controls_fit_the_deck_width(app):
    view = ALCScanView()
    try:
        button = _find_rf_fit_button(view)
        # The container holding the RF fit rows (seeds + button) must not require
        # more than the deck's default width, or it clips with no horizontal scroll.
        container = button.parentWidget()
        required = container.minimumSizeHint().width()
        assert required <= _DECK_DEFAULT_WIDTH, (
            f"RF fit row needs {required}px > deck {_DECK_DEFAULT_WIDTH}px — it clips "
            "(ALCScanView's scroll is vertical-only). Wrap the row or add h-scroll."
        )
    finally:
        view.close()
        view.deleteLater()
