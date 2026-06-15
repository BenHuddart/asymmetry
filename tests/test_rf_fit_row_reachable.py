"""RED target for branch ``fix/rf-fit-row-clipping`` (now widened to cover B5).

Round-3 GUI findings (``_findings/windows-gui/Round3_progress.md``):

* **B2** — Benzene RF: the #105 "RF resonance (A_µ, A_p)" fit row (v_RF · A_µ₀ ·
  A_p₀ · **Fit RF resonance**) was wider than the inspector deck and clipped. Fixed
  by wrapping its seed spinboxes (``_build_rf_group``).
* **B5** — ALC in TCNQ: the *same* clipping affects the **Baseline** ("Fit baseline",
  ``_build_baseline_group``) and **Peaks** ("Fit peaks", ``_build_peaks_group``)
  rows. Both append their Fit button to the right end of an over-wide horizontal row,
  and ``ALCScanView``'s analysis area is a ``QScrollArea`` with the horizontal
  scrollbar **explicitly off** (``ScrollBarAlwaysOff``) — so the Fit buttons overflow
  past the right edge with no way to reach them. Measured required widths (pre-fix):
  Fit baseline ≈ 386 px, Fit peaks ≈ 536 px, both > the ~360 px deck. The RF row,
  after the B2 fix, is ≈ 224 px (passes).

Contract: **every** ALC fit-control container (Baseline, Peaks, RF resonance) must be
reachable at the deck's default width (~360 px), i.e. its ``minimumSizeHint().width()``
must not exceed the deck. Today RF passes (regression guard) while Baseline and Peaks
are RED until each Fit button is wrapped onto its own line (or h-scroll is enabled).
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

# Every ALC fit action lives at the right end of a horizontal row inside the
# vertical-only analysis scroll area. If any row's container needs more than the
# deck width, that Fit button clips off the right edge unreachably.
_FIT_BUTTON_LABELS = ["Fit baseline", "Fit peaks", "Fit RF resonance"]


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _find_fit_button(view, label: str) -> QPushButton:
    for btn in view.findChildren(QPushButton):
        if btn.text() == label:
            return btn
    raise AssertionError(f"{label!r} button not found in ALCScanView")


@pytest.mark.parametrize("label", _FIT_BUTTON_LABELS)
def test_alc_fit_controls_fit_the_deck_width(app, label):
    view = ALCScanView()
    try:
        button = _find_fit_button(view, label)
        # The container holding this fit row (controls + button) must not require
        # more than the deck's default width, or it clips with no horizontal scroll.
        container = button.parentWidget()
        required = container.minimumSizeHint().width()
        assert required <= _DECK_DEFAULT_WIDTH, (
            f"{label!r} row needs {required}px > deck {_DECK_DEFAULT_WIDTH}px — it "
            "clips (ALCScanView's scroll is vertical-only; horizontal scrollbar is "
            "off). Wrap the Fit button onto its own line or enable horizontal scroll."
        )
    finally:
        view.close()
        view.deleteLater()
