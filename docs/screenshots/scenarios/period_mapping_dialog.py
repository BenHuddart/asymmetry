"""Period → red/green/ignore mapping dialog on a two-period silicon run.

Companion screenshot to :doc:`/workflows/photomusr_silicon_periods`. Drives
:class:`~asymmetry.gui.windows.period_mapping_dialog.PeriodMappingDialog` with
the two per-period datasets of the photo-μSR silicon run
(:func:`~docs.screenshots.data.archetypes.make_silicon_photomusr_periods`):
period 1 is *red* (laser ON) and period 2 is *green* (laser OFF), each with its
own good-frame count (≈ 28,108). The dialog's defaults follow WiMDA — the first
period maps to **Red** and the second to **Green** — so the captured state shows
exactly the red = period 1 / green = period 2 convention the page describes.

No fit runs, so this scenario is cheap and needs no ``requires_fit`` flag.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_silicon_photomusr_periods
from ._base import Scenario, register


class PeriodMappingDialogScenario(Scenario):
    name = "period_mapping_dialog"
    description = (
        "Period mapping dialog on a two-period silicon photo-μSR run, with the "
        "first period mapped to Red and the second to Green."
    )
    size = (640, 150)

    def build(self) -> QWidget:
        from asymmetry.gui.windows.period_mapping_dialog import PeriodMappingDialog

        periods = make_silicon_photomusr_periods()
        return PeriodMappingDialog(periods)


register(PeriodMappingDialogScenario())
