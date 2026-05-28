"""Side-by-side raw vs rebinned visualisation of a low-statistics TF run.

Demonstrates the classic noise-reduction trade-off of binning: combining
*N* consecutive bins reduces the per-bin uncertainty by ~1/√N at the cost
of time resolution. We render the same plot panel twice — once with raw
data, once after a ×8 rebin — and composite the two pixmaps onto a single
caption-aware canvas (same composition pattern as
:mod:`docs.screenshots.scenarios.data_browser_filter`).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.rebin import rebin

from ..data import make_generic_tf_for_processing
from ._base import CaptureContext, Scenario, register


class DataProcessingRebinScenario(Scenario):
    name = "data_processing_rebin"
    description = "Composite plot of a TF dataset before and after an ×8 rebin."
    size = (1400, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.panels.plot_panel import PlotPanel

        raw_dataset = make_generic_tf_for_processing()
        t_reb, v_reb, e_reb = rebin(
            raw_dataset.time, raw_dataset.asymmetry, raw_dataset.error, factor=4
        )
        rebinned_dataset = MuonDataset(
            time=t_reb,
            asymmetry=v_reb,
            error=e_reb,
            metadata={**raw_dataset.metadata, "title": "TF 100G rebinned ×4"},
        )

        raw_pix = _grab_plot(raw_dataset, caption="Raw (bin width = 16.7 ns)")
        reb_pix = _grab_plot(rebinned_dataset, caption="Rebinned ×4 (bin width = 66.7 ns)")

        canvas = _compose(raw_pix, reb_pix)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not canvas.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _pump_events(50)
        return out_path


def _grab_plot(dataset: MuonDataset, *, caption: str) -> QPixmap:
    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel(domain="time")
    panel.resize(600, 520)
    panel.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    panel.show()
    _pump_events(120)
    panel.plot_dataset(dataset)
    _pump_events(120)

    pix = panel.grab()
    panel.close()
    panel.deleteLater()

    # Tag the pixmap with a caption using a small QPainter overlay.
    captioned = QPixmap(pix.width(), pix.height() + 40)
    captioned.fill(QColor("#f8fafc"))
    painter = QPainter(captioned)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = QFont(painter.font())
    font.setBold(True)
    font.setPointSizeF(14)
    painter.setFont(font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(
        0, 0, pix.width(), 40,
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        caption,
    )
    painter.drawPixmap(0, 40, pix)
    painter.end()
    return captioned


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _compose(left: QPixmap, right: QPixmap) -> QPixmap:
    padding = 24
    gap = 24
    canvas_w = left.width() + right.width() + padding * 2 + gap
    canvas_h = max(left.height(), right.height()) + padding * 2
    canvas = QPixmap(canvas_w, canvas_h)
    canvas.fill(QColor("#f8fafc"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.drawPixmap(padding, padding, left)
    painter.drawPixmap(padding + left.width() + gap, padding, right)
    painter.end()
    return canvas


register(DataProcessingRebinScenario())
