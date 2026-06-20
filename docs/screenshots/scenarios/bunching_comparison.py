"""Three-panel comparison of bunching (rebin) factors ×1, ×4, ×16.

Drives the practical-guidance section in
:doc:`/reference/data_processing` — illustrates the noise-reduction
vs time-resolution trade-off across three rebin factors. The ×16
panel is intentionally over-bunched relative to the 1.4 MHz Larmor
signal so the visual cost of going too aggressive is obvious.
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


class BunchingComparisonScenario(Scenario):
    name = "bunching_comparison"
    description = "Three-panel composite of ×1, ×4, ×16 bunched data."
    size = (1600, 540)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        raw_dataset = make_generic_tf_for_processing()
        panels: list[QPixmap] = []
        for factor, caption in (
            (1, "×1 (raw, 16.7 ns)"),
            (4, "×4 (66.7 ns)"),
            (16, "×16 (267 ns)"),
        ):
            if factor == 1:
                ds = raw_dataset
            else:
                t, v, e = rebin(
                    raw_dataset.time, raw_dataset.asymmetry,
                    raw_dataset.error, factor=factor,
                )
                ds = MuonDataset(
                    time=t, asymmetry=v, error=e,
                    metadata={**raw_dataset.metadata,
                              "title": f"TF 100G bunched ×{factor}"},
                )
            panels.append(_grab_plot(ds, caption=caption))

        canvas = _compose(panels)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not canvas.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _pump_events(40)
        return out_path


def _grab_plot(dataset: MuonDataset, *, caption: str) -> QPixmap:
    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel(domain="time")
    panel.resize(480, 440)
    panel.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    panel.show()
    _pump_events(120)
    panel.plot_dataset(dataset)
    _pump_events(120)
    pix = panel.grab()
    panel.close()
    panel.deleteLater()

    captioned = QPixmap(pix.width(), pix.height() + 36)
    captioned.fill(QColor("#f8fafc"))
    painter = QPainter(captioned)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = QFont(painter.font())
    font.setBold(True)
    font.setPointSizeF(13)
    painter.setFont(font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(
        0, 0, pix.width(), 36,
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        caption,
    )
    painter.drawPixmap(0, 36, pix)
    painter.end()
    return captioned


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _compose(panels: list[QPixmap]) -> QPixmap:
    padding = 20
    gap = 20
    total_w = sum(p.width() for p in panels) + gap * (len(panels) - 1) + padding * 2
    total_h = max(p.height() for p in panels) + padding * 2
    canvas = QPixmap(total_w, total_h)
    canvas.fill(QColor("#f8fafc"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    x = padding
    for p in panels:
        painter.drawPixmap(x, padding, p)
        x += p.width() + gap
    painter.end()
    return canvas


register(BunchingComparisonScenario())
