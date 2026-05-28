"""Excel-style column filter dialog alongside the data browser.

``QWidget.grab`` does not capture top-level child windows, so we render
the dialog and the populated data browser separately and composite the
two pixmaps onto a single canvas. The resulting screenshot shows both
in spatial context without depending on OS window chrome.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from ..data import make_euo_tf_tscan
from ._base import CaptureContext, Scenario, register


class DataBrowserFilterScenario(Scenario):
    """Composite screenshot: data browser + filter dialog side by side."""

    name = "data_browser_filter"
    description = "Data browser populated with a T-scan plus the column filter dialog."
    # ``size`` is unused for this scenario; capture() composes the output
    # directly from two child widget grabs.
    size = (1200, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401 — overrides parent
        from asymmetry.gui.panels.data_browser import DataBrowserPanel, FilterDialog

        browser = DataBrowserPanel()
        for dataset in make_euo_tf_tscan():
            browser.add_dataset(dataset)
        browser.resize(680, 540)
        browser.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        browser.show()
        _pump_events(160)

        unique_temps = sorted(
            {f"{int(ds.metadata['temperature'])}" for ds in make_euo_tf_tscan()},
            key=int,
        )
        # Select the three runs at or below Tc=69 K — the ordered ferromagnetic
        # state where the spontaneous precession dominates the signal.
        selected = {"30", "50", "65"}
        dialog = FilterDialog("T (K)", unique_temps, selected)
        dialog.resize(320, 420)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(160)

        browser_pix = browser.grab()
        dialog_pix = dialog.grab()

        canvas = _compose(browser_pix, dialog_pix)

        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not canvas.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        browser.close()
        browser.deleteLater()
        dialog.close()
        dialog.deleteLater()
        _pump_events(50)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _compose(browser_pix: QPixmap, dialog_pix: QPixmap) -> QPixmap:
    """Composite browser + dialog pixmaps side-by-side with a caption."""
    padding = 24
    gap = 36
    caption_height = 32

    canvas_w = browser_pix.width() + dialog_pix.width() + gap + padding * 2
    canvas_h = (
        max(browser_pix.height(), dialog_pix.height() + caption_height)
        + padding * 2
    )

    canvas = QPixmap(canvas_w, canvas_h)
    canvas.fill(QColor("#f8fafc"))

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    browser_y = padding + (canvas_h - 2 * padding - browser_pix.height()) // 2
    painter.drawPixmap(padding, browser_y, browser_pix)

    dialog_x = padding + browser_pix.width() + gap
    caption_y = padding
    dialog_y = caption_y + caption_height

    font = QFont(painter.font())
    font.setBold(True)
    font.setPointSizeF(15)
    painter.setFont(font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(
        dialog_x,
        caption_y,
        dialog_pix.width(),
        caption_height,
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        "Filter — T (K)",
    )

    painter.drawPixmap(dialog_x, dialog_y, dialog_pix)
    painter.end()
    return canvas


register(DataBrowserFilterScenario())
