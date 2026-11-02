"""Three-panel comparison of bunching (rebin) factors ×1, ×4, ×16.

Drives the practical-guidance section in
:doc:`/reference/data_processing` — illustrates the noise-reduction
vs time-resolution trade-off across three rebin factors. The ×16
panel is intentionally over-bunched relative to the 1.4 MHz Larmor
signal so the visual cost of going too aggressive is obvious.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.simulate import BUILTIN_TEMPLATES, simulate_run
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

from ._base import CaptureContext, Scenario, register


class BunchingComparisonScenario(Scenario):
    name = "bunching_comparison"
    description = "Toolbar counts-first bunching of a seeded TF run at ×1, ×4 and ×16."
    size = (1600, 540)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.mainwindow import MainWindow

        template = replace(
            BUILTIN_TEMPLATES["ideal_pulsed_fb"],
            n_detectors=2,
            n_bins=480,
            bin_width_us=8 / 479,
            t0_bin=0,
            forward_detectors=(1,),
            backward_detectors=(2,),
        ).build()
        field = 100.0
        run = simulate_run(
            template,
            MODELS["Oscillatory"].function,
            {
                "A0": 22.0,
                "frequency": MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * field,
                "phase": 0.0,
                "Lambda": 0.1,
                "baseline": 0.4,
            },
            total_events=7.2e6,
            seed=107,
            run_number=9501,
        )
        run.metadata.update(field=field, temperature=20.0, title="TF count-binning demo")
        raw_dataset = reduce_run(run, ReductionSettings())
        window = MainWindow()
        panels: list[QPixmap] = []
        try:
            window._data_browser.add_dataset(raw_dataset)
            window._on_dataset_selected(raw_dataset.run_number)
            for factor, caption in (
                (1, "×1 (raw, 16.7 ns)"),
                (4, "×4 (66.7 ns)"),
                (16, "×16 (267 ns)"),
            ):
                window._view_bunch_spin.setValue(factor)
                analysis = window._plot_panel.get_analysis_dataset(raw_dataset)
                assert analysis.n_points == raw_dataset.n_points // factor
                assert raw_dataset.run.grouping["bunching_factor"] == 1
                # The separate capture panel receives the already reduced curve.
                panels.append(_grab_plot(replace(analysis, run=None), caption=caption))
        finally:
            window._clear_dirty()
            window.close()

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
        0,
        0,
        pix.width(),
        36,
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
