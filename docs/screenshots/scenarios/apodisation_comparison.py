"""Three-up FFT spectrum comparison: no apodisation, Gaussian, Lorentzian.

Companion to :doc:`/user_guide/fourier_analysis` — illustrates how
apodisation trades line-sharpness for sidelobe suppression on the
same TF signal. Uses the existing YBCO vortex-lattice synthetic
dataset so the asymmetric P(B) lineshape is recognisable in each
panel.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_vortex_lattice
from ..data.archetypes import GAMMA_MU_MHZ_PER_G
from ._base import CaptureContext, Scenario, register


class ApodisationComparisonScenario(Scenario):
    name = "apodisation_comparison"
    description = "Three-up FFT spectrum showing none / Gaussian / Lorentzian apodisation."
    size = (1700, 620)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        dataset = make_ybco_vortex_lattice()
        # Subtract mean and compute the FFT magnitude in the
        # asymmetric peak window only. We synthesise three apodised
        # variants directly in numpy so the comparison is reproducible
        # without reaching into the GUI's grouped-FFT pipeline.
        signal = dataset.asymmetry - dataset.asymmetry.mean()
        time = dataset.time
        dt = time[1] - time[0]
        center_mhz = GAMMA_MU_MHZ_PER_G * 2000.0    # 27.1 MHz

        panels: list[QPixmap] = []
        for caption, window in (
            ("No apodisation", np.ones_like(signal)),
            ("Gaussian (σ = 4 μs)",
             np.exp(-0.5 * (time / 4.0) ** 2)),
            ("Lorentzian (τ = 3 μs)",
             np.exp(-time / 3.0)),
        ):
            apodised = signal * window
            spectrum = np.abs(np.fft.rfft(apodised))
            freqs = np.fft.rfftfreq(len(apodised), dt)
            mask = (freqs >= center_mhz - 1.0) & (freqs <= center_mhz + 1.5)

            figure = Figure(figsize=(5.5, 5.0), dpi=120, tight_layout=True)
            ax = figure.add_subplot(1, 1, 1)
            ax.plot(freqs[mask], spectrum[mask], color="#1f77b4", lw=1.4)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("|F(t)|  (arb)")
            ax.set_title(caption, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.25)
            canvas = FigureCanvasQTAgg(figure)
            canvas.draw()
            pix = QPixmap(canvas.size())
            canvas.render(pix)
            panels.append(pix)

        canvas_pix = _compose(panels)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not canvas_pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _compose(panels: list[QPixmap]) -> QPixmap:
    padding = 20
    gap = 24
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


register(ApodisationComparisonScenario())
