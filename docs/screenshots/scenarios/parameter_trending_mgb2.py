"""Parameter-trending output: MgB₂ σ(T) two-gap superconductor fit.

Renders the σ(T) data produced by ``make_mgb2_sigma_t`` together with the
``SC_TwoGap_SS`` parametric-model fit curve as a single composite plot.
This shows what the parameter-trending workflow produces: a science-ready
σ(T) → λ(T) inference for a multiband superconductor (Niedermayer
*et al.* PRB 65, 094512 (2002); Sonier, Brewer & Kiefl, Rev. Mod.
Phys. 72, 769 (2000)).

Marked ``requires_fit = False`` because the fit curve here is computed
analytically from the SC_TwoGap_SS evaluator at the literature-default
parameters rather than re-fitted at capture time — that keeps the
scenario fast and dev-safe while still demonstrating the workflow output.
The shared image is referenced from both ``parameter_trending.rst`` and
``sc_penetration_depth.rst``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from ..data import make_mgb2_sigma_t
from ._base import CaptureContext, Scenario, register


class ParameterTrendingMgb2Scenario(Scenario):
    name = "parameter_trending_mgb2"
    description = "σ(T) plot of synthetic MgB₂ data with the two-gap SC_TwoGap_SS fit overlaid."
    size = (1200, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from asymmetry.core.fitting.sc.models import sc_two_gap_ss

        from .. data.archetypes import TC_MGB2_K

        payload = make_mgb2_sigma_t()
        t_data = payload["T_K"]
        sigma_data = payload["sigma"]
        sigma_err = payload["sigma_err"]

        t_dense = np.linspace(0.5, TC_MGB2_K - 0.5, 400)
        sigma_fit = sc_two_gap_ss(
            t_dense,
            sigma_0=1.25,
            Tc=TC_MGB2_K,
            gap_ratio_1=1.1,
            gap_ratio_2=2.3,
            weight=0.55,
            sigma_bg=0.03,
        )

        figure = Figure(figsize=(9.5, 5.5), dpi=120, tight_layout=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.errorbar(
            t_data,
            sigma_data,
            yerr=sigma_err,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=0.8,
            markersize=5,
            label="σ(T) data (MgB₂ synthetic)",
        )
        ax.plot(
            t_dense,
            sigma_fit,
            color="#d62728",
            lw=1.6,
            label="SC_TwoGap_SS fit (gap ratios 1.1, 2.3; w = 0.55)",
        )
        ax.axvline(TC_MGB2_K, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.text(
            TC_MGB2_K - 0.4, ax.get_ylim()[1] * 0.92, "T$_c$ = 36 K",
            color="grey", ha="right", fontsize=10,
        )
        ax.set_xlabel("Temperature  T (K)")
        ax.set_ylabel("Muon depolarization rate  σ (μs⁻¹)")
        ax.set_title("Parameter trending: MgB₂ σ(T) — two-gap superconductor fit")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.25)

        canvas = FigureCanvasQTAgg(figure)
        canvas.draw()
        pix = QPixmap(canvas.size())
        canvas.render(pix)

        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(ParameterTrendingMgb2Scenario())
