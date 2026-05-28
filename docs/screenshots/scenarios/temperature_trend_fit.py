"""Order-parameter ν(T) plot for the EuO temperature scan.

Companion screenshot to :doc:`/user_guide/workflows/temperature_scan_magnetism`.
Synthesises the EuO ν(T) trend that a user would extract by fitting
each run of the ZF temperature scan, then overlays the Landau power-
law fit ν(T) = ν₀·(1 − T/Tc)^β. Produced via matplotlib (same approach
as ``parameter_trending_mgb2.py``) since the parameter-trending GUI
panel is laborious to populate from synthetic per-run results.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from ..data.archetypes import TC_EUO_K
from ._base import CaptureContext, Scenario, register


class TemperatureTrendFitScenario(Scenario):
    name = "temperature_trend_fit"
    description = "EuO order-parameter ν(T) trend with Landau power-law fit overlaid."
    size = (1200, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        # Synthetic ν(T) values (same generator the EuO scan uses).
        nu_0_mhz = 28.0
        beta = 0.40
        T_data = np.array([30.0, 50.0, 65.0, 69.0])
        # The 69 K point is right at Tc → ν is near zero in practice.
        order_data = np.clip(1.0 - T_data / TC_EUO_K, 0.0, None) ** beta
        rng = np.random.default_rng(31)
        nu_data = nu_0_mhz * order_data + rng.normal(0.0, 0.4, T_data.shape)
        nu_err = np.full_like(T_data, 0.6)
        # Suppress the last point's noisy "above zero" if needed.
        nu_data[T_data >= TC_EUO_K] = 0.0
        nu_err[T_data >= TC_EUO_K] = 0.3

        T_fit = np.linspace(0.5, TC_EUO_K - 0.1, 500)
        nu_fit = nu_0_mhz * (1.0 - T_fit / TC_EUO_K) ** beta

        figure = Figure(figsize=(9.0, 5.5), dpi=120, tight_layout=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.errorbar(
            T_data, nu_data, yerr=nu_err,
            fmt="o", color="#1f77b4", ecolor="#1f77b4",
            elinewidth=0.8, markersize=6,
            label="ν(T) from per-run fits (synthetic EuO)",
        )
        ax.plot(
            T_fit, nu_fit, color="#d62728", lw=1.6,
            label=fr"Landau fit ν$_0$(1 − T/T$_c$)$^β$  "
                  fr"(ν$_0$ = {nu_0_mhz:.1f} MHz, T$_c$ = {TC_EUO_K:.0f} K, β = {beta:.2f})",
        )
        ax.axvline(TC_EUO_K, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.set_xlabel("Temperature T (K)")
        ax.set_ylabel(r"Precession frequency ν$_\mu$ (MHz)")
        ax.set_title("Order-parameter temperature scan: EuO ν(T)")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.0, 90.0)
        ax.set_ylim(-1.0, max(nu_0_mhz, nu_data.max()) * 1.1)

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


register(TemperatureTrendFitScenario())
