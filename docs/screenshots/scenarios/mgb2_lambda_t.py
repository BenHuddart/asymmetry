"""Penetration depth λ_L(T) for MgB₂ derived from σ(T).

Companion screenshot to
:doc:`/user_guide/workflows/superconductor_penetration_depth`. The σ(T)
trend from ``make_mgb2_sigma_t`` is converted to a London penetration
depth using Asymmetry's canonical
:func:`asymmetry.core.fitting.sc.constants.sigma_to_lambda_nm`, with the
fitted background subtracted before the conversion. The plot is rendered
as a matplotlib figure (same pattern as ``parameter_trending_mgb2``)
because the parameter-trending GUI panel exposes σ(T) but does not yet
have a built-in σ→λ conversion view.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from ..data import make_mgb2_sigma_t
from ._base import CaptureContext, Scenario, register


class Mgb2LambdaTScenario(Scenario):
    name = "mgb2_lambda_t"
    description = (
        "λ_L(T) for synthetic MgB₂ data, converted from σ(T) using the "
        "Brandt triangular-lattice formula."
    )
    size = (1200, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from asymmetry.core.fitting.sc.constants import sigma_to_lambda_nm
        from asymmetry.core.fitting.sc.models import sc_two_gap_ss

        from ..data.archetypes import TC_MGB2_K

        payload = make_mgb2_sigma_t()
        t_data = payload["T_K"]
        sigma_data = payload["sigma"]
        sigma_err = payload["sigma_err"]

        # Fit-recovered parameters (see workflow text). Background sigma is
        # subtracted before the conversion so λ_L reflects the
        # superconducting contribution alone.
        sigma_bg = 0.03
        sigma_sc = np.clip(sigma_data - sigma_bg, 1e-6, None)
        sigma_sc_err = sigma_err

        # Propagate the error linearly through the inverse-square root.
        lambda_data = sigma_to_lambda_nm(sigma_sc)
        lambda_err = 0.5 * lambda_data * sigma_sc_err / sigma_sc

        # Smooth model curve from the SC_TwoGap_SS fit at the literature
        # decomposition (recovered values are close to these; see prose).
        t_dense = np.linspace(0.5, TC_MGB2_K - 0.5, 400)
        sigma_dense = sc_two_gap_ss(
            t_dense,
            sigma_0=1.25,
            Tc=TC_MGB2_K,
            gap_ratio_1=1.1,
            gap_ratio_2=2.3,
            weight=0.55,
            sigma_bg=sigma_bg,
        )
        sigma_dense_sc = np.clip(sigma_dense - sigma_bg, 1e-6, None)
        lambda_dense = sigma_to_lambda_nm(sigma_dense_sc)

        figure = Figure(figsize=(9.5, 5.5), dpi=120, tight_layout=True)
        ax = figure.add_subplot(1, 1, 1)
        ax.errorbar(
            t_data,
            lambda_data,
            yerr=lambda_err,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=0.8,
            markersize=5,
            label="λ$_L$(T) from σ(T) data (Brandt inversion)",
        )
        ax.plot(
            t_dense,
            lambda_dense,
            color="#d62728",
            lw=1.6,
            label="SC_TwoGap_SS gap-model curve, inverted",
        )
        ax.axvline(TC_MGB2_K, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.text(
            TC_MGB2_K - 0.4,
            ax.get_ylim()[1] * 0.92,
            "T$_c$ = 36 K",
            color="grey",
            ha="right",
            fontsize=10,
        )
        ax.set_xlabel("Temperature  T (K)")
        ax.set_ylabel("London penetration depth  λ$_L$ (nm)")
        ax.set_title(
            "MgB₂ penetration depth λ$_L$(T) derived from σ(T) "
            "via the Brandt formula"
        )
        ax.legend(loc="upper left", frameon=True)
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


register(Mgb2LambdaTScenario())
