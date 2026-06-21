"""Angle-dependent Knight shift K(θ) with a joint two-site anisotropy fit.

Companion screenshot to :doc:`/workflows/knight_shift_angle`. Synthesises the
K(θ) values a user extracts by fitting the precession frequency at each crystal
angle for two muon sites, then runs the *real* joint K(θ) anisotropy fit
(:func:`asymmetry.core.fitting.angular_assignment.fit_assigned_angular_curves`)
and overlays the two fitted branches. The sites share an isotropic shift but
have opposite-sign axial shifts, so their branches cross where the axial term
(3cos²θ − 1)/2 vanishes (the magic angle, 54.7°, and again at 125.3°) — the case
the per-angle component assignment exists to resolve. Rendered with matplotlib,
like the other parameter-trend figures (``temperature_trend_fit``,
``parameter_trending_mgb2``), since the trend GUI panel is laborious to populate
from synthetic per-run results.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from ._base import CaptureContext, Scenario, register


class KnightShiftAngleScenario(Scenario):
    name = "knight_shift_angle"
    description = "Angle-dependent Knight shift K(θ) with a joint two-site anisotropy fit."
    size = (1200, 720)
    requires_fit = True  # runs the iminuit-backed joint K(θ) fit

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from asymmetry.core.fitting.angular_assignment import fit_assigned_angular_curves

        # Two sites: a shared isotropic shift, opposite-sign axial shifts. K (%).
        k_iso = 0.40
        k_ax_a, k_ax_b = -0.30, +0.30
        angles = np.arange(0.0, 180.0, 15.0)  # 0..165°, 12 points

        def k_of(theta_deg: np.ndarray, k_ax: float) -> np.ndarray:
            axial = (3.0 * np.cos(np.radians(theta_deg)) ** 2 - 1.0) / 2.0
            return k_iso + k_ax * axial

        rng = np.random.default_rng(73)
        sigma = 0.02
        meas_a = k_of(angles, k_ax_a) + rng.normal(0.0, sigma, angles.shape)
        meas_b = k_of(angles, k_ax_b) + rng.normal(0.0, sigma, angles.shape)
        err = np.full_like(angles, sigma)

        # The real joint fit: one curve per site, with per-angle assignment.
        result = fit_assigned_angular_curves(
            angles.tolist(),
            np.column_stack([meas_a, meas_b]).tolist(),
            np.column_stack([err, err]).tolist(),
            model_name="KnightAnisotropy",
            max_iter=25,
        )

        theta_fit = np.linspace(angles.min(), angles.max(), 400)
        axial_fit = (3.0 * np.cos(np.radians(theta_fit)) ** 2 - 1.0) / 2.0
        colours = ["#1f77b4", "#d62728"]
        labels = ["Site 1", "Site 2"]

        figure = Figure(figsize=(9.0, 5.5), dpi=120, tight_layout=True)
        ax = figure.add_subplot(1, 1, 1)
        for m in range(len(result.curves)):
            kiso = result.curves[m].parameters["K_iso"].value
            kax = result.curves[m].parameters["K_ax"].value
            ax.errorbar(
                angles,
                result.curve_values[m],
                yerr=result.curve_errors[m],
                fmt="o",
                color=colours[m],
                ecolor=colours[m],
                elinewidth=0.8,
                markersize=6,
                label=fr"{labels[m]}: K$_\mathrm{{iso}}$={kiso:.2f}%, "
                fr"K$_\mathrm{{ax}}$={kax:+.2f}%",
            )
            ax.plot(theta_fit, kiso + kax * axial_fit, color=colours[m], lw=1.6)

        for crossing in (54.7, 125.3):
            ax.axvline(crossing, color="grey", ls="--", lw=0.7, alpha=0.55)
        ax.annotate(
            "branches cross where\n3cos²θ − 1 = 0\n(assignment resolves the labels)",
            xy=(54.7, k_iso),
            xytext=(70.0, 0.80),
            fontsize=8,
            color="dimgrey",
            ha="center",
        )
        ax.set_xlabel("Crystal angle θ (°)")
        ax.set_ylabel("Knight shift K (%)")
        ax.set_title("Angle-dependent Knight shift: joint two-site K(θ) fit")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-6.0, 171.0)
        ax.set_ylim(0.0, 0.95)

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


register(KnightShiftAngleScenario())
