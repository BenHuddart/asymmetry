"""Knight shift analysis window on a two-site angle-scan series.

Opens the standalone :class:`~asymmetry.gui.windows.knight_shift_window.
KnightShiftWindow` directly (it needs only an optional parent) and feeds it a
synthetic :class:`~asymmetry.core.fitting.knight_analysis.KnightAnalysisInput`
snapshot built from the same two-site angle scan used by
:doc:`/workflows/knight_shift_angle` (shared contact shift, opposite-sign axial
shifts, crossing at both magic angles). This is the window opened from the
Analysis menu's **Knight shift analysis…** entry or the Fit Parameters panel's
**Knight shift window…** button; the sidebar shows the *Source*, *Conversion*,
and *Branches* sections with the applied-field reference already converting
both frequency components, and the plot shows both K(θ) branches with their
crossing markers.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ._base import Scenario, register, _process_events_for


class KnightShiftWindowScenario(Scenario):
    name = "knight_shift_window"
    description = (
        "Knight shift analysis window: Source/Conversion/Branches sidebar and "
        "K(theta) plot with crossing markers for a two-site angle scan."
    )
    size = (980, 620)

    def build(self) -> QWidget:
        import numpy as np

        from asymmetry.core.fitting.knight_analysis import KnightAnalysisInput, KnightPoint
        from asymmetry.gui.windows.knight_shift_window import KnightShiftWindow

        # Same two-site scan as the knight_shift_angle workflow figure: shared
        # contact shift, opposite-sign axial shifts, crossing at both magic
        # angles (54.7 deg and 125.3 deg). Frequencies are expressed in MHz so
        # the Applied field reference (nu_ref = gamma_mu * B) converts them.
        gamma_mu_mhz_per_g = 0.0135538  # MHz/G
        field_g = 2000.0
        nu_ref = gamma_mu_mhz_per_g * field_g
        k_iso = 0.0040
        k_ax_a, k_ax_b = -0.0030, 0.0030
        angles = np.arange(0.0, 180.0, 15.0)  # 0..165 deg, 12 points

        def nu_of(theta_deg: np.ndarray, k_ax: float) -> np.ndarray:
            axial = (3.0 * np.cos(np.radians(theta_deg)) ** 2 - 1.0) / 2.0
            k = k_iso + k_ax * axial
            return nu_ref * (1.0 + k)

        rng = np.random.default_rng(73)
        sigma_nu = 0.00003 * nu_ref  # matches the 0.02%-K noise of knight_shift_angle.py
        nu_a = nu_of(angles, k_ax_a) + rng.normal(0.0, sigma_nu, angles.shape)
        nu_b = nu_of(angles, k_ax_b) + rng.normal(0.0, sigma_nu, angles.shape)

        points = tuple(
            KnightPoint(
                run_number=20000 + i,
                run_label=f"Run {20000 + i}",
                x=float(theta),
                field_gauss=field_g,
                values={"frequency": float(nu_a[i]), "frequency_2": float(nu_b[i])},
                errors={"frequency": float(sigma_nu), "frequency_2": float(sigma_nu)},
            )
            for i, theta in enumerate(angles)
        )
        snapshot = KnightAnalysisInput(
            x_key="angle",
            x_label="Angle (°)",
            components=(("frequency", "frequency"), ("frequency_2", "frequency")),
            points=points,
            source_label="YBCO angle scan (synthetic)",
            batch_id="knight-shift-window-demo",
            group_id=None,
        )

        window = KnightShiftWindow()
        window.set_snapshot(snapshot)
        _process_events_for(milliseconds=150)
        return window


register(KnightShiftWindowScenario())
