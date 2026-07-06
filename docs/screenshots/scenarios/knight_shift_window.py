"""Knight shift analysis window on a two-site angle-scan series.

Opens the standalone :class:`~asymmetry.gui.windows.knight_shift_window.
KnightShiftWindow` directly (it needs only an optional parent) and feeds it a
synthetic :class:`~asymmetry.core.fitting.knight_analysis.KnightAnalysisInput`
snapshot built from the same two-site angle scan used by
:doc:`/workflows/knight_shift_angle` (shared contact shift, opposite-sign axial
shifts, crossing at both magic angles). This is the window opened from the
Analysis menu's **Knight shift analysis…** entry or the Fit Parameters panel's
**Knight shift window…** button; the sidebar shows the *Source*, *Conversion*,
*Branches*, and *Model fit* sections with the applied-field reference already
converting both frequency components, a completed joint K(θ) fit applied (run
synchronously with the real core :func:`~asymmetry.core.fitting.
knight_analysis.run_joint_fit`, then injected via ``restore_state`` +
``set_snapshot`` rather than the off-thread button so the capture stays
deterministic), and the plot showing both realigned K(θ) branches with their
fitted curves and assignment-swap markers.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ._base import Scenario, register, _process_events_for


def _build_snapshot():
    import numpy as np

    from asymmetry.core.fitting.knight_analysis import KnightAnalysisInput, KnightPoint

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
    return KnightAnalysisInput(
        x_key="angle",
        x_label="Angle (°)",
        components=(("frequency", "frequency"), ("frequency_2", "frequency")),
        points=points,
        source_label="YBCO angle scan (synthetic)",
        batch_id="knight-shift-window-demo",
        group_id=None,
    )


class KnightShiftWindowScenario(Scenario):
    name = "knight_shift_window"
    description = (
        "Knight shift analysis window: Source/Conversion/Branches/Model fit "
        "sidebar and a completed joint K(theta) fit for a two-site angle scan."
    )
    size = (980, 810)
    requires_fit = True  # runs the real iminuit-backed joint K(theta) fit

    def build(self) -> QWidget:
        from asymmetry.core.fitting.knight_analysis import (
            KnightAnalysisState,
            evaluate,
            run_joint_fit,
        )
        from asymmetry.core.fitting.knight_shift import KnightShiftConfig
        from asymmetry.gui.windows.knight_shift_window import KnightShiftWindow

        snapshot = _build_snapshot()

        # Run the real joint fit synchronously (off the TaskRunner's worker
        # thread, which the capture driver's single-threaded event loop can't
        # wait on deterministically) and inject the resulting state directly,
        # the same way a restored project would.
        config = KnightShiftConfig(enabled=True)
        result = evaluate(snapshot, config)
        joint = run_joint_fit(result, model_name="KnightAnisotropy", max_iter=25)
        state = KnightAnalysisState(config=config, x_key="angle", joint=joint).to_dict()

        window = KnightShiftWindow()
        window.restore_state(state)
        window.set_snapshot(snapshot)
        _process_events_for(milliseconds=150)
        return window


register(KnightShiftWindowScenario())
