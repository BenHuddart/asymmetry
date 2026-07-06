"""Trend model-fit dialog fitting an EuO order parameter ν(T).

Drives the real :class:`~asymmetry.gui.panels.model_fit_dialog.ModelFitDialog`
on a synthetic EuO spontaneous-precession-frequency trend ν(T) crossing the
Curie point (Tc = 69 K), the same archetype the parameter-trending page uses.
The dialog defaults to the ``OrderParameter`` component (an order-parameter
observable versus temperature), so we drive its live ``_run_fit`` to
convergence and grab the split view: controls and range cards on the left, the
live preview with the converged curve on the right.

Companion screenshot to the "Fitting a trend model" section of
:doc:`/reference/parameter_trending`. Marked ``requires_fit = True`` because it
runs a real iminuit fit at capture time.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data.archetypes import TC_EUO_K
from ._base import CaptureContext, Scenario, register


class TrendModelFitDialogScenario(Scenario):
    name = "trend_model_fit_dialog"
    description = (
        "Trend model-fit dialog: OrderParameter fit of an EuO ν(T) trend, "
        "with range cards, parameter table, and the live preview curve."
    )
    # Keep the width comfortably above the dialog's 900px preview-collapse
    # threshold so the split view (controls + preview) is captured intact.
    size = (1180, 720)
    requires_fit = True

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np

        from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

        # Synthetic ν(T) from the EuO order parameter, one point per scan run
        # (deterministic: fixed seed). ν(T) = ν0 (1 − T/Tc)^β below Tc, 0 above.
        nu_0_mhz = 28.0
        beta = 0.40
        temperature = np.array([30.0, 40.0, 50.0, 58.0, 63.0, 66.0, 68.0])
        order = np.clip(1.0 - temperature / TC_EUO_K, 0.0, None) ** beta
        rng = np.random.default_rng(31)
        nu = nu_0_mhz * order + rng.normal(0.0, 0.25, temperature.shape)
        nu_err = np.full_like(temperature, 0.4)

        dialog = ModelFitDialog(
            parameter_name="frequency",
            x_key="temperature",
            x_values=temperature,
            y_values=nu,
            y_errors=nu_err,
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(200)

        # Drive the live fit on the (single) default OrderParameter range, then
        # poll to completion rather than sleeping — the fit and the preview
        # sample both run off the GUI thread. A 7-point fit is milliseconds; the
        # correctness of the wait is what keeps placeholder junk out of the shot.
        dialog._run_fit(0)
        _wait_until(
            lambda: (
                not dialog._fit_in_progress
                and dialog._fit.ranges[0].result is not None
                and getattr(dialog._fit.ranges[0].result, "success", False)
            ),
            timeout_ms=20000,
        )
        # The solid/converged preview curve is sampled off-thread on the
        # active-range refresh; wait for that sampler to settle too.
        _wait_until(
            lambda: not dialog._preview_active,
            timeout_ms=10000,
        )
        _pump_events(250)

        result = dialog._fit.ranges[0].result
        if result is None or not getattr(result, "success", False):
            raise RuntimeError("EuO order-parameter fit did not converge for the screenshot")

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    """Pump the event loop until *predicate* holds or the timeout elapses."""
    elapsed = 0
    while elapsed < timeout_ms:
        QApplication.processEvents()
        if predicate():
            return
        _pump_events(poll_ms)
        elapsed += poll_ms


register(TrendModelFitDialogScenario())
