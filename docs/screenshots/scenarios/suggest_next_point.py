"""Suggest-next-point dialog: refinement suggestion and model comparison.

Drives the real :class:`~asymmetry.gui.panels.model_fit_dialog.ModelFitDialog`
on a synthetic order-parameter trend (Tc = 100 K), fits the default
``OrderParameter`` model, opens the "Suggest next point" section, and runs a
c-optimal suggestion for ``Tc`` with a precision goal so the events-factor and
MC-calibrated sigma both render (:class:`SuggestNextPointScenario`). A second
scenario continues from the same fit and additionally fits a "Compare
against" alternative (``PowerLaw``), populating the model-discrimination
overlay and AIC evidence line (:class:`SuggestNextPointCompareScenario`).

Companion screenshots to the "Suggest next point" section of
:doc:`/reference/suggest_next_point`. Both marked ``requires_fit = True``
because they run real iminuit fits (and, for the first, a Monte-Carlo
calibration pass) at capture time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ._base import CaptureContext, Scenario, register

_TC_K = 100.0
_SIZE = (1400, 900)


def _build_fitted_dialog() -> Any:
    """Build the dialog on a synthetic order-parameter trend and fit it.

    Shared by both scenarios: synthetic order parameter
    y(T) = y0 (1 - T/Tc)^beta, Tc = 100 K, a coarse-but-adequate scan (9
    points) so the fit is well-conditioned and the suggestion targets the
    informative region just below Tc.
    """
    import numpy as np

    from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

    y0 = 20.0
    beta = 0.35
    temperature = np.array([10.0, 25.0, 40.0, 55.0, 68.0, 78.0, 86.0, 92.0, 97.0])
    order = np.clip(1.0 - temperature / _TC_K, 0.0, None) ** beta
    rng = np.random.default_rng(7)
    y = y0 * order + rng.normal(0.0, 0.15, temperature.shape)
    y_err = np.full_like(temperature, 0.2)

    dialog = ModelFitDialog(
        parameter_name="frequency",
        x_key="temperature",
        x_values=temperature,
        y_values=y,
        y_errors=y_err,
    )
    dialog.resize(*_SIZE)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    dialog.show()
    _pump_events(200)

    # Drive the primary OrderParameter fit to convergence (same
    # poll-to-completion pattern as trend_model_fit_dialog.py).
    dialog._run_fit(0)
    _wait_until(
        lambda: (
            not dialog._fit_in_progress
            and dialog._fit.ranges[0].result is not None
            and getattr(dialog._fit.ranges[0].result, "success", False)
        ),
        timeout_ms=20000,
    )
    _wait_until(lambda: not dialog._preview_active, timeout_ms=10000)
    _pump_events(150)

    result = dialog._fit.ranges[0].result
    if result is None or not getattr(result, "success", False):
        raise RuntimeError("order-parameter fit did not converge for the screenshot")

    # NOTE: the candidate-range fields default to a hardcoded [0, 1] and are
    # only auto-seeded to the measured x span the first time the section is
    # shown for a genuinely fresh range; as of this writing that auto-seed
    # has a latent bug (the "are the fields still at their constructor
    # default" guard compares 0.0 to 1.0, which is never equal, so it never
    # fires) and the range silently stays [0, 1]. Set it explicitly here to
    # the measured span so the screenshots show the feature working as
    # intended rather than that bug.
    dialog._suggest_min_field.setValue(float(temperature.min()))
    dialog._suggest_max_field.setValue(float(temperature.max()))
    return dialog


def _run_suggestion(dialog: Any) -> None:
    """Target Tc (c-optimal) with a precision goal, and wait for calibration."""
    idx = dialog._suggest_target_combo.findData("Tc")
    if idx < 0:
        raise RuntimeError("Tc is not offered as a suggestion target")
    dialog._suggest_target_combo.setCurrentIndex(idx)
    dialog._suggest_goal_edit.setText("1.0")
    dialog._on_suggest_clicked()
    _wait_until(lambda: dialog._last_suggestion is not None, timeout_ms=5000)
    # Wait for the off-thread Monte-Carlo calibration pass so the result line
    # shows the MC-calibrated sigma, not just the analytic one.
    _wait_until(lambda: not dialog._calibration_in_progress, timeout_ms=20000)
    _pump_events(150)


def _grab_and_save(dialog: Any, out_path: Path) -> Path:
    pix = dialog.grab()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not pix.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    return out_path


def _teardown(dialog: Any) -> None:
    dialog.close()
    dialog.deleteLater()
    _pump_events(40)


class SuggestNextPointScenario(Scenario):
    name = "suggest_next_point"
    description = (
        "Suggest next point: c-optimal Tc suggestion with precision goal and "
        "MC-calibrated sigma on a synthetic order-parameter trend."
    )
    size = _SIZE
    requires_fit = True

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        dialog = _build_fitted_dialog()
        _run_suggestion(dialog)
        out_path = _grab_and_save(dialog, ctx.output_dir / f"{self.name}.png")
        _teardown(dialog)
        return out_path


class SuggestNextPointCompareScenario(Scenario):
    name = "suggest_next_point_compare"
    description = (
        "Suggest next point, Compare against: a PowerLaw alternative fitted "
        "for AIC comparison, with the model-discrimination overlay."
    )
    size = _SIZE
    requires_fit = True

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        dialog = _build_fitted_dialog()
        _run_suggestion(dialog)

        # "Compare against": fit PowerLaw as an alternative candidate and
        # capture the discrimination overlay + AIC evidence line.
        compare_idx = dialog._compare_model_combo.findData("PowerLaw")
        if compare_idx < 0:
            raise RuntimeError("PowerLaw is not offered as a compare-against candidate")
        dialog._compare_model_combo.setCurrentIndex(compare_idx)
        dialog._on_compare_fit_clicked()
        _wait_until(lambda: not dialog._compare_fit_in_progress, timeout_ms=20000)
        _wait_until(lambda: dialog._last_discrimination is not None, timeout_ms=5000)
        _pump_events(150)

        out_path = _grab_and_save(dialog, ctx.output_dir / f"{self.name}.png")
        _teardown(dialog)
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


register(SuggestNextPointScenario())
register(SuggestNextPointCompareScenario())
