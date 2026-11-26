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
from PySide6.QtWidgets import QApplication, QWidget

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
    include_y_errors = True
    stabilise_errors = False

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        import numpy as np

        from asymmetry.core.fitting.trend_result import ErrorMode
        from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

        # Synthetic ν(T) from the EuO order parameter, one point per scan run
        # (deterministic: fixed seed). ν(T) = ν0 (1 − T/Tc)^β below Tc, 0 above.
        nu_0_mhz = 28.0
        beta = 0.40
        temperature = np.array([30.0, 40.0, 50.0, 58.0, 63.0, 66.0, 68.0])
        order = np.clip(1.0 - temperature / TC_EUO_K, 0.0, None) ** beta
        rng = np.random.default_rng(31)
        nu = nu_0_mhz * order + rng.normal(0.0, 0.25, temperature.shape)
        nu_err = np.full_like(temperature, 0.4) if self.include_y_errors else None
        if self.stabilise_errors:
            nu_err[0] = 0.08

        dialog = ModelFitDialog(
            parameter_name="frequency",
            x_key="temperature",
            x_values=temperature,
            y_values=nu,
            y_errors=nu_err,
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        if self.stabilise_errors:
            picker = dialog._error_mode_combo
            picker.setCurrentIndex(picker.findData(ErrorMode.COLUMN_STABILIZED.value))
            dialog._show_residuals_check.setChecked(True)
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
                and dialog._fit.ranges[0].result.success
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
        if result is None or not result.success:
            raise RuntimeError("EuO order-parameter fit did not converge for the screenshot")
        if self.stabilise_errors and not result.column_error_floor_applied:
            raise RuntimeError("The stabilised trend screenshot requires an active error floor")
        if (
            not self.include_y_errors
            and "No χ² quality verdict" not in dialog._quality_label.text()
        ):
            raise RuntimeError("The unit-weight trend result must show its quality explanation")

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


class TrendModelFitUnitWeightsScenario(TrendModelFitDialogScenario):
    name = "trend_model_fit_unit_weights"
    description = "Column trend fit with assumed unit y errors and its quality explanation"
    include_y_errors = False


register(TrendModelFitUnitWeightsScenario())


class TrendModelFitStabilisedScenario(TrendModelFitDialogScenario):
    name = "trend_model_fit_stabilised"
    description = "Trend fit with the column error floor and weighted residuals"
    stabilise_errors = True


register(TrendModelFitStabilisedScenario())


class ZeroResidualDegreesFitScenario(Scenario):
    name = "fit_zero_residual_degrees"
    description = "Bounded one-point asymmetry fit with the raw χ² and unavailable reduced χ²"
    size = (480, 160)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.engine import FitEngine
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit.single_tab import SingleFitTab

        model = CompositeModel(["Constant"])
        parameters = ParameterSet([Parameter(model.param_names[0], 0.5, min=0, max=1)])
        result = FitEngine().fit_arrays([0.0], [5.0], [0.1], model.function, parameters)
        if not result.success or result.dof != 0:
            raise RuntimeError("The single-point screenshot fit must converge with ν = 0")
        tab = SingleFitTab()
        tab._show_fit_summary(result, tag="Fit ✓", tone="ok", headline="Run 101")
        card = tab._results_card
        if "unavailable" not in card.content_html():
            raise RuntimeError("The single-point fit card must report the unavailable reduced χ²")
        card.setParent(None)
        tab.deleteLater()
        return card


register(ZeroResidualDegreesFitScenario())


class TrendGaussianWidthConversionScenario(Scenario):
    """Show the explicit Gaussian-width conversion at its Brandt fit consumer."""

    name = "trend_gaussian_width_conversion"
    description = "Brandt fit after explicit Gaussian-width and error conversion"
    size = (1180, 720)
    requires_fit = True

    def capture(self, ctx: CaptureContext) -> Path:
        import numpy as np

        from asymmetry.core.fitting.axis_transforms import AxisTransform
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.core.fitting.sc import GaussianWidth
        from asymmetry.core.fitting.sc.models import brandt_field_width_sigma
        from asymmetry.core.fitting.sc.trend_width import (
            TrendWidthContext,
            WidthSource,
            model_width_sources,
        )
        from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

        field = np.linspace(100.0, 12000.0, 16)
        sigma = brandt_field_width_sigma(field, 220.0, 2.0)
        sigma /= GaussianWidth.GAUSSIAN_COMPONENT.value
        transform = AxisTransform.custom("sqrt(2)*x")
        values, errors = transform.apply(sigma, np.full_like(sigma, 0.01))
        source = model_width_sources(CompositeModel.from_expression("Gaussian"))["sigma"]
        context = TrendWidthContext(
            "sigma", (WidthSource.from_dict(source),) * len(field), transform
        )
        model = ParameterCompositeModel.from_expression("SC_Brandt_VortexLattice")
        parameters = ParameterSet(
            [
                Parameter("lambda_ab", 215.0, min=1.0),
                Parameter("Bc2", 2.0, fixed=True),
                Parameter("sigma_bg", 0.0, fixed=True),
            ]
        )
        fit = ParameterModelFit(
            "sigma",
            "field",
            ranges=[ModelFitRange(float(field.min()), float(field.max()), model, parameters)],
        )
        dialog = ModelFitDialog(
            "sigma",
            "field",
            field,
            values,
            errors,
            existing_fit=fit,
            width_context=context,
            x_label="B (G)",
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        dialog._run_fit(0)
        _wait_until(lambda: not dialog._fit_in_progress, timeout_ms=20000)
        _wait_until(lambda: not dialog._preview_active, timeout_ms=10000)
        result = dialog._fit.ranges[0].result
        if result is None or not result.success:
            raise RuntimeError("Converted Gaussian-width Brandt fit did not converge")
        if not np.isclose(result.parameters["lambda_ab"].value, 220.0, rtol=1e-5):
            raise RuntimeError("Converted Gaussian-width fit did not recover the control")
        _pump_events(250)
        path = ctx.output_dir / f"{self.name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(path)):
            raise RuntimeError("Failed to save the converted-width screenshot")
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return path


register(TrendGaussianWidthConversionScenario())
