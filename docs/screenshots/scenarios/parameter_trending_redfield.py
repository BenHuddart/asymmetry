"""Parameter-trending panel: Redfield linearisation via axis transforms.

Captures the real :class:`~asymmetry.gui.panels.fit_parameters_panel.
FitParametersPanel` with a longitudinal-field relaxation-rate scan
:math:`\\lambda(B)` plotted as the **Redfield** linearisation — the Y axis
transformed to ``1/x  (reciprocal)`` and the X axis to ``x²  (square)``, so
:math:`1/\\lambda` versus :math:`(\\mu_0 H)^2` is a straight line whose slope and
intercept give the field-fluctuation rate and width. A ``Linear`` model fit runs
on the transformed coordinates (the transform feeds the fit, not just the plot),
and the high-field saturated point is excluded from the trend so it sits off the
line. This is the µSR presentation the axis-transform feature exists for
(Baker, Lord & Prabhakaran, *J. Phys.: Condens. Matter* **23**, 306001 (2011)).

The series is injected through the panel's ``load_representation_series`` entry
point, the transforms are set on the panel, and the trend line is produced by
the real core ``fit_parameter_model`` minimiser on the transformed arrays — the
exact machinery the Model Fit dialog drives — then injected as a
``ParameterModelFit``. Marked ``requires_fit = True``: the iminuit-backed Linear
fit runs at capture time. Referenced from ``parameter_trending.rst``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


def _redfield_lambda_of_field() -> dict:
    """Deterministic λ(B) following a Redfield law 1/λ = a + b·B² (G units).

    Returns field (Gauss), λ (µs⁻¹) and its error; the top point is saturated
    (pulled above the line) and flagged for exclusion from the trend.
    """
    field_g = np.array(
        [2000.0, 5000.0, 8000.0, 12000.0, 16000.0, 20000.0, 25000.0, 30000.0, 35000.0, 38000.0]
    )
    a = 0.42  # µs (H → 0 intercept)
    b = 3.0e-9  # µs · G⁻²
    inv_lambda = a + b * field_g**2
    lam = 1.0 / inv_lambda
    lam[-1] *= 0.82  # saturated high-field point sits off the Redfield line
    lam_err = 0.02 * lam
    return {"field_g": field_g, "lam": lam, "lam_err": lam_err}


def _build_redfield_linear_fit(payload: dict):
    """Fit ``Linear`` to the *transformed* (1/λ vs B²) plateau via the core minimiser."""
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    field_g = payload["field_g"][:-1]  # exclude the saturated point from the fit
    lam = payload["lam"][:-1]
    lam_err = payload["lam_err"][:-1]

    x2, _ = AxisTransform.preset("square").apply(field_g)
    inv_lam, inv_lam_err = AxisTransform.preset("reciprocal").apply(lam, lam_err)

    model = ParameterCompositeModel(["Linear"])
    seed = ParameterSet(
        [
            Parameter("m", value=3.0e-9, min=0.0, max=1.0e-6),
            Parameter("b", value=0.4, min=0.0, max=5.0),
        ]
    )
    result = fit_parameter_model(
        x2, inv_lam, inv_lam_err, model, seed, x_min=float(x2.min()), x_max=float(x2.max())
    )
    if not result.success:
        raise RuntimeError("Redfield Linear fit did not converge for the screenshot")

    fit_range = ModelFitRange(
        x_min=float(x2.min()),
        x_max=float(x2.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    return ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[fit_range],
        active=True,
    )


class ParameterTrendingRedfieldScenario(Scenario):
    name = "parameter_trending_redfield"
    description = (
        "Fit Parameters trending panel: Redfield linearisation with the Y axis "
        "transformed to 1/λ and the X axis to B², a Linear fit on the plateau."
    )
    size = (1240, 760)
    requires_fit = True  # runs the real iminuit-backed Linear trend fit

    def build(self) -> QWidget:
        from asymmetry.core.fitting.axis_transforms import AxisTransform
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        payload = _redfield_lambda_of_field()
        field_g = payload["field_g"]
        lam = payload["lam"]
        lam_err = payload["lam_err"]

        batch_id = "redfield-lambda-b"
        row_dicts = [
            {
                "run_number": 9031 + i,
                "run_label": f"{b_g / 10000.0:.2f} T",
                "field": float(b_g),
                "temperature": 15.0,
                "values": {"Lambda": float(lam[i])},
                "errors": {"Lambda": float(lam_err[i])},
                # The saturated top point stays visible but is excluded from the
                # trend (and so from the Redfield line).
                "include_in_trend": i < len(field_g) - 1,
            }
            for i, b_g in enumerate(field_g)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "λ(B) — Ca₃Co₂O₆", row_dicts)],
            select_id=batch_id,
        )

        # The Redfield linearisation: reciprocal Y, square X.
        panel._set_axis_transform("y", AxisTransform.preset("reciprocal"))
        panel._set_axis_transform("x", AxisTransform.preset("square"))

        # Inject the real Linear fit computed on the transformed plateau, tagged
        # with the active transform so its overlay is drawn (not suppressed).
        fit = _build_redfield_linear_fit(payload)
        panel._model_fits["Lambda"] = fit
        panel._model_fit_transform_sig["Lambda"] = panel._transform_signature()
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        widget._refresh_plot()
        _wait_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20000,
        )
        _process_events_for(milliseconds=200)


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    """Pump the event loop until *predicate* holds or the timeout elapses."""
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


register(ParameterTrendingRedfieldScenario())
