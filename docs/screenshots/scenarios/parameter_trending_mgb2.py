"""Parameter-trending panel: MgB₂ σ(T) two-gap superconductor fit.

Captures the real :class:`~asymmetry.gui.panels.fit_parameters_panel.
FitParametersPanel` — the **Fit Parameters** trending workspace — with the
MgB₂ σ(T) series from :func:`make_mgb2_sigma_t` loaded as trend points and the
``SC_TwoGap_SS`` parametric-model fit overlaid as the red trend curve. This is
what the parameter-trending workflow produces on screen: a science-ready
σ(T) → λ(T) inference for a multiband superconductor (Niedermayer *et al.*
PRB 65, 094512 (2002); Sonier, Brewer & Kiefl, Rev. Mod. Phys. 72, 769 (2000)).

The series is injected through the panel's ``load_representation_series`` pull
entry point (the same route ``MainWindow`` uses when a batch series is
recorded), then the trend curve is produced by running the real core
``fit_parameter_model`` machinery — the exact minimiser the panel's Model Fit
dialog calls — synchronously and injecting the resulting ``ParameterModelFit``,
so the overlay is a genuine fit rather than an analytic curve. Marked
``requires_fit = True`` because that iminuit fit runs at capture time (it trips
numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3). The shared image is
referenced from ``parameter_trending.rst``, ``sc_penetration_depth.rst``, and
``workflows/superconductor_penetration_depth.rst``.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_mgb2_sigma_t
from ._base import Scenario, _process_events_for, register


def _build_sc_two_gap_fit(payload: dict):
    """Fit ``SC_TwoGap_SS`` to the MgB₂ σ(T) via the real core minimiser.

    Returns a converged ``ParameterModelFit`` (parameter ``sigma`` vs
    temperature) ready to inject into the panel. Deterministic: no RNG, a fixed
    literature-seeded start, single least-squares solve.
    """
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    t = payload["T_K"]
    sigma = payload["sigma"]
    sigma_err = payload["sigma_err"]

    model = ParameterCompositeModel(["SC_TwoGap_SS"])
    # Seeded at the MgB₂ alpha-model decomposition (Niedermayer et al. 2002);
    # bounds keep the two-gap ratios / weight physical.
    seed = ParameterSet(
        [
            Parameter("sigma_0", value=1.2, min=0.0, max=5.0),
            Parameter("Tc", value=35.0, min=10.0, max=50.0),
            Parameter("gap_ratio_1", value=1.0, min=0.2, max=5.0),
            Parameter("gap_ratio_2", value=2.5, min=0.2, max=6.0),
            Parameter("weight", value=0.5, min=0.0, max=1.0),
            Parameter("sigma_bg", value=0.02, min=-0.5, max=0.5),
        ]
    )
    result = fit_parameter_model(
        t,
        sigma,
        sigma_err,
        model,
        seed,
        x_min=float(t.min()),
        x_max=float(t.max()),
    )
    if not result.success:
        raise RuntimeError("SC_TwoGap_SS σ(T) fit did not converge for the screenshot")

    fit_range = ModelFitRange(
        x_min=float(t.min()),
        x_max=float(t.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    return ParameterModelFit(
        parameter_name="sigma",
        x_key="temperature",
        ranges=[fit_range],
        active=True,
    )


class ParameterTrendingMgb2Scenario(Scenario):
    name = "parameter_trending_mgb2"
    description = (
        "Fit Parameters trending panel: MgB₂ σ(T) trend points with the fitted "
        "two-gap SC_TwoGap_SS curve overlaid."
    )
    size = (1240, 760)
    requires_fit = True  # runs the real iminuit-backed SC_TwoGap_SS trend fit

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        payload = make_mgb2_sigma_t()
        t = payload["T_K"]
        sigma = payload["sigma"]
        sigma_err = payload["sigma_err"]

        # One trend point per temperature — the shape a per-run batch σ fit
        # feeds into the trending panel. field held constant so the panel infers
        # temperature as the trend abscissa.
        batch_id = "mgb2-sigma-t"
        row_dicts = [
            {
                "run_number": 40000 + i,
                "run_label": f"{temp:.1f} K",
                "field": 0.0,
                "temperature": float(temp),
                "values": {"sigma": float(sigma[i])},
                "errors": {"sigma": float(sigma_err[i])},
            }
            for i, temp in enumerate(t)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "σ(T) — MgB₂", row_dicts)],
            select_id=batch_id,
        )

        # Run the real trend fit and inject it as the active model fit for σ.
        fit = _build_sc_two_gap_fit(payload)
        panel._model_fits["sigma"] = fit
        panel._sync_active_group_state()
        # Reflect the attached fit in the chrome: the Y row's button relabels to
        # "Model Fit*" the same way it does after a real Model Fit dialog run.
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        # Trigger the trend redraw now the panel is shown, then wait for the
        # off-thread overlay-curve sampler (TaskRunner worker) to populate the
        # cache so the fitted curve is present in the grab, not a blank overlay.
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


register(ParameterTrendingMgb2Scenario())
