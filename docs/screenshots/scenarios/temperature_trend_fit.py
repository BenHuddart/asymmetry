"""Parameter-trending panel: EuO order-parameter ν(T) with a Landau fit.

Companion screenshot to :doc:`/workflows/temperature_scan_magnetism`. Captures
the real :class:`~asymmetry.gui.panels.fit_parameters_panel.FitParametersPanel`
trending workspace with the EuO spontaneous-precession-frequency trend ν(T)
loaded as trend points and the ``OrderParameter`` (Landau power-law)
ν(T) = y₀·(1 − (T/Tc)^α)^β fit overlaid as the trend curve.

The six-temperature scan crosses the Curie point (Tc = 69 K): the three points
at and above Tc (69/73/90 K) carry ν = 0 — the spontaneous frequency vanishes in
the paramagnetic state — and sit on the axis, while the sub-Tc points trace the
ordered moment. The series is injected through the panel's
``load_representation_series`` pull entry point, then the overlay is produced by
running the real core ``fit_parameter_model`` minimiser — the exact machinery
the panel's Model Fit dialog calls, with the same ``suggest_trend_seeds``
data-aware seeding — synchronously and injecting the resulting
``ParameterModelFit``. Marked ``requires_fit = True`` because that iminuit fit
runs at capture time (it trips numpy ≥ 2.3 in dev environments; CI keeps
numpy < 2.3).
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data.archetypes import TC_EUO_K
from ._base import Scenario, _process_events_for, register


def _make_euo_nu_t():
    """Synthetic EuO ν(T): six scan temperatures, ν → 0 at and above Tc.

    Deterministic (fixed seed). ν(T) = ν0 (1 − T/Tc)^β below Tc, 0 at/above it —
    the archetype the temperature-scan workflow describes (30/50/65 K ordered;
    69/73/90 K paramagnetic).
    """
    import numpy as np

    nu_0_mhz = 28.0
    beta = 0.40
    temperature = np.array([30.0, 50.0, 65.0, 69.0, 73.0, 90.0])
    order = np.clip(1.0 - temperature / TC_EUO_K, 0.0, None) ** beta
    rng = np.random.default_rng(31)
    nu = nu_0_mhz * order + rng.normal(0.0, 0.25, temperature.shape)
    nu_err = np.full_like(temperature, 0.4)
    # At/above Tc the spontaneous precession frequency is identically zero.
    nu[temperature >= TC_EUO_K] = 0.0
    nu_err[temperature >= TC_EUO_K] = 0.3
    return temperature, nu, nu_err


def _build_order_parameter_fit(temperature, nu, nu_err):
    """Fit the ``OrderParameter`` (Landau) model to ν(T) via the core minimiser.

    Returns a converged ``ParameterModelFit`` (parameter ``frequency`` vs
    temperature) ready to inject. Seeds come from ``suggest_trend_seeds`` — the
    same data-aware seeding the Model Fit dialog uses for critical-temperature
    trend components — so the recovered (y0, Tc, β) match the input archetype.
    """
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
        suggest_trend_seeds,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = ParameterCompositeModel(["OrderParameter"])
    seeds = suggest_trend_seeds(model, temperature, nu)
    params = ParameterSet(
        [
            Parameter(name=pname, value=float(seeds.get(pname, model.param_defaults[pname])))
            for pname in model.param_names
        ]
    )
    result = fit_parameter_model(
        temperature,
        nu,
        nu_err,
        model,
        params,
        x_min=float(temperature.min()),
        x_max=float(temperature.max()),
    )
    if not result.success:
        raise RuntimeError("EuO OrderParameter ν(T) fit did not converge for the screenshot")

    fit_range = ModelFitRange(
        x_min=float(temperature.min()),
        x_max=float(temperature.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    return ParameterModelFit(
        parameter_name="frequency",
        x_key="temperature",
        ranges=[fit_range],
        active=True,
    )


class TemperatureTrendFitScenario(Scenario):
    name = "temperature_trend_fit"
    description = (
        "Fit Parameters trending panel: EuO order-parameter ν(T) trend points "
        "with the fitted Landau (OrderParameter) curve overlaid."
    )
    size = (1240, 760)
    requires_fit = True  # runs the real iminuit-backed OrderParameter trend fit

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temperature, nu, nu_err = _make_euo_nu_t()

        batch_id = "euo-nu-t"
        row_dicts = [
            {
                "run_number": 41000 + i,
                "run_label": f"{temp:.0f} K",
                "field": 0.0,
                "temperature": float(temp),
                "values": {"frequency": float(nu[i])},
                "errors": {"frequency": float(nu_err[i])},
            }
            for i, temp in enumerate(temperature)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — EuO", row_dicts)],
            select_id=batch_id,
        )

        fit = _build_order_parameter_fit(temperature, nu, nu_err)
        panel._model_fits["frequency"] = fit
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


register(TemperatureTrendFitScenario())
