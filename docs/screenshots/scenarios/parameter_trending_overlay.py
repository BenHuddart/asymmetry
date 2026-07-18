"""Parameter-trending panel: two-field σ(T) multi-series overlay.

Captures the real :class:`~asymmetry.gui.panels.fit_parameters_panel.
FitParametersPanel` with two recorded σ(T) series — a 400 G and a 200 G
transverse-field scan of a high-Tc cuprate — overlaid on one axis (colour =
series, with a legend flagging the active series). This is the field-comparison
presentation the multi-series overlay exists for: the 200 G plateau sits below
the 400 G one, the pancake-vortex field dependence of the London second moment
(cf. the BiSCCO corpus example).

No fit runs at capture time (``requires_fit = False``): the point is the overlay
itself, driven through the public :meth:`FitParametersPanel.select_series` API.
Referenced from ``parameter_trending.rst``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


def _sigma_of_t(plateau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic σ(T): a low-T plateau collapsing to ~0 at Tc ≈ 107 K."""
    temperature = np.array([10.0, 30.0, 50.0, 70.0, 85.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    tc = 107.0
    reduced = np.clip(1.0 - (temperature / tc) ** 2, 0.0, None)
    sigma = plateau * np.sqrt(reduced) + 0.06 * (temperature < tc)
    sigma_err = 0.02 * sigma + 0.005
    return temperature, sigma, sigma_err


class ParameterTrendingOverlayScenario(Scenario):
    name = "parameter_trending_overlay"
    description = (
        "Fit Parameters trending panel: σ(T) at 400 G and 200 G overlaid as two "
        "colour-coded series with a legend."
    )
    size = (1240, 760)
    requires_fit = False

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        series_specs = [
            ("bscco-400", "σ(T) — 400 G", 400.0, 1.16),
            ("bscco-200", "σ(T) — 200 G", 200.0, 0.91),
        ]
        series_payload = []
        for batch_id, name, field, plateau in series_specs:
            temperature, sigma, sigma_err = _sigma_of_t(plateau)
            row_dicts = [
                {
                    "run_number": 1200 + int(field) + i,
                    "run_label": f"{temp:.0f} K",
                    "field": field,
                    "temperature": float(temp),
                    "values": {"sigma": float(sigma[i])},
                    "errors": {"sigma": float(sigma_err[i])},
                }
                for i, temp in enumerate(temperature)
            ]
            series_payload.append((batch_id, name, row_dicts))

        panel = FitParametersPanel()
        panel.load_representation_series(series_payload, select_id="bscco-400")
        # Public overlay API — the equivalent of Shift-clicking the second pill.
        panel.select_series(["bscco-400", "bscco-200"])
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        widget._refresh_plot()
        _process_events_for(milliseconds=200)


register(ParameterTrendingOverlayScenario())
