"""Parameter-trending panel: a phase-owned series with its range and boundary.

Captures the real :class:`~asymmetry.gui.panels.fit_parameters_panel.
FitParametersPanel` showing one phase's local-parameter trend (the damped
oscillation frequency of the cold phase of the synthetic two-phase ZF scan,
:func:`make_two_phase_zf_tscan`), injected as a ``PhaseDecoration`` through
``load_representation_series``' ``phase_by_id`` — the same route
``MainWindow._refresh_trend_panel`` uses for a series bound to a real phase
data group. The series' own colour matches the phase swatch, the phase's run
range is shaded, and its upper boundary is drawn as a dashed line with a
faint uncertainty band, exactly as :meth:`FitParametersPanel._draw_phase_band`
renders a real phase in the live app.

No fit runs at capture time (``requires_fit = False``): the frequency values
plotted are the same deterministic values the archetype used to build the
phase's synthetic data, not a live analysis.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_two_phase_zf_tscan
from ._base import Scenario, _process_events_for, register


class ParameterTrendingPhaseScenario(Scenario):
    name = "parameter_trending_phase"
    description = (
        "Fit Parameters trending panel showing a phase-owned series with its "
        "range shaded and its boundary drawn as a dashed line."
    )
    size = (1080, 860)
    requires_fit = False

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, PhaseDecoration
        from asymmetry.gui.utils.phase_colors import phase_color

        datasets = make_two_phase_zf_tscan()
        cold = datasets[0:5]
        tc_planted_k = 20.0

        row_dicts = []
        for dataset in cold:
            temperature = float(dataset.metadata["temperature"])
            order = (1.0 - temperature / tc_planted_k) ** 0.35
            frequency = 6.0 * order
            row_dicts.append(
                {
                    "run_number": int(dataset.run_number),
                    "run_label": dataset.run_label,
                    "field": 0.0,
                    "temperature": temperature,
                    "values": {"frequency": frequency},
                    "errors": {"frequency": 0.05 * frequency + 0.02},
                }
            )

        phase = PhaseDecoration(
            color=phase_color(1),
            color_dark=phase_color(1, dark=True),
            ordinal=1,
            name="Phase I",
            axis_key="temperature",
            range=(4.0, 18.0),
            lower=None,
            upper=(21.0, 3.0),
        )

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("phase-1", "Phase I · 4.0 – 18.0 K", row_dicts)],
            select_id="phase-1",
            phase_by_id={"phase-1": phase},
        )
        _process_events_for(milliseconds=200)
        return panel


register(ParameterTrendingPhaseScenario())
