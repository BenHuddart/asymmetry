"""Helpers for driving :class:`FitParametersPanel`'s chip rail and card stack.

The panel's y selection is a strip of checkable chips and its Subplots plot is
one figure per card, so a test that used to select a row in the Y-parameter
table and read ``panel._figure.axes`` now goes through these three helpers.
They are deliberately thin: ``select_params`` performs the user's gesture
(check exactly these chips) and forces the redraw the debounce timer would
otherwise deliver on the next event-loop turn.
"""

from __future__ import annotations

from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.widgets.parameter_card import ParameterCard

__all__ = ["axes_for", "card", "select_params"]


def select_params(panel: FitParametersPanel, names: list[str]) -> None:
    """Check exactly the chips in *names* and redraw, as a user's clicks would."""
    for name, chip in panel._y_chips.items():
        chip.setChecked(name in names)
    panel._refresh_plot()


def card(panel: FitParametersPanel, name: str) -> ParameterCard:
    """The card plotting *name* (its chip must be checked)."""
    return panel._card_stack.card(name)


def axes_for(panel: FitParametersPanel, name: str):
    """The axes *name* is drawn on, in either plot mode.

    In Subplots that is its card's own axes; in Overlay the panel tags each
    axis of the shared figure with the parameter it carries ("main" when one
    axis carries several).
    """
    if panel._plot_mode() == "Subplots":
        return card(panel, name).figure.axes[0]
    for axes in panel._figure.axes:
        if panel._axes_tag_map.get(id(axes)) == name:
            return axes
    return panel._figure.axes[0]


def forbid_trend_optimization(monkeypatch) -> None:
    """Fail actual history restoration if a frontend or optimizer starts a fit."""
    import pytest
    from iminuit import Minuit

    def unexpected(*args, **kwargs):
        pytest.fail("restoring width history must not fit or optimize")

    for target in (
        "asymmetry.core.fitting.parameter_models.fit_parameter_model",
        "asymmetry.core.fitting.parameter_models.global_fit_parameter_model",
        "asymmetry.gui.panels.fit_parameters_panel.fit_parameter_model",
        "asymmetry.gui.panels.model_fit_dialog.fit_parameter_model",
        "asymmetry.gui.panels.cross_group_config.global_fit_parameter_model",
        "asymmetry.gui.panels.cross_group_fit_dialog.global_fit_parameter_model",
    ):
        monkeypatch.setattr(target, unexpected)
    for method in ("migrad", "simplex", "hesse"):
        monkeypatch.setattr(Minuit, method, unexpected)
