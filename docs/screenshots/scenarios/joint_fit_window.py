"""Joint-fit window: two series, two models, a suggested shared table.

Opens the standalone :class:`~asymmetry.gui.windows.joint_fit_window.
JointFitWindow` directly (``docs/plans/joint-fit.md``) and feeds it two
synthetic :class:`~asymmetry.gui.windows.joint_fit_window.JointSeriesEntry`
records — the motivating example from the plan: a sample crossing a phase
transition, fitted with an oscillating model in the magnetically ordered
phase (``Oscillatory * Exponential + Constant``, runs 101-102) and a plain
relaxation in the paramagnetic phase (``Exponential + Constant``, runs
201-202). Both series classify their overall amplitude (``A_1``) and their
background (``A_bg``) as Global; every other parameter (``frequency``,
``phase``, the two series' own ``Lambda``) is Local, so it never appears on
the shared table.

Both series are ticked (the window's own overlap rule and eligibility list
are irrelevant here — the entries are fabricated eligible from the start),
and the window's real ``_on_suggest_clicked`` is called once, synchronously,
so the shared table shows both suggestion tiers for real:

* ``A_bg`` — the same full name, same unit, Global in both models — is
  proposed **exact** and ticked by default.
* ``A_1`` — the same full name and unit, Global in both, but owned by a
  different component in each model (the product ``Oscillatory *
  Exponential`` in the ordered phase, a bare ``Exponential`` in the
  paramagnetic one) — is proposed **candidate** and left unticked. This is
  the initial-asymmetry caveat the docs page calls out: each model exposes
  its own amplitude as one parameter, but that does not make the two
  amplitudes the same physical quantity, so the tiering leaves the call to
  the user.

No fit runs at capture time (``requires_fit = False``): the window only
needs its picker ticked and its shared table suggested, never a converged
result, so nothing here depends on iminuit or numpy's fitting stack.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


def _build_entries():
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.representation.base import RepresentationType
    from asymmetry.gui.windows.joint_fit_window import JointSeriesEntry

    ordered_model = CompositeModel.from_expression("Oscillatory * Exponential + Constant")
    paramagnetic_model = CompositeModel.from_expression("Exponential + Constant")

    ordered = JointSeriesEntry(
        batch_id="series-ordered",
        label="Ordered phase",
        rep_type=RepresentationType.TIME_FB_ASYMMETRY,
        model_text="Oscillatory * Exponential + Constant",
        members=(101, 102),
        status="2/2",
        blocked_reason="",
        model=ordered_model,
        roles={
            "A_1": "global",
            "frequency": "local",
            "phase": "local",
            "Lambda": "local",
            "A_bg": "global",
        },
        recipe={
            "parameters": [
                {"name": "A_1", "value": 0.20, "bounds": "0,1"},
                {"name": "frequency", "value": 6.0, "bounds": "0,inf"},
                {"name": "phase", "value": 0.0, "bounds": "-inf,inf"},
                {"name": "Lambda", "value": 0.30, "bounds": "0,inf"},
                {"name": "A_bg", "value": 0.0, "bounds": "-inf,inf"},
            ],
        },
    )
    paramagnetic = JointSeriesEntry(
        batch_id="series-paramagnetic",
        label="Paramagnetic phase",
        rep_type=RepresentationType.TIME_FB_ASYMMETRY,
        model_text="Exponential + Constant",
        members=(201, 202),
        status="2/2",
        blocked_reason="",
        model=paramagnetic_model,
        roles={"A_1": "global", "Lambda": "local", "A_bg": "global"},
        recipe={
            "parameters": [
                {"name": "A_1", "value": 0.18, "bounds": "0,1"},
                {"name": "Lambda", "value": 0.15, "bounds": "0,inf"},
                {"name": "A_bg", "value": 0.0, "bounds": "-inf,inf"},
            ],
        },
    )
    return [ordered, paramagnetic]


class JointFitWindowScenario(Scenario):
    name = "joint_fit_window"
    description = (
        "Joint-fit window with an ordered-phase oscillating series and a "
        "paramagnetic relaxing series ticked, and a suggested shared table "
        "showing both tiers (A_bg exact/ticked, A_1 candidate/unticked)."
    )
    size = (1050, 480)
    requires_fit = False

    def build(self) -> QWidget:
        from asymmetry.gui.windows.joint_fit_window import JointFitWindow

        entries = _build_entries()

        window = JointFitWindow()
        window.set_providers(lambda: entries, lambda _batch_id: [])
        window.start_new(entries[0].rep_type)
        # Tick both series directly — the picker's own checkbox is exercised
        # by the GUI test suite; the screenshot only needs the end state.
        window._checked = [entry.batch_id for entry in entries]
        window.refresh_series()
        window._on_suggest_clicked()
        # The per-series combo columns default to a narrow interactive width
        # that clips the second series' header ("Paramagnetic phase"); widen
        # them to their content, exactly as a user dragging the header would.
        window._shared_table.resizeColumnsToContents()
        _process_events_for(milliseconds=150)
        return window


register(JointFitWindowScenario())
