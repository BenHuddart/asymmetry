"""A fit completion reads its launch, and the form it launched from is frozen.

Two complementary halves, both pinned here:

1. Every worker fit binds a :class:`FitLaunch` — the model, roles, members and
   fit span it was started with — into its completion, so a result is rendered
   and emitted against what was actually fitted rather than whatever the form
   says when the worker lands.
2. While a fit runs the form that defines it is disabled, so the roles and
   model ``MainWindow`` reads back off the tab when it records the series are
   the launch's own by construction.

The engines are stubbed (a blocking call the test releases) so the freeze can
be observed from the GUI thread mid-fit without running a real minimisation.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.core.fitting.composite import CompositeModel  # noqa: E402
from asymmetry.core.fitting.engine import FitResult  # noqa: E402
from asymmetry.core.fitting.global_search.heuristics import (  # noqa: E402
    is_amplitude_parameter,
)
from asymmetry.core.fitting.grouped_time_domain import (  # noqa: E402
    build_grouped_count_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.gui.panels.fit import global_tab as global_tab_module  # noqa: E402
from asymmetry.gui.panels.fit.global_tab import FitLaunch, GlobalFitTab  # noqa: E402
from asymmetry.gui.panels.fit.single_tab import SingleFitTab  # noqa: E402

#: How long a stubbed engine waits for the test to release it before failing
#: the run rather than hanging the suite.
_RELEASE_TIMEOUT_S = 20.0

_MODEL_A_VALUES = {"A_1": 0.2, "Lambda": 0.4, "A_bg": 0.01}
#: The grouped count model's own per-group nuisances, beside the physics.
_GROUP_NUISANCES = {"N0": 120.0, "background": 2.0, "amplitude": 0.21, "relative_phase": 0.0}
_GROUPED_PHYSICS = {"A_1": 0.2, "field": 151.0, "phase": 0.0}


def _dataset(run_number: int) -> MuonDataset:
    time = np.linspace(0.1, 8.0, 32)
    return MuonDataset(
        time=time,
        asymmetry=0.2 * np.exp(-0.5 * time),
        error=np.full_like(time, 0.01),
        metadata={"run_number": run_number},
    )


def _grouped_dataset(member_key: int, group_id: int) -> MuonDataset:
    time = np.linspace(0.0, 4.0, 16)
    return MuonDataset(
        time=time,
        asymmetry=np.full_like(time, 100.0),
        error=np.full_like(time, 1.0),
        metadata={
            "run_number": member_key,
            "source_run_number": 9,
            "group_id": group_id,
            "grouped_time_domain": True,
        },
    )


def _result(values: dict[str, float]) -> FitResult:
    return FitResult(
        success=True,
        chi_squared=30.0,
        reduced_chi_squared=1.0,
        dof=30,
        parameters=ParameterSet([Parameter(name=k, value=v) for k, v in values.items()]),
        uncertainties=dict.fromkeys(values, 0.01),
    )


def _blocking_series(release: threading.Event, results: dict[int, FitResult]):
    """Stand in for ``fit_asymmetry_series``, held until *release* is set."""

    def _call(*_args, **_kwargs):
        assert release.wait(_RELEASE_TIMEOUT_S), "the test never released the stub engine"
        return SimpleNamespace(results=results, fitted_global=ParameterSet())

    return _call


# ── half 1: the completion reads its launch ─────────────────────────────────


def test_a_batch_renders_the_model_it_was_launched_with(qapp: QApplication, monkeypatch) -> None:
    """Switching the function mid-fit cannot retarget the batch that is running.

    The emitted curves and component names must come from the launch model,
    not from the one the form is showing when the worker lands.
    """
    tab = GlobalFitTab(member_kind="runs")
    release = threading.Event()
    try:
        tab.set_datasets([_dataset(3001), _dataset(3002)])
        model_a = tab._composite_model
        results = {3001: _result(_MODEL_A_VALUES), 3002: _result(_MODEL_A_VALUES)}
        monkeypatch.setattr(
            global_tab_module, "fit_asymmetry_series", _blocking_series(release, results)
        )
        emitted: dict[str, object] = {}
        tab.global_fit_completed.connect(lambda res, _glob: emitted.update(res=res))

        tab._run_global_fit()
        # The user replaces the function while the worker is still running.
        tab._set_composite_model(CompositeModel(["Gaussian", "Constant"], operators=["+"]))
        assert tab._composite_model.param_names != model_a.param_names

        release.set()
        assert tab.wait_for_fit()

        _result_payload, (t_fit, y_fit), components = emitted["res"][3001]
        assert y_fit == pytest.approx(model_a.function(t_fit, **_MODEL_A_VALUES))
        assert [name for name, _curve in components] == list(model_a.component_names)
    finally:
        release.set()
        tab.shutdown_workers()
        tab.close()
        tab.deleteLater()


def test_b_a_grouped_series_completion_ignores_a_live_extra_amplitude(
    qapp: QApplication,
) -> None:
    """A term added mid-fit neither seeds an amplitude nor redraws the curves.

    The completion used to read ``_grouped_fit_model()``: the live model's
    amplitude parameters were defaulted into the curve's parameters and the
    curve itself was evaluated from the live function — wrong values on a
    ``CompositeModel``, and an unexpected-keyword error on a user function.
    """
    tab = GlobalFitTab(member_kind="groups")
    try:
        model_a = tab._grouped_fit_model()
        grouped_datasets = [_grouped_dataset(9001, 1), _grouped_dataset(9002, 2)]
        launch = FitLaunch(model=model_a, global_params=(), datasets=tuple(grouped_datasets))

        # Mid-fit the user subtracts a second term carrying its own amplitude.
        tab._set_composite_model(
            CompositeModel(["OscillatoryField", "Exponential"], operators=["-"])
        )
        live_only = set(tab._grouped_fit_model().param_names) - set(model_a.param_names)
        assert [name for name in live_only if is_amplitude_parameter(name)] == ["A_2"]

        member = _result({**_GROUP_NUISANCES, **_GROUPED_PHYSICS})
        series_result = SimpleNamespace(
            member_results={9001: member, 9002: member},
            member_source_run={9001: 9, 9002: 9},
            seeding_reason="",
        )
        emitted: dict[str, object] = {}
        tab.grouped_fit_completed.connect(lambda _ds, res: emitted.update(res=res))

        tab._on_grouped_series_fit_finished(launch, series_result)

        assert set(emitted["res"]) == {9001, 9002}
        _payload, (t_fit, y_fit), _components = emitted["res"][9001]
        expected = build_grouped_count_model(model_a.function)(
            t_fit, **{**_GROUP_NUISANCES, **_GROUPED_PHYSICS}
        )
        assert y_fit == pytest.approx(expected)
    finally:
        tab.shutdown_workers()
        tab.close()
        tab.deleteLater()


# ── half 2: the form is frozen while the fit runs ───────────────────────────


def test_c_the_batch_form_is_frozen_while_its_fit_runs(qapp: QApplication, monkeypatch) -> None:
    tab = GlobalFitTab(member_kind="runs")
    release = threading.Event()
    try:
        tab.set_datasets([_dataset(3001), _dataset(3002)])
        monkeypatch.setattr(
            global_tab_module,
            "fit_asymmetry_series",
            _blocking_series(release, {3001: _result(_MODEL_A_VALUES)}),
        )
        frozen = (
            tab._edit_model_btn,
            tab._fit_wizard_btn,
            tab._param_table,
            tab._group_model_table,
            tab._group_param_table,
            tab._group_param_reset_btn,
            tab._initial_values_btn,
            tab._seeding_combo,
            tab._preview_btn,
        )

        tab._run_global_fit()
        assert [widget for widget in frozen if widget.isEnabled()] == []
        # A selection change mid-fit must not hand the form back to the user.
        tab.set_datasets([_dataset(3001), _dataset(3002), _dataset(3003)])
        assert [widget for widget in frozen if widget.isEnabled()] == []

        release.set()
        assert tab.wait_for_fit()

        # Preview stays off outside grouped mode; everything else comes back.
        back = [widget for widget in frozen if widget is not tab._preview_btn]
        assert [widget for widget in back if not widget.isEnabled()] == []
    finally:
        release.set()
        tab.shutdown_workers()
        tab.close()
        tab.deleteLater()


def test_d_the_single_form_is_frozen_while_its_fit_runs(qapp: QApplication) -> None:
    tab = SingleFitTab()
    release = threading.Event()
    try:
        tab.set_dataset(_dataset(3001))

        def _fit(_ds, _model_fn, parameters, *, minos=False, cancel_callback=None, **_kwargs):
            assert release.wait(_RELEASE_TIMEOUT_S), "the test never released the stub engine"
            return FitResult(
                success=True,
                chi_squared=30.0,
                reduced_chi_squared=1.0,
                dof=30,
                parameters=parameters,
                uncertainties={p.name: 0.01 for p in parameters},
            )

        tab._fit_engine = SimpleNamespace(fit=_fit)
        frozen = (
            tab._edit_model_btn,
            tab._fit_wizard_btn,
            tab._param_table,
            tab._reset_btn,
            tab._preview_btn,
        )

        tab._run_fit()
        assert [widget for widget in frozen if widget.isEnabled()] == []

        release.set()
        assert tab.wait_for_fit()
        assert [widget for widget in frozen if not widget.isEnabled()] == []
    finally:
        release.set()
        tab.shutdown_workers()
        tab.close()
        tab.deleteLater()
