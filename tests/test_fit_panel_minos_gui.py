"""GUI wiring for MINOS / quality / seeding / abort (fit-workflow-diagnostics)."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab, _ValueUncertaintyDelegate


def _dataset() -> MuonDataset:
    t = np.linspace(0.0, 8.0, 200)
    err = np.full_like(t, 0.01)
    y = 0.2 * np.exp(-0.5 * t)
    return MuonDataset(time=t, asymmetry=y, error=err, metadata={"run_number": 1})


def test_single_fit_minos_toggle_threads_and_populates_role(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_dataset(_dataset())
    model = tab._composite_model
    captured: dict = {}

    def _fit(ds, model_fn, parameters, *, minos=False):
        captured["minos"] = minos
        names = list(model.param_names)
        return FitResult(
            success=True,
            chi_squared=190.0,
            reduced_chi_squared=1.0,
            dof=190,
            parameters=ParameterSet(
                [Parameter(name=p, value=float(i + 1)) for i, p in enumerate(names)]
            ),
            uncertainties={p: 0.01 for p in names},
            minos_errors={names[0]: (-0.012, 0.009)} if minos else None,
        )

    from types import SimpleNamespace

    tab._fit_engine = SimpleNamespace(fit=_fit)
    tab._minos_checkbox.setChecked(True)
    tab._run_fit()

    assert captured["minos"] is True
    # The first parameter's value cell carries the asymmetric interval role.
    value_item = tab._param_table.item(0, 1)
    assert value_item.data(_ValueUncertaintyDelegate._MINOS_ROLE) == (-0.012, 0.009)
    # The result label gained a teaching tooltip.
    assert "quality" in tab._result_label.toolTip().lower()


def test_single_fit_minos_off_clears_role(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_dataset(_dataset())
    model = tab._composite_model

    def _fit(ds, model_fn, parameters, *, minos=False):
        names = list(model.param_names)
        return FitResult(
            success=True,
            chi_squared=190.0,
            reduced_chi_squared=1.0,
            dof=190,
            parameters=ParameterSet([Parameter(name=p, value=1.0) for p in names]),
            uncertainties={p: 0.01 for p in names},
        )

    from types import SimpleNamespace

    tab._fit_engine = SimpleNamespace(fit=_fit)
    tab._minos_checkbox.setChecked(False)
    tab._run_fit()
    value_item = tab._param_table.item(0, 1)
    assert value_item.data(_ValueUncertaintyDelegate._MINOS_ROLE) is None


def test_batch_seeding_mode_setter(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    assert tab._batch_seeding_mode == "auto"
    tab.set_batch_seeding_mode("chain")
    assert tab._batch_seeding_mode == "chain"


def test_stop_button_hidden_until_busy(qapp: QApplication) -> None:
    # isHidden() reflects the explicit hide flag regardless of ancestor visibility
    # (the tab is never shown on screen in the offscreen test).
    tab = GlobalFitTab(member_kind="runs")
    assert tab._stop_btn.isHidden()
    tab._set_series_busy(True)
    assert not tab._stop_btn.isHidden()
    assert tab._fit_btn.isHidden()
    tab._set_series_busy(False)
    assert tab._stop_btn.isHidden()
    assert not tab._fit_btn.isHidden()


def test_order_key_from_group_metadata(qapp: QApplication) -> None:
    class _Group:
        def __init__(self, temperature):
            self.metadata = {"temperature": temperature}

    members = {10: [_Group(5.0)], 11: [_Group(10.0)]}
    order = GlobalFitTab._grouped_series_order_key(members)
    assert order == {10: 5.0, 11: 10.0}
    # No usable metadata -> None (Auto then falls back to independent seeds).
    assert GlobalFitTab._grouped_series_order_key({10: [object()]}) is None
