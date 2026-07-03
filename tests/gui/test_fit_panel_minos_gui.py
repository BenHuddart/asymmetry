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

    def _fit(ds, model_fn, parameters, *, minos=False, cancel_callback=None):
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
    assert tab.wait_for_fit()

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

    def _fit(ds, model_fn, parameters, *, minos=False, cancel_callback=None):
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
    assert tab.wait_for_fit()
    value_item = tab._param_table.item(0, 1)
    assert value_item.data(_ValueUncertaintyDelegate._MINOS_ROLE) is None


def test_batch_seeding_mode_setter(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    assert tab._batch_seeding_mode == "auto"
    tab.set_batch_seeding_mode("chain")
    assert tab._batch_seeding_mode == "chain"


def test_batch_seeding_on_tab_combo_syncs_with_mode(qapp: QApplication) -> None:
    """The on-tab seeding combo and the mode stay in lock-step both ways (Bug #7)."""
    tab = GlobalFitTab(member_kind="runs")
    # Default reflects the mode.
    assert tab._seeding_combo.currentData() == "auto"

    # mode -> combo: set_batch_seeding_mode drives the combo without re-emitting.
    emitted: list[str] = []
    tab.batch_seeding_mode_changed.connect(emitted.append)
    tab.set_batch_seeding_mode("chain")
    assert tab._seeding_combo.currentData() == "chain"
    assert emitted == []  # programmatic sync must not bounce back

    # combo -> mode: a user selection updates the mode and notifies listeners once.
    idx = tab._seeding_combo.findData("as_provided")
    tab._seeding_combo.setCurrentIndex(idx)
    assert tab._batch_seeding_mode == "as_provided"
    assert emitted == ["as_provided"]


def test_seeding_combo_absent_on_single_grouped_surface(qapp: QApplication) -> None:
    """Batch-series seeding has no meaning on the single grouped surface (one
    dataset's groups, no run series), so the on-tab combo is omitted there but
    present on a real batch-series tab."""
    single_grouped = GlobalFitTab(member_kind="groups", grouped_single=True)
    assert single_grouped._seeding_combo is None
    # set_batch_seeding_mode stays safe with no combo present.
    single_grouped.set_batch_seeding_mode("chain")
    assert single_grouped._batch_seeding_mode == "chain"

    batch_series = GlobalFitTab(member_kind="groups")
    assert batch_series._seeding_combo is not None


def test_fit_quality_tooltip_explains_high_ndof_band(qapp: QApplication) -> None:
    """The verdict tooltip explains the confidence band tightens with ν (Bug #3)."""
    from asymmetry.gui.styles.widgets import fit_quality_tooltip

    quality = {
        "verdict": "poor",
        "band_low": 0.960,
        "band_high": 1.041,
        "confidence": 0.95,
        "dof": 4756,
    }
    tip = fit_quality_tooltip(quality)
    assert "Rgoodfit" in tip
    assert "ν" in tip  # explains the dependence on degrees of freedom
    assert "Fit quality confidence" in tip  # points to the configurable setting


def test_fit_quality_chip_softens_marginal_high_ndof(qapp: QApplication) -> None:
    """A near-unity χ²ᵣ that only reads "poor" at high ν leads with a neutral
    "near-ideal (band-tight)" amber chip rather than an alarming red "poor"
    (the cuprate case); the verdict itself stays in the tooltip (P3-1)."""
    from asymmetry.gui.styles import tokens
    from asymmetry.gui.styles.widgets import fit_quality_chip_html, fit_quality_tooltip

    marginal = {"verdict": "poor", "chi2_reduced": 1.10, "marginal": True}
    plain = {"verdict": "poor", "chi2_reduced": 8.0, "marginal": False}

    chip_marginal = fit_quality_chip_html(marginal)
    assert "near-ideal (band-tight)" in chip_marginal
    assert "poor" not in chip_marginal  # the alarming word is not in the at-a-glance chip
    assert tokens.WARN in chip_marginal  # amber, not the alarming error red
    assert tokens.ERROR not in chip_marginal
    # The verdict is still explained on hover.
    assert "poor" in fit_quality_tooltip(marginal)

    chip_plain = fit_quality_chip_html(plain)
    assert "near-ideal" not in chip_plain
    assert tokens.ERROR in chip_plain  # genuine poor stays red


def test_fit_quality_chip_and_tooltip_flag_params_at_bound(qapp: QApplication) -> None:
    """A free param pinned on its bound adds an "at bound" chip + tooltip note,
    even when no χ² verdict is available."""
    from asymmetry.gui.styles.widgets import fit_quality_chip_html, fit_quality_tooltip

    chip = fit_quality_chip_html(None, ["r"])
    assert "at bound" in chip

    tip = fit_quality_tooltip(None, ["r"])
    assert "at a bound" in tip
    assert "r" in tip
    assert "poorly constrained" in tip

    # No bound params -> no badge.
    assert "at bound" not in fit_quality_chip_html({"verdict": "good"}, [])


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
