"""Single-fit carry-forward provenance badge (D2/F6).

Selecting an unseen run kept showing the previously-fit model *and its fitted
values* with nothing indicating the new run had never been fit itself (F6);
after a project reload the mix-up got worse (F21). Carry-forward is useful
(a run series naturally inherits the model of its neighbours) and is kept,
but the panel must say so explicitly via a dismissable badge, cleared the
moment a real fit is recorded for the run.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import FitPanel


def _dataset(run_number: int) -> MuonDataset:
    time = np.linspace(0.1, 8.0, 8)
    return MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=np.ones_like(time),
        metadata={"run_number": run_number},
    )


def _fit_result() -> FitResult:
    ps = ParameterSet()
    ps.add(Parameter(name="A_1", value=0.2))
    return FitResult(success=True, reduced_chi_squared=1.0, parameters=ps)


def test_first_ever_selection_shows_no_badge(qapp: QApplication) -> None:
    """The very first run shown in a fresh panel holds the untouched default,

    not content inherited from a real prior selection -- badging it "carried"
    would be exactly the kind of false provenance claim this feature exists
    to prevent.
    """
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: None)

    panel.set_dataset(_dataset(2960))

    assert panel._single_fit_provenance == "representation_default"
    assert panel._single_tab._carry_forward_badge.isHidden()


def test_own_slot_and_unseen_run_carries_with_named_source(qapp: QApplication) -> None:
    """Fit run A, select unseen run B: badge names A as the carry source."""
    panel = FitPanel()
    stored: dict[int, dict] = {}
    panel.set_single_fit_restore_provider(lambda ds: stored.get(int(ds.run_number)) if ds else None)

    panel.set_dataset(_dataset(2960))
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)
    stored[2960] = panel.get_single_form_state()

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._carry_forward_badge.isHidden()

    panel.set_dataset(_dataset(2963))

    assert panel._single_fit_provenance == "carried_from_run"
    assert panel._single_fit_carry_source_run == 2960
    badge_text = panel._single_tab._carry_forward_badge_label.text()
    assert "2960" in badge_text
    assert "not fitted for this run" in badge_text
    assert not panel._single_tab._carry_forward_badge.isHidden()


def test_badge_clears_when_fit_recorded_for_carried_run(qapp: QApplication) -> None:
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: None)

    panel.set_dataset(_dataset(2960))
    panel.set_dataset(_dataset(2963))
    assert not panel._single_tab._carry_forward_badge.isHidden()

    panel._single_tab.fit_completed.emit(_fit_result(), None, None)

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._carry_forward_badge.isHidden()


def test_session_cached_unfit_form_shows_generic_badge(qapp: QApplication) -> None:
    """Revisiting a never-fit run shows the generic wording (no known source)."""
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: None)

    panel.set_dataset(_dataset(2960))  # first-ever run: untouched default, no badge
    panel.set_dataset(_dataset(2963))  # unseen, carries from 2960
    panel.set_dataset(_dataset(2960))  # revisits 2960 -- session-cached, never fit

    assert panel._single_fit_provenance == "carried_session"
    badge_text = panel._single_tab._carry_forward_badge_label.text()
    assert "not fitted for this run" in badge_text
    assert not panel._single_tab._carry_forward_badge.isHidden()


def test_reload_of_own_slot_shows_no_badge(qapp: QApplication) -> None:
    """A form restored from its own persisted slot must not be badged."""
    panel = FitPanel()
    stored = {2960: {"composite_model": None, "parameters": [], "result_html": "fit"}}
    panel.set_single_fit_restore_provider(lambda ds: stored.get(int(ds.run_number)) if ds else None)

    panel.set_dataset(_dataset(2960))

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._carry_forward_badge.isHidden()


def test_representation_default_shows_no_badge(qapp: QApplication) -> None:
    """An explicitly-blanked unfit projection is not "carried" -- no badge."""
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: {})

    panel.set_dataset(_dataset(2960))

    assert panel._single_fit_provenance == "representation_default"
    assert panel._single_tab._carry_forward_badge.isHidden()
