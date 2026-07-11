"""Single-fit carry-forward provenance badge (D2/F6) and D5 refresh-unless-fitted.

Selecting an unseen run kept showing the previously-fit model *and its fitted
values* with nothing indicating the new run had never been fit itself (F6);
after a project reload the mix-up got worse (F21). Carry-forward is useful
(a run series naturally inherits the model of its neighbours) and is kept,
but the panel must say so explicitly via a dismissable badge, cleared the
moment a real fit is recorded for the run.

D5 (`docs/studies/datagroup-fitseries-unification.md`) later upgraded this to
"refresh-unless-fitted": a run with a recorded fit result (single *or*
batch/global member) is PROTECTED and never auto-overwritten; everything else
is REFRESHABLE and, on selection, refreshes from the session's most recently
*fitted* single-tab state (superseding any stale cached carry for that run)
rather than restoring whatever it — or the previously active run — last
displayed. "Share with Group" is retired in the same change.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import FitPanel

pytestmark = [pytest.mark.gui]


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


def test_no_fit_yet_falls_back_to_last_displayed_carry(qapp: QApplication) -> None:
    """D5 TEST 7: with nothing fitted anywhere in the session yet, a refresh
    falls back to today's last-*displayed* carry-forward (unchanged
    behaviour) rather than refusing to show anything.
    """
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: None)

    panel.set_dataset(_dataset(2960))  # first-ever run: untouched default, no badge
    panel.set_dataset(_dataset(2963))  # unseen, no fit exists yet -> carries from 2960

    assert panel._single_fit_provenance == "carried_from_run"
    assert panel._single_fit_carry_source_run == 2960


def test_protected_single_fit_survives_a_different_model_fit_elsewhere(
    qapp: QApplication,
) -> None:
    """D5 TEST 1: run A fitted; fit a different model on run B; select A ->
    A's own fitted state (its model) is intact, not refreshed onto B's model.
    """
    panel = FitPanel()
    stored: dict[int, dict] = {}
    panel.set_single_fit_restore_provider(lambda ds: stored.get(int(ds.run_number)) if ds else None)

    a_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    panel.set_dataset(_dataset(3001))
    panel._single_tab._set_composite_model(a_model)
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)
    stored[3001] = panel.get_single_form_state()
    assert panel._single_fit_provenance == "own_slot"

    b_model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    panel.set_dataset(_dataset(3002))
    panel._single_tab._set_composite_model(b_model)
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)
    stored[3002] = panel.get_single_form_state()

    panel.set_dataset(_dataset(3001))

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._composite_model.component_names == a_model.component_names


def test_protected_batch_member_survives_a_different_single_fit_elsewhere(
    qapp: QApplication,
) -> None:
    """D5 TEST 2: a batch/global write-back protects its member runs exactly
    like a single fit -- this is the "pointer slot" path (batch members carry
    a real result but never a ``ui_state`` of their own; the persisted-slot
    reconstruction that makes this possible lives in
    ``MainWindow._single_fit_restore_payload`` /
    ``FitPanel.build_single_fit_payload_from_slot``, exercised for real in
    ``tests/gui/test_fit_slot_orchestration.py``). Here the panel's own
    batch-write-back cache (``register_global_fit_results`` writes exactly
    the state shape a reconstructed slot would) stands in for the persisted
    project-model slot a real mediator would return.
    """
    panel = FitPanel()
    stored: dict[int, dict] = {}
    panel.set_single_fit_restore_provider(lambda ds: stored.get(int(ds.run_number)) if ds else None)

    batch_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    panel._global_tab._set_composite_model(batch_model)
    panel.register_global_fit_results(
        {
            1: (_fit_result(), None, []),
            2: (_fit_result(), None, []),
        }
    )
    stored[1] = panel._single_state_by_run[1]
    stored[2] = panel._single_state_by_run[2]

    other_model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    panel.set_dataset(_dataset(3))
    panel._single_tab._set_composite_model(other_model)
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)

    panel.set_dataset(_dataset(1))

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._composite_model.component_names == batch_model.component_names


def test_refresh_replaces_stale_cached_carry_with_latest_fitted_function(
    qapp: QApplication,
) -> None:
    """D5 TEST 4: viewing an unfit run C caches a carried form; fitting an
    improved model on D and re-selecting C must show D's function (the stale
    cached carry is replaced), name D in the badge, and clear the result label
    -- not resurrect the earlier, now-stale carry.
    """
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: None)

    panel.set_dataset(_dataset(4001))  # first-ever run: untouched default
    panel.set_dataset(_dataset(4002))  # C: unseen, no fit yet -> carries from 4001
    assert panel._single_fit_carry_source_run == 4001

    d_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    panel.set_dataset(_dataset(4003))  # D
    panel._single_tab._set_composite_model(d_model)
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)

    panel.set_dataset(_dataset(4002))  # revisit C

    assert panel._single_fit_provenance == "carried_from_run"
    assert panel._single_fit_carry_source_run == 4003
    assert panel._single_tab._composite_model.component_names == d_model.component_names
    assert panel._single_tab._result_label.text() == "No fit performed yet"
    badge_text = panel._single_tab._carry_forward_badge_label.text()
    assert "4003" in badge_text


def test_hand_edited_unfit_form_is_refreshable_by_design(qapp: QApplication) -> None:
    """D5 TEST 5: editing a run's model without fitting it does not protect it
    -- D5 deliberately has no dirty tracking; the protection trigger is "did
    you commit by fitting", so navigating away and a fit landing elsewhere
    means this run refreshes on return, same as any other unfit run.
    """
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: None)

    panel.set_dataset(_dataset(5001))  # first-ever run
    panel.set_dataset(_dataset(5002))  # E: unseen
    hand_edited_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    panel._single_tab._set_composite_model(hand_edited_model)  # hand-edited, never fit

    d_model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    panel.set_dataset(_dataset(5003))  # navigate away
    panel._single_tab._set_composite_model(d_model)
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)  # fit elsewhere

    panel.set_dataset(_dataset(5002))  # return to E

    assert panel._single_fit_provenance == "carried_from_run"  # refreshed, not protected
    assert panel._single_tab._composite_model.component_names == d_model.component_names


def test_reload_of_own_slot_shows_no_badge(qapp: QApplication) -> None:
    """A form restored from its own persisted slot must not be badged."""
    panel = FitPanel()
    stored = {2960: {"composite_model": None, "parameters": [], "result_html": "fit"}}
    panel.set_single_fit_restore_provider(lambda ds: stored.get(int(ds.run_number)) if ds else None)

    panel.set_dataset(_dataset(2960))

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._carry_forward_badge.isHidden()


def test_protected_across_reload_with_no_session_cache(qapp: QApplication) -> None:
    """D5 TEST 3: fit run A, "save + reload" (a fresh panel with no session
    cache but a mediator that still resolves A's persisted slot), fit a new
    model on B, select A -> A's fitted state is restored via the mediator/slot
    path alone -- ``_single_state_by_run`` is never consulted for protection.
    """
    a_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    persisted = {
        6001: {
            "model_name": "Composite",
            "composite_model": a_model.to_dict(),
            "parameters": [],
            "result_html": "fit",
        }
    }

    # Fresh panel simulating a just-reloaded project: no run has ever been
    # shown, so ``_single_state_by_run`` starts empty; the mediator alone
    # supplies A's persisted state.
    panel = FitPanel()
    panel.set_single_fit_restore_provider(
        lambda ds: persisted.get(int(ds.run_number)) if ds else None
    )
    assert panel._single_state_by_run == {}

    b_model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    panel.set_dataset(_dataset(6002))
    panel._single_tab._set_composite_model(b_model)
    panel._single_tab.fit_completed.emit(_fit_result(), None, None)

    panel.set_dataset(_dataset(6001))

    assert panel._single_fit_provenance == "own_slot"
    assert panel._single_tab._composite_model.component_names == a_model.component_names


def test_representation_default_shows_no_badge(qapp: QApplication) -> None:
    """An explicitly-blanked unfit projection is not "carried" -- no badge."""
    panel = FitPanel()
    panel.set_single_fit_restore_provider(lambda ds: {})

    panel.set_dataset(_dataset(2960))

    assert panel._single_fit_provenance == "representation_default"
    assert panel._single_tab._carry_forward_badge.isHidden()
