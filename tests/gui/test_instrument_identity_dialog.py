"""Identity confirmation preserves authored grouping until its caller applies the choice."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.instruments.resolution import (
    GenericInstrumentIdentity,
    InstrumentSelection,
    RecordedInstrumentChoice,
    UnresolvedInstrumentIdentity,
    resolve_instrument,
)
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.gui.windows.instrument_identity_dialog import InstrumentIdentityDialog

pytestmark = pytest.mark.gui


@pytest.fixture
def run():
    return Run(
        histograms=[Histogram(np.full(8, 100.0), 0.1, 0, 0, 7) for _ in range(8)],
        metadata={"facility": "PSI", "instrument": "FLAME"},
        grouping={"groups": {1: [1, 3], 2: [2, 4]}, "alpha_x": 0.93},
        instrument_choice=RecordedInstrumentChoice(
            InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
        ),
    )


def select(dialog, key, acquisition):
    index = next(
        i
        for i in range(dialog._choices.count())
        if isinstance(dialog._choices.itemData(i), InstrumentSelection)
        and dialog._choices.itemData(i).identity == (key, acquisition)
    )
    dialog._choices.setCurrentIndex(index)


def test_conflict_shows_retained_observed_and_raw_evidence_and_captures_acceptance(qapp, run):
    dialog = InstrumentIdentityDialog(run)
    dialog.show()
    assert "GPS" in dialog._retained.text()
    assert "FLAME" in dialog._observed.text()
    assert "facility: PSI" in dialog._evidence.toPlainText()
    assert "instrument: FLAME" in dialog._evidence.toPlainText()
    assert "conflict" in dialog._observed.text().lower()
    identities = [
        item.identity
        for i in range(dialog._choices.count())
        if isinstance(item := dialog._choices.itemData(i), InstrumentSelection)
    ]
    assert len(identities) == len(set(identities))
    assert ("psi.lem",) == tuple(
        dialog._choices.itemData(i).profile_key
        for i in range(dialog._choices.count())
        if isinstance(dialog._choices.itemData(i), InstrumentSelection)
        and dialog._choices.itemData(i).profile_key == "psi.lem"
    )
    select(dialog, "psi.flame", "eight_channel")
    QTest.mouseClick(dialog._buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted
    choice = dialog.get_choice()
    assert choice.selection.identity == ("psi.flame", "eight_channel")
    assert choice.evidence == resolve_instrument(run).observed.evidence
    assert run.instrument_choice.selection.profile_key == "psi.gps"


def test_cancel_after_draft_selection_does_not_change_run_or_following_profile(qapp, run):
    profile = GroupingProfile("Saved GPS", ProfileFingerprint("psi.gps", 8, "bin_combined"))
    before = deepcopy((run.grouping, run.metadata, run.instrument_choice, profile.to_dict()))
    draft = InstrumentSelection("psi.flame", "eight_channel", "1", "manual")
    dialog = InstrumentIdentityDialog(run, profile, selection=draft)
    dialog.show()
    assert dialog._choices.currentData().identity == draft.identity
    assert "stop following" in dialog._consequence.text().lower()
    assert "Saved GPS" in dialog._consequence.text()
    assert "base" in dialog._consequence.text().lower()
    QTest.mouseClick(dialog._buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.LeftButton)
    assert dialog.result() == QDialog.Rejected
    assert (run.grouping, run.metadata, run.instrument_choice, profile.to_dict()) == before


def test_retaining_compatible_identity_keeps_following_and_unknown_label_is_preserved(qapp, run):
    profile = GroupingProfile("Saved GPS", ProfileFingerprint("psi.gps", 8, "bin_combined"))
    dialog = InstrumentIdentityDialog(run, profile)
    assert "keep following" in dialog._consequence.text().lower()
    assert "no calculation" in dialog._consequence.text().lower()
    dialog.close()
    unknown = UnresolvedInstrumentIdentity("Uncatalogued <special> apparatus")
    dialog = InstrumentIdentityDialog(run, selection=unknown)
    assert unknown.label in dialog._choices.currentText()
    assert dialog.get_choice().selection == unknown
    assert dialog._choices.count() == 1
    assert not dialog._choices.isEnabled()
    dialog.close()


def test_unavailable_retained_profile_is_not_replaced_with_a_bundled_revision(qapp, run):
    stored = InstrumentSelection("external.apparatus", "saved-config", "7", "project")
    run.instrument_choice = RecordedInstrumentChoice(stored, ())
    dialog = InstrumentIdentityDialog(run)
    assert "external.apparatus" in dialog._retained.text()
    assert "saved-config" in dialog._retained.text()
    assert "unavailable" in dialog._retained.text()
    assert dialog.get_choice().selection is stored
    dialog.close()


def test_no_resolved_or_retained_identity_requires_an_explicit_choice(qapp):
    dialog = InstrumentIdentityDialog(Run())
    assert not dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    with pytest.raises(ValueError, match="must be selected"):
        dialog.get_choice()
    dialog.close()


def test_layout_proposal_is_fixed_while_standalone_review_offers_other_choices(qapp, run):
    proposed = InstrumentSelection("psi.flame", "eight_channel", "1", "manual")
    confirmation = InstrumentIdentityDialog(run, selection=proposed)
    assert confirmation._choices.count() == 1
    assert not confirmation._choices.isEnabled()
    assert confirmation.get_choice().selection.identity == proposed.identity
    review = InstrumentIdentityDialog(run)
    assert review._choices.isEnabled()
    assert any(
        isinstance(review._choices.itemData(i), GenericInstrumentIdentity)
        for i in range(review._choices.count())
    )
    confirmation.done(QDialog.Rejected)
    review.done(QDialog.Rejected)
