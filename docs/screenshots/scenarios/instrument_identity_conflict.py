"""Review synthetic identity evidence while retaining the pending scientific request."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QDialog, QMessageBox

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
    accept_instrument_choice,
)
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from asymmetry.gui.windows.grouping.layout_editor import InstrumentEdit
from asymmetry.gui.windows.instrument_identity_dialog import InstrumentIdentityDialog

from ._base import Scenario, register


def identity_conflict_run():
    """An eight-channel synthetic run with saved GPS and observed FLAME identities."""
    return Run(
        run_number=8104,
        histograms=[Histogram(np.full(8, 100.0), 0.01, 0, 0, 7) for _ in range(8)],
        metadata={"facility": "PSI", "instrument": "FLAME"},
        grouping={"groups": {1: [1, 3], 2: [2, 4]}, "alpha": 0.93},
        instrument_choice=RecordedInstrumentChoice(
            InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
        ),
    )


class InstrumentIdentityConflictScenario(Scenario):
    name = "instrument_identity_conflict"
    description = (
        "Retained GPS identity, contrary FLAME evidence and explicit follower consequence."
    )
    size = (620, 360)

    def build(self):
        run = identity_conflict_run()
        profile = GroupingProfile("Saved GPS", ProfileFingerprint("psi.gps", 8, "bin_combined"))
        dialog = InstrumentIdentityDialog(run, profile)
        index = next(
            i
            for i in range(dialog._choices.count())
            if isinstance(dialog._choices.itemData(i), InstrumentSelection)
            and dialog._choices.itemData(i).identity == ("psi.flame", "eight_channel")
        )
        dialog._choices.setCurrentIndex(index)
        return dialog


class GroupingPendingIdentityScenario(Scenario):
    name = "grouping_pending_identity"
    description = "Another run's identity review preserves the first pending identity and settings."
    size = (620, 160)

    def build(self):
        run = identity_conflict_run()
        other = deepcopy(run)
        other.run_number = 8105
        other.instrument_choice = accept_instrument_choice(other, other.instrument_choice.selection)
        datasets = [
            MuonDataset(
                np.arange(8) * 0.01,
                np.zeros(8),
                np.ones(8),
                metadata={"run_number": item.run_number},
                run=item,
            )
            for item in (run, other)
        ]
        profile = GroupingProfile("Saved GPS", ProfileFingerprint("psi.gps", 8, "bin_combined"))
        grouping = GroupingDialog(
            datasets, profiles=[profile], assigned_profiles={8104: profile.name, 8105: profile.name}
        )
        grouping._instrument_edit = InstrumentEdit(
            accept_instrument_choice(run, run.instrument_choice.selection),
            GroupingProfile.from_dict(grouping._draft.to_dict()),
            run.run_number,
        )
        with patch.object(QMessageBox, "information") as notice:
            grouping._on_scope_run_selected(8105)
            grouping._on_instrument_identity()
        notice.assert_called_once()
        _, title, text = notice.call_args.args
        if grouping._instrument_edit.run_number != 8104:
            raise RuntimeError("The second review must retain the first pending identity")
        grouping.done(QDialog.Rejected)
        grouping.deleteLater()
        return QMessageBox(QMessageBox.Information, title, text, QMessageBox.Ok)


register(InstrumentIdentityConflictScenario())
register(GroupingPendingIdentityScenario())
