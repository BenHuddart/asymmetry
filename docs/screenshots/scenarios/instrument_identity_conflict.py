"""Review contrary synthetic file evidence without discarding the saved request."""

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.instruments.resolution import InstrumentSelection, RecordedInstrumentChoice
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.gui.windows.instrument_identity_dialog import InstrumentIdentityDialog

from ._base import Scenario, register


class InstrumentIdentityConflictScenario(Scenario):
    name = "instrument_identity_conflict"
    description = (
        "Retained GPS identity, contrary FLAME evidence and explicit follower consequence."
    )
    size = (620, 360)

    def build(self):
        run = Run(
            run_number=8104,
            histograms=[Histogram(np.full(8, 100.0), 0.01, 0, 0, 7) for _ in range(8)],
            metadata={"facility": "PSI", "instrument": "FLAME"},
            grouping={"groups": {1: [1, 3], 2: [2, 4]}, "alpha": 0.93},
            instrument_choice=RecordedInstrumentChoice(
                InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
            ),
        )
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


register(InstrumentIdentityConflictScenario())
