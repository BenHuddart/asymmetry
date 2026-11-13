"""A source question over synthetic, owner-labelled provenance facts."""

from asymmetry.core.data.acquisition import (
    ContentIdentity,
    SourceDeclaration,
    SourceOwner,
    SourceVerification,
)
from asymmetry.gui.utils.project_sources import source_change_question

from ._base import Scenario, register


class ProjectSourceDecisionScenario(Scenario):
    name = "project_source_decision"
    description = "Explicit source replacement, history, and cancellation choices."
    size = (650, 370)

    def build(self):
        old, new = ContentIdentity("a" * 64), ContentIdentity("b" * 64)
        facts = tuple(
            SourceVerification(
                SourceDeclaration(
                    role, name, old, run_number=number, owner=SourceOwner(address, True)
                ),
                name,
                "changed",
                new,
            )
            for role, name, number, address in (
                ("sample", "synthetic-ag.nxs", 701, ("datasets", 0)),
                (
                    "reference",
                    "synthetic-reference.nxs",
                    702,
                    ("datasets", 0, "grouping_overrides", "background_run"),
                ),
            )
        )
        box, _, _ = source_change_question(None, facts, (0, 1))
        return box


register(ProjectSourceDecisionScenario())
