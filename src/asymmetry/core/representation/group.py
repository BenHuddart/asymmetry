"""Core ``DataGroup``: the canonical, persisted registry entry for a named
group of dataset runs.

This is the schema counterpart to the GUI-only ``DataGroup`` in
``asymmetry.gui.panels.data_browser`` (which additionally carries a
``collapsed`` display flag — a view-state concern that stays out of core).

Historical note (D1, Option B "linked"): groups and series were originally
only weakly coupled — a :class:`FitSeries` recorded the group it was launched
from as pure provenance (``source_group_id``) and membership was frozen at
record time. That description is now historical. Under the unification
(D1/D7) the group becomes the **canonical batch vehicle**: a run-membered
series belongs structurally to a group (``FitSeries.group_id``) and its
effective membership is live-derived from the group's members. A run may
belong to any number of groups — **multi-group membership is explicitly
permitted at the core layer** (no single-membership partition is assumed
here; the browser's one-row-per-membership presentation is a GUI concern).

``kind`` distinguishes ``"user"`` groups (named by the user) from ``"auto"``
groups (minted automatically for an ad-hoc batch selection so every batch fit
has an explicit group). Renaming an ``"auto"`` group promotes it to
``"user"`` — that promotion lives in the ``ProjectModel`` mutation API, not
here.
"""

from __future__ import annotations

from typing import Any

from asymmetry.core.utils.constants import ORDER_KEYS

#: Allowed :attr:`DataGroup.kind` discriminators. ``"user"`` groups are named
#: by the user; ``"auto"`` groups are minted for ad-hoc batch selections.
DATA_GROUP_KINDS = ("user", "auto")


class DataGroup:
    """A named, ordered group of dataset run numbers.

    ``order_key`` is the trend-X convention already used by
    :attr:`FitSeries.order_key` (``"run"``, ``"field"`` or ``"temperature"``),
    so a series built from a group can inherit it. ``kind`` is one of
    :data:`DATA_GROUP_KINDS`; an unrecognised value coerces to ``"user"``
    (mirroring the ``order_key`` coercion above).
    """

    def __init__(
        self,
        group_id: str,
        name: str,
        member_run_numbers: list[int] | None = None,
        order_key: str = "run",
        kind: str = "user",
    ) -> None:
        self.group_id = str(group_id)
        self.name = str(name)
        self.member_run_numbers: list[int] = [int(r) for r in (member_run_numbers or [])]
        self.order_key = order_key if order_key in ORDER_KEYS else "run"
        self.kind = kind if kind in DATA_GROUP_KINDS else "user"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "member_run_numbers": list(self.member_run_numbers),
            "order_key": self.order_key,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataGroup:
        return cls(
            group_id=str(data["group_id"]),
            name=str(data.get("name") or ""),
            member_run_numbers=data.get("member_run_numbers"),
            order_key=str(data.get("order_key", "run")),
            # Tolerant read: pre-v15 saves have no ``kind`` — default to "user".
            kind=str(data.get("kind", "user")),
        )
