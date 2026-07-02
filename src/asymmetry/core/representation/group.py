"""Core ``DataGroup``: the canonical, persisted registry entry for a named
group of dataset runs.

This is the schema/provenance counterpart to the GUI-only ``DataGroup`` in
``asymmetry.gui.panels.data_browser`` (which additionally carries a
``collapsed`` display flag — a view-state concern that stays out of core).
A :class:`FitSeries` built from a group's members can record the group's id
as :attr:`~asymmetry.core.representation.series.FitSeries.source_group_id`
(D1, Option B: "linked" — the series remains an independent object; the
back-reference from group to series is computed, not stored).
"""

from __future__ import annotations

from typing import Any

from asymmetry.core.utils.constants import ORDER_KEYS


class DataGroup:
    """A named, ordered group of dataset run numbers.

    ``order_key`` is the trend-X convention already used by
    :attr:`FitSeries.order_key` (``"run"``, ``"field"`` or ``"temperature"``),
    so a series built from a group can inherit it.
    """

    def __init__(
        self,
        group_id: str,
        name: str,
        member_run_numbers: list[int] | None = None,
        order_key: str = "run",
    ) -> None:
        self.group_id = str(group_id)
        self.name = str(name)
        self.member_run_numbers: list[int] = [int(r) for r in (member_run_numbers or [])]
        self.order_key = order_key if order_key in ORDER_KEYS else "run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "member_run_numbers": list(self.member_run_numbers),
            "order_key": self.order_key,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataGroup:
        return cls(
            group_id=str(data["group_id"]),
            name=str(data.get("name") or ""),
            member_run_numbers=data.get("member_run_numbers"),
            order_key=str(data.get("order_key", "run")),
        )
