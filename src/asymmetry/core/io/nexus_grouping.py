"""NeXus-declared memberships and optional independent default pair."""

from dataclasses import dataclass

import numpy as np

from asymmetry.core.instruments.layout import PresetGrouping


@dataclass(frozen=True)
class GroupingSelection:
    groups: dict[int, list[int]]
    pair: tuple[int, int] | None

    @property
    def payload(self) -> dict:
        return {
            "groups": self.groups,
            **(
                {"forward_group": self.pair[0], "backward_group": self.pair[1]} if self.pair else {}
            ),
        }


def nexus_grouping(
    values, n_detectors: int, default: PresetGrouping | None = None
) -> GroupingSelection:
    """Present tables retain their memberships; absence alone permits a default."""
    if values is None:
        if default is not None:
            return GroupingSelection(
                {gid: list(group.detector_ids) for gid, group in default.groups.items()},
                (default.forward_group, default.backward_group),
            )
        split = max(1, n_detectors // 2)
        return GroupingSelection(
            {
                1: list(range(1, split + 1)),
                2: list(range(split + 1, n_detectors + 1)) or list(range(1, n_detectors + 1)),
            },
            (1, 2),
        )
    table = np.asarray(values)
    if table.shape != (n_detectors,) or not np.issubdtype(table.dtype, np.integer):
        raise ValueError("Grouping declaration must contain one integer per acquired detector")
    groups = {}
    for ordinal, group in enumerate(table, start=1):
        if group > 0:
            groups.setdefault(int(group), []).append(ordinal)
    if not groups:
        raise ValueError("Grouping declaration contains no positive detector group")
    ordered = sorted(groups)
    return GroupingSelection(groups, tuple(ordered[:2]) if len(ordered) >= 2 else None)


def nexus_period_grouping(values, n_periods: int, n_detectors: int):
    """Decode exact shared-detector or period-major histogram declarations.

    NeXus muon V1, Cottrell Rev. 8, notes 4 and 6: grouping indexes
    histogram_number, including switching-state doubling.
    """
    if values is None:
        return [None] * n_periods
    table = np.asarray(values)
    if table.shape == (n_detectors,):
        return [table] * n_periods
    if table.shape in {(n_periods * n_detectors,), (n_periods, n_detectors)}:
        return list(table.reshape(n_periods, n_detectors))
    raise ValueError(
        "Grouping declaration must contain one integer per acquired detector or histogram"
    )


def grouping_membership(grouping: dict) -> tuple:
    """Compare detector membership and ordered legs independently of presentation."""
    return (
        tuple(
            sorted(
                (int(gid), tuple(sorted(members)))
                for gid, members in grouping.get("groups", {}).items()
            )
        ),
        grouping.get("forward_group"),
        grouping.get("backward_group"),
    )
