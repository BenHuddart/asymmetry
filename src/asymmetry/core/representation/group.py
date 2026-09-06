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

Phase groups (Global Fit Wizard transitions, D1 of the transitions plan): a
group is a *phase* when :attr:`DataGroup.parent_group_id` names another
group — the series group it partitions. A phase's members are always a
subset of its parent's members (enforced by :meth:`ProjectModel.
create_phase_groups`, not here — this class stores whatever shape it is
given). The phase-only fields (:attr:`phase_ordinal`, :attr:`phase_range`,
:attr:`phase_boundaries`, :attr:`phase_color`, :attr:`phase_provenance`) are
meaningless for an ordinary group and stay at their "not a phase" defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asymmetry.core.utils.constants import ORDER_KEYS

#: Allowed :attr:`DataGroup.kind` discriminators. ``"user"`` groups are named
#: by the user; ``"auto"`` groups are minted for ad-hoc batch selections.
DATA_GROUP_KINDS = ("user", "auto")


@dataclass(frozen=True)
class PhaseSpec:
    """One phase of a series partition, as handed to :meth:`ProjectModel.create_phase_groups`.

    A plain, immutable description — not yet a registered :class:`DataGroup` —
    produced by the Global Fit Wizard's partition search (or, in tests, built
    by hand). ``member_run_numbers`` need not be sorted; ``create_phase_groups``
    is responsible for checking it is a subset of the parent's members and that
    phases are mutually disjoint.
    """

    ordinal: int
    name: str
    member_run_numbers: tuple[int, ...]
    phase_range: tuple[float, float] | None
    phase_boundaries: dict[str, tuple[float, float] | None]
    phase_color: str | None
    phase_provenance: dict[str, Any] = field(default_factory=dict)


#: Roman numerals for :func:`phase_group_name`, high value first — the full
#: subtractive table, so a long series with many phases still reads
#: conventionally ("Phase XL", not "Phase XXXX").
_ROMAN_NUMERALS: tuple[tuple[int, str], ...] = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def phase_group_name(ordinal: int) -> str:
    """The display name of phase *ordinal*: ``"Phase I"``, ``"Phase II"``, ….

    Roman numerals keep a phase's identity visually distinct from the run
    numbers and axis values it sits next to, so "Phase II" is never misread as a
    temperature or a run.
    """
    remaining = int(ordinal)
    if remaining < 1:
        raise ValueError(f"Phase ordinals are 1-based; got {ordinal!r}.")
    numeral: list[str] = []
    for value, symbol in _ROMAN_NUMERALS:
        while remaining >= value:
            numeral.append(symbol)
            remaining -= value
    return f"Phase {''.join(numeral)}"


def _as_pair(value: object) -> tuple[float, float] | None:
    """Coerce a 2-element sequence (tuple, list, or JSON array) to ``(float, float)``.

    ``None`` passes through unchanged — both :attr:`DataGroup.phase_range` and
    each end of :attr:`DataGroup.phase_boundaries` are legitimately absent (no
    range yet computed; a series end has no boundary on that side).
    """
    if value is None:
        return None
    first, second = value  # type: ignore[misc]
    return (float(first), float(second))


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
        *,
        parent_group_id: str | None = None,
        phase_ordinal: int | None = None,
        phase_range: tuple[float, float] | None = None,
        phase_boundaries: dict[str, tuple[float, float] | None] | None = None,
        phase_color: str | None = None,
        phase_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.group_id = str(group_id)
        self.name = str(name)
        self.member_run_numbers: list[int] = [int(r) for r in (member_run_numbers or [])]
        self.order_key = order_key if order_key in ORDER_KEYS else "run"
        self.kind = kind if kind in DATA_GROUP_KINDS else "user"
        #: The series group this phase partitions, or ``None`` for an ordinary
        #: group. A group *is a phase* exactly when this is set — see
        #: :attr:`is_phase`.
        self.parent_group_id: str | None = str(parent_group_id) if parent_group_id else None
        #: 1-based position of this phase along the sweep axis (phases only).
        self.phase_ordinal: int | None = int(phase_ordinal) if phase_ordinal is not None else None
        #: (first, last) axis value spanned by this phase's members.
        self.phase_range: tuple[float, float] | None = _as_pair(phase_range)
        #: Break estimates at the two ends of this phase — ``"lower"``/
        #: ``"upper"`` each an ``(estimate, half_gap)`` pair, or ``None`` at a
        #: series end. Both keys are always present, so a reader never needs a
        #: defensive ``.get(..., None)``.
        boundaries = phase_boundaries or {}
        self.phase_boundaries: dict[str, tuple[float, float] | None] = {
            "lower": _as_pair(boundaries.get("lower")),
            "upper": _as_pair(boundaries.get("upper")),
        }
        #: Swatch colour for this phase (assigned from ``tokens.PHASE_COLORS``).
        self.phase_color: str | None = str(phase_color) if phase_color else None
        #: Plain, JSON-able record of how this phase was found: wizard run
        #: date (ISO string), ``selected_breaks``, ``gains``, ``model_title``,
        #: ``confidence``, ``axis_key``.
        self.phase_provenance: dict[str, Any] = dict(phase_provenance) if phase_provenance else {}

    @property
    def is_phase(self) -> bool:
        """``True`` when this group is a phase nested under a parent group."""
        return self.parent_group_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "member_run_numbers": list(self.member_run_numbers),
            "order_key": self.order_key,
            "kind": self.kind,
            "parent_group_id": self.parent_group_id,
            "phase_ordinal": self.phase_ordinal,
            "phase_range": list(self.phase_range) if self.phase_range is not None else None,
            "phase_boundaries": {
                side: (list(pair) if pair is not None else None)
                for side, pair in self.phase_boundaries.items()
            },
            "phase_color": self.phase_color,
            "phase_provenance": dict(self.phase_provenance),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataGroup:
        phase_range = data.get("phase_range")
        phase_boundaries = data.get("phase_boundaries")
        phase_provenance = data.get("phase_provenance")
        return cls(
            group_id=str(data["group_id"]),
            name=str(data.get("name") or ""),
            member_run_numbers=data.get("member_run_numbers"),
            order_key=str(data.get("order_key", "run")),
            # Tolerant read: pre-v15 saves have no ``kind`` — default to "user".
            kind=str(data.get("kind", "user")),
            # Tolerant reads (file boundary): pre-v19 saves carry none of the
            # phase fields — every one defaults to "not a phase group".
            parent_group_id=data.get("parent_group_id"),
            phase_ordinal=data.get("phase_ordinal"),
            phase_range=phase_range
            if isinstance(phase_range, (list, tuple)) and len(phase_range) == 2
            else None,
            phase_boundaries=phase_boundaries if isinstance(phase_boundaries, dict) else None,
            phase_color=data.get("phase_color"),
            phase_provenance=phase_provenance if isinstance(phase_provenance, dict) else None,
        )
