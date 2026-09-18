"""The persisted *joint fit* record: several series coupled by shared parameters.

A :class:`JointFit` composes :class:`~asymmetry.core.representation.series.FitSeries`
already made in the Batch tab (D2 of ``docs/plans/joint-fit.md``) — it never
holds curves or per-run state of its own, only which series belong to it, the
shared-parameter table that couples them, and a summary of the last run. The
per-run and per-series results of a joint run live exactly where a solo run's
would: under each member series' own ``batch_id`` in ``results_by_run`` (D8).
This mirrors :class:`~asymmetry.core.representation.global_fit_study.GlobalFitStudy`
— canonical ``to_dict``, tolerant ``from_dict`` that drops a malformed entry
rather than failing the whole project, and a staleness verdict computed at
read time from live project state rather than stored on the record (D9): a
stamp goes stale the moment a member is deleted, detached, or re-run on its
own, and persisting "stale" would just be a second place for that fact to go
out of date.

Qt-free: no GUI code may be imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from asymmetry.core.representation.base import RepresentationType

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from asymmetry.core.representation.project_model import ProjectModel

__all__ = ["JointFit"]


def _optional_bound(value: float) -> float | None:
    """Render one shared-row bound for disk: ``None`` for an unbounded ±inf."""
    return None if math.isinf(value) else float(value)


def _bound_from_optional(value: object, default: float) -> float:
    """Read one shared-row bound: ``None``/absent becomes *default* (an inf sentinel)."""
    return default if value is None else float(value)


def _normalise_shared_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one ``shared`` entry in its canonical in-memory shape.

    Bounds are held as float ±inf sentinels internally (matching
    :class:`asymmetry.core.fitting.joint.SharedParameter`, which this row feeds
    at run time) so every reader can compare/format them uniformly; only
    :meth:`JointFit.to_dict` turns an infinite bound into ``None`` for disk.
    """
    members = row.get("members") or {}
    return {
        "name": str(row.get("name", "")),
        "members": {str(batch_id): str(param) for batch_id, param in members.items()},
        "value": float(row.get("value", 0.0)),
        "min": _bound_from_optional(row.get("min"), -math.inf),
        "max": _bound_from_optional(row.get("max"), math.inf),
    }


def _shared_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Canonical, JSON-safe serialization of one ``shared`` entry."""
    return {
        "name": row["name"],
        "members": dict(row["members"]),
        "value": float(row["value"]),
        "min": _optional_bound(row["min"]),
        "max": _optional_bound(row["max"]),
    }


def _shared_row_from_dict(entry: object) -> dict[str, Any] | None:
    """Tolerant read of one persisted ``shared`` entry, or ``None`` when unusable.

    A row naming fewer than two members shares nothing (D4's own invariant),
    so it is dropped rather than carried through as a row that could never
    have come from a legitimate run.
    """
    if not isinstance(entry, dict):
        return None
    members = entry.get("members")
    if not entry.get("name") or not isinstance(members, dict) or len(members) < 2:
        return None
    try:
        return _normalise_shared_row(entry)
    except (TypeError, ValueError):
        return None


@dataclass
class JointFit:
    """A named, persisted joint fit over two or more member series.

    ``joint_id`` is caller-supplied (mirroring :class:`GlobalFitStudy`, this
    module never mints one). ``member_batch_ids`` is ordered — the order the
    member series were ticked in, which is also the seed order D6 uses to pick
    a shared row's initial value/bounds from "the first contributing series".
    ``shared`` is a plain list of dicts rather than a dataclass: it is exactly
    what the window's shared-parameter table edits, and a fourth column type
    (min/max/value) would buy nothing a dict does not already give a JSON
    round trip. ``result`` holds only the last run's *summary* — shared
    values, uncertainties, the shared covariance block, combined and
    per-series χ², a timestamp — never curves; the curves are each member
    series' own ``results_by_run``, already persisted there.
    """

    joint_id: str
    label: str | None
    rep_type: RepresentationType | str
    member_batch_ids: list[str] = field(default_factory=list)
    shared: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.joint_id = str(self.joint_id)
        self.label = str(self.label).strip() or None if self.label else None
        self.rep_type = (
            self.rep_type
            if isinstance(self.rep_type, RepresentationType)
            else RepresentationType(str(self.rep_type))
        )
        self.member_batch_ids = [str(batch_id) for batch_id in self.member_batch_ids]
        self.shared = [_normalise_shared_row(row) for row in self.shared]
        self.result = dict(self.result) if isinstance(self.result, dict) else None

    # ── label ──────────────────────────────────────────────────────────────

    def display_name(self, fallback: str) -> str:
        """Return the user-assigned label, or *fallback* when none is set."""
        return self.label or fallback

    # ── staleness (D9; runtime only, never persisted) ───────────────────────

    def is_stale(self, model: ProjectModel) -> bool:
        """Return ``True`` when this record no longer describes a live joint fit."""
        return bool(self.stale_reason(model))

    def stale_reason(self, model: ProjectModel) -> str:
        """Return the reason this joint fit is stale, or ``""`` when it is fresh.

        Checked in member order, first match wins: a member missing entirely
        (deleted), a member whose own :attr:`FitSeries.joint_fit_id` no longer
        points back at this record (it was re-run solo, per D3 of the series
        workflow, and its results no longer honour the shared constraint), or
        a member that is itself stale against its owning group
        (:meth:`FitSeries.is_stale`). ``is_stale`` is exactly ``bool`` of this,
        so the two can never disagree on whether a record is fresh.
        """
        for batch_id in self.member_batch_ids:
            series = model.batch(batch_id)
            if series is None:
                return f"member series {batch_id!r} was deleted"
            if series.joint_fit_id != self.joint_id:
                return f"member series {series.display_name(batch_id)!r} was re-run on its own"
            group = model.data_group(series.group_id) if series.group_id is not None else None
            if series.is_stale(group):
                return (
                    f"member series {series.display_name(batch_id)!r} is stale "
                    "(its membership changed)"
                )
        return ""

    # ── persistence ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "label": self.label,
            "rep_type": self.rep_type.value,
            "member_batch_ids": list(self.member_batch_ids),
            "shared": [_shared_row_to_dict(row) for row in self.shared],
            "result": dict(self.result) if self.result is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> JointFit | None:
        """Tolerant deserialization.

        Returns ``None`` — rather than raising — for a payload that cannot be
        a joint fit at all: not a mapping, an unrecognised ``rep_type``, or
        fewer than two ``member_batch_ids`` (D2's own invariant: a joint fit
        of one series composes nothing). A caller iterating a persisted
        ``joint_fits`` list skips a ``None`` entry, exactly as
        :meth:`GlobalFitStudy.from_dict`'s callers do. A malformed ``shared``
        row is dropped individually rather than failing the whole record; an
        unusable ``result`` block degrades to ``None`` rather than failing it
        either, since a joint fit's identity is its members and shared table,
        not its last result.
        """
        if not isinstance(data, dict):
            return None
        try:
            rep_type = RepresentationType(str(data["rep_type"]))
        except (KeyError, ValueError):
            return None

        raw_members = data.get("member_batch_ids")
        if not isinstance(raw_members, list) or len(raw_members) < 2:
            return None

        shared: list[dict[str, Any]] = []
        for entry in data.get("shared") or []:
            row = _shared_row_from_dict(entry)
            if row is not None:
                shared.append(row)

        result = data.get("result")
        result = dict(result) if isinstance(result, dict) else None

        return cls(
            joint_id=str(data.get("joint_id", "")),
            label=data.get("label"),
            rep_type=rep_type,
            member_batch_ids=[str(batch_id) for batch_id in raw_members],
            shared=shared,
            result=result,
        )
