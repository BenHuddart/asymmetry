"""Project-level owner of per-dataset representations and batches.

``ProjectModel`` is the in-memory home for the redesign's representation state.
It is a *view* keyed by run number: the per-dataset ``representations`` map and
the top-level ``batches`` list of a schema-v6 project dict.  Source files,
metadata, grouping overrides, browser/plot state, etc. remain owned by the
surrounding project dict; this model only carries representations + batches.

On project load, :meth:`recompute_all` walks every representation and rebuilds
its transient arrays from the recipe — this is the recipe-only recompute that
replaces storing computed spectra/asymmetry in the project file.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from asymmetry.core.data.dataset import Run
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.container import DatasetRepresentations
from asymmetry.core.representation.group import DataGroup, PhaseSpec
from asymmetry.core.representation.series import FitSeries


def _axis_value(run_number: int, order_key: str, runs_by_number: dict[int, Run]) -> float:
    """The sweep-axis value of *run_number* under *order_key* (mirrors ``FitSeries.sort_members``).

    Falls back to the run number itself for ``order_key == "run"`` or when the
    run's metadata is not supplied — the same fallback ``FitSeries.sort_members``
    uses, so a phase's axis ordering agrees with its owning series'.
    """
    if order_key == "run":
        return float(run_number)
    run = runs_by_number.get(run_number)
    if run is None:
        return float(run_number)
    return float(run.field if order_key == "field" else run.temperature)


class ProjectModel:
    """Holds the representations (per run) and batches for one project."""

    def __init__(
        self,
        datasets: dict[int, DatasetRepresentations] | None = None,
        batches: dict[str, FitSeries] | None = None,
        data_groups: dict[str, DataGroup] | None = None,
        active_series: dict[str, str] | None = None,
    ) -> None:
        self.datasets: dict[int, DatasetRepresentations] = dict(datasets or {})
        self.batches: dict[str, FitSeries] = dict(batches or {})
        #: DataGroup registry (D1, Option B: "linked"). Additive/optional in the
        #: schema — projects saved before Phase 7 have no ``data_groups`` block
        #: and load with an empty dict. A group's back-references to the series
        #: built from it are computed on demand (:meth:`series_for_group`), not
        #: stored, so editing a group never invalidates or re-fits its series.
        self.data_groups: dict[str, DataGroup] = dict(data_groups or {})
        #: One active series per representation (D5), keyed by
        #: :class:`RepresentationType` *value*. The Parameters chip, the Batch
        #: tab's open series and the plot overlay all read and write this one
        #: pointer. A representation with no entry has no active series.
        self.active_series: dict[str, str] = {
            str(rep): str(batch_id) for rep, batch_id in (active_series or {}).items()
        }

    # ── access ───────────────────────────────────────────────────────────────

    def ensure_dataset(self, run_number: int) -> DatasetRepresentations:
        """Return the container for *run_number*, creating it if needed."""
        run_number = int(run_number)
        existing = self.datasets.get(run_number)
        if existing is None:
            existing = DatasetRepresentations(run_number)
            self.datasets[run_number] = existing
        return existing

    def representation(self, run_number: int, rep_type: RepresentationType):
        """Return the representation of *rep_type* for *run_number*, or ``None``."""
        container = self.datasets.get(int(run_number))
        return None if container is None else container.get(rep_type)

    def batch(self, batch_id: str) -> FitSeries | None:
        """Return the batch with *batch_id*, or ``None``."""
        return self.batches.get(str(batch_id))

    def add_batch(self, batch: FitSeries) -> None:
        """Register *batch* by its id."""
        self.batches[batch.batch_id] = batch

    def data_group(self, group_id: str) -> DataGroup | None:
        """Return the DataGroup with *group_id*, or ``None``."""
        return self.data_groups.get(str(group_id))

    def add_data_group(self, group: DataGroup) -> None:
        """Register *group* by its id."""
        self.data_groups[group.group_id] = group

    # ── active series (D5) ───────────────────────────────────────────────────

    @staticmethod
    def _rep_key(rep_type: RepresentationType | str) -> str:
        """Return the :attr:`active_series` key for *rep_type* (its enum value)."""
        return RepresentationType(rep_type).value

    def set_active_series(self, rep_type: RepresentationType | str, batch_id: str | None) -> None:
        """Make *batch_id* the active series of *rep_type*, or clear it with ``None``.

        The active series is the one the plot overlays for every run it covers,
        the one the Parameters chip rail shows pressed, and the one the Batch
        tab reopens on load (D5).
        """
        key = self._rep_key(rep_type)
        if batch_id is None:
            self.active_series.pop(key, None)
        else:
            self.active_series[key] = str(batch_id)

    def active_series_id(self, rep_type: RepresentationType | str) -> str | None:
        """Return the active series id for *rep_type*, or ``None``."""
        return self.active_series.get(self._rep_key(rep_type))

    # ── group mutation API (D1/D4/D7; plain methods, no Qt) ────────────────────

    def create_data_group(
        self,
        name: str,
        member_run_numbers: list[int] | None = None,
        *,
        kind: str = "user",
        group_id: str | None = None,
        order_key: str = "run",
        parent_group_id: str | None = None,
    ) -> DataGroup:
        """Create, register, and return a new :class:`DataGroup`.

        Mints a fresh ``uuid4`` id when *group_id* is not supplied. *kind* is
        ``"user"`` (default) or ``"auto"`` (an auto-group minted for an ad-hoc
        batch selection). Multi-group membership is permitted, so this does
        **not** strip *member_run_numbers* from any other group.

        *parent_group_id* recreates a group that was already a phase — the
        Data Browser's ``sync_groups_from_project_model`` legacy-seed
        reconstruction is the only caller that passes it; a fresh partition
        goes through :meth:`create_phase_groups` instead, which also fills in
        the ordinal/range/boundaries/colour/provenance this generic
        constructor leaves at their "not yet known" defaults.
        """
        gid = str(group_id) if group_id else str(uuid.uuid4())
        group = DataGroup(
            group_id=gid,
            name=name,
            member_run_numbers=member_run_numbers,
            order_key=order_key,
            kind=kind,
            parent_group_id=parent_group_id,
        )
        self.data_groups[gid] = group
        return group

    def rename_data_group(self, group_id: str, name: str) -> bool:
        """Set the display *name* of the group with *group_id*.

        Renaming a ``kind="auto"`` group **promotes it to ``"user"``** (D4): a
        user who names an auto-group has adopted it, so it should no longer read
        as a machine-minted group. Returns ``True`` on success, ``False`` when
        *group_id* is unknown.
        """
        group = self.data_groups.get(str(group_id))
        if group is None:
            return False
        group.name = str(name)
        if group.kind == "auto":
            group.kind = "user"
        return True

    def set_data_group_members(self, group_id: str, member_run_numbers: list[int] | None) -> bool:
        """Replace the membership of the group with *group_id*.

        Returns ``True`` on success, ``False`` when *group_id* is unknown. Does
        not touch any series built from the group (staleness is derived on
        demand via :meth:`FitSeries.is_stale`, not pushed).
        """
        group = self.data_groups.get(str(group_id))
        if group is None:
            return False
        group.member_run_numbers = [int(r) for r in (member_run_numbers or [])]
        return True

    def find_auto_group(self, member_run_numbers: list[int] | None) -> DataGroup | None:
        """Return an existing ``kind="auto"`` group with the identical member *set*.

        Used to reuse an auto-group when a batch is re-run over the same ad-hoc
        selection (D3) rather than proliferating near-duplicate auto-groups. The
        comparison is set-based (order-insensitive); only ``"auto"`` groups are
        candidates — a user group with the same members is never silently
        reused. A phase group is never returned: it is always ``kind="user"``
        (:meth:`create_phase_groups`), so this exclusion is redundant in
        practice, but it documents the invariant an ad-hoc batch selection must
        never resolve to a structural phase.
        """
        target = {int(r) for r in (member_run_numbers or [])}
        for group in self.data_groups.values():
            if (
                group.kind == "auto"
                and not group.is_phase
                and set(group.member_run_numbers) == target
            ):
                return group
        return None

    def remove_data_group(self, group_id: str, *, orphan_series: bool) -> list[str]:
        """Remove the group with *group_id* and dispose of its series (D7).

        *orphan_series* selects the disposition of the run series owned by the
        group (matched via :meth:`series_for_group`, i.e. on ``group_id`` with a
        legacy ``source_group_id`` fallback):

        * ``True`` — **freeze** each owned series into a standalone legacy
          analysis: its structural ``group_id`` is cleared to ``None`` and its
          ``member_run_numbers`` are snapshotted to ``last_fitted_members`` so
          the frozen membership is exactly what was last fit (not the live,
          possibly-drifted group membership). No series is deleted and the
          returned list is empty.
        * ``False`` — **delete** each owned series from :attr:`batches` via
          :meth:`remove_batch` (which also clears an :attr:`active_series`
          pointer at it) and return the removed ``batch_id``\\ s so a GUI caller
          can clean up any further pointers it holds.

        Any phase groups nested under *group_id* (:meth:`phase_groups_for`) are
        removed first, recursively, with the *same* ``orphan_series`` choice —
        a series group's disposal must not leave its phases dangling, and the
        choice of "keep the fits" vs. "delete the fits" is one decision for the
        whole series, not one the caller re-makes per phase.

        Groups with no series behave identically under either flag (nothing to
        dispose). Unknown *group_id* removes nothing and returns ``[]``.
        """
        group = self.data_groups.pop(str(group_id), None)
        if group is None:
            return []
        removed: list[str] = []
        for phase in self.phase_groups_for(group_id):
            removed.extend(self.remove_data_group(phase.group_id, orphan_series=orphan_series))
        owned = self.series_for_group(group_id)
        if orphan_series:
            for series in owned:
                # Only structurally-owned run series are re-snapshotted and
                # frozen; a legacy series matched purely by the source_group_id
                # fallback is already frozen (group_id is None) — leave its
                # snapshot untouched.
                if series.group_id == str(group_id):
                    # An empty last_fitted_members means the series was never
                    # fit — keep its existing member list rather than freezing
                    # it to an empty snapshot.
                    if series.last_fitted_members:
                        series.member_run_numbers = list(series.last_fitted_members)
                    series.group_id = None
            return removed
        for series in owned:
            if self.remove_batch(series.batch_id) is not None:
                removed.append(series.batch_id)
        return removed

    def series_for_group(self, group_id: str) -> list[FitSeries]:
        """Return the series owned by the group with *group_id*.

        Matches on the structural :attr:`FitSeries.group_id` (D1/D7), falling
        back to the legacy :attr:`FitSeries.source_group_id` provenance pointer
        so series migrated from — or frozen out of — an older project still
        resolve. A group's back-references are always computed from the batches,
        never stored on the group.
        """
        group_id = str(group_id)
        return [
            batch
            for batch in self.batches.values()
            if batch.group_id == group_id
            or (batch.group_id is None and batch.source_group_id == group_id)
        ]

    # ── phase groups (Global Fit Wizard transitions, D1) ────────────────────

    def phase_groups_for(self, parent_id: str) -> list[DataGroup]:
        """Return the phase groups nested under *parent_id*, in ordinal order."""
        parent_id = str(parent_id)
        return sorted(
            (g for g in self.data_groups.values() if g.parent_group_id == parent_id),
            key=lambda g: g.phase_ordinal if g.phase_ordinal is not None else 0,
        )

    def excluded_runs_for(self, parent_id: str) -> list[int]:
        """Return the parent's members claimed by none of its phases.

        Preserves the parent's own member order. Unknown *parent_id* has no
        members to exclude and returns ``[]``.
        """
        parent = self.data_groups.get(str(parent_id))
        if parent is None:
            return []
        claimed: set[int] = set()
        for phase in self.phase_groups_for(parent_id):
            claimed.update(phase.member_run_numbers)
        return [r for r in parent.member_run_numbers if r not in claimed]

    def create_phase_groups(self, parent_id: str, phases: Sequence[PhaseSpec]) -> list[str]:
        """Partition the group *parent_id* into phase groups, replacing any existing ones.

        Each :class:`~asymmetry.core.representation.group.PhaseSpec` becomes a
        new :class:`DataGroup` nested under *parent_id*
        (``parent_group_id=parent_id``). *phases* must together describe a
        genuine partition of a *subset* of the parent's members: every run a
        phase names must also be a member of the parent, and no run number may
        appear in more than one phase (a run in no phase is allowed — it reads
        back from :meth:`excluded_runs_for`). Both violations raise
        ``ValueError`` before any group is created or removed.

        Any phase groups the parent already owns are removed first, cascading
        their series with ``orphan_series=False`` — i.e. **deleted**, not
        frozen. A re-partition replaces the previous phases outright: freezing
        the old phases' fits as standalone legacy analyses named after phases
        that no longer exist (e.g. a stale "Phase II" group once the break
        count changes) would only confuse the project, and the wizard that
        calls this always re-fits the new phases from scratch.

        Returns the new phase groups' ids, in ordinal order.
        """
        parent = self.data_groups.get(str(parent_id))
        if parent is None:
            raise ValueError(f"Unknown parent group id: {parent_id!r}")

        parent_members = set(parent.member_run_numbers)
        claimed: set[int] = set()
        for phase in phases:
            members = {int(r) for r in phase.member_run_numbers}
            outside = members - parent_members
            if outside:
                raise ValueError(
                    f"Phase {phase.ordinal} ({phase.name!r}) names members outside "
                    f"parent group {parent_id!r}: {sorted(outside)}"
                )
            overlap = members & claimed
            if overlap:
                raise ValueError(
                    f"Phase {phase.ordinal} ({phase.name!r}) overlaps an earlier "
                    f"phase: {sorted(overlap)}"
                )
            claimed |= members

        for existing in self.phase_groups_for(parent_id):
            self.remove_data_group(existing.group_id, orphan_series=False)

        new_ids: list[str] = []
        for phase in sorted(phases, key=lambda p: p.ordinal):
            # Members keep the parent's order — the sweep-axis order the series
            # was built in — not numeric run order, which need not follow the
            # axis at all.
            members = {int(r) for r in phase.member_run_numbers}
            group = DataGroup(
                group_id=str(uuid.uuid4()),
                name=phase.name,
                member_run_numbers=[r for r in parent.member_run_numbers if r in members],
                order_key=parent.order_key,
                kind="user",
                parent_group_id=str(parent_id),
                phase_ordinal=phase.ordinal,
                phase_range=phase.phase_range,
                phase_boundaries=phase.phase_boundaries,
                phase_color=phase.phase_color,
                phase_provenance=phase.phase_provenance,
            )
            self.add_data_group(group)
            new_ids.append(group.group_id)
        return new_ids

    def remove_phase_groups(self, parent_id: str, *, orphan_series: bool) -> list[str]:
        """Dissolve every phase of *parent_id*, leaving the parent group intact.

        The un-partitioning counterpart of :meth:`create_phase_groups`: the
        parent keeps all of its members (a phase's members were always a subset
        of the parent's, so nothing is lost), and each phase's own series is
        disposed of per *orphan_series* exactly as :meth:`remove_data_group`
        defines it — ``True`` freezes them into standalone legacy analyses,
        ``False`` deletes them and returns the removed ``batch_id``\\ s.

        Distinct from :meth:`create_phase_groups`'s internal re-partition
        cascade, which always deletes: that path is replacing the phases with
        fresh ones, whereas this one is the user's explicit "Ungroup phases",
        where keeping the fits is a legitimate choice.
        """
        removed: list[str] = []
        for phase in self.phase_groups_for(parent_id):
            removed.extend(self.remove_data_group(phase.group_id, orphan_series=orphan_series))
        return removed

    def move_run_to_phase(
        self,
        run_number: int,
        phase_id: str | None,
        *,
        series_group_id: str,
        runs_by_number: dict[int, Run],
    ) -> None:
        """Move *run_number* to the phase *phase_id* (``None`` = excluded) within one series.

        *series_group_id* names the partitioned series the move belongs to. A
        run may be a member of several data groups, and two of them may both
        be partitioned; only the phases of *this* series are touched, so
        excluding a run from one series' partition never edits another's.
        Removes the run from whichever phase of that series currently holds it
        (a no-op if it was already excluded), then adds it to the target phase.
        Parent membership is untouched — a phase's members are always a subset
        of the parent's, never a separate list to keep in sync. Both the
        vacated and the target phase (whichever of the two exist) have their
        members re-ordered along the sweep axis and their
        :attr:`DataGroup.phase_range` recomputed from *runs_by_number*, the
        same metadata :meth:`FitSeries.sort_members` uses.

        Raises ``ValueError`` when *phase_id* names something other than a
        live phase group of *series_group_id*, or when *run_number* is not a
        member of that series — moving a run into a phase whose parent never
        had it would silently violate the subset invariant
        :meth:`create_phase_groups` enforces at creation time.

        Any phase series bound to an affected phase group reads stale
        afterwards through the existing membership-snapshot mechanism
        (:attr:`FitSeries.group_id` / :meth:`FitSeries.is_stale` against
        :meth:`FitSeries.effective_members`) — this method does not touch
        series state directly.
        """
        run_number = int(run_number)
        series_id = str(series_group_id)
        parent = self.data_groups.get(series_id)
        if parent is None or run_number not in parent.member_run_numbers:
            raise ValueError(f"Run {run_number} is not a member of series group {series_id!r}")
        affected: list[DataGroup] = []

        for phase in self.phase_groups_for(series_id):
            if run_number in phase.member_run_numbers:
                phase.member_run_numbers = [r for r in phase.member_run_numbers if r != run_number]
                affected.append(phase)

        if phase_id is not None:
            target = self.data_groups.get(str(phase_id))
            if target is None or target.parent_group_id != series_id:
                raise ValueError(f"{phase_id!r} is not a phase of series group {series_id!r}")
            if run_number not in target.member_run_numbers:
                target.member_run_numbers = [*target.member_run_numbers, run_number]
            if target not in affected:
                affected.append(target)

        for phase in affected:
            phase.member_run_numbers.sort(
                key=lambda r: _axis_value(r, phase.order_key, runs_by_number)
            )
            if phase.member_run_numbers:
                values = [
                    _axis_value(r, phase.order_key, runs_by_number)
                    for r in phase.member_run_numbers
                ]
                phase.phase_range = (values[0], values[-1])
            else:
                phase.phase_range = None

    def remove_batch(self, batch_id: str) -> FitSeries | None:
        """Remove and return the series with *batch_id*, or ``None`` if unknown.

        Deleting a series touches only that series (D6): per-run state is the
        single fit alone, so there are no member pointers to clear and no other
        series to re-evaluate. An :attr:`active_series` entry pointing at the
        removed series is cleared — that representation simply has no active
        series until one is chosen.
        """
        batch_id = str(batch_id)
        series = self.batches.pop(batch_id, None)
        if series is None:
            return None
        self.active_series = {
            rep: active for rep, active in self.active_series.items() if active != batch_id
        }
        return series

    def set_trend_excluded(self, batch_id: str, run_number: int, excluded: bool) -> None:
        """Include or exclude one member of *batch_id* from that series' trend (D4).

        *run_number* is a member key: a run number for a run series, a synthetic
        detector-group key for a group series. Exclusion is per series, so a run
        dropped from one series' trend still trends in every other series that
        contains it. Unknown *batch_id* changes nothing.
        """
        series = self.batches.get(str(batch_id))
        if series is None:
            return
        remaining = set(series.trend_excluded_runs)
        if excluded:
            remaining.add(int(run_number))
        else:
            remaining.discard(int(run_number))
        series.trend_excluded_runs = sorted(remaining)

    def rename_batch(self, batch_id: str, label: str | None) -> bool:
        """Set the display label of the batch with *batch_id*.

        Pass ``None`` or ``""`` to clear the label (reverts to the positional
        fallback rendered by the GUI).  Returns ``True`` on success, ``False``
        when *batch_id* is not found.
        """
        series = self.batches.get(str(batch_id))
        if series is None:
            return False
        series.label = str(label).strip() or None if label else None
        return True

    # ── recompute-on-load ──────────────────────────────────────────────────────

    def recompute_all(self, runs_by_number: dict[int, Run]) -> None:
        """Rebuild every representation's transient arrays from its recipe.

        Representations whose run is missing, whose recipe cannot currently be
        computed, or which opt out of load-time recomputation
        (``recompute_on_load`` is false, e.g. the expensive MaxEnt iteration)
        are left uncomputed rather than aborting the whole load.
        """
        for run_number, container in self.datasets.items():
            run = runs_by_number.get(run_number)
            if run is None:
                continue
            for representation in container:
                if not representation.recompute_on_load:
                    continue
                try:
                    representation.invalidate()
                    representation.ensure_computed(run)
                except Exception:  # noqa: BLE001 - one bad recipe must not abort load
                    representation.invalidate()

    # ── standalone persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a standalone serialisation of representations + batches."""
        return {
            "representations_by_run": {
                str(run_number): container.to_dict()
                for run_number, container in self.datasets.items()
            },
            "batches": [batch.to_dict() for batch in self.batches.values()],
            "data_groups": [group.to_dict() for group in self.data_groups.values()],
            "active_series": dict(self.active_series),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ProjectModel:
        """Inverse of :meth:`to_dict`."""
        datasets: dict[int, DatasetRepresentations] = {}
        batches: dict[str, FitSeries] = {}
        data_groups: dict[str, DataGroup] = {}
        active_series: dict | None = None
        if isinstance(data, dict):
            raw_reps = data.get("representations_by_run")
            if isinstance(raw_reps, dict):
                for run_key, container_data in raw_reps.items():
                    if not isinstance(container_data, dict):
                        continue
                    payload = dict(container_data)
                    payload.setdefault("run_number", run_key)
                    container = DatasetRepresentations.from_dict(payload)
                    datasets[container.run_number] = container
            for batch_data in data.get("batches", []) or []:
                if isinstance(batch_data, dict):
                    batch = FitSeries.from_dict(batch_data)
                    batches[batch.batch_id] = batch
            for group_data in data.get("data_groups", []) or []:
                if isinstance(group_data, dict):
                    group = DataGroup.from_dict(group_data)
                    data_groups[group.group_id] = group
            active_series = data.get("active_series")
        return cls(datasets, batches, data_groups, active_series)

    # ── project-dict integration ───────────────────────────────────────────────

    @classmethod
    def from_project_state(cls, project: dict) -> ProjectModel:
        """Build a model from a schema-v6 project dict.

        Reads ``datasets[i].representations``, the top-level ``batches``, the
        optional top-level ``data_groups`` block (Phase 7, additive — absent on
        a project saved before this phase, which loads with an empty registry
        rather than failing) and the top-level ``active_series`` pointer map
        (D5; absent before v20, which loads with no active series).
        """
        datasets: dict[int, DatasetRepresentations] = {}
        batches: dict[str, FitSeries] = {}
        data_groups: dict[str, DataGroup] = {}
        if not isinstance(project, dict):
            return cls()

        for entry in project.get("datasets", []) or []:
            if not isinstance(entry, dict):
                continue
            run_number = int(entry.get("run_number", 0))
            reps = entry.get("representations")
            if isinstance(reps, dict) and reps:
                container = DatasetRepresentations.from_dict(
                    {"run_number": run_number, "representations": reps}
                )
                datasets[run_number] = container

        for batch_data in project.get("batches", []) or []:
            if isinstance(batch_data, dict):
                batch = FitSeries.from_dict(batch_data)
                batches[batch.batch_id] = batch

        for group_data in project.get("data_groups", []) or []:
            if isinstance(group_data, dict):
                group = DataGroup.from_dict(group_data)
                data_groups[group.group_id] = group

        return cls(datasets, batches, data_groups, project.get("active_series"))

    def write_to_project_state(self, project: dict) -> None:
        """Write representations onto each dataset entry, and the top-level blocks.

        ``project['datasets']`` entries are matched by ``run_number``; entries
        with no representations get an empty ``representations`` map. The
        ``batches``, ``data_groups`` and ``active_series`` blocks are written at
        the top level.
        """
        for entry in project.get("datasets", []) or []:
            if not isinstance(entry, dict):
                continue
            container = self.datasets.get(int(entry.get("run_number", 0)))
            entry["representations"] = (
                container.to_dict()["representations"] if container is not None else {}
            )
        project["batches"] = [batch.to_dict() for batch in self.batches.values()]
        project["data_groups"] = [group.to_dict() for group in self.data_groups.values()]
        project["active_series"] = dict(self.active_series)
