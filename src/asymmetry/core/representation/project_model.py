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

import json

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.container import DatasetRepresentations
from asymmetry.core.representation.group import DataGroup
from asymmetry.core.representation.naming import default_series_label
from asymmetry.core.representation.series import FitSeries, canonical_model_matches


def _series_signature(series: FitSeries) -> tuple:
    """A comparable identity for a fit series: same → a duplicate re-run.

    Captures the representation, member kind + physical source runs, and the
    normalised canonical model. Parameter classification (``param_roles``) and
    the fit window (``fit_range``) are deliberately **excluded**: they are
    attributes of a run, not identity, so re-running the same members+model with
    a different Global/Local split (or a different fit window) supersedes the
    earlier series in place (D4) rather than accumulating a duplicate trend pill.
    """
    # Key on the actual member keys, not the deduplicated physical runs: for a
    # group series ``member_run_numbers`` are the synthetic ``(run, detector-group)``
    # keys, so two grouped fits over the *same* source runs but *different*
    # detector-group subsets stay distinct (keying on ``source_runs()`` would
    # collapse them and let one wrongly supersede/merge the other). The keys are
    # always populated — even for a legacy group series with an empty
    # ``member_source_run`` map — so this never degrades to an empty ``()`` set
    # the way ``member_source_run.values()`` would. For a run series the keys are
    # the physical run numbers, so this is unchanged from ``source_runs()``.
    members = tuple(sorted(series.member_run_numbers))
    model_key: str | None = None
    if series.canonical_model is not None:
        try:
            normalised = CompositeModel.from_dict(series.canonical_model).to_dict()
        except (ValueError, KeyError, TypeError):
            normalised = series.canonical_model
        model_key = json.dumps(normalised, sort_keys=True, default=str)
    return (
        str(series.rep_type),
        series.member_kind,
        members,
        model_key,
    )


def _inherit_label(survivor: FitSeries, donors: list[FitSeries | None]) -> None:
    """Give *survivor* a user label from the first *donor* that has one.

    Only fills a *missing* label — a label already on the survivor (e.g. a rename
    of the incoming series) is never clobbered. Shared by the two twin-collapsing
    paths (live re-run supersession and load-time dedupe) so they name a
    collapsed series identically.
    """
    if survivor.label:
        return
    for donor in donors:
        if donor is not None and donor.label:
            survivor.label = donor.label
            return


class ProjectModel:
    """Holds the representations (per run) and batches for one project."""

    def __init__(
        self,
        datasets: dict[int, DatasetRepresentations] | None = None,
        batches: dict[str, FitSeries] | None = None,
        data_groups: dict[str, DataGroup] | None = None,
    ) -> None:
        self.datasets: dict[int, DatasetRepresentations] = dict(datasets or {})
        self.batches: dict[str, FitSeries] = dict(batches or {})
        #: DataGroup registry (D1, Option B: "linked"). Additive/optional in the
        #: schema — projects saved before Phase 7 have no ``data_groups`` block
        #: and load with an empty dict. A group's back-references to the series
        #: built from it are computed on demand (:meth:`series_for_group`), not
        #: stored, so editing a group never invalidates or re-fits its series.
        self.data_groups: dict[str, DataGroup] = dict(data_groups or {})

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

    def remove_data_group(self, group_id: str) -> DataGroup | None:
        """Remove and return the DataGroup with *group_id*, or ``None``.

        Series built from the group are left untouched (D1, Option B): their
        ``source_group_id`` becomes a dangling provenance pointer, same as any
        other reference to a deleted upstream object.
        """
        return self.data_groups.pop(str(group_id), None)

    def series_for_group(self, group_id: str) -> list[FitSeries]:
        """Return batches whose ``source_group_id`` is *group_id*.

        A group's back-references to its series are always computed from the
        batches, never stored on the group — editing a group's membership does
        not retroactively touch any series already built from it.
        """
        group_id = str(group_id)
        return [batch for batch in self.batches.values() if batch.source_group_id == group_id]

    def superseded_batch_ids(self, series: FitSeries) -> list[str]:
        """Return ids of existing batches that *series* makes redundant.

        A batch is superseded when it has the same identity signature as
        *series* (same representation, members and model) — i.e. it is an earlier
        run of the very same batch, even if its Global/Local classification or
        fit window differs. Computed (model-less) series are never matched, since
        two scans over the same runs are legitimately distinct results.
        """
        if series.is_computed:
            return []
        target = _series_signature(series)
        return [
            batch_id
            for batch_id, existing in self.batches.items()
            if batch_id != series.batch_id
            and not existing.is_computed
            and _series_signature(existing) == target
        ]

    def remove_superseded_batches(self, series: FitSeries) -> list[str]:
        """Remove prior batches identical to *series* and inherit their identity.

        Stops re-running the same batch from accumulating duplicate trend
        "pills": the freshly recorded series replaces its older twins. So the
        chip and any back-references stay stable across the re-run, *series*
        inherits the superseded twin's ``batch_id`` and — unless it already
        carries one — its user ``label``. Mutating ``series.batch_id`` here means
        callers must write member ``FitSlot`` pointers from ``series.batch_id``
        (not a pre-allocated id) *after* this returns.
        """
        removed = self.superseded_batch_ids(series)
        if not removed:
            return removed
        # The signatures are identical, so there is normally exactly one twin;
        # if a legacy project carried several, inherit the first (oldest) one's
        # identity and drop the rest. Defer divergence to a single sweep.
        predecessor = self.batches.get(removed[0])
        for batch_id in removed:
            self.remove_batch(batch_id, refresh=False)
        self.refresh_divergence()
        if predecessor is not None:
            _inherit_label(series, [predecessor])
            series.batch_id = predecessor.batch_id
        return removed

    def dedupe_batches(self) -> list[dict]:
        """Collapse batches sharing an identity signature (load-time migration).

        Projects saved during the duplicate era (before replace-in-place, D4)
        can carry several identically-keyed batch series over the same
        members+model. For each such group keep the **most recently recorded**
        one (its results are freshest and its id is the one member ``FitSlot``\\ s
        already reference), drop the rest, and carry a user ``label`` forward onto
        the survivor when it has none. Computed (model-less) series are never
        merged. Returns one record ``{"kept", "dropped", "label"}`` per collapsed
        group so the caller can log what changed; no schema bump — this is
        tolerant reading of an already-valid project.
        """
        groups: dict[tuple, list[str]] = {}
        for batch_id, series in self.batches.items():
            if series.is_computed:
                continue
            groups.setdefault(_series_signature(series), []).append(batch_id)

        records: list[dict] = []
        for ids in groups.values():
            if len(ids) < 2:
                continue
            # ``self.batches`` preserves the saved (recording) order, so the last
            # id is the most recent — keep it and drop its earlier twins. Its
            # member FitSlots already reference it, so dropping earlier twins
            # never clears the keeper's slots.
            keeper_id = ids[-1]
            dropped = ids[:-1]
            keeper = self.batches[keeper_id]
            _inherit_label(keeper, [self.batches.get(other_id) for other_id in dropped])
            for other_id in dropped:
                self.remove_batch(other_id, refresh=False)
            # ``remove_batch`` clears a member FitSlot only when it referenced the
            # dropped id. Normally every slot already points at the keeper (the
            # most recent twin) so nothing is cleared — but a legacy project whose
            # slot pointed at a dropped twin would just have lost its series link.
            # Re-point any now-unlinked keeper member back to the keeper so no run
            # is silently orphaned from the surviving series.
            self._relink_unlinked_members(keeper)
            records.append(
                {
                    "kept": keeper_id,
                    "dropped": list(dropped),
                    # Friendly default (model · run-range) for the log, not the
                    # opaque batch id, when the survivor carries no user label.
                    "label": keeper.label or default_series_label(keeper),
                }
            )
        if records:
            self.refresh_divergence()
        return records

    def remove_batch(self, batch_id: str, *, refresh: bool = True) -> FitSeries | None:
        """Remove and return the batch with *batch_id*, clearing member FitSlot pointers.

        For each member the corresponding FitSlot's ``batch_id`` is cleared and
        ``provenance`` is reset to ``"single"`` (the fit result is preserved;
        only the series association is dropped).  Divergence state for the
        removed batch is no longer relevant; ``refresh_divergence`` is called so
        other batches remain consistent — pass ``refresh=False`` when removing a
        run of batches to defer to a single sweep by the caller.
        """
        series = self.batches.pop(str(batch_id), None)
        if series is None:
            return None
        for run_number in series.source_runs():
            representation = self.representation(run_number, series.rep_type)
            if representation is None:
                continue
            slot = representation.fit
            if slot.batch_id == batch_id:
                slot.batch_id = None
                slot.provenance = "single"
                slot.diverged = False
                slot.include_in_trend = True
        if refresh:
            self.refresh_divergence()
        return series

    def _relink_unlinked_members(self, batch: FitSeries) -> None:
        """Point *batch*'s currently-unlinked member FitSlots back at *batch*.

        Only touches slots whose ``batch_id`` is ``None`` (either never linked or
        cleared while dropping a duplicate twin) — a slot already owned by another
        batch is left alone. Restores the batch's provenance so the run rejoins the
        series' trend.
        """
        provenance = "global" if batch.is_global() else "batch"
        for run_number in batch.source_runs():
            representation = self.representation(run_number, batch.rep_type)
            if representation is None:
                continue
            slot = representation.fit
            if slot.batch_id is None:
                slot.batch_id = batch.batch_id
                slot.provenance = provenance

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

    # ── divergence & trending ──────────────────────────────────────────────────

    def refresh_divergence(self) -> None:
        """Re-evaluate divergence of every batch member against its canonical model.

        A member whose stored fit model no longer matches the batch's canonical
        model is flagged ``diverged`` and excluded from trending **by default**
        (the first time it diverges).  A member whose model matches again is
        un-flagged and re-included.  A manual trend re-inclusion of a still-
        diverged member is preserved across refreshes.

        For group series (``member_kind == "groups"``) the divergence check is
        done at the *source run* level — each synthetic member key maps back to
        one physical run whose ``TIME_GROUPS`` representation holds the
        canonical ``FitSlot``.  All synthetic members sharing a source run
        therefore diverge/reconverge together.
        """
        for batch in self.batches.values():
            # A computed series (e.g. an integral scan) has no fit model and no
            # per-run FitSlots; divergence is meaningless and would wrongly flag
            # a real fit that happens to share the same run numbers.
            if batch.is_computed:
                continue
            if batch.member_kind == "groups":
                self._refresh_group_series_divergence(batch)
            else:
                self._refresh_run_series_divergence(batch)

    def _refresh_run_series_divergence(self, batch: FitSeries) -> None:
        """Divergence check for run-membered series."""
        for run_number in batch.member_run_numbers:
            representation = self.representation(run_number, batch.rep_type)
            if representation is None:
                continue
            fit = representation.fit
            matches = canonical_model_matches(fit.model, batch.canonical_model)
            was_diverged = batch.is_diverged(run_number)
            if matches:
                batch.clear_diverged(run_number)
                if was_diverged:
                    fit.diverged = False
                    fit.include_in_trend = True
                else:
                    fit.diverged = False
            else:
                batch.mark_diverged(run_number)
                fit.diverged = True
                if not was_diverged:
                    fit.include_in_trend = False

    def _refresh_group_series_divergence(self, batch: FitSeries) -> None:
        """Divergence check for group-membered series.

        Group series store one :class:`~asymmetry.core.representation.base.FitSlot`
        per *source run*, not per synthetic member key.  Divergence is therefore
        evaluated once per unique source run; all synthetic members belonging to
        that source run are diverged/reconverged together.
        """
        # Collect unique source runs and the synthetic member keys they own.
        source_to_members: dict[int, list[int]] = {}
        for member_key in batch.member_run_numbers:
            src = batch.source_run_for(member_key)
            source_to_members.setdefault(src, []).append(member_key)

        for source_run, member_keys in source_to_members.items():
            representation = self.representation(source_run, batch.rep_type)
            if representation is None:
                continue
            fit = representation.fit
            matches = canonical_model_matches(fit.model, batch.canonical_model)
            # Snapshot the per-key diverged state BEFORE mutating diverged_runs so
            # that subsequent iterations within this source run see the pre-refresh
            # state.  All keys are evaluated at the same logical point in time.
            # "Any key was diverged" → the source run was previously diverged.
            was_any_diverged = any(batch.is_diverged(k) for k in member_keys)
            # Update all synthetic member keys in the batch first.
            for member_key in member_keys:
                if matches:
                    batch.clear_diverged(member_key)
                else:
                    batch.mark_diverged(member_key)
            # Now write the shared FitSlot exactly once, using the pre-mutation state.
            if matches:
                if was_any_diverged:
                    # All re-converged: restore inclusion.
                    fit.diverged = False
                    fit.include_in_trend = True
                else:
                    fit.diverged = False
            else:
                fit.diverged = True
                if not was_any_diverged:
                    # Newly diverged: exclude from trending by default.
                    fit.include_in_trend = False

    def trend_runs_for_batch(self, batch: FitSeries) -> list[int]:
        """Return the ordered member runs currently included in trending.

        For run series, inclusion follows each member's ``fit.include_in_trend``
        flag.  For group series the flag lives on the *source run*'s
        representation; all synthetic members of an included source run are
        returned.
        """
        runs: list[int] = []
        for member_key in batch.member_run_numbers:
            if batch.member_kind == "groups":
                source_run = batch.source_run_for(member_key)
                representation = self.representation(source_run, batch.rep_type)
            else:
                representation = self.representation(member_key, batch.rep_type)
            if representation is not None and representation.fit.include_in_trend:
                runs.append(member_key)
        return runs

    def set_member_trend_inclusion(self, batch_id: str, run_number: int, include: bool) -> None:
        """Manually include/exclude a batch member from trending.

        For group series *run_number* may be a synthetic member key; the
        inclusion flag is applied to the corresponding source run's
        representation so all groups from that source run are toggled together.
        """
        batch = self.batches.get(str(batch_id))
        if batch is None:
            return
        if batch.member_kind == "groups":
            resolve_run = batch.source_run_for(int(run_number))
        else:
            resolve_run = int(run_number)
        representation = self.representation(resolve_run, batch.rep_type)
        if representation is not None:
            representation.fit.include_in_trend = bool(include)

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
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ProjectModel:
        """Inverse of :meth:`to_dict`."""
        datasets: dict[int, DatasetRepresentations] = {}
        batches: dict[str, FitSeries] = {}
        data_groups: dict[str, DataGroup] = {}
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
        return cls(datasets, batches, data_groups)

    # ── project-dict integration ───────────────────────────────────────────────

    @classmethod
    def from_project_state(cls, project: dict) -> ProjectModel:
        """Build a model from a schema-v6 project dict.

        Reads ``datasets[i].representations``, the top-level ``batches``, and
        the optional top-level ``data_groups`` block (Phase 7, additive —
        absent on a project saved before this phase, which loads with an empty
        registry rather than failing).
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

        return cls(datasets, batches, data_groups)

    def write_to_project_state(self, project: dict) -> None:
        """Write representations onto each dataset entry and batches/groups at top level.

        ``project['datasets']`` entries are matched by ``run_number``; entries
        with no representations get an empty ``representations`` map.
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
