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

from asymmetry.core.data.dataset import Run
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.container import DatasetRepresentations
from asymmetry.core.representation.series import FitSeries, canonical_model_matches


class ProjectModel:
    """Holds the representations (per run) and batches for one project."""

    def __init__(
        self,
        datasets: dict[int, DatasetRepresentations] | None = None,
        batches: dict[str, FitSeries] | None = None,
    ) -> None:
        self.datasets: dict[int, DatasetRepresentations] = dict(datasets or {})
        self.batches: dict[str, FitSeries] = dict(batches or {})

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

    def remove_batch(self, batch_id: str) -> FitSeries | None:
        """Remove and return the batch with *batch_id*, clearing member FitSlot pointers.

        For each member the corresponding FitSlot's ``batch_id`` is cleared and
        ``provenance`` is reset to ``"single"`` (the fit result is preserved;
        only the series association is dropped).  Divergence state for the
        removed batch is no longer relevant; ``refresh_divergence`` is called so
        other batches remain consistent.
        """
        series = self.batches.pop(str(batch_id), None)
        if series is None:
            return None
        if series.member_kind == "groups":
            source_runs = sorted(set(series.member_source_run.values()))
        else:
            source_runs = list(series.member_run_numbers)
        for run_number in source_runs:
            representation = self.representation(run_number, series.rep_type)
            if representation is None:
                continue
            slot = representation.fit
            if slot.batch_id == batch_id:
                slot.batch_id = None
                slot.provenance = "single"
                slot.diverged = False
                slot.include_in_trend = True
        self.refresh_divergence()
        return series

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
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ProjectModel:
        """Inverse of :meth:`to_dict`."""
        datasets: dict[int, DatasetRepresentations] = {}
        batches: dict[str, FitSeries] = {}
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
        return cls(datasets, batches)

    # ── project-dict integration ───────────────────────────────────────────────

    @classmethod
    def from_project_state(cls, project: dict) -> ProjectModel:
        """Build a model from a schema-v6 project dict.

        Reads ``datasets[i].representations`` and the top-level ``batches``.
        """
        datasets: dict[int, DatasetRepresentations] = {}
        batches: dict[str, FitSeries] = {}
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

        return cls(datasets, batches)

    def write_to_project_state(self, project: dict) -> None:
        """Write representations onto each dataset entry and batches at top level.

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
