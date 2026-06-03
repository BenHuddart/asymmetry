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
from asymmetry.core.representation.batch import Batch, canonical_model_matches
from asymmetry.core.representation.container import DatasetRepresentations


class ProjectModel:
    """Holds the representations (per run) and batches for one project."""

    def __init__(
        self,
        datasets: dict[int, DatasetRepresentations] | None = None,
        batches: dict[str, Batch] | None = None,
    ) -> None:
        self.datasets: dict[int, DatasetRepresentations] = dict(datasets or {})
        self.batches: dict[str, Batch] = dict(batches or {})

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

    def batch(self, batch_id: str) -> Batch | None:
        """Return the batch with *batch_id*, or ``None``."""
        return self.batches.get(str(batch_id))

    def add_batch(self, batch: Batch) -> None:
        """Register *batch* by its id."""
        self.batches[batch.batch_id] = batch

    # ── divergence & trending ──────────────────────────────────────────────────

    def refresh_divergence(self) -> None:
        """Re-evaluate divergence of every batch member against its canonical model.

        A member whose stored fit model no longer matches the batch's canonical
        model is flagged ``diverged`` and excluded from trending **by default**
        (the first time it diverges).  A member whose model matches again is
        un-flagged and re-included.  A manual trend re-inclusion of a still-
        diverged member is preserved across refreshes.
        """
        for batch in self.batches.values():
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
                        # Model re-converged: drop the flag and re-include.
                        fit.diverged = False
                        fit.include_in_trend = True
                    else:
                        fit.diverged = False
                else:
                    batch.mark_diverged(run_number)
                    fit.diverged = True
                    if not was_diverged:
                        # Newly diverged: exclude from trending by default.
                        fit.include_in_trend = False

    def trend_runs_for_batch(self, batch: Batch) -> list[int]:
        """Return the ordered member runs currently included in trending.

        Inclusion follows each member's ``fit.include_in_trend`` flag, so a
        non-diverged member is included and a manually re-included diverged
        member is too.
        """
        runs: list[int] = []
        for run_number in batch.member_run_numbers:
            representation = self.representation(run_number, batch.rep_type)
            if representation is not None and representation.fit.include_in_trend:
                runs.append(run_number)
        return runs

    def set_member_trend_inclusion(self, batch_id: str, run_number: int, include: bool) -> None:
        """Manually include/exclude a batch member from trending."""
        batch = self.batches.get(str(batch_id))
        if batch is None:
            return
        representation = self.representation(int(run_number), batch.rep_type)
        if representation is not None:
            representation.fit.include_in_trend = bool(include)

    # ── recompute-on-load ──────────────────────────────────────────────────────

    def recompute_all(self, runs_by_number: dict[int, Run]) -> None:
        """Rebuild every representation's transient arrays from its recipe.

        Representations whose run is missing, or whose recipe cannot currently
        be computed (e.g. unimplemented MaxEnt), are left uncomputed rather than
        aborting the whole load.
        """
        for run_number, container in self.datasets.items():
            run = runs_by_number.get(run_number)
            if run is None:
                continue
            for representation in container:
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
        batches: dict[str, Batch] = {}
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
                    batch = Batch.from_dict(batch_data)
                    batches[batch.batch_id] = batch
        return cls(datasets, batches)

    # ── project-dict integration ───────────────────────────────────────────────

    @classmethod
    def from_project_state(cls, project: dict) -> ProjectModel:
        """Build a model from a schema-v6 project dict.

        Reads ``datasets[i].representations`` and the top-level ``batches``.
        """
        datasets: dict[int, DatasetRepresentations] = {}
        batches: dict[str, Batch] = {}
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
                batch = Batch.from_dict(batch_data)
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
