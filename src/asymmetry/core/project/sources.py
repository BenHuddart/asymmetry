"""Verify current project inputs and explicitly accept selected replacements.

Historical declarations are informational. Decisions never rewrite result recipes,
calibration expectations, or the observations and execution recorded by a fit.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from asymmetry.core.data.acquisition import SourceBinding, SourceVerification, verify_sources
from asymmetry.core.project.preparation import PreparedProject, prepare_project


def source_candidates(prepared, project_path, *, directory=None):
    """Resolve project-relative locators; an optional directory only proposes basenames."""
    import ntpath

    base = Path(project_path).absolute().parent
    paths = []
    for source in prepared.sources:
        path = Path(source.source_file)
        if not path.is_absolute():
            path = base / path
        if directory is not None and not path.is_file():
            path = Path(directory) / ntpath.basename(source.source_file)
        paths.append(str(path))
    return tuple(paths)


@dataclass(frozen=True)
class ProjectSourceVerification:
    """Fresh facts bound to the actual current owners of a detached project."""

    prepared: PreparedProject
    facts: tuple[SourceVerification, ...]


def verify_project_sources(
    prepared,
    project_path,
    *,
    candidates=None,
    checkpoint=lambda: None,
    progress=lambda done, total: None,
):
    """Verify bytes and bind following-profile requests to actual acquired fingerprints.

    Profile selection uses the existing loader only when following inputs need it.
    Decoded facts remain detached, and a capture mismatch aborts before application.
    """
    from asymmetry.core.io import load
    from asymmetry.core.io.periods import source_run_of
    from asymmetry.core.project.profiles import default_profile_for_run, named_profile_for_run

    facts = verify_sources(
        prepared.sources,
        candidates=source_candidates(prepared, project_path) if candidates is None else candidates,
        checkpoint=checkpoint,
        progress=progress,
    )
    profile_refs = [
        f
        for f in facts
        if f.source.role == "reference" and f.source.owner.address[0] == "grouping_profiles"
    ]
    used = set()
    loaded = {}
    if any(f.source.owner.active for f in profile_refs):
        for fact in facts:
            checkpoint()
            if (
                fact.source.role != "sample"
                or not fact.source.owner.active
                or fact.observed is None
            ):
                continue
            entry = source_record(prepared.state, fact.source.owner.address)
            if entry.get("grouping_overrides") is not None:
                continue
            path = fact.candidate_file
            if path not in loaded:
                try:
                    loaded[path] = load(path, checkpoint=checkpoint)
                except InterruptedError:
                    raise
                except (OSError, ValueError):
                    loaded[path] = None
            value = loaded[path]
            datasets = value if isinstance(value, list) else [value]
            for dataset in datasets:
                if dataset is None or dataset.run is None:
                    continue
                capture = dataset.run.source_binding.observed
                legacy_unknown = fact.status == "legacy_unknown" and capture is None
                if capture != fact.observed and not legacy_unknown:
                    raise ValueError(
                        "Source changed after verification; project application cancelled"
                    )
                if (
                    len(datasets) != 1
                    and dataset.run_number != fact.source.run_number
                    and source_run_of(dataset.run) != fact.source.run_number
                ):
                    continue
                profile = named_profile_for_run(
                    prepared.profiles, entry.get("profile", ""), dataset.run
                )
                if profile is None:
                    profile = default_profile_for_run(prepared.profiles, dataset.run)
                if profile is not None:
                    used.add(
                        next(
                            i
                            for i, candidate in enumerate(prepared.profiles)
                            if candidate is profile
                        )
                    )
    sources = tuple(
        replace(
            source,
            owner=replace(
                source.owner, active=source.owner.active and source.owner.address[1] in used
            ),
        )
        if source.role == "reference" and source.owner.address[0] == "grouping_profiles"
        else source
        for source in prepared.sources
    )
    rebound = replace(prepared, sources=sources)
    facts = tuple(replace(fact, source=source) for fact, source in zip(facts, sources, strict=True))
    checkpoint()
    return ProjectSourceVerification(rebound, facts)


def source_record(state, address):
    """Access a typed owner address supplied by project preparation, without searching."""
    for component in address:
        state = state[component]
    return state


@dataclass(frozen=True)
class ProjectSourceApplication:
    """Detached decisions and accepted read facts for one deliberate project application."""

    prepared: PreparedProject
    verification: tuple[SourceVerification, ...]
    replacements: frozenset[int]

    def available(self, index):
        """Whether this input may be consumed by the current operation."""
        return (
            self.verification[index].status in {"matched", "legacy_unknown"}
            or index in self.replacements
        )

    def input_for(self, address):
        """Return the verification index for a current owner, or None for absent provenance."""
        return next(
            (
                i
                for i, fact in enumerate(self.verification)
                if fact.source.owner.active and fact.source.owner.address == address
            ),
            None,
        )

    def capture_matches(self, index, observed):
        """Legacy unknown capture stays unverified; accepted known content must match."""
        fact = self.verification[index]
        if not self.available(index):
            return False
        if observed is None and fact.status == "legacy_unknown" and index not in self.replacements:
            return True
        return fact.observed is None or observed == fact.observed

    def check_capture(self, index, observed):
        """Reject changed captures and unknown capture for accepted known inputs."""
        if not self.capture_matches(index, observed):
            raise ValueError(
                "Source changed after verification or is unavailable; reopen to choose its replacement"
            )

    def reference_inputs(self, payload):
        """Carry the exact prepared reference owners through mutable legacy restoration."""
        indices = tuple(
            i
            for i, fact in enumerate(self.verification)
            if fact.source.owner.active
            and fact.source.role == "reference"
            and source_record(self.prepared.state, fact.source.owner.address) is payload
        )
        if not indices:
            raise ValueError("Reference request lost its prepared project owner")
        for index in indices:
            source = self.verification[index].source
            if source.source_file or source.expected is not None:
                if not self.available(index):
                    raise ValueError("Reference source is unavailable pending explicit replacement")
        return indices

    def check_reference_capture(self, indices, reference):
        """Compare actual reference counts against the pre-resolution accepted request."""
        for index in indices:
            source = self.verification[index].source
            if source.source_file or source.expected is not None:
                self.check_capture(index, reference.content_identity)

    def checked_samples(self, loaded):
        """Exclude changed captures before any acquired facts can heal project policies."""
        checked = dict(loaded)
        for i, fact in enumerate(self.verification):
            if (
                not fact.source.owner.active
                or fact.source.role != "sample"
                or not self.available(i)
            ):
                continue
            value = checked.get(fact.candidate_file)
            candidates = value if isinstance(value, list) else [value]
            for dataset in candidates:
                if dataset is None or isinstance(dataset, Exception):
                    continue
                observed = dataset.run.source_binding.observed if dataset.run is not None else None
                if not self.capture_matches(i, observed):
                    checked[fact.candidate_file] = ValueError(
                        "Source changed after verification; reopen to choose its replacement"
                    )
        return checked

    def dataset_paths(self):
        """Accepted candidate targets, keyed by the existing current dataset owner."""
        return {
            entry["run_number"]: (
                self.verification[i].candidate_file if self.available(i) else None
            )
            for i, fact in enumerate(self.verification)
            if fact.source.owner.active and fact.source.role == "sample"
            for entry in [source_record(self.prepared.state, fact.source.owner.address)]
        }

    def check_constituents(self, index, datasets):
        """Verify consumed constituent captures; preserve their ordered selections."""
        from asymmetry.core.data.combine import CombineError
        from asymmetry.core.io.reference_selection import select_reference_run

        for i, fact in enumerate(self.verification):
            if not fact.source.owner.active or fact.source.owner.address[:2] != (
                "combined_datasets",
                index,
            ):
                continue
            source = fact.source
            if not self.available(i):
                raise CombineError("Constituent source is unavailable pending explicit replacement")
            request = {"source_file": fact.candidate_file, "run_number": source.run_number}
            if source.period_number is not None:
                request["period_number"] = source.period_number
            runs = [
                selected
                for ds in datasets
                if ds.run is not None
                for selected in [select_reference_run(ds.run, request)]
                if selected is not None
            ]
            if not runs:
                raise CombineError("Constituent acquisition is unavailable")
            for run in runs:
                self.check_capture(i, run.source_binding.observed)

    def report(self):
        """Publish statuses and owner addresses without locators or private diagnostics."""
        return [
            {
                "index": i,
                "owner": list(fact.source.owner.address),
                "active": fact.source.owner.active,
                "role": fact.source.role,
                "status": fact.status,
                "replacement": i in self.replacements,
                "expected": fact.source.expected.to_dict() if fact.source.expected else None,
                "observed": fact.observed.to_dict() if fact.observed else None,
                "run_number": fact.source.run_number,
                "period_number": fact.source.period_number,
                "coefficient": fact.source.coefficient,
                "input_number": fact.source.input_number,
                "dae_period_number": fact.source.dae_period_number,
            }
            for i, fact in enumerate(self.verification)
        ]


def apply_source_decisions(verification, *, replacements=(), checkpoint=lambda: None):
    """Accept only explicitly selected current owners from this verification request.

    Unselected changed inputs remain unavailable (history only). Matched locators
    may move, legacy expectations remain unknown, and historical records stay intact.
    Callers must compare actual acquisition capture before using accepted inputs.
    No reduction, model fitting, calibration, or file writes occur here.
    """
    checkpoint()
    prepared = verification.prepared
    facts = verification.facts
    selected = frozenset(replacements)
    if tuple(f.source for f in facts) != prepared.sources:
        raise ValueError("Source decisions require this prepared project's complete verification")
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(facts):
            raise ValueError("Replacement must identify a prepared source index")
        fact = facts[index]
        if (
            not fact.source.owner.active
            or fact.status not in {"matched", "changed", "legacy_unknown"}
            or fact.observed is None
        ):
            raise ValueError("Replacement requires a readable current input owner")
    state = deepcopy(prepared.state)
    for index, fact in enumerate(facts):
        checkpoint()
        if not fact.source.owner.active:
            continue
        row = source_record(state, fact.source.owner.address)
        if fact.status in {"matched", "legacy_unknown"} or index in selected:
            row["source_file"] = fact.candidate_file
            if index in selected:
                if fact.source.role == "reference":
                    row["content_identity"] = fact.observed.to_dict()
                else:
                    row["source_binding"] = SourceBinding(fact.observed, fact.observed).to_dict()
    updated = prepare_project(state, checkpoint=checkpoint)
    updated = replace(
        updated,
        sources=tuple(
            replace(source, owner=fact.source.owner)
            for source, fact in zip(updated.sources, facts, strict=True)
        ),
    )
    return ProjectSourceApplication(updated, facts, selected)
