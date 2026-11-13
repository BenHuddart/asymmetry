"""Immutable byte identity, historical expectations, and explicit verification facts.

Content identity says nothing about computation compatibility or statistical
independence. A missing expected digest cannot be reconstructed from today's file.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from asymmetry.core.utils.content import ContentReadError, hash_file


@dataclass(frozen=True)
class ContentIdentity:
    """SHA-256 of the complete acquisition, independent of its locator."""

    sha256: str

    def __post_init__(self):
        if not isinstance(self.sha256, str) or re.fullmatch("[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("Content identity requires a lowercase SHA-256 digest")

    def to_dict(self) -> dict:
        """Return the fixed-algorithm canonical codec."""
        return {"sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict | None):
        """Decode a recorded identity; absence remains unknown."""
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"sha256"}:
            raise ValueError("Content identity requires only sha256")
        return cls(value["sha256"])

    @classmethod
    def read(cls, path: str | Path, **kwargs):
        """Observe complete current bytes, with hash_file checkpoints/progress."""
        return cls(hash_file(path, **kwargs))


@dataclass(frozen=True)
class SourceBinding:
    """Saved expectation and actual read fact, with independent unknowns.

    On a fresh acquisition both describe that read. Restoration retains the
    saved expectation (including None) and the new read's observed fact.
    """

    expected: ContentIdentity | None = None
    observed: ContentIdentity | None = None

    def to_dict(self) -> dict:
        """Encode both contexts without resolving any locator."""
        return {
            name: value.to_dict() if value is not None else None
            for name, value in (("expected", self.expected), ("observed", self.observed))
        }

    @classmethod
    def from_dict(cls, value: dict | None):
        """Decode optional provenance; never promote an observed fact to expected."""
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) != {"expected", "observed"}:
            raise ValueError("Source binding requires expected and observed")
        return cls(*(ContentIdentity.from_dict(value[key]) for key in ("expected", "observed")))


@dataclass(frozen=True)
class SourceDeclaration:
    """One ordered source role; matching content never merges scientific terms."""

    role: str
    source_file: str
    expected: ContentIdentity | None
    run_number: int | None = None
    period_number: int | None = None
    coefficient: float = 1.0
    input_number: int | None = None
    dae_period_number: int | None = None


@dataclass(frozen=True)
class SourceVerification:
    """Fresh comparison only, never a persisted certification or state mutation."""

    source: SourceDeclaration
    candidate_file: str
    status: Literal["matched", "changed", "missing", "legacy_unknown", "unreadable"]
    observed: ContentIdentity | None = None
    diagnostic: str = ""


def verify_sources(
    sources: Iterable[SourceDeclaration],
    *,
    candidates: Iterable[str | Path] | None = None,
    checkpoint: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> tuple[SourceVerification, ...]:
    """Compare caller-selected candidates; read each physical path once per request.

    No path search, substitution, history edits, or recomputation occurs. Missing
    expectations remain legacy_unknown even when current bytes can be observed.
    Missing/unreadable candidates retain their separate availability outcomes.
    Cancellation raised by checkpoint aborts without returning partial facts.
    """
    sources = tuple(sources)
    paths = tuple(candidates) if candidates is not None else tuple(s.source_file for s in sources)
    if len(paths) != len(sources):
        raise ValueError("Verification requires one candidate per declaration")
    cache = {}
    results = []
    for source, candidate in zip(sources, paths, strict=True):
        checkpoint()
        path = Path(candidate)
        try:
            key = path.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(
                SourceVerification(source, str(candidate), "unreadable", diagnostic=str(exc))
            )
            continue
        if key not in cache:
            try:
                observed = ContentIdentity.read(path, checkpoint=checkpoint, progress=progress)
            except ContentReadError as exc:
                cache[key] = (
                    None,
                    "missing" if isinstance(exc.__cause__, FileNotFoundError) else "unreadable",
                    str(exc),
                )
            else:
                cache[key] = (observed, "", "")
        observed, unavailable, diagnostic = cache[key]
        status = unavailable or (
            "legacy_unknown"
            if source.expected is None
            else "matched"
            if source.expected == observed
            else "changed"
        )
        results.append(SourceVerification(source, str(candidate), status, observed, diagnostic))
    checkpoint()
    return tuple(results)


def constituent_sources_for_project(dataset) -> list[dict]:
    """Retain saved term expectations, with fresh observations kept separate.

    Original role/order/selection/coefficient context belongs to the saved
    combination. A current reconstructed curve has its own result recipe.
    """
    from copy import deepcopy

    current = dataset.metadata.get("combination", {}).get("constituents", [])
    saved = dataset.metadata.get("constituent_source_expectations")
    if saved is None:
        return deepcopy(current)
    result = deepcopy(saved)
    for row in result:
        original = SourceBinding.from_dict(row.get("source_binding"))
        matches = [
            r
            for r in current
            if all(
                row.get(k) is None or r.get(k) == row[k]
                for k in (
                    "source_run_number",
                    "data_period_number",
                    "dae_period_number",
                    "input_ordinal",
                )
            )
        ]
        match = matches[0] if len(matches) == 1 else None
        observed = (
            SourceBinding.from_dict(match.get("source_binding")).observed
            if match is not None
            else None
        )
        row["source_binding"] = SourceBinding(original.expected, observed).to_dict()
    return result


def restore_constituent_expectations(dataset, saved: list[dict] | None) -> None:
    """Transfer historical expected rows; absence stays unknown after reconstruction."""
    from copy import deepcopy

    rows = deepcopy(
        saved
        if saved is not None
        else dataset.metadata.get("combination", {}).get("constituents", [])
    )
    if saved is None:
        for row in rows:
            observed = SourceBinding.from_dict(row.get("source_binding")).observed
            row["source_binding"] = SourceBinding(None, observed).to_dict()
    dataset.metadata["constituent_source_expectations"] = rows
