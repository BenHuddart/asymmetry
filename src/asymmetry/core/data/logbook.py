"""Logbook — run-table management for collections of μSR runs.

A Logbook maintains a table of loaded runs with their key metadata,
supports filtering / sorting / tagging, and can be persisted to disk.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from asymmetry.core.data.dataset import MuonDataset


@dataclass
class LogbookEntry:
    """One row in the logbook."""

    run_number: int
    title: str = ""
    temperature: float = 0.0
    field: float = 0.0
    comment: str = ""
    tags: list[str] = dc_field(default_factory=list)
    source_file: str = ""
    extra: dict[str, Any] = dc_field(default_factory=dict)


class Logbook:
    """A searchable, sortable table of μSR runs."""

    def __init__(self) -> None:
        self._entries: dict[int, LogbookEntry] = {}
        self._datasets: dict[int, MuonDataset] = {}
        self._collections: dict[str, list[int]] = {}

    # --- entry management -----------------------------------------------

    def add(self, dataset: MuonDataset, tags: list[str] | None = None) -> LogbookEntry:
        """Register a dataset in the logbook."""
        meta = dataset.metadata
        entry = LogbookEntry(
            run_number=dataset.run_number,
            title=meta.get("title", ""),
            temperature=float(meta.get("temperature", 0.0)),
            field=float(meta.get("field", 0.0)),
            comment=meta.get("comment", ""),
            tags=tags or [],
            source_file=meta.get("source_file", ""),
        )
        self._entries[entry.run_number] = entry
        self._datasets[entry.run_number] = dataset
        return entry

    def remove(self, run_number: int) -> None:
        self._entries.pop(run_number, None)
        self._datasets.pop(run_number, None)

    def get_entry(self, run_number: int) -> LogbookEntry | None:
        return self._entries.get(run_number)

    def get_dataset(self, run_number: int) -> MuonDataset | None:
        return self._datasets.get(run_number)

    @property
    def run_numbers(self) -> list[int]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LogbookEntry]:
        for rn in sorted(self._entries):
            yield self._entries[rn]

    # --- filtering & searching ------------------------------------------

    def filter(self, **criteria: Any) -> list[LogbookEntry]:
        """Return entries matching all *criteria* (field=value pairs)."""
        results = []
        for entry in self:
            if all(getattr(entry, k, None) == v for k, v in criteria.items()):
                results.append(entry)
        return results

    def search(self, text: str) -> list[LogbookEntry]:
        """Free-text search across title, comment and tags."""
        text_lower = text.lower()
        results = []
        for entry in self:
            haystack = f"{entry.title} {entry.comment} {' '.join(entry.tags)}".lower()
            if text_lower in haystack:
                results.append(entry)
        return results

    # --- collections (named groups of runs) -----------------------------

    def create_collection(self, name: str, run_numbers: list[int]) -> None:
        self._collections[name] = list(run_numbers)

    def get_collection(self, name: str) -> list[int]:
        return list(self._collections.get(name, []))

    @property
    def collections(self) -> list[str]:
        return sorted(self._collections)

    # --- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save logbook metadata to a JSON file."""
        data = {
            "entries": [
                {
                    "run_number": e.run_number,
                    "title": e.title,
                    "temperature": e.temperature,
                    "field": e.field,
                    "comment": e.comment,
                    "tags": e.tags,
                    "source_file": e.source_file,
                    "extra": e.extra,
                }
                for e in self
            ],
            "collections": self._collections,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_metadata(self, path: str | Path) -> None:
        """Restore logbook entries from a JSON file (data files must be reloaded separately)."""
        raw = json.loads(Path(path).read_text())
        for item in raw.get("entries", []):
            entry = LogbookEntry(**item)
            self._entries[entry.run_number] = entry
        self._collections = raw.get("collections", {})
