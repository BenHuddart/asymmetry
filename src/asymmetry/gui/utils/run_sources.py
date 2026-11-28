"""Source locations used for run-number ownership during desktop imports."""

import os
from collections.abc import Iterable, Mapping

from asymmetry.core.data.dataset import MuonDataset


def normalise_source_path(path: str) -> str:
    """Compare source locators independently of spelling and symbolic links."""
    return os.path.normcase(os.path.realpath(path)) if path else ""


def dataset_source_file(dataset: MuonDataset) -> str:
    """Return the acquisition path, using metadata for curve-only imports."""
    path = dataset.run.source_file if dataset.run is not None else ""
    return str(path or dataset.metadata.get("source_file") or "")


def first_run_source_collision(
    existing: Mapping[int, MuonDataset], incoming: Iterable[MuonDataset]
) -> str | None:
    """Find a row whose run number belongs to a different source."""
    for dataset in incoming:
        number = int(dataset.run_number)
        if number not in existing:
            continue
        current = dataset_source_file(existing[number])
        candidate = dataset_source_file(dataset)
        if normalise_source_path(current) != normalise_source_path(candidate):
            current_label = f'"{current}"' if current else "an in-memory dataset"
            candidate_label = f'"{candidate}"' if candidate else "an in-memory dataset"
            return (
                f"Run {number} is already loaded from {current_label}. "
                f"Could not load {candidate_label}. Use File → New Project for the other source."
            )
    return None
