"""Shared helpers for corpus-driven screenshot scenarios.

Corpus scenarios subclass :class:`CorpusScenario`, which extends the standard
:class:`~docs.screenshots.scenarios._base.Scenario` with corpus-root
resolution. The root comes from ``ASYMMETRY_CORPUS_ROOT`` so the same
scenarios run locally and in CI (where the corpus is provisioned separately).

House rules (same as the synthetic scenarios, see ``docs/README.md``):
deterministic, cropped to the panel the prose will discuss, fast, and within
the 600 KB per-PNG budget. Scenario names must start with ``corpus_``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from .._base import Scenario, _process_events_for, register  # noqa: F401  (re-exported)

DEFAULT_CORPUS_ROOT = "/home/ben/Documents/WiMDA muon school-indexed"


def corpus_root() -> Path:
    """Return the corpus root, honouring ``ASYMMETRY_CORPUS_ROOT``."""
    root = Path(os.environ.get("ASYMMETRY_CORPUS_ROOT", DEFAULT_CORPUS_ROOT))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus root not found: {root} — set ASYMMETRY_CORPUS_ROOT to the "
            "'WiMDA muon school' corpus directory."
        )
    return root


def corpus_path(relative: str) -> Path:
    """Resolve *relative* against the corpus root, requiring it to exist."""
    path = corpus_root() / relative
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return path


def load_corpus_datasets(relative_paths: list[str], **load_kwargs):
    """Load corpus files through the real loader; returns a flat dataset list.

    ``load()`` may return one dataset or a list (multi-period files); the
    result here is always flat so callers can iterate uniformly.
    """
    from asymmetry.core.io import load

    datasets = []
    for rel in relative_paths:
        loaded = load(str(corpus_path(rel)), **load_kwargs)
        if isinstance(loaded, list):
            datasets.extend(loaded)
        else:
            datasets.append(loaded)
    return datasets


class CorpusScenario(Scenario):
    """Base class for scenarios that render real corpus data.

    Subclasses set :attr:`example` to the corpus-relative example folder
    (e.g. ``"Magnetism/Magnetic ordering in EuO"``) so tooling can associate
    scenarios with their ``GROUND_TRUTH.md``.
    """

    example: ClassVar[str] = ""

    def add_to_browser(self, window, datasets) -> None:
        """Add *datasets* to the main window's data browser."""
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
