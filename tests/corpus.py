"""The WiMDA muon-school corpus, for tests that read real instrument data.

The corpus is never committed; ``ASYMMETRY_CORPUS_ROOT`` names the extracted
release (see AGENTS.md § "Test Corpus"). Tests that need it skip when the
variable is unset, which is always the case on CI. Once it is set, a missing
file is a configuration error (the wrong release, a renamed run), not a skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CORPUS_ROOT = (
    Path(os.environ["ASYMMETRY_CORPUS_ROOT"]) if "ASYMMETRY_CORPUS_ROOT" in os.environ else None
)


def corpus_path(relative: str) -> Path | None:
    """*relative* under the corpus root, or None when no corpus is configured."""
    if CORPUS_ROOT is None:
        return None
    path = CORPUS_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(
            f"{relative!r} is not in the corpus at ASYMMETRY_CORPUS_ROOT={CORPUS_ROOT}"
        )
    return path


def requires_corpus(path: Path | None) -> pytest.MarkDecorator:
    """Skip unless a corpus is configured (*path* is from :func:`corpus_path`)."""
    return pytest.mark.skipif(
        path is None, reason="WiMDA corpus not configured (set ASYMMETRY_CORPUS_ROOT)"
    )
