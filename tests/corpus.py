"""The WiMDA muon-school corpus, for tests that read real instrument data.

The corpus is never committed; ``ASYMMETRY_CORPUS_ROOT`` names the extracted
release (see AGENTS.md § "Test corpus"). Tests that need it skip when it is
absent, which is always the case on CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CORPUS_ROOT = (
    Path(os.environ["ASYMMETRY_CORPUS_ROOT"]) if "ASYMMETRY_CORPUS_ROOT" in os.environ else None
)


def corpus_path(relative: str) -> Path | None:
    """*relative* under the corpus root, or None when the corpus or that path is absent."""
    if CORPUS_ROOT is None:
        return None
    path = CORPUS_ROOT / relative
    return path if path.exists() else None


def requires_corpus(path: Path | None) -> pytest.MarkDecorator:
    """Skip unless *path* (from :func:`corpus_path`) exists."""
    return pytest.mark.skipif(
        path is None, reason="WiMDA corpus not available (set ASYMMETRY_CORPUS_ROOT)"
    )
