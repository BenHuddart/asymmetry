"""Corpus-driven documentation scenarios (WiMDA muon school examples).

Unlike the synthetic-archetype scenarios in the parent package, these load
**real data files** from the WiMDA muon school corpus and drive the GUI
through the worked analyses graded by each example's ``GROUND_TRUTH.md``.

The corpus root is resolved by :mod:`._corpus` from ``ASYMMETRY_CORPUS_ROOT``
(default: the local corpus path). Capture is driven by
``docs.screenshots.capture_corpus`` — these scenarios are deliberately *not*
imported by the standard ``docs.screenshots.capture`` driver, so the normal
docs build and ``--check-refs`` are unaffected until individual scenarios
graduate into the published docs.

Every ``*.py`` module in this package that does not start with an underscore
is auto-imported below, so adding a new example module requires no edits to
any shared file.
"""

from __future__ import annotations

import importlib
import pkgutil


def import_all_scenario_modules() -> list[str]:
    """Import every non-underscore module in this package; return their names."""
    imported: list[str] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")
        imported.append(info.name)
    return imported
