"""Corpus-driven documentation scenarios (WiMDA muon school examples).

Unlike the synthetic-archetype scenarios in the parent package, these load
**real data files** from the WiMDA muon school corpus and drive the GUI
through the worked analyses graded by each example's ``GROUND_TRUTH.md``.

The corpus root is resolved by :mod:`._corpus` from ``ASYMMETRY_CORPUS_ROOT``.
:mod:`docs.screenshots.capture_corpus` statically discovers scenarios in
non-underscore modules and imports the modules selected for capture.
"""
