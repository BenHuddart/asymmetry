"""Runnable, paper-shaped example datasets built from the public core API.

Each module here writes loadable synthetic data (through the repository's own
NeXus writer and simulation engine) whose *truth* is a set of published
parameter values, so a full analysis in the GUI or the API reproduces the
paper's results. The examples are deliberately GUI-free: they live in
``asymmetry.core``-only territory and are importable without PySide6.

Available examples
------------------
- :mod:`asymmetry.examples.ybzn2gao5` — the longitudinal-field spin-dynamics
  study of the Dirac U(1) quantum spin liquid YbZn2GaO5 (Wu *et al.*,
  arXiv:2502.00130). Generates ~160 synthetic runs (eight temperatures ×
  twenty fields) whose λ(B) relaxation follows the paper's Eq. (1), so a
  two-level batch-then-global-fit analysis recovers the Table I parameters.
"""

from __future__ import annotations

from asymmetry.examples import ybzn2gao5

__all__ = ["ybzn2gao5"]
