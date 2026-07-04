"""Detector-grouping dialog package.

Milestone-M1 mechanical split of the former monolithic
``grouping_dialog.py`` into a dependency-ordered package. The ``.grp``
file Load/Save feature (and its ``grp_io`` module) was retired once
project persistence (grouping profiles saved in ``.asymp``) and
instrument presets made it redundant; the alpha-estimate display helpers
that used to live alongside it now live in :mod:`asymmetry.gui.windows.
grouping.format`.

The public API is re-exported here and, for backward compatibility,
through the thin ``asymmetry.gui.windows.grouping_dialog`` shim module
(which existing imports and tests use). New code should import
:class:`GroupingDialog` from this package root and the formatting
helpers from :mod:`asymmetry.gui.windows.grouping.format`.
"""

from __future__ import annotations

from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from asymmetry.gui.windows.grouping.format import (
    ALPHA_METHOD_ITEMS,
    format_value_with_uncertainty,
)

__all__ = [
    "ALPHA_METHOD_ITEMS",
    "GroupingDialog",
    "format_value_with_uncertainty",
]
