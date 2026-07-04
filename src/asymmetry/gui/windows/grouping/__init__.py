"""Detector-grouping dialog package.

Milestone-M1 mechanical split of the former monolithic
``grouping_dialog.py`` into a dependency-ordered package:

``grp_io`` (leaf: ``.grp`` serialization + formatting helpers) → ``dialog``
(the :class:`GroupingDialog` shell).

No behaviour changed in the split. The public API is re-exported here and,
for backward compatibility, through the thin
``asymmetry.gui.windows.grouping_dialog`` shim module (which existing imports
and tests use). New code should import :class:`GroupingDialog` from this
package root and the ``.grp`` helpers from
:mod:`asymmetry.gui.windows.grouping.grp_io`.
"""

from __future__ import annotations

from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from asymmetry.gui.windows.grouping.grp_io import (
    ALPHA_METHOD_ITEMS,
    format_value_with_uncertainty,
    parse_grp,
    serialize_grp,
)

__all__ = [
    "ALPHA_METHOD_ITEMS",
    "GroupingDialog",
    "format_value_with_uncertainty",
    "parse_grp",
    "serialize_grp",
]
