"""Deprecated shim — ``grouping_dialog`` was split into the ``grouping`` package.

Import :class:`GroupingDialog` from :mod:`asymmetry.gui.windows.grouping` (or the
``.grp`` helpers from :mod:`asymmetry.gui.windows.grouping.grp_io`) instead. This
module re-exports the public API for backward compatibility so existing imports
and tests keep working unchanged after the milestone-M1 package split.

``QMessageBox`` is re-exported here because tests monkeypatch
``grouping_dialog.QMessageBox.warning``; patching an attribute on the class object
affects every reference to it, so the dialog's warnings are intercepted. The
underscore-prefixed ``_format_value_with_uncertainty`` / ``_ALPHA_METHOD_ITEMS``
aliases preserve the historical module-level names.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from asymmetry.gui.windows.grouping.grp_io import (  # noqa: F401  (re-export)
    ALPHA_METHOD_ITEMS as _ALPHA_METHOD_ITEMS,
)
from asymmetry.gui.windows.grouping.grp_io import (  # noqa: F401  (re-export)
    format_value_with_uncertainty as _format_value_with_uncertainty,
)

__all__ = [
    "GroupingDialog",
    "QMessageBox",
]
