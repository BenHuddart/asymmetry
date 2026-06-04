"""A members × parameters initial-value editor for batch fits.

Exposes every parameter's per-member initial value in one table. ``Local``
parameters are editable per member; ``Global`` / ``Fixed`` parameters are shown
read-only (one shared value across members — they are set on the parameter
table). On accept, :meth:`InitialValuesDialog.edited_values` returns the
per-member values of the editable (local) parameters.

Members are generic ``(key, label)`` pairs — run numbers for F-B asymmetry, or
synthetic ``(run, group)`` keys for grouped fits — so the same dialog serves
both representations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class InitialValuesDialog(QDialog):
    """Editable grid of per-member initial parameter values."""

    def __init__(
        self,
        members: Sequence[tuple[int, str]],
        params: Sequence[tuple[str, str, str]],
        values: Mapping[int, Mapping[str, float]],
        *,
        parent=None,
        title: str = "Initial values",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        # members: [(key, label)]; params: [(name, label, role)] role∈{global,local,fixed}
        self._members = list(members)
        self._params = list(params)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Editable cells are per-member <b>Local</b> initial values. "
            "<b>Global</b>/<b>Fixed</b> parameters are shared (set them on the parameter table)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._table = QTableWidget(len(self._members), len(self._params))
        self._table.setHorizontalHeaderLabels([label for _name, label, _role in self._params])
        self._table.setVerticalHeaderLabels([label for _key, label in self._members])
        for row, (key, _member_label) in enumerate(self._members):
            row_values = values.get(key, {})
            for col, (name, _param_label, role) in enumerate(self._params):
                value = float(row_values.get(name, 0.0))
                item = QTableWidgetItem(f"{value:.6g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if role != "local":
                    # Shared (global) / held (fixed) — read-only, visually muted.
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip(f"{role.capitalize()} parameter (shared); edit on the table.")
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def edited_values(self) -> dict[int, dict[str, float]]:
        """Return ``{member_key: {local_param: value}}`` parsed from the table.

        Only ``Local`` (editable) parameters are returned; non-finite/blank cells
        are skipped.
        """
        result: dict[int, dict[str, float]] = {}
        for row, (key, _member_label) in enumerate(self._members):
            per_member: dict[str, float] = {}
            for col, (name, _param_label, role) in enumerate(self._params):
                if role != "local":
                    continue
                item = self._table.item(row, col)
                if item is None:
                    continue
                try:
                    per_member[name] = float(item.text())
                except (TypeError, ValueError):
                    continue
            if per_member:
                result[int(key)] = per_member
        return result
