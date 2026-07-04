"""Fingerprint scope panel for the grouping profile editor.

Replaces the old broadcast tick-list. Instead of choosing which runs to *push*
grouping onto, this panel shows every run of the editor's current fingerprint
with a status chip — either ``inherits <profile>`` or ``override`` — and lets the
user *release* a run from its profile (freeze its current grouping as an explicit
per-run override) or *reattach* it (drop the override so it inherits again).

The panel owns no project state: it holds a set of released run numbers, exposes
it via :meth:`released_run_numbers`, and emits :attr:`changed` when the user
toggles a run. The dialog reconciles that set into the project on Apply.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ScopePanel(QWidget):
    """List runs of one fingerprint and edit their inherit/override state.

    Parameters
    ----------
    parent
        Parent Qt widget.
    """

    #: Emitted whenever the released-run set changes (release / reattach).
    changed = Signal()
    #: Emitted (with the run number) when the user asks to edit an override.
    edit_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty scope panel; call :meth:`set_runs` to populate it."""
        super().__init__(parent)
        #: run_number -> already-overridden (released from the profile) at open.
        self._initial_overridden: dict[int, bool] = {}
        #: run_number -> currently released (drives the chip + apply reconciliation).
        self._released: dict[int, bool] = {}
        #: run_number -> display label.
        self._labels: dict[int, str] = {}
        self._profile_name = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._heading = QLabel("Runs of this instrument")
        layout.addWidget(self._heading)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._update_button_states)
        layout.addWidget(self._list)

        button_row = QHBoxLayout()
        self._release_btn = QPushButton("Release from profile")
        self._release_btn.setAutoDefault(False)
        self._release_btn.setDefault(False)
        self._release_btn.setToolTip(
            "Freeze the selected runs' current grouping as a per-run override, so "
            "applying this profile leaves them unchanged."
        )
        self._release_btn.clicked.connect(self._on_release)
        self._reattach_btn = QPushButton("Reattach to profile")
        self._reattach_btn.setAutoDefault(False)
        self._reattach_btn.setDefault(False)
        self._reattach_btn.setToolTip(
            "Drop the per-run override so the selected runs inherit this profile again."
        )
        self._reattach_btn.clicked.connect(self._on_reattach)
        self._edit_btn = QPushButton("Edit…")
        self._edit_btn.setAutoDefault(False)
        self._edit_btn.setDefault(False)
        self._edit_btn.setToolTip(
            "Edit the selected overridden run's own grouping — changes apply to "
            "that run only. Selects it as the preview run."
        )
        self._edit_btn.clicked.connect(self._on_edit)
        button_row.addWidget(self._release_btn)
        button_row.addWidget(self._reattach_btn)
        button_row.addWidget(self._edit_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

    def set_runs(
        self,
        runs: list[tuple[int, str, bool]],
        *,
        profile_name: str,
    ) -> None:
        """Populate the panel.

        Parameters
        ----------
        runs
            ``(run_number, label, overridden)`` triples for every run of the
            current fingerprint. ``overridden`` is the run's state at open.
        profile_name
            Name of the profile inheriting runs are attached to (for the chips).
        """
        self._profile_name = str(profile_name)
        self._initial_overridden = {int(rn): bool(ov) for rn, _label, ov in runs}
        self._released = dict(self._initial_overridden)
        self._labels = {int(rn): str(label) for rn, label, _ov in runs}
        self._rebuild()

    def _rebuild(self) -> None:
        self._list.clear()
        for run_number in sorted(self._labels):
            released = self._released.get(run_number, False)
            chip = "override" if released else f"inherits {self._profile_name}"
            item = QListWidgetItem(f"{self._labels[run_number]}  —  {chip}")
            item.setData(Qt.ItemDataRole.UserRole, int(run_number))
            self._list.addItem(item)
        self._update_button_states()

    def _selected_run_numbers(self) -> list[int]:
        return [int(item.data(Qt.ItemDataRole.UserRole)) for item in self._list.selectedItems()]

    def _update_button_states(self) -> None:
        selected = self._selected_run_numbers()
        any_inherit = any(not self._released.get(rn, False) for rn in selected)
        any_override = any(self._released.get(rn, False) for rn in selected)
        self._release_btn.setEnabled(any_inherit)
        self._reattach_btn.setEnabled(any_override)
        # Edit… acts on exactly one overridden run.
        overridden_selected = [rn for rn in selected if self._released.get(rn, False)]
        self._edit_btn.setEnabled(len(overridden_selected) == 1)

    def _on_edit(self) -> None:
        overridden_selected = [
            rn for rn in self._selected_run_numbers() if self._released.get(rn, False)
        ]
        if len(overridden_selected) == 1:
            self.edit_requested.emit(int(overridden_selected[0]))

    def _on_release(self) -> None:
        changed = False
        for rn in self._selected_run_numbers():
            if not self._released.get(rn, False):
                self._released[rn] = True
                changed = True
        if changed:
            self._rebuild()
            self.changed.emit()

    def _on_reattach(self) -> None:
        changed = False
        for rn in self._selected_run_numbers():
            if self._released.get(rn, False):
                self._released[rn] = False
                changed = True
        if changed:
            self._rebuild()
            self.changed.emit()

    def released_run_numbers(self) -> set[int]:
        """Run numbers currently marked as released (per-run override)."""
        return {rn for rn, released in self._released.items() if released}

    def inheriting_run_numbers(self) -> set[int]:
        """Run numbers currently inheriting the profile (no override)."""
        return {rn for rn in self._labels if not self._released.get(rn, False)}

    def newly_released(self) -> set[int]:
        """Runs released in this session that were not overridden at open."""
        return {
            rn for rn in self.released_run_numbers() if not self._initial_overridden.get(rn, False)
        }

    def newly_reattached(self) -> set[int]:
        """Runs reattached in this session that were overridden at open."""
        return {
            rn for rn in self.inheriting_run_numbers() if self._initial_overridden.get(rn, False)
        }


__all__ = ["ScopePanel"]
