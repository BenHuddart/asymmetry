"""Fingerprint scope panel for the grouping profile editor.

Replaces the old broadcast tick-list. Instead of choosing which runs to *push*
grouping onto, this panel lists every run of the editor's current fingerprint
with a status chip — either ``inherits <profile>`` or ``override`` — and is the
window's **selector**: the run selected here is the one the form previews and
edits (an inheriting run edits the profile draft; an overridden run edits that
run's own override draft). It also lets the user *release* a run from its
profile (freeze its current grouping as an explicit per-run override) or
*reattach* it (drop the override so it inherits again).

The panel owns no project state: it holds a set of released run numbers plus the
set of runs whose override draft has uncommitted edits this session, exposes
them via :meth:`released_run_numbers` / marker text, and emits :attr:`changed`
when the user toggles a run and :attr:`selected` when the current run changes.
The dialog reconciles those sets into the project on Apply.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
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

from asymmetry.gui.styles import tokens


class ScopePanel(QWidget):
    """List runs of one fingerprint and edit their inherit/override state.

    Parameters
    ----------
    parent
        Parent Qt widget.
    """

    #: Emitted whenever the released-run set changes (release / reattach).
    changed = Signal()
    #: Emitted (with the run number) when the current/selected run changes.
    selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty scope panel; call :meth:`set_runs` to populate it."""
        super().__init__(parent)
        #: run_number -> already-overridden (released from the profile) at open.
        self._initial_overridden: dict[int, bool] = {}
        #: run_number -> currently released (drives the chip + apply reconciliation).
        self._released: dict[int, bool] = {}
        #: run_number -> display label.
        self._labels: dict[int, str] = {}
        #: run numbers whose override draft has uncommitted edits this session.
        self._override_dirty: set[int] = set()
        self._profile_name = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._heading = QLabel("Runs of this instrument")
        layout.addWidget(self._heading)

        self._list = QListWidget()
        # Extended selection still drives Release/Reattach on multiple runs, but
        # the *current* item (single) is what the form previews and edits.
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._update_button_states)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
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
        button_row.addWidget(self._release_btn)
        button_row.addWidget(self._reattach_btn)
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
        # A repopulate (instrument switch) drops any override-dirty markers for
        # runs no longer listed.
        self._override_dirty = {rn for rn in self._override_dirty if rn in self._labels}
        self._rebuild()

    def _rebuild(self) -> None:
        current = self.current_run_number()
        blocked = self._list.blockSignals(True)
        try:
            self._list.clear()
            for run_number in sorted(self._labels):
                released = self._released.get(run_number, False)
                chip = "override" if released else f"inherits {self._profile_name}"
                if released and run_number in self._override_dirty:
                    chip += " *"
                item = QListWidgetItem(f"{self._labels[run_number]}  —  {chip}")
                item.setData(Qt.ItemDataRole.UserRole, int(run_number))
                self._tint_item(item, released)
                self._list.addItem(item)
        finally:
            self._list.blockSignals(blocked)
        # Restore (or default) the current row without re-emitting selection.
        target = (
            current if current in self._labels else (min(self._labels) if self._labels else None)
        )
        if target is not None:
            self._set_current_silent(target)
        self._update_button_states()

    @staticmethod
    def _tint_item(item: QListWidgetItem, released: bool) -> None:
        """Tint a row to match the editing-target strip (accent vs warning)."""
        if released:
            item.setForeground(QColor(tokens.WARN))
        else:
            item.setForeground(QColor(tokens.ACCENT))

    def _set_current_silent(self, run_number: int) -> None:
        """Select *run_number* as the current row without emitting :attr:`selected`."""
        for row in range(self._list.count()):
            item = self._list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(run_number):
                blocked = self._list.blockSignals(True)
                try:
                    self._list.setCurrentItem(item)
                finally:
                    self._list.blockSignals(blocked)
                return

    def set_current_run(self, run_number: int) -> None:
        """Programmatically select *run_number*, emitting :attr:`selected` if it changes."""
        if int(run_number) == self.current_run_number():
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(run_number):
                self._list.setCurrentItem(item)
                return

    def current_run_number(self) -> int | None:
        """The run number currently selected (the preview + editing target)."""
        item = self._list.currentItem()
        if item is None:
            return None
        return int(item.data(Qt.ItemDataRole.UserRole))

    def _on_current_item_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            return
        self._update_button_states()
        self.selected.emit(int(current.data(Qt.ItemDataRole.UserRole)))

    def _selected_run_numbers(self) -> list[int]:
        return [int(item.data(Qt.ItemDataRole.UserRole)) for item in self._list.selectedItems()]

    def _update_button_states(self) -> None:
        selected = self._selected_run_numbers()
        any_inherit = any(not self._released.get(rn, False) for rn in selected)
        any_override = any(self._released.get(rn, False) for rn in selected)
        self._release_btn.setEnabled(any_inherit)
        self._reattach_btn.setEnabled(any_override)

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
                # The dialog owns the pending-override-draft confirm/discard (it
                # holds the drafts); it clears the marker via mark_override_dirty.
                changed = True
        if changed:
            self._rebuild()
            self.changed.emit()

    def set_released(self, run_number: int, released: bool) -> None:
        """Set a run's released state programmatically (e.g. to undo a reattach)."""
        run_number = int(run_number)
        if self._released.get(run_number, False) == bool(released):
            return
        self._released[run_number] = bool(released)
        self._rebuild()

    def mark_override_dirty(self, run_number: int, dirty: bool = True) -> None:
        """Flag (or clear) *run_number* as having uncommitted override edits.

        Updates the run's row chip ("override *") without re-emitting selection.
        """
        run_number = int(run_number)
        was = run_number in self._override_dirty
        if dirty:
            self._override_dirty.add(run_number)
        else:
            self._override_dirty.discard(run_number)
        if (run_number in self._override_dirty) != was:
            self._refresh_row_text(run_number)

    def _refresh_row_text(self, run_number: int) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) != int(run_number):
                continue
            released = self._released.get(run_number, False)
            chip = "override" if released else f"inherits {self._profile_name}"
            if released and run_number in self._override_dirty:
                chip += " *"
            item.setText(f"{self._labels[run_number]}  —  {chip}")
            self._tint_item(item, released)
            return

    def override_dirty_run_numbers(self) -> set[int]:
        """Run numbers whose override draft has uncommitted edits this session."""
        return {rn for rn in self._override_dirty if self._released.get(rn, False)}

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
