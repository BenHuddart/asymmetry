"""Fingerprint scope panel for the grouping profile editor.

Replaces the old broadcast tick-list. Instead of choosing which runs to *push*
grouping onto, this panel lists every run of the editor's current fingerprint
with a status chip — ``follows <profile>`` (its assigned profile) or
``override`` — and is the window's **selector**: the run selected here is the
one the form previews and edits (a following run edits the profile draft; an
overridden run edits that run's own override draft). It lets the user
*release* a run from its profile (freeze its current grouping as an explicit
per-run override), *reattach* it (drop the override so it follows its
assigned profile again), or *assign* it to another profile of the fingerprint
(schema v17 — several profiles can be in concurrent use, e.g. one per
sample).

The panel owns no project state: it holds working released/assigned maps
seeded by :meth:`set_runs`, exposes them via :meth:`released_run_numbers` /
:meth:`assignments` / marker text, and emits :attr:`changed` when the user
toggles or reassigns a run and :attr:`selected` when the current run changes.
The dialog reconciles those maps into the project on Apply.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens


class ScopePanel(QWidget):
    """List runs of one fingerprint and edit their follow/override/assignment state.

    Parameters
    ----------
    parent
        Parent Qt widget.
    """

    #: Emitted whenever the released-run set or an assignment changes.
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
        #: run_number -> assigned profile name at open (for newly_assigned()).
        self._initial_assigned: dict[int, str] = {}
        #: run_number -> currently assigned profile name.
        self._assigned: dict[int, str] = {}
        #: profile names of the current fingerprint (the Assign-to menu).
        self._profile_names: list[str] = []
        #: profile name -> identity colour (hex), for row tints and swatches.
        self._profile_colors: dict[str, str] = {}
        #: run numbers whose override draft has uncommitted edits this session.
        self._override_dirty: set[int] = set()
        self._profile_name = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._heading = QLabel("Runs of this instrument")
        layout.addWidget(self._heading)

        self._list = QListWidget()
        # Extended selection still drives Release/Reattach/Assign on multiple
        # runs, but the *current* item (single) is what the form previews and
        # edits.
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
            "applying their profile leaves them unchanged."
        )
        self._release_btn.clicked.connect(self._on_release)
        self._reattach_btn = QPushButton("Reattach to profile")
        self._reattach_btn.setAutoDefault(False)
        self._reattach_btn.setDefault(False)
        self._reattach_btn.setToolTip(
            "Drop the per-run override so the selected runs follow their assigned profile again."
        )
        self._reattach_btn.clicked.connect(self._on_reattach)
        self._assign_btn = QPushButton("Assign to ▸")
        self._assign_btn.setAutoDefault(False)
        self._assign_btn.setDefault(False)
        self._assign_btn.setToolTip(
            "Assign the selected runs to another grouping profile of this "
            "instrument (e.g. a different sample). A released run keeps its "
            "override; the chosen profile becomes its reattach target."
        )
        self._assign_btn.clicked.connect(self._on_assign_menu)
        button_row.addWidget(self._release_btn)
        button_row.addWidget(self._reattach_btn)
        button_row.addWidget(self._assign_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

    def set_runs(
        self,
        runs: list[tuple[int, str, bool, str]],
        *,
        profile_name: str,
        profile_names: list[str] | None = None,
        profile_colors: dict[str, str] | None = None,
    ) -> None:
        """Populate the panel.

        Parameters
        ----------
        runs
            ``(run_number, label, overridden, assigned_profile)`` tuples for
            every run of the current fingerprint. ``overridden`` and
            ``assigned_profile`` are the run's state at open.
        profile_name
            Name of the profile the editor is currently editing (rows assigned
            to it are the emphasised "will receive edits" set).
        profile_names
            Every profile name of the fingerprint, for the Assign-to menu.
            Defaults to just *profile_name*.
        profile_colors
            Each profile's identity colour (hex), for the row tints and the
            Assign-to menu swatches.
        """
        self._profile_name = str(profile_name)
        self._profile_names = [str(n) for n in (profile_names or [profile_name])]
        self._profile_colors = {str(k): str(v) for k, v in (profile_colors or {}).items()}
        self._initial_overridden = {int(rn): bool(ov) for rn, _label, ov, _assigned in runs}
        self._released = dict(self._initial_overridden)
        self._labels = {int(rn): str(label) for rn, label, _ov, _assigned in runs}
        self._initial_assigned = {int(rn): str(assigned) for rn, _l, _ov, assigned in runs}
        self._assigned = dict(self._initial_assigned)
        # A repopulate (instrument switch) drops any override-dirty markers for
        # runs no longer listed.
        self._override_dirty = {rn for rn in self._override_dirty if rn in self._labels}
        self._rebuild()

    def _chip_text(self, run_number: int) -> str:
        """The status chip for a run: its override state or assigned profile."""
        assigned = self._assigned.get(run_number, self._profile_name)
        if self._released.get(run_number, False):
            chip = "override" if assigned == self._profile_name else f"override · {assigned}"
            if run_number in self._override_dirty:
                chip += " *"
            return chip
        return f"follows {assigned}"

    def _rebuild(self) -> None:
        current = self.current_run_number()
        blocked = self._list.blockSignals(True)
        try:
            self._list.clear()
            for run_number in sorted(self._labels):
                item = QListWidgetItem(
                    f"{self._labels[run_number]}  —  {self._chip_text(run_number)}"
                )
                item.setData(Qt.ItemDataRole.UserRole, int(run_number))
                self._tint_item(item, run_number)
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

    def _profile_color(self, name: str) -> QColor:
        """The identity colour for profile *name* (accent when unmapped)."""
        return QColor(self._profile_colors.get(str(name), tokens.ACCENT))

    def _tint_item(self, item: QListWidgetItem, run_number: int) -> None:
        """Style a row by its profile identity and state.

        Every row's text wears its assigned profile's identity colour — the
        same colour the Data Browser and the editing strip use. The runs
        following the *edited* profile additionally get the strip's emphasis
        (bold text on a soft tint of that colour), so the strip and the rows
        it "applies to" read as one thing. Released rows are warning-tinted:
        the diverged state outranks sample identity.
        """
        font = item.font()
        if self._released.get(run_number, False):
            item.setForeground(QColor(tokens.WARN))
            font.setBold(False)
            item.setBackground(QBrush())
        else:
            assigned = self._assigned.get(run_number, self._profile_name)
            color = self._profile_color(assigned)
            item.setForeground(color)
            if assigned == self._profile_name:
                font.setBold(True)
                soft = QColor(color)
                soft.setAlpha(30)
                item.setBackground(soft)
            else:
                font.setBold(False)
                item.setBackground(QBrush())
        item.setFont(font)

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
        self._assign_btn.setEnabled(bool(selected) and len(self._profile_names) > 1)

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

    def _on_assign_menu(self) -> None:
        """Show the Assign-to menu and reassign the selected runs."""
        selected = self._selected_run_numbers()
        if not selected or len(self._profile_names) < 2:
            return
        menu = QMenu(self)
        for name in self._profile_names:
            action = menu.addAction(name)
            action.setData(name)
            swatch = QPixmap(10, 10)
            swatch.fill(self._profile_color(name))
            action.setIcon(QIcon(swatch))
        chosen = menu.exec(self._assign_btn.mapToGlobal(self._assign_btn.rect().bottomLeft()))
        if chosen is None:
            return
        self.assign_runs(selected, str(chosen.data()))

    def assign_runs(self, run_numbers: list[int], profile_name: str) -> None:
        """Assign *run_numbers* to *profile_name*, emitting :attr:`changed`.

        A released run keeps its override; the assignment only retargets the
        profile it reattaches to. No-op assignments emit nothing.
        """
        changed = False
        for rn in run_numbers:
            rn = int(rn)
            if rn in self._labels and self._assigned.get(rn) != str(profile_name):
                self._assigned[rn] = str(profile_name)
                changed = True
        if changed:
            self._rebuild()
            self.changed.emit()

    def rename_profile(self, old_name: str, new_name: str) -> None:
        """Follow a profile rename: rewrite every reference to *old_name*."""
        old_name, new_name = str(old_name), str(new_name)
        if self._profile_name == old_name:
            self._profile_name = new_name
        self._profile_names = [new_name if n == old_name else n for n in self._profile_names]
        if old_name in self._profile_colors:
            self._profile_colors[new_name] = self._profile_colors.pop(old_name)
        for mapping in (self._assigned, self._initial_assigned):
            for rn, name in list(mapping.items()):
                if name == old_name:
                    mapping[rn] = new_name
        self._rebuild()

    def remove_profile(self, name: str, reassign_to: str) -> None:
        """Drop *name* from the profile list, moving its runs to *reassign_to*.

        The initial-assignment record is deliberately left untouched so the
        forced moves surface in :meth:`newly_assigned` (the dialog reconciles
        them into the project on Apply).
        """
        name, reassign_to = str(name), str(reassign_to)
        self._profile_names = [n for n in self._profile_names if n != name]
        self._profile_colors.pop(name, None)
        for rn, assigned in list(self._assigned.items()):
            if assigned == name:
                self._assigned[rn] = reassign_to
        self._rebuild()

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
            item.setText(f"{self._labels[run_number]}  —  {self._chip_text(run_number)}")
            self._tint_item(item, run_number)
            return

    def override_dirty_run_numbers(self) -> set[int]:
        """Run numbers whose override draft has uncommitted edits this session."""
        return {rn for rn in self._override_dirty if self._released.get(rn, False)}

    def released_run_numbers(self) -> set[int]:
        """Run numbers currently marked as released (per-run override)."""
        return {rn for rn, released in self._released.items() if released}

    def inheriting_run_numbers(self) -> set[int]:
        """Run numbers currently following a profile (no override)."""
        return {rn for rn in self._labels if not self._released.get(rn, False)}

    def runs_following(self, profile_name: str) -> set[int]:
        """Non-released runs assigned to *profile_name* (the Apply target set)."""
        return {
            rn
            for rn in self.inheriting_run_numbers()
            if self._assigned.get(rn, self._profile_name) == str(profile_name)
        }

    def assignments(self) -> dict[int, str]:
        """The current run→profile assignment map (released runs included)."""
        return dict(self._assigned)

    def assigned_profile(self, run_number: int) -> str | None:
        """The profile *run_number* is currently assigned to, if listed."""
        return self._assigned.get(int(run_number))

    def newly_assigned(self) -> dict[int, str]:
        """Runs whose assignment changed this session (run → new profile)."""
        return {
            rn: name
            for rn, name in self._assigned.items()
            if name != self._initial_assigned.get(rn)
        }

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
