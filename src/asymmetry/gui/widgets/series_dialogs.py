"""The two prompts a recorded series' or saved fit's name and deletion go through.

The Parameters panel's chip menu, the Batch tab's series row and the Single
tab's fit row all offer ``Rename…`` and ``Delete…``, so the wording lives here
once: the delete confirmation spells out what survives it (series D6), and the
rename dialog opens on the name the user is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget


@dataclass(frozen=True)
class RecordKind:
    """What is being renamed or deleted: its noun and what a delete leaves behind."""

    noun: str
    delete_keeps: str


SERIES = RecordKind(
    "series",
    "Removes this series and its trend. Other series and single fits on these runs are kept.",
)
SAVED_FIT = RecordKind(
    "fit",
    "Removes this saved fit from the run. The run's other fits and every series are kept.",
)


def prompt_rename(parent: QWidget, kind: RecordKind, current_name: str) -> str | None:
    """Ask for a new name for *current_name*, or ``None`` when cancelled.

    The returned name is stripped; an empty string means "drop the user label"
    and is the caller's to interpret.
    """
    new_name, ok = QInputDialog.getText(
        parent,
        f"Rename {kind.noun}",
        f"{kind.noun.capitalize()} name:",
        text=current_name,
    )
    return new_name.strip() if ok else None


def confirm_delete(parent: QWidget, kind: RecordKind, name: str) -> bool:
    """Confirm deleting *name*, naming what a delete keeps."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(f"Delete {kind.noun}")
    box.setText(f'Delete {kind.noun} "{name}"?')
    box.setInformativeText(kind.delete_keeps)
    box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return box.exec() == QMessageBox.StandardButton.Ok
