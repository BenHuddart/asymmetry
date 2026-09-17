"""The two prompts a recorded series' name and deletion go through.

Both the Parameters panel's chip menu and the Batch tab's series row offer
``Rename…`` and ``Delete…``, so the wording lives here once: the delete
confirmation spells out what survives it (D6), and the rename dialog opens on
the name the user is looking at.
"""

from __future__ import annotations

from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget


def prompt_series_rename(parent: QWidget, current_name: str) -> str | None:
    """Ask for a new name for *current_name*, or ``None`` when cancelled.

    The returned name is stripped; an empty string means "drop the user label"
    and is the caller's to interpret.
    """
    new_name, ok = QInputDialog.getText(
        parent,
        "Rename series",
        "Series name:",
        text=current_name,
    )
    return new_name.strip() if ok else None


def confirm_series_delete(parent: QWidget, series_name: str) -> bool:
    """Confirm deleting *series_name*, naming what a delete keeps (D6)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Delete series")
    box.setText(f'Delete series "{series_name}"?')
    box.setInformativeText(
        "Removes this series and its trend. Other series and single fits on these runs are kept."
    )
    box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return box.exec() == QMessageBox.StandardButton.Ok
