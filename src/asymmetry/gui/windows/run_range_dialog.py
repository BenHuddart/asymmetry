"""Collect a folder + run-number range for the "Load run range…" workflow.

The native Open dialog's File-name field truncates a long quoted list, so a
contiguous run series (e.g. BiSCCO 1276–1289) had to be loaded in batches. This
dialog gathers a folder, an optional prefix, and an inclusive first/last run;
the main window then calls :func:`asymmetry.core.io.resolve_run_range` to expand
that into the existing files and loads them through the usual batch path.

The dialog is presentation only — it never opens data files. When the user
picks a folder it prefills the prefix and first/last from the run files found
there (via :func:`asymmetry.core.io.scan_run_files`), so the common case is a
single click plus OK.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from asymmetry.core.io import scan_run_files
from asymmetry.gui.styles import tokens

_MAX_RUN = 99_999_999


class RunRangeDialog(QDialog):
    """Collect ``(folder, first, last, prefix)`` for a run-range load."""

    def __init__(self, parent=None, *, initial_dir: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Run Range")

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Load a contiguous run series by folder and run number, bypassing "
                "the Open dialog's file-name length limit. Missing runs in the range "
                "are skipped."
            )
        )

        form = QFormLayout()

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit(self)
        self._folder_edit.setPlaceholderText("Folder containing the run files")
        if initial_dir:
            self._folder_edit.setText(initial_dir)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._on_browse)
        folder_row.addWidget(self._folder_edit, 1)
        folder_row.addWidget(browse)
        form.addRow("Folder:", folder_row)

        self._prefix_edit = QLineEdit(self)
        self._prefix_edit.setPlaceholderText("e.g. MUSR (blank = auto-detect)")
        form.addRow("Prefix:", self._prefix_edit)

        self._first_spin = QSpinBox(self)
        self._first_spin.setRange(0, _MAX_RUN)
        form.addRow("First run:", self._first_spin)

        self._last_spin = QSpinBox(self)
        self._last_spin.setRange(0, _MAX_RUN)
        form.addRow("Last run:", self._last_spin)

        layout.addLayout(form)

        # Shown only when the folder scan hit the entry cap (see
        # scan_run_files.DEFAULT_MAX_SCAN_ENTRIES) — the prefill above then
        # reflects only the inspected prefix of the folder, not the whole
        # thing, so the range may need manual adjustment.
        self._scan_truncated_label = QLabel()
        self._scan_truncated_label.setStyleSheet(f"color: {tokens.WARN};")
        self._scan_truncated_label.setWordWrap(True)
        self._scan_truncated_label.hide()
        layout.addWidget(self._scan_truncated_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        if initial_dir:
            self._prefill_from_folder(initial_dir)

    # ── interaction ──────────────────────────────────────────────────────

    def _on_browse(self) -> None:
        start = self._folder_edit.text().strip() or ""
        folder = QFileDialog.getExistingDirectory(self, "Select run folder", start)
        if folder:
            self._folder_edit.setText(folder)
            self._prefill_from_folder(folder)

    def _prefill_from_folder(self, folder: str) -> None:
        """Prefill prefix and first/last from the run files in ``folder``."""
        try:
            result = scan_run_files(folder)
        except ValueError:
            return
        if result.truncated:
            self._scan_truncated_label.setText(
                f"This folder has too many files to scan in full — showing the first "
                f"{len(result.entries)} run files found. Adjust the range by hand if "
                "runs are missing."
            )
            self._scan_truncated_label.show()
        else:
            self._scan_truncated_label.hide()
        found = result.entries
        if not found:
            return
        runs = [run for _, run, _ in found]
        self._first_spin.setValue(min(runs))
        self._last_spin.setValue(max(runs))
        if not self._prefix_edit.text().strip():
            prefixes = [prefix for prefix, _, _ in found if prefix]
            if prefixes:
                most_common = Counter(prefixes).most_common(1)[0][0]
                self._prefix_edit.setText(most_common)

    def _on_accept(self) -> None:
        if not self.folder():
            self._warn("Choose a folder containing the run files.")
            return
        if not Path(self.folder()).is_dir():
            self._warn("The chosen folder does not exist.")
            return
        if self.first_run() > self.last_run():
            self._warn("The first run must not be after the last run.")
            return
        self.accept()

    def _warn(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(self, "Load Run Range", message)

    # ── results ──────────────────────────────────────────────────────────

    def folder(self) -> str:
        return self._folder_edit.text().strip()

    def prefix(self) -> str | None:
        text = self._prefix_edit.text().strip()
        return text or None

    def first_run(self) -> int:
        return int(self._first_spin.value())

    def last_run(self) -> int:
        return int(self._last_spin.value())
