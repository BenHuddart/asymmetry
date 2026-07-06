"""GLE executable location settings and auto-detection.

In a frozen macOS app the PATH is minimal and shutil.which("gle") fails even
when GLE is installed via Homebrew.  This module searches a wider set of
candidate paths and persists the user-chosen path across sessions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles.metrics import dialog_width

_SETTINGS_KEY = "gle/executable_path"

# Common GLE install locations on macOS (Homebrew Intel/ARM, MacPorts, system)
_CANDIDATE_PATHS: list[str] = [
    "/opt/homebrew/bin/gle",
    "/usr/local/bin/gle",
    "/opt/local/bin/gle",
    "/usr/bin/gle",
    "/Applications/QGLE.app/Contents/Resources/gle/bin/gle",
]


def find_gle_executable() -> str | None:
    """Return the GLE executable path, or None if not found.

    Checks PATH first (works when running from source), then falls back to
    candidate paths so a frozen macOS app can still locate GLE.
    """
    found = shutil.which("gle")
    if found:
        return found
    for candidate in _CANDIDATE_PATHS:
        if Path(candidate).is_file():
            return candidate
    return None


def get_gle_executable() -> str | None:
    """Return the user-configured or auto-detected GLE executable path."""
    settings = QSettings()
    saved = settings.value(_SETTINGS_KEY, "", str)
    if saved and Path(saved).is_file():
        return saved
    return find_gle_executable()


def save_gle_executable(path: str) -> None:
    QSettings().setValue(_SETTINGS_KEY, path)


def validate_gle_executable(path: str) -> str | None:
    """Return a problem description for *path*, or None when it runs.

    Actually executes ``<path> -v`` (GLE prints its version banner) so a
    saved path that exists but cannot run — lost execute bit, a directory,
    a non-GLE binary — is caught here in the setup dialog instead of
    surfacing later as an opaque export failure.
    """
    p = Path(path)
    if not p.is_file():
        return "No file at this path."
    if not os.access(p, os.X_OK):
        return "The file is not executable."
    try:
        result = subprocess.run([str(p), "-v"], capture_output=True, text=True, timeout=10)
    except OSError as exc:
        return f"The file could not be run: {exc}"
    except subprocess.TimeoutExpired:
        return "The executable did not respond within 10 seconds."
    banner = (result.stdout or "") + (result.stderr or "")
    if "GLE" not in banner:
        return "The executable does not look like GLE (no GLE version banner)."
    return None


class GleSetupDialog(QDialog):
    """Dialog for configuring the GLE executable path."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GLE Setup")
        self.setMinimumWidth(dialog_width(69))  # ~500px at default scale
        self._build_ui()
        self._refresh_status()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._status_label = QLabel()
        layout.addWidget(self._status_label)

        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to GLE executable…")
        path_layout.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        auto_btn = QPushButton("Auto-detect")
        auto_btn.clicked.connect(self._on_auto_detect)
        layout.addWidget(auto_btn)

        hint = QLabel(
            "GLE is used to compile exported .gle scripts to PDF or EPS.\n"
            "Download from: glx.sourceforge.net"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_status(self) -> None:
        settings = QSettings()
        saved = settings.value(_SETTINGS_KEY, "", str)
        self._path_edit.setText(saved)

        found = get_gle_executable()
        if found:
            self._status_label.setText(f"GLE found: {found}")
        else:
            self._status_label.setText(
                "GLE not found on PATH or common install locations. "
                "Specify the path below or click Auto-detect."
            )

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select GLE Executable")
        if path:
            self._path_edit.setText(path)
            if Path(path).is_file():
                self._status_label.setText(f"GLE found: {path}")

    def _on_auto_detect(self) -> None:
        found = find_gle_executable()
        if found:
            self._path_edit.setText(found)
            self._status_label.setText(f"GLE found: {found}")
        else:
            self._status_label.setText(
                "Could not auto-detect GLE. Install GLE or browse to its location."
            )

    def _on_accept(self) -> None:
        path = self._path_edit.text().strip()
        if path:
            problem = validate_gle_executable(path)
            if problem is not None:
                # Keep the dialog open: a path that cannot run would only
                # resurface later as an opaque export failure.
                self._status_label.setText(f"Cannot use this GLE path: {problem}")
                return
        save_gle_executable(path)
        self.accept()
