"""Shared helpers for remembering export locations across GUI dialogs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings


_EXPORT_DIR_KEY = "io/last_export_dir"
_OPEN_DIR_KEY = "io/last_open_dir"


def default_export_path(default_name: str) -> str:
    """Return a save-path suggestion rooted at the remembered export directory."""
    settings = QSettings()
    base_dir = settings.value(_EXPORT_DIR_KEY, "", str)
    if not base_dir:
        base_dir = settings.value(_OPEN_DIR_KEY, "", str)

    if base_dir:
        return str(Path(base_dir) / default_name)
    return str(Path.home() / default_name)


def remember_export_path(selected_path: str) -> None:
    """Persist parent directory for future export dialogs."""
    try:
        parent = str(Path(selected_path).expanduser().resolve().parent)
    except Exception:
        parent = str(Path(selected_path).expanduser().parent)

    if parent:
        settings = QSettings()
        settings.setValue(_EXPORT_DIR_KEY, parent)
