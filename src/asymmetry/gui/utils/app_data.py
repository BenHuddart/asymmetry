"""Resolve the shared application-data location without creating it."""

from pathlib import Path

from PySide6.QtCore import QStandardPaths


def app_data_dir() -> Path:
    """Return the platform application-data directory; writes create their parents."""
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
