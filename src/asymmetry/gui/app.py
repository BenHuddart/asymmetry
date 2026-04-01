"""Application entry point for the Asymmetry GUI.

Launch with::

    python -m asymmetry.gui.app
    # or via the installed console script:
    asymmetry-gui
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication

from asymmetry.gui.mainwindow import MainWindow


def _load_app_icon() -> QIcon | None:
    """Load application icon from package resources.

    Returns None if icon cannot be loaded.
    """
    # Try importlib.resources (preferred for installed packages)
    try:
        from importlib.resources import files

        logo = files("asymmetry.resources").joinpath("logo_256x256.png")
        if logo.is_file():
            pixmap = QPixmap()
            if pixmap.loadFromData(logo.read_bytes(), "PNG"):
                icon = QIcon(pixmap)
                if not icon.isNull():
                    return icon
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError, OSError):
        pass

    # Fallback: try direct path (for development)
    try:
        resources_dir = Path(__file__).parent.parent / "resources"
        icon_path = resources_dir / "logo_256x256.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon
    except (OSError, ValueError):
        pass

    return None


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Asymmetry")
    app.setOrganizationName("Asymmetry")

    # Set application icon
    icon = _load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
