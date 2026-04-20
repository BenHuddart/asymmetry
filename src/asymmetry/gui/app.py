"""Application entry point for the Asymmetry GUI.

Launch with::

    python -m asymmetry.gui.app
    # or via the installed console script:
    asymmetry-gui
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path


QApplication = None
MainWindow = None


def _load_app_icon():
    """Load application icon from package resources.

    Returns None if icon cannot be loaded.
    """
    from PySide6.QtGui import QIcon, QPixmap

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
    mp.freeze_support()

    global QApplication, MainWindow

    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        # Force a headless backend so this check can run on CI runners.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.argv = [arg for arg in sys.argv if arg != "--smoke-test"]

    if QApplication is None:
        from PySide6.QtWidgets import QApplication as _QApplication

        QApplication = _QApplication

    if MainWindow is None:
        from asymmetry.gui.mainwindow import MainWindow as _MainWindow

        MainWindow = _MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Asymmetry")
    app.setOrganizationName("Asymmetry")

    # Set application icon
    icon = _load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    window = MainWindow()
    window.show()

    if smoke_test:
        app.processEvents()
        window.close()
        return

    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()
