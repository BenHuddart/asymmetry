"""Application entry point for the Asymmetry GUI.

Launch with::

    python -m asymmetry.gui.app
    # or via the installed console script:
    asymmetry-gui
"""

from __future__ import annotations

import multiprocessing as mp
import sys

QApplication = None
MainWindow = None


def _resource_file_path(filename: str) -> str:
    """Return a direct filesystem path for an installed package resource."""
    package_root = __file__.replace("\\", "/").rsplit("/", 2)[0]
    return f"{package_root}/resources/{filename}"


def _load_startup_pixmap(filename: str):
    """Load a startup image without importing resource helpers."""
    from PySide6.QtGui import QPixmap

    pixmap = QPixmap(_resource_file_path(filename))
    if not pixmap.isNull():
        return pixmap
    return None


def _load_resource_pixmap(filename: str):
    """Load a packaged resource image into a ``QPixmap``.

    This helper intentionally imports only QtGui and importlib.resources so it
    can be used before the heavier main-window module is imported.
    """
    from PySide6.QtGui import QPixmap

    pixmap = _load_startup_pixmap(filename)
    if pixmap is not None:
        return pixmap

    try:
        from importlib.resources import files

        image = files("asymmetry.resources").joinpath(filename)
        if image.is_file():
            pixmap = QPixmap()
            if pixmap.loadFromData(image.read_bytes(), "PNG") and not pixmap.isNull():
                return pixmap
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError, OSError):
        pass

    return None


def _icon_from_pixmap(pixmap):
    """Build an application icon from an existing pixmap."""
    if pixmap is None:
        return None
    from PySide6.QtGui import QIcon

    icon = QIcon(pixmap)
    if not icon.isNull():
        return icon
    return None


def _load_app_icon(pixmap=None):
    """Load application icon from package resources.

    Returns None if icon cannot be loaded.
    """
    icon = _icon_from_pixmap(pixmap)
    if icon is not None:
        return icon

    pixmap = _load_resource_pixmap("logo_256x256.png")
    return _icon_from_pixmap(pixmap)


def _create_splash_screen(app, logo=None):
    """Create and show the startup splash screen.

    Keep this dependency-light: it runs before importing ``MainWindow`` and the
    rest of the plotting/fitting stack.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPainter, QPixmap
        from PySide6.QtWidgets import QSplashScreen
    except (ImportError, ModuleNotFoundError):
        return None

    canvas = QPixmap(420, 300)
    canvas.fill(QColor("#f8fafc"))

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QColor("#cbd5e1"))
    painter.drawRect(0, 0, canvas.width() - 1, canvas.height() - 1)

    if logo is not None:
        scaled_logo = logo.scaled(
            156,
            156,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (canvas.width() - scaled_logo.width()) // 2
        painter.drawPixmap(x, 32, scaled_logo)

    font = painter.font()
    font.setPointSize(24)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(0, 205, canvas.width(), 34, Qt.AlignmentFlag.AlignHCenter, "Asymmetry")

    font.setPointSize(10)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor("#475569"))
    painter.drawText(
        0,
        246,
        canvas.width(),
        24,
        Qt.AlignmentFlag.AlignHCenter,
        "Loading analysis tools...",
    )
    painter.end()

    splash = QSplashScreen(canvas)
    splash.show()
    app.processEvents()
    return splash


def main() -> None:
    mp.freeze_support()

    global QApplication, MainWindow

    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        import os

        # Force a headless backend so this check can run on CI runners.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.argv = [arg for arg in sys.argv if arg != "--smoke-test"]

    if QApplication is None:
        from PySide6.QtWidgets import QApplication as _QApplication

        QApplication = _QApplication

    app = QApplication(sys.argv)

    startup_pixmap = _load_startup_pixmap("logo_256x256.png")
    icon = _load_app_icon(startup_pixmap)
    if icon is not None:
        app.setWindowIcon(icon)

    splash = _create_splash_screen(app, startup_pixmap)

    app.setApplicationName("Asymmetry")
    app.setOrganizationName("Asymmetry")

    if icon is None:
        icon = _load_app_icon()
        if icon is not None:
            app.setWindowIcon(icon)

    if MainWindow is None:
        from asymmetry.gui.mainwindow import MainWindow as _MainWindow

        MainWindow = _MainWindow

    window = MainWindow()
    window.show()
    if splash is not None:
        splash.finish(window)

    if smoke_test:
        app.processEvents()
        window.close()
        return

    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()
