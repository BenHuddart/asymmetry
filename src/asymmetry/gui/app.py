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
MACOS_ICON_TILE_SCALE = 0.82
_SMOKE_QT_PREVIOUS_HANDLER = None
_SMOKE_QT_MESSAGE_HANDLER = None


def _install_smoke_qt_message_filter() -> None:
    global _SMOKE_QT_PREVIOUS_HANDLER, _SMOKE_QT_MESSAGE_HANDLER

    if _SMOKE_QT_MESSAGE_HANDLER is not None:
        return

    from PySide6.QtCore import qInstallMessageHandler

    def _handler(msg_type, context, message):
        text = str(message)
        if "Populating font family aliases took" in text and '"Sans Serif"' in text:
            return
        if "This plugin does not support propagateSizeHints()" in text:
            return
        if _SMOKE_QT_PREVIOUS_HANDLER is not None:
            _SMOKE_QT_PREVIOUS_HANDLER(msg_type, context, message)

    _SMOKE_QT_MESSAGE_HANDLER = _handler
    _SMOKE_QT_PREVIOUS_HANDLER = qInstallMessageHandler(_handler)


def _resource_file_path(filename: str) -> str:
    """Return a direct filesystem path for an installed package resource."""
    package_root = __file__.replace("\\", "/").rsplit("/", 2)[0]
    return f"{package_root}/resources/{filename}"


def _load_bench_stylesheet() -> str:
    """Read bench.qss from gui/styles/ and return its contents."""
    from pathlib import Path

    qss_path = Path(__file__).parent / "styles" / "bench.qss"
    try:
        return qss_path.read_text(encoding="utf-8")
    except OSError:
        return ""


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


def _macos_icon_pixmap(pixmap):
    """Return a rounded-square macOS icon pixmap from a source logo pixmap."""
    if pixmap is None or sys.platform != "darwin":
        return pixmap

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPainterPath, QPixmap

    size = max(pixmap.width(), pixmap.height())
    if size <= 0:
        return pixmap
    tile_size = max(1, round(size * MACOS_ICON_TILE_SCALE))
    tile_offset = (size - tile_size) // 2

    icon = QPixmap(size, size)
    icon.fill(Qt.GlobalColor.transparent)

    painter = QPainter(icon)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    radius = round(tile_size * 0.223)
    path = QPainterPath()
    path.addRoundedRect(tile_offset, tile_offset, tile_size, tile_size, radius, radius)
    painter.setClipPath(path)
    painter.fillPath(path, QColor("#ffffff"))
    painter.drawPixmap(
        tile_offset,
        tile_offset,
        pixmap.scaled(
            tile_size,
            tile_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ),
    )
    painter.end()

    return icon


def _load_app_icon(pixmap=None):
    """Load application icon from package resources.

    Returns None if icon cannot be loaded.
    """
    icon = _icon_from_pixmap(_macos_icon_pixmap(pixmap))
    if icon is not None:
        return icon

    pixmap = _load_resource_pixmap("logo_256x256.png")
    return _icon_from_pixmap(_macos_icon_pixmap(pixmap))


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
    root_smoke_test = "--smoke-test-root" in sys.argv
    if smoke_test:
        import os

        # Force a headless backend so this check can run on CI runners.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.argv = [arg for arg in sys.argv if arg != "--smoke-test"]
    if root_smoke_test:
        sys.argv = [arg for arg in sys.argv if arg != "--smoke-test-root"]
        import awkward  # noqa: F401
        import awkward_cpp  # noqa: F401
        import uproot  # noqa: F401

        from asymmetry.core.io.root import RootLoader

        RootLoader

    if QApplication is None:
        from PySide6.QtWidgets import QApplication as _QApplication

        QApplication = _QApplication

    if smoke_test:
        _install_smoke_qt_message_filter()

    app = QApplication(sys.argv)

    startup_pixmap = _load_startup_pixmap("logo_256x256.png")
    if startup_pixmap is None:
        startup_pixmap = _load_resource_pixmap("logo_256x256.png")
    icon = _load_app_icon(startup_pixmap)
    if icon is not None:
        app.setWindowIcon(icon)

    splash = _create_splash_screen(app, startup_pixmap)

    app.setApplicationName("Asymmetry")
    app.setOrganizationName("Asymmetry")

    from asymmetry.gui.styles.fonts import configure_plot_fonts, register_bundled_fonts

    register_bundled_fonts()
    configure_plot_fonts()

    bench_css = _load_bench_stylesheet()
    if bench_css and hasattr(app, "setStyleSheet"):
        app.setStyleSheet(bench_css)

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
