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
ProjectShell = None
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
    """Return the BENCH stylesheet with design-token colours substituted in.

    ``bench.qss`` is a template whose colours are ``@TOKEN@`` placeholders;
    :func:`~asymmetry.gui.styles.stylesheet.render_bench_stylesheet` loads it
    (on-disk first, then ``importlib.resources`` for frozen builds) and fills the
    tokens. The resources fallback matters in a frozen (PyInstaller) build, where
    ``app.py`` is the entry script and its ``__file__`` is relocated to the
    archive root rather than the ``asymmetry/gui/`` package tree.
    """
    from asymmetry.gui.styles.stylesheet import render_bench_stylesheet

    return render_bench_stylesheet()


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


def _load_app_icon():
    """Load every pre-rendered size, shared with the native installer icons."""
    from PySide6.QtGui import QIcon

    from asymmetry.resources import ICON_SIZES

    platform = "macos" if sys.platform == "darwin" else "windows"
    icon = QIcon()
    for size in ICON_SIZES:
        filename = f"icon_{platform}_{size}.png"
        pixmap = _load_resource_pixmap(filename)
        if pixmap is None:
            raise ValueError(f"packaged application icon cannot be decoded: {filename}")
        icon.addPixmap(pixmap)
    return icon


def _create_splash_screen(app, logo=None):
    """Create and show the startup splash screen.

    Keep this dependency-light: it runs before importing ``MainWindow`` and the
    rest of the plotting/fitting stack.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPainter, QPixmap
        from PySide6.QtWidgets import QSplashScreen

        from asymmetry.gui.styles import tokens
    except (ImportError, ModuleNotFoundError):
        return None

    # Literal px here are intentional: the splash paints before QSettings /
    # QApplication font are read, so there is no UI scale to derive from yet.
    width, height = 420, 300
    dpr = max(2.0, app.devicePixelRatio())
    canvas = QPixmap(round(width * dpr), round(height * dpr))
    canvas.setDevicePixelRatio(dpr)
    canvas.fill(QColor(tokens.BG))

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QColor(tokens.BORDER))
    painter.drawRect(0, 0, width - 1, height - 1)

    if logo is not None:
        scaled_logo = logo.scaled(
            round(156 * dpr),
            round(156 * dpr),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled_logo.setDevicePixelRatio(dpr)
        x = round((width - scaled_logo.deviceIndependentSize().width()) / 2)
        painter.drawPixmap(x, 32, scaled_logo)

    font = painter.font()
    font.setPointSize(24)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(tokens.TEXT))
    painter.drawText(0, 205, width, 34, Qt.AlignmentFlag.AlignHCenter, "Asymmetry")

    font.setPointSize(10)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor(tokens.TEXT_MUTED))
    painter.drawText(
        0,
        246,
        width,
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

    if "--scientific-acceptance" in sys.argv:
        from asymmetry.gui.scientific_acceptance import main as acceptance_main

        sys.exit(acceptance_main([arg for arg in sys.argv[1:] if arg != "--scientific-acceptance"]))

    global QApplication, ProjectShell

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

        # Verify the bundled HDF4 runtime loads in the frozen app (pyhdf + its
        # HDF4 C library DLLs/dylibs). Raises ImportError if the runtime is not
        # bundled/findable, failing the packaging smoke test.
        from asymmetry.core.io.hdf4 import _require_pyhdf

        _require_pyhdf()

    if QApplication is None:
        from PySide6.QtWidgets import QApplication as _QApplication

        QApplication = _QApplication

    if smoke_test:
        _install_smoke_qt_message_filter()

    app = QApplication(sys.argv)

    # Guard the QScreen wrappers' lifetime: PySide's .screen() bindings can
    # let the Python GC delete the live C++ QScreen (the FB↔FFT crash). Pins
    # every screen wrapper, arms a destruction tripwire, and re-anchors
    # windows on real display removal (docs/GUI_GUIDELINES.md).
    from asymmetry.gui.screen_guard import install_screen_change_guard

    install_screen_change_guard(app)

    from asymmetry.gui.styles.palette import apply_bench_style

    apply_bench_style(app)

    startup_pixmap = _load_startup_pixmap("logo.png")
    if startup_pixmap is None:
        startup_pixmap = _load_resource_pixmap("logo.png")
    icon = _load_app_icon()
    app.setWindowIcon(icon)

    splash = _create_splash_screen(app, startup_pixmap)

    app.setApplicationName("Asymmetry")
    app.setOrganizationName("Asymmetry")

    from asymmetry.gui.styles.fonts import configure_plot_fonts, register_bundled_fonts
    from asymmetry.gui.ui_manager import set_base_stylesheet

    register_bundled_fonts()
    configure_plot_fonts()

    # Registered as the base, not set: the first MainWindow composes it with
    # the scale block and applies the sheet once its widget tree is built.
    set_base_stylesheet(app, _load_bench_stylesheet())

    # Discover user functions before the main window is built so plugin
    # components are in the registries when the pickers first render. The
    # loader confines plugin failures to its report (surfaced in the log
    # panel and Setup → User Functions…); the blanket guard here only covers
    # discovery infrastructure itself — startup must never fail on user code.
    try:
        from asymmetry.core.plugins import load_user_functions

        load_user_functions()
    except Exception:
        pass

    if ProjectShell is None:
        from asymmetry.gui.shell import ProjectShell as _ProjectShell

        ProjectShell = _ProjectShell

    from PySide6.QtCore import Qt, QTimer

    shell = ProjectShell()
    # Destroy the owned widget tree while Qt can still service layout callbacks.
    shell.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    shell.add_project()
    shell.show()
    if splash is not None:
        splash.finish(shell)

    if smoke_test:
        QTimer.singleShot(0, shell.close)

    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()
