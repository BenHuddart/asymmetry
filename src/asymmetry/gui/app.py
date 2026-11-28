"""Application entry point for the Asymmetry GUI.

Launch with::

    python -m asymmetry.gui.app
    # or via the installed console script:
    asymmetry-gui
"""

from __future__ import annotations

import multiprocessing as mp
import sys

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
    from PySide6.QtCore import QEvent, Qt, QTimer
    from PySide6.QtGui import QColor, QPainter, QPixmap
    from PySide6.QtWidgets import QWidget

    from asymmetry.gui.styles import tokens

    class SplashWindow(QWidget):
        def __init__(self, pixmap):
            super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
            self.pixmap = pixmap
            self.resize(pixmap.deviceIndependentSize().toSize())

        def paintEvent(self, _event):  # noqa: N802 — Qt override
            with QPainter(self) as painter:
                painter.drawPixmap(0, 0, self.pixmap)

        def mousePressEvent(self, _event):  # noqa: N802 — Qt override
            self.hide()

        def finish_when_painted(self, window):
            window.installEventFilter(self)

        def eventFilter(self, window, event):  # noqa: N802 — Qt override
            if event.type() in (QEvent.Type.Paint, QEvent.Type.Close):
                window.removeEventFilter(self)
                QTimer.singleShot(0, self, self.close)
            return False

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

    # A plain widget publishes its first frame on Cocoa without waiting for
    # the native window to be mapped inside Qt's splash Show handler.
    splash = SplashWindow(canvas)
    screen = app.primaryScreen()
    if screen is not None:
        splash.move(screen.geometry().center() - splash.rect().center())
    splash.show()
    app.processEvents()
    return splash


def _prepare_startup(worker):
    """Load definitions and prepare plot fonts before building the project window."""
    from asymmetry.gui.styles.fonts import configure_plot_fonts
    from asymmetry.gui.tasks import check_task_cancelled

    configure_plot_fonts()
    check_task_cancelled(worker)
    from asymmetry.gui.shell import ProjectShell

    check_task_cancelled(worker)
    return ProjectShell, _load_bench_stylesheet()


def _start_project_window(app, splash, *, smoke_test: bool) -> int:
    """Keep the event loop running while startup dependencies load."""
    from PySide6.QtCore import Qt, QTimer

    from asymmetry.gui.tasks import TaskCancelledError, TaskRunner
    from asymmetry.gui.ui_manager import set_base_stylesheet

    startup = TaskRunner(app)
    shell = None
    failure = None

    def fail(error: BaseException) -> None:
        nonlocal failure
        failure = error
        app.exit(1)

    def prepare(worker):
        try:
            return _prepare_startup(worker)
        except TaskCancelledError:
            raise
        except BaseException as error:
            return error

    def show_project(prepared) -> None:
        nonlocal shell
        if isinstance(prepared, BaseException):
            fail(prepared)
            return
        # Exceptions crossing a Qt slot boundary do not unwind app.exec().
        try:
            # Plugins register before the first model picker is constructed.
            try:
                from asymmetry.core.plugins import load_user_functions

                load_user_functions()
            except Exception:
                pass
            shell_type, stylesheet = prepared
            set_base_stylesheet(app, stylesheet)
            shell = shell_type()
            shell.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            shell.add_project()
            splash.finish_when_painted(shell)
            shell.show()
            if smoke_test:
                QTimer.singleShot(0, shell, shell.close)
        except BaseException as error:
            fail(error)

    QTimer.singleShot(0, app, lambda: startup.start(prepare, on_finished=show_project))
    try:
        exit_code = app.exec()
    finally:
        startup.shutdown()
    if failure is not None:
        raise failure
    return exit_code


def main() -> None:
    mp.freeze_support()

    if "--scientific-acceptance" in sys.argv:
        from asymmetry.gui.scientific_acceptance import main as acceptance_main

        sys.exit(acceptance_main([arg for arg in sys.argv[1:] if arg != "--scientific-acceptance"]))

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

    from PySide6.QtWidgets import QApplication

    from asymmetry.gui.crash_log import CrashLog

    if smoke_test:
        _install_smoke_qt_message_filter()

    app = QApplication(sys.argv)
    app.setApplicationName("Asymmetry")
    app.setOrganizationName("Asymmetry")
    CrashLog(app)

    # Guard the QScreen wrappers' lifetime: PySide's .screen() bindings can
    # let the Python GC delete the live C++ QScreen (the FB↔FFT crash). Pins
    # every screen wrapper, arms a destruction tripwire, and re-anchors
    # windows on real display removal (docs/GUI_GUIDELINES.md).
    from asymmetry.gui.screen_guard import install_screen_change_guard

    install_screen_change_guard(app)

    splash = _create_splash_screen(app, _load_resource_pixmap("logo.png"))

    from asymmetry.gui.styles.palette import apply_bench_style

    apply_bench_style(app)
    app.setWindowIcon(_load_app_icon())

    from asymmetry.gui.styles.fonts import register_bundled_fonts

    register_bundled_fonts()
    sys.exit(_start_project_window(app, splash, smoke_test=smoke_test))


if __name__ == "__main__":
    mp.freeze_support()
    main()
