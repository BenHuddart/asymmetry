"""Guards against the FB↔FFT ``QScreen`` crash (PySide wrapper lifetime bug).

**Root cause (captured live under guard malloc + lldb, 2026-07-09):** the crash
long attributed to a Qt/macOS-26 defect is a **PySide/shiboken object-lifetime
bug in the application process**. ``QScreen::~QScreen`` was caught firing
mid-session from ``SbkDeallocWrapperCommon`` inside ``gc_collect_main`` — the
Python cyclic GC reclaimed a shiboken wrapper and shiboken deleted the **live
C++ ``QScreen``** with it. Nothing removes it from
``QGuiApplication::screens()`` (``~QScreen``'s "must go through
handleScreenRemoved" check is a debug-only assert), so the very next DPI
resolution (``QWidget::metric`` → ``QWidget::screen()`` →
``QGuiApplication::screenAt()`` → ``QScreen::virtualSiblings()``) walks the
freed object: the byte-identical address faulted in the subsequent UAF. On
Windows the same deletion empties the screen list instead (``Cannot create
window: no screens available`` + fatal exit). Full evidence trail:
``docs/investigations/tahoe-qscreen-uaf.md``.

The wrapper relationship that exposes the screen is created by the
``QWidget.screen()`` / ``QWindow.screen()`` bindings: shiboken's return-value
heuristic attaches a **parent link** from the process-wide cached ``QScreen``
wrapper to the *receiver's* wrapper (verified with ``shiboken6.dump``; same
mechanism as PYSIDE-3380). When the receiver is a transient dialog/window, the
screen wrapper's fate becomes tied to garbage collection of that dialog.

Defenses provided here:

- :func:`screen_for` — resolve a widget's screen via
  ``QGuiApplication.screenAt()`` / ``primaryScreen()``, which create **no**
  parent link (verified). App code must use this instead of calling
  ``.screen()`` on widgets/windows.
- :func:`pin_screens` — hold module-level strong references to every screen
  wrapper so the cyclic GC can never deallocate one (the binding map hands out
  one wrapper per C++ ``QScreen``, so pinning covers wrappers acquired
  anywhere). Refreshed on screen add/remove.
- :func:`reanchor_stale_windows` — the pre-existing partial guard for genuine
  display **removal** (external monitor unplugged / virtual display torn
  down): re-point windows whose associated screen is gone at the primary.
"""

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import QPoint
from PySide6.QtGui import QGuiApplication, QScreen
from PySide6.QtWidgets import QWidget

#: Strong references to the current screen wrappers; see :func:`pin_screens`.
_PINNED_SCREENS: list[QScreen] = []

#: ``id()`` of wrappers already wired to the destruction tripwire.
_TRIPWIRED: set[int] = set()


def _on_pinned_screen_destroyed(*_args: object) -> None:
    """Tripwire: a live ``QScreen`` was destroyed mid-session.

    With the wrappers pinned this should never fire before shutdown. If it
    does, the wrapper-lifetime bug (module docstring) has another path — log
    loudly with the Python stack so the deleting context is captured without
    needing a debugger.
    """
    print(
        "screen_guard: a QScreen was destroyed while the application is "
        "running — QScreen wrapper-lifetime bug triggered. Python stack:",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_stack(file=sys.stderr)


def screen_for(widget: QWidget) -> QScreen | None:
    """Resolve ``widget``'s screen without ``QWidget.screen()``.

    ``QWidget.screen()`` / ``QWindow.screen()`` parent the shared ``QScreen``
    wrapper to the receiver's wrapper (shiboken return-value heuristic), tying
    the screen's lifetime to whatever transient widget asked — the root of the
    FB↔FFT crash (module docstring). ``screenAt`` / ``primaryScreen`` create no
    such link, and mirror Qt's own fallback for a not-yet-shown widget
    (``screenAt(topLevel geometry centre)``, then the primary screen).
    """
    app = QGuiApplication.instance()
    if app is None:
        return None
    top = widget.window()
    if top is not None:
        center = top.frameGeometry().center()
        screen = QGuiApplication.screenAt(QPoint(center.x(), center.y()))
        if screen is not None:
            return screen
    return QGuiApplication.primaryScreen()


def pin_screens() -> None:
    """Refresh module-level strong references to all current screen wrappers.

    Shiboken's binding map returns one Python wrapper per C++ ``QScreen``, so
    holding these keeps every screen wrapper reachable — the cyclic GC can
    then never deallocate one and take the C++ ``QScreen`` down with it. Cheap
    (a single-display machine pins one object); refreshed on add/remove so a
    genuinely removed screen's wrapper is released again.
    """
    app = QGuiApplication.instance()
    if app is None:
        return
    _PINNED_SCREENS[:] = app.screens()
    for screen in _PINNED_SCREENS:
        if id(screen) not in _TRIPWIRED:
            _TRIPWIRED.add(id(screen))
            screen.destroyed.connect(_on_pinned_screen_destroyed)


def reanchor_stale_windows() -> None:
    """Point any top-level window whose screen is missing/gone at the primary.

    A window already on a live screen is left untouched, so multi-monitor
    placement is preserved; only a window whose associated ``QScreen`` is ``None``
    or no longer in :meth:`QGuiApplication.screens` (the display-removal case) is
    moved. Wrapped defensively — this runs from a display-change signal and must
    never itself bring down the application.

    ``window.screen()`` here does create the parent link described in the
    module docstring, tying the screen wrapper to a transient ``QWindow``
    wrapper — acceptable on this rare display-removal path because
    :func:`pin_screens` runs immediately afterwards and re-pins fresh wrappers.
    """
    app = QGuiApplication.instance()
    if app is None:
        return
    primary = app.primaryScreen()
    if primary is None:
        return
    live = set(app.screens())
    for window in app.topLevelWindows():
        try:
            current = window.screen()
            if current is None or current not in live:
                window.setScreen(primary)
        except (RuntimeError, ReferenceError):
            # A window torn down concurrently with the reconfiguration; skip it.
            continue


def install_screen_change_guard(app: QGuiApplication) -> None:
    """Install the screen-lifetime guards; call once after app creation.

    Pins the current screen wrappers immediately (see :func:`pin_screens`),
    and on display add/remove re-anchors windows off any dead screen and
    re-pins the new wrapper set.
    """

    def _on_screen_change(_screen: QScreen | None = None) -> None:
        reanchor_stale_windows()
        pin_screens()

    pin_screens()
    app.screenAdded.connect(_on_screen_change)
    app.screenRemoved.connect(_on_screen_change)
    app.primaryScreenChanged.connect(_on_screen_change)
