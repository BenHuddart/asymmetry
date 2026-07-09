"""Open top-level windows at a comfortable, screen-aware default size.

Several auxiliary windows and dialogs hard-coded generous default sizes
(``resize(1280, 920)`` etc.) that exceed a 13-inch laptop's ~800 px-high work
area, so they opened with the title bar clipped above the menu bar. This helper
caps the *initial* size to a fraction of the available geometry (excluding the
taskbar/dock) while preferring the spacious size on larger displays, mirroring
the screen-aware sizing the main window already uses.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from asymmetry.gui.screen_guard import screen_for


def resize_to_available(
    window: QWidget,
    preferred_width: int,
    preferred_height: int,
    *,
    width_fraction: float = 0.92,
    height_fraction: float = 0.9,
    min_width: int = 640,
    min_height: int = 480,
    center: bool = False,
) -> None:
    """Resize ``window`` to ``preferred`` size, capped to the available screen.

    The width/height are each capped at the given fraction of the screen's
    available geometry (work area), so the window never opens taller or wider
    than fits on the current display, then floored at ``min_*`` so it stays
    usable on a tiny screen. Positioning is left to Qt by default (a parented
    dialog keeps its parent-relative placement); pass ``center=True`` to centre
    the window in the available geometry instead. Falls back to the preferred
    size verbatim when no screen can be resolved (e.g. some headless contexts).
    """
    # Never QWidget.screen(): callers here are transient dialogs, and PySide's
    # .screen() binding ties the shared QScreen wrapper's lifetime to the
    # receiver's wrapper (see gui/screen_guard.py — the FB↔FFT crash).
    screen = screen_for(window)
    if screen is None:
        window.resize(preferred_width, preferred_height)
        return

    available = screen.availableGeometry()
    width = max(min_width, min(preferred_width, round(available.width() * width_fraction)))
    height = max(min_height, min(preferred_height, round(available.height() * height_fraction)))
    window.resize(width, height)

    if center:
        frame = window.frameGeometry()
        frame.moveCenter(available.center())
        window.move(frame.topLeft())
