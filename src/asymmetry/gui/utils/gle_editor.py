"""Launch the gleplot editor GUI in-process for exported ``.gle`` figures.

Asymmetry ships gleplot as a library dependency (the ``gle`` extra), and
gleplot ≥ 1.6 exposes an embedding API — ``gleplot.gui.open_editor`` — that
opens its editor window inside the host's ``QApplication``. Because the window
joins Asymmetry's application it inherits the bench stylesheet, fonts, and UI
zoom, so the editor looks native.

This module is the single integration point:

* :func:`gle_editor_available` — feature-detects the API so callers can fall
  back to the legacy static PNG preview against older gleplot installs.
* :func:`launch_gle_editor` — opens the editor on a ``.gle`` file (or blank),
  wiring Asymmetry's configured GLE binary into the editor's preview renderer.
* :func:`close_all_gle_editors` — shutdown hook for ``MainWindow.closeEvent``.

Editor windows are top-level and parentless (they outlive the panel that
exported the figure), so this module holds the strong references Qt does not:
a parentless window with no Python reference is garbage-collected and vanishes.
Windows are created with ``WA_DeleteOnClose`` and drop out of the registry via
their ``destroyed`` signal.
"""

from __future__ import annotations

from pathlib import Path

_open_windows: list = []


def gle_editor_available() -> bool:
    """Return True when the installed gleplot provides the embedding API.

    False against gleplot builds that predate ``gleplot.gui`` (< 1.2) or its
    ``open_editor`` entry point (< 1.6), and when gleplot itself is missing.
    """
    try:
        from gleplot.gui import open_editor  # noqa: F401
    except Exception:
        return False
    return True


def _prune_dead_windows(*_args) -> None:
    """Drop strong references to destroyed editor windows.

    Connected to each window's ``destroyed`` signal. The signal argument is
    useless for identification — PySide delivers a fresh ``QWidget`` wrapper
    for the dying object, which does not compare equal to the stored
    ``MainWindow`` wrapper — so this sweeps by liveness instead.
    """
    import shiboken6

    _open_windows[:] = [w for w in _open_windows if shiboken6.isValid(w)]


def launch_gle_editor(gle_path: Path | str | None = None) -> bool:
    """Open a gleplot editor window on ``gle_path`` (or blank when ``None``).

    Passes Asymmetry's configured GLE executable (Setup ▸ GLE Setup…) into the
    editor so its live preview compiles with the same binary as Asymmetry's
    exports; ``None`` lets gleplot auto-detect. Returns True when a window was
    opened, False when the embedding API is unavailable or launching failed —
    callers should then fall back to the legacy static preview.
    """
    try:
        from gleplot.gui import open_editor
    except Exception:
        return False

    from PySide6.QtCore import Qt

    from asymmetry.gui.gle_settings import get_gle_executable

    try:
        window = open_editor(
            str(gle_path) if gle_path is not None else None,
            gle_executable=get_gle_executable(),
        )
    except Exception:
        # Best-effort like the legacy preview: a failed launch must never
        # break the export that triggered it.
        return False

    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _prune_dead_windows()
    _open_windows.append(window)
    window.destroyed.connect(_prune_dead_windows)
    window.raise_()
    window.activateWindow()
    return True


def open_gle_editor_count() -> int:
    """Return the number of live editor windows (test/introspection hook)."""
    _prune_dead_windows()
    return len(_open_windows)


def close_all_gle_editors() -> None:
    """Close every open editor window (``MainWindow.closeEvent`` hook).

    Uses gleplot's ``force_close`` when available: a plain ``close()`` on a
    dirty editor document pops a modal discard-confirmation, which cannot be
    answered during host shutdown and would hang a headless test run.
    """
    for window in list(_open_windows):
        try:
            closer = getattr(window, "force_close", None) or window.close
            closer()
        except RuntimeError:
            # C++ side already gone; the destroyed hook will have pruned it.
            pass
    _open_windows.clear()
