"""Scenario base class and shared rendering helpers for GUI screenshots.

The :class:`Scenario` base class is intentionally tiny: subclasses implement
:meth:`build` (construct a widget tree and return the widget to grab) and
optionally override :meth:`settle` (a coroutine of ``processEvents`` calls
that lets layouts / canvases finish rendering before the grab).

All scenarios are run inside one shared ``QApplication`` boot performed by
:mod:`docs.screenshots.capture`, so they should not call ``QApplication`` or
mutate global Qt state themselves.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import QCoreApplication, QEventLoop, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

try:
    import oxipng
except ImportError:  # pragma: no cover - exercised when the optional dep is absent
    oxipng = None

_warned_missing_oxipng = False


@dataclass
class CaptureContext:
    """Runtime context handed to each scenario by the capture driver."""

    output_dir: Path
    device_pixel_ratio: float = 2.0


class Scenario:
    """Abstract base for one documentation screenshot.

    Subclasses must set :attr:`name` and implement :meth:`build`. The
    returned widget will be sized to :attr:`size` (a ``(width, height)``
    tuple in logical pixels) and grabbed at :attr:`CaptureContext.device_pixel_ratio`.

    Scenarios that perform a fit at capture time should set
    :attr:`requires_fit` to ``True`` so they can be filtered out by the
    ``--skip-fits`` CLI flag. The fit backend (iminuit/numba) is
    incompatible with numpy ≥ 2.3, which trips dev environments; CI keeps
    numpy < 2.3 via ``constraints.txt`` so fit-bearing scenarios run there.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    size: ClassVar[tuple[int, int]] = (1280, 800)
    requires_fit: ClassVar[bool] = False

    def build(self) -> QWidget:  # pragma: no cover - abstract
        raise NotImplementedError

    def settle(self, widget: QWidget) -> None:
        """Process pending events so the widget is fully laid out before grab."""
        _process_events_for(milliseconds=200)

    def teardown(self, widget: QWidget) -> None:
        """Best-effort cleanup; safe to leave default."""
        # Screenshots are never saved, so suppress the unsaved-changes guard:
        # a MainWindow with a loaded session is "dirty", and its closeEvent
        # would otherwise block forever on a modal save prompt offscreen.
        if hasattr(widget, "_dirty"):
            widget._dirty = False
        widget.close()
        widget.deleteLater()
        _process_events_for(milliseconds=50)

    def capture(self, ctx: CaptureContext) -> Path:
        widget = self.build()
        widget.resize(*self.size)
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        widget.show()
        self.settle(widget)
        # Grab at requested DPR for crisp display on retina screens.
        pixmap = _grab_at_dpr(widget, ctx.device_pixel_ratio)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _optimize_png(out_path)
        self.teardown(widget)
        return out_path


_REGISTRY: dict[str, Scenario] = {}


def register(scenario: Scenario) -> Scenario:
    if not scenario.name:
        raise ValueError("Scenario.name must be set")
    if scenario.name in _REGISTRY:
        raise ValueError(f"Duplicate scenario name: {scenario.name}")
    _REGISTRY[scenario.name] = scenario
    return scenario


def registered_scenarios() -> dict[str, Scenario]:
    return dict(_REGISTRY)


_SB = QMessageBox.StandardButton

# Preference order used when a message box does not name a default button.
# Most dismissive first: a capture run must never *confirm* anything, so the
# guard picks the button a user pressing Esc would get.
_DISMISSIVE_PRIORITY: tuple[QMessageBox.StandardButton, ...] = (
    _SB.Cancel,
    _SB.No,
    _SB.Close,
    _SB.Abort,
    _SB.Ignore,
    _SB.Ok,
    _SB.Yes,
)


def _resolve_dismissal(
    buttons: QMessageBox.StandardButton | int,
    default_button: QMessageBox.StandardButton | int | None,
) -> QMessageBox.StandardButton:
    """Return the button an auto-dismissed message box should answer with.

    The dialog's own default button wins when it names one that is actually on
    offer (that is what Esc / Return would trigger for a real user); otherwise
    the most dismissive available button is chosen.
    """
    mask = int(buttons or 0)
    default = int(default_button or 0)
    if default and default & mask:
        return _SB(default)
    for candidate in _DISMISSIVE_PRIORITY:
        if int(candidate) & mask:
            return candidate
    return _SB.NoButton


def _log_dismissal(kind: str, title: str, chosen: QMessageBox.StandardButton) -> None:
    print(
        f"[screenshots] auto-dismissed modal: {title or '(untitled)'} [{kind}] -> {chosen.name}",
        flush=True,
    )


def _static_dismisser(
    kind: str, fallback_buttons: QMessageBox.StandardButton
) -> Callable[..., object]:
    """Build a replacement for one ``QMessageBox`` static convenience method."""

    def _dismiss(
        parent=None,  # noqa: ANN001 - mirrors the Qt signature
        title: str = "",
        text: str = "",
        buttons: QMessageBox.StandardButton | int = fallback_buttons,
        defaultButton: QMessageBox.StandardButton | int = _SB.NoButton,  # noqa: N803
    ) -> QMessageBox.StandardButton:
        chosen = _resolve_dismissal(buttons, defaultButton)
        _log_dismissal(kind, title, chosen)
        return chosen

    return _dismiss


def _dismiss_exec(box: QMessageBox, *_args, **_kwargs) -> int:
    """Replacement for ``QMessageBox.exec``: close immediately, never block.

    Only ``QMessageBox`` is patched, never ``QDialog`` — scenarios whose subject
    *is* a dialog (the calibration, simulate, and wizard captures) must keep
    working normally.
    """
    default = box.defaultButton()
    chosen = _resolve_dismissal(
        box.standardButtons(),
        box.standardButton(default) if default is not None else _SB.NoButton,
    )
    _log_dismissal("exec", box.windowTitle(), chosen)
    box.done(int(chosen))
    return int(chosen)


@contextmanager
def auto_dismiss_modals() -> Iterator[None]:
    """Neutralise blocking ``QMessageBox`` modals for the duration of a capture.

    Screenshots are captured under ``QT_QPA_PLATFORM=offscreen``, where nobody
    can click a modal: any message box that reaches its own event loop blocks
    the whole process until the capture watchdog hard-exits, and every scenario
    after it is silently never written. That is exactly how the published docs
    lost most of their GUI images — ``GroupingDialog.closeEvent`` runs an
    unsaved-changes guard (``QMessageBox.question("Discard changes", ...)``), so
    ``alpha_calibration_dialog`` wedged on ``dialog.close()`` and took the ~30
    alphabetically later scenarios with it.

    Rather than teach each scenario to tiptoe around its own dirty-state guard,
    the harness makes the failure mode impossible: inside this context every
    ``QMessageBox`` answers instantly with its default (or most dismissive)
    button and logs one ``[screenshots] auto-dismissed modal: ...`` line, so a
    future scenario hitting a new guard degrades to a logged line instead of a
    hung build. Nothing here is destructive — the guards it answers are
    "discard the draft?" prompts on windows the capture is about to throw away.
    """
    patches: dict[str, object] = {
        "question": staticmethod(_static_dismisser("question", _SB.Yes | _SB.No)),
        "information": staticmethod(_static_dismisser("information", _SB.Ok)),
        "warning": staticmethod(_static_dismisser("warning", _SB.Ok)),
        "critical": staticmethod(_static_dismisser("critical", _SB.Ok)),
        "exec": _dismiss_exec,
        "exec_": _dismiss_exec,
    }
    originals = {name: QMessageBox.__dict__.get(name, None) for name in patches}
    saved = {name: getattr(QMessageBox, name) for name in patches}
    try:
        for name, replacement in patches.items():
            setattr(QMessageBox, name, replacement)
        yield
    finally:
        for name in patches:
            if originals[name] is None:
                # ``exec``/``exec_`` are inherited from QDialog; deleting the
                # attribute we added restores the inherited slot cleanly.
                try:
                    delattr(QMessageBox, name)
                except AttributeError:  # pragma: no cover - defensive
                    setattr(QMessageBox, name, saved[name])
            else:
                setattr(QMessageBox, name, originals[name])


def _process_events_for(milliseconds: int) -> None:
    """Pump the Qt event loop for at least ``milliseconds`` ms."""
    app = QApplication.instance()
    if app is None:
        return
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, milliseconds)


def _optimize_png(path: Path) -> None:
    """Losslessly re-encode ``path`` in place with ``oxipng``, if available.

    Optimisation is best-effort: a missing ``pyoxipng`` install (an optional
    dependency, see ``docs/requirements.txt``) must never fail a capture run,
    so this prints one warning for the whole process and leaves the
    Qt-written PNG as-is. ``level=4`` matches ``level=6`` byte-for-byte on
    every screenshot sampled during development but runs in roughly half the
    time; ``StripChunks.safe()`` drops incidental metadata (e.g. timestamps)
    without touching color/gamma data, which keeps the output byte-identical
    across repeated runs of the same input — required so Pages deploy diffs
    only reflect real pixel changes.
    """
    global _warned_missing_oxipng
    if oxipng is None:
        if not _warned_missing_oxipng:
            print(
                "[screenshots] warning: pyoxipng not installed; skipping PNG "
                "optimisation (see docs/requirements.txt)",
                file=sys.stderr,
                flush=True,
            )
            _warned_missing_oxipng = True
        return
    oxipng.optimize(str(path), str(path), level=4, strip=oxipng.StripChunks.safe())


def _grab_at_dpr(widget: QWidget, dpr: float) -> QPixmap:
    """Render ``widget`` to a QPixmap at the requested device-pixel ratio.

    ``QWidget.grab`` honors the widget's own ``devicePixelRatio``. We resize
    a backing pixmap accordingly so the output is sharp at the configured
    DPR rather than the platform default.
    """
    width = int(widget.width() * dpr)
    height = int(widget.height() * dpr)
    pixmap = QPixmap(width, height)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    widget.render(pixmap)
    return pixmap


def call_with_event_loop(callable_: Callable[[], None], *, timeout_ms: int = 500) -> None:
    """Run ``callable_`` after the event loop has pumped a short interval."""
    QTimer.singleShot(0, callable_)
    _process_events_for(timeout_ms)
