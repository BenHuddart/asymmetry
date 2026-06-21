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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

from PySide6.QtCore import QCoreApplication, QEventLoop, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget


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


def _process_events_for(milliseconds: int) -> None:
    """Pump the Qt event loop for at least ``milliseconds`` ms."""
    app = QApplication.instance()
    if app is None:
        return
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, milliseconds)


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
