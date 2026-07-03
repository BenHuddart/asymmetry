"""Shared matplotlib figure/canvas construction.

Converges the ``Figure(...)`` + ``FigureCanvasQTAgg(...)`` (+ optional
``NavigationToolbar2QT(...)``) triples that grew up independently in
``plot_panel.py``, ``fit_parameters_panel.py``, ``global_parameter_fit_window.py``,
and ``alc_panel.py``. :func:`create_canvas` centralizes the matplotlib import
and the layout-engine choice so each call site expresses only what differs
(layout engine, whether a nav toolbar is needed, and the parent for that
toolbar).

**Lazy import is load-bearing.** Several panels guard canvas creation behind
a ``try/except ImportError`` so the module still imports (and degrades to a
"matplotlib not installed" label) in headless/no-matplotlib contexts. The
``matplotlib`` imports below live inside :func:`create_canvas`'s body, not at
module top level, so importing ``mpl_canvas`` itself never pulls in
matplotlib — callers keep their own ``try/except ImportError`` around the
call to :func:`create_canvas`.

Return contract: :func:`create_canvas` returns ``(figure, canvas)`` when
``toolbar=False`` (the default), and ``(figure, canvas, nav_toolbar)`` when
``toolbar=True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QWidget

_Layout = Literal["tight", "constrained", "none"]


@overload
def create_canvas(
    *,
    layout: _Layout = ...,
    toolbar: Literal[False] = ...,
    parent: QWidget | None = ...,
    figsize: tuple[float, float] | None = ...,
) -> tuple[Figure, FigureCanvasQTAgg]: ...


@overload
def create_canvas(
    *,
    layout: _Layout = ...,
    toolbar: Literal[True],
    parent: QWidget | None = ...,
    figsize: tuple[float, float] | None = ...,
) -> tuple[Figure, FigureCanvasQTAgg, NavigationToolbar2QT]: ...


def create_canvas(
    *,
    layout: str | None = "tight",
    toolbar: bool = False,
    parent: QWidget | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Build a matplotlib ``Figure`` + Qt canvas (+ optional nav toolbar).

    Args:
        layout: Layout engine for the figure — ``"tight"`` for
            ``Figure(tight_layout=True)``, ``"constrained"`` for
            ``Figure(constrained_layout=True)``, or ``"none"``/``None`` for a
            plain ``Figure()``. Any other value raises ``ValueError``.
        toolbar: When ``True``, also build a ``NavigationToolbar2QT`` bound to
            the new canvas and return it as a third element.
        parent: Parent widget passed to ``NavigationToolbar2QT`` (ignored
            when ``toolbar=False``). Matplotlib's ``NavigationToolbar2QT``
            accepts ``None``.
        figsize: Optional ``(width, height)`` in inches passed to ``Figure``.

    Returns:
        ``(figure, canvas)`` when ``toolbar=False``; ``(figure, canvas,
        nav_toolbar)`` when ``toolbar=True``.

    Raises:
        ValueError: if ``layout`` is not one of ``"tight"``, ``"constrained"``,
            ``"none"``, or ``None``.
    """
    # Imported lazily (not at module scope) so importing this module never
    # requires matplotlib to be installed — see module docstring.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    figure_kwargs = {}
    if figsize is not None:
        figure_kwargs["figsize"] = figsize

    if layout == "tight":
        figure_kwargs["tight_layout"] = True
    elif layout == "constrained":
        figure_kwargs["constrained_layout"] = True
    elif layout in ("none", None):
        pass
    else:
        raise ValueError(
            f"create_canvas: unknown layout {layout!r}; expected 'tight', "
            "'constrained', 'none', or None"
        )

    figure = Figure(**figure_kwargs)
    canvas = FigureCanvasQTAgg(figure)

    if toolbar:
        nav_toolbar = NavigationToolbar2QT(canvas, parent)
        return figure, canvas, nav_toolbar

    return figure, canvas
