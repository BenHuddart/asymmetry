"""``gui/utils/errorbar_dots.py`` — the fast dots-with-error-bars idiom.

``Axes.errorbar`` builds one Path per point for the bars; the shared helper
draws the same picture as a marker-only ``Line2D`` plus one single-path
``LineCollection`` (NaN-separated), which is what keeps a run switch on a
4k-point display budget off the errorbar construction cost.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from asymmetry.gui.utils.errorbar_dots import (  # noqa: E402
    add_errorbar_dots,
    errorbar_dots_line_class,
)


def _axes():
    fig = Figure(figsize=(4, 3), dpi=72)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def test_adds_marker_line_and_single_path_bar_collection() -> None:
    fig, ax = _axes()
    x = np.linspace(0.0, 1.0, 50)
    y = np.sin(x)
    err = np.full_like(x, 0.1)

    marker, bars = add_errorbar_dots(ax, x, y, err, fmt=".", markersize=3, color="C1", label="run")

    assert isinstance(marker, errorbar_dots_line_class())
    assert marker in ax.lines and marker.get_linestyle() == "None"
    assert marker.get_marker() == "." and marker.get_label() == "run"
    assert isinstance(bars, LineCollection) and bars in ax.collections
    # One path carrying every bar, not one path per point.
    assert len(bars.get_paths()) == 1
    verts = bars.get_paths()[0].vertices
    assert verts.shape == (3 * x.size, 2)
    np.testing.assert_allclose(verts[0::3, 1], y - err)
    np.testing.assert_allclose(verts[1::3, 1], y + err)
    assert np.all(np.isnan(verts[2::3, 1]))
    assert marker.bar_line is bars
    # No ErrorbarContainer: the errorbar path was never taken.
    assert ax.containers == []
    fig.canvas.draw()  # renders without error


def test_non_finite_error_draws_no_bar_for_that_point() -> None:
    _fig, ax = _axes()
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.5, 0.6, 0.7])
    err = np.array([0.1, np.nan, 0.1])

    _marker, bars = add_errorbar_dots(ax, x, y, err)

    verts = bars.get_paths()[0].vertices
    assert np.isnan(verts[3, 1]) and np.isnan(verts[4, 1])  # the NaN-error bar
    assert np.isfinite(verts[0, 1]) and np.isfinite(verts[6, 1])


def test_bar_extents_feed_the_axes_data_limits() -> None:
    _fig, ax = _axes()
    x = np.linspace(0.0, 1.0, 10)
    y = np.zeros_like(x)
    err = np.full_like(x, 2.5)

    add_errorbar_dots(ax, x, y, err)

    assert ax.dataLim.y1 == pytest.approx(2.5)
    assert ax.dataLim.y0 == pytest.approx(-2.5)
    # ``autoscale_view`` at draw time picks the bars up, as after ``errorbar``.
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    assert lo <= -2.5 and hi >= 2.5


def test_none_colour_advances_the_cycle_and_bars_follow_the_marker() -> None:
    _fig, ax = _axes()
    x = np.arange(3.0)
    m1, b1 = add_errorbar_dots(ax, x, x, np.ones(3), color=None)
    m2, b2 = add_errorbar_dots(ax, x, x, np.ones(3), color=None)

    assert m1.get_color() != m2.get_color()
    assert tuple(b1.get_color()[0]) == matplotlib.colors.to_rgba(m1.get_color())
    assert tuple(b2.get_color()[0]) == matplotlib.colors.to_rgba(m2.get_color())


def test_ecolor_overrides_bar_colour_only() -> None:
    _fig, ax = _axes()
    x = np.arange(3.0)
    marker, bars = add_errorbar_dots(ax, x, x, np.ones(3), color="C0", ecolor="0.6")
    assert marker.get_color() == "C0"
    assert tuple(bars.get_color()[0]) == matplotlib.colors.to_rgba("0.6")


def test_legend_glyph_keeps_the_error_bar() -> None:
    fig, ax = _axes()
    x = np.arange(5.0)
    add_errorbar_dots(ax, x, x, np.ones(5), color="C2", label="run 1")

    legend = ax.legend()
    fig.canvas.draw()

    assert [t.get_text() for t in legend.get_texts()] == ["run 1"]
    handle = legend.legend_handles[0]
    # The registered handler puts a vertical bar glyph behind the marker.
    assert isinstance(handle, Line2D)
    xs, ys = handle.get_xdata(), handle.get_ydata()
    assert len(xs) == 2 and xs[0] == xs[1] and ys[0] != ys[1]
    assert matplotlib.colors.to_rgba(handle.get_color()) == matplotlib.colors.to_rgba("C2")


def test_never_calls_axes_errorbar(monkeypatch: pytest.MonkeyPatch) -> None:
    _fig, ax = _axes()

    def _boom(*_args, **_kwargs):
        raise AssertionError("errorbar must not be used")

    monkeypatch.setattr(ax, "errorbar", _boom)
    add_errorbar_dots(ax, np.arange(4.0), np.arange(4.0), np.ones(4))
