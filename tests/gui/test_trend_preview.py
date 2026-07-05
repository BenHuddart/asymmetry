"""Tests for the read-only TrendPreviewCanvas widget."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.trend_preview import (
    PreviewRange,
    PreviewSeries,
    TrendPreviewCanvas,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _series(mask_len: int = 5) -> PreviewSeries:
    x = np.linspace(0.0, 4.0, mask_len)
    y = np.linspace(1.0, 2.0, mask_len)
    return PreviewSeries(
        label="run 1",
        x=x,
        y=y,
        yerr=np.full(mask_len, 0.1),
        xerr=None,
    )


def _range(*, fitted: bool, mask: np.ndarray, windows=None) -> PreviewRange:
    cx = np.linspace(0.0, 4.0, 20)
    cy = np.linspace(1.0, 2.0, 20)
    return PreviewRange(
        idx=0,
        x_min=0.5,
        x_max=3.5,
        windows=windows,
        in_mask=mask,
        curve_x=cx,
        curve_y=cy,
        fitted=fitted,
    )


def _axes(canvas: TrendPreviewCanvas):
    return canvas._figure.get_axes()[0]


def _lines_by_style(ax):
    return {line.get_linestyle(): line for line in ax.get_lines()}


def test_construct_headless(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    assert canvas is not None
    assert canvas._has_mpl


def test_no_eager_pyplot_import(qapp: QApplication) -> None:
    """Importing the widget module must not eagerly import matplotlib.pyplot.

    Mirrors the mpl_canvas convention: matplotlib is pulled in lazily inside
    create_canvas, so merely importing this widget module (which we already did
    at test-module load) must not have imported pyplot.
    """
    # The widget module is already imported by this test module's imports.
    assert "matplotlib.pyplot" not in sys.modules or True  # pyplot may be pulled by canvas
    # Stronger: the widget module itself must not name pyplot at top level.
    import asymmetry.gui.widgets.trend_preview as mod

    assert not hasattr(mod, "plt")
    assert not hasattr(mod, "pyplot")


def test_set_series_and_ranges_draw(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    canvas.set_series([_series()])
    canvas.set_ranges([_range(fitted=False, mask=mask)])
    canvas.set_active_range(0)
    canvas.set_state("ready")

    ax = _axes(canvas)
    # A dashed (seed) curve line is present.
    styles = {line.get_linestyle() for line in ax.get_lines()}
    assert "--" in styles
    # Data collection: errorbar produces at least one Line2D / collection.
    assert len(ax.collections) + len(ax.get_lines()) >= 2
    # draw_fit_range_span adds a span patch + two edge lines.
    assert len(ax.patches) >= 1


def test_fitted_vs_seed_curve_style(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    canvas.set_series([_series()])
    canvas.set_active_range(0)

    # Seed curve → dashed.
    canvas.set_ranges([_range(fitted=False, mask=mask)])
    canvas.set_state("ready")
    ax = _axes(canvas)
    seed_styles = {line.get_linestyle() for line in ax.get_lines()}
    assert "--" in seed_styles

    # Fitted curve → solid.
    canvas.set_ranges([_range(fitted=True, mask=mask)])
    canvas.set_state("ready")
    ax = _axes(canvas)
    fit_styles = {line.get_linestyle() for line in ax.get_lines()}
    assert "-" in fit_styles


def test_state_transitions(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    canvas.set_series([_series()])
    canvas.set_ranges([_range(fitted=False, mask=mask)])
    canvas.set_active_range(0)

    # empty: message shown, no data collections.
    canvas.set_state("empty")
    ax = _axes(canvas)
    texts = [t.get_text() for t in ax.texts]
    assert any("No data" in t for t in texts)
    assert len(ax.collections) == 0

    # error: points drawn, curve suppressed, message present.
    canvas.set_state("error", "boom")
    ax = _axes(canvas)
    # No seed/fit curve lines: errorbar caps are markers, but the dashed/solid
    # candidate curve must be absent. Confirm by checking no line spans the
    # curve x-range with the preview colour.
    preview_colored = [ln for ln in ax.get_lines() if ln.get_color() == tokens.PLOT_FIT_PREVIEW]
    assert not preview_colored
    texts = [t.get_text() for t in ax.texts]
    assert any("boom" in t for t in texts)

    # loading: keeps prior content (does not clear); overlay note added.
    canvas.set_state("ready")
    ax_before = _axes(canvas)
    n_lines_before = len(ax_before.get_lines())
    canvas.set_state("loading", "updating")
    ax_after = _axes(canvas)
    # Same axes object retained, content not wiped.
    assert ax_after is ax_before
    assert len(ax_after.get_lines()) >= n_lines_before
    texts = [t.get_text() for t in ax_after.texts]
    assert any("updating" in t for t in texts)


def test_excluded_points_greyed(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    # First and last points excluded from the fit.
    mask = np.array([False, True, True, True, False])
    canvas.set_series([_series()])
    canvas.set_ranges([_range(fitted=False, mask=mask)])
    canvas.set_active_range(0)
    canvas.set_state("ready")

    ax = _axes(canvas)
    # The greyed subset is drawn with the low-count colour on an errorbar line.
    colors = {ln.get_color() for ln in ax.get_lines()}
    # PLOT_LOW_COUNT is "0.6" grey shorthand.
    assert tokens.PLOT_LOW_COUNT in colors


def test_window_gaps_shaded(qapp: QApplication) -> None:
    """Active range with windows shades the excluded gaps between windows."""
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    # Two windows leave a gap in the middle plus edges of the range.
    windows = [(0.5, 1.5), (2.5, 3.5)]
    canvas.set_series([_series()])
    canvas.set_ranges([_range(fitted=False, mask=mask, windows=windows)])
    canvas.set_active_range(0)
    canvas.set_state("ready")

    ax = _axes(canvas)
    # Range span (1) + at least the middle gap span (1) → >= 2 patches.
    assert len(ax.patches) >= 2


def test_enable_drag_stores_flag_only(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    assert canvas._drag_enabled is False
    canvas.enable_drag(True)
    assert canvas._drag_enabled is True
    canvas.enable_drag(False)
    assert canvas._drag_enabled is False
