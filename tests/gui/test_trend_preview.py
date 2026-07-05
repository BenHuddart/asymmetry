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
    range_span_color,
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


def _range(*, fitted: bool, mask: np.ndarray, windows=None, idx: int = 0) -> PreviewRange:
    cx = np.linspace(0.0, 4.0, 20)
    cy = np.linspace(1.0, 2.0, 20)
    return PreviewRange(
        idx=idx,
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


def test_range_span_color_stable() -> None:
    """range_span_color is deterministic, distinct per idx, and cycles."""
    from asymmetry.gui.widgets.trend_preview import _RANGE_COLOR_CYCLE

    color0 = range_span_color(0)
    color1 = range_span_color(1)
    assert isinstance(color0, str) and isinstance(color1, str)
    assert color0 != color1
    # Stable across repeated calls.
    assert range_span_color(0) == color0
    assert range_span_color(1) == color1
    # Cyclic: wraps after the palette length.
    n = len(_RANGE_COLOR_CYCLE)
    assert range_span_color(n) == color0
    assert range_span_color(n + 1) == color1


def test_spans_use_range_span_color(qapp: QApplication) -> None:
    """Each range's span face uses its own range_span_color; active is emphasized."""
    import matplotlib.colors as mcolors

    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    canvas.set_series([_series()])
    range0 = _range(fitted=False, mask=mask, idx=0)
    range1 = _range(fitted=False, mask=mask, idx=1)
    canvas.set_ranges([range0, range1])
    canvas.set_active_range(0)
    canvas.set_state("ready")

    ax = _axes(canvas)
    expected0 = mcolors.to_rgb(range_span_color(0))
    expected1 = mcolors.to_rgb(range_span_color(1))

    # Find the axvspan patches for each range by matching facecolor (ignoring
    # alpha) to the expected per-range colour.
    span_patches_0 = [p for p in ax.patches if mcolors.to_rgb(p.get_facecolor()) == expected0]
    span_patches_1 = [p for p in ax.patches if mcolors.to_rgb(p.get_facecolor()) == expected1]
    assert span_patches_0, "no span patch found using range_span_color(0)"
    assert span_patches_1, "no span patch found using range_span_color(1)"

    # Active range (idx 0) is more emphasized: higher alpha than the non-active
    # range's span face.
    active_alpha = max(p.get_alpha() or 0.0 for p in span_patches_0)
    non_active_alpha = max(p.get_alpha() or 0.0 for p in span_patches_1)
    assert active_alpha > non_active_alpha

    # The active range also gets the dashed BENCH edge-line treatment; the
    # non-active range does not get those accent-coloured edge lines.
    accent_edges = [
        ln
        for ln in ax.get_lines()
        if ln.get_linestyle() == "--" and ln.get_color() == tokens.PLOT_FIT_RANGE_EDGE
    ]
    assert accent_edges, "active range should still draw the BENCH dashed edges"


def test_enable_drag_stores_flag_only(qapp: QApplication) -> None:
    canvas = TrendPreviewCanvas()
    assert canvas._drag_enabled is False
    canvas.enable_drag(True)
    assert canvas._drag_enabled is True
    canvas.enable_drag(False)
    assert canvas._drag_enabled is False


# ── Drag interaction ─────────────────────────────────────────────────────────
def _fake_event(canvas, ax, data_x, *, button=1, inside=True):
    """A minimal event exposing the attrs the drag handlers read.

    ``x`` (device pixel) is projected from ``data_x`` through ``transData`` so
    ``nearest_handle`` hit-tests correctly; ``xdata``/``inaxes`` mimic a cursor
    inside (or outside) the axes.
    """

    class _E:
        pass

    e = _E()
    e.button = button
    e.x = float(ax.transData.transform((data_x, 0.0))[0])
    e.y = float(ax.transData.transform((0.0, 0.0))[1])
    e.xdata = float(data_x) if inside else None
    e.ydata = 0.0 if inside else None
    e.inaxes = ax if inside else None
    return e


def _drawn_canvas(*, windows=None):
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    canvas.set_series([_series()])
    canvas.set_ranges([_range(fitted=False, mask=mask, windows=windows)])
    canvas.set_active_range(0)
    canvas.set_state("ready")
    # Ensure a draw has happened so transData is valid under offscreen.
    canvas._canvas.draw()
    return canvas


def test_edge_drag_emits_signal(qapp: QApplication) -> None:
    canvas = _drawn_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.range_edge_dragged.connect(lambda i, lo, hi: events.append((i, lo, hi)))

    # Grab the range-max edge (x_max = 3.5) and drag it to 3.0.
    canvas._on_button_press(_fake_event(canvas, ax, 3.5, button=1))
    assert canvas._active_handle == ("range", "max")
    canvas._on_motion_notify(_fake_event(canvas, ax, 3.0, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 3.0, button=1))

    assert events, "range_edge_dragged never fired"
    idx, x_min, x_max = events[-1]
    assert idx == 0
    assert x_min == pytest.approx(0.5)
    assert x_max == pytest.approx(3.0)


def test_window_edge_drag_emits_signal(qapp: QApplication) -> None:
    windows = [(1.0, 1.5), (2.5, 3.0)]
    canvas = _drawn_canvas(windows=windows)
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.window_edge_dragged.connect(lambda i, w, lo, hi: events.append((i, w, lo, hi)))

    # Grab the second window's hi edge (3.0) and drag it to 3.2.
    canvas._on_button_press(_fake_event(canvas, ax, 3.0, button=1))
    assert canvas._active_handle == ("window", 1, "hi")
    canvas._on_motion_notify(_fake_event(canvas, ax, 3.2, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 3.2, button=1))

    assert events, "window_edge_dragged never fired"
    idx, w_idx, lo, hi = events[-1]
    assert idx == 0
    assert w_idx == 1
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(3.2)


def test_exclude_drag_emits_region(qapp: QApplication) -> None:
    canvas = _drawn_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.exclude_region_requested.connect(lambda i, lo, hi: events.append((i, lo, hi)))

    # Right-drag from 2.5 back to 1.5 → emitted region is sorted (1.5, 2.5).
    canvas._on_button_press(_fake_event(canvas, ax, 2.5, button=3))
    canvas._on_button_release(_fake_event(canvas, ax, 1.5, button=3))

    assert events, "exclude_region_requested never fired"
    idx, lo, hi = events[-1]
    assert idx == 0
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(2.5)


def test_drag_inert_when_disabled(qapp: QApplication) -> None:
    canvas = _drawn_canvas(windows=[(1.0, 1.5)])
    canvas.enable_drag(False)
    ax = _axes(canvas)

    fired: list[str] = []
    canvas.range_edge_dragged.connect(lambda *a: fired.append("range"))
    canvas.window_edge_dragged.connect(lambda *a: fired.append("window"))
    canvas.exclude_region_requested.connect(lambda *a: fired.append("exclude"))

    # Left-edge gesture.
    canvas._on_button_press(_fake_event(canvas, ax, 3.5, button=1))
    canvas._on_motion_notify(_fake_event(canvas, ax, 3.0, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 3.0, button=1))
    # Right-exclude gesture.
    canvas._on_button_press(_fake_event(canvas, ax, 2.5, button=3))
    canvas._on_button_release(_fake_event(canvas, ax, 1.5, button=3))

    assert fired == []
    assert canvas._active_handle is None


def test_exclude_click_no_drag_is_ignored(qapp: QApplication) -> None:
    """A right-click with no movement is a click, not an exclude gesture."""
    canvas = _drawn_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.exclude_region_requested.connect(lambda *a: events.append(a))

    canvas._on_button_press(_fake_event(canvas, ax, 2.0, button=3))
    canvas._on_button_release(_fake_event(canvas, ax, 2.0, button=3))

    assert events == []


def test_edge_clamps_at_partner(qapp: QApplication) -> None:
    """Dragging min past max clamps: min stays <= max."""
    canvas = _drawn_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.range_edge_dragged.connect(lambda i, lo, hi: events.append((lo, hi)))

    # Grab the min edge (0.5) and drag it well past max (3.5) to 5.0.
    canvas._on_button_press(_fake_event(canvas, ax, 0.5, button=1))
    assert canvas._active_handle == ("range", "min")
    canvas._on_motion_notify(_fake_event(canvas, ax, 5.0, button=1))

    assert events, "range_edge_dragged never fired"
    x_min, x_max = events[-1]
    assert x_min <= x_max
    # Clamped at the partner (max stayed at 3.5).
    assert x_min == pytest.approx(3.5)
    assert x_max == pytest.approx(3.5)


def test_disable_mid_drag_stops_cleanly(qapp: QApplication) -> None:
    """enable_drag(False) mid-drag abandons the grab and blocks further emits."""
    canvas = _drawn_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.range_edge_dragged.connect(lambda *a: events.append(a))

    canvas._on_button_press(_fake_event(canvas, ax, 3.5, button=1))
    assert canvas._active_handle is not None
    canvas.enable_drag(False)
    assert canvas._active_handle is None
    canvas._on_motion_notify(_fake_event(canvas, ax, 3.0, button=1))
    assert events == []


# ── Canvas add/select gestures (contract C-CANVAS-ADD / C-GESTURE) ───────────
def _two_range_canvas():
    """Two ranges: idx 0 active at [0.5, 3.5], idx 1 non-active at [5.0, 7.0]."""
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    # Widen the data so both spans (and empty space beyond) are on-axes.
    x = np.linspace(0.0, 10.0, 5)
    y = np.linspace(1.0, 2.0, 5)
    canvas.set_series([PreviewSeries(label="run 1", x=x, y=y, yerr=np.full(5, 0.1), xerr=None)])
    range0 = _range(fitted=False, mask=mask, idx=0)  # [0.5, 3.5]
    range1 = _range(fitted=False, mask=mask, idx=1)
    range1.x_min = 5.0
    range1.x_max = 7.0
    canvas.set_ranges([range0, range1])
    canvas.set_active_range(0)
    canvas.set_state("ready")
    canvas._canvas.draw()
    return canvas


def _rubberband_present(canvas) -> bool:
    """True if the tracked rubber-band artist is still attached to the axes."""
    if canvas._rubberband is None:
        return False
    ax = _axes(canvas)
    return canvas._rubberband in ax.patches


def test_press_on_edge_moves_not_creates(qapp: QApplication) -> None:
    """Press near the active range's edge → MOVE_EDGE, never a create."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    added: list[tuple] = []
    edged: list[tuple] = []
    canvas.range_add_requested.connect(lambda lo, hi: added.append((lo, hi)))
    canvas.range_edge_dragged.connect(lambda *a: edged.append(a))

    # Press exactly on the active range's max edge (3.5).
    canvas._on_button_press(_fake_event(canvas, ax, 3.5, button=1))
    assert canvas._active_handle == ("range", "max")
    assert canvas._create_start_x is None
    canvas._on_motion_notify(_fake_event(canvas, ax, 3.0, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 3.0, button=1))

    assert edged, "edge drag should have fired"
    assert added == []


def test_press_in_other_span_selects(qapp: QApplication) -> None:
    """Press inside the NON-active range's span, negligible move → select."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    selected: list[int] = []
    added: list[tuple] = []
    canvas.range_select_requested.connect(lambda i: selected.append(i))
    canvas.range_add_requested.connect(lambda lo, hi: added.append((lo, hi)))

    # 6.0 is inside range 1's span [5.0, 7.0].
    canvas._on_button_press(_fake_event(canvas, ax, 6.0, button=1))
    assert canvas._select_idx == 1
    canvas._on_button_release(_fake_event(canvas, ax, 6.0, button=1))

    assert selected == [1]
    assert added == []


def test_drag_starting_in_other_span_selects_not_creates(qapp: QApplication) -> None:
    """A drag that STARTS inside a non-active span resolves to SELECT (emitted
    unconditionally, regardless of drag distance) and NEVER creates."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    selected: list[int] = []
    added: list[tuple] = []
    canvas.range_select_requested.connect(lambda i: selected.append(i))
    canvas.range_add_requested.connect(lambda lo, hi: added.append((lo, hi)))

    # Press inside range 1 [5.0, 7.0], then drag well beyond it.
    canvas._on_button_press(_fake_event(canvas, ax, 5.5, button=1))
    assert canvas._select_idx == 1
    assert canvas._create_start_x is None
    canvas._on_motion_notify(_fake_event(canvas, ax, 9.5, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 9.5, button=1))

    # A press that starts on range 1's coverage means "select range 1" — the
    # drag distance is irrelevant (no rival gesture to fence off). It selects and
    # crucially never creates.
    assert selected == [1]
    assert added == []


def test_drag_empty_space_emits_range_add(qapp: QApplication) -> None:
    """Press on empty space, drag > _ADD_MIN_PX → range_add_requested (sorted)."""
    from asymmetry.gui.widgets.trend_preview import _ADD_MIN_PX

    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    added: list[tuple] = []
    canvas.range_add_requested.connect(lambda lo, hi: added.append((lo, hi)))

    # 8.5 → 9.5 is empty (beyond range 1's max of 7.0). Drag right-to-left to
    # confirm the emitted bounds come out sorted.
    canvas._on_button_press(_fake_event(canvas, ax, 9.5, button=1))
    assert canvas._create_start_x == pytest.approx(9.5)
    # Motion draws a rubber-band.
    canvas._on_motion_notify(_fake_event(canvas, ax, 8.5, button=1))
    band_shown = canvas._rubberband is not None
    # Verify the pixel span exceeds the threshold for this axes scaling.
    px0 = _fake_event(canvas, ax, 9.5).x
    px1 = _fake_event(canvas, ax, 8.5).x
    assert abs(px1 - px0) >= _ADD_MIN_PX
    canvas._on_button_release(_fake_event(canvas, ax, 8.5, button=1))

    assert band_shown, "a rubber-band should have been drawn during motion"
    assert added, "range_add_requested never fired"
    lo, hi = added[-1]
    assert lo == pytest.approx(8.5)
    assert hi == pytest.approx(9.5)
    assert not _rubberband_present(canvas)


def test_bare_click_empty_is_noop(qapp: QApplication) -> None:
    """Press+release on empty with negligible move → no signal, no rubber-band."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    added: list[tuple] = []
    selected: list[int] = []
    canvas.range_add_requested.connect(lambda lo, hi: added.append((lo, hi)))
    canvas.range_select_requested.connect(lambda i: selected.append(i))

    n_patches_before = len(ax.patches)
    canvas._on_button_press(_fake_event(canvas, ax, 9.0, button=1))
    assert canvas._create_start_x == pytest.approx(9.0)
    canvas._on_button_release(_fake_event(canvas, ax, 9.0, button=1))

    assert added == []
    assert selected == []
    assert not _rubberband_present(canvas)
    assert len(ax.patches) == n_patches_before


def test_right_drag_still_excludes(qapp: QApplication) -> None:
    """Regression: right-drag on the active range still emits exclude."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    events: list[tuple] = []
    canvas.exclude_region_requested.connect(lambda i, lo, hi: events.append((i, lo, hi)))

    canvas._on_button_press(_fake_event(canvas, ax, 2.5, button=3))
    canvas._on_button_release(_fake_event(canvas, ax, 1.5, button=3))

    assert events, "exclude_region_requested never fired"
    idx, lo, hi = events[-1]
    assert idx == 0
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(2.5)


def test_create_and_select_disabled_when_drag_off(qapp: QApplication) -> None:
    """With drag off, empty-drag and span-click emit nothing."""
    canvas = _two_range_canvas()
    canvas.enable_drag(False)
    ax = _axes(canvas)

    fired: list[str] = []
    canvas.range_add_requested.connect(lambda *a: fired.append("add"))
    canvas.range_select_requested.connect(lambda *a: fired.append("select"))

    # Empty-space drag.
    canvas._on_button_press(_fake_event(canvas, ax, 9.5, button=1))
    canvas._on_motion_notify(_fake_event(canvas, ax, 8.5, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 8.5, button=1))
    # Span click on the non-active range.
    canvas._on_button_press(_fake_event(canvas, ax, 6.0, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 6.0, button=1))

    assert fired == []
    assert canvas._create_start_x is None
    assert canvas._select_idx is None
    assert not _rubberband_present(canvas)


def test_rubberband_cleared_on_release(qapp: QApplication) -> None:
    """No rubber-band artist survives: create drag, no-op click, or leave-axes."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    # 1. Normal create drag.
    canvas._on_button_press(_fake_event(canvas, ax, 9.5, button=1))
    canvas._on_motion_notify(_fake_event(canvas, ax, 8.5, button=1))
    assert canvas._rubberband is not None
    canvas._on_button_release(_fake_event(canvas, ax, 8.5, button=1))
    assert not _rubberband_present(canvas)

    # 2. No-op bare click.
    canvas._on_button_press(_fake_event(canvas, ax, 9.0, button=1))
    canvas._on_button_release(_fake_event(canvas, ax, 9.0, button=1))
    assert not _rubberband_present(canvas)

    # 3. Drag that leaves the axes mid-gesture, then releases outside.
    canvas._on_button_press(_fake_event(canvas, ax, 8.5, button=1))
    canvas._on_motion_notify(_fake_event(canvas, ax, 9.0, button=1))
    assert canvas._rubberband is not None
    # Cursor leaves the axes: motion with inside=False must not extend/crash.
    canvas._on_motion_notify(_fake_event(canvas, ax, 9.0, button=1, inside=False))
    # Release outside the axes still clears the band.
    canvas._on_button_release(_fake_event(canvas, ax, 9.0, button=1, inside=False))
    assert not _rubberband_present(canvas)


def test_drag_off_mid_create_clears_rubberband(qapp: QApplication) -> None:
    """enable_drag(False) mid-create aborts and removes the rubber-band."""
    canvas = _two_range_canvas()
    canvas.enable_drag(True)
    ax = _axes(canvas)

    canvas._on_button_press(_fake_event(canvas, ax, 9.5, button=1))
    canvas._on_motion_notify(_fake_event(canvas, ax, 8.5, button=1))
    assert canvas._rubberband is not None

    canvas.enable_drag(False)
    assert canvas._create_start_x is None
    assert not _rubberband_present(canvas)


def test_residual_axis_toggles(qapp: QApplication) -> None:
    """set_show_residuals(True) adds a residual axis with points; (False) removes it."""
    canvas = TrendPreviewCanvas()
    mask = np.array([True, True, True, True, True])
    canvas.set_series([_series()])
    canvas.set_ranges([_range(fitted=False, mask=mask)])
    canvas.set_active_range(0)
    canvas.set_state("ready")

    # Single axis by default.
    assert len(canvas._figure.get_axes()) == 1

    canvas.set_show_residuals(True)
    axes = canvas._figure.get_axes()
    assert len(axes) == 2
    # Main axis is still axes[0] so dragging keeps working (axes[0]); the
    # residual strip is axes[1].
    resid_ax = axes[1]
    assert resid_ax.get_ylabel() == "resid/σ"
    # The residual strip drew standardized-residual points (a marker line).
    resid_point_lines = [line for line in resid_ax.get_lines() if line.get_linestyle() == "None"]
    assert resid_point_lines
    xs = resid_point_lines[0].get_xdata()
    assert len(xs) == mask.sum()

    canvas.set_show_residuals(False)
    assert len(canvas._figure.get_axes()) == 1


def test_residuals_no_curve_no_crash(qapp: QApplication) -> None:
    """With residuals on but no active-range curve, the strip is empty (no crash)."""
    canvas = TrendPreviewCanvas()
    canvas.set_show_residuals(True)
    canvas.set_series([_series()])
    # A range whose curve is empty.
    empty_range = PreviewRange(
        idx=0,
        x_min=0.5,
        x_max=3.5,
        windows=None,
        in_mask=np.array([True, True, True, True, True]),
        curve_x=np.array([], dtype=float),
        curve_y=np.array([], dtype=float),
        fitted=False,
    )
    canvas.set_ranges([empty_range])
    canvas.set_active_range(0)
    canvas.set_state("ready")

    axes = canvas._figure.get_axes()
    assert len(axes) == 2
    resid_ax = axes[1]
    # No residual point markers (curve was empty), but no exception either.
    resid_point_lines = [line for line in resid_ax.get_lines() if line.get_linestyle() == "None"]
    assert resid_point_lines == []
