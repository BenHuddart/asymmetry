#!/usr/bin/env python
"""Wall-clock call tree of a data-browser run switch in a live ``MainWindow``.

cProfile drops the frames Qt dispatches (a slot reached from a C++ signal
emission never appears under its caller), so its cumulative view of a
selection change is misleading. This tool wraps the run-switch pipeline
with timing wrappers instead, opens a project exactly as the GUI does,
selects runs through the real browser path (``DataBrowserPanel.select_runs``
→ ``selection_changed`` → ``dataset_selected``) and prints, per switch, the
synchronous cost, an indented call tree (plot build, canvas draw, fit-panel
binds, grouped-context rebuilds, garbage-collector pauses) and the deferred
event-loop turns that follow (draw_idle, single-shot timers).

Run it on the real display for numbers that include Retina rasterisation::

    .venv/bin/python tools/trace_run_switch.py path/to/project.asymp 374 375 376

Headless (``QT_QPA_PLATFORM=offscreen``) works too but under-reports the
draw. ``--threshold`` hides calls faster than N ms; ``--draw`` also wraps
matplotlib's per-artist draw calls; ``--xmax`` first narrows the view to an
early-time window, the case the decimation must keep cheap.
"""

from __future__ import annotations

import argparse
import functools
import gc
import os
import sys
import time
from collections.abc import Callable

_depth = 0
_records: list[tuple[int, str, float, str]] = []


def wrap(
    cls: object, name: str, label: str | None = None, arg_repr: Callable | None = None
) -> None:
    """Replace ``cls.name`` with a timing wrapper recording depth and elapsed ms."""
    original = getattr(cls, name)
    text = label or f"{getattr(cls, '__name__', cls)}.{name}"

    @functools.wraps(original)
    def inner(*args, **kwargs):
        global _depth
        depth = _depth
        _depth += 1
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _depth -= 1
            elapsed = (time.perf_counter() - started) * 1000.0
            extra = arg_repr(*args, **kwargs) if arg_repr else ""
            _records.append((depth, text, elapsed, extra))

    setattr(cls, name, inner)


def wrap_all(cls: type, prefixes: tuple[str, ...]) -> None:
    """Wrap every plain method of *cls* whose name starts with one of *prefixes*.

    Static/class methods and properties are left alone: re-binding them
    through ``setattr`` would turn them into instance methods.
    """
    for name, attr in list(vars(cls).items()):
        if not name.startswith(prefixes):
            continue
        if isinstance(attr, (staticmethod, classmethod, property)) or not callable(attr):
            continue
        wrap(cls, name)


def dump(threshold_ms: float) -> None:
    # Records are appended on exit; reversing yields parent-before-child order.
    for depth, text, elapsed, extra in reversed(_records):
        if elapsed >= threshold_ms:
            print(f"{'  ' * depth}{elapsed:7.1f} ms  {text} {extra}")
    _records.clear()


def _install_wrappers(*, draw: bool) -> None:
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    import asymmetry.gui.windows.multi_group_fit_window as mgfw
    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.panels.data_browser import DataBrowserPanel
    from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
    from asymmetry.gui.panels.fit.panel import FitPanel
    from asymmetry.gui.panels.fit.single_tab import SingleFitTab
    from asymmetry.gui.panels.fit.tab_base import FitTabBase
    from asymmetry.gui.panels.plot_panel import PlotPanel

    for name in (
        "_on_dataset_selected",
        "_update_selected_datasets",
        "_render_current_selection_plot",
        "_sync_fourier_panel_for_dataset",
        "_refresh_fourier_compute_scope",
        "_synchronize_targets_to_axis",
        "_sync_frequency_plot_for_current_dataset",
        "_refresh_vector_axis_selector",
        "_update_fit_block_state",
        "_get_fit_dataset",
        "_store_fourier_group_phase_state_for_dataset",
        "_store_maxent_panel_state_for_dataset",
        "_update_grouping_hint",
        "_selected_or_current_datasets",
        "_current_single_fit_projection",
        "_refresh_spectral_moments",
    ):
        if hasattr(MainWindow, name):
            wrap(MainWindow, name)
    wrap_all(
        PlotPanel,
        (
            "plot_",
            "_plot_",
            "_apply_limits",
            "_schedule_viewport",
            "_visible_plot",
            "_decimated",
            "_draw",
            "_refresh",
            "_update",
            "_render",
            "_set",
            "set_",
            "clear",
            "_clear",
            "_sync",
        ),
    )
    wrap_all(FitPanel, ("set_", "_"))
    tab_prefixes = (
        "set_",
        "_update",
        "_rebuild",
        "_compute",
        "_grouped",
        "_seed",
        "_refresh",
        "_sync",
        "_populate",
        "_apply",
    )
    for cls in (GlobalFitTab, SingleFitTab, FitTabBase):
        wrap_all(cls, tab_prefixes)
    wrap_all(
        DataBrowserPanel,
        ("_on_selection_changed", "get_selected", "select_runs", "_restore_selection"),
    )
    wrap_all(mgfw.MultiGroupFitWindow, ("set_", "_update", "_rebuild", "_refresh", "_sync"))
    wrap(FigureCanvasQTAgg, "draw")
    wrap(FigureCanvasQTAgg, "paintEvent")
    wrap(RendererAgg, "draw_path")
    wrap(RendererAgg, "draw_path_collection")
    if draw:
        from matplotlib import layout_engine
        from matplotlib.axes import Axes
        from matplotlib.axis import Axis
        from matplotlib.collections import Collection
        from matplotlib.figure import Figure
        from matplotlib.legend import Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.text import Text

        for cls in (Figure, Axes, Axis, Legend, Line2D, Collection, Patch):
            wrap(cls, "draw")
        wrap(Text, "draw", arg_repr=lambda self, *a, **k: repr(self.get_text())[:40])
        for cls in (layout_engine.TightLayoutEngine, layout_engine.ConstrainedLayoutEngine):
            wrap(cls, "execute")
        wrap(
            RendererAgg,
            "draw_text",
            arg_repr=lambda self, gc_, x, y, s, prop, *a, **k: repr(s)[:40],
        )


def _install_gc_probe() -> None:
    starts: dict[int, float] = {}

    def callback(phase: str, info: dict) -> None:
        generation = int(info["generation"])
        if phase == "start":
            starts[generation] = time.perf_counter()
            return
        elapsed = (time.perf_counter() - starts.pop(generation, time.perf_counter())) * 1000.0
        _records.append(
            (_depth, f"<gc gen{generation} collected={info['collected']}>", elapsed, "")
        )

    gc.callbacks.append(callback)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", help="an .asymp project whose data files are reachable")
    parser.add_argument("runs", nargs="*", type=int, help="run numbers to switch through, in order")
    parser.add_argument("--threshold", type=float, default=2.0, help="hide calls faster than N ms")
    parser.add_argument("--draw", action="store_true", help="also wrap matplotlib per-artist draws")
    parser.add_argument(
        "--xmax", type=float, default=None, help="narrow the view to [0, xmax] first"
    )
    parser.add_argument("--settle", type=float, default=1.0, help="seconds to watch deferred turns")
    args = parser.parse_args(argv)

    os.environ.setdefault("ASYMMETRY_PERF_LOGGING", "0")
    from PySide6.QtWidgets import QApplication

    from asymmetry.gui.mainwindow import MainWindow

    _install_wrappers(draw=args.draw)
    _install_gc_probe()

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1600, 1000)
    window.show()
    app.processEvents()
    window._open_project_file(args.project)
    for _ in range(50):
        app.processEvents()
    if args.xmax is not None:
        window._plot_panel.set_view_limits(
            0.0, args.xmax, *window._plot_panel.get_view_limits()[2:]
        )
        for _ in range(20):
            app.processEvents()
    _records.clear()

    runs = args.runs or [ds.run_number for ds in window._data_browser.get_all_datasets()[:4]]
    panel = window._plot_panel
    print(
        f"view x=({panel._x_min.value():.3g}, {panel._x_max.value():.3g}) "
        f"dpr={panel._canvas.devicePixelRatioF():.1f} "
        f"tracked objects={len(gc.get_objects())}"
    )
    for run_number in runs:
        print(f"\n########## switch to {run_number}")
        started = time.perf_counter()
        window._data_browser.select_runs({int(run_number)})
        print(f"--- synchronous: {(time.perf_counter() - started) * 1000:.0f} ms")
        dump(args.threshold)
        print(f"--- deferred (event-loop turns > 5 ms within {args.settle:.1f} s):")
        deadline = time.perf_counter() + args.settle
        while time.perf_counter() < deadline:
            turn_started = time.perf_counter()
            app.processEvents()
            turn_ms = (time.perf_counter() - turn_started) * 1000.0
            if turn_ms > 5:
                print(f"  [turn {turn_ms:.0f} ms]")
                dump(args.threshold)
            else:
                _records.clear()
            time.sleep(0.002)
    window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
