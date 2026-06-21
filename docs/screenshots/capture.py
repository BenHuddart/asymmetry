"""CLI entry point for generating GUI screenshots used in the Sphinx docs.

Usage::

    python -m docs.screenshots.capture --out docs/_generated/screenshots
    python -m docs.screenshots.capture --list
    python -m docs.screenshots.capture --only main_window fourier_tf

The script defaults to ``QT_QPA_PLATFORM=offscreen`` so no display server is
required. Run it from the project root.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

# Hard cap on the entire capture process — if any scenario deadlocks or a fit
# stalls, the watchdog dumps every thread's stack and exits rather than blocking
# the CI job indefinitely.  ``continue-on-error: true`` on the workflow step
# means the Sphinx build proceeds even when we exit with a non-zero code.
_CAPTURE_TIMEOUT_S = 8 * 60  # 8 minutes


def _start_watchdog(timeout_s: int = _CAPTURE_TIMEOUT_S) -> None:
    """Hard-exit the process if capture wedges for *timeout_s* seconds.

    Uses :func:`faulthandler.dump_traceback_later` rather than a pure-Python
    daemon thread calling ``os._exit``. A fit runs on a Qt worker thread, and a
    long ``iminuit``/``migrad`` minimisation holds the GIL inside its native
    (C++/pybind11) call. That starves *every* Python thread — including a
    pure-Python watchdog daemon, which can then never wake to call ``os._exit``.
    This is why a previous 20-minute daemon watchdog let a CI run hang for hours
    instead of self-terminating.

    ``faulthandler``'s timer fires from C without needing the GIL, dumps the
    stacks of all threads (so the CI log shows exactly where it wedged), and
    exits the process. ``os`` import is retained for the env defaults below.
    """
    faulthandler.enable()
    print(
        f"[screenshots] watchdog armed: hard exit + thread dump after {timeout_s}s.",
        flush=True,
    )
    faulthandler.dump_traceback_later(timeout_s, exit=True)


def _ensure_offscreen_default() -> None:
    """Default to the offscreen Qt platform unless the caller overrides it."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")


def _boot_qapplication():
    """Construct a QApplication that matches the real GUI startup styling."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    from asymmetry.gui import app as app_module
    from asymmetry.gui.styles.fonts import (
        configure_plot_fonts,
        register_bundled_fonts,
    )

    # Use a dedicated org/app so we never read or write the user's QSettings.
    app.setApplicationName("AsymmetryScreenshots")
    app.setOrganizationName("Asymmetry")
    settings = QSettings()
    settings.clear()
    settings.setValue("ui/scale", 1.0)

    register_bundled_fonts()
    configure_plot_fonts()

    bench_css = app_module._load_bench_stylesheet()
    if bench_css:
        app.setStyleSheet(bench_css)

    _apply_determinism_patches()

    return app


def _apply_determinism_patches() -> None:
    """Patch sources of non-determinism so PNGs are byte-stable across builds.

    Currently this freezes the wall-clock used by the log panel for its
    leading ``HH:MM:SS`` timestamps; without this the log column varies on
    every CI run and bloats Pages deploy diffs.
    """
    from datetime import datetime as _real_datetime

    from asymmetry.gui.panels import log_panel as _log_panel_module

    class _FrozenDatetime(_real_datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return _real_datetime(2026, 1, 1, 9, 30, 0, tzinfo=tz)

    _log_panel_module.datetime = _FrozenDatetime  # type: ignore[attr-defined]


def _import_scenarios() -> None:
    """Import all scenario modules so they register with the base registry."""
    from .scenarios import (  # noqa: F401
        apodisation_comparison,
        bunching_comparison,
        composite_fractions_dialog,
        composite_models_builder,
        data_browser_filter,
        data_processing_rebin,
        euo_fit_oscillatory,
        fit_wizard_gkt,
        fit_wizard_portfolio,
        fourier_tf,
        # TODO: re-enable global_fit_lfkt and lf_kt_global_results once the global
        # fit wizard has been further developed.  Both scenarios are temporarily
        # excluded because lf_kt_global_results runs a synchronous 4-dataset global
        # fit that takes several minutes on CI, and global_fit_lfkt is the companion
        # setup screenshot for the same feature.  See docs/screenshots/scenarios/
        # global_fit_lfkt.py and lf_kt_global_results.py — the files are intact and
        # ready to be re-imported when the feature is ready.
        #
        # global_fit_lfkt,
        # lf_kt_global_results,
        grouped_fit_ybco_knight,
        lf_kt_series_plot,
        logbook_view,
        main_window,
        maxent_ybco,
        mgb2_lambda_t,
        muon_fluorine_pbf2,
        parameter_trending_mgb2,
        temperature_trend_fit,
        vector_polarization_emu,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/_generated/screenshots"),
        help="Output directory for PNGs (default: docs/_generated/screenshots).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="NAME",
        help="Capture only the named scenarios. Default: all registered.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered scenarios and exit.",
    )
    parser.add_argument(
        "--dpr",
        type=float,
        default=2.0,
        help="Device-pixel ratio for the output PNGs (default: 2.0).",
    )
    parser.add_argument(
        "--skip-fits",
        action="store_true",
        help=(
            "Skip scenarios that require a working fit backend "
            "(``Scenario.requires_fit = True``). Useful on dev environments "
            "where numpy>=2.3 breaks iminuit/numba — CI does not need this."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _start_watchdog()
    args = _parse_args(argv)

    _ensure_offscreen_default()
    _boot_qapplication()
    _import_scenarios()

    from .scenarios._base import CaptureContext, registered_scenarios

    scenarios = registered_scenarios()
    if args.list:
        for name, scenario in scenarios.items():
            description = scenario.description or scenario.__class__.__doc__ or ""
            print(f"{name}\t{description.strip().splitlines()[0] if description else ''}")
        return 0

    if args.only:
        unknown = [n for n in args.only if n not in scenarios]
        if unknown:
            print(f"Unknown scenarios: {', '.join(unknown)}", file=sys.stderr)
            print(f"Known: {', '.join(scenarios)}", file=sys.stderr)
            return 2
        selected = {name: scenarios[name] for name in args.only}
    else:
        selected = scenarios

    if args.skip_fits:
        skipped = [
            name for name, scenario in selected.items() if scenario.requires_fit
        ]
        for name in skipped:
            del selected[name]
            print(f"[screenshots] skipping {name} (requires_fit=True)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = CaptureContext(output_dir=args.out, device_pixel_ratio=args.dpr)

    for name, scenario in selected.items():
        print(f"[screenshots] capturing {name}...", flush=True)
        _t0 = time.monotonic()
        path = scenario.capture(ctx)
        _elapsed = time.monotonic() - _t0
        print(f"[screenshots] wrote {path} ({_elapsed:.1f}s)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
