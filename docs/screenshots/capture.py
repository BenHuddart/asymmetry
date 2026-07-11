"""CLI entry point for generating GUI screenshots used in the Sphinx docs.

Usage::

    python -m docs.screenshots.capture --out docs/_generated/screenshots
    python -m docs.screenshots.capture --list
    python -m docs.screenshots.capture --only main_window fourier_tf
    python -m docs.screenshots.capture --check-refs

The script defaults to ``QT_QPA_PLATFORM=offscreen`` so no display server is
required. Run it from the project root.
"""

from __future__ import annotations

import argparse
import ast
import faulthandler
import os
import re
import sys
import time
from pathlib import Path

# Hard cap on the entire capture process — if any scenario deadlocks or a fit
# stalls, the watchdog dumps every thread's stack and exits rather than blocking
# the CI job indefinitely.  ``continue-on-error: true`` on the workflow step
# means the Sphinx build proceeds even when we exit with a non-zero code.
_CAPTURE_TIMEOUT_S = 8 * 60  # 8 minutes

DOCS_DIR = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
GENERATED_SCREENSHOTS_DIR = DOCS_DIR / "_generated" / "screenshots"

# Per-image size budget after lossless optimisation (see
# ``scenarios/_base.py::_optimize_png``). The largest current screenshot is
# ~450 KB pre-optimisation; 600 KB leaves headroom as the corpus grows while
# still catching an accidentally huge capture (e.g. a scenario sized far
# beyond the usual (1280, 800)).
SCREENSHOT_SIZE_BUDGET_BYTES = 600 * 1024

# A screenshot reference in the .rst sources: `/_generated/screenshots/<name>.png`
# inside an `image::`/`figure::` directive (matching the path token alone keeps
# the scan robust to directive style and indentation).
_SCREENSHOT_REF_RE = re.compile(r"/_generated/screenshots/([A-Za-z0-9_-]+)\.png")


def scenario_names_from_source(scenarios_dir: Path = SCENARIOS_DIR) -> set[str]:
    """Return every scenario ``name`` declared in the scenario modules.

    Scans class bodies statically (AST) rather than importing the runtime
    registry: the check then needs no Qt runtime, and it still sees scenarios
    whose import is temporarily commented out in :func:`_import_scenarios`
    (their module files remain the source of truth). Only class-level
    ``name = "..."`` assignments count, so a ``name="..."`` keyword argument
    inside a builder call is not mistaken for a scenario name.

    Each scenario emits exactly one ``<name>.png`` — see
    ``scenarios/_base.py::Scenario.capture``. If a scenario ever grows multiple
    outputs, this scan (and the reference check built on it) must learn about
    them.
    """
    names: set[str] = set()
    for path in sorted(scenarios_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "name"
                        for target in stmt.targets
                    )
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    names.add(stmt.value.value)
    return names


def referenced_screenshot_names(docs_dir: Path = DOCS_DIR) -> dict[str, list[str]]:
    """Map each screenshot name referenced from the .rst sources to its locations."""
    references: dict[str, list[str]] = {}
    for path in sorted(docs_dir.rglob("*.rst")):
        if "_build" in path.parts or "_generated" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _SCREENSHOT_REF_RE.finditer(line):
                references.setdefault(match.group(1), []).append(f"{path}:{lineno}")
    return references


def _oversized_paths(paths: list[Path], budget_bytes: int) -> list[str]:
    """Return one ``"<name> (<size> KB)"`` entry per path over ``budget_bytes``.

    Shared by :func:`oversized_screenshots` (scans a directory) and
    :func:`main` (checks exactly the paths captured this run), so both report
    the same message shape and stay in sync if the format ever changes.
    """
    return [
        f"{path.name} ({path.stat().st_size / 1024:.1f} KB)"
        for path in paths
        if path.stat().st_size > budget_bytes
    ]


def oversized_screenshots(
    generated_dir: Path = GENERATED_SCREENSHOTS_DIR,
    budget_bytes: int = SCREENSHOT_SIZE_BUDGET_BYTES,
) -> list[str]:
    """Return one message per generated PNG that exceeds ``budget_bytes``.

    Silently returns ``[]`` when ``generated_dir`` does not exist: screenshots
    are gitignored build output, so ``structural`` (and this module's own
    tmp-path tests, which pass an unrelated ``docs_dir``) must stay green when
    no capture has ever run in that tree.
    """
    if not generated_dir.is_dir():
        return []

    oversized = _oversized_paths(sorted(generated_dir.glob("*.png")), budget_bytes)
    return [f"screenshot exceeds {budget_bytes // 1024} KB budget: {entry}" for entry in oversized]


def check_screenshot_references(
    docs_dir: Path = DOCS_DIR, scenarios_dir: Path = SCENARIOS_DIR
) -> list[str]:
    """Return screenshot-reference inconsistencies between docs and scenarios.

    Two failure modes, both reported: an .rst reference with no registered
    scenario would render as a permanently missing image, and a scenario never
    referenced from any .rst costs capture time on every full docs build for
    an image nobody embeds. A third check, the per-image size budget, is
    folded in here too so both ``--check-refs`` and the ``structural`` harness
    command report it for free; it looks for generated PNGs alongside
    ``docs_dir`` and is a no-op when that directory does not exist.
    """
    scenario_names = scenario_names_from_source(scenarios_dir)
    references = referenced_screenshot_names(docs_dir)

    problems: list[str] = []
    for name in sorted(set(references) - scenario_names):
        locations = ", ".join(references[name])
        problems.append(f"referenced screenshot has no registered scenario: {name} ({locations})")
    for name in sorted(scenario_names - set(references)):
        problems.append(f"registered scenario is never referenced from any .rst: {name}")
    problems.extend(oversized_screenshots(docs_dir / "_generated" / "screenshots"))
    return problems


def run_reference_check() -> int:
    """CLI wrapper for ``--check-refs``: print problems and return an exit code."""
    problems = check_screenshot_references()
    if problems:
        print("screenshot references: failed", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("screenshot references: ok")
    return 0


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
        alc_field_scan,
        alc_scan_exclusion,
        alpha_calibration_dialog,
        alpha_count_calibration,
        apodisation_comparison,
        batch_tab_group_binding,
        bunching_comparison,
        composite_fractions_dialog,
        composite_models_builder,
        data_browser_filter,
        data_browser_groups,
        data_processing_rebin,
        emu_longitudinal_layout,
        euo_fit_oscillatory,
        fit_asymmetric_errors,
        fit_wizard_gkt,
        fit_wizard_result,
        fourier_tf,
        global_fit_lfkt,
        global_fit_wizard_result,
        global_fit_wizard_running,
        global_fit_wizard_setup,
        grouped_fit_ybco_knight,
        grouping_window_profile_editor,
        hifi_transverse_layout,
        knight_shift_window,
        lf_kt_global_results,
        lf_kt_series_plot,
        logbook_view,
        main_window,
        maxent_ybco,
        mgb2_lambda_t,
        muon_fluorine_pbf2,
        new_user_function_dialog,
        parameter_trending_mgb2,
        period_mapping_dialog,
        quickstart_first_fit,
        run_info_provenance,
        simulate_dialog,
        spectral_moments_readout,
        suggest_next_point,
        temperature_trend_fit,
        trend_model_fit_dialog,
        vector_polarization_emu,
        waterfall_overlay,
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
        "--check-refs",
        action="store_true",
        help=(
            "Check that every docs/**/*.rst screenshot reference has a "
            "registered scenario and vice versa, then exit (no Qt needed)."
        ),
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
    args = _parse_args(argv)

    # Pure static check: no watchdog, no QApplication, no scenario imports.
    if args.check_refs:
        return run_reference_check()

    _start_watchdog()
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
        skipped = [name for name, scenario in selected.items() if scenario.requires_fit]
        for name in skipped:
            del selected[name]
            print(f"[screenshots] skipping {name} (requires_fit=True)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = CaptureContext(output_dir=args.out, device_pixel_ratio=args.dpr)

    captured_paths: list[Path] = []
    failed: list[str] = []
    for name, scenario in selected.items():
        print(f"[screenshots] capturing {name}...", flush=True)
        _t0 = time.monotonic()
        # One broken scenario must not blank out every scenario after it: a
        # raised exception here used to abort the whole run, and with
        # `continue-on-error: true` on the CI step the deploy then silently
        # published a mostly-imageless site. Capture the traceback, carry on,
        # and fail the run at the end instead.
        try:
            path = scenario.capture(ctx)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"[screenshots] FAILED {name}", file=sys.stderr, flush=True)
            failed.append(name)
            continue
        _elapsed = time.monotonic() - _t0
        print(f"[screenshots] wrote {path} ({_elapsed:.1f}s)", flush=True)
        captured_paths.append(path)

    if failed:
        print(
            f"[screenshots] {len(failed)} scenario(s) failed: {', '.join(failed)}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    oversized = _oversized_paths(captured_paths, SCREENSHOT_SIZE_BUDGET_BYTES)
    if oversized:
        budget_kb = SCREENSHOT_SIZE_BUDGET_BYTES // 1024
        print(f"[screenshots] {len(oversized)} PNG(s) exceed the {budget_kb} KB budget:", file=sys.stderr)
        for entry in oversized:
            print(f"  - {entry}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
