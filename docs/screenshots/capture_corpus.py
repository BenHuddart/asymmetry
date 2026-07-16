"""CLI entry point for corpus-driven documentation screenshots.

Usage::

    .venv/bin/python -m docs.screenshots.capture_corpus --list
    .venv/bin/python -m docs.screenshots.capture_corpus --only corpus_euo_zf_fit
    .venv/bin/python -m docs.screenshots.capture_corpus            # all corpus scenarios
    .venv/bin/python -m docs.screenshots.capture_corpus --stubs    # placeholders, no corpus

Identical machinery to :mod:`docs.screenshots.capture` (offscreen Qt,
deterministic boot, watchdog, size budget), but imports only the scenarios in
``scenarios/corpus/`` and writes to ``docs/_generated/corpus_screenshots``.
Kept separate so the standard docs build never depends on the corpus being
present; the corpus root is resolved from ``ASYMMETRY_CORPUS_ROOT`` (see
``scenarios/corpus/_corpus.py``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .capture import (
    SCREENSHOT_SIZE_BUDGET_BYTES,
    _boot_qapplication,
    _ensure_offscreen_default,
    _oversized_paths,
    _parse_args,
    _start_watchdog,
)

CORPUS_SCREENSHOTS_DIR = Path(__file__).resolve().parents[1] / "_generated" / "corpus_screenshots"

# Uniform placeholder dimensions (px) for --stubs renders.
STUB_SIZE = (600, 380)
STUB_BACKGROUND = (232, 232, 232)  # light grey
STUB_TEXT_COLOUR = (96, 96, 96)


def _write_stub_png(path: Path, name: str) -> None:
    """Write a small uniform placeholder PNG with *name* centred.

    Used by ``--stubs`` so Sphinx builds that embed corpus renders never break
    when the corpus is absent (CI without ``ASYMMETRY_CORPUS_ROOT``). PIL only —
    no Qt, no corpus, no fit backend.
    """
    from PIL import Image, ImageDraw

    width, height = STUB_SIZE
    image = Image.new("RGB", (width, height), STUB_BACKGROUND)
    draw = ImageDraw.Draw(image)
    # A thin border makes the placeholder read as a deliberate frame, not a
    # broken/blank image, in the docs.
    draw.rectangle([0, 0, width - 1, height - 1], outline=(200, 200, 200))
    draw.text(
        (width / 2, height / 2),
        name,
        fill=STUB_TEXT_COLOUR,
        anchor="mm",
    )
    image.save(str(path), "PNG")


def _write_stubs(scenario_names: list[str], out_dir: Path) -> int:
    """Write one placeholder PNG per registered corpus scenario into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in scenario_names:
        path = out_dir / f"{name}.png"
        _write_stub_png(path, name)
        print(f"[corpus-screenshots] stub {path}", flush=True)
    print(f"[corpus-screenshots] wrote {len(scenario_names)} stub PNG(s)", flush=True)
    return 0


def _import_corpus_scenarios() -> dict:
    """Import and return the registered ``corpus_*`` scenarios (no Qt required)."""
    from . import scenarios as _scenarios_pkg  # noqa: F401  (registers _base)
    from .scenarios import corpus as corpus_pkg
    from .scenarios._base import registered_scenarios

    corpus_pkg.import_all_scenario_modules()
    return {name: s for name, s in registered_scenarios().items() if name.startswith("corpus_")}


def _module_of(scenario) -> str:
    """Return the scenario's defining module basename (e.g. ``euo_ordering``)."""
    return type(scenario).__module__.rsplit(".", 1)[-1]


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # ``--stubs``, ``--stubs-missing``, ``--list-modules`` and ``--module`` are
    # corpus-capture-only, so they are handled here rather than in the shared
    # ``_parse_args``.
    stubs = "--stubs" in raw_argv
    stubs_missing = "--stubs-missing" in raw_argv
    list_modules = "--list-modules" in raw_argv
    module_filter: str | None = None
    parsed_argv: list[str] = []
    skip_next = False
    for i, arg in enumerate(raw_argv):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--stubs", "--stubs-missing", "--list-modules"):
            continue
        if arg == "--module":
            if i + 1 >= len(raw_argv):
                print("--module requires a module name", file=sys.stderr)
                return 2
            module_filter = raw_argv[i + 1]
            skip_next = True
            continue
        parsed_argv.append(arg)

    args = _parse_args(parsed_argv)

    if list_modules:
        # One module name per line — CI iterates these to keep each capture
        # process inside the watchdog budget. No Qt boot needed.
        scenarios = _import_corpus_scenarios()
        for module in sorted({_module_of(s) for s in scenarios.values()}):
            print(module)
        return 0
    if args.check_refs:
        print("capture_corpus has no --check-refs (corpus scenarios are unpublished).")
        return 2
    if args.out == Path("docs/_generated/screenshots"):
        args.out = Path("docs/_generated/corpus_screenshots")

    if stubs or stubs_missing:
        # Placeholder path: no corpus, no Qt boot, no fit backend needed. The
        # scenario modules import cleanly (corpus data is resolved lazily at
        # capture time), so registration alone gives us every scenario name.
        # ``--stubs-missing`` writes placeholders only where no PNG exists,
        # so a partially-successful capture sweep keeps its real renders.
        scenarios = _import_corpus_scenarios()
        names = list(args.only) if args.only else list(scenarios)
        unknown = [n for n in names if n not in scenarios]
        if unknown:
            print(f"Unknown corpus scenarios: {', '.join(unknown)}", file=sys.stderr)
            return 2
        if stubs_missing:
            names = [n for n in names if not (args.out / f"{n}.png").exists()]
            if not names:
                print("[corpus-screenshots] no missing renders; nothing to stub", flush=True)
                return 0
        return _write_stubs(names, args.out)

    _start_watchdog()
    _ensure_offscreen_default()
    _boot_qapplication()

    from .scenarios._base import CaptureContext

    scenarios = _import_corpus_scenarios()

    if args.list:
        for name, scenario in scenarios.items():
            description = scenario.description or scenario.__class__.__doc__ or ""
            first = description.strip().splitlines()[0] if description else ""
            print(f"{name}\t{first}")
        return 0

    if module_filter is not None:
        selected = {name: s for name, s in scenarios.items() if _module_of(s) == module_filter}
        if not selected:
            known = sorted({_module_of(s) for s in scenarios.values()})
            print(f"Unknown corpus module: {module_filter}", file=sys.stderr)
            print(f"Known: {', '.join(known)}", file=sys.stderr)
            return 2
    elif args.only:
        unknown = [n for n in args.only if n not in scenarios]
        if unknown:
            print(f"Unknown corpus scenarios: {', '.join(unknown)}", file=sys.stderr)
            print(f"Known: {', '.join(scenarios)}", file=sys.stderr)
            return 2
        selected = {name: scenarios[name] for name in args.only}
    else:
        selected = scenarios

    if args.skip_fits:
        for name in [n for n, s in selected.items() if s.requires_fit]:
            del selected[name]
            print(f"[corpus-screenshots] skipping {name} (requires_fit=True)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = CaptureContext(output_dir=args.out, device_pixel_ratio=args.dpr)

    captured: list[Path] = []
    failed: list[str] = []
    for name, scenario in selected.items():
        print(f"[corpus-screenshots] capturing {name}...", flush=True)
        t0 = time.monotonic()
        try:
            path = scenario.capture(ctx)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"[corpus-screenshots] FAILED {name}", file=sys.stderr, flush=True)
            failed.append(name)
            continue
        print(f"[corpus-screenshots] wrote {path} ({time.monotonic() - t0:.1f}s)", flush=True)
        captured.append(path)

    if failed:
        print(
            f"[corpus-screenshots] {len(failed)} scenario(s) failed: {', '.join(failed)}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    oversized = _oversized_paths(captured, SCREENSHOT_SIZE_BUDGET_BYTES)
    if oversized:
        for entry in oversized:
            print(f"[corpus-screenshots] over budget: {entry}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
