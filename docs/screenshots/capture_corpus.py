"""CLI entry point for corpus-driven documentation screenshots.

Usage::

    .venv/bin/python -m docs.screenshots.capture_corpus --list
    .venv/bin/python -m docs.screenshots.capture_corpus --only corpus_euo_zf_fit
    .venv/bin/python -m docs.screenshots.capture_corpus            # all corpus scenarios

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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check_refs:
        print("capture_corpus has no --check-refs (corpus scenarios are unpublished).")
        return 2
    if args.out == Path("docs/_generated/screenshots"):
        args.out = Path("docs/_generated/corpus_screenshots")

    _start_watchdog()
    _ensure_offscreen_default()
    _boot_qapplication()

    from . import scenarios as _scenarios_pkg  # noqa: F401  (registers _base)
    from .scenarios import corpus as corpus_pkg
    from .scenarios._base import CaptureContext, registered_scenarios

    corpus_pkg.import_all_scenario_modules()
    scenarios = {
        name: s for name, s in registered_scenarios().items() if name.startswith("corpus_")
    }

    if args.list:
        for name, scenario in scenarios.items():
            description = scenario.description or scenario.__class__.__doc__ or ""
            first = description.strip().splitlines()[0] if description else ""
            print(f"{name}\t{first}")
        return 0

    if args.only:
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
