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
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from pathlib import Path

from .capture import (
    DOCS_DIR,
    SCREENSHOT_SIZE_BUDGET_BYTES,
    _boot_qapplication,
    _ensure_offscreen_default,
    _oversized_paths,
    _parse_args,
    _start_watchdog,
    referenced_screenshot_names,
)

CORPUS_SCREENSHOTS_DIR = Path(__file__).resolve().parents[1] / "_generated" / "corpus_screenshots"

# Uniform placeholder dimensions (px) for --stubs renders.
STUB_SIZE = (600, 380)
STUB_BACKGROUND = (232, 232, 232)  # light grey
STUB_TEXT_COLOUR = (96, 96, 96)
_PNG_END = b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"


def _capture_receipt(path: Path) -> Path:
    return path.with_suffix(".sha256")


def write_capture_receipt(path: Path) -> None:
    """Record the bytes of a completed corpus capture for the Pages gate."""
    _capture_receipt(path).write_text(
        sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )


def corpus_capture_problem(path: Path) -> str | None:
    """Return the reason a referenced corpus PNG cannot be published."""
    from PIL import Image

    if not path.is_file():
        return f"Missing corpus capture: {path.name}"
    receipt = _capture_receipt(path)
    if not receipt.is_file():
        return f"No completed corpus capture: {path.name}"
    if path.stat().st_size > SCREENSHOT_SIZE_BUDGET_BYTES:
        return f"Corpus capture exceeds PNG size budget: {path.name}"
    content = path.read_bytes()
    if receipt.read_bytes().strip() != sha256(content).hexdigest().encode("ascii"):
        return f"Corpus capture checksum mismatch: {path.name}"
    if not content.endswith(_PNG_END):
        return f"Invalid corpus capture PNG: {path.name}"
    try:
        with Image.open(BytesIO(content), formats=["PNG"]) as image:
            image.verify()
        with Image.open(BytesIO(content), formats=["PNG"]) as image:
            image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        return f"Invalid corpus capture PNG: {path.name}"
    return None


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
        name.removeprefix("corpus_").replace("_", " "),
        fill=STUB_TEXT_COLOUR,
        anchor="mm",
    )
    draw.text(
        (width / 2, height / 2 + 24),
        "Example data capture unavailable",
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


def _import_corpus_scenarios(modules: list[str]) -> dict:
    """Import and return the scenarios registered by the selected modules."""
    from .scenarios._base import registered_scenarios

    imported = {
        import_module(f"{__package__}.scenarios.corpus.{module}").__name__ for module in modules
    }
    return {
        name: scenario
        for name, scenario in registered_scenarios().items()
        if name.startswith("corpus_") and type(scenario).__module__ in imported
    }


CORPUS_SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios" / "corpus"


def _static_scenarios_by_module(
    scenarios_dir: Path = CORPUS_SCENARIOS_DIR,
) -> dict[str, list[str]]:
    """Map each corpus scenario module to its declared scenario names.

    Static AST scan (same approach as ``capture.scenario_names_from_source``,
    but per-file so the module mapping survives): the scenario modules import
    PySide6 at module level, so the stubs and module-listing paths must never
    import them — CI's PR smoke has no Qt runtime libraries.
    """
    import ast

    by_module: dict[str, list[str]] = {}
    for path in sorted(scenarios_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        names: list[str] = []
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
                    and stmt.value.value.startswith("corpus_")
                ):
                    names.append(stmt.value.value)
        if names:
            by_module[path.stem] = sorted(names)
    return by_module


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # The shared parser owns the options common to both capture drivers.
    stubs = "--stubs" in raw_argv
    stubs_missing = "--stubs-missing" in raw_argv
    list_modules = "--list-modules" in raw_argv
    published = "--published" in raw_argv
    module_filter: str | None = None
    parsed_argv: list[str] = []
    skip_next = False
    for i, arg in enumerate(raw_argv):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--stubs", "--stubs-missing", "--list-modules", "--published"):
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
    by_module = _static_scenarios_by_module()
    required = (
        referenced_screenshot_names(DOCS_DIR, namespace="corpus_screenshots") if published else None
    )
    modules = [
        module
        for module, names in by_module.items()
        if required is None or any(name in required for name in names)
    ]
    if list_modules:
        for module in modules:
            print(module)
        return 0
    if args.check_refs:
        print("Use the Sphinx documentation gate to check corpus screenshot references.")
        return 2
    if args.out == Path("docs/_generated/screenshots"):
        args.out = Path("docs/_generated/corpus_screenshots")

    if stubs or stubs_missing:
        # Placeholder path: no corpus, no Qt, no fit backend, no imports — the
        # scenario names come from a static AST scan (the scenario modules
        # import PySide6 at module level, which the PR smoke cannot satisfy).
        # ``--stubs-missing`` writes placeholders only where no PNG exists,
        # so a partially-successful capture sweep keeps its real renders.
        all_names = {n for names in by_module.values() for n in names}
        names = list(args.only) if args.only else sorted(all_names)
        unknown = [n for n in names if n not in all_names]
        if unknown:
            print(f"Unknown corpus scenarios: {', '.join(unknown)}", file=sys.stderr)
            return 2
        if stubs_missing:
            names = [n for n in names if not (args.out / f"{n}.png").exists()]
            if not names:
                print("[corpus-screenshots] no missing renders; nothing to stub", flush=True)
                return 0
        return _write_stubs(names, args.out)

    if module_filter is not None:
        if module_filter not in by_module:
            print(f"Unknown corpus module: {module_filter}", file=sys.stderr)
            print(f"Known: {', '.join(by_module)}", file=sys.stderr)
            return 2
        modules = [module for module in modules if module == module_filter]
    elif args.only:
        modules = [module for module in modules if set(by_module[module]).intersection(args.only)]

    _start_watchdog()
    _ensure_offscreen_default()
    _boot_qapplication()

    from .scenarios._base import CaptureContext, auto_dismiss_modals

    scenarios = _import_corpus_scenarios(modules)

    if args.list:
        for name, scenario in scenarios.items():
            description = scenario.description or scenario.__class__.__doc__ or ""
            first = description.strip().splitlines()[0] if description else ""
            print(f"{name}\t{first}")
        return 0

    if args.only and module_filter is None:
        unknown = [name for name in args.only if name not in scenarios]
        if unknown:
            print(f"Unknown corpus scenarios: {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected = {name: scenarios[name] for name in args.only}
    else:
        selected = scenarios

    if required is not None:
        selected = {name: scenario for name, scenario in selected.items() if name in required}

    if args.skip_fits:
        for name in [n for n, s in selected.items() if s.requires_fit]:
            del selected[name]
            print(f"[corpus-screenshots] skipping {name} (requires_fit=True)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = CaptureContext(output_dir=args.out, device_pixel_ratio=args.dpr)

    captured: list[Path] = []
    failed: list[str] = []
    # Same guard as the main capture driver: offscreen, an unanswerable modal
    # would wedge the run and blank every scenario after it.
    with auto_dismiss_modals():
        for name, scenario in selected.items():
            print(f"[corpus-screenshots] capturing {name}...", flush=True)
            t0 = time.monotonic()
            _capture_receipt(args.out / f"{name}.png").unlink(missing_ok=True)
            try:
                path = scenario.capture(ctx)
            except Exception:
                import traceback

                traceback.print_exc()
                print(f"[corpus-screenshots] FAILED {name}", file=sys.stderr, flush=True)
                failed.append(name)
                continue
            write_capture_receipt(path)
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
