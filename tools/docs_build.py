"""Run the shared, fresh Sphinx HTML gate without importing Sphinx in the harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_HTML_DIR = ROOT / "docs" / "_build" / "html"
DOCS_PAGE_SIZE_BUDGET_BYTES = 600_000


def oversized_docs_pages(
    html_dir: Path = DOCS_HTML_DIR, budget_bytes: int = DOCS_PAGE_SIZE_BUDGET_BYTES
) -> list[str]:
    """List over-budget HTML pages, largest first."""
    oversized = [(p, p.stat().st_size) for p in html_dir.rglob("*.html")]
    oversized.sort(key=lambda item: item[1], reverse=True)
    return [
        f"{p.relative_to(html_dir).as_posix()} ({size / 1000:.1f} KB)"
        for p, size in oversized
        if size > budget_bytes
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "publish"), required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--output-dir", type=Path, default=DOCS_HTML_DIR)
    args = parser.parse_args(argv)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-a",
            "-W",
            "--keep-going",
            "-N",
            "-D",
            f"docs_build_mode={args.mode}",
            "-d",
            str(args.output_dir.parent / "doctrees"),
            "-b",
            "html",
            str(args.source_dir),
            str(args.output_dir),
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode
    oversized = oversized_docs_pages(args.output_dir)
    for entry in oversized:
        print(
            f"[docs] page exceeds {DOCS_PAGE_SIZE_BUDGET_BYTES // 1000} KB: {entry}",
            file=sys.stderr,
        )
    return int(bool(oversized))


if __name__ == "__main__":
    raise SystemExit(main())
