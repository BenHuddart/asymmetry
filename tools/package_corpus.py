"""Package the WiMDA muon-school corpus for CI consumption.

Builds a ``wimda-corpus-<version>.tar.zst`` containing everything the corpus
screenshot scenarios need — instrument data files, logbooks, ground-truth
documents, grouping presets, and the small WiMDA reference outputs — while
excluding what they don't (papers, guides, format-duplicate copies, macOS
AppleDouble litter, the WiMDA installation).

Usage::

    python tools/package_corpus.py --corpus "<corpus root>" --version 2026.07.16
    # → dist/wimda-corpus-2026.07.16.tar.zst (+ .sha256)

The archive extracts to a single ``wimda-corpus/`` directory; point
``ASYMMETRY_CORPUS_ROOT`` at it. Publish as a GitHub release asset on the
corpus repository (see docs/screenshots/scenarios/corpus/README.md).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

# Directory subtrees that no scenario reads.
EXCLUDE_DIRS = {
    "wimda installation",
    "_sync",
    "_findings",
    "nxs4to5",
    ".git",
}

# File patterns that no scenario reads. The TRSB ``.nxs_v2``/``.RAW`` copies
# duplicate the loadable HDF4 ``.nxs`` files; papers and guides are for humans.
EXCLUDE_PATTERNS = [
    "._*",  # macOS AppleDouble
    ".DS_Store",
    "desktop.ini",
    "*.docx",
    "*.pdf",
    "*.pptx",
    "*.nxs_v2",
    "*.RAW",
    "*.log",  # per-run ICP text logs (TRSB ships 400 of them)
    "ICP*.txt",
    "Status*.txt",
    "~$*",
]


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    return any(fnmatch.fnmatch(rel.name, pat) for pat in EXCLUDE_PATTERNS)


def collect(corpus: Path) -> list[Path]:
    files = []
    for path in sorted(corpus.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(corpus)
        if _excluded(rel):
            continue
        files.append(rel)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus root directory.")
    parser.add_argument("--version", required=True, help="Version tag, e.g. 2026.07.16.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("dist"), help="Output directory (default: dist/)."
    )
    parser.add_argument(
        "--list-only", action="store_true", help="List selected files and total size, then exit."
    )
    args = parser.parse_args(argv)

    corpus = args.corpus.expanduser().resolve()
    if not corpus.is_dir():
        print(f"corpus root not found: {corpus}", file=sys.stderr)
        return 2

    files = collect(corpus)
    total = sum((corpus / rel).stat().st_size for rel in files)
    print(f"{len(files)} files selected, {total / 1e9:.2f} GB uncompressed")
    if args.list_only:
        for rel in files[:20]:
            print(f"  {rel}")
        print(f"  ... ({len(files)} total)")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"wimda-corpus-{args.version}.tar.zst"

    # Stream through zstd for speed and ratio; fall back to Python-only xz if
    # the zstd binary is unavailable. File names go via stdin (--files-from=-):
    # thousands of arguments would exceed ARG_MAX on the command line.
    if _have("zstd"):
        cmd = [
            "tar",
            "--create",
            f"--directory={corpus}",
            "--transform=s|^|wimda-corpus/|",
            "--use-compress-program=zstd -T0 -12",
            f"--file={out}",
            "--files-from=-",
        ]
        names = "\n".join(str(rel) for rel in files)
        subprocess.run(cmd, input=names, text=True, check=True)
    else:
        out = out.with_suffix("")  # .tar
        out = out.with_name(out.name + ".tar.xz")
        with tarfile.open(out, "w:xz") as tar:
            for rel in files:
                tar.add(corpus / rel, arcname=f"wimda-corpus/{rel}")

    digest = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    (out.parent / (out.name + ".sha256")).write_text(f"{sha}  {out.name}\n")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB), sha256 {sha[:16]}…")
    return 0


def _have(binary: str) -> bool:
    from shutil import which

    return which(binary) is not None


if __name__ == "__main__":
    raise SystemExit(main())
