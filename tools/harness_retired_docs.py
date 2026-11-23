"""Retired planning locations and their existing shrinking path baseline."""

import posixpath
import re
from pathlib import Path

from harness_checks import HarnessFailure

ROOT = Path(__file__).resolve().parents[1]
RETIRED_DOC_PATHS = (
    "docs/plans",
    "docs/porting",
    "docs/studies",
    "docs/audit",
    "docs/investigations",
    "docs/PLANS.md",
)
RETIRED_DOCS_BASELINE = Path("tools/baselines/retired_docs.txt")


def find_retired_doc_violations(root: Path = ROOT) -> list[HarnessFailure]:
    """Reject new historical plans and require deletions to shrink their baseline."""

    baseline = root / RETIRED_DOCS_BASELINE
    if not baseline.is_file():
        return [HarnessFailure(baseline, 0, "Missing retired-docs baseline.")]
    failures: list[HarnessFailure] = []
    entries: set[str] = set()
    previous = ""
    for line, relative in enumerate(baseline.read_text(encoding="utf-8").splitlines(), 1):
        path = Path(relative)
        if (
            not relative
            or relative != relative.strip()
            or path.is_absolute()
            or "\\" in relative
            or ".." in path.parts
            or path.as_posix() != relative
            or not any(relative == p or relative.startswith(f"{p}/") for p in RETIRED_DOC_PATHS)
            or relative <= previous
        ):
            failures.append(
                HarnessFailure(
                    baseline,
                    line,
                    "Malformed retired-docs baseline entry; use strictly sorted, unique repo-relative POSIX file paths in retired locations.",
                )
            )
        else:
            entries.add(relative)
            if not (root / relative).is_file():
                failures.append(
                    HarnessFailure(
                        baseline,
                        line,
                        f"Retired file {relative} is gone; delete line {line} from {RETIRED_DOCS_BASELINE} so it cannot return.",
                    )
                )
        previous = relative
    for relative in RETIRED_DOC_PATHS:
        location = root / relative
        paths = [location] if location.is_file() else sorted(location.rglob("*"))
        for path in paths:
            parts = path.relative_to(root).parts
            if not path.is_file() or any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            if path.relative_to(root).as_posix() not in entries:
                failures.append(
                    HarnessFailure(
                        path,
                        0,
                        "New file in a retired planning location; plans go in a GitHub issue, lasting content in the owning Sphinx page, a regression test, or an invariant comment.",
                    )
                )
    return failures + find_retired_doc_links(root)


def find_retired_doc_links(root: Path) -> list[HarnessFailure]:
    """Reject explicit local Markdown/RST targets, not prose or historical URLs."""
    targets = re.compile(
        r"\]\(([^)\s]+)(?:\s+[^)]*)?\)|^\s*\[[^\]]+\]:\s*(\S+)"
        r"|<([^<>\s]+)>|^\s*\.\.\s+(?:include|literalinclude)::\s*(\S+)"
        r"|:doc:`([^`]+)`"
    )
    paths = list(root.glob("*"))
    for directory in ("docs", "tools", "tests", "src", ".github", ".claude"):
        paths.extend((root / directory).rglob("*"))
    failures = []
    for path in paths:
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.suffix not in {".md", ".rst"}
            or any(
                (p.startswith(".") and p not in {".github", ".claude"})
                or p in {"_build", "_generated", "__pycache__"}
                for p in relative.parts
            )
        ):
            continue
        for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in targets.finditer(text):
                target = next(value for value in match.groups() if value is not None)
                target = target.rsplit("<", 1)[-1].rstrip(">")
                if re.match(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//|#)", target):
                    continue
                target = target.split("#", 1)[0]
                parent = relative.parent.as_posix()
                if target.startswith("/"):
                    parent = "docs" if match.group(4) or match.group(5) else "."
                resolved = posixpath.normpath(posixpath.join(parent, target.lstrip("/")))
                if match.group(5) and resolved == "docs/PLANS":
                    resolved += ".md"
                if any(resolved == p or resolved.startswith(p + "/") for p in RETIRED_DOC_PATHS):
                    failures.append(
                        HarnessFailure(
                            path,
                            line,
                            "Local link targets retired planning material; use its maintained owner or issue.",
                        )
                    )
    return failures
