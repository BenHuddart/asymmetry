"""Retired planning locations and their existing shrinking path baseline."""

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
    return failures
