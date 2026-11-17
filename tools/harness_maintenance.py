"""Read-only maintenance ratchets; Git is confined to this repository boundary."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from harness_checks import HarnessFailure
from owned_probes import Ownership

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "tools/baselines/maintenance.json"
# Only this immutable pre-rollout commit may supply an absent initial baseline.
BOOTSTRAP_BASE = "606992f5a4f6de2a3781f60c3e5444ffbaae6069"
OWNED_ROOTS = frozenset({"src", "tests", "tools", "docs", "examples", "packaging", "scripts"})
EXCLUDED = frozenset(
    {
        "build",
        "dist",
        "_build",
        "_generated",
        "__pycache__",
        "node_modules",
        "site-packages",
        "vendor",
        "vendored",
    }
)


def owned_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        bool(parts)
        and path == PurePosixPath(path).as_posix()
        and "\\" not in path
        and ".." not in parts
        and not path.startswith("/")
        and path.endswith(".py")
        and (len(parts) == 1 or parts[0] in OWNED_ROOTS)
        and not any(
            p.startswith(".") or p in EXCLUDED or p in {"venv", "env"} or p.endswith(".egg-info")
            for p in parts
        )
    )


def physical_lines(source: bytes) -> int:
    """LF-delimited physical lines, including a nonempty unterminated final line."""
    return source.count(b"\n") + bool(source and not source.endswith(b"\n"))


def current_sources(root: Path) -> dict[str, bytes]:
    sources = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        here = Path(directory)
        dirs[:] = sorted(
            d
            for d in dirs
            if not (here / d).is_symlink()
            and not d.startswith(".")
            and d not in EXCLUDED | {"venv", "env"}
            and not d.endswith(".egg-info")
            and not (here / d / "pyvenv.cfg").is_file()
            and (here != root or d in OWNED_ROOTS)
        )
        for name in sorted(files):
            path = here / name
            relative = path.relative_to(root).as_posix()
            if not path.is_symlink() and owned_path(relative):
                sources[relative] = path.read_bytes()
    return sources


def git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], input=input_bytes, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(
            f"Git base check failed: {result.stderr.decode('utf-8', errors='replace').strip()}; provide an available immutable --base commit (no automatic fetch)."
        )
    return result.stdout


def base_tree(root: Path, requested: str) -> tuple[str, dict[str, bytes], bytes | None, set[str]]:
    if not requested or requested.startswith("-"):
        raise ValueError("Requested --base must name an available commit.")
    commit = git(root, "rev-parse", "--verify", f"{requested}^{{commit}}").decode().strip()
    entries = {}
    environments = set()
    for record in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        path = os.fsdecode(raw_path)
        if mode in {b"100644", b"100755"} and kind == b"blob":
            entries[path] = object_id
            if path.endswith("/pyvenv.cfg"):
                environments.add(path.removesuffix("/pyvenv.cfg"))
    entries = {
        path: oid
        for path, oid in entries.items()
        if (owned_path(path) or path == BASELINE)
        and not any(path.startswith(f"{directory}/") for directory in environments)
    }
    # Read raw Git blobs: archive export-ignore/export-subst attributes must not
    # change the compared physical source or hide debt from the reviewed base.
    objects = list(dict.fromkeys(entries.values()))
    output = git(root, "cat-file", "--batch", input_bytes=b"".join(oid + b"\n" for oid in objects))
    blobs = {}
    offset = 0
    for oid in objects:
        end = output.index(b"\n", offset)
        actual_oid, kind, size = output[offset:end].split()
        length = int(size)
        if (
            actual_oid != oid
            or kind != b"blob"
            or output[end + 1 + length : end + 2 + length] != b"\n"
        ):
            raise ValueError("Git base check returned an invalid blob frame.")
        blobs[oid] = output[end + 1 : end + 1 + length]
        offset = end + 2 + length
    if offset != len(output):
        raise ValueError("Git base check returned trailing blob data.")
    sources = {path: blobs[oid] for path, oid in entries.items() if path != BASELINE}
    baseline = blobs[entries[BASELINE]] if BASELINE in entries else None
    # Index changes count as touches too, even if an unstaged edit restores base bytes.
    touched = set()
    for mode in (("--cached",), ()):
        touched.update(
            os.fsdecode(p)
            for p in git(root, "diff", *mode, "--name-only", "-z", commit, "--").split(b"\0")
            if p
        )
    return commit, sources, baseline, touched


def ci_base(event_name: str, event: dict) -> str:
    """Select the immutable event base, including GitHub's synthetic merge checkout."""
    if not isinstance(event, dict):
        raise ValueError("CI event must be a JSON object.")
    match event_name, event:
        case "pull_request", {"pull_request": {"base": {"sha": base}}}:
            pass
        case "push", {"before": base}:
            pass
        case "workflow_dispatch", {"inputs": {"structural_base": base}}:
            pass
        case _:
            raise ValueError(f"Missing or unsupported structural CI event base: {event_name!r}.")
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base) or base == "0" * 40:
        raise ValueError(
            "CI requires an immutable 40-character structural base SHA in the event; dispatch requires structural_base."
        )
    return base


def requested_base(base: str | None, ci: bool, snapshot: bool) -> str | None:
    if ci:
        if base is not None or snapshot:
            raise ValueError(
                "--ci selects the event base; do not combine it with --base or --snapshot."
            )
        try:
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
            return ci_base(os.environ["GITHUB_EVENT_NAME"], event)
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read required CI event base: {exc}") from exc
    if snapshot:
        if base is not None:
            raise ValueError("Choose --base or --snapshot, not both.")
        return None
    return base if base is not None else os.environ.get("ASYMMETRY_STRUCTURAL_BASE")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate baseline key {name!r}.")
        result[name] = value
    return result


def read_baseline(source: bytes) -> dict:
    try:
        value = json.loads(source, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Malformed maintenance baseline: {exc}.") from exc
    if not isinstance(value, dict) or set(value) != {"sizes", "probes"}:
        raise ValueError(
            "Malformed maintenance baseline: require exactly sizes and probes objects."
        )
    for rule in ("sizes", "probes"):
        entries = value[rule]
        if not isinstance(entries, dict):
            raise ValueError(f"Malformed {rule} baseline: require an object.")
        for path, allowance in entries.items():
            if not owned_path(path):
                raise ValueError(f"Out-of-scope maintenance allowance {path!r}; remove it.")
            if rule == "sizes":
                if type(allowance) is not int or allowance < 1000:
                    raise ValueError(
                        f"Malformed module-size allowance for {path}; require an integer >= 1000."
                    )
            elif (
                not isinstance(allowance, dict)
                or not allowance
                or any(
                    not isinstance(identity, str)
                    or not identity
                    or type(count) is not int
                    or count < 1
                    for identity, count in allowance.items()
                )
            ):
                raise ValueError(
                    f"Malformed owned-probe allowance for {path}; require nonempty semantic identities with positive integer multiplicities."
                )
    return value


def debt(sources: dict[str, bytes]) -> dict:
    return {
        "sizes": {
            path: count
            for path, source in sorted(sources.items())
            if (count := physical_lines(source)) >= 1000
        },
        "probes": {
            path: dict(sorted(counts.items()))
            for path, counts in sorted(Ownership(sources).probes().items())
        },
    }


def compare_debt(
    root: Path,
    current: dict,
    allowance: dict,
    *,
    base_debt: dict | None = None,
    base_allowance: dict | None = None,
    touched: set[str] = frozenset(),
) -> list[HarnessFailure]:
    """Pure rules on explicit snapshots; a path change is deletion plus new work."""
    failures = []
    for rule in ("sizes", "probes"):
        actual = current[rule]
        permitted = allowance[rule]
        for path in sorted(actual.keys() | permitted.keys()):
            value = actual.get(path, 0 if rule == "sizes" else {})
            limit = permitted.get(path, 0 if rule == "sizes" else {})
            label = "module-size" if rule == "sizes" else "owned-probe"
            if value != limit:
                details = [
                    (
                        f"{value} physical lines"
                        if path in actual
                        else "no oversized file (deleted or below 1000 lines)",
                        f"{limit} lines",
                    )
                ]
                if rule == "probes":
                    details = [
                        (
                            f"{identity}: {value.get(identity, 0)} occurrences",
                            f"{limit.get(identity, 0)} occurrences",
                        )
                        for identity in sorted(value.keys() | limit.keys())
                        if value.get(identity, 0) != limit.get(identity, 0)
                    ]
                for observed, ceiling in details:
                    failures.append(
                        HarnessFailure(
                            root / path,
                            0,
                            f"{label}: actual {observed}, frozen allowance {ceiling}; remove new debt, lower resolved allowances to the exact remaining debt, or delete stale entries."
                            + (
                                " New modules must have fewer than 1000 physical lines."
                                if rule == "sizes"
                                else " Use construction-guaranteed state directly or model optional state explicitly."
                            ),
                        )
                    )
            if base_debt is not None and base_allowance is not None:
                prior = base_allowance[rule].get(path, 0 if rule == "sizes" else {})
                increased = (
                    limit > prior if rule == "sizes" else bool(Counter(limit) - Counter(prior))
                )
                if increased:
                    failures.append(
                        HarnessFailure(
                            root / path,
                            0,
                            f"{label}: allowance added/raised/restored against the defined base; remove it. New paths, renames and copies receive no transferable allowance.",
                        )
                    )
        if rule == "sizes" and base_debt is not None:
            for path, count in base_debt[rule].items():
                if path in touched and path in current[rule] and current[rule][path] >= count:
                    failures.append(
                        HarnessFailure(
                            root / path,
                            0,
                            f"module-size: touched oversized base file ({count} lines) must shrink, including same-size/comment edits; extract or delete code and lower its ceiling.",
                        )
                    )
    return failures


def find_maintenance_violations(
    root: Path = ROOT, *, base: str | None = None, ci: bool = False, snapshot: bool = False
) -> list[HarnessFailure]:
    """Validate the final worktree, optionally against a required immutable Git base."""
    try:
        requested = requested_base(base, ci, snapshot)
        sources = current_sources(root)
        current = debt(sources)
        path = root / BASELINE
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "Missing regular maintenance baseline; restore it (checks never regenerate allowances)."
            )
        allowance = read_baseline(path.read_bytes())
        if requested is None:
            return compare_debt(root, current, allowance)
        commit, old_sources, old_bytes, staged = base_tree(root, requested)
        old_debt = debt(old_sources)
        if old_bytes is None:
            if commit != BOOTSTRAP_BASE:
                raise ValueError(
                    "Base has no maintenance baseline and is not the reviewed one-time bootstrap commit; choose a post-rollout base."
                )
            old_allowance = old_debt
        else:
            old_allowance = read_baseline(old_bytes)
            old_errors = compare_debt(root, old_debt, old_allowance)
            if old_errors:
                raise ValueError(
                    "Defined base contains stale maintenance allowances; select a valid reviewed base: "
                    + old_errors[0].message
                )
        touched = staged | {
            p for p in sources.keys() | old_sources.keys() if sources.get(p) != old_sources.get(p)
        }
        return compare_debt(
            root,
            current,
            allowance,
            base_debt=old_debt,
            base_allowance=old_allowance,
            touched=touched,
        )
    except (ValueError, OSError, SyntaxError) as exc:
        return [HarnessFailure(root / BASELINE, 0, f"maintenance: {exc}")]


if __name__ == "__main__":
    # The workflow fetches this validated immutable SHA before invoking the gate.
    # Local structural checks never perform network operations.
    try:
        print(requested_base(None, True, False))
    except ValueError as error:
        raise SystemExit(str(error)) from error
