#!/usr/bin/env python3
"""Agent-friendly repository validation harness for Asymmetry."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "src" / "asymmetry" / "core"
GUI_ROOT = ROOT / "src" / "asymmetry" / "gui"
TESTS_ROOT = ROOT / "tests"

CORE_IMPORT_BANS = ("PySide6", "matplotlib", "asymmetry.gui")
CORE_DEPENDENCY_BANS = ("PySide6", "matplotlib")
# Canonical home for the shared numeric-limit line edit. Any other
# `class *LimitField` definition duplicates this foundation.
LIMIT_FIELD_HOME = GUI_ROOT / "widgets" / "axis_limits.py"
LIMIT_FIELD_CLASS_RE = re.compile(r"^\s*class\s+\w*LimitField\b")
# Canonical home for `FigureCanvasQTAgg` construction. Everything else should
# go through `asymmetry.gui.widgets.mpl_canvas.create_canvas`.
MPL_CANVAS_HOME = GUI_ROOT / "widgets" / "mpl_canvas.py"
# Pre-audit sites Phase 1b left out of scope for the `create_canvas` migration.
# They may migrate later -- see docs/audit/shared-foundations/FOLLOW-UPS.md.
MPL_CANVAS_CONSTRUCTION_ALLOWLIST = frozenset(
    {
        GUI_ROOT / "windows" / "fit_wizard_window.py",
        GUI_ROOT / "widgets" / "detector_schematic.py",
    }
)
# Canonical home for manual QThread lifecycles. Everything else in gui/ should
# run background work via `asymmetry.gui.tasks.TaskRunner`.
TASK_RUNNER_HOME = GUI_ROOT / "tasks.py"
# Sanctioned tests/ subpackages (the Phase-4 taxonomy plus pre-existing ones).
SANCTIONED_TEST_SUBPACKAGES = frozenset(
    {
        "core",
        "gui",
        "io",
        "project",
        "tools",
        "integration",
        "negmu",
        "docs",
        "porting",
    }
)
REQUIRED_KNOWLEDGE_FILES = (
    "AGENTS.md",
    "README.md",
    "CONTRIBUTING.md",
    "docs/INDEX.md",
    "docs/ARCHITECTURE.md",
    "docs/HARNESS.md",
    "docs/porting/README.md",
    "docs/porting/index.json",
    "docs/QUALITY.md",
    "docs/PLANS.md",
)
LINT_TARGETS = ("src", "tests", "tools")
PORTING_REQUIRED_STUDY_FILES = (
    "README.md",
    "comparison.md",
    "implementation-options.md",
    "test-data.md",
    "verification-plan.md",
)
PORTING_REQUIRED_CANDIDATE_FILES = (
    "README.md",
    "comparison.md",
    "scoring.md",
)
# Subdirectories of docs/porting/ that hold reference or workflow docs, not study artifacts.
PORTING_NON_STUDY_DIRS = frozenset({"practical-workflows", "reference"})
PORTING_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class HarnessFailure:
    """A structural harness failure with a location and remediation hint."""

    path: Path
    line: int
    message: str

    def format(self) -> str:
        try:
            relpath = self.path.relative_to(ROOT)
        except ValueError:
            relpath = self.path
        location = f"{relpath}:{self.line}" if self.line else str(relpath)
        return f"{location}: {self.message}"


def _module_matches(module: str, banned: str) -> bool:
    return module == banned or module.startswith(f"{banned}.")


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def find_core_boundary_violations(core_root: Path = CORE_ROOT) -> list[HarnessFailure]:
    """Return imports that would make the core depend on GUI/runtime UI packages."""

    failures: list[HarnessFailure] = []
    for path in _iter_python_files(core_root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(
                HarnessFailure(path, exc.lineno or 0, f"Python syntax error: {exc.msg}")
            )
            continue

        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]

            for module in modules:
                for banned in CORE_IMPORT_BANS:
                    if _module_matches(module, banned):
                        failures.append(
                            HarnessFailure(
                                path,
                                getattr(node, "lineno", 0),
                                (
                                    f"`asymmetry.core` must not import `{module}`. "
                                    "Move UI or plotting behavior to `asymmetry.gui`."
                                ),
                            )
                        )
    return failures


def find_duplicate_limit_field_violations(gui_root: Path = GUI_ROOT) -> list[HarnessFailure]:
    """Return `class *LimitField` definitions outside the shared widget home."""

    failures: list[HarnessFailure] = []
    for path in _iter_python_files(gui_root):
        if path == LIMIT_FIELD_HOME:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if LIMIT_FIELD_CLASS_RE.match(line):
                failures.append(
                    HarnessFailure(
                        path,
                        lineno,
                        (
                            "Duplicate limit-field class definition. "
                            "Use `asymmetry.gui.widgets.axis_limits.FloatLimitField`."
                        ),
                    )
                )
    return failures


def find_duplicate_mpl_canvas_violations(gui_root: Path = GUI_ROOT) -> list[HarnessFailure]:
    """Return direct `FigureCanvasQTAgg(` construction outside the shared factory."""

    failures: list[HarnessFailure] = []
    for path in _iter_python_files(gui_root):
        if path == MPL_CANVAS_HOME or path in MPL_CANVAS_CONSTRUCTION_ALLOWLIST:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "FigureCanvasQTAgg(" in line:
                failures.append(
                    HarnessFailure(
                        path,
                        lineno,
                        (
                            "Direct `FigureCanvasQTAgg(` construction. "
                            "Use `asymmetry.gui.widgets.mpl_canvas.create_canvas`."
                        ),
                    )
                )
    return failures


def find_bespoke_qthread_violations(gui_root: Path = GUI_ROOT) -> list[HarnessFailure]:
    """Return manual `QThread(` construction outside the shared task runner."""

    failures: list[HarnessFailure] = []
    for path in _iter_python_files(gui_root):
        if path == TASK_RUNNER_HOME:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "QThread(" in line:
                failures.append(
                    HarnessFailure(
                        path,
                        lineno,
                        (
                            "Bespoke `QThread(` construction. "
                            "Run background work via `asymmetry.gui.tasks.TaskRunner`."
                        ),
                    )
                )
    return failures


def find_test_placement_violations(tests_root: Path = TESTS_ROOT) -> list[HarnessFailure]:
    """Return `test_*.py` files that live outside a sanctioned tests/ subpackage."""

    failures: list[HarnessFailure] = []
    if not tests_root.exists():
        return failures

    for path in sorted(tests_root.rglob("test_*.py")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue

        relative = path.relative_to(tests_root)
        parts = relative.parts
        # A sanctioned placement is `tests/<subpackage>/.../test_*.py` -- i.e. at
        # least one directory component between `tests/` and the file itself,
        # and that first component must be a sanctioned subpackage.
        if len(parts) < 2 or parts[0] not in SANCTIONED_TEST_SUBPACKAGES:
            failures.append(
                HarnessFailure(
                    path,
                    0,
                    (
                        "Test file is not under a sanctioned tests/ subpackage. "
                        "Place test files under a sanctioned tests/ subpackage "
                        "(see tests/README.md)."
                    ),
                )
            )
    return failures


def _requirement_name(requirement: str) -> str:
    head = re.split(r"[<>=!~;@ \[]", requirement, maxsplit=1)[0]
    return head.strip().lower().replace("_", "-")


def _project_dependencies(pyproject_path: Path) -> list[str]:
    text = pyproject_path.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        return list(data.get("project", {}).get("dependencies", []))

    dependencies: list[str] = []
    in_project = False
    in_dependencies = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            in_dependencies = False
            continue
        if in_project and stripped.startswith("dependencies"):
            in_dependencies = True
        if in_dependencies:
            dependencies.extend(re.findall(r'"([^"]+)"', line))
            if "]" in line:
                in_dependencies = False
    return dependencies


def find_dependency_boundary_violations(
    pyproject_path: Path = ROOT / "pyproject.toml",
) -> list[HarnessFailure]:
    """Return core dependency declarations that belong in optional GUI extras."""

    if not pyproject_path.exists():
        return [HarnessFailure(pyproject_path, 0, "Missing pyproject.toml")]

    dependencies = _project_dependencies(pyproject_path)
    banned_names = {_requirement_name(name) for name in CORE_DEPENDENCY_BANS}

    failures: list[HarnessFailure] = []
    for dependency in dependencies:
        if _requirement_name(dependency) in banned_names:
            failures.append(
                HarnessFailure(
                    pyproject_path,
                    0,
                    (
                        f"`{dependency}` is a GUI/runtime plotting dependency. "
                        "Keep it in an optional extra instead of core dependencies."
                    ),
                )
            )
    return failures


def find_knowledge_base_violations(root: Path = ROOT) -> list[HarnessFailure]:
    """Return missing agent-facing knowledge-base files."""

    failures: list[HarnessFailure] = []
    for relative in REQUIRED_KNOWLEDGE_FILES:
        path = root / relative
        if not path.is_file():
            failures.append(
                HarnessFailure(
                    path,
                    0,
                    "Missing required knowledge-base file referenced by the agent harness.",
                )
            )
    return failures


def find_porting_policy_violations(root: Path = ROOT) -> list[HarnessFailure]:
    """Return porting-study layout issues that break the study-first workflow."""

    failures: list[HarnessFailure] = []
    porting_root = root / "docs" / "porting"
    index_path = porting_root / "index.json"

    if not porting_root.is_dir():
        failures.append(
            HarnessFailure(
                porting_root,
                0,
                "Missing `docs/porting/` directory for study-first feature ports.",
            )
        )
        return failures

    if not index_path.is_file():
        failures.append(
            HarnessFailure(
                index_path,
                0,
                "Missing `docs/porting/index.json` machine-readable study index.",
            )
        )
        return failures

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(
            HarnessFailure(index_path, exc.lineno, f"Invalid JSON: {exc.msg}"),
        )
        return failures

    if not isinstance(index_data, dict):
        failures.append(
            HarnessFailure(index_path, 0, "Porting index must be a JSON object."),
        )
        return failures

    version = index_data.get("version")
    studies = index_data.get("studies")
    if not isinstance(version, int) or version < 1:
        failures.append(
            HarnessFailure(index_path, 0, "Porting index must define an integer `version` >= 1."),
        )
    if not isinstance(studies, list):
        failures.append(
            HarnessFailure(index_path, 0, "Porting index must define a `studies` array."),
        )
        return failures

    # Top-level dirs that hold documentation or candidate containers, not study artifacts.
    _skip_dirs = PORTING_NON_STUDY_DIRS | {"candidates"}
    study_dirs = {
        path.name: path
        for path in sorted(porting_root.iterdir())
        if path.is_dir() and path.name not in _skip_dirs
    }

    candidates_root = porting_root / "candidates"
    candidate_dirs = {
        path.name: path
        for path in (sorted(candidates_root.iterdir()) if candidates_root.is_dir() else [])
        if path.is_dir()
    }

    indexed_study_slugs: dict[str, dict[str, object]] = {}
    indexed_candidate_slugs: dict[str, dict[str, object]] = {}
    all_indexed_slugs: set[str] = set()

    for entry in studies:
        if not isinstance(entry, dict):
            failures.append(
                HarnessFailure(index_path, 0, "Each porting study entry must be a JSON object."),
            )
            continue

        slug = entry.get("slug")
        feature_name = entry.get("feature_name")
        status = entry.get("status")
        path_value = entry.get("path")
        references = entry.get("references")
        docs = entry.get("docs")

        if not isinstance(slug, str) or not PORTING_SLUG_RE.fullmatch(slug):
            failures.append(
                HarnessFailure(index_path, 0, "Each study entry needs a kebab-case `slug`."),
            )
            continue
        if slug in all_indexed_slugs:
            failures.append(
                HarnessFailure(index_path, 0, f"Duplicate porting study slug `{slug}` in index."),
            )
            continue
        all_indexed_slugs.add(slug)

        is_candidate = status == "candidate"
        if is_candidate:
            indexed_candidate_slugs[slug] = entry
        else:
            indexed_study_slugs[slug] = entry

        if not isinstance(feature_name, str) or not feature_name.strip():
            failures.append(
                HarnessFailure(
                    index_path, 0, f"Porting study `{slug}` needs a non-empty `feature_name`."
                ),
            )
        if not isinstance(status, str) or not status.strip():
            failures.append(
                HarnessFailure(
                    index_path, 0, f"Porting study `{slug}` needs a non-empty `status`."
                ),
            )

        if is_candidate:
            expected_path = f"docs/porting/candidates/{slug}"
            expected_docs: dict[str, str] = {
                "readme": f"{expected_path}/README.md",
                "comparison": f"{expected_path}/comparison.md",
                "scoring": f"{expected_path}/scoring.md",
            }
        else:
            expected_path = f"docs/porting/{slug}"
            expected_docs = {
                "readme": f"{expected_path}/README.md",
                "comparison": f"{expected_path}/comparison.md",
                "implementation_options": f"{expected_path}/implementation-options.md",
                "test_data": f"{expected_path}/test-data.md",
                "verification_plan": f"{expected_path}/verification-plan.md",
            }

        if path_value != expected_path:
            failures.append(
                HarnessFailure(
                    index_path, 0, f"Porting study `{slug}` must use `path: {expected_path}`."
                ),
            )
        if not isinstance(references, list) or not all(
            isinstance(item, str) for item in references
        ):
            failures.append(
                HarnessFailure(
                    index_path, 0, f"Porting study `{slug}` needs a string `references` list."
                ),
            )
        if not isinstance(docs, dict):
            failures.append(
                HarnessFailure(index_path, 0, f"Porting study `{slug}` needs a `docs` object."),
            )
            continue

        for key, expected_doc_path in expected_docs.items():
            if docs.get(key) != expected_doc_path:
                failures.append(
                    HarnessFailure(
                        index_path,
                        0,
                        f"Porting study `{slug}` must define `docs.{key}` as `{expected_doc_path}`.",
                    )
                )

    for slug, study_dir in study_dirs.items():
        if not PORTING_SLUG_RE.fullmatch(slug):
            failures.append(
                HarnessFailure(
                    study_dir,
                    0,
                    "Porting study directories must use kebab-case feature slugs.",
                )
            )

        for filename in PORTING_REQUIRED_STUDY_FILES:
            file_path = study_dir / filename
            if not file_path.is_file():
                failures.append(
                    HarnessFailure(
                        file_path,
                        0,
                        "Missing required study-pass artifact for feature port.",
                    )
                )

        if slug not in indexed_study_slugs:
            failures.append(
                HarnessFailure(
                    study_dir,
                    0,
                    "Porting study directory is missing from `docs/porting/index.json`.",
                )
            )

    for slug in indexed_study_slugs:
        if slug not in study_dirs:
            failures.append(
                HarnessFailure(
                    index_path,
                    0,
                    f"Porting index entry `{slug}` does not have a matching study directory.",
                )
            )

    for slug, candidate_dir in candidate_dirs.items():
        if not PORTING_SLUG_RE.fullmatch(slug):
            failures.append(
                HarnessFailure(
                    candidate_dir,
                    0,
                    "Porting candidate directories must use kebab-case feature slugs.",
                )
            )

        for filename in PORTING_REQUIRED_CANDIDATE_FILES:
            file_path = candidate_dir / filename
            if not file_path.is_file():
                failures.append(
                    HarnessFailure(
                        file_path,
                        0,
                        "Missing required candidate artifact for feature port.",
                    )
                )

        if slug not in indexed_candidate_slugs:
            failures.append(
                HarnessFailure(
                    candidate_dir,
                    0,
                    "Porting candidate directory is missing from `docs/porting/index.json`.",
                )
            )

    for slug in indexed_candidate_slugs:
        if slug not in candidate_dirs:
            failures.append(
                HarnessFailure(
                    index_path,
                    0,
                    f"Porting index candidate `{slug}` does not have a matching candidate directory.",
                )
            )

    return failures


def run_structural_checks() -> int:
    """Run fast structural checks that do not require third-party packages."""

    failures = [
        *find_knowledge_base_violations(),
        *find_porting_policy_violations(),
        *find_dependency_boundary_violations(),
        *find_core_boundary_violations(),
        *find_duplicate_limit_field_violations(),
        *find_duplicate_mpl_canvas_violations(),
        *find_bespoke_qthread_violations(),
        *find_test_placement_violations(),
    ]
    if not failures:
        print("structural: ok")
        return 0

    print("structural: failed")
    for failure in failures:
        print(f"- {failure.format()}")
    return 1


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    mpl_config_dir = Path(tempfile.gettempdir()) / "asymmetry-matplotlib"
    xdg_cache_dir = Path(tempfile.gettempdir()) / "asymmetry-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    env.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
    return env


def _preferred_venv_python(root: Path = ROOT) -> Path | None:
    candidates = (
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _maybe_reexec_with_venv(argv: Sequence[str] | None) -> None:
    """Prefer the project venv when the harness is launched from another Python.

    On POSIX ``os.execv`` replaces the current process, so the venv interpreter
    inherits the caller's exit-code contract for free. On Windows there is no
    real ``exec``: ``os.execv`` spawns a *child* and terminates the parent
    immediately with exit code 0, so a failing pytest run inside the child would
    be reported to the shell as success (observed: 70 failing tests, ``$LASTEXITCODE``
    still 0). To preserve the failure signal there we run the child synchronously
    and exit with its return code instead.
    """

    if os.environ.get("ASYMMETRY_HARNESS_NO_VENV") == "1":
        return

    venv_python = _preferred_venv_python()
    if venv_python is None:
        return

    venv_root = ROOT / ".venv"
    if Path(sys.prefix).resolve() == venv_root.resolve():
        return

    args = list(sys.argv[1:] if argv is None else argv)
    child_argv = [str(venv_python), str(Path(__file__).resolve()), *args]
    print(f"Re-executing harness with {venv_python}", file=sys.stderr)
    if os.name == "nt":
        completed = subprocess.run(child_argv, check=False)  # noqa: S603
        sys.exit(completed.returncode)
    os.execv(str(venv_python), child_argv)


def _run_command(args: Sequence[str]) -> int:
    print("+", " ".join(args))
    completed = subprocess.run(args, cwd=ROOT, env=_command_env(), check=False)
    return completed.returncode


def _strip_passthrough(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def cmd_lint(_args: argparse.Namespace) -> int:
    format_result = _run_command([sys.executable, "-m", "ruff", "format", "--check", *LINT_TARGETS])
    if format_result:
        return format_result
    return _run_command([sys.executable, "-m", "ruff", "check", *LINT_TARGETS])


def cmd_lint_all(_args: argparse.Namespace) -> int:
    return cmd_lint(_args)


_TIER_MARKER: dict[str, str | None] = {
    "fast": "unit and not slow and not gui and not io and not integration",
    "standard": "not slow and not integration",
    "full": None,
}

# CI shards the standard/full tier across two runners by Qt involvement: the GUI
# tests carry the per-test MainWindow construction cost, so splitting them onto
# their own runner roughly halves wall-clock. `all` is the default (no split).
_SUBSET_MARKER: dict[str, str | None] = {
    "all": None,
    "gui": "gui",
    "non-gui": "not gui",
}


def _has_explicit_targets(pytest_args: Sequence[str]) -> bool:
    """Return True if the caller named specific test files or node ids.

    When you point the harness at exact targets (``-- tests/test_x.py`` or a
    ``...::test_case`` node id) you want *those* tests to run, not a tier-marker
    subset silently deselecting half of them. We detect a positional that looks
    like a path or node id; option *values* (e.g. a ``-k`` expression) are not
    paths and so do not trip this.
    """
    for arg in pytest_args:
        # Skip option flags and `key=value` option values (e.g. `-o x=/tmp/c`):
        # a real test path or node id never contains "=", so this keeps an
        # option value that happens to look path-like from bypassing the marker.
        if arg.startswith("-") or "=" in arg:
            continue
        if arg.endswith(".py") or "::" in arg or "/" in arg or os.sep in arg:
            return True
        if (ROOT / arg).exists():
            return True
    return False


def build_pytest_command(
    pytest_args: Sequence[str],
    *,
    tier: str = "standard",
    subset: str = "all",
    parallel: bool = True,
    shard: str | None = None,
) -> list[str]:
    """Build the pytest argv for a tier/subset run (pure, so it is unit-tested).

    Composes the tier and subset markers into a single ``-m`` expression, unless
    the caller already passed ``-m`` or named explicit targets (in which case
    those run verbatim). xdist parallelism is added for every tier unless
    ``--no-parallel`` is given (the fast tier measured 71s serial vs 25s with
    ``-n auto``, so worker startup is worth it even there).

    ``--subset`` only shards the gui/non-gui split of the standard/full tiers; the
    ``fast`` tier is non-GUI by definition, so combining it with a subset is
    rejected rather than silently composing a contradictory marker (``fast`` +
    ``gui`` would select zero tests and still exit 0 — a false green).

    ``shard`` (``"K/N"``) is forwarded to pytest's ``--shard`` (a conftest option)
    to run a stable 1-of-N slice of the selection — used to split the GUI subset
    across several CI runners. It is appended after marker composition so its
    ``"K/N"`` value is never mistaken for a test target.
    """
    if tier == "fast" and subset != "all":
        raise ValueError(
            f"--subset {subset!r} is incompatible with --tier fast "
            "(the fast tier is already non-GUI); use --tier standard/full to shard."
        )

    pytest_args = list(pytest_args)

    user_marker = any(a == "-m" or a.startswith("-m") for a in pytest_args)
    if not user_marker and not _has_explicit_targets(pytest_args):
        marker_parts = [
            f"({expr})" for expr in (_TIER_MARKER[tier], _SUBSET_MARKER[subset]) if expr is not None
        ]
        if marker_parts:
            pytest_args = ["-m", " and ".join(marker_parts)] + pytest_args

    if shard:
        pytest_args = pytest_args + ["--shard", shard]

    parallel_args: list[str] = []
    if parallel:
        parallel_args = ["-n", "auto", "--dist", "load"]

    return [sys.executable, "-m", "pytest", *parallel_args, *pytest_args]


def cmd_test(args: argparse.Namespace) -> int:
    command = build_pytest_command(
        _strip_passthrough(list(args.pytest_args)),
        tier=getattr(args, "tier", "standard"),
        subset=getattr(args, "subset", "all"),
        parallel=not getattr(args, "no_parallel", False),
        shard=getattr(args, "shard", None),
    )
    return _run_command(command)


def cmd_docs(_args: argparse.Namespace) -> int:
    return _run_command(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-d",
            "docs/_build/doctrees",
            "-b",
            "html",
            "docs",
            "docs/_build/html",
        ]
    )


def cmd_examples(_args: argparse.Namespace) -> int:
    return _run_command([sys.executable, "examples/run_all.py"])


def cmd_gui_smoke(_args: argparse.Namespace) -> int:
    return _run_command([sys.executable, "-m", "asymmetry.gui.app", "--smoke-test"])


def cmd_structural(_args: argparse.Namespace) -> int:
    return run_structural_checks()


def cmd_validate(args: argparse.Namespace) -> int:
    if not hasattr(args, "tier"):
        args.tier = "standard"
    if not hasattr(args, "no_parallel"):
        args.no_parallel = False
    steps = [
        ("structural", cmd_structural),
        ("lint", cmd_lint),
        ("test", cmd_test),
    ]
    for name, command in steps:
        print(f"\n== {name} ==")
        result = command(args)
        if result:
            return result
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    structural_parser = subparsers.add_parser("structural", help="Run fast repository-shape checks")
    structural_parser.set_defaults(func=cmd_structural)

    lint_parser = subparsers.add_parser("lint", help="Run Ruff checks")
    lint_parser.set_defaults(func=cmd_lint)

    lint_all_parser = subparsers.add_parser("lint-all", help="Run Ruff checks across the repo")
    lint_all_parser.set_defaults(func=cmd_lint_all)

    test_parser = subparsers.add_parser("test", help="Run pytest, optionally with passthrough args")
    test_parser.add_argument(
        "--tier",
        choices=["fast", "standard", "full"],
        default="standard",
        help="Test tier: fast (<30s unit-only), standard (default, excludes slow/integration), full (everything)",
    )
    test_parser.add_argument(
        "--subset",
        choices=["all", "gui", "non-gui"],
        default="all",
        help="Restrict to GUI or non-GUI tests (composes with --tier; used to shard CI)",
    )
    test_parser.add_argument(
        "--shard",
        default=None,
        metavar="K/N",
        help="Run a stable 1-of-N slice of the selection (e.g. 1/3); splits the GUI subset across runners",
    )
    test_parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Disable pytest-xdist parallelization",
    )
    test_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test_parser.set_defaults(func=cmd_test)

    docs_parser = subparsers.add_parser("docs", help="Build Sphinx documentation")
    docs_parser.set_defaults(func=cmd_docs)

    examples_parser = subparsers.add_parser("examples", help="Run documentation examples")
    examples_parser.set_defaults(func=cmd_examples)

    gui_parser = subparsers.add_parser(
        "gui-smoke",
        help="Launch the GUI in headless smoke-test mode",
    )
    gui_parser.set_defaults(func=cmd_gui_smoke)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run structural checks, lint, and tests (standard tier by default)",
    )
    validate_parser.add_argument(
        "--tier",
        choices=["fast", "standard", "full"],
        default="standard",
        help="Test tier passed through to the test step",
    )
    validate_parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Disable pytest-xdist parallelization",
    )
    validate_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    validate_parser.set_defaults(func=cmd_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _maybe_reexec_with_venv(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
