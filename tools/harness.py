#!/usr/bin/env python3
"""Agent-friendly repository validation harness for Asymmetry."""

from __future__ import annotations

import argparse
import ast
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

CORE_IMPORT_BANS = ("PySide6", "matplotlib", "asymmetry.gui")
CORE_DEPENDENCY_BANS = ("PySide6", "matplotlib")
REQUIRED_KNOWLEDGE_FILES = (
    "AGENTS.md",
    "README.md",
    "CONTRIBUTING.md",
    "docs/INDEX.md",
    "docs/ARCHITECTURE.md",
    "docs/HARNESS.md",
    "docs/QUALITY.md",
    "docs/PLANS.md",
)
LINT_TARGETS = ("src", "tests", "tools")


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


def run_structural_checks() -> int:
    """Run fast structural checks that do not require third-party packages."""

    failures = [
        *find_knowledge_base_violations(),
        *find_dependency_boundary_violations(),
        *find_core_boundary_violations(),
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
    """Prefer the project venv when the harness is launched from another Python."""

    if os.environ.get("ASYMMETRY_HARNESS_NO_VENV") == "1":
        return

    venv_python = _preferred_venv_python()
    if venv_python is None:
        return

    venv_root = ROOT / ".venv"
    if Path(sys.prefix).resolve() == venv_root.resolve():
        return

    args = list(sys.argv[1:] if argv is None else argv)
    print(f"Re-executing harness with {venv_python}", file=sys.stderr)
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *args])


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


def cmd_test(args: argparse.Namespace) -> int:
    pytest_args = _strip_passthrough(list(args.pytest_args))
    return _run_command([sys.executable, "-m", "pytest", *pytest_args])


def cmd_docs(_args: argparse.Namespace) -> int:
    return _run_command([sys.executable, "-m", "sphinx", "-b", "html", "docs", "docs/_build/html"])


def cmd_examples(_args: argparse.Namespace) -> int:
    return _run_command([sys.executable, "examples/run_all.py"])


def cmd_gui_smoke(_args: argparse.Namespace) -> int:
    return _run_command([sys.executable, "-m", "asymmetry.gui.app", "--smoke-test"])


def cmd_structural(_args: argparse.Namespace) -> int:
    return run_structural_checks()


def cmd_validate(args: argparse.Namespace) -> int:
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
        help="Run structural checks, lint, and tests",
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
