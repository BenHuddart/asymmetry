"""Production-entry regressions for size and semantic-debt ratchets."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests._harness import _load_harness

BASELINE = Path("tools/baselines/maintenance.json")


def write(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def baseline(root: Path, sizes=None, probes=None) -> None:
    write(
        root,
        str(BASELINE),
        json.dumps({"sizes": sizes or {}, "probes": probes or {}}, indent=2) + "\n",
    )


def check(root: Path, base: str | None = None):
    return _load_harness().find_maintenance_violations(root, base=base)


@pytest.mark.parametrize("lines", [999, 1000, 1001])
def test_new_module_threshold(tmp_path: Path, lines: int) -> None:
    baseline(tmp_path)
    write(tmp_path, "src/new.py", "# line\n" * lines)
    failures = check(tmp_path)
    assert bool(failures) == (lines >= 1000)
    if failures:
        assert "module-size" in failures[0].message
        assert "src/new.py" in str(failures[0].path)


def test_owned_probe_is_rejected_at_harness_entry(tmp_path: Path) -> None:
    baseline(tmp_path)
    write(
        tmp_path,
        "src/panel.py",
        """class Panel:
    def __init__(self):
        self._datasets = []
    def update(self):
        return getattr(self, "_datasets", None)
""",
    )
    failures = check(tmp_path)
    assert any("owned-probe" in failure.message for failure in failures)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return result.stdout.decode().strip()


def commit(root: Path) -> str:
    git(root, "add", "--all")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return git(root, "rev-parse", "HEAD")


def freeze(root: Path) -> dict:
    rules = _load_harness()._maintenance
    value = rules.debt(rules.current_sources(root))
    write(root, str(BASELINE), json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    git(tmp_path, "init", "-q")
    write(tmp_path, "src/old module.py", "# old\n" * 1004)
    freeze(tmp_path)
    return tmp_path, commit(tmp_path)


@pytest.mark.parametrize(
    "source,expected",
    [(b"", 0), (b"\n", 1), (b"a", 1), (b"a\n", 1), (b"a\r\nb\r\n", 2), (b"a\rb", 1)],
)
def test_physical_line_definition(source: bytes, expected: int) -> None:
    assert _load_harness()._maintenance.physical_lines(source) == expected


def test_unchanged_old_debt_passes_and_never_mutates(repo) -> None:
    root, base = repo
    before = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    assert check(root, base) == []
    after = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    assert before == after
    assert git(root, "status", "--porcelain") == ""


@pytest.mark.parametrize("mode", ["unstaged", "staged", "committed"])
@pytest.mark.parametrize("lines", [1004, 1005])
def test_touched_old_file_must_shrink_even_with_raised_baseline(repo, mode, lines) -> None:
    root, base = repo
    write(root, "src/old module.py", "# edit\n" * lines)
    freeze(root)
    if mode == "staged":
        git(root, "add", "--all")
    if mode == "committed":
        commit(root)
    failures = check(root, base)
    assert any("must shrink" in f.message for f in failures)
    if lines > 1004:
        assert any("added/raised/restored" in f.message for f in failures)


def test_staged_touch_cannot_hide_behind_unstaged_base_bytes(repo) -> None:
    root, base = repo
    write(root, "src/old module.py", "# edit\n" * 1004)
    git(root, "add", "--all")
    write(root, "src/old module.py", "# old\n" * 1004)
    assert any("must shrink" in f.message for f in check(root, base))


@pytest.mark.parametrize("lines", [999, 1000, 1003])
def test_shrink_requires_exact_lowered_allowance(repo, lines) -> None:
    root, base = repo
    write(root, "src/old module.py", "# old\n" * lines)
    assert check(root, base)
    freeze(root)
    assert check(root, base) == []


def test_deleted_debt_requires_allowance_removal(repo) -> None:
    root, base = repo
    (root / "src/old module.py").unlink()
    assert check(root, base)
    freeze(root)
    assert check(root, base) == []


@pytest.mark.parametrize("staged", [False, True])
def test_new_oversized_file_cannot_be_laundered(repo, staged) -> None:
    root, base = repo
    write(root, "tools/new file.py", "# new\n" * 1000)
    freeze(root)
    if staged:
        git(root, "add", "--all")
    assert any("added/raised/restored" in f.message for f in check(root, base))


@pytest.mark.parametrize("operation", ["rename", "copy"])
def test_rename_copy_do_not_transfer_allowances(repo, operation) -> None:
    root, base = repo
    write(root, "src/new name.py", (root / "src/old module.py").read_text())
    if operation == "rename":
        (root / "src/old module.py").unlink()
    freeze(root)
    assert any("New paths, renames and copies" in f.message for f in check(root, base))
    write(root, "src/new name.py", "# small\n")
    freeze(root)
    assert check(root, base) == []


@pytest.mark.parametrize("delete_baseline", [False, True])
def test_size_removal_then_restoration_fails_across_commits(repo, delete_baseline) -> None:
    root, _ = repo
    write(root, "src/old module.py", "# small\n")
    freeze(root)
    if delete_baseline:
        (root / BASELINE).unlink()
    base = commit(root)
    write(root, "src/old module.py", "# old\n" * 1004)
    freeze(root)
    failures = check(root, base)
    assert failures
    assert any(
        "not the reviewed one-time" in f.message
        if delete_baseline
        else "added/raised/restored" in f.message
        for f in failures
    )


PROBE = """class Panel:
    first: int
    second: int
    def inspect(self):
        return getattr(self, "first", None)
"""


@pytest.mark.parametrize("update_allowance", [False, True])
def test_same_count_probe_substitution_fails(tmp_path: Path, update_allowance: bool) -> None:
    git(tmp_path, "init", "-q")
    path = write(tmp_path, "src/panel.py", PROBE)
    freeze(tmp_path)
    base = commit(tmp_path)
    path.write_text(PROBE.replace('"first"', '"second"'))
    if update_allowance:
        freeze(tmp_path)
    assert any("owned-probe" in f.message for f in check(tmp_path, base))


def test_probe_removal_then_reintroduction_fails_across_commits(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    path = write(tmp_path, "src/panel.py", PROBE)
    freeze(tmp_path)
    initial = commit(tmp_path)
    path.write_text(PROBE.replace('getattr(self, "first", None)', "self.first"))
    freeze(tmp_path)
    assert check(tmp_path, initial) == []
    removed = commit(tmp_path)
    path.write_text(PROBE)
    freeze(tmp_path)
    assert any("added/raised/restored" in f.message for f in check(tmp_path, removed))


def test_probe_identity_survives_line_shifts_but_not_multiplicity(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    path = write(tmp_path, "src/panel.py", PROBE)
    freeze(tmp_path)
    base = commit(tmp_path)
    path.write_text("# unrelated\n\n" + PROBE)
    assert check(tmp_path, base) == []
    path.write_text(
        PROBE.replace("return getattr", 'getattr(self, "first", None)\n        return getattr')
    )
    assert check(tmp_path, base)


@pytest.mark.parametrize(
    "entry",
    [
        '{"sizes":{},"sizes":{},"probes":{}}',
        '{"sizes":{"src/a.py":true},"probes":{}}',
        '{"sizes":{"src/a.py":999},"probes":{}}',
        '{"sizes":{"../a.py":1000},"probes":{}}',
        '{"sizes":{"other/a.py":1000},"probes":{}}',
        '{"sizes":{"src//a.py":1000},"probes":{}}',
        '{"sizes":{},"probes":{"src/a.py":{}}}',
        '{"sizes":{},"probes":{"src/a.py":{"id":0}}}',
        '{"sizes":{},"probes":{"src/a.py":{"id":1,"id":1}}}',
        '{"sizes":{},"probes":{},"reset":true}',
        "[]",
        "invalid",
    ],
)
def test_malformed_and_out_of_scope_allowances_fail(tmp_path: Path, entry: str) -> None:
    write(tmp_path, str(BASELINE), entry)
    assert check(tmp_path)


def test_stale_size_or_probe_entries_fail(tmp_path: Path) -> None:
    baseline(tmp_path, {"src/gone.py": 1004}, {"src/gone.py": {"old identity": 1}})
    assert len(check(tmp_path)) == 2


def test_missing_requested_base_fails_instead_of_snapshot(repo) -> None:
    root, _ = repo
    assert any("Git base check failed" in f.message for f in check(root, "missing-commit"))


def test_empty_requested_base_does_not_silently_skip(repo) -> None:
    root, _ = repo
    assert check(root, "")


def test_wrong_bootstrap_commit_fails(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    write(tmp_path, "src/old.py", "# old\n" * 1001)
    base = commit(tmp_path)
    freeze(tmp_path)
    assert any("not the reviewed one-time" in f.message for f in check(tmp_path, base))


def test_one_time_bootstrap_uses_base_debt_not_candidate_growth(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    write(tmp_path, "src/old.py", "# old\n" * 1001)
    base = commit(tmp_path)
    freeze(tmp_path)
    harness = _load_harness()
    harness._maintenance.BOOTSTRAP_BASE = base
    assert harness.find_maintenance_violations(tmp_path, base=base) == []
    write(tmp_path, "src/new.py", "# new\n" * 1001)
    freeze(tmp_path)
    assert any(
        "added/raised/restored" in f.message
        for f in harness.find_maintenance_violations(tmp_path, base=base)
    )


def test_scope_covers_new_local_files_and_ignores_external_symlinks(tmp_path: Path) -> None:
    baseline(tmp_path)
    for root in (
        "src",
        "tools",
        "tests",
        "docs",
        "examples",
        "scripts",
        "packaging",
        ".github",
        ".claude/hooks",
    ):
        write(tmp_path, f"{root}/new.py", "# new\n" * 1000)
    write(tmp_path, "new.py", "# new\n" * 1000)
    external = write(tmp_path, "external/lib.py", "# external\n" * 1200)
    for path in ("docs/_build/generated.py", "tools/.venv/lib.py", "src/vendor/lib.py"):
        write(tmp_path, path, "# external\n" * 1200)
    (tmp_path / "src/dependency").symlink_to(external.parent, target_is_directory=True)
    (tmp_path / "src/linked.py").symlink_to(external)
    assert len(check(tmp_path)) == 10


def test_shallow_clone_missing_base_then_available_base(tmp_path: Path, repo) -> None:
    root, base = repo
    write(root, "README", "second\n")
    commit(root)
    clone = tmp_path / "shallow"
    subprocess.run(["git", "clone", "--quiet", "--depth=1", root.as_uri(), str(clone)], check=True)
    assert check(clone, base)
    git(clone, "fetch", "--quiet", "--unshallow")
    assert check(clone, base) == []


def test_merge_checkout_compares_event_base_not_head_parent(repo) -> None:
    root, base = repo
    git(root, "checkout", "-qb", "topic")
    write(root, "src/old module.py", "# topic\n" * 1004)
    topic = commit(root)
    git(root, "checkout", "-qb", "target", base)
    write(root, "README", "base advance\n")
    event_base = commit(root)
    git(root, "merge", "--no-ff", "-qm", "synthetic merge", topic)
    assert any("must shrink" in f.message for f in check(root, event_base))
    assert check(root, "HEAD") == []  # Shows why CI must never select HEAD versus itself.


def test_parser_and_validate_propagate_base(monkeypatch) -> None:
    h = _load_harness()
    observed = []
    monkeypatch.setattr(h, "run_structural_checks", lambda **kwargs: observed.append(kwargs) or 1)
    for command in ("structural", "validate"):
        args = h.build_parser().parse_args([command, "--base", "immutable"])
        assert args.func(args) == 1
    assert observed == [{"base": "immutable", "ci": False, "snapshot": False}] * 2


def test_structural_orchestrator_runs_maintenance(monkeypatch) -> None:
    h = _load_harness()
    for name in vars(h):
        if name.startswith("find_"):
            monkeypatch.setattr(h, name, lambda: [])
    monkeypatch.setattr(
        h,
        "find_maintenance_violations",
        lambda **kwargs: [h.HarnessFailure(Path("src/bad.py"), 0, "module-size: bad")],
    )
    assert h.run_structural_checks(base="immutable") == 1


def test_git_attributes_cannot_hide_or_substitute_base_source(repo) -> None:
    root, _ = repo
    write(root, ".gitattributes", '"src/old module.py" export-ignore export-subst\n')
    base = commit(root)
    write(root, "src/old module.py", "# changed\n" * 1004)
    assert any("must shrink" in f.message for f in check(root, base))


def test_named_virtual_environment_is_excluded_in_both_trees(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    write(tmp_path, "tools/custom-runtime/pyvenv.cfg", "home = /unused\n")
    write(tmp_path, "tools/custom-runtime/library.py", "# dependency\n" * 1001)
    freeze(tmp_path)
    base = commit(tmp_path)
    assert check(tmp_path, base) == []


def test_unstaged_mode_only_touch_requires_shrink(repo) -> None:
    root, base = repo
    git(root, "config", "core.filemode", "true")
    path = root / "src/old module.py"
    path.chmod(path.stat().st_mode | 0o111)
    assert any("must shrink" in f.message for f in check(root, base))


def test_actual_structural_command_rejects_new_debt(tmp_path: Path) -> None:
    import os
    import shutil
    import sys

    root = Path(__file__).resolve().parents[2]
    for name in (
        "harness.py",
        "harness_checks.py",
        "owned_probes.py",
        "harness_maintenance.py",
        "harness_retired_docs.py",
    ):
        target = tmp_path / "tools" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(root / "tools" / name, target)
    write(
        tmp_path,
        "src/asymmetry/core/utils/constants.py",
        (root / "src/asymmetry/core/utils/constants.py").read_text(),
    )
    baseline(tmp_path)
    write(tmp_path, "AGENTS.md", "")
    write(tmp_path, "CLAUDE.md", "@AGENTS.md\n")
    write(tmp_path, "src/asymmetry/resources/skills/asymmetry-analysis/SKILL.md", "")
    write(tmp_path, "src/oversized.py", "# line\n" * 1000)
    write(tmp_path, "src/owned.py", PROBE)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "tools/harness.py"), "structural", "--snapshot"],
        capture_output=True,
        text=True,
        env={**os.environ, "ASYMMETRY_HARNESS_NO_VENV": "1"},
    )
    assert result.returncode == 1
    assert "src/oversized.py: module-size" in result.stdout
    assert "src/owned.py: owned-probe" in result.stdout
    assert "snapshot only" in result.stdout


def test_current_repository_debt_matches_reviewed_baseline_without_mutation() -> None:
    root = Path(__file__).resolve().parents[2]
    original = (root / BASELINE).read_bytes()
    assert check(root) == []
    assert (root / BASELINE).read_bytes() == original


@pytest.mark.parametrize("operation", ["rename", "copy"])
def test_probe_allowance_does_not_transfer_to_a_new_path(tmp_path: Path, operation: str) -> None:
    git(tmp_path, "init", "-q")
    old = write(tmp_path, "src/old panel.py", PROBE)
    freeze(tmp_path)
    base = commit(tmp_path)
    write(tmp_path, "src/new panel.py", PROBE)
    if operation == "rename":
        old.unlink()
    freeze(tmp_path)
    assert any("added/raised/restored" in f.message for f in check(tmp_path, base))


def test_maintained_hidden_tools_exclude_local_config_and_nested_worktrees(tmp_path: Path) -> None:
    baseline(tmp_path)
    for path in (".claude/hooks/check.py", ".github/scripts/check.py"):
        write(tmp_path, path, "# maintained\n" * 1000)
    for path in (
        ".claude/worktrees/other/src/private.py",
        ".claude/local.py",
        ".github/build/generated.py",
    ):
        write(tmp_path, path, "# excluded\n" * 2000)
    assert {f.path.relative_to(tmp_path).as_posix() for f in check(tmp_path)} == {
        ".claude/hooks/check.py",
        ".github/scripts/check.py",
    }


def test_hidden_roots_have_identical_base_and_current_scope(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    for path in (".claude/hooks/check.py", ".github/scripts/check.py"):
        write(tmp_path, path, "# maintained\n" * 1001)
    write(tmp_path, ".claude/worktrees/other/private.py", "# excluded\n" * 2000)
    freeze(tmp_path)
    base = commit(tmp_path)
    assert check(tmp_path, base) == []
    write(tmp_path, ".claude/hooks/check.py", "# smaller\n" * 1000)
    freeze(tmp_path)
    assert check(tmp_path, base) == []
