from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "tools" / "harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("asymmetry_harness", HARNESS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_core_satisfies_gui_boundary() -> None:
    harness = _load_harness()

    assert harness.find_core_boundary_violations() == []


def test_required_knowledge_files_exist() -> None:
    harness = _load_harness()

    assert harness.find_knowledge_base_violations() == []


def test_core_boundary_check_reports_gui_imports(tmp_path: Path) -> None:
    core_root = tmp_path / "core"
    core_root.mkdir()
    bad_module = core_root / "bad.py"
    bad_module.write_text(
        "import PySide6\nfrom asymmetry.gui import app\nfrom matplotlib.figure import Figure\n",
        encoding="utf-8",
    )
    harness = _load_harness()

    failures = harness.find_core_boundary_violations(core_root)

    assert len(failures) == 3
    assert all("must not import" in failure.message for failure in failures)


def test_core_dependencies_do_not_include_gui_runtime_packages() -> None:
    harness = _load_harness()

    assert harness.find_dependency_boundary_violations() == []


def test_current_gui_has_no_duplicate_limit_field_classes() -> None:
    harness = _load_harness()

    assert harness.find_duplicate_limit_field_violations() == []


def test_duplicate_limit_field_check_reports_stray_class_definition(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "widgets").mkdir(parents=True)
    # The canonical home is exempt even though it defines the class.
    (gui_root / "widgets" / "axis_limits.py").write_text(
        "class FloatLimitField(QLineEdit):\n    pass\n", encoding="utf-8"
    )
    stray = gui_root / "panels" / "weird_panel.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("class WeirdLimitField(QLineEdit):\n    pass\n", encoding="utf-8")
    harness = _load_harness()
    harness.LIMIT_FIELD_HOME = gui_root / "widgets" / "axis_limits.py"

    failures = harness.find_duplicate_limit_field_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "FloatLimitField" in failures[0].message


def test_current_gui_has_no_stray_mpl_canvas_construction() -> None:
    harness = _load_harness()

    assert harness.find_duplicate_mpl_canvas_violations() == []


def test_mpl_canvas_check_reports_stray_construction_outside_allowlist(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "widgets").mkdir(parents=True)
    (gui_root / "widgets" / "mpl_canvas.py").write_text(
        "canvas = FigureCanvasQTAgg(figure)\n", encoding="utf-8"
    )
    stray = gui_root / "panels" / "rogue_panel.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("self.canvas = FigureCanvasQTAgg(self.figure)\n", encoding="utf-8")
    harness = _load_harness()
    harness.MPL_CANVAS_HOME = gui_root / "widgets" / "mpl_canvas.py"
    harness.MPL_CANVAS_CONSTRUCTION_ALLOWLIST = frozenset()

    failures = harness.find_duplicate_mpl_canvas_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "create_canvas" in failures[0].message


def test_mpl_canvas_check_is_silent_for_allowlisted_survivors(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "widgets").mkdir(parents=True)
    (gui_root / "windows").mkdir(parents=True)
    (gui_root / "widgets" / "mpl_canvas.py").write_text(
        "canvas = FigureCanvasQTAgg(figure)\n", encoding="utf-8"
    )
    allowlisted = gui_root / "windows" / "fit_wizard_window.py"
    allowlisted.write_text("self.canvas = FigureCanvasQTAgg(self.figure)\n", encoding="utf-8")
    harness = _load_harness()
    harness.MPL_CANVAS_HOME = gui_root / "widgets" / "mpl_canvas.py"
    harness.MPL_CANVAS_CONSTRUCTION_ALLOWLIST = frozenset({allowlisted})

    assert harness.find_duplicate_mpl_canvas_violations(gui_root) == []


def test_current_gui_has_no_bespoke_qthread_construction() -> None:
    harness = _load_harness()

    assert harness.find_bespoke_qthread_violations() == []


def test_qthread_check_reports_construction_outside_tasks_module(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    gui_root.mkdir(parents=True)
    (gui_root / "tasks.py").write_text("self._thread = QThread()\n", encoding="utf-8")
    stray = gui_root / "windows" / "legacy_window.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("self._thread = QThread(self)\n", encoding="utf-8")
    harness = _load_harness()
    harness.TASK_RUNNER_HOME = gui_root / "tasks.py"

    failures = harness.find_bespoke_qthread_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "TaskRunner" in failures[0].message


def test_current_tests_directory_satisfies_placement_rule() -> None:
    harness = _load_harness()

    assert harness.find_test_placement_violations() == []


def test_test_placement_check_reports_file_at_tests_root(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    stray = tests_root / "test_stray.py"
    stray.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    sanctioned = tests_root / "core" / "test_fine.py"
    sanctioned.parent.mkdir(parents=True)
    sanctioned.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    harness = _load_harness()

    failures = harness.find_test_placement_violations(tests_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "sanctioned tests/ subpackage" in failures[0].message


def test_test_placement_check_reports_unsanctioned_subpackage(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    misplaced = tests_root / "scratch" / "test_misplaced.py"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    harness = _load_harness()

    failures = harness.find_test_placement_violations(tests_root)

    assert len(failures) == 1
    assert failures[0].path == misplaced


def _install_fake_os(monkeypatch, harness, *, name: str, execv=None):
    """Swap ``harness.os`` for a shim with a chosen ``name``.

    ``_maybe_reexec_with_venv`` branches on ``os.name``, but that attribute is
    shared with ``pathlib`` (which picks Windows/Posix Path at construction).
    Mutating the real ``os.name`` would make ``Path()`` raise on the running
    platform, so we give the harness its own ``os`` view instead. ``environ``
    and ``execv`` stay wired to the real module unless overridden.
    """
    fake_os = SimpleNamespace(
        name=name,
        environ=os.environ,
        execv=execv if execv is not None else os.execv,
    )
    monkeypatch.setattr(harness, "os", fake_os)
    return fake_os


def test_windows_reexec_propagates_child_exit_code(monkeypatch, tmp_path: Path) -> None:
    # On Windows os.execv spawns a *child* and exits the parent with 0, so a
    # failing pytest run inside the re-exec would be reported to the shell as a
    # success. The nt branch must instead run the child synchronously and exit
    # with its return code so failures still fail.
    harness = _load_harness()

    fake_venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(harness, "_preferred_venv_python", lambda root=None: fake_venv_python)
    # Do not let the "already inside the venv" short-circuit fire.
    monkeypatch.setattr(harness.sys, "prefix", str(tmp_path / "not-the-venv"))
    _install_fake_os(monkeypatch, harness, name="nt")

    recorded: dict[str, object] = {}

    class _Completed:
        returncode = 7

    def _fake_run(argv, check=False):
        recorded["argv"] = list(argv)
        recorded["check"] = check
        return _Completed()

    monkeypatch.setattr(harness.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as excinfo:
        harness._maybe_reexec_with_venv(["test", "--", "tests/does_not_exist.py"])

    assert excinfo.value.code == 7
    assert recorded["argv"][0] == str(fake_venv_python)
    assert recorded["check"] is False


def test_posix_reexec_replaces_process_with_execv(monkeypatch, tmp_path: Path) -> None:
    # POSIX keeps the real exec semantics: os.execv replaces the process, so the
    # venv interpreter inherits the caller's exit-code contract directly and we
    # must not fall back to subprocess.run.
    harness = _load_harness()

    fake_venv_python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(harness, "_preferred_venv_python", lambda root=None: fake_venv_python)
    monkeypatch.setattr(harness.sys, "prefix", str(tmp_path / "not-the-venv"))

    recorded: dict[str, object] = {}

    def _fake_execv(path, argv):
        recorded["path"] = path
        recorded["argv"] = list(argv)

    _install_fake_os(monkeypatch, harness, name="posix", execv=_fake_execv)

    def _forbidden_run(*args, **kwargs):
        raise AssertionError("POSIX re-exec must use os.execv, not subprocess.run")

    monkeypatch.setattr(harness.subprocess, "run", _forbidden_run)

    # os.execv is stubbed so control returns normally instead of replacing us.
    harness._maybe_reexec_with_venv(["structural"])

    assert recorded["path"] == str(fake_venv_python)
    assert recorded["argv"][0] == str(fake_venv_python)


def test_harness_test_command_fails_on_failing_tests(tmp_path: Path) -> None:
    # End-to-end guard: the harness `test` subcommand must surface a non-zero
    # exit code when the underlying pytest run fails. Run with the re-exec
    # disabled so we exercise this interpreter's entry point directly and stay
    # independent of whether a project .venv exists.
    failing_test = tmp_path / "test_deliberate_failure.py"
    failing_test.write_text("def test_fails():\n    assert False\n", encoding="utf-8")

    env = {**os.environ, "ASYMMETRY_HARNESS_NO_VENV": "1"}
    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_PATH),
            "test",
            "--no-parallel",
            "--",
            str(failing_test),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode != 0, (
        "harness `test` returned 0 despite a failing pytest run\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _marker(command: list[str]) -> str | None:
    """Return the pytest ``-m`` marker expression in a built command, if any.

    Skips past the ``python -m pytest`` prefix so the leading ``-m`` (whose value
    is ``pytest``) is not mistaken for the marker selector.
    """
    rest = command[command.index("pytest") + 1 :]
    if "-m" in rest:
        return rest[rest.index("-m") + 1]
    return None


def test_default_run_uses_standard_tier_marker_and_parallel() -> None:
    harness = _load_harness()

    command = harness.build_pytest_command([])

    assert _marker(command) == "(not slow and not integration)"
    assert "-n" in command and "auto" in command


def test_fast_tier_marker_is_unit_only_and_parallel() -> None:
    harness = _load_harness()

    command = harness.build_pytest_command([], tier="fast")

    assert _marker(command) == "(unit and not slow and not gui and not io and not integration)"
    # At ~2,400 tests the fast tier is no longer worker-startup bound: measured
    # 71s serial vs 25s with `-n auto`, so it parallelizes like the other tiers.
    assert "-n" in command and "auto" in command


def test_subset_composes_with_tier_marker() -> None:
    harness = _load_harness()

    gui = harness.build_pytest_command([], tier="standard", subset="gui")
    non_gui = harness.build_pytest_command([], tier="standard", subset="non-gui")

    # The two shards together partition exactly the standard tier.
    assert _marker(gui) == "(not slow and not integration) and (gui)"
    assert _marker(non_gui) == "(not slow and not integration) and (not gui)"


def test_full_tier_subset_has_no_tier_clause() -> None:
    harness = _load_harness()

    command = harness.build_pytest_command([], tier="full", subset="gui")

    # Full tier has no exclusion marker, so only the subset clause remains.
    assert _marker(command) == "(gui)"


def test_explicit_targets_bypass_tier_marker() -> None:
    harness = _load_harness()

    for target in ("tests/test_plot_panel.py", "tests/test_x.py::test_case"):
        command = harness.build_pytest_command([target])
        assert _marker(command) is None, f"{target} should run verbatim"
        assert target in command


def test_explicit_marker_is_not_overridden() -> None:
    harness = _load_harness()

    command = harness.build_pytest_command(["-m", "gui"])

    assert _marker(command) == "gui"


def test_k_expression_is_not_mistaken_for_a_target() -> None:
    harness = _load_harness()

    # A -k filter value is an option value, not a path: the tier marker must
    # still be injected.
    command = harness.build_pytest_command(["-k", "fourier"])

    assert _marker(command) == "(not slow and not integration)"


def test_path_like_option_value_does_not_bypass_marker() -> None:
    harness = _load_harness()

    # `-o cache_dir=/tmp/x` is an option value, not a test target — its slash must
    # not be mistaken for a path and bypass the tier marker.
    command = harness.build_pytest_command(["-o", "cache_dir=/tmp/x"])

    assert _marker(command) == "(not slow and not integration)"


def test_fast_tier_rejects_subset() -> None:
    harness = _load_harness()

    # fast is non-GUI by definition; fast+gui would compose to a contradictory
    # marker selecting zero tests yet exiting 0, so it must be rejected loudly.
    for subset in ("gui", "non-gui"):
        with pytest.raises(ValueError, match="fast"):
            harness.build_pytest_command([], tier="fast", subset=subset)


def test_shard_is_forwarded_after_marker() -> None:
    harness = _load_harness()

    command = harness.build_pytest_command([], subset="gui", shard="1/3")

    # The gui marker is still injected and `--shard 1/3` is appended for the
    # conftest hook. The "1/3" value must not be mistaken for a test target.
    assert _marker(command) == "(not slow and not integration) and (gui)"
    assert command[command.index("--shard") + 1] == "1/3"
