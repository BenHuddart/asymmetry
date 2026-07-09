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


def test_current_gui_has_no_widget_screen_calls() -> None:
    harness = _load_harness()

    assert harness.find_widget_screen_call_violations() == []


def test_widget_screen_check_reports_calls_outside_screen_guard(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    gui_root.mkdir(parents=True)
    (gui_root / "screen_guard.py").write_text("current = window.screen()\n", encoding="utf-8")
    stray = gui_root / "windows" / "some_dialog.py"
    stray.parent.mkdir(parents=True)
    stray.write_text(
        "# prose may mention .screen() without tripping the rule\n"
        "screen = self.screen()  # trailing comment\n",
        encoding="utf-8",
    )
    harness = _load_harness()
    harness.SCREEN_GUARD_HOME = gui_root / "screen_guard.py"

    failures = harness.find_widget_screen_call_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert failures[0].line == 2
    assert "screen_for" in failures[0].message


def test_current_gui_has_no_bespoke_section_definitions() -> None:
    harness = _load_harness()

    assert harness.find_bespoke_section_violations() == []


def test_current_gui_has_no_bespoke_gle_export_fragments() -> None:
    harness = _load_harness()

    assert harness.find_bespoke_gle_export_violations() == []


def test_gle_export_check_reports_fragments_outside_shared_home(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    utils = gui_root / "utils"
    utils.mkdir(parents=True)
    (utils / "gle_export.py").write_text(
        'glp = importlib.import_module("gleplot")\ncompile_gle(exe, f, "pdf", cwd=d)\n',
        encoding="utf-8",
    )
    stray = gui_root / "panels" / "legacy_panel.py"
    stray.parent.mkdir(parents=True)
    stray.write_text(
        'glp = importlib.import_module("gleplot")\n'
        'compile_gle(exe, f, "pdf", cwd=d)\n'
        "def _show_gle_preview(self, gle_path):\n"
        "    pass\n",
        encoding="utf-8",
    )
    harness = _load_harness()
    harness.GLE_EXPORT_UTILS = frozenset({utils / "gle_export.py"})

    failures = harness.find_bespoke_gle_export_violations(gui_root)

    assert len(failures) == 3
    assert all(f.path == stray for f in failures)
    assert any("run_gle_export" in f.message for f in failures)


def test_bespoke_section_check_reports_class_and_deleted_import(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "widgets").mkdir(parents=True)
    # The canonical home is exempt.
    (gui_root / "widgets" / "panel_section.py").write_text(
        "class PanelSection(QWidget):\n    pass\n", encoding="utf-8"
    )
    stray_class = gui_root / "panels" / "rogue.py"
    stray_class.parent.mkdir(parents=True)
    stray_class.write_text("class MyCollapsibleSection(QWidget):\n    pass\n", encoding="utf-8")
    stray_import = gui_root / "windows" / "legacy.py"
    stray_import.parent.mkdir(parents=True)
    stray_import.write_text(
        "from asymmetry.gui.widgets.collapsible_section import CollapsibleSection\n",
        encoding="utf-8",
    )
    harness = _load_harness()
    harness.PANEL_SECTION_HOME = gui_root / "widgets" / "panel_section.py"

    failures = harness.find_bespoke_section_violations(gui_root)

    assert {f.path for f in failures} == {stray_class, stray_import}
    assert all("PanelSection" in f.message for f in failures)


def test_bespoke_section_check_ignores_panel_local_wrapper(tmp_path: Path) -> None:
    # A `_collapsible_group` factory that *wraps* PanelSection is the sanctioned
    # convention (alc_panel), not a bespoke reimplementation.
    gui_root = tmp_path / "gui"
    (gui_root / "widgets").mkdir(parents=True)
    (gui_root / "widgets" / "panel_section.py").write_text(
        "class PanelSection(QWidget):\n    pass\n", encoding="utf-8"
    )
    wrapper = gui_root / "panels" / "alc_panel.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "def _collapsible_group(title):\n    return PanelSection(title, collapsible=True)\n",
        encoding="utf-8",
    )
    harness = _load_harness()
    harness.PANEL_SECTION_HOME = gui_root / "widgets" / "panel_section.py"

    assert harness.find_bespoke_section_violations(gui_root) == []


def test_current_gui_has_no_raw_hex_colour_literals() -> None:
    harness = _load_harness()

    assert harness.find_raw_hex_colour_violations() == []


def test_raw_hex_colour_check_reports_stray_literal(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "styles").mkdir(parents=True)
    # styles/ is exempt (the token source).
    (gui_root / "styles" / "tokens.py").write_text('ACCENT = "#1f4d8a"\n', encoding="utf-8")
    stray = gui_root / "panels" / "rogue.py"
    stray.parent.mkdir(parents=True)
    stray.write_text('label.setStyleSheet("color: #a8332a;")\n', encoding="utf-8")
    harness = _load_harness()

    failures = harness.find_raw_hex_colour_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "tokens" in failures[0].message


def test_raw_hex_colour_check_ignores_qss_id_selectors(tmp_path: Path) -> None:
    # `#addFieldRail` is a QSS id selector, not a hex colour — the trailing
    # non-identifier guard must reject it.
    gui_root = tmp_path / "gui"
    (gui_root / "panels").mkdir(parents=True)
    qss = gui_root / "panels" / "data_browser.py"
    qss.write_text(
        'rail.setStyleSheet("#addFieldRail { background-color: red; }")\n'
        'filler.setStyleSheet("#addFieldFiller { background-color: red; }")\n',
        encoding="utf-8",
    )
    harness = _load_harness()

    assert harness.find_raw_hex_colour_violations(gui_root) == []


def test_raw_hex_colour_check_honours_allowlist(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "widgets").mkdir(parents=True)
    allowed = gui_root / "widgets" / "detector_schematic.py"
    allowed.write_text('ax.text(0, 0, "s", color="#333333")\n', encoding="utf-8")
    harness = _load_harness()
    harness.HEX_COLOUR_ALLOWLIST = {allowed: "matplotlib diagram colours"}

    assert harness.find_raw_hex_colour_violations(gui_root) == []


def test_current_gui_has_no_literal_pixel_geometry() -> None:
    harness = _load_harness()

    assert harness.find_literal_pixel_geometry_violations() == []


def test_literal_pixel_geometry_check_reports_large_literal(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "panels").mkdir(parents=True)
    stray = gui_root / "panels" / "rogue.py"
    stray.write_text("widget.setFixedWidth(120)\n", encoding="utf-8")
    harness = _load_harness()
    harness.PIXEL_GEOMETRY_ALLOWLIST = {}

    failures = harness.find_literal_pixel_geometry_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "metrics" in failures[0].message


def test_literal_pixel_geometry_check_allows_small_paddings(tmp_path: Path) -> None:
    # Literals below the threshold (paddings / hairlines / icon swatches) are fine.
    gui_root = tmp_path / "gui"
    (gui_root / "panels").mkdir(parents=True)
    small = gui_root / "panels" / "fine.py"
    small.write_text(
        "pad.setFixedWidth(6)\ntable.setMinimumHeight(0)\nicon.setFixedSize(20, 20)\n",
        encoding="utf-8",
    )
    harness = _load_harness()
    harness.PIXEL_GEOMETRY_ALLOWLIST = {}

    assert harness.find_literal_pixel_geometry_violations(gui_root) == []


def test_literal_pixel_geometry_check_honours_allowlist(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    (gui_root / "panels").mkdir(parents=True)
    allowed = gui_root / "panels" / "canvas_panel.py"
    allowed.write_text("self._canvas.setMinimumHeight(200)\n", encoding="utf-8")
    harness = _load_harness()
    harness.PIXEL_GEOMETRY_ALLOWLIST = {allowed: "canvas floor"}

    assert harness.find_literal_pixel_geometry_violations(gui_root) == []


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


def test_current_tests_directory_satisfies_gui_marker_rule() -> None:
    harness = _load_harness()

    assert harness.find_missing_gui_marker_violations() == []


def test_gui_marker_check_reports_unmarked_qapp_function(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    gui_dir = tests_root / "gui"
    gui_dir.mkdir(parents=True)
    unmarked = gui_dir / "test_unmarked_gui.py"
    unmarked.write_text(
        "def test_needs_qt(qapp) -> None:\n    assert qapp is not None\n",
        encoding="utf-8",
    )
    harness = _load_harness()

    failures = harness.find_missing_gui_marker_violations(tests_root)

    assert len(failures) == 1
    assert failures[0].path == unmarked
    assert "pytest.mark.gui" in failures[0].message


def test_gui_marker_check_accepts_file_level_pytestmark(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    gui_dir = tests_root / "gui"
    gui_dir.mkdir(parents=True)
    marked = gui_dir / "test_marked_gui.py"
    marked.write_text(
        "import pytest\n\npytestmark = [pytest.mark.gui]\n\n\ndef test_needs_qt(qapp) -> None:\n"
        "    assert qapp is not None\n",
        encoding="utf-8",
    )
    harness = _load_harness()

    assert harness.find_missing_gui_marker_violations(tests_root) == []


def test_gui_marker_check_accepts_per_function_decorator(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    gui_dir = tests_root / "gui"
    gui_dir.mkdir(parents=True)
    mixed = gui_dir / "test_mixed_gui.py"
    mixed.write_text(
        "import pytest\n\n\ndef test_pure() -> None:\n    assert True\n\n\n"
        "@pytest.mark.gui\ndef test_needs_qt(qapp) -> None:\n    assert qapp is not None\n",
        encoding="utf-8",
    )
    harness = _load_harness()

    assert harness.find_missing_gui_marker_violations(tests_root) == []


def test_gui_marker_check_reports_unmarked_module_usefixtures(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    gui_dir = tests_root / "gui"
    gui_dir.mkdir(parents=True)
    unmarked = gui_dir / "test_unmarked_usefixtures.py"
    unmarked.write_text(
        'import pytest\n\npytestmark = pytest.mark.usefixtures("qapp")\n\n\n'
        "def test_needs_qt() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    harness = _load_harness()

    failures = harness.find_missing_gui_marker_violations(tests_root)

    assert len(failures) == 1
    assert failures[0].path == unmarked


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


def test_current_gui_has_no_stray_process_events_calls() -> None:
    harness = _load_harness()

    assert harness.find_process_events_violations() == []


def test_process_events_check_reports_calls_outside_allowlist(tmp_path: Path) -> None:
    gui_root = tmp_path / "gui"
    gui_root.mkdir(parents=True)
    allowed = gui_root / "app.py"
    allowed.write_text("app.processEvents()\n", encoding="utf-8")
    stray = gui_root / "windows" / "busy_dialog.py"
    stray.parent.mkdir(parents=True)
    stray.write_text(
        "def _progress(done, total):\n    QApplication.processEvents()\n",
        encoding="utf-8",
    )
    benign = gui_root / "tasks.py"
    # sendPostedEvents is the queued-event drain in TaskRunner.shutdown — not
    # an event-loop pump; it must not trip the check.
    benign.write_text("app.sendPostedEvents(self, 0)\n", encoding="utf-8")
    harness = _load_harness()
    harness.PROCESS_EVENTS_ALLOWLIST = frozenset({allowed})

    failures = harness.find_process_events_violations(gui_root)

    assert len(failures) == 1
    assert failures[0].path == stray
    assert "GUI_GUIDELINES" in failures[0].message
