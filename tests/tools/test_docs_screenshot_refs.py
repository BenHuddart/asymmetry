"""Tests for the docs screenshot-reference consistency check.

Covers the static check itself (``docs/screenshots/capture.py``), its harness
wiring (``find_screenshot_reference_violations`` in the ``structural`` command),
and the ``--check-refs`` CLI mode.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "tools" / "harness.py"
CAPTURE_PATH = ROOT / "docs" / "screenshots" / "capture.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_harness():
    return _load_module("asymmetry_harness", HARNESS_PATH)


def _load_capture():
    return _load_module("asymmetry_screenshot_capture_test", CAPTURE_PATH)


def _write_scenario(scenarios_dir: Path, module_name: str, scenario_name: str) -> Path:
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    path = scenarios_dir / f"{module_name}.py"
    path.write_text(
        f'class _Scenario:\n    name = "{scenario_name}"\n',
        encoding="utf-8",
    )
    return path


def _write_rst(docs_dir: Path, relative: str, *screenshot_names: str) -> Path:
    path = docs_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f".. image:: /_generated/screenshots/{name}.png" for name in screenshot_names]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_current_tree_screenshot_references_are_consistent() -> None:
    harness = _load_harness()

    assert harness.find_screenshot_reference_violations() == []


def test_check_refs_cli_mode_passes_on_current_tree(capsys) -> None:
    # The CLI mode must run the static check without booting Qt or arming the
    # watchdog, and exit 0 on the current (consistent) tree.
    capture = _load_capture()

    assert capture.main(["--check-refs"]) == 0
    assert "screenshot references: ok" in capsys.readouterr().out


def test_unregistered_reference_is_reported(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    scenarios_dir = docs_dir / "screenshots" / "scenarios"
    _write_scenario(scenarios_dir, "real_scenario", "real_scenario")
    _write_rst(docs_dir, "reference/page.rst", "real_scenario", "ghost_scenario")
    capture = _load_capture()

    problems = capture.check_screenshot_references(docs_dir, scenarios_dir)

    assert len(problems) == 1
    assert "ghost_scenario" in problems[0]
    assert "no registered scenario" in problems[0]
    # The failing reference is located for the reader.
    assert "page.rst:2" in problems[0]


def test_unreferenced_scenario_is_reported(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    scenarios_dir = docs_dir / "screenshots" / "scenarios"
    _write_scenario(scenarios_dir, "used_scenario", "used_scenario")
    _write_scenario(scenarios_dir, "orphan_scenario", "orphan_scenario")
    _write_rst(docs_dir, "reference/page.rst", "used_scenario")
    capture = _load_capture()

    problems = capture.check_screenshot_references(docs_dir, scenarios_dir)

    assert len(problems) == 1
    assert "orphan_scenario" in problems[0]
    assert "never referenced" in problems[0]


def test_scenario_scan_ignores_call_kwargs_and_private_modules(tmp_path: Path) -> None:
    # `name="..."` keyword arguments inside builder calls (dataset titles etc.)
    # and underscore-prefixed modules (_base.py) must not register names.
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "fancy.py").write_text(
        "class FancyScenario:\n"
        '    name = "fancy"\n'
        "\n"
        "    def build(self):\n"
        '        return make_series([1, 2], name="LF decoupling - Ag")\n',
        encoding="utf-8",
    )
    (scenarios_dir / "_base.py").write_text(
        'class Scenario:\n    name = "should_not_count"\n', encoding="utf-8"
    )
    capture = _load_capture()

    assert capture.scenario_names_from_source(scenarios_dir) == {"fancy"}


def test_reference_scan_skips_build_and_generated_trees(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    _write_rst(docs_dir, "reference/page.rst", "real_scenario")
    _write_rst(docs_dir, "_build/html/_sources/stale.rst", "stale_scenario")
    _write_rst(docs_dir, "_generated/copy.rst", "generated_scenario")
    capture = _load_capture()

    assert set(capture.referenced_screenshot_names(docs_dir)) == {"real_scenario"}


def test_harness_wraps_problems_as_structural_failures(tmp_path: Path) -> None:
    root = tmp_path
    docs_dir = root / "docs"
    scenarios_dir = docs_dir / "screenshots" / "scenarios"
    _write_scenario(scenarios_dir, "orphan_scenario", "orphan_scenario")
    _write_rst(docs_dir, "reference/page.rst", "ghost_scenario")
    # The harness loads the real capture module but points it at *root*'s docs.
    (docs_dir / "screenshots" / "capture.py").write_text("", encoding="utf-8")
    harness = _load_harness()

    failures = harness.find_screenshot_reference_violations(root)

    messages = [failure.message for failure in failures]
    assert len(messages) == 2
    assert any("ghost_scenario" in message for message in messages)
    assert any("orphan_scenario" in message for message in messages)


def test_docs_command_runs_capture_first_with_screenshots_flag(monkeypatch) -> None:
    harness = _load_harness()
    recorded: list[list[str]] = []

    def _fake_run(command):
        recorded.append(list(command))
        return 0

    monkeypatch.setattr(harness, "_run_command", _fake_run)

    from types import SimpleNamespace

    assert harness.cmd_docs(SimpleNamespace(screenshots=True)) == 0
    assert len(recorded) == 2
    assert recorded[0][1:] == [
        "-m",
        "docs.screenshots.capture",
        "--out",
        "docs/_generated/screenshots",
    ]
    assert "sphinx" in recorded[1]

    recorded.clear()
    assert harness.cmd_docs(SimpleNamespace(screenshots=False)) == 0
    assert len(recorded) == 1
    assert "sphinx" in recorded[0]


def test_oversized_paths_flags_only_files_over_budget(tmp_path: Path) -> None:
    # Exercises the exact helper `main()` calls on the just-captured paths
    # (as opposed to `oversized_screenshots`, which re-globs a directory) so
    # the CLI's non-zero-exit path has a direct, Qt-free unit test.
    capture = _load_capture()
    small = tmp_path / "small.png"
    big = tmp_path / "big.png"
    small.write_bytes(b"0" * 100)
    big.write_bytes(b"0" * 1000)

    entries = capture._oversized_paths([small, big], budget_bytes=500)

    assert len(entries) == 1
    assert "big.png" in entries[0]
    assert "small.png" not in entries[0]


def test_oversized_paths_empty_when_all_within_budget(tmp_path: Path) -> None:
    capture = _load_capture()
    path = tmp_path / "fine.png"
    path.write_bytes(b"0" * 100)

    assert capture._oversized_paths([path], budget_bytes=500) == []


def test_oversized_screenshots_skips_silently_when_dir_absent(tmp_path: Path) -> None:
    capture = _load_capture()

    assert capture.oversized_screenshots(tmp_path / "does_not_exist") == []


def test_oversized_screenshots_flags_files_over_budget(tmp_path: Path) -> None:
    capture = _load_capture()
    generated_dir = tmp_path / "screenshots"
    generated_dir.mkdir()
    small = generated_dir / "small.png"
    big = generated_dir / "big.png"
    small.write_bytes(b"0" * 100)
    big.write_bytes(b"0" * 1000)

    problems = capture.oversized_screenshots(generated_dir, budget_bytes=500)

    assert len(problems) == 1
    assert "big.png" in problems[0]
    assert "small.png" not in problems[0]


def test_oversized_screenshots_respects_default_budget_on_current_tree() -> None:
    # Guards against accidentally lowering the shared budget below the
    # largest screenshot actually committed to the generated output, which
    # would make `structural`/`--check-refs` fail on a clean, in-budget tree.
    capture = _load_capture()

    assert capture.SCREENSHOT_SIZE_BUDGET_BYTES >= 600 * 1024


def test_check_screenshot_references_reports_oversized_generated_png(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    scenarios_dir = docs_dir / "screenshots" / "scenarios"
    _write_scenario(scenarios_dir, "real_scenario", "real_scenario")
    _write_rst(docs_dir, "reference/page.rst", "real_scenario")
    generated_dir = docs_dir / "_generated" / "screenshots"
    generated_dir.mkdir(parents=True)
    (generated_dir / "real_scenario.png").write_bytes(b"0" * (capture_budget_bytes() + 1))
    capture = _load_capture()

    problems = capture.check_screenshot_references(docs_dir, scenarios_dir)

    assert len(problems) == 1
    assert "real_scenario.png" in problems[0]
    assert "budget" in problems[0]


def capture_budget_bytes() -> int:
    return _load_capture().SCREENSHOT_SIZE_BUDGET_BYTES
