from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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
