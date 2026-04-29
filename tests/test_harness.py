from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
