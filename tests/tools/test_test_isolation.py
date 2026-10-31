from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_core_harness_never_imports_qt(tmp_path: Path) -> None:
    plugin = tmp_path / "block_qt_plugin.py"
    plugin.write_text(
        """import sys
import importlib.abc

attempted = []

class BlockQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PySide6" or fullname.startswith(("PySide6.", "asymmetry.gui")):
            attempted.append(fullname)
            raise AssertionError(f"Core tests imported {fullname}")

sys.meta_path.insert(0, BlockQt())

def pytest_sessionfinish(session, exitstatus):
    leaked = [name for name in sys.modules if name.startswith(("PySide6", "asymmetry.gui"))]
    assert not attempted, attempted
    assert not leaked, leaked
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [
            sys.executable,
            "tools/harness.py",
            "test",
            "--no-parallel",
            "--",
            "tests/core/test_transforms.py",
            "-p",
            "block_qt_plugin",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Unwritable-home check uses Linux Qt paths")
def test_gui_harness_with_unwritable_home(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")
    env = dict(os.environ)
    env["HOME"] = str(blocker)
    result = subprocess.run(
        [
            sys.executable,
            "tools/harness.py",
            "test",
            "--no-parallel",
            "--",
            "tests/gui/test_visual_grammar.py::test_deck_docks_use_slim_title_bars",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
