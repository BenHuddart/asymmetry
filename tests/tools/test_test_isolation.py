from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.io
@pytest.mark.skipif(sys.platform == "win32", reason="Child cleanup uses POSIX process groups")
@pytest.mark.parametrize("slow_test", [False, True], ids=["healthy-session", "stalled-test"])
def test_native_watchdog_tracks_each_test_and_preserves_capture(
    tmp_path: Path, slow_test: bool
) -> None:
    child = tmp_path / "watchdog"
    child.mkdir()
    (child / "conftest.py").write_text(
        'pytest_plugins = ["tests.conftest"]\nfrom asymmetry.core import plugins\n'
    )
    (child / "test_timer.py").write_text(
        "import time\nimport pytest\n"
        '@pytest.mark.parametrize("index", range(20))\n'
        "def test_timer_capture(index):\n"
        f"    time.sleep({0.3 if slow_test else 0.02})\n"
    )
    workflow = yaml.safe_load((ROOT / ".github/workflows/full-suite.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["full-suite"]["steps"]
        if step["name"] == "Run full test suite"
    )
    options = shlex.split(step["run"].partition(" -- ")[2])
    timer = options.index("faulthandler_timeout=600")
    options[timer] = "faulthandler_timeout=0.1" if slow_test else "faulthandler_timeout=1"
    env = dict(os.environ)
    env["ASYMMETRY_FAULTHANDLER_TIMEOUT"] = "0.2"
    with subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "tools/harness.py"),
            "test",
            "--",
            str(child),
            "-n",
            "2",
            *options,
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail("Native watchdog child did not finish: " + stdout + stderr)
        output = stdout + stderr
        assert process.returncode == 0, output
    assert ("Timeout (" in output) is slow_test, output
    if slow_test:
        assert "test_timer_capture" in output
