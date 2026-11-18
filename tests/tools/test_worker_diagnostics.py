"""Worker termination evidence survives without changing the failing exit status."""

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


@pytest.mark.io
@pytest.mark.skipif(sys.platform != "linux", reason="Controlled native abort uses Linux signals")
@pytest.mark.parametrize(
    ("mode", "enabled"), [("healthy", True), ("healthy", False), ("timeout", True), ("abort", True)]
)
def test_worker_diagnostics_retain_timeout_and_native_failure(tmp_path, mode, enabled):
    child = tmp_path / "child"
    child.mkdir()
    (child / "conftest.py").write_text('pytest_plugins = ["tests.conftest"]\n')
    deadline = (
        "@pytest.mark.timeout(1, method='thread', func_only=True)\n" if mode == "timeout" else ""
    )
    (child / "test_worker.py").write_text(
        "import os\nimport resource\nimport time\nimport pytest\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        + deadline
        + "def test_worker_exit():\n"
        + {
            "healthy": "    time.sleep(0.02)\n",
            "timeout": "    time.sleep(3)\n",
            "abort": "    os.abort()\n",
        }[mode]
    )
    diagnostics = tmp_path / "diagnostics"
    env = dict(os.environ)
    env.pop("ASYMMETRY_PYTEST_DIAGNOSTICS", None)
    if enabled:
        env["ASYMMETRY_PYTEST_DIAGNOSTICS"] = str(diagnostics)
    process = subprocess.Popen(
        [
            sys.executable,
            "tools/harness.py",
            "test",
            "--",
            "-c",
            str(ROOT / "pyproject.toml"),
            str(child / "test_worker.py"),
            "-n",
            "1",
            "--max-worker-restart=0",
            "-q",
            "-o",
            "faulthandler_timeout=0.25" if mode == "timeout" else "faulthandler_timeout=90",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        pytest.fail("Diagnostic child did not finish: " + stdout + stderr)
    output = stdout + stderr
    assert process.returncode == (0 if mode == "healthy" else 1), output
    if not enabled:
        assert not diagnostics.exists()
        return
    files = list(diagnostics.glob("worker-*.log"))
    assert len(files) == 1, output
    worker_output = files[0].read_text()
    assert "worker=gw0 pid=" in worker_output
    if mode == "timeout":
        assert " Timeout " in worker_output
        assert "test_worker_exit" in worker_output
        assert "Timeout (0:00:00.250000)!" in output
    elif mode == "abort":
        assert "Fatal Python error: Aborted" in output
        assert "test_worker_exit" in output


@pytest.mark.skipif(sys.platform != "linux", reason="CI test step uses Bash pipefail")
def test_ci_diagnostics_preserve_the_test_command_exit(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["test"]
    step = next(step for step in job["steps"] if step.get("name") == "Run test suite")
    command = step["run"]
    original = next(line for line in command.splitlines() if "python tools/harness.py test" in line)
    fake = tmp_path / "failed_test.py"
    fake.write_text("import sys\nprint('worker stderr evidence', file=sys.stderr)\nsys.exit(7)\n")
    replacement = shlex.join([sys.executable, str(fake)])
    command = command.replace(original.split(" 2>&1")[0], replacement)
    env = dict(os.environ)
    env["ASYMMETRY_PYTEST_DIAGNOSTICS"] = str(tmp_path / "diagnostics")
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert "worker stderr evidence" in (tmp_path / "diagnostics" / "runner.log").read_text()
    retention = next(step for step in job["steps"] if step.get("name") == "Retain test diagnostics")
    assert retention["if"] == "always()"
    assert retention["with"]["path"] == step["env"]["ASYMMETRY_PYTEST_DIAGNOSTICS"]
    assert retention["with"]["path"] == "${{ runner.temp }}/pytest-diagnostics"
    assert retention["with"]["retention-days"] == 7
