"""Worker termination evidence survives without changing the failing exit status."""

from __future__ import annotations

import json
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
    ("mode", "enabled", "restarts"),
    [
        (mode, enabled, 0)
        for mode in ("healthy", "nonzero", "timeout", "abort", "kill")
        for enabled in (True, False)
    ]
    + [("kill", enabled, 1) for enabled in (True, False)],
)
def test_worker_diagnostics_retain_exit_evidence(tmp_path, mode, enabled, restarts):
    child = tmp_path / "child"
    child.mkdir()
    (child / "conftest.py").write_text('pytest_plugins = ["tests._worker_diagnostics"]\n')
    deadline = (
        "@pytest.mark.timeout(1, method='thread', func_only=True)\n" if mode == "timeout" else ""
    )
    (child / "test_worker.py").write_text(
        "import os\nimport resource\nimport threading\nimport time\nimport pytest\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        + deadline
        + "def test_worker_exit():\n"
        + {
            "healthy": "    time.sleep(0.02)\n",
            "timeout": "    threading.Event().wait()\n",
            "abort": "    os.abort()\n",
            "nonzero": "    os._exit(7)\n",
            "kill": "    os.kill(os.getpid(), 9)\n",
        }[mode]
    )
    if restarts:
        (child / "test_worker.py").write_text(
            (child / "test_worker.py").read_text()
            + "\ndef test_replacement():\n    assert os.environ['PYTEST_XDIST_WORKER'] == 'gw1'\n"
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
            f"--max-worker-restart={restarts}",
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
    if restarts:
        assert "1 passed" in output, output
    if not enabled:
        assert not diagnostics.exists(), output
        return
    files = list(diagnostics.glob("worker-*.log"))
    assert len(files) == 1 + restarts, output
    worker_output = next(file for file in files if "gw0" in file.name).read_text()
    records = [
        json.loads(line) for line in (diagnostics / "controller.jsonl").read_text().splitlines()
    ]
    for worker in range(1 + restarts):
        events = [record for record in records if record["worker"] == f"gw{worker}"]
        assert [event["event"] for event in events] == ["started", "disconnected", "process_exit"]
        pid = events[0]["pid"]
        assert isinstance(pid, int) and pid > 0
        assert all(event["pid"] == pid for event in events)
        abnormal = mode != "healthy" and worker == 0
        assert events[1]["abnormal"] is abnormal
        exit_event = events[2]
        expected = {"healthy": 0, "nonzero": 7, "timeout": 1, "abort": -6, "kill": -9}
        code = expected[mode] if worker == 0 else 0
        assert exit_event["status"] == "observed", output
        assert exit_event["returncode"] == code, exit_event
        assert exit_event["signal"] == (-code if code < 0 else None), exit_event
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    assert "worker=gw0 pid=" in worker_output, output
    if mode == "timeout":
        assert " Timeout " in worker_output, output
        assert "test_worker_exit" in worker_output, output
        assert "Timeout (0:00:00.250000)!" in output, output
    elif mode == "abort":
        assert "Fatal Python error: Aborted" in output, output
        assert "test_worker_exit" in output, output


@pytest.mark.skipif(sys.platform != "linux", reason="CI test step uses Bash pipefail")
@pytest.mark.parametrize("exitcode", [0, 7])
def test_ci_diagnostics_preserve_the_test_command_exit(tmp_path, exitcode):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["test"]
    step = next(step for step in job["steps"] if step.get("name") == "Run test suite")
    command = step["run"]
    original = next(line for line in command.splitlines() if "python tools/harness.py test" in line)
    fake = tmp_path / "failed_test.py"
    fake.write_text(
        f"import sys\nprint('worker stderr evidence', file=sys.stderr)\nsys.exit({exitcode})\n"
    )
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
    assert result.returncode == exitcode, result.stdout + result.stderr
    assert "worker stderr evidence" in (tmp_path / "diagnostics" / "runner.log").read_text()
    for phase in ("before", "after"):
        record = json.loads((tmp_path / "diagnostics" / f"oom-{phase}.json").read_text())
        assert record["status"] in ("observed", "unavailable")
        assert record["source"] == "/proc/vmstat"

    retention = next(step for step in job["steps"] if step.get("name") == "Retain test diagnostics")
    assert retention["if"] == "always()"
    assert retention["with"]["path"] == step["env"]["ASYMMETRY_PYTEST_DIAGNOSTICS"]
    assert retention["with"]["path"] == "${{ runner.temp }}/pytest-diagnostics"
    assert retention["with"]["retention-days"] == 7


@pytest.mark.parametrize(
    ("content", "status", "count"),
    [
        ("pgpgin 13\noom_kill 4\n", "observed", 4),
        ("oom_kill 0\n", "observed", 0),
        ("pgpgin 13\n", "unavailable", None),
        ("oom_kill invalid\n", "unavailable", None),
        (None, "unavailable", None),
    ],
)
def test_memory_counter_evidence(tmp_path, content, status, count):
    from tools.ci_memory import record_oom_counter

    vmstat = tmp_path / "vmstat"
    if content is not None:
        vmstat.write_text(content)
    destination = tmp_path / "diagnostics"
    record_oom_counter(destination, "before", vmstat)
    record = json.loads((destination / "oom-before.json").read_text())
    assert record["status"] == status
    assert record["oom_kill"] == count
    assert record["source"] == str(vmstat)
    if status == "unavailable":
        assert record["reason"]


@pytest.mark.parametrize(
    "transport", ["socket", "proxy", "unsupported_io", "missing_io", "missing_process"]
)
def test_unsupported_worker_boundary_is_explicit(tmp_path, transport):
    from types import SimpleNamespace

    from execnet.gateway_io import Popen2IOMaster

    from tests._worker_diagnostics import WorkerExitRecords

    records = WorkerExitRecords(tmp_path)
    node = SimpleNamespace(
        gateway=SimpleNamespace(
            id="gw0",
            spec=SimpleNamespace(popen=transport != "socket", via=transport == "proxy"),
            _io=object(),
        )
    )
    if transport == "missing_io":
        del node.gateway._io
    elif transport == "missing_process":
        node.gateway._io = object.__new__(Popen2IOMaster)
    records.pytest_configure_node(node)
    records.pytest_testnodedown(node, "lost connection")
    finish = records.pytest_sessionfinish()
    next(finish)
    assert list(finish) == []
    events = [json.loads(line) for line in records.path.read_text().splitlines()]
    assert events[0]["status"] == "unavailable"
    assert events[1]["abnormal"] is True
    assert events[2]["status"] == "unavailable"
    assert events[2]["reason"] == "local process unavailable"
    assert all(event["pid"] is None for event in events)
    assert events[2]["returncode"] is None
    assert events[2]["signal"] is None


@pytest.mark.parametrize("eventual_code", [None, 7])
def test_cached_exit_observation_is_bounded_without_wait_or_poll(tmp_path, eventual_code):
    import time

    from tests._worker_diagnostics import WorkerExitRecords

    class CachedProcess:
        pid = 123

        def __init__(self):
            self.observations = 0

        @property
        def returncode(self):
            self.observations += 1
            return eventual_code if self.observations >= 3 else None

    process = CachedProcess()
    records = WorkerExitRecords(tmp_path)
    records.processes["gw0"] = process
    finish = records.pytest_sessionfinish()
    next(finish)
    start = time.monotonic()
    assert list(finish) == []
    assert time.monotonic() - start < 2
    event = json.loads(records.path.read_text())
    assert process.observations >= 3
    assert event["returncode"] == eventual_code
    assert event["signal"] is None
    assert event["status"] == ("observed" if eventual_code is not None else "unavailable")
    if eventual_code is None:
        assert event["reason"] == "exit code not observed before deadline"
