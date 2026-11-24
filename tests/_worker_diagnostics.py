"""Opt-in worker output and controller observations; execnet owns process cleanup."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from subprocess import Popen

import pytest
from execnet.gateway_io import Popen2IOMaster
from xdist import get_xdist_worker_id


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    destination = os.environ.get("ASYMMETRY_PYTEST_DIAGNOSTICS")
    worker = get_xdist_worker_id(session)
    if not destination:
        return
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    if worker == "master":
        session.config.pluginmanager.register(WorkerExitRecords(directory))
        return
    stream = (directory / f"worker-{worker}-{os.getpid()}.log").open("w", buffering=1)
    stream.write(f"worker={worker} pid={os.getpid()}\n")
    # execnet discards worker stdout; timeout diagnostics must survive worker exit.
    session.config.get_terminal_writer()._file = stream
    session.config.add_cleanup(stream.close)


class WorkerExitRecords:
    def __init__(self, directory: Path) -> None:
        self.path = directory / "controller.jsonl"
        self.processes: dict[str, Popen | None] = {}

    def record(self, worker: str, event: str, **evidence: object) -> None:
        process = self.processes[worker]
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "worker": worker,
                        "pid": process.pid if process else None,
                        "event": event,
                        **evidence,
                    }
                )
                + "\n"
            )

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node) -> None:
        gateway = node.gateway
        # execnet 2.x exposes its local subprocess only through this private IO boundary.
        io = getattr(gateway, "_io", None)
        local = gateway.spec.popen and not gateway.spec.via
        candidate = getattr(io, "popen", None) if local and isinstance(io, Popen2IOMaster) else None
        process = candidate if isinstance(candidate, Popen) else None
        self.processes[gateway.id] = process
        self.record(gateway.id, "started", status="observed" if process else "unavailable")

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error) -> None:
        self.record(node.gateway.id, "disconnected", abnormal=error is not None)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(self):
        result = yield
        # Normal execnet teardown waits/reaps; only observe its cached return codes.
        deadline = time.monotonic() + 0.25
        pending = dict(self.processes)
        while pending:
            for worker, process in list(pending.items()):
                code = process.returncode if process else None
                if code is None and process and time.monotonic() < deadline:
                    continue
                reason = None
                if code is None:
                    reason = (
                        "local process unavailable"
                        if process is None
                        else "exit code not observed before deadline"
                    )
                self.record(
                    worker,
                    "process_exit",
                    status="observed" if code is not None else "unavailable",
                    reason=reason,
                    returncode=code,
                    signal=-code if code is not None and code < 0 and os.name == "posix" else None,
                )
                del pending[worker]
            if pending:
                time.sleep(0.01)
        return result
