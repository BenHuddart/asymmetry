"""Wait for the real application diagnostic and its fresh per-invocation report."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def payload_environment() -> dict[str, str]:
    """Keep OS runtime access while removing developer Python/native search overrides."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "PIP_", "CONDA", "DYLD_", "ASYMMETRY_"))
        and key
        not in {
            "VIRTUAL_ENV",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "QT_PLUGIN_PATH",
            "QML2_IMPORT_PATH",
        }
    }
    if os.name == "nt":
        windows = Path(environment["SystemRoot"])
        environment["PATH"] = os.pathsep.join(map(str, (windows / "System32", windows)))
    else:
        environment["PATH"] = os.defpath
    environment.update(
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen"
    )
    return environment


def terminate_tree(process: subprocess.Popen) -> dict:
    """Preserve cleanup failures separately from the primary scientific/timeout failure."""
    result = {}
    try:
        if os.name == "nt":
            cleanup = subprocess.run(
                [
                    str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"),
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                timeout=10,
            )
            result["tree_cleanup_exit"] = cleanup.returncode
        else:
            os.killpg(process.pid, signal.SIGKILL)
            result["tree_cleanup_exit"] = 0
    except ProcessLookupError:
        result["tree_cleanup_exit"] = 0
    except (OSError, subprocess.SubprocessError) as exc:
        result["tree_cleanup_failure"] = type(exc).__name__
    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        result["child_reaped"] = True
    except (OSError, subprocess.SubprocessError) as exc:
        result["reap_failure"] = type(exc).__name__
    return result


def run_acceptance(
    command: list[str],
    inputs: Path,
    identity: dict,
    work: Path,
    *,
    timeout: float = 60,
    environment: dict | None = None,
) -> dict:
    """Execute one owned child tree; reject failures, timeout, and stale/missing reports."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("acceptance timeout must be finite and positive")
    work.mkdir(parents=True, exist_ok=False)
    runtime = work / "runtime"
    runtime.mkdir()
    child_environment = payload_environment() if environment is None else environment.copy()
    for key in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
    ):
        child_environment[key] = str(runtime.resolve())
    token = uuid.uuid4().hex
    root = work / "journey"
    arguments = [
        *command,
        "--scientific-acceptance",
        "--acceptance-inputs",
        str(inputs.resolve()),
        "--acceptance-root",
        str(root.resolve()),
        "--acceptance-token",
        token,
        "--acceptance-identity",
        json.dumps(identity, allow_nan=False),
    ]
    started = time.monotonic()
    outcome = {"token": token, "success": False, "timeout_seconds": timeout}
    try:
        with (
            (work / "launcher.log").open("wb") as stream,
            (work / "launcher-error.log").open("wb") as errors,
        ):
            process = subprocess.Popen(
                arguments,
                cwd=work,
                env=child_environment,
                stdout=stream,
                stderr=errors,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
    except OSError as exc:
        outcome["failure"] = "launcher failed"
        outcome["failure_type"] = type(exc).__name__
        (work / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("desktop acceptance failed: launcher failed") from None
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            outcome["failure"] = "timeout"
            outcome.update(terminate_tree(process))
        outcome["exit_code"] = process.returncode
        path = root / "result.json"
        if path.is_file():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                json.dumps(report, allow_nan=False)
            except (OSError, ValueError):
                report = {}
                outcome.setdefault("failure", "unreadable report")
            if not isinstance(report, dict):
                report = {}
                outcome.setdefault("failure", "malformed report")
            outcome["report"] = report
            if report.get("token") != token or report.get("identity") != identity:
                outcome.setdefault("failure", "stale or mismatched report")
            elif (
                report.get("success") is not True
                or not isinstance(report.get("stages"), list)
                or not all(isinstance(stage, str) for stage in report["stages"])
                or report["stages"][-1:] != ["complete"]
            ):
                outcome.setdefault("failure", "scientific stage failed")
        else:
            outcome.setdefault("failure", "missing report")
        if process.returncode != 0:
            outcome.setdefault("failure", "process failed")
        outcome["success"] = "failure" not in outcome
    finally:
        outcome["elapsed_seconds"] = time.monotonic() - started
        if process.poll() is None:
            outcome.update(terminate_tree(process))
        if not outcome["success"]:
            log = work / "launcher-error.log"
            if log.stat().st_size == 0:
                log = work / "launcher.log"
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                diagnostic = stream.read().decode("utf-8", errors="replace")
            outcome["launcher_output_tail"] = re.sub(
                r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s'\"<>]+", "<path>", diagnostic
            )
        (work / "outcome.json").write_text(
            json.dumps(outcome, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if not outcome["success"]:
        raise RuntimeError(f"desktop acceptance failed: {outcome['failure']}")
    return outcome


def main() -> None:
    """Run an existing launcher; input generation and native lifecycle stay separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a fresh report destination")
    with tempfile.TemporaryDirectory(prefix="asymmetry-science-") as temporary:
        work = Path(temporary) / "invocation"
        try:
            run_acceptance(
                [str(args.executable.resolve())],
                args.inputs,
                json.loads(args.identity.read_text(encoding="utf-8")),
                work,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        finally:
            outcome = work / "outcome.json"
            if outcome.is_file():
                args.output.write_bytes(outcome.read_bytes())


if __name__ == "__main__":
    main()
