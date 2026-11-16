"""Execute one pinned external START module in a separate Python process."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PULSE_IDENTITY = Path(__file__).parent / "fixtures" / "reference_cases" / "pulse-source.json"


def run_start(source: Path, n_pulses: int) -> list[list[float]]:
    identity = json.loads(PULSE_IDENTITY.read_text())
    module = source / identity["module"]
    assert hashlib.sha256(module.read_bytes()).hexdigest() == identity["sha256"], (
        "Mantid START module identity mismatch"
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    assert commit == identity["commit"], "Mantid source checkout identity mismatch"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(module), str(n_pulses)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    response = json.loads(result.stdout)
    assert len(response) == identity["settings"]["n_spectrum"], "empty or truncated START output"
    assert all(len(row) == 2 for row in response), "invalid START complex output"
    return response


if __name__ == "__main__":
    import importlib.util

    import numpy as np

    class SilentLog:
        def notice(self, *_args, **_kwargs):
            pass

    settings = json.loads(PULSE_IDENTITY.read_text())["settings"]
    spec = importlib.util.spec_from_file_location("_external_start", sys.argv[1])
    assert spec is not None and spec.loader is not None, "unloadable START module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, response = module.START(
        settings["npts"],
        int(sys.argv[2]),
        settings["resolution_us"],
        settings["n_spectrum"],
        settings["fine_offset_us"],
        SilentLog(),
    )
    print(json.dumps(np.column_stack((response.real, response.imag)).tolist(), allow_nan=False))
