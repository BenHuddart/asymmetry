"""Retain Linux's runner-wide OOM counter without attributing a victim process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def record_oom_counter(directory: Path, phase: str, vmstat: Path = Path("/proc/vmstat")) -> None:
    record: dict[str, object] = {
        "source": str(vmstat),
        "status": "unavailable",
        "oom_kill": None,
    }
    try:
        lines = vmstat.read_text().splitlines()
    except OSError as error:
        record["reason"] = type(error).__name__
    else:
        counter = next((line.split() for line in lines if line.split()[:1] == ["oom_kill"]), [])
        if len(counter) == 2 and counter[1].isascii() and counter[1].isdecimal():
            record.update(status="observed", oom_kill=int(counter[1]))
        else:
            record["reason"] = "missing or invalid oom_kill counter"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"oom-{phase}.json").write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    record_oom_counter(args.directory, args.phase)
