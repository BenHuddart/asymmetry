#!/usr/bin/env python3
"""PostToolUse hook: run the structural harness after edits that can affect a
repository-shape invariant, and feed any violation back to the agent.

This wraps `tools/harness.py structural` — the single source of truth that CI
also runs — so it never re-implements a check. It only adds a fast feedback
loop: instead of waiting for the agent to remember to run the harness (or for
CI), a boundary violation surfaces immediately after the offending edit.

Input: the PostToolUse JSON payload on stdin (tool_name, tool_input, ...).
Output / exit codes:
- 0: edit is irrelevant to structural invariants, or the checks pass (silent).
- 2: structural checks failed; the failure text is written to stderr, which
     Claude Code feeds back to the agent so it can fix the violation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Editing a file under one of these prefixes (or one of the named files) can
# change a structural invariant: the core import boundary, the core dependency
# boundary, or the porting-study layout. Edits anywhere else cannot break
# `harness.py structural`, so the hook stays silent and fast for them.
RELEVANT_PREFIXES = (
    "src/asymmetry/core/",
    "docs/porting/",
)
RELEVANT_FILES = ("pyproject.toml",)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # No usable payload — do nothing rather than risk spurious noise.
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return 0

    repo_root = Path(__file__).resolve().parents[2]
    try:
        rel = Path(file_path).resolve().relative_to(repo_root).as_posix()
    except ValueError:
        # Edit outside the repo tree — nothing structural to check.
        return 0

    relevant = rel in RELEVANT_FILES or any(rel.startswith(p) for p in RELEVANT_PREFIXES)
    if not relevant:
        return 0

    env = os.environ.copy()
    # The structural checks are pure-stdlib, so skip the harness venv re-exec:
    # it would only add latency and a re-exec banner on stderr.
    env["ASYMMETRY_HARNESS_NO_VENV"] = "1"
    result = subprocess.run(
        [sys.executable, "tools/harness.py", "structural"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return 0

    sys.stderr.write(
        "Structural harness check failed after this edit "
        "(`python tools/harness.py structural`):\n\n"
    )
    sys.stderr.write((result.stdout or "") + (result.stderr or ""))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
