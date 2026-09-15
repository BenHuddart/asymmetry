"""Tests for ``tools/agent_eval/run_eval.py``, the agent-evaluation runner.

Nothing here invokes the real Claude Code CLI: the exit-code tests point
``--claude`` at a stub script written into ``tmp_path``, and the allow-list
tests are pure string construction. No research data and no private paths —
the "dataset" is an empty directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools" / "agent_eval" / "run_eval.py"


def _load_runner():
    """The evaluation runner, loaded from ``tools/`` by path."""
    spec = importlib.util.spec_from_file_location("asymmetry_run_eval", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- the allow-list ---------------------------------------------------------


def test_read_and_write_are_scoped_to_the_run_s_own_data_copy() -> None:
    """The agent may read and write inside its copy, and nowhere else by rule.

    Writes are granted as ``Edit(...)``: Claude Code consults ``Edit`` and
    ``Read`` path rules only, and accepts but never consults a
    ``Write(<path>)`` rule.
    """
    runner = _load_runner()

    allowed = runner.allowed_tools(Path("/private/tmp/evals/run-1/data"))

    assert "Read(//private/tmp/evals/run-1/data/**)" in allowed
    assert "Edit(//private/tmp/evals/run-1/data/**)" in allowed
    assert "Read(./**)" in allowed
    assert "Edit(./**)" in allowed


def test_no_file_tool_is_allowed_without_a_path_scope() -> None:
    """A bare ``Read``/``Write``/``Edit`` entry would make the scoping pointless."""
    runner = _load_runner()

    allowed = runner.allowed_tools(Path("/private/tmp/evals/run-1/data"))

    assert "Read" not in allowed
    assert "Write" not in allowed
    assert "Edit" not in allowed


def test_the_skill_and_the_analysis_cli_stay_allowed() -> None:
    """Without ``Skill`` the run measures nothing; without the CLI there is no analysis."""
    runner = _load_runner()

    allowed = runner.allowed_tools(Path("/private/tmp/evals/run-1/data"))

    assert "Skill" in allowed
    assert "Bash(asymmetry:*)" in allowed
    # Glob and Grep take their path in the field Claude Code refuses to match
    # rules against, so they cannot be scoped by an allow rule — see the
    # harness README.
    assert "Glob" in allowed
    assert "Grep" in allowed


def test_the_network_and_subagent_tools_are_denied_but_edit_is_not() -> None:
    """A bare ``Edit`` deny would also stop the scoped writes the allow-list grants."""
    runner = _load_runner()

    assert set(runner.DISALLOWED_TOOLS) >= {"WebFetch", "WebSearch", "Agent"}
    assert "Edit" not in runner.DISALLOWED_TOOLS


# -- the exit code ----------------------------------------------------------


def _stub_claude(tmp_path: Path, *, body: str) -> Path:
    """A fake Claude Code CLI: *body* is the shell it runs instead."""
    path = tmp_path / "claude-stub"
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def eval_inputs(tmp_path: Path):
    """``(data, out)`` for a run: an empty dataset folder and an unused output path."""
    data = tmp_path / "data-in"
    data.mkdir()
    (data / "README.txt").write_text("not a run file", encoding="utf-8")
    return data, tmp_path / "out"


def _run(runner, *, data: Path, out: Path, claude: Path) -> int:
    return runner.main(
        [
            "--data",
            str(data),
            "--rubric",
            "fmuf-ptfe",
            "--out",
            str(out),
            "--claude",
            str(claude),
            "--max-turns",
            "1",
        ]
    )


def test_a_failing_agent_exits_nonzero_and_still_writes_every_artefact(
    eval_inputs, tmp_path: Path
) -> None:
    """An auth error or a CLI crash must never be filed as a scored evaluation."""
    runner = _load_runner()
    data, out = eval_inputs
    claude = _stub_claude(tmp_path, body="echo 'not logged in' >&2\nexit 3")

    assert _run(runner, data=data, out=out, claude=claude) == 3

    cost = json.loads((out / "cost.json").read_text(encoding="utf-8"))
    assert cost["returncode"] == 3
    assert (out / "transcript.jsonl").is_file()
    assert (out / "summary.md").is_file()
    assert (out / "rubric.md").is_file()
    assert "not logged in" in (out / "agent-stderr.txt").read_text(encoding="utf-8")


def test_an_agent_that_produced_no_result_event_exits_nonzero(eval_inputs, tmp_path: Path) -> None:
    runner = _load_runner()
    data, out = eval_inputs
    claude = _stub_claude(tmp_path, body='echo \'{"type": "assistant"}\'\nexit 0')

    assert _run(runner, data=data, out=out, claude=claude) != 0

    cost = json.loads((out / "cost.json").read_text(encoding="utf-8"))
    assert cost["returncode"] == 0
    assert cost["n_events"] == 1


def test_a_clean_run_exits_zero(eval_inputs, tmp_path: Path) -> None:
    runner = _load_runner()
    data, out = eval_inputs
    event = json.dumps({"type": "result", "subtype": "success", "num_turns": 1})
    claude = _stub_claude(tmp_path, body=f"echo '{event}'\nexit 0")

    assert _run(runner, data=data, out=out, claude=claude) == 0

    cost = json.loads((out / "cost.json").read_text(encoding="utf-8"))
    assert cost["returncode"] == 0
    assert cost["result_subtype"] == "success"


def test_the_runner_hands_the_agent_the_scoped_allow_list(
    eval_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flags the subprocess is actually launched with, not just their construction."""
    runner = _load_runner()
    data, out = eval_inputs
    recorded = tmp_path / "argv.txt"
    claude = _stub_claude(tmp_path, body=f'printf "%s\\n" "$@" > {recorded}\nexit 0')

    _run(runner, data=data, out=out, claude=claude)

    argv = recorded.read_text(encoding="utf-8").split("\n")
    work = out / "data"
    assert f"Read(//{str(work).lstrip(os.sep)}/**)" in argv
    assert f"Edit(//{str(work).lstrip(os.sep)}/**)" in argv
    assert "--disallowedTools" in argv
    assert "WebFetch" in argv
