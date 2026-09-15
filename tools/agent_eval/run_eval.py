#!/usr/bin/env python3
"""Run one agent evaluation of the ``asymmetry-analysis`` skill.

Copies a dataset folder into the output directory (the corpus itself is never
written to), makes an empty project directory beside it holding only the
installed skill, runs the Claude Code CLI headless *in the project directory*
with the fixed analysis prompt and the data copy's path, and saves everything
a human needs to tick the dataset's rubric:

``data/``             the read-only copy of the dataset the agent analysed
``project/``          the directory the agent worked in (its ``asymmetry-work/``
                      and the installed skill)
``workdir/``          the work directory the agent built, copied out
``transcript.jsonl``  the raw ``stream-json`` event stream
``summary.md``        the agent's report — the only thing the rubric scores
``assistant-text.md`` every assistant message, to check the report was picked right
``commands.txt``      every Bash command the agent ran, in order
``cost.json``         usage, cost, wall time, exit code, permission denials
``rubric.md``         the dataset's rubric, to tick

This script itself creates, modifies and removes nothing outside the output
directory, and the agent's own file tools are scoped to those two directories
— the data copy readable, the project writable (see :func:`allowed_tools`), so
that the skill's "never write into the data folder" rule is enforced rather
than merely asked for: an attempt shows up as a ``permission_denials`` entry
in ``cost.json``. It exits nonzero when the agent did not finish, so a broken
run is never filed as a scoreable one.

The agent it launches is an ordinary Claude Code session, so that
session's own state (its transcript and any auto-memory it writes) lands under
``~/.claude/`` keyed on the project directory, exactly as for a hand-run
session in that folder.

Usage::

    python tools/agent_eval/run_eval.py \\
        --data "~/Documents/WiMDA muon school/.../Data" \\
        --rubric fmuf-ptfe --out /tmp/evals/pass1-fmuf-ptfe
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"


def asymmetry_command() -> list[str]:
    """How to invoke the ``asymmetry`` CLI from this interpreter's environment.

    The console script installed beside the running interpreter when there is
    one (a venv, a pipx environment, a CI install), else the package's
    ``python -m asymmetry`` entry point, which works wherever the package
    imports. Nothing here assumes a ``.venv`` at the repository root.
    """
    script = Path(sys.executable).parent / "asymmetry"
    if script.exists():
        return [str(script)]
    return [sys.executable, "-m", "asymmetry"]


def cli_bin_dir() -> Path:
    """The directory the agent needs first on ``PATH`` to find ``asymmetry``."""
    return Path(sys.executable).parent


#: The prompt the plan fixes for every evaluation.
DEFAULT_PROMPT = (
    "This directory contains the data from a recent muSR experiment. Can you "
    "analyse these using Asymmetry and present me a summary of what they show?"
)

#: Bash commands the agent may run: the analysis CLI and two read-only
#: helpers, so the eval measures the skill rather than the agent's ability to
#: reach around it (a Python one-liner could invent any number).
_ALLOWED_BASH = ("Bash(asymmetry:*)", "Bash(ls:*)", "Bash(cat:*)")

#: Tools allowed without a path scope. ``Skill`` is how a Claude Code agent
#: loads the installed skill at all — without it the skill is listed at
#: startup and can never be read. ``Glob`` and ``Grep`` take their path in the
#: field Claude Code refuses to match rules against, so neither can be scoped
#: by an allow rule (see the README); they are allowed as they are.
_ALLOWED_UNSCOPED = ("Glob", "Grep", "Skill")

#: Tools denied outright, so a run cannot reach the network or spawn helpers
#: whose own tool use this harness would never see. ``--tools`` already leaves
#: them out of the session; these say so a second time, at the permission
#: layer. ``Edit`` is deliberately *not* here: Claude Code checks file writes
#: against ``Edit(<path>)`` rules, so a bare ``Edit`` deny would also stop the
#: scoped writes :func:`allowed_tools` grants.
DISALLOWED_TOOLS = ("WebFetch", "WebSearch", "Agent", "Task")

#: Built-in tools the session is given at all, before the allow-list narrows it.
TOOLS = ("Bash", "Read", "Glob", "Grep", "Write", "Skill")

#: The work directory the CLI writes, looked for under the project directory.
WORKDIR_NAME = "asymmetry-work"


def _absolute_pattern(path: Path) -> str:
    """*path* as a permission-rule pattern covering it and everything under it.

    A single leading slash anchors a rule at its settings source, not at the
    filesystem root; ``//`` is the absolute form (see the permissions
    documentation, "Read and Edit").
    """
    return "//" + str(path).lstrip("/") + "/**"


def allowed_tools(data: Path) -> tuple[str, ...]:
    """The ``--allowedTools`` list for a run analysing the copy at *data*.

    The agent runs in the project directory, so ``./**`` is that directory:
    it may read and write there, and it may **read** the data copy and not
    write it. Nothing else is granted, so an agent cannot read this repository
    (its rubrics included) or write outside the project without a permission
    prompt, which in headless mode is a refusal recorded in ``cost.json``.
    That the data copy is readable but not writable is the point: the skill
    tells the agent never to write into the data folder, and a run that tries
    leaves a ``permission_denials`` entry saying so. Writes are granted as
    ``Edit(...)``: Claude Code consults ``Edit`` and ``Read`` path rules only,
    and accepts but never consults a ``Write(<path>)`` rule.
    """
    return (
        *_ALLOWED_BASH,
        f"Read({_absolute_pattern(data)})",
        "Read(./**)",
        "Edit(./**)",
        *_ALLOWED_UNSCOPED,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments, with the paths resolved."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Dataset folder to copy and analyse")
    parser.add_argument(
        "--rubric",
        required=True,
        help=f"Rubric name (a file stem in {RUBRIC_DIR})",
    )
    parser.add_argument("--out", required=True, help="Output directory; must not already exist")
    parser.add_argument("--model", default="sonnet", help="Model alias for the agent")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to hand the agent")
    parser.add_argument("--max-turns", type=int, default=80, help="Turn budget for the agent")
    parser.add_argument(
        "--claude",
        default=str(Path.home() / ".local" / "bin" / "claude"),
        help="Path to the Claude Code CLI",
    )
    args = parser.parse_args(argv)
    args.data = Path(args.data).expanduser().resolve()
    args.out = Path(args.out).expanduser().resolve()
    args.rubric_path = RUBRIC_DIR / f"{args.rubric}.md"
    return args


def check_inputs(args: argparse.Namespace) -> None:
    """Fail before anything is written if an input is not what it must be."""
    if not args.data.is_dir():
        sys.exit(f"--data {args.data} is not a directory")
    if not args.rubric_path.is_file():
        available = ", ".join(sorted(p.stem for p in RUBRIC_DIR.glob("*.md")))
        sys.exit(f"--rubric {args.rubric!r} has no file in {RUBRIC_DIR} (have: {available})")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"--out {args.out} already exists and is not empty; choose another directory")
    if not Path(args.claude).exists():
        sys.exit(f"Claude Code CLI not found at {args.claude}; pass --claude")


def stage(args: argparse.Namespace) -> tuple[Path, Path]:
    """Copy the dataset to ``<out>/data`` and build the agent's ``<out>/project``.

    The project directory starts out holding nothing but the installed skill —
    it is the directory an analyst would open beside their data, and where the
    work directory is expected to appear.
    """
    data = args.out / "data"
    project = args.out / "project"
    data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.data, data)
    project.mkdir()
    subprocess.run(
        [*asymmetry_command(), "skill", "install", "--agent", "claude", "--project"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return data, project


def agent_env() -> dict[str, str]:
    """The environment the agent runs in: this interpreter's ``bin`` first on ``PATH``."""
    env = dict(os.environ)
    env["PATH"] = f"{cli_bin_dir()}{os.pathsep}{env.get('PATH', '')}"
    return env


def full_prompt(args: argparse.Namespace, data: Path) -> str:
    """The prompt the agent is handed, naming where the data is.

    The agent's own directory no longer holds the runs, so the path has to be
    said. For the plan's fixed sentence that means "The directory <path>
    contains ..." in place of "This directory contains ..." — the same
    request, pointed at the copy; a custom prompt gets the path stated in
    front of it. The whole thing is recorded in ``cost.json``.
    """
    if args.prompt == DEFAULT_PROMPT:
        return DEFAULT_PROMPT.replace(
            "This directory contains", f"The directory {data} contains", 1
        )
    return f"The data is in {data}. {args.prompt}"


def run_agent(args: argparse.Namespace, data: Path, project: Path) -> tuple[list[dict], float, int]:
    """Run the agent in *project*, streaming its events to ``transcript.jsonl``.

    Returns the parsed events, the wall time in seconds, and the CLI's exit
    code — a failed run (an auth error, a CLI crash) produces a transcript
    that looks merely short, so the exit code is the only thing that tells it
    from an evaluation worth scoring.
    """
    command = [
        args.claude,
        "-p",
        full_prompt(args, data),
        "--model",
        args.model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(args.max_turns),
        "--setting-sources",
        "project",
        "--tools",
        *TOOLS,
        "--allowedTools",
        *allowed_tools(data),
        "--disallowedTools",
        *DISALLOWED_TOOLS,
    ]
    transcript = args.out / "transcript.jsonl"
    events: list[dict] = []
    started = time.monotonic()
    # The agent's stderr goes straight to its own file rather than a pipe:
    # this process only drains stdout, so a pipe that filled would deadlock a
    # long run.
    with (
        transcript.open("w", encoding="utf-8") as stream,
        (args.out / "agent-stderr.txt").open("w", encoding="utf-8") as errors,
    ):
        process = subprocess.Popen(
            command,
            cwd=project,
            env=agent_env(),
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            line = line.strip()
            if line.startswith("{"):
                events.append(json.loads(line))
                print(f"  ... {len(events)} events", end="\r", file=sys.stderr)
        returncode = process.wait()
    return events, time.monotonic() - started, returncode


def _content_blocks(event: dict) -> list[dict]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def extract_commands(events: list[dict]) -> list[str]:
    """Every Bash command the agent ran, in order."""
    commands = []
    for event in events:
        for block in _content_blocks(event):
            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                commands.append(str(block.get("input", {}).get("command", "")))
    return commands


def assistant_texts(events: list[dict]) -> list[str]:
    """Every assistant text block, in order."""
    return [
        block.get("text", "")
        for event in events
        if event.get("type") == "assistant"
        for block in _content_blocks(event)
        if block.get("type") == "text" and block.get("text", "").strip()
    ]


def extract_summary(events: list[dict]) -> str:
    """The agent's report — what the rubric scores.

    The report is the longest assistant message plus everything the agent said
    after it. Taking the ``result`` event alone is not enough: an agent that
    writes its summary and then does one more thing (saving a note, tidying up)
    ends the run on a one-line sign-off, and the report is the message before
    it. The longest message is the report in every run observed.
    """
    texts = assistant_texts(events)
    if not texts:
        return ""
    start = max(range(len(texts)), key=lambda index: len(texts[index]))
    return "\n\n".join(texts[start:])


def skill_was_invoked(events: list[dict]) -> bool:
    """Whether the agent actually loaded the skill (a ``Skill`` tool use).

    Being *listed* at init only means the skill was installed and discovered;
    the trigger check is whether the agent chose to read it.
    """
    for event in events:
        for block in _content_blocks(event):
            if block.get("type") == "tool_use" and block.get("name") == "Skill":
                if "asymmetry-analysis" in json.dumps(block.get("input", {})):
                    return True
    return False


def skill_was_available(events: list[dict]) -> bool:
    """Whether the installed skill was discovered at session start."""
    return any(
        event.get("subtype") == "init" and "asymmetry-analysis" in (event.get("skills") or [])
        for event in events
    )


def write_outputs(
    args: argparse.Namespace,
    data: Path,
    project: Path,
    events: list[dict],
    elapsed: float,
    returncode: int,
) -> None:
    """Save the summary, the command list, the cost record, the work directory and the rubric.

    Written for a failed run too: the transcript and stderr of a run that
    crashed are exactly what says why, and ``cost.json`` records *returncode*
    so a later reader can tell a scoreable evaluation from a broken one.
    """
    (args.out / "summary.md").write_text(extract_summary(events) + "\n", encoding="utf-8")
    (args.out / "assistant-text.md").write_text(
        "\n\n---\n\n".join(assistant_texts(events)) + "\n", encoding="utf-8"
    )
    (args.out / "commands.txt").write_text(
        "\n".join(extract_commands(events)) + "\n", encoding="utf-8"
    )

    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    cost = {
        "dataset": str(args.data),
        "rubric": args.rubric,
        "model": args.model,
        "prompt": full_prompt(args, data),
        "max_turns": args.max_turns,
        "returncode": returncode,
        "wall_seconds": round(elapsed, 1),
        "num_turns": result.get("num_turns"),
        "total_cost_usd": result.get("total_cost_usd"),
        "usage": result.get("usage"),
        "is_error": result.get("is_error"),
        "result_subtype": result.get("subtype"),
        "permission_denials": result.get("permission_denials"),
        "skill_available": skill_was_available(events),
        "skill_invoked": skill_was_invoked(events),
        "n_events": len(events),
    }
    (args.out / "cost.json").write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")

    produced = project / WORKDIR_NAME
    if produced.is_dir():
        shutil.copytree(produced, args.out / "workdir")

    shutil.copy2(args.rubric_path, args.out / "rubric.md")


def has_result(events: list[dict]) -> bool:
    """Whether the stream carried a ``result`` event — the agent's own sign-off.

    Without one the run ended some other way (killed, a CLI failure part-way
    through), and there is no evaluation to score however much text arrived.
    """
    return any(event.get("type") == "result" for event in events)


def report(args: argparse.Namespace) -> None:
    """Print the rubric to tick and where the evidence is."""
    cost = json.loads((args.out / "cost.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 72)
    print(
        f"{args.rubric} — {args.model} — {cost['wall_seconds']} s, "
        f"{cost['num_turns']} turns, ${cost['total_cost_usd']}"
    )
    print(f"skill available: {cost['skill_available']}, invoked: {cost['skill_invoked']}")
    if cost["returncode"] != 0:
        print(f"claude exited {cost['returncode']} — this run is NOT scoreable")
    if cost["permission_denials"]:
        print(f"permission denials: {len(cost['permission_denials'])}")
    print("=" * 72)
    print(args.rubric_path.read_text(encoding="utf-8"))
    print("=" * 72)
    print(f"summary to score : {args.out / 'summary.md'}")
    print(f"commands run     : {args.out / 'commands.txt'}")
    print(f"transcript       : {args.out / 'transcript.jsonl'}")
    print(f"work directory   : {args.out / 'workdir'}")
    print(f"project directory: {args.out / 'project'}")
    print(f"rubric to tick   : {args.out / 'rubric.md'}")


def main(argv: list[str] | None = None) -> int:
    """Stage, run, save, report. Returns an exit code.

    Nonzero when the agent exited nonzero or never reached a ``result`` event:
    a run that failed to start (an expired login, a CLI that crashed) must not
    be mistaken for an evaluation whose agent simply said very little. Every
    artefact is written either way — the evidence of *why* it failed is in
    them.
    """
    args = parse_args(argv)
    check_inputs(args)

    print(f"staging {args.data.name} -> {args.out / 'data'}", file=sys.stderr)
    data, project = stage(args)
    print(f"running {args.model} in {project}", file=sys.stderr)
    events, elapsed, returncode = run_agent(args, data, project)
    write_outputs(args, data, project, events, elapsed, returncode)
    report(args)

    if returncode != 0:
        print(
            f"claude exited {returncode}; see {args.out / 'agent-stderr.txt'}. "
            "Nothing here is scoreable.",
            file=sys.stderr,
        )
        return returncode
    if not has_result(events):
        print(
            f"the agent produced no result event ({len(events)} events); see "
            f"{args.out / 'transcript.jsonl'}. Nothing here is scoreable.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
