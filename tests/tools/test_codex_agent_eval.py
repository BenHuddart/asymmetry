"""Tests for the Codex/Luna agent-evaluation runner.

These are construction and transcript-parsing tests only. They never invoke a
real model or copy research data.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools" / "agent_eval" / "run_codex_eval.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("asymmetry_run_codex_eval", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(runner, tmp_path: Path):
    data = tmp_path / "input"
    data.mkdir()
    return runner.parse_args(
        [
            "--data",
            str(data),
            "--rubric",
            "fmuf-ptfe",
            "--out",
            str(tmp_path / "output"),
            "--codex",
            sys.executable,
        ]
    )


def test_codex_runner_defaults_to_luna(tmp_path: Path) -> None:
    runner = _load_runner()

    args = _args(runner, tmp_path)

    assert args.model == "gpt-5.6-luna"


def test_command_is_persistent_and_uses_the_controlled_workspace(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(runner, tmp_path)
    data = args.out / "data"
    project = args.out / "project"

    command = runner.build_command(args, data, project)

    assert command[:2] == [sys.executable, "exec"]
    assert "--ephemeral" not in command
    assert "--ignore-user-config" in command
    assert "--approve-for-me" in command
    enabled = [command[index + 1] for index, value in enumerate(command) if value == "--enable"]
    assert enabled == ["view_image", "skip_host_skill_discovery"]
    assert "--sandbox" not in command
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert command[command.index("--cd") + 1] == str(project)
    assert str(data) in command[-1]
    assert runner.IMAGE_REVIEW_START in command[-1]


def test_review_images_are_project_local_hashed_and_capped(tmp_path: Path) -> None:
    runner = _load_runner()
    project = tmp_path / "project"
    plots = project / "asymmetry-work" / "plots"
    plots.mkdir(parents=True)
    for index in range(1, 6):
        (plots / f"plot-{index}.png").write_bytes(f"image-{index}".encode())
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    manifest = "\n".join(
        [
            runner.IMAGE_REVIEW_START,
            *(f"asymmetry-work/plots/plot-{index}.png" for index in range(1, 6)),
            "../outside.png",
            runner.IMAGE_REVIEW_END,
        ]
    )
    events = [
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": manifest},
        }
    ]

    images = runner.review_images(events, project)

    assert [item["relative_path"] for item in images] == [
        f"asymmetry-work/plots/plot-{index}.png" for index in range(1, 5)
    ]
    assert all(len(item["sha256"]) == 64 for item in images)


def test_resume_command_attaches_each_nominated_image(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(runner, tmp_path)
    project = args.out / "project"
    image = project / "asymmetry-work" / "plots" / "fft.png"
    images = [{"relative_path": "asymmetry-work/plots/fft.png", "path": image}]

    command = runner.build_resume_command(args, project, "thread-1", images)

    assert command[:3] == [sys.executable, "exec", "resume"]
    enabled = [command[index + 1] for index, value in enumerate(command) if value == "--enable"]
    assert enabled == ["view_image", "skip_host_skill_discovery"]
    assert command[command.index("--image") + 1] == str(image)
    assert "thread-1" in command
    assert "asymmetry-work/plots/fft.png" in command[-1]


def test_codex_jsonl_extracts_messages_commands_and_usage() -> None:
    runner = _load_runner()
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {
                "id": "item-1",
                "type": "command_execution",
                "command": "asymmetry survey data",
                "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "item-2", "type": "agent_message", "text": "Result."},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 120, "output_tokens": 30},
        },
    ]

    assert runner.extract_commands(events) == ["asymmetry survey data"]
    assert runner.assistant_texts(events) == ["Result."]
    assert runner.usage(events) == {"input_tokens": 120, "output_tokens": 30}
    assert runner.has_completed_turn(events)
    assert runner.thread_id(events) == "thread-1"


def test_usage_is_summed_across_both_turns() -> None:
    runner = _load_runner()
    events = [
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
        {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 3}},
    ]

    assert runner.usage(events) == {"input_tokens": 30, "output_tokens": 5}
    assert runner.completed_turns(events) == 2


def test_hdf4_runtime_is_forwarded_to_the_agent(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(runner, tmp_path)
    runtime = tmp_path / "hdf4-runtime"
    runtime.mkdir()
    args.hdf4_dll_dir = runtime

    assert runner.agent_env(args)["ASYMMETRY_HDF4_DLL_DIR"] == str(runtime)


def test_summary_prefers_the_explicit_last_message(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(runner, tmp_path)
    args.out.mkdir()
    (args.out / "last-message.md").write_text("Final report\n", encoding="utf-8")
    events = [
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Intermediate"},
        }
    ]

    assert runner.extract_summary(args, events) == "Final report"


def test_cost_record_names_codex_and_exact_model(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(runner, tmp_path)
    data = args.out / "data"
    project = args.out / "project"
    data.mkdir(parents=True)
    (project / ".agents" / "skills" / "asymmetry-analysis").mkdir(parents=True)
    (args.out / "last-message.md").write_text("A report", encoding="utf-8")
    events = [
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
        {"type": "turn.completed", "usage": {"input_tokens": 2}},
    ]

    runner.write_outputs(args, data, project, events, 1.25, 0, [])

    record = json.loads((args.out / "cost.json").read_text(encoding="utf-8"))
    assert record["agent"] == "Codex"
    assert record["model"] == "gpt-5.6-luna"
    assert record["turn_completed"] is True
    assert record["turns_completed"] == 2
    assert record["protocol"] == "two_turn_image_review_v1"
    assert record["skill_installed"] is True


def test_report_replaces_characters_the_console_cannot_encode(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    args = _args(runner, tmp_path)
    args.out.mkdir()
    (args.out / "cost.json").write_text(
        json.dumps(
            {
                "wall_seconds": 1.0,
                "skill_installed": True,
                "skill_invoked": True,
                "returncode": 0,
                "turn_completed": True,
            }
        ),
        encoding="utf-8",
    )
    rubric = tmp_path / "rubric.md"
    rubric.write_text("χ² and µs⁻¹", encoding="utf-8")
    args.rubric_path = rubric
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)

    runner.report(args)

    stream.flush()
