"""Tests for the packaged ``asymmetry-analysis`` skill and its install/check/uninstall CLI.

Covers the two things Phase 3 adds around the skill: that it ships as package
data with valid frontmatter (:mod:`asymmetry.cli.skill`'s
:func:`~asymmetry.cli.skill.skill_source_dir`, located exactly as the module
documents — via :mod:`importlib.resources`), and that ``install``/``check``/
``uninstall`` behave as :doc:`docs/plans/agent-cli-skill.md` specifies. No
research data or private paths — everything here is a ``tmp_path``.
"""

from __future__ import annotations

import argparse
import importlib.resources
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from asymmetry import __version__, cli
from asymmetry.cli import skill
from asymmetry.cli._output import UserError

ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = ROOT / "tools" / "agent_eval" / "render_command_reference.py"


def _load_renderer():
    """The command-reference generator, loaded from ``tools/`` by path."""
    spec = importlib.util.spec_from_file_location("asymmetry_command_reference", RENDERER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _subcommand_names() -> list[str]:
    """Every subcommand the CLI registers, top level and nested."""
    names: list[str] = []

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:  # noqa: SLF001 — argparse exposes no public walk
            if isinstance(action, argparse._SubParsersAction):
                for name, subparser in action.choices.items():
                    names.append(name)
                    walk(subparser)

    walk(cli.build_parser())
    return names


def _parse_frontmatter(text: str) -> dict[str, str]:
    """A minimal ``---``-delimited YAML-frontmatter parser (``key: value`` only).

    Skills use plain scalar frontmatter, so a full YAML parser is not needed
    here; this reads exactly what :func:`test_skill_frontmatter_has_name_and_description`
    checks for.
    """
    assert text.startswith("---\n")
    _, _, rest = text.partition("---\n")
    body, _, _ = rest.partition("\n---")
    fields: dict[str, str] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        assert separator, f"Frontmatter line {line!r} is not key: value"
        fields[key.strip()] = value.strip()
    return fields


# -- packaging ----------------------------------------------------------


def test_skill_is_reachable_via_importlib_resources() -> None:
    packaged = importlib.resources.files("asymmetry.resources") / "skills" / "asymmetry-analysis"
    assert (packaged / "SKILL.md").is_file()
    assert skill.skill_source_dir() == Path(str(packaged))


def test_skill_frontmatter_has_a_name_and_a_description() -> None:
    text = (skill.skill_source_dir() / "SKILL.md").read_text(encoding="utf-8")
    fields = _parse_frontmatter(text)
    assert fields["name"] == "asymmetry-analysis"
    assert fields["description"]
    # The description is the trigger text an agent's skill-selection reads;
    # it must actually say what the skill is for.
    assert "asymmetry" in fields["description"].lower()
    # Skill descriptions are matched, not read: an over-long one is truncated
    # by the agent before it ever reaches the selection.
    assert len(fields["description"]) <= 300


def test_skill_body_names_every_subcommand() -> None:
    """A command the skill never mentions is a command the agent never runs."""
    body = (skill.skill_source_dir() / "SKILL.md").read_text(encoding="utf-8")

    missing = [name for name in _subcommand_names() if name not in body]

    assert missing == []


# -- references/commands.md ------------------------------------------------


def test_command_reference_is_current() -> None:
    """The bundled ``--help`` reference matches a fresh render of the parser.

    Regenerate with ``python tools/agent_eval/render_command_reference.py``
    whenever a flag changes; a stale reference misleads the agent about the
    exact spelling of an option, which is the one thing it cannot guess.
    """
    renderer = _load_renderer()
    bundled = (skill.skill_source_dir() / "references" / "commands.md").read_text(encoding="utf-8")

    assert bundled == renderer.render()


def test_command_reference_ships_with_the_skill(tmp_path: Path) -> None:
    result = skill.install("claude", into=tmp_path)

    assert (result.path / "references" / "commands.md").is_file()


# -- install / uninstall (asymmetry.cli.skill) ---------------------------


def test_install_writes_skill_md_and_a_manifest(tmp_path: Path) -> None:
    result = skill.install("claude", into=tmp_path)

    assert result.agent == "claude"
    assert result.path == tmp_path / "asymmetry-analysis"
    assert (result.path / "SKILL.md").is_file()
    manifest = json.loads((result.path / skill.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == result.manifest
    assert manifest["schema"] == 1
    assert manifest["asymmetry_version"] == __version__
    assert manifest["agent"] == "claude"
    assert manifest["installed_at"]


def test_a_second_install_overwrites_the_first(tmp_path: Path) -> None:
    first = skill.install("claude", into=tmp_path)
    stray = first.path / "leftover.txt"
    stray.write_text("stale", encoding="utf-8")

    second = skill.install("claude", into=tmp_path)

    assert second.path == first.path
    assert not stray.exists()
    assert (second.path / "SKILL.md").is_file()


def test_a_foreign_directory_is_refused_without_force_and_accepted_with_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asymmetry-analysis"
    target.mkdir()
    (target / "not-ours.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(UserError, match="not removing it|not written by"):
        skill.install("claude", into=tmp_path)
    with pytest.raises(UserError):
        skill.uninstall("claude", into=tmp_path)

    result = skill.install("claude", into=tmp_path, force=True)
    assert not (result.path / "not-ours.txt").exists()
    assert (result.path / "SKILL.md").is_file()


def test_uninstall_removes_only_our_directory(tmp_path: Path) -> None:
    result = skill.install("claude", into=tmp_path)

    removed = skill.uninstall("claude", into=tmp_path)

    assert removed == result.path
    assert not result.path.exists()


def test_uninstall_refuses_a_directory_with_no_manifest(tmp_path: Path) -> None:
    target = tmp_path / "asymmetry-analysis"
    target.mkdir()
    with pytest.raises(UserError):
        skill.uninstall("claude", into=tmp_path)
    assert target.exists()  # never touched


def test_target_dir_rejects_an_unknown_agent(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="Unknown agent"):
        skill.target_dir("gpt", into=tmp_path)


# -- check ----------------------------------------------------------------


def test_check_reports_a_version_mismatch_after_the_manifest_is_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Nothing installed yet: informational, not a failure, as long as the
    # environment otherwise has matplotlib and a loader.
    report = skill.run_check(("claude",))
    assert report["skills"]["claude"]["installed"] is False
    assert report["skills"]["claude"]["up_to_date"] is True

    result = skill.install("claude")  # into the monkeypatched home
    report = skill.run_check(("claude",))
    assert report["skills"]["claude"] == {
        "installed": True,
        "path": str(result.path),
        "version": __version__,
        "up_to_date": True,
    }
    assert report["ok"] is True

    manifest_path = result.path / skill.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["asymmetry_version"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = skill.run_check(("claude",))
    assert report["skills"]["claude"]["up_to_date"] is False
    assert report["ok"] is False
    assert any("rerun 'asymmetry skill install'" in line["text"] for line in report["lines"])


# -- the CLI wiring (asymmetry skill <sub>) --------------------------------


def _json_output(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_cli_skill_install_check_uninstall_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cli.main(["skill", "install", "--agent", "claude", "--json"])
    installed = _json_output(capsys)["install"]
    assert Path(installed["path"]).is_file() is False
    assert Path(installed["path"], "SKILL.md").is_file()

    cli.main(["skill", "check", "--agent", "claude", "--json"])
    checked = _json_output(capsys)["check"]
    assert checked["skills"]["claude"]["up_to_date"] is True

    cli.main(["skill", "uninstall", "--agent", "claude", "--json"])
    uninstalled = _json_output(capsys)["agent"]
    assert uninstalled == "claude"
    assert not Path(installed["path"]).exists()
