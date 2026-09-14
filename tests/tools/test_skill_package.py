"""Tests for the packaged ``asymmetry-analysis`` skill and its install/check/uninstall CLI.

Covers the two things Phase 3 adds around the skill: that it ships as package
data with valid frontmatter (:mod:`asymmetry.cli.skill`'s
:func:`~asymmetry.cli.skill.skill_source_dir`, located exactly as the module
documents — via :mod:`importlib.resources`), and that ``install``/``check``/
``uninstall`` behave as :doc:`docs/plans/agent-cli-skill.md` specifies. No
research data or private paths — everything here is a ``tmp_path``.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import pytest

from asymmetry import __version__, cli
from asymmetry.cli import skill
from asymmetry.cli._output import UserError


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
