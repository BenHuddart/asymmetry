"""Packaging and installing the ``asymmetry-analysis`` agent skill.

The skill ships as package data
(``asymmetry/resources/skills/asymmetry-analysis/``); this module copies it
into an agent's skill directory and stamps a manifest
(``.asymmetry-skill.json``) recording which version installed it, so
``check`` can tell a stale copy from a current one and ``install``/
``uninstall`` can tell a directory this command wrote from one that just
happens to be in the way.

``install --link`` is the development variant: instead of copying, it points a
symlink at :func:`skill_source_dir`, so edits in a checkout reach the agent
without reinstalling. A link carries no manifest — nothing is written into the
package directory — and needs none: a symlink resolving to the packaged skill
*is* the proof this command made it, and it is always current by construction.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asymmetry import __version__
from asymmetry.cli._output import UserError

#: Name of the skill this CLI ships, and its folder under ``resources/skills/``.
SKILL_NAME = "asymmetry-analysis"

#: Manifest file stamped beside ``SKILL.md`` on install.
MANIFEST_NAME = ".asymmetry-skill.json"

#: Schema version of the manifest this command writes and recognises.
MANIFEST_SCHEMA = 1

#: The agent skill ecosystems this CLI installs into.
AGENTS = ("claude", "codex")

#: Skill directory for each agent, relative to its root (the home directory,
#: or the current directory for ``--project``) — the same relative layout
#: either way.
_AGENT_SKILL_SUBDIR = {"claude": Path(".claude") / "skills", "codex": Path(".agents") / "skills"}

#: Optional loader modules ``check`` reports on, and the extra that installs each.
LOADER_MODULES = {"h5py": "hdf5", "pyhdf": "hdf4", "uproot": "root"}


def skill_source_dir() -> Path:
    """Where the packaged skill lives on disk (an editable checkout or an install)."""
    return Path(str(importlib.resources.files("asymmetry.resources") / "skills" / SKILL_NAME))


def target_dir(agent: str, *, project: bool = False, into: str | Path | None = None) -> Path:
    """Where the skill is (or would be) installed for *agent*.

    Raises :class:`UserError` for an *agent* outside :data:`AGENTS`.
    """
    if agent not in AGENTS:
        raise UserError(f"Unknown agent {agent!r}; expected one of {', '.join(AGENTS)}.")
    if into is not None:
        return Path(into) / SKILL_NAME
    root = Path.cwd() if project else Path.home()
    return root / _AGENT_SKILL_SUBDIR[agent] / SKILL_NAME


def _manifest_path(target: Path) -> Path:
    return target / MANIFEST_NAME


def read_manifest(target: Path) -> dict[str, Any] | None:
    """The manifest *this command* wrote in *target*, or ``None`` when there is none.

    ``install --force`` and ``uninstall`` ``rmtree`` the directory this returns
    a manifest for, so "there is a readable JSON file at the manifest path" is
    not a good enough answer: it would hand a foreign directory that happens to
    hold a ``.asymmetry-skill.json`` to :func:`shutil.rmtree`. The file is
    parsed as the record :func:`install` writes — a JSON object carrying
    ``schema`` :data:`MANIFEST_SCHEMA`, a string ``asymmetry_version``, a
    string ``installed_at`` and an ``agent`` in :data:`AGENTS`. Anything else
    (unreadable bytes, malformed JSON, a missing or wrong field) is *no
    manifest*, so the directory is refused rather than removed.
    """
    path = _manifest_path(target)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("schema") != MANIFEST_SCHEMA:
        return None
    if not isinstance(manifest.get("asymmetry_version"), str):
        return None
    if not isinstance(manifest.get("installed_at"), str):
        return None
    if manifest.get("agent") not in AGENTS:
        return None
    return manifest


def links_into_package(target: Path) -> bool:
    """Whether *target* is a symlink pointing at the packaged skill.

    That is the whole proof a linked install needs: only ``install --link``
    puts a symlink to :func:`skill_source_dir` at an agent's skill path, so a
    link resolving there is one this command made and is safe to replace or
    remove. A symlink to anywhere else belongs to whoever made it.
    """
    return target.is_symlink() and target.resolve() == skill_source_dir().resolve()


def _clear_target(target: Path, *, force: bool) -> None:
    """Make way for a new install at *target*, or refuse.

    A symlink is replaced only when it is one of ours or *force* is given; a
    directory only when it carries this command's manifest or *force* is given.
    """
    if target.is_symlink():
        if not (links_into_package(target) or force):
            raise UserError(
                f"{target} is a symlink to {os.readlink(target)}, which is not the "
                "packaged skill; pass --force to replace it."
            )
        target.unlink()
        return
    if not target.exists():
        return
    if not force and read_manifest(target) is None:
        raise UserError(
            f"{target} already exists and was not written by 'asymmetry skill install' "
            "(no manifest found there); pass --force to overwrite it anyway."
        )
    shutil.rmtree(target)


@dataclass(frozen=True)
class InstallResult:
    """What :func:`install` wrote.

    ``manifest`` is ``None`` exactly when ``linked`` is true: a development
    link writes nothing into the package directory, and needs no manifest.
    """

    agent: str
    path: Path
    linked: bool
    manifest: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "agent": self.agent,
            "path": str(self.path),
            "linked": self.linked,
            "manifest": None if self.manifest is None else dict(self.manifest),
        }


def install(
    agent: str,
    *,
    project: bool = False,
    into: str | Path | None = None,
    force: bool = False,
    link: bool = False,
) -> InstallResult:
    """Put the packaged skill in *agent*'s skill directory.

    Copies it and stamps a manifest, or with *link* points a symlink at
    :func:`skill_source_dir` instead, so a checkout's edits reach the agent
    without reinstalling. Raises :class:`UserError` when the target exists and
    is neither this command's own directory nor its own link, unless *force* is
    given — what this command did not write is not this command's to overwrite.
    """
    target = target_dir(agent, project=project, into=into)
    _clear_target(target, force=force)
    target.parent.mkdir(parents=True, exist_ok=True)

    if link:
        target.symlink_to(skill_source_dir(), target_is_directory=True)
        return InstallResult(agent=agent, path=target, linked=True, manifest=None)

    shutil.copytree(skill_source_dir(), target)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "asymmetry_version": __version__,
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent,
    }
    _manifest_path(target).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return InstallResult(agent=agent, path=target, linked=False, manifest=manifest)


def uninstall(agent: str, *, project: bool = False, into: str | Path | None = None) -> Path:
    """Remove *agent*'s skill, only if this command put it there.

    A development link is removed on the strength of where it points (see
    :func:`links_into_package`) — the link itself, never what it points at.
    Otherwise the directory must carry this command's manifest; raises
    :class:`UserError` when it does not — including when it does not exist at
    all — so an unrelated directory (or one already removed) is never silently
    accepted as a no-op.
    """
    target = target_dir(agent, project=project, into=into)
    if links_into_package(target):
        target.unlink()
        return target
    if read_manifest(target) is None:
        raise UserError(
            f"{target} was not written by 'asymmetry skill install' (no manifest found "
            "there); not removing it."
        )
    shutil.rmtree(target)
    return target


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_check(agents: tuple[str, ...] = AGENTS) -> dict[str, Any]:
    """Everything ``asymmetry skill check`` reports, as a JSON-safe dict.

    ``"lines"`` is one entry per reported fact, each carrying whether it
    counts against ``"ok"`` — the CLI on ``PATH``, matplotlib importable, at
    least one loader extra installed, and every *installed* skill's manifest
    matching :data:`asymmetry.__version__` are the pass/fail facts; which
    individual loader extras are present, and which agents have no skill
    installed at all, are informational only.
    """
    lines: list[dict[str, Any]] = []

    on_path = shutil.which("asymmetry") is not None
    lines.append({"ok": on_path, "text": f"asymmetry on PATH: {'ok' if on_path else 'missing'}"})

    matplotlib_available = _module_available("matplotlib")
    lines.append(
        {
            "ok": matplotlib_available,
            "text": f"matplotlib importable: {'ok' if matplotlib_available else 'missing'}",
        }
    )

    loaders = {module: _module_available(module) for module in LOADER_MODULES}
    for module, available in loaders.items():
        extra = LOADER_MODULES[module]
        lines.append(
            {
                "ok": True,  # informational: no single loader is required
                "text": f"{module} ({extra}) importable: {'yes' if available else 'no'}",
            }
        )
    loader_available = any(loaders.values())
    lines.append(
        {
            "ok": loader_available,
            "text": (f"at least one loader extra installed: {'yes' if loader_available else 'no'}"),
        }
    )

    skills: dict[str, dict[str, Any]] = {}
    for agent in agents:
        target = target_dir(agent)
        if links_into_package(target):
            # A link *is* the running package's skill directory, so it cannot be
            # a stale copy; there is nothing to compare and nothing to rerun.
            skills[agent] = {
                "installed": True,
                "linked": True,
                "path": str(target),
                "version": __version__,
                "up_to_date": True,
            }
            lines.append(
                {
                    "ok": True,
                    "text": (
                        f"{agent} skill installed: linked (development) → {skill_source_dir()}"
                    ),
                }
            )
            continue
        manifest = read_manifest(target)
        if manifest is None:
            skills[agent] = {
                "installed": False,
                "linked": False,
                "path": str(target),
                "version": None,
                "up_to_date": True,
            }
            lines.append({"ok": True, "text": f"{agent} skill installed: no ({target})"})
            continue
        version = manifest.get("asymmetry_version")
        up_to_date = version == __version__
        skills[agent] = {
            "installed": True,
            "linked": False,
            "path": str(target),
            "version": version,
            "up_to_date": up_to_date,
        }
        text = f"{agent} skill installed: yes ({target}, version {version})"
        if not up_to_date:
            text += f" — current is {__version__}; rerun 'asymmetry skill install'"
        lines.append({"ok": up_to_date, "text": text})

    ok = all(line["ok"] for line in lines)
    return {
        "ok": ok,
        "asymmetry_on_path": on_path,
        "matplotlib_available": matplotlib_available,
        "loaders": loaders,
        "loader_available": loader_available,
        "skills": skills,
        "lines": lines,
    }


__all__ = [
    "AGENTS",
    "LOADER_MODULES",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "SKILL_NAME",
    "InstallResult",
    "install",
    "links_into_package",
    "read_manifest",
    "run_check",
    "skill_source_dir",
    "target_dir",
    "uninstall",
]
