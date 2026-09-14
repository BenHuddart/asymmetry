"""``asymmetry skill`` — install, check, or remove the agent skill."""

from __future__ import annotations

import argparse

from asymmetry.cli._output import UserError, emit_json, payload


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``skill`` subcommand and its ``install``/``check``/``uninstall`` children."""
    parser = subparsers.add_parser(
        "skill",
        help="Install, check or remove the asymmetry-analysis agent skill",
    )
    children = parser.add_subparsers(dest="skill_command", required=True)

    install_parser = children.add_parser("install", help="Install the skill for an agent")
    install_parser.add_argument("--agent", choices=["claude", "codex"], required=True)
    install_parser.add_argument(
        "--project",
        action="store_true",
        help="Install under ./.claude or ./.agents instead of the home directory",
    )
    install_parser.add_argument(
        "--into",
        default=None,
        help="Install under this directory instead of the agent's usual location",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a target directory even if it was not written by a previous install",
    )
    install_parser.add_argument(
        "--json", action="store_true", help="Emit the machine-readable payload"
    )
    install_parser.set_defaults(func=_run_install)

    check_parser = children.add_parser(
        "check", help="Report whether the CLI, loaders and skill are ready for agent use"
    )
    check_parser.add_argument(
        "--agent",
        choices=["claude", "codex"],
        default=None,
        help="Check only this agent's skill install (default: every agent)",
    )
    check_parser.add_argument(
        "--json", action="store_true", help="Emit the machine-readable payload"
    )
    check_parser.set_defaults(func=_run_check)

    uninstall_parser = children.add_parser("uninstall", help="Remove an installed skill")
    uninstall_parser.add_argument("--agent", choices=["claude", "codex"], required=True)
    uninstall_parser.add_argument(
        "--project",
        action="store_true",
        help="Remove ./.claude or ./.agents instead of the home directory's copy",
    )
    uninstall_parser.add_argument(
        "--into",
        default=None,
        help="The directory it was installed under, if not the agent's usual location",
    )
    uninstall_parser.add_argument(
        "--json", action="store_true", help="Emit the machine-readable payload"
    )
    uninstall_parser.set_defaults(func=_run_uninstall)


def _run_install(args: argparse.Namespace) -> None:
    from asymmetry.cli import skill

    result = skill.install(args.agent, project=args.project, into=args.into, force=args.force)

    if args.json:
        emit_json(payload(install=result.to_dict()))
        return

    print(f"Installed the {skill.SKILL_NAME!r} skill for {result.agent} into {result.path}")
    print("Restart the agent (or start a new session) so it picks up the skill.")


def _run_uninstall(args: argparse.Namespace) -> None:
    from asymmetry.cli import skill

    path = skill.uninstall(args.agent, project=args.project, into=args.into)

    if args.json:
        emit_json(payload(agent=args.agent, path=str(path), removed=True))
        return

    print(f"Removed {path}")


def _run_check(args: argparse.Namespace) -> None:
    from asymmetry.cli import skill

    agents = skill.AGENTS if args.agent is None else (args.agent,)
    report = skill.run_check(agents)

    if args.json:
        emit_json(payload(check=report))
    else:
        print(_render_check(report))

    if not report["ok"]:
        failing = "; ".join(line["text"] for line in report["lines"] if not line["ok"])
        raise UserError(failing)


def _render_check(report: dict) -> str:
    """The human-readable check report: one line per fact, then a verdict."""
    lines = [line["text"] for line in report["lines"]]
    lines.append("")
    lines.append("ready for agent use" if report["ok"] else "NOT ready for agent use")
    return "\n".join(lines)
