"""Read-only project source verification and explicit selected-owner replacement."""

from asymmetry.cli._output import UserError, emit_json, payload


def add_parser(subparsers):
    """Register project source actions with explicit output intent."""
    parser = subparsers.add_parser(
        "project-sources", help="Verify project inputs or accept selected replacements"
    )
    parser.add_argument("project", help="Project to inspect")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="INDEX=PATH",
        help="Propose a file for one source index",
    )
    parser.add_argument(
        "--replace",
        action="append",
        type=int,
        default=[],
        metavar="INDEX",
        help="Explicitly accept a readable current input",
    )
    parser.add_argument("--output", help="Write the updated project atomically; requires --replace")
    parser.add_argument(
        "--json", action="store_true", help="Emit statuses without locators or private diagnostics"
    )
    parser.set_defaults(func=run)


def run(args):
    """Inspect without writes, or save explicitly selected replacements without fitting."""
    from asymmetry.core.project import (
        apply_source_decisions,
        load_prepared_project,
        save_project,
        verify_project_sources,
    )
    from asymmetry.core.project.sources import source_candidates

    if bool(args.replace) != bool(args.output):
        raise UserError("--replace and --output are required together")
    try:
        prepared = load_prepared_project(args.project)
        candidates = list(source_candidates(prepared, args.project))
        seen = set()
        for entry in args.candidate:
            index_text, separator, path = entry.partition("=")
            index = int(index_text)
            if not separator or not path or index in seen or not 0 <= index < len(candidates):
                raise ValueError("--candidate requires a unique valid INDEX=PATH")
            seen.add(index)
            candidates[index] = path
        verified = verify_project_sources(prepared, args.project, candidates=candidates)
        application = apply_source_decisions(verified, replacements=args.replace)
        if args.output:
            save_project(application.prepared.state, args.output)
    except (ValueError, OSError) as exc:
        raise UserError(str(exc)) from None
    report = application.report()
    if args.json:
        emit_json(payload(sources=report, written=bool(args.output)))
    else:
        for row in report:
            kind = "current input" if row["active"] else "history"
            action = "; replacement accepted" if row["replacement"] else ""
            print(f"{row['index']}: {kind} {row['role']}: {row['status']}{action}")
