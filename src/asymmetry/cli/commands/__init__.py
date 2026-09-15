"""One module per ``asymmetry`` subcommand.

Each module exposes ``add_parser(subparsers)`` (declaring the command's
arguments) and ``run(args)`` (doing the work). Imports of the analysis engine
happen inside ``run`` so ``asymmetry --help`` stays fast.
"""
