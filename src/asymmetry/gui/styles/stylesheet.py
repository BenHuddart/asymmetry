"""Render the BENCH application stylesheet from ``bench.qss`` + design tokens.

``bench.qss`` is a *template*: every colour is written as an ``@TOKEN@``
placeholder that names a constant in :mod:`asymmetry.gui.styles.tokens`. This
module substitutes the token values at load time so the stylesheet has a single
source of truth (the tokens) rather than a parallel set of hardcoded hex
literals that could drift. Scale-dependent metrics and chrome font-sizes are
appended separately by :meth:`UIManager.build_stylesheet`.
"""

from __future__ import annotations

import re
from pathlib import Path

from asymmetry.gui.styles import tokens

_TEMPLATE_NAME = "bench.qss"
#: Matches an ``@TOKEN@`` placeholder (upper-snake-case identifier).
_PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")


def substitute_tokens(template: str) -> str:
    """Replace every ``@TOKEN@`` placeholder with its ``tokens`` colour value.

    Raises:
        KeyError: if a placeholder names a token that does not exist or is not a
            string colour — a templating typo that should fail loudly in tests
            rather than ship a broken stylesheet.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        value = getattr(tokens, name, None)
        if not isinstance(value, str):
            raise KeyError(f"bench.qss references unknown colour token @{name}@")
        return value

    return _PLACEHOLDER.sub(_sub, template)


def load_template() -> str:
    """Return the raw ``bench.qss`` template text (disk first, then resources).

    Mirrors the dual lookup the app relies on: the on-disk path for source and
    editable installs, and ``importlib.resources`` for a frozen (PyInstaller)
    build where ``__file__`` paths do not resolve into the bundled package tree.
    """
    path = Path(__file__).parent / _TEMPLATE_NAME
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        pass

    try:
        from importlib.resources import files

        resource = files("asymmetry.gui.styles").joinpath(_TEMPLATE_NAME)
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError, OSError):
        pass

    return ""


def render_bench_stylesheet() -> str:
    """Return the colour-substituted BENCH stylesheet, or ``""`` if unavailable.

    A templating typo (a placeholder naming a missing token) makes
    :func:`substitute_tokens` raise — which a test catches loudly — but at
    runtime we must never crash app startup over chrome, so the production entry
    point degrades to bare Fusion (``""``) instead, matching the prior loader's
    "ugly but runs" failure mode.
    """
    template = load_template()
    if not template:
        return ""
    try:
        return substitute_tokens(template)
    except KeyError:
        return ""
