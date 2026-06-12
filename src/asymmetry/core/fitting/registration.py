"""Shared insertion core for the name-keyed fit-function registries.

``COMPONENTS``, ``MODELS``, and ``PARAMETER_MODEL_COMPONENTS`` are all plain
name-keyed dicts. Built-in entries are mostly literals, but every entry added
through *code* — ``models._register`` and the public user-function facade in
:mod:`asymmetry.core.fitting.user_functions` — funnels through
:func:`insert_definition`, so there is exactly one registration path and one
place that enforces the name rules.

Name grammar: a registry key must be a bare expression-grammar atom
(``[A-Za-z_][A-Za-z0-9_]*``) so it can appear verbatim in composite
expressions, and must not be a grammar-reserved token.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

#: The expression-tokenizer atom (see ``composite._tokenize_component_expression``).
REGISTRY_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

#: Tokens with grammar meaning that must never become component names:
#: the ``{frac}`` group decorator and the internal unit-amplitude sentinel.
RESERVED_NAMES = frozenset({"frac", "__UNIT_AMPLITUDE__"})


class NamedDefinition(Protocol):
    """The slice of the definition dataclasses the registration core needs."""

    name: str


def validate_registry_name(name: object, *, registry: dict[str, Any], registry_label: str) -> str:
    """Validate *name* as a new key for *registry*, returning it on success.

    Raises ``ValueError`` for a non-string, grammar-incompatible, or reserved
    name, and for a name already present in *registry*.
    """
    if not isinstance(name, str) or not REGISTRY_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid registry name {name!r}: names must match [A-Za-z_][A-Za-z0-9_]* "
            "so they can appear in composite expressions."
        )
    if name in RESERVED_NAMES:
        raise ValueError(f"Invalid registry name {name!r}: reserved grammar token.")
    if name in registry:
        raise ValueError(f"Name {name!r} is already registered in {registry_label}.")
    return name


def insert_definition(
    registry: dict[str, Any],
    definition: NamedDefinition,
    *,
    registry_label: str,
) -> None:
    """Insert *definition* into *registry* under its own name, validated."""
    validate_registry_name(definition.name, registry=registry, registry_label=registry_label)
    registry[definition.name] = definition
