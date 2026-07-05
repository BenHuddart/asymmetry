"""Dialog for building composite fit functions.

Thin subclass of the shared
:class:`~asymmetry.gui.widgets.function_builder.dialog.FunctionBuilderDialog`
that wires the fit grammar and domain filtering. The dialog opens with a
searchable component library, a structured row editor, and an honest text
mode that speaks the canonical ``{frac}`` syntax.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from PySide6.QtWidgets import QWidget

from asymmetry.core.fitting.composite import (
    COMPONENTS,
    CompositeModel,
    UnknownComponentError,
    parse_composite_expression,
    placeholder_component_definition,
)
from asymmetry.core.fitting.domain_library import (
    coerce_domain,
    components_for_domain,
    default_model_for_domain,
)
from asymmetry.gui.widgets.function_builder.dialog import FunctionBuilderDialog


class FitFunctionBuilderDialog(FunctionBuilderDialog):
    """Compose a custom fit function from predefined components."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_model: CompositeModel | None = None,
        domain: str = "time",
    ) -> None:
        self._domain = coerce_domain(domain)
        domain_components = components_for_domain(self._domain)
        self._allowed_components = set(domain_components)

        model = (
            initial_model if initial_model is not None else default_model_for_domain(self._domain)
        )

        # A project model may reference a user function that is not registered in
        # this session. Materialise per-instance placeholder definitions for
        # those names so the row list renders them (with a warning tint) rather
        # than the library dropping them silently. The placeholders are also
        # injected into the parser's name pool (below) so the honest initial
        # expression seeds instead of failing to parse.
        self._placeholder_definitions = {
            name: placeholder_component_definition(name)
            for name in getattr(model, "missing_component_names", ())
        }
        component_definitions = dict(domain_components)
        component_definitions.update(self._placeholder_definitions)

        if self._domain == "frequency":
            expression_prefix = "S(ν)"
        else:
            expression_prefix = "A(t)"

        super().__init__(
            title="Build Fit Function",
            expression_prefix=expression_prefix,
            component_definitions=component_definitions,
            model_parser=self._parse_model,
            expression_parser=self._expression_parser,
            initial_expression=model.component_expression_string(),
            enable_fraction_groups=True,
            syntax_help_text=(
                "Select two or more '+'-joined rows and press 'Group as fractions'; "
                "the last term's weight is the remainder to 1. "
                "Del removes · Alt+↑/↓ moves · drag the grip to reorder."
            ),
            parent=parent,
        )

    # ---------------------------------------------------------- domain guard
    @contextlib.contextmanager
    def _placeholders_registered(self) -> Iterator[None]:
        """Temporarily expose placeholder names to the core parsers.

        The core expression parsers validate names against the global
        ``COMPONENTS`` registry. A project model referencing an unregistered
        user function must still round-trip through the dialog, so its
        per-instance placeholder definitions are injected for the duration of a
        parse and removed afterwards (they are never persisted into
        ``COMPONENTS``).
        """
        if not self._placeholder_definitions:
            yield
            return
        added = [name for name in self._placeholder_definitions if name not in COMPONENTS]
        for name in added:
            COMPONENTS[name] = self._placeholder_definitions[name]
        try:
            yield
        finally:
            for name in added:
                COMPONENTS.pop(name, None)

    def _domain_hint_error(self, exc: UnknownComponentError) -> UnknownComponentError | ValueError:
        """Upgrade an unknown-name error with a domain hint when applicable.

        A component that exists in the global registry but belongs to the other
        analysis domain gets an explanatory message instead of the generic
        "unknown component" error; a genuinely unknown name propagates as-is so
        its close-name suggestions survive.
        """
        definition = COMPONENTS.get(exc.name)
        if definition is not None and definition.domain != self._domain:
            return ValueError(
                f"'{definition.name}' is a {definition.domain}-domain component "
                f"and is not available when fitting in the {self._domain} domain."
            )
        return exc

    def _check_domain(self, component_names: list[str]) -> None:
        """Raise a domain-hint error if any name is out of the domain pool."""
        for name in component_names:
            if name in self._allowed_components or name not in COMPONENTS:
                # In-pool, or a missing/user placeholder (unknown to COMPONENTS)
                # which is surfaced separately via missing_component_names.
                continue
            definition = COMPONENTS[name]
            if definition.domain != self._domain:
                raise ValueError(
                    f"'{definition.name}' is a {definition.domain}-domain component "
                    f"and is not available when fitting in the {self._domain} domain."
                )

    def _expression_parser(
        self, expression: str
    ) -> tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]:
        """Parse *expression*, rejecting components outside the domain pool."""
        try:
            with self._placeholders_registered():
                names, operators, opens, closes, fractions = parse_composite_expression(expression)
        except UnknownComponentError as exc:
            raise self._domain_hint_error(exc) from exc
        self._check_domain(names)
        return names, operators, opens, closes, fractions

    def _parse_model(self, expression: str) -> CompositeModel:
        """Build a :class:`CompositeModel`, enforcing the domain pool.

        Structure comes from :meth:`_expression_parser` (which registers
        placeholder names for the parse, upgrades domain-hint errors, and
        rejects out-of-pool names) so a missing-user-function model still
        round-trips; the model itself is built with ``allow_missing`` so its
        ``missing_component_names`` provenance survives (a placeholder is never
        persisted into COMPONENTS).
        """
        names, operators, opens, closes, fractions = self._expression_parser(expression)
        return CompositeModel(
            names,
            operators=operators,
            open_parentheses=opens,
            close_parentheses=closes,
            fraction_groups=fractions,
            allow_missing=True,
        )

    # --------------------------------------------------------------- result
    def get_composite_model(self) -> CompositeModel | None:
        """Return the model produced when the dialog is accepted."""
        model = self.built_model()
        return model if isinstance(model, CompositeModel) else None
