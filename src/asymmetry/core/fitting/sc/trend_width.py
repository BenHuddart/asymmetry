"""Captured Gaussian-width provenance at parameter-fit boundaries.

The documented Custom transform is recognised structurally, never by sampling
values. This records a width conversion, not nuclear subtraction or geometry.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, replace

from asymmetry.core.fitting.axis_transforms import AxisTransform
from asymmetry.core.fitting.composite import QUADRATURE_OPERATOR
from asymmetry.core.fitting.expression_tree import ExprLeaf, ExprProduct, build_expression_tree
from asymmetry.core.fitting.parameter_carry import ComponentParameter
from asymmetry.core.fitting.sc.constants import GaussianWidth

SECOND_MOMENT_MODELS = frozenset({"SC_Brandt_VortexLattice", "SC_Brandt_VortexLattice_Powder"})


class WidthConventionError(ValueError):
    """A known width input does not satisfy the requested second-moment contract."""


@dataclass(frozen=True)
class WidthSource:
    """A captured component parameter and its known width conventions.

    Conventions describe known inputs; unverified transforms retain an altered
    known origin without certifying its output as a physical width. An empty
    convention tuple means the source is unknown.
    """

    component: str
    index: int
    parameter: str
    widths: tuple[GaussianWidth, ...] = ()
    unverified_transforms: tuple[str, ...] = ()
    conversion_expression: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-safe captured source record."""
        return {
            "component": self.component,
            "index": self.index,
            "parameter": self.parameter,
            "widths": [w.name for w in self.widths],
            "unverified_transforms": list(self.unverified_transforms),
            "conversion_expression": self.conversion_expression,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WidthSource:
        """Read a recorded source, preserving unknown legacy conventions."""
        return cls(
            data["component"],
            data["index"],
            data["parameter"],
            tuple(GaussianWidth[name] for name in data["widths"]),
            tuple(data.get("unverified_transforms", ())),
            data.get("conversion_expression"),
        )


def _additive_width_components(model) -> set[int]:
    """Find components joined to the output only by addition or quadrature.

    Both operations preserve units. An unknown multiplier/divisor stops
    attribution along that branch, without hiding a bare width in another term.
    """
    tree = build_expression_tree(
        len(model.components),
        ["+" if operator == QUADRATURE_OPERATOR else operator for operator in model.operators],
        model.open_parentheses,
        model.close_parentheses,
    )
    pending = [tree]
    indices = set()
    while pending:
        node = pending.pop()
        if isinstance(node, ExprProduct):
            continue
        if isinstance(node, ExprLeaf):
            indices.add(node.index)
        else:
            pending.extend(node.terms)
    return indices


def model_width_sources(model, inherited=(), unverified_transforms=()):
    """Capture canonical identities, independent of aliases and display labels."""
    from asymmetry.core.fitting.parameter_models import SUPERCONDUCTING_GAP_LAWS

    additive = _additive_width_components(model)
    sources = {}
    for name, identity in model.parameter_identities().items():
        if not isinstance(identity, ComponentParameter):
            continue
        component = model.components[identity.component]
        widths = ()
        unverified = ()
        if component.name == "Gaussian" and identity.local_name == "sigma":
            widths = (GaussianWidth.GAUSSIAN_COMPONENT,)
        elif identity.component in additive and (
            (component.name, identity.local_name) in {("Constant", "c"), ("Linear", "b")}
            or (
                component.name in SUPERCONDUCTING_GAP_LAWS
                and identity.local_name
                in {
                    "sigma_0",
                    "sigma_bg",
                    "sigma_sc",
                    "sigma_nm",
                }
            )
        ):
            widths = tuple(inherited)
            unverified = tuple(unverified_transforms)
        elif component.name in SECOND_MOMENT_MODELS and identity.local_name == "sigma_bg":
            widths = (GaussianWidth.SECOND_MOMENT,)
        sources[name] = WidthSource(
            component.name, identity.component, identity.local_name, widths, unverified
        ).to_dict()
    return sources


@dataclass(frozen=True)
class TrendWidthContext:
    """Source identities aligned to the actual Y array, with its captured lens."""

    column: str
    sources: tuple[WidthSource | None, ...]
    transform: AxisTransform = AxisTransform()

    @classmethod
    def for_column(cls, column: str, source_maps: Sequence[dict], transform=AxisTransform()):
        """Capture one column's per-row source records at data assembly."""
        return cls(
            column,
            tuple(
                WidthSource.from_dict(source) if source else None
                for sources in source_maps
                for source in [sources.get(column)]
            ),
            transform,
        )

    def select(self, mask: Sequence[bool]) -> TrendWidthContext:
        """Apply the numerical fit mask to its aligned source identities."""
        if len(mask) != len(self.sources):
            raise ValueError("Width source count must match fitted Y observations")
        return TrendWidthContext(
            self.column,
            tuple(source for source, kept in zip(self.sources, mask, strict=True) if kept),
            self.transform,
        )

    def to_dict(self) -> dict:
        """Capture source identities and the actual Y transform."""
        return {
            "column": self.column,
            "sources": [s.to_dict() if s else None for s in self.sources],
            "transform": self.transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> TrendWidthContext | None:
        """Restore captured context; absent historical attribution stays unknown."""
        if data is None:
            return None
        return cls(
            data["column"],
            tuple(WidthSource.from_dict(s) if s else None for s in data["sources"]),
            AxisTransform.from_dict(data["transform"]),
        )

    @property
    def converts_gaussian(self):
        """Only the documented sqrt(2)*x expression declares this conversion."""
        if self.transform.kind != "custom":
            return False
        # AxisTransform's safe expression boundary validates execution separately.
        node = ast.parse(self.transform.expression_text, mode="eval").body
        return ast.dump(node) in {
            ast.dump(ast.parse(expression, mode="eval").body)
            for expression in ("sqrt(2)*x", "x*sqrt(2)")
        }

    @property
    def preserves_width(self) -> bool:
        """Recognise identity without evaluating sampled values."""
        if self.transform.is_identity:
            return True
        if self.transform.kind != "custom":
            return False
        node = ast.parse(self.transform.expression_text, mode="eval").body
        return isinstance(node, ast.Name) and node.id == "x"

    def known_widths(self) -> tuple[GaussianWidth, ...]:
        """Return the known subset without inferring facts about unknown rows."""
        return tuple(
            sorted(
                {width for source in self.sources if source for width in source.widths},
                key=lambda width: width.name,
            )
        )

    def unverified_transforms(self):
        """Retain unverified changes to known widths across fitted width outputs."""
        expressions = {e for source in self.sources if source for e in source.unverified_transforms}
        widths = self.known_widths()
        if (
            widths
            and not self.preserves_width
            and not (self.converts_gaussian and set(widths) == {GaussianWidth.GAUSSIAN_COMPONENT})
        ):
            expressions.add(self.transform.expression_text)
        return tuple(sorted(expressions))

    def effective_widths(self):
        """Known inherited conventions; arbitrary transforms establish none."""
        if self.unverified_transforms():
            return ()
        widths = set(self.known_widths())
        if self.preserves_width:
            return tuple(sorted(widths, key=lambda width: width.name))
        if self.converts_gaussian and widths == {GaussianWidth.GAUSSIAN_COMPONENT}:
            return (GaussianWidth.SECOND_MOMENT,)
        return ()

    def validate(self, model):
        """Refuse known incompatible inputs at a bare second-moment consumer."""
        consumers = {
            model.components[index].name for index in _additive_width_components(model)
        }.intersection(SECOND_MOMENT_MODELS)
        if not consumers:
            return
        inherited_unverified = {
            e for source in self.sources if source for e in source.unverified_transforms
        }
        if inherited_unverified:
            raise WidthConventionError(
                f"Column {self.column!r} has an unverified transform of a known width "
                f"({', '.join(sorted(inherited_unverified))}); a second-moment fit requires "
                "an explicitly established width conversion."
            )
        widths = set(self.known_widths())
        gaussian = GaussianWidth.GAUSSIAN_COMPONENT in widths
        if gaussian and not (
            self.converts_gaussian and widths == {GaussianWidth.GAUSSIAN_COMPONENT}
        ):
            raise WidthConventionError(
                f"Column {self.column!r} contains Gaussian-component width in exp(-sigma^2*t^2); "
                f"{', '.join(sorted(consumers))} expects second-moment width in exp(-sigma^2*t^2/2). "
                "Apply Custom Y transform sqrt(2)*x to Gaussian-component values and errors before fitting; "
                "mixed conventions require separate conversion."
            )
        if widths == {GaussianWidth.SECOND_MOMENT} and not self.preserves_width:
            raise WidthConventionError(
                f"Column {self.column!r} already has second-moment width; "
                "use untransformed values and errors for a second-moment model."
            )


def fitted_width_sources(model, recorded_context):
    """Attribute width outputs to the actual selected fit input."""
    context = TrendWidthContext.from_dict(recorded_context)
    if context is None:
        return model_width_sources(model)
    unverified = context.unverified_transforms()
    return model_width_sources(
        model, context.known_widths() if unverified else context.effective_widths(), unverified
    )


def cross_group_width_sources(model, result, group_id=None):
    """Shared widths inherit all groups; local widths inherit their own group."""
    contexts = [TrendWidthContext.from_dict(c) for c in result.input_width_contexts.values()]
    unverified = {e for context in contexts for e in context.unverified_transforms()}
    inherited = {
        w
        for context in contexts
        for w in (
            context.known_widths()
            if context.unverified_transforms()
            else context.effective_widths()
        )
    }
    sources = model_width_sources(
        model, tuple(sorted(inherited, key=lambda w: w.name)), tuple(sorted(unverified))
    )
    if group_id is not None:
        local = fitted_width_sources(model, result.input_width_contexts.get(group_id))
        for name in result.local_parameters.get(group_id, ()):
            sources[name.name] = local[name.name]
    return sources


def derived_width_source(
    expression: str, symbols: Sequence[str], sources: dict
) -> WidthSource | None:
    """Recognise only a literal copy or the documented one-source conversion.

    Arbitrary derived quantities remain unclassified; no sampled multiplier or
    nonlinear expression establishes a physical width convention.
    """
    if len(symbols) != 1:
        return None
    source_data = sources.get(symbols[0])
    if source_data is None:
        return None
    source = WidthSource.from_dict(source_data)
    if not source.widths:
        return None
    node = ast.parse(expression, mode="eval")
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id == symbols[0]:
            item.id = "x"
    context = TrendWidthContext(symbols[0], (source,), AxisTransform.custom(ast.unparse(node)))
    if context.preserves_width:
        return source
    if not context.converts_gaussian:
        return None
    unverified = context.unverified_transforms()
    return replace(
        source,
        widths=context.known_widths() if unverified else context.effective_widths(),
        unverified_transforms=unverified,
        conversion_expression=expression,
    )
