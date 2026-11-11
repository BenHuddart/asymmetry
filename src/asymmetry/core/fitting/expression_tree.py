"""Arithmetic structure and fraction spans of composite source models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class _ClosingParenthesis:
    """One closing parenthesis: the component span it delimits, and its decorator."""

    span: tuple[int, int]
    is_fraction_group: bool


def _closing_parentheses(
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    fraction_groups: Sequence[tuple[int, int]] = (),
) -> list[list[_ClosingParenthesis]]:
    """Return, per component index, the parentheses closing there, innermost first.

    Two parentheses can delimit the same component span (``((a + b))``), while a
    ``{frac}`` decorator belongs to exactly one pair: it lands on the innermost
    pair with that span — the pair :func:`parse_composite_expression` read it
    from. Everything that has to know *which* parenthesis carries a group (the
    tree builder, the expression writer, the formula and mathtext renderers)
    reads it from here, so they all agree.
    """
    pending = Counter(fraction_groups)
    stack: list[int] = []
    closures: list[list[_ClosingParenthesis]] = []
    for index in range(len(open_parentheses)):
        stack.extend([index] * open_parentheses[index])
        closing_here: list[_ClosingParenthesis] = []
        for _ in range(close_parentheses[index]):
            if not stack:
                raise ValueError("Invalid parentheses: closing before opening")
            span = (stack.pop(), index)
            is_fraction_group = pending[span] > 0
            if is_fraction_group:
                pending[span] -= 1
            closing_here.append(_ClosingParenthesis(span, is_fraction_group))
        closures.append(closing_here)
    if stack:
        raise ValueError("Invalid parentheses: unbalanced expression")
    return closures


@dataclass(frozen=True)
class ExprLeaf:
    """One component, addressed by its index in ``component_names``."""

    index: int


@dataclass(frozen=True)
class ExprProduct:
    """A ``*``/``/`` chain; ``ops[k]`` joins ``factors[k]`` to ``factors[k + 1]``.

    Normalised so that no factor is itself an :class:`ExprProduct`: every
    factor is an :class:`ExprLeaf` or an :class:`ExprSum`.
    """

    factors: tuple[ExpressionNode, ...]
    ops: tuple[str, ...]


@dataclass(frozen=True)
class ExprSum:
    """A ``+``/``-`` chain; ``signs[k]`` is ``+1``/``-1`` for ``terms[k]``.

    ``fraction_group`` carries the ``(start, end)`` component range when the sum
    is a fraction group: its group amplitude scales the weighted sum of its
    terms, and its terms stay aligned one-for-one with the group's additive term
    ranges (so a fraction sum is never flattened into a parent sum).
    """

    terms: tuple[ExpressionNode, ...]
    signs: tuple[int, ...]
    fraction_group: tuple[int, int] | None = None


ExpressionNode = ExprLeaf | ExprProduct | ExprSum

_INVERTED_OPERATOR = {"*": "/", "/": "*"}


def _node_children(node: ExpressionNode) -> tuple[ExpressionNode, ...]:
    if isinstance(node, ExprLeaf):
        return ()
    return node.factors if isinstance(node, ExprProduct) else node.terms


def leaf_indices(node: ExpressionNode) -> tuple[int, ...]:
    """Return the component indices under ``node``, left to right."""
    if isinstance(node, ExprLeaf):
        return (node.index,)
    return tuple(index for child in _node_children(node) for index in leaf_indices(child))


def iter_nodes(node: ExpressionNode) -> Iterator[ExpressionNode]:
    """Yield ``node`` and every node beneath it, parents before children."""
    yield node
    for child in _node_children(node):
        yield from iter_nodes(child)


def _flattened_sum(terms: list[ExpressionNode], signs: list[int]) -> ExprSum:
    """Return a plain sum with plain-sum terms lifted into it.

    ``a - (b + c)`` is ``a - b - c``, so a lifted term's sign multiplies the
    signs it brings with it. Fraction sums own a scale and are never lifted.
    """
    flat_terms: list[ExpressionNode] = []
    flat_signs: list[int] = []
    for sign, term in zip(signs, terms, strict=True):
        if isinstance(term, ExprSum) and term.fraction_group is None:
            for inner_sign, inner_term in zip(term.signs, term.terms, strict=True):
                flat_terms.append(inner_term)
                flat_signs.append(sign * inner_sign)
        else:
            flat_terms.append(term)
            flat_signs.append(sign)
    return ExprSum(tuple(flat_terms), tuple(flat_signs), None)


def _flattened_product(factors: list[ExpressionNode], ops: list[str]) -> ExprProduct:
    """Return a product with sub-products lifted into it.

    ``x / (a * b)`` is ``x / a / b``: dividing *into* a sub-product inverts the
    operators the sub-product brings with it.
    """
    flat_factors: list[ExpressionNode] = []
    flat_ops: list[str] = []
    for order, factor in enumerate(factors):
        join = ops[order - 1] if order else ""
        if isinstance(factor, ExprProduct):
            inverted = join == "/"
            for inner_order, inner_factor in enumerate(factor.factors):
                if inner_order:
                    inner_op = factor.ops[inner_order - 1]
                    flat_ops.append(_INVERTED_OPERATOR[inner_op] if inverted else inner_op)
                elif join:
                    flat_ops.append(join)
                flat_factors.append(inner_factor)
        else:
            if join:
                flat_ops.append(join)
            flat_factors.append(factor)
    return ExprProduct(tuple(flat_factors), tuple(flat_ops))


def build_expression_tree(
    component_count: int,
    operators: Sequence[str],
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    fraction_groups: Sequence[tuple[int, int]] = (),
) -> ExpressionNode:
    """Return the expression tree denoted by a composite model's flat form.

    ``*``/``/`` bind tighter than ``+``/``-``; parentheses group. The result is
    normalised so that only the *structure* survives: a group holding a single
    node **is** that node (redundant parentheses vanish), a product factor that
    is itself a product is flattened into its parent, and a plain sum term that
    is itself a plain sum is flattened. A fraction-group sum keeps its own terms.

    The caller has already validated the parenthesis balance, and the fraction
    groups are the ones *requested*: a group is tagged onto the sum of the
    parenthesis whose span it names, and
    :meth:`CompositeModel._validate_fraction_groups` then reads the tagged sums
    back to accept or reject the request.
    """
    # ``(`` and ``)`` tokens both carry the parenthesis they delimit, so an
    # opening parenthesis knows whether it starts a fraction group.
    closures = _closing_parentheses(open_parentheses, close_parentheses, fraction_groups)
    tokens: list[tuple[str, int | str | _ClosingParenthesis | None]] = []
    open_positions: list[int] = []
    for index in range(component_count):
        for _ in range(open_parentheses[index]):
            open_positions.append(len(tokens))
            tokens.append(("(", None))
        tokens.append(("leaf", index))
        for closure in closures[index]:
            tokens[open_positions.pop()] = ("(", closure)
            tokens.append((")", closure))
        if index < component_count - 1:
            tokens.append(("op", operators[index]))

    position = 0

    def at_operator(symbols: set[str]) -> bool:
        return (
            position < len(tokens)
            and tokens[position][0] == "op"
            and tokens[position][1] in symbols
        )

    def parse_group(fraction_group: tuple[int, int] | None) -> ExpressionNode:
        nonlocal position
        terms = [parse_product()]
        signs = [1]
        while at_operator({"+", "-"}):
            signs.append(1 if tokens[position][1] == "+" else -1)
            position += 1
            terms.append(parse_product())
        if fraction_group is not None:
            # A tagged sum keeps its own terms, however many: validation counts
            # them here and rejects a group that does not span two or more.
            return ExprSum(tuple(terms), tuple(signs), fraction_group)
        if len(terms) == 1:
            return terms[0]
        return _flattened_sum(terms, signs)

    def parse_product() -> ExpressionNode:
        nonlocal position
        factors = [parse_atom()]
        ops: list[str] = []
        while at_operator({"*", "/"}):
            ops.append(cast(str, tokens[position][1]))
            position += 1
            factors.append(parse_atom())
        if len(factors) == 1:
            return factors[0]
        return _flattened_product(factors, ops)

    def parse_atom() -> ExpressionNode:
        nonlocal position
        kind, payload = tokens[position]
        position += 1
        if kind == "leaf":
            return ExprLeaf(index=cast(int, payload))
        closure = cast(_ClosingParenthesis, payload)
        node = parse_group(closure.span if closure.is_fraction_group else None)
        position += 1  # the matching ``)``
        return node

    return parse_group(None)


def _fraction_group_nodes(tree: ExpressionNode) -> dict[tuple[int, int], ExprSum]:
    """Return the sum node tagged with each fraction group, keyed by group."""
    return {
        node.fraction_group: node
        for node in iter_nodes(tree)
        if isinstance(node, ExprSum) and node.fraction_group is not None
    }
