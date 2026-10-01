"""A model's signal terms as one fraction group: a total and fractions.

``A + B + Constant`` and ``(A + B){frac} + Constant`` are the same model family
written two ways: an amplitude per signal term, or one total with each term's
share of it. A series whose volume fraction changes through a transition has a
constant total and no constant amplitude, so the second form is the one whose
total can be shared (``docs/plans/global-wizard-trend-objective.md``, D11, D16).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.composite import CompositeModel, ExprSum, iter_nodes, leaf_indices
from asymmetry.core.fitting.parameter_carry import GroupAmplitude, carried_names

__all__ = ["SignalFractionForm", "signal_fraction_form"]

_SCALE_KINDS = (ParameterKind.AMPLITUDE, ParameterKind.BACKGROUND)


@dataclass(frozen=True)
class SignalFractionForm:
    """The fraction-group form of a plain additive model, and the map between their values.

    Both models have the same number of parameters: ``n`` signal amplitudes
    become one total and ``n − 1`` fractions, the last term taking the
    remainder. Every other parameter is carried, possibly under another name.
    """

    grouped: CompositeModel
    #: The plain model's signal amplitudes, in term order.
    amplitudes: tuple[str, ...]
    #: The grouped model's total: the sum of ``amplitudes``.
    total: str
    #: The grouped model's free fractions, one per signal term but the last.
    fractions: tuple[str, ...]
    #: Grouped name → plain name of every parameter both models have.
    carried: Mapping[str, str]

    def grouped_values(self, plain: Mapping[str, float]) -> dict[str, float]:
        """Express the plain model's values in the grouped model's parameters.

        ``total = Σ Aᵢ`` and ``fractionᵢ = Aᵢ / total``. The two models then
        evaluate alike when every fraction lies in [0, 1], which is where the
        grouped model clamps them: amplitudes of one sign.
        """
        total = sum(plain[name] for name in self.amplitudes)
        if total == 0.0:
            raise ValueError("signal amplitudes summing to zero have no fractions")
        return {
            self.total: total,
            **{
                fraction: plain[amplitude] / total
                for fraction, amplitude in zip(self.fractions, self.amplitudes, strict=False)
            },
            **{name: plain[plain_name] for name, plain_name in self.carried.items()},
        }

    def plain_values(self, grouped: Mapping[str, float]) -> dict[str, float]:
        """The inverse of :meth:`grouped_values`: ``Aᵢ = total · fractionᵢ``."""
        total = grouped[self.total]
        shares = [grouped[name] for name in self.fractions]
        return {
            **{
                amplitude: total * share
                for amplitude, share in zip(
                    self.amplitudes, [*shares, 1.0 - sum(shares)], strict=True
                )
            },
            **{plain_name: grouped[name] for name, plain_name in self.carried.items()},
        }

    def plain_uncertainties(
        self, grouped: Mapping[str, float], uncertainties: Mapping[str, float]
    ) -> dict[str, float]:
        """1σ of the plain model's parameters, from the grouped model's.

        ``σ²(Aᵢ) = shareᵢ² σ²(total) + total² σ²(shareᵢ)``, the last share being
        one minus the others. The covariance between the total and a share is
        left out, so these are for judging whether a term is there at all, not
        for quoting. An amplitude has no entry when the total or one of the
        fractions its share is built from has none.
        """
        total = grouped[self.total]
        shares = [grouped[name] for name in self.fractions]
        variances = [uncertainties.get(name, math.nan) ** 2 for name in self.fractions]
        total_variance = uncertainties.get(self.total, math.nan) ** 2
        amplitudes = {
            amplitude: math.sqrt(share**2 * total_variance + total**2 * variance)
            for amplitude, share, variance in zip(
                self.amplitudes,
                [*shares, 1.0 - sum(shares)],
                [*variances, sum(variances)],
                strict=True,
            )
        }
        return {
            **{name: sigma for name, sigma in amplitudes.items() if math.isfinite(sigma)},
            **{
                plain_name: uncertainties[name]
                for name, plain_name in self.carried.items()
                if name in uncertainties
            },
        }


def signal_fraction_form(model: CompositeModel) -> SignalFractionForm | None:
    """Return ``model``'s signal terms grouped under one total, or ``None``.

    A model has this form when it is a sum of two or more added signal terms
    standing side by side, and nothing else but background terms. A signal term
    is a component or a product of components whose one scale is an amplitude;
    a background term is one whose scale is a background (plan D16: it stays
    outside the total). A subtracted term, a sum inside a term, an existing
    fraction group, a term with no scale of its own, or a background standing
    between two signal terms leaves the model without the form.
    """
    root = model.expression_tree()
    if not isinstance(root, ExprSum) or any(
        isinstance(node, ExprSum) for term in root.terms for node in iter_nodes(term)
    ):
        return None
    kinds = model.parameter_kinds()
    identities = model.parameter_identities()
    positions: list[int] = []
    spans: list[tuple[int, int]] = []
    amplitudes: list[str] = []
    for position, (term, sign) in enumerate(zip(root.terms, root.signs, strict=True)):
        components = leaf_indices(term)
        scales = [
            name
            for name, identity in identities.items()
            if kinds[name] in _SCALE_KINDS and identity.component in components
        ]
        owned = [name for index in components if (name := model.scale_parameter_name(index))]
        if scales != owned or len(scales) != 1:
            return None
        if kinds[scales[0]] is ParameterKind.BACKGROUND:
            continue
        if sign != 1:
            return None
        positions.append(position)
        spans.append((min(components), max(components)))
        amplitudes.extend(scales)
    if len(positions) < 2 or positions[-1] - positions[0] != len(positions) - 1:
        return None

    group = (spans[0][0], spans[-1][1])
    open_parentheses = list(model.open_parentheses)
    close_parentheses = list(model.close_parentheses)
    open_parentheses[group[0]] += 1
    close_parentheses[group[1]] += 1
    grouped = CompositeModel(
        component_names=list(model.component_names),
        operators=list(model.operators),
        open_parentheses=open_parentheses,
        close_parentheses=close_parentheses,
        fraction_groups=[group],
    )
    (fractions,) = grouped.fraction_parameter_groups()
    return SignalFractionForm(
        grouped=grouped,
        amplitudes=tuple(amplitudes),
        total=next(
            name
            for name, identity in grouped.parameter_identities().items()
            if isinstance(identity, GroupAmplitude)
        ),
        fractions=tuple(fractions),
        carried=carried_names(
            identities, grouped.parameter_identities(), range(len(model.component_names))
        ),
    )
