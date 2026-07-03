"""Discoverability hints for unknown component names in fit expressions.

These guard the single most-repeated clean-room API miss: using a *MODELS*
name (e.g. ``GaussianRelaxation``) where a fit expression requires the
*COMPONENTS* name (``Gaussian``). The error must name the correct expression
token rather than a bare ``Unknown component`` message.
"""

from __future__ import annotations

import pytest

from asymmetry.core.fitting.composite import (
    _MODEL_NAME_TO_COMPONENT,
    COMPONENTS,
    CompositeModel,
    UnknownComponentError,
    parse_component_expression,
)
from asymmetry.core.fitting.models import MODELS


@pytest.mark.parametrize(
    ("model_name", "expression_name"),
    sorted(_MODEL_NAME_TO_COMPONENT.items()),
)
def test_model_name_in_expression_points_to_component(
    model_name: str, expression_name: str
) -> None:
    with pytest.raises(UnknownComponentError) as excinfo:
        CompositeModel.from_expression(model_name)

    message = str(excinfo.value)
    assert model_name in message
    # The corrective name must be the expression (COMPONENTS) name...
    assert f"use '{expression_name}'" in message
    # ...and the message must explain the MODELS-vs-COMPONENTS distinction.
    assert "MODELS" in message
    assert "COMPONENTS" in message
    assert excinfo.value.suggestions == (expression_name,)


def test_alias_map_matches_live_registries() -> None:
    """The alias map must track reality: keys are MODELS-only names, values are
    real COMPONENTS names. Catches drift if a model or component is renamed."""
    for model_name, expression_name in _MODEL_NAME_TO_COMPONENT.items():
        assert model_name in MODELS, model_name
        assert model_name not in COMPONENTS, model_name
        assert expression_name in COMPONENTS, expression_name

    # Every MODELS key that is NOT also a COMPONENTS name should have an alias,
    # so no MODELS-only name slips through with an unhelpful generic message.
    models_only = {name for name in MODELS if name not in COMPONENTS}
    assert models_only <= set(_MODEL_NAME_TO_COMPONENT)


def test_typo_suggests_nearest_component() -> None:
    with pytest.raises(UnknownComponentError) as excinfo:
        CompositeModel.from_expression("Gausian")

    assert "Gaussian" in excinfo.value.suggestions
    assert "Did you mean" in str(excinfo.value)


def test_unrelated_unknown_name_still_raises_cleanly() -> None:
    with pytest.raises(UnknownComponentError) as excinfo:
        CompositeModel.from_expression("TotallyMadeUpThing")

    assert excinfo.value.name == "TotallyMadeUpThing"
    assert excinfo.value.suggestions == ()


def test_restricted_grammar_does_not_suggest_disallowed_names() -> None:
    """In a restricted grammar (e.g. parameter-domain models) a name that is a
    valid global component but disallowed here must NOT be echoed back as a
    suggestion, nor should any other globally-valid-but-disallowed name."""
    with pytest.raises(UnknownComponentError) as excinfo:
        parse_component_expression("Exponential", allowed_components={"Constant"})

    assert "Exponential" not in excinfo.value.suggestions
    assert all(s in {"Constant"} for s in excinfo.value.suggestions)


def test_valid_expression_still_parses() -> None:
    model = CompositeModel.from_expression("Gaussian + Constant")
    assert model.component_names == ["Gaussian", "Constant"]
