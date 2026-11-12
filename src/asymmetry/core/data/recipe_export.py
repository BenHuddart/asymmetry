"""Disclosure and actual written-observation context for reduced-curve exports."""

from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe


def exported_recipe(payload, *, x_range=None, bake_waterfall_offset=True):
    """Describe selection and applied display offset without changing exported numbers."""
    value = payload.get("effective_recipe")
    if value is None:
        return None
    recipe = EffectiveReductionRecipe.from_dict(value)
    offset = float(payload.get("waterfall_offset", 0.0))
    return recipe.curve_operation(
        "export_curve",
        x_range=None if x_range is None else sorted(x_range),
        applied_value_offset=offset if bake_waterfall_offset else 0.0,
        plotted_value_offset=offset,
    ).publishable()


def disclose_result_recipes(value):
    """Project only recipe fields at a public JSON boundary, including nested fits/scans."""
    if isinstance(value, dict):
        return {
            key: (
                EffectiveReductionRecipe.from_dict(item).publishable() if item is not None else None
            )
            if key == "effective_recipe"
            else disclose_result_recipes(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [disclose_result_recipes(item) for item in value]
    return value
