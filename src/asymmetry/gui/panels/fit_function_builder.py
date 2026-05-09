"""Dialog for building composite fit functions."""

from __future__ import annotations

from collections import defaultdict

from PySide6.QtWidgets import QWidget

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.gui.widgets.function_expression_builder import (
    ComponentSelectorButton as _ComponentSelectorButton,  # noqa: F401
)
from asymmetry.gui.widgets.function_expression_builder import (
    FunctionExpressionBuilderDialog,
)


def _build_components_by_category() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, definition in COMPONENTS.items():
        category = (definition.category or "General").strip() or "General"
        grouped[category].append(name)

    for names in grouped.values():
        names.sort()
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


class FitFunctionBuilderDialog(FunctionExpressionBuilderDialog):
    """Compose a custom fit function from predefined components."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_model: CompositeModel | None = None,
    ) -> None:
        initial_expression = (
            initial_model.component_expression_string()
            if initial_model is not None
            else "Exponential + Constant"
        )
        super().__init__(
            title="Build Fit Function",
            expression_prefix="A(t)",
            components_by_category=_build_components_by_category(),
            component_definitions=COMPONENTS,
            model_parser=CompositeModel.from_expression,
            initial_expression=initial_expression,
            expression_placeholder="e.g. Exponential + ( Gaussian * Constant )",
            parent=parent,
        )

    def get_composite_model(self) -> CompositeModel | None:
        """Return the model produced when the dialog is accepted."""
        model = self.built_model()
        return model if isinstance(model, CompositeModel) else None
