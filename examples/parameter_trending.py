"""Evaluate parameter-vs-field composite models."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting import ParameterCompositeModel, component_names_for_x


def main() -> None:
    print("field components:", ", ".join(component_names_for_x("field")[:8]), "...")

    model = ParameterCompositeModel(
        component_names=["DiffusionLF_2D", "Redfield", "Lambda_bg"],
        operators=["+", "+"],
    )
    print("formula:", model.formula_string())

    field = np.array([20.0, 50.0, 100.0, 300.0, 1000.0], dtype=float)
    values = model.function(
        field,
        A=3.0,
        D_2D=0.7,
        D_perp=0.1,
        D=14.0,
        nu=9.0,
        m=2.0,
        lambda_BG=0.05,
    )

    for b, lam in zip(field, values, strict=True):
        print(f"B = {b:7.1f} G -> lambda = {lam:.4f} us^-1")


if __name__ == "__main__":
    main()
