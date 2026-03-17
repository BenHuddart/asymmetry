"""Build and evaluate a composite time-domain fit model."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting import CompositeModel


def main() -> None:
    model = CompositeModel(["Exponential", "Oscillatory", "Constant"], operators=["+", "+"])
    print("formula:", model.formula_string())

    kwargs = {
        "A_1": 18.0,
        "Lambda": 0.45,
        "A_2": 6.5,
        "frequency": 1.8,
        "phase": 0.2,
        "A_bg": 0.4,
    }

    time = np.linspace(0.0, 6.0, 6)
    total = model.function(time, **kwargs)
    components = model.evaluate_components(time, additive_only=True, **kwargs)

    damped_model = CompositeModel(["Exponential", "Oscillatory"], operators=["*"])
    damped_kwargs = {
        "A_1": 12.0,
        "Lambda": 0.35,
        "frequency": 2.2,
        "phase": 0.0,
    }
    damped_total = damped_model.function(time, **damped_kwargs)

    print("t:", np.array2string(time, precision=2))
    print("A(t):", np.array2string(total, precision=3))
    print("additive components:", [name for name, _ in components])
    print("damped formula:", damped_model.formula_string())
    print("damped A(t):", np.array2string(damped_total, precision=3))
    print("damped parameters:", sorted(damped_kwargs))


if __name__ == "__main__":
    main()
