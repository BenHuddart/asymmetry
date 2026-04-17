"""Run the global fit wizard on a synthetic field series."""

from __future__ import annotations

import sys

import numpy as np

from asymmetry.core.data import MuonDataset
from asymmetry.core.fitting import (
    CompositeModel,
    build_global_fit_wizard_recommendation,
)


def _synthetic_dataset(
    run_number: int,
    *,
    field: float,
    temperature: float,
    lambda_value: float,
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 240)
    error = np.full_like(time, 0.01)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    asymmetry = model.function(
        time,
        A_1=0.2,
        Lambda=lambda_value,
        A_bg=0.01,
    )
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": run_number,
            "run_label": str(run_number),
            "field": field,
            "temperature": temperature,
        },
    )


def main() -> None:
    strategy = sys.argv[1] if len(sys.argv) > 1 else "legacy"
    instrumentation: dict[str, object] = {}
    datasets = [
        _synthetic_dataset(
            4100 + index,
            field=50.0 * (index + 1),
            temperature=5.0,
            lambda_value=lambda_value,
        )
        for index, lambda_value in enumerate((0.15, 0.25, 0.55, 0.90))
    ]

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        search_strategy=strategy,
        instrumentation=instrumentation,
    )
    assessment = recommendation.recommended_assessment

    print(f"strategy = {strategy}")
    print(f"axis = {recommendation.series_axis_label}")
    print(f"summary = {recommendation.summary}")
    counters = instrumentation.get("counters")
    if isinstance(counters, dict):
        print("counters:")
        for key in sorted(counters):
            print(f"  {key} = {counters[key]}")
    if assessment is None:
        return

    print(f"model = {assessment.template.title}")
    print("parameter roles:")
    for parameter in assessment.parameter_recommendations:
        print(
            f"  {parameter.name}: {parameter.recommended_role} (delta={parameter.score_delta:.3f})"
        )


if __name__ == "__main__":
    main()
