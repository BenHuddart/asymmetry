"""Run the global fit wizard on a synthetic field series."""

from __future__ import annotations

import sys

import numpy as np

from asymmetry.core.data import MuonDataset
from asymmetry.core.fitting import (
    CompositeModel,
    build_global_fit_wizard_screening_recommendation,
    build_global_fit_wizard_recommendation,
    merge_global_fit_wizard_recommendations,
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
    datasets = [
        _synthetic_dataset(
            4100 + index,
            field=50.0 * (index + 1),
            temperature=5.0,
            lambda_value=lambda_value,
        )
        for index, lambda_value in enumerate((0.15, 0.25, 0.55, 0.90))
    ]

    screening = build_global_fit_wizard_screening_recommendation(
        datasets,
    )
    top_screening = screening.sorted_prescreen_assessments()[:2]
    selected_keys = tuple(assessment.template.key for assessment in top_screening)

    optimized = build_global_fit_wizard_recommendation(
        datasets,
        selected_template_keys=selected_keys,
    )
    recommendation = merge_global_fit_wizard_recommendations(screening, optimized)
    assessment = recommendation.recommended_assessment

    print(f"axis = {recommendation.series_axis_label}")
    print("screening:")
    for candidate in screening.sorted_prescreen_assessments()[:5]:
        print(
            f"  {candidate.template.title}: {screening.metric.value}={candidate.metric_value(screening.metric):.3f} "
            f"[{screening.optimization_status_for_key(candidate.template.key)}]"
        )
    print(f"selected for optimisation = {selected_keys}")
    print(f"summary = {recommendation.summary}")
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
