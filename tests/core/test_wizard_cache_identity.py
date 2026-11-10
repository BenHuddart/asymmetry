"""Derived tables cannot stand in for complete single-run analyses."""

import pytest

import asymmetry.core.fitting.fit_wizard as fit_wizard_module
import asymmetry.core.fitting.global_fit_wizard as global_fit_wizard_module
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    SelectionMetric,
    build_fit_wizard_recommendation_for_templates,
)
from asymmetry.core.fitting.global_fit_wizard import (
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio,
)
from tests.core.test_global_fit_wizard import (
    _dataset_for,
    _force_serial_phase_one,
    _restrict_to_exp_constant_template,
    _stub_single_run_wizard,
)


@pytest.mark.parametrize("role", ["templates", "completed"])
def test_derived_score_table_cannot_replace_a_full_single_run_analysis(monkeypatch, role):
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=940 + idx,
            field=40.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + 0.1 * idx, "A_bg": 0.01},
        )
        for idx in (1, 2)
    ]
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    generated = []
    _stub_single_run_wizard(monkeypatch, (template,), calls=generated)
    _force_serial_phase_one(monkeypatch)
    source = build_fit_wizard_recommendation_for_templates(datasets[0], (template,))
    if role == "completed":
        plan = global_fit_wizard_module._plan_run_completion(
            datasets[0],
            (template,),
            source,
            rebin_factor=1,
            metric=SelectionMetric.AICC,
            sibling_values_by_template={},
        )
        fitted = {
            task.template.key: fit_wizard_module._execute_assessment_task(task)
            for task in plan.tasks
        }
        derived = global_fit_wizard_module._assemble_completed_run(
            plan, fitted, (template,), rebin_factor=1, metric=SelectionMetric.AICC
        )
    else:
        derived = source
    assert derived.build_signature != fit_wizard_module.single_fit_build_signature(
        None, dataset=datasets[0]
    )
    result = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
        existing_recommendations_by_run={int(datasets[0].run_number): derived},
    )
    assert set(generated) == {int(dataset.run_number) for dataset in datasets}
    assert result.single_fit_recommendations_by_run[int(datasets[0].run_number)] is not derived
