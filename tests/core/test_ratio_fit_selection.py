"""Histogram Gaussian selection uses mathematical observations, never display sentinels."""

import pickle

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine, _dataset_observations
from asymmetry.core.fitting.likelihood import FitInputError
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform.preparation import prepare_wizard_dataset
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.core.test_binning_entry_points import _run


def constant(time, c):
    return np.full_like(time, c)


def test_production_fit_omits_empty_and_one_sided_without_changing_grid():
    dataset = reduce_run(_run([0, 0, 10, 60, 40], [0, 10, 0, 40, 60]), ReductionSettings())
    result = FitEngine().fit(dataset, constant, ParameterSet([Parameter("c", 0)]))
    assert result.success
    assert result.dof == 1
    assert len(result.residuals) == 2
    assert result.chi_squared == pytest.approx(2 * 20**2 / 96)
    assert result.observation_report.to_dict() == {
        "used": 2,
        "omitted": 3,
        "undefined_ratio": 1,
        "zero_variance": 2,
    }
    assert dataset.n_points == 5
    np.testing.assert_array_equal(dataset.run.histograms[0].counts, [0, 0, 10, 60, 40])


def test_crop_counts_first_merge_and_worker_preserve_final_selection():
    dataset = reduce_run(_run([0, 10, 0, 20, 30, 0], [0, 0, 10, 20, 10, 0]), ReductionSettings())
    cropped = dataset.time_range(dataset.time[1], dataset.time[4])
    assert len(_dataset_observations(cropped.fit_record(), None, None).time) == 2
    merged = prepare_wizard_dataset(cropped, 2)
    record = pickle.loads(pickle.dumps(merged.fit_record()))
    observations = _dataset_observations(record, None, None)
    assert observations.report.used == 2
    assert observations.report.omitted == 0
    np.testing.assert_allclose(observations.values, [0, 25])
    assert record.histogram_observation is None
    assert not record.run.histograms


def test_custom_sentinel_values_remain_strict_generic_observations():
    dataset = MuonDataset(
        np.arange(3.0), np.zeros(3), np.full(3, 100.0), run=_run([0] * 3, [0] * 3)
    )
    assert len(_dataset_observations(dataset, None, None).time) == 3
    dataset.error[0] = 0
    with pytest.raises(FitInputError, match="finite positive"):
        _dataset_observations(dataset, None, None)


def test_no_usable_observations_is_a_fit_error_with_reasons():
    dataset = reduce_run(_run([0, 1], [0, 0]), ReductionSettings())
    assert dataset.n_points == 2
    with pytest.raises(FitInputError, match="used 0.*undefined ratio 1.*zero variance 1"):
        FitEngine().fit(dataset, constant, ParameterSet([Parameter("c", 0)]))


@pytest.mark.parametrize("mode", ["green_minus_red", "green_plus_red"])
def test_period_final_variance_and_undefined_constituents(mode):
    from asymmetry.core.transform.preparation import prepare_histogram_asymmetry

    red = _run([0, 0, 10, 10], [10, 0, 0, 10])
    green = _run([60, 60, 0, 10], [40, 40, 10, 10])
    run = _run(
        [1] * 4, [1] * 4, period_mode=mode, period_histograms=[red.histograms, green.histograms]
    )
    result = prepare_histogram_asymmetry(run, merge=1, facility=None, reference=None)
    if mode == "green_plus_red":
        np.testing.assert_array_equal(result.ratio_validity.reasons, [0, 0, 0, 0])
        np.testing.assert_allclose(result.asymmetry, [100 / 11, 20, 0, 0])
        np.testing.assert_allclose(
            result.error,
            200
            * np.sqrt(
                np.array([60 * 50 / 110**3, 60 * 40 / 100**3, 10 * 10 / 20**3, 20 * 20 / 40**3])
            ),
        )
        return
    np.testing.assert_array_equal(result.ratio_validity.reasons, [0, 1, 2, 0])
    # One-sided red has zero mathematical variance, rather than a 100%-error contribution.
    assert result.error[0] == pytest.approx(np.sqrt(96))
    assert result.error[3] == pytest.approx(np.sqrt(1000))
    expected = 120 if mode == "green_minus_red" else -80
    assert result.asymmetry[0] == pytest.approx(expected)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (1.25, 0.8)])
def test_signed_corrected_zero_and_tiny_defined_ratio_remain_usable(alpha, beta):
    from asymmetry.core.transform.asymmetry import ratio_with_count_errors

    f = np.array([-5.0, 0.0, -10.0 + 1e-8, 0.0])
    b = np.array([10.0, 10.0, beta / alpha * 10.0, 0.0])
    ef, eb = np.sqrt([5.0, 5.0, 5.0, 0.0]), np.sqrt([10.0, 10.0, 10.0, 0.0])
    a, error, validity = ratio_with_count_errors(f, b, ef, eb, alpha, beta)
    np.testing.assert_array_equal(validity.reasons, [0, 0, 0, 1])
    d = beta * f[:3] + alpha * b[:3]
    np.testing.assert_allclose(a[:3], (f[:3] - alpha * b[:3]) / d)
    expected = (
        abs(alpha * (1 + beta))
        * np.sqrt(b[:3] ** 2 * ef[:3] ** 2 + f[:3] ** 2 * eb[:3] ** 2)
        / d**2
    )
    np.testing.assert_allclose(error[:3], expected)


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
def test_global_objective_and_dof_use_selected_points(strategy):
    datasets = []
    for run_number in [1, 2]:
        run = _run([0, 0, 60, 40, 55, 45], [0, 10, 40, 60, 45, 55])
        run.run_number = run_number
        datasets.append(reduce_run(run, ReductionSettings()))
    results, _ = FitEngine().global_fit(
        datasets,
        constant,
        ["c"],
        [],
        {ds.run_number: ParameterSet([Parameter("c", 0)]) for ds in datasets},
        strategy=strategy,
    )
    for result in results.values():
        assert result.success
        assert len(result.residuals) == 4
        assert result.dof == 3
        assert result.observation_report.used == 4
        assert result.observation_report.omitted == 2


def test_single_wizard_real_fit_completion_rebuild_and_history():
    from dataclasses import replace

    import asymmetry.core.fitting.fit_wizard as single
    import asymmetry.core.fitting.global_fit_wizard as global_wizard
    from asymmetry.core.fitting.information_criteria import SelectionMetric

    ds = reduce_run(_run([0, 0, 51, 49] * 16, [0, 10, 49, 51] * 16), ReductionSettings())
    template = single.build_null_baseline_templates()[0]
    recommendation = single.build_fit_wizard_recommendation_for_templates(
        ds, (template,), variant_budget=1
    )
    row = recommendation.assessments[0]
    assert row.fit_result.success
    assert row.sample_count == 32
    assert row.fit_result.observation_report.omitted == 32
    assert np.isnan(row.residual_fft_peak_snr)
    assert any("FFT unavailable" in message for message in row.fit_result.warnings)
    observations = _dataset_observations(ds, None, None)
    predicted = template.model.function(
        observations.time, **{p.name: p.value for p in row.fit_result.parameters}
    )
    assert row.fit_result.chi_squared == pytest.approx(
        np.sum(((observations.values - predicted) / observations.sigma) ** 2)
    )
    source = replace(
        recommendation, build_signature=single.single_fit_build_signature(None, dataset=ds)
    )
    plan = global_wizard._plan_run_completion(
        ds,
        (template,),
        source,
        rebin_factor=2,
        metric=SelectionMetric.AICC,
        sibling_values_by_template={},
    )
    fitted = single._execute_assessment_task(plan.tasks[0])
    assert fitted.sample_count == 16
    assert fitted.fit_result.observation_report.omitted == 16
    completed = global_wizard._assemble_completed_run(
        plan, {template.key: fitted}, (template,), rebin_factor=2, metric=SelectionMetric.AICC
    )
    rebuilt = single.assessment_with_curves(completed, template.key, ds)
    assert rebuilt.fit_result.chi_squared == fitted.fit_result.chi_squared
    restored = single.deserialize_fit_wizard_recommendation(
        single.serialize_fit_wizard_recommendation(completed)
    )
    assert (
        restored.assessments[0].fit_result.observation_report
        == fitted.fit_result.observation_report
    )
    assert restored.assessments[0].fit_result.dof == fitted.fit_result.dof


def test_insufficient_and_empty_global_cohort_are_explicit():
    usable = reduce_run(_run([60, 40], [40, 60]), ReductionSettings())
    insufficient = reduce_run(_run([0, 60], [0, 40]), ReductionSettings())
    failed = FitEngine().fit(insufficient, constant, ParameterSet([Parameter("c", 0)]))
    assert not failed.success
    assert "Insufficient usable" in failed.message
    assert failed.observation_report.used == 1
    empty_run = _run([0, 0], [0, 0])
    empty_run.run_number = 2
    empty = reduce_run(empty_run, ReductionSettings())
    with pytest.raises(FitInputError, match="run 2: no usable"):
        FitEngine().global_fit(
            [usable, empty],
            constant,
            ["c"],
            [],
            {n: ParameterSet([Parameter("c", 0)]) for n in [1, 2]},
        )


def test_rebinned_comparison_uses_exact_fit_residual_support():
    from dataclasses import replace

    import asymmetry.core.fitting.fit_wizard as single
    from asymmetry.core.fitting.model_comparison import summarise_single_candidates

    ds = reduce_run(_run([0, 0, 51, 49] * 16, [0, 10, 49, 51] * 16), ReductionSettings())
    prepared = prepare_wizard_dataset(ds, 2)
    template = single.build_null_baseline_templates()[0]
    rec = single.build_fit_wizard_recommendation_for_templates(
        prepared, (template,), variant_budget=1
    )
    rec = replace(rec, rebin_factor=2)
    curves = summarise_single_candidates(rec, ds, rec.metric)[0].runs[0].curves
    row = rec.assessments[0]
    assert len(curves.residuals[0]) == row.sample_count
    np.testing.assert_array_equal(curves.residuals[0], row.fit_result.residual_time)
    assert np.sum(curves.residuals[1] ** 2) == pytest.approx(row.fit_result.chi_squared)


def test_generic_shape_and_list_boundaries_are_preserved():
    ds = MuonDataset([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    assert len(_dataset_observations(ds, None, None).time) == 3
    ds.error = [1.0, 1.0]
    with pytest.raises(FitInputError, match="same length"):
        _dataset_observations(ds, None, None)


def test_old_counts_first_signature_rebuild_does_not_become_curve_mean():
    import json

    from asymmetry.core.transform.preparation import (
        wizard_observation_identity,
        wizard_observation_matches,
        wizard_rebuild_dataset,
    )

    ds = reduce_run(_run([90, 1] * 4, [10, 9] * 4), ReductionSettings())
    policy = "counts_first_observation_v1"
    signature = json.dumps(
        {
            "wizard_preparation": policy,
            "observation": wizard_observation_identity(ds, policy=policy),
        }
    )
    rebuilt = wizard_rebuild_dataset(ds, 2, signature)
    np.testing.assert_allclose(rebuilt.asymmetry, 100 * 72 / 110)
    assert not wizard_observation_matches(ds, signature)


def test_cosubtract_known_signed_variance_and_counts_first_merge():
    from asymmetry.core.data.combine import combine_runs, reduce_combined_run

    ds = reduce_combined_run(
        combine_runs(
            [_run([5, 10, 0, 5], [10, 10, 0, 10]), _run([10, 10, 0, 15], [0, 0, 0, 0])], sign=-1
        )
    )
    np.testing.assert_array_equal(ds.ratio_validity.reasons, [0, 0, 1, 1])
    assert ds.asymmetry[0] == -300
    assert ds.error[0] > 0
    assert ds.histogram_observation is not None
    merged = prepare_wizard_dataset(ds, 2)
    assert len(merged.time) == 2
    assert merged.ratio_validity.report().used == 1


@pytest.mark.parametrize("invalid", [256, -256, 0.5, 3, np.nan])
def test_stored_reason_codes_cannot_wrap_into_usable(invalid):
    from asymmetry.core.data.ratio_validity import RatioValidity

    with pytest.raises(ValueError, match="recognised reason"):
        RatioValidity(np.array([invalid]))


def test_joint_different_series_share_selected_objective():
    from asymmetry.core.fitting.joint import JointSeriesProblem, SharedParameter, fit_joint

    problems = []
    for n in [1, 2]:
        run = _run([0, 0, 60, 40, 55, 45], [0, 10, 40, 60, 45, 55])
        run.run_number = n
        ds = reduce_run(run, ReductionSettings())
        problems.append(
            JointSeriesProblem(
                key=n,
                datasets=[ds],
                model_fn=constant,
                global_params=["c"],
                local_params=[],
                initial_params={n: ParameterSet([Parameter("c", 0)])},
            )
        )
    result = fit_joint(
        problems, [SharedParameter(name="shared", members={1: "c", 2: "c"}, value=0)]
    )
    assert result.success
    assert result.dof == 7
    for series in result.series_results.values():
        for fit in series.values():
            assert fit.observation_report.used == 4
            assert fit.residuals.size == 4


def test_historical_curve_mean_retains_display_error_arithmetic():
    from asymmetry.core.transform.preparation import wizard_rebuild_dataset

    ds = reduce_run(_run([0, 10], [10, 0]), ReductionSettings())
    old = wizard_rebuild_dataset(ds, 2, "")
    assert old.ratio_validity is None
    assert old.error[0] == pytest.approx(100 / np.sqrt(2))


def test_cosubtract_worker_drops_detector_variance_arrays():
    from asymmetry.core.data.combine import combine_runs, reduce_combined_run

    ds = reduce_combined_run(
        combine_runs([_run([5] * 1000, [10] * 1000), _run([10] * 1000, [0] * 1000)], sign=-1)
    )
    record = ds.fit_record()
    assert "detector_variance" not in record.metadata["combination"]
    assert "detector_variance" not in record.run.metadata["combination"]
    assert record.ratio_validity.report().used == 1000
    assert len(pickle.dumps(record)) < len(pickle.dumps(ds)) / 2


def test_known_fit_record_mean_retains_final_definedness_and_variance():
    ds = reduce_run(_run([0, 60, 0, 40], [0, 40, 10, 60]), ReductionSettings())
    merged = prepare_wizard_dataset(ds.fit_record(), 2)
    np.testing.assert_array_equal(merged.ratio_validity.reasons, [1, 0])
    observed = _dataset_observations(merged, None, None)
    assert observed.report.used == 1
    np.testing.assert_allclose(observed.values, [-60.0])
    np.testing.assert_allclose(observed.sigma, [np.sqrt(24.0)])
    assert not pickle.loads(pickle.dumps(merged)).ratio_validity.reasons.flags.writeable


def test_existing_positional_gaussian_window_still_selects_same_support():
    from asymmetry.core.fitting.likelihood import GaussianObservations

    observed = GaussianObservations(np.arange(5.0), np.zeros(5), np.ones(5), None, 1.0, 2.0)
    np.testing.assert_array_equal(observed.time, [1.0, 2.0])
    assert observed.report is None
    restored = pickle.loads(pickle.dumps(observed))
    np.testing.assert_array_equal(restored.time, [1.0, 2.0])


def test_global_prescreen_completion_and_partition_count_only_fitted_observations():
    from dataclasses import replace

    import asymmetry.core.fitting.fit_wizard as single
    import asymmetry.core.fitting.global_fit_wizard as global_wizard

    template = single.build_null_baseline_templates()[0]
    datasets = []
    sources = {}
    for number in [1, 2]:
        run = _run([0, 0, 52, 48] * 12, [0, 10, 48, 52] * 12)
        run.run_number = number
        dataset = reduce_run(run, ReductionSettings())
        datasets.append(dataset)
        source = single.build_fit_wizard_recommendation_for_templates(
            dataset, (template,), variant_budget=1
        )
        assert source.analysed_points == 24
        sources[number] = replace(
            source, build_signature=single.single_fit_build_signature(None, dataset=dataset)
        )
    rows, _ = global_wizard._build_single_fit_prescreen_assessments(
        datasets,
        {n: r.fingerprint for n, r in sources.items()},
        [template],
        single_fit_recommendations_by_run=sources,
        current_parameter_types={},
        current_values={},
        parameter_bounds={},
        axis_key="run_number",
        metric=sources[1].metric,
        repair_partial_incomplete=False,
    )
    row = rows[template.key]
    assert row.sample_count == 48
    assert any("undefined ratio" in message for message in row.series_warnings)
    table, _, count = global_wizard._partition_inputs_from_prescreen(
        datasets, rows, analysed_points_by_run={n: r.analysed_points for n, r in sources.items()}
    )
    assert count == 48
    for number in [1, 2]:
        assert table[number][template.key] == pytest.approx(
            row.fit_results_by_run[number].chi_squared + np.log(24)
        )
    plan = global_wizard._plan_run_completion(
        datasets[0],
        (template,),
        sources[1],
        rebin_factor=2,
        metric=sources[1].metric,
        sibling_values_by_template={},
    )
    assert plan.analysed_points == 12
