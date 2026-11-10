"""Wizard observations preserve histogram membership rather than averaging ratios."""

import pickle
from dataclasses import replace

import numpy as np
import pytest

import asymmetry.core.fitting.fit_wizard as single
import asymmetry.core.fitting.global_fit_wizard as global_wizard
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import PhysicsClass
from asymmetry.core.fitting.global_fit_wizard import _separable_search_datasets
from asymmetry.core.fitting.wizard_scope import WizardScope
from asymmetry.core.transform.preparation import (
    histogram_observation,
    prepare_histogram_asymmetry,
    prepare_wizard_dataset,
    wizard_observation_identity,
)
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.core.test_binning_entry_points import _run


def test_global_search_prepares_count_ratio_and_independent_error():
    dataset = reduce_run(_run([90, 1], [10, 9]), ReductionSettings())
    prepared = _separable_search_datasets([dataset], search_rebin_factor=2, instrumentation=None)[0]
    np.testing.assert_allclose(prepared.asymmetry, [100 * 72 / 110])
    np.testing.assert_allclose(prepared.error, [200 * np.sqrt(19**2 * 91 + 91**2 * 19) / 110**2])
    np.testing.assert_allclose(prepared.time, [0.05])


def observed(run, *, merge=1, reference=None):
    reduced = prepare_histogram_asymmetry(run, merge=merge, facility=None, reference=reference)
    return MuonDataset(
        reduced.time,
        reduced.asymmetry,
        reduced.error,
        run=run,
        histogram_observation=histogram_observation(
            run,
            reference=reference,
            binning=replace(
                BinningSpec(
                    mode=run.grouping.get("binning_mode", "fixed"),
                    factor=run.grouping.get("bunching_factor", 1),
                    bin0_us=run.grouping.get("bin0_us", 0.08),
                    bin10_us=run.grouping.get("bin10_us", 0.25),
                ),
                merge=merge,
            ),
        ),
    )


@pytest.mark.parametrize("factor", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "forward,backward", [([90, 1], [10, 9]), ([900, 10], [100, 90]), ([9, 18], [1, 2])]
)
def test_factor_controls_keep_complete_groups_and_independent_error(factor, forward, backward):
    dataset = observed(_run(forward * 3, backward * 3))
    result = prepare_wizard_dataset(dataset, factor)
    n = dataset.n_points // factor * factor
    f = np.array(forward * 3)[:n].reshape(-1, factor).sum(axis=1)
    b = np.array(backward * 3)[:n].reshape(-1, factor).sum(axis=1)
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(result.error, 200 * np.sqrt(b * b * f + f * f * b) / (f + b) ** 2)


@pytest.mark.parametrize("mode", ["fixed", "variable", "constant_error"])
def test_base_toolbar_crop_and_wizard_preserve_exact_members(mode):
    f, b = np.arange(1.0, 102), np.arange(103.0, 204)
    run = _run(f, b, binning_mode=mode, bunching_factor=2, bin0_us=0.04, bin10_us=0.04)
    run.histograms = [replace(h, t0_time_us=0.073) for h in run.histograms]
    dataset = observed(run, merge=2)
    # Crop starts at output index 1. Applying factor 3 on the full grid first
    # would include the distinct first group's observations and change every ratio.
    cropped = dataset.time_range(dataset.time[1], dataset.time[7])
    result = prepare_wizard_dataset(cropped, 3)
    base = dataset.histogram_observation.binning.edges(len(f), 0.1, -0.023)
    edges = base[1:9:3]
    fs = np.array([f[a:z].sum() for a, z in zip(edges[:-1], edges[1:])])
    bs = np.array([b[a:z].sum() for a, z in zip(edges[:-1], edges[1:])])
    np.testing.assert_allclose(result.asymmetry, 100 * (fs - bs) / (fs + bs))
    np.testing.assert_allclose(
        result.error, 200 * np.sqrt(bs * bs * fs + fs * fs * bs) / (fs + bs) ** 2
    )
    np.testing.assert_allclose(result.time, (edges[:-1] + edges[1:]) * 0.05 - 0.073, atol=1e-14)
    np.testing.assert_array_equal(run.histograms[0].counts, f)


def test_workflow_window_is_selected_before_further_merging():
    f, b = np.arange(1.0, 21), np.arange(21.0, 41)
    dataset = reduce_run(_run(f, b), ReductionSettings(rebin=2, t_min=0.25, t_max=1.5))
    result = prepare_wizard_dataset(dataset, 3)
    fs = f[2:14].reshape(2, 6).sum(axis=1)
    bs = b[2:14].reshape(2, 6).sum(axis=1)
    np.testing.assert_allclose(result.asymmetry, 100 * (fs - bs) / (fs + bs))
    np.testing.assert_allclose(result.time, [0.45, 1.05])


@pytest.mark.parametrize("mode", ["red", "green", "green_minus_red", "green_plus_red"])
def test_period_observations_and_worker_payload(mode):
    red, green = _run([90, 1] * 100, [10, 9] * 100), _run([2, 80] * 100, [8, 10] * 100)
    run = _run(
        [1] * 200,
        [1] * 200,
        period_mode=mode,
        period_histograms=[red.histograms, green.histograms],
        period_reduced=[{}, {}],
    )
    result = prepare_wizard_dataset(observed(run), 2)
    ra, ga = 100 * 72 / 110, 100 * 64 / 100
    rv, gv = 40000 * 91 * 19 / 110**3, 40000 * 82 * 18 / 100**3
    np.testing.assert_allclose(
        result.asymmetry,
        {"red": ra, "green": ga, "green_minus_red": ga - ra, "green_plus_red": 100 * 136 / 210}[
            mode
        ],
    )
    np.testing.assert_allclose(
        result.error,
        np.sqrt(
            {
                "red": rv,
                "green": gv,
                "green_minus_red": rv + gv,
                "green_plus_red": 40000 * 173 * 37 / 210**3,
            }[mode]
        ),
    )
    record = result.fit_record()
    assert record.histogram_observation is None
    assert record.run.histograms == []
    assert "period_histograms" not in record.run.grouping
    assert "period_reduced" not in record.run.grouping
    assert len(pickle.dumps(record)) < len(pickle.dumps(result)) / 2


def test_background_shared_variance_and_signed_count_support():
    run = _run(
        [8, 8, 10, 1, 20, 1],
        [8, 8, 5, 2, 7, 3],
        first_good_bin=2,
        last_good_bin=5,
        background_correction=True,
        background_mode="range",
        background_ranges=[0, 1],
    )
    result = prepare_wizard_dataset(observed(run), 4)
    f, b = 32 - 32, 17 - 32
    vf, vb = 32 + 16 * 4, 17 + 16 * 4
    np.testing.assert_allclose(result.asymmetry, [100 * (f - b) / (f + b)])
    np.testing.assert_allclose(
        result.error, [200 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2]
    )


@pytest.mark.parametrize("rotated", [False, True])
def test_custom_and_rotated_curves_keep_means_even_with_raw_run(rotated):
    dataset = observed(_run([90, 1], [10, 9]))
    if rotated:
        dataset.metadata["rotating_frame"] = {"frequency": 1}
    else:
        dataset.histogram_observation = None
    np.testing.assert_allclose(prepare_wizard_dataset(dataset, 2).asymmetry, [0])
    assert dataset.merge_curve_bins(2).histogram_observation is None


def test_context_snapshot_identity_and_cancellation():
    run = _run([90, 1] * 8, [10, 9] * 8)
    dataset = observed(run)
    identity = wizard_observation_identity(dataset)
    run.grouping["alpha"] = 1.2
    assert wizard_observation_identity(dataset) == identity
    np.testing.assert_allclose(prepare_wizard_dataset(dataset, 2).asymmetry, 100 * 72 / 110)
    assert wizard_observation_identity(observed(run)) != identity
    assert wizard_observation_identity(dataset.time_range(0.1, 1)) != identity

    def cancel():
        raise single.FitCancelledError("cancelled")

    with pytest.raises(single.FitCancelledError, match="cancelled"):
        prepare_wizard_dataset(dataset, 2, cancel_check=cancel)


def test_global_decimated_search_uses_actual_count_observations(monkeypatch):
    dataset = observed(_run([90, 1] * 16, [10, 9] * 16))
    monkeypatch.setattr(global_wizard, "_decimation_factor_for_engine", lambda engine: 2)
    monkeypatch.setattr(global_wizard, "_decimation_is_nyquist_safe", lambda *args: True)
    result, factor = global_wizard._decimated_datasets_for_search(
        [dataset], engine="test", aggregate_fingerprint=None, instrumentation=None
    )
    assert factor == 2
    np.testing.assert_allclose(result[0].asymmetry, 100 * 72 / 110)


def test_single_real_fit_scores_and_rebuilds_the_prepared_record(monkeypatch):
    dataset = observed(_run([90, 1] * 64, [10, 9] * 64))
    monkeypatch.setattr(single, "analysis_rebin_factor", lambda *args, **kwargs: 2)
    seen = []
    execute = single._execute_assessment_task

    def capture(task, cancel_callback=None):
        seen.append(task.dataset)
        return execute(task, cancel_callback)

    monkeypatch.setattr(single, "_execute_assessment_task", capture)
    scope = WizardScope(physics=frozenset({PhysicsClass.GENERIC_RELAXATION}))
    recommendation = single.build_fit_wizard_recommendation(
        dataset, scope=scope, max_workers=1, refine_top_candidates=0
    )
    assert recommendation.rebin_factor == 2
    assert seen
    for record in seen:
        np.testing.assert_allclose(record.asymmetry, 100 * 72 / 110)
        assert record.histogram_observation is None
        assert not record.run.histograms
    assessment = recommendation.assessments[0]
    dataset.run.histograms[0].counts.view()[:] = 10
    rebuilt = single.assessment_with_curves(recommendation, assessment.template.key, dataset)
    expected = single._with_dense_curves(assessment, prepare_wizard_dataset(dataset, 2))
    np.testing.assert_array_equal(rebuilt.fitted_time, expected.fitted_time)
    np.testing.assert_array_equal(rebuilt.fitted_curve, expected.fitted_curve)
    assert assessment.sample_count == 64
    prepared = prepare_wizard_dataset(dataset, 2)
    for row in recommendation.assessments:
        predicted = row.template.model.function(
            prepared.time, **{p.name: p.value for p in row.fit_result.parameters}
        )
        objective = np.sum(((prepared.asymmetry - predicted) / prepared.error) ** 2)
        np.testing.assert_allclose(row.fit_result.chi_squared, objective, rtol=1e-10, atol=1e-10)
    historical = replace(recommendation, build_signature="")
    old = single.assessment_with_curves(historical, assessment.template.key, dataset)
    old_expected = single._with_dense_curves(assessment, dataset.merge_curve_bins(2))
    np.testing.assert_array_equal(old.fitted_time, old_expected.fitted_time)
    restored = single.deserialize_fit_wizard_recommendation(
        single.serialize_fit_wizard_recommendation(historical)
    )
    assert restored.build_signature == ""
    assert restored.assessments[0].fit_result.chi_squared == assessment.fit_result.chi_squared


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_actual_nonuniform_wizard_detection_retains_fft_diagnostic(mode):
    run = _run(
        np.arange(1.0, 102), np.arange(103.0, 204), binning_mode=mode, bin0_us=0.14, bin10_us=0.5
    )
    dataset = observed(run)
    assert np.ptp(np.diff(dataset.time)) > 0.01
    with pytest.raises(ValueError, match="uniform"):
        single.build_fit_wizard_recommendation(dataset, max_workers=1)


def test_global_completion_fits_and_rebuilds_count_record():
    from asymmetry.core.fitting.information_criteria import SelectionMetric

    dataset = observed(_run([55, 1] * 32, [45, 1] * 32))
    template = single.build_null_baseline_templates()[0]
    source = single.build_fit_wizard_recommendation_for_templates(
        dataset, (template,), variant_budget=1
    )
    source = replace(
        source, build_signature=single.single_fit_build_signature(None, dataset=dataset)
    )
    plan = global_wizard._plan_run_completion(
        dataset,
        (template,),
        source,
        rebin_factor=2,
        metric=SelectionMetric.AICC,
        sibling_values_by_template={},
    )
    assert len(plan.tasks) == 1
    record = plan.tasks[0].dataset
    np.testing.assert_allclose(record.asymmetry, 100 * 10 / 102)
    assert record.histogram_observation is None
    assert not record.run.histograms
    fitted = single._execute_assessment_task(plan.tasks[0])
    predicted = template.model.function(
        record.time, **{p.name: p.value for p in fitted.fit_result.parameters}
    )
    np.testing.assert_allclose(
        fitted.fit_result.chi_squared,
        np.sum(((record.asymmetry - predicted) / record.error) ** 2),
        rtol=1e-10,
        atol=1e-10,
    )
    completed = global_wizard._assemble_completed_run(
        plan, {template.key: fitted}, (template,), rebin_factor=2, metric=SelectionMetric.AICC
    )
    dataset.run.histograms[0].counts.view()[:] = 10
    rebuilt = single.assessment_with_curves(completed, template.key, dataset)
    eager = single._with_dense_curves(fitted, plan.analysis_dataset)
    np.testing.assert_array_equal(rebuilt.fitted_time, eager.fitted_time)
    np.testing.assert_array_equal(rebuilt.fitted_curve, eager.fitted_curve)
    # A cell with old curve-mean semantics cannot qualify at the same factor.
    old = replace(completed, build_signature="")
    cold = global_wizard._plan_run_completion(
        dataset,
        (template,),
        old,
        rebin_factor=2,
        metric=SelectionMetric.AICC,
        sibling_values_by_template={},
    )
    assert len(cold.tasks) == 1
    warm = global_wizard._plan_run_completion(
        dataset,
        (template,),
        completed,
        rebin_factor=2,
        metric=SelectionMetric.AICC,
        sibling_values_by_template={},
    )
    assert not warm.tasks
    fresh = observed(dataset.run)
    changed = global_wizard._plan_run_completion(
        fresh,
        (template,),
        completed,
        rebin_factor=2,
        metric=SelectionMetric.AICC,
        sibling_values_by_template={},
    )
    assert len(changed.tasks) == 1
    with pytest.raises(ValueError, match="observation changed"):
        single.assessment_with_curves(completed, template.key, fresh)


def test_reference_clock_and_both_variances_through_wizard_preparation():
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest

    run = _run(
        [200] * 8,
        [150] * 8,
        first_good_bin=2,
        last_good_bin=5,
        background_correction=True,
        background_mode="reference_run",
    )
    run.histograms = [
        replace(h, count_variance=np.full(8, v)) for h, v in zip(run.histograms, [17, 23])
    ]
    reference_run = _run(np.arange(10.0, 20), np.arange(20.0, 30))
    histograms = [
        replace(h, t0_bin=1, count_variance=np.full(10, v))
        for h, v in zip(reference_run.histograms, [5, 7])
    ]
    sample_exposure, reference_exposure = (
        ReferenceExposure(200, "file"),
        ReferenceExposure(100, "file"),
    )
    run.grouping["reference_exposure"] = sample_exposure.to_dict()
    reference = BackgroundReference(
        histograms,
        ReferenceScaleRequest().resolve(sample_exposure, reference_exposure),
        run_number=2,
    )
    dataset = observed(run, reference=reference)
    result = prepare_wizard_dataset(dataset, 2)
    f, b = np.array([346.0, 338.0]), np.array([206.0, 198.0])
    vf, vb = 2 * 17 + 4 * 2 * 5, 2 * 23 + 4 * 2 * 7
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(result.error, 200 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2)
    np.testing.assert_allclose(result.time, [0.25, 0.45])
    assert result.fit_record().histogram_observation is None
    altered = histogram_observation(
        run, reference=replace(reference, histograms=[replace(h, t0_bin=2) for h in histograms])
    )
    assert altered.identity != dataset.histogram_observation.identity


@pytest.mark.parametrize("canonical", [False, True])
def test_selected_vector_balance_and_deadtime_through_wizard(canonical):
    from tests.core.test_selected_pair_balances import pair_run

    run = pair_run(canonical)
    run.grouping.update(forward_group=3, backward_group=4)
    dataset = observed(run)
    result = prepare_wizard_dataset(dataset, 2)
    np.testing.assert_allclose(result.asymmetry, 100 * (120 - 1.1 * 60) / (120 + 1.1 * 60))

    corrected = _run(
        [100, 80],
        [60, 50],
        deadtime_correction=True,
        deadtime_mode="manual",
        dead_time_us=[0.01, 0.02],
        good_frames=1000,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[7, 3],
    )
    prepared = prepare_wizard_dataset(observed(corrected), 2)
    f = sum(c / (1 - c * 0.01 / (1000 * 0.1)) for c in [100, 80]) - 14
    b = sum(c / (1 - c * 0.02 / (1000 * 0.1)) for c in [60, 50]) - 6
    np.testing.assert_allclose(prepared.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(
        prepared.error, 200 * np.sqrt(b * b * 180 + f * f * 110) / (f + b) ** 2
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("field", 1200.0),
        ("temperature", 80.0),
        ("field_direction", "LF"),
        ("field_state", "TF"),
        ("title", "CaF2"),
        ("sample", "PbF2"),
    ],
)
def test_scope_seed_and_series_metadata_edits_change_cache_identity(key, value):
    dataset = observed(_run([90, 1] * 8, [10, 9] * 8))
    before = wizard_observation_identity(dataset)
    dataset.metadata[key] = value
    assert wizard_observation_identity(dataset) != before


def test_display_metadata_does_not_change_numerical_identity():
    dataset = observed(_run([90, 1] * 8, [10, 9] * 8))
    before = wizard_observation_identity(dataset)
    dataset.metadata["plot_colour"] = "blue"
    assert wizard_observation_identity(dataset) == before


@pytest.mark.parametrize("mode", ["single", "red", "green"])
def test_simulation_base_and_merge_replay_its_exact_uncorrected_source(mode):
    from asymmetry.core.simulate import reduce_run_to_dataset

    red = _run(
        [90, 1, 30, 2] * 8,
        [10, 9, 20, 8] * 8,
        bunching_factor=2,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[10, 20],
    )
    green = _run([2, 80, 10, 2] * 8, [8, 10, 30, 8] * 8)
    if mode != "single":
        red.grouping.update(period_mode=mode, period_histograms=[red.histograms, green.histograms])
    dataset = reduce_run_to_dataset(red)
    assert dataset.histogram_observation is not None
    assert not dataset.histogram_observation.source.grouping["background_correction"]
    result = prepare_wizard_dataset(dataset, 2)
    source = green if mode == "green" else red
    f, b = (h.counts.reshape(-1, 4).sum(axis=1) for h in source.histograms)
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(result.error, 200 * np.sqrt(b * b * f + f * f * b) / (f + b) ** 2)


def test_matched_period_simulation_merges_each_period_counts_first():
    from asymmetry.core.simulate import reduce_run_to_dataset

    red, green = _run([90, 1] * 8, [10, 9] * 8), _run([2, 80] * 8, [8, 10] * 8)
    red.grouping.update(
        period_mode="green_minus_red", period_histograms=[red.histograms, green.histograms]
    )
    dataset = reduce_run_to_dataset(red)
    assert dataset.histogram_observation is not None
    prepared = prepare_wizard_dataset(dataset, 2)
    np.testing.assert_allclose(prepared.asymmetry, 100 * 64 / 100 - 100 * 72 / 110)
    assert not np.allclose(prepared.asymmetry, dataset.merge_curve_bins(2).asymmetry)


@pytest.mark.parametrize("mode,sign", [("green_minus_red", -1), ("green_plus_red", 1)])
def test_simulated_period_context_records_implicit_acquired_windows(mode, sign):
    from asymmetry.core.simulate import reduce_run_to_dataset

    red, green = (
        _run(np.arange(20.0, 52), np.arange(60.0, 92)),
        _run(np.arange(90.0, 122), np.arange(50.0, 82)),
    )
    red.histograms = [
        replace(h, bin_width=0.125, t0_bin=2, good_bin_start=4, good_bin_end=19)
        for h in red.histograms
    ]
    green.histograms = [
        replace(h, bin_width=0.125, t0_bin=3, good_bin_start=5, good_bin_end=20)
        for h in green.histograms
    ]
    red.grouping.update(
        t0_bin=2,
        first_good_bin=4,
        last_good_bin=19,
        bunching_factor=2,
        detector_t0_bins=[2, 2],
        detector_first_good_bins=[4, 4],
        detector_last_good_bins=[19, 19],
        period_mode=mode,
        period_histograms=[red.histograms, green.histograms],
    )
    dataset = reduce_run_to_dataset(red)
    assert dataset.histogram_observation.source.grouping["good_window_policy"] == {
        "mode": "from_file"
    }
    prepared = prepare_wizard_dataset(dataset, 2)
    f, b = (h.counts[4:20].reshape(-1, 4).sum(axis=1) for h in red.histograms)
    gf, gb = (h.counts[5:21].reshape(-1, 4).sum(axis=1) for h in green.histograms)
    if mode == "green_plus_red":
        f, b = f + gf, b + gb
        expected, variance = 100 * (f - b) / (f + b), 40000 * f * b / (f + b) ** 3
    else:
        expected = 100 * (gf - gb) / (gf + gb) - 100 * (f - b) / (f + b)
        variance = 40000 * gf * gb / (gf + gb) ** 3 + 40000 * f * b / (f + b) ** 3
    np.testing.assert_allclose(prepared.asymmetry, expected)
    np.testing.assert_allclose(prepared.error, np.sqrt(variance))
    np.testing.assert_allclose(
        prepared.time, (np.arange(4, 20).reshape(-1, 4).mean(axis=1) - 2) * 0.125
    )


def test_rotation_quantity_changes_identity_and_refuses_count_based_cells():
    from asymmetry.core.transform.preparation import wizard_observation_matches

    dataset = observed(_run([90, 1] * 16, [10, 9] * 16))
    templates = single.build_null_baseline_templates()[:1]
    recommendation = single.build_fit_wizard_recommendation_for_templates(
        dataset, templates, variant_budget=1
    )
    identity = wizard_observation_identity(dataset)
    np.testing.assert_allclose(prepare_wizard_dataset(dataset, 2).asymmetry, 100 * 72 / 110)
    dataset.metadata["rotating_frame"] = {}
    assert wizard_observation_identity(dataset) != identity
    assert not wizard_observation_matches(dataset, recommendation.build_signature)
    np.testing.assert_allclose(prepare_wizard_dataset(dataset, 2).asymmetry, 0)
    assert not global_wizard.single_fit_table_covers_portfolio(
        [dataset], templates, {int(dataset.run_number): recommendation}
    )
    with pytest.raises(ValueError, match="observation changed"):
        single.assessment_with_curves(recommendation, templates[0].key, dataset)


def test_nullable_held_periods_preserve_single_run_workflow_observation():
    source = _run([90, 1] * 8, [10, 9] * 8)
    ordinary = reduce_run(source, ReductionSettings())
    nullable = reduce_run(
        replace(source, grouping=source.grouping | {"period_histograms": None}), ReductionSettings()
    )
    for name in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(nullable, name), getattr(ordinary, name))
    np.testing.assert_allclose(prepare_wizard_dataset(nullable, 2).asymmetry, 100 * 72 / 110)
