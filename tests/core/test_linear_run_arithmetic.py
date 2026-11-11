"""Independent arithmetic and likelihood oracles through production entries."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.combine import CombineError, combine_runs, reduce_combined_run
from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.count_domain import build_count_groups
from asymmetry.core.fitting.grouped_time_domain import group_observations
from asymmetry.core.fitting.likelihood import FitInputError, Objective


def source(number, counts=10.0, variance=None, *, periodic=False):
    histograms = [
        Histogram(
            np.full(12, counts * factor),
            0.1,
            count_variance=None if variance is None else np.full(12, variance),
        )
        for factor in (1, 2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if periodic:
        grouping.update(
            period_histograms=[histograms, [replace(h, counts=h.counts * 3) for h in histograms]],
            period_good_frames=[100.0, 200.0],
            period_mode="red",
        )
    return Run(number, histograms, {}, grouping)


@pytest.mark.parametrize(
    "counts,variances,scales,sign,expected,noise",
    [
        ((10, 20), (None, None), (1, 2), 1, 50, 90),
        ((10, 20), (25, 49), (1, 1), 1, 30, 74),
        ((100, 20), (None, None), (2, 0.5), -1, 190, 405),
        ((100, 20), (25, 49), (1, 0.5), -1, 90, 37.25),
        ((10, 20), (25, None), (-1, 0.5), 1, 0, 30),
        ((-10, 20), (25, 49), (1, -0.5), 1, -20, 37.25),
    ],
)
def test_independent_count_and_noise_oracles(counts, variances, scales, sign, expected, noise):
    runs = [source(i, n, v) for i, (n, v) in enumerate(zip(counts, variances, strict=True))]
    before = [h.counts.copy() for run in runs for h in run.histograms]
    result = combine_runs(runs, scales=scales, sign=sign)
    np.testing.assert_allclose(result.histograms[0].counts, expected)
    np.testing.assert_allclose(result.histograms[0].count_variance, noise)
    assert "detector_variance" not in result.metadata["combination"]
    for h, original in zip([h for run in runs for h in run.histograms], before, strict=True):
        np.testing.assert_array_equal(h.counts, original)


@pytest.mark.parametrize("sign", [1, -1])
def test_both_period_counterparts_use_coefficients(sign):
    a, b = source(1, 100, periodic=True), source(2, 20, periodic=True)
    result = combine_runs([a, b], scales=[2, 0.5], sign=sign)
    for period, multiplier in zip(result.grouping["period_histograms"], [1, 3], strict=True):
        np.testing.assert_allclose(period[0].counts, (200 + sign * 10) * multiplier)
        np.testing.assert_allclose(period[0].count_variance, 405 * multiplier)


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficients_fail_at_operation_boundary(scale):
    with pytest.raises(CombineError, match="finite"):
        combine_runs([source(1), source(2)], scales=[1, scale])


def test_all_zero_request_has_no_acquisition():
    with pytest.raises(CombineError, match="nonzero"):
        combine_runs([source(1), source(2)], scales=[0, 0])


def test_zero_term_does_not_change_domain_or_overlap():
    a, b = source(1), source(2, periodic=True)
    b.histograms[0].bin_width = 2
    result = combine_runs([a, b], scales=[1, 0])
    np.testing.assert_array_equal(result.histograms[0].counts, a.histograms[0].counts)
    assert result.metadata["combination"]["scales"] == [1, 0]
    combine_runs([a, a], scales=[1, 0])


@pytest.mark.parametrize("mismatch", ["mixed", "length"])
def test_period_payload_errors_request_selection(mismatch):
    a, b = source(1, periodic=True), source(2, periodic=mismatch != "mixed")
    if mismatch == "length":
        b.grouping["period_histograms"].append(b.histograms)
    with pytest.raises(CombineError, match="period.*select|select.*period"):
        combine_runs([a, b])


def test_known_direct_reuse_rejected_equal_numbers_are_not_identity():
    a = source(1)
    with pytest.raises(CombineError, match="independent"):
        combine_runs([a, a])
    combine_runs([a, source(1)])
    with pytest.raises(CombineError, match="independent"):
        combine_runs([a, replace(a)])


def test_weighted_integer_counts_rejected_by_actual_poisson_boundary():
    run = combine_runs([source(1), source(2, 20)], scales=[1, 2])
    group = build_count_groups(reduce_combined_run(run), [1], lifetime_corrected=False)[0]
    with pytest.raises(FitInputError, match="raw counts"):
        group_observations(group, Objective.POISSON)
    observations = group_observations(group, Objective.GAUSSIAN)
    np.testing.assert_allclose(observations.sigma, np.sqrt(90))


def test_gaussian_arithmetic_uses_variance_below_one_without_floor():
    run = combine_runs([source(1), source(2, 20)], scales=[0.1, 0.1])
    group = build_count_groups(reduce_combined_run(run), [1], lifetime_corrected=False)[0]
    observations = group_observations(group, Objective.GAUSSIAN)
    np.testing.assert_allclose(observations.values, 3)
    np.testing.assert_allclose(observations.sigma, np.sqrt(0.3))


def test_raw_unit_sum_remains_poisson():
    run = combine_runs([source(1), source(2, 20)])
    group = build_count_groups(reduce_combined_run(run), [1], lifetime_corrected=False)[0]
    observations = group_observations(group, Objective.POISSON)
    np.testing.assert_allclose(observations.counts, 30)


def test_inherited_nonpoisson_origin_survives_unit_sum():
    weighted = combine_runs([source(1), source(2)], scales=[1, 2])
    result = combine_runs([weighted, source(3)])
    group = build_count_groups(reduce_combined_run(result), [1], lifetime_corrected=False)[0]
    with pytest.raises(FitInputError):
        group_observations(group, Objective.POISSON)


def test_supplied_noise_cannot_be_promoted_to_poisson_by_equal_mean():
    result = combine_runs([source(1, 10, 10), source(2, 20, 20)])
    group = build_count_groups(reduce_combined_run(result), [1], lifetime_corrected=False)[0]
    with pytest.raises(FitInputError):
        group_observations(group, Objective.POISSON)


def test_zero_retained_gaussian_error_has_typed_diagnostic():
    result = combine_runs([source(1, 10, 0), source(2, 20, 0)])
    group = build_count_groups(reduce_combined_run(result), [1], lifetime_corrected=False)[0]
    with pytest.raises(FitInputError, match="finite and positive"):
        group_observations(group, Objective.GAUSSIAN)


def test_known_file_reuse_and_distinct_acquired_periods():
    a, b = source(1), source(1)
    a.source_file = b.source_file = "same-acquisition.nxs"
    with pytest.raises(CombineError, match="independent"):
        combine_runs([a, b])
    a.metadata["data_period_number"] = 1
    b.metadata["data_period_number"] = 2
    combine_runs([a, b])


def test_explicit_selected_periods_and_mapped_sources_keep_noise():
    from asymmetry.core.io.periods import period_run

    a, b = source(1, 100, periodic=True), source(2, 20, periodic=True)
    selected = [period_run(run, 1) for run in (a, b)]
    result = combine_runs(selected, scales=[1, -0.5])
    np.testing.assert_allclose(result.histograms[0].counts, 270)
    np.testing.assert_allclose(result.histograms[0].count_variance, 315)


def test_normal_reduction_and_integral_have_independent_ratio_uncertainties():
    from asymmetry.core.transform.integral import integrate_run

    result = combine_runs([source(1, 10, 25), source(2, 20, 49)])
    dataset = reduce_combined_run(result)
    expected_error = 200 * np.sqrt(60**2 * 74 + 30**2 * 74) / 90**2
    np.testing.assert_allclose(dataset.asymmetry, -100 / 3)
    np.testing.assert_allclose(dataset.error, expected_error)
    value, error = integrate_run(result)
    assert value == pytest.approx(-1 / 3)
    assert error == pytest.approx(expected_error / 100 / np.sqrt(dataset.n_points))


def test_count_first_replay_keeps_noise_and_capture_is_cancellable():
    from asymmetry.core.transform.preparation import histogram_observation, prepare_wizard_dataset
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    result = combine_runs([source(1, 10, 25), source(2, 20, 49)])
    dataset = reduce_run(result, ReductionSettings())
    expected = dataset.error[0] / np.sqrt(4)
    cropped = dataset.time_range(dataset.time[2], dataset.time[9])
    merged = prepare_wizard_dataset(prepare_wizard_dataset(cropped, 2), 2)
    np.testing.assert_allclose(merged.error, expected)
    result.histograms[0].counts[:] = 1
    np.testing.assert_allclose(prepare_wizard_dataset(cropped, 4).error, expected)
    record = merged.fit_record()
    assert record.histogram_observation is None
    assert record.run.histograms == []
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("cancelled capture")

    with pytest.raises(RuntimeError, match="cancelled capture"):
        histogram_observation(result, cancel_check=cancel)
    assert len(calls) == 2


def test_arithmetic_policy_scopes_fourier_and_wizard_identity(monkeypatch):
    import asymmetry.core.io.period_counts as policy
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest
    from asymmetry.core.transform.preparation import wizard_observation_identity

    result = combine_runs([source(1), source(2)])
    dataset = reduce_combined_run(result)
    raw = reduce_combined_run(source(3))
    before = (fourier_grouping_digest(result), wizard_observation_identity(dataset))
    raw_before = (fourier_grouping_digest(raw.run), wizard_observation_identity(raw))
    monkeypatch.setattr(policy, "RUN_ARITHMETIC_POLICY", "independent-test-revision")
    after = (fourier_grouping_digest(result), wizard_observation_identity(dataset))
    assert all(a != b for a, b in zip(before, after, strict=True))
    assert raw_before == (fourier_grouping_digest(raw.run), wizard_observation_identity(raw))


def test_unknown_signed_noise_cannot_be_hidden_by_positive_total():
    result = combine_runs([source(1, -10), source(2, 100)])
    assert np.isnan(result.histograms[0].count_variance).all()
    with pytest.raises(ValueError, match="supplied variance"):
        reduce_combined_run(result)


def test_period_reference_retains_sample_normalisers():
    result = combine_runs([source(1, periodic=True), source(2, periodic=True)], sign=-1)
    assert result.grouping["period_good_frames"] == [100, 200]


def test_shared_green_buffers_are_direct_overlap():
    a, b = source(1, periodic=True), source(2, periodic=True)
    b.grouping["period_histograms"][1] = a.grouping["period_histograms"][1]
    with pytest.raises(CombineError, match="independent"):
        combine_runs([a, b])


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
def test_unknown_noise_is_scoped_to_consumed_bins_and_survives_nested_pooling(periodic, supplied):
    a, b = source(1, periodic=periodic), source(2, 20, periodic=periodic)
    for run in (a, b):
        run.grouping.update(first_good_bin=1, last_good_bin=11)
    periods = a.grouping["period_histograms"] if periodic else [a.histograms]
    for histograms in periods:
        if supplied:
            variance = np.full(12, 25.0)
            variance[0] = np.nan
            histograms[0] = replace(histograms[0], count_variance=variance)
        else:
            histograms[0].counts[0] = -5
    if supplied and periodic:
        a.histograms = periods[0]
    result = combine_runs([a, b], scales=[1, 0.5])
    assert np.isnan(result.histograms[0].count_variance[0])
    reduced = reduce_combined_run(result)
    np.testing.assert_allclose(reduced.asymmetry, -100 / 3)
    if periodic:
        assert all(
            np.isnan(period[0].count_variance[0]) for period in result.grouping["period_histograms"]
        )
    # Inherited unknown noise cannot be lost in a later unit sum.
    c = source(3, periodic=periodic)
    c.grouping.update(first_good_bin=1, last_good_bin=11)
    nested = combine_runs([result, c])
    assert np.isnan(nested.histograms[0].count_variance[0])
    np.testing.assert_allclose(reduce_combined_run(nested).asymmetry, -100 / 3)
    nested.grouping["first_good_bin"] = 0
    with pytest.raises(ValueError, match="unknown independent count noise"):
        reduce_combined_run(nested)


def test_unused_detector_does_not_enter_unknown_noise_domain():
    a, b = source(1), source(2, 20)
    a.histograms.append(Histogram(np.full(12, -10.0), 0.1))
    b.histograms.append(Histogram(np.full(12, 30.0), 0.1))
    result = combine_runs([a, b])
    assert np.isnan(result.histograms[2].count_variance).all()
    np.testing.assert_allclose(reduce_combined_run(result).asymmetry, -100 / 3)
    result.grouping["groups"][1] = [1, 3]
    with pytest.raises(ValueError, match="supplied variance"):
        reduce_combined_run(result)


def test_fractional_unknown_sources_cannot_gain_poisson_origin():
    result = combine_runs([source(1, 0.5), source(2, 0.5)])
    np.testing.assert_allclose(result.histograms[0].counts, 1)
    assert np.isnan(result.histograms[0].count_variance).all()
    with pytest.raises(ValueError, match="supplied variance"):
        reduce_combined_run(result)


@pytest.mark.parametrize("selection", ["time", "exclude"])
def test_actual_count_inputs_validate_only_selected_noise(selection):
    from asymmetry.core.data.dataset import MuonDataset

    a, b = source(1), source(2, 20)
    a.histograms[0].counts[0] = -5
    result = combine_runs([a, b], scales=[0.1, 0.1])
    data = MuonDataset(np.arange(12) * 0.1, np.zeros(12), np.ones(12), {}, result)
    kwargs = {"t_min": 0.1} if selection == "time" else {"exclude": (-0.01, 0.01)}
    group = build_count_groups(data, [1], lifetime_corrected=False, **kwargs)[0]
    observations = group_observations(group, Objective.GAUSSIAN)
    np.testing.assert_allclose(observations.sigma, np.sqrt(0.3))
    assert len(observations.time) == 11
    with pytest.raises(ValueError, match="supplied variance"):
        build_count_groups(data, [1], lifetime_corrected=False)


def test_unknown_noise_capture_owns_context_and_identity_sees_known_noise_edits():
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.transform.deadtime import DeadtimeDomain
    from asymmetry.core.transform.fb_counts import BinWindow
    from asymmetry.core.transform.preparation import (
        histogram_observation,
        prepare_histogram_asymmetry,
        prepare_wizard_dataset,
    )
    from asymmetry.core.transform.scoped_counts import scoped_source_digest

    a, b = source(1), source(2, 20)
    a.histograms[0].counts[0] = -5
    result = combine_runs([a, b])
    result.grouping.update(first_good_bin=1, last_good_bin=11)
    snapshot = histogram_observation(result)
    assert np.isnan(snapshot.source.histograms[0].count_variance[0])
    prepared = prepare_histogram_asymmetry(result, merge=1, facility=None, reference=None)
    data = MuonDataset(
        prepared.time, prepared.asymmetry, prepared.error, result.metadata, result, snapshot
    )
    initial = prepare_wizard_dataset(data, 2)
    domain = DeadtimeDomain(((0, (BinWindow(0, 11),)),))
    before = scoped_source_digest(result.histograms, domain)
    result.histograms[0].count_variance.setflags(write=True)
    result.histograms[0].count_variance[0] = 25
    assert scoped_source_digest(result.histograms, domain) != before
    assert histogram_observation(result).identity != snapshot.identity
    assert np.isnan(snapshot.source.histograms[0].count_variance[0])
    replayed = prepare_wizard_dataset(data, 2)
    np.testing.assert_array_equal(replayed.error, initial.error)
    assert not snapshot.source.histograms[0].count_variance.flags.writeable


def test_file_identity_survives_renumbered_rows():
    a, b = source(1), source(2)
    a.source_file = b.source_file = "one-acquisition.nxs"
    with pytest.raises(CombineError, match="independent"):
        combine_runs([a, b])


def test_actual_gaussian_fit_uses_subunit_variance_and_selected_observations():
    from asymmetry.core.fitting.engine import GAUSSIAN_COST, FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    result = combine_runs([source(1), source(2, 20)], scales=[0.1, 0.1])
    group = build_count_groups(
        reduce_combined_run(result), [1], t_min=0.2, lifetime_corrected=False
    )[0]
    parameters = ParameterSet()
    parameters.add(Parameter("level", 2.5))

    def constant(time, level):
        return np.full_like(time, level)

    fit = FitEngine().fit_arrays(
        group.time, group.counts, group.error, constant, parameters, cost_factory=GAUSSIAN_COST
    )
    assert fit.success
    assert fit.parameters["level"].value == pytest.approx(3, abs=1e-5)
    assert fit.uncertainties["level"] == pytest.approx(np.sqrt(0.3 / 10), rel=1e-4)
    assert fit.dof == 9
    assert len(fit.residuals) == 10
    np.testing.assert_allclose(fit.residual_sigma, np.sqrt(0.3))


def test_arithmetic_policy_changes_only_multisource_workdir_identity(tmp_path, monkeypatch):
    import asymmetry.core.workflow.workdir as workdir
    from asymmetry.core.workflow.reduction import ReductionSettings

    paths = [tmp_path / "a.nxs", tmp_path / "b.nxs"]
    for path in paths:
        path.write_bytes(b"synthetic identity fixture")
    kwargs = {"grouping": source(1).grouping, "settings": ReductionSettings()}
    multiple = workdir.reduction_digest(source_files=paths, **kwargs)
    single = workdir.reduction_digest(source_files=paths[:1], **kwargs)
    monkeypatch.setattr(workdir, "RUN_ARITHMETIC_POLICY", "test-revision")
    assert workdir.reduction_digest(source_files=paths, **kwargs) != multiple
    assert workdir.reduction_digest(source_files=paths[:1], **kwargs) == single


def test_historical_wizard_score_identity_cannot_match_fresh_arithmetic(monkeypatch):
    import json

    import asymmetry.core.io.period_counts as policy
    from asymmetry.core.transform.preparation import (
        WIZARD_PREPARATION,
        wizard_observation_identity,
        wizard_observation_matches,
        wizard_rebuild_dataset,
    )

    dataset = reduce_combined_run(combine_runs([source(1), source(2)]))
    with monkeypatch.context() as old:
        old.setattr(policy, "RUN_ARITHMETIC_POLICY", "historical-arithmetic")
        signature = json.dumps(
            {
                "wizard_preparation": WIZARD_PREPARATION,
                "observation": wizard_observation_identity(dataset),
            }
        )
    before = (dataset.time.copy(), dataset.asymmetry.copy(), dataset.error.copy())
    assert not wizard_observation_matches(dataset, signature)
    with pytest.raises(ValueError, match="observation changed"):
        wizard_rebuild_dataset(dataset, 2, signature)
    for actual, historical in zip(
        (dataset.time, dataset.asymmetry, dataset.error), before, strict=True
    ):
        np.testing.assert_array_equal(actual, historical)
