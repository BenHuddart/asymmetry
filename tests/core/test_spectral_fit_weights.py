"""Verify the fitted objective and statistics for spectral weight policies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.fit_log import FitLog
from asymmetry.core.fitting.result_payload import _deserialize_fit_result, _serialize_fit_result
from asymmetry.core.fitting.result_summary import fit_result_summary, reduced_statistic_summary
from asymmetry.core.fitting.spectral import default_frequency_model
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import FitSeries


@pytest.mark.parametrize("partial", [False, True])
def test_spectral_fit_uses_only_stated_positive_errors(partial):
    x = np.arange(8, dtype=float)
    y = np.array([0.1, 1.5, 2.1, 0.7, 1.1, 1.9, 3.1, 4.2])
    error = np.array([0.3, 0.1, 0.6, 1.2, 0.2, 0.4, 0.25, 0.1])
    if partial:
        error[[2, 3, 5]] = [0, np.nan, -1]
    dataset = MuonDataset(x, y, error, metadata={"plot_domain": "frequency"})
    params = ParameterSet([Parameter("c", value=1.5)])

    def constant(f, c):
        return np.full_like(f, c)

    for low, high in ((0.0, 7.0), (1.0, 6.0)):
        selected = (x >= low) & (x <= high) & (error > 0)
        weights = 1 / np.square(error[selected])
        expected = np.average(y[selected], weights=weights)
        fit = FitEngine().fit(dataset, constant, params, t_min=low, t_max=high)
        summary = fit_result_summary(fit)
        attribution = summary["input_attribution"]
        weighting = attribution["spectral_weighting"]

        assert fit.success
        assert fit.parameters["c"].value == pytest.approx(expected, abs=1e-6)
        assert fit.uncertainties["c"] == pytest.approx(weights.sum() ** -0.5, rel=1e-5)
        assert fit.chi_squared == pytest.approx(np.dot(weights, np.square(y[selected] - expected)))
        np.testing.assert_array_equal(fit.residual_time, x[selected])
        np.testing.assert_array_equal(fit.residual_sigma, error[selected])
        assert weighting["policy"] == ("positive_only" if partial else "diagonal")
        assert weighting["omitted_bin_ranges"] == (
            [[int(2 - low), int(4 - low)], [int(5 - low), int(6 - low)]] if partial else []
        )
        assert attribution["n"] == selected.sum()
        assert (attribution["min"], attribution["max"]) == (low, high)
        assert summary["quality"] is None
        assert "Correlated spectral bins" in summary["quality_reason"]
        assert "Approximate spectral fit" in FitLog().format_record(summary)

        reloaded = _deserialize_fit_result(_serialize_fit_result(fit, include_residuals=False))
        assert reloaded.input_attribution == attribution
        assert reloaded.uncertainties == fit.uncertainties


def test_partial_spectrum_with_too_few_positive_errors_names_remedies():
    x = np.arange(8, dtype=float)
    ds = MuonDataset(x, x, np.r_[0.2, np.zeros(7)], metadata={"plot_domain": "frequency"})
    model = default_frequency_model()
    result = FitEngine().fit(
        ds,
        model.function,
        ParameterSet([Parameter(name, value) for name, value in model.param_defaults.items()]),
    )

    assert not result.success
    assert (
        "Only 1 of the selected spectral bins have positive errors for 4 free parameters."
        in result.message
    )
    assert "Widen the fit range or fix parameters." in result.message


def test_coupled_fit_suppresses_errors_only_in_connected_members_and_round_trips(tmp_path):
    x = np.linspace(0, 1, 8)
    declared = np.linspace(0.35, 1, x.size)
    measured = 1.0 + 1.1 * x + 0.05 * np.sin(20 * x)
    unit = 2.0 + 0.8 * x + 0.1 * np.cos(10 * x)
    independent = -0.5 + 1.7 * x + 0.03 * np.cos(15 * x)
    spectra = [
        MuonDataset(x, values, errors, metadata={"plot_domain": "frequency", "run_number": run})
        for run, values, errors in [
            (11, measured, declared),
            (12, unit, np.zeros_like(x)),
            (13, independent, declared),
        ]
    ]

    def line(f, c, slope):
        return c + slope * f

    starts = {
        ds.run_number: ParameterSet(
            [
                Parameter("c", -0.5 if ds.run_number == 13 else 1.0, fixed=ds.run_number == 13),
                Parameter("slope", 1.0),
            ]
        )
        for ds in spectra
    }
    results, _ = FitEngine().global_fit(spectra, line, ["c"], ["slope"], starts)
    design = np.vstack(
        [
            np.column_stack((1 / declared, x / declared, np.zeros_like(x))),
            np.column_stack((np.ones_like(x), np.zeros_like(x), x)),
        ]
    )
    target = np.r_[measured / declared, unit]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    assert all(result.success for result in results.values())
    assert results[11].parameters["c"].value == pytest.approx(expected[0], abs=1e-5)
    assert results[11].parameters["slope"].value == pytest.approx(expected[1], abs=1e-5)
    assert results[12].parameters["slope"].value == pytest.approx(expected[2], abs=1e-5)
    for run in (11, 12):
        assert not results[run].uncertainties
        assert results[run].covariance is None
        assert results[run].minos_errors is None
    assert results[11].input_attribution["coupled_unit_weights"] is True
    assert math.isnan(results[12].reduced_chi_squared)
    assert results[12].chi_squared == pytest.approx(
        np.sum((unit - line(x, expected[0], expected[2])) ** 2)
    )
    assert results[13].uncertainties["slope"] == pytest.approx(
        1 / np.linalg.norm(x / declared), rel=1e-5
    )

    model = ProjectModel()
    model.add_batch(
        FitSeries(
            "joint-weights",
            RepresentationType.FREQ_FFT,
            canonical_model=default_frequency_model().to_dict(),
            member_run_numbers=list(results),
            results_by_run={run: fit_result_summary(result) for run, result in results.items()},
        )
    )
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [{"run_number": run} for run in results],
    }
    model.write_to_project_state(state)
    path = tmp_path / "spectra.asymp"
    save_project(state, path)
    reopened = ProjectModel.from_project_state(load_project(path))
    records = reopened.batch("joint-weights").results_by_run
    assert records[11]["input_attribution"]["coupled_unit_weights"]
    assert "linked member" in records[11]["quality_reason"]
    assert records[12]["quality"] is None and records[12]["reduced_chi_squared"] is None
    assert records[13]["uncertainties"]["slope"] == results[13].uncertainties["slope"]
    assert "Mixed spectral weights" in reduced_statistic_summary(list(records.values()))
    report = FitLog().format_report(records.values())
    assert "Residual sum of squares =" in report
    assert "Unit weights in a linked member" in report
    assert "Diagonal spectral errors on all 8 fitted bins" in report
    assert "target" not in report


@pytest.mark.parametrize(("bridge", "supported", "n_free"), [(False, 2, 3), (True, 3, 4)])
def test_coupled_spectral_support_counts_rows_only_against_columns_they_can_constrain(
    bridge, supported, n_free
):
    from asymmetry.core.fitting.likelihood import FitInputError

    x = np.linspace(0, 1, 8)

    def polynomial(f, c, slope, quadratic, cubic):
        return c + slope * f + quadratic * f**2 + cubic * f**3

    starts = {}
    spectra = []
    for run in (1, 2, 3):
        errors = np.ones_like(x)
        if run in (1, 2):
            errors[:] = 0
            errors[(run - 1) * (x.size - 1)] = 0.2
        spectra.append(
            MuonDataset(
                x,
                polynomial(x, 1, 2, 3, 4),
                errors,
                metadata={"plot_domain": "frequency", "run_number": run},
            )
        )
        starts[run] = ParameterSet(
            [
                Parameter("c", 1, fixed=run == 3),
                Parameter("slope", 2, fixed=run == 3),
                Parameter("quadratic", 3, fixed=run == 3),
                Parameter("cubic", 3, fixed=run != 3 and not bridge),
            ]
        )
    shared = ["c", "slope", "quadratic"]
    if bridge:
        shared.append("cubic")

    with pytest.raises(FitInputError, match=f"support {supported} of {n_free} free parameters"):
        FitEngine().global_fit(spectra, polynomial, shared, [] if bridge else ["cubic"], starts)


@pytest.mark.parametrize("unit", [False, True])
def test_joint_spectral_fit_qualifies_its_aggregate_objective(unit):
    from asymmetry.core.fitting.joint import JointSeriesProblem, SharedParameter, fit_joint

    x = np.arange(6, dtype=float)
    rows = (1.0 + 0.05 * np.sin(x), 2.0 + 0.05 * np.cos(x), 3.0 - 0.05 * x, 3.0 + 0.02 * x)

    def constant(f, c):
        return np.full_like(f, c)

    errors = [0.5, 0.0 if unit else 0.8, 0.2, 0.3]
    problems = []
    for run, (y, error) in enumerate(zip(rows, errors, strict=True), start=1):
        dataset = MuonDataset(
            x, y, np.full_like(x, error), metadata={"plot_domain": "frequency", "run_number": run}
        )
        problems.append(
            JointSeriesProblem(
                key=run,
                datasets=[dataset],
                model_fn=constant,
                global_params=["c"],
                local_params=[],
                initial_params={run: ParameterSet([Parameter("c", 1.0)])},
            )
        )

    result = fit_joint(
        problems,
        [
            SharedParameter("common", {1: "c", 2: "c"}, 1.0),
            SharedParameter("independent", {3: "c", 4: "c"}, 3.0),
        ],
    )
    weights = np.repeat([1.0 / sigma**2 if sigma else 1.0 for sigma in errors], x.size).reshape(
        2, -1
    )
    observations = np.concatenate(rows).reshape(2, -1)
    expected = np.average(observations, weights=weights, axis=1)
    objective = np.sum(weights * np.square(observations - expected[:, None]))
    variances = 1 / weights.sum(axis=1)
    assert result.success
    assert result.shared_parameters["common"].value == pytest.approx(expected[0], abs=1e-8)
    assert result.shared_parameters["independent"].value == pytest.approx(expected[1], abs=1e-8)
    assert result.objective_value == pytest.approx(objective)
    assert result.series_reduced_chi_squared[1] == pytest.approx(
        np.sum(((rows[0] - expected[0]) / errors[0]) ** 2) / (x.size - 1)
    )
    assert result.shared_uncertainties["independent"] == pytest.approx(variances[1] ** 0.5)
    assert result.shared_covariance[1, 1] == pytest.approx(variances[1])
    if unit:
        assert "common" not in result.shared_uncertainties
        assert np.isnan(result.shared_covariance[0]).all()
        assert np.isnan(result.shared_covariance[:, 0]).all()
        assert math.isnan(result.chi_squared)
        assert math.isnan(result.reduced_chi_squared)
        assert math.isnan(result.series_reduced_chi_squared[2])
        assert "unit-weight spectral residuals" in result.quality_reason
    else:
        assert result.shared_uncertainties["common"] == pytest.approx(variances[0] ** 0.5)
        assert result.chi_squared == pytest.approx(objective)
        assert result.reduced_chi_squared == pytest.approx(objective / (4 * x.size - 2))
        assert "Correlated spectral bins" in result.quality_reason
