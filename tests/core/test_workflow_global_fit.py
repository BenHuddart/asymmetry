"""Tests for true simultaneous fitting through the workflow façade."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.workflow.global_fit import fit_global
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.series import scan_axis, supplied_axis


def test_fit_global_recovers_parameters_shared_across_runs() -> None:
    time = np.linspace(0.0, 5.0, 180)
    error = np.full(time.size, 0.05)
    datasets = {
        1: MuonDataset(
            time,
            16.0 * np.exp(-0.4 * time) + 1.5,
            error,
            {"run_number": 1, "field": 0.0},
        ),
        2: MuonDataset(
            time,
            20.0 * np.exp(-0.4 * time) + 1.5,
            error,
            {"run_number": 2, "field": 10.0},
        ),
    }
    recipe = FitRecipe.from_expression("Exponential + Constant", dataset=datasets[1])

    outcome = fit_global(
        datasets,
        recipe,
        shared_params=["Lambda", "A_bg"],
        axis=scan_axis(datasets, "run"),
        strategy="least_squares",
    )

    assert outcome.shared["Lambda"] == pytest.approx(0.4, abs=0.01)
    assert outcome.shared["A_bg"] == pytest.approx(1.5, abs=0.03)
    assert all(result["success"] for result in outcome.results)


def test_fit_global_sets_a_field_parameter_per_run_and_holds_it() -> None:
    time = np.linspace(0.0, 4.0, 120)
    error = np.full(time.size, 0.05)
    recipe = FitRecipe.from_expression("LongitudinalFieldKT")
    model = recipe.model()
    datasets = {
        run: MuonDataset(
            time,
            model.function(time, A_1=18.0, Delta=0.3, B_L=field),
            error,
            {"run_number": run, "field": field},
        )
        for run, field in ((10, 0.0), (11, 8.0))
    }

    outcome = fit_global(
        datasets,
        recipe,
        shared_params=["A_1", "Delta"],
        axis=scan_axis(datasets, "field"),
        field_params=["B_L"],
        strategy="least_squares",
    )

    by_run = {result["run"]: result for result in outcome.results}
    assert by_run[10]["parameters"]["B_L"] == pytest.approx(0.0)
    assert by_run[11]["parameters"]["B_L"] == pytest.approx(8.0)
    assert "B_L" not in by_run[11]["uncertainties"]


def test_fit_global_tabulates_its_run_local_parameters_along_the_axis() -> None:
    time = np.linspace(0.0, 5.0, 180)
    error = np.full(time.size, 0.05)
    rates = {1: 0.9, 2: 0.3, 3: 0.6}
    datasets = {
        run: MuonDataset(
            time, 18.0 * np.exp(-rate * time) + 1.5, error, {"run_number": run, "field": 0.0}
        )
        for run, rate in rates.items()
    }
    recipe = FitRecipe.from_expression("Exponential + Constant", dataset=datasets[1])
    concentration = {1: 1.0, 2: 0.0, 3: 0.5}

    outcome = fit_global(
        datasets,
        recipe,
        shared_params=["A_1", "A_bg"],
        axis=supplied_axis("concentration", concentration, datasets),
        strategy="least_squares",
    )

    assert outcome.free_params == ["Lambda"]
    assert outcome.trend.order_key == "concentration"
    assert outcome.trend.columns == ["key", "x", "Lambda", "Lambda_err", "flags"]
    assert [row["key"] for row in outcome.trend.rows] == ["2", "3", "1"]
    assert [row["x"] for row in outcome.trend.rows] == [0.0, 0.5, 1.0]
    for row in outcome.trend.rows:
        assert row["Lambda"] == pytest.approx(rates[int(row["key"])], abs=0.02)
    assert outcome.to_dict()["trend"] == outcome.trend.to_dict()


#: k_B in meV/K, as the Arrhenius trend law writes it.
_KB_MEV = 8.617333262e-2


def _relaxing_run(run: int, *, amplitude: float, rate: float, temperature: float, seed: int):
    time = np.linspace(0.0, 5.0, 200)
    error = np.full(time.size, 0.05)
    noise = np.random.default_rng(seed).normal(0.0, 0.05, time.size)
    return MuonDataset(
        time,
        amplitude * np.exp(-rate * time) + 1.5 + noise,
        error,
        {"run_number": run, "field": 0.0, "temperature": temperature},
    )


def test_a_batch_trends_each_groups_shared_parameter_against_the_group_temperature() -> None:
    from asymmetry.core.workflow.global_fit import fit_global_batch
    from asymmetry.core.workflow.series import group_axis

    def rate(temperature: float) -> float:
        return 0.1 + 0.004 * temperature

    groups = {
        f"batch-{index}": {
            run: _relaxing_run(
                run,
                amplitude=amplitude,
                rate=rate(temperature),
                temperature=temperature,
                seed=run,
            )
            for run, amplitude in (
                (10 * index, 14.0),
                (10 * index + 1, 18.0),
                (10 * index + 2, 22.0),
            )
        }
        # Listed out of temperature order: the batch orders its rows itself.
        for index, temperature in ((1, 150.0), (2, 50.0), (3, 100.0))
    }
    every_run = {run: dataset for runs in groups.values() for run, dataset in runs.items()}
    recipe = FitRecipe.from_expression("Exponential + Constant", dataset=every_run[10])

    batch = fit_global_batch(
        groups,
        recipe,
        shared_params=["Lambda", "A_bg"],
        axis=scan_axis(every_run, "run"),
        batch_axis=group_axis(groups, "temperature"),
        strategy="least_squares",
    )

    assert list(batch.groups) == ["batch-2", "batch-3", "batch-1"]
    assert batch.free_params == ["Lambda", "A_bg"]
    assert batch.trend.order_key == "temperature"
    assert batch.trend.columns == ["key", "x", "Lambda", "Lambda_err", "A_bg", "A_bg_err", "flags"]
    assert [row["key"] for row in batch.trend.rows] == ["batch-2", "batch-3", "batch-1"]
    assert [row["x"] for row in batch.trend.rows] == [50.0, 100.0, 150.0]
    for row in batch.trend.rows:
        assert row["Lambda"] == pytest.approx(rate(row["x"]), abs=3.0 * row["Lambda_err"])
        assert row["A_bg"] == pytest.approx(1.5, abs=0.05)
    # Each group is a simultaneous fit of its own, with the run-local amplitude.
    assert batch.groups["batch-1"].free_params == ["A_1"]


def test_a_batch_refuses_a_group_of_one_run_naming_it() -> None:
    from asymmetry.core.workflow.global_fit import fit_global_batch
    from asymmetry.core.workflow.series import group_axis

    groups = {
        "b-1": {1: _relaxing_run(1, amplitude=18.0, rate=0.3, temperature=5.0, seed=1)},
        "b-2": {
            run: _relaxing_run(run, amplitude=18.0, rate=0.3, temperature=10.0, seed=run)
            for run in (2, 3)
        },
    }
    every_run = {run: dataset for runs in groups.values() for run, dataset in runs.items()}
    with pytest.raises(ValueError, match="too few in b-1"):
        fit_global_batch(
            groups,
            FitRecipe.from_expression("Exponential + Constant", dataset=every_run[2]),
            shared_params=["A_bg"],
            axis=scan_axis(every_run, "run"),
            batch_axis=group_axis(groups, "temperature"),
        )


def test_a_rate_constant_per_temperature_carries_an_arrhenius_law() -> None:
    """The two-level chain: λ against concentration per temperature, then k(T)."""
    from asymmetry.core.workflow.global_fit import fit_global_batch
    from asymmetry.core.workflow.series import group_axis
    from asymmetry.core.workflow.trend_fit import fit_trend, fit_trend_table

    energy_mev, prefactor, base_rate = 100.0, 47.9, 0.2
    concentrations = (0.0, 0.5, 1.0, 1.5)

    def rate_constant(temperature: float) -> float:
        return prefactor * np.exp(-energy_mev / (_KB_MEV * temperature))

    groups: dict[str, dict[int, MuonDataset]] = {}
    concentration_of: dict[int, float] = {}
    for index, temperature in enumerate((250.0, 275.0, 300.0, 325.0), start=1):
        groups[f"mu-{index}"] = {}
        for offset, concentration in enumerate(concentrations):
            run = 100 * index + offset
            concentration_of[run] = concentration
            groups[f"mu-{index}"][run] = _relaxing_run(
                run,
                amplitude=18.0,
                rate=base_rate + rate_constant(temperature) * concentration,
                temperature=temperature,
                seed=run,
            )
    every_run = {run: dataset for runs in groups.values() for run, dataset in runs.items()}
    batch_axis = group_axis(groups, "temperature")

    batch = fit_global_batch(
        groups,
        FitRecipe.from_expression("Exponential + Constant", dataset=every_run[100]),
        shared_params=["A_1", "A_bg"],
        axis=supplied_axis("concentration", concentration_of, every_run),
        batch_axis=batch_axis,
        strategy="least_squares",
    )
    slopes = {
        name: fit_trend(outcome.trend, "Lambda", "Linear").to_dict()
        for name, outcome in batch.groups.items()
    }
    for name, fit in slopes.items():
        expected = rate_constant(batch_axis.values[name])
        assert fit["parameters"]["m"] == pytest.approx(expected, rel=0.1)

    law = fit_trend(fit_trend_table(slopes, "m", batch_axis), "m", "Arrhenius")

    assert law.success
    assert law.keys == ["mu-1", "mu-2", "mu-3", "mu-4"]
    assert law.parameters["Ea"] == pytest.approx(energy_mev, abs=3.0 * law.uncertainties["Ea"])
    assert law.parameters["Ea"] == pytest.approx(energy_mev, rel=0.15)
