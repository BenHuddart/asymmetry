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
    assert outcome.trend.columns == ["run", "x", "Lambda", "Lambda_err", "flags"]
    assert [row["run"] for row in outcome.trend.rows] == [2, 3, 1]
    assert [row["x"] for row in outcome.trend.rows] == [0.0, 0.5, 1.0]
    for row in outcome.trend.rows:
        assert row["Lambda"] == pytest.approx(rates[row["run"]], abs=0.02)
    assert outcome.to_dict()["trend"] == outcome.trend.to_dict()
