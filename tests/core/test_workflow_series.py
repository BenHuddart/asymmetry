"""Tests for :mod:`asymmetry.core.workflow.series`."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.series import ORDER_KEYS, fit_one, fit_series, order_values
from asymmetry.core.workflow.workdir import WorkDir
from tests.core.conftest import (
    FLAT_RUN,
    SCAN_RUNS,
    SCAN_TEMPERATURES,
    ZF_RUNS,
    scan_rate,
)

_EXPRESSION = "Exponential + Constant"


@pytest.fixture(scope="module")
def recipe(reduced_workdir: WorkDir) -> FitRecipe:
    """A relaxation recipe seeded from the coldest run of the scan."""
    return FitRecipe.from_expression(_EXPRESSION, dataset=reduced_workdir.reduced(SCAN_RUNS[0]))


@pytest.fixture(scope="module")
def outcome(reduced_workdir: WorkDir, recipe: FitRecipe):
    """The whole zero-field scan fitted once, broken run included."""
    datasets = {run: reduced_workdir.reduced(run) for run in ZF_RUNS}
    return fit_series(datasets, recipe, order_key="temperature", name="zf")


# -- order keys -------------------------------------------------------------


def test_order_values_reads_the_scan_coordinate_of_every_run(
    reduced_workdir: WorkDir,
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS}
    assert order_values(datasets, "temperature") == {
        run: temperature for run, temperature in zip(SCAN_RUNS, SCAN_TEMPERATURES)
    }
    assert order_values(datasets, "run") == {run: float(run) for run in SCAN_RUNS}
    assert "run" in ORDER_KEYS


def test_order_values_names_the_runs_that_do_not_record_the_quantity() -> None:
    bare = MuonDataset(
        time=np.linspace(0.0, 8.0, 10),
        asymmetry=np.zeros(10),
        error=np.ones(10),
        metadata={"run_number": 7},
    )
    with pytest.raises(ValueError, match="Run\\(s\\) 7 record no temperature"):
        order_values({7: bare}, "temperature")
    with pytest.raises(ValueError, match="Unknown order key"):
        order_values({7: bare}, "pressure")


# -- one run ----------------------------------------------------------------


def test_fit_one_recovers_the_simulated_parameters(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    run = SCAN_RUNS[0]
    result = fit_one(reduced_workdir.reduced(run), recipe)

    assert result["run"] == run
    assert result["success"] is True
    assert result["free_params"] == ["A_1", "Lambda", "A_bg"]

    expected = scan_rate(SCAN_TEMPERATURES[0])
    rate = result["parameters"]["Lambda"]
    sigma = result["uncertainties"]["Lambda"]
    assert sigma > 0.0
    assert abs(rate - expected) <= 3.0 * sigma
    assert result["quality"]["verdict"] == "good"


# -- the scan ---------------------------------------------------------------


def test_fit_series_recovers_the_simulated_rate_curve(outcome) -> None:
    by_run = {entry["run"]: entry for entry in outcome.results}
    for run, temperature in zip(SCAN_RUNS, SCAN_TEMPERATURES):
        entry = by_run[run]
        rate = entry["parameters"]["Lambda"]
        sigma = entry["uncertainties"]["Lambda"]
        assert sigma > 0.0
        assert abs(rate - scan_rate(temperature)) <= 3.0 * sigma, run


def test_fit_series_chains_along_the_scan_and_reports_its_seeding(outcome) -> None:
    assert outcome.seeding_used == "chain"
    assert outcome.seeding_reason
    assert [entry["run"] for entry in outcome.results] == list(ZF_RUNS)
    assert [entry["x"] for entry in outcome.results] == pytest.approx([*SCAN_TEMPERATURES, 60.0])


def test_fit_series_flags_the_broken_run_without_dropping_it(outcome) -> None:
    broken = next(entry for entry in outcome.results if entry["run"] == FLAT_RUN)

    # The flat run has no signal: its amplitude collapses away from the trend
    # the good runs set, which is exactly what the series' own diagnosis is for.
    assert "spurious_reseeded" in broken["quality_flags"]
    assert broken["parameters"]["A_1"] < 1.0
    # ... and it keeps its row. Excluding a point is the analyst's call.
    assert FLAT_RUN in [row["run"] for row in outcome.trend.rows]


def test_the_trend_table_carries_every_free_parameter_with_its_error(outcome) -> None:
    assert outcome.free_params == ["A_1", "Lambda", "A_bg"]
    assert outcome.trend.columns == [
        "run",
        "x",
        "A_1",
        "A_1_err",
        "Lambda",
        "Lambda_err",
        "A_bg",
        "A_bg_err",
        "flags",
    ]
    assert len(outcome.trend.rows) == len(ZF_RUNS)
    first = outcome.trend.rows[0]
    assert first["Lambda"] == pytest.approx(
        next(e["parameters"]["Lambda"] for e in outcome.results if e["run"] == first["run"])
    )


def test_the_trend_csv_has_one_header_line_and_one_line_per_run(outcome) -> None:
    lines = outcome.trend.to_csv().strip().split("\n")
    assert lines[0] == ",".join(outcome.trend.columns)
    assert len(lines) == 1 + len(ZF_RUNS)
    # Flags are a list in JSON; in CSV they join on ';' so a cell never splits
    # the row.
    assert all(line.count(",") == len(outcome.trend.columns) - 1 for line in lines)


def test_the_outcome_round_trips_through_its_dict(outcome) -> None:
    data = outcome.to_dict()
    assert data["name"] == "zf"
    assert data["order_key"] == "temperature"
    assert data["expression"] == _EXPRESSION
    assert data["trend"]["columns"] == outcome.trend.columns
    assert [entry["run"] for entry in data["results"]] == list(ZF_RUNS)


# -- pinned (global) parameters ---------------------------------------------


def test_a_global_parameter_is_pinned_in_every_run(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS[:3]}
    pinned = fit_series(
        datasets,
        recipe.with_overrides(fix={"A_bg": 0.0}),
        order_key="temperature",
        global_params=["A_bg"],
        name="pinned",
    )

    assert pinned.global_params == ["A_bg"]
    assert pinned.free_params == ["A_1", "Lambda"]
    assert "A_bg" not in pinned.trend.columns
    for entry in pinned.results:
        assert entry["parameters"]["A_bg"] == pytest.approx(0.0)


def test_a_global_parameter_the_recipe_does_not_have_is_rejected(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS[:2]}
    with pytest.raises(KeyError, match="Nope"):
        fit_series(datasets, recipe, order_key="temperature", global_params=["Nope"], name="x")
