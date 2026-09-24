"""Tests for :mod:`asymmetry.core.workflow.series`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.series import (
    ORDER_KEYS,
    fit_one,
    fit_series,
    scan_axis,
    supplied_axis,
)
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
    return fit_series(datasets, recipe, axis=scan_axis(datasets, "temperature"), name="zf")


# -- order keys -------------------------------------------------------------


def test_scan_axis_reads_the_scan_coordinate_of_every_run(
    reduced_workdir: WorkDir,
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS}
    axis = scan_axis(datasets, "temperature")
    assert axis.name == "temperature"
    assert axis.values == {
        run: temperature for run, temperature in zip(SCAN_RUNS, SCAN_TEMPERATURES)
    }
    assert scan_axis(datasets, "run").values == {run: float(run) for run in SCAN_RUNS}
    assert "run" in ORDER_KEYS


def test_scan_axis_reads_the_logged_sample_temperature_apart_from_the_setpoint() -> None:
    logged = MuonDataset(
        time=np.linspace(0.0, 8.0, 10),
        asymmetry=np.zeros(10),
        error=np.ones(10),
        metadata={"run_number": 7, "temperature": 50.0, "sample_temperature_logged": 56.4},
    )
    assert scan_axis({7: logged}, "temperature").values == {7: 50.0}
    assert scan_axis({7: logged}, "sample_temperature_logged").values == {7: 56.4}


def test_scan_axis_names_the_runs_that_do_not_record_the_quantity() -> None:
    bare = MuonDataset(
        time=np.linspace(0.0, 8.0, 10),
        asymmetry=np.zeros(10),
        error=np.ones(10),
        metadata={"run_number": 7},
    )
    with pytest.raises(ValueError, match="Run\\(s\\) 7 record no temperature"):
        scan_axis({7: bare}, "temperature")
    with pytest.raises(ValueError, match="'pressure' is not recorded in the files"):
        scan_axis({7: bare}, "pressure")


def test_a_supplied_axis_takes_exactly_one_value_per_run() -> None:
    axis = supplied_axis("concentration", {1: 0.0, 2: 0.5}, [1, 2])
    assert (axis.name, axis.values) == ("concentration", {1: 0.0, 2: 0.5})
    with pytest.raises(ValueError, match="No concentration value for run\\(s\\) 2"):
        supplied_axis("concentration", {1: 0.0}, [1, 2])
    with pytest.raises(ValueError, match="given for run\\(s\\) 3, which are not in the series"):
        supplied_axis("concentration", {1: 0.0, 2: 0.5, 3: 1.0}, [1, 2])
    with pytest.raises(ValueError, match="'field' is read from the files"):
        supplied_axis("field", {1: 0.0, 2: 0.5}, [1, 2])


def test_a_supplied_axis_orders_the_series_and_labels_its_trend(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    runs = SCAN_RUNS[:3]
    datasets = {run: reduced_workdir.reduced(run) for run in runs}
    # Reversed against run number, so ordering by it is visible.
    axis = supplied_axis("foils", {run: float(len(runs) - i) for i, run in enumerate(runs)}, runs)
    outcome = fit_series(datasets, recipe, axis=axis, name="foils")

    assert outcome.order_key == "foils"
    assert outcome.trend.order_key == "foils"
    assert [row["run"] for row in outcome.trend.rows] == list(reversed(runs))
    assert [row["x"] for row in outcome.trend.rows] == [1.0, 2.0, 3.0]


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
    assert [branch.seeding_used for branch in outcome.branches] == ["chain"]
    assert outcome.branches[0].seeding_reason
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


# -- where the chain starts --------------------------------------------------


def _parameters_by_run(outcome) -> dict[int, dict[str, float]]:
    return {entry["run"]: entry["parameters"] for entry in outcome.results}


def test_a_default_series_is_one_chain_over_the_whole_scan(outcome) -> None:
    assert outcome.start_run is None
    assert [branch.direction for branch in outcome.branches] == ["ascending"]
    assert outcome.branches[0].runs == list(ZF_RUNS)
    assert outcome.branches[0].seeding_used == "chain"


def test_starting_at_the_first_run_is_the_default(
    reduced_workdir: WorkDir, recipe: FitRecipe, outcome
) -> None:
    # The descending branch would hold only that run, so there is nothing to
    # chain downward and the whole scan is one ascending chain — the same fits,
    # in the same order, as omitting --start entirely.
    datasets = {run: reduced_workdir.reduced(run) for run in ZF_RUNS}
    started = fit_series(
        datasets, recipe, axis=scan_axis(datasets, "temperature"), start_run=ZF_RUNS[0], name="zf"
    )

    assert [branch.direction for branch in started.branches] == ["ascending"]
    assert _parameters_by_run(started) == _parameters_by_run(outcome)
    assert started.trend.to_dict() == outcome.trend.to_dict()
    assert started.reseeded_runs == outcome.reseeded_runs


def test_starting_mid_scan_merges_the_two_branches_unchanged(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    # The merge must be exactly the two branches, not a re-fit of anything: each
    # branch is reproduced here on its own runs (where the start run sits at one
    # end, so that call is a single chain in the same direction) and every
    # fitted value must match.
    datasets = {run: reduced_workdir.reduced(run) for run in ZF_RUNS}
    start = ZF_RUNS[2]
    below = {run: datasets[run] for run in ZF_RUNS[: ZF_RUNS.index(start) + 1]}
    above = {run: datasets[run] for run in ZF_RUNS[ZF_RUNS.index(start) :]}

    merged = fit_series(
        datasets, recipe, axis=scan_axis(datasets, "temperature"), start_run=start, name="outward"
    )
    descending = fit_series(
        below, recipe, axis=scan_axis(below, "temperature"), start_run=start, name="down"
    )
    ascending = fit_series(
        above, recipe, axis=scan_axis(above, "temperature"), start_run=start, name="up"
    )

    assert [branch.direction for branch in merged.branches] == ["descending", "ascending"]
    assert merged.branches[0].runs == list(reversed(below))
    assert merged.branches[1].runs == list(above)
    assert [branch.direction for branch in descending.branches] == ["descending"]
    assert [branch.direction for branch in ascending.branches] == ["ascending"]

    values = _parameters_by_run(merged)
    assert [entry["run"] for entry in merged.results] == list(ZF_RUNS)
    for run, expected in _parameters_by_run(descending).items():
        # The start run belongs to the ascending branch in the merge.
        if run != start:
            assert values[run] == expected, run
    for run, expected in _parameters_by_run(ascending).items():
        assert values[run] == expected, run


def test_the_start_run_is_fitted_from_the_recipe_itself(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    # Nothing chains into the start run, so it must land exactly where a single
    # fit of that run with the same recipe lands.
    datasets = {run: reduced_workdir.reduced(run) for run in ZF_RUNS}
    start = ZF_RUNS[2]

    merged = fit_series(
        datasets, recipe, axis=scan_axis(datasets, "temperature"), start_run=start, name="outward"
    )
    alone = fit_one(datasets[start], recipe)

    assert _parameters_by_run(merged)[start] == alone["parameters"]


def test_a_start_run_outside_the_series_is_rejected(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS[:3]}
    with pytest.raises(ValueError, match="Start run 999 is not in this series"):
        fit_series(
            datasets, recipe, axis=scan_axis(datasets, "temperature"), start_run=999, name="nope"
        )


# -- pinned (global) parameters ---------------------------------------------


def test_a_global_parameter_is_pinned_in_every_run(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS[:3]}
    pinned = fit_series(
        datasets,
        recipe.with_overrides(fix={"A_bg": 0.0}),
        axis=scan_axis(datasets, "temperature"),
        global_params=["A_bg"],
        name="pinned",
    )

    assert pinned.global_params == ["A_bg"]
    assert pinned.free_params == ["A_1", "Lambda"]
    assert "A_bg" not in pinned.trend.columns
    for entry in pinned.results:
        assert entry["parameters"]["A_bg"] == pytest.approx(0.0)


# -- run-bound values --------------------------------------------------------

#: A longitudinal-field decoupling set: the applied field is the thing that
#: changes run to run, and the model carries it as ``B_L``.
_LF_EXPRESSION = "LongitudinalFieldKT + Constant"
_LF_FIELDS = (50.0, 100.0, 200.0)


def _lf_datasets() -> dict[int, MuonDataset]:
    """Three synthetic longitudinal-field runs, each at its own field."""
    model = CompositeModel.from_expression(_LF_EXPRESSION)
    time = np.linspace(0.05, 8.0, 240)
    datasets: dict[int, MuonDataset] = {}
    for index, field in enumerate(_LF_FIELDS):
        run = 200 + index
        asymmetry = model.function(time, A_1=18.0, Delta=0.3, B_L=field, A_bg=0.5)
        datasets[run] = MuonDataset(
            time=time,
            asymmetry=asymmetry,
            error=np.full_like(time, 0.2),
            metadata={"run_number": run, "field": field, "temperature": 10.0},
        )
    return datasets


def _lf_recipe(datasets: dict[int, MuonDataset]) -> FitRecipe:
    """An LF recipe seeded from the first run, with ``B_L`` held as the wizard would."""
    first = datasets[min(datasets)]
    recipe = FitRecipe.from_expression(_LF_EXPRESSION, dataset=first)
    return replace(
        recipe,
        parameters=tuple(
            replace(parameter, fixed=True) if parameter.name == "B_L" else parameter
            for parameter in recipe.parameters
        ),
    )


def test_fit_series_re_seeds_a_run_bound_parameter_from_each_runs_own_record() -> None:
    # The recipe carries one field — the first run's. Every other run must be
    # fitted at *its* field, or a decoupling scan is fitted at the wrong field
    # everywhere but its first point. B_L stays held throughout.
    datasets = _lf_datasets()

    outcome = fit_series(
        datasets, _lf_recipe(datasets), axis=scan_axis(datasets, "field"), name="lf"
    )

    fitted = {entry["run"]: entry["parameters"]["B_L"] for entry in outcome.results}
    assert list(fitted.values()) == pytest.approx(list(_LF_FIELDS))
    assert "B_L" not in outcome.free_params


def test_a_hand_pinned_run_bound_parameter_keeps_the_value_a_person_chose() -> None:
    # --fix is a decision, not a measurement off a run, so nothing re-seeds it.
    datasets = _lf_datasets()
    recipe = _lf_recipe(datasets).with_overrides(fix={"B_L": 7.0})

    outcome = fit_series(datasets, recipe, axis=scan_axis(datasets, "field"), name="lf-pinned")

    assert [entry["parameters"]["B_L"] for entry in outcome.results] == pytest.approx(
        [7.0] * len(_LF_FIELDS)
    )


def test_a_global_run_bound_parameter_is_held_at_the_recipes_value() -> None:
    # "Global" means identical across the scan; re-seeding it per run would
    # contradict the word.
    datasets = _lf_datasets()
    recipe = _lf_recipe(datasets)

    outcome = fit_series(
        datasets, recipe, axis=scan_axis(datasets, "field"), global_params=["B_L"], name="lf-global"
    )

    recipe_field = next(p.value for p in recipe.parameters if p.name == "B_L")
    assert [entry["parameters"]["B_L"] for entry in outcome.results] == pytest.approx(
        [recipe_field] * len(_LF_FIELDS)
    )


def test_fit_one_starts_from_the_recipe_as_written() -> None:
    # No re-seeding for a single fit: the caller aimed this recipe at this run.
    datasets = _lf_datasets()
    recipe = _lf_recipe(datasets)
    last = max(datasets)

    result = fit_one(datasets[last], recipe)

    assert result["parameters"]["B_L"] == pytest.approx(_LF_FIELDS[0])


def test_a_global_parameter_the_recipe_does_not_have_is_rejected(
    reduced_workdir: WorkDir, recipe: FitRecipe
) -> None:
    datasets = {run: reduced_workdir.reduced(run) for run in SCAN_RUNS[:2]}
    with pytest.raises(KeyError, match="Nope"):
        fit_series(
            datasets,
            recipe,
            axis=scan_axis(datasets, "temperature"),
            global_params=["Nope"],
            name="x",
        )


def test_a_frequency_trend_sets_the_surveyed_line_beside_the_fit() -> None:
    from asymmetry.core.workflow.series import build_trend_table, survey_line

    def dataset(metadata: dict) -> MuonDataset:
        return MuonDataset(np.zeros(3), np.zeros(3), np.ones(3), metadata)

    assert survey_line(dataset({"precession": "other", "precession_frequency_mhz": 29.9})) == 29.9
    assert survey_line(dataset({"precession": "none", "precession_frequency_mhz": None})) is None
    assert survey_line(dataset({})) is None

    results = [
        {
            "run": run,
            "x": x,
            "parameters": {"frequency": f},
            "uncertainties": {"frequency": 0.1},
            "quality_flags": [],
        }
        for run, x, f in ((1, 10.0, 29.8), (2, 60.0, 0.4))
    ]
    trend = build_trend_table(
        results, ["frequency"], "temperature", survey_lines={1: 29.9, 2: None}
    )
    assert "survey_line_mhz" in trend.columns
    assert [row["survey_line_mhz"] for row in trend.rows] == [29.9, None]
    assert "survey_line_mhz" not in build_trend_table(results, ["frequency"], "run").columns


def test_amplitudes_far_beyond_the_record_are_flagged() -> None:
    from asymmetry.core.workflow.series import amplitude_exceeds_data

    time = np.linspace(0.0, 8.0, 400)
    # A 20 % precession: its early mean is near zero, its scale is not.
    record = MuonDataset(time, 20.0 * np.cos(2 * np.pi * 1.4 * time), np.ones(400), {})
    assert not amplitude_exceeds_data(record, {"A_1": 20.0, "frequency": 1.4, "A_bg": 0.5})
    # Two amplitudes cancelling to describe the same 20 % are not.
    assert amplitude_exceeds_data(record, {"A_1": 143.6, "Lambda": 3.0, "A_bg": -120.0})
    # Non-amplitude parameters never count.
    assert not amplitude_exceeds_data(record, {"A_1": 18.0, "nu": 16351.0})
