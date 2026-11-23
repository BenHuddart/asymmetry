"""Saved combination definitions preserve conditions, diagnostics, and fit history."""

import json

import numpy as np
import pytest

from asymmetry.core.project import load_project, save_project
from asymmetry.core.representation import FitSeries, RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from tests.gui.source_replay import sample_decoder
from tests.gui.test_data_browser_combine import _dataset

pytestmark = [pytest.mark.gui]


@pytest.mark.parametrize("conflict", [False, True])
def test_project_restores_target_override_or_unavailable_definition_with_history(
    qapp, tmp_path, monkeypatch, conflict
):
    window = MainWindow()
    for number in (401, 402):
        dataset = _dataset(number)
        path = tmp_path / f"run-{number}.nxs"
        path.touch()
        dataset.run.source_file = str(path)
        dataset.metadata["temperature"] = None if number == 401 else 20
        window._data_browser.add_dataset(dataset)
    browser = window._data_browser
    browser.add_combined_dataset([401, 402], combined_run_number=-1)
    browser._table.item(0, 3).setText("42")
    window._project_model.add_batch(
        FitSeries(
            "history",
            RepresentationType.TIME_FB_ASYMMETRY,
            member_run_numbers=[-1],
            results_by_run={-1: {"field": 17.5, "temperature": 12, "parameters": {"A": 0.2}}},
        )
    )
    state = window.collect_project_state()
    assert state["combined_datasets"][0]["metadata_overrides"] == {"field": 42}
    project = tmp_path / "combined.asymp"
    save_project(state, project)
    restored = MainWindow()

    def load_source(path):
        number = 401 if "401" in path else 402
        dataset = _dataset(number)
        dataset.run.source_file = path
        dataset.metadata["temperature"] = None if number == 401 else 20
        dataset.run.grouping["histogram_labels"] = (
            ["Backward", "Forward"] if conflict and number == 402 else ["Forward", "Backward"]
        )
        return dataset

    monkeypatch.setattr(restored, "_load_file", sample_decoder(load_source))
    restored.restore_project_state(load_project(project), str(project))
    result = restored._data_browser.get_dataset(-1)
    assert result is not None
    assert restored._project_model.batch("history").results_by_run[-1] == {
        "field": 17.5,
        "temperature": 12,
        "parameters": {"A": 0.2},
    }
    if conflict:
        assert "correspondence" in result.metadata["correction_error"]
        assert {401, 402, -1} <= set(restored._data_browser._datasets)
        assert result.run is None
        assert result.n_points == 0
        assert restored._data_browser.displayed_coordinate(-1)["field"] == 42
    else:
        assert result.metadata["temperature"] is None
        assert result.metadata["field"] == 42
        assert restored._data_browser.displayed_coordinate(-1)["temperature"] is None
        assert np.isfinite(result.asymmetry).all()
    historical = restored._project_model.representation(-1, RepresentationType.TIME_FB_ASYMMETRY)
    original = state["combined_datasets"][0]["representations"]["time_fb_asymmetry"][
        "result_metadata"
    ]
    assert historical.result_metadata == original
    saved = restored.collect_project_state()
    if conflict:
        assert json.loads(json.dumps(saved["combined_datasets"])) == json.loads(
            json.dumps(state["combined_datasets"])
        )
    else:
        current = saved["combined_datasets"][0]
        expected = json.loads(json.dumps(state["combined_datasets"][0]))
        expected["grouping_overrides"]["histogram_labels"] = ["Forward", "Backward"]
        expected["grouping_overrides"]["deadtime_file_us"] = []
        assert json.loads(
            json.dumps({key: value for key, value in current.items() if key != "representations"})
        ) == {key: value for key, value in expected.items() if key != "representations"}
        records = current["representations"]["time_fb_asymmetry"]["result_metadata"][
            "effective_recipes"
        ]
        assert records == original["effective_recipes"]


@pytest.mark.parametrize("sign,scales", [(1, [1, 2]), (-1, [2, 0.5])])
def test_unavailable_weighted_definition_survives_rebuild(
    qapp, tmp_path, monkeypatch, sign, scales
):
    from asymmetry.core.data.acquisition import constituent_sources_for_project
    from asymmetry.core.data.combine import combine_runs, reduce_combined_run

    window = MainWindow()
    sources = []
    for number in (401, 402):
        dataset = _dataset(number, seed=number)
        path = tmp_path / f"run-{number}.nxs"
        path.touch()
        dataset.run.source_file = str(path)
        sources.append(dataset)
        window._data_browser.add_dataset(dataset)
    operation = "coadd" if sign == 1 else "subtract_signed"
    window._data_browser.add_combined_dataset(
        [401, 402], sign=sign, operation=operation, combined_run_number=-1
    )
    weighted = reduce_combined_run(
        combine_runs(
            [dataset.run for dataset in sources],
            scales=scales,
            sign=sign,
            subtract_method="signed",
            run_number=-1,
        )
    )
    window._data_browser._datasets[-1] = weighted
    window._project_model.add_batch(
        FitSeries(
            "history",
            RepresentationType.TIME_FB_ASYMMETRY,
            member_run_numbers=[-1],
            results_by_run={-1: {"parameters": {"A": 0.2}}},
        )
    )
    project = tmp_path / "weighted.asymp"
    state = window.collect_project_state()
    terms = state["combined_datasets"][0]["constituent_sources"]
    assert [row["coefficient"] for row in terms] == [scales[0], sign * scales[1]]
    save_project(state, project)
    restored = MainWindow()

    def load_source(path):
        number = 401 if "401" in path else 402
        dataset = _dataset(number, seed=number)
        dataset.run.source_file = path
        return dataset

    monkeypatch.setattr(restored, "_load_file", sample_decoder(load_source))
    restored.restore_project_state(load_project(project), str(project))
    browser = restored._data_browser
    unavailable = browser.get_dataset(-1)
    assert unavailable.run is None
    saved_terms = constituent_sources_for_project(unavailable)
    assert [row["coefficient"] for row in saved_terms] == [scales[0], sign * scales[1]]
    assert browser.rebuild_combined_dataset(-1) is None
    assert browser.get_dataset(-1) is unavailable
    assert unavailable.run is None
    assert unavailable.n_points == 0
    assert constituent_sources_for_project(unavailable) == saved_terms
    assert (
        "Saved combination terms cannot be reconstructed"
        in unavailable.metadata["correction_error"]
    )
    assert json.loads(
        json.dumps(restored.collect_project_state()["combined_datasets"])
    ) == json.loads(json.dumps(state["combined_datasets"]))
    assert restored._project_model.batch("history").results_by_run[-1] == {"parameters": {"A": 0.2}}


@pytest.mark.parametrize("operation", ["coadd", "subtract_signed", "subtract_reference"])
def test_supported_saved_terms_survive_rebuild(qapp, tmp_path, monkeypatch, operation):
    from copy import deepcopy

    window = MainWindow()
    for number in (401, 402):
        dataset = _dataset(number, expected=500 if number == 401 else 100, seed=number)
        path = tmp_path / f"run-{number}.nxs"
        path.touch()
        dataset.run.source_file = str(path)
        window._data_browser.add_dataset(dataset)
    window._data_browser.add_combined_dataset(
        [401, 402],
        sign=1 if operation == "coadd" else -1,
        operation=operation,
        combined_run_number=-1,
    )
    state = window.collect_project_state()
    project = tmp_path / "supported.asymp"
    save_project(state, project)
    restored = MainWindow()

    def load_source(path):
        number = 401 if "401" in path else 402
        dataset = _dataset(number, expected=500 if number == 401 else 100, seed=number)
        dataset.run.source_file = path
        return dataset

    monkeypatch.setattr(restored, "_load_file", sample_decoder(load_source))
    restored.restore_project_state(load_project(project), str(project))
    row = restored._data_browser.get_dataset(-1)
    before = tuple(array.copy() for array in (row.time, row.asymmetry, row.error))
    terms = deepcopy(row.metadata["constituent_source_expectations"])
    assert restored._data_browser.rebuild_combined_dataset(-1) is row
    for actual, expected in zip((row.time, row.asymmetry, row.error), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert row.metadata["constituent_source_expectations"] == terms
    expected = json.loads(json.dumps(state["combined_datasets"]))
    expected[0]["grouping_overrides"]["deadtime_file_us"] = []
    assert json.loads(json.dumps(restored.collect_project_state()["combined_datasets"])) == expected


def test_registered_core_weighted_result_cannot_rebuild_as_unit_sum(qapp):
    from asymmetry.core.data.combine import combine_runs, reduce_combined_run

    window = MainWindow()
    sources = [_dataset(number, seed=number) for number in (401, 402)]
    for dataset in sources:
        window._data_browser.add_dataset(dataset)
    window._data_browser.add_combined_dataset([401, 402], combined_run_number=-1)
    weighted = reduce_combined_run(
        combine_runs([dataset.run for dataset in sources], scales=[1, 2], run_number=-1)
    )
    window._data_browser.add_dataset(weighted)
    before = tuple(array.copy() for array in (weighted.time, weighted.asymmetry, weighted.error))
    assert window._data_browser.rebuild_combined_dataset(-1) is None
    assert window._data_browser.get_dataset(-1) is weighted
    for actual, expected in zip(
        (weighted.time, weighted.asymmetry, weighted.error), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert [row["coefficient"] for row in weighted.metadata["combination"]["constituents"]] == [
        1,
        2,
    ]
