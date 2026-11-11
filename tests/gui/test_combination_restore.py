"""Saved combination definitions preserve conditions, diagnostics, and fit history."""

import numpy as np
import pytest

from asymmetry.core.project import load_project, save_project
from asymmetry.core.representation import FitSeries, RepresentationType
from asymmetry.gui.mainwindow import MainWindow
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

    monkeypatch.setattr(restored, "_load_file", load_source)
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
    saved = restored.collect_project_state()
    assert saved["combined_datasets"] == state["combined_datasets"]
