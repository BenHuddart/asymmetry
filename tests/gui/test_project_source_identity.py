"""Actual project restoration and collection preserve historical byte expectations."""

import pytest

from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding
from asymmetry.core.io import load
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, load_prepared_project, save_project
from asymmetry.gui.mainwindow import MainWindow
from tests.io.test_acquisition_content import acquisition

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("expected", [None, ContentIdentity("a" * 64)])
def test_actual_open_and_resave_retains_expected_and_observed(qapp, tmp_path, expected):
    path = acquisition(tmp_path)
    actual = load(str(path)).run.source_binding.observed
    entry = {"run_number": load(str(path)).run_number, "source_file": str(path)}
    if expected is not None:
        entry["source_binding"] = SourceBinding(expected, expected).to_dict()
    project = tmp_path / "source.asymp"
    save_project({"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]}, project)
    window = MainWindow()
    window._open_project_file(str(project))
    ds = window._data_browser.get_dataset(entry["run_number"])
    assert ds is not None
    assert ds.run.source_binding == SourceBinding(expected, actual)
    state = window.collect_project_state()
    assert state["datasets"][0]["source_binding"] == SourceBinding(expected, actual).to_dict()
    save_project(state, project)
    reopened = load_prepared_project(project)
    assert reopened.sources[0].expected == expected
    window._clear_dirty()
    window.close()
