"""The public first-analysis route keeps its data, fit, and readable export."""

import re

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.gui.mainwindow import MainWindow
from docs.screenshots.scenarios.quickstart_first_fit import run_quickstart

pytestmark = [pytest.mark.gui, pytest.mark.real_save_guard]


def test_public_synthetic_first_analysis(qapp, tmp_path):
    pytest.importorskip("h5py")
    shell = run_quickstart(tmp_path)
    page = shell.current_page()
    source = tmp_path / "first_run.nxs"
    project = tmp_path / "first_analysis.asymp"
    dataset = page._data_browser.get_all_datasets()[0]
    assert dataset.run.source_file == str(source)
    assert not page._unsaved_synthetic_run_labels()
    _, record = page._collect_latest_fit_records()[0]
    assert record["parameters"]["Lambda"] == pytest.approx(0.3, abs=0.005)
    assert record["uncertainties"]["Lambda"] > 0
    report = (tmp_path / "first_fit.txt").read_text()
    assert "Model:" in report
    assert 0.5 < record["reduced_chi_squared"] < 1.5
    match = re.search(r"Lambda\s+=\s+([\d.]+)\s+\+/-\s+([\d.]+)", report)
    assert match is not None
    assert float(match[1]) == pytest.approx(record["parameters"]["Lambda"], abs=5e-7)
    assert float(match[2]) > 0
    restored = MainWindow()
    restored._open_project_file(str(project))
    again = restored._data_browser.get_all_datasets()[0]
    for original, reopened in zip(dataset.run.histograms, again.run.histograms, strict=True):
        np.testing.assert_array_equal(original.counts, reopened.counts)
    assert restored._collect_latest_fit_records()[0][1]["parameters"] == record["parameters"]
    assert again.run.source_binding.expected == load(source).run.source_binding.observed
    restored._clear_dirty()
    restored.close()
    for open_page in shell.pages():
        open_page._clear_dirty()
    shell.close()
