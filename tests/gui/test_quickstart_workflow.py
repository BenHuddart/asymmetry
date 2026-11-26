"""The first-analysis route keeps its source, specified fit, and readable export."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

import asymmetry
from asymmetry.core import computation
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import build_builtin_template, simulate_run
from docs.screenshots.scenarios.corpus.quickstart import (
    ARGUS_COPPER_RUN,
    fit_quickstart,
    trigger_menu,
    wait_for_quickstart,
)
from tests.corpus import corpus_path, requires_corpus

pytestmark = pytest.mark.gui

_ARGUS_SOURCE = corpus_path(ARGUS_COPPER_RUN)


@pytest.fixture
def source(request, tmp_path) -> Path:
    if request.param is not None:
        return request.param
    pytest.importorskip("h5py")
    source = tmp_path / "test_zf.nxs"
    template = build_builtin_template("ideal_pulsed_fb", n_bins=320, bin_width_us=0.04)

    def signal(time):
        scaled_time_squared = (0.4 * time) ** 2
        return 21 * (1 + 2 * (1 - scaled_time_squared) * np.exp(-scaled_time_squared / 2)) / 3 + 5

    run = simulate_run(template, signal, total_events=10e6, seed=23, run_number=123)
    write_nexus_v1(run, source)
    return source


@pytest.mark.parametrize(
    ("source", "run_number", "expected_width", "expected_chi2", "expected_verdict"),
    [
        pytest.param(
            None,
            123,
            pytest.approx(0.4, abs=0.005),
            pytest.approx(1.0, abs=0.2),
            "good",
            id="offline",
        ),
        pytest.param(
            _ARGUS_SOURCE,
            76935,
            pytest.approx(0.394, abs=0.0005),
            pytest.approx(1.45, abs=0.005),
            "poor",
            marks=[requires_corpus(_ARGUS_SOURCE), pytest.mark.io],
            id="argus-copper",
        ),
    ],
    indirect=["source"],
)
def test_public_first_analysis(
    qapp, tmp_path, source, run_number, expected_width, expected_chi2, expected_verdict
):
    project = tmp_path / "first_analysis.asymp"
    report = tmp_path / "first_fit.txt"
    shell = fit_quickstart(source)
    page = shell.current_page()
    (dataset,) = page._data_browser.get_all_datasets()
    assert dataset.run_number == run_number
    ((title, record),) = page._collect_latest_fit_records()
    assert record["parameters"]["Delta"] == expected_width
    assert record["reduced_chi_squared"] == expected_chi2
    assert record["quality"]["verdict"] == expected_verdict

    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(project), "")):
        trigger_menu(shell, "File", "Save Project")
    wait_for_quickstart(lambda: not page._project_save_active, "project saving")
    assert project.is_file() and not page._dirty
    shell.close_page(page)
    assert page not in shell.pages()

    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(project), "")):
        trigger_menu(shell, "File", "Open Project…")
    reopened = shell.current_page()
    (restored,) = reopened._data_browser.get_all_datasets()
    assert restored.run_number == run_number
    assert restored.run.source_file == dataset.run.source_file == str(source)
    binding = dataset.run.source_binding.observed
    assert binding is not None
    assert restored.run.source_binding.expected == restored.run.source_binding.observed == binding
    assert reopened._collect_latest_fit_records() == [(title, record)]

    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(report), "")):
        trigger_menu(shell, "Analysis", "Export fit report…")
    contents = report.read_text(encoding="utf-8")
    assert "Model:  StaticGKT_ZF + Constant" in contents
    for name in ("A_1", "Delta", "A_bg"):
        match = re.search(rf"^\s*{name}\s+=\s+(\S+)\s+\+/-\s+(\S+)$", contents, re.MULTILINE)
        assert match is not None
        assert float(match[1]) == pytest.approx(record["parameters"][name], rel=5e-6)
        assert float(match[2]) == pytest.approx(record["uncertainties"][name], rel=5e-6)
        assert float(match[2]) > 0
    shell.close()


@pytest.mark.parametrize("source", [None], indirect=True)
def test_report_preserves_each_producing_version_across_reopen(qapp, tmp_path, source, monkeypatch):
    project = tmp_path / "versions.asymp"
    report = tmp_path / "versions.txt"
    version_a, version_b = "0.29.0", "0.30.0"
    monkeypatch.setattr(asymmetry, "__version__", version_a)
    shell = fit_quickstart(source)

    def save_and_close():
        page = shell.current_page()
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(project), "")):
            trigger_menu(shell, "File", "Save Project")
        wait_for_quickstart(lambda: not page._project_save_active, "project saving")
        assert project.is_file() and not page._dirty
        shell.close_page(page)
        assert page not in shell.pages()

    def open_project():
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(project), "")):
            trigger_menu(shell, "File", "Open Project…")
        return shell.current_page()

    def export_versions():
        report.unlink(missing_ok=True)
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(report), "")):
            trigger_menu(shell, "Analysis", "Export fit report…")
        return dict(
            re.findall(
                r"^=== (.+) ===\nAsymmetry version: (.+)$",
                report.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )

    try:
        ((title_a, original),) = shell.current_page()._collect_latest_fit_records()
        save_and_close()
        monkeypatch.setattr(asymmetry, "__version__", version_b)
        page = open_project()
        assert export_versions() == {title_a: version_a}

        (dataset,) = page._data_browser.get_all_datasets()
        page._data_browser.select_runs([dataset.run_number])
        trigger_menu(shell, "Analysis", "Fit")
        tab = page._fit_panel._single_tab
        assert tab._new_fit_btn.isEnabled()
        tab._new_fit_btn.click()
        tab._fit_btn.click()
        assert tab.wait_for_fit(timeout_ms=10_000)
        ((_, record_a), (title_b, record_b)) = page._collect_latest_fit_records()
        assert title_a != title_b
        assert record_a == original and record_b["success"]
        assert export_versions() == {title_a: version_a, title_b: version_b}

        save_and_close()
        page = open_project()
        assert export_versions() == {title_a: version_a, title_b: version_b}
        records = dict(page._collect_latest_fit_records())
        assert records[title_a] == original and records[title_b] == record_b
        assert original["computation_revision"] == computation.COMPUTATION_REVISION

        save_and_close()
        state = json.loads(project.read_text(encoding="utf-8"))
        representation = state["datasets"][0]["representations"]["time_fb_asymmetry"]
        legacy = representation["single_fits"][""]["fits"][0]["result"]
        assert legacy.pop("producing_version") == version_a
        project.write_text(json.dumps(state), encoding="utf-8")

        page = open_project()
        assert export_versions() == {title_a: "unknown", title_b: version_b}
        assert dict(page._collect_latest_fit_records()) == {title_a: legacy, title_b: record_b}
        save_and_close()
        page = open_project()
        assert export_versions() == {title_a: "unknown", title_b: version_b}
        assert dict(page._collect_latest_fit_records()) == {title_a: legacy, title_b: record_b}
    finally:
        for page in shell.pages():
            page._clear_dirty()
        shell.close()
