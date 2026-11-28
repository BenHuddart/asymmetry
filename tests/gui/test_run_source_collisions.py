"""Import preserves the analysis owner of each occupied run number."""

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.io import load
from asymmetry.core.io.periods import encode_period_run_number
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.data_browser import DataBrowserPanel
from tests.io.test_load_source_defaults import source_file

pytestmark = pytest.mark.gui


@pytest.fixture
def sources(tmp_path):
    specs = (
        ("first", 44989, b"EMU", 1),
        ("second", 44989, b"MUSR", 2),
        ("other", 44990, b"EMU", 3),
    )
    paths = []
    for folder, number, instrument, scale in specs:
        path = tmp_path / folder / "run.nxs"
        path.parent.mkdir()
        source_file(path, file_anchor=2)
        with h5py.File(path, "r+") as handle:
            entry = handle["raw_data_1"]
            entry["run_number"][()] = number
            del entry["name"]
            entry.create_dataset("name", data=instrument)
            counts = entry["instrument/detector_1/counts"]
            counts[...] *= scale
        paths.append(path)
    return paths


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: pytest.fail(str(args[2])))
    instance = MainWindow()
    yield instance
    instance.close()


def test_browser_rejects_a_different_source(qapp, sources):
    first, second, _ = sources
    browser = DataBrowserPanel()
    original = load(first)
    browser.add_dataset(original)

    with pytest.raises(ValueError) as error:
        browser.add_dataset(load(second))

    assert all(str(path) in str(error.value) for path in (first, second))
    assert browser.get_all_datasets() == [original]


def test_conflict_in_a_batch_preserves_the_existing_fit(window, sources):
    first, second, other = sources
    window._load_files([str(first)])
    original = window._data_browser.get_dataset(44989)
    counts = original.run.histograms[0].counts.copy()
    window._plot_panel.plot_fit(
        original.time, np.full_like(original.time, 0.3), label="Original fit"
    )
    fit = window._plot_panel._fit_curves[44989]

    window._load_files([str(second), str(other)])

    assert window.statusBar().currentMessage() == "Loaded 1 file(s), 1 failed"
    log = window._log_panel.to_plain_text()
    assert "File → New Project" in log
    assert any(all(str(path) in line for path in (first, second)) for line in log.splitlines())
    assert window._data_browser.get_dataset(44990).run.source_file == str(other)
    assert window._data_browser.get_dataset(44989) is original
    np.testing.assert_array_equal(original.run.histograms[0].counts, counts)
    assert window._plot_panel._fit_curves[44989] is fit


def test_first_source_in_a_batch_claims_the_run_number(window, sources):
    first, second, _ = sources

    window._load_files([str(first), str(second)])

    assert window.statusBar().currentMessage() == "Loaded 1 file(s), 1 failed"
    assert [ds.run.source_file for ds in window._data_browser.get_all_datasets()] == [str(first)]


def test_refused_reload_keeps_the_old_rows_for_both_files(window, sources, monkeypatch):
    first, _, other = sources
    window._load_files([str(first), str(other)])
    originals = window._data_browser.get_all_datasets()
    with h5py.File(other, "r+") as handle:
        handle["raw_data_1/run_number"][()] = 44989
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)

    window._load_files([str(other)])

    assert window.statusBar().currentMessage() == "Failed to load 1 file(s)"
    assert window._data_browser.get_all_datasets() == originals


def test_conflict_in_one_period_refuses_the_whole_file(window, sources):
    first, second, other = sources
    occupied_number = encode_period_run_number(44989, 2)
    with h5py.File(first, "r+") as handle:
        handle["raw_data_1/run_number"][()] = occupied_number
    with h5py.File(second, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        counts = np.stack([detector["counts"][()]] * 3)
        del detector["counts"]
        detector.create_dataset("counts", data=counts)
    window._load_files([str(first)])

    window._load_files([str(second), str(other)])

    assert window.statusBar().currentMessage() == "Loaded 1 file(s), 1 failed"
    assert f"Run {occupied_number} is already loaded from" in window._log_panel.to_plain_text()
    datasets = window._data_browser.get_all_datasets()
    assert {ds.run_number: ds.run.source_file for ds in datasets} == {
        occupied_number: str(first),
        44990: str(other),
    }


def test_symlink_to_the_same_file_keeps_the_reload_choice(window, sources, monkeypatch):
    first, _, _ = sources
    alias = first.with_name("alias.nxs")
    try:
        alias.symlink_to(first)
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable: {error}")
    window._load_files([str(first)])
    original = window._data_browser.get_dataset(44989)
    prompts = []

    def choose(_parent, title, *_args):
        prompts.append(title)
        return (
            QMessageBox.StandardButton.No if len(prompts) == 1 else QMessageBox.StandardButton.Yes
        )

    monkeypatch.setattr(QMessageBox, "question", choose)
    window._load_files([str(alias)])
    assert window._data_browser.get_dataset(44989) is original

    window._load_files([str(alias)])
    assert window._data_browser.get_dataset(44989) is not original
    assert window._data_browser.get_dataset(44989).run.source_file == str(alias)
    assert prompts == ["Data File Already Loaded"] * 2
