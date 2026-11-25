"""A synthetic preset can acquire verified file backing without rewriting its fits."""

import threading
import time
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

import asymmetry.core.io.persist_run as persistence
from asymmetry.core.data.acquisition import ContentIdentity
from asymmetry.core.io import load
from asymmetry.core.project import save_project
from asymmetry.core.representation import RepresentationType
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import SINGLE_FIT_ID
from asymmetry.gui.utils.scientific_input import ScientificInputDialog
from tests.gui.test_project_save_synthetic_warning import _add_synthetic
from tests.gui.test_saved_single_fits import _EXP, _fit

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()
    return window


def save_action(browser):
    browser.select_runs({90001})
    menu = browser._create_table_context_menu()
    assert menu is not None
    return next((action for action in menu.actions() if action.text() == "Save as NeXus…"), None)


def test_preset_has_save_route(window, qapp):
    _add_synthetic(window)
    assert save_action(window._data_browser) is not None


def test_path_without_capture_stays_unsaved(window, qapp, tmp_path):
    _add_synthetic(window)
    window._data_browser.get_dataset(90001).run.source_file = str(tmp_path / "absent.nxs")
    assert window._unsaved_synthetic_run_labels()


def test_verified_save_preserves_settings_history_and_actual_project_reopen(
    window, qapp, tmp_path, monkeypatch
):
    _add_synthetic(window)
    browser = window._data_browser
    ds = browser.get_dataset(90001)
    ds.run.grouping["alpha"] = 1.17
    ds = reduce_run_to_dataset(ds.run)
    browser.add_dataset(ds)
    window._on_dataset_selected(90001)
    _fit(window, _EXP)
    rep = window._project_model.representation(90001, RepresentationType.TIME_FB_ASYMMETRY)
    history = deepcopy(rep.fit_set(None).to_dict())
    old_recipe = ds.effective_recipe
    old_observation = ds.histogram_observation
    raw = [hist.counts.copy() for hist in ds.run.histograms]
    metadata = deepcopy(ds.run.metadata)
    grouping = deepcopy(ds.run.grouping)
    path = tmp_path / "preset.nxs"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    save_action(browser).trigger()
    assert ds.run.source_file == str(path)
    identity = ContentIdentity.read(path)
    assert ds.run.source_binding.expected == ds.run.source_binding.observed == identity
    assert ds.run.metadata == metadata
    assert ds.run.grouping == grouping
    assert rep.fit_set(None).to_dict() == history
    assert old_recipe.sources()[0].expected is None
    assert old_observation.source.source_binding.observed is None
    assert ds.histogram_observation.source.source_binding.observed == identity
    assert ds.effective_recipe.sources()[0].expected == identity
    assert window._unsaved_synthetic_run_labels() == []
    assert save_action(browser) is None
    for expected, hist in zip(raw, load(str(path)).run.histograms, strict=True):
        np.testing.assert_array_equal(expected, hist.counts)
    project = tmp_path / "preset.asymp"
    save_project(window.collect_project_state(), project)
    restored = MainWindow()
    restored._open_project_file(str(project))
    reopened = restored._data_browser.get_dataset(90001)
    assert reopened is not None
    assert reopened.run.grouping["alpha"] == 1.17
    assert reopened.run.source_binding == ds.run.source_binding
    reopened_rep = restored._project_model.representation(
        90001, RepresentationType.TIME_FB_ASYMMETRY
    )
    assert reopened_rep.fit_set(None).to_dict() == history
    restored._plot_workspace.set_active_view("fb_asymmetry")
    restored._on_dataset_selected(90001)
    restored._on_saved_fit_open_requested(reopened_rep.fit_set(None).open_id)
    assert SINGLE_FIT_ID in restored._plot_panel.stored_fit_ids(90001)
    for expected, hist in zip(raw, reopened.run.histograms, strict=True):
        np.testing.assert_array_equal(expected, hist.counts)
    restored._clear_dirty()
    restored.close()


def test_file_dialog_cancel_keeps_memory_owner(window, qapp, monkeypatch):
    _add_synthetic(window)
    ds = window._data_browser.get_dataset(90001)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: ("", ""))
    save_action(window._data_browser).trigger()
    assert ds.run.source_file == ""
    assert ds.run.source_binding.observed is None


@pytest.mark.parametrize("failure", ["write", "verification"])
def test_failure_leaves_owner_and_existing_destination_intact(
    window, qapp, tmp_path, monkeypatch, failure
):
    _add_synthetic(window)
    ds = window._data_browser.get_dataset(90001)
    path = tmp_path / "existing.nxs"
    path.write_bytes(b"old destination")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: messages.append(a[2]))
    if failure == "write":

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr(persistence, "write_nexus_v1", fail)
    else:
        real = persistence.load

        def corrupt(*args, **kwargs):
            result = real(*args, **kwargs)
            result.run.histograms[0].counts[0] += 1
            return result

        monkeypatch.setattr(persistence, "load", corrupt)
    save_action(window._data_browser).trigger()
    assert messages
    assert ds.run.source_file == ""
    assert ds.run.source_binding.observed is None
    assert all(hist.counts.flags.writeable for hist in ds.run.histograms)
    assert path.read_bytes() == b"old destination"
    assert list(tmp_path.glob("*.nxs")) == [path]
    assert not list(tmp_path.glob(".*.nxs"))


@pytest.mark.parametrize(
    "change", ["cancel", "remove", "replace", "project", "counts", "grouping", "source"]
)
def test_worker_io_keeps_heartbeat_and_discards_obsolete_completion(
    window, qapp, tmp_path, monkeypatch, change
):
    _add_synthetic(window)
    browser = window._data_browser
    ds = browser.get_dataset(90001)
    path = tmp_path / "captured.nxs"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    entered = threading.Event()
    release = threading.Event()
    threads = []
    real = persistence.write_nexus_v1

    def write(*args):
        threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(3)
        return real(*args)

    monkeypatch.setattr(persistence, "write_nexus_v1", write)
    beats = []
    timer = QTimer()
    timer.setInterval(1)

    def tick():
        beats.append(time.monotonic())
        if not entered.is_set():
            return
        if change == "cancel":
            browser.findChild(ScientificInputDialog).canceled.emit()
        elif change == "remove":
            browser.clear()
        elif change == "replace":
            browser.add_dataset(reduce_run_to_dataset(deepcopy(ds.run)))
        elif change == "grouping":
            window._apply_grouping_settings_to_dataset(ds, dict(ds.run.grouping, alpha=1.23))
        elif change == "source":
            ds.run.source_file = "different.nxs"
        elif change == "counts":
            ds.run.histograms[0].counts = ds.run.histograms[0].counts + 1
        else:
            from asymmetry.core.representation.project_model import ProjectModel

            browser.set_project_model(ProjectModel())
        release.set()
        timer.stop()

    timer.timeout.connect(tick)
    timer.start()
    save_action(browser).trigger()
    assert beats
    assert threads and all(thread != qapp.thread() for thread in threads)
    assert ds.run.source_file == ("different.nxs" if change == "source" else "")
    assert ds.run.source_binding.observed is None
    if change == "cancel":
        assert not path.exists()
    assert not list(tmp_path.glob(".*.nxs"))


@pytest.mark.parametrize("excluded", ["file", "combined"])
def test_save_route_excludes_other_owners(window, qapp, tmp_path, excluded):
    _add_synthetic(window)
    browser = window._data_browser
    dataset = browser.get_dataset(90001)
    if excluded == "file":
        dataset.run = persistence.persist_histogram_run(dataset.run, tmp_path / "source.nxs")
    else:
        browser._combined_datasets[90001] = [90001]
    assert save_action(browser) is None
