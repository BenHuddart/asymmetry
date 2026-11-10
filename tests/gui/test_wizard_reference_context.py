"""Resolved acquisition identity belongs to warm toolbar/wizard provenance."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.preparation import histogram_observation, prepare_histogram_asymmetry
from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest
from asymmetry.gui.mainwindow import MainWindow
from tests.core.test_reference_timing import timing_runs

pytestmark = pytest.mark.gui


def test_equal_reference_counts_from_another_acquisition_refresh_provenance(qapp, monkeypatch):
    mainwindow = MainWindow()
    sample, source = timing_runs()
    sample.grouping["background_run"] = {"run_number": 2, "period_number": 1}
    scale = ReferenceScaleRequest("manual", 1).resolve(ReferenceExposure(), ReferenceExposure())
    reference = BackgroundReference(
        source.histograms, scale, run_number=2, source_file="reference-a.nxs"
    )
    base = prepare_histogram_asymmetry(sample, merge=1, facility=None, reference=reference)
    dataset = MuonDataset(
        base.time,
        base.asymmetry,
        base.error,
        run=sample,
        histogram_observation=histogram_observation(sample, reference=reference),
    )
    selected = [reference]
    monkeypatch.setattr(mainwindow, "_resolve_background_reference", lambda grouping: selected[0])
    first = mainwindow._counts_first_rebunched_dataset(dataset, 2)
    selected[0] = replace(reference, source_file="reference-b.nxs")
    second = mainwindow._counts_first_rebunched_dataset(dataset, 2)
    np.testing.assert_array_equal(first.asymmetry, second.asymmetry)
    np.testing.assert_array_equal(first.error, second.error)
    outcome = second.metadata["correction_outcomes"]["red"]["background"]["effective"]
    assert outcome["reference"]["source_file"] == "reference-b.nxs"
    assert first.histogram_observation.identity != second.histogram_observation.identity


@pytest.mark.parametrize("nullable", [False, True, "ignored"])
def test_large_observation_snapshot_runs_on_existing_worker(qapp, monkeypatch, nullable):
    from PySide6.QtCore import QThread

    import asymmetry.core.transform.preparation as preparation
    from asymmetry.gui.utils.reduction_cache import capture_histogram_observation
    from tests.core.test_binning_entry_points import _run

    source = _run(np.full(8192, 100.0), np.full(8192, 80.0))
    if nullable == "ignored":
        from asymmetry.core.io.periods import combine_mapped_periods
        from tests.io.test_period_mapping import _four_periods

        periods = _four_periods()
        periods[3].run.histograms = source.histograms
        source = combine_mapped_periods(periods, {1: "red", 4: "ignore"}).run
    elif nullable:
        source.grouping["period_histograms"] = None
    original = preparation.histogram_observation
    worker_threads = []

    def record(*args, **kwargs):
        worker_threads.append(QThread.currentThread() != qapp.thread())
        assert kwargs["cancel_check"] is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(preparation, "histogram_observation", record)
    result = capture_histogram_observation(source, facility=None, reference=None, parent=None)
    assert worker_threads == [True]
    assert not np.shares_memory(result.source.histograms[0].counts, source.histograms[0].counts)
    assert not result.source.histograms[0].counts.flags.writeable


def test_large_observation_snapshot_cancels_without_publishing_context(qapp, monkeypatch):
    import threading

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton, QWidget

    import asymmetry.core.transform.preparation as preparation
    from asymmetry.gui.utils.reduction_cache import capture_histogram_observation
    from tests.core.test_binning_entry_points import _run

    source = _run(np.full(8192, 100.0), np.full(8192, 80.0))

    def wait_for_cancel(*args, cancel_check, **kwargs):
        while True:
            cancel_check()
            threading.Event().wait(0.005)

    def cancel():
        dialog = QApplication.activeModalWidget()
        assert dialog is not None
        dialog.findChild(QPushButton).click()

    parent = QWidget()
    parent.show()
    monkeypatch.setattr(preparation, "histogram_observation", wait_for_cancel)
    QTimer.singleShot(20, cancel)
    with pytest.raises(ValueError, match="snapshot cancelled"):
        capture_histogram_observation(source, facility=None, reference=None, parent=parent)


def test_warm_toolbar_refreshes_current_metadata_and_copies_reduction_provenance(qapp):
    from asymmetry.core.transform.preparation import wizard_observation_identity
    from tests.core.test_binning_entry_points import _run
    from tests.core.test_wizard_preparation import observed

    window = MainWindow()
    dataset = observed(_run([90, 1] * 16, [10, 9] * 16))
    first = window._counts_first_rebunched_dataset(dataset, 2)
    dataset.metadata.update(field=100.0, title="CaF2", field_direction="LF")
    second = window._counts_first_rebunched_dataset(dataset, 2)
    assert second.metadata["field"] == 100.0
    assert second.metadata["title"] == "CaF2"
    assert second.metadata["field_direction"] == "LF"
    assert wizard_observation_identity(first) != wizard_observation_identity(second)
    second.metadata["field"] = 999.0
    second.metadata["correction_outcomes"]["red"]["background"]["status"] = "edited"
    third = window._counts_first_rebunched_dataset(dataset, 2)
    assert third.metadata["field"] == 100.0
    assert third.metadata["correction_outcomes"]["red"]["background"]["status"] != "edited"
    window.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_mapping_constructs_snapshot_on_worker_and_registers_only_completed_result(
    qapp, monkeypatch, cancelled
):
    import threading

    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QApplication, QPushButton

    import asymmetry.core.io.periods as period_io
    from tests.io.test_period_mapping import _four_periods

    window = MainWindow()
    window.show()
    periods = _four_periods()
    for dataset in periods:
        window._data_browser.add_dataset(dataset)
    original = period_io.combine_mapped_periods
    worker_threads, applied = [], []

    def construct(*args, cancel_check, **kwargs):
        worker_threads.append(QThread.currentThread() != qapp.thread())
        if cancelled:
            while True:
                cancel_check()
                threading.Event().wait(0.005)
        return original(*args, cancel_check=cancel_check, **kwargs)

    def apply(dataset, grouping):
        assert QThread.currentThread() == qapp.thread()
        applied.append(dataset)

    def cancel():
        dialog = QApplication.activeModalWidget()
        assert dialog is not None
        dialog.findChild(QPushButton).click()

    monkeypatch.setattr(period_io, "combine_mapped_periods", construct)
    monkeypatch.setattr(window, "_apply_grouping_settings_to_dataset", apply)
    if cancelled:
        QTimer.singleShot(20, cancel)
    window._create_period_mapped_dataset(
        {
            "period_run_numbers": [ds.run_number for ds in periods],
            "source_run_number": 900,
            "mapping": {1: "red", 4: "ignore"},
        },
        {},
    )
    assert worker_threads == [True]
    mapped = window._data_browser.get_dataset(900)
    if cancelled:
        assert mapped is None
        assert applied == []
    else:
        assert applied == [mapped]
        assert len(mapped.histogram_observation.source.grouping["period_inputs"]) == 4
