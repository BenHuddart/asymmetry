"""Histogram grid edits invalidate both cached count views and recorded spectra."""

import numpy as np
import pytest

from asymmetry.gui.mainwindow import MainWindow
from tests._qt_helpers import wait_for
from tests.gui import test_fourier_staleness_gui as staleness_helpers
from tests.gui.test_reduction_cache import _count_builds, _multi_group_dataset

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("field", ["t0_bin", "t0_time_us", "bin_width"])
def test_grid_edit_refreshes_cold_warm_grouped_count_views(qapp, monkeypatch, field):
    dataset = _multi_group_dataset(
        grouping_extra={
            "t0_bin": 20,
            "first_good_bin": 20,
            "last_good_bin": 350,
            "background_correction": True,
            "background_mode": "range",
            "background_range": [0, 5],
        }
    )
    for histogram in dataset.run.histograms:
        histogram.t0_bin = 20
    calls = _count_builds(monkeypatch)
    window = MainWindow()
    try:
        first = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        warm = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        assert calls["n"] == 1
        np.testing.assert_array_equal(warm[0].time, first[0].time)
        for histogram in dataset.run.histograms:
            value = {
                "t0_bin": 21,
                "t0_time_us": 20.5 * histogram.bin_width + 0.001,
                "bin_width": histogram.bin_width * 2,
            }[field]
            setattr(histogram, field, value)
        changed = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        changed_warm = window._grouped_time_domain_display_datasets(
            dataset, lifetime_corrected=False
        )
        assert calls["n"] == 2
        assert not (
            np.array_equal(changed[0].time, first[0].time)
            and np.array_equal(changed[0].asymmetry, first[0].asymmetry)
        )
        np.testing.assert_array_equal(changed_warm[0].time, changed[0].time)
        np.testing.assert_array_equal(changed_warm[0].asymmetry, changed[0].asymmetry)
    finally:
        window.close()


@pytest.mark.parametrize("field", ["t0_bin", "t0_time_us", "bin_width"])
def test_grid_edit_marks_computed_range_spectrum_stale_then_refreshes(qapp, monkeypatch, field):
    run = staleness_helpers._tf_run()
    for histogram in run.histograms:
        histogram.t0_bin = 20
    run.grouping.update(
        first_good_bin=20,
        last_good_bin=450,
        background_correction=True,
        background_mode="range",
        background_range=[0, 5],
    )
    monkeypatch.setattr(staleness_helpers, "_tf_run", lambda: run)
    window = MainWindow()
    try:
        staleness_helpers._computed_window(window)
        assert not window._fourier_staleness_for_current_run()[0]
        for histogram in run.histograms:
            value = {
                "t0_bin": 21,
                "t0_time_us": 20.5 * histogram.bin_width + 0.001,
                "bin_width": histogram.bin_width * 2,
            }[field]
            setattr(histogram, field, value)
        stale, reason = window._fourier_staleness_for_current_run()
        assert stale
        assert "grouping changed" in reason
        window._on_compute_fourier()
        wait_for(lambda: not window._fourier_compute_active, qapp, timeout_s=15)
        assert not window._fourier_staleness_for_current_run()[0]
    finally:
        window.close()
