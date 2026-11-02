"""Display bunching replays counts for every binning and period mode.

Corrected counts and their shared errors are packed before the ratio. Curves
without raw histograms and rotated projections use independent curve means.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.counts import aligned_counts

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.rebin import binned_fb_asymmetry, binning_spec, merge_curve_bins
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import PlotPanel

BIN_WIDTH_US = 0.016


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    """Create a mainwindow for testing."""
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _histogram_run_dataset(
    counts_f: np.ndarray,
    counts_b: np.ndarray,
    *,
    run_number: int = 4242,
    base_bunch: int = 1,
    grouping_extra: dict | None = None,
) -> MuonDataset:
    """A run with two raw histograms and its recorded (counts-first) reduction."""
    n = counts_f.size
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": n - 1,
        "bunching_factor": base_bunch,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if grouping_extra:
        grouping.update(grouping_extra)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts_f.copy(), bin_width=BIN_WIDTH_US, t0_bin=0),
            Histogram(counts=counts_b.copy(), bin_width=BIN_WIDTH_US, t0_bin=0),
        ],
        metadata={"run_number": run_number},
        grouping=grouping,
    )
    time, asym, err = binned_fb_asymmetry(
        aligned_counts(
            counts_f,
            counts_b,
            common_t0=0,
            bin_width_us=BIN_WIDTH_US,
            first_good_bin=0,
            last_good_bin=n - 1,
        ),
        binning=binning_spec(grouping),
        alpha=1.0,
    )
    return MuonDataset(
        time=time,
        asymmetry=asym * 100.0,
        error=err * 100.0,
        metadata={"run_number": run_number},
        run=run,
    )


def _sparse_counts(n: int = 1200, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Sparse Poisson traces with plenty of one-sided (F·B = 0) raw bins."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * BIN_WIDTH_US
    decay = np.exp(-t / MUON_LIFETIME_US)
    counts_f = rng.poisson(np.clip(3.0 * decay, 0.0, None)).astype(float)
    counts_b = rng.poisson(np.clip(2.5 * decay, 0.0, None)).astype(float)
    assert np.count_nonzero(counts_f * counts_b == 0) > 50
    return counts_f, counts_b


def _packed_reference(
    counts_f: np.ndarray, counts_b: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Percent-scale counts-first packing: sum blocks, then one asymmetry."""
    trimmed = (counts_f.size // factor) * factor
    f_packed = counts_f[:trimmed].reshape(-1, factor).sum(axis=1)
    b_packed = counts_b[:trimmed].reshape(-1, factor).sum(axis=1)
    asym, err = compute_asymmetry(f_packed, b_packed, alpha=1.0)
    return f_packed, b_packed, asym * 100.0, err * 100.0


class TestPanelBunchCountsFirst:
    def test_bunched_display_matches_counts_first_packing_on_sparse_run(
        self, mainwindow: MainWindow
    ) -> None:
        """Panel bunching on a histogram-backed sparse run packs counts, not values."""
        counts_f, counts_b = _sparse_counts()
        dataset = _histogram_run_dataset(counts_f, counts_b)
        factor = 8

        mainwindow._plot_panel.set_bunch_factor(factor, emit_signal=False)
        analysis = mainwindow._plot_panel.get_analysis_dataset(dataset)

        f_packed, b_packed, asym_ref, err_ref = _packed_reference(counts_f, counts_b, factor)
        np.testing.assert_allclose(analysis.asymmetry, asym_ref)
        np.testing.assert_allclose(analysis.error, err_ref)

        # The time axis must not move relative to the old value-domain path.
        time_ref, _, err_old = merge_curve_bins(
            dataset.time, dataset.asymmetry, dataset.error, factor
        )
        np.testing.assert_allclose(analysis.time, time_ref)

        # And the value-domain merge is strictly noisier here: the σ = 100 %
        # sentinels of one-sided raw bins inflate its quadrature.
        populated = (f_packed > 0) & (b_packed > 0)
        assert np.median(err_old[populated] / analysis.error[populated]) > 1.05

        # Provenance: the analysis dataset is a copy tied to the same run.
        assert analysis is not dataset
        assert analysis.run is dataset.run

    def test_panel_factor_multiplies_grouping_bunching_factor(self, mainwindow: MainWindow) -> None:
        """Effective packing = grouping ``bunching_factor`` × panel factor."""
        counts_f, counts_b = _sparse_counts(seed=17)
        dataset = _histogram_run_dataset(counts_f, counts_b, base_bunch=2)

        mainwindow._plot_panel.set_bunch_factor(3, emit_signal=False)
        analysis = mainwindow._plot_panel.get_analysis_dataset(dataset)

        _, _, asym_ref, err_ref = _packed_reference(counts_f, counts_b, 6)
        np.testing.assert_allclose(analysis.asymmetry, asym_ref)
        np.testing.assert_allclose(analysis.error, err_ref)
        assert analysis.time.size == counts_f.size // 6

    def test_recorded_corrections_flow_through_rebunch(self, mainwindow: MainWindow) -> None:
        """Panel bunching equals the grouping chokepoint re-reduce — deadtime included."""
        rng = np.random.default_rng(23)
        n = 400
        counts_f = rng.poisson(500.0, n).astype(float)
        counts_b = rng.poisson(420.0, n).astype(float)
        dataset = _histogram_run_dataset(
            counts_f,
            counts_b,
            grouping_extra={
                "deadtime_correction": True,
                "deadtime_mode": "manual",
                "dead_time_us": [0.01, 0.0],
                "good_frames": 25000.0,
            },
        )
        factor = 4

        mainwindow._plot_panel.set_bunch_factor(factor, emit_signal=False)
        analysis = mainwindow._plot_panel.get_analysis_dataset(dataset)

        # The deadtime correction must actually bite: the result differs from
        # a raw-counts packing (which would mean the correction was bypassed).
        _, _, asym_plain, _ = _packed_reference(counts_f, counts_b, factor)
        assert not np.allclose(analysis.asymmetry, asym_plain, rtol=1e-4)

        # And it matches the normal grouping re-reduce at the same effective
        # factor — the same chokepoint, so all recorded corrections apply.
        twin = MuonDataset(
            time=dataset.time.copy(),
            asymmetry=dataset.asymmetry.copy(),
            error=dataset.error.copy(),
            metadata=dict(dataset.metadata),
            run=dataset.run,
        )
        payload = dict(dataset.run.grouping)
        payload["bunching_factor"] = factor
        applied, _ = mainwindow._apply_grouping_settings_to_dataset(twin, payload)
        assert applied is True
        np.testing.assert_allclose(analysis.time, twin.time)
        np.testing.assert_allclose(analysis.asymmetry, twin.asymmetry)
        np.testing.assert_allclose(analysis.error, twin.error)


class TestPanelProviderContract:
    """Panel-side contract, no MainWindow: the hook gates on raw histograms."""

    def _curve_dataset(self, run: Run | None = None) -> MuonDataset:
        t = np.arange(12, dtype=float) * BIN_WIDTH_US
        return MuonDataset(
            time=t,
            asymmetry=np.linspace(20.0, 8.0, t.size),
            error=np.full(t.size, 1.5),
            metadata={"run_number": 7},
            run=run,
        )

    def test_histogram_less_dataset_never_calls_provider(self, qapp: QApplication) -> None:
        panel = PlotPanel()
        calls: list[tuple[int, int]] = []
        panel.set_counts_rebunch_provider(
            lambda ds, factor: calls.append((int(ds.metadata["run_number"]), factor))
        )
        dataset = self._curve_dataset(run=None)

        panel.set_bunch_factor(3, emit_signal=False)
        analysis = panel.get_analysis_dataset(dataset)

        assert calls == []
        t_ref, a_ref, e_ref = merge_curve_bins(dataset.time, dataset.asymmetry, dataset.error, 3)
        np.testing.assert_allclose(analysis.time, t_ref)
        np.testing.assert_allclose(analysis.asymmetry, a_ref)
        np.testing.assert_allclose(analysis.error, e_ref)

    def test_provider_none_result_raises_a_named_error(self, qapp: QApplication) -> None:
        panel = PlotPanel()
        run = Run(
            run_number=7,
            histograms=[Histogram(counts=np.ones(12), bin_width=BIN_WIDTH_US)],
            metadata={},
            grouping={},
        )
        dataset = self._curve_dataset(run=run)
        panel.set_counts_rebunch_provider(lambda ds, factor: None)

        panel.set_bunch_factor(4, emit_signal=False)
        with pytest.raises(ValueError, match="provider must return"):
            panel.get_analysis_dataset(dataset)


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_nonfixed_display_merges_recorded_count_bins(mainwindow, mode):
    f, b = np.arange(1.0, 12.0), np.arange(21.0, 32.0)
    dataset = _histogram_run_dataset(
        f, b, grouping_extra={"binning_mode": mode, "bin0_us": 0.04, "bin10_us": 0.04}
    )
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._on_dataset_selected(dataset.run_number)
    mainwindow._view_bunch_spin.setValue(2)
    actual = mainwindow._plot_panel.get_analysis_dataset(dataset)
    forward = np.array([f[:6].sum(), f[6:].sum()])
    backward = np.array([b[:6].sum(), b[6:].sum()])
    expected = 100 * (forward - backward) / (forward + backward)
    error = 200 * np.sqrt(backward**2 * forward + forward**2 * backward) / (forward + backward) ** 2
    np.testing.assert_allclose(actual.time, [0.04, 0.128], rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual.asymmetry, expected, rtol=1e-12)
    np.testing.assert_allclose(actual.error, error, rtol=1e-12)


@pytest.mark.parametrize("mode,sign", [("green_minus_red", -1), ("green_plus_red", 1)])
def test_period_display_bunches_each_period_before_combining(mainwindow, mode, sign):
    red_f, red_b = np.array([90.0, 1, 30, 2]), np.array([10.0, 9, 20, 8])
    green_f, green_b = np.array([2.0, 80, 10, 2]), np.array([8.0, 10, 30, 8])
    dataset = _histogram_run_dataset(red_f, red_b)
    green = _histogram_run_dataset(green_f, green_b, run_number=4243)
    dataset.run.grouping.update(
        period_mode=mode, period_histograms=[dataset.run.histograms, green.run.histograms]
    )
    dataset.asymmetry = green.asymmetry + sign * dataset.asymmetry
    dataset.error = np.sqrt(green.error**2 + dataset.error**2)
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._on_dataset_selected(dataset.run_number)
    mainwindow._view_bunch_spin.setValue(2)
    actual = mainwindow._plot_panel.get_analysis_dataset(dataset)
    _, _, ra, re = _packed_reference(red_f, red_b, 2)
    _, _, ga, ge = _packed_reference(green_f, green_b, 2)
    np.testing.assert_allclose(actual.asymmetry, ga + sign * ra, rtol=1e-12)
    np.testing.assert_allclose(actual.error, np.sqrt(ge**2 + re**2), rtol=1e-12)


def test_bunching_larger_than_the_record_returns_no_points(mainwindow):
    dataset = _histogram_run_dataset(np.array([90.0, 1]), np.array([10.0, 9]))
    mainwindow._plot_panel.set_bunch_factor(8, emit_signal=False)
    assert mainwindow._plot_panel.get_analysis_dataset(dataset).time.size == 0


def test_toolbar_merge_also_sums_raw_group_counts(mainwindow):
    dataset = _histogram_run_dataset(np.array([90.0, 1, 30, 2]), np.array([10.0, 9, 20, 8]))
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._on_dataset_selected(dataset.run_number)
    mainwindow._view_bunch_spin.setValue(2)
    groups = mainwindow._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
    by_group = {item.metadata["group_id"]: item for item in groups}
    np.testing.assert_array_equal(by_group[1].asymmetry, [91.0, 32.0])
    np.testing.assert_array_equal(by_group[2].asymmetry, [19.0, 28.0])
    assert dataset.run.grouping["bunching_factor"] == 1
