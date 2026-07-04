"""Counts-first display bunching through the plot panel's bunch control.

When the displayed run still carries its raw histograms, the panel bunch
factor must not value-domain rebin the reduced asymmetry curve: one-sided raw
bins (F·B = 0) carry the σ = 100 % no-information sentinel, which a quadrature
merge folds into every bunched error bar on sparse data. Instead the panel's
``get_analysis_dataset`` re-runs the recorded reduction (via the MainWindow
chokepoint, deadtime/background included) with the factor multiplied onto the
grouping ``bunching_factor``, packing counts before the ratio. Histogram-less
curves (co-added data, G±R period combinations) keep the curve-level
``rebin``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.rebin import binned_fb_asymmetry, rebin
from asymmetry.core.utils.constants import MUON_LIFETIME_US, PeriodMode
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import PlotPanel

BIN_WIDTH_US = 0.016


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
        counts_f,
        counts_b,
        grouping=grouping,
        common_t0=0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.0,
        first_good_bin=0,
        last_good_bin=n - 1,
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
        time_ref, _, err_old = rebin(dataset.time, dataset.asymmetry, dataset.error, factor)
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

    def test_period_combined_and_nonfixed_runs_fall_back_to_curve_rebin(
        self, mainwindow: MainWindow
    ) -> None:
        """G±R combinations and non-fixed binning keep the value-domain rebin."""
        counts_f, counts_b = _sparse_counts(seed=29)
        gr_dataset = _histogram_run_dataset(
            counts_f,
            counts_b,
            grouping_extra={
                "period_mode": str(PeriodMode.GREEN_MINUS_RED),
                "period_histograms": [[1], [2]],
            },
        )
        factor = 5
        assert mainwindow._counts_first_rebunched_arrays(gr_dataset, factor) is None

        mainwindow._plot_panel.set_bunch_factor(factor, emit_signal=False)
        analysis = mainwindow._plot_panel.get_analysis_dataset(gr_dataset)
        t_ref, a_ref, e_ref = rebin(gr_dataset.time, gr_dataset.asymmetry, gr_dataset.error, factor)
        np.testing.assert_allclose(analysis.time, t_ref)
        np.testing.assert_allclose(analysis.asymmetry, a_ref)
        np.testing.assert_allclose(analysis.error, e_ref)

        variable_dataset = _histogram_run_dataset(
            counts_f, counts_b, grouping_extra={"binning_mode": "variable"}
        )
        assert mainwindow._counts_first_rebunched_arrays(variable_dataset, factor) is None


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
        t_ref, a_ref, e_ref = rebin(dataset.time, dataset.asymmetry, dataset.error, 3)
        np.testing.assert_allclose(analysis.time, t_ref)
        np.testing.assert_allclose(analysis.asymmetry, a_ref)
        np.testing.assert_allclose(analysis.error, e_ref)

    def test_provider_none_result_falls_back_to_rebin(self, qapp: QApplication) -> None:
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
        analysis = panel.get_analysis_dataset(dataset)

        t_ref, a_ref, e_ref = rebin(dataset.time, dataset.asymmetry, dataset.error, 4)
        np.testing.assert_allclose(analysis.time, t_ref)
        np.testing.assert_allclose(analysis.asymmetry, a_ref)
        np.testing.assert_allclose(analysis.error, e_ref)

    def test_provider_arrays_are_used_verbatim(self, qapp: QApplication) -> None:
        panel = PlotPanel()
        run = Run(
            run_number=7,
            histograms=[Histogram(counts=np.ones(12), bin_width=BIN_WIDTH_US)],
            metadata={},
            grouping={},
        )
        dataset = self._curve_dataset(run=run)
        provided = (
            np.array([0.0, 1.0]),
            np.array([10.0, 9.0]),
            np.array([0.5, 0.6]),
        )
        seen: list[int] = []

        def provider(ds: MuonDataset, factor: int):
            seen.append(factor)
            return provided

        panel.set_counts_rebunch_provider(provider)
        panel.set_bunch_factor(6, emit_signal=False)
        analysis = panel.get_analysis_dataset(dataset)

        assert seen == [6]
        np.testing.assert_allclose(analysis.time, provided[0])
        np.testing.assert_allclose(analysis.asymmetry, provided[1])
        np.testing.assert_allclose(analysis.error, provided[2])
        assert analysis.run is run
