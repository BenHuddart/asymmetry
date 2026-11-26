"""Display bunching replays counts for every binning and period mode.

Corrected counts and their shared errors are packed before the ratio. Curves
without raw histograms and rotated projections use independent curve means.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.counts import aligned_counts
from tests.gui.gle_helpers import wait_for_gle

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication, QFileDialog  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.preparation import histogram_observation
from asymmetry.core.transform.rebin import binned_fb_asymmetry, binning_spec, merge_curve_bins
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.utils import gle_export
from asymmetry.gui.utils.errorbar_dots import errorbar_dots_line_class

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
        histogram_observation=histogram_observation(run),
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

    def test_custom_curve_with_histograms_keeps_curve_means(self, qapp: QApplication) -> None:
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
        actual = panel.get_analysis_dataset(dataset)
        expected = dataset.merge_curve_bins(4)
        np.testing.assert_array_equal(actual.asymmetry, expected.asymmetry)


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
    dataset.histogram_observation = histogram_observation(dataset.run)
    dataset.asymmetry = green.asymmetry + sign * dataset.asymmetry
    dataset.error = np.sqrt(green.error**2 + dataset.error**2)
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._on_dataset_selected(dataset.run_number)
    mainwindow._view_bunch_spin.setValue(2)
    actual = mainwindow._plot_panel.get_analysis_dataset(dataset)
    _, _, ra, re = _packed_reference(red_f, red_b, 2)
    _, _, ga, ge = _packed_reference(green_f, green_b, 2)
    if mode == "green_plus_red":
        _, _, a, e = _packed_reference(red_f + green_f, red_b + green_b, 2)
    else:
        a, e = ga - ra, np.hypot(ge, re)
    np.testing.assert_allclose(actual.asymmetry, a, rtol=1e-12)
    np.testing.assert_allclose(actual.error, e, rtol=1e-12)


def test_bunching_larger_than_the_record_returns_no_points(mainwindow):
    dataset = _histogram_run_dataset(np.array([90.0, 1]), np.array([10.0, 9]))
    mainwindow._plot_panel.set_bunch_factor(8, emit_signal=False)
    assert mainwindow._plot_panel.get_analysis_dataset(dataset).time.size == 0


@pytest.mark.parametrize("view", ["raw_counts", "groups"])
@pytest.mark.parametrize("base_bunch,bunch", [(1, 2), (1, 1), (3, 2)])
def test_grouped_toolbar_bunch_plot_and_exports(
    mainwindow, tmp_path, monkeypatch, view, base_bunch, bunch
):
    forward = np.arange(1.0, 257.0)
    backward = 2 * forward + 11
    names = {1: "Upstream", 2: "Downstream"}
    dataset = _histogram_run_dataset(
        forward, backward, base_bunch=base_bunch, grouping_extra={"group_names": names}
    )
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({dataset.run_number})
    if view == "groups":
        mainwindow._domain_buttons_by_token["groups"].click()
    else:
        mainwindow._raw_counts_action.trigger()
    assert mainwindow._plot_workspace.active_view() == view
    if bunch == 1:
        mainwindow._view_bunch_spin.setValue(2)
    mainwindow._view_bunch_spin.setValue(bunch)
    assert dataset.run.grouping["bunching_factor"] == base_bunch

    factor = base_bunch * bunch
    usable = forward.size // factor * factor
    time = (np.arange(0, usable, factor) + (factor - 1) / 2) * BIN_WIDTH_US
    quantity = "Lifetime-corrected counts" if view == "groups" else "Counts"
    scale = np.exp(time / MUON_LIFETIME_US) if view == "groups" else 1
    expected = {}
    panel = mainwindow._plot_panel
    for axis, name, counts in zip(
        panel._figure.axes, names.values(), [forward, backward], strict=True
    ):
        summed = counts[:usable].reshape(-1, factor).sum(axis=1)
        values, errors = summed * scale, np.sqrt(summed) * scale
        expected[name] = np.column_stack((time, values, errors))
        assert axis.get_title(loc="left") == name
        assert axis.get_ylabel() == quantity
        (markers,) = [line for line in axis.lines if isinstance(line, errorbar_dots_line_class())]
        np.testing.assert_allclose(markers.get_xdata(), time, atol=1e-15)
        np.testing.assert_allclose(markers.get_ydata(), values)
        (path,) = markers.bar_line.get_paths()
        bounds = path.vertices.reshape(-1, 3, 2)[:, :2]
        np.testing.assert_allclose(bounds[..., 0], np.repeat(time[:, None], 2, axis=1), atol=1e-15)
        np.testing.assert_allclose(
            bounds[..., 1], np.column_stack((values - errors, values + errors))
        )
    assert panel._figure.axes[-1].get_xlabel() == "Time (µs)"

    text_dir = tmp_path / "text"
    text_dir.mkdir()
    monkeypatch.setattr(panel, "_prompt_text_export_options", lambda: ("data", False))
    monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *_args: None)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(text_dir))
    panel.export_plotted_data_as_text()

    gle_dir = tmp_path / "counts.gleplot"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(gle_dir), ""))
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    panel.export_plots_to_gle()
    wait_for_gle(panel)
    script = (gle_dir / "counts.gle").read_text()
    assert script.count(f'ytitle "{quantity}"') == 2
    for name in names.values():
        assert f'title "{name}"' in script
    gle_y_limits = [
        (float(low), float(high))
        for low, high in re.findall(r"^\s*yaxis min (\S+) max (\S+)", script, re.MULTILINE)
    ]
    np.testing.assert_allclose(
        gle_y_limits, [ax.get_ylim() for ax in panel._figure.axes], rtol=5e-6
    )

    for directory in (text_dir, gle_dir):
        files = sorted(directory.glob("*.dat"))
        assert len(files) == len(names)
        for path in files:
            header = path.read_text()
            group_id, name = next(
                (gid, name) for gid, name in names.items() if name.lower() in path.stem.lower()
            )
            assert f"!  Run number  : {dataset.run_number}\n" in header
            assert f"!  Detector group: {name} (ID {group_id})\n" in header
            assert f"!  Datarow: (Time) ({quantity}) (Error)\n" in header
            assert f"!  Ylabel: {quantity}\n" in header
            assert f"!  Display bunching factor: {factor}\n" in header
            assert "!  Deadtime correction off\n" in header
            assert "asymmetry" not in header.lower()
            np.testing.assert_allclose(np.loadtxt(path, comments="!"), expected[name])
    assert dataset.run.grouping["bunching_factor"] == base_bunch


@pytest.mark.parametrize(
    "settings,status",
    [
        (
            {"deadtime_mode": "manual", "dead_time_us": [0.01, 0.01], "good_frames": 25000},
            "applied",
        ),
        ({"deadtime_mode": "from_file", "deadtime_file_us": []}, "skipped"),
    ],
)
def test_grouped_export_reports_prepared_deadtime(mainwindow, tmp_path, settings, status):
    dataset = _histogram_run_dataset(
        np.arange(1.0, 65.0),
        np.arange(101.0, 165.0),
        grouping_extra={"deadtime_correction": True, **settings},
    )
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({dataset.run_number})
    mainwindow._raw_counts_action.trigger()
    mainwindow._view_bunch_spin.setValue(2)

    panel = mainwindow._plot_panel
    payload = panel.get_current_plot_export_data()[0]
    path = tmp_path / "counts.dat"
    panel._write_data_file(path, payload)
    assert f"!  Deadtime correction {status}\n" in path.read_text()


@pytest.mark.parametrize("complete", [False, True], ids=["preview", "completed"])
def test_grouped_fit_overlays_keep_the_toolbar_bins(mainwindow, complete):
    time = np.arange(256) * BIN_WIDTH_US
    field = 200.0
    frequency = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * field
    decay = np.exp(-time / MUON_LIFETIME_US)
    counts = [
        np.rint(n0 * (1 + 0.12 * np.cos(2 * np.pi * frequency * time + phase)) * decay)
        for n0, phase in [(6000, 0.0), (5000, np.pi)]
    ]
    dataset = _histogram_run_dataset(*counts)
    dataset.run.metadata["field"] = field
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({dataset.run_number})
    mainwindow._domain_buttons_by_token["groups"].click()
    mainwindow._view_bunch_spin.setValue(2)

    tab = mainwindow._multi_group_fit_window._single_fit_tab
    signal = tab.grouped_fit_completed if complete else tab.grouped_preview_requested
    emitted = []
    signal.connect(lambda _datasets, curves: emitted.append(curves))
    button = tab._fit_btn if complete else tab._preview_btn
    assert button.isEnabled()
    button.click()
    if complete:
        assert tab.wait_for_fit()
    (base_curves,) = emitted
    if complete:
        assert all(result.success for result, *_ in base_curves.values())

    panel = mainwindow._plot_panel
    centres = time.reshape(-1, 2).mean(axis=1)
    correction = np.exp(centres / MUON_LIFETIME_US)
    fit_label = "Grouped Fit" if complete else "Grouped Preview"
    for group, axis, raw, exported in zip(
        build_grouped_time_domain_datasets(dataset),
        panel._figure.axes,
        counts,
        panel.get_current_plot_export_data(),
        strict=True,
    ):
        (markers,) = [line for line in axis.lines if isinstance(line, errorbar_dots_line_class())]
        np.testing.assert_allclose(markers.get_xdata(), centres, atol=1e-15)
        np.testing.assert_allclose(markers.get_ydata(), raw.reshape(-1, 2).sum(axis=1) * correction)

        native_t, native_y = base_curves[group.run_number][1]
        pairs = time.reshape(-1, 2)
        covered = (pairs[:, 0] >= native_t[0] - 1e-12) & (pairs[:, 1] <= native_t[-1] + 1e-12)
        native_counts = np.interp(time, native_t, native_y) * decay
        prediction = native_counts.reshape(-1, 2).sum(axis=1) * correction
        (line,) = [
            line
            for line, _, name in panel._time_view.fits
            if line.axes is axis and name.endswith(fit_label)
        ]
        np.testing.assert_allclose(line.get_xdata(), centres[covered], atol=1e-15)
        np.testing.assert_allclose(line.get_ydata(), prediction[covered])
        np.testing.assert_allclose(exported["fit"]["t"], centres[covered], atol=1e-15)
        np.testing.assert_allclose(exported["fit"]["y"], prediction[covered])
        assert ("chi_squared" in exported["fit_metadata"]) == complete

    mainwindow._view_bunch_spin.setValue(1)
    for exported, (_, (native_t, native_y), _) in zip(
        panel.get_current_plot_export_data(), base_curves.values(), strict=True
    ):
        np.testing.assert_array_equal(exported["fit"]["t"], native_t)
        np.testing.assert_array_equal(exported["fit"]["y"], native_y)


def test_fb_toolbar_bunch_plot_and_text_export(mainwindow, tmp_path, monkeypatch):
    forward = np.arange(1.0, 257.0)
    backward = 2 * forward + 11
    dataset = _histogram_run_dataset(forward, backward)
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({dataset.run_number})
    mainwindow._view_bunch_spin.setValue(2)
    time = (np.arange(0, forward.size, 2) + 0.5) * BIN_WIDTH_US
    _, _, values, errors = _packed_reference(forward, backward, 2)

    panel = mainwindow._plot_panel
    (markers,) = [line for line in panel._ax.lines if isinstance(line, errorbar_dots_line_class())]
    np.testing.assert_allclose(markers.get_xdata(), time)
    np.testing.assert_allclose(markers.get_ydata(), values)

    target = tmp_path / "asymmetry.dat"
    monkeypatch.setattr(panel, "_prompt_text_export_options", lambda: ("data", False))
    monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *_args: None)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(target), ""))
    panel.export_plotted_data_as_text()
    header = target.read_text()
    assert "!  Datarow: (Time) (FB asymmetry) (Error)\n" in header
    assert "!  Ylabel: % Asymmetry\n" in header
    np.testing.assert_allclose(
        np.loadtxt(target, comments="!"), np.column_stack((time, values, errors))
    )
    assert dataset.run.grouping["bunching_factor"] == 1


@pytest.mark.parametrize("view", ["groups", "raw_counts"])
def test_reference_count_views_show_diagnostic_and_recover(mainwindow, view):
    dataset = _histogram_run_dataset(np.full(32, 200.0), np.full(32, 150.0))
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._on_dataset_selected(dataset.run_number)
    mainwindow._plot_panel.set_time_view_modes(["fb_asymmetry", "groups", "raw_counts"], view)
    mainwindow._render_current_selection_plot()
    assert len(mainwindow._plot_panel._grouped_time_subplot_datasets) == 2

    dataset.run.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 42},
    )
    mainwindow._on_plot_time_view_changed(view)
    message = " ".join(text.get_text() for text in mainwindow._plot_panel._ax.texts)
    assert "reference-run subtraction is unsupported" in message
    assert "Disable background correction or choose a non-reference background mode" in message
    assert not mainwindow._plot_panel._grouped_time_subplot_datasets
    assert dataset.run.grouping["background_mode"] == "reference_run"
    np.testing.assert_array_equal(dataset.run.histograms[0].counts, 200.0)
    np.testing.assert_array_equal(dataset.run.histograms[1].counts, 150.0)

    dataset.run.grouping["background_correction"] = False
    mainwindow._render_current_selection_plot()
    groups = mainwindow._plot_panel._grouped_time_subplot_datasets
    assert len(groups) == 2
    assert not any("unsupported" in text.get_text() for text in mainwindow._plot_panel._ax.texts)
