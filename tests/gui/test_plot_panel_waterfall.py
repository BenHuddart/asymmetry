"""Tests for PlotPanel's waterfall overlay stacking (single-axis overlay path)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.utils.waterfall import auto_waterfall_delta


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def time_panel(qapp: QApplication) -> PlotPanel:
    panel = PlotPanel(domain="time")
    yield panel
    panel.close()
    panel.deleteLater()


@pytest.fixture
def freq_panel(qapp: QApplication) -> PlotPanel:
    panel = PlotPanel(domain="frequency")
    yield panel
    panel.close()
    panel.deleteLater()


def _time_dataset(run_number: int, amplitude: float) -> MuonDataset:
    t = np.linspace(0.0, 10.0, 200)
    a = amplitude * np.exp(-0.3 * t) * np.cos(2.0 * np.pi * 1.0 * t)
    e = np.full_like(t, 0.005)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": run_number})


def _freq_dataset(run_number: int, peak: float) -> MuonDataset:
    f = np.linspace(0.0, 5.0, 200)
    y = peak * np.exp(-(((f - 1.0) / 0.15) ** 2))
    e = np.full_like(f, 0.01)
    return MuonDataset(
        time=f,
        asymmetry=y,
        error=e,
        metadata={
            "run_number": run_number,
            "plot_domain": "frequency",
            "x_label": "Frequency (MHz)",
            "y_label": "|F| (arb.)",
        },
    )


def _spy_errorbar(panel: PlotPanel) -> list[np.ndarray]:
    """Record the asymmetry array handed to each main errorbar draw."""
    calls: list[np.ndarray] = []
    original = panel._plot_errorbar_masked

    def _record(ax, time, asymmetry, error, mask, **kwargs):
        if kwargs.get("color") not in ("0.6",):  # skip the grey low-count series
            calls.append(np.asarray(asymmetry, dtype=float).copy())
        return original(ax, time, asymmetry, error, mask, **kwargs)

    panel._plot_errorbar_masked = _record  # type: ignore[assignment]
    return calls


def _spy_frequency(panel: PlotPanel) -> list[np.ndarray]:
    calls: list[np.ndarray] = []
    original = panel._plot_frequency_line_masked

    def _record(ax, freq, values, error, mask, **kwargs):
        calls.append(np.asarray(values, dtype=float).copy())
        return original(ax, freq, values, error, mask, **kwargs)

    panel._plot_frequency_line_masked = _record  # type: ignore[assignment]
    return calls


def _baseline_count(panel: PlotPanel) -> int:
    """Count the faint waterfall baseline hairlines (linewidth 0.6)."""
    return sum(
        1
        for child in panel._ax.get_children()
        if isinstance(child, Line2D) and abs(child.get_linewidth() - 0.6) < 1e-9
    )


class TestWaterfallOffsets:
    def test_auto_offsets_applied_per_trace(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2), _time_dataset(3, 0.2)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)  # auto spacing

        calls = _spy_errorbar(time_panel)
        time_panel.plot_datasets(datasets)

        assert len(calls) == 3
        delta = auto_waterfall_delta([ds.asymmetry for ds in datasets])
        assert time_panel._waterfall_resolved_delta == pytest.approx(delta)
        for i, ds in enumerate(datasets):
            np.testing.assert_allclose(calls[i], ds.asymmetry + i * delta)

    def test_manual_offset_overrides_auto(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.set_waterfall_offset(5.0)

        calls = _spy_errorbar(time_panel)
        time_panel.plot_datasets(datasets)

        assert time_panel._waterfall_resolved_delta == pytest.approx(5.0)
        np.testing.assert_allclose(calls[0], datasets[0].asymmetry)
        np.testing.assert_allclose(calls[1], datasets[1].asymmetry + 5.0)

    def test_no_offset_when_waterfall_disabled(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2)]
        time_panel.set_overlay_enabled(True)
        # waterfall left off

        calls = _spy_errorbar(time_panel)
        time_panel.plot_datasets(datasets)

        assert time_panel._waterfall_resolved_delta == 0.0
        for ds, drawn in zip(datasets, calls):
            np.testing.assert_allclose(drawn, ds.asymmetry)

    def test_source_arrays_not_mutated(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2)]
        originals = [ds.asymmetry.copy() for ds in datasets]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)
        for ds, original in zip(datasets, originals):
            np.testing.assert_array_equal(ds.asymmetry, original)

    def test_single_dataset_ignores_waterfall(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        # A one-element list delegates to plot_dataset (no overlay stacking).
        time_panel.plot_datasets([_time_dataset(1, 0.2)])
        assert time_panel._waterfall_resolved_delta == 0.0
        assert _baseline_count(time_panel) == 0

    def test_fit_curve_shares_dataset_offset(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2)]
        t_fit = np.linspace(0.0, 10.0, 50)
        y_fit = 0.15 * np.exp(-0.3 * t_fit)
        time_panel._fit_curves[2] = (t_fit, y_fit, "Fit")
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)

        delta = time_panel._waterfall_resolved_delta
        # The fit is drawn at linewidth 2; its ydata must carry the trace offset.
        fit_lines = [
            ln
            for ln in time_panel._ax.get_children()
            if isinstance(ln, Line2D) and abs(ln.get_linewidth() - 2.0) < 1e-9
        ]
        assert fit_lines
        drawn_fit = fit_lines[0].get_ydata()
        np.testing.assert_allclose(drawn_fit, y_fit + delta)


class TestWaterfallFraming:
    def test_auto_y_covers_top_trace(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2), _time_dataset(3, 0.2)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)

        delta = time_panel._waterfall_resolved_delta
        top_baseline = (len(datasets) - 1) * delta
        # The top trace's baseline must sit inside the framed y-range.
        assert time_panel._y_max.value() >= top_baseline
        # And waterfall visibly expands the range beyond the un-stacked signal.
        assert time_panel._y_max.value() > 0.25


class TestWaterfallBaselines:
    def test_baselines_drawn_in_time_domain(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2), _time_dataset(3, 0.2)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)
        assert _baseline_count(time_panel) == 3

    def test_no_baselines_when_waterfall_off(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 0.2), _time_dataset(2, 0.2)]
        time_panel.set_overlay_enabled(True)
        time_panel.plot_datasets(datasets)
        assert _baseline_count(time_panel) == 0

    def test_frequency_domain_has_no_baselines(self, freq_panel: PlotPanel) -> None:
        if not freq_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_freq_dataset(1, 1.0), _freq_dataset(2, 1.0)]
        freq_panel.set_overlay_enabled(True)
        freq_panel.set_waterfall_enabled(True)

        calls = _spy_frequency(freq_panel)
        freq_panel.plot_datasets(datasets)

        # Spectra are stacked but carry no baseline hairlines.
        assert _baseline_count(freq_panel) == 0
        delta = freq_panel._waterfall_resolved_delta
        assert delta > 0.0
        np.testing.assert_allclose(calls[0], datasets[0].asymmetry)
        np.testing.assert_allclose(calls[1], datasets[1].asymmetry + delta)


class TestWaterfallControlCoupling:
    def test_waterfall_disabled_until_overlay_on(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        # Overlay starts off -> the waterfall checkbox is disabled.
        assert time_panel._waterfall_checkbox.isEnabled() is False
        time_panel.set_overlay_enabled(True)
        assert time_panel._waterfall_checkbox.isEnabled() is True

    def test_turning_overlay_off_unchecks_waterfall(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        assert time_panel.is_waterfall_enabled() is True

        time_panel.set_overlay_enabled(False)
        assert time_panel.is_waterfall_enabled() is False
        assert time_panel._waterfall_checkbox.isEnabled() is False
        assert time_panel._waterfall_delta_field.isEnabled() is False

    def test_delta_field_enabled_only_with_waterfall(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        time_panel.set_overlay_enabled(True)
        assert time_panel._waterfall_delta_field.isEnabled() is False
        time_panel.set_waterfall_enabled(True)
        assert time_panel._waterfall_delta_field.isEnabled() is True

    def test_waterfall_changed_emitted_on_toggle(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        time_panel.set_overlay_enabled(True)
        received: list[bool] = []
        time_panel.waterfall_changed.connect(lambda: received.append(True))
        time_panel._waterfall_checkbox.setChecked(True)
        assert received == [True]


class TestWaterfallSaturationSentinels:
    def test_auto_y_covers_percent_scale_stack(self, time_panel: PlotPanel) -> None:
        # ±25 % traces stacked by Δ≈55 legitimately cross the ±100 % sentinel
        # threshold by trace 2–3; classifying sentinels on the displayed
        # (offset) values masked those traces out of framing, so auto-Y never
        # covered the top of the stack.
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(i + 1, 25.0) for i in range(4)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)

        delta = time_panel._waterfall_resolved_delta
        assert delta > 30.0
        assert time_panel._y_max.value() >= 3 * delta

    def test_shifted_raw_sentinel_still_excluded(self, time_panel: PlotPanel) -> None:
        # A genuine −100 % sentinel on trace 1 is displayed at −100 + Δ (well
        # inside the normal range), while trace 2's stacked points sit above
        # +100: the stored raw-value mask must exclude the former from framing
        # and keep the latter.
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(1, 5.0), _time_dataset(2, 5.0), _time_dataset(3, 5.0)]
        sentinel_idx = [10, 11, 12]
        datasets[1].asymmetry[sentinel_idx] = -100.0
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.set_waterfall_offset(60.0)
        time_panel.plot_datasets(datasets)

        stored = time_panel._last_plot_sentinel_mask
        assert stored is not None
        assert int(stored.sum()) == len(sentinel_idx)

        displayed = time_panel._last_plot_asymmetry
        keep = time_panel._mask_without_saturation_sentinels(
            displayed, np.ones_like(displayed, dtype=bool)
        )
        # The shifted sentinels (displayed ≈ −40) are excluded...
        assert not np.any(keep & stored)
        # ...while stacked points legitimately above +100 are kept.
        over_100 = displayed >= 100.0
        assert np.any(over_100)
        assert np.all(keep[over_100])


class TestWaterfallReframing:
    def test_toggling_waterfall_reframes_y_limits(self, time_panel: PlotPanel) -> None:
        # The frame identity must include the waterfall state: checking the box
        # on an already-framed overlay used to keep the stale y-limits and draw
        # the stack off-screen.
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(i + 1, 0.2) for i in range(3)]
        time_panel.set_overlay_enabled(True)
        time_panel.plot_datasets(datasets)
        y_max_flat = time_panel._y_max.value()

        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)  # the replot the toggle handler triggers
        delta = time_panel._waterfall_resolved_delta
        assert delta > 0.0
        assert time_panel._y_max.value() >= 2 * delta
        assert time_panel._y_max.value() > y_max_flat

    def test_editing_manual_delta_reframes_y_limits(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(i + 1, 0.2) for i in range(3)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)

        time_panel.set_waterfall_offset(5.0)
        time_panel.plot_datasets(datasets)
        assert time_panel._y_max.value() >= 10.0  # covers 2 × Δ


class TestWaterfallFrequencyXFraming:
    @staticmethod
    def _highfield_spectra(n: int = 3) -> list[MuonDataset]:
        """High-TF-like spectra: a narrow line at 800 MHz over a flat floor."""
        rng = np.random.default_rng(7)
        f = np.linspace(0.0, 1000.0, 4096)
        spectra = []
        for i in range(n):
            y = 0.01 + 0.001 * rng.random(f.size)
            y = y + np.exp(-(((f - 800.0) / 0.5) ** 2))
            spectra.append(
                MuonDataset(
                    time=f,
                    asymmetry=y,
                    error=np.zeros_like(y),
                    metadata={
                        "run_number": 100 + i,
                        "plot_domain": "frequency",
                        "x_label": "Frequency (MHz)",
                        "y_label": "|F| (arb.)",
                    },
                )
            )
        return spectra

    def test_x_framing_identical_with_and_without_waterfall(self, qapp: QApplication) -> None:
        # First-paint x-framing must be computed from the RAW spectra: offset
        # pedestals corrupt the block-median baseline in the peak finder
        # (non-monotonic np.interp across trace boundaries) and used to
        # spuriously veto the centered high-TF window, collapsing to full span.
        # A manual Δ gives pedestals comparable to the peak height, the worst case.
        limits: dict[bool, tuple[float, float]] = {}
        for waterfall in (False, True):
            panel = PlotPanel(domain="frequency")
            if not panel._has_mpl:
                panel.deleteLater()
                pytest.skip("matplotlib not available")
            try:
                panel.set_overlay_enabled(True)
                panel.set_waterfall_enabled(waterfall)
                panel.set_waterfall_offset(2.0)
                panel.plot_datasets(self._highfield_spectra())
                limits[waterfall] = (panel._x_min.value(), panel._x_max.value())
            finally:
                panel.close()
                panel.deleteLater()

        # The plain overlay engages the centered high-TF window (not full span)...
        assert limits[False][1] < 999.0
        assert limits[False][0] > 0.0
        # ...and the waterfall overlay frames the identical x-window.
        assert limits[True][0] == pytest.approx(limits[False][0])
        assert limits[True][1] == pytest.approx(limits[False][1])


class TestWaterfallWindowedAutoDelta:
    @staticmethod
    def _fft_like_spectra(n: int = 3) -> list[MuonDataset]:
        """Full-Nyquist FFT-magnitude shape: dominant peaks below ~2 MHz over a
        long near-zero tail to 50 MHz, so first-paint frames only the peak
        region and full-array percentiles land in the tail (the field defect).
        """
        f = np.linspace(0.0, 50.0, 2000)
        spectra = []
        for i in range(n):
            y = 7.0e7 * np.exp(-(((f - (0.6 + 0.6 * i)) / 0.3) ** 2)) + 7.0e3
            spectra.append(
                MuonDataset(
                    time=f,
                    asymmetry=y,
                    error=np.zeros_like(y),
                    metadata={
                        "run_number": 4000 + i,
                        "plot_domain": "frequency",
                        "x_label": "Frequency (MHz)",
                        "y_label": "|F| (arb.)",
                    },
                )
            )
        return spectra

    def test_frequency_auto_delta_measures_displayed_window(self, freq_panel: PlotPanel) -> None:
        # Auto Δ must be resolved from the samples inside the x-window actually
        # shown: full-array percentiles over the near-zero tail deflated Δ to
        # ~10 % of the visible spans and the stack looked like a no-op.
        if not freq_panel._has_mpl:
            pytest.skip("matplotlib not available")
        spectra = self._fft_like_spectra()
        freq_panel.set_overlay_enabled(True)
        freq_panel.set_waterfall_enabled(True)
        freq_panel.plot_datasets(spectra)

        # First-paint framed only the peak region, not the full Nyquist span.
        x_lo, x_hi = freq_panel._x_min.value(), freq_panel._x_max.value()
        assert x_hi < 25.0

        # Adjacent baselines (separated by Δ) must clear the median in-window
        # robust span, so the stack actually resolves.
        delta = freq_panel._waterfall_resolved_delta
        spans = []
        for ds in spectra:
            in_window = (ds.time >= x_lo) & (ds.time <= x_hi)
            vals = ds.asymmetry[in_window]
            spans.append(float(np.percentile(vals, 98) - np.percentile(vals, 2)))
        median_span = float(np.median(spans))
        assert delta >= median_span
        # The full-array spacing (the old bug) is several times smaller.
        assert delta > 3.0 * auto_waterfall_delta([ds.asymmetry for ds in spectra])

    def test_zoom_re_render_keeps_plot_time_delta(self, freq_panel: PlotPanel) -> None:
        # A zoom schedules a decimation re-render of the same content through
        # plot_datasets; the auto Δ must stay at its plot-time value rather
        # than re-resolving from the zoomed window and re-spacing the stack.
        if not freq_panel._has_mpl:
            pytest.skip("matplotlib not available")
        spectra = self._fft_like_spectra()
        freq_panel.set_overlay_enabled(True)
        freq_panel.set_waterfall_enabled(True)
        freq_panel.plot_datasets(spectra)
        delta_at_plot_time = freq_panel._waterfall_resolved_delta
        assert delta_at_plot_time > 0.0

        # User zooms into the near-zero tail; the viewport refresh replots.
        freq_panel._x_min.setValue(30.0)
        freq_panel._x_max.setValue(50.0)
        freq_panel.plot_datasets(spectra)
        assert freq_panel._waterfall_resolved_delta == pytest.approx(delta_at_plot_time)


class TestWaterfallExportRecord:
    def test_grouped_view_export_carries_no_stale_offsets(self, time_panel: PlotPanel) -> None:
        # The export stamps offsets from the last actually-drawn stack, never
        # from the checkbox: after switching to a grouped-subplot view (with
        # Waterfall still checked) the payloads must carry no offsets.
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(i + 1, 0.2) for i in range(3)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)
        payloads = time_panel.get_current_plot_export_data()
        assert payloads is not None
        assert all("waterfall_offset" in p for p in payloads)

        grouped = [_time_dataset(10, 0.2), _time_dataset(11, 0.2)]
        time_panel.plot_grouped_time_domain_subplots(grouped)
        payloads = time_panel.get_current_plot_export_data()
        assert payloads is not None
        assert all("waterfall_offset" not in p for p in payloads)

    def test_single_view_export_carries_no_stale_offsets(self, time_panel: PlotPanel) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        datasets = [_time_dataset(i + 1, 0.2) for i in range(3)]
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.plot_datasets(datasets)

        time_panel.plot_dataset(_time_dataset(9, 0.2))
        payloads = time_panel.get_current_plot_export_data()
        assert payloads is not None
        assert all("waterfall_offset" not in p for p in payloads)


class TestWaterfallDeltaField:
    def test_clearing_delta_field_restores_auto(self, time_panel: PlotPanel) -> None:
        # Clearing the manual-Δ field must stick as "Auto"; the field used to
        # rewrite the stored value back on commit, making auto unreachable.
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        time_panel.set_overlay_enabled(True)
        time_panel.set_waterfall_enabled(True)
        time_panel.set_waterfall_offset(2.5)
        field = time_panel._waterfall_delta_field
        assert field.text() == "2.500"

        field.setFocus()
        field.clear()
        QTest.keyClick(field, Qt.Key.Key_Return)

        assert time_panel.waterfall_offset() is None
        assert field.text() == ""
        assert field.placeholderText() == "Auto"

    def test_type_then_clear_does_not_resurrect_restored_offset(
        self, time_panel: PlotPanel
    ) -> None:
        if not time_panel._has_mpl:
            pytest.skip("matplotlib not available")
        time_panel.set_overlay_enabled(True)
        # A project restore carrying a manual Δ seeds the field's stored value.
        time_panel.restore_state(
            {
                "bunch_factor": 1,
                "x_min": 0.0,
                "x_max": 10.0,
                "y_min": -30.0,
                "y_max": 30.0,
                "waterfall": {"enabled": True, "offset": 2.5},
            }
        )
        field = time_panel._waterfall_delta_field
        field.setFocus()
        field.selectAll()
        QTest.keyClicks(field, "4.0")
        QTest.keyClick(field, Qt.Key.Key_Return)
        assert time_panel.waterfall_offset() == pytest.approx(4.0)

        field.selectAll()
        QTest.keyClick(field, Qt.Key.Key_Delete)
        QTest.keyClick(field, Qt.Key.Key_Return)

        assert time_panel.waterfall_offset() is None
        assert field.text() == ""
        assert time_panel.get_state()["waterfall"]["offset"] is None
