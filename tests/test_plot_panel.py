"""Tests for PlotPanel."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import PySide6 conditionally
pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton  # type: ignore

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.export_paths import resolve_gle_export_paths
from asymmetry.gui.panels.plot_panel import PlotPanel, _FloatLimitField
from asymmetry.gui.styles import tokens


class _FakeAxis:
    def __init__(self) -> None:
        self.errorbar_calls: list[dict[str, object]] = []
        self.plot_calls: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []
        self.xlim_calls: list[tuple[float, float]] = []
        self.ylim_calls: list[tuple[float, float]] = []
        self.xlabel_calls: list[str] = []
        self.ylabel_calls: list[str] = []
        self.tick_params_calls: list[dict[str, object]] = []
        self.legend_call_count = 0
        self.xaxis = SimpleNamespace(
            label=SimpleNamespace(get_color=lambda: "black", set_color=self._set_x_label_color)
        )
        self._x_label_color = "black"

    def _set_x_label_color(self, color: str) -> None:
        self._x_label_color = color

    def errorbar(self, *args, **kwargs) -> None:
        self.errorbar_calls.append({"args": args, "kwargs": kwargs})

    def plot(self, *args, **kwargs) -> None:
        self.plot_calls.append({"args": args, "kwargs": kwargs})

    def text(self, *args, **kwargs) -> None:
        self.text_calls.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, label: str, *_args, **_kwargs) -> None:
        self.xlabel_calls.append(label)

    def set_ylabel(self, label: str, *_args, **_kwargs) -> None:
        self.ylabel_calls.append(label)

    def tick_params(self, *args, **kwargs) -> None:
        self.tick_params_calls.append({"args": args, "kwargs": kwargs})

    def legend(self, *_args, **_kwargs) -> None:
        self.legend_call_count += 1
        return

    def set_xlim(self, xmin: float, xmax: float) -> None:
        self.xlim_calls.append((xmin, xmax))

    def set_ylim(self, ymin: float, ymax: float) -> None:
        self.ylim_calls.append((ymin, ymax))


class _FakeFigure:
    def __init__(
        self,
        axis: _FakeAxis,
        figsize: tuple[float, float] | None = None,
        *,
        generate_data_files: bool = False,
    ) -> None:
        self._axis = axis
        self.saved_paths: list[str] = []
        self.saved_kwargs: list[dict[str, object]] = []
        self.figsize = figsize
        self.generate_data_files = generate_data_files

    def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
        return self._axis

    def savefig(self, path: str, **kwargs) -> None:
        self.saved_kwargs.append(dict(kwargs))
        output_path = Path(path)
        if kwargs.get("folder"):
            output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
            export_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(str(output_path))
        output_path.write_text("! fake gle", encoding="utf-8")
        if not self.generate_data_files:
            return

        out_dir = output_path.parent
        for call in self._axis.errorbar_calls:
            kwargs = call.get("kwargs", {})
            data_name = kwargs.get("data_name")
            if not data_name:
                continue
            args = call.get("args", ())
            if len(args) < 2:
                continue
            x_vals = np.asarray(args[0], dtype=float)
            y_vals = np.asarray(args[1], dtype=float)
            e_vals = kwargs.get("yerr")
            err = np.asarray(e_vals, dtype=float) if e_vals is not None else np.zeros_like(y_vals)
            data_path = out_dir / f"{data_name}.dat"
            with open(data_path, "w", encoding="utf-8") as f:
                for xv, yv, ev in zip(x_vals, y_vals, err):
                    f.write(f"{float(xv):.10g} {float(yv):.10g} {float(ev):.10g}\n")


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp: QApplication) -> PlotPanel:
    """Create a PlotPanel for testing."""
    widget = PlotPanel()
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.fixture
def sample_dataset() -> MuonDataset:
    """Create a sample dataset."""
    t = np.linspace(0, 10, 100)
    a = 0.2 * np.exp(-0.5 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 12345})


class TestPlotPanel:
    def test_initialization(self, panel: PlotPanel) -> None:
        """Test panel initializes correctly."""
        assert panel is not None
        if hasattr(panel, "_canvas"):
            assert panel._canvas is not None
        if hasattr(panel, "_canvas_scroll_area"):
            # Canvas lives inside _canvas_host; the scroll area holds the host
            assert panel._canvas_scroll_area.widget() is panel._canvas_host

    def test_toolbar_does_not_show_apply_or_bunch_controls(self, panel: PlotPanel) -> None:
        """Main plot toolbar should not expose Apply or Bunch controls."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        buttons = panel.findChildren(QPushButton)
        button_texts = {btn.text() for btn in buttons}
        assert "Apply" not in button_texts
        assert "Auto" not in button_texts
        assert "Auto X" in button_texts
        assert "Auto Y" in button_texts
        assert "Pan" in button_texts
        assert "Zoom" in button_texts

        labels = panel.findChildren(QLabel)
        label_texts = {lbl.text() for lbl in labels}
        assert "Bunch:" not in label_texts

    def test_plot_parameters_are_shown_inline(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        assert isinstance(panel._x_min, _FloatLimitField)
        assert isinstance(panel._x_max, _FloatLimitField)
        assert isinstance(panel._y_min, _FloatLimitField)
        assert isinstance(panel._y_max, _FloatLimitField)

    def test_overlay_checkbox_defaults_to_disabled(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        assert panel.is_overlay_enabled() is False
        assert panel._overlay_checkbox.text() == "Overlay"

    def test_second_toolbar_row_places_label_overlay_then_annotation_export(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        top_layout = panel.layout()
        assert top_layout is not None
        nav_row = top_layout.itemAt(1).layout()
        assert nav_row is not None

        label_combo_pos = nav_row.indexOf(panel._label_field_combo)
        overlay_pos = nav_row.indexOf(panel._overlay_checkbox)

        assert label_combo_pos >= 0
        assert overlay_pos > label_combo_pos
        assert nav_row.indexOf(panel._add_label_btn) == -1
        assert nav_row.indexOf(panel._export_gle_btn) == -1

    def test_third_toolbar_row_right_aligns_annotation_and_export_controls(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        footer_row = panel._plot_footer.layout()
        assert footer_row is not None

        annotation_pos = footer_row.indexOf(panel._add_label_btn)
        export_pos = footer_row.indexOf(panel._export_gle_btn)
        format_pos = footer_row.indexOf(panel._gle_format_combo)

        assert annotation_pos >= 0
        assert export_pos > annotation_pos
        assert format_pos > export_pos

    def test_pan_and_zoom_buttons_live_in_separate_right_aligned_row(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        row0 = panel._limit_toolbar.itemAt(0).layout()
        assert row0 is not None
        assert row0.indexOf(panel._pan_btn) == -1
        assert row0.indexOf(panel._zoom_btn) == -1

        top_layout = panel.layout()
        assert top_layout is not None
        nav_row = top_layout.itemAt(1).layout()
        assert nav_row is not None
        assert nav_row.indexOf(panel._pan_btn) >= 0
        assert nav_row.indexOf(panel._zoom_btn) > nav_row.indexOf(panel._pan_btn)

    def test_pan_and_zoom_buttons_toggle_matplotlib_navigation_mode(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel._set_navigation_mode("none")
        assert panel._current_navigation_mode() == "none"

        panel._pan_btn.click()
        assert panel._current_navigation_mode() == "pan"
        assert panel._pan_btn.isChecked()
        assert not panel._zoom_btn.isChecked()

        panel._zoom_btn.click()
        assert panel._current_navigation_mode() == "zoom"
        assert panel._zoom_btn.isChecked()
        assert not panel._pan_btn.isChecked()

        panel._zoom_btn.click()
        assert panel._current_navigation_mode() == "none"
        assert not panel._pan_btn.isChecked()
        assert not panel._zoom_btn.isChecked()

    def test_pan_and_zoom_buttons_have_explicit_checked_style(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        assert "QPushButton:checked" in panel._pan_btn.styleSheet()
        assert "#1f4d8a" in panel._pan_btn.styleSheet()
        assert panel._pan_btn.styleSheet() == panel._zoom_btn.styleSheet()

    def test_limit_fields_follow_axis_limit_changes(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)

        panel._ax.set_xlim(1.1, 3.9)
        panel._ax.set_ylim(-0.15, 0.25)
        panel._canvas.draw()

        assert panel._x_min.value() == pytest.approx(1.1)
        assert panel._x_max.value() == pytest.approx(3.9)
        assert panel._y_min.value() == pytest.approx(-0.15)
        assert panel._y_max.value() == pytest.approx(0.25)

    def test_auto_x_and_auto_y_change_only_their_axes(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)

        panel._x_min.setValue(1.0)
        panel._x_max.setValue(2.0)
        panel._y_min.setValue(-10.0)
        panel._y_max.setValue(10.0)
        panel._apply_limits()

        x_before = panel._ax.get_xlim()
        y_before = panel._ax.get_ylim()

        panel._auto_x_limits()
        x_after_x = panel._ax.get_xlim()
        y_after_x = panel._ax.get_ylim()

        assert x_after_x != x_before
        assert y_after_x == pytest.approx(y_before)

        panel._auto_y_limits()
        x_after_y = panel._ax.get_xlim()

        assert x_after_y == pytest.approx(x_after_x)

    def test_auto_x_and_auto_y_buttons_are_persistent_toggles(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)

        assert panel._auto_x_btn.isCheckable()
        assert panel._auto_y_btn.isCheckable()
        assert not panel._auto_x_btn.isChecked()
        assert not panel._auto_y_btn.isChecked()

        panel._auto_x_btn.click()
        panel._auto_y_btn.click()
        assert panel._auto_x_btn.isChecked()
        assert panel._auto_y_btn.isChecked()

        panel._auto_x_btn.click()
        panel._auto_y_btn.click()
        assert not panel._auto_x_btn.isChecked()
        assert not panel._auto_y_btn.isChecked()

    def test_auto_x_and_auto_y_buttons_have_explicit_checked_style(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        assert "QPushButton:checked" in panel._auto_x_btn.styleSheet()
        assert "#1f4d8a" in panel._auto_x_btn.styleSheet()
        assert panel._auto_x_btn.styleSheet() == panel._auto_y_btn.styleSheet()

    def test_active_auto_limit_toggles_reapply_on_new_dataset(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel._auto_x_btn.click()
        panel._auto_y_btn.click()

        first_x = panel._ax.get_xlim()
        first_y = panel._ax.get_ylim()

        shifted_dataset = MuonDataset(
            time=sample_dataset.time + 50.0,
            asymmetry=sample_dataset.asymmetry * 4.0,
            error=sample_dataset.error,
            metadata=dict(sample_dataset.metadata),
            run=sample_dataset.run,
        )
        panel.plot_dataset(shifted_dataset)

        second_x = panel._ax.get_xlim()
        second_y = panel._ax.get_ylim()

        assert second_x != pytest.approx(first_x)
        assert second_y != pytest.approx(first_y)
        assert second_x[0] > 40.0
        assert second_y[1] > first_y[1]

    def test_auto_y_uses_current_x_range_and_ignores_low_count_points(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        counts = np.full(100, 1000.0)
        hist = Histogram(counts=counts, bin_width=0.1, t0_bin=0)
        run = Run(
            run_number=321,
            histograms=[hist],
            grouping={"first_good_bin": 10, "last_good_bin": 80},
        )

        time = hist.time_axis.copy()
        asym = np.full_like(time, 0.2, dtype=float)
        err = np.full_like(time, 0.01, dtype=float)
        # Outliers in low-count bins (outside good-bin window): should be ignored by Auto Y.
        asym[5] = 100.0
        asym[95] = -100.0
        # Outlier in good bins but outside current x-range: also ignored by Auto Y.
        asym[60] = 5.0

        ds = MuonDataset(
            time=time, asymmetry=asym, error=err, metadata={"run_number": 321}, run=run
        )
        panel.plot_dataset(ds)

        panel._x_min.setValue(1.0)
        panel._x_max.setValue(3.0)
        panel._apply_limits()
        panel._auto_y_limits()

        assert panel._y_max.value() < 1.0
        assert panel._y_min.value() > -1.0

    def test_plot_dataset_draws_low_count_points_gray_without_histograms(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        time = np.arange(20, dtype=float)
        asym = np.linspace(0.25, 0.05, time.size)
        err = np.full_like(time, 0.01)
        run = Run(
            run_number=322,
            grouping={"first_good_bin": 5, "last_good_bin": 14},
        )
        ds = MuonDataset(
            time=time,
            asymmetry=asym,
            error=err,
            metadata={"run_number": 322},
            run=run,
        )

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_dataset(ds)

        assert len(errorbar_calls) >= 2
        low_call = errorbar_calls[0]
        main_call = errorbar_calls[1]
        low_x = np.asarray(low_call["args"][0], dtype=float)
        main_x = np.asarray(main_call["args"][0], dtype=float)

        assert low_call["kwargs"].get("color") == "0.6"
        assert np.all((low_x < 5.0) | (low_x > 14.0))
        assert np.all((main_x >= 5.0) & (main_x <= 14.0))

    def test_plot_dataset_low_count_mask_uses_source_time_when_rebinned_without_histograms(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel._bunch_factor.setValue(2)

        time = np.arange(20, dtype=float)
        asym = np.linspace(0.3, 0.1, time.size)
        err = np.full_like(time, 0.01)
        run = Run(
            run_number=323,
            grouping={"first_good_bin": 5, "last_good_bin": 14},
        )
        ds = MuonDataset(
            time=time,
            asymmetry=asym,
            error=err,
            metadata={"run_number": 323},
            run=run,
        )

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_dataset(ds)

        assert len(errorbar_calls) >= 2
        low_call = errorbar_calls[0]
        main_call = errorbar_calls[1]
        low_x = np.asarray(low_call["args"][0], dtype=float)
        main_x = np.asarray(main_call["args"][0], dtype=float)

        assert low_call["kwargs"].get("color") == "0.6"
        assert np.all((low_x < 5.0) | (low_x > 14.0))
        assert np.all((main_x >= 5.0) & (main_x <= 14.0))

    def test_plot_dataset_without_histograms_applies_good_window(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        time = np.arange(20, dtype=float)
        asym = np.linspace(0.25, 0.05, time.size)
        err = np.full_like(time, 0.01)
        run = Run(
            run_number=324,
            grouping={"first_good_bin": 5, "last_good_bin": 14},
            source_file="/tmp/run_324.nxs",
        )
        ds = MuonDataset(
            time=time,
            asymmetry=asym,
            error=err,
            metadata={"run_number": 324},
            run=run,
        )

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_dataset(ds)

        assert len(errorbar_calls) >= 2
        low_call = errorbar_calls[0]
        main_call = errorbar_calls[1]
        low_x = np.asarray(low_call["args"][0], dtype=float)
        main_x = np.asarray(main_call["args"][0], dtype=float)

        assert low_call["kwargs"].get("color") == "0.6"
        assert np.all((low_x < 5.0) | (low_x > 14.0))
        assert np.all((main_x >= 5.0) & (main_x <= 14.0))

    def test_low_count_mask_marks_saturated_and_zero_denominator_bins_in_good_window(
        self,
        panel: PlotPanel,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        f_counts = np.array([80, 70, 60, 50, 40, 2, 1, 0, 0, 0], dtype=float)
        b_counts = np.array([80, 70, 60, 50, 40, 0, 0, 1, 0, 0], dtype=float)
        denominator = f_counts + b_counts
        asym = np.zeros_like(denominator)
        safe = denominator > 0.0
        asym[safe] = ((f_counts[safe] - b_counts[safe]) / denominator[safe]) * 100.0
        err = np.ones_like(asym)

        run = Run(
            run_number=324,
            histograms=[
                Histogram(counts=f_counts, bin_width=1.0, t0_bin=0),
                Histogram(counts=b_counts, bin_width=1.0, t0_bin=0),
            ],
            grouping={
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": 9,
            },
        )
        ds = MuonDataset(
            time=np.arange(10, dtype=float),
            asymmetry=asym,
            error=err,
            metadata={"run_number": 324},
            run=run,
        )

        mask = panel._low_count_mask_for_dataset(ds, source_dataset=ds)

        assert mask.shape == (10,)
        assert np.array_equal(
            mask, np.array([False, False, False, False, False, True, True, True, True, True])
        )

    def test_low_count_mask_skips_raw_denominator_under_variable_binning(
        self,
        panel: PlotPanel,
    ) -> None:
        """Under variable binning the per-raw-bin denominator no longer maps to
        the displayed bins, so the mask returns the saturation flags only — an
        explicit skip, not a silent array-shape mismatch. Same counts as the
        fixed-binning test above, where bins 8-9 (zero denominator) are flagged;
        here they are not."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        f_counts = np.array([80, 70, 60, 50, 40, 2, 1, 0, 0, 0], dtype=float)
        b_counts = np.array([80, 70, 60, 50, 40, 0, 0, 1, 0, 0], dtype=float)
        denominator = f_counts + b_counts
        asym = np.zeros_like(denominator)
        safe = denominator > 0.0
        asym[safe] = ((f_counts[safe] - b_counts[safe]) / denominator[safe]) * 100.0

        run = Run(
            run_number=3240,
            histograms=[
                Histogram(counts=f_counts, bin_width=1.0, t0_bin=0),
                Histogram(counts=b_counts, bin_width=1.0, t0_bin=0),
            ],
            grouping={
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": 9,
                "binning_mode": "variable",
                "bin0_us": 0.08,
                "bin10_us": 0.25,
            },
        )
        ds = MuonDataset(
            time=np.arange(10, dtype=float),
            asymmetry=asym,
            error=np.ones_like(asym),
            metadata={"run_number": 3240},
            run=run,
        )

        mask = panel._low_count_mask_for_dataset(ds, source_dataset=ds)

        # Saturation only: bins 5-7 (±100%); zero-denominator bins 8-9 are NOT
        # added because the raw-bin reduction is explicitly skipped here.
        assert np.array_equal(
            mask, np.array([False, False, False, False, False, True, True, True, False, False])
        )

    def test_low_count_mask_uses_background_corrected_psi_denominator(
        self,
        panel: PlotPanel,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        f_raw = np.array([10.0, 10.0, 110.0, 80.0, 50.0, 25.0, 13.0, 10.1, 9.9, 10.2])
        b_raw = np.array([12.0, 12.0, 90.0, 70.0, 48.0, 25.0, 13.1, 11.9, 12.2, 12.0])
        f_corr = f_raw - 10.0
        b_corr = b_raw - 12.0
        denominator = f_corr + b_corr
        asym = np.zeros_like(denominator)
        safe = denominator != 0.0
        asym[safe] = ((f_corr[safe] - b_corr[safe]) / denominator[safe]) * 100.0

        run = Run(
            run_number=326,
            histograms=[
                Histogram(counts=f_raw, bin_width=1.0, t0_bin=0),
                Histogram(counts=b_raw, bin_width=1.0, t0_bin=0),
            ],
            metadata={"facility": "PSI"},
            grouping={
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": 9,
                "background_correction": True,
                "background_range": [0, 1],
            },
            source_file="/tmp/run_326.bin",
        )
        ds = MuonDataset(
            time=np.arange(10, dtype=float),
            asymmetry=asym,
            error=np.ones_like(asym),
            metadata={"run_number": 326, "facility": "PSI"},
            run=run,
        )

        mask = panel._low_count_mask_for_dataset(ds, source_dataset=ds)

        assert np.array_equal(
            mask,
            np.array([True, True, False, False, False, False, False, True, True, True]),
        )

    def test_low_count_mask_projects_saturated_source_bins_after_rebin(
        self,
        panel: PlotPanel,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel._bunch_factor.setValue(2)

        f_counts = np.array([80, 70, 60, 50, 40, 2, 1, 0, 0, 0], dtype=float)
        b_counts = np.array([80, 70, 60, 50, 40, 0, 0, 1, 0, 0], dtype=float)
        denominator = f_counts + b_counts
        asym = np.zeros_like(denominator)
        safe = denominator > 0.0
        asym[safe] = ((f_counts[safe] - b_counts[safe]) / denominator[safe]) * 100.0
        err = np.ones_like(asym)

        run = Run(
            run_number=325,
            histograms=[
                Histogram(counts=f_counts, bin_width=1.0, t0_bin=0),
                Histogram(counts=b_counts, bin_width=1.0, t0_bin=0),
            ],
            grouping={
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": 9,
            },
        )
        ds = MuonDataset(
            time=np.arange(10, dtype=float),
            asymmetry=asym,
            error=err,
            metadata={"run_number": 325},
            run=run,
        )

        analysis_ds = panel.get_analysis_dataset(ds)
        assert analysis_ds is not None

        mask = panel._low_count_mask_for_dataset(analysis_ds, source_dataset=ds)

        assert np.array_equal(mask, np.array([False, False, True, True, True]))

    def test_low_count_mask_does_not_treat_grouped_time_domain_counts_as_saturated(
        self,
        panel: PlotPanel,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = MuonDataset(
            time=np.array([0.0, 1.0, 2.0], dtype=float),
            asymmetry=np.array([80.0, 120.0, 160.0], dtype=float),
            error=np.ones(3, dtype=float),
            metadata={
                "run_number": 999001,
                "grouped_time_domain": True,
                "y_label": "Lifetime-corrected counts",
            },
            run=None,
        )

        mask = panel._low_count_mask_for_dataset(ds, source_dataset=ds)

        assert np.array_equal(mask, np.array([False, False, False]))

    def test_axis_limits_persist_across_redraw_and_dataset_switch(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds1 = sample_dataset
        ds2 = MuonDataset(
            time=sample_dataset.time,
            asymmetry=sample_dataset.asymmetry * 0.5,
            error=sample_dataset.error,
            metadata={"run_number": 67890},
        )

        panel.plot_dataset(ds1)
        panel._x_min.setValue(0.5)
        panel._x_max.setValue(2.5)
        panel._y_min.setValue(-0.1)
        panel._y_max.setValue(0.4)
        panel._apply_limits()

        panel.plot_dataset(ds2)
        panel.plot_dataset(ds1)

        assert panel._x_min.value() == pytest.approx(0.5)
        assert panel._x_max.value() == pytest.approx(2.5)
        assert panel._y_min.value() == pytest.approx(-0.1)
        assert panel._y_max.value() == pytest.approx(0.4)

    def test_restored_limits_are_preserved_when_plotting_after_restore_without_dataset(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        restored_state = {
            "x_min": 0.25,
            "x_max": 2.75,
            "y_min": -0.2,
            "y_max": 0.35,
            "fit_x_min": None,
            "fit_x_max": None,
        }
        panel.restore_state(restored_state, dataset=None)
        panel.plot_dataset(sample_dataset)

        assert panel._x_min.value() == pytest.approx(0.25)
        assert panel._x_max.value() == pytest.approx(2.75)
        assert panel._y_min.value() == pytest.approx(-0.2)
        assert panel._y_max.value() == pytest.approx(0.35)

    def test_plot_dataset(self, panel: PlotPanel, sample_dataset: MuonDataset) -> None:
        """Test plotting a single dataset."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        panel.plot_dataset(sample_dataset)

    def test_plot_dataset_decimates_dense_display_but_keeps_full_cached_arrays(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 101)
        ds = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.25 * t),
            error=np.full_like(t, 0.01),
            metadata={"run_number": 7771},
        )
        panel._max_render_points_per_trace = 10

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_dataset(ds)

        assert panel._last_plot_time is not None
        assert len(panel._last_plot_time) == len(t)
        assert errorbar_calls
        plotted_x = np.asarray(errorbar_calls[-1]["args"][0], dtype=float)
        assert plotted_x.size < t.size
        assert plotted_x.size <= 11
        # The corner chip flags the decimated view ("11 of 101 pts"); the
        # x-axis itself is left untouched (no more alarm-red labels).
        chip_text = panel.decimation_chip_text()
        assert chip_text is not None
        assert "of" in chip_text and "pts" in chip_text
        assert panel._ax.xaxis.label.get_color() == panel._default_x_axis_label_color
        assert panel._canvas.toolTip() != ""

    def test_plot_dataset_can_disable_decimation_for_dense_display(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 101)
        ds = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.25 * t),
            error=np.full_like(t, 0.01),
            metadata={"run_number": 7775},
        )
        panel._max_render_points_per_trace = 10
        panel.set_decimation_enabled(False, redraw=False)

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_dataset(ds)

        plotted_x = np.asarray(errorbar_calls[-1]["args"][0], dtype=float)
        assert plotted_x.size == t.size
        # No decimation -> no chip, no tooltip.
        assert panel.decimation_chip_text() is None
        assert panel._canvas.toolTip() == ""

    def test_polarization_combo_uses_subscript_labels(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_x")
        labels = [
            panel._polarization_combo.itemText(i) for i in range(panel._polarization_combo.count())
        ]

        assert labels == ["x", "y", "z"]

    def test_polarization_combo_change_updates_current_axis_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_z")
        idx = panel._polarization_combo.findData("P_x")
        assert idx >= 0
        panel._polarization_combo.setCurrentIndex(idx)

        assert panel.get_current_polarization_axis() == "P_x"

    def test_polarization_combo_supports_all_option(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "ALL")
        labels = [
            panel._polarization_combo.itemText(i) for i in range(panel._polarization_combo.count())
        ]

        assert labels[0] == "All"

    def test_time_view_selector_supports_group_mode(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_time_view_modes(["fb_asymmetry", "groups"], current_mode="groups")

        labels = [panel._time_view_combo.itemText(i) for i in range(panel._time_view_combo.count())]

        assert labels == ["FB Asymmetry", "Individual Groups"]
        assert panel.current_time_view_mode() == "groups"

    def test_grouped_subplots_expand_canvas_height_for_scrolling(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        original_height = panel._canvas.minimumHeight()
        datasets = [
            MuonDataset(
                time=np.array([0.0, 1.0, 2.0]),
                asymmetry=np.array([1.0, 0.9, 0.8]),
                error=np.array([0.01, 0.01, 0.01]),
                metadata={"run_number": -(index + 1)},
            )
            for index in range(6)
        ]

        panel.plot_grouped_time_domain_subplots(datasets)

        assert panel._canvas.minimumHeight() > original_height

    def test_grouped_subplots_enable_vertical_scrollbar(
        self,
        qapp: QApplication,
        panel: PlotPanel,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        datasets = [
            MuonDataset(
                time=np.array([0.0, 1.0, 2.0]),
                asymmetry=np.array([1.0, 0.9, 0.8]),
                error=np.array([0.01, 0.01, 0.01]),
                metadata={"run_number": -(index + 1)},
            )
            for index in range(8)
        ]

        panel.resize(640, 280)
        panel.show()
        qapp.processEvents()

        panel.plot_grouped_time_domain_subplots(datasets)
        qapp.processEvents()

        assert panel._canvas_scroll_area.verticalScrollBar().maximum() > 0

    def test_grouped_subplots_shrink_with_panel_width(
        self,
        qapp: QApplication,
        panel: PlotPanel,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        datasets = [
            MuonDataset(
                time=np.array([0.0, 1.0, 2.0]),
                asymmetry=np.array([1.0, 0.9, 0.8]),
                error=np.array([0.01, 0.01, 0.01]),
                metadata={"run_number": -(index + 1)},
            )
            for index in range(6)
        ]

        panel.resize(900, 420)
        panel.show()
        qapp.processEvents()
        panel.plot_grouped_time_domain_subplots(datasets)
        qapp.processEvents()
        wide_width = panel._canvas.width()

        panel.resize(520, 420)
        qapp.processEvents()
        narrow_width = panel._canvas.width()

        assert narrow_width < wide_width

    def test_single_plot_resets_canvas_height_after_grouped_view(
        self,
        panel: PlotPanel,
        sample_dataset: MuonDataset,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        datasets = [
            MuonDataset(
                time=np.array([0.0, 1.0, 2.0]),
                asymmetry=np.array([1.0, 0.9, 0.8]),
                error=np.array([0.01, 0.01, 0.01]),
                metadata={"run_number": -(index + 1)},
            )
            for index in range(4)
        ]
        panel.plot_grouped_time_domain_subplots(datasets)

        panel.plot_dataset(sample_dataset)

        assert panel._canvas.minimumHeight() == panel._default_canvas_min_height

    def test_grouped_subplots_redraw_preserves_subplot_mode(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        datasets = [
            MuonDataset(
                time=np.array([0.0, 1.0, 2.0]),
                asymmetry=np.array([1.0, 0.9, 0.8]),
                error=np.array([0.01, 0.01, 0.01]),
                metadata={"run_number": -(index + 1)},
            )
            for index in range(3)
        ]
        panel.plot_grouped_time_domain_subplots(datasets)

        grouped_calls: list[list[int]] = []

        def _capture_grouped(captured_datasets: list[MuonDataset]) -> None:
            grouped_calls.append([int(ds.run_number) for ds in captured_datasets])

        def _fail_overlay(_datasets: list[MuonDataset]) -> None:
            raise AssertionError("grouped redraw regressed to overlay mode")

        monkeypatch.setattr(panel, "plot_grouped_time_domain_subplots", _capture_grouped)
        monkeypatch.setattr(panel, "plot_datasets", _fail_overlay)

        panel._redraw_current_view()

        assert grouped_calls == [[-1, -2, -3]]

    def test_polarization_axis_remembers_separate_y_limits(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_x")
        panel._y_min.setValue(-0.1)
        panel._y_max.setValue(0.3)
        panel._apply_limits()

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_y")
        panel._y_min.setValue(-1.0)
        panel._y_max.setValue(1.0)
        panel._apply_limits()

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_x")
        assert panel._y_min.value() == pytest.approx(-0.1)
        assert panel._y_max.value() == pytest.approx(0.3)

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_y")
        assert panel._y_min.value() == pytest.approx(-1.0)
        assert panel._y_max.value() == pytest.approx(1.0)
        # Check if plot was created (canvas should have drawn something)
        assert panel._canvas is not None

    def test_single_dataset_shows_alpha_value_in_plot(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0, 10, 100)
        a = 0.2 * np.exp(-0.5 * t)
        e = np.full_like(t, 0.01)
        run = Run(run_number=4321, grouping={"alpha": 1.2345})
        ds = MuonDataset(
            time=t,
            asymmetry=a,
            error=e,
            metadata={"run_number": 4321},
            run=run,
        )

        panel.plot_dataset(ds)
        assert panel._header_meta_label.text() == "(alpha = 1.2345)"

    def test_single_dataset_uses_axis_specific_alpha_in_vector_mode(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0, 10, 100)
        a = 0.2 * np.exp(-0.5 * t)
        e = np.full_like(t, 0.01)
        run = Run(
            run_number=4322,
            grouping={
                "alpha": 1.0,
                "alpha_x": 1.1,
                "alpha_y": 1.2,
                "alpha_z": 1.3,
                "vector_axis": "P_y",
            },
        )
        ds = MuonDataset(
            time=t,
            asymmetry=a,
            error=e,
            metadata={"run_number": 4322},
            run=run,
        )

        panel.plot_dataset(ds)
        assert panel._header_meta_label.text() == "(alpha = 1.2)"

    def test_vector_all_mode_hides_alpha_label(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 5.0, 40)
        e = np.full_like(t, 0.01)
        base = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.3 * t),
            error=e,
            metadata={"run_number": 9991},
            run=Run(run_number=9991, grouping={"alpha": 1.5}),
        )

        panel.plot_vector_subplots({"P_x": [base], "P_y": [base], "P_z": [base]})
        assert panel._header_meta_label.text() == ""

    def test_axis_specific_fits_persist_separately_per_run(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 80)
        e = np.full_like(t, 0.01)
        y_px = 0.21 * np.exp(-0.30 * t)
        y_py = 0.16 * np.exp(-0.24 * t)

        run_px = Run(run_number=9901, grouping={"vector_axis": "P_x"})
        run_py = Run(run_number=9901, grouping={"vector_axis": "P_y"})
        ds_px = MuonDataset(
            time=t, asymmetry=y_px, error=e, metadata={"run_number": 9901}, run=run_px
        )
        ds_py = MuonDataset(
            time=t, asymmetry=y_py, error=e, metadata={"run_number": 9901}, run=run_py
        )

        fit_px = 0.20 * np.exp(-0.28 * t)
        fit_py = 0.15 * np.exp(-0.22 * t)

        panel.plot_dataset(ds_px)
        panel.plot_fit(t, fit_px, label="Fit Px")

        panel.plot_dataset(ds_py)
        panel.plot_fit(t, fit_py, label="Fit Py")

        assert (9901, "P_x") in panel._fit_curves_by_key
        assert (9901, "P_y") in panel._fit_curves_by_key
        np.testing.assert_allclose(panel._fit_curves_by_key[(9901, "P_x")][1], fit_px)
        np.testing.assert_allclose(panel._fit_curves_by_key[(9901, "P_y")][1], fit_py)

    def test_all_mode_axis_plotting_uses_matching_axis_fit_curve(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 80)
        e = np.full_like(t, 0.01)
        run_px = Run(run_number=9902, grouping={"vector_axis": "P_x"})
        run_py = Run(run_number=9902, grouping={"vector_axis": "P_y"})
        ds_px = MuonDataset(
            time=t,
            asymmetry=0.22 * np.exp(-0.30 * t),
            error=e,
            metadata={"run_number": 9902},
            run=run_px,
        )
        ds_py = MuonDataset(
            time=t,
            asymmetry=0.17 * np.exp(-0.24 * t),
            error=e,
            metadata={"run_number": 9902},
            run=run_py,
        )

        fit_px = 0.21 * np.exp(-0.27 * t)
        fit_py = 0.14 * np.exp(-0.20 * t)

        panel._fit_curves_by_key[(9902, "P_x")] = (t, fit_px, "Fit Px")
        panel._fit_curves_by_key[(9902, "P_y")] = (t, fit_py, "Fit Py")

        ax_px = _FakeAxis()
        ax_py = _FakeAxis()
        panel._plot_datasets_on_axis(ax_px, [ds_px], "P_x")
        panel._plot_datasets_on_axis(ax_py, [ds_py], "P_y")

        assert ax_px.plot_calls
        assert ax_py.plot_calls
        np.testing.assert_allclose(np.asarray(ax_px.plot_calls[-1]["args"][1], dtype=float), fit_px)
        np.testing.assert_allclose(np.asarray(ax_py.plot_calls[-1]["args"][1], dtype=float), fit_py)

    def test_axis_specific_fit_does_not_leak_to_other_polarization(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 80)
        e = np.full_like(t, 0.01)
        ds_pz = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.3 * t),
            error=e,
            metadata={"run_number": 9903},
            run=Run(run_number=9903, grouping={"vector_axis": "P_z"}),
        )
        ds_py = MuonDataset(
            time=t,
            asymmetry=0.18 * np.exp(-0.25 * t),
            error=e,
            metadata={"run_number": 9903},
            run=Run(run_number=9903, grouping={"vector_axis": "P_y"}),
        )

        fit_pz = 0.19 * np.exp(-0.22 * t)
        panel._fit_curves_by_key[(9903, "P_z")] = (t, fit_pz, "Fit Pz")
        # Legacy run-only cache should not override axis-specific separation.
        panel._fit_curves[9903] = (t, fit_pz, "Fit")

        assert panel._fit_curve_for_dataset(ds_pz) is not None
        assert panel._fit_curve_for_dataset(ds_py) is None

    def test_period_mode_color_mapping(self, panel: PlotPanel) -> None:
        red_hist = Histogram(counts=np.array([1.0, 2.0, 3.0]), bin_width=0.01)
        run = Run(
            run_number=9001,
            histograms=[red_hist],
            grouping={
                "period_histograms": [[red_hist], [red_hist]],
                "period_mode": str(PeriodMode.GREEN_MINUS_RED),
            },
        )
        ds = MuonDataset(
            time=np.array([0.0, 0.01, 0.02]),
            asymmetry=np.array([0.1, 0.2, 0.3]),
            error=np.array([0.01, 0.01, 0.01]),
            metadata={"run_number": 9001},
            run=run,
        )
        assert panel._period_mode_color_for_dataset(ds) == "#0000c0"

    def test_multi_dataset_same_period_mode_uses_distinct_colors(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.array([0.0, 0.01, 0.02, 0.03])
        err = np.full_like(t, 0.01)

        def _two_period_ds(run_number: int) -> MuonDataset:
            red_hist = Histogram(counts=np.array([50.0, 60.0, 55.0, 58.0]), bin_width=0.01)
            run = Run(
                run_number=run_number,
                histograms=[red_hist],
                grouping={
                    "period_histograms": [[red_hist], [red_hist]],
                    "period_mode": str(PeriodMode.RED),
                },
            )
            return MuonDataset(
                time=t,
                asymmetry=np.array([0.1, 0.11, 0.09, 0.1]),
                error=err,
                metadata={"run_number": run_number},
                run=run,
            )

        ds1 = _two_period_ds(9101)
        ds2 = _two_period_ds(9102)

        panel.plot_datasets([ds1, ds2])
        handles, labels = panel._ax.get_legend_handles_labels()
        assert len(handles) >= 2

        def _handle_color(handle) -> str | None:
            if hasattr(handle, "lines") and getattr(handle, "lines"):
                return handle.lines[0].get_color()
            if hasattr(handle, "get_color"):
                return handle.get_color()
            return None

        first = _handle_color(handles[0])
        second = _handle_color(handles[1])
        assert first is not None
        assert second is not None
        assert first != second
        assert labels[0] == str(ds1.run_label)
        assert labels[1] == str(ds2.run_label)

    def test_period_mode_fit_line_uses_contrast_color(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.array([0.0, 0.01, 0.02, 0.03])
        err = np.full_like(t, 0.01)
        red_hist = Histogram(counts=np.array([40.0, 41.0, 39.0, 38.0]), bin_width=0.01)
        run = Run(
            run_number=9103,
            histograms=[red_hist],
            grouping={
                "period_histograms": [[red_hist], [red_hist]],
                "period_mode": str(PeriodMode.RED),
            },
        )
        ds = MuonDataset(
            time=t,
            asymmetry=np.array([0.1, 0.09, 0.11, 0.1]),
            error=err,
            metadata={"run_number": 9103},
            run=run,
        )

        panel.plot_dataset(ds)
        point_color = panel._period_mode_color_for_dataset(ds)
        assert point_color is not None

        original_plot = panel._ax.plot
        plot_calls: list[dict[str, object]] = []

        def _capture_plot(*args, **kwargs):
            plot_calls.append(kwargs)
            return original_plot(*args, **kwargs)

        panel._ax.plot = _capture_plot
        try:
            fit_y = 0.095 * np.exp(-0.1 * t)
            panel.plot_fit(t, fit_y, label="Fit")
        finally:
            panel._ax.plot = original_plot

        assert plot_calls
        fit_color = plot_calls[-1].get("color")
        assert isinstance(fit_color, str)
        assert fit_color != point_color

    def test_preview_fit_line_uses_red_overlay(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 100)
        ds = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.2 * t),
            error=np.full_like(t, 0.01),
            metadata={"run_number": 9901},
        )

        panel.plot_dataset(ds)

        original_plot = panel._ax.plot
        plot_calls: list[dict[str, object]] = []

        def _capture_plot(*args, **kwargs):
            plot_calls.append(kwargs)
            return original_plot(*args, **kwargs)

        panel._ax.plot = _capture_plot
        try:
            fit_y = 0.18 * np.exp(-0.15 * t)
            panel.plot_fit(t, fit_y, label="Preview")
        finally:
            panel._ax.plot = original_plot

        assert plot_calls
        assert plot_calls[-1].get("color") == "#d73a49"

    def test_fit_line_uses_red_overlay(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 100)
        ds = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.2 * t),
            error=np.full_like(t, 0.01),
            metadata={"run_number": 9902},
        )

        panel.plot_dataset(ds)

        original_plot = panel._ax.plot
        plot_calls: list[dict[str, object]] = []

        def _capture_plot(*args, **kwargs):
            plot_calls.append(kwargs)
            return original_plot(*args, **kwargs)

        panel._ax.plot = _capture_plot
        try:
            fit_y = 0.18 * np.exp(-0.15 * t)
            panel.plot_fit(t, fit_y, label="Fit")
        finally:
            panel._ax.plot = original_plot

        assert plot_calls
        assert plot_calls[-1].get("color") == tokens.PLOT_FIT

    def test_plot_multiple_datasets(self, panel: PlotPanel, sample_dataset: MuonDataset) -> None:
        """Test plotting multiple datasets."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        t2 = np.linspace(0, 10, 100)
        a2 = 0.15 * np.exp(-0.7 * t2)
        e2 = np.full_like(t2, 0.01)
        ds2 = MuonDataset(time=t2, asymmetry=a2, error=e2, metadata={"run_number": 67890})

        panel.plot_dataset(sample_dataset)
        panel.plot_dataset(ds2)
        # Panel should handle multiple datasets
        assert panel._canvas is not None

    def test_multi_dataset_legend_labels_follow_selected_label_field(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        panel.set_overlay_enabled(True)

        t = np.linspace(0, 5, 50)
        err = np.full_like(t, 0.01)
        ds1 = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.4 * t),
            error=err,
            metadata={"run_number": 101, "temperature": 2.5},
        )
        ds2 = MuonDataset(
            time=t,
            asymmetry=0.15 * np.exp(-0.3 * t),
            error=err,
            metadata={"run_number": 102, "temperature": 7.25},
        )

        panel.plot_datasets([ds1, ds2])
        _, labels = panel._ax.get_legend_handles_labels()
        assert str(ds1.run_label) in labels
        assert str(ds2.run_label) in labels

        idx = panel._label_field_combo.findData("temperature")
        assert idx >= 0
        panel._label_field_combo.setCurrentIndex(idx)

        _, labels = panel._ax.get_legend_handles_labels()
        assert "2.50 K" in labels
        assert "7.25 K" in labels

    def test_multi_dataset_plot_does_not_show_alpha_overlay(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0, 5, 50)
        err = np.full_like(t, 0.01)
        run1 = Run(run_number=701, grouping={"alpha": 1.1})
        run2 = Run(run_number=702, grouping={"alpha": 1.2})
        ds1 = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.4 * t),
            error=err,
            metadata={"run_number": 701},
            run=run1,
        )
        ds2 = MuonDataset(
            time=t,
            asymmetry=0.15 * np.exp(-0.3 * t),
            error=err,
            metadata={"run_number": 702},
            run=run2,
        )

        panel.plot_datasets([ds1, ds2])
        assert panel._header_meta_label.text() == ""

    def test_plot_datasets_decimates_each_dense_trace_independently(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 121)
        err = np.full_like(t, 0.01)
        ds1 = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.2 * t),
            error=err,
            metadata={"run_number": 7772},
        )
        ds2 = MuonDataset(
            time=t,
            asymmetry=0.15 * np.exp(-0.15 * t),
            error=err,
            metadata={"run_number": 7773},
        )
        panel._max_render_points_per_trace = 12

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_datasets([ds1, ds2])

        labelled_calls = {
            str(call["kwargs"].get("label")): np.asarray(call["args"][0], dtype=float)
            for call in errorbar_calls
            if call["kwargs"].get("label") not in {None, "_nolegend_"}
        }
        assert str(ds1.run_label) in labelled_calls
        assert str(ds2.run_label) in labelled_calls
        assert labelled_calls[str(ds1.run_label)].size < t.size
        assert labelled_calls[str(ds2.run_label)].size < t.size
        assert labelled_calls[str(ds1.run_label)].size <= 13
        assert labelled_calls[str(ds2.run_label)].size <= 13

    def test_set_view_limits_rebuilds_dense_trace_for_visible_window(
        self,
        panel: PlotPanel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 101)
        err = np.full_like(t, 0.01)
        ds = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.2 * t),
            error=err,
            metadata={"run_number": 7774},
        )
        panel._max_render_points_per_trace = 50

        errorbar_calls: list[dict[str, object]] = []
        original_errorbar = panel._ax.errorbar

        def _capture_errorbar(*args, **kwargs):
            errorbar_calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_errorbar(*args, **kwargs)

        monkeypatch.setattr(panel._ax, "errorbar", _capture_errorbar)

        panel.plot_dataset(ds)
        initial_visible = np.asarray(errorbar_calls[-1]["args"][0], dtype=float)
        assert panel.decimation_chip_text() is not None

        panel.set_view_limits(2.0, 4.0, -0.1, 0.3)
        QApplication.processEvents()

        refreshed_visible = np.asarray(errorbar_calls[-1]["args"][0], dtype=float)
        assert refreshed_visible.size < initial_visible.size
        assert np.all(refreshed_visible >= 2.0)
        assert np.all(refreshed_visible <= 4.0)
        # Zoomed view renders every visible point, so the chip disappears.
        assert panel.decimation_chip_text() is None
        assert panel._canvas.toolTip() == ""

    def test_add_label_keeps_multi_dataset_redraw(
        self, panel: PlotPanel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        panel.set_overlay_enabled(True)

        t = np.linspace(0, 5, 50)
        err = np.full_like(t, 0.01)
        ds1 = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.4 * t),
            error=err,
            metadata={"run_number": 201},
        )
        ds2 = MuonDataset(
            time=t,
            asymmetry=0.15 * np.exp(-0.3 * t),
            error=err,
            metadata={"run_number": 202},
        )

        panel._current_datasets = [ds1, ds2]
        panel._current_dataset = ds2

        redraw_calls: list[str] = []
        monkeypatch.setattr(panel, "plot_datasets", lambda datasets: redraw_calls.append("multi"))
        monkeypatch.setattr(panel, "plot_dataset", lambda dataset: redraw_calls.append("single"))
        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QInputDialog.getText",
            lambda *_args, **_kwargs: ("peak", True),
        )

        event = SimpleNamespace(inaxes=panel._ax, xdata=1.0, ydata=0.1)
        panel._add_annotation_at_event(event)

        assert len(panel._annotations) == 1
        assert redraw_calls == ["multi"]

    def test_label_field_selection_persists_in_panel_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        idx = panel._label_field_combo.findData("temperature")
        assert idx >= 0
        panel._label_field_combo.setCurrentIndex(idx)

        state = panel.get_state()

        restored = PlotPanel()
        if not hasattr(restored, "_has_mpl") or not restored._has_mpl:
            pytest.skip("matplotlib not available")
        restored.restore_state(state, dataset=None)

        assert restored._label_field_combo.currentData() == "temperature"

    def test_overlay_selection_persists_in_panel_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_overlay_enabled(False)
        state = panel.get_state()

        restored = PlotPanel()
        if not hasattr(restored, "_has_mpl") or not restored._has_mpl:
            pytest.skip("matplotlib not available")
        restored.restore_state(state, dataset=None)

        assert restored.is_overlay_enabled() is False

    def test_time_view_selection_persists_in_panel_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_time_view_modes(["fb_asymmetry", "groups"], current_mode="fb_asymmetry")
        panel.set_current_time_view_mode("groups")
        state = panel.get_state()

        restored = PlotPanel()
        if not hasattr(restored, "_has_mpl") or not restored._has_mpl:
            pytest.skip("matplotlib not available")
        restored.restore_state(state, dataset=None)
        restored.set_time_view_modes(
            ["fb_asymmetry", "groups"], current_mode=state["time_view_mode"]
        )

        assert restored.current_time_view_mode() == "groups"

    def test_axis_specific_fit_curves_round_trip_in_panel_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.array([0.0, 0.5, 1.0], dtype=float)
        fit_px = np.array([0.2, 0.15, 0.1], dtype=float)
        fit_py = np.array([0.18, 0.12, 0.08], dtype=float)

        panel._fit_curves_by_key[(1101, "P_x")] = (t, fit_px, "Fit Px")
        panel._fit_curves_by_key[(1101, "P_y")] = (t, fit_py, "Fit Py")
        panel._fit_components_by_key[(1101, "P_x")] = [("Component", fit_px)]
        panel._fit_metadata_by_key[(1101, "P_x")] = {"fit_function": "A0*exp(-lambda*t)"}

        state = panel.get_state()

        restored = PlotPanel()
        if not hasattr(restored, "_has_mpl") or not restored._has_mpl:
            pytest.skip("matplotlib not available")
        restored.restore_state(state, dataset=None)

        assert (1101, "P_x") in restored._fit_curves_by_key
        assert (1101, "P_y") in restored._fit_curves_by_key
        np.testing.assert_allclose(restored._fit_curves_by_key[(1101, "P_x")][1], fit_px)
        np.testing.assert_allclose(restored._fit_curves_by_key[(1101, "P_y")][1], fit_py)
        assert restored._fit_metadata_by_key[(1101, "P_x")]["fit_function"] == "A0*exp(-lambda*t)"

    def test_label_field_selection_is_tracked_per_data_group(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        idx_temperature = panel._label_field_combo.findData("temperature")
        idx_field = panel._label_field_combo.findData("field")
        assert idx_temperature >= 0
        assert idx_field >= 0

        panel.set_active_label_group("g1")
        panel._label_field_combo.setCurrentIndex(idx_temperature)

        panel.set_active_label_group("g2")
        panel._label_field_combo.setCurrentIndex(idx_field)

        panel.set_active_label_group("g1")
        assert panel._label_field_combo.currentData() == "temperature"

        panel.set_active_label_group("g2")
        assert panel._label_field_combo.currentData() == "field"

    def test_group_label_field_preferences_persist_in_panel_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        idx_temperature = panel._label_field_combo.findData("temperature")
        idx_field = panel._label_field_combo.findData("field")
        assert idx_temperature >= 0
        assert idx_field >= 0

        panel.set_active_label_group("g1")
        panel._label_field_combo.setCurrentIndex(idx_temperature)
        panel.set_active_label_group("g2")
        panel._label_field_combo.setCurrentIndex(idx_field)

        state = panel.get_state()

        restored = PlotPanel()
        if not hasattr(restored, "_has_mpl") or not restored._has_mpl:
            pytest.skip("matplotlib not available")
        restored.restore_state(state, dataset=None)

        restored.set_active_label_group("g1")
        assert restored._label_field_combo.currentData() == "temperature"

        restored.set_active_label_group("g2")
        assert restored._label_field_combo.currentData() == "field"

    def test_annotations_are_scoped_per_data_group(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel._annotations.append({"x": 0.2, "y": 0.1, "text": "default", "artist": None})

        panel.set_active_label_group("g1")
        assert panel._annotations == []
        panel._annotations.append({"x": 0.3, "y": 0.2, "text": "g1", "artist": None})

        panel.set_active_label_group("g2")
        assert panel._annotations == []
        panel._annotations.append({"x": 0.4, "y": 0.3, "text": "g2", "artist": None})

        panel.set_active_label_group(None)
        assert [ann["text"] for ann in panel._annotations] == ["default"]

        panel.set_active_label_group("g1")
        assert [ann["text"] for ann in panel._annotations] == ["g1"]

        panel.set_active_label_group("g2")
        assert [ann["text"] for ann in panel._annotations] == ["g2"]

    def test_group_annotations_persist_in_panel_state(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel._annotations.append({"x": 0.1, "y": 0.1, "text": "default", "artist": None})
        panel.set_active_label_group("g1")
        panel._annotations.append({"x": 0.2, "y": 0.2, "text": "g1", "artist": None})
        panel.set_active_label_group("g2")
        panel._annotations.append({"x": 0.3, "y": 0.3, "text": "g2", "artist": None})

        state = panel.get_state()

        restored = PlotPanel()
        if not hasattr(restored, "_has_mpl") or not restored._has_mpl:
            pytest.skip("matplotlib not available")
        restored.restore_state(state, dataset=None)

        assert [ann["text"] for ann in restored._annotations] == ["default"]

        restored.set_active_label_group("g1")
        assert [ann["text"] for ann in restored._annotations] == ["g1"]

        restored.set_active_label_group("g2")
        assert [ann["text"] for ann in restored._annotations] == ["g2"]

    def test_dataset_label_falls_back_to_run_label_when_field_missing(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = MuonDataset(
            time=np.array([0.0, 1.0]),
            asymmetry=np.array([0.1, 0.09]),
            error=np.array([0.01, 0.01]),
            metadata={"run_number": 111},
        )
        idx = panel._label_field_combo.findData("comment")
        assert idx >= 0
        panel._label_field_combo.setCurrentIndex(idx)

        assert panel._dataset_label_for(ds) == str(ds.run_label)

    def test_bunching_only_changes_plotted_representation(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """Bunching should preserve the source dataset and create a fit-ready copy."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        original_time = sample_dataset.time.copy()
        original_asymmetry = sample_dataset.asymmetry.copy()
        original_error = sample_dataset.error.copy()

        panel._bunch_factor.setValue(5)
        panel.plot_dataset(sample_dataset)
        analysis_dataset = panel.get_analysis_dataset(sample_dataset)

        assert panel._current_dataset is sample_dataset
        assert analysis_dataset is not None
        assert analysis_dataset is not sample_dataset
        assert len(analysis_dataset.time) < len(sample_dataset.time)
        np.testing.assert_array_equal(sample_dataset.time, original_time)
        np.testing.assert_array_equal(sample_dataset.asymmetry, original_asymmetry)
        np.testing.assert_array_equal(sample_dataset.error, original_error)

    def test_oversized_bunch_factor_falls_back_to_source_dataset(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([0.2, 0.15, 0.1]),
            error=np.array([0.01, 0.01, 0.01]),
            metadata={"run_number": 9981},
        )

        panel._bunch_factor.setValue(10)
        analysis_dataset = panel.get_analysis_dataset(ds)

        assert analysis_dataset is ds

    def test_plot_dataset_with_oversized_bunch_factor_keeps_data_visible(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([0.2, 0.15, 0.1]),
            error=np.array([0.01, 0.01, 0.01]),
            metadata={"run_number": 9982},
        )

        panel._bunch_factor.setValue(10)
        panel.plot_dataset(ds)

        assert panel._current_dataset is ds
        assert panel._last_plot_time is not None
        assert panel._last_plot_time.size == 3

    def test_clear_plot(self, panel: PlotPanel, sample_dataset: MuonDataset) -> None:
        """Test clearing the plot."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        panel.plot_dataset(sample_dataset)
        panel.clear()
        # Should clear without error
        assert panel._canvas is not None

    def test_log_scale(self, panel: PlotPanel, sample_dataset: MuonDataset) -> None:
        """Test setting log scale."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        panel.plot_dataset(sample_dataset)
        # Panel should have log scale controls
        if hasattr(panel, "set_xscale"):
            panel.set_xscale("log")
        if hasattr(panel, "set_yscale"):
            panel.set_yscale("log")
        # No error should occur
        assert True

    def test_fit_range_defaults_to_data_extent(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """Fit range should initialize to the currently plotted x-range."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        x_min, x_max = panel.get_fit_range()
        assert x_min == pytest.approx(float(sample_dataset.time.min()))
        assert x_max == pytest.approx(float(sample_dataset.time.max()))

    def test_grouped_subplots_draw_fit_range_handles(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        datasets = [
            MuonDataset(
                time=np.array([0.0, 1.0, 2.0]),
                asymmetry=np.array([1.0, 0.9, 0.8]),
                error=np.array([0.01, 0.01, 0.01]),
                metadata={"run_number": -(index + 1)},
            )
            for index in range(3)
        ]

        panel.plot_grouped_time_domain_subplots(datasets)

        assert len(panel._fit_span_artists) == len(datasets)
        assert len(panel._fit_min_handles) == len(datasets)
        assert len(panel._fit_max_handles) == len(datasets)

    def test_vector_all_subplots_draw_fit_range_handles(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.array([0.0, 1.0, 2.0])
        e = np.array([0.01, 0.01, 0.01])
        datasets_by_axis = {
            "P_x": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.3, 0.2, 0.1]),
                    error=e,
                    metadata={"run_number": 6101},
                )
            ],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.2, 0.1, 0.05]),
                    error=e,
                    metadata={"run_number": 6101},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.1, 0.05, 0.02]),
                    error=e,
                    metadata={"run_number": 6101},
                )
            ],
        }

        panel._current_polarization_axis = "ALL"
        panel.plot_vector_subplots(datasets_by_axis)

        assert len(panel._fit_span_artists) == 3
        assert len(panel._fit_min_handles) == 3
        assert len(panel._fit_max_handles) == 3

    def test_get_fit_dataset_applies_selected_range(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """Only points inside the selected fit range should be returned."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel.set_fit_range(2.0, 4.0)

        fit_ds = panel.get_fit_dataset(sample_dataset)
        assert fit_ds is not None
        assert np.all(fit_ds.time >= 2.0)
        assert np.all(fit_ds.time <= 4.0)
        assert len(fit_ds.time) < len(sample_dataset.time)

    def test_fit_range_can_extend_beyond_data_extent(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """Out-of-data fit bounds should be preserved without dropping overlap."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel.set_fit_range(-1.0, 12.0)

        x_min, x_max = panel.get_fit_range()
        assert x_min == pytest.approx(-1.0)
        assert x_max == pytest.approx(12.0)
        assert panel._x_min.value() == pytest.approx(-1.0)
        assert panel._x_max.value() == pytest.approx(12.0)

        fit_ds = panel.get_fit_dataset(sample_dataset)
        assert fit_ds is not None
        assert len(fit_ds.time) == len(sample_dataset.time)
        assert fit_ds.time[0] == pytest.approx(float(sample_dataset.time.min()))
        assert fit_ds.time[-1] == pytest.approx(float(sample_dataset.time.max()))

    def test_fit_range_outside_data_survives_dataset_replot(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """Replotting should not snap an out-of-data fit range back to the data extent."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel.set_fit_range(-1.0, 12.0)
        panel.plot_dataset(sample_dataset)

        x_min, x_max = panel.get_fit_range()
        assert x_min == pytest.approx(-1.0)
        assert x_max == pytest.approx(12.0)
        assert panel._x_min.value() == pytest.approx(-1.0)
        assert panel._x_max.value() == pytest.approx(12.0)

    def test_fit_range_prompt_allows_min_below_dataset_extent(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """The exact-value fit-range dialog should allow bounds below the data minimum."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)

        with patch(
            "asymmetry.gui.panels.plot_panel.QInputDialog.getDouble",
            return_value=(-1.0, True),
        ):
            panel._prompt_handle_value_edit("min")

        x_min, x_max = panel.get_fit_range()
        assert x_min == pytest.approx(-1.0)
        assert x_max == pytest.approx(float(sample_dataset.time.max()))
        assert panel._x_min.value() == pytest.approx(-1.0)

    def test_get_current_plot_export_data_available_with_plotted_data(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        payloads = panel.get_current_plot_export_data()
        assert payloads is not None
        assert len(payloads) == 1
        assert payloads[0]["run_number"] == sample_dataset.run_number
        assert payloads[0]["fit"] is None

    def test_export_controls_enabled_after_plotting_data(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        assert panel._export_gle_btn.isEnabled() is False
        assert panel._gle_format_combo.isEnabled() is False

        panel.plot_dataset(sample_dataset)

        assert panel._export_gle_btn.isEnabled() is True
        assert panel._gle_format_combo.isEnabled() is True

    def test_get_current_plot_export_data_includes_components_and_annotations(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        t_fit = np.linspace(0.0, 10.0, 120)
        y_fit = 0.18 * np.exp(-0.45 * t_fit)
        panel.plot_fit(
            t_fit,
            y_fit,
            label="Fit",
            component_curves=[
                ("Exponential", y_fit - 0.01),
                ("Constant", np.full_like(t_fit, 0.01)),
            ],
        )
        panel._annotations = [{"x": 1.0, "y": 0.12, "text": "peak", "artist": None}]

        payloads = panel.get_current_plot_export_data()
        assert payloads is not None
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["run_number"] == sample_dataset.run_number
        assert len(payload["components"]) == 2
        assert payload["components"][0]["name"] == "Exponential"
        assert payload["annotations"][0]["text"] == "peak"

    def test_single_fit_curve_is_restored_when_returning_to_dataset(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds1 = sample_dataset
        ds2 = MuonDataset(
            time=sample_dataset.time,
            asymmetry=sample_dataset.asymmetry,
            error=sample_dataset.error,
            metadata={"run_number": 999},
        )

        t_fit = np.linspace(0.0, 10.0, 120)
        y_fit = 0.18 * np.exp(-0.45 * t_fit)

        panel.plot_dataset(ds1)
        panel.plot_fit(t_fit, y_fit, label="Fit")
        assert panel.get_current_plot_export_data() is not None

        panel.plot_dataset(ds2)
        payload_no_fit = panel.get_current_plot_export_data()
        assert payload_no_fit is not None
        assert payload_no_fit[0]["run_number"] == ds2.run_number
        assert payload_no_fit[0]["fit"] is None

        panel.plot_dataset(ds1)
        restored = panel.get_current_plot_export_data()
        assert restored is not None
        assert restored[0]["run_number"] == ds1.run_number

    def test_export_current_plot_warns_when_no_data(
        self, panel: PlotPanel, sample_dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        warnings: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **_kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
        )

        panel.export_current_plot()

        assert warnings
        assert "No plotted data" in warnings[0]

    def test_export_current_plot_writes_gle_and_compiles_pdf(
        self,
        panel: PlotPanel,
        sample_dataset: MuonDataset,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        t_fit = np.linspace(0.0, 10.0, 120)
        y_fit = 0.18 * np.exp(-0.45 * t_fit)
        fit_function = "A0*exp(-lambda*t)+C"
        panel.plot_fit(t_fit, y_fit, label="Fit", fit_function=fit_function)
        panel._annotations = [{"x": 2.0, "y": 0.09, "text": "note", "artist": None}]

        target_gle = tmp_path / "asymmetry_plot.gle"
        resolved_gle, _ = resolve_gle_export_paths(target_gle, folder=True)
        axis = _FakeAxis()
        fig = _FakeFigure(axis)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        subprocess_calls: list[tuple[list[str], str | None]] = []
        dialogs: list[tuple[str, str, str]] = []
        previews: list[str] = []

        panel._x_min.setValue(1.25)
        panel._x_max.setValue(8.75)
        panel._y_min.setValue(-0.3)
        panel._y_max.setValue(0.4)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr(
            "subprocess.run",
            lambda args, **kwargs: subprocess_calls.append((list(args), kwargs.get("cwd"))),
        )
        monkeypatch.setattr(
            panel,
            "_show_export_result_dialog",
            lambda title, summary, details: dialogs.append((title, summary, details)),
        )
        monkeypatch.setattr(
            panel,
            "_show_gle_preview",
            lambda gle_path: previews.append(str(gle_path)),
        )

        panel.export_current_plot()

        assert resolved_gle.exists()
        assert axis.errorbar_calls
        assert axis.plot_calls
        assert axis.text_calls
        assert axis.xlim_calls
        assert axis.ylim_calls
        assert axis.xlim_calls[-1] == (1.25, 8.75)
        assert axis.ylim_calls[-1] == (-0.3, 0.4)
        assert "folder" not in fig.saved_kwargs[-1]
        assert axis.xlabel_calls[-1] == "Time (µs)"
        assert subprocess_calls
        assert subprocess_calls[0][0][:3] == ["gle", "-d", "pdf"]
        assert str(resolved_gle) in subprocess_calls[0][0]
        assert subprocess_calls[0][1] == str(resolved_gle.parent)

        fit_files = sorted(resolved_gle.parent.glob("*.fit"))
        assert fit_files
        fit_text = fit_files[0].read_text(encoding="utf-8")
        assert f"! fit_function: {fit_function}" in fit_text
        assert dialogs
        assert dialogs[0][0] == "Export Successful"
        assert "Data/fit files:" in dialogs[0][2]
        assert previews == [str(resolved_gle)]

    def test_show_gle_preview_returns_early_under_pytest(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        gle_path = tmp_path / "preview.gle"
        gle_path.write_text("! fake gle", encoding="utf-8")

        monkeypatch.setenv(
            "PYTEST_CURRENT_TEST",
            "tests/test_plot_panel.py::test_show_gle_preview_returns_early_under_pytest",
        )

        panel._show_gle_preview(gle_path)

    def test_export_current_plot_sanitizes_gle_text(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 60)
        e = np.full_like(t, 0.01)
        payload = {
            "run_number": 901,
            "label": "\x1b[91mRed Label\x1b[0m μ-test",
            "data": {"t": t, "y": 0.2 * np.exp(-0.3 * t), "err": e},
            "fit": {"t": t, "y": 0.19 * np.exp(-0.28 * t), "label": "Fit"},
            "fit_metadata": {},
            "annotations": [{"x": 1.0, "y": 0.1, "text": "note \x1b[92mOK\x1b[0m"}],
        }

        target_gle = tmp_path / "sanitize_export.gle"
        axis = _FakeAxis()
        fig = _FakeFigure(axis)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(panel, "get_current_plot_export_data", lambda: [payload])
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        label = axis.errorbar_calls[0]["kwargs"].get("label")
        assert "\x1b" not in str(label)
        assert "Red Label" in str(label)

        ann_text = axis.text_calls[0]["args"][2]
        assert "\x1b" not in str(ann_text)

    def test_export_current_plot_dat_header_includes_grouping(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 80)
        e = np.full_like(t, 0.01)
        run = Run(
            run_number=2401,
            grouping={
                "forward": [1, 2, 3],
                "backward": [4, 5, 6],
                "alpha": 1.125,
                "first_good_bin": 8,
                "last_good_bin": 72,
            },
        )
        ds = MuonDataset(
            time=t,
            asymmetry=0.2 * np.exp(-0.3 * t),
            error=e,
            metadata={"run_number": 2401},
            run=run,
        )
        panel.plot_dataset(ds)

        target_gle = tmp_path / "grouping_export.gle"
        resolved_gle, _ = resolve_gle_export_paths(target_gle, folder=True)
        axis = _FakeAxis()
        fig = _FakeFigure(axis)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        dat_files = sorted(resolved_gle.parent.glob("*.dat"))
        assert dat_files
        dat_text = dat_files[0].read_text(encoding="utf-8")
        assert "! START OF RUN INFORMATION" in dat_text
        assert "!  Run number  : 2401" in dat_text
        assert "! END OF RUN INFORMATION" in dat_text
        assert "! START OF GROUPING INFORMATION" in dat_text
        assert "!  Group#01  Hist(t0): 01, 02, 03" in dat_text
        assert "!  Group#02  Hist(t0): 04, 05, 06" in dat_text
        assert "!  Forward Group = forward, Backward Group = backward, Alpha = 1.1250" in dat_text
        assert "!  Offset to first good bin = 8, Last good bin = 72" in dat_text
        assert "! END OF GROUPING INFORMATION" in dat_text
        assert "! START OF DATA SET INFORMATION" in dat_text
        assert "! END OF DATA SET INFORMATION" in dat_text

    def test_export_current_plot_dat_header_survives_gleplot_save_overwrite(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 40)
        e = np.full_like(t, 0.01)
        run = Run(
            run_number=2410,
            grouping={"forward": [1, 2], "backward": [3, 4], "alpha": 0.95},
        )
        ds = MuonDataset(
            time=t,
            asymmetry=0.12 * np.exp(-0.25 * t),
            error=e,
            metadata={"run_number": 2410},
            run=run,
        )
        panel.plot_dataset(ds)

        target_gle = tmp_path / "overwrite_export.gle"
        resolved_gle, _ = resolve_gle_export_paths(target_gle, folder=True)
        axis = _FakeAxis()
        fig = _FakeFigure(axis, generate_data_files=True)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        dat_files = sorted(resolved_gle.parent.glob("*.dat"))
        assert dat_files
        dat_text = dat_files[0].read_text(encoding="utf-8")
        assert dat_text.startswith("! START OF RUN INFORMATION")
        assert "! START OF GROUPING INFORMATION" in dat_text
        assert "!  Group#01  Hist(t0): 01, 02" in dat_text
        assert "!  Group#02  Hist(t0): 03, 04" in dat_text
        assert "!  Forward Group = forward, Backward Group = backward, Alpha = 0.9500" in dat_text

    def test_export_current_plot_multi_uses_matching_colors_and_clean_legend(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 10.0, 80)
        e = np.full_like(t, 0.01)
        ds1 = MuonDataset(
            time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 1001}
        )
        ds2 = MuonDataset(
            time=t, asymmetry=0.16 * np.exp(-0.22 * t), error=e, metadata={"run_number": 1002}
        )
        panel.plot_datasets([ds1, ds2])

        panel._fit_curves = {
            1001: (t, 0.2 * np.exp(-0.28 * t), "Fit"),
            1002: (t, 0.16 * np.exp(-0.2 * t), "Fit"),
        }
        panel._fit_components_by_run = {1001: [], 1002: []}

        target_gle = tmp_path / "multi_export.gle"
        axis = _FakeAxis()
        created_figs: list[_FakeFigure] = []

        def _make_fig(**kwargs):
            fig = _FakeFigure(axis, figsize=kwargs.get("figsize"))
            created_figs.append(fig)
            return fig

        fake_glp = SimpleNamespace(figure=_make_fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        assert created_figs
        assert created_figs[0].figsize is not None
        width, height = created_figs[0].figsize
        assert width == 6.0
        assert height > 4.2

        assert len(axis.errorbar_calls) >= 2
        assert len(axis.plot_calls) >= 2

        first_data_color = axis.errorbar_calls[0]["kwargs"].get("color")
        first_fit_color = axis.plot_calls[0]["kwargs"].get("color")
        second_data_color = axis.errorbar_calls[1]["kwargs"].get("color")
        second_fit_color = axis.plot_calls[1]["kwargs"].get("color")

        assert first_data_color == first_fit_color
        assert second_data_color == second_fit_color
        assert axis.plot_calls[0]["kwargs"].get("label") is None
        assert axis.plot_calls[1]["kwargs"].get("label") is None

    def test_export_current_plot_vector_all_generates_subplots(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 50)
        e = np.full_like(t, 0.01)
        base = MuonDataset(
            time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 3001}
        )

        panel._current_polarization_axis = "ALL"
        panel._vector_subplot_datasets = {
            "P_x": [base],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=0.16 * np.exp(-0.25 * t),
                    error=e,
                    metadata={"run_number": 3001},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=0.12 * np.exp(-0.2 * t),
                    error=e,
                    metadata={"run_number": 3001},
                )
            ],
        }
        panel._y_limits_by_polarization = {
            "P_x": (-0.2, 0.4),
            "P_y": (-0.1, 0.3),
            "P_z": (-0.05, 0.2),
        }

        target_gle = tmp_path / "vector_all_export.gle"

        class _MultiAxisFigure:
            def __init__(self):
                self.axes: list[_FakeAxis] = []
                self.saved_paths: list[str] = []

            def add_subplot(self, *_args, **_kwargs):
                axis = _FakeAxis()
                self.axes.append(axis)
                return axis

            def savefig(self, path: str, **kwargs) -> None:
                output_path = Path(path)
                if kwargs.get("folder"):
                    output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
                    export_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                self.saved_paths.append(str(output_path))
                output_path.write_text("! fake gle", encoding="utf-8")

        fig = _MultiAxisFigure()
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        assert len(fig.axes) == 3
        y_labels = [axis.ylabel_calls[-1] for axis in fig.axes if axis.ylabel_calls]
        assert "a_0 P_{x}(t) (%)" in y_labels
        assert "a_0 P_{y}(t) (%)" in y_labels
        assert "a_0 P_{z}(t) (%)" in y_labels

        x_labels = [axis.xlabel_calls[-1] for axis in fig.axes if axis.xlabel_calls]
        assert x_labels
        assert x_labels[-1] == "Time (µs)"

    def test_apply_limits_in_all_mode_preserves_per_axis_limits(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ax_px = _FakeAxis()
        ax_py = _FakeAxis()
        ax_pz = _FakeAxis()
        panel._subplot_axes_by_polarization = {"P_x": ax_px, "P_y": ax_py, "P_z": ax_pz}
        panel._current_polarization_axis = "ALL"
        panel._y_limits_by_polarization = {
            "P_x": (-0.2, 0.4),
            "P_y": (-1.0, 1.0),
            "P_z": (-0.05, 0.2),
        }

        panel._x_min.setValue(0.5)
        panel._x_max.setValue(8.5)
        panel._y_min.setValue(-3.0)
        panel._y_max.setValue(4.0)

        panel._apply_limits()

        assert panel._y_limits_by_polarization["P_x"] == pytest.approx((-0.2, 0.4))
        assert panel._y_limits_by_polarization["P_y"] == pytest.approx((-1.0, 1.0))
        assert panel._y_limits_by_polarization["P_z"] == pytest.approx((-0.05, 0.2))
        assert ax_px.ylim_calls[-1] == pytest.approx((-0.2, 0.4))
        assert ax_py.ylim_calls[-1] == pytest.approx((-1.0, 1.0))
        assert ax_pz.ylim_calls[-1] == pytest.approx((-0.05, 0.2))

    def test_all_mode_disables_y_controls_and_sets_tooltips(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel._subplot_axes_by_polarization = {
            "P_x": _FakeAxis(),
            "P_y": _FakeAxis(),
            "P_z": _FakeAxis(),
        }
        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "ALL")

        assert not panel._y_min.isEnabled()
        assert not panel._y_max.isEnabled()
        assert panel._auto_y_btn.isEnabled()
        assert "x, y, and z" in panel._y_min.toolTip()
        assert "Auto Y updates" in panel._auto_y_btn.toolTip()
        assert panel._y_min.toolTip() == panel._y_max.toolTip()

        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "P_x")
        assert panel._y_min.isEnabled()
        assert panel._y_max.isEnabled()
        assert panel._auto_y_btn.isEnabled()
        assert panel._y_min.toolTip() == ""

    def test_auto_y_in_all_mode_updates_each_polarization(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 4.0, 5)
        e = np.full_like(t, 0.01)
        datasets_by_axis = {
            "P_x": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.10, 0.20, 0.15, 0.25, 0.30]),
                    error=e,
                    metadata={"run_number": 4301},
                )
            ],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([-0.30, -0.20, -0.10, 0.00, 0.10]),
                    error=e,
                    metadata={"run_number": 4301},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.02, 0.03, 0.01, 0.04, 0.02]),
                    error=e,
                    metadata={"run_number": 4301},
                )
            ],
        }

        panel._current_polarization_axis = "ALL"
        panel.plot_vector_subplots(datasets_by_axis)
        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "ALL")
        panel._x_min.setValue(0.0)
        panel._x_max.setValue(4.0)

        panel._auto_y_limits()

        px_limits = panel._y_limits_by_polarization["P_x"]
        py_limits = panel._y_limits_by_polarization["P_y"]
        pz_limits = panel._y_limits_by_polarization["P_z"]

        assert px_limits[1] > pz_limits[1]
        assert py_limits[0] < px_limits[0]
        assert panel._subplot_axes_by_polarization["P_x"].get_ylim() == pytest.approx(px_limits)
        assert panel._subplot_axes_by_polarization["P_y"].get_ylim() == pytest.approx(py_limits)
        assert panel._subplot_axes_by_polarization["P_z"].get_ylim() == pytest.approx(pz_limits)

    def test_auto_y_toggle_reapplies_on_vector_all_redraw(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 4.0, 5)
        e = np.full_like(t, 0.01)
        initial = {
            "P_x": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.10, 0.20, 0.15, 0.25, 0.30]),
                    error=e,
                    metadata={"run_number": 4401},
                )
            ],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([-0.30, -0.20, -0.10, 0.00, 0.10]),
                    error=e,
                    metadata={"run_number": 4401},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.02, 0.03, 0.01, 0.04, 0.02]),
                    error=e,
                    metadata={"run_number": 4401},
                )
            ],
        }
        updated = {
            "P_x": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.50, 0.55, 0.60, 0.58, 0.62]),
                    error=e,
                    metadata={"run_number": 4402},
                )
            ],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([-0.60, -0.55, -0.50, -0.45, -0.40]),
                    error=e,
                    metadata={"run_number": 4402},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=np.array([0.08, 0.10, 0.12, 0.09, 0.11]),
                    error=e,
                    metadata={"run_number": 4402},
                )
            ],
        }

        panel._current_polarization_axis = "ALL"
        panel.plot_vector_subplots(initial)
        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "ALL")
        panel._auto_y_btn.click()

        first_px_limits = panel._subplot_axes_by_polarization["P_x"].get_ylim()

        panel.plot_vector_subplots(updated)

        second_px_limits = panel._subplot_axes_by_polarization["P_x"].get_ylim()
        second_py_limits = panel._subplot_axes_by_polarization["P_y"].get_ylim()

        assert panel._auto_y_btn.isChecked()
        assert second_px_limits != pytest.approx(first_px_limits)
        assert second_px_limits[1] > first_px_limits[1]
        assert second_py_limits[0] < first_px_limits[0]

    def test_switching_from_all_mode_preserves_zoomed_x_limits(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 80)
        e = np.full_like(t, 0.01)
        datasets_by_axis = {
            "P_x": [
                MuonDataset(
                    time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 4101}
                )
            ],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=0.16 * np.exp(-0.25 * t),
                    error=e,
                    metadata={"run_number": 4101},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=0.12 * np.exp(-0.2 * t),
                    error=e,
                    metadata={"run_number": 4101},
                )
            ],
        }

        panel._current_polarization_axis = "ALL"
        panel.plot_vector_subplots(datasets_by_axis)
        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "ALL")

        panel._x_min.setValue(1.5)
        panel._x_max.setValue(5.5)
        panel._apply_limits()

        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "P_x")
        panel.plot_dataset(datasets_by_axis["P_x"][0])

        assert panel._x_min.value() == pytest.approx(1.5)
        assert panel._x_max.value() == pytest.approx(5.5)
        assert panel._ax.get_xlim() == pytest.approx((1.5, 5.5))

    def test_stale_axis_limit_callback_does_not_reset_all_mode_x_limits(
        self, panel: PlotPanel
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 80)
        e = np.full_like(t, 0.01)
        datasets_by_axis = {
            "P_x": [
                MuonDataset(
                    time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 4201}
                )
            ],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=0.16 * np.exp(-0.25 * t),
                    error=e,
                    metadata={"run_number": 4201},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=0.12 * np.exp(-0.2 * t),
                    error=e,
                    metadata={"run_number": 4201},
                )
            ],
        }

        panel._current_polarization_axis = "ALL"
        panel.plot_vector_subplots(datasets_by_axis)
        stale_axis = panel._subplot_axes_by_polarization["P_x"]

        panel._x_min.setValue(1.2)
        panel._x_max.setValue(6.4)
        panel._apply_limits()

        # Rebuild ALL-mode subplots as happens when moving between datasets.
        panel.plot_vector_subplots(datasets_by_axis)
        assert all(stale_axis is not ax for ax in panel._subplot_axes_by_polarization.values())

        # Simulate a late callback from an axis that is no longer active.
        panel._on_axis_limits_changed(stale_axis)

        assert panel._x_min.value() == pytest.approx(1.2)
        assert panel._x_max.value() == pytest.approx(6.4)

    def test_export_vector_all_uses_subplots_sharex_when_available(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 50)
        e = np.full_like(t, 0.01)
        base = MuonDataset(
            time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 3002}
        )

        panel._current_polarization_axis = "ALL"
        panel._vector_subplot_datasets = {
            "P_x": [base],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=0.16 * np.exp(-0.25 * t),
                    error=e,
                    metadata={"run_number": 3002},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=0.12 * np.exp(-0.2 * t),
                    error=e,
                    metadata={"run_number": 3002},
                )
            ],
        }

        target_gle = tmp_path / "vector_all_subplots_sharex.gle"

        class _SubplotFigure:
            def __init__(self):
                self.saved_paths: list[str] = []
                self.subplots_adjust_calls: list[dict[str, float]] = []

            def savefig(self, path: str, **kwargs) -> None:
                output_path = Path(path)
                if kwargs.get("folder"):
                    output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
                    export_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                self.saved_paths.append(str(output_path))
                output_path.write_text("! fake gle", encoding="utf-8")

            def subplots_adjust(self, **kwargs) -> None:
                self.subplots_adjust_calls.append(kwargs)

        subplot_fig = _SubplotFigure()
        axes = [_FakeAxis(), _FakeAxis(), _FakeAxis()]
        subplots_calls: list[dict[str, object]] = []

        def _subplots(**kwargs):
            subplots_calls.append(kwargs)
            return subplot_fig, axes

        fake_glp = SimpleNamespace(figure=lambda **_kwargs: _SubplotFigure(), subplots=_subplots)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        assert subplots_calls
        assert subplots_calls[0]["nrows"] == 3
        assert subplots_calls[0]["ncols"] == 1
        assert subplots_calls[0]["sharex"] is True
        assert subplot_fig.subplots_adjust_calls

    def test_export_vector_all_single_series_does_not_add_legend(
        self,
        panel: PlotPanel,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 50)
        e = np.full_like(t, 0.01)
        base = MuonDataset(
            time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 3003}
        )

        panel._current_polarization_axis = "ALL"
        panel._vector_subplot_datasets = {
            "P_x": [base],
            "P_y": [
                MuonDataset(
                    time=t,
                    asymmetry=0.16 * np.exp(-0.25 * t),
                    error=e,
                    metadata={"run_number": 3003},
                )
            ],
            "P_z": [
                MuonDataset(
                    time=t,
                    asymmetry=0.12 * np.exp(-0.2 * t),
                    error=e,
                    metadata={"run_number": 3003},
                )
            ],
        }

        target_gle = tmp_path / "vector_all_single_no_legend.gle"

        class _MultiAxisFigure:
            def __init__(self):
                self.axes: list[_FakeAxis] = []

            def add_subplot(self, *_args, **_kwargs):
                axis = _FakeAxis()
                self.axes.append(axis)
                return axis

            def savefig(self, path: str, **kwargs) -> None:
                output_path = Path(path)
                if kwargs.get("folder"):
                    output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
                    export_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("! fake gle", encoding="utf-8")

        fig = _MultiAxisFigure()
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr(
            "importlib.import_module", lambda name: fake_glp if name == "gleplot" else None
        )
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        assert len(fig.axes) == 3
        assert all(axis.legend_call_count == 0 for axis in fig.axes)


class TestDecimationStrategies:
    """Domain-specific decimation: stride for time scatter, min-max for spectra."""

    def test_frequency_minmax_decimation_preserves_narrow_peak(self, qapp: QApplication) -> None:
        """A 3-bin spectral peak must survive decimation of a 100k-point spectrum.

        A uniform stride of ~100 would drop it entirely — and in a spectrum the
        peaks ARE the physics — so frequency panels bucket by min/max instead.
        """
        panel = PlotPanel(domain="frequency")
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        n = 100_000
        freq = np.linspace(0.0, 50.0, n)
        amplitude = np.zeros(n)
        amplitude[50_001:50_004] = 1.0  # narrow peak, offset from any stride grid
        mask = np.ones(n, dtype=bool)
        panel._max_render_points_per_trace = 1000

        indices = panel._decimated_plot_indices(freq, mask, values=amplitude)

        assert indices.size < n
        assert indices.size <= 1002  # 2 per bucket + endpoints
        assert amplitude[indices].max() == 1.0, "min-max bucketing must keep the peak"

    def test_time_domain_keeps_uniform_stride(self, qapp: QApplication) -> None:
        """Time-domain scatter stays a uniform sample (an unbiased visual of noise)."""
        panel = PlotPanel()
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        n = 100_000
        t = np.linspace(0.0, 32.0, n)
        rng = np.random.default_rng(7)
        asym = rng.normal(0.0, 0.05, n)
        mask = np.ones(n, dtype=bool)
        panel._max_render_points_per_trace = 1000

        indices = panel._decimated_plot_indices(t, mask, values=asym)

        assert indices.size <= 1001
        # All gaps equal except possibly the appended final point.
        gaps = np.diff(indices)
        assert np.all(gaps[:-1] == gaps[0])
