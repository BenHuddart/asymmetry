"""Tests for PlotPanel."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import PySide6 conditionally
pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton  # type: ignore

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.panels.plot_panel import PlotPanel


class _FakeAxis:
    def __init__(self) -> None:
        self.errorbar_calls: list[dict[str, object]] = []
        self.plot_calls: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []
        self.xlim_calls: list[tuple[float, float]] = []
        self.ylim_calls: list[tuple[float, float]] = []
        self.xlabel_calls: list[str] = []
        self.ylabel_calls: list[str] = []
        self.legend_call_count = 0

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
        self.figsize = figsize
        self.generate_data_files = generate_data_files

    def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
        return self._axis

    def savefig(self, path: str) -> None:
        self.saved_paths.append(path)
        Path(path).write_text("! fake gle", encoding="utf-8")
        if not self.generate_data_files:
            return

        out_dir = Path(path).parent
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

    def test_limit_spinboxes_follow_axis_limit_changes(self, panel: PlotPanel, sample_dataset: MuonDataset) -> None:
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

    def test_auto_y_uses_current_x_range_and_ignores_low_count_points(self, panel: PlotPanel) -> None:
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

        ds = MuonDataset(time=time, asymmetry=asym, error=err, metadata={"run_number": 321}, run=run)
        panel.plot_dataset(ds)

        panel._x_min.setValue(1.0)
        panel._x_max.setValue(3.0)
        panel._apply_limits()
        panel._auto_y_limits()

        assert panel._y_max.value() < 1.0
        assert panel._y_min.value() > -1.0

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

    def test_polarization_combo_uses_subscript_labels(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.set_polarization_axes(["P_x", "P_y", "P_z"], "P_x")
        labels = [panel._polarization_combo.itemText(i) for i in range(panel._polarization_combo.count())]

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
        labels = [panel._polarization_combo.itemText(i) for i in range(panel._polarization_combo.count())]

        assert labels[0] == "All"

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
        assert not panel._alpha_label.isHidden()
        assert panel._alpha_label.text() == "(alpha = 1.2345)"

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
        assert not panel._alpha_label.isHidden()
        assert panel._alpha_label.text() == "(alpha = 1.2)"

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
        assert panel._alpha_label.isHidden()
        assert panel._alpha_label.text() == ""

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

    def test_multi_dataset_legend_labels_follow_selected_label_field(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

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
        assert panel._alpha_label.isHidden()
        assert panel._alpha_label.text() == ""

    def test_add_label_keeps_multi_dataset_redraw(
        self, panel: PlotPanel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

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

    def test_dataset_label_falls_back_to_run_label_when_field_missing(self, panel: PlotPanel) -> None:
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
            component_curves=[("Exponential", y_fit - 0.01), ("Constant", np.full_like(t_fit, 0.01))],
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
        axis = _FakeAxis()
        fig = _FakeFigure(axis)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        subprocess_calls: list[list[str]] = []
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
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr(
            "subprocess.run",
            lambda args, **_kwargs: subprocess_calls.append(list(args)),
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

        assert target_gle.exists()
        assert axis.errorbar_calls
        assert axis.plot_calls
        assert axis.text_calls
        assert axis.xlim_calls
        assert axis.ylim_calls
        assert axis.xlim_calls[-1] == (1.25, 8.75)
        assert axis.ylim_calls[-1] == (-0.3, 0.4)
        assert axis.xlabel_calls[-1] == "Time (µs)"
        assert subprocess_calls
        assert subprocess_calls[0][:3] == ["gle", "-d", "pdf"]
        assert str(target_gle) in subprocess_calls[0]

        fit_files = sorted(tmp_path.glob("*.fit"))
        assert fit_files
        fit_text = fit_files[0].read_text(encoding="utf-8")
        assert f"! fit_function: {fit_function}" in fit_text
        assert dialogs
        assert dialogs[0][0] == "Export Successful"
        assert "Data/fit files:" in dialogs[0][2]
        assert previews == [str(target_gle)]

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
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
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
        axis = _FakeAxis()
        fig = _FakeFigure(axis)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        dat_files = sorted(tmp_path.glob("*.dat"))
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
        axis = _FakeAxis()
        fig = _FakeFigure(axis, generate_data_files=True)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        dat_files = sorted(tmp_path.glob("*.dat"))
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
        ds1 = MuonDataset(time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 1001})
        ds2 = MuonDataset(time=t, asymmetry=0.16 * np.exp(-0.22 * t), error=e, metadata={"run_number": 1002})
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
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
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
        base = MuonDataset(time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 3001})

        panel._current_polarization_axis = "ALL"
        panel._vector_subplot_datasets = {
            "P_x": [base],
            "P_y": [MuonDataset(time=t, asymmetry=0.16 * np.exp(-0.25 * t), error=e, metadata={"run_number": 3001})],
            "P_z": [MuonDataset(time=t, asymmetry=0.12 * np.exp(-0.2 * t), error=e, metadata={"run_number": 3001})],
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

            def savefig(self, path: str) -> None:
                self.saved_paths.append(path)
                Path(path).write_text("! fake gle", encoding="utf-8")

        fig = _MultiAxisFigure()
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
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
        assert not panel._auto_y_btn.isEnabled()
        assert "x, y, and z" in panel._y_min.toolTip()
        assert panel._y_min.toolTip() == panel._y_max.toolTip()
        assert panel._y_min.toolTip() == panel._auto_y_btn.toolTip()

        panel.set_polarization_axes(["ALL", "P_x", "P_y", "P_z"], "P_x")
        assert panel._y_min.isEnabled()
        assert panel._y_max.isEnabled()
        assert panel._auto_y_btn.isEnabled()
        assert panel._y_min.toolTip() == ""

    def test_switching_from_all_mode_preserves_zoomed_x_limits(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        t = np.linspace(0.0, 8.0, 80)
        e = np.full_like(t, 0.01)
        datasets_by_axis = {
            "P_x": [MuonDataset(time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 4101})],
            "P_y": [MuonDataset(time=t, asymmetry=0.16 * np.exp(-0.25 * t), error=e, metadata={"run_number": 4101})],
            "P_z": [MuonDataset(time=t, asymmetry=0.12 * np.exp(-0.2 * t), error=e, metadata={"run_number": 4101})],
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
        base = MuonDataset(time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 3002})

        panel._current_polarization_axis = "ALL"
        panel._vector_subplot_datasets = {
            "P_x": [base],
            "P_y": [MuonDataset(time=t, asymmetry=0.16 * np.exp(-0.25 * t), error=e, metadata={"run_number": 3002})],
            "P_z": [MuonDataset(time=t, asymmetry=0.12 * np.exp(-0.2 * t), error=e, metadata={"run_number": 3002})],
        }

        target_gle = tmp_path / "vector_all_subplots_sharex.gle"

        class _SubplotFigure:
            def __init__(self):
                self.saved_paths: list[str] = []
                self.subplots_adjust_calls: list[dict[str, float]] = []

            def savefig(self, path: str) -> None:
                self.saved_paths.append(path)
                Path(path).write_text("! fake gle", encoding="utf-8")

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
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
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
        base = MuonDataset(time=t, asymmetry=0.2 * np.exp(-0.3 * t), error=e, metadata={"run_number": 3003})

        panel._current_polarization_axis = "ALL"
        panel._vector_subplot_datasets = {
            "P_x": [base],
            "P_y": [MuonDataset(time=t, asymmetry=0.16 * np.exp(-0.25 * t), error=e, metadata={"run_number": 3003})],
            "P_z": [MuonDataset(time=t, asymmetry=0.12 * np.exp(-0.2 * t), error=e, metadata={"run_number": 3003})],
        }

        target_gle = tmp_path / "vector_all_single_no_legend.gle"

        class _MultiAxisFigure:
            def __init__(self):
                self.axes: list[_FakeAxis] = []

            def add_subplot(self, *_args, **_kwargs):
                axis = _FakeAxis()
                self.axes.append(axis)
                return axis

            def savefig(self, path: str) -> None:
                Path(path).write_text("! fake gle", encoding="utf-8")

        fig = _MultiAxisFigure()
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_gle), "GLE files (*.gle)"),
        )
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_show_gle_preview", lambda *args, **kwargs: None)

        panel.export_current_plot()

        assert len(fig.axes) == 3
        assert all(axis.legend_call_count == 0 for axis in fig.axes)
