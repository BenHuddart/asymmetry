"""Tests for PlotPanel."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Import PySide6 conditionally
pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton  # type: ignore

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.panels.plot_panel import PlotPanel


class _FakeAxis:
    def __init__(self) -> None:
        self.errorbar_calls: list[dict[str, object]] = []
        self.plot_calls: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []

    def errorbar(self, *args, **kwargs) -> None:
        self.errorbar_calls.append({"args": args, "kwargs": kwargs})

    def plot(self, *args, **kwargs) -> None:
        self.plot_calls.append({"args": args, "kwargs": kwargs})

    def text(self, *args, **kwargs) -> None:
        self.text_calls.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return

    def set_ylabel(self, *_args, **_kwargs) -> None:
        return

    def legend(self, *_args, **_kwargs) -> None:
        return


class _FakeFigure:
    def __init__(self, axis: _FakeAxis) -> None:
        self._axis = axis
        self.saved_paths: list[str] = []

    def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
        return self._axis

    def savefig(self, path: str) -> None:
        self.saved_paths.append(path)
        Path(path).write_text("! fake gle", encoding="utf-8")


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
    return PlotPanel()


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

        labels = panel.findChildren(QLabel)
        label_texts = {lbl.text() for lbl in labels}
        assert "Bunch:" not in label_texts

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

    def test_plot_dataset(self, panel: PlotPanel, sample_dataset: MuonDataset) -> None:
        """Test plotting a single dataset."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")
        panel.plot_dataset(sample_dataset)
        # Check if plot was created (canvas should have drawn something)
        assert panel._canvas is not None

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

    def test_get_current_plot_export_data_requires_fitted_curve(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        assert panel.get_current_plot_export_data() is None

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

        payload = panel.get_current_plot_export_data()
        assert payload is not None
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
        assert panel.get_current_plot_export_data() is None

        panel.plot_dataset(ds1)
        restored = panel.get_current_plot_export_data()
        assert restored is not None
        assert restored["run_number"] == ds1.run_number

    def test_export_current_plot_warns_when_no_fit(
        self, panel: PlotPanel, sample_dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        warnings: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            lambda *args, **_kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
        )

        panel.export_current_plot()

        assert warnings
        assert "No fitted curve" in warnings[0]

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
        panel.plot_fit(t_fit, y_fit, label="Fit")
        panel._annotations = [{"x": 2.0, "y": 0.09, "text": "note", "artist": None}]

        target_pdf = tmp_path / "current_plot.pdf"
        axis = _FakeAxis()
        fig = _FakeFigure(axis)
        fake_glp = SimpleNamespace(figure=lambda **_kwargs: fig)

        subprocess_calls: list[list[str]] = []
        infos: list[str] = []

        monkeypatch.setattr(
            "asymmetry.gui.panels.plot_panel.QFileDialog.getSaveFileName",
            lambda *_a, **_k: (str(target_pdf), "PDF files (*.pdf)"),
        )
        monkeypatch.setattr("importlib.import_module", lambda name: fake_glp if name == "gleplot" else None)
        monkeypatch.setattr("shutil.which", lambda _name: "gle")
        monkeypatch.setattr(
            "subprocess.run",
            lambda args, **_kwargs: subprocess_calls.append(list(args)),
        )
        monkeypatch.setattr(
            QMessageBox,
            "information",
            lambda *args, **_kwargs: infos.append(str(args[2]) if len(args) > 2 else ""),
        )

        panel.export_current_plot()

        gle_path = target_pdf.with_suffix(".gle")
        assert gle_path.exists()
        assert axis.errorbar_calls
        assert axis.plot_calls
        assert axis.text_calls
        assert subprocess_calls
        assert subprocess_calls[0][:3] == ["gle", "-d", "pdf"]
        assert str(gle_path) in subprocess_calls[0]
        assert infos
