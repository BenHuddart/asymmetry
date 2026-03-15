"""Tests for PlotPanel."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Import PySide6 conditionally
pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore

from asymmetry.core.data.dataset import MuonDataset
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

    def test_analysis_dataset_is_pass_through(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        """Plot analysis dataset should now match the source dataset."""
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        analysis_dataset = panel.get_analysis_dataset(sample_dataset)

        assert panel._current_dataset is sample_dataset
        assert analysis_dataset is sample_dataset

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

    def test_plot_does_not_reset_user_limits_on_replot(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel._x_min.setValue(1.23)
        panel._x_max.setValue(4.56)
        panel._y_min.setValue(-7.89)
        panel._y_max.setValue(9.87)
        panel._apply_limits()

        panel.plot_dataset(sample_dataset)

        assert panel._x_min.value() == pytest.approx(1.23)
        assert panel._x_max.value() == pytest.approx(4.56)
        assert panel._y_min.value() == pytest.approx(-7.89)
        assert panel._y_max.value() == pytest.approx(9.87)

    def test_auto_x_keeps_y_limits(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel._y_min.setValue(-11.0)
        panel._y_max.setValue(12.0)
        panel._auto_x_limits()

        assert panel._y_min.value() == pytest.approx(-11.0)
        assert panel._y_max.value() == pytest.approx(12.0)

    def test_auto_y_ignores_unreliable_points(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = MuonDataset(
            time=np.array([0.0, 1.0], dtype=float),
            asymmetry=np.array([0.0, 100.0], dtype=float),
            error=np.array([1.0, 1.0], dtype=float),
            metadata={"run_number": 1},
        )
        panel.plot_dataset(ds)
        panel._compute_plot_valid_mask = lambda _dataset: np.array([True, False], dtype=bool)

        panel._auto_y_limits()

        # Reliable foreground is the first point only: 0 +/- 1 with small padding.
        assert panel._y_max.value() < 10.0

    def test_auto_y_uses_points_in_current_x_range(self, panel: PlotPanel) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = MuonDataset(
            time=np.array([0.0, 1.0, 2.0], dtype=float),
            asymmetry=np.array([0.0, 1.0, 200.0], dtype=float),
            error=np.array([0.2, 0.2, 0.2], dtype=float),
            metadata={"run_number": 1},
        )
        panel.plot_dataset(ds)
        panel._compute_plot_valid_mask = lambda _dataset: np.array([True, True, True], dtype=bool)

        panel._x_min.setValue(0.0)
        panel._x_max.setValue(1.1)
        panel._auto_y_limits()

        # The outlier at x=2 should be excluded from auto-Y.
        assert panel._y_max.value() < 10.0

    def test_limit_spinbox_editing_finished_applies_limits(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        panel._x_min.setValue(2.0)
        panel._x_max.setValue(3.0)
        panel._x_max.editingFinished.emit()

        x_lo, x_hi = panel._ax.get_xlim()
        assert x_lo == pytest.approx(2.0)
        assert x_hi == pytest.approx(3.0)

    def test_restore_state_keeps_saved_limits_when_dataset_replotted(
        self, panel: PlotPanel, sample_dataset: MuonDataset
    ) -> None:
        if not hasattr(panel, "_has_mpl") or not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.plot_dataset(sample_dataset)
        state = panel.get_state()
        state["x_min"] = 2.5
        state["x_max"] = 4.5
        state["y_min"] = -3.0
        state["y_max"] = 6.0

        panel2 = PlotPanel()
        if not panel2._has_mpl:
            pytest.skip("matplotlib not available")
        panel2.restore_state(state, sample_dataset)

        assert panel2._x_min.value() == pytest.approx(2.5)
        assert panel2._x_max.value() == pytest.approx(4.5)
        assert panel2._y_min.value() == pytest.approx(-3.0)
        assert panel2._y_max.value() == pytest.approx(6.0)

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
