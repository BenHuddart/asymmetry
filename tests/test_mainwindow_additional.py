"""Additional tests for mainwindow functionality."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import numpy as np

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QToolBar  # type: ignore

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.gui.mainwindow import MainWindow


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
    return MainWindow()


class TestMainWindowBasic:
    def test_initialization(self, mainwindow: MainWindow) -> None:
        """Test mainwindow initializes correctly."""
        assert mainwindow is not None
        assert mainwindow.windowTitle() != ""

    def test_has_menu_bar(self, mainwindow: MainWindow) -> None:
        """Test menubar exists."""
        assert mainwindow.menuBar() is not None

    def test_has_central_widget(self, mainwindow: MainWindow) -> None:
        """Test central widget exists."""
        assert mainwindow.centralWidget() is not None

    def test_window_size(self, mainwindow: MainWindow) -> None:
        """Test window has reasonable size."""
        size = mainwindow.size()
        assert size.width() > 0
        assert size.height() > 0

    def test_on_fit_shows_fit_dock(self, mainwindow: MainWindow) -> None:
        """Fit action should unhide the fit dock if it starts hidden."""
        assert mainwindow._dock_fit.isHidden()
        mainwindow._on_fit()
        assert not mainwindow._dock_fit.isHidden()

    def test_on_fourier_shows_fourier_dock(self, mainwindow: MainWindow) -> None:
        """Fourier action should unhide the Fourier dock if it starts hidden."""
        assert mainwindow._dock_fourier.isHidden()
        mainwindow._on_fourier()
        assert not mainwindow._dock_fourier.isHidden()

    def test_on_fit_parameters_shows_params_dock(self, mainwindow: MainWindow) -> None:
        """Fit Parameters action should unhide the dock if it starts hidden."""
        assert mainwindow._dock_fit_parameters.isHidden()
        mainwindow._on_fit_parameters()
        assert not mainwindow._dock_fit_parameters.isHidden()

    def test_on_export_current_plot_delegates_to_plot_panel(self, mainwindow: MainWindow) -> None:
        """Export menu/toolbar handler should delegate to PlotPanel exporter."""
        called = {"count": 0}

        def _mark_called() -> None:
            called["count"] += 1

        mainwindow._plot_panel.export_current_plot = _mark_called
        mainwindow._on_export_current_plot()
        assert called["count"] == 1

    def test_toolbar_grouping_before_fit(self, mainwindow: MainWindow) -> None:
        """Toolbar should expose Grouping action before Fit for discoverability."""
        toolbar = mainwindow.findChild(QToolBar)
        assert toolbar is not None
        texts = [action.text() for action in toolbar.actions()]
        assert "Grouping" in texts
        assert "Fit" in texts
        assert texts.index("Grouping") < texts.index("Fit")

    def test_deadtime_correction_uses_good_frames(self, mainwindow: MainWindow) -> None:
        """Deadtime correction should include good-frame normalization (Mantid-style)."""
        counts = np.array([100.0], dtype=float)
        corrected = mainwindow._apply_deadtime_correction(
            counts,
            tau_us=0.01,
            bin_width_us=0.02,
            num_good_frames=1000.0,
        )
        expected = 100.0 / (1.0 - (100.0 * 0.01 / (0.02 * 1000.0)))
        assert corrected[0] == pytest.approx(expected)

    def test_run_info_inclusion_handler_updates_data_browser(self, mainwindow: MainWindow) -> None:
        """Run Info include/exclude signal should add/remove data-browser columns."""
        calls: list[tuple[str, str]] = []

        mainwindow._data_browser.add_extra_column = lambda key: calls.append(("add", key))
        mainwindow._data_browser.remove_extra_column = lambda key: calls.append(("remove", key))

        mainwindow._on_run_info_field_inclusion_changed("run_info.points", True)
        mainwindow._on_run_info_field_inclusion_changed("run_info.points", False)

        assert calls == [("add", "run_info.points"), ("remove", "run_info.points")]

    def test_cross_group_completion_shows_global_parameter_window(self, mainwindow: MainWindow) -> None:
        """Accepted cross-group fit should open and focus the global-fit result window."""
        model = ParameterCompositeModel(["Linear"])
        fit_result = CrossGroupFitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=1.0,
            message="Fit successful",
        )
        groups = [
            ParameterGroupData(
                group_id="g0",
                group_name="G0",
                x=np.array([1.0, 2.0], dtype=float),
                y=np.array([0.1, 0.2], dtype=float),
                yerr=np.array([0.01, 0.01], dtype=float),
                group_variable_value=1.0,
            )
        ]
        output = SimpleNamespace(model=model, fit_result=fit_result)

        mainwindow._on_cross_group_fit_completed("Lambda", groups, output)

        assert mainwindow._global_parameter_fit_window is not None
        assert mainwindow._global_parameter_fit_window.isVisible()
