"""Additional tests for mainwindow functionality."""

from __future__ import annotations

import sys

import pytest

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore

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
