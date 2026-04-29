"""Tests for app initialization."""

from __future__ import annotations

import pytest

pyside6 = pytest.importorskip("PySide6")


class TestAppImports:
    def test_gui_app_import(self) -> None:
        """Test that gui.app can be imported."""
        from asymmetry.gui import app

        assert app is not None

    def test_app_has_run_function(self) -> None:
        """Test that app module has main function."""
        from asymmetry.gui import app

        assert hasattr(app, "main")

    def test_mainwindow_import(self) -> None:
        """Test that mainwindow can be imported."""
        from asymmetry.gui import mainwindow

        assert mainwindow is not None
        assert hasattr(mainwindow, "MainWindow")


class TestModuleStructure:
    def test_core_modules_exist(self) -> None:
        """Test that core modules can be imported."""
        from asymmetry import core

        assert core is not None

        from asymmetry.core import data

        assert data is not None

        from asymmetry.core import fitting

        assert fitting is not None

        from asymmetry.core import fourier

        assert fourier is not None

        from asymmetry.core import transform

        assert transform is not None

    def test_gui_modules_exist(self) -> None:
        """Test that gui modules can be imported."""
        from asymmetry import gui

        assert gui is not None

        from asymmetry.gui import panels

        assert panels is not None
