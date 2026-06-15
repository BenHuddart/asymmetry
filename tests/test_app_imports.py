"""Tests for app initialization."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]

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


class TestBenchStylesheetLoading:
    def test_loads_from_source_tree(self) -> None:
        """The bench stylesheet loads via the on-disk path in a source install."""
        from asymmetry.gui import app

        css = app._load_bench_stylesheet()
        assert css
        # Sentinel rule: scrollbar end-arrow buttons are zeroed out. If this is
        # missing the app falls back to bare Fusion chrome (arrows reappear).
        assert "QScrollBar::add-line" in css

    def test_frozen_entry_point_file_falls_back_to_resources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bench.qss still loads when __file__ is relocated outside the package.

        Reproduces the PyInstaller frozen build, where the entry script's
        ``__file__`` points at the archive root rather than asymmetry/gui/,
        so the ``Path(__file__).parent`` lookup misses and the
        ``importlib.resources`` fallback must take over.
        """
        from asymmetry.gui import app

        monkeypatch.setattr(app, "__file__", "/nonexistent/frozen/app.py")
        css = app._load_bench_stylesheet()
        assert css
        assert "QScrollBar::add-line" in css


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
