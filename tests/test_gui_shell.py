"""Headless smoke tests for GUI shell modules."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget

import asymmetry.gui.app as app_module
import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import MuonDataset


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)


class _StubDataBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_selected = _DummySignal()
        self.selection_changed = _DummySignal()
        self._datasets = {}

    def add_dataset(self, dataset):
        self._datasets[dataset.run_number] = dataset

    def get_dataset(self, run_number):
        return self._datasets.get(run_number)

    def get_selected_datasets(self):
        return list(self._datasets.values())


class _StubFitPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.fit_completed = _DummySignal()
        self.global_fit_completed = _DummySignal()
        self.last_dataset = None
        self.last_datasets = None

    def set_datasets(self, datasets):
        self.last_datasets = datasets
        return

    def set_dataset(self, dataset):
        self.last_dataset = dataset
        return


class _StubPlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._fit_curve = None
        self._fit_curves = {}
        self.fit_range_changed = _DummySignal()
        self.last_plotted_dataset = None

    def plot_dataset(self, dataset):
        self.last_plotted_dataset = dataset
        return

    def plot_fit(self, *_args, **_kwargs):
        return

    def set_global_fits(self, _curves):
        return

    def get_analysis_dataset(self, dataset):
        return dataset

    def get_fit_dataset(self, dataset):
        return dataset

    def clear(self):
        self.last_plotted_dataset = None
        return


class _StubLogPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.messages = []

    def log(self, message: str):
        self.messages.append(message)


class _StubFourier(QWidget):
    pass


class _StubFitParams(QWidget):
    def set_fit_results(self, *_args, **_kwargs):
        return


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_mainwindow_smoke_paths(monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowser)
    monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanel)
    monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanel)
    monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
    monkeypatch.setattr(mw_module, "FourierPanel", _StubFourier)
    monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParams)

    window = mw_module.MainWindow()

    # Exercise simple actions and status/log flows.
    window._on_fit()
    window._on_fourier()
    window._on_fit_parameters()

    ds = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.array([0.2, 0.1]),
        error=np.array([0.01, 0.01]),
        metadata={"run_number": 42, "field_comment_candidate": 150.0, "field_header": 0.0},
    )
    window._data_browser.add_dataset(ds)
    window._on_dataset_selected(42)
    assert window._fit_panel.last_dataset is ds

    choice = window._maybe_apply_comment_field(ds, "dummy.wim", apply_to_all=True)
    assert choice == "yes_to_all"
    assert ds.metadata["field"] == pytest.approx(150.0)

    window._plot_panel.fit_range_changed.emit(0.0, 1.0)
    assert window._fit_panel.last_dataset is ds

    results_dict = {
        42: (
            SimpleNamespace(success=True, reduced_chi_squared=1.2),
            (np.array([0.0, 1.0]), np.array([0.1, 0.1])),
        )
    }
    window._on_global_fit_completed(results_dict, SimpleNamespace())
    assert "Global fit completed" in window.statusBar().currentMessage()


def test_app_main_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeApp:
        def __init__(self, _argv):
            self.name = None
            self.org = None

        def setApplicationName(self, name):
            self.name = name

        def setOrganizationName(self, org):
            self.org = org

        def setWindowIcon(self, _icon):
            return

        def exec(self):
            return 0

    class _FakeWindow:
        def show(self):
            return

    monkeypatch.setattr(app_module, "QApplication", _FakeApp)
    monkeypatch.setattr(app_module, "MainWindow", _FakeWindow)
    monkeypatch.setattr(app_module, "_load_app_icon", lambda: None)

    captured = {}

    def _fake_exit(code):
        captured["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(app_module.sys, "exit", _fake_exit)

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 0
    assert captured["code"] == 0
