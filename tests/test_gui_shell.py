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

    def get_group_id_for_run(self, run_number):
        if run_number in self._datasets:
            return "g1"
        return None

    def get_group_member_run_numbers(self, group_id):
        if group_id == "g1":
            return [42, 43]
        return []

    def get_group_name(self, group_id):
        if group_id == "g1":
            return "Group 1"
        return None


class _StubFitPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.fit_completed = _DummySignal()
        self.global_fit_completed = _DummySignal()
        self.last_dataset = None
        self.last_datasets = None
        self.last_global_results = None
        self.shared_calls = []

    def set_datasets(self, datasets):
        self.last_datasets = datasets
        return

    def set_dataset(self, dataset):
        self.last_dataset = dataset
        return

    def register_global_fit_results(self, results_by_run):
        self.last_global_results = results_by_run
        return

    def share_single_function_state(self, source_run_number, target_run_numbers):
        self.shared_calls.append((source_run_number, list(target_run_numbers)))
        return len(target_run_numbers)


class _StubPlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._fit_curve = None
        self._fit_curves = {}
        self.bunch_factor_changed = _DummySignal()
        self.fit_range_changed = _DummySignal()
        self.factor = 1
        self.last_plotted_dataset = None

    def plot_dataset(self, dataset):
        self.last_plotted_dataset = dataset
        return

    def plot_datasets(self, datasets):
        self.last_plotted_dataset = datasets[-1] if datasets else None
        return

    def plot_fit(self, *_args, **_kwargs):
        return

    def set_global_fits(self, _curves):
        return

    def get_analysis_dataset(self, dataset):
        if dataset is None or self.factor <= 1:
            return dataset
        return MuonDataset(
            time=dataset.time[:: self.factor],
            asymmetry=dataset.asymmetry[:: self.factor],
            error=dataset.error[:: self.factor],
            metadata={**dataset.metadata, "analysis_factor": self.factor},
            run=dataset.run,
        )

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

    choice = window._maybe_apply_comment_field(ds, "dummy.nxs", apply_to_all=True)
    assert choice == "yes_to_all"
    assert ds.metadata["field"] == pytest.approx(150.0)

    window._plot_panel.factor = 2
    window._plot_panel.bunch_factor_changed.emit(2)
    assert window._fit_panel.last_dataset is not ds
    assert window._fit_panel.last_dataset.metadata["analysis_factor"] == 2
    assert all(d.metadata["analysis_factor"] == 2 for d in window._fit_panel.last_datasets)

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
            events.append("app-init")

        def setApplicationName(self, name):  # noqa: N802
            self.name = name
            events.append("set-name")

        def setOrganizationName(self, org):  # noqa: N802
            self.org = org

        def setWindowIcon(self, _icon):  # noqa: N802
            events.append("set-icon")
            return

        def processEvents(self):  # noqa: N802
            events.append("process-events")

        def exec(self):
            events.append("exec")
            return 0

    class _FakeWindow:
        def __init__(self):
            events.append("window-init")

        def show(self):
            events.append("window-show")
            return

    class _FakeSplash:
        def finish(self, _window):
            events.append("splash-finish")

    events = []
    startup_pixmap = object()

    def _fake_startup_pixmap(_filename):
        events.append("load-startup-pixmap")
        return startup_pixmap

    def _fake_create_splash(_app, logo=None):
        assert logo is startup_pixmap
        events.append("splash-show")
        return _FakeSplash()

    monkeypatch.setattr(app_module, "QApplication", _FakeApp)
    monkeypatch.setattr(app_module, "MainWindow", _FakeWindow)
    monkeypatch.setattr(app_module, "_load_startup_pixmap", _fake_startup_pixmap)
    monkeypatch.setattr(
        app_module,
        "_load_app_icon",
        lambda _pixmap=None: events.append("load-icon") or object(),
    )
    monkeypatch.setattr(app_module, "_create_splash_screen", _fake_create_splash)

    captured = {}

    def _fake_exit(code):
        captured["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(app_module.sys, "exit", _fake_exit)

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 0
    assert captured["code"] == 0
    assert events == [
        "app-init",
        "load-startup-pixmap",
        "load-icon",
        "set-icon",
        "splash-show",
        "set-name",
        "window-init",
        "window-show",
        "splash-finish",
        "exec",
    ]


def test_mainwindow_share_single_fit_function_with_group(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowser)
    monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanel)
    monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanel)
    monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
    monkeypatch.setattr(mw_module, "FourierPanel", _StubFourier)
    monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParams)

    window = mw_module.MainWindow()
    ds42 = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.array([0.2, 0.1]),
        error=np.array([0.01, 0.01]),
        metadata={"run_number": 42},
    )
    ds43 = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.array([0.15, 0.08]),
        error=np.array([0.01, 0.01]),
        metadata={"run_number": 43},
    )
    window._data_browser.add_dataset(ds42)
    window._data_browser.add_dataset(ds43)

    window._on_share_single_function_with_group(42)

    assert window._fit_panel.shared_calls
    source, targets = window._fit_panel.shared_calls[-1]
    assert source == 42
    assert targets == [43]
    assert "Shared fit function" in window.statusBar().currentMessage()
