"""Headless smoke tests for GUI shell modules."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPixmap
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
        self.grouped_fit_completed = _DummySignal()
        self.grouped_time_domain_mode_changed = _DummySignal()
        self.last_dataset = None
        self.last_datasets = None
        self.last_global_results = None
        self.shared_calls = []
        self._grouped_mode = False

    def set_datasets(self, datasets):
        self.last_datasets = datasets
        return

    def set_dataset(self, dataset):
        self.last_dataset = dataset
        return

    def is_grouped_time_domain_mode(self):
        return bool(self._grouped_mode)

    def register_global_fit_results(self, results_by_run):
        self.last_global_results = results_by_run
        return

    def share_single_function_state(
        self, source_run_number, target_run_numbers, datasets_by_run=None
    ):
        self.shared_calls.append((source_run_number, list(target_run_numbers)))
        return len(target_run_numbers)


class _StubPlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._fit_curve = None
        self._fit_curves = {}
        self.bunch_factor_changed = _DummySignal()
        self.fit_range_changed = _DummySignal()
        self.time_view_changed = _DummySignal()
        self.factor = 1
        self.last_plotted_dataset = None
        self.last_grouped_datasets = None
        self._time_view_mode = "fb_asymmetry"
        self._time_view_modes = ["fb_asymmetry"]

    def plot_dataset(self, dataset):
        self.last_plotted_dataset = dataset
        return

    def plot_datasets(self, datasets):
        self.last_plotted_dataset = datasets[-1] if datasets else None
        return

    def plot_grouped_time_domain_subplots(self, datasets):
        self.last_grouped_datasets = list(datasets)
        return

    def current_time_view_mode(self):
        return self._time_view_mode

    def set_time_view_modes(self, modes, current_mode=None):
        self._time_view_modes = list(modes)
        if current_mode is not None:
            self._time_view_mode = current_mode
        elif self._time_view_mode not in self._time_view_modes:
            self._time_view_mode = self._time_view_modes[0]

    def set_current_time_view_mode(self, mode, *, emit_signal=False):
        self._time_view_mode = mode
        if emit_signal:
            self.time_view_changed.emit(mode)

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

    def get_fit_range(self):
        return (None, None)

    def set_fit_range(self, x_min, x_max):
        return

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
    def restore_group_phase_state(self, *_args, **_kwargs):
        return

    def clear_group_phase_state(self):
        return

    def set_group_definitions(self, *_args, **_kwargs):
        return

    def set_fft_status(self, message: str, *, success: bool = False) -> None:
        return


class _StubFitParams(QWidget):
    def set_fit_results(self, *_args, **_kwargs):
        return


class _StubMultiGroupFitWindow(QWidget):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.grouped_fit_completed = _DummySignal()
        self.grouped_preview_requested = _DummySignal()
        self.last_dataset = None
        self.last_block_state = None
        self._title = "Multi-Group Fit"

    def set_dataset(self, dataset):
        self.last_dataset = dataset
        if dataset is None:
            self._title = "Multi-Group Fit"
            return
        self._title = f"Multi-Group Fit — {getattr(dataset, 'run_label', dataset.run_number)}"

    def set_fit_blocked(self, blocked, reason=""):
        self.last_block_state = (blocked, reason)

    def dock_title(self):
        return self._title

    def grouped_fit_formula_string(self):
        return "A(t)"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _set_default_ui_scale() -> None:
    settings = QSettings()
    settings.setValue("ui/scale", 1.0)


def test_mainwindow_smoke_paths(monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowser)
    monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanel)
    monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanel)
    monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
    monkeypatch.setattr(mw_module, "FourierPanel", _StubFourier)
    monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParams)
    monkeypatch.setattr(mw_module, "MultiGroupFitWindow", _StubMultiGroupFitWindow)

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

    monkeypatch.setattr(
        window,
        "_grouped_time_domain_display_datasets",
        lambda dataset=None: [
            MuonDataset(
                time=np.array([0.0, 1.0]),
                asymmetry=np.array([1.0, 0.9]),
                error=np.array([0.0, 0.0]),
                metadata={"run_number": -4201},
            )
        ],
    )
    window._plot_panel.set_current_time_view_mode("groups")
    window._render_current_selection_plot()
    assert window._plot_panel.last_grouped_datasets is not None
    window._on_fit()
    assert isinstance(window._multi_group_fit_window, _StubMultiGroupFitWindow)
    assert window._fit_stack.currentWidget() is window._multi_group_fit_window
    assert not window._dock_fit.isHidden()

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


def test_macos_app_icon_pixmap_uses_rounded_mask(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("#ffffff"))

    rounded = app_module._macos_icon_pixmap(pixmap).toImage()

    assert rounded.pixelColor(0, 0).alpha() == 0
    assert rounded.pixelColor(32, 0).alpha() == 0
    assert rounded.pixelColor(32, 6).alpha() == 255
    assert rounded.pixelColor(32, 32).alpha() == 255


def test_main_uses_resource_fallback_for_splash_logo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeApp:
        def __init__(self, _argv):
            return

        def setApplicationName(self, _name):  # noqa: N802
            return

        def setOrganizationName(self, _name):  # noqa: N802
            return

        def setWindowIcon(self, _icon):  # noqa: N802
            return

        def processEvents(self):  # noqa: N802
            return

        def exec(self):
            return 0

    class _FakeWindow:
        def show(self):
            return

    class _FakeSplash:
        def finish(self, _window):
            return

    startup_fallback = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr(app_module, "QApplication", _FakeApp)
    monkeypatch.setattr(app_module, "MainWindow", _FakeWindow)
    monkeypatch.setattr(app_module, "_load_startup_pixmap", lambda _filename: None)
    monkeypatch.setattr(
        app_module,
        "_load_resource_pixmap",
        lambda _filename: startup_fallback,
    )
    monkeypatch.setattr(app_module, "_load_app_icon", lambda _pixmap=None: object())

    def _fake_create_splash(_app, logo=None):
        seen["logo"] = logo
        return _FakeSplash()

    monkeypatch.setattr(app_module, "_create_splash_screen", _fake_create_splash)
    monkeypatch.setattr(
        app_module.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 0
    assert seen["logo"] is startup_fallback


def test_non_macos_app_icon_pixmap_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(app_module.sys, "platform", "linux")
    pixmap = QPixmap(64, 64)

    assert app_module._macos_icon_pixmap(pixmap) is pixmap


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
