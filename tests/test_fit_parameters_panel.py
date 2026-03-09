"""Tests for FitParametersPanel GLE export helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore

from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp: QApplication) -> FitParametersPanel:
    """Create a panel with predictable test data."""
    w = FitParametersPanel()
    w._rows = [
        _FitRow(
            run_number=2,
            field=200.0,
            temperature=10.0,
            values={"A0": 0.22, "Lambda": 0.12},
            errors={"A0": 0.02, "Lambda": 0.01},
        ),
        _FitRow(
            run_number=1,
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        ),
    ]
    w._varying_params = ["A0", "Lambda"]
    w._inferred_x_key = "field"
    w._x_combo.setCurrentText("Auto")
    return w


def test_gle_columns_for_param(panel: FitParametersPanel) -> None:
    assert panel._gle_columns_for_param("A0") == (2, 3)
    assert panel._gle_columns_for_param("Lambda") == (4, 5)
    assert panel._gle_columns_for_param("missing") is None


def test_write_gle_data_file_contains_column_map_and_sorted_rows(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
    out = tmp_path / "fit_parameters.dat"
    panel._write_gle_data_file(out)
    text = out.read_text(encoding="utf-8")

    assert "! Column map:" in text
    assert "!   c 1 = B_field(G)" in text
    assert "!   c 2 = A0" in text
    assert "!   c 3 = err_A0" in text
    assert "!   c 4 = Lambda" in text
    assert "!   c 5 = err_Lambda" in text

    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
    assert len(data_lines) == 2

    # Rows should be sorted by inferred x-axis (field): 100 before 200.
    assert float(data_lines[0].split()[0]) == pytest.approx(100.0)
    assert float(data_lines[1].split()[0]) == pytest.approx(200.0)


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.xscale_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.yscale_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.ylabel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def errorbar_from_file(self, *args, **kwargs) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return

    def set_ylabel(self, *_args, **_kwargs) -> None:
        self.ylabel_calls.append((_args, _kwargs))
        return

    def set_xscale(self, *_args, **_kwargs) -> None:
        self.xscale_calls.append((_args, _kwargs))
        return

    def set_yscale(self, *_args, **_kwargs) -> None:
        self.yscale_calls.append((_args, _kwargs))
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


def test_generate_gle_plot_uses_errorbar_from_file(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot.gle"

    axis = _FakeAxis()
    fig = _FakeFigure(axis)

    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        figure=lambda **_kwargs: fig,
    )

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr("shutil.which", lambda _name: "gle")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(panel, "_show_gle_preview", lambda *_a, **_k: None)

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert axis.calls, "Expected at least one errorbar_from_file call"
    first = axis.calls[0]
    assert first["args"][0] == data_path.name
    assert first["kwargs"]["x_col"] == 1
    assert first["kwargs"]["y_col"] == 2
    assert first["kwargs"]["yerr_col"] == 3
    assert str(gle_path) in fig.saved_paths


def test_generate_gle_plot_warns_for_old_gleplot(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot.gle"

    warnings: list[str] = []
    fake_glp = SimpleNamespace(Axes=type("OldAxes", (), {}))

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **_kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert warnings
    assert "gleplot" in warnings[0]


def test_effective_x_key_mapping(panel: FitParametersPanel) -> None:
    panel._x_combo.setCurrentText("𝐵 (G)")
    assert panel._effective_x_key() == "field"

    panel._x_combo.setCurrentText("𝑇 (K)")
    assert panel._effective_x_key() == "temperature"

    panel._x_combo.setCurrentText("Run")
    assert panel._effective_x_key() == "run"

    panel._x_combo.setCurrentText("Auto")
    panel._inferred_x_key = "temperature"
    assert panel._effective_x_key() == "temperature"


def test_generate_gle_plot_subplots_mode_uses_black_series(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot_subplots.gle"

    ax1 = _FakeAxis()
    ax2 = _FakeAxis()
    fig = _FakeFigure(ax1)

    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        subplots=lambda **_kwargs: (fig, [ax1, ax2]),
        figure=lambda **_kwargs: fig,
    )

    panel._plot_mode_combo.setCurrentText("Subplots")
    panel._log_x_check.setChecked(True)
    panel._log_y_check.setChecked(True)
    monkeypatch.setattr(panel, "_get_selected_y_parameters", lambda: ["A0", "Lambda"])

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(QMessageBox, "exec", lambda _self: None)
    monkeypatch.setattr(panel, "_show_gle_preview", lambda *_a, **_k: None)

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert ax1.calls and ax2.calls
    assert ax1.calls[0]["kwargs"]["color"] == "black"
    assert ax2.calls[0]["kwargs"]["color"] == "black"
    assert ax1.xscale_calls and ax1.yscale_calls
    assert ax2.xscale_calls and ax2.yscale_calls


def test_generate_gle_plot_dual_axis_assigns_y_and_y2(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot_dual.gle"

    axis = _FakeAxis()
    fig = _FakeFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        figure=lambda **_kwargs: fig,
    )

    panel._plot_mode_combo.setCurrentText("Single Axes")
    monkeypatch.setattr(panel, "_get_selected_y_parameters", lambda: ["A0", "Lambda"])

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr("shutil.which", lambda _name: "gle")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(panel, "_show_gle_preview", lambda *_a, **_k: None)

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert len(axis.calls) == 2
    assert axis.calls[0]["kwargs"]["yaxis"] == "y"
    assert axis.calls[1]["kwargs"]["yaxis"] == "y2"
